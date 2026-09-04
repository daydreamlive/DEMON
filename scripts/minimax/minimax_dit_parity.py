"""Numeric parity gate for ``acestep.engine.minimax_dit`` vs the reference.

The reference implementation of MiniMax-Music3 lives in ``diffusers >= 0.40``.
DEMON pins ``diffusers==0.37.1`` for ACE-Step, so the two implementations
cannot be imported into the same interpreter. The run is therefore split into
two stages that talk to each other through a safetensors fixture:

  stage ``ref``   runs in a throwaway venv with ``diffusers>=0.40``. It draws
                  every input from a fixed seed, runs the reference modules,
                  and writes inputs + reference outputs to
                  ``<scratch>/parity_fixtures.safetensors``.

  stage ``mine``  runs in the DEMON venv. It loads the SAME input tensors
                  (bit-identical, straight off disk), runs
                  ``acestep.engine.minimax_dit``, and reports cosine
                  similarity and relative RMS against the saved outputs.

Stage ``both`` (the default) shells out to ``--ref-python`` for the first half
and then does the second half in-process.

Coverage:
  * DiT at t = 0.05, 0.3, 0.6, 0.95, B=1, on REAL conditioning (the condition
    encoder's output for real autoregressive frame hiddens).
  * DiT at B=4 where every row carries a different t, a different latent and a
    different conditioning slice; the four rows are exactly the four B=1 cases,
    so batching is also checked row-against-row.
  * DAV decoder on a REAL latent produced by the reference flow-matching loop
    (30 Euler steps, CFG 1.7, upstream's inverted-sigma schedule).
  * Condition encoder on real frame hiddens, at a 200-frame chunk and at the
    full 500-frame take.

Bar: cosine >= 0.9999 per forward in fp32.

Usage::

    python scripts/minimax/minimax_dit_parity.py --ref-python <path-to-ref-venv-python>
    python scripts/minimax/minimax_dit_parity.py --stage ref    # in the ref venv
    python scripts/minimax/minimax_dit_parity.py --stage mine   # in the DEMON venv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ID = "MiniMaxAI/MiniMax-Music3"
COSINE_BAR = 0.9999

# DiT cases: (name, flow-matching t, conditioning start frame).
DIT_CASES = (
    ("t005", 0.05, 0),
    ("t030", 0.30, 100),
    ("t060", 0.60, 200),
    ("t095", 0.95, 300),
)
DIT_LATENT_FRAMES = 256
COND_CHUNK_FRAMES = 200  # upstream's denoising window
DENOISE_STEPS = 30
GUIDANCE_SCALE = 1.7
DAV_LATENT_FRAMES = 344  # ~4 s of audio out of the real latent


def default_scratch() -> Path:
    root = os.environ.get("MINIMAX_PARITY_SCRATCH")
    if root:
        return Path(root)
    return Path(
        r"C:\Users\ryanf\AppData\Local\Temp\claude\C---dev-projects-DEMON"
        r"\1cfc013b-c5f5-47d6-a49b-071d5ca255b9\scratchpad\minimax_dit"
    )


def resolve_model_dir(explicit: str | None) -> Path:
    """Locate the already-downloaded snapshot. Never triggers a download."""
    if explicit:
        return Path(explicit)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(REPO_ID, local_files_only=True))
    except Exception:
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        snapshots = sorted((cache / f"models--{REPO_ID.replace('/', '--')}" / "snapshots").glob("*"))
        if not snapshots:
            raise SystemExit(f"{REPO_ID} is not in the local HF cache; nothing to compare against.")
        return snapshots[-1]


def exact_fp32() -> None:
    """No TF32 anywhere: the bar is a true fp32 comparison."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_frame_hiddens(path: Path, device: torch.device) -> torch.Tensor:
    """Real per-frame hidden states from an autoregressive take, ``(1, F, 8*4096)``."""
    if not path.is_file():
        raise SystemExit(
            f"missing real frame hiddens at {path}; pass --frame-hiddens with a dump of the AR stage's output"
        )
    return torch.load(path, map_location="cpu")["fh"].float().to(device)


def make_inputs(condition: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    """Seeded latents + the real conditioning slices, identical across stages."""
    generator = torch.Generator(device="cpu").manual_seed(20260826)
    inputs: dict[str, torch.Tensor] = {}
    for name, t_value, cond_start in DIT_CASES:
        latent = torch.randn(1, 128, DIT_LATENT_FRAMES, generator=generator, dtype=torch.float32)
        inputs[f"dit/{name}/hidden_states"] = latent.to(device)
        inputs[f"dit/{name}/encoder_hidden_states"] = condition[
            :, cond_start : cond_start + DIT_LATENT_FRAMES
        ].contiguous()
        inputs[f"dit/{name}/timestep"] = torch.full((1,), t_value, dtype=torch.float32, device=device)
    # The B=4 case is exactly the four B=1 cases stacked, so a row-vs-row check
    # is a direct proof that per-row timesteps are honoured.
    inputs["dit/batch4/hidden_states"] = torch.cat([inputs[f"dit/{n}/hidden_states"] for n, _, _ in DIT_CASES])
    inputs["dit/batch4/encoder_hidden_states"] = torch.cat(
        [inputs[f"dit/{n}/encoder_hidden_states"] for n, _, _ in DIT_CASES]
    )
    inputs["dit/batch4/timestep"] = torch.cat([inputs[f"dit/{n}/timestep"] for n, _, _ in DIT_CASES])
    return inputs


def real_latent(transformer, scheduler, condition: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run upstream's own flow-matching loop to get a genuinely on-manifold
    latent for the decoder check. Sigmas are inverted by the shipped scheduler
    config, so t runs 0 (noise) -> ~1 (data)."""
    import numpy as np

    sigmas = np.linspace(1.0, 1.0 / DENOISE_STEPS, DENOISE_STEPS)
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    generator = torch.Generator(device="cpu").manual_seed(4242)
    latents = torch.randn(1, 128, condition.shape[1], generator=generator, dtype=torch.float32).to(device)
    uncond = torch.zeros_like(condition)
    for t in scheduler.timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        v_cond = transformer(
            hidden_states=latents, timestep=timestep, encoder_hidden_states=condition, return_dict=False
        )[0]
        v_uncond = transformer(
            hidden_states=latents, timestep=timestep, encoder_hidden_states=uncond, return_dict=False
        )[0]
        velocity = v_uncond + GUIDANCE_SCALE * (v_cond - v_uncond)
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
    return latents[..., :DAV_LATENT_FRAMES].contiguous()


def stage_ref(args) -> None:
    import diffusers
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )

    exact_fp32()
    device = pick_device(args.device)
    model_dir = resolve_model_dir(args.model_dir)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"[ref] diffusers {diffusers.__version__} torch {torch.__version__} device {device}", flush=True)

    frame_hiddens = load_frame_hiddens(Path(args.frame_hiddens), device)
    tensors: dict[str, torch.Tensor] = {"cond/full/hidden_states": frame_hiddens[:, :].cpu()}

    # --- condition encoder -------------------------------------------------
    encoder = (
        MiniMaxMusic3ConditionEncoder.from_pretrained(model_dir, subfolder="condition_encoder")
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    with torch.no_grad():
        chunk = frame_hiddens[:, :COND_CHUNK_FRAMES].contiguous()
        cond_chunk = encoder(chunk)
        cond_full = encoder(frame_hiddens)
    tensors["cond/chunk/hidden_states"] = chunk.cpu()
    tensors["cond/chunk/ref_out"] = cond_chunk.cpu()
    tensors["cond/full/ref_out"] = cond_full.cpu()
    print(f"[ref] cond encoder: {tuple(chunk.shape)} -> {tuple(cond_chunk.shape)}", flush=True)
    del encoder

    condition = cond_chunk.contiguous()
    inputs = make_inputs(condition, device)
    for key, value in inputs.items():
        tensors[key] = value.cpu()

    # --- transformer -------------------------------------------------------
    transformer = (
        MiniMaxMusic3Transformer1DModel.from_pretrained(model_dir, subfolder="transformer")
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    with torch.no_grad():
        for name, _, _ in DIT_CASES + (("batch4", 0.0, 0),):
            out = transformer(
                hidden_states=inputs[f"dit/{name}/hidden_states"],
                timestep=inputs[f"dit/{name}/timestep"],
                encoder_hidden_states=inputs[f"dit/{name}/encoder_hidden_states"],
                return_dict=False,
            )[0]
            tensors[f"dit/{name}/ref_out"] = out.cpu()
            print(f"[ref] dit {name}: {tuple(out.shape)}", flush=True)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
        latent = real_latent(transformer, scheduler, condition, device)
    tensors["dav/latents"] = latent.cpu()
    print(f"[ref] real latent: {tuple(latent.shape)} std {float(latent.std()):.4f}", flush=True)
    del transformer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- DAV decoder -------------------------------------------------------
    vocoder = (
        MiniMaxMusic3Vocoder.from_pretrained(model_dir, subfolder="vocoder")
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    assert_no_sampling(vocoder)
    with torch.no_grad():
        waveform = vocoder(latent)
    tensors["dav/ref_out"] = waveform.cpu()
    print(f"[ref] dav: {tuple(latent.shape)} -> {tuple(waveform.shape)}", flush=True)

    out_path = scratch / "parity_fixtures.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out_path))
    (scratch / "parity_fixtures.json").write_text(
        json.dumps(
            {
                "diffusers": diffusers.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "model_dir": str(model_dir),
                "cases": [c[0] for c in DIT_CASES],
                "timesteps": [c[1] for c in DIT_CASES],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ref] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)


def assert_no_sampling(module) -> None:
    """The decoder must be deterministic; there is no randn anywhere in it."""
    import inspect

    seen = set()
    for sub in module.modules():
        cls = type(sub)
        if cls in seen:
            continue
        seen.add(cls)
        try:
            # forward() only: torch's own __init__/reset_parameters mention randn.
            src = inspect.getsource(cls.forward)
        except (OSError, TypeError):
            continue
        for token in ("randn", "rand(", "multinomial", "bernoulli", "torch.normal", "generator"):
            if token in src:
                raise AssertionError(f"sampling op {token!r} found in {cls.__name__}.forward")


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a = reference.detach().to(torch.float64).flatten()
    b = candidate.detach().to(torch.float64).flatten()
    cosine = float(torch.dot(a, b) / (a.norm() * b.norm()))
    rel_rms = float((a - b).norm() / a.norm())
    return {
        "cosine": cosine,
        "rel_rms": rel_rms,
        "max_abs_diff": float((a - b).abs().max()),
        "ref_rms": float(a.pow(2).mean().sqrt()),
        "pass": cosine >= COSINE_BAR,
    }


def stage_mine(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from acestep.engine.minimax_dit import MiniMaxConditionEncoder, MiniMaxDAV, MiniMaxDiT

    exact_fp32()
    device = pick_device(args.device)
    model_dir = resolve_model_dir(args.model_dir)
    scratch = Path(args.scratch)
    fixtures = load_file(str(scratch / "parity_fixtures.safetensors"))
    manifest = json.loads((scratch / "parity_fixtures.json").read_text(encoding="utf-8"))
    import diffusers

    print(
        f"[mine] diffusers {diffusers.__version__} torch {torch.__version__} device {device} "
        f"(fixtures from diffusers {manifest['diffusers']})",
        flush=True,
    )

    results: dict[str, dict] = {}

    # --- condition encoder -------------------------------------------------
    encoder = MiniMaxConditionEncoder.from_pretrained(model_dir, dtype=torch.float32, device=device)
    with torch.no_grad():
        for case in ("chunk", "full"):
            out = encoder(fixtures[f"cond/{case}/hidden_states"].to(device))
            results[f"cond/{case}"] = metrics(fixtures[f"cond/{case}/ref_out"], out.cpu())
    del encoder

    # Length-policy probe. We resample with the exact integer ratio 441/128;
    # upstream evaluates the same product in float64 and truncates. Sweep the
    # frame counts to find where (if anywhere) the two disagree.
    probe_max = 16384
    divergences = []
    for n in range(1, probe_max + 1):
        exact = n * 441 // 128
        upstream = int(n * 44100 / 24000 * 960 / 512)
        if exact != upstream:
            divergences.append((n, upstream, exact))
    results["cond/length_policy"] = {
        "probe_max_frames": probe_max,
        "num_divergent": len(divergences),
        "first_divergences_frames_upstream_exact": divergences[:4],
    }

    # --- transformer -------------------------------------------------------
    transformer = MiniMaxDiT.from_pretrained(model_dir, dtype=torch.float32, device=device)
    outs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, _, _ in DIT_CASES + (("batch4", 0.0, 0),):
            out = transformer(
                fixtures[f"dit/{name}/hidden_states"].to(device),
                fixtures[f"dit/{name}/timestep"].to(device),
                fixtures[f"dit/{name}/encoder_hidden_states"].to(device),
            ).cpu()
            outs[name] = out
            results[f"dit/{name}"] = metrics(fixtures[f"dit/{name}/ref_out"], out)
    # Row-vs-row: batching must not change any row, and each row carries its own t.
    for index, (name, _, _) in enumerate(DIT_CASES):
        results[f"dit/batch4_row{index}_vs_{name}"] = metrics(outs[name][0], outs["batch4"][index])
    del transformer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- DAV decoder -------------------------------------------------------
    decoder = MiniMaxDAV.from_pretrained(model_dir, dtype=torch.float32, device=device)
    assert_no_sampling(decoder)
    with torch.no_grad():
        waveform = decoder(fixtures["dav/latents"].to(device)).cpu()
    results["dav/real_latent"] = metrics(fixtures["dav/ref_out"], waveform)

    # --- report ------------------------------------------------------------
    width = max(len(k) for k in results)
    print("", flush=True)
    print(f"{'case'.ljust(width)}   {'cosine':>15}  {'1-cosine':>11}  {'rel_rms':>11}  {'max_abs':>11}   verdict")
    failures = 0
    for key, value in results.items():
        if "cosine" not in value:
            continue
        verdict = "PASS" if value["pass"] else "FAIL"
        failures += 0 if value["pass"] else 1
        print(
            f"{key.ljust(width)}   {value['cosine']:>15.12f}  {1.0 - value['cosine']:>11.3e}  "
            f"{value['rel_rms']:>11.3e}  {value['max_abs_diff']:>11.3e}   {verdict}"
        )
    print("", flush=True)
    print(f"length policy vs upstream's float64 truncation: {results['cond/length_policy']}", flush=True)
    (scratch / "parity_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"bar: cosine >= {COSINE_BAR} in fp32 -- {'ALL PASS' if failures == 0 else f'{failures} FAILED'}", flush=True)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("ref", "mine", "both"), default="both")
    parser.add_argument("--ref-python", default=None, help="interpreter of the diffusers>=0.40 venv")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--scratch", default=str(default_scratch()))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--frame-hiddens",
        default=str(default_scratch() / "frame_hiddens_20s_seed7.pt"),
        help="dump of real per-frame hidden states from the autoregressive stage",
    )
    args = parser.parse_args()

    if args.stage == "ref":
        stage_ref(args)
        return 0
    if args.stage == "mine":
        return stage_mine(args)

    if not args.ref_python:
        raise SystemExit("--stage both needs --ref-python (the diffusers>=0.40 interpreter)")
    command = [
        args.ref_python,
        str(Path(__file__).resolve()),
        "--stage",
        "ref",
        "--scratch",
        args.scratch,
        "--device",
        args.device,
        "--frame-hiddens",
        args.frame_hiddens,
    ]
    if args.model_dir:
        command += ["--model-dir", args.model_dir]
    subprocess.run(command, check=True)
    return stage_mine(args)


if __name__ == "__main__":
    raise SystemExit(main())

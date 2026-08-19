"""SA3 decode-determinism gate: prove or disprove decode-stage stochasticity.

Loads the SA3 checkpoint locally (same loader as production), generates ONE
real latent, then interrogates the latent->audio decode stage in-process:

  [gen]        Is the offline generation path bit-stable for a fixed seed?
               (two `sam.generate(..., return_latents=True)` calls, same seed)
  [decode-on]  Production small-model path (`decode_sa3_latent`, i.e. what
               `SA3SAMECodec.decode_full` runs): decode the SAME latent twice
               with checkpoint-default noise flags. Any diff here is decode
               noise, full stop.
  [decode-off] Same, under `sa3_decode_noise_mode(enabled=False)` — the mode
               the SAME-L windowed production path already uses. Expect
               bit-identical.
  [decode-seeded] Same, with noise flags untouched but the global RNG
               reseeded before each decode. Expect bit-identical — shows
               seeding (keep the noise, pin the RNG) is a viable fix shape.
  [on-vs-off]  Magnitude of the noise contribution in audio space (how much
               the inference-time regularizer noise actually changes the
               waveform), for the "is disabling audible/harmful" call.

Writes listening WAVs and prints a stats table. No production code is
touched; this is the pre-change gate.

Run:
    .venv/Scripts/python.exe scripts/sa3/sa3_decode_determinism_gate.py \
        --model small-music --prompt "warm analog house groove, 124 bpm" \
        --duration 10 --steps 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
for _p in (str(_REPO_ROOT),):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from acestep import paths  # noqa: E402
from acestep.engine.sa3_helpers import ensure_sa3_paths, sa3_checkpoint_dir  # noqa: E402

ensure_sa3_paths()

import sa3_reference_generate as ref  # noqa: E402

from acestep.engine.sa3_stream_helpers import (  # noqa: E402
    decode_sa3_latent,
    sa3_decode_noise_mode,
)


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Objective diff stats between two [B?, C, N] float audio tensors."""
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    sig_rms = a.pow(2).mean().sqrt().item()
    diff_rms = (a - b).pow(2).mean().sqrt().item()
    return {
        "bit_identical": bool(torch.equal(a, b)),
        "max_abs_diff": d.max().item(),
        "rms_diff": diff_rms,
        "diff_db_rel_signal": (
            20.0 * torch.log10(torch.tensor(diff_rms / sig_rms)).item()
            if diff_rms > 0 and sig_rms > 0 else float("-inf")
        ),
    }


def fmt(name: str, s: dict) -> str:
    return (
        f"{name:<28} identical={str(s['bit_identical']):<5} "
        f"max_abs={s['max_abs_diff']:.3e} rms={s['rms_diff']:.3e} "
        f"rel_dB={s['diff_db_rel_signal']:+.1f}"
    )


def save_wav(path: Path, audio_cn: torch.Tensor, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio_cn.float().cpu().numpy().T, samplerate=sr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="small-music")
    ap.add_argument("--prompt", default="warm analog house groove, 124 bpm, deep bassline")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else (
        paths.models_dir() / "sa3" / "spike_out" / "decode_determinism"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = sa3_checkpoint_dir(args.model)
    cfg = json.loads((dest / "model_config.json").read_text(encoding="utf-8"))
    print(f"[load] {dest} device={device}")
    t0 = time.time()
    sam = ref.load_local_model(dest, device=device, model_half=(device == "cuda"))
    print(f"[load] done in {time.time() - t0:.1f}s")
    sr = int(sam.model.sample_rate)

    bottleneck = getattr(sam.model.pretransform.model, "bottleneck", None)
    print(f"[flags] bottleneck={type(bottleneck).__name__} "
          f"noise_regularize={getattr(bottleneck, 'noise_regularize', None)} "
          f"noise_augment_dim={getattr(bottleneck, 'noise_augment_dim', None)} "
          f"pretransform_training={sam.model.pretransform.training}")
    mask_noise_mods = [
        m for m in sam.model.pretransform.model.decoder.modules()
        if hasattr(m, "mask_noise")
    ]
    print(f"[flags] decoder modules with mask_noise: {len(mask_noise_mods)} "
          f"values={sorted({float(m.mask_noise) for m in mask_noise_mods})}")

    gen_kwargs = dict(
        prompt=args.prompt, duration=args.duration, steps=args.steps,
        seed=args.seed, cfg_scale=1.0, return_latents=True,
    )
    print(f"[gen] latent 1/2 (seed={args.seed}) ...")
    latent1 = sam.generate(**gen_kwargs)
    print(f"[gen] latent 2/2 (same seed) ...")
    latent2 = sam.generate(**gen_kwargs)
    print(f"[gen] latent shape={tuple(latent1.shape)} dtype={latent1.dtype}")

    results: dict[str, dict] = {}
    results["gen (same seed, twice)"] = stats(latent1.float(), latent2.float())

    # --- decode-on: production SA3SAMECodec.decode_full semantics ---------
    a1 = decode_sa3_latent(sam, latent1)
    a2 = decode_sa3_latent(sam, latent1)
    results["decode-on (default flags)"] = stats(a1, a2)

    # --- decode-off: noise sources disabled (SAME-L windowed-path mode) ---
    with sa3_decode_noise_mode(sam, enabled=False):
        b1 = decode_sa3_latent(sam, latent1)
    with sa3_decode_noise_mode(sam, enabled=False):
        b2 = decode_sa3_latent(sam, latent1)
    results["decode-off (noise disabled)"] = stats(b1, b2)

    # --- decode-seeded: noise kept, RNG pinned per decode ------------------
    torch.manual_seed(args.seed)
    c1 = decode_sa3_latent(sam, latent1)
    torch.manual_seed(args.seed)
    c2 = decode_sa3_latent(sam, latent1)
    results["decode-seeded (RNG pinned)"] = stats(c1, c2)

    # --- decompose the two noise sources -----------------------------------
    # bottleneck-only: mask_noise forced 0, noise_regularize left on
    saved_mask = [(m, float(m.mask_noise)) for m in mask_noise_mods]
    for m, _ in saved_mask:
        m.mask_noise = 0.0
    d1 = decode_sa3_latent(sam, latent1)
    d2 = decode_sa3_latent(sam, latent1)
    for m, v in saved_mask:
        m.mask_noise = v
    results["bottleneck-only (2 decodes)"] = stats(d1, d2)

    # mask-only: noise_regularize off, mask_noise left on
    if bottleneck is not None:
        saved_reg = bottleneck.noise_regularize
        bottleneck.noise_regularize = False
        e1 = decode_sa3_latent(sam, latent1)
        e2 = decode_sa3_latent(sam, latent1)
        bottleneck.noise_regularize = saved_reg
        results["mask-only (2 decodes)"] = stats(e1, e2)

    # --- how big is the noise contribution, audibly? -----------------------
    results["on-vs-off (same latent)"] = stats(a1, b1)
    results["seeded-vs-off (same latent)"] = stats(c1, b1)

    tag = f"{args.model}_seed{args.seed}_{int(args.duration)}s"
    wavs = {
        f"gate_{tag}_noise_on_run1.wav": a1,
        f"gate_{tag}_noise_on_run2.wav": a2,
        f"gate_{tag}_noise_off_run1.wav": b1,
        f"gate_{tag}_noise_off_run2.wav": b2,
        f"gate_{tag}_noise_seeded_run1.wav": c1,
    }
    for name, audio in wavs.items():
        save_wav(out_dir / name, audio[0], sr)
        print(f"[save] {out_dir / name}")

    print("\n=== gate results ===")
    for name, s in results.items():
        print(fmt(name, s))

    decode_stochastic = not results["decode-on (default flags)"]["bit_identical"]
    off_deterministic = results["decode-off (noise disabled)"]["bit_identical"]
    seeded_deterministic = results["decode-seeded (RNG pinned)"]["bit_identical"]
    print(f"\nGATE: decode stage stochastic = {decode_stochastic} "
          f"(noise-off deterministic = {off_deterministic}, "
          f"seeded deterministic = {seeded_deterministic})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

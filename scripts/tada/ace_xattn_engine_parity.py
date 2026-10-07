"""Eager vs TensorRT parity for the ACE cross-attention steering site (GPU).

Records every eager decoder forward of a real streaming drain (fixture
source + prompt, 8 steps) through the production ``StreamPipeline``
with its eager steering hooks, then replays the identical inputs, and
the identical steering tensors the pipeline built for that forward,
through the TensorRT engines:

* ``spectral``: the previous production engine (post-block input only),
  replayed for the unsteered case: the engine-numerics floor;
* ``tada``: the new engine with ``steering_xattn``.

Cases: unsteered; TADA-style cross-attention steering on blocks 6 and 7
with renorm (the paper's benchmark setting); the same without renorm;
post-block steering (the existing slot, must still hold on the new
engine). Reports per-forward cosine of the velocity against eager and
the relative size of the steering effect. Bar: cos >= 0.9998 per forward
wherever the previous engine meets it.

Run (PYTHONUTF8=1)::

    python scripts/tada/ace_xattn_engine_parity.py [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
while str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

CHECKPOINT = "acestep-v15-turbo"
FIXTURE = "low_fi_Gm_loop_60s_gnm.wav"
PROMPT = "driving cinematic synthwave, analog arpeggios, 152 bpm, G minor"
SEED = 1528
STEPS = 8
BAR = 0.9998
TADA_BLOCKS = (6, 7)


def _load_audio(path: Path, duration: float = 30.0):
    import soundfile as sf

    from acestep.nodes.types import Audio

    data, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio

        wav = torchaudio.transforms.Resample(sr, 48000)(wav)
    wav = wav[:2, : int(duration * 48000)]
    pool = 1920 * 5
    rem = wav.shape[-1] % pool
    if rem:
        wav = wav[:, : wav.shape[-1] - rem]
    return Audio(waveform=wav, sample_rate=48000)


def _unit(seed: int, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g)
    return v / v.norm()


def record(session, source, cond, configs_fn) -> list:
    """Drain one request; return per-forward inputs, steering tensors, output."""
    from acestep.steering.layout import HOOK_CROSS_ATTN_OUTPUT

    handle = session.stream(source=source, conditioning=cond, steps=STEPS,
                            shift=3.0, method="ode", pipeline_depth=1,
                            dcw_enabled=False)
    # First tick builds the pipeline (and runs step 0 unsteered); drop it.
    handle.tick(denoise=1.0, seed=SEED, source_latent=source.latent, steps=STEPS)
    pipe = handle.stream_node.pipeline
    pipe.set_steering(configs_fn(pipe))
    dec = pipe.decoder
    orig = dec.forward
    recs: list = []

    def rec(**kw):
        out = orig(**kw)
        B = kw["hidden_states"].shape[0]
        L, H = len(dec.layers), dec.config.hidden_size
        dev = kw["hidden_states"].device
        sp = torch.zeros(B, L, H, device=dev)
        sx = torch.zeros(B, L, H, device=dev)
        rn = torch.zeros(B, L, device=dev)
        pipe._fill_steering_rows(sp, B)
        pipe._fill_steering_rows(sx, B, HOOK_CROSS_ATTN_OUTPUT, rn)
        recs.append({
            "hs": kw["hidden_states"].detach().clone(),
            "t": kw["timestep"].detach().float().clone(),
            "enc": kw["encoder_hidden_states"].detach().clone(),
            "ctx": kw["context_latents"].detach().clone(),
            "sp": sp, "sx": sx, "rn": rn,
            "out": out[0].detach().float().clone(),
        })
        return out

    dec.forward = rec
    try:
        for _ in range(STEPS + 1):
            handle.tick(denoise=1.0, seed=SEED, source_latent=source.latent, steps=STEPS)
    finally:
        dec.forward = orig
        pipe.set_steering([])
    handle.close()
    return recs


def replay(engine, recs, use_xattn: bool) -> list:
    outs = []
    for r in recs:
        kw = {}
        if use_xattn:
            kw = {"steering_xattn": r["sx"], "steering_xattn_renorm": r["rn"]}
        o = engine._trt_decoder_step(
            r["hs"], r["t"], r["enc"], r["ctx"], steering=r["sp"], **kw,
        )
        outs.append(o.detach().float().clone())
    return outs


def cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.flatten().double(), b.flatten().double(), dim=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=None,
                    help="cross-attn shift size (default: 1.0 x mean token norm "
                         "of the block's cross-attention output)")
    args = ap.parse_args()

    from acestep.engine.session import Session
    from acestep.fixtures import audio_fixture
    from acestep.paths import checkpoints_dir, trt_engine_path
    from acestep.steering.layout import HOOK_CROSS_ATTN_OUTPUT

    session = Session(project_root=str(checkpoints_dir()), config_path=CHECKPOINT,
                      decoder_backend="eager", vae_backend="eager")
    torch.manual_seed(SEED)
    audio = _load_audio(audio_fixture(FIXTURE))
    source = session.prepare_source(audio)
    cond = session.encode_text(tags=PROMPT, duration=audio.waveform.shape[-1] / 48000)

    H = None
    xnorm = {}

    def probe(pipe):
        nonlocal H
        dec = pipe.decoder
        H = dec.config.hidden_size
        return []

    # Unsteered, plus cross-attn output token norms at the TADA blocks.
    norms: dict = {b: [] for b in TADA_BLOCKS}
    hooks = []
    recs_none = None

    def none_cfg(pipe):
        probe(pipe)
        for b in TADA_BLOCKS:
            hooks.append(pipe.decoder.layers[b].cross_attn.register_forward_hook(
                lambda m, i, o, b=b: norms[b].append(float(o[0].float().norm(dim=-1).mean()))
            ))
        return []

    recs_none = record(session, source, cond, none_cfg)
    for h in hooks:
        h.remove()
    for b in TADA_BLOCKS:
        xnorm[b] = sum(norms[b]) / max(1, len(norms[b]))
    print(f"[eager] forwards={len(recs_none)} hidden={H} "
          f"xattn token norm {', '.join(f'b{b}={xnorm[b]:.2f}' for b in TADA_BLOCKS)}")

    def xattn_cfg(renorm):
        def fn(pipe):
            cfgs = []
            for k, b in enumerate(TADA_BLOCKS):
                a = args.alpha if args.alpha is not None else xnorm[b]
                cfgs.append({
                    "layer": b, "step": -1, "weights": (1.0,) * STEPS,
                    "vector": _unit(100 + k, H).to(pipe._device, torch.float32),
                    "magnitude": 1.0, "alpha": a,
                    "hook": HOOK_CROSS_ATTN_OUTPUT, "renorm": renorm, "cond_only": True,
                })
            return cfgs
        return fn

    def post_cfg(pipe):
        return [{
            "layer": 12, "step": -1, "weights": (1.0,) * STEPS,
            "vector": _unit(200, H).to(pipe._device, torch.float32),
            "magnitude": 1.0, "alpha": 20.0,
        }]

    cases = {
        "none": recs_none,
        "xattn_renorm": record(session, source, cond, xattn_cfg(True)),
        "xattn_plain": record(session, source, cond, xattn_cfg(False)),
        "post_block": record(session, source, cond, post_cfg),
    }

    eng = session.model.handler._diffusion_engine
    report: dict = {"bar": BAR, "xattn_token_norm": xnorm, "cases": {}}

    for engine_name, use_x in (("spectral_decoder_mixed_refit_b8_60s", False),
                               ("tada_decoder_mixed_refit_b8_60s", True)):
        eng.load_trt_engine(str(trt_engine_path(engine_name)))
        for case, recs in cases.items():
            if not use_x and case.startswith("xattn"):
                continue
            outs = replay(eng, recs, use_x)
            cs = [cos(o, r["out"]) for o, r in zip(outs, recs)]
            eff = [
                float((r["out"] - n["out"]).norm() / n["out"].norm())
                for r, n in zip(recs, cases["none"])
            ]
            key = f"{engine_name}:{case}"
            report["cases"][key] = {
                "cos_min": min(cs), "cos_mean": sum(cs) / len(cs),
                "n": len(cs), "pass": min(cs) >= BAR,
                "eager_effect_rel": max(eff),
            }
            print(f"{key:55s} cos min {min(cs):.6f} mean {sum(cs)/len(cs):.6f} "
                  f"effect {max(eff):.3f} {'PASS' if min(cs) >= BAR else 'FAIL'}")
        eng.unload_trt_engine()

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    session.close()
    ok = all(v["pass"] for v in report["cases"].values())
    print("PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

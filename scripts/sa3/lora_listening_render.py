"""Render SA3 LoRA listening WAVs (notes/SA3_LORA_PLAN.md Phase 3 signoff).

Fixed prompt + seed through upstream's own ``StableAudioModel.generate``
(the real SA3 pipeline: T5Gemma conditioning, pingpong sampler, VAE
decode), producing:

* baseline (no LoRA), eager DiT;
* the LoRA at chosen strengths, eager DiT (parametrized weights);
* the LoRA at 1.0 with the DiT routed through the refit-mirrored TRT
  engine — the same upstream sampler loop, ``sam.model.model`` swapped
  for a shim that calls ``SA3TRTDit.step_bundle``.

Duration defaults to 54 s (the refit engine's all-valid L=646 window)
with the saved WAV trimmed to ``--save-seconds``, so the eager and TRT
renders share identical latent geometry and noise draws.

Run:
    .venv/Scripts/python.exe scripts/sa3/lora_listening_render.py \
        --lora <file> --out-dir notes/sa3_lora_listening
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
for _p in (str(_REPO_ROOT),):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import torch  # noqa: E402


class _TRTShim(torch.nn.Module):
    """Routes DiT forwards inside upstream's sampler to the TRT engine.
    Holds the eager module so ``parameters()``-based dtype probes in
    ``generate`` keep working."""

    def __init__(self, trt_dit, eager_dit):
        super().__init__()
        self._trt = trt_dit
        self._eager = eager_dit

    def forward(self, x, t, **kwargs):
        t_val = float(t.flatten()[0].item()) if torch.is_tensor(t) else float(t)
        # Persistent output buffer -> materialize; match eager dtype so
        # sampler arithmetic and randn_like draws stay path-identical.
        return self._trt.step_bundle(x, t_val, kwargs).to(dtype=x.dtype, copy=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--prompt",
                    default="oldschool goa trance, 145 bpm, hypnotic acid leads, "
                            "rolling bassline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--duration", type=float, default=54.0)
    ap.add_argument("--save-seconds", type=float, default=30.0)
    ap.add_argument("--strengths", type=float, nargs="*", default=[0.5, 1.0, 1.5])
    ap.add_argument("--trt-strength", type=float, default=1.0)
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "notes" / "sa3_lora_listening"))
    args = ap.parse_args()

    import soundfile as sf
    import torch.nn.utils.parametrize as parametrize

    from acestep.engine.sa3_context import SA3Context
    from acestep.engine.sa3_lora import SA3LoRAManager

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading SA3Context({args.model!r})...")
    ctx = SA3Context(model_id=args.model)
    sam = ctx.sam
    sr = sam.model.sample_rate
    save_samples = int(args.save_seconds * sr)

    def render(tag: str) -> Path:
        with torch.no_grad(), parametrize.cached():
            audio = sam.generate(
                prompt=args.prompt,
                duration=args.duration,
                steps=args.steps,
                seed=args.seed,
                cfg_scale=1.0,
            )
        wav = audio[0, :, :save_samples].T.float().cpu().numpy()
        path = out_dir / f"{tag}.wav"
        sf.write(str(path), wav, samplerate=sr)
        peak = float(abs(audio).max())
        print(f"[render] {path.name}  peak={peak:.3f}")
        return path

    stem = Path(args.lora).stem

    # --- baseline (no LoRA), eager -------------------------------------
    render("01_baseline_eager")

    # --- eager strengths ----------------------------------------------
    mgr = SA3LoRAManager(
        model_root=sam.model.model,
        conditioner_root=sam.model.conditioner,
        checkpoint_id=ctx.model_id,
    )
    lid = mgr.register_lora(args.lora)
    mgr.prewarm_lora(lid).result(timeout=180)
    first = True
    for i, s in enumerate(args.strengths):
        if first:
            mgr.enable_lora(lid, strength=s)
            first = False
        else:
            mgr.set_lora_strength(lid, s)
        render(f"{i + 2:02d}_{stem}_s{s:g}_eager")

    # --- TRT refit render ----------------------------------------------
    from acestep.engine.sa3_trt import SA3TRTDit, find_dit_engine
    from acestep.engine.sa3_trt_lora import SA3TRTRefitMirror, find_refit_manifest

    # Latent geometry of the padded window generate() will use.
    cond = ctx.prepare_cond(prompt=args.prompt, duration=args.duration,
                            steps=args.steps)
    L = cond.latent_frames
    engine_path = find_dit_engine(args.model, L, want_refittable=True)
    if engine_path is None:
        raise RuntimeError(f"no refittable TRT DiT engine for L={L}")
    trt_dit = SA3TRTDit(engine_path, latent_frames=L, seconds_total=args.duration)
    manifest = find_refit_manifest(engine_path)
    if manifest is None:
        raise RuntimeError("no refit manifest; run gen_sa3_refit_manifest.py")
    mirror = SA3TRTRefitMirror(trt_dit.engine, sam.model.model, manifest)

    mgr.set_lora_strength(lid, args.trt_strength)
    mirror.sync(reason="listening-render")
    real_dit = sam.model.model
    sam.model.model = _TRTShim(trt_dit, real_dit)
    try:
        render(f"{len(args.strengths) + 2:02d}_{stem}_s{args.trt_strength:g}_trt_refit")
    finally:
        sam.model.model = real_dit

    mgr.close()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

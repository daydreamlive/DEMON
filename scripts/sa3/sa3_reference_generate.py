"""SA3 offline reference generation (Phase 0 spike).

Loads the `small-music` checkpoint ENTIRELY from the project model dir
(`<MODELS_DIR>/sa3/checkpoints/stable-audio-3-small-music/`, resolved via
`acestep.paths`) and produces a one-shot text->audio reference output by
replicating `StableAudioModel.from_pretrained(...).generate(...)` with
explicit local paths.

Why not call `StableAudioModel.from_pretrained("small-music")` directly:
its `ModelConfig.resolve()` re-fetches via `hf_hub_download(repo_id, ...)`
into the HF cache, ignoring our project model dir. We instead load the
config/ckpt by path and patch the t5gemma conditioner to load the BUNDLED
encoder (`<DEST>/t5gemma-b-b-ul2/`) instead of the gated `google/...` repo.

This is the reference output the streaming fork (Phase 2) must match.

Run:
    .venv/Scripts/python.exe scripts/sa3/sa3_reference_generate.py \
        --prompt "warm analog house groove, 124 bpm" --duration 10 --steps 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# --- sys.path: repo root FIRST (so `acestep` is ours, not a sibling shadow).
#     The managed SA3 vendor path is added through acestep.engine.sa3_helpers.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
# Force repo root to the FRONT. A sibling ACE-Step editable install
# (_editable_impl_ace_step.pth -> C:/_dev/projects/ACE-Step-1.5_alt) injects
# its own `acestep`; our repo path is also injected but lands after it, so an
# "insert only if absent" leaves the sibling shadowing our edited acestep.
# Remove-then-insert guarantees ours wins.
for _p in (str(_REPO_ROOT),):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from acestep import paths  # noqa: E402
from acestep.engine.sa3_helpers import (  # noqa: E402
    ensure_sa3_paths,
    require_sa3_vendor,
    sa3_checkpoint_dir,
)

ensure_sa3_paths()


def checkpoint_dir(model_name: str = "small-music") -> Path:
    """Return the local checkpoint dir for a given SA3 model name.

    Defaults to ``small-music``; pass ``"medium"`` (or any other registry
    name) to point at a different checkpoint. Single-sourced from
    :func:`acestep.engine.sa3_helpers.sa3_checkpoint_dir`.
    """
    return sa3_checkpoint_dir(model_name)


def spike_out_dir() -> Path:
    """Where spike scripts write generated audio. Lives under MODELS_DIR
    (outside the repo tree), so generated WAVs are never tracked."""
    d = paths.models_dir() / "sa3" / "spike_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_local_model(dest: Path, device: str, model_half: bool):
    """Replicate StableAudioModel.from_pretrained but from local paths,
    pointing the t5gemma conditioner at the bundled encoder subfolder."""
    require_sa3_vendor()
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel

    config_path = dest / "model_config.json"
    ckpt_path = dest / "model.safetensors"
    if not config_path.is_file() or not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint files under {dest} "
            f"(need model_config.json + model.safetensors)."
        )

    model_config = json.loads(config_path.read_text(encoding="utf-8"))

    # Patch the t5gemma conditioner to load the BUNDLED encoder locally.
    # T5GemmaConditioner resolves load_from = model_path or repo_id or model_name,
    # and still honors `subfolder`, so model_path=<DEST> + subfolder=t5gemma-b-b-ul2
    # resolves to <DEST>/t5gemma-b-b-ul2/ on disk. We drop repo_id so a stale
    # network repo can never win over the local copy.
    patched = False
    for c in model_config["model"]["conditioning"]["configs"]:
        if c.get("type") == "t5gemma":
            c["config"]["model_path"] = str(dest)
            c["config"].pop("repo_id", None)
            patched = True
    if not patched:
        raise RuntimeError("No t5gemma conditioner found in model_config to patch.")

    model = load_diffusion_cond(
        model_config, str(ckpt_path), device=device, model_half=model_half
    )
    model.use_lora = False
    model.lora_names = []
    return StableAudioModel(model, model_config, device, model_half)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="warm analog house groove, 124 bpm, deep bassline")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if omitted)")
    ap.add_argument(
        "--out",
        default=None,
        help="output wav path (default: <MODELS_DIR>/sa3/spike_out/<seed>_<dur>s.wav)",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="set HF_HUB_OFFLINE=1 to PROVE the load is fully local (no network).",
    )
    args = ap.parse_args()

    if args.offline:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_half = device == "cuda"

    dest = checkpoint_dir()
    print(f"[load] checkpoint dir: {dest}")
    print(f"[load] device={device} model_half={model_half} offline={args.offline}")

    t0 = time.time()
    sam = load_local_model(dest, device=device, model_half=model_half)
    t_load = time.time() - t0

    sr = sam.model.sample_rate
    ds = sam.model.pretransform.downsampling_ratio
    print(f"[model] sample_rate={sr} downsampling_ratio={ds} "
          f"latent_rate={sr / ds:.4f} Hz io_channels={sam.model.io_channels} "
          f"objective={sam.model.diffusion_objective}")
    print(f"[model] loaded in {t_load:.1f}s")

    t0 = time.time()
    audio = sam.generate(
        prompt=args.prompt,
        duration=args.duration,
        steps=args.steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
    )
    t_gen = time.time() - t0

    # audio: [B, 2, samples] fp32 in [-1, 1]
    print(f"[gen] audio shape={tuple(audio.shape)} dtype={audio.dtype} "
          f"peak={audio.abs().max().item():.3f} in {t_gen:.1f}s "
          f"({args.duration / max(t_gen, 1e-6):.2f}x realtime)")

    out = Path(args.out) if args.out else (
        spike_out_dir() / f"seed{args.seed}_{int(args.duration)}s.wav"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # soundfile rather than torchaudio.save: torchaudio's default backend is now
    # torchcodec, which needs FFmpeg shared libs not installed on this Windows box.
    # soundfile (libsndfile) is already a DEMON dep and wants [frames, channels].
    import soundfile as sf

    sf.write(str(out), audio[0].T.cpu().numpy(), samplerate=sr)
    print(f"[save] {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the ACE-Step v1.5 decoder engine with TADA's steering site (GPU).

TADA (Staniszewski et al., arXiv 2602.11910) steers the OUTPUT of a
block's cross-attention. The production ACE decoder engine
(``spectral_decoder_mixed_refit_b8_60s``) only carries the post-block
``steering`` input, so this exports the same bf16-hybrid refit ONNX with
two more inputs (``steering_xattn`` ``[B, 24, 2048]``,
``steering_xattn_renorm`` ``[B, 24]``; see
``acestep.engine.trt.export._patch_cross_attn_steering``) and builds it
with the production 60 s profile under a new name
(``tada_decoder_mixed_refit_b8_60s``). The previous engine is left in
place. Needs an idle GPU (an engine built under contention can segfault
on load).

Run (PYTHONUTF8=1 on Windows)::

    python scripts/tada/ace_build_xattn_engine.py [--skip-export] [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
while str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

CHECKPOINT = "acestep-v15-turbo"
DURATION_S = 60


def onnx_path() -> Path:
    from acestep.paths import trt_engines_dir

    return trt_engines_dir() / "_onnx_tada" / "decoder_refit" / "decoder_refit.onnx"


def export(device: str) -> Path:
    import torch

    from acestep.engine.model_context import ModelContext
    from acestep.engine.trt.export import OnnxExportConfig, export_decoder_onnx

    path = onnx_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] loading {CHECKPOINT} ...", flush=True)
    handler = ModelContext(
        config_path=CHECKPOINT,
        device=device,
        use_flash_attention=False,
        compile_decoder=False,
        compile_vae=False,
        skip_vae=True,
    )
    t0 = time.time()
    with handler._load_model_context("model"):
        export_decoder_onnx(
            handler.model, path, device=device,
            config=OnnxExportConfig(
                mixed_precision=True, for_refit=True, xattn_steering=True,
            ),
        )
    print(f"[export] {path} in {time.time() - t0:.0f}s", flush=True)
    del handler
    torch.cuda.empty_cache()
    return path


def build(onnx: Path, force: bool) -> Path:
    from acestep.engine.trt.build import _preflight
    from acestep.engine.trt.export import TRTBuildConfig, build_trt_engine
    from acestep.engine.trt._engine_metadata import expected_metadata, write_metadata
    from acestep.paths import trt_engines_dir

    env = _preflight("cuda")
    config = TRTBuildConfig(
        fp16=True,
        strongly_typed=True,
        refit=True,
        workspace_gb=16.0,
        batch_max=8,
        seq_opt=min(DURATION_S * 25, 1500),
        seq_max=DURATION_S * 25,
        variant="turbo",
        onnx_precision="fp16_mixed",
        xattn_steering=True,
    )
    name = config.engine_filename().replace(".engine", "")
    assert name == f"tada_decoder_mixed_refit_b8_{DURATION_S}s", name
    engine = trt_engines_dir() / name / f"{name}.engine"
    if engine.exists() and not force:
        print(f"[build] exists, skipping: {engine}")
        return engine
    engine.parent.mkdir(parents=True, exist_ok=True)
    meta = expected_metadata(component="decoder_refit", onnx_path=onnx, config=config, env=env)
    t0 = time.time()
    build_trt_engine(onnx, engine, config=config)
    write_metadata(engine_path=str(engine), expected=meta, env=env)
    manifest = Path(str(onnx) + ".refit_manifest.json")
    if manifest.is_file():
        Path(str(engine) + ".refit_manifest.json").write_text(
            manifest.read_text(encoding="utf-8"), encoding="utf-8",
        )
    print(f"[build] {engine} in {time.time() - t0:.0f}s", flush=True)
    return engine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild an existing engine")
    args = ap.parse_args()
    onnx = onnx_path() if args.skip_export else export("cuda")
    if not onnx.exists():
        print(f"missing ONNX {onnx}")
        return 2
    build(onnx, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate XL turbo TRT decoder vs PyTorch decoder.

Loads the model, runs both PT and TRT on the same input, reports accuracy.
"""

import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
torch.set_grad_enabled(False)

from acestep.engine.model_context import ModelContext
from acestep.engine.trt.export import validate_trt_vs_pytorch
from acestep.paths import checkpoints_dir, trt_engine_path

ENGINE_NAME = os.environ.get("ENGINE_NAME", "decoder_xl-turbo_bf16_b1_60s")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "acestep-v15-xl-turbo")


def main():
    engine = trt_engine_path(ENGINE_NAME)
    print(f"engine: {engine}")
    print(f"checkpoint: {CONFIG_PATH}")

    ctx = ModelContext(
        project_root=str(checkpoints_dir()),
        config_path=CONFIG_PATH,
        device="cuda",
        use_flash_attention=False,
        skip_vae=True,
    )

    print("\n--- 30s window (T=750, L=200) ---")
    r = validate_trt_vs_pytorch(
        ctx.model, engine,
        device="cuda", dtype=torch.bfloat16,
        seq_len=750, enc_len=200, seed=42,
    )
    print()
    for k, v in r.items():
        print(f"  {k:<20s} {v:.6f}")

    print("\n--- 60s window (T=1500, L=200) ---")
    r2 = validate_trt_vs_pytorch(
        ctx.model, engine,
        device="cuda", dtype=torch.bfloat16,
        seq_len=1500, enc_len=200, seed=42,
    )
    print()
    for k, v in r2.items():
        print(f"  {k:<20s} {v:.6f}")


if __name__ == "__main__":
    main()

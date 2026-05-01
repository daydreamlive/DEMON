"""Find which op produces a complex tensor when tracing the XL turbo decoder in bf16.

Strategy: install a torch._C._jit_set_logging_options-style hook? No, simpler:
patch torch.Tensor.__torch_function__ via a tracer that snapshots dtypes per op.

We just call the wrapper's forward in bf16 (no trace), and walk the graph
afterward inspecting the produced graph nodes for complex outputs. Or even
simpler: run forward with hooks on every nn.Module, inspect every output dtype.
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
from acestep.engine.trt.export import DecoderForExport
from acestep.paths import checkpoints_dir


def main():
    ctx = ModelContext(
        project_root=str(checkpoints_dir()),
        config_path="acestep-v15-xl-turbo",
        device="cuda",
        use_flash_attention=False,
        skip_vae=True,
    )

    wrapper = DecoderForExport(ctx.model.decoder, mixed_precision=False, precision="bf16").eval()
    wrapper = wrapper.to("cuda")

    # Hook every leaf module
    complex_modules = []
    def hook_fn(name):
        def hook(_mod, _inp, out):
            t = out
            if isinstance(t, tuple):
                t = t[0]
            if isinstance(t, torch.Tensor):
                if t.is_complex():
                    complex_modules.append((name, str(t.dtype), tuple(t.shape)))
                    print(f"!! COMPLEX: {name}  dtype={t.dtype}  shape={tuple(t.shape)}", flush=True)
        return hook

    handles = []
    for name, mod in wrapper.decoder.named_modules():
        if name == "":
            continue
        handles.append(mod.register_forward_hook(hook_fn(name)))

    torch.manual_seed(42)
    B, T, L = 1, 750, 200
    inputs = dict(
        hidden_states=torch.randn(B, T, 64, device="cuda", dtype=torch.bfloat16),
        timestep=torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(B, L, 2048, device="cuda", dtype=torch.bfloat16),
        context_latents=torch.randn(B, T, 128, device="cuda", dtype=torch.bfloat16),
    )

    print("--- forward pass with bf16 inputs (no trace) ---")
    out = wrapper(**inputs)
    print(f"forward output: dtype={out.dtype}  shape={tuple(out.shape)}  is_complex={out.is_complex()}")

    for h in handles:
        h.remove()

    if complex_modules:
        print(f"\nFound {len(complex_modules)} modules producing complex outputs:")
        for n, dt, sh in complex_modules:
            print(f"  {n}: {dt} {sh}")
    else:
        print("\nNo modules produced complex outputs at the leaf level.")
        print("The complex tensor must be created inside a module's internal ops.")

    # Now actually trace and catch the error
    print("\n--- attempting torch.onnx.export trace in bf16 ---")
    import traceback as tb
    try:
        torch.onnx.export(
            wrapper,
            (
                inputs["hidden_states"],
                inputs["timestep"],
                inputs["encoder_hidden_states"],
                inputs["context_latents"],
            ),
            "/tmp/_debug_bf16.onnx",
            input_names=["hidden_states", "timestep", "encoder_hidden_states", "context_latents"],
            output_names=["velocity"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        print("trace succeeded (no complex error)")
    except Exception as e:
        print(f"trace failed: {type(e).__name__}: {e}")
        tb.print_exc()


if __name__ == "__main__":
    main()

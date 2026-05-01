"""Investigate XL turbo decoder fp16 behavior.

Three probes:
  A. Weight range scan -- any parameter that overflows on .half()?
  B. Activation probe (bf16 ground truth) -- value range per layer.
  C. fp16 mixed-precision probe -- find first layer where outputs go bad.

Goal: understand where the existing mixed-precision recipe breaks for XL
so we can design a recipe based on evidence, not the 2B pattern.
"""

import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import copy
import math

import torch
import torch.nn as nn
torch.set_grad_enabled(False)

from acestep.engine.model_context import ModelContext
from acestep.engine.trt.export import DecoderForExport
from acestep.paths import checkpoints_dir

CONFIG_PATH = os.environ.get("PROBE_CKPT", "acestep-v15-xl-turbo")

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05  # smallest positive fp16 normal
SEQ_LEN = 750
ENC_LEN = 200


def _stat(t: torch.Tensor) -> dict:
    """Return value statistics for a tensor (cast to fp32 for safety)."""
    f = t.detach().float()
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "min":   f.min().item() if f.numel() else 0.0,
        "max":   f.max().item() if f.numel() else 0.0,
        "absmax": f.abs().max().item() if f.numel() else 0.0,
        "mean":  f.mean().item() if f.numel() else 0.0,
        "std":   f.std().item() if f.numel() > 1 else 0.0,
        "nan":   torch.isnan(f).any().item(),
        "inf":   torch.isinf(f).any().item(),
    }


# ---------------------------------------------------------------------------
# Probe A: weight range scan
# ---------------------------------------------------------------------------

def probe_weights(model) -> None:
    print("\n" + "=" * 78)
    print("PROBE A: weight range scan (looking for fp16 overflow on .half())")
    print("=" * 78)

    decoder = model.decoder
    overflowing = []
    near_overflow = []
    all_max = []

    for name, p in decoder.named_parameters():
        amax = p.detach().abs().max().item()
        all_max.append((name, amax, tuple(p.shape), str(p.dtype)))
        if amax > FP16_MAX:
            overflowing.append((name, amax, tuple(p.shape)))
        elif amax > FP16_MAX * 0.5:
            near_overflow.append((name, amax, tuple(p.shape)))

    all_max.sort(key=lambda x: -x[1])
    print(f"\nTotal decoder params scanned: {len(all_max)}")
    print(f"Top 15 by abs-max:")
    for name, amax, shape, dtype in all_max[:15]:
        flag = ""
        if amax > FP16_MAX:
            flag = "  <-- OVERFLOWS fp16"
        elif amax > FP16_MAX * 0.5:
            flag = "  <-- near fp16 max"
        print(f"  {amax:>10.4f}  {dtype:<14s} {str(shape):<30s} {name}{flag}")

    if overflowing:
        print(f"\n!! {len(overflowing)} parameter(s) OVERFLOW fp16:")
        for name, amax, shape in overflowing:
            print(f"   {name}  shape={shape}  abs_max={amax:.4f}")
    else:
        print("\nOK: no decoder parameter overflows fp16 directly.")

    if near_overflow:
        print(f"\n{len(near_overflow)} parameter(s) above fp16_max/2 (close call):")
        for name, amax, shape in near_overflow:
            print(f"   {name}  shape={shape}  abs_max={amax:.4f}")


# ---------------------------------------------------------------------------
# Probe B / C: activation probe
# ---------------------------------------------------------------------------

class ForwardProbe:
    """Wraps the model with hooks that record per-module output statistics."""

    def __init__(self, decoder: nn.Module):
        self.records: list[tuple[str, dict]] = []
        self._hooks = []
        self._first_bad: tuple[str, dict] | None = None

        # Hook on every nn.Module that has at least one parameter or known output
        for name, mod in decoder.named_modules():
            if name == "":
                continue
            # Skip container-only modules; hook the leaves and the per-layer wrappers
            if not list(mod.children()) or "layers." in name:
                self._hooks.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_mod, _inp, out):
            t = out
            if isinstance(t, tuple):
                t = t[0]
            if not isinstance(t, torch.Tensor):
                return
            s = _stat(t)
            self.records.append((name, s))
            if (s["nan"] or s["inf"]) and self._first_bad is None:
                self._first_bad = (name, s)
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def _build_inputs(device, dtype, ts_dtype=None):
    torch.manual_seed(42)
    if ts_dtype is None:
        ts_dtype = torch.float32 if dtype == torch.float16 else dtype
    return dict(
        hidden_states=torch.randn(1, SEQ_LEN, 64, device=device, dtype=dtype),
        timestep=torch.full((1,), 0.5, device=device, dtype=ts_dtype),
        encoder_hidden_states=torch.randn(1, ENC_LEN, 2048, device=device, dtype=dtype),
        context_latents=torch.randn(1, SEQ_LEN, 128, device=device, dtype=dtype),
    )


def _call_wrapper(wrapper, inputs):
    return wrapper(
        hidden_states=inputs["hidden_states"],
        timestep=inputs["timestep"],
        encoder_hidden_states=inputs["encoder_hidden_states"],
        context_latents=inputs["context_latents"],
    )


def probe_activations(model, label: str, mixed_precision: bool, dtype: torch.dtype):
    print("\n" + "=" * 78)
    print(f"PROBE: {label}  (mixed_precision={mixed_precision} input_dtype={dtype})")
    print("=" * 78)

    # Deep-copy the decoder so the wrapper's mutations (Lambda replace,
    # forward patch, dtype changes) don't pollute later probes.
    dec_copy = copy.deepcopy(model.decoder)
    fake_model = type("M", (), {"decoder": dec_copy})()

    wrapper = DecoderForExport(fake_model.decoder, mixed_precision=mixed_precision).eval()
    if not mixed_precision:
        wrapper = wrapper.to(dec_copy.parameters().__next__().device)

    probe = ForwardProbe(wrapper.decoder)
    inputs = _build_inputs(dec_copy.parameters().__next__().device, dtype)

    out = _call_wrapper(wrapper, inputs)
    probe.remove()

    print(f"output stats: {_stat(out)}")
    if probe._first_bad is not None:
        bad_name, bad_stat = probe._first_bad
        print(f"\n!! FIRST NaN/Inf in: {bad_name}")
        print(f"   {bad_stat}")
        # Print the surrounding 6 records to see the upstream values
        idx = next(i for i, (n, _) in enumerate(probe.records) if n == bad_name)
        lo = max(0, idx - 5)
        print(f"\n   Last {idx - lo + 1} records before/at the failure:")
        for n, s in probe.records[lo: idx + 1]:
            flag = "  <<< BAD" if (s["nan"] or s["inf"]) else ""
            print(f"     [{n}] absmax={s['absmax']:.3e}  mean={s['mean']:.3e}  "
                  f"std={s['std']:.3e}{flag}")
    else:
        print("\nOK: no NaN/Inf in any layer output.")
        # Show top-10 absmax records to see if anything is dangerously large
        top = sorted(probe.records, key=lambda r: -r[1]["absmax"])[:10]
        print("\n  top 10 layers by absmax:")
        for n, s in top:
            print(f"     [{n}] absmax={s['absmax']:.3e}")

    del wrapper, dec_copy, fake_model, probe
    torch.cuda.empty_cache()


def quick_qnorm_summary(model):
    """One-line summary: max abs of q_norm/k_norm weights across all layers."""
    print("\nq_norm / k_norm weight magnitudes (per layer, self_attn AND cross_attn):")
    print(f"  {'layer':>5s}  {'self q':>10s} {'self k':>10s}   {'cross q':>10s} {'cross k':>10s}")
    for i, layer in enumerate(model.decoder.layers):
        sq = layer.self_attn.q_norm.weight.detach().abs().max().item()
        sk = layer.self_attn.k_norm.weight.detach().abs().max().item()
        cq = ck = float("nan")
        if hasattr(layer, "cross_attn"):
            cq = layer.cross_attn.q_norm.weight.detach().abs().max().item()
            ck = layer.cross_attn.k_norm.weight.detach().abs().max().item()
        flag = ""
        if max(sq, sk, cq if cq == cq else 0, ck if ck == ck else 0) > 5:
            flag = "  <-- outlier"
        print(f"  {i:>5d}  {sq:>10.4f} {sk:>10.4f}   {cq:>10.4f} {ck:>10.4f}{flag}")


def main():
    print(f"loading {CONFIG_PATH} ...")
    ctx = ModelContext(
        project_root=str(checkpoints_dir()),
        config_path=CONFIG_PATH,
        device="cuda",
        use_flash_attention=False,
        skip_vae=True,
    )
    print("loaded.")
    print(f"native dtype: {next(ctx.model.parameters()).dtype}")

    # Probe A: weight range
    probe_weights(ctx.model)

    # Per-layer q_norm / k_norm summary
    quick_qnorm_summary(ctx.model)

    # Probe B: bf16 ground truth (no precision change)
    probe_activations(
        ctx.model,
        label="bf16 ground-truth",
        mixed_precision=False,
        dtype=torch.bfloat16,
    )

    # Probe C: fp16 mixed precision (current export recipe)
    probe_activations(
        ctx.model,
        label="fp16 mixed (current export recipe)",
        mixed_precision=True,
        dtype=torch.float16,
    )


if __name__ == "__main__":
    main()

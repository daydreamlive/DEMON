"""Map a ComfyUI-layout MiniMax-Music3 LoRA onto our DiT.

Community LoRAs for this model — step-distillation "turbo" adapters
included — ship in the ComfyUI/native weight layout, which differs from
the diffusers layout this tree reimplements in one structural way: the
native checkpoint fuses attention projections into a single
``self_attn.to_qkv`` of shape ``[3*dim, dim]``, where we keep separate
``to_q`` / ``to_k`` / ``to_v``. Everything else lines up name for name.

**The packing order of that fused matrix is verified, not assumed.** It
is contiguous ``[q; k; v]``. Measured by fetching layer 0's
``to_qkv.weight`` out of ``Comfy-Org/MiniMax-Music-3`` and comparing it
against the diffusers checkpoint's three separate matrices:

    contiguous [q;k;v]              max|diff|  ~1.9e-3   (fp16 rounding)
    head-interleaved [32,3,64,dim]  max|diff|  ~6.4e-1

Two orders of magnitude apart, so there is no ambiguity. This matters
more than it looks: a head-interleaved split would still produce
plausible music, just subtly wrong music, and nothing downstream would
raise. It is exactly the class of error this model punishes quietly.

The mapping, per transformer block ``i``:

===========================================  =============================
native                                       ours
===========================================  =============================
``layers.i.self_attn.to_qkv``                ``transformer_blocks.i.attn.to_{q,k,v}``
``layers.i.self_attn.to_out``                ``transformer_blocks.i.attn.to_out``
``layers.i.ff.ff.0.proj``                    ``transformer_blocks.i.ff_in``
``layers.i.ff.ff.2``                         ``transformer_blocks.i.ff_out``
===========================================  =============================

This module only produces weight DELTAS. It takes no view on whether a
caller merges them, keeps them as an adapter, or refits a TensorRT
engine with them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import torch

#: Every native key starts with this.
NATIVE_PREFIX = "diffusion_transformer.transformer.layers."

#: native leaf -> (our leaf, number of contiguous row-blocks it fuses)
_TARGETS = {
    "self_attn.to_qkv": (("attn.to_q", "attn.to_k", "attn.to_v"), 3),
    "self_attn.to_out": (("attn.to_out",), 1),
    "ff.ff.0.proj": (("ff_in",), 1),
    "ff.ff.2": (("ff_out",), 1),
}

_KEY = re.compile(
    re.escape(NATIVE_PREFIX) + r"(\d+)\.(.+?)\.(lora_down|lora_up|alpha)"
    r"(?:\.weight)?$"
)


class MiniMaxLoRAError(ValueError):
    """A LoRA that cannot be mapped, rather than one silently half-applied."""


def load_native_lora(
    path: "str | Path", *, strength: float = 1.0, dtype=torch.float32,
    allow_noop: bool = False,
) -> Dict[str, torch.Tensor]:
    """``path`` -> ``{our_state_dict_key: delta_weight}``.

    ``strength`` scales every delta linearly.

    Raises rather than skipping on anything unrecognized. A LoRA that
    applies to 143 of 144 projections is not a slightly weaker LoRA, it
    is a differently-broken model, and the failure would be inaudible
    until it was expensive.

    Also raises when every delta comes out zero, unless ``allow_noop``.
    That is not a hypothetical: the community "8-step turbo" adapter for
    this model (``guillaume127/MiniMax-Music-3-Turbo-FP8``, and the
    byte-identical ``modulsx`` duplicate, sha256 ``eed5f8c9...``) ships
    with all 144 ``lora_up`` tensors exactly zero and ``lora_down`` still
    at Kaiming init. ``B @ A`` with ``B = 0`` is zero at every strength,
    so it changes nothing while looking, loading, and merging perfectly.
    An adapter that silently does nothing is worse than one that fails,
    because the surrounding benchmark will still move for other reasons
    and get credited to it.
    """
    from safetensors.torch import load_file

    raw = load_file(str(path))
    parts: Dict[tuple, Dict[str, torch.Tensor]] = {}
    for key, value in raw.items():
        m = _KEY.match(key)
        if m is None:
            raise MiniMaxLoRAError(
                f"unrecognized LoRA key {key!r}. Expected the ComfyUI "
                f"layout, {NATIVE_PREFIX}<block>.<target>.<lora_up|"
                "lora_down|alpha>"
            )
        block, target, kind = int(m.group(1)), m.group(2), m.group(3)
        if target not in _TARGETS:
            raise MiniMaxLoRAError(
                f"LoRA targets {target!r} on block {block}, which this "
                f"model has no counterpart for. Known: {sorted(_TARGETS)}"
            )
        parts.setdefault((block, target), {})[kind] = value

    out: Dict[str, torch.Tensor] = {}
    for (block, target), tensors in sorted(parts.items()):
        missing = {"lora_up", "lora_down"} - set(tensors)
        if missing:
            raise MiniMaxLoRAError(
                f"block {block} target {target!r} is missing {sorted(missing)}"
            )
        up = tensors["lora_up"].to(dtype)
        down = tensors["lora_down"].to(dtype)
        rank = down.shape[0]
        if up.shape[1] != rank:
            raise MiniMaxLoRAError(
                f"block {block} {target!r}: lora_up has inner dim "
                f"{up.shape[1]} but lora_down has rank {rank}"
            )
        # Standard LoRA scaling. ``alpha`` is stored per-target as a
        # 0-dim tensor; absent it, convention is alpha == rank (scale 1).
        alpha = tensors.get("alpha")
        scale = (float(alpha) / rank) if alpha is not None else 1.0
        delta = (up @ down) * (scale * float(strength))

        names, chunks = _TARGETS[target]
        if delta.shape[0] % chunks:
            raise MiniMaxLoRAError(
                f"block {block} {target!r}: {delta.shape[0]} rows do not "
                f"divide into {chunks} projections"
            )
        step = delta.shape[0] // chunks
        for j, leaf in enumerate(names):
            # Contiguous [q; k; v] -- see the module docstring; this is
            # the line a head-interleaved checkpoint would silently break.
            out[f"transformer_blocks.{block}.{leaf}.weight"] = (
                delta[j * step:(j + 1) * step].contiguous()
            )

    if not allow_noop and out and not any(v.any() for v in out.values()):
        zeros = sum(1 for k, v in raw.items() if "lora_up" in k and not v.any())
        total = sum(1 for k in raw if "lora_up" in k)
        raise MiniMaxLoRAError(
            f"{Path(path).name} is a no-op: every one of its {len(out)} "
            f"weight deltas is exactly zero ({zeros}/{total} lora_up "
            "tensors are all-zero, i.e. still at initialization). This "
            "adapter was published untrained. Pass allow_noop=True only "
            "if you are deliberately measuring a null control."
        )
    return out


def apply_native_lora(
    model: torch.nn.Module, path: "str | Path", *, strength: float = 1.0,
) -> int:
    """Merge a native-layout LoRA into ``model`` in place.

    Returns the number of parameters modified, so a caller can assert on
    it instead of trusting that anything happened at all.
    """
    deltas = load_native_lora(path, strength=strength)
    params = dict(model.named_parameters())
    unknown = sorted(set(deltas) - set(params))
    if unknown:
        raise MiniMaxLoRAError(
            f"LoRA maps to {len(unknown)} parameters this model does not "
            f"have, e.g. {unknown[:3]}"
        )
    with torch.no_grad():
        for name, delta in deltas.items():
            p = params[name]
            if p.shape != delta.shape:
                raise MiniMaxLoRAError(
                    f"{name}: model has {tuple(p.shape)}, LoRA delta is "
                    f"{tuple(delta.shape)}"
                )
            p.add_(delta.to(device=p.device, dtype=p.dtype))
    return len(deltas)

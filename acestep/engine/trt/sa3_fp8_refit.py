"""Refit manifest for Stability's SA3 medium fp8 DiT graph (fp8 refit Q2a).

Stability's ``onnx/sa3-m/dit_fp8.onnx`` (what ``sa3_build --fp8`` compiles)
keeps every quantized Linear weight as an **fp32 initializer** feeding
``QuantizeLinear(scale, zero_point, axis=0) -> DequantizeLinear ->
Transpose -> MatMul`` (read 2026-10-12: 222 sites, 220 per-output-channel
scales, 2 per-tensor). The fp32 initializers are bit-equal to the stock
medium checkpoint's fp32 weights; the baked scales are amax(W)/448 to ~1e-6.

A refit-built engine of that graph (``sa3_build --fp8-refit``: the graph
with fp16 attention cores, ``BuilderFlag.REFIT_INDIVIDUAL`` marking only
these 222 weights) exposes each weight as an **fp32 slot**: the mirror
pushes fp32 W' = W + dW and TensorRT re-quantizes it with the baked scale
at refit (TRT 10.16, :func:`refit_inventory`: 222 refittable, prototype
FLOAT). The weight and activation scales are build-time constants, fixed
by design: a LoRA delta whose amax exceeds scale * 448 saturates instead
of rescaling (``scripts/sa3/sa3_fp8_lora_saturation.py`` measures it).
The manifest records, per mapped module:

* ``initializer``: ONNX weight initializer name (the refit name);
* ``transposed``: whether the initializer is stored ``[in, out]``;
* ``scale_init`` / ``axis`` / ``per_channel``: the baked weight scale
  (values in the ``.scales.safetensors`` sidecar, keyed by initializer
  name; activation scale values live there too so a refit that reports
  them missing can re-submit them unchanged).

The base the mirror composes on is the checkpoint fp32 weight (verified
bit-equal to the ONNX initializer at manifest time), never the fp16 eager
weight: a base-value refit must reproduce the build-time quantization
bit for bit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch

FP8_MANIFEST_VERSION = 2
E4M3_MAX = 448.0
SCALES_SUFFIX = ".scales.safetensors"


@dataclass
class FP8WeightSite:
    fqn: str
    initializer: str
    scale_init: str
    zero_point: Optional[str]
    axis: Optional[int]
    per_channel: bool
    transposed: bool
    shape: list


class OnnxInitReader:
    """Reads named initializers (inline or external data) without loading
    the whole multi-GB sidecar."""

    def __init__(self, model, onnx_path: Path):
        self._dir = Path(onnx_path).parent
        self._by_name = {i.name: i for i in model.graph.initializer}
        self._maps: Dict[str, np.memmap] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> torch.Tensor:
        import onnx
        from onnx import numpy_helper

        init = self._by_name[name]
        if init.data_location != onnx.TensorProto.EXTERNAL:
            return torch.from_numpy(numpy_helper.to_array(init).copy()).float()
        if init.data_type != onnx.TensorProto.FLOAT:
            raise TypeError(f"{name}: expected an fp32 external initializer")
        ext = {e.key: e.value for e in init.external_data}
        loc = ext["location"]
        mm = self._maps.get(loc)
        if mm is None:
            mm = np.memmap(self._dir / loc, dtype=np.uint8, mode="r")
            self._maps[loc] = mm
        off, length = int(ext.get("offset", 0)), int(ext["length"])
        arr = np.frombuffer(mm[off:off + length].tobytes(), dtype=np.float32)
        return torch.from_numpy(arr.copy()).reshape(tuple(init.dims))


def _consumers(model) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for n in model.graph.node:
        for i in n.input:
            out.setdefault(i, []).append(n)
    return out


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return list(a.ints)
            return a.i
    return default


def _stored_transposed(dq_out: str, cons: Dict[str, list], dims) -> bool:
    """Orientation of the initializer relative to torch's ``[out, in]``,
    from how the dequantized weight reaches its GEMM."""
    users = cons.get(dq_out, [])
    if len(users) != 1:
        raise RuntimeError(f"{dq_out}: expected one consumer, got {len(users)}")
    u = users[0]
    if u.op_type == "Transpose":
        perm = _attr(u, "perm")
        if perm not in (None, [1, 0]):
            raise RuntimeError(f"{u.name}: unexpected Transpose perm {perm}")
        return False          # [out, in] -> Transpose -> MatMul(x, W^T)
    if u.op_type == "MatMul":
        return True           # [in, out] fed straight to MatMul
    if u.op_type == "Gemm":
        return not bool(_attr(u, "transB", 0))
    raise RuntimeError(f"{u.name}: unsupported weight consumer {u.op_type}")


def fp8_weight_sites(
    model, *, graph_prefix: str = "dit.", module_prefix: str = "model.",
) -> list:
    """Every QuantizeLinear whose input is an initializer (the weight Q)."""
    inits = {i.name: i for i in model.graph.initializer}
    cons = _consumers(model)
    sites = []
    for n in model.graph.node:
        if n.op_type != "QuantizeLinear" or n.input[0] not in inits:
            continue
        w = inits[n.input[0]]
        name = w.name
        if not (name.startswith(graph_prefix) and name.endswith(".weight")):
            raise RuntimeError(f"weight initializer {name!r} outside {graph_prefix!r}*.weight")
        dqs = [c for c in cons.get(n.output[0], []) if c.op_type == "DequantizeLinear"]
        if len(dqs) != 1:
            raise RuntimeError(f"{n.name}: expected one DequantizeLinear")
        scale = inits.get(n.input[1])
        if scale is None:
            raise RuntimeError(f"{n.name}: weight scale {n.input[1]!r} is not an initializer")
        per_channel = int(np.prod(scale.dims)) > 1 if len(scale.dims) else False
        axis = _attr(n, "axis", 1 if per_channel else None)
        transposed = _stored_transposed(dqs[0].output[0], cons, list(w.dims))
        fqn = module_prefix + name[len(graph_prefix):-len(".weight")]
        sites.append(FP8WeightSite(
            fqn=fqn, initializer=name, scale_init=scale.name,
            zero_point=n.input[2] if len(n.input) > 2 and n.input[2] else None,
            axis=axis if per_channel else None, per_channel=per_channel,
            transposed=transposed, shape=list(w.dims),
        ))
    return sites


def activation_scale_inits(model) -> list:
    """Scale initializers of the activation (non-initializer-input) Qs."""
    inits = {i.name for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in inits and n.input[1] in inits:
            if n.input[1] not in out:
                out.append(n.input[1])
    return out


def checkpoint_base_reader(
    checkpoint_path, *, prefix: str = "model.",
) -> Callable[[str], torch.Tensor]:
    """``fqn -> fp32 weight`` from the stock checkpoint (``prefix + fqn +
    '.weight'``; medium's DiT tree sits under ``model.`` relative to the
    ``sam.model.model`` root the LoRA manager and mirror use)."""
    from safetensors import safe_open

    handle = safe_open(str(checkpoint_path), framework="pt")

    def read(fqn: str) -> torch.Tensor:
        return handle.get_tensor(f"{prefix}{fqn}.weight").to(torch.float32)

    return read


def build_fp8_manifest(
    onnx_path,
    *,
    base_reader: Optional[Callable[[str], torch.Tensor]] = None,
    base_checkpoint: Optional[str] = None,
    graph_prefix: str = "dit.",
    module_prefix: str = "model.",
) -> tuple:
    """(manifest dict, {initializer name: fp32 scale tensor}).

    With ``base_reader`` every weight initializer is compared bit-for-bit
    against the base the mirror will compose on; any mismatch raises (a
    base-value refit would otherwise not reproduce the built engine)."""
    import onnx

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path), load_external_data=False)
    reader = OnnxInitReader(model, onnx_path)
    sites = fp8_weight_sites(model, graph_prefix=graph_prefix, module_prefix=module_prefix)
    if not sites:
        raise RuntimeError(f"{onnx_path}: no weight QuantizeLinear sites (not an fp8 graph?)")
    scales: Dict[str, torch.Tensor] = {}
    mismatched = []
    for s in sites:
        scales[s.scale_init] = reader.get(s.scale_init).reshape(-1).contiguous()
        if base_reader is not None:
            ref = reader.get(s.initializer)
            base = base_reader(s.fqn)
            if s.transposed:
                base = base.transpose(0, 1)
            if tuple(base.shape) != tuple(ref.shape) or not torch.equal(base.contiguous(), ref):
                mismatched.append(s.fqn)
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} fp8 weights differ from the base checkpoint "
            f"(first: {mismatched[:3]}); the mirror's base would not reproduce the engine"
        )
    act = activation_scale_inits(model)
    for name in act:
        scales[name] = reader.get(name).reshape(-1).contiguous()
    manifest = {
        "version": FP8_MANIFEST_VERSION,
        "precision": "fp8",
        "onnx": onnx_path.name,
        "graph_prefix": graph_prefix,
        "module_prefix": module_prefix,
        "base": "checkpoint_fp32",
        "base_checkpoint": str(base_checkpoint) if base_checkpoint else None,
        "base_verified": base_reader is not None,
        "weights": {
            s.fqn: {k: v for k, v in asdict(s).items() if k != "fqn"} for s in sites
        },
        "activation_scales": act,
        "unmatched": [],
        "ambiguous": [],
    }
    return manifest, scales


def write_fp8_manifest(manifest: dict, scales: Dict[str, torch.Tensor], out_path) -> Path:
    from safetensors.torch import save_file

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    side = out_path.with_name(out_path.name + SCALES_SUFFIX)
    save_file({k: v.contiguous() for k, v in scales.items()}, str(side))
    manifest = dict(manifest, scales_file=side.name)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def load_fp8_scales(manifest_path, manifest: dict) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    side = Path(manifest_path).with_name(manifest["scales_file"])
    return {k: v.to(torch.float32) for k, v in load_file(str(side)).items()}


def quantize_e4m3_fixed(
    w: torch.Tensor, scale: torch.Tensor, axis: Optional[int],
) -> torch.Tensor:
    """Saturating E4M3 quantization of ``w`` (engine orientation) under a
    FIXED scale, as uint8 storage. ``axis`` is the per-channel axis (None =
    per-tensor). Values past the bound clip to +-448 (ONNX ``saturate=1``
    and TRT semantics); a bare ``float8_e4m3fn`` cast would make them NaN."""
    w = w.to(torch.float32)
    s = scale.to(device=w.device, dtype=torch.float32)
    if axis is not None and s.numel() > 1:
        shape = [1] * w.dim()
        shape[axis] = s.numel()
        s = s.view(shape)
    else:
        s = s.reshape(())
    q = (w / s).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.view(torch.uint8)


def refit_inventory(refitter, manifest: dict, trt_mod=None) -> dict:
    """What a refit-built fp8 engine exposes, against the manifest: which
    weight initializers and scale initializers are refittable, and the
    prototype dtypes of the weight slots."""
    names = set(refitter.get_all_weights())
    w = manifest.get("weights", {})
    weight_inits = [r["initializer"] for r in w.values()]
    scale_inits = sorted({r["scale_init"] for r in w.values()})
    act = list(manifest.get("activation_scales", []))
    dtypes: Dict[str, int] = {}
    if hasattr(refitter, "get_weights_prototype"):
        for n in weight_inits:
            if n in names:
                try:
                    dt = str(refitter.get_weights_prototype(n).dtype)
                except Exception as exc:  # noqa: BLE001
                    dt = f"error:{type(exc).__name__}"
                dtypes[dt] = dtypes.get(dt, 0) + 1
    return {
        "refittable_total": len(names),
        "weights_mapped": len(weight_inits),
        "weights_refittable": sum(1 for n in weight_inits if n in names),
        "weight_scales": len(scale_inits),
        "weight_scales_refittable": sum(1 for n in scale_inits if n in names),
        "activation_scales": len(act),
        "activation_scales_refittable": sum(1 for n in act if n in names),
        "weight_prototype_dtypes": dtypes,
        "other_refittable_sample": sorted(
            names - set(weight_inits) - set(scale_inits) - set(act)
        )[:20],
    }


__all__: Iterable[str] = (
    "FP8_MANIFEST_VERSION", "E4M3_MAX", "FP8WeightSite", "OnnxInitReader",
    "fp8_weight_sites", "activation_scale_inits", "checkpoint_base_reader",
    "build_fp8_manifest", "write_fp8_manifest", "load_fp8_scales",
    "quantize_e4m3_fixed", "refit_inventory",
)

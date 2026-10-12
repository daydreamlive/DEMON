"""fp16mixed-style ONNX export of an SA3 DiT-like torch module (DEMON's
own SA3 producer, the first stage of the fp8 graft recipe).

:func:`export_fp16mixed` traces a module whose weights are fp16 with the
legacy TorchScript exporter (opset 17, external data) while
:func:`island_patches` pins upstream's fp16mixed fp32 island set by module
patching (vendored ``build_dit_fp16.py`` / README):

* ``rmsnorm``    RMSNorm (and LayerNorm) statistics, normalisation and gain
                 in fp32, result cast back to the input dtype.
* ``softmax``    the attention softmax in fp32 (QK^T and P.V stay fp16).
* ``rope``       rotary angle generation + rotation in fp32. The vendored
                 code already computes RoPE in fp32 (``@autocast(enabled=
                 False)``); the patch only promotes an fp16 ``inv_freq``
                 so the traced Einsum has one dtype. Always on.
* ``fourier``    the timestep / seconds Fourier front-ends
                 (``FourierFeatures``, ``ExpoFourierFeatures``,
                 ``LearnedPositionalEmbedding``) in fp32.
* ``cond_embed`` ``to_cond_embed`` (prompt projection) in fp32.

Everything else (every other GEMM, the attention BMMs, adapters) stays fp16.
Nothing is pinned for accumulation (the private "pinned" recipe's fp32
Linear weights are exactly the defect this replaces).

The export runs with ``do_constant_folding=False`` on purpose: folding the
Linear weight Transposes renames the weights to ``onnx::MatMul_N``, and the
fp8 graft (:mod:`acestep.engine.trt.sa3_fp8`) maps every MatMul to its
calibration key by the module-qualified weight initializer name (reached
directly or through Transpose/Cast). TensorRT folds those constants itself.

The saved graph is Kahn topo-sorted, written with all tensors in one
``<name>.onnx.data`` file and reloaded to assert every initializer survived
byte-identical (onnx external-data saves can silently revert tensors).

The graph helpers (:func:`topo_sort`, :func:`linear_sites`,
:func:`save_onnx_checked`, :func:`onnx_sha256`) are shared with the graft.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import heapq
import math
import os
import sys
import tempfile
import types
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

EXPORT_OPSET = 17
DEFAULT_MIN_FEATURES = 128

ISLANDS = ("rmsnorm", "softmax", "rope", "fourier", "cond_embed")
ISLAND_SETS = {
    "upstream": ISLANDS,
    "none": ("rope",),
}
FOURIER_CLASS_NAMES = frozenset({"FourierFeatures", "ExpoFourierFeatures", "LearnedPositionalEmbedding"})
COND_EMBED_NAMES = frozenset({"to_cond_embed"})

# Ops counted as computing (not plumbing) when listing fp32 island nodes.
_COMPUTE_OPS = frozenset({
    "MatMul", "Gemm", "Softmax", "ReduceMean", "Pow", "Sqrt", "Reciprocal", "Div", "Mul",
    "Add", "Sub", "Cos", "Sin", "Exp", "Einsum", "Neg", "Tanh", "Sigmoid", "Erf",
})
_WEIGHT_PASSTHROUGH = frozenset({"Transpose", "Cast", "Identity"})


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------


class _Report:
    def _asdict(self) -> dict:
        return asdict(self)


@dataclass
class ExportReport(_Report):
    onnx_path: str
    node_count: int
    island_nodes: list[str]
    linear_count: int
    input_names: list[str]
    sha256: str
    islands: list[str] = field(default_factory=list)
    island_ops: dict[str, int] = field(default_factory=dict)
    linear_weights: list[str] = field(default_factory=list)
    opset: int = EXPORT_OPSET
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# island patching
# ---------------------------------------------------------------------------


def resolve_islands(islands: str | Iterable[str]) -> tuple[str, ...]:
    """``"upstream"`` (all five), ``"none"`` (rope only, it cannot be turned
    off), a comma list, or an iterable of island names."""
    if isinstance(islands, str):
        if islands in ISLAND_SETS:
            names = ISLAND_SETS[islands]
        else:
            names = tuple(s.strip() for s in islands.split(",") if s.strip())
    else:
        names = tuple(islands)
    bad = [n for n in names if n not in ISLANDS]
    if bad:
        raise ValueError(f"unknown islands {bad}; known {ISLANDS} or sets {tuple(ISLAND_SETS)}")
    out = set(names) | {"rope"}
    return tuple(n for n in ISLANDS if n in out)


def _cast_tree(obj, dtype):
    if torch.is_tensor(obj):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, tuple):
        return tuple(_cast_tree(o, dtype) for o in obj)
    if isinstance(obj, list):
        return [_cast_tree(o, dtype) for o in obj]
    return obj


def _fp32_forward(module: nn.Module):
    """Instance forward that runs ``module`` with fp32 parameters/buffers and
    fp32 floating inputs, casting outputs back to the first input's dtype."""
    def fwd(*args, **kwargs):
        dtypes = [a.dtype for a in args if torch.is_tensor(a) and a.is_floating_point()]
        out_dtype = dtypes[0] if dtypes else torch.float32
        # Swap every floating param/buffer in the subtree for an fp32 view
        # (traced as Cast(initializer)); torch.func.functional_call refuses
        # to run under the TorchScript tracer, so do it by hand.
        saved = []
        for sub in module.modules():
            for store in (sub._parameters, sub._buffers):
                for name, p in list(store.items()):
                    if p is not None and p.is_floating_point() and p.dtype != torch.float32:
                        saved.append((store, name, p))
                        store[name] = p.float()
        args32 = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
        try:
            out = type(module).forward(module, *args32, **kwargs)
        finally:
            for store, name, p in reversed(saved):
                store[name] = p
        return _cast_tree(out, out_dtype)

    return fwd


def _make_apply_attn(softmax_fp32: bool):
    def apply_attn(self, q, k, v, causal=None, flex_attention_block_mask=None, flex_attention_score_mod=None,
                   flash_attn_sliding_window=None, padding_mask=None, varlen_metadata=None, **_unused):
        if flash_attn_sliding_window is not None or causal:
            raise RuntimeError("sa3_export: sliding-window / causal attention is not covered")
        if getattr(self, "num_heads", None) != getattr(self, "kv_heads", getattr(self, "num_heads", None)):
            raise RuntimeError("sa3_export: grouped KV heads are not covered")
        if padding_mask is not None and varlen_metadata is None:
            raise RuntimeError("sa3_export: padding mask without varlen metadata")
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if varlen_metadata is not None:
            mask = varlen_metadata["mask"]                       # [B, N] bool
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        if softmax_fp32:
            probs = scores.float().softmax(dim=-1).to(v.dtype)
        else:
            probs = scores.softmax(dim=-1)
        out = torch.matmul(probs, v)
        if varlen_metadata is not None:
            out = out * mask[:, None, :, None].to(out.dtype)
        return out.to(v.dtype)

    return apply_attn


def _ff_forward(self, x, varlen_metadata=None, **_unused):
    out = self.ff(x)
    if varlen_metadata is not None:
        out = out * varlen_metadata["mask"][..., None].to(out.dtype)
    return out


def _rope_forward(module: nn.Module):
    cls_forward = type(module).forward

    def fwd(*args, **kwargs):
        inv = getattr(module, "inv_freq", None)
        if inv is None or inv.dtype == torch.float32:
            return cls_forward(module, *args, **kwargs)
        module.inv_freq = inv.float()
        try:
            return cls_forward(module, *args, **kwargs)
        finally:
            module.inv_freq = inv

    return fwd


@contextlib.contextmanager
def island_patches(module: nn.Module, islands: str | Iterable[str] = "upstream"):
    """Patch ``module`` (and torch.nn.functional norms) for export with the
    given fp32 islands; everything is restored on exit. Also swaps the
    vendored flash-attention / varlen paths for exact exportable ones
    (always, independent of the island set)."""
    active = set(resolve_islands(islands))
    restore: list = []

    norm_fp32 = "rmsnorm" in active
    orig_rms = getattr(F, "rms_norm", None)
    orig_ln = F.layer_norm

    def rms_norm(x, normalized_shape, weight=None, eps=None):
        # aten::rms_norm has no opset-17 symbolic; statistics always in fp32.
        dims = tuple(range(-len(normalized_shape), 0))
        if eps is None:
            eps = torch.finfo(x.dtype).eps
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=dims, keepdim=True) + eps)
        if norm_fp32:
            if weight is not None:
                normed = normed * weight.float()
            return normed.to(x.dtype)
        out = normed.to(x.dtype)
        return out if weight is None else out * weight.to(x.dtype)

    def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
        if not norm_fp32 or x.dtype == torch.float32:
            return orig_ln(x, normalized_shape, weight, bias, eps)
        out = orig_ln(x.float(), normalized_shape, None if weight is None else weight.float(),
                      None if bias is None else bias.float(), eps)
        return out.to(x.dtype)

    F.rms_norm = rms_norm
    F.layer_norm = layer_norm
    restore.append(lambda: setattr(F, "layer_norm", orig_ln))
    restore.append(lambda: setattr(F, "rms_norm", orig_rms))

    apply_attn = _make_apply_attn("softmax" in active)
    patched_pymods: dict[str, object] = {}
    counts: Counter = Counter()

    def instance_patch(m: nn.Module, attr: str, value) -> None:
        had = attr in m.__dict__
        old = m.__dict__.get(attr)
        setattr(m, attr, value)

        def undo(m=m, attr=attr, had=had, old=old):
            if had:
                setattr(m, attr, old)
            else:
                delattr(m, attr)

        restore.append(undo)

    for name, m in module.named_modules():
        cls = type(m)
        leaf = name.rsplit(".", 1)[-1]
        if callable(getattr(cls, "apply_attn", None)):
            instance_patch(m, "apply_attn", types.MethodType(apply_attn, m))
            counts["attention"] += 1
            pymod = sys.modules.get(cls.__module__)
            if pymod is not None and cls.__module__ not in patched_pymods:
                patched_pymods[cls.__module__] = pymod
        elif cls.__name__ == "FeedForward" and hasattr(m, "ff"):
            instance_patch(m, "forward", types.MethodType(_ff_forward, m))
        if hasattr(m, "inv_freq") and torch.is_tensor(getattr(m, "inv_freq", None)):
            instance_patch(m, "forward", _rope_forward(m))
            counts["rope"] += 1
        if "fourier" in active and cls.__name__ in FOURIER_CLASS_NAMES:
            instance_patch(m, "forward", _fp32_forward(m))
            counts["fourier"] += 1
        if "cond_embed" in active and leaf in COND_EMBED_NAMES:
            instance_patch(m, "forward", _fp32_forward(m))
            counts["cond_embed"] += 1

    for modname, pymod in patched_pymods.items():
        for attr, value in (
            ("precompute_varlen_metadata", lambda padding_mask: {"mask": padding_mask.bool()}),
            ("flash_attn_varlen_func", getattr(pymod, "flash_attn_varlen_func", None) or (lambda *a, **k: None)),
            ("index_first_axis", getattr(pymod, "index_first_axis", None) or (lambda *a, **k: None)),
        ):
            if hasattr(pymod, attr):
                old = getattr(pymod, attr)
                setattr(pymod, attr, value)
                restore.append(lambda pymod=pymod, attr=attr, old=old: setattr(pymod, attr, old))

    logger.info("sa3_export island patches: islands={} patched={}", sorted(active), dict(counts))
    try:
        yield dict(counts)
    finally:
        for undo in reversed(restore):
            undo()


# ---------------------------------------------------------------------------
# ONNX graph helpers (shared with sa3_fp8)
# ---------------------------------------------------------------------------


def topo_sort(graph) -> None:
    """Kahn topological sort of ``graph.node`` in place (stable: ready nodes
    keep their original relative order). Raises on a cycle or a dangling
    input."""
    nodes = list(graph.node)
    available = {i.name for i in graph.initializer} | {i.name for i in graph.input} | {""}
    producer = {}
    for idx, n in enumerate(nodes):
        for o in n.output:
            if o:
                producer[o] = idx
    indeg = [0] * len(nodes)
    consumers: dict[int, list[int]] = {}
    for idx, n in enumerate(nodes):
        deps = set()
        for i in n.input:
            if i in available and i not in producer:
                continue
            if i not in producer:
                raise RuntimeError(f"topo_sort: node {n.name!r} ({n.op_type}) reads undefined tensor {i!r}")
            deps.add(producer[i])
        indeg[idx] = len(deps)
        for d in deps:
            consumers.setdefault(d, []).append(idx)
    ready = [i for i, d in enumerate(indeg) if d == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for c in consumers.get(i, ()):
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(ready, c)
    if len(order) != len(nodes):
        raise RuntimeError(f"topo_sort: cycle; {len(nodes) - len(order)} nodes unplaced")
    del graph.node[:]
    graph.node.extend(nodes[i] for i in order)


def is_topologically_sorted(graph) -> bool:
    seen = {i.name for i in graph.initializer} | {i.name for i in graph.input} | {""}
    for n in graph.node:
        if any(i not in seen for i in n.input):
            return False
        seen.update(n.output)
    return True


@dataclass
class LinearSite:
    """A MatMul/Gemm whose B operand is a 2-D weight initializer, reached
    directly or through Transpose/Cast/Identity."""
    node: object
    init_name: str
    chain: list            # passthrough nodes, initializer side first
    module_path: str       # init name minus a trailing ".weight"
    shape: tuple           # effective [in, out] (MatMul) / as consumed (Gemm)

    @property
    def name(self) -> str:
        return self.node.name or self.init_name


def _module_path(init_name: str) -> str:
    return init_name[: -len(".weight")] if init_name.endswith(".weight") else init_name


def _effective_shape(init, chain) -> tuple:
    shape = list(init.dims)
    for n in chain:
        if n.op_type == "Transpose":
            perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
            shape = [shape[p] for p in perm] if perm else shape[::-1]
    return tuple(shape)


def linear_sites(graph, *, min_features: int = 0) -> list[LinearSite]:
    """Every weight-backed MatMul/Gemm with a 2-D weight whose smaller dim is
    ``>= min_features``, in graph order."""
    inits = {i.name: i for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    sites = []
    for n in graph.node:
        if n.op_type not in ("MatMul", "Gemm") or len(n.input) < 2:
            continue
        chain = []
        t = n.input[1]
        while t not in inits:
            p = producer.get(t)
            if p is None or p.op_type not in _WEIGHT_PASSTHROUGH or not p.input:
                break
            chain.append(p)
            t = p.input[0]
        if t not in inits:
            continue
        init = inits[t]
        if len(init.dims) != 2 or min(init.dims) < min_features:
            continue
        chain.reverse()
        sites.append(LinearSite(n, t, chain, _module_path(t), _effective_shape(init, chain)))
    return sites


def tensor_sha256(tensor) -> str:
    from onnx import numpy_helper

    arr = numpy_helper.to_array(tensor)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def onnx_sha256(path: str | os.PathLike[str]) -> str:
    """sha256 over the ``.onnx`` proto AND every external data file it
    references (a proto hash alone misses a weight-only change)."""
    import onnx
    from onnx.external_data_helper import ExternalDataInfo, uses_external_data

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    locations = set()
    for t in model.graph.initializer:
        if uses_external_data(t):
            locations.add(ExternalDataInfo(t).location)
    h = hashlib.sha256()
    for f in [path] + [path.parent / loc for loc in sorted(locations)]:
        h.update(f.name.encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


EXTERNAL_SIZE_THRESHOLD = 1024
_EXTERNAL_ALIGN = 64


def load_onnx(path: str | os.PathLike[str], *, load_external_data: bool = True):
    """``onnx.load`` with every external tensor read by Python in binary
    mode into ``raw_data`` and its external-data metadata cleared, so
    nothing in memory references the old file."""
    import onnx
    from onnx.external_data_helper import ExternalDataInfo, uses_external_data

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    if not load_external_data:
        return model
    handles: dict[str, object] = {}
    try:
        for t in _all_tensors(model):
            if not uses_external_data(t):
                continue
            info = ExternalDataInfo(t)
            loc = (path.parent / info.location).resolve()
            if path.parent.resolve() not in loc.parents:
                raise ValueError(f"{t.name}: external data {info.location!r} escapes {path.parent}")
            fh = handles.get(str(loc))
            if fh is None:
                fh = handles[str(loc)] = open(loc, "rb")
            fh.seek(int(info.offset or 0))
            data = fh.read() if info.length is None else fh.read(int(info.length))
            if info.length is not None and len(data) != int(info.length):
                raise RuntimeError(f"{t.name}: short read {len(data)} of {info.length} from {loc}")
            t.raw_data = data
            t.data_location = onnx.TensorProto.DEFAULT
            del t.external_data[:]
    finally:
        for fh in handles.values():
            fh.close()
    return model


def _all_tensors(model):
    from onnx.external_data_helper import _get_all_tensors

    return _get_all_tensors(model)


def _materialise(model) -> None:
    """Pull any tensor still pointing at external data into memory (raises
    if one cannot be resolved: a model to save must hold all its bytes)."""
    import onnx
    from onnx.external_data_helper import uses_external_data

    for t in _all_tensors(model):
        if uses_external_data(t):
            if not t.HasField("raw_data"):
                raise RuntimeError(f"{t.name}: external tensor without in-memory data; load with load_onnx()")
            t.data_location = onnx.TensorProto.DEFAULT
            del t.external_data[:]


def _write_external_binary(model, data_path: Path, location: str,
                           size_threshold: int = EXTERNAL_SIZE_THRESHOLD) -> None:
    """Move every initializer with >= ``size_threshold`` raw bytes into
    ``data_path`` (one file, written by Python in binary mode, 64-byte
    aligned) and point the tensor at it.

    Not onnx's writer: on Windows ``onnx.external_data_helper.
    save_external_data`` reopens the data file per tensor with ``os.open``
    and no ``O_BINARY``; the CRT text-mode open strips a trailing Ctrl-Z
    (0x1A), so a tensor whose LAST byte is 0x1A loses it and the next
    tensor is written one byte early (reproduced 2026-10-12 on the stock
    medium: to_timestep_embed.2.bias and layers.22.to_local_embed.0.weight,
    the same two tensors as June 2026)."""
    import onnx
    from onnx.external_data_helper import set_external_data

    with open(data_path, "wb") as fh:
        for t in model.graph.initializer:
            if not t.HasField("raw_data") or len(t.raw_data) < size_threshold:
                continue
            pad = (-fh.tell()) % _EXTERNAL_ALIGN
            if pad:
                fh.write(b"\x00" * pad)
            offset = fh.tell()
            fh.write(t.raw_data)
            set_external_data(t, location, offset, len(t.raw_data))
            t.ClearField("raw_data")
            assert t.data_location == onnx.TensorProto.EXTERNAL


def _nonfinite_count(tensor) -> int:
    import numpy as np
    from onnx import TensorProto, numpy_helper

    if tensor.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE):
        return 0
    return int((~np.isfinite(numpy_helper.to_array(tensor))).sum())


def save_onnx_checked(model, out_path: str | os.PathLike[str]) -> str:
    """Save ``model`` to ``out_path`` + ``<name>.data`` (all tensors >= 1 KiB
    in one binary-written file, see :func:`_write_external_binary`),
    replacing any previous pair atomically, then reload (binary,
    :func:`load_onnx`) and assert every initializer is byte-identical, no
    new NaN/inf appeared, and the graph is topo-sorted. Returns
    :func:`onnx_sha256` of the result. ``model``'s large tensors become
    external references to the new file."""
    import onnx

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not is_topologically_sorted(model.graph):
        raise RuntimeError("save_onnx_checked: graph is not topologically sorted")
    _materialise(model)
    expected = {t.name: tensor_sha256(t) for t in model.graph.initializer}
    nonfinite = {t.name: _nonfinite_count(t) for t in model.graph.initializer}
    n_nodes = len(model.graph.node)
    data_name = out_path.name + ".data"
    with tempfile.TemporaryDirectory(dir=out_path.parent, prefix=".onnx_save_") as tmp:
        tmp_onnx = Path(tmp) / out_path.name
        _write_external_binary(model, Path(tmp) / data_name, data_name)
        onnx.save_model(model, str(tmp_onnx), save_as_external_data=False)
        os.replace(Path(tmp) / data_name, out_path.parent / data_name)
        os.replace(tmp_onnx, out_path)
    reloaded = load_onnx(out_path)
    if len(reloaded.graph.node) != n_nodes:
        raise RuntimeError(f"reload: {len(reloaded.graph.node)} nodes != {n_nodes} saved")
    got = {t.name: tensor_sha256(t) for t in reloaded.graph.initializer}
    if got.keys() != expected.keys():
        raise RuntimeError(f"reload: initializer set changed ({len(got)} vs {len(expected)})")
    bad = [k for k in expected if got[k] != expected[k]]
    if bad:
        raise RuntimeError(f"reload: {len(bad)} initializers changed on disk, e.g. {bad[:3]}")
    new_nan = [t.name for t in reloaded.graph.initializer if _nonfinite_count(t) > nonfinite[t.name]]
    if new_nan:
        raise RuntimeError(f"reload: NaN/inf appeared in {new_nan[:3]}")
    if not is_topologically_sorted(reloaded.graph):
        raise RuntimeError("reload: graph is not topologically sorted")
    del reloaded
    return onnx_sha256(out_path)


def elem_types(model) -> dict[str, int]:
    """Tensor name -> ONNX elem type, via shape inference on a weight-free
    copy of ``model``."""
    import onnx

    m = onnx.ModelProto()
    m.CopyFrom(model)
    for t in m.graph.initializer:           # dims + dtype are enough
        t.ClearField("raw_data")
        del t.external_data[:]
        t.data_location = onnx.TensorProto.DEFAULT
        for f in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data"):
            t.ClearField(f)
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as exc:  # pragma: no cover - inference is best effort
        logger.warning("shape inference failed: {}", exc)
    types_ = {}
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        if vi.type.HasField("tensor_type"):
            types_[vi.name] = vi.type.tensor_type.elem_type
    for t in m.graph.initializer:
        types_[t.name] = t.data_type
    return types_


def island_nodes_of(model) -> list[str]:
    """Names of compute nodes whose first output is fp32."""
    import onnx

    types_ = elem_types(model)
    return [n.name for n in model.graph.node
            if n.op_type in _COMPUTE_OPS and n.output and types_.get(n.output[0]) == onnx.TensorProto.FLOAT]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _ensure_fp16(module: nn.Module) -> None:
    other = sorted({str(p.dtype) for p in module.parameters() if p.is_floating_point() and p.dtype != torch.float16})
    if other:
        logger.info("sa3_export: casting module to fp16 in place (found {})", other)
        module.half()


def export_fp16mixed(
    module: nn.Module,
    example_inputs: Mapping[str, torch.Tensor],
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]] | None,
    out_onnx: str | os.PathLike[str],
    islands: str | Iterable[str] = "upstream",
    opset: int = EXPORT_OPSET,
    min_features: int = DEFAULT_MIN_FEATURES,
) -> ExportReport:
    """Export ``module`` (cast to fp16 in place if needed) to ``out_onnx``
    with the fp32 island set ``islands``; see the module docstring.
    ``example_inputs`` is keyed by input name and passed positionally in
    ``input_names`` order. Returns an :class:`ExportReport`
    (``linear_count`` counts weight-backed Linears with min dim >=
    ``min_features``, the graft's eligible set)."""
    import time

    import onnx

    out_onnx = Path(out_onnx)
    missing = [n for n in input_names if n not in example_inputs]
    if missing:
        raise ValueError(f"example_inputs lacks {missing}")
    extra = [n for n in example_inputs if n not in input_names]
    if extra:
        raise ValueError(f"example_inputs has names not in input_names: {extra}")
    active = resolve_islands(islands)
    module.eval()
    _ensure_fp16(module)
    args = tuple(example_inputs[n] for n in input_names)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=out_onnx.parent, prefix=".export_") as tmp:
        raw = Path(tmp) / "raw.onnx"
        with torch.no_grad(), island_patches(module, active):
            torch.onnx.export(
                module, args, str(raw),
                input_names=list(input_names), output_names=list(output_names),
                dynamic_axes={k: dict(v) for k, v in (dynamic_axes or {}).items()},
                opset_version=int(opset), do_constant_folding=False, dynamo=False,
                external_data=True,
            )
        model = load_onnx(raw)
    topo_sort(model.graph)
    sites = linear_sites(model.graph, min_features=min_features)
    unnamed = [s.init_name for s in sites if not s.init_name.endswith(".weight")]
    if unnamed:
        raise RuntimeError(f"export lost module weight names for {len(unnamed)} Linears, e.g. {unnamed[:3]}")
    islands_found = island_nodes_of(model)
    ops = Counter()
    by_name = {n.name: n.op_type for n in model.graph.node}
    for name in islands_found:
        ops[by_name[name]] += 1
    node_count = len(model.graph.node)
    sha = save_onnx_checked(model, out_onnx)
    report = ExportReport(
        onnx_path=str(out_onnx), node_count=node_count, island_nodes=islands_found,
        linear_count=len(sites), input_names=list(input_names), sha256=sha,
        islands=list(active), island_ops=dict(ops), linear_weights=[s.init_name for s in sites],
        opset=int(opset), seconds=round(time.time() - t0, 2),
    )
    logger.info("sa3_export: {} nodes, {} Linears (min dim >= {}), {} fp32 island nodes {} -> {}",
                node_count, len(sites), min_features, len(islands_found), dict(ops), out_onnx)
    return report


def match_patterns(path: str, patterns: Iterable[str]) -> bool:
    """fnmatch ``path`` against ``patterns`` (exact names match too)."""
    return any(path == p or fnmatch.fnmatchcase(path, p) for p in patterns)


__all__ = [
    "ExportReport", "ISLANDS", "ISLAND_SETS", "LinearSite", "export_fp16mixed", "island_nodes_of",
    "island_patches", "is_topologically_sorted", "linear_sites", "load_onnx", "match_patterns", "onnx_sha256",
    "resolve_islands", "save_onnx_checked", "topo_sort",
]

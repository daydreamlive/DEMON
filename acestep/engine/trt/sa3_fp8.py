"""DEMON's SA3 fp8 producer: activation calibration capture and the fp8
E4M3 Q/DQ graft onto an fp16mixed-style export (runbook 2026-10-12 D1).

Recipe (planner decision D1): the fp16 trunk exported by
:func:`acestep.engine.trt.sa3_export.export_fp16mixed` keeps upstream's
fp32 islands; every eligible trunk Linear gets fp8 Q/DQ on its activation
(per-tensor scale = calibrated max absmax x margin / 448) and its weight
(per-output-channel scales by default); scales are in the GEMM's own dtype
(fp16 in the trunk, fp32 inside an fp32 island such as ``to_cond_embed``)
floored at 1e-4; compiled STRONGLY_TYPED. Mechanics mined from the
vendored ``make_dit_fp8_smalldit.py``; max calibration only (percentile
clipping collapses parity, June 2026).

Linears are found by weight initializer (never by node name) and mapped to
calibration keys by module path: exact, either path a dotted suffix of the
other, or equal after dropping each path's first component. An eligible
Linear with no scale is an error unless ``exclude`` names it.

Stock medium reference (Stability's HF ``onnx/sa3-m/dit_fp8.onnx``, read
2026-10-12): 222 fp8 Linears = every Linear with min dim >= 128 except
``to_timestep_embed.*`` and ``transformer.global_cond_embedder.*``
(:data:`HF_MEDIUM_EXCLUDE`); per-channel weight scales on 220, per-tensor on 2.

CLI::

    python -m acestep.engine.trt.sa3_fp8 graft --fp16-onnx A.onnx --act-scales S.json --out B.onnx
    python -m acestep.engine.trt.sa3_fp8 build --onnx B.onnx --engine-dir D --profile x=1x256x140,... \
        --recipe demon_fp8_graft --act-scales S.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
from loguru import logger
from torch import nn

from .sa3_export import (
    DEFAULT_MIN_FEATURES,
    elem_types,
    linear_sites,
    match_patterns,
    save_onnx_checked,
    topo_sort,
)

E4M3_MAX = 448.0
SCALE_FLOOR = 1e-4
GRAFT_OPSET = 19
HF_MEDIUM_EXCLUDE = ("*to_timestep_embed.*", "*transformer.global_cond_embedder.*")
HF_MEDIUM_FP8_LINEARS = 222

# Reduce ops whose ``axes`` moved from attribute to input at opset 18.
_REDUCE_AXES_TO_INPUT = frozenset({
    "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
    "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
})


# ---------------------------------------------------------------------------
# calibration capture
# ---------------------------------------------------------------------------


def capture_activation_scales(module: nn.Module, run: Callable[[], None], *,
                              min_features: int = DEFAULT_MIN_FEATURES) -> dict[str, float]:
    """Max |input| of every ``nn.Linear`` in ``module`` with
    ``min(in_features, out_features) >= min_features`` over everything
    ``run()`` does. Keys are module qualified names (relative to
    ``module``); values plain floats (JSON-serialisable). Linears never
    called are left out (logged). Non-finite maxima raise."""
    amax: dict[str, torch.Tensor] = {}
    handles = []
    eligible = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= min_features:
            eligible.append(name)

            def hook(_mod, args, _name=name):
                x = args[0] if args else None
                if not torch.is_tensor(x) or x.numel() == 0:
                    return
                v = x.detach().abs().amax().float()
                prev = amax.get(_name)
                amax[_name] = v if prev is None else torch.maximum(prev, v)

            handles.append(m.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            run()
    finally:
        for h in handles:
            h.remove()
    out = {k: float(v) for k, v in sorted(amax.items())}
    bad = [k for k, v in out.items() if v != v or v in (float("inf"), float("-inf"))]
    if bad:
        raise ValueError(f"non-finite activation absmax on {len(bad)} Linears, e.g. {bad[:3]}")
    never = [n for n in eligible if n not in out]
    logger.info("sa3_fp8 capture: {} of {} eligible Linears called ({} never: {})",
                len(out), len(eligible), len(never), never[:5])
    return out


def act_scales_sha256(act_scales: Mapping[str, float]) -> str:
    payload = json.dumps({k: float(act_scales[k]) for k in sorted(act_scales)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_act_scales(path: str | os.PathLike[str]) -> dict[str, float]:
    """A plain ``{module: absmax}`` JSON, or one wrapped as ``{"scales": {...}}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("scales"), dict):
        data = data["scales"]
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a non-empty {{module: absmax}} dict")
    return {str(k): float(v) for k, v in data.items()}


def write_act_scales(path: str | os.PathLike[str], act_scales: Mapping[str, float]) -> str:
    """Write the plain dict; returns its sha256 (of the file bytes)."""
    payload = json.dumps({k: float(act_scales[k]) for k in sorted(act_scales)}, indent=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# graft
# ---------------------------------------------------------------------------


class UnmatchedLinearError(RuntimeError):
    """An eligible Linear has no activation scale and is not excluded."""


@dataclass
class GraftReport:
    linears_total: int
    linears_fp8: int
    unmatched: list[str]                 # eligible Linears left fp16 because excluded
    scale_stats: dict
    out_onnx: str = ""
    sha256: str = ""
    act_scales_sha256: str = ""
    unused_scales: list[str] = field(default_factory=list)
    margin: float = 1.0
    scale_floor: float = SCALE_FLOOR
    weight_scales: str = "per_channel"
    sites: list[dict] = field(default_factory=list)

    def _asdict(self) -> dict:
        return asdict(self)


def _drop_first(path: str) -> str:
    return path.split(".", 1)[1] if "." in path else path


class _ScaleResolver:
    def __init__(self, act_scales: Mapping[str, float]):
        self.scales = dict(act_scales)
        self.used: set[str] = set()

    def lookup(self, path: str) -> str | None:
        if path in self.scales:
            return path
        hits = [k for k in self.scales
                if path.endswith("." + k) or k.endswith("." + path)
                or ("." in k and "." in path and _drop_first(k) == _drop_first(path))]
        if not hits:
            return None
        hits.sort(key=len, reverse=True)
        if len(hits) > 1 and len(hits[0]) == len(hits[1]):
            raise UnmatchedLinearError(f"ambiguous activation scale for {path!r}: {hits[:4]}")
        return hits[0]


def _upgrade_opset(model, target: int = GRAFT_OPSET) -> None:
    """Declare the default-domain opset ``target`` (fp8 Q/DQ needs 19),
    moving Reduce* ``axes`` attributes to inputs (the one 17->19 break)."""
    from onnx import TensorProto, helper

    found = False
    current = 0
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            found = True
            current = op.version
            if op.version < target:
                op.version = target
    if not found:
        model.opset_import.append(helper.make_opsetid("", target))
    if model.ir_version < 9:
        model.ir_version = 9
    if current >= 18:
        return
    for n in model.graph.node:
        if n.op_type not in _REDUCE_AXES_TO_INPUT:
            continue
        axes = next((a for a in n.attribute if a.name == "axes"), None)
        if axes is None:
            continue
        name = f"{n.name or n.output[0]}__axes"
        model.graph.initializer.append(helper.make_tensor(name, TensorProto.INT64, [len(axes.ints)], list(axes.ints)))
        del n.input[1:]
        n.input.append(name)
        keep = [a for a in n.attribute if a.name != "axes"]
        del n.attribute[:]
        n.attribute.extend(keep)


def _effective_weight(init, chain):
    import numpy as np
    from onnx import helper, numpy_helper

    w = numpy_helper.to_array(init)
    for n in chain:
        if n.op_type == "Transpose":
            perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
            w = np.transpose(w, perm) if perm else w.T
        elif n.op_type == "Cast":
            to = next(a.i for a in n.attribute if a.name == "to")
            w = w.astype(helper.tensor_dtype_to_np_dtype(to))
    return w


def _stats(values: list[float], floored: int) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "max": max(values),
            "median": statistics.median(values), "floored": floored}


def graft_fp8(fp16_onnx: str | os.PathLike[str], out_onnx: str | os.PathLike[str],
              act_scales: Mapping[str, float], *, margin: float = 1.0, scale_floor: float = SCALE_FLOOR,
              weight_scales: str = "per_channel", exclude: Iterable[str] = (),
              min_features: int = DEFAULT_MIN_FEATURES) -> GraftReport:
    """Graft fp8 E4M3 Q/DQ onto every eligible Linear of ``fp16_onnx``
    (weight-backed MatMul/Gemm, min weight dim >= ``min_features``) and
    write ``out_onnx`` (+ ``.data``). ``act_scales``: max absmax per module
    (:func:`capture_activation_scales`). ``exclude``: module paths / ONNX
    node names / fnmatch patterns left unquantized. ``weight_scales``:
    ``"per_channel"`` (output channel) or ``"per_tensor"``."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    if weight_scales not in ("per_channel", "per_tensor"):
        raise ValueError(f"weight_scales must be per_channel|per_tensor, got {weight_scales!r}")
    if margin <= 0 or scale_floor <= 0:
        raise ValueError("margin and scale_floor must be > 0")
    exclude = tuple(exclude)
    model = onnx.load(str(fp16_onnx), load_external_data=True)
    _upgrade_opset(model)
    g = model.graph
    types_ = elem_types(model)
    inits = {i.name: i for i in g.initializer}
    sites = linear_sites(g, min_features=min_features)
    resolver = _ScaleResolver(act_scales)

    plan, unmatched, missing = [], [], []
    for s in sites:
        if match_patterns(s.module_path, exclude) or match_patterns(s.name, exclude):
            unmatched.append(s.module_path)
            continue
        key = resolver.lookup(s.module_path)
        if key is None:
            missing.append(s.module_path)
            continue
        plan.append((s, key))
    if missing:
        raise UnmatchedLinearError(
            f"{len(missing)} eligible Linears have no activation scale (pass exclude= to leave them fp16): "
            f"{missing[:8]}")

    used_names: set[str] = {n.name for n in g.node} | set(inits)

    def uniq(base: str) -> str:
        name, i = base, 1
        while name in used_names:
            name = f"{base}_{i}"
            i += 1
        used_names.add(name)
        return name

    new_nodes, new_inits, site_rows = [], [], []
    a_vals, w_vals, a_floor, w_floor = [], [], 0, 0
    for s, key in plan:
        node = s.node
        w = _effective_weight(inits[s.init_name], s.chain)
        w_dtype = TensorProto.FLOAT16 if w.dtype == np.float16 else TensorProto.FLOAT
        a_type = types_.get(node.input[0])
        if a_type is not None and a_type != w_dtype:
            raise RuntimeError(f"{s.name}: activation dtype {a_type} != weight dtype {w_dtype}")
        np_dt = np.float16 if w_dtype == TensorProto.FLOAT16 else np.float32
        absw = np.abs(w.astype(np.float32))
        if node.op_type == "Gemm":
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            axis = 0 if trans_b else 1
        else:
            axis = 1                                     # MatMul B is [in, out]
        pfx = re.sub(r"[^0-9A-Za-z_.]", "_", (node.name or s.init_name).strip("/").replace("/", "_"))

        raw_a = float(act_scales[key]) * margin / E4M3_MAX
        a_scale = max(raw_a, scale_floor)
        a_floor += raw_a < scale_floor
        a_vals.append(a_scale)
        if weight_scales == "per_channel":
            raw_w = absw.max(axis=1 - axis) / E4M3_MAX
            w_scale = np.maximum(raw_w, scale_floor).astype(np_dt)
            w_floor += int((raw_w < scale_floor).sum())
            w_vals.extend(float(v) for v in (w_scale.min(), w_scale.max()))
            w_dims, zp_dims, w_attr = [int(w_scale.shape[0])], [int(w_scale.shape[0])], {"axis": axis}
        else:
            raw_w = float(absw.max()) / E4M3_MAX
            w_scale = np.asarray(max(raw_w, scale_floor), dtype=np_dt)
            w_floor += raw_w < scale_floor
            w_vals.append(float(w_scale))
            w_dims, zp_dims, w_attr = [], [], {}

        n_as, n_az = uniq(f"{pfx}_ascale"), uniq(f"{pfx}_azero")
        n_ws, n_wz = uniq(f"{pfx}_wscale"), uniq(f"{pfx}_wzero")
        new_inits += [
            helper.make_tensor(n_as, w_dtype, [], np.asarray([a_scale], dtype=np_dt).tobytes(), raw=True),
            helper.make_tensor(n_az, TensorProto.FLOAT8E4M3FN, [], b"\x00", raw=True),
            helper.make_tensor(n_ws, w_dtype, w_dims, np.ascontiguousarray(w_scale).tobytes(), raw=True),
            helper.make_tensor(n_wz, TensorProto.FLOAT8E4M3FN, zp_dims,
                               bytes(int(np.prod(zp_dims)) if zp_dims else 1), raw=True),
        ]
        aq, adq, wq, wdq = (uniq(f"{pfx}_{t}") for t in ("aq", "adq", "wq", "wdq"))
        new_nodes += [
            helper.make_node("QuantizeLinear", [node.input[0], n_as, n_az], [aq], name=uniq(f"{pfx}_Qa")),
            helper.make_node("DequantizeLinear", [aq, n_as, n_az], [adq], name=uniq(f"{pfx}_DQa")),
            helper.make_node("QuantizeLinear", [node.input[1], n_ws, n_wz], [wq], name=uniq(f"{pfx}_Qw"), **w_attr),
            helper.make_node("DequantizeLinear", [wq, n_ws, n_wz], [wdq], name=uniq(f"{pfx}_DQw"), **w_attr),
        ]
        node.input[0] = adq
        node.input[1] = wdq
        site_rows.append({"node": node.name, "module": s.module_path, "key": key,
                          "act_scale": a_scale, "dtype": "fp16" if np_dt is np.float16 else "fp32"})

    g.initializer.extend(new_inits)
    g.node.extend(new_nodes)
    topo_sort(g)
    sha = save_onnx_checked(model, out_onnx)
    keys_used = {k for _, k in plan}
    report = GraftReport(
        linears_total=len(sites), linears_fp8=len(plan), unmatched=unmatched,
        scale_stats={"act": _stats(a_vals, a_floor), "weight": _stats(w_vals, w_floor)},
        out_onnx=str(out_onnx), sha256=sha, act_scales_sha256=act_scales_sha256(act_scales),
        unused_scales=sorted(set(act_scales) - keys_used), margin=margin, scale_floor=scale_floor,
        weight_scales=weight_scales, sites=site_rows,
    )
    logger.info("sa3_fp8 graft: fp8 on {}/{} Linears ({} excluded, {} unused scales) -> {}",
                report.linears_fp8, report.linears_total, len(unmatched), len(report.unused_scales), out_onnx)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_profile(items: list[str]) -> dict:
    """``name=1x256x140`` (static) or ``name=1x256x140:1x256x140:1x256x140``."""
    out = {}
    for item in items:
        for part in item.split(","):
            if not part.strip():
                continue
            name, spec = part.split("=", 1)
            shapes = [tuple(int(d) for d in s.split("x")) for s in spec.split(":")]
            if len(shapes) == 1:
                shapes *= 3
            if len(shapes) != 3:
                raise ValueError(f"profile {part!r}: give one shape or min:opt:max")
            out[name.strip()] = tuple(shapes)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("graft", help="fp8 Q/DQ graft onto an fp16mixed-style export")
    g.add_argument("--fp16-onnx", type=Path, required=True)
    g.add_argument("--act-scales", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--margin", type=float, default=1.0)
    g.add_argument("--scale-floor", type=float, default=SCALE_FLOOR)
    g.add_argument("--weight-scales", choices=("per_channel", "per_tensor"), default="per_channel")
    g.add_argument("--exclude", action="append", default=[], help="module path / node name / glob (repeat)")
    g.add_argument("--hf-medium-exclude", action="store_true",
                   help=f"add {HF_MEDIUM_EXCLUDE} (the HF medium fp8 set, {HF_MEDIUM_FP8_LINEARS} Linears)")
    g.add_argument("--expect-linears", type=int, default=None)
    b = sub.add_parser("build", help="STRONGLY_TYPED engine from an ONNX (sa3_build.build_engine_from_onnx)")
    b.add_argument("--onnx", type=Path, required=True)
    b.add_argument("--engine-dir", type=Path, required=True)
    b.add_argument("--profile", action="append", required=True,
                   help="name=1x256x140[:opt:max], comma-separated or repeated")
    b.add_argument("--recipe", required=True)
    b.add_argument("--act-scales", type=Path, default=None)
    b.add_argument("--workspace-gb", type=float, default=16.0)
    b.add_argument("--timing-cache", type=Path, default=None)
    b.add_argument("--not-strongly-typed", action="store_true")
    b.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "graft":
        exclude = list(a.exclude) + (list(HF_MEDIUM_EXCLUDE) if a.hf_medium_exclude else [])
        report = graft_fp8(a.fp16_onnx, a.out, read_act_scales(a.act_scales), margin=a.margin,
                           scale_floor=a.scale_floor, weight_scales=a.weight_scales, exclude=exclude)
        side = a.out.with_suffix(".graft.json")
        side.write_text(json.dumps(report._asdict(), indent=2), encoding="utf-8")
        print(f"linears_fp8={report.linears_fp8} total={report.linears_total} "
              f"excluded={len(report.unmatched)} -> {a.out} ({side.name})")
        if a.expect_linears is not None and report.linears_fp8 != a.expect_linears:
            print(f"FAIL: expected {a.expect_linears} fp8 Linears", file=sys.stderr)
            return 2
        return 0

    from .sa3_build import build_engine_from_onnx

    path = build_engine_from_onnx(
        a.onnx, a.engine_dir, profiles=_parse_profile(a.profile), strongly_typed=not a.not_strongly_typed,
        workspace_gb=a.workspace_gb, timing_cache=a.timing_cache, recipe=a.recipe,
        extra_metadata={}, act_scales_path=a.act_scales, force_rebuild=a.force,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stock-trunk export driver for DEMON's SA3 fp8 producer proof (runbook C3).

Three sub-commands, run one at a time by the GPU conductor:

``--export``  loads the stock medium DiT through ``SA3Context`` (the same
              module root as ``sa3_fp8_stock_calib.py``: ``sam.model.model``),
              builds example inputs from a production cond bundle at the demo
              window (10 s + 3 s wrap = L=140) and calls
              ``acestep.engine.trt.sa3_export.export_fp16mixed`` ->
              ``H/onnx/stock_fp16mixed.onnx``.
``--graft``   ``acestep.engine.trt.sa3_fp8.graft_fp8`` with
              ``H/calib/stock_act_scales.json`` -> ``H/onnx/stock_fp8.onnx``
              with ``exclude=HF_MEDIUM_EXCLUDE`` (the HF medium fp8 set: exactly
              222 Linears).
``--build``   ``acestep.engine.trt.sa3_build.build_engine_from_onnx`` for both
              graphs at profile (1,140,140) into the fixed engine dirs
              ``H/engines/stock/sa3_m_dit_ours_{fp16mixed,fp8}_l1_140_140``.
              The only GPU-heavy step (needs an idle GPU).

Graph I/O (what ``sa3_fp8_stock_proof.py`` binds by name): inputs ``x``
[1,256,L] fp32, ``t`` [1] fp32, then the cond-bundle tensors among
``cross_attn_cond, cross_attn_mask, global_cond, padding_mask,
local_add_cond`` that the bundle actually carries; one output ``velocity``
[1,256,L] fp32. Floating inputs are cast to the module dtype in-graph so the
trunk sees what the eager fp16 DiT sees. The export wrapper registers the
DiT's ``model`` child under the same attribute name, so ONNX weight names
equal the calibration keys (``model.transformer....``).

D1's modules are imported lazily; a missing one stops with a clear message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

H = Path(r"D:\hot\DEMON\musetimbre\fp8_2026-10-12")
DEFAULT_FP16_ONNX = H / "onnx" / "stock_fp16mixed.onnx"
DEFAULT_FP8_ONNX = H / "onnx" / "stock_fp8.onnx"
DEFAULT_SCALES = H / "calib" / "stock_act_scales.json"
ENGINE_ROOT = H / "engines" / "stock"
PROFILE = (1, 140, 140)
ENGINE_FP16MIXED = f"sa3_m_dit_ours_fp16mixed_l{PROFILE[0]}_{PROFILE[1]}_{PROFILE[2]}"
ENGINE_FP8 = f"sa3_m_dit_ours_fp8_l{PROFILE[0]}_{PROFILE[1]}_{PROFILE[2]}"
COND_INPUTS = ("cross_attn_cond", "cross_attn_mask", "global_cond", "padding_mask",
               "local_add_cond")
OUTPUT_NAME = "velocity"
# The HF medium fp8 set (D1, read at source): every Linear with min dim >= 128
# except to_timestep_embed.* and transformer.global_cond_embedder.*.
STOCK_LINEARS = 222
DEMO_DURATION_S = 10.0
DEMO_LATENTS = 140
EXPORT_PROMPT = "funky ass shit"
# Axis carrying the latent length per input (the dynamic axis).
LATENT_AXES = {"x": 2, "local_add_cond": 2, "padding_mask": 1}


def _require(module: str, attr: str, marker: str):
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise SystemExit(f"{module} is not available yet (lane D1, {marker}): {exc}")
    fn = getattr(mod, attr, None)
    if fn is None:
        raise SystemExit(f"{module}.{attr} is not available yet (lane D1, {marker})")
    return fn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def select_cond_inputs(bundle: dict) -> list[str]:
    """The ``COND_INPUTS`` the bundle carries as tensors, in contract order."""
    import torch

    return [k for k in COND_INPUTS if torch.is_tensor(bundle.get(k))]


def bundle_constants(bundle: dict) -> dict:
    """Non-tensor bundle entries (cfg_scale, batch_cfg, ...) baked as constants."""
    import torch

    return {k: v for k, v in bundle.items()
            if not torch.is_tensor(v) and v is not None}


def make_export_wrapper(dit, cond_names: Sequence[str], constants: dict, dtype):
    """Positional-input module around a ``DiTWrapper``-like ``dit``.

    ``dit.model`` is registered as ``self.model`` and ``dit`` itself is kept
    out of the module tree, so parameter (and ONNX initializer) names are
    exactly the calibration keys taken on ``dit``."""
    import torch

    class StockDiTExport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = dit.model
            object.__setattr__(self, "_dit", dit)
            self.cond_names = tuple(cond_names)
            self.constants = dict(constants)
            self.dtype_ = dtype

        def forward(self, x, t, *cond):
            kw = {}
            for name, value in zip(self.cond_names, cond):
                kw[name] = value.to(self.dtype_) if value.is_floating_point() else value
            kw.update(self.constants)
            v = type(self._dit).forward(self._dit, x.to(self.dtype_), t.to(self.dtype_), **kw)
            return v.float()

    return StockDiTExport()


def export_spec(bundle: dict, latent_channels: int, latent_frames: int, device="cpu"):
    """``(example_inputs, input_names, dynamic_axes)`` for the export."""
    import torch

    names = select_cond_inputs(bundle)
    ex = {"x": torch.randn(1, latent_channels, latent_frames, device=device),
          "t": torch.full((1,), 0.7, device=device)}
    for n in names:
        v = bundle[n]
        ex[n] = v.float() if v.is_floating_point() else v
    input_names = ["x", "t", *names]
    dyn = {n: {LATENT_AXES[n]: "latent_frames"} for n in input_names if n in LATENT_AXES}
    dyn[OUTPUT_NAME] = {2: "latent_frames"}
    for n in input_names:
        if n in LATENT_AXES and ex[n].shape[LATENT_AXES[n]] != latent_frames:
            raise ValueError(f"{n} axis {LATENT_AXES[n]} is {ex[n].shape[LATENT_AXES[n]]}, "
                             f"expected latent_frames={latent_frames}")
    return ex, input_names, dyn


def _report_dict(report) -> dict:
    if hasattr(report, "_asdict"):
        d = report._asdict()
    elif hasattr(report, "__dataclass_fields__"):
        import dataclasses

        d = dataclasses.asdict(report)
    elif isinstance(report, dict):
        d = dict(report)
    else:
        d = {"repr": repr(report)}
    return json.loads(json.dumps(d, default=str))


def cmd_export(args) -> int:
    export_fp16mixed = _require("acestep.engine.trt.sa3_export", "export_fp16mixed",
                                "d1_export.ready")
    import torch

    from acestep.engine.sa3_context import SA3Context

    print("[load] SA3 medium via SA3Context", flush=True)
    context = SA3Context("medium", device=args.device)
    dit = context.dit
    dit.eval()
    cond = context.prepare_cond(prompt=EXPORT_PROMPT, duration=DEMO_DURATION_S, steps=8)
    L = int(cond.latent_frames)
    if L != DEMO_LATENTS:
        raise SystemExit(f"window L={L}, expected {DEMO_LATENTS}")
    bundle = cond.cond_bundle
    ex, input_names, dyn = export_spec(bundle, context.latent_channels, L,
                                       device=context.device)
    dropped = [n for n in COND_INPUTS if n not in input_names]
    wrapper = make_export_wrapper(dit, input_names[2:], bundle_constants(bundle),
                                  context.dtype).eval()
    print(f"[export] inputs={input_names} (absent in bundle: {dropped}) "
          f"constants={sorted(bundle_constants(bundle))}", flush=True)
    args.fp16_onnx.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        report = export_fp16mixed(
            wrapper, ex, input_names=input_names, output_names=[OUTPUT_NAME],
            dynamic_axes=dyn, out_onnx=args.fp16_onnx,
        )
    rd = _report_dict(report)
    rd.update({"input_names_requested": input_names, "absent_cond_inputs": dropped,
               "latent_frames": L, "prompt": EXPORT_PROMPT, "module_root": "sam.model.model"})
    side = args.fp16_onnx.with_suffix(".export.json")
    side.write_text(json.dumps(rd, indent=2), encoding="utf-8")
    print(f"[export] {args.fp16_onnx} ({side.name})", flush=True)
    return 0


# ---------------------------------------------------------------------------
# graft
# ---------------------------------------------------------------------------

def load_scales(path: Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"{path}: expected a non-empty {{module: absmax}} dict")
    return {str(k): float(v) for k, v in data.items()}


def cmd_graft(args) -> int:
    graft_fp8 = _require("acestep.engine.trt.sa3_fp8", "graft_fp8", "d1_graft.ready")
    exclude = _require("acestep.engine.trt.sa3_fp8", "HF_MEDIUM_EXCLUDE", "d1_graft.ready")
    if not args.fp16_onnx.is_file():
        raise SystemExit(f"{args.fp16_onnx} missing: run --export first")
    scales = load_scales(args.scales)
    report = graft_fp8(args.fp16_onnx, args.fp8_onnx, scales, exclude=tuple(exclude))
    rd = _report_dict(report)
    rd["act_scales_path"] = str(args.scales)
    rd["exclude"] = list(exclude)
    rd["act_scales_sha256"] = sha256_file(args.scales)
    side = args.fp8_onnx.with_suffix(".graft.json")
    side.write_text(json.dumps(rd, indent=2), encoding="utf-8")
    n = rd.get("linears_fp8")
    print(f"[graft] linears_fp8={n} total={rd.get('linears_total')} "
          f"unmatched={rd.get('unmatched')} -> {args.fp8_onnx}", flush=True)
    if n != args.expect_linears:
        raise SystemExit(f"graft put fp8 on {n} Linears, expected {args.expect_linears}")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def onnx_profiles(onnx_path: Path, profile: tuple[int, int, int]) -> dict:
    """``{input: (min, opt, max)}`` from the graph's input shapes: the dynamic
    (``dim_param``) dims take ``profile``, static dims stay as they are."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    inits = {i.name for i in model.graph.initializer}
    out = {}
    for inp in model.graph.input:
        if inp.name in inits:
            continue
        dims = inp.type.tensor_type.shape.dim
        shapes = []
        for value in profile:
            shape = []
            for d in dims:
                if d.HasField("dim_value") and d.dim_value > 0:
                    shape.append(int(d.dim_value))
                else:
                    shape.append(int(value))
            shapes.append(tuple(shape))
        out[inp.name] = tuple(shapes)
    return out


def build_jobs(args) -> list[dict]:
    jobs = []
    for which, onnx_path, name, recipe in (
        ("fp16mixed", args.fp16_onnx, ENGINE_FP16MIXED, "demon_fp16mixed_export"),
        ("fp8", args.fp8_onnx, ENGINE_FP8, "demon_fp8_graft"),
    ):
        if args.only and which != args.only:
            continue
        extra = {"arm": f"ours_{which}", "profile": list(PROFILE), "source": "stock medium"}
        act = args.scales if which == "fp8" else None
        if act is not None and not act.is_file():
            raise SystemExit(f"{act} missing: the fp8 engine records its act-scales sha")
        jobs.append({"which": which, "onnx": onnx_path, "engine_dir": args.engine_root / name,
                     "recipe": recipe, "extra_metadata": extra, "act_scales_path": act})
    return jobs


def cmd_build(args) -> int:
    build = _require("acestep.engine.trt.sa3_build", "build_engine_from_onnx", "d1_build.ready")
    for job in build_jobs(args):
        if not job["onnx"].is_file():
            raise SystemExit(f"{job['onnx']} missing: run --export/--graft first")
        profiles = onnx_profiles(job["onnx"], PROFILE)
        print(f"[build] {job['engine_dir'].name} from {job['onnx'].name} profiles={profiles}",
              flush=True)
        job["engine_dir"].mkdir(parents=True, exist_ok=True)
        path = build(job["onnx"], job["engine_dir"], profiles=profiles, strongly_typed=True,
                     workspace_gb=args.workspace_gb, timing_cache=args.timing_cache,
                     recipe=job["recipe"], extra_metadata=job["extra_metadata"],
                     act_scales_path=job["act_scales_path"])
        print(f"[build] -> {path}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    step = ap.add_mutually_exclusive_group(required=True)
    step.add_argument("--export", action="store_true")
    step.add_argument("--graft", action="store_true")
    step.add_argument("--build", action="store_true")
    ap.add_argument("--fp16-onnx", type=Path, default=DEFAULT_FP16_ONNX)
    ap.add_argument("--fp8-onnx", type=Path, default=DEFAULT_FP8_ONNX)
    ap.add_argument("--scales", type=Path, default=DEFAULT_SCALES)
    ap.add_argument("--engine-root", type=Path, default=ENGINE_ROOT)
    ap.add_argument("--only", choices=("fp16mixed", "fp8"), default=None,
                    help="--build just one of the two engines")
    ap.add_argument("--expect-linears", type=int, default=STOCK_LINEARS)
    ap.add_argument("--workspace-gb", type=float, default=16.0)
    ap.add_argument("--timing-cache", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.export:
        return cmd_export(args)
    if args.graft:
        return cmd_graft(args)
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())

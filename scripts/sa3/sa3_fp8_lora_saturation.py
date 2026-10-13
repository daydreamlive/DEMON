"""Fixed-scale fp8 refit viability for an SA3 LoRA (CPU only).

For every weight-quantized Linear of Stability's ``onnx/sa3-m/dit_fp8.onnx``
(fp32 weight initializer -> QuantizeLinear(scale) -> DequantizeLinear ->
Transpose -> MatMul; 220 per-output-channel scales, 2 per-tensor) and each
LoRA strength, compute for W' = W + dW (dW = strength * alpha/rank * B @ A,
the vendored ``LoRAParametrization.lora_forward`` math, torch [out, in]):

* ``sat_frac``: fraction of output channels whose amax(W') exceeds the baked
  scale * 448 by more than ``SAT_TOL`` (0.1%; the baked scales are
  amax(W)/448 to ~1e-6, so a strict ``>`` flags ~half of all channels at
  strength 0 on fp32 rounding alone). ``sat_frac_bin`` uses 3.6% (half the
  top E4M3 step, 448 vs 416: clipping beyond what rounding would do anyway).
* ``chan_err_ratio_*``: per saturating channel, rel. error under the baked
  scale over rel. error under a fresh per-channel scale (p50 / max).
* ``clip_elem_frac``: fraction of weight elements beyond the baked range.
* ``err_baked`` / ``err_fresh``: relative Frobenius error
  ||fq(W') - W'|| / ||W'|| of E4M3 fake-quant (saturating, the ONNX
  ``saturate=1`` / TRT behaviour) under the baked scale vs a fresh
  per-channel scale amax(W')/448. ``err_base`` is W under its baked scale.

W is read from the graph's own fp32 initializers (exactly what TRT quantizes
at build/refit). Linears the LoRA does not touch are reported with dW = 0.

    python scripts/sa3/sa3_fp8_lora_saturation.py --onnx <dit_fp8.onnx> \
        --lora <lora_sa3.safetensors> --out-dir <dir> [--strengths 1.0 0.5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from acestep.engine.trt.sa3_fp8_refit import OnnxInitReader, fp8_weight_sites  # noqa: E402

E4M3_MAX = 448.0
SAT_TOL = 1e-3
SAT_TOL_BIN = 16.0 / 448.0


def fake_quant_e4m3(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Saturating E4M3 fake quant; ``scale`` broadcasts over rows ([out] or scalar)."""
    s = scale.view(-1, 1) if scale.dim() == 1 else scale
    q = (w / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    return q * s


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a - ref).norm() / ref.norm().clamp_min(1e-30))


def lora_deltas(lora_path: Path):
    """{graph_weight_name: (A, B, scaling)} from a vendored-format SA3 LoRA file.
    Key ``model.<path>.parametrizations.weight.0.lora_A`` -> graph ``dit.<path>.weight``."""
    from safetensors import safe_open

    f = safe_open(str(lora_path), framework="pt")
    cfg = json.loads((f.metadata() or {}).get("lora_config", "{}"))
    rank = float(cfg.get("rank", 0)) or None
    alpha = float(cfg.get("alpha", rank or 1.0))
    out = {}
    marker = ".parametrizations.weight.0."
    for k in f.keys():
        if not k.endswith(".lora_A"):
            continue
        mod = k[: k.index(marker)]
        A = f.get_tensor(k).float()
        B = f.get_tensor(k[: -len("lora_A")] + "lora_B").float()
        r = rank or A.shape[0]
        path = mod.split(".", 1)[1] if mod.startswith("model.") else mod
        out[f"dit.{path}.weight"] = (A, B, alpha / r)
    return out, cfg


def analyse(onnx_path: Path, lora_path: Path, strengths: list[float]) -> dict:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    site_recs = fp8_weight_sites(model)
    sites = [r.initializer for r in site_recs]
    reader = OnnxInitReader(model, onnx_path)
    deltas, cfg = lora_deltas(lora_path)
    missing = sorted(set(deltas) - set(sites))
    rows = []
    for rec in site_recs:
        name, sname, axis = rec.initializer, rec.scale_init, rec.axis
        if rec.transposed:
            raise RuntimeError(f"{name}: [in, out] storage not handled here")
        W = reader.get(name)
        s = reader.get(sname).reshape(-1)
        per_channel = s.numel() > 1
        if per_channel and (axis not in (0, None) or s.numel() != W.shape[0]):
            raise RuntimeError(f"{name}: unexpected scale layout axis={axis} n={s.numel()}")
        s_b = s if per_channel else s.reshape(())
        bound = (s_b if per_channel else s_b.expand(W.shape[0])) * E4M3_MAX
        row = {
            "weight": name, "shape": list(W.shape), "per_channel": per_channel,
            "lora": name in deltas,
            "baked_vs_amax_W": float(((W.abs().amax(1) / E4M3_MAX) / (s_b if per_channel else s_b)).median()),
            "err_base": rel_err(fake_quant_e4m3(W, s_b), W),
            "by_strength": {},
        }
        dW_unit = None
        if name in deltas:
            A, B, scaling = deltas[name]
            dW_unit = (B @ A) * scaling
            row["dW_over_W_fro"] = float(dW_unit.norm() / W.norm())
        for st in strengths:
            Wp = W if dW_unit is None else W + st * dW_unit
            amax = Wp.abs().amax(1)
            ratio = amax / bound
            sat = ratio > 1 + SAT_TOL
            fresh = (amax.clamp_min(1e-12) / E4M3_MAX)
            fq_b = fake_quant_e4m3(Wp, s_b)
            fq_f = fake_quant_e4m3(Wp, fresh)
            cr = []
            if bool(sat.any()):
                ref = Wp[sat]
                e_b = (fq_b[sat] - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
                e_f = (fq_f[sat] - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
                cr = (e_b / e_f.clamp_min(1e-30)).tolist()
            row["by_strength"][str(st)] = {
                "sat_frac": float(sat.float().mean()),
                "sat_frac_bin": float((ratio > 1 + SAT_TOL_BIN).float().mean()),
                "chan_err_ratio_p50": _q(cr, .5) if cr else 1.0,
                "chan_err_ratio_max": max(cr) if cr else 1.0,
                "max_amax_over_bound": float((amax / bound).max()),
                "clip_elem_frac": float((Wp.abs() > bound.view(-1, 1)).float().mean()),
                "err_baked": rel_err(fq_b, Wp),
                "err_fresh": rel_err(fq_f, Wp),
            }
        rows.append(row)
    return {"onnx": str(onnx_path), "lora": str(lora_path), "lora_config": cfg,
            "n_fp8_linears": len(sites), "n_lora_targets": len(deltas),
            "lora_targets_not_fp8": missing, "strengths": strengths, "rows": rows}


def _q(vals, p):
    vals = sorted(vals)
    if not vals:
        return float("nan")
    i = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
    return vals[i]


def summarise(res: dict) -> dict:
    summ = {}
    for st in res["strengths"]:
        k = str(st)
        for subset, rows in (("all", res["rows"]), ("lora", [r for r in res["rows"] if r["lora"]])):
            b = [r["by_strength"][k] for r in rows]
            sat = [x["sat_frac"] for x in b]
            ratio = [x["err_baked"] / max(x["err_fresh"], 1e-30) for x in b]
            summ[f"{k}/{subset}"] = {
                "n": len(b),
                "linears_with_any_saturation": sum(1 for v in sat if v > 0),
                "sat_frac_mean": statistics.fmean(sat) if sat else 0.0,
                "sat_frac_p50": _q(sat, .5), "sat_frac_p90": _q(sat, .9), "sat_frac_max": max(sat, default=0.0),
                "sat_frac_bin_mean": statistics.fmean([x["sat_frac_bin"] for x in b]) if b else 0.0,
                "sat_frac_bin_max": max((x["sat_frac_bin"] for x in b), default=0.0),
                "linears_with_bin_saturation": sum(1 for x in b if x["sat_frac_bin"] > 0),
                "chan_err_ratio_p50_median": _q([x["chan_err_ratio_p50"] for x in b], .5),
                "chan_err_ratio_max": max((x["chan_err_ratio_max"] for x in b), default=1.0),
                "clip_elem_frac_max": max((x["clip_elem_frac"] for x in b), default=0.0),
                "max_amax_over_bound": max((x["max_amax_over_bound"] for x in b), default=0.0),
                "err_baked_p50": _q([x["err_baked"] for x in b], .5),
                "err_baked_max": max((x["err_baked"] for x in b), default=0.0),
                "err_fresh_p50": _q([x["err_fresh"] for x in b], .5),
                "err_fresh_max": max((x["err_fresh"] for x in b), default=0.0),
                "err_ratio_baked_over_fresh_p50": _q(ratio, .5),
                "err_ratio_baked_over_fresh_max": max(ratio, default=0.0),
            }
    return summ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    args = ap.parse_args()
    t0 = time.time()
    res = analyse(Path(args.onnx), Path(args.lora), args.strengths)
    res["summary"] = summarise(res)
    res["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "q1_saturation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["summary"], indent=2))
    print(f"targets_not_fp8={res['lora_targets_not_fp8']} elapsed={res['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

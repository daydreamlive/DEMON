"""Stock-trunk proof for DEMON's SA3 fp8 producer (runbook D3).

Builds nothing. Given engine dirs for up to three arms, runs the stock
medium eager DiT against each engine on real conditioning bundles at the
demo window (L=140) and judges the D2 bars:

* ``hf_fp8``          control: main's ``sa3_build --fp8`` engine (Stability's
                      HF dit_fp8.onnx).
* ``ours_fp16mixed``  our fp16mixed-style export built plain.
* ``ours_fp8``        our fp16mixed-style export + fp8 graft.

Bars: ours_fp16mixed worst step >= 0.9998; ours_fp8 worst step >= 0.90 and
>= hf_fp8 worst - 0.01, compounded >= hf_fp8 compounded - 0.01, ms/step <=
1.05 x hf_fp8. The tier bars come from ``sa3_parity_tier``.

Method (per arm, per prompt x seed case): deterministic 8-step euler on the
session schedule. Per-step cos: the eager trajectory drives ``x_t`` and the
engine is evaluated on the same ``x_t`` (isolates engine numerics). Compounded
cos: each backend's own euler trajectory from the same ``x_T``, final-latent
cos. Every arm runs two warm-up rollouts first (the first fp8 rollout in a
process is non-deterministic). Any non-finite engine output fails the arm.
ms/step is the median engine step wall time over the measured rollouts.

Conditioning is the production path ``sa3_trt_dit_cond_parity.py --production``
uses: ``SA3Context.prepare_cond`` (song-length label + loop-wrap pad), so
``--duration 10`` lands on L=140. Engines whose inputs are the HF signature
(``t5_hidden``/``seconds_total``) run through DEMON's ``SA3TRTDit``; any other
signature (our export) is bound by input name from ``{x, t, **cond_bundle}``
through the same ``TrtLaunch`` (CUDA graph on both, so ms/step is comparable).

Run (GPU; the conductor only):
    python scripts/sa3/sa3_fp8_stock_proof.py \
        --arm hf_fp8=<engine dir> --arm ours_fp16mixed=<engine dir> \
        --arm ours_fp8=<engine dir> \
        --out D:/hot/DEMON/musetimbre/fp8_2026-10-12/reports/stock_proof.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "scripts" / "sa3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sa3_parity_tier import TIER_FP8, TIER_FP16MIXED, judge  # noqa: E402

STEPS = 8
WARMUPS = 2
DEMO_DURATION_S = 10.0
DEMO_LATENTS = 140
ARMS = ("hf_fp8", "ours_fp16mixed", "ours_fp8")
ARM_TIERS = {"hf_fp8": TIER_FP8, "ours_fp16mixed": TIER_FP16MIXED, "ours_fp8": TIER_FP8}
FP16MIXED_BAR = 0.9998
FP8_FLOOR = 0.90
REL_TOL = 0.01
SPEED_TOL = 1.05
# Held out from calibration (which uses upstream's Music few-shot examples):
# the parity script's prompt plus the demo's music presets.
PROOF_PROMPTS = (
    "funky ass shit",
    "An intimate solo cello with a warm, resonant body, expressive bow pressure, "
    "and the quiet detail of fingers shifting on the strings.",
    "A softly played grand piano in a spacious room, with clear felted attacks, "
    "lingering harmonics, and natural pauses between phrases.",
    "A close, richly voiced brass ensemble with rounded horns, gentle breath in the "
    "attacks, and a broad cinematic sense of space.",
    "A fingerpicked acoustic guitar with woody resonance, delicate string noise, "
    "and an unhurried, human touch.",
    "A luminous analog synthesizer with a rounded lead tone, subtle filter movement, "
    "and a wide halo of shimmering ambience.",
    "A tactile ensemble of hand drums, brushed metal, and dry wooden strikes, layered "
    "with small room reflections and lively dynamics.",
)
SEEDS = (7, 1528)
DEFAULT_OUT = Path(r"D:\hot\DEMON\musetimbre\fp8_2026-10-12\reports\stock_proof.json")
DEFAULT_GRAFT_REPORT = Path(r"D:\hot\DEMON\musetimbre\fp8_2026-10-12\onnx\stock_fp8.graft.json")
# HF's dit_fp8.onnx quantizes 222 Linears; ours uses DEMON_MEDIUM_EXCLUDE (198,
# to_local_embed.0 kept fp16), so the control is not like-for-like on GEMM count.
HF_FP8_LINEARS = 222

StepFn = Callable  # (x [1,C,L] fp32, t: float, bundle: dict) -> v [1,C,L]


# ---------------------------------------------------------------------------
# Pure metrics (CPU-testable with fakes)
# ---------------------------------------------------------------------------

def _cos(a, b) -> float:
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _finite(t) -> bool:
    import torch

    return bool(torch.isfinite(t).all())


def euler_rollout(step: StepFn, x_T, sched: Sequence[float], bundle: dict, *,
                  shadow: Optional[StepFn] = None, clock: Optional[Callable] = None):
    """Deterministic euler over ``sched`` (``x <- x + (t_nxt - t_cur) v``).

    With ``shadow``, ``shadow`` is also evaluated on every driven ``x_t`` and
    its cos vs ``step``'s velocity recorded. With ``clock``, ``step``'s wall
    time per call is recorded. Returns ``(x_0, step_cos, step_ms, nonfinite)``
    where ``nonfinite`` counts non-finite shadow (or step, when unshadowed)
    outputs."""
    x = x_T.clone().float()
    step_cos, step_ms, nonfinite = [], [], 0
    for i in range(len(sched) - 1):
        t_cur, t_nxt = float(sched[i]), float(sched[i + 1])
        t0 = clock() if clock else None
        v = step(x, t_cur, bundle).float().clone()
        if clock:
            step_ms.append((clock() - t0) * 1e3)
        if shadow is not None:
            v_s = shadow(x, t_cur, bundle).float().clone()
            if not _finite(v_s):
                nonfinite += 1
                step_cos.append(float("nan"))
            else:
                step_cos.append(_cos(v, v_s))
        elif not _finite(v):
            nonfinite += 1
        x = x + (t_nxt - t_cur) * v
    return x, step_cos, step_ms, nonfinite


def measure_arm(eager: StepFn, engine: StepFn, cases: Sequence[dict], *,
                warmups: int = WARMUPS, clock: Optional[Callable] = None,
                eager_finals: Optional[dict] = None) -> dict:
    """Per-step and compounded cos of ``engine`` vs ``eager`` over ``cases``.

    ``cases``: ``{"id", "x_T", "sched", "bundle"}``. ``eager_finals`` caches
    the eager trajectory's x_0 per case id across arms (eager is the same
    reference for every arm)."""
    if not cases:
        raise ValueError("no cases")
    eager_finals = {} if eager_finals is None else eager_finals
    w = cases[0]
    warm_nonfinite = 0
    for _ in range(max(0, int(warmups))):
        warm_nonfinite += euler_rollout(engine, w["x_T"], w["sched"], w["bundle"])[3]
    per_case, all_cos, all_ms, nonfinite = [], [], [], warm_nonfinite
    for c in cases:
        x_e, cos_steps, _, nf_shadow = euler_rollout(
            eager, c["x_T"], c["sched"], c["bundle"], shadow=engine)
        eager_finals.setdefault(c["id"], x_e)
        x_g, _, ms, nf_own = euler_rollout(
            engine, c["x_T"], c["sched"], c["bundle"], clock=clock)
        nf = nf_shadow + nf_own + (0 if _finite(x_g) else 1)
        comp = _cos(eager_finals[c["id"]], x_g) if _finite(x_g) else float("nan")
        finite_cos = [v for v in cos_steps if not math.isnan(v)]
        per_case.append({
            "id": c["id"], "step_cos": cos_steps,
            "worst_step_cos": min(finite_cos) if len(finite_cos) == len(cos_steps) else float("nan"),
            "compounded_cos": comp, "nonfinite": nf,
            "ms_step_median": statistics.median(ms) if ms else None,
        })
        all_cos.extend(cos_steps)
        all_ms.extend(ms)
        nonfinite += nf
    clean = [v for v in all_cos if not math.isnan(v)]
    worst_case = min(per_case, key=lambda r: _nan_low(r["worst_step_cos"]))
    comps = [_nan_low(r["compounded_cos"]) for r in per_case]
    # Worst step index across all cases (position inside the 8-step schedule).
    n_steps = len(cases[0]["sched"]) - 1
    by_step = [min(_nan_low(r["step_cos"][i]) for r in per_case) for i in range(n_steps)]
    return {
        "cases": per_case,
        "n_cases": len(per_case),
        "worst_step_cos": min(clean) if len(clean) == len(all_cos) else float("nan"),
        "mean_step_cos": statistics.fmean(clean) if clean else float("nan"),
        "worst_by_step": by_step,
        "worst_case": worst_case["id"],
        "compounded_cos_min": min(comps),
        "compounded_cos_mean": statistics.fmean(comps),
        "ms_step_median": statistics.median(all_ms) if all_ms else None,
        "ms_step_mean": statistics.fmean(all_ms) if all_ms else None,
        "nonfinite": nonfinite,
        "warmup_nonfinite": warm_nonfinite,
    }


def _nan_low(v) -> float:
    return -1.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def judge_arms(arms: dict) -> dict:
    """D2 bars over ``{arm: measure_arm(...)}``. Missing arms make the gate
    ``incomplete``; a non-finite output fails its arm."""
    checks: list[dict] = []

    def add(name, value, op, bar, ok):
        checks.append({"check": name, "value": value, "op": op, "bar": bar, "pass": bool(ok)})

    tiers = {}
    for name, m in arms.items():
        tier = ARM_TIERS.get(name, TIER_FP8 if "fp8" in name else TIER_FP16MIXED)
        v = judge([_nan_low(c) for c in m["worst_by_step"]], tier)
        v["pass"] = bool(v["pass"] and m["nonfinite"] == 0)
        tiers[name] = v
        add(f"{name} non-finite outputs", m["nonfinite"], "==", 0, m["nonfinite"] == 0)

    fp16 = arms.get("ours_fp16mixed")
    if fp16 is not None:
        w = _nan_low(fp16["worst_step_cos"])
        add("ours_fp16mixed worst step", w, ">=", FP16MIXED_BAR, w >= FP16MIXED_BAR)
    ours, hf = arms.get("ours_fp8"), arms.get("hf_fp8")
    if ours is not None:
        w = _nan_low(ours["worst_step_cos"])
        add("ours_fp8 worst step", w, ">=", FP8_FLOOR, w >= FP8_FLOOR)
        if hf is not None:
            hw = _nan_low(hf["worst_step_cos"])
            add("ours_fp8 worst step vs hf_fp8", w, ">=", hw - REL_TOL, w >= hw - REL_TOL)
            oc, hc = _nan_low(ours["compounded_cos_min"]), _nan_low(hf["compounded_cos_min"])
            add("ours_fp8 compounded vs hf_fp8", oc, ">=", hc - REL_TOL, oc >= hc - REL_TOL)
            om, hm = ours["ms_step_median"], hf["ms_step_median"]
            if om is not None and hm is not None:
                add("ours_fp8 ms/step vs hf_fp8", om, "<=", SPEED_TOL * hm, om <= SPEED_TOL * hm)
    missing = [a for a in ARMS if a not in arms]
    all_pass = all(c["pass"] for c in checks)
    gate = "incomplete" if missing else ("PASS" if all_pass else "FAIL")
    return {"gate": gate, "missing_arms": missing, "checks": checks, "tiers": tiers}


def markdown_table(report: dict) -> str:
    lines = [
        f"# SA3 fp8 stock-trunk proof (L={report.get('latent_frames')}, "
        f"{report.get('n_cases')} cases, euler {STEPS} steps, {WARMUPS} warm-ups)",
        "",
        "| arm | engine | worst step cos | mean step cos | compounded min | "
        "compounded mean | ms/step | non-finite | tier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["arms"].items():
        tier = report["verdict"]["tiers"].get(name, {})
        ms = m.get("ms_step_median")
        lines.append(
            f"| {name} | {m.get('engine', '')} | {m['worst_step_cos']:.6f} | "
            f"{m['mean_step_cos']:.6f} | {m['compounded_cos_min']:.6f} | "
            f"{m['compounded_cos_mean']:.6f} | {'' if ms is None else f'{ms:.2f}'} | "
            f"{m['nonfinite']} | {tier.get('tier', '')} "
            f"{'PASS' if tier.get('pass') else 'FAIL'} |")
    eager_ms = report.get("eager_ms_step_median")
    if eager_ms is not None:
        lines.append(f"| eager | fp16 torch | 1 | 1 | 1 | 1 | {eager_ms:.2f} | 0 | ref |")
    lines += ["", "| check | value | bar | pass |", "|---|---|---|---|"]
    for c in report["verdict"]["checks"]:
        val = c["value"]
        val_s = f"{val:.6f}" if isinstance(val, float) else str(val)
        bar_s = f"{c['bar']:.6f}" if isinstance(c["bar"], float) else str(c["bar"])
        lines.append(f"| {c['check']} | {val_s} | {c['op']} {bar_s} | "
                     f"{'PASS' if c['pass'] else 'FAIL'} |")
    counts = report.get("fp8_linears") or {}
    if counts:
        hf_n, ours_n = counts.get("hf_fp8"), counts.get("ours_fp8")
        lines += ["", f"ours_fp8: {'unknown' if ours_n is None else ours_n} fp8 Linears "
                  f"(to_local_embed.0 kept fp16) vs HF {hf_n}"
                  + ("" if ours_n is None or hf_n is None else
                     f" ({hf_n - ours_n} fewer fp8 Linears in ours; the control is not "
                     "like-for-like on quantized GEMM count)") + "."]
    v = report["verdict"]
    lines += ["", f"GATE: {v['gate']}"
              + (f" (missing arms: {', '.join(v['missing_arms'])})" if v["missing_arms"] else "")]
    return "\n".join(lines) + "\n"


def fp8_linear_counts(arms: Sequence[str], graft_report: Optional[Path]) -> dict:
    """fp8 Linear count per fp8 arm: HF's fixed set, ours from the graft sidecar."""
    out: dict = {}
    if "hf_fp8" in arms:
        out["hf_fp8"] = HF_FP8_LINEARS
    if "ours_fp8" in arms:
        n = None
        if graft_report is not None and Path(graft_report).is_file():
            n = json.loads(Path(graft_report).read_text(encoding="utf-8")).get("linears_fp8")
        out["ours_fp8"] = n
    return out


def parse_arms(specs: Sequence[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--arm wants name=engine_dir, got {s!r}")
        name, path = s.split("=", 1)
        name = name.strip()
        if name not in ARMS:
            raise ValueError(f"unknown arm {name!r}; expected one of {ARMS}")
        if name in arms:
            raise ValueError(f"arm {name!r} given twice")
        arms[name] = Path(path.strip())
    if not arms:
        raise ValueError("no --arm given")
    return arms


def resolve_engine(path: Path) -> Path:
    """An engine dir (``<dir>/<dir name>.trt``, else its single ``*.trt``) or
    a ``.trt`` file."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    named = path / f"{path.name}.trt"
    if named.is_file():
        return named
    found = sorted(path.glob("*.trt"))
    if len(found) != 1:
        raise FileNotFoundError(f"{path}: expected one .trt, found {[f.name for f in found]}")
    return found[0]


# ---------------------------------------------------------------------------
# GPU runners
# ---------------------------------------------------------------------------

HF_SIGNATURE = {"x", "t", "t5_hidden", "t5_mask", "seconds_total", "local_add_cond"}


class GenericTRTDit:
    """A DiT engine bound by input name from ``{x, t, **cond_bundle}``: for
    our own export, whose inputs are whatever the wrapped module's forward
    takes. Same ``TrtLaunch`` (dedicated stream + CUDA graph) as SA3TRTDit."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt

        from acestep.engine.sa3_trt import _deserialize_engine, _trt_dtype_to_torch
        from acestep.engine.trt.launch import TrtLaunch

        self._trt = trt
        self._to_torch = _trt_dtype_to_torch
        self.engine = _deserialize_engine(Path(engine_path))
        self.engine_path = Path(engine_path)
        self._ctx = self.engine.create_execution_context()
        self._launch = TrtLaunch(self.engine, self._ctx, label="sa3_fp8_proof_dit")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = trt.TensorIOMode.INPUT
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == mode]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) != mode]
        if len(self.outputs) != 1:
            raise RuntimeError(f"{engine_path.name}: expected one output, got {self.outputs}")
        self._bufs: dict = {}
        self._out = None
        self._staged = None

    def _dtype(self, name):
        return self._to_torch(self._trt, self.engine.get_tensor_dtype(name))

    def _bind_all(self, feeds: dict) -> None:
        import torch

        for name in self.inputs:
            src = feeds[name]
            buf = torch.empty(tuple(src.shape), dtype=self._dtype(name), device="cuda")
            self._bufs[name] = buf
            self._launch.bind(name, buf, is_input=True)
        out = self.outputs[0]
        shape = tuple(self._ctx.get_tensor_shape(out))
        self._out = torch.empty(shape, dtype=self._dtype(out), device="cuda")
        self._launch.bind(out, self._out, is_input=False)

    def step_bundle(self, x, t: float, bundle: dict):
        import torch

        t_tensor = torch.full((x.shape[0],), float(t), device="cuda")
        feeds = {"x": x, "t": t_tensor}
        for name in self.inputs:
            if name in ("x", "t"):
                continue
            v = bundle.get(name)
            if v is None:
                raise KeyError(f"engine input {name!r} not in cond bundle "
                               f"(bundle keys {sorted(k for k, v in bundle.items() if v is not None)})")
            feeds[name] = v
        if not self._bufs:
            self._bind_all(feeds)
        for name in ("x", "t"):
            self._bufs[name].copy_(feeds[name].to(self._bufs[name].dtype))
        if self._staged is not bundle:
            for name in self.inputs:
                if name not in ("x", "t"):
                    self._bufs[name].copy_(feeds[name].to(self._bufs[name].dtype))
            self._staged = bundle
        self._launch.launch()
        return self._out.float()


def engine_input_names(engine_path: Path) -> set[str]:
    import tensorrt as trt

    from acestep.engine.sa3_trt import _deserialize_engine

    eng = _deserialize_engine(Path(engine_path))
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    return {n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT}


def make_engine_step(engine_path: Path, *, latent_frames: int, seconds_total: float):
    """``(step_fn, runner_label)`` for one engine."""
    names = engine_input_names(engine_path)
    if HF_SIGNATURE <= names:
        from acestep.engine.sa3_trt import SA3TRTDit

        dit = SA3TRTDit(engine_path, latent_frames=latent_frames, seconds_total=seconds_total)
        return (lambda x, t, b: dit.step_bundle(x, t, b)), "SA3TRTDit"
    dit = GenericTRTDit(engine_path)
    return (lambda x, t, b: dit.step_bundle(x, t, b)), f"generic[{','.join(dit.inputs)}]"


def _wall_clock():
    import torch

    torch.cuda.synchronize()
    return time.perf_counter()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[],
                    help=f"name=engine_dir, name in {ARMS}; repeat per arm")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--duration", type=float, default=DEMO_DURATION_S)
    ap.add_argument("--expect-latents", type=int, default=DEMO_LATENTS,
                    help="assert the conditioning window (0 disables)")
    ap.add_argument("--n-prompts", type=int, default=len(PROOF_PROMPTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--warmups", type=int, default=WARMUPS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--graft-report", type=Path, default=DEFAULT_GRAFT_REPORT,
                    help="ours_fp8 graft sidecar (linears_fp8 is recorded in the report)")
    args = ap.parse_args(argv)
    arm_dirs = parse_arms(args.arm)
    engines = {n: resolve_engine(p) for n, p in arm_dirs.items()}

    import torch

    from acestep.engine.sa3_context import SA3Context

    print(f"[load] SA3 {args.model} via SA3Context (production cond)", flush=True)
    context = SA3Context(args.model)
    context.dit.eval()
    dtype = context.dtype
    seconds_label = context.cond_seconds_total(args.duration)

    cases = []
    L = None
    for pi, prompt in enumerate(PROOF_PROMPTS[: args.n_prompts]):
        cond = context.prepare_cond(prompt=prompt, duration=args.duration, steps=STEPS)
        L = int(cond.latent_frames)
        if args.expect_latents and L != args.expect_latents:
            raise SystemExit(f"window L={L}, expected {args.expect_latents}")
        sched = context.make_schedule_builder(cond, STEPS)(1.0).tolist()
        for seed in args.seeds:
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            x_T = torch.randn(1, context.latent_channels, L, device="cuda", generator=g)
            cases.append({"id": f"p{pi}_s{seed}", "prompt": prompt, "seed": int(seed),
                          "x_T": x_T, "sched": sched, "bundle": cond.cond_bundle})
    print(f"[cond] L={L} label={seconds_label} cases={len(cases)} "
          f"schedule={[round(v, 4) for v in cases[0]['sched']]}", flush=True)

    def eager(x, t, bundle):
        t_b = torch.tensor([t], device="cuda", dtype=dtype)
        with torch.no_grad():
            return context.dit(x.to(dtype), t_b, **bundle).float()

    # Eager warm-up + its own ms/step (reference only).
    euler_rollout(eager, cases[0]["x_T"], cases[0]["sched"], cases[0]["bundle"])
    _, _, eager_ms, _ = euler_rollout(eager, cases[0]["x_T"], cases[0]["sched"],
                                      cases[0]["bundle"], clock=_wall_clock)

    eager_finals: dict = {}
    arms = {}
    for name in ARMS:
        if name not in engines:
            continue
        path = engines[name]
        step, runner = make_engine_step(path, latent_frames=L, seconds_total=seconds_label)
        print(f"[arm] {name}: {path.parent.name} via {runner}", flush=True)
        m = measure_arm(eager, step, cases, warmups=args.warmups, clock=_wall_clock,
                        eager_finals=eager_finals)
        m.update({"engine": path.parent.name, "engine_path": str(path), "runner": runner})
        arms[name] = m
        print(f"[arm] {name}: worst={m['worst_step_cos']:.6f} mean={m['mean_step_cos']:.6f} "
              f"compounded_min={m['compounded_cos_min']:.6f} ms/step={m['ms_step_median']:.2f} "
              f"nonfinite={m['nonfinite']}", flush=True)
        del step
        torch.cuda.empty_cache()

    verdict = judge_arms(arms)
    report = {
        "model": args.model, "latent_frames": L, "duration_s": args.duration,
        "seconds_label": seconds_label, "outro_pad_s": context.outro_pad_s,
        "sampler": "euler", "steps": STEPS, "warmups": args.warmups,
        "seeds": args.seeds, "prompts": list(PROOF_PROMPTS[: args.n_prompts]),
        "n_cases": len(cases), "eager_dtype": str(dtype),
        "eager_ms_step_median": statistics.median(eager_ms) if eager_ms else None,
        "demon_commit": _git_head(), "arms": arms, "verdict": verdict,
        "fp8_linears": fp8_linear_counts(list(arms), args.graft_report),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = args.out.with_suffix(".md")
    md.write_text(markdown_table(report), encoding="utf-8")
    print(markdown_table(report), flush=True)
    print(f"[out] {args.out} + {md.name}", flush=True)
    return 0 if verdict["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

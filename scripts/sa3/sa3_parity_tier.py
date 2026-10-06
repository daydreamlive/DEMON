"""Parity tiers for the SA3 TensorRT DiT engines.

Two tiers, chosen from the engine dir name (``_fp8_`` => fp8, else
fp16mixed) unless the caller overrides it:

* ``fp16mixed`` is the fidelity tier: min per-step velocity cosine vs the
  eager reference >= 0.9998.
* ``fp8`` is a speed tier judged at upstream's bar (Stability-AI/
  stable-audio-3 PR #86, in our vendored pin 960da1f8): worst-step cosine
  vs fp32 on adversarial seeds ~0.92-0.94 with the early steps near the
  calibrated reference, default stays fp16mixed. An fp8 engine passes at
  min per-step cos >= 0.90 and is never judged at the fidelity gate.

Pure Python (no torch) so the unit tests run without a GPU.
"""

from __future__ import annotations

from typing import Optional

TIER_FP16MIXED = "fp16mixed"
TIER_FP8 = "fp8"
TIERS = (TIER_FP16MIXED, TIER_FP8)

TIER_BARS = {TIER_FP16MIXED: 0.9998, TIER_FP8: 0.90}
TIER_LABELS = {
    TIER_FP16MIXED: "fidelity tier",
    TIER_FP8: "speed tier (upstream #86 bar)",
}


def infer_tier(engine_name: str, override: Optional[str] = None) -> str:
    """Tier for an engine dir/file name; ``override`` wins when given."""
    if override:
        if override not in TIERS:
            raise ValueError(f"unknown parity tier {override!r}; expected one of {TIERS}")
        return override
    return TIER_FP8 if "_fp8_" in str(engine_name) else TIER_FP16MIXED


def tier_bar(tier: str) -> float:
    """Min per-step cosine an engine of ``tier`` must reach."""
    return TIER_BARS[tier]


def tier_label(tier: str) -> str:
    return TIER_LABELS[tier]


def judge(step_cos: list[float], tier: str) -> dict:
    """Verdict fields for a list of per-step cosines under ``tier``.

    Returns tier, bar, label, min cos, worst step index and pass; these are
    the keys the parity scripts store in their JSON."""
    if not step_cos:
        raise ValueError("no per-step cosines to judge")
    worst_step = min(range(len(step_cos)), key=lambda i: step_cos[i])
    worst = float(step_cos[worst_step])
    bar = tier_bar(tier)
    return {"tier": tier, "tier_label": tier_label(tier), "gate": bar,
            "min_step_cos": worst, "worst_step": worst_step,
            "pass": bool(worst >= bar)}


def verdict_line(v: dict) -> str:
    return (f"tier={v['tier']} [{v['tier_label']}] bar={v['gate']} "
            f"min per-step cos={v['min_step_cos']:.6f} at step {v['worst_step']} "
            f"-> {'PASS' if v['pass'] else 'FAIL'}")

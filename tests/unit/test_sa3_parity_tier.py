"""SA3 TRT DiT parity tiers: tier inference from engine names and bar selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sa3"))

from sa3_parity_tier import (  # noqa: E402
    TIER_FP8, TIER_FP16MIXED, infer_tier, judge, tier_bar, tier_label, verdict_line,
)


@pytest.mark.parametrize("name,tier", [
    ("sa3_m_dit_fp8_l1_614_646", TIER_FP8),
    ("sa3_m_dit_l1_614_646", TIER_FP16MIXED),
    ("sa3_sm_dit_l1_646_646", TIER_FP16MIXED),
    ("sa3_sfx_dit_l1_76_162", TIER_FP16MIXED),
    ("sa3_m_dit_fp8_l1_614_646/sa3_m_dit_fp8_l1_614_646.trt", TIER_FP8),
])
def test_infer_tier_from_engine_name(name, tier):
    assert infer_tier(name) == tier


def test_override_wins():
    assert infer_tier("sa3_m_dit_l1_614_646", "fp8") == TIER_FP8
    assert infer_tier("sa3_m_dit_fp8_l1_614_646", "fp16mixed") == TIER_FP16MIXED
    with pytest.raises(ValueError):
        infer_tier("sa3_m_dit_l1_614_646", "bf16")


def test_bars():
    assert tier_bar(TIER_FP16MIXED) == 0.9998
    assert tier_bar(TIER_FP8) == 0.90
    assert "upstream #86" in tier_label(TIER_FP8)


def test_fp8_engine_at_upstream_bar_passes():
    cos = [0.9995, 0.9993, 0.999, 0.998, 0.995, 0.99, 0.98, 0.93]
    v = judge(cos, infer_tier("sa3_m_dit_fp8_l1_614_646"))
    assert v["pass"] and v["tier"] == TIER_FP8 and v["gate"] == 0.90
    assert v["worst_step"] == 7 and v["min_step_cos"] == 0.93
    line = verdict_line(v)
    assert "speed tier (upstream #86 bar)" in line and "step 7" in line and "PASS" in line


def test_same_numbers_fail_fidelity_tier():
    cos = [0.9995, 0.93]
    v = judge(cos, infer_tier("sa3_m_dit_l1_614_646"))
    assert not v["pass"] and v["gate"] == 0.9998


def test_fp8_below_bar_fails_and_fidelity_passes():
    assert not judge([0.99, 0.89], TIER_FP8)["pass"]
    assert judge([0.99999, 0.9998], TIER_FP16MIXED)["pass"]

"""Latency characterization: soft ceilings + a tracked report artifact.

Philosophy: hard, tight latency gates in shared infrastructure die of
flakiness; what kills the product is an ARCHITECTURAL regression (e.g.
sliding back toward chunk-scale response times). So the assertions here
are deliberately coarse ceilings that only such a regression would
trip, every ceiling is env-overridable, and the full percentile detail
lands in a JSON report meant to be diffed between builds.

Report: ``runs/latency-reports/latency-<scenario>.json`` (override the
directory with ``DEMON_LAT_REPORT_DIR``).
"""

import json
import os
from pathlib import Path

import pytest

from .scenarios import SCENARIOS

_NAMES = [s.name for s in SCENARIOS]


def _ceiling(env: str, default: float) -> float:
    return float(os.environ.get(env, default))


# Coarse architectural ceilings (seconds unless noted). A healthy
# streaming build sits far below all of these; chunk-style generation
# sits far above the slice/action ones.
READY_S = _ceiling("DEMON_LAT_CEILING_READY_S", 240.0)         # cold init
FIRST_SLICE_S = _ceiling("DEMON_LAT_CEILING_FIRST_SLICE_S", 20.0)
SLICE_GAP_P95_MS = _ceiling("DEMON_LAT_CEILING_SLICE_GAP_P95_MS", 3000.0)
ACTION_TO_SLICE_MS = _ceiling("DEMON_LAT_CEILING_ACTION_SLICE_MS", 5000.0)
# Knob-to-ear, full-effect bound (playhead reaches the action-time
# generation frontier). Dominated by the playback-lead buffer (~0.2 s at
# depth 4 on the reference 5090) plus enqueue depth; a chunk-scale
# architecture would put this at whole-chunk latency, far above this.
ACTION_TO_AUDIBLE_MS = _ceiling("DEMON_LAT_CEILING_ACTION_AUDIBLE_MS",
                                8000.0)
REALTIME_FACTOR_MIN = _ceiling("DEMON_LAT_FLOOR_REALTIME_FACTOR", 1.0)


def _report_dir() -> Path:
    d = Path(os.environ.get("DEMON_LAT_REPORT_DIR",
                            "runs/latency-reports"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.parametrize("name", _NAMES)
def test_latency_envelope(name, scenario_runs):
    result, _run_dir = scenario_runs(name)
    if result["status"] == "skipped":
        pytest.skip(f"{name}: {result.get('reason')}")
    assert result["status"] == "ok", (
        f"{name} run failed: {result.get('error', result['status'])}")

    # Always persist the report, even when ceilings fail below.
    report = {k: result.get(k) for k in (
        "scenario", "t_config_to_ready_s", "t_ready_to_first_slice_s",
        "slice_gap_ms", "dec_ms", "tick_ms", "lead_s", "n_slices",
        "gen_samples", "wall_s", "realtime_factor", "actions")}
    out = _report_dir() / f"latency-{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    problems = []
    if result["t_config_to_ready_s"] > READY_S:
        problems.append(f"config->ready {result['t_config_to_ready_s']}s "
                        f"> {READY_S}s")
    first = result.get("t_ready_to_first_slice_s")
    if first is None:
        problems.append("no slices received at all")
    elif first > FIRST_SLICE_S:
        problems.append(f"ready->first-slice {first}s > {FIRST_SLICE_S}s")
    gap_p95 = (result.get("slice_gap_ms") or {}).get("p95")
    if gap_p95 is not None and gap_p95 > SLICE_GAP_P95_MS:
        problems.append(f"slice gap p95 {gap_p95}ms > "
                        f"{SLICE_GAP_P95_MS}ms")
    rtf = result.get("realtime_factor")
    if rtf is not None and rtf < REALTIME_FACTOR_MIN:
        problems.append(f"realtime factor {rtf} < {REALTIME_FACTOR_MIN} "
                        f"(generation slower than playback)")
    for a in result.get("actions", []):
        gap = a.get("next_slice_gap_ms")
        if gap is None:
            problems.append(f"{a['kind']}: no slice after the action")
        elif gap > ACTION_TO_SLICE_MS:
            problems.append(f"{a['kind']}->next-slice {gap}ms > "
                            f"{ACTION_TO_SLICE_MS}ms")
        audible = a.get("audible_full_ms")
        if audible is not None and audible > ACTION_TO_AUDIBLE_MS:
            problems.append(f"{a['kind']}->audible(full) {audible}ms > "
                            f"{ACTION_TO_AUDIBLE_MS}ms")

    assert not problems, (
        f"{name} blew a latency ceiling (report: {out}):\n"
        + "\n".join(f"  - {p}" for p in problems))

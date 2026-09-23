"""Unit tests for continuous-knob slew / rate-limiting (knobs.py).

The bug: a fast knob sweep applied raw discontinuous values to the
running stream pipeline with no smoothing, so a single sweep could drive
the audio into distortion/clipping. The fix rate-limits continuous knobs
at the engine's once-per-tick knob read (:class:`KnobSlewLimiter`),
keyed by per-knob ceilings in the registry
(:func:`effective_slew_max_per_s`). Discrete knobs are never touched.

These are pure (torch-free, no GPU) tests over the registry + limiter:
they pin the two regression-critical properties — a large instantaneous
jump is broken into bounded per-tick deltas, and discrete / non-registry
keys pass through unchanged — plus the supporting metadata behavior.
"""

import math

from acestep.streaming.knobs import (
    DEFAULT_SLEW_FRACTION_PER_S,
    KnobSlewLimiter,
    KnobSpec,
    catalog_from_specs,
    effective_slew_max_per_s,
    knob_catalog,
    knob_specs,
)


class _FakeClock:
    """A hand-advanced monotonic clock so the slew math is deterministic
    and independent of wall time."""

    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _specs_by_name(*specs) -> dict:
    return {s.name: s for s in specs}


# --------------------------------------------------------------------------
# effective_slew_max_per_s: which knobs are continuous, and at what rate
# --------------------------------------------------------------------------

def test_effective_slew_default_is_range_relative():
    # A float knob with no explicit ceiling slews at the default fraction
    # of its full [min, max] span per second.
    spec = KnobSpec("denoise", min_val=0.0, max_val=1.0)  # span 1.0
    assert effective_slew_max_per_s(spec) == DEFAULT_SLEW_FRACTION_PER_S * 1.0

    bipolar = KnobSpec("steer", min_val=-30.0, max_val=30.0)  # span 60.0
    assert effective_slew_max_per_s(bipolar) == DEFAULT_SLEW_FRACTION_PER_S * 60.0


def test_effective_slew_explicit_override_and_optout():
    explicit = KnobSpec("x", min_val=0.0, max_val=1.0, slew_max_per_s=0.5)
    assert effective_slew_max_per_s(explicit) == 0.5

    # 0 / negative opts a continuous knob OUT of slewing.
    assert effective_slew_max_per_s(
        KnobSpec("y", max_val=1.0, slew_max_per_s=0.0)
    ) is None
    assert effective_slew_max_per_s(
        KnobSpec("z", max_val=1.0, slew_max_per_s=-1.0)
    ) is None


def test_effective_slew_none_for_discrete_and_degenerate():
    # Discrete knob types are never slewed.
    assert effective_slew_max_per_s(KnobSpec("seed", type="int", max_val=9.0)) is None
    assert effective_slew_max_per_s(
        KnobSpec("mode", type="enum", options=("a", "b"))
    ) is None
    assert effective_slew_max_per_s(
        KnobSpec("flag", type="bool", options=(False, True))
    ) is None
    # A zero-width float range has no meaningful rate.
    assert effective_slew_max_per_s(
        KnobSpec("pin", min_val=1.0, max_val=1.0)
    ) is None


def test_registry_knobs_slew_only_continuous_floats():
    # Tie the registry to expected behavior: the load-bearing continuous
    # knobs slew; the discrete control knobs do not.
    by_name = {s.name: s for s in knob_specs(sde=False)}
    slewed = {n for n, s in by_name.items()
              if effective_slew_max_per_s(s) is not None}

    for cont in ("denoise", "feedback", "shift", "hint_strength",
                 "x0_target", "guidance_scale", "cfg_rescale",
                 "dcw_scaler", "ch_g0", "ch13"):
        assert cont in slewed, cont
    for discrete in ("seed", "feedback_depth", "steps_override",
                     "rcfg_mode", "dcw_enabled", "dcw_mode"):
        assert discrete not in slewed, discrete


# --------------------------------------------------------------------------
# KnobSlewLimiter: the actual rate limiting
# --------------------------------------------------------------------------

def test_large_jump_is_broken_into_bounded_per_tick_deltas():
    # The core regression guard: an instantaneous full-range jump must be
    # ramped, never applied as one step.
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock)
    specs = _specs_by_name(KnobSpec("denoise", min_val=0.0, max_val=1.0))
    ceiling = effective_slew_max_per_s(specs["denoise"])

    # Seed at 0.0 (first sighting snaps, no startup ramp).
    assert lim.apply({"denoise": 0.0}, specs)["denoise"] == 0.0

    dt = 1.0 / 60.0  # ~60 Hz tick
    prev = 0.0
    seen = [prev]
    for _ in range(200):
        clock.advance(dt)
        cur = lim.apply({"denoise": 1.0}, specs)["denoise"]
        step = abs(cur - prev)
        # No tick may move more than the ceiling allows (+ float slop).
        assert step <= ceiling * dt + 1e-9, (step, ceiling * dt)
        prev = cur
        seen.append(cur)
        if math.isclose(cur, 1.0):
            break

    # It did eventually converge, and it took more than one tick (i.e. it
    # actually ramped instead of snapping).
    assert math.isclose(prev, 1.0)
    assert len([v for v in seen if 0.0 < v < 1.0]) >= 1


def test_slew_converges_and_is_monotonic_toward_target():
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock)
    specs = _specs_by_name(KnobSpec("ch_g0", default=1.0, max_val=3.0))
    lim.apply({"ch_g0": 1.0}, specs)  # seed
    last = 1.0
    for _ in range(500):
        clock.advance(0.01)
        cur = lim.apply({"ch_g0": 3.0}, specs)["ch_g0"]
        assert cur >= last - 1e-9  # never overshoots backward
        assert cur <= 3.0 + 1e-9   # never overshoots the target
        last = cur
        if math.isclose(cur, 3.0):
            break
    assert math.isclose(last, 3.0)


def test_negative_direction_slews_symmetrically():
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock)
    specs = _specs_by_name(KnobSpec("denoise", min_val=0.0, max_val=1.0))
    ceiling = effective_slew_max_per_s(specs["denoise"])
    lim.apply({"denoise": 1.0}, specs)  # seed high
    clock.advance(0.05)
    cur = lim.apply({"denoise": 0.0}, specs)["denoise"]
    assert math.isclose(cur, 1.0 - ceiling * 0.05)
    assert cur < 1.0


def test_slow_changes_pass_through_without_lag():
    # A change slower than the ceiling is applied in full on the same tick
    # — a slew limiter adds zero latency below its rate.
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock)
    specs = _specs_by_name(KnobSpec("denoise", min_val=0.0, max_val=1.0))
    ceiling = effective_slew_max_per_s(specs["denoise"])
    lim.apply({"denoise": 0.0}, specs)  # seed
    clock.advance(0.1)
    small = ceiling * 0.1 * 0.5  # half of what the tick budget allows
    cur = lim.apply({"denoise": small}, specs)["denoise"]
    assert math.isclose(cur, small)


def test_discrete_and_unknown_keys_pass_through_unchanged():
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock)
    specs = _specs_by_name(
        KnobSpec("denoise", min_val=0.0, max_val=1.0),
        KnobSpec("seed", type="int", max_val=9999.0),
        KnobSpec("steps_override", type="int", min_val=1.0, max_val=16.0),
        KnobSpec("rcfg_mode", type="enum", options=("off", "full")),
    )
    lim.apply(
        {"denoise": 0.0, "seed": 0, "steps_override": 8, "rcfg_mode": "off"},
        specs,
    )
    clock.advance(1.0 / 60.0)
    out = lim.apply(
        {
            "denoise": 1.0,            # slewed
            "seed": 4242,             # int: verbatim
            "steps_override": 16,      # int: verbatim
            "rcfg_mode": "full",      # enum: verbatim
            "playback_pos": 12.5,      # not in registry: verbatim
            "sde_denoise_curve": {"type": "constant"},  # curve spec: verbatim
        },
        specs,
    )
    assert out["denoise"] < 1.0          # ramping
    assert out["seed"] == 4242           # instant
    assert out["steps_override"] == 16   # instant (no rebuild-detection drift)
    assert out["rcfg_mode"] == "full"
    assert out["playback_pos"] == 12.5
    assert out["sde_denoise_curve"] == {"type": "constant"}


def test_first_sighting_snaps_no_startup_ramp():
    # A session that starts with a non-zero default must not ramp up from
    # 0 on the first tick.
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock)
    specs = _specs_by_name(KnobSpec("ch_g0", default=1.0, max_val=3.0))
    assert lim.apply({"ch_g0": 1.0}, specs)["ch_g0"] == 1.0


def test_dt_is_capped_so_idle_resume_cannot_release_a_full_jump():
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock, max_dt_s=0.2)
    specs = _specs_by_name(KnobSpec("denoise", min_val=0.0, max_val=1.0))
    ceiling = effective_slew_max_per_s(specs["denoise"])
    lim.apply({"denoise": 0.0}, specs)  # seed
    clock.advance(100.0)  # simulate a long idle gap
    cur = lim.apply({"denoise": 1.0}, specs)["denoise"]
    # Bounded by the capped dt, NOT a full snap to 1.0.
    assert math.isclose(cur, ceiling * 0.2)
    assert cur < 1.0


def test_limits_rebuild_when_spec_map_identity_changes():
    # The session swaps its spec map wholesale when the LoRA / steering
    # knob set changes; the limiter must pick up the new continuous knobs.
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock)
    base = _specs_by_name(KnobSpec("denoise", min_val=0.0, max_val=1.0))
    lim.apply({"denoise": 0.0}, base)

    # A new map (new dict object) that adds a runtime LoRA strength knob.
    extended = dict(base)
    extended["lora_str_x"] = KnobSpec("lora_str_x", max_val=2.0)
    extended_map = dict(extended)
    lim.apply({"denoise": 0.0, "lora_str_x": 0.0}, extended_map)
    clock.advance(1.0 / 60.0)
    out = lim.apply({"denoise": 0.0, "lora_str_x": 2.0}, extended_map)
    assert out["lora_str_x"] < 2.0  # the runtime knob is now slewed too


def test_reset_clears_accumulated_state():
    clock = _FakeClock()
    lim = KnobSlewLimiter(clock=clock)
    specs = _specs_by_name(KnobSpec("denoise", min_val=0.0, max_val=1.0))
    lim.apply({"denoise": 0.0}, specs)
    clock.advance(1.0 / 60.0)
    lim.apply({"denoise": 1.0}, specs)  # mid-ramp
    lim.reset()
    # After reset the next sighting snaps straight to the target again.
    clock.advance(1.0 / 60.0)
    assert lim.apply({"denoise": 0.9}, specs)["denoise"] == 0.9


# --------------------------------------------------------------------------
# Catalog projection: the slew ceiling is discoverable on the wire
# --------------------------------------------------------------------------

def test_catalog_exposes_slew_for_continuous_knobs_only():
    cat = catalog_from_specs([
        KnobSpec("denoise", min_val=0.0, max_val=1.0),
        KnobSpec("seed", type="int", max_val=9999.0),
        KnobSpec("rcfg_mode", type="enum", options=("off", "full")),
    ])
    assert cat["denoise"]["slew_max_per_s"] == DEFAULT_SLEW_FRACTION_PER_S * 1.0
    assert "slew_max_per_s" not in cat["seed"]
    assert "slew_max_per_s" not in cat["rcfg_mode"]


def test_full_registry_catalog_has_slew_on_floats():
    cat = knob_catalog(sde=False)
    assert "slew_max_per_s" in cat["denoise"]
    assert "slew_max_per_s" in cat["guidance_scale"]
    assert "slew_max_per_s" not in cat["seed"]
    assert "slew_max_per_s" not in cat["steps_override"]

"""knob_response.json sidecar: perceptual knob positions mapped through a
per-knob response curve (v3 smoke S4 format)."""

from __future__ import annotations

import json

import pytest
import torch

from acestep.paths import STEERING_PACKS_ENV
from acestep.steering.packs import (
    KNOB_RESPONSE_NAME,
    SteeringPack,
    load_session_packs,
    save_pack,
)
from acestep.streaming.knobs import catalog_from_specs

GRID = [i / 10 for i in range(11)]
# r(u) = u^2: an easy curve to check by hand.
R_SQ = [round(u * u, 6) for u in GRID]


def _pack(name, *, gain=None, seed=0, hidden=16):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden, generator=g)
    prov = {"category": "descriptor"}
    if gain is not None:
        prov["calibrated_gain"] = gain
    return SteeringPack(
        family="sa3", checkpoint="medium", block=3, hidden_size=hidden,
        name=name, vector=v / v.norm(), label=name.title(), norm=10.0,
        magnitude=1.0, provenance=prov,
    )


def _sidecar(knobs, **top):
    doc = {
        "version": 1, "kind": "steering_knob_response", "model": "sa3/medium",
        "u_grid": GRID, "interp": "piecewise_linear", "headroom": 1.25,
        "knobs": knobs,
    }
    doc.update(top)
    return doc


@pytest.fixture
def packs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(STEERING_PACKS_ENV, str(tmp_path))
    monkeypatch.setenv("DEMON_STEERING_STACKING", "none")
    save_pack(_pack("curved", gain={"pos": {"median": 40.0}, "neg": {"median": 20.0}}),
              tmp_path / "curved")
    save_pack(_pack("plain", gain={"pos": 10.0}, seed=1), tmp_path / "plain")
    save_pack(_pack("uncal", seed=2), tmp_path / "uncal")
    return tmp_path


def _signed_alpha(ps, knob, value):
    (c,) = ps.build_configs({knob: value}, 1)
    # Format-1 packs negate the positive row: alpha carries the sign.
    return c["alpha"]


def test_no_sidecar_keeps_raw_linear_knob(packs_dir):
    ps = load_session_packs(family="sa3", checkpoint="medium")
    assert ps.responses == {}
    assert _signed_alpha(ps, "steer_curved", 7.5) == 7.5
    spec = {s.name: s for s in ps.knob_specs()}["steer_curved"]
    assert spec.max_val == 50.0 and spec.min_val == -25.0
    assert spec.meta["mapping"] == "linear"
    assert spec.meta["cutoff_position"] == 0.8


def test_sidecar_maps_perceptual_position(packs_dir):
    (packs_dir / KNOB_RESPONSE_NAME).write_text(json.dumps(_sidecar({
        "curved": {"pos": {"r": R_SQ}, "neg": {"r": GRID}},
        "uncal": {"pos": {"r": R_SQ}},
    })))
    ps = load_session_packs(family="sa3", checkpoint="medium")
    # Only calibrated packs with an entry get a response.
    assert set(ps.responses) == {"steer_curved"}
    a = lambda v: _signed_alpha(ps, "steer_curved", v)  # noqa: E731
    # The cutoff sits at 80% of the throw: |v| = 0.8 -> u = 1 -> gain.
    assert a(0.8) == pytest.approx(40.0)
    assert a(-0.8) == pytest.approx(-20.0)
    # Below the cutoff the curve applies: v = 0.4 -> u = 0.5 -> r = 0.25.
    assert a(0.4) == pytest.approx(40.0 * 0.25)
    # Between grid points it is piecewise linear: u = 0.55 -> (0.25+0.36)/2.
    assert a(0.44) == pytest.approx(40.0 * (0.25 + 0.36) / 2)
    # Headroom (0.8..1.0 -> u 1..1.25) stays linear in u.
    assert a(0.9) == pytest.approx(40.0 * 1.125)
    assert a(1.0) == pytest.approx(40.0 * 1.25)
    assert a(-1.0) == pytest.approx(-20.0 * 1.25)
    # Linear neg curve: v = -0.4 -> u = 0.5 -> r = 0.5.
    assert a(-0.4) == pytest.approx(-20.0 * 0.5)
    # Knobs without an entry keep the raw map.
    assert _signed_alpha(ps, "steer_plain", 7.5) == 7.5
    assert _signed_alpha(ps, "steer_uncal", 7.5) == 7.5


def test_sidecar_catalog_meta_says_perceptual(packs_dir):
    (packs_dir / KNOB_RESPONSE_NAME).write_text(json.dumps(_sidecar({
        "curved": {"pos": {"r": R_SQ}, "neg": {"r": GRID}},
    })))
    ps = load_session_packs(family="sa3", checkpoint="medium")
    cat = catalog_from_specs(ps.knob_specs())
    entry = cat["steer_curved"]
    spec = {s.name: s for s in ps.knob_specs()}["steer_curved"]
    assert (spec.min_val, spec.max_val) == (-1.0, 1.0)
    assert spec.meta["mapping"] == "perceptual"
    assert spec.meta["cutoff_position"] == 0.8
    assert spec.meta["cutoff"] == {"pos": 40.0, "neg": 20.0}
    plain = {s.name: s for s in ps.knob_specs()}["steer_plain"]
    assert plain.meta["mapping"] == "linear" and plain.max_val == 12.5
    assert entry["meta"]["mapping"] == "perceptual"


def test_sidecar_one_sign_curve_other_sign_linear_on_same_throw(packs_dir):
    (packs_dir / KNOB_RESPONSE_NAME).write_text(json.dumps(_sidecar({
        "curved": {"pos": {"r": R_SQ}},
    })))
    ps = load_session_packs(family="sa3", checkpoint="medium")
    # Neg has no curve: r(u) = u, cutoff still at 0.8.
    assert _signed_alpha(ps, "steer_curved", -0.8) == pytest.approx(-20.0)
    assert _signed_alpha(ps, "steer_curved", -0.4) == pytest.approx(-10.0)


@pytest.mark.parametrize("bad", [
    {"kind": "something_else"},
    {"model": "sa3/small"},
    {"version": 2},
    {"u_grid": [0.0, 0.5]},
])
def test_bad_sidecar_is_ignored(packs_dir, bad):
    (packs_dir / KNOB_RESPONSE_NAME).write_text(json.dumps(_sidecar({
        "curved": {"pos": {"r": R_SQ}},
    }, **bad)))
    ps = load_session_packs(family="sa3", checkpoint="medium")
    assert ps.responses == {}
    assert _signed_alpha(ps, "steer_curved", 7.5) == 7.5


def test_wrong_length_curve_falls_back_to_linear_on_the_throw(packs_dir):
    (packs_dir / KNOB_RESPONSE_NAME).write_text(json.dumps(_sidecar({
        "curved": {"pos": {"r": R_SQ[:5]}},
    })))
    ps = load_session_packs(family="sa3", checkpoint="medium")
    assert ps.responses["steer_curved"].r["pos"] is None
    assert _signed_alpha(ps, "steer_curved", 0.4) == pytest.approx(20.0)


def test_stacking_applies_after_the_perceptual_map(packs_dir, monkeypatch):
    monkeypatch.setenv("DEMON_STEERING_STACKING", "inv_n")
    (packs_dir / KNOB_RESPONSE_NAME).write_text(json.dumps(_sidecar({
        "curved": {"pos": {"r": R_SQ}},
    })))
    ps = load_session_packs(family="sa3", checkpoint="medium")
    cfgs = ps.build_configs({"steer_curved": 0.8, "steer_plain": 4.0}, 1)
    alphas = sorted(c["alpha"] for c in cfgs)
    assert alphas == pytest.approx([2.0, 20.0])

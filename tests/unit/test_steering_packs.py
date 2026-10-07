"""Steering packs: file round-trip, boot filtering, knob exposure, and the
SA3 backend driving the pipeline's steering slot from ``steer_*`` knobs.

CPU only; the SA3 backend runs over the mock DiT fixtures of
``test_sa3_backend``.
"""

from __future__ import annotations

import pytest
import torch

from acestep.steering.layout import SteeringLayout
from acestep.steering.packs import (
    PackSteering,
    SteeringPack,
    discover_packs,
    load_pack,
    policy_weights,
    save_pack,
)
from acestep.streaming.knobs import (
    STEERING_ALPHA_MAX,
    steering_axis_spec,
    steering_pack_spec,
)


def _pack(name="bright", *, family="sa3", checkpoint="medium", block=3,
          hidden=16, seed=0, **kw) -> SteeringPack:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden, generator=g)
    return SteeringPack(
        family=family, checkpoint=checkpoint, block=block, hidden_size=hidden,
        name=name, vector=v / v.norm(), label=name.title(), norm=12.5,
        magnitude=1.25,
        provenance=kw.pop("provenance", {
            "pos": ["bright airy"], "neg": ["dark muffled"], "pairs": 32,
        }),
        **kw,
    )


def test_round_trip_preserves_vector_and_metadata(tmp_path):
    p = _pack(policy={"kind": "range", "start": 0.25, "end": 1.0})
    path = save_pack(p, tmp_path / "sa3" / "bright")
    assert path.suffix == ".safetensors"
    q = load_pack(path)
    assert torch.equal(q.vector, p.vector)
    assert q.metadata() == p.metadata()
    assert q.knob_name == "steer_bright"
    assert q.path == path


def test_invalid_packs_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_pack(_pack(name="Bad Name"), tmp_path / "x")
    bad = _pack()
    bad.vector = torch.zeros(5)
    with pytest.raises(ValueError):
        save_pack(bad, tmp_path / "y")
    with pytest.raises(ValueError):
        save_pack(_pack(policy={"kind": "nope"}), tmp_path / "z")


def test_discovery_filters_family_checkpoint_layout_and_names(tmp_path):
    save_pack(_pack("bright"), tmp_path / "a")
    save_pack(_pack("warm", checkpoint="small-music"), tmp_path / "b")
    save_pack(_pack("rough", family="acestep"), tmp_path / "c")
    save_pack(_pack("dense", block=40), tmp_path / "d")          # out of layout
    save_pack(_pack("wide", hidden=8), tmp_path / "e")           # wrong hidden
    save_pack(_pack("bright", seed=9), tmp_path / "z_dup")      # duplicate name
    save_pack(_pack("reserved"), tmp_path / "f")
    (tmp_path / "junk.safetensors").write_bytes(b"not a pack")
    lay = SteeringLayout(num_blocks=24, hidden_size=16)
    packs = discover_packs(
        tmp_path, family="sa3", checkpoint="medium", layout=lay,
        reserved_names=["steer_reserved"],
    )
    assert [p.name for p in packs] == ["bright"]
    assert torch.equal(packs[0].vector, _pack("bright").vector)
    # No layout: only family/checkpoint filtering.
    loose = discover_packs(tmp_path, family="sa3", checkpoint="medium")
    assert sorted(p.name for p in loose) == ["bright", "dense", "reserved", "wide"]
    assert discover_packs(tmp_path / "missing", family="sa3", checkpoint="medium") == []
    assert discover_packs(None, family="sa3", checkpoint="medium") == []


def test_policy_weights():
    assert policy_weights(None, 4) == (1.0, 1.0, 1.0, 1.0)
    assert policy_weights({"kind": "range", "start": 0.0, "end": 0.5}, 8) == (
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    )
    assert policy_weights({"kind": "range", "start": 0.5, "end": 1.0}, 4) == (
        0.0, 0.0, 1.0, 1.0,
    )
    assert policy_weights({"kind": "step", "step": 3, "of": 8}, 8) == (
        0, 0, 0, 1, 0, 0, 0, 0,
    )
    assert policy_weights({"kind": "curve", "weights": [1.0, 0.5]}, 4) == (
        1.0, 1.0, 0.5, 0.5,
    )


def test_pack_knob_spec_shares_axis_semantics():
    a = steering_axis_spec("steer_bright")
    b = steering_pack_spec("steer_bright", label="Bright", block=7)
    for f in ("type", "default", "min_val", "max_val", "options", "group", "bank"):
        assert getattr(a, f) == getattr(b, f)
    assert b.max_val == STEERING_ALPHA_MAX and b.min_val == -STEERING_ALPHA_MAX
    assert "block 7" in b.description


def test_pack_surface_configs_and_snapshot():
    ps = PackSteering([_pack("bright", block=2), _pack("warm", block=5, seed=1)])
    assert [s.name for s in ps.knob_specs()] == ["steer_bright", "steer_warm"]
    raw = {"steer_bright": 2.0, "steer_warm": 0.0}
    cfgs = ps.build_configs(raw, 4)
    assert len(cfgs) == 1
    c = cfgs[0]
    assert c["layer"] == 2 and c["alpha"] == 2.0 and c["magnitude"] == 1.25
    assert c["weights"] == (1.0, 1.0, 1.0, 1.0)
    assert ps.snapshot_key(raw, 4) != ps.snapshot_key({**raw, "steer_warm": 1.0}, 4)
    assert ps.snapshot_key(raw, 4) != ps.snapshot_key(raw, 8)
    assert PackSteering().is_loaded is False


def test_calibrated_pack_range_and_manifest_meta():
    from acestep.streaming.knobs import STEERING_PACK_HEADROOM, catalog_from_specs

    cal = _pack("riser", provenance={
        "category": "sound_effect",
        "calibrated_gain": {"pos": 48.0, "neg": 4.0,
                            "pos_reached": True, "neg_reached": False},
        "screening": {"flags": ["needs_ear"]},
    }, blurb="positive adds a riser")
    plain = _pack("bright", seed=1)
    specs = PackSteering([cal, plain]).knob_specs()
    by = {s.name: s for s in specs}

    # Per-sign range = calibrated gain x 1.25 (cutoff at 80% of throw);
    # values past the old +-30 cap are representable.
    r = by["steer_riser"]
    assert r.max_val == pytest.approx(48.0 * STEERING_PACK_HEADROOM)
    assert r.min_val == pytest.approx(-4.0 * STEERING_PACK_HEADROOM)
    assert r.max_val > STEERING_ALPHA_MAX
    # Uncalibrated pack keeps the shared bipolar range.
    p = by["steer_bright"]
    assert (p.min_val, p.max_val) == (-STEERING_ALPHA_MAX, STEERING_ALPHA_MAX)

    cat = catalog_from_specs(specs)
    m = cat["steer_riser"]["meta"]
    assert cat["steer_riser"]["max"] == pytest.approx(60.0)
    assert m["category"] == "sound_effect" and m["label"] == "Riser"
    assert m["blurb"] == "positive adds a riser"
    assert m["calibrated"] is True
    assert m["cutoff"] == {"pos": 48.0, "neg": 4.0}
    assert m["cutoff_reached"] == {"pos": True, "neg": False}
    assert m["flags"] == ["needs_ear"]
    mp = cat["steer_bright"]["meta"]
    assert mp["category"] == "" and mp["calibrated"] is False
    # Non-steering knobs carry no meta key.
    from acestep.streaming.knobs import KnobSpec
    assert "meta" not in catalog_from_specs([KnobSpec("x")])["x"]

    # A wire value past the old cap survives coercion up to the new max.
    from acestep.streaming.knobs import coerce_knob_values
    clean, _ = coerce_knob_values({"steer_riser": 55.0}, by)
    assert clean["steer_riser"] == pytest.approx(55.0)


def test_calibrated_gain_reached_shorthand_and_mirrored_neg():
    spec = steering_pack_spec(
        "steer_x", gain={"pos": 8.0, "reached": False},
    )
    assert spec.min_val == pytest.approx(-10.0) and spec.max_val == pytest.approx(10.0)
    assert spec.meta["cutoff_reached"] == {"pos": False, "neg": False}


# ---- SA3 backend: manifest + slot wiring (mock DiT) -----------------------


def _sa3_backend():
    from test_sa3_backend import _backend

    return _backend()


def _sa3_pack(name="bright", block=1, alpha_policy=None):
    # The mock DiT has no trunk blocks; the slot is still driven, which is
    # what the backend owns. Delivery is covered by test_steering_seam.
    return SteeringPack(
        family="sa3", checkpoint="medium", block=block, hidden_size=4,
        name=name, vector=torch.ones(4) / 2.0, magnitude=3.0,
        policy=alpha_policy or {"kind": "range", "start": 0.0, "end": 1.0},
    )


def test_sa3_manifest_exposes_steer_knobs_only_when_packs_attached():
    b = _sa3_backend()
    names = {s.name for s in b.knob_specs()}
    assert not any(n.startswith("steer_") for n in names)
    b.attach_steering_packs(PackSteering([_sa3_pack()]))
    specs = {s.name: s for s in b.knob_specs()}
    assert "steer_bright" in specs
    assert specs["steer_bright"].group == "steering"


def test_sa3_backend_pushes_pack_configs_into_pipeline_slot():
    from acestep.streaming.generator_backend import TickContext
    from test_sa3_backend import _knobs

    b = _sa3_backend()
    b.attach_steering_packs(PackSteering([_sa3_pack()]))
    ctx = TickContext(playhead_s=0.0, buffer_duration_s=1.0)
    b.produce(_knobs(b), ctx, "generate")
    assert b.pipeline._steering_by_layer == {}
    b.produce(_knobs(b, steer_bright=2.0), ctx, "generate")
    applies = b.pipeline._steering_by_layer[1]
    assert len(applies) == 1
    assert applies[0].scale == pytest.approx(6.0)
    assert applies[0].weights == (1.0, 1.0, 1.0)
    # Back to 0 clears the slot.
    b.produce(_knobs(b, steer_bright=0.0), ctx, "generate")
    assert b.pipeline._steering_by_layer == {}


def _shift_norms(ps, raw):
    """Per-knob applied shift norm (|alpha| x magnitude x |vector|)."""
    out = {}
    for c in ps.build_configs(raw, 1):
        out[c["layer"]] = float(c["vector"].norm()) * c["magnitude"] * c["alpha"]
    return out


def test_stacking_inv_n_halves_two_active_knobs():
    from acestep.steering.packs import STACKING_INV_N, STACKING_NONE

    packs = [_pack("bright", block=2), _pack("warm", block=5, seed=1),
             _pack("rough", block=7, seed=2)]
    ps = PackSteering(packs, stacking=STACKING_INV_N)
    one = _shift_norms(ps, {"steer_bright": 4.0})
    assert one == pytest.approx(_shift_norms(PackSteering(packs, stacking=STACKING_NONE),
                                             {"steer_bright": 4.0}))
    solo_warm = _shift_norms(ps, {"steer_warm": -3.0})
    two = _shift_norms(ps, {"steer_bright": 4.0, "steer_warm": -3.0, "steer_rough": 0.0})
    assert two[2] == pytest.approx(one[2] / 2)
    assert two[5] == pytest.approx(solo_warm[5] / 2)
    three = _shift_norms(ps, {"steer_bright": 4.0, "steer_warm": -3.0, "steer_rough": 1.0})
    assert three[2] == pytest.approx(one[2] / 3)
    unscaled = _shift_norms(PackSteering(packs, stacking=STACKING_NONE),
                            {"steer_bright": 4.0, "steer_warm": -3.0})
    assert unscaled[2] == pytest.approx(one[2])


def test_stacking_rule_env_toggle(monkeypatch):
    from acestep.steering.packs import STACKING_ENV, stacking_rule

    monkeypatch.delenv(STACKING_ENV, raising=False)
    assert stacking_rule() == "inv_n"
    monkeypatch.setenv(STACKING_ENV, "none")
    assert stacking_rule() == "none"
    assert PackSteering().stacking == "none"
    monkeypatch.setenv(STACKING_ENV, "bogus")
    assert stacking_rule() == "inv_n"

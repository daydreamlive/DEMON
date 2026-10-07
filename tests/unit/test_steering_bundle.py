"""Steering bundles: one safetensors file per model holding every shipped
knob, its per-sign quality bar, and the one-sided clamp that bar drives.

CPU only. The installed-bundle test runs only when a bundle is installed
beside the loose sa3/medium packs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from acestep.steering.packs import (
    BUNDLE_NAME,
    PackSteering,
    SteeringPack,
    build_bundle,
    discover_packs,
    load_bundle,
    load_pack,
    read_bundle_manifest,
    save_pack,
)
from acestep.streaming.knobs import (
    STEERING_PACK_HEADROOM,
    catalog_from_specs,
    coerce_knob_values,
)

H = 1536
REPO = Path(__file__).resolve().parents[2]


def _unit(seed, n=H):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g)
    return v / v.norm()


def _gain(pos=40.0, neg=20.0):
    def side(m):
        return {"median": m, "min": m * 0.9, "max": m * 1.1, "reached": [True, True, False],
                "seeds": [0, 1, 2], "cutoff": [3.1, 3.2, 3.3]}
    return {"pos": side(pos), "neg": side(neg)}


def _pack(name, **kw):
    d = dict(family="sa3", checkpoint="medium", block=3, hidden_size=H, name=name,
             vector=_unit(sum(map(ord, name))), norm=20.0, magnitude=2.0,
             category="timbre", applies_to=["music"], variant="a",
             description=f"{name} description", pos_anchor=f"{name} +",
             neg_anchor=f"{name} -",
             provenance={"calibrated_gain": _gain(), "knob_source": f"v_{name}"})
    d.update(kw)
    return SteeringPack(**d)


def _packs():
    vecs = torch.stack([_unit(10), _unit(11) * 7.5, _unit(12)])
    return [
        _pack("single"),
        _pack("single_neg", vector_neg=_unit(5)),
        _pack("multi", vectors=vecs, blocks=(4, 7, 20), block=4, vector=vecs[0],
              vectors_neg=torch.stack([_unit(20), _unit(21), _unit(22)]),
              norms=(1.0, 2.0, 3.0)),
    ]


BARS = {
    "single": {"pos": True, "neg": True},
    "single_neg": {"pos": True, "neg": False},
    "multi": {"pos": False, "neg": True},
}


def _tensors(p):
    names = ("vector", "vector_neg", "vectors", "vectors_neg")
    return {n: getattr(p, n) for n in names if getattr(p, n) is not None}


def _same_pack(a, b):
    ta, tb = _tensors(a), _tensors(b)
    assert ta.keys() == tb.keys()
    for k in ta:
        assert ta[k].dtype == tb[k].dtype and torch.equal(ta[k], tb[k]), (a.name, k)
    assert tuple(a.all_blocks) == tuple(b.all_blocks)
    ma, mb = a.metadata(), b.metadata()
    ga = ma["provenance"].pop("calibrated_gain")
    gb = mb["provenance"].pop("calibrated_gain")
    for s in ("pos", "neg"):
        for k in ("median", "min", "max", "reached", "seeds", "cutoff"):
            assert ga[s][k] == gb[s][k], (a.name, s, k)
    assert a.effective_category == b.effective_category
    ma.pop("category", None), mb.pop("category", None)
    assert ma == mb
    for x, y in zip(a.terms(), b.terms()):
        assert x.block == y.block and x.pos_magnitude == y.pos_magnitude
        assert torch.equal(x.pos, y.pos)
        assert (x.neg is None) == (y.neg is None)
        if x.neg is not None:
            assert torch.equal(x.neg, y.neg) and x.neg_magnitude == y.neg_magnitude


def test_bundle_round_trip(tmp_path):
    packs = _packs()
    manifest = build_bundle(packs, BARS, tmp_path / BUNDLE_NAME, created="2026-10-06T00:00:00Z")
    assert manifest == read_bundle_manifest(tmp_path / BUNDLE_NAME)
    assert manifest["version"] == 1 and manifest["model"] == "sa3/medium"
    assert manifest["created"] == "2026-10-06T00:00:00Z"
    by = {k["name"]: k for k in manifest["knobs"]}
    assert set(by) == set(BARS)
    for name, k in by.items():
        assert k["bar_pass"] == BARS[name]
        assert set(k["calibrated_gain"]) == {"pos", "neg"}
        for s in ("pos", "neg"):
            assert set(k["calibrated_gain"][s]) == {"median", "min", "max", "reached",
                                                    "seeds", "cutoff"}
        assert "calibrated_gain" not in k["provenance"]
        assert k["anchors"] == {"pos": f"{name} +", "neg": f"{name} -"}
    assert by["multi"]["blocks"] == [4, 7, 20]
    loaded = {p.name: p for p in load_bundle(tmp_path / BUNDLE_NAME)}
    for p in packs:
        _same_pack(p, loaded[p.name])
        assert loaded[p.name].bar_pass == BARS[p.name]


@pytest.mark.parametrize("breakage", [
    "no_gain", "flat_gain", "gain_missing_key", "no_bar", "bar_one_sign",
])
def test_bundle_refuses_incomplete_metadata(tmp_path, breakage):
    packs = _packs()
    bars = dict(BARS)
    prov = packs[0].provenance
    if breakage == "no_gain":
        prov.pop("calibrated_gain")
    elif breakage == "flat_gain":
        prov["calibrated_gain"] = {"pos": 40.0, "neg": 20.0}
    elif breakage == "gain_missing_key":
        prov["calibrated_gain"]["neg"].pop("cutoff")
    elif breakage == "no_bar":
        bars.pop("single")
    else:
        bars["single"] = {"pos": True}
    with pytest.raises(ValueError, match="single"):
        build_bundle(packs, bars, tmp_path / BUNDLE_NAME)
    assert not (tmp_path / BUNDLE_NAME).exists()


def test_discover_prefers_bundle_and_keeps_loose_user_packs(tmp_path):
    packs = _packs()
    d = tmp_path / "sa3" / "medium"
    build_bundle(packs, BARS, d / BUNDLE_NAME)
    # A stale loose copy of a bundled knob (different vector): shadowed.
    save_pack(_pack("single", vector=_unit(999)), d / "single")
    save_pack(_pack("user_made", provenance={}), d / "user_made")
    found = {p.name: p for p in discover_packs(tmp_path, family="sa3", checkpoint="medium")}
    assert set(found) == {"single", "single_neg", "multi", "user_made"}
    assert found["single"].path.name == BUNDLE_NAME
    assert torch.equal(found["single"].vector, packs[0].vector)
    assert found["user_made"].path.name == "user_made.safetensors"
    assert found["user_made"].bar_pass is None


def test_one_sided_clamp_in_knob_manifest(tmp_path):
    build_bundle(_packs(), BARS, tmp_path / BUNDLE_NAME)
    specs = PackSteering(load_bundle(tmp_path / BUNDLE_NAME)).knob_specs()
    cat = catalog_from_specs(specs)
    pos_hi = round(40.0 * STEERING_PACK_HEADROOM, 4)
    neg_lo = -round(20.0 * STEERING_PACK_HEADROOM, 4)

    both = cat["steer_single"]
    assert both["meta"]["bar_pass"] == {"pos": True, "neg": True}
    assert (both["min"], both["max"]) == (neg_lo, pos_hi)

    pos_only = cat["steer_single_neg"]
    assert pos_only["meta"]["bar_pass"] == {"pos": True, "neg": False}
    assert (pos_only["min"], pos_only["max"]) == (0.0, pos_hi)
    assert pos_only["meta"]["cutoff"]["neg"] == 20.0  # calibration still reported

    neg_only = cat["steer_multi"]
    assert neg_only["meta"]["bar_pass"] == {"pos": False, "neg": True}
    assert (neg_only["min"], neg_only["max"]) == (neg_lo, 0.0)

    by_name = {s.name: s for s in specs}
    clean, errors = coerce_knob_values(
        {"steer_single_neg": -10.0, "steer_multi": 10.0}, by_name,
    )
    assert clean == {"steer_single_neg": 0.0, "steer_multi": 0.0} and len(errors) == 2


def test_loose_pack_has_no_bar_and_stays_bipolar(tmp_path):
    save_pack(_pack("loose"), tmp_path / "loose")
    (spec,) = PackSteering([load_pack(tmp_path / "loose.safetensors")]).knob_specs()
    assert "bar_pass" not in spec.meta
    assert spec.min_val < 0 < spec.max_val


def test_builder_cli(tmp_path, capsys):
    sys.path.insert(0, str(REPO / "scripts" / "steering"))
    import build_bundle as cli

    src = tmp_path / "packs"
    for p in _packs():
        save_pack(p, src / p.name)
    # Keyed by knob_source with the ship pass's measured shape.
    bar = {f"v_{n}": {"pos": {"good": b["pos"]}, "neg": {"good": b["neg"]}}
           for n, b in BARS.items()}
    (tmp_path / "bar.json").write_text(json.dumps(bar))
    assert cli.main(["--packs", str(src), "--bar", str(tmp_path / "bar.json")]) == 0
    human = json.loads((src / "bundle.json").read_text())
    assert human == read_bundle_manifest(src / BUNDLE_NAME)
    assert "3 knobs, 2 one-sided" in capsys.readouterr().out
    # Missing a sign for one knob: refuses, writes nothing new.
    bar.pop("v_multi")
    (tmp_path / "bar.json").write_text(json.dumps(bar))
    out2 = tmp_path / "out2"
    assert cli.main(["--packs", str(src), "--bar", str(tmp_path / "bar.json"),
                     "--out", str(out2)]) == 1
    assert not (out2 / BUNDLE_NAME).exists()


def test_installed_bundle_matches_loose_packs():
    from acestep.paths import steering_packs_dir

    d = steering_packs_dir() / "sa3" / "medium"
    bundle = d / BUNDLE_NAME
    if not bundle.is_file():
        pytest.skip("no installed sa3/medium bundle")
    packs = load_bundle(bundle)
    checked = 0
    for p in packs:
        loose = d / f"{p.name}.safetensors"
        if not loose.is_file():
            continue
        _same_pack(load_pack(loose), p)
        checked += 1
    if not checked:
        pytest.skip("no loose packs beside the installed bundle")
    assert checked == len(packs)


def _applied_shift(steer, knob, value):
    (c,) = steer.build_configs({knob: value}, 1)
    return float(c["vector"].float().norm()) * c["magnitude"] * c["alpha"]


def test_installed_bundle_c_pack_neg_shift_matches_manifest():
    """Variant-c packs carry the neg gain in the dn vector's own magnitude
    units (provenance.dn_magnitude_own); a negative knob must apply exactly
    that shift, and the positive side the pack magnitude."""
    from acestep.paths import steering_packs_dir

    bundle = steering_packs_dir() / "sa3" / "medium" / BUNDLE_NAME
    if not bundle.is_file():
        pytest.skip("no installed sa3/medium bundle")
    manifest = {k["name"]: k for k in read_bundle_manifest(bundle)["knobs"]}
    c_packs = [p for p in load_bundle(bundle) if p.vectors_neg is not None
               and "dn_magnitude_own" in (manifest[p.name].get("provenance") or {})]
    if not c_packs:
        pytest.skip("no variant-c packs in the installed bundle")
    for p in c_packs:
        k = manifest[p.name]
        steer = PackSteering([p])
        dn_mag = float(k["provenance"]["dn_magnitude_own"])
        pos_mag = float(k["apply"]["magnitude"])
        for sign, expect in (("pos", pos_mag), ("neg", dn_mag)):
            g = float(k["calibrated_gain"][sign]["median"])
            value = g if sign == "pos" else -g
            got = _applied_shift(steer, p.knob_name, value)
            assert got == pytest.approx(g * expect, rel=1e-5), (p.name, sign)

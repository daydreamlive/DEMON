"""Steering pack format 2: multi-block ``vectors``/``blocks``, optional
negative directions, the new header keys, and byte-compatibility of every
format-1 pack (synthetic and, when present, the installed ones).

CPU only. The summed engine input is built by the real
``StreamPipeline`` slot fill (the same code that feeds the SA3 TRT
steering input), so these tests pin the tensor the engine receives.
"""

from __future__ import annotations

import json

import pytest
import torch

from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import StreamPipeline
from acestep.steering.layout import SteeringLayout
from acestep.steering.packs import (
    PackSteering,
    SteeringPack,
    discover_packs,
    load_pack,
    policy_weights,
    save_pack,
)
from acestep.streaming.knobs import catalog_from_specs

NB, H = 24, 1536


class _EngineAdapter:
    """Minimal family whose 'engine' takes the [B, NB, H] steering input."""

    name = "fake"
    latent_channels = 4
    latent_rate_hz = 10.0
    sample_rate = 16000
    accepts_steering = True

    def __init__(self, nb=NB, h=H):
        self.nb, self.h = nb, h

    def steering_layout(self):
        return SteeringLayout(num_blocks=self.nb, hidden_size=self.h, engine_input=True)

    def steering_blocks(self):
        return None


def summed_input(configs, *, steps=4, nb=NB, h=H) -> torch.Tensor:
    """The ONE summed steering input ``[steps, nb, h]`` (row i = step i)."""
    cfg = DiffusionConfig(infer_steps=steps, infer_method="ode", noise_on_cpu=True,
                          dcw_enabled=False)
    pipe = StreamPipeline(None, cfg, pipeline_depth=1, adapter=_EngineAdapter(nb, h))
    pipe.set_steering(configs)
    pipe._current_step_per_row = list(range(steps))
    buf = torch.zeros(steps, nb, h, dtype=torch.float32)
    pipe._fill_steering_rows(buf, steps)
    return buf


def _v1_reference(path, alpha, steps):
    """Format-1 semantics straight from the file, independent of packs.py:
    ``alpha * magnitude * vector`` at ``block``, gated by the policy."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = json.loads(f.metadata()["steering_pack"])
        vec = f.get_tensor("vector").to(torch.float32)
    out = torch.zeros(steps, NB, H)
    w = policy_weights(meta.get("policy"), steps)
    scale = alpha * float(meta["magnitude"])
    for i in range(steps):
        if w[i] != 0.0:
            out[i, int(meta["block"]), :] += (scale * w[i]) * vec
    return out


def _unit(seed, n=H):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g)
    return v / v.norm()


def _base(name, **kw):
    d = dict(family="sa3", checkpoint="medium", block=3, hidden_size=H, name=name,
             vector=_unit(0), norm=20.0, magnitude=2.0)
    d.update(kw)
    return SteeringPack(**d)


# ---- format 1 stays byte-compatible -------------------------------------


def test_v1_pack_header_and_tensors_unchanged(tmp_path):
    from safetensors import safe_open

    path = save_pack(_base("bright", provenance={"category": "timbre"}), tmp_path / "b")
    with safe_open(str(path), framework="pt") as f:
        assert set(f.keys()) == {"vector"}
        meta = json.loads(f.metadata()["steering_pack"])
    assert meta["format"] == 1
    for k in ("vectors", "blocks", "norms", "category", "variant", "description"):
        assert k not in meta
    p = load_pack(path)
    assert not p.is_v2 and p.all_blocks == (3,)
    (t,) = p.terms()
    assert t.pos is p.vector and t.pos_magnitude == 2.0 and t.neg is None


def test_v1_synthetic_summed_input_matches_file_semantics(tmp_path):
    save_pack(_base("a", block=2, policy={"kind": "range", "start": 0.25, "end": 1.0}),
              tmp_path / "a")
    save_pack(_base("b", block=2, vector=_unit(1)), tmp_path / "b")
    save_pack(_base("c", block=9, vector=_unit(2), magnitude=0.7), tmp_path / "c")
    packs = discover_packs(tmp_path, family="sa3", checkpoint="medium")
    ps = PackSteering(packs)
    raw = {"steer_a": 3.0, "steer_b": -1.5, "steer_c": 12.0}
    got = summed_input(ps.build_configs(raw, 4), steps=4)
    want = sum(_v1_reference(p.path, raw[p.knob_name], 4) for p in packs)
    assert torch.allclose(got, want, atol=1e-6)


def _installed():
    from acestep.paths import steering_packs_dir

    d = steering_packs_dir()
    lay = SteeringLayout(num_blocks=NB, hidden_size=H, engine_input=True)
    packs = discover_packs(d, family="sa3", checkpoint="medium", layout=lay)
    return [p for p in packs if not p.is_v2]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_installed_v1_packs_summed_input_identical(sign):
    packs = _installed()
    if not packs:
        pytest.skip("no installed format-1 sa3/medium packs")
    ps = PackSteering(packs)
    raw = {p.knob_name: sign * (1.0 + 0.37 * i) for i, p in enumerate(packs)}
    steps = 8
    cfgs = ps.build_configs(raw, steps)
    # One config per format-1 pack, carrying the stored tensor itself.
    assert len(cfgs) == len(packs)
    for c, p in zip(cfgs, packs):
        assert c["vector"] is p.vector and c["layer"] == p.block
        assert c["magnitude"] == float(p.magnitude) and c["alpha"] == raw[p.knob_name]
    got = summed_input(cfgs, steps=steps)
    want = torch.zeros(steps, NB, H)
    # Same accumulation order as the slot fill (by layer, then pack order).
    order = sorted({p.block for p in packs}, key=[p.block for p in packs].index)
    for b in order:
        for p in packs:
            if p.block == b:
                want += _v1_reference(p.path, raw[p.knob_name], steps)
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-6)


# ---- format 2 shapes ----------------------------------------------------


def _v2(name="multi", **kw):
    vecs = torch.stack([_unit(10), _unit(11), _unit(12)])
    d = dict(vectors=vecs, blocks=(4, 7, 20), block=4, vector=vecs[0])
    d.update(kw)
    return _base(name, **d)


def test_v2_multi_block_round_trip_and_sum(tmp_path):
    from safetensors import safe_open

    path = save_pack(_v2(category="genre", variant="wide", applies_to=["sa3/medium"],
                         seeds=[1, 2]), tmp_path / "m")
    with safe_open(str(path), framework="pt") as f:
        assert set(f.keys()) == {"vectors", "blocks"}
        assert f.get_tensor("blocks").dtype == torch.int64
        meta = json.loads(f.metadata()["steering_pack"])
    assert meta["format"] == 2 and meta["category"] == "genre"
    p = load_pack(path)
    assert p.is_v2 and p.all_blocks == (4, 7, 20)
    assert p.variant == "wide" and p.applies_to == ["sa3/medium"] and p.seeds == [1, 2]

    ps = PackSteering([p])
    for alpha in (2.5, -2.5):
        got = summed_input(ps.build_configs({"steer_multi": alpha}, 2), steps=2)
        want = torch.zeros(2, NB, H)
        for k, b in enumerate(p.blocks):
            want[:, b, :] += alpha * 2.0 * p.vectors[k]
        assert torch.allclose(got, want, atol=1e-6)
        assert torch.count_nonzero(got[:, [i for i in range(NB) if i not in (4, 7, 20)]]) == 0


def test_v2_raw_rows_use_their_own_norm_and_header_norms():
    raw_rows = torch.stack([_unit(1) * 30.0, _unit(2) * 5.0])
    p = _base("raw", vectors=raw_rows, blocks=(1, 2), vector=raw_rows[0])
    (t0, t1) = p.terms()
    assert t0.pos_magnitude == pytest.approx(3.0) and t1.pos_magnitude == pytest.approx(0.5)
    assert float(t0.pos.norm()) == pytest.approx(1.0, abs=1e-5)
    q = _v2("normed", norms=(10.0, 20.0, 30.0))
    assert [t.pos_magnitude for t in q.terms()] == pytest.approx([1.0, 2.0, 3.0])


def test_v2_negative_directions(tmp_path):
    neg = torch.stack([_unit(20), _unit(21), _unit(22)])
    p = load_pack(save_pack(_v2("bi", vectors_neg=neg), tmp_path / "bi"))
    ps = PackSteering([p])
    pos = summed_input(ps.build_configs({"steer_bi": 3.0}, 1), steps=1)
    got = summed_input(ps.build_configs({"steer_bi": -3.0}, 1), steps=1)
    want = torch.zeros(1, NB, H)
    for k, b in enumerate(p.blocks):
        want[:, b, :] += 3.0 * 2.0 * neg[k]
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got, -pos, atol=1e-3)

    # (c) beside a legacy single vector.
    q = load_pack(save_pack(_base("one", vector_neg=_unit(30)), tmp_path / "one"))
    assert q.is_v2 and q.all_blocks == (3,)
    qs = PackSteering([q])
    got = summed_input(qs.build_configs({"steer_one": -4.0}, 1), steps=1)
    assert torch.allclose(got[0, 3], 4.0 * 2.0 * _unit(30), atol=1e-6)
    got = summed_input(qs.build_configs({"steer_one": 4.0}, 1), steps=1)
    assert torch.allclose(got[0, 3], 4.0 * 2.0 * _unit(0), atol=1e-6)


def test_v2_layout_filter_checks_every_block(tmp_path):
    save_pack(_v2("ok"), tmp_path / "ok")
    save_pack(_v2("far", blocks=(4, 30, 5)), tmp_path / "far")
    lay = SteeringLayout(num_blocks=NB, hidden_size=H)
    names = [p.name for p in discover_packs(tmp_path, family="sa3", checkpoint="medium",
                                            layout=lay)]
    assert names == ["ok"]


def test_v2_malformed_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_pack(_v2("bad", blocks=(1, 2)), tmp_path / "a")          # K mismatch
    with pytest.raises(ValueError):
        save_pack(_v2("bad", vectors_neg=torch.zeros(2, H)), tmp_path / "b")
    with pytest.raises(ValueError):
        save_pack(_base("bad", vector_neg=torch.zeros(5)), tmp_path / "c")


# ---- knob meta ------------------------------------------------------------


def test_knob_meta_category_precedence_and_v2_keys():
    top = _v2("t", category="genre", applies_to=["sa3/medium"], variant="v2a",
              description="Pushes toward drum and bass.", pos_anchor="drum and bass",
              neg_anchor="ambient", provenance={"category": "misc"},
              blurb="'dnb' by CLAP text similarity.")
    prov = _base("p", provenance={"category": "instrument"},
                 blurb="text only unless Essentia is installed")
    legacy = _base("bright", blurb="positive brightens")
    cat = catalog_from_specs(PackSteering([top, prov, legacy]).knob_specs())
    mt, mp, ml = cat["steer_t"]["meta"], cat["steer_p"]["meta"], cat["steer_bright"]["meta"]
    assert mt["category"] == "genre" and mp["category"] == "instrument"
    assert mt["applies_to"] == ["sa3/medium"] and mt["variant"] == "v2a"
    assert mt["blocks"] == [4, 7, 20]
    assert mt["pos_anchor"] == "drum and bass" and mt["neg_anchor"] == "ambient"
    assert "variant" not in mp and "blocks" not in mp
    # Pack description wins; catalogue blurbs never reach the description;
    # a legacy (uncategorised) pack keeps its blurb as before.
    assert cat["steer_t"]["description"] == "Pushes toward drum and bass."
    assert "Essentia" not in cat["steer_p"]["description"]
    assert cat["steer_bright"]["description"].endswith("positive brightens")

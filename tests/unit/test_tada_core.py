"""TADA core (acestep.tada) on tiny synthetic models. CPU only.

Covers: the concept data imported verbatim from the reference code, the
patching sweep finding a planted block, CAA recovering a planted
difference, the CAA intervention matching the reference math, packs
round-tripping and loading as ``steer_<concept>`` knobs, the knob at zero
being a bit-identical no-op through the real StreamPipeline (and nonzero
touching only the cross-attention output of the conditional pass), and
the evaluation port reproducing the paper's published AUC numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import SlotCondition, SlotRequest, StreamPipeline
from acestep.steering.layout import (
    HOOK_CROSS_ATTN_OUTPUT,
    HOOK_POST_BLOCK_RESIDUAL,
    SteeringLayout,
)
from acestep.steering.packs import PackSteering, discover_packs, load_pack
from acestep.tada import concepts
from acestep.tada import metrics as M
from acestep.tada.caa import (
    alphas_from_range,
    combine,
    caa_vectors,
    stack_vectors,
    steer_activation,
)
from acestep.tada.austeer import austeer_scores, austeer_vectors, select_top_s
from acestep.tada.packs import (
    METHOD_AUSTEER,
    METHOD_TADA_CAA,
    austeer_pack,
    caa_pack,
    write_caa_pack,
)
from acestep.tada.patching import (
    impact_score,
    localize,
    patching_sweep,
    run_patched,
    select_layers,
)
from acestep.tada.target import ActivationRecorder, ModuleTarget, forward_counter, frames

H = 8     # hidden size
DC = 6    # text-conditioning width
T = 5     # time frames
NB = 4    # blocks


# ---------------------------------------------------------------------------
# A tiny cross-attention model
# ---------------------------------------------------------------------------

class _XAttn(torch.nn.Module):
    """Stands in for a cross-attention module: its output depends only on
    the conditioning (mean over text tokens, projected), broadcast over the
    query frames, plus a small term in the hidden state."""

    def __init__(self, gain: float, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(gain * torch.randn(DC, H, generator=g), requires_grad=False)

    def forward(self, hidden_states, encoder_hidden_states=None):
        c = encoder_hidden_states.mean(dim=1, keepdim=True) @ self.w  # [B, 1, H]
        return c.expand(-1, hidden_states.shape[1], -1) + 0.01 * hidden_states


class _Block(torch.nn.Module):
    def __init__(self, i: int, gain: float):
        super().__init__()
        self.cross_attn = _XAttn(gain, seed=100 + i)
        # Identity tap after the cross-attention: tests observe the value
        # that actually joins the residual (after any steering hook).
        self.tap = torch.nn.Identity()
        self.k = 0.9 + 0.01 * i

    def forward(self, h, enc):
        h = h + self.tap(self.cross_attn(h, encoder_hidden_states=enc))
        return h * self.k


class _Model(torch.nn.Module):
    def __init__(self, gains=(1.0,) * NB):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block(i, g) for i, g in enumerate(gains))

    def forward(self, x, enc):
        h = x
        for blk in self.layers:
            h = blk(h, enc)
        return h

    def target(self) -> ModuleTarget:
        return ModuleTarget(
            {HOOK_CROSS_ATTN_OUTPUT: [b.cross_attn for b in self.layers]},
            cond_keys=("encoder_hidden_states",),
        )


def _enc(seed: int, n_tok: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n_tok, DC, generator=g)


def _x(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(10_000 + seed)
    return torch.randn(1, T, H, generator=g)


# ---------------------------------------------------------------------------
# Concept data (verbatim from the reference code)
# ---------------------------------------------------------------------------

def test_concept_data_matches_reference_tables():
    assert concepts.STEERING_CONCEPTS == tuple(
        concepts.load("steering_concepts")["concepts"]
    )
    for c in concepts.STEERING_CONCEPTS:
        pos, neg, _ = concepts.prompt_pairs(c)
        assert len(pos) == len(neg) == 50
    pos, neg, lyrics = concepts.prompt_pairs("piano")
    assert pos[0] == "a song, with piano" and neg[0] == "a song" and lyrics == "[inst]"
    pos, neg, lyrics = concepts.prompt_pairs("vocal_gender")
    assert pos[0].endswith(", with female vocal") and neg[0].endswith(", with male vocal")
    assert lyrics == ""
    pos, neg, _ = concepts.prompt_pairs("tempo")
    assert pos[0].startswith("fast song, ") and neg[0].startswith("slow song, ")
    assert concepts.eval_queries("piano")["clap"] == "a piano song"
    prompts, lyr = concepts.benchmark_prompts()
    assert len(prompts) == len(lyr) == 100
    assert len(concepts.benchmark_prompts(holdout=True)[0]) == 20
    assert concepts.benchmark_lyrics("piano") == ["[inst]"] * 100
    cfg = concepts.caa_eval_config("piano", "loc")
    assert cfg["layers"] == "tf6tf7" and cfg["steer-mode"] == "cond_only"
    assert cfg["min-range"] == pytest.approx(-132.0726)
    assert cfg["method-kwargs"] == {"renorm": True}
    assert concepts.localization_concepts("ace") == [
        "female", "male", "fast", "slow", "happy", "sad", "drums", "violin", "reggae", "jazz",
    ]
    assert concepts.localization_eval_prompts("violin")["muqt"] == ["violin-based"]
    assert concepts.similarity_templates()["muqt"] == "A {p} track"


def test_alpha_grid_matches_reference():
    a = alphas_from_range(-132.0726, 121.6698, 15)
    assert len(a) == 31 and a[15] == 0.0
    assert a[0] == pytest.approx(-132.0726) and a[-1] == pytest.approx(121.6698)
    assert a[16] == round(121.6698 / 15, 4)
    assert alphas_from_range(0, 12.12, 15)[0] == 0.0


# ---------------------------------------------------------------------------
# Patching sweep
# ---------------------------------------------------------------------------

def test_patching_sweep_picks_the_planted_block():
    planted = 2
    gains = [0.02] * NB
    gains[planted] = 2.0
    model = _Model(gains)
    target = model.target()
    pairs = [(_enc(2 * i), _enc(2 * i + 1)) for i in range(4)]
    # Concept direction: what the planted layer writes for clean - corrupted.
    w = model.layers[planted].cross_attn.w

    def generate(patch_blocks):
        outs = []
        for i, (clean, corrupt) in enumerate(pairs):
            x = _x(i)
            _, patched = run_patched(
                target, patch_blocks,
                lambda: model(x, clean), lambda: model(x, corrupt),
            )
            outs.append((patched, i))
        return outs

    def similarity(outs):
        vals = []
        for out, i in outs:
            clean, corrupt = pairs[i]
            u = ((clean - corrupt).mean(dim=1) @ w).reshape(-1)
            vals.append(float((out.mean(dim=1).reshape(-1) @ u) / u.norm()))
        return float(np.mean(vals))

    res = patching_sweep("planted", NB, generate, similarity)
    assert res.n_samples == len(pairs)
    best = int(np.nanargmax(res.impacts))
    assert best == planted
    assert res.impacts[planted] > 0.5
    agg, chosen = localize([res])
    assert chosen == [planted]
    assert select_layers([0.05, 0.1, 0.5]) == [1, 2]


def test_patched_run_equals_clean_when_every_block_is_patched():
    model = _Model()
    target = model.target()
    x, clean, corrupt = _x(0), _enc(0), _enc(1)
    c_out, p_out = run_patched(
        target, range(NB), lambda: model(x, clean), lambda: model(x, corrupt),
    )
    assert torch.equal(c_out, p_out)
    _, unpatched = run_patched(target, (), lambda: model(x, clean), lambda: model(x, corrupt))
    assert torch.equal(unpatched, model(x, corrupt))


def test_impact_score_floor_and_no_signal():
    assert impact_score(0.5, 0.0, 1.0) == 0.5
    assert impact_score(-0.2, 0.0, 1.0) == 0.0          # floored
    assert impact_score(1.3, 0.0, 1.0) == pytest.approx(1.3)  # not capped
    assert np.isnan(impact_score(0.3, 0.2, 0.2))


# ---------------------------------------------------------------------------
# CAA
# ---------------------------------------------------------------------------

def _record_cfg_run(model, target, enc, null_enc, x, steps=3):
    """Two sequential forwards per step (conditional, then unconditional),
    the reference ACE-Step CFG layout; only the conditional pass records."""
    rec = ActivationRecorder(target, context=forward_counter(passes_per_step=2, cond_pass=0))
    with rec:
        h = x
        for _ in range(steps):
            vc = model(h, enc)
            model(h, null_enc)
            h = h + 0.1 * vc
    return rec.steps()


def test_caa_vector_equals_planted_difference():
    model = _Model()
    target = model.target()
    delta = _enc(999)[:, :1, :].expand(1, 3, DC) * 0.7   # planted text-space shift
    null = torch.zeros(1, 3, DC)
    pos_runs, neg_runs = [], []
    for i in range(5):
        base = _enc(i)
        pos_runs.append(_record_cfg_run(model, target, base + delta, null, _x(i)))
        neg_runs.append(_record_cfg_run(model, target, base, null, _x(i)))
    vecs = caa_vectors(pos_runs, neg_runs)
    assert sorted(vecs) == [0, 1, 2]
    for b in range(NB):
        planted = (delta.mean(dim=1) @ model.layers[b].cross_attn.w).reshape(-1)
        planted = planted / planted.norm()
        for s in vecs:
            # The hidden-state term (0.01 h) differs slightly between the
            # paired runs from block 1 on; the conditioning term dominates.
            cos = torch.nn.functional.cosine_similarity(vecs[s][b], planted, dim=0)
            assert cos > 0.999, (b, s, float(cos))
            assert float(vecs[s][b].norm()) == pytest.approx(1.0, abs=1e-5)
    # Block 0 sees identical hidden states in both runs: exact.
    p0 = (delta.mean(dim=1) @ model.layers[0].cross_attn.w).reshape(-1)
    assert torch.allclose(vecs[0][0], p0 / p0.norm(), atol=1e-6)
    stacked = stack_vectors(vecs, [1, 2])
    assert stacked.shape == (2, 3, H)


def test_recorder_skips_the_unconditional_pass():
    model = _Model()
    target = model.target()
    run = _record_cfg_run(model, target, _enc(0), torch.zeros(1, 3, DC), _x(0), steps=2)
    assert sorted(run) == [0, 1]
    assert all(len(run[s][b]) == 1 for s in run for b in range(NB))


def test_steer_activation_matches_reference_vectorstore_math():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(1, T, H, generator=g)
    v = torch.randn(H, generator=g)
    v = v / v.norm()
    alpha = 20.0
    # Reference: vector + alpha * sv.view(1,1,-1).expand(1, T, -1)
    ref = h + alpha * v.view(1, 1, -1).expand(1, T, -1)
    assert torch.allclose(steer_activation(h, v, alpha), ref)
    norm = torch.norm(h, dim=2, keepdim=True)
    ref_r = ref / torch.norm(ref, dim=2, keepdim=True) * norm
    out_r = steer_activation(h, v, alpha, renorm=True)
    assert torch.allclose(out_r, ref_r, atol=1e-6)
    assert steer_activation(h, v, 0.0) is h


# ---------------------------------------------------------------------------
# Packs
# ---------------------------------------------------------------------------

def _tada_pack(blocks=(1, 2), steps=4, name="piano", **kw):
    g = torch.Generator().manual_seed(7)
    v = torch.randn(len(blocks), steps, H, generator=g)
    v = v / v.norm(dim=-1, keepdim=True)
    return caa_pack(
        v, family="fake", checkpoint="tiny", concept=name, blocks=list(blocks),
        magnitude=kw.pop("magnitude", 10.0), provenance={"pairs": 50}, **kw,
    )


def test_pack_round_trip_and_discovery(tmp_path: Path):
    pack = _tada_pack()
    path = write_caa_pack(pack, tmp_path)
    back = load_pack(path)
    assert back.blocks == [1, 2] and back.block == 1
    assert back.hook == HOOK_CROSS_ATTN_OUTPUT and back.method == METHOD_TADA_CAA
    assert back.cond_only and not back.renorm
    assert torch.equal(back.vector, pack.vector)
    assert back.provenance["pairs"] == 50 and "2602.11910" in back.provenance["citation"]

    post = SteeringLayout(num_blocks=NB, hidden_size=H)
    xattn = SteeringLayout(num_blocks=NB, hidden_size=H,
                           extra_hooks=(HOOK_CROSS_ATTN_OUTPUT,))
    found = discover_packs(tmp_path, family="fake", checkpoint="tiny", layout=xattn)
    assert [p.knob_name for p in found] == ["steer_piano"]
    # A layout without the cross-attention hook point does not take it.
    assert discover_packs(tmp_path, family="fake", checkpoint="tiny", layout=post) == []
    # Too few blocks for block 2: rejected.
    small = SteeringLayout(num_blocks=2, hidden_size=H,
                           extra_hooks=(HOOK_CROSS_ATTN_OUTPUT,))
    assert discover_packs(tmp_path, family="fake", checkpoint="tiny", layout=small) == []

    surf = PackSteering(found)
    spec = surf.knob_specs()[0]
    assert spec.name == "steer_piano" and spec.group == "steering"
    assert "cross-attention output of blocks 1,2" in spec.description


def test_pack_configs_map_steps_by_position():
    pack = _tada_pack(steps=4)
    surf = PackSteering([pack])
    assert surf.build_configs({"steer_piano": 0.0}, 8) == []
    cfgs = surf.build_configs({"steer_piano": 1.5}, 8)
    # One config per localised block, its vector one row per live step.
    assert [c["layer"] for c in cfgs] == [1, 2]
    for k, c in enumerate(cfgs):
        assert c["hook"] == HOOK_CROSS_ATTN_OUTPUT and c["cond_only"] and not c["renorm"]
        assert tuple(c["vector"].shape) == (8, H)
        for i in range(8):
            assert torch.equal(c["vector"][i], pack.vector[k, min(3, int(i * 4 / 8))])
        assert c["alpha"] * c["magnitude"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Through the real StreamPipeline
# ---------------------------------------------------------------------------

class _CfgTrunk(torch.nn.Module):
    """velocity = model(x, enc); records cross-attention outputs per call."""

    def __init__(self):
        super().__init__()
        self.model = _Model()
        self.seen: list = []
        for i, b in enumerate(self.model.layers):
            b.tap.register_forward_hook(self._spy(i))

    def _spy(self, i):
        def hook(_m, _inp, out):
            self.seen.append((i, out.detach().clone()))
        return hook

    def forward(self, x, enc):
        return self.model(x, enc)


class _FakeAdapter:
    name = "fake"
    latent_channels = H
    latent_rate_hz = 10.0
    sample_rate = 16000
    accepts_steering = False

    def __init__(self):
        self.trunk = _CfgTrunk()

    def build_schedule(self, config, denoise, device, dtype):
        return torch.linspace(float(denoise), 0.0, config.infer_steps + 1)

    def request_frames(self, request):
        return int(request.latent_frames)

    def request_device_dtype(self, request):
        return torch.device("cpu"), torch.float32

    def steering_layout(self):
        return SteeringLayout(num_blocks=NB, hidden_size=H,
                              extra_hooks=(HOOK_CROSS_ATTN_OUTPUT,))

    def steering_blocks(self):
        return self.trunk.model.layers

    def steering_hook_modules(self, hook):
        if hook == HOOK_CROSS_ATTN_OUTPUT:
            return [b.cross_attn for b in self.trunk.model.layers]
        return None

    def batched_forward(self, xt_batch, timestep_list, enc_list, mask_list,
                        ctx_list, aux_list):
        enc = torch.cat(enc_list, dim=0)
        return self.trunk(xt_batch, enc)


def _pipe(adapter, steps=4):
    cfg = DiffusionConfig(infer_steps=steps, infer_method="ode", noise_on_cpu=True,
                          dcw_enabled=False)
    return StreamPipeline(None, cfg, pipeline_depth=1, adapter=adapter)


def _cfg_req(seed):
    enc = _enc(seed)
    return SlotRequest(
        seed=seed, denoise=1.0, aux_cond={}, latent_frames=T,
        encoder_hidden_states=enc, encoder_attention_mask=torch.ones(1, enc.shape[1]),
        neg_conditions=[SlotCondition(
            encoder_hidden_states=torch.zeros_like(enc),
            encoder_attention_mask=torch.ones(1, enc.shape[1]),
        )],
        guidance_curve=torch.full((T,), 3.0),
    )


def _drain(pipe, seeds, ticks=8):
    out = []
    q = list(seeds)
    for _ in range(ticks):
        if q:
            pipe.submit(_cfg_req(q.pop(0)))
        fin = pipe.tick()
        if fin is not None:
            out.append(fin.clone())
    return out


def test_zero_strength_is_a_bit_identical_noop_through_streampipeline():
    pack = _tada_pack(blocks=(1, 2), steps=4)
    surf = PackSteering([pack])

    base = _drain(_pipe(_FakeAdapter()), [1, 2])
    assert base

    pipe = _pipe(_FakeAdapter())
    pipe.set_steering(surf.build_configs({"steer_piano": 0.0}, 4))
    same = _drain(pipe, [1, 2])
    assert len(same) == len(base)
    for a, b in zip(base, same):
        assert torch.equal(a, b)

    pipe = _pipe(_FakeAdapter())
    pipe.set_steering(surf.build_configs({"steer_piano": 2.0}, 4))
    moved = _drain(pipe, [1, 2])
    assert any(not torch.equal(a, b) for a, b in zip(base, moved))
    pipe.set_steering([])
    assert not pipe._steering_by_layer


def test_nonzero_steers_only_cross_attention_on_the_conditional_pass():
    pack = _tada_pack(blocks=(1,), steps=4, renorm=False)
    surf = PackSteering([pack])
    a0, a1 = _FakeAdapter(), _FakeAdapter()
    _drain(_pipe(a0), [3])
    p1 = _pipe(a1)
    p1.set_steering(surf.build_configs({"steer_piano": 1.0}, 4))
    _drain(p1, [3])
    seen0, seen1 = a0.trunk.seen, a1.trunk.seen
    assert len(seen0) == len(seen1) and seen0
    # The first two forwards of the request (conditional pass, then the CFG
    # negative pass) see identical inputs in both pipelines, so every
    # difference in them is the intervention itself.
    for f in (0, 1):
        for i in range(NB):
            b0, o0 = seen0[f * NB + i]
            b1, o1 = seen1[f * NB + i]
            assert b0 == b1 == i
            if i == 0 or f == 1:
                assert torch.equal(o0, o1)      # upstream, or the negative pass
            elif i == 1:
                # Conditional pass, steered block: + alpha * magnitude * v_0.
                shift = o1 - o0
                ref = (10.0 * pack.vector[0, 0]).expand_as(shift)
                assert torch.allclose(shift, ref, atol=1e-5)


def test_renorm_keeps_steered_frame_norms():
    pack = _tada_pack(blocks=(1,), steps=4, renorm=True)
    surf = PackSteering([pack])
    a0, a1 = _FakeAdapter(), _FakeAdapter()
    _drain(_pipe(a0), [5])
    p1 = _pipe(a1)
    p1.set_steering(surf.build_configs({"steer_piano": 3.0}, 4))
    _drain(p1, [5])
    # First forward of the request: identical inputs up to block 1, so the
    # steered block-1 output is the reference intervention on the plain one.
    _, plain = a0.trunk.seen[1]
    _, steered = a1.trunk.seen[1]
    ref = steer_activation(plain, pack.vector[0, 0], 30.0, renorm=True)
    assert torch.allclose(steered, ref, atol=1e-5)
    assert torch.allclose(steered.norm(dim=-1), plain.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(steered, plain)


# ---------------------------------------------------------------------------
# Evaluation port
# ---------------------------------------------------------------------------

DATA = Path(__file__).parent / "data" / "tada_published_piano.csv"


def _cutoff(df, d):
    vals = []
    for c in ("pci_all", "pci_loc"):
        x = df[df.cell == c]
        x = x[x.alpha >= 0] if d == "pos" else x[x.alpha <= 0]
        vals.append(x.lpaps.max())
    return M.lpaps_cutoff(vals)


@pytest.mark.parametrize("cell,muq,clap,smooth", [
    # Reference results/tables_lpaps.md, piano rows "CAA (loc.)" and "CAA".
    ("caa_loc", 0.107, 0.053, 0.042),
    ("caa_all", 0.052, 0.025, 0.052),
])
def test_auc_and_smoothness_reproduce_published_piano(cell, muq, clap, smooth):
    df = pd.read_csv(DATA)
    cp, cn = _cutoff(df, "pos"), _cutoff(df, "neg")
    assert cp == pytest.approx(3.804, abs=5e-4) and cn == pytest.approx(3.492, abs=5e-4)
    x = df[df.cell == cell].sort_values("alpha")

    def both(fn, col):
        return (fn(x.alpha, x.lpaps, x[col], cp, "pos") + fn(x.alpha, x.lpaps, x[col], cn, "neg")) / 2

    assert round(both(M.alignment_auc, "muqt"), 3) == muq
    assert round(both(M.alignment_auc, "clap"), 3) == clap
    assert round(both(M.smoothness, "muqt"), 3) == smooth


def test_lpaps_math_and_windowing():
    g = torch.Generator().manual_seed(0)
    f = [torch.randn(2, 6, 4, generator=g) for _ in range(4)]
    assert torch.allclose(M.lpaps_from_features(f, f), torch.zeros(2))
    f2 = [x + 0.5 for x in f]
    assert bool((M.lpaps_from_features(f, f2) > 0).all())
    a = np.zeros((1, 100))
    calls = []
    M.windowed(lambda u, v: calls.append(u.shape[-1]) or 1.0, a, a, 4, 4, win_s=10)
    # 40-sample windows stepped by 36: starts 0, 36, 72.
    assert calls == [40, 40, 28]


def test_quality_sampling_and_mir_axes():
    alphas = [-2, -1, 0, 1, 2]
    lp = [2.0, 1.0, 0.0, 1.0, 2.0]
    q = [5.0, 6.0, 7.0, 6.0, 5.0]
    assert M.quality_at_lpaps(alphas, lp, q, cutoff=2.0, direction="pos", n_points=3) == pytest.approx(6.0)
    c = np.abs(np.random.default_rng(0).normal(size=(12, 20)))
    assert M.harmony_distance(c, c) == pytest.approx(0.0, abs=1e-6)
    assert M.structure_distance(c, c) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# AUSteer and multi-concept combination
# ---------------------------------------------------------------------------

def _planted_momentum_runs(pairs=6, frames=5, steps=2):
    """Paired recordings (``reduce=frames`` layout) with planted sign-stable
    dimensions: block 0 dim 2 always up, block 1 dim 5 always down, block 1
    dim 0 up on all but one sample; everything else is noise."""
    g = torch.Generator().manual_seed(3)
    pos_runs, neg_runs = [], []
    for i in range(pairs):
        pos, neg = {}, {}
        for s in range(steps):
            pos[s], neg[s] = {}, {}
            for b in range(2):
                base = torch.randn(frames, H, generator=g)
                m = torch.randn(frames, H, generator=g)
                if b == 0:
                    m[:, 2] = m[:, 2].abs() + 0.1
                else:
                    m[:, 5] = -(m[:, 5].abs() + 0.1)
                    m[:, 0] = m[:, 0].abs() + 0.1
                    if i == 0:
                        m[0, 0] = -1.0
                pos[s][b] = [base + m]
                neg[s][b] = [base]
        pos_runs.append(pos)
        neg_runs.append(neg)
    return pos_runs, neg_runs


def _reference_betas(pos_list, neg_list):
    """``compute_austeer_scores`` from the reference code, for one layer."""
    mom = np.concatenate([p.numpy() - n.numpy() for p, n in zip(pos_list, neg_list)], axis=0)
    n = mom.shape[0]
    r_pos = (mom > 0).sum(axis=0) / n
    r_neg = (mom < 0).sum(axis=0) / n
    scores = np.maximum(r_pos, r_neg)
    return np.where(r_pos >= r_neg, scores, -scores)


def test_austeer_selects_planted_sign_consistent_dimensions():
    pos_runs, neg_runs = _planted_momentum_runs()
    betas = austeer_scores(pos_runs, neg_runs)
    for s in betas:
        for b in betas[s]:
            ref = _reference_betas([r[s][b][0] for r in pos_runs], [r[s][b][0] for r in neg_runs])
            assert np.allclose(betas[s][b].numpy(), ref)
            assert float(betas[s][b].abs().min()) >= 0.5 and float(betas[s][b].abs().max()) <= 1.0
    assert float(betas[0][0][2]) == 1.0 and float(betas[0][1][5]) == -1.0
    assert float(betas[0][1][0]) == pytest.approx(29 / 30)

    sel = select_top_s(betas, s=2)
    for s in sel:
        nz = {(b, int(d)) for b in sel[s] for d in torch.nonzero(sel[s][b]).reshape(-1)}
        assert nz == {(0, 2), (1, 5)}
        assert float(sel[s][0][2]) == 1.0 and float(sel[s][1][5]) == -1.0
    sel3 = select_top_s(betas, s=3)
    assert {(b, int(d)) for b in sel3[0] for d in torch.nonzero(sel3[0][b]).reshape(-1)} == {
        (0, 2), (1, 5), (1, 0)}
    # Localised: only block 1 competes for the budget.
    loc = select_top_s(betas, s=2, blocks=[1])
    assert sorted(loc[0]) == [1]
    assert {int(d) for d in torch.nonzero(loc[0][1]).reshape(-1)} == {5, 0}
    assert torch.equal(austeer_vectors(pos_runs, neg_runs, s=2)[1][0], sel[1][0])
    # Ties keep block order, then dimension order (the reference stable sort).
    tie = select_top_s({0: {0: torch.full((H,), -0.5), 1: torch.full((H,), 0.5)}}, s=3)
    assert torch.nonzero(tie[0][0]).reshape(-1).tolist() == [0, 1, 2]
    assert not tie[0][1].any()


def test_austeer_scores_from_recorded_frames():
    model = _Model()
    target = model.target()

    def run(enc, x):
        rec = ActivationRecorder(target, context=forward_counter(1), reduce=frames)
        with rec:
            for _ in range(2):
                model(x, enc)
        return rec.steps()

    pos = [run(_enc(i) + 0.5, _x(i)) for i in range(3)]
    neg = [run(_enc(i), _x(i)) for i in range(3)]
    assert tuple(pos[0][0][0][0].shape) == (T, H)
    betas = austeer_scores(pos, neg)
    assert sorted(betas) == [0, 1] and sorted(betas[0]) == list(range(NB))
    # The conditioning shift is the same for every frame and pair: every
    # dimension of block 0 is perfectly sign-consistent.
    assert torch.all(betas[0][0].abs() == 1.0)


def test_austeer_pack_round_trip_and_zero_strength_noop(tmp_path: Path):
    pos_runs, neg_runs = _planted_momentum_runs(steps=4)
    sel = select_top_s(austeer_scores(pos_runs, neg_runs), s=2)
    vec = stack_vectors({s: {1: sel[s][0], 2: sel[s][1]} for s in sel}, [1, 2])
    pack = austeer_pack(vec, top_s=2, family="fake", checkpoint="tiny",
                        concept="piano", blocks=[1, 2], magnitude=10.0)
    back = load_pack(write_caa_pack(pack, tmp_path))
    assert back.method == METHOD_AUSTEER == "auscore"
    assert back.provenance["top_s"] == 2 and back.cond_only and not back.renorm
    assert torch.equal(back.vector, vec)

    surf = PackSteering([back])
    base = _drain(_pipe(_FakeAdapter()), [1, 2])
    pipe = _pipe(_FakeAdapter())
    pipe.set_steering(surf.build_configs({"steer_piano": 0.0}, 4))
    same = _drain(pipe, [1, 2])
    assert base and len(same) == len(base)
    assert all(torch.equal(a, b) for a, b in zip(base, same))
    pipe = _pipe(_FakeAdapter())
    pipe.set_steering(surf.build_configs({"steer_piano": 2.0}, 4))
    assert any(not torch.equal(a, b) for a, b in zip(base, _drain(pipe, [1, 2])))


def test_multi_concept_combination_is_unit_weight_sum_with_negation():
    g = torch.Generator().manual_seed(1)
    a, b, c = (torch.randn(2, 3, H, generator=g) for _ in range(3))
    assert torch.allclose(combine([a, b]), a + b)
    assert torch.allclose(combine([a, b, c], negate=[False, True, False]), a - b + c)
    assert torch.allclose(combine([a]), a)
    with pytest.raises(ValueError):
        combine([a, torch.zeros(H)])
    with pytest.raises(ValueError):
        combine([a, b], negate=[True])


def test_austeer_released_budgets_are_data():
    assert concepts.austeer_eval_config("piano", "loc")["method-kwargs"]["k"] == 4096
    assert concepts.austeer_eval_config("violin", "all")["method-kwargs"]["k"] == 256
    assert concepts.austeer_eval_config("piano", "loc")["layers"] == "tf6tf7"

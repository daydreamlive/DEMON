"""TADA SAE port (acestep.tada.sae) on synthetic data. CPU only.

Covers: the BatchTopK SAE and trainer recovering a planted sparse
dictionary; held-out FVU per bucket with the absolute gate failing a
broken dictionary; the sweep choice; generation-time token capture
(conditional rows, every k-th step) and the on-disk cache; TF-IDF scoring
ranking a planted concept feature first, through hooks on a target;
``v_SAE`` construction; packs round-tripping as ``steer_<concept>`` knobs
with zero strength a no-op.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from acestep.steering.packs import PackSteering, load_pack
from acestep.tada.caa import steer_activation
from acestep.tada.sae import (
    ActivationStore,
    FeatureMeanRecorder,
    Sae,
    SaeConfig,
    SaeTrainer,
    TokenRecorder,
    TrainConfig,
    absolute_bucket_gate,
    cache_generations,
    choose_config,
    concept_vectors,
    fvu_by_bucket,
    pooled_scores,
    sae_pack,
    sae_vector,
    score_tables,
    select_k,
    stack_blocks,
    sweep_configs,
    tfidf,
    top_features,
    write_sae_pack,
)
from acestep.tada.sae.packs import METHOD_TADA_SAE
from acestep.tada.sae.train import linear_schedule, sample_batches
from acestep.tada.target import HOOK_CROSS_ATTN_OUTPUT, ModuleTarget, forward_counter

D = 16       # activation width
F_TRUE = 24  # planted dictionary size
SEQ = 8      # tokens per sample


def _planted(n_samples: int, seed: int = 0, active: int = 2):
    """Samples whose tokens are sparse positive mixes of ``F_TRUE`` unit
    atoms plus a constant offset; returns ``(acts [n, SEQ, D], atoms)``."""
    g = torch.Generator().manual_seed(seed)
    atoms = torch.randn(F_TRUE, D, generator=g)
    atoms = atoms / atoms.norm(dim=1, keepdim=True)
    n_tok = n_samples * SEQ
    idx = torch.stack([torch.randperm(F_TRUE, generator=g)[:active] for _ in range(n_tok)])
    coef = 1.0 + torch.rand(n_tok, active, generator=g)
    x = (coef[..., None] * atoms[idx]).sum(1) + 0.5
    return x.reshape(n_samples, SEQ, D), atoms


def _train(acts, cfg_sae, steps=1500, lr=4e-3, seed=0):
    torch.manual_seed(seed)
    sae = Sae(D, cfg_sae)
    cfg = TrainConfig(effective_batch_size=256, lr=lr, lr_warmup_steps=50,
                      num_epochs=1000, device="cpu", dead_feature_threshold=20_000)
    stats = SaeTrainer({"s": sae}, cfg).fit(acts, max_steps=steps)
    return sae, stats["s"]


# ---------------------------------------------------------------------------
# model + trainer
# ---------------------------------------------------------------------------

def test_planted_dictionary_recovered():
    allacts, atoms = _planted(1700)
    acts, held = allacts[:1500], allacts[1500:]  # held-out samples, same dictionary
    sae, st = _train(acts, SaeConfig(expansion_factor=4, k=2, batch_topk=True))
    cos = torch.nn.functional.normalize(sae.W_dec.detach(), dim=1) @ atoms.T  # [L, F_TRUE]
    best = cos.max(dim=0).values
    assert (best > 0.9).float().mean() >= 0.9, best
    assert st.fvu < 0.15
    assert 0.0 <= st.dead_pct <= 1.0 and 0.0 <= st.fire_pct <= 1.0
    buckets = np.repeat(np.arange(4), 50)
    bf = fvu_by_bucket(sae, held, buckets, device="cpu")
    assert set(bf) == {0, 1, 2, 3, "all"}
    assert absolute_bucket_gate(bf, bar=0.5)


def test_absolute_gate_fails_broken_dictionary():
    held, _ = _planted(100, seed=3)
    broken = Sae(D, SaeConfig(expansion_factor=4, k=2, batch_topk=True))
    with torch.no_grad():
        broken.W_dec.mul_(25.0)  # blows the reconstruction up
    bf = fvu_by_bucket(broken, held, np.zeros(100, dtype=int), device="cpu")
    assert not absolute_bucket_gate(bf, bar=0.5)
    assert not absolute_bucket_gate({0: 0.2, 1: float("nan")})


def test_reference_mechanics():
    sae = Sae(D, SaeConfig(expansion_factor=2, k=3, batch_topk=True))
    assert sae.num_latents == 2 * D
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(sae.num_latents), atol=1e-5)
    assert torch.equal(sae.encoder.bias, torch.zeros(sae.num_latents))
    x = torch.randn(4, SEQ, D)
    out = sae(x)
    # BatchTopK keeps k * rows activations over the whole batch.
    assert int((out.latent_acts != 0).sum()) <= 3 * 4 * SEQ
    assert out.latent_indices.shape == (4 * SEQ, 3)
    # AuxK only with dead latents.
    dead = torch.ones(sae.num_latents, dtype=torch.bool)
    assert float(sae(x, dead_mask=dead).auxk_loss) > 0
    assert float(sae(x, dead_mask=~dead).auxk_loss) == 0
    assert linear_schedule(0, 10, 100) == 0.0
    assert linear_schedule(10, 10, 100) == 1.0
    assert linear_schedule(100, 10, 100) == 0.0
    b = sample_batches(100, SEQ, 4096 // 64 * SEQ, seed=42)
    assert all(len(i) == 64 for i in b) and len(b) == 1


def test_save_load_reference_layout(tmp_path):
    sae = Sae(D, SaeConfig(expansion_factor=2, k=4, batch_topk=True))
    p = sae.save_to_disk(tmp_path / "tf11")
    assert (p / "cfg.json").exists() and (p / "sae.safetensors").exists()
    back = Sae.load_from_disk(p)
    assert torch.equal(back.W_dec, sae.W_dec) and back.cfg.k == 4
    assert back.cfg.batch_topk is False  # the reference switches to per-token TopK


def test_choose_config_and_sweep_grid():
    grid = sweep_configs()
    assert len(grid) == 12 and all(c.batch_topk for c in grid)
    rows = [
        {"name": "a", "val_fvu": 0.10, "dead_pct": 0.05, "fire_pct": 0.1, "gate": True},
        {"name": "b", "val_fvu": 0.20, "dead_pct": 0.001, "fire_pct": 0.2, "gate": True},
        {"name": "c", "val_fvu": 0.15, "dead_pct": 0.001, "fire_pct": 0.2, "gate": False},
        {"name": "d", "val_fvu": 0.18, "dead_pct": 0.002, "fire_pct": 0.4, "gate": True},
    ]
    assert choose_config(rows)["name"] == "b"
    assert choose_config(rows[:1]) is None


# ---------------------------------------------------------------------------
# capture during generation
# ---------------------------------------------------------------------------

class _XA(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x, context=None):
        return x @ self.w + context.mean(1, keepdim=True)


class _Model(torch.nn.Module):
    """Two blocks with a cross-attention module each; ``generate`` runs
    ``steps`` denoise steps as ``[uncond; cond]`` row pairs."""

    def __init__(self, steps=6):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.xa = torch.nn.ModuleList([_XA(torch.randn(D, D, generator=g) * 0.1) for _ in range(2)])
        self.steps = steps

    def generate(self, ctx, seed):
        g = torch.Generator().manual_seed(seed)
        b = ctx.shape[0]
        x = torch.randn(b, SEQ, D, generator=g)
        for _ in range(self.steps):
            xb = torch.cat([x, x])
            cb = torch.cat([torch.zeros_like(ctx), ctx])
            for m in self.xa:
                xb = xb + m(xb, context=cb)
            x = 0.5 * (xb[:b] + xb[b:])
        return x


def _target(model):
    return ModuleTarget({HOOK_CROSS_ATTN_OUTPUT: list(model.xa)}, cond_keys=("context",))


class _CondRows:
    """Context for ``[uncond; cond]`` rows in one forward: the second half."""

    def __init__(self):
        self.forwards = -1

    def tick(self):
        self.forwards += 1

    def __call__(self, batch):
        return [(slice(batch // 2, batch), self.forwards)]


def test_token_recorder_every_k_cond_rows(tmp_path):
    model = _Model(steps=6)
    ctx = torch.randn(1, 3, D)
    rec = TokenRecorder(_target(model), [0, 1], every=2, context=_CondRows(),
                        sigma_fn=lambda: 0.5)
    with rec:
        model.generate(ctx, seed=3)
    acts, steps, sigmas, rows = rec.samples(1)
    assert rows == [0, 0, 0]
    assert steps == [0, 2, 4] and acts.shape == (3, SEQ, D) and acts.dtype == torch.float16
    assert sigmas == [0.5, 0.5, 0.5]
    # The recorded rows are the conditional ones: recompute step 0 by hand.
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, SEQ, D, generator=g)
    h0 = model.xa[0](x, context=ctx)
    xb1 = x + h0
    h1 = model.xa[1](xb1, context=ctx)
    assert torch.allclose(acts[0].float(), h1[0], atol=1e-2)

    store = ActivationStore(tmp_path / "cache")
    prompts = [torch.randn(1, 3, D) for _ in range(4)]
    n = cache_generations(
        lambda batch, s: model.generate(torch.cat(list(batch)), s), prompts,
        lambda: TokenRecorder(_target(model), [0, 1], every=3, context=_CondRows()),
        store, seeds=[10, 11], batch_size=2,
    )
    assert n == 4 and store.blocks() == [0, 1]
    a, meta = store.load(0)
    assert a.shape == (8, SEQ, D)
    assert meta["prompt"].tolist() == [0, 1, 0, 1, 2, 3, 2, 3]
    assert meta["step"].tolist() == [0, 0, 3, 3, 0, 0, 3, 3]
    a2, _ = store.load(0, prompts=[2])
    assert a2.shape == (2, SEQ, D) and torch.equal(a2, a[[4, 6]])


# ---------------------------------------------------------------------------
# scoring and vectors
# ---------------------------------------------------------------------------

def _identity_sae():
    """An SAE whose encoder reads coordinate ``j`` as feature ``j``."""
    sae = Sae(D, SaeConfig(num_latents=D, k=4))
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(D))
        sae.encoder.bias.zero_()
        sae.W_dec.copy_(torch.eye(D))
        sae.b_dec.zero_()
    return sae


def test_tfidf_ranks_planted_concept_feature_first():
    sae = _identity_sae()

    class _Out(torch.nn.Module):
        def forward(self, x, context=None):
            return x + context

    mods = [_Out()]
    target = ModuleTarget({HOOK_CROSS_ATTN_OUTPUT: mods})
    g = torch.Generator().manual_seed(0)

    sig = {"v": 1.0}

    def run(planted: bool, n=8, steps=3):
        rec = FeatureMeanRecorder(target, {0: sae}, context=forward_counter(1),
                                  sigma_fn=lambda: sig["v"])
        with rec:
            for _ in range(n):
                rec.reset_steps()
                for _s in range(steps):
                    sig["v"] = 1.0 - 0.25 * _s
                    base = torch.rand(1, SEQ, D, generator=g) * 0.5  # shared "music"
                    ctx = torch.zeros(1, SEQ, D)
                    if planted:
                        ctx[..., 5] = 2.0   # the concept feature
                    mods[0](base, context=ctx)
        assert rec.step_sigmas() == [1.0, 0.75, 0.5]
        return rec.result()[0]

    mp, mn = run(True), run(False)
    assert mp.shape == (3, D)
    per_step = top_features(tfidf(mp, mn), 1)
    assert all(v == [5] for v in per_step.values())
    assert top_features(pooled_scores(mp, mn), 1) == [5]
    tables = score_tables(mp, mn)
    assert set(tables) == {"tfidf", "diff", "mean_pos"}
    assert int(tables["tfidf"][0].argmax()) == 5
    # Eq. 12 exactly
    assert torch.allclose(tfidf(torch.tensor([2.0]), torch.tensor([0.0])),
                          torch.tensor([2.0 * math.log(1 + 1 / 1e-6)]))


def test_sae_vector_and_blocks():
    W = torch.randn(10, D)
    assert torch.allclose(sae_vector(W, [1, 3]), W[1] + W[3])
    per = sae_vector(W, {0: [1], 1: [2, 4]})
    assert per.shape == (2, D) and torch.allclose(per[1], W[2] + W[4])
    sae = _identity_sae()
    mp = torch.zeros(3, D)
    mp[:, 7] = 1.0
    mn = torch.zeros(3, D)
    v = concept_vectors({4: sae}, {4: mp}, {4: mn}, {4: 1})
    assert v[4].shape == (3, D) and torch.equal(v[4][0], torch.eye(D)[7])
    vp = concept_vectors({4: sae}, {4: mp}, {4: mn}, {4: 1}, per_step=False)
    assert vp[4].shape == (D,)
    st = stack_blocks({4: v[4], 5: v[4]}, [4, 5])
    assert st.shape == (2, 3, D)
    assert stack_blocks({4: vp[4]}, [4]).shape == (1, 1, D)


def test_select_k_grid():
    best, table = select_k([11, 12], lambda ks: -abs(ks[11] - 20) - abs(ks[12] - 100))
    assert best == {11: 20, 12: 100} and len(table) == 36


# ---------------------------------------------------------------------------
# packs and zero strength
# ---------------------------------------------------------------------------

def test_pack_roundtrip_knob_and_zero_noop(tmp_path):
    vec = torch.randn(2, 4, D)
    pack = sae_pack(vec, family="sa3", checkpoint="sa3-medium", concept="piano",
                    blocks=[11, 12], k_per_block={11: 5, 12: 20},
                    sigmas=[1.0, 0.9, 0.5, 0.1])
    assert pack.method == METHOD_TADA_SAE and pack.hook == HOOK_CROSS_ATTN_OUTPUT
    assert pack.cond_only and pack.renorm
    path = write_sae_pack(pack, tmp_path)
    back = load_pack(path)
    assert torch.equal(back.vector, pack.vector) and back.blocks == [11, 12]
    assert back.provenance["k_c"] == {"11": 5, "12": 20}
    assert back.provenance["sigmas"] == [1.0, 0.9, 0.5, 0.1]
    with pytest.raises(ValueError):
        sae_pack(vec, family="sa3", checkpoint="x", concept="c", blocks=[11, 12],
                 k_per_block={11: 5, 12: 5}, sigmas=[1.0])
    ps = PackSteering([back])
    assert [s.name for s in ps.knob_specs()] == ["steer_piano"]
    assert ps.build_configs({"steer_piano": 0.0}, 8) == []
    cfgs = ps.build_configs({"steer_piano": 2.0}, 8)
    assert {c["layer"] for c in cfgs} == {11, 12}
    assert all(c["hook"] == HOOK_CROSS_ATTN_OUTPUT and c["cond_only"] for c in cfgs)
    # The intervention itself is bit-identical at zero strength.
    h = torch.randn(1, SEQ, D)
    assert steer_activation(h, vec[0, 0], 0.0, renorm=True) is h
    single = sae_pack(torch.randn(1, 1, D), family="sa3", checkpoint="sa3-medium",
                      concept="tempo", blocks=[12], k_per_block={12: 10})
    assert single.provenance["per_step"] is False
    with pytest.raises(ValueError):
        sae_pack(torch.randn(4, D), family="sa3", checkpoint="x", concept="c",
                 blocks=[1], k_per_block={1: 5})


# ---------------------------------------------------------------------------
# audio tokens only (memory tokens and padding excluded)
# ---------------------------------------------------------------------------

def test_token_select_excludes_memory_rows():
    from acestep.tada.sae import select_tokens

    hs = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    shared = torch.tensor([False, False, True, True, True])
    assert torch.equal(select_tokens(hs, lambda h: shared), hs[:, 2:])
    per_row = torch.tensor([[False, True, True, True, True], [False, False, True, True, True]])
    assert select_tokens(hs, lambda h: per_row).shape == (7, 3)
    assert select_tokens(hs, None) is hs


def test_recorders_skip_memory_tokens():
    mem = 3

    class _WithMemory(torch.nn.Module):
        """Output = [memory rows carrying feature 9 hugely; audio rows]."""

        def forward(self, x, context=None):
            m = torch.zeros(x.shape[0], mem, D)
            m[..., 9] = 50.0
            return torch.cat([m, x + context], dim=1)

    mods = [_WithMemory()]
    target = ModuleTarget({HOOK_CROSS_ATTN_OUTPUT: mods})
    keep = lambda h: torch.arange(h.shape[1]) >= mem  # noqa: E731
    rec = TokenRecorder(target, [0], tokens=keep)
    with rec:
        mods[0](torch.zeros(2, SEQ, D), context=torch.ones(2, SEQ, D))
    acts, _, _, rows = rec.samples(0)
    assert acts.shape == (2, SEQ, D) and rows == [0, 1]
    assert float(acts[..., 9].abs().max()) == 1.0  # no memory row leaked in

    sae = _identity_sae()

    def run(planted, tokens):
        r = FeatureMeanRecorder(target, {0: sae}, context=forward_counter(1), tokens=tokens)
        with r:
            ctx = torch.zeros(1, SEQ, D)
            if planted:
                ctx[..., 5] = 2.0
            mods[0](torch.rand(1, SEQ, D) * 0.1, context=ctx)
        return r.result()[0]

    with_mem = tfidf(run(True, None), run(False, None))[0]
    audio_only = tfidf(run(True, keep), run(False, keep))[0]
    assert int(audio_only.argmax()) == 5
    assert float(run(True, keep)[0, 9]) < 1.0 < float(run(True, None)[0, 9])
    assert with_mem.shape == audio_only.shape


def test_sa3_audio_token_mask():
    from acestep.engine.sa3_tada_tokens import SA3AudioTokens

    class _Trunk(torch.nn.Module):
        num_memory_tokens = 4

        def forward(self, x, padding_mask=None):
            return x

    t = _Trunk()
    sel = SA3AudioTokens(t)
    hs = torch.zeros(2, 4 + 6, D)
    t(torch.zeros(1), padding_mask=None)
    m = sel(hs)
    assert m.shape == (2, 10) and not m[:, :4].any() and m[:, 4:].all()
    pm = torch.tensor([[True] * 5 + [False]])
    t(torch.zeros(1), padding_mask=pm)
    m = sel(hs)
    assert m[:, 4:].sum().item() == 10 and not m[:, -1].any()
    sel.remove()
    with pytest.raises(ValueError):
        SA3AudioTokens(t, num_memory_tokens=20)(hs)

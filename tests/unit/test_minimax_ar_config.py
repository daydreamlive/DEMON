"""Guard the MiniMax-Music3 language-model config shim.

The checkpoint's ``config.json`` was written by a transformers version
newer than this repo pins, and the failure mode is silent rather than
loud: ``AutoConfig.from_pretrained`` SUCCEEDS on it. The v5-style
``rope_parameters`` dict lands in ``**kwargs``, gets stashed as an
inert attribute, and ``rope_theta`` quietly keeps the 4.x default of
10000 instead of the checkpoint's 1000000. A 36-layer model then loads,
runs, and produces confidently wrong audio with nothing raised.

That is worth a test on its own, because the shim looks removable to
anyone who checks only whether the config parses.

CPU-only; skipped when the checkpoint is not installed.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("transformers")

from acestep.engine.minimax_ar import load_qwen3_config  # noqa: E402
from acestep.engine.minimax_helpers import resolve_model_dir  # noqa: E402

# What the checkpoint actually specifies, as opposed to what an
# unshimmed transformers 4.x would silently substitute.
CHECKPOINT_ROPE_THETA = 1000000.0
TRANSFORMERS_V4_DEFAULT = 10000.0


@pytest.fixture(scope="module")
def lm_dir():
    try:
        root = resolve_model_dir()
    except FileNotFoundError:
        pytest.skip("MiniMax-Music3 checkpoint not installed")
    lm = root / "language_model"
    if not (lm / "config.json").is_file():
        pytest.skip("MiniMax-Music3 language_model/config.json not present")
    return lm


def test_checkpoint_still_uses_the_v5_rope_layout(lm_dir):
    """If upstream ever rewrites the config in the old layout, the shim
    stops being load-bearing and this test says so rather than letting
    it rot in place unexamined."""
    raw = json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))
    assert "rope_parameters" in raw, (
        "checkpoint no longer uses the v5 rope_parameters layout; "
        "re-evaluate whether load_qwen3_config's shim is still needed"
    )
    assert raw["rope_parameters"]["rope_theta"] == CHECKPOINT_ROPE_THETA


def test_rope_theta_survives_the_shim(lm_dir):
    """The whole point: rope_theta must arrive as the checkpoint's
    value, not the 4.x default."""
    cfg = load_qwen3_config(lm_dir)
    assert float(cfg.rope_theta) == CHECKPOINT_ROPE_THETA
    assert float(cfg.rope_theta) != TRANSFORMERS_V4_DEFAULT


def test_core_architecture_fields_round_trip(lm_dir):
    raw = json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))
    cfg = load_qwen3_config(lm_dir)
    for field in ("hidden_size", "num_hidden_layers", "vocab_size"):
        assert getattr(cfg, field) == raw[field], field


def test_naive_autoconfig_would_get_it_wrong(lm_dir):
    """Documents the trap directly.

    On a transformers new enough to understand ``rope_parameters`` this
    assertion stops being interesting, so it only asserts the failure
    when the plain loader actually disagrees with the checkpoint --
    which is exactly the condition under which the shim is required.
    """
    from transformers import AutoConfig

    raw = json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))
    want = float(raw["rope_parameters"]["rope_theta"])

    naive = AutoConfig.from_pretrained(str(lm_dir), trust_remote_code=False)
    naive_theta = float(getattr(naive, "rope_theta", -1.0))

    if naive_theta == want:
        pytest.skip(
            "installed transformers understands rope_parameters natively; "
            "the shim is a no-op on this version"
        )
    # The installed version silently substitutes its own default...
    assert naive_theta != want
    # ...and the shim is what stands between that and wrong audio.
    assert float(load_qwen3_config(lm_dir).rope_theta) == want

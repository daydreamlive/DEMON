"""SA3 checkpoint identity + context cache keying (Tier 1, no GPU).

The process context cache must key on WHAT was loaded, not merely where
it came from. An operator-supplied checkpoint directory can be rewritten
in place between runs (a training loop that keeps overwriting the same
weights at the same path), so a path-only key would silently hand back
the previously loaded model.
"""

from __future__ import annotations

import json

from acestep.engine.sa3_helpers import (
    sa3_checkpoint_identity,
    sa3_custom_checkpoint_status,
)
from acestep.streaming.sa3_session import sa3_context_key


def _write_checkpoint(base, *, weights: bytes = b"w" * 64):
    base.mkdir(parents=True, exist_ok=True)
    (base / "model_config.json").write_text(
        json.dumps({"model": {"diffusion": {}}}), encoding="utf-8",
    )
    (base / "model.safetensors").write_bytes(weights)
    return base


def test_identity_changes_when_checkpoint_is_rewritten_in_place(tmp_path):
    base = _write_checkpoint(tmp_path / "ckpt")
    before = sa3_checkpoint_identity(base)

    # Same path, different content — the exact shape a training run
    # produces when it overwrites its "best" checkpoint.
    _write_checkpoint(base, weights=b"w" * 128)
    after = sa3_checkpoint_identity(base)

    assert before != after


def test_identity_is_stable_for_an_untouched_checkpoint(tmp_path):
    base = _write_checkpoint(tmp_path / "ckpt")
    assert sa3_checkpoint_identity(base) == sa3_checkpoint_identity(base)


def test_identity_survives_a_missing_file(tmp_path):
    # Used for cache keys, so it must not raise; the preflight is what
    # reports a missing checkpoint with an actionable message.
    identity = sa3_checkpoint_identity(tmp_path / "absent")
    assert isinstance(identity, tuple)


def test_context_key_distinguishes_rewritten_checkpoints(tmp_path):
    base = _write_checkpoint(tmp_path / "ckpt")
    before = sa3_context_key("medium", checkpoint_dir=base)

    _write_checkpoint(base, weights=b"w" * 128)
    after = sa3_context_key("medium", checkpoint_dir=base)

    assert before != after


def test_context_key_is_unchanged_for_the_catalog_path():
    # No custom directory and no extension means the catalog location for
    # the model id, whose identity is the model id itself — stock
    # behavior must not start stat-ing the managed models tree on every
    # session create.
    assert sa3_context_key("small-music") == ("small-music", None, None)
    assert sa3_context_key("small-music") != sa3_context_key("medium")


def test_custom_checkpoint_status_reports_missing_layout(tmp_path):
    ok, msg = sa3_custom_checkpoint_status(tmp_path / "nope")
    assert not ok
    assert "not found" in msg

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "model_config.json").write_text("{}", encoding="utf-8")
    ok, msg = sa3_custom_checkpoint_status(partial)
    assert not ok
    assert "model.safetensors" in msg

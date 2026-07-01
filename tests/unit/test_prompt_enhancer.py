"""Unit tests for the backend-specific prompt enhancer.

Covers the policy split added for Stable Audio 3: the pod infers the enhancer
backend from its own resolved checkpoint FAMILY (``_BACKEND_FAMILY``: the ``sa3``
family selects the SA3 natural-language policy, everything else keeps ACE-Step
tags), and an optional client ``backend=`` query param can override it. The LLM
call itself is mocked (``_ask_haiku``) so these tests are deterministic and need
no network / API key.

The one live-model check — does the SA3 few-shot actually steer Haiku — lives
outside the suite (it needs a real ANTHROPIC_API_KEY); this file pins the policy
selection, sanitize behavior, and failure shape.
"""

import pytest

from demos.realtime_motion_graph_web import prompt_enhancer as pe
from demos.realtime_motion_graph_web import server


@pytest.fixture(autouse=True)
def _restore_family():
    """Keep _BACKEND_FAMILY mutations from leaking across tests."""
    saved = server._BACKEND_FAMILY
    yield
    server._BACKEND_FAMILY = saved


def _capture_system(monkeypatch):
    """Patch _ask_haiku to record the system prompt it was handed and echo a
    canned reply, so a test can assert WHICH policy was selected without a
    network call. Returns a dict that receives the system + user text.
    """
    seen = {}

    def fake(system, user):
        seen["system"] = system
        seen["user"] = user
        # A reply carrying a bpm number so _sanitize's strip is exercised too.
        return "melodic techno, hypnotic arpeggio, deep kick, 128 bpm, wide stereo"

    monkeypatch.setattr(pe, "_ask_haiku", fake)
    return seen


# ── policy selection in enhance_prompt ──────────────────────────────────────

def test_default_backend_uses_acestep_policy(monkeypatch):
    """No backend arg == legacy ACE-Step path, byte-identical selection."""
    seen = _capture_system(monkeypatch)
    out, ok = pe.enhance_prompt("dreamy synthwave")
    assert ok is True
    assert seen["system"] is pe._SYSTEM
    assert "ACE-Step" in seen["user"]


def test_acestep_backend_uses_acestep_policy(monkeypatch):
    seen = _capture_system(monkeypatch)
    pe.enhance_prompt("dreamy synthwave", "acestep")
    assert seen["system"] is pe._SYSTEM


def test_sa3_backend_uses_sa3_policy(monkeypatch):
    seen = _capture_system(monkeypatch)
    out, ok = pe.enhance_prompt("dreamy synthwave", "sa3")
    assert ok is True
    assert seen["system"] is pe._SYSTEM_SA3
    assert "Stable Audio 3" in seen["user"]


def test_bpm_number_stripped_for_both_backends(monkeypatch):
    """_sanitize's bpm-strip is shared: neither backend emits a numeric bpm."""
    for backend in ("acestep", "sa3"):
        _capture_system(monkeypatch)
        out, ok = pe.enhance_prompt("techno", backend)
        assert ok is True
        assert "bpm" not in out.lower()


def test_sa3_system_prompt_is_bpm_free_and_natural_language():
    """The SA3 few-shot must not seed the model with numeric bpm, and must ask
    for descriptive phrases rather than a bare tag list (guide-grounded)."""
    import re
    sys_l = pe._SYSTEM_SA3.lower()
    assert not re.search(r"\d+\s*bpm", sys_l)  # no numeric bpm in examples/instructions
    assert "description" in sys_l
    assert "stable audio 3" in sys_l


def test_empty_prompt_returns_unchanged_not_ok(monkeypatch):
    # Should never even reach the model.
    called = {"n": 0}
    monkeypatch.setattr(pe, "_ask_haiku", lambda s, u: called.__setitem__("n", called["n"] + 1) or "x")
    for backend in ("acestep", "sa3"):
        out, ok = pe.enhance_prompt("   ", backend)
        assert out == ""
        assert ok is False
    assert called["n"] == 0


def test_model_failure_echoes_input_unchanged(monkeypatch):
    """No key / API error path: _ask_haiku returns None -> (idea, False)."""
    monkeypatch.setattr(pe, "_ask_haiku", lambda s, u: None)
    for backend in ("acestep", "sa3", "unknown-passed-straight-through"):
        out, ok = pe.enhance_prompt("dreamy synthwave", backend)
        assert out == "dreamy synthwave"
        assert ok is False


# ── backend resolution in the server route ──────────────────────────────────

def test_resolve_infers_from_backend_family():
    """With no override, the resolved checkpoint family drives the policy.

    _BACKEND_FAMILY is set in main() from resolve_checkpoint(); e.g. an
    ``sa3-medium`` checkpoint resolves to family "sa3" (model id "medium").
    """
    server._BACKEND_FAMILY = "acestep"
    assert server._resolve_enhance_backend("") == "acestep"
    server._BACKEND_FAMILY = "sa3"
    assert server._resolve_enhance_backend("") == "sa3"


def test_resolve_prefers_valid_override():
    server._BACKEND_FAMILY = "acestep"
    assert server._resolve_enhance_backend("sa3") == "sa3"
    server._BACKEND_FAMILY = "sa3"
    assert server._resolve_enhance_backend("acestep") == "acestep"


def test_resolve_falls_back_on_unknown_override():
    """Unknown override never errors: falls back to the inferred family."""
    server._BACKEND_FAMILY = "sa3"
    assert server._resolve_enhance_backend("bogus") == "sa3"
    server._BACKEND_FAMILY = "acestep"
    assert server._resolve_enhance_backend("bogus") == "acestep"


def test_resolve_unknown_family_defaults_to_acestep():
    """A family not in the enhancer set (future model) can't crash enhance."""
    server._BACKEND_FAMILY = "some-future-family"
    assert server._resolve_enhance_backend("") == "acestep"


def test_resolve_override_is_case_insensitive():
    server._BACKEND_FAMILY = "acestep"
    assert server._resolve_enhance_backend("SA3") == "sa3"
    assert server._resolve_enhance_backend(" Sa3 ") == "sa3"


def test_family_matches_resolve_checkpoint_for_sa3():
    """Guard the inference contract end-to-end: an sa3 alias really does resolve
    to family 'sa3' via the same helper main() uses, so a pod started with
    --checkpoint sa3-medium enhances with the SA3 policy."""
    from acestep.streaming.families import resolve_checkpoint
    family, model_id = resolve_checkpoint("sa3-medium")
    assert family == "sa3"
    server._BACKEND_FAMILY = family
    assert server._resolve_enhance_backend("") == "sa3"
    # And a plain ACE checkpoint name resolves to acestep.
    fam2, _ = resolve_checkpoint("acestep-v15-turbo")
    assert fam2 == "acestep"
    server._BACKEND_FAMILY = fam2
    assert server._resolve_enhance_backend("") == "acestep"

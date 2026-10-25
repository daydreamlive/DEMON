"""SA3Backend hands an extension the prompt text it decorates (no GPU).

An extension whose ``decorate_conditioning`` declares ``text`` receives
the exact prompt string each bundle was encoded from; one that does not
keeps the original two-argument call.
"""

from __future__ import annotations

import torch

from acestep.plugins import ModelExtensionRuntime

from tests.unit.test_sa3_backend import C, T, _schedule_builder_factory
from tests.unit.test_sa3_extension_integration import _backend, _cond, _selected


class _TextRuntime(ModelExtensionRuntime):
    def __init__(self):
        self.calls = []

    def decorate_conditioning(self, bundle, source=None, text=None):
        self.calls.append(text)
        return {**bundle, "probe_text": text}


class _LegacyRuntime(ModelExtensionRuntime):
    def __init__(self):
        self.calls = 0

    def decorate_conditioning(self, bundle, source=None):
        self.calls += 1
        return {**bundle, "probe_legacy": True}


def _rebuilder(tags, steps, duration_s):
    return _cond(), _schedule_builder_factory


def test_text_extension_receives_the_prompt_at_every_call_site():
    runtime = _TextRuntime()
    b = _backend(
        extension=_selected(), runtime=runtime,
        prompt_tags="start prompt", prompt_rebuilder=_rebuilder,
        source_encoder=lambda *_a: torch.randn(1, C, T),
    )
    assert runtime.calls == ["start prompt"]
    assert b._cond.cond_bundle["probe_text"] == "start prompt"

    b.handle_set_prompt("prompt A\n@drums: tight kick")
    assert runtime.calls[-1] == "prompt A\n@drums: tight kick"
    assert b._cond.cond_bundle["probe_text"] == "prompt A\n@drums: tight kick"
    assert b._cond_b is b._cond

    b.handle_set_prompt("prompt A", tags_b="prompt B")
    assert runtime.calls[-2:] == ["prompt A", "prompt B"]
    assert b._cond_b.cond_bundle["probe_text"] == "prompt B"

    # Source swap re-decorates the live pair with the live prompts.
    n = len(runtime.calls)
    b.handle_swap_source(torch.zeros(2, 48000), 48000)
    assert runtime.calls[n:] == ["prompt A", "prompt B"]


def test_legacy_extension_keeps_the_two_argument_call():
    runtime = _LegacyRuntime()
    b = _backend(
        extension=_selected(), runtime=runtime,
        prompt_tags="start prompt", prompt_rebuilder=_rebuilder,
    )
    b.handle_set_prompt("prompt A")
    assert runtime.calls == 2
    assert b._cond.cond_bundle["probe_legacy"] is True
    assert "probe_text" not in b._cond.cond_bundle


def test_helper_inspects_the_signature_once_per_runtime(monkeypatch):
    import inspect as _inspect

    from acestep.plugins.model_extensions import decorate_conditioning

    runtime = _TextRuntime()
    seen = []
    real = _inspect.signature
    monkeypatch.setattr(
        _inspect, "signature", lambda f, *a, **k: seen.append(f) or real(f, *a, **k),
    )
    for t in ("p", "q", "r"):
        decorate_conditioning(runtime, {}, None, text=t)
    assert len(seen) == 1
    assert runtime.calls == ["p", "q", "r"]

    legacy = _LegacyRuntime()
    decorate_conditioning(legacy, {}, None, text="x")
    decorate_conditioning(legacy, {}, None, text="y")
    assert len(seen) == 2
    assert legacy.calls == 2


def test_helper_without_weakref_still_dispatches_on_the_signature():
    from acestep.plugins.model_extensions import decorate_conditioning

    class _Slotted:
        __slots__ = ()

        def decorate_conditioning(self, bundle, source=None, text=None):
            return {**bundle, "probe_text": text}

    class _SlottedLegacy:
        __slots__ = ()

        def decorate_conditioning(self, bundle, source=None):
            return {**bundle, "probe_legacy": True}

    assert decorate_conditioning(_Slotted(), {}, text="t") == {"probe_text": "t"}
    assert decorate_conditioning(_SlottedLegacy(), {}, text="t") == {
        "probe_legacy": True,
    }

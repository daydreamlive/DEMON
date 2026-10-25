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


def test_signature_is_inspected_once_per_runtime(monkeypatch):
    import inspect as _inspect

    runtime = _TextRuntime()
    b = _backend(
        extension=_selected(), runtime=runtime,
        prompt_tags="p", prompt_rebuilder=_rebuilder,
    )
    seen = []
    real = _inspect.signature
    monkeypatch.setattr(
        _inspect, "signature", lambda f, *a, **k: seen.append(f) or real(f, *a, **k),
    )
    b.handle_set_prompt("q")
    b.handle_set_prompt("r")
    assert seen == []
    assert runtime.calls == ["p", "q", "r"]

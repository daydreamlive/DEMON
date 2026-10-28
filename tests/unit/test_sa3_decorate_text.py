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


class _BlendRuntime(ModelExtensionRuntime):
    """Returns a marker mapping from blend_conditioning and records calls."""

    def __init__(self, raises=False, returns_none=False):
        self.calls = []
        self.raises = raises
        self.returns_none = returns_none

    def blend_conditioning(self, a, b, v):
        self.calls.append((a, b, v))
        if self.raises:
            raise RuntimeError("probe blend failure")
        if self.returns_none:
            return None
        return {**a, "probe_blend": v}


def _two_prompt_backend(runtime):
    b = _backend(
        extension=_selected(), runtime=runtime,
        prompt_tags="start prompt", prompt_rebuilder=_rebuilder,
    )
    b.handle_set_prompt("prompt A", tags_b="prompt B")
    assert b._cond_b is not b._cond
    return b


def test_blend_conditioning_mapping_is_the_active_bundle():
    runtime = _BlendRuntime()
    b = _two_prompt_backend(runtime)
    b.handle_set_prompt_blend(0.5)
    a_bundle = b._cond.cond_bundle
    b_bundle = b._cond_b.cond_bundle
    assert runtime.calls[-1] == (a_bundle, b_bundle, 0.5)
    active = b._active_bundle
    assert active["probe_blend"] == 0.5
    assert type(active) is dict
    assert active is not a_bundle and active is not b_bundle


def test_blend_endpoints_skip_the_hook():
    runtime = _BlendRuntime()
    b = _two_prompt_backend(runtime)
    runtime.calls.clear()
    b.handle_set_prompt_blend(0.0)
    assert b._active_bundle is b._cond.cond_bundle
    b.handle_set_prompt_blend(1.0)
    assert b._active_bundle is b._cond_b.cond_bundle
    assert runtime.calls == []


def test_runtime_without_the_hook_keeps_the_slerp():
    # _TextRuntime inherits the default (None); returns_none declines too.
    for runtime in (_TextRuntime(), _BlendRuntime(returns_none=True)):
        b = _two_prompt_backend(runtime)
        b.handle_set_prompt_blend(0.5)
        active = b._active_bundle
        assert "probe_blend" not in active
        assert active is not b._cond.cond_bundle
        assert torch.equal(
            active["cross_attn_cond"], b._cond.cond_bundle["cross_attn_cond"],
        )


def test_blend_hook_exception_falls_back_and_logs_once():
    from acestep.engine.obs import logger

    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        runtime = _BlendRuntime(raises=True)
        b = _two_prompt_backend(runtime)
        b.handle_set_prompt_blend(0.5)
        b.handle_set_prompt_blend(0.4)
    finally:
        logger.remove(sink)
    assert len(runtime.calls) == 2
    assert "probe_blend" not in b._active_bundle
    assert "cross_attn_cond" in b._active_bundle
    hits = [m for m in messages if "sa3_blend_conditioning_failed" in m]
    assert len(hits) == 1

"""Session-level LoRA plumbing through the backend facade (D2).

The session's pending-drain (``_apply_lora_pending``), knob-manifest
rebuild, and enabled-id tracking must speak only to
``self.backend.*`` — never to the ACE ``engine_obj`` — so the SA3
family (and any future one) rides the identical plumbing. Exercised
with a recording stub backend and unbound session methods: no GPU, no
model load, same pattern as test_lora_resolution.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.streaming.knobs import KnobState, lora_strength_spec
from acestep.streaming.session import StreamingSession


class _Desc:
    def __init__(self, lora_id, state="registered", strength=0.0):
        self.id = lora_id
        self.name = lora_id
        self.path = f"/nonexistent/{lora_id}.safetensors"
        self.state = state
        self.strength = strength
        self.materialized_bytes = 0


class _RecordingBackend:
    """Facade-shaped stub: catalog + mutation recording. ``fail_ids``
    simulate engine-side enable failures (wrong family, bad file)."""

    def __init__(self, ids=(), fail_ids=()):
        self._descs = {i: _Desc(i) for i in ids}
        self.fail_ids = set(fail_ids)
        self.calls: list = []

    def list_loras(self):
        return list(self._descs.values())

    def enable_lora(self, lora_id, strength=None):
        self.calls.append(("enable", lora_id, strength))
        if lora_id in self.fail_ids:
            raise RuntimeError(f"stub enable failure for {lora_id}")
        d = self._descs.setdefault(lora_id, _Desc(lora_id))
        d.state = "enabled"
        if strength is not None:
            d.strength = float(strength)

    def disable_lora(self, lora_id):
        self.calls.append(("disable", lora_id))
        if lora_id in self._descs:
            self._descs[lora_id].state = "registered"

    def knob_specs(self, lora_ids=()):
        return [lora_strength_spec(lid) for lid in lora_ids]

    def lora_compatible(self, metadata):
        return True


class _Bus:
    def __init__(self):
        self.events: list = []

    def publish(self, event):
        self.events.append(event)


class _DrainSession:
    """Just enough StreamingSession surface for _apply_lora_pending."""

    lora_available = True
    use_lora = True

    _apply_lora_pending = StreamingSession._apply_lora_pending
    _enabled_lora_ids = StreamingSession._enabled_lora_ids
    _rebuild_knob_specs = StreamingSession._rebuild_knob_specs
    lora_catalog_payload = StreamingSession.lora_catalog_payload

    def __init__(self, backend):
        self.backend = backend
        self.state = types.SimpleNamespace(
            _lock=threading.Lock(),
            pending_enable=[],
            pending_disable=[],
        )
        self.virtual_knobs = KnobState([])
        self.bus = _Bus()
        self._knob_specs_by_name = {}


def test_drain_enables_through_facade_and_allocates_knob():
    be = _RecordingBackend(ids=["ambient"])
    ss = _DrainSession(be)
    ss.state.pending_enable.append(("ambient", 0.7))

    ss._apply_lora_pending()

    assert ("enable", "ambient", 0.7) in be.calls
    # Knob slot allocated with the enable strength as its default.
    values = ss.virtual_knobs.get_all_values()
    assert values["lora_str_ambient"] == 0.7
    # The cached validation spec map was rebuilt from the backend's
    # manifest for the new enabled set.
    assert "lora_str_ambient" in ss._knob_specs_by_name
    # Catalog broadcast published.
    assert len(ss.bus.events) == 1
    catalog = ss.bus.events[0].catalog
    assert [e["id"] for e in catalog] == ["ambient"]
    assert catalog[0]["state"] == "enabled"


def test_drain_disable_removes_knob_and_rebuilds():
    be = _RecordingBackend(ids=["ambient"])
    ss = _DrainSession(be)
    ss.state.pending_enable.append(("ambient", 1.0))
    ss._apply_lora_pending()
    assert "lora_str_ambient" in ss.virtual_knobs.get_all_values()

    ss.state.pending_disable.append("ambient")
    ss._apply_lora_pending()

    assert ("disable", "ambient") in be.calls
    assert "lora_str_ambient" not in ss.virtual_knobs.get_all_values()
    assert "lora_str_ambient" not in ss._knob_specs_by_name


def test_drain_enable_failure_is_contained():
    """A backend-side enable failure (wrong family, bad file) is logged,
    allocates no knob, and doesn't break the drain for other entries."""
    be = _RecordingBackend(ids=["good", "bad"], fail_ids=["bad"])
    ss = _DrainSession(be)
    ss.state.pending_enable.append(("bad", 1.0))
    ss.state.pending_enable.append(("good", 0.5))

    ss._apply_lora_pending()  # must not raise

    assert ("enable", "bad", 1.0) in be.calls
    assert ("enable", "good", 0.5) in be.calls
    values = ss.virtual_knobs.get_all_values()
    assert "lora_str_bad" not in values
    assert values["lora_str_good"] == 0.5
    assert ss._enabled_lora_ids() == ["good"]


def test_drain_noop_without_pending():
    be = _RecordingBackend(ids=["ambient"])
    ss = _DrainSession(be)
    ss._apply_lora_pending()
    assert be.calls == []
    assert ss.bus.events == []


def test_enabled_lora_ids_reads_backend_catalog():
    be = _RecordingBackend(ids=["a", "b"])
    ss = _DrainSession(be)
    assert ss._enabled_lora_ids() == []
    be.enable_lora("b", strength=1.0)
    assert ss._enabled_lora_ids() == ["b"]
    ss2 = _DrainSession(be)
    ss2.use_lora = False
    assert ss2._enabled_lora_ids() == []

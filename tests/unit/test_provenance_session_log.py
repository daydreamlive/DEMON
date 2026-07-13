"""Local JSONL session log: bus tap wiring, line schema, hash chain.

Pure Python — no GPU, no c2pa. Drives a real
:class:`acestep.streaming.events.EventBus` through the registry hook
(:func:`acestep.streaming.registry.register` attaches the tap when the
handle carries a ``bus``) and asserts the on-disk JSONL record: one
``session_start`` with model/LoRA identity, timestamped typed events,
diffed + rate-limited ``params`` user actions, output-chain
checkpoints, and a sealing ``session_end``.

Determinism: :meth:`SessionLogTap.close` (via ``unregister``)
unsubscribes and joins the drainer, and the drainer delivers all
already-queued events before exiting — so once ``unregister`` returns,
every published event is on disk and ``session_end`` is the last line.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import acestep.provenance.session_log as session_log_mod
from acestep.provenance.session_log import (
    SESSION_LOG_SCHEMA_VERSION,
    get_tap,
    latest_session_summary,
    record_user_action,
)
from acestep.streaming import registry
from acestep.streaming.events import (
    AudioReady,
    EventBus,
    PromptApplied,
    SwapFailed,
)


def _slice(tag: int) -> AudioReady:
    audio = np.full((16, 2), float(tag), dtype=np.float32)
    return AudioReady(
        audio=audio, start_sample=tag * 16, num_samples=16, channels=2,
        tick_ms=0.0, dec_ms=0.0, num_gens=tag, params={},
    )


def _register(bus: EventBus, session_id: str, **kwargs) -> registry.SessionHandle:
    handle = registry.SessionHandle(
        id=session_id,
        started_at=time.time(),
        inject=lambda data, audio: None,
        snapshot=kwargs.pop("snapshot", lambda: {}),
        bus=bus,
        provenance_meta=kwargs.pop("provenance_meta", None),
    )
    assert not kwargs
    registry.register(handle)
    return handle


def _read_log(tmp_path: Path, session_id: str) -> list[dict]:
    p = tmp_path / "provenance" / "sessions" / f"{session_id}.jsonl"
    assert p.is_file(), f"missing session log at {p}"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def _by_event(lines: list[dict], event: str) -> list[dict]:
    return [l for l in lines if l["event"] == event]


def test_session_log_schema_and_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))
    bus = EventBus()
    snapshot = lambda: {  # noqa: E731 — mirrors adapter snapshot shape
        "lora_catalog": [
            {"id": "lead-guitar", "state": "enabled"},
            {"id": "muted", "state": "disabled"},
        ],
        "prompt": "warm analog dub",
        "bpm": 120,
        "key": "C minor",
        "time_signature": "4",
    }
    _register(
        bus, "sess-schema",
        snapshot=snapshot,
        provenance_meta={"checkpoint": "ace_step_v1", "fixture_name": "loop60"},
    )

    bus.publish(PromptApplied(tags="dark techno"))
    bus.publish(SwapFailed(error="no such fixture"))
    record_user_action("sess-schema", "prompt", {"type": "prompt", "tags": "dub 2"})
    registry.unregister("sess-schema")

    lines = _read_log(tmp_path, "sess-schema")

    # Every line carries the shared envelope: schema version, a
    # wall-clock UTC "Z" timestamp, the session id, and an event type.
    for line in lines:
        assert line["schema"] == SESSION_LOG_SCHEMA_VERSION
        assert line["session_id"] == "sess-schema"
        assert isinstance(line["event"], str)
        assert line["ts"].endswith("Z")
        datetime.fromisoformat(line["ts"].replace("Z", "+00:00"))  # parses

    # session_start is first and carries the identity fields.
    start = lines[0]
    assert start["event"] == "session_start"
    assert start["model"] == "ace_step_v1"
    assert start["loras"] == ["lead-guitar"]
    assert start["fixture_name"] == "loop60"
    assert start["prompt"] == "warm analog dub"
    assert start["bpm"] == 120

    # Typed bus events land with their payload fields.
    prompt = _by_event(lines, "prompt_change")
    assert len(prompt) == 1 and prompt[0]["tags"] == "dark techno"
    errors = _by_event(lines, "error")
    assert len(errors) == 1
    assert errors[0]["kind"] == "SwapFailed"
    assert errors[0]["error"] == "no such fixture"

    # User actions carry action / source / bounded payload summary.
    actions = _by_event(lines, "user_action")
    assert len(actions) == 1
    assert actions[0]["action"] == "prompt"
    assert actions[0]["source"] == "ws"
    assert actions[0]["summary"] == {"tags": "dub 2"}

    # session_end seals the log: chain head + timeline summary counts.
    end = lines[-1]
    assert end["event"] == "session_end"
    assert end["slices_hashed"] == 0
    assert end["output_chain_head"] == hashlib.sha256(b"").hexdigest()
    summary = end["timeline_summary"]
    assert summary["prompt_changes"] == 2  # bus PromptApplied + user action
    assert summary["user_actions"] == 1
    assert summary["distinct_prompts"] == 3  # snapshot + applied + user action
    # The seal's summary is computed before the session_end line itself
    # is counted.
    assert summary["events"] == len(lines) - 1
    assert summary["duration_s"] >= 0.0

    # Detach removed the tap from the process-global registry.
    assert get_tap("sess-schema") is None
    assert latest_session_summary() is None


def test_output_chain_checkpoints_and_seal(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))
    monkeypatch.setattr(session_log_mod, "_CHAIN_CHECKPOINT_EVERY", 4)
    bus = EventBus()
    _register(bus, "sess-chain")

    slices = [_slice(i) for i in range(6)]
    for s in slices:
        bus.publish(s)
    registry.unregister("sess-chain")

    # Recompute the expected rolling chain: head = sha256(head_hex || bytes).
    head = hashlib.sha256(b"").hexdigest()
    heads = []
    for s in slices:
        head = hashlib.sha256(
            head.encode("ascii") + np.ascontiguousarray(s.audio).tobytes(),
        ).hexdigest()
        heads.append(head)

    lines = _read_log(tmp_path, "sess-chain")
    checkpoints = _by_event(lines, "output_chain_checkpoint")
    assert len(checkpoints) == 1  # 6 slices, checkpoint every 4
    assert checkpoints[0]["slices"] == 4
    assert checkpoints[0]["chain_head"] == heads[3]
    assert checkpoints[0]["start_sample"] == slices[3].start_sample

    end = lines[-1]
    assert end["event"] == "session_end"
    assert end["slices_hashed"] == 6
    assert end["output_chain_head"] == heads[-1]
    assert end["timeline_summary"]["slices_hashed"] == 6


def test_params_actions_are_diffed_and_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))
    bus = EventBus()
    _register(bus, "sess-params")
    tap = get_tap("sess-params")
    assert tap is not None

    # First params action logs only the changed knobs; telemetry keys
    # are stripped.
    record_user_action(
        "sess-params", "params",
        {"type": "params", "steer": 0.5, "flow": 0.1, "playback_pos": 12.5},
    )
    # Inside the rate-limit window: suppressed even though a knob moved.
    record_user_action("sess-params", "params", {"steer": 0.9})
    # Identical values are suppressed regardless of the window.
    monkeypatch.setattr(session_log_mod.time, "monotonic", lambda: 1e9)
    record_user_action(
        "sess-params", "params", {"steer": 0.5, "flow": 0.1},
    )
    # Outside the window with a real change: logs the diff only.
    record_user_action("sess-params", "params", {"steer": 0.7, "flow": 0.1})

    registry.unregister("sess-params")
    lines = _read_log(tmp_path, "sess-params")
    actions = _by_event(lines, "user_action")
    assert [a["summary"] for a in actions] == [
        {"steer": 0.5, "flow": 0.1},
        {"steer": 0.7},
    ]
    assert lines[-1]["timeline_summary"]["param_changes"] == 2


def test_registry_hook_is_inert_without_bus(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))
    handle = registry.SessionHandle(
        id="sess-no-bus",
        started_at=time.time(),
        inject=lambda data, audio: None,
        snapshot=lambda: {},
    )
    registry.register(handle)
    registry.unregister("sess-no-bus")

    assert get_tap("sess-no-bus") is None
    assert not (tmp_path / "provenance" / "sessions").exists()

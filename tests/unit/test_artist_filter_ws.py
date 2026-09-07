"""Enforcement wiring for the artist-name filter on the WS surface.

Two layers, matching what a dev box can actually import:

* Source-structure guards (AST/text, no torch): the prompt dispatch arm must
  consult ``scan_prompt_slots`` BEFORE ``set_prompt``, and session-init must
  substitute the default prompt. These run everywhere and catch the
  "enforcement quietly removed" regression.
* Behavior tests on ``scan_prompt_slots`` itself, skipped where the full
  runtime (torch/torchaudio) isn't installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WS_SRC = (
    Path(__file__).resolve().parents[2]
    / "demos/realtime_motion_graph_web/ws_adapter.py"
).read_text(encoding="utf-8")


# ── source-structure guards (no heavy imports) ──────────────────────────────

def test_prompt_dispatch_gates_before_set_prompt():
    arm = re.search(
        r'elif mtype == "prompt":(.*?)elif mtype == ', WS_SRC, re.DOTALL
    ).group(1)
    gate = arm.index("scan_prompt_slots")
    apply = arm.index("streaming.set_prompt")
    assert gate < apply, "policy scan must run before set_prompt"
    assert "_send_json(rejection)" in arm


def test_session_init_substitutes_default_prompt():
    assert "init_prompt_rejection = scan_prompt_slots(" in WS_SRC
    assert "cfg.prompt = SessionConfig.prompt" in WS_SRC


def test_rejection_never_logs_prompt_text():
    helper = re.search(
        r"def scan_prompt_slots.*?\n\n\n", WS_SRC, re.DOTALL
    ).group(0)
    for logline in re.findall(r"logger\.\w+\(\s*\"([^\"]+)\"", helper):
        assert "tags" not in logline and "text" not in logline


# ── behavior (needs the runtime deps ws_adapter pulls in) ───────────────────
#
# importorskip lives in a fixture, not at module scope, so the source guards
# above still run on a dev box without torch.


@pytest.fixture()
def wa():
    return pytest.importorskip(
        "demos.realtime_motion_graph_web.ws_adapter",
        reason="ws_adapter needs torch/torchaudio",
    )


def test_scan_prompt_slots_rejects_with_event_payload(wa, monkeypatch):
    monkeypatch.delenv("DEMON_ARTIST_FILTER", raising=False)
    payload = wa.scan_prompt_slots("taylor swift style pop", None, surface="ws")
    assert payload["type"] == "prompt_rejected"
    assert payload["slot"] == "a"
    assert payload["reason"] == "artist_name"
    assert payload["matched"] == "Taylor Swift"
    assert payload["filter_version"] == "artists.v1"
    assert "Taylor Swift" in payload["detail"]


def test_scan_prompt_slots_slot_attribution(wa, monkeypatch):
    monkeypatch.delenv("DEMON_ARTIST_FILTER", raising=False)
    assert wa.scan_prompt_slots(
        "warm pads", "daft punk style", surface="ws")["slot"] == "b"
    assert wa.scan_prompt_slots(
        "daft punk style", "taylor swift style", surface="ws")["slot"] == "both"


def test_scan_prompt_slots_clean_and_modes(wa, monkeypatch):
    monkeypatch.delenv("DEMON_ARTIST_FILTER", raising=False)
    assert wa.scan_prompt_slots("warm dub techno chords", None, surface="ws") is None
    monkeypatch.setenv("DEMON_ARTIST_FILTER", "log")
    assert wa.scan_prompt_slots("taylor swift style", None, surface="ws") is None
    monkeypatch.setenv("DEMON_ARTIST_FILTER", "off")
    assert wa.scan_prompt_slots("taylor swift style", None, surface="ws") is None

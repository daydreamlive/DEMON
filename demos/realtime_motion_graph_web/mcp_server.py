"""Onboard MCP server for the DEMON realtime motion-graph demo.

Exposes every user-facing demo action as an MCP tool so an LLM (Claude
Code or any MCP client) can drive an already-running session for
automated testing. The MCP attaches to a *live* session — the user
opens the demo in their browser as usual, and the MCP injects commands
into that session over an HTTP control bus the server hosts on
``127.0.0.1:1319``. The front-end's own WebSocket stays primary, so
MCP-driven changes propagate back to the browser via the same ack
messages the UI already listens to.

Run as a stdio MCP server. Example Claude Code config:

    {
      "mcpServers": {
        "demon": {
          "command": "uv",
          "args": [
            "run", "python", "-u",
            "-m", "demos.realtime_motion_graph_web.mcp_server"
          ],
          "cwd": "C:/_dev/projects/DEMON"
        }
      }
    }

Override the backend host/port via ``DEMON_HOST`` / ``DEMON_PORT``
(backend's main HTTP+WS port, default 1318) and ``DEMON_CONTROL_HOST`` /
``DEMON_CONTROL_PORT`` (control bus, default 127.0.0.1:1319).
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import urllib.error
import urllib.request
from math import gcd
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
from loguru import logger
from mcp.server.fastmcp import FastMCP

from acestep.steering import (
    MANUAL_SLOT_CAP,
    ensure_steering_vectors,
    enumerate_catalog,
)
from acestep.streaming.knobs import (
    KNOB_SCHEMA_VERSION,
    KnobSpec,
    coerce_knob_values,
    knob_catalog,
    knob_specs,
)
from .protocol import coerce_command_payload, wire_contract

# MCP runs as a single global process, so we pre-fetch the canonical
# 2B turbo bundle at module init. Fetch failures leave the cache empty;
# the next streaming session retries.
_MANUAL_VECTOR_DIR = ensure_steering_vectors("acestep-v15-turbo")


# MCP wire protocol owns stdout — every log MUST go to stderr. Lazy
# configure so this module stays importable without a hard dependency on
# the rest of the engine package's logging config. The backend logs
# every dispatched command on its end (origin="control"), so this side
# stays intentionally light.
_mcp_log = logger.bind(component="mcp")
_logger_configured = False


def _ensure_logger() -> None:
    # loguru's logger is a process-global singleton. If a co-located
    # backend (or anything else) already configured sinks, calling
    # remove() here would wipe them — so only attach our stderr sink
    # when no handler is already attached. This module is normally a
    # separate stdio subprocess, but the guard makes a same-process
    # import safe. ``logger._core.handlers`` is private but stable
    # across loguru versions; the try/except keeps the guard from
    # crashing if loguru ever renames it.
    global _logger_configured
    if _logger_configured:
        return
    try:
        already_configured = bool(logger._core.handlers)
    except AttributeError:
        already_configured = False
    if not already_configured:
        logger.add(sys.stderr, level="INFO")
    _logger_configured = True


def _log(*parts: Any) -> None:
    _ensure_logger()
    _mcp_log.info(" ".join(str(p) for p in parts))


BACKEND_HOST = os.environ.get("DEMON_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("DEMON_PORT", "1318"))
CONTROL_HOST = os.environ.get("DEMON_CONTROL_HOST", "127.0.0.1")
CONTROL_PORT = int(os.environ.get("DEMON_CONTROL_PORT", "1319"))
BACKEND_HTTP = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
CONTROL_HTTP = f"http://{CONTROL_HOST}:{CONTROL_PORT}"
TARGET_SR = 48000

mcp = FastMCP("demon")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_json(method: str, url: str, body: bytes = b"",
               timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, data=body if body else None, method=method)
    if body:
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("Content-Length", str(len(body)))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8")) if data else {}
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
        except Exception:
            err_body = {"error": e.reason}
        raise RuntimeError(
            f"{method} {url} -> {e.code}: {err_body.get('error', e.reason)}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"{method} {url}: {e.reason} "
            f"(is the demo backend running on {BACKEND_HOST}:{BACKEND_PORT}?)"
        ) from e


def _http_get_bytes(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Session selection — by default we drive the most-recently-started session
# ---------------------------------------------------------------------------


def _list_sessions() -> list[dict]:
    return _http_json("GET", f"{CONTROL_HTTP}/sessions", timeout=5.0)


def _resolve_session_id(session_id: Optional[str]) -> str:
    sessions = _list_sessions()
    if not sessions:
        raise RuntimeError(
            "No active session. Open the demo in your browser first "
            f"(http://localhost:{BACKEND_PORT}/) — the MCP attaches to the "
            "live session; it does not spawn its own.",
        )
    if session_id is not None:
        for s in sessions:
            if s.get("id") == session_id:
                return session_id
        raise RuntimeError(
            f"session_id {session_id!r} not found. Live sessions: "
            f"{[s.get('id') for s in sessions]}"
        )
    # default: pick most recent (registry returns newest first)
    return sessions[0]["id"]


def _encode_cmd(data: dict, audio: Optional[bytes] = None) -> bytes:
    json_bytes = json.dumps(data).encode("utf-8")
    prefix = struct.pack("<I", len(json_bytes))
    if audio is None:
        return prefix + json_bytes
    return prefix + json_bytes + audio


def _validate_command(data: dict) -> dict:
    """Coerce + validate a command envelope against the wire contract
    (:func:`demos.realtime_motion_graph_web.protocol.coerce_command_payload`).

    Raises ``ValueError`` on any contract violation so the agent gets explicit
    feedback instead of the backend silently ignoring a malformed command.
    This is the command-envelope counterpart to the knob-level validation in
    ``_validate_against_session``: every MCP command now derives its validity
    from the same registry the backend and a re-skinned UI build against.
    Returns the cleaned envelope to forward. The free-form ``params.raw`` knob
    dict rides through untouched here — its schema is the /api/knobs manifest,
    enforced separately by ``_validate_against_session``.
    """
    clean, errors = coerce_command_payload(str(data.get("type")), data)
    if errors:
        raise ValueError("; ".join(errors))
    return clean


def _send_cmd(session_id: Optional[str], data: dict,
              audio: Optional[bytes] = None) -> dict:
    data = _validate_command(data)
    sid = _resolve_session_id(session_id)
    body = _encode_cmd(data, audio)
    return _http_json("POST", f"{CONTROL_HTTP}/sessions/{sid}/cmd",
                      body=body, timeout=120.0)


# ---------------------------------------------------------------------------
# Audio helpers — local audio file → wire format expected by the backend
# ---------------------------------------------------------------------------


def _resample_to_target(arr: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return arr
    from scipy.signal import resample_poly
    g = gcd(sr, TARGET_SR)
    up = TARGET_SR // g
    down = sr // g
    out = np.stack([resample_poly(arr[c], up, down) for c in range(arr.shape[0])])
    return out.astype(np.float32, copy=False)


def _load_audio(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    arr, sr = sf.read(str(p), always_2d=True)
    arr = arr.T.astype(np.float32, copy=False)
    if arr.shape[0] > 2:
        arr = arr[:2]
    return _resample_to_target(arr, sr)


def _waveform_to_audio_bytes(waveform: np.ndarray) -> bytes:
    """Channel-major (channels, samples) -> backend wire format (``<II``
    header + interleaved float32 PCM)."""
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be 2D; got {waveform.shape}")
    channels, samples = int(waveform.shape[0]), int(waveform.shape[1])
    interleaved = waveform.T.astype(np.float32, copy=False).tobytes()
    return struct.pack("<II", channels, samples) + interleaved


# ---------------------------------------------------------------------------
# Manual steering vector catalog (local view of the prefetched bundle)
# ---------------------------------------------------------------------------


def _enumerate_manual_catalog() -> list[dict]:
    """Manual steering catalog flattened to wire-stable dicts."""
    if _MANUAL_VECTOR_DIR is None:
        return []
    return [
        {
            "index": entry.index,
            "axis": entry.axis,
            "build_layer": entry.build_layer,
            "build_step": entry.build_step,
            "filename": entry.filename,
        }
        for entry in enumerate_catalog(_MANUAL_VECTOR_DIR)
    ]


# ---------------------------------------------------------------------------
# Tools — discovery
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_sessions() -> list[dict]:
    """List active demo sessions (newest first).

    Each entry includes ``id``, ``started_at``, current prompt, fixture,
    bpm/key/time_signature, knob_values, lora_catalog, etc. Pass ``id``
    to other tools as ``session_id`` if you want to target a specific
    session; tools default to the most-recently-started one.
    """
    return _list_sessions()


@mcp.tool()
async def session_state(session_id: Optional[str] = None) -> dict:
    """Full snapshot of one session (defaults to the most recent)."""
    sid = _resolve_session_id(session_id)
    return _http_json("GET", f"{CONTROL_HTTP}/sessions/{sid}", timeout=5.0)


@mcp.tool()
async def list_fixtures() -> list[str]:
    """Canonical audio fixture names from the daydreamlive/demon-fixtures-v2
    Hugging Face dataset. Any name here can be passed to swap_to_fixture
    / set_timbre_fixture / set_structure_fixture.
    """
    return _http_json("GET", f"{BACKEND_HTTP}/api/fixtures", timeout=10.0)


@mcp.tool()
async def list_loras() -> dict:
    """List all LoRAs discoverable in the server's MODELS_DIR/loras.

    Each entry has id, name, path, state, strength, materialized_bytes,
    and a ``metadata`` blob with the normalized sidecar record:
    primary_trigger_word, trigger_words, description, recommended_*,
    classification, etc. Use ``id`` with enable_lora/disable_lora; the
    metadata is most useful for picking which LoRA to enable and at
    what strength.
    """
    return _http_json("GET", f"{BACKEND_HTTP}/api/loras", timeout=10.0)


@mcp.tool()
async def get_lora_metadata(lora_id: str) -> dict:
    """Return the full metadata record for a single LoRA by id (stem).

    Mirrors the ``metadata`` block on each ``list_loras`` entry but
    saves the agent from parsing the whole catalog. Returns a sparse
    record (mostly nulls) for LoRAs without a sidecar; ``has_metadata``
    is True iff a real ``<stem>.metadata.json`` was loaded. Returns
    ``{"error": "not_found"}`` if no LoRA with that id exists.
    """
    catalog = _http_json("GET", f"{BACKEND_HTTP}/api/loras", timeout=10.0)
    for entry in catalog.get("loras", []):
        if entry.get("id") == lora_id:
            return entry.get("metadata") or {}
    return {"error": "not_found", "lora_id": lora_id}


@mcp.tool()
async def describe_protocol() -> dict:
    """The full WebSocket wire contract: ``{version, commands, events}``.

    Same self-describing manifest the backend serves at ``GET /api/protocol``.
    ``commands`` is every client->server message type (name -> fields / binary
    / origin_sensitive / description); ``events`` is every server->client
    message type. A consumer can build every control message and decode every
    server event from this alone, modulo the binary framing noted per entry
    and the separate ``/api/knobs`` payload schema for the ``params`` command.

    Static (session-independent): the vocabulary doesn't change per session,
    only the ``params`` knob set does (see list_knobs).
    """
    return wire_contract()


@mcp.tool()
async def list_knobs(session_id: Optional[str] = None) -> dict:
    """Knob catalog (name → type/default/min/max/group/options/description/
    bank) plus the session's current knob_values dict.

    Same manifest the backend serves at ``/api/knobs`` — the complete
    operator-knob set, including the non-bank params (guidance_scale,
    cfg_rescale, dcw_*, steps_override) and the string-valued enums
    (rcfg_mode, dcw_mode, dcw_wavelet). ``bank: false`` marks a knob that
    rides the params channel but isn't stored in KnobState as a live knob;
    its default is still seeded into the snapshot, so every manifest knob
    reports a value in ``current``. ``version`` is the schema version
    (bump = the contract changed shape; re-skins/agents can detect a stale
    build).

    Knob set depends on whether the session was started in SDE mode and
    which LoRAs are currently enabled — pulled from the live snapshot.
    """
    snap = await session_state(session_id)
    # Prefer the snapshot's backend-owned manifest (Phase 2): it is the
    # session's LIVE knob universe, including the backend-specific knobs
    # the static registry projection can't reproduce (the steering
    # steer_* axes and the per-slot man_*_<N> quadruples).
    manifest = snap.get("knob_manifest") or {}
    if isinstance(manifest, dict) and manifest.get("knobs"):
        return {
            "version": manifest.get("version", KNOB_SCHEMA_VERSION),
            "knobs": manifest["knobs"],
            "current": snap.get("knob_values") or {},
        }
    # Older server snapshot without a manifest: re-derive from the
    # shared registry (no steering surface on those servers anyway).
    sde, enabled = _session_knob_shape(snap)
    return {
        "version": KNOB_SCHEMA_VERSION,
        "knobs": knob_catalog(sde=sde, loras=enabled),
        "current": snap.get("knob_values") or {},
    }


def _session_knob_shape(snap: dict) -> tuple[bool, list]:
    """(sde, enabled_lora_ids) for a session snapshot — the two inputs
    knob_specs/knob_catalog need to reproduce that session's knob set."""
    sde = snap.get("sde")
    if not isinstance(sde, bool):
        # Older server whose snapshot doesn't carry ``sde``: infer it from
        # the SDE-only knob's presence.
        sde = "sde_amp" in (snap.get("knob_values") or {})
    enabled = [
        d.get("id") for d in (snap.get("lora_catalog") or [])
        if d.get("state") == "enabled" and d.get("id")
    ]
    return sde, enabled


def _specs_from_snapshot(snap: dict) -> dict:
    """``{name: KnobSpec}`` for a session snapshot.

    Reconstructed from the snapshot's backend-owned ``knob_manifest``
    when present (so validation covers backend-specific knobs like the
    steering surface); falls back to the shared registry for older
    servers. The manifest is itself a registry projection
    (``catalog_from_specs``), so this stays single-source."""
    manifest = (snap.get("knob_manifest") or {}).get("knobs")
    if isinstance(manifest, dict) and manifest:
        return {
            name: KnobSpec(
                name=name,
                default=e.get("default", 0.0),
                min_val=e.get("min"),
                max_val=e.get("max", 1.0),
                type=e.get("type", "float"),
                options=tuple(e.get("options") or ()),
                group=e.get("group", "core"),
                bank=bool(e.get("bank", True)),
            )
            for name, e in manifest.items()
        }
    sde, enabled = _session_knob_shape(snap)
    return {s.name: s for s in knob_specs(sde=sde, loras=enabled)}


async def _validate_against_session(
    raw: dict, session_id: Optional[str]
) -> dict:
    """Validate a raw knob dict against the live session's schema. Returns
    the coerced dict to send; raises ValueError (surfaced to the agent) if
    any value is out of range or not an allowed enum/bool option. Reuses
    the same coerce_knob_values the server enforces, so MCP can't drift."""
    snap = await session_state(session_id)
    clean, errors = coerce_knob_values(raw, _specs_from_snapshot(snap))
    if errors:
        raise ValueError("; ".join(errors))
    return clean


@mcp.tool()
async def add_manual_slot(session_id: Optional[str] = None) -> dict:
    """Spawn the next manual steering slot (LIFO; alpha defaults to 0).

    Refused (no-op echo) at MANUAL_SLOT_CAP.
    """
    _send_cmd(session_id, {"type": "manual_slot_add"})
    snap = await session_state(session_id)
    return {
        "count": int(snap.get("manual_slot_count") or 0),
        "cap": MANUAL_SLOT_CAP,
    }


@mcp.tool()
async def pop_manual_slot(session_id: Optional[str] = None) -> dict:
    """Remove the highest-numbered manual steering slot.

    LIFO; interior deletion is not supported. Refused (no-op echo) on
    an empty registry.
    """
    _send_cmd(session_id, {"type": "manual_slot_pop"})
    snap = await session_state(session_id)
    return {
        "count": int(snap.get("manual_slot_count") or 0),
        "cap": MANUAL_SLOT_CAP,
    }


@mcp.tool()
async def list_manual_steering_vectors() -> dict:
    """Catalog of pre-built steering vectors for the manual slots.

    Returns ``{"count": N, "vectors": [...]}``. Each entry's ``index``
    is the value to set on ``man_src_<slot>``. Order is stable across
    sessions (axis-major, then build_layer asc, then build_step asc).
    """
    cat = _enumerate_manual_catalog()
    return {"count": len(cat), "vectors": cat}


# ---------------------------------------------------------------------------
# Tools — prompt
# ---------------------------------------------------------------------------


@mcp.tool()
async def set_prompt(
    prompt: str,
    prompt_b: Optional[str] = None,
    key: Optional[str] = None,
    time_signature: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Change the live prompt. Pass ``prompt_b`` to cache a second prompt
    for A/B blending via set_prompt_blend. ``key`` accepts strings like
    ``"C major"`` / ``"A minor"``; ``time_signature`` accepts ``"3"`` /
    ``"4"`` / ``"6"`` etc.
    """
    msg: dict[str, Any] = {"type": "prompt", "tags": prompt}
    if prompt_b is not None:
        msg["tags_b"] = prompt_b
    if key is not None:
        msg["key"] = key
    if time_signature is not None:
        msg["time_signature"] = time_signature
    return _send_cmd(session_id, msg)


@mcp.tool()
async def set_prompt_blend(value: float, session_id: Optional[str] = None) -> dict:
    """Lerp between cached prompt A and B (0.0 = A, 1.0 = B).

    Cheap; no text-encoder pass. Requires a prior set_prompt with
    ``prompt_b=...``.
    """
    v = max(0.0, min(1.0, float(value)))
    return _send_cmd(session_id, {"type": "set_prompt_blend", "value": v})


# ---------------------------------------------------------------------------
# Tools — knobs
# ---------------------------------------------------------------------------


@mcp.tool()
async def set_knob(name: str, value: float,
                   session_id: Optional[str] = None) -> dict:
    """Set a single knob (see list_knobs).

    Backend merges into its current state. The browser's UI mirrors
    MCP-driven knob changes via the new ``params_echo`` message; the
    next UI param tick then carries the same value back so it sticks.

    Validated against the live session's schema: an out-of-range value or
    an unknown knob raises rather than silently clamping. Enum/bool knobs
    (rcfg_mode, dcw_mode/wavelet, dcw_enabled) are float-rejected here —
    use set_rcfg_mode and friends for those.
    """
    clean = await _validate_against_session({name: value}, session_id)
    return _send_cmd(session_id, {
        "type": "params",
        "raw": clean,
        "playback_pos": 0.0,
    })


@mcp.tool()
async def set_knobs(values: dict[str, float],
                    session_id: Optional[str] = None) -> dict:
    """Bulk knob update. Validated against the live session's schema; any
    out-of-range or unknown knob raises (the whole batch is rejected) so the
    agent gets explicit feedback instead of a silent clamp. Numeric parsing
    is owned by coerce_knob_values, so a bad value yields a contract error
    naming the knob rather than a bare float() crash."""
    clean = await _validate_against_session(dict(values), session_id)
    return _send_cmd(session_id, {
        "type": "params",
        "raw": clean,
        "playback_pos": 0.0,
    })


@mcp.tool()
async def get_knob(name: str, session_id: Optional[str] = None) -> dict:
    """Return a knob's value from the session's current state."""
    snap = await session_state(session_id)
    kv = snap.get("knob_values") or {}
    return {"name": name, "value": kv.get(name)}


@mcp.tool()
async def set_rcfg_mode(mode: str, session_id: Optional[str] = None) -> dict:
    """Set the RCFG (Residual CFG) mode. String-valued, so it can't ride
    set_knob (which is float-only).

    Modes (the authoritative set is the ``rcfg_mode`` entry in list_knobs):
      "off"        — no guidance (turbo default; free)
      "self"       — virtual uncond from initial noise (~1.06x cost)
      "initialize" — uncond run once per slot then cached (~1.07x cost)
      "full"       — standard two-pass CFG (~2x cost; not in the UI
                     dropdown because turbo is CFG-distilled, but
                     pipeline.py accepts it for test scripts)

    Pairs with the ``guidance_scale`` and ``cfg_rescale`` knobs (only
    consumed when mode != "off"). Rides the ``params`` control channel;
    useMcpMirror has a string-value branch that drives setRcfgMode so
    the value persists across the next UI param tick.
    """
    # Enum membership comes from the knob registry (coerce_knob_values
    # checks ``mode`` against the rcfg_mode spec's options), so adding a
    # mode is a one-place edit in knobs.py.
    clean = await _validate_against_session({"rcfg_mode": mode}, session_id)
    return _send_cmd(session_id, {
        "type": "params",
        "raw": clean,
        "playback_pos": 0.0,
    })


# ---------------------------------------------------------------------------
# Tools — LoRA
# ---------------------------------------------------------------------------


@mcp.tool()
async def enable_lora(lora_id: str, strength: Optional[float] = None,
                      session_id: Optional[str] = None) -> dict:
    """Enable a LoRA by id (see list_loras). Optional ``strength`` sets the
    target value the refit lands at (avoids the first-window-without-LoRA
    artifact you'd get if you enabled at 0 and ramped via set_knob).

    The LoRA's trigger token (if any) is prepended to the next text encode
    by the server.
    """
    msg: dict[str, Any] = {"type": "enable_lora", "id": lora_id}
    if strength is not None:
        msg["strength"] = float(strength)
    return _send_cmd(session_id, msg)


@mcp.tool()
async def disable_lora(lora_id: str, session_id: Optional[str] = None) -> dict:
    """Disable a LoRA by id."""
    return _send_cmd(session_id, {"type": "disable_lora", "id": lora_id})


# ---------------------------------------------------------------------------
# Tools — timbre reference
# ---------------------------------------------------------------------------


@mcp.tool()
async def set_timbre_strength(value: float,
                              session_id: Optional[str] = None) -> dict:
    """Live blend between the silence-baseline and full timbre-ref
    conditioning. 1.0 = full reference; 0.0 = silence baseline.
    """
    v = max(0.0, min(1.0, float(value)))
    return _send_cmd(session_id, {"type": "set_timbre_strength", "value": v})


@mcp.tool()
async def set_timbre_fixture(name: str,
                             session_id: Optional[str] = None) -> dict:
    """Use a server-side fixture (from list_fixtures) as the timbre reference.

    Avoids the round-trip of downloading and re-uploading PCM; the server
    resolves the WAV from its local HF cache.
    """
    return _send_cmd(session_id, {"type": "set_timbre_fixture", "name": name})


@mcp.tool()
async def set_timbre_audio(audio_file: str, name: Optional[str] = None,
                           session_id: Optional[str] = None) -> dict:
    """Upload a local audio file as the timbre reference.

    File is resampled to 48 kHz, capped to ≤2 channels, and the server
    will further cap its length to the playback source's duration.
    """
    waveform = _load_audio(audio_file)
    label = name or Path(audio_file).name
    return _send_cmd(
        session_id,
        {"type": "set_timbre_source", "name": label},
        audio=_waveform_to_audio_bytes(waveform),
    )


@mcp.tool()
async def clear_timbre(session_id: Optional[str] = None) -> dict:
    """Drop the timbre override; server falls back to self-timbre
    (encode against the playback source's own latent).
    """
    return _send_cmd(session_id, {"type": "clear_timbre_source"})


# ---------------------------------------------------------------------------
# Tools — structure reference
# ---------------------------------------------------------------------------


@mcp.tool()
async def set_structure_fixture(name: str,
                                session_id: Optional[str] = None) -> dict:
    """Use a server-side fixture as the structure (semantic-hint) reference."""
    return _send_cmd(session_id, {"type": "set_structure_fixture", "name": name})


@mcp.tool()
async def set_structure_audio(audio_file: str, name: Optional[str] = None,
                              session_id: Optional[str] = None) -> dict:
    """Upload a local audio file as the structure reference.

    Server pads/trims it to match the playback source's exact sample count
    before extracting the context_latent.
    """
    waveform = _load_audio(audio_file)
    label = name or Path(audio_file).name
    return _send_cmd(
        session_id,
        {"type": "set_structure_source", "name": label},
        audio=_waveform_to_audio_bytes(waveform),
    )


@mcp.tool()
async def clear_structure(session_id: Optional[str] = None) -> dict:
    """Drop the structure override; server restores the playback source's
    own context_latent.
    """
    return _send_cmd(session_id, {"type": "clear_structure_source"})


# ---------------------------------------------------------------------------
# Tools — swap playback source
# ---------------------------------------------------------------------------


@mcp.tool()
async def swap_to_fixture(
    name: str,
    prompt: Optional[str] = None,
    key: Optional[str] = None,
    time_signature: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Swap the playback source to a server-side fixture (from list_fixtures).

    Pulls the fixture bytes from the backend's HTTP endpoint and sends a
    ``swap_source`` command. Server runs the sidecar fast path when
    available (skips BPM/key detection and prepare_source).
    """
    audio_bytes = _http_get_bytes(f"{BACKEND_HTTP}/fixtures/{name}")
    arr, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    arr = arr.T.astype(np.float32, copy=False)
    if arr.shape[0] > 2:
        arr = arr[:2]
    arr = _resample_to_target(arr, sr)
    msg: dict[str, Any] = {"type": "swap_source", "fixture_name": name}
    if prompt is not None:
        msg["tags"] = prompt
    if key is not None:
        msg["key"] = key
    if time_signature is not None:
        msg["time_signature"] = time_signature
    return _send_cmd(session_id, msg, audio=_waveform_to_audio_bytes(arr))


@mcp.tool()
async def swap_to_audio(
    audio_file: str,
    name: Optional[str] = None,
    prompt: Optional[str] = None,
    key: Optional[str] = None,
    time_signature: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Swap the playback source to a local audio file (resampled to 48 kHz).

    ``name`` is the label echoed back to the front-end so the fixture
    dropdown can adopt it (uploads stay in-session via customTracks).
    Defaults to the file's basename.
    """
    arr = _load_audio(audio_file)
    label = name or Path(audio_file).name
    # The backend's sidecar lookup keys off fixture_name; an upload's
    # label won't match any known fixture, so the lookup misses and the
    # live BPM/key path runs (intended). The label still flows back
    # through swap_ready.fixture_name so the UI mirror can adopt it.
    msg: dict[str, Any] = {"type": "swap_source", "fixture_name": label}
    if prompt is not None:
        msg["tags"] = prompt
    if key is not None:
        msg["key"] = key
    if time_signature is not None:
        msg["time_signature"] = time_signature
    return _send_cmd(session_id, msg, audio=_waveform_to_audio_bytes(arr))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _log(f"starting MCP server; backend={BACKEND_HTTP}, control={CONTROL_HTTP}")
    mcp.run()


if __name__ == "__main__":
    main()

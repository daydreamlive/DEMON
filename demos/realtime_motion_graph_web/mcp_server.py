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
from mcp.server.fastmcp import FastMCP


def _log(*parts: Any) -> None:
    """Stderr-only logging. stdout belongs to the MCP wire protocol."""
    print("[demon-mcp]", *parts, file=sys.stderr, flush=True)


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


def _send_cmd(session_id: Optional[str], data: dict,
              audio: Optional[bytes] = None) -> dict:
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
# Knob catalog (mirrors demos/realtime_motion_graph_web/knobs.py)
# ---------------------------------------------------------------------------


_GROUP_NAMES = [
    "ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7",
]
_KEYSTONE_NAMES = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"]


def _build_knob_catalog(sde: bool, enabled_lora_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if sde:
        out["sde_amp"] = {"default": 0.0, "max": 1.0, "group": "core",
                          "description": "SDE diffusion amplitude (replaces denoise in SDE mode)"}
        out["periodicity"] = {"default": 0.0, "max": 12.5, "group": "core",
                              "description": "SDE periodicity"}
    else:
        out["denoise"] = {"default": 0.0, "max": 1.0, "group": "core",
                          "description": "ODE denoise strength"}
    out["seed"] = {"default": 0.0, "max": 1.0, "group": "core",
                   "description": "Stream seed control"}
    out["feedback"] = {"default": 0.0, "max": 1.0, "group": "core",
                       "description": "Feedback amount"}
    out["shift"] = {"default": 0.5, "max": 1.0, "group": "core",
                    "description": "Flow shift (timing/curve shape)"}
    for lid in enabled_lora_ids:
        out[f"lora_str_{lid}"] = {
            "default": 0.0, "max": 2.0, "group": "core",
            "description": f"Strength for LoRA {lid!r}",
        }
    out["hint_strength"] = {"default": 1.0, "max": 1.0, "group": "core",
                            "description": "Structure (semantic hint) blend strength"}
    out["ode_noise"] = {"default": 0.0, "max": 0.5, "group": "core",
                        "description": "ODE noise injection amount"}
    for name in _GROUP_NAMES:
        out[name] = {"default": 1.0, "max": 3.0, "group": "groups",
                     "description": f"Channel-group amplifier {name}"}
    for name in _KEYSTONE_NAMES:
        out[name] = {"default": 1.0, "max": 3.0, "group": "keystones",
                     "description": f"Keystone channel amplifier {name}"}
    return out


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
    """Canonical audio fixture names from the daydreamlive/demon-fixtures
    Hugging Face dataset. Any name here can be passed to swap_to_fixture
    / set_timbre_fixture / set_structure_fixture.
    """
    return _http_json("GET", f"{BACKEND_HTTP}/api/fixtures", timeout=10.0)


@mcp.tool()
async def list_loras() -> dict:
    """List all LoRAs discoverable in the server's MODELS_DIR/loras.

    Each entry has id, name, path, trigger, state, strength,
    materialized_bytes. Use ``id`` with enable_lora/disable_lora.
    """
    return _http_json("GET", f"{BACKEND_HTTP}/api/loras", timeout=10.0)


@mcp.tool()
async def list_knobs(session_id: Optional[str] = None) -> dict:
    """Knob catalog (name → default/max/group/description) plus the
    session's current knob_values dict.

    Knob set depends on whether the session was started in SDE mode and
    which LoRAs are currently enabled — pulled from the live snapshot.
    """
    snap = await session_state(session_id)
    sde = "sde_amp" in (snap.get("knob_values") or {})
    enabled = [
        d.get("id") for d in (snap.get("lora_catalog") or [])
        if d.get("state") == "enabled" and d.get("id")
    ]
    return {
        "knobs": _build_knob_catalog(sde=sde, enabled_lora_ids=enabled),
        "current": snap.get("knob_values") or {},
    }


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
    """
    return _send_cmd(session_id, {
        "type": "params",
        "raw": {name: float(value)},
        "playback_pos": 0.0,
    })


@mcp.tool()
async def set_knobs(values: dict[str, float],
                    session_id: Optional[str] = None) -> dict:
    """Bulk knob update."""
    coerced = {k: float(v) for k, v in values.items()}
    return _send_cmd(session_id, {
        "type": "params",
        "raw": coerced,
        "playback_pos": 0.0,
    })


@mcp.tool()
async def get_knob(name: str, session_id: Optional[str] = None) -> dict:
    """Return a knob's value from the session's current state."""
    snap = await session_state(session_id)
    kv = snap.get("knob_values") or {}
    return {"name": name, "value": kv.get(name)}


_RCFG_MODES = ("off", "self", "initialize", "full")


@mcp.tool()
async def set_rcfg_mode(mode: str, session_id: Optional[str] = None) -> dict:
    """Set the RCFG (Residual CFG) mode. String-valued, so it can't ride
    set_knob (which is float-only).

    Modes:
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
    if mode not in _RCFG_MODES:
        raise ValueError(
            f"mode must be one of {list(_RCFG_MODES)}; got {mode!r}"
        )
    return _send_cmd(session_id, {
        "type": "params",
        "raw": {"rcfg_mode": mode},
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

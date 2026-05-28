"""WebSocket protocol shared by the client, the server, and the full demo.

Torch-free. Only depends on numpy, websockets, and zstandard.
"""

import json
import struct
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

# Legacy module constants. Since the Phase-2 contract surface, the
# DECLARED truth for a session's audio shape is the backend-sourced
# ``ready.geometry`` block (sample_rate / channels / chunk_rate_hz /
# duration_s); these constants remain for the ACE-only code paths and
# tooling that predate it.
SAMPLE_RATE = 48000
T = 1500  # 60s at 25fps latents
CROSSFADE_SECONDS = 0.025

# Binary slice header: flags, start_sample, num_samples, channels, tick_ms, dec_ms, num_gens
# flags: 0 = raw float16, 1 = zstd-compressed float16 delta
SLICE_HDR_FMT = "<BIIHffI"
SLICE_HDR_SIZE = struct.calcsize(SLICE_HDR_FMT)
SLICE_FLAG_RAW = 0
SLICE_FLAG_DELTA = 1


class RemoteBackend:
    """WebSocket connection to a remote GPU pipeline server.

    Accepts the source audio as either a torch.Tensor with shape
    ``(channels, samples)`` (via ``.numpy()``) or a plain numpy array with
    the same shape. This keeps the thin client torch-free while remaining
    compatible with the full local+remote demo.
    """

    def __init__(self, url: str, waveform, config: dict):
        from websockets.sync.client import connect as ws_connect

        print(f"[Remote] Connecting to {url}...")
        self.ws = ws_connect(url, max_size=50 * 1024 * 1024, open_timeout=30)
        self._send_lock = threading.Lock()

        print("[Remote] Sending config...")
        self.ws.send(json.dumps(config))

        if hasattr(waveform, "numpy"):
            audio_np = waveform.numpy()
        else:
            audio_np = np.asarray(waveform)
        # Normalize to (samples, channels) layout for the wire format.
        # Callers pass (channels, samples); transpose here.
        audio_np = audio_np.T
        hdr = struct.pack("<II", audio_np.shape[1], audio_np.shape[0])
        self.ws.send(hdr + audio_np.astype(np.float32).tobytes())
        print(f"[Remote] Uploaded audio ({audio_np.nbytes / 1024 / 1024:.1f} MB)")

        print("[Remote] Waiting for server init...")
        ready = json.loads(self.ws.recv())
        self.duration = ready["duration"]
        self.channels = ready["channels"]
        self.sample_rate = ready["sample_rate"]
        self.lora_count = ready.get("lora_count", 0)

        init_bytes = self.ws.recv()
        self.initial_buffer = (
            np.frombuffer(init_bytes, dtype=np.float16)
            .astype(np.float32)
            .reshape(-1, self.channels)
        )
        print(f"[Remote] Ready: {self.duration:.1f}s, {self.channels}ch")

    def send_raw(self, raw_dict: dict, playback_pos: float):
        msg = {"type": "params", "raw": raw_dict, "playback_pos": playback_pos}
        try:
            with self._send_lock:
                self.ws.send(json.dumps(msg))
        except Exception:
            pass

    def send_prompt(self, tags: str):
        try:
            with self._send_lock:
                self.ws.send(json.dumps({"type": "prompt", "tags": tags}))
        except Exception:
            pass

    def recv(self, timeout: float = 0.01):
        try:
            msg = self.ws.recv(timeout=timeout)
            if isinstance(msg, bytes) and len(msg) > SLICE_HDR_SIZE:
                hdr = struct.unpack(SLICE_HDR_FMT, msg[:SLICE_HDR_SIZE])
                flags = hdr[0]
                payload = msg[SLICE_HDR_SIZE:]
                if flags == SLICE_FLAG_DELTA:
                    import zstandard as zstd
                    payload = zstd.decompress(payload)
                audio_f16 = np.frombuffer(payload, dtype=np.float16)
                return ("audio", {
                    "flags": flags,
                    "start_sample": hdr[1],
                    "num_samples": hdr[2],
                    "channels": hdr[3],
                    "tick_ms": hdr[4],
                    "dec_ms": hdr[5],
                    "num_gens": hdr[6],
                    "audio": audio_f16.astype(np.float32).reshape(hdr[2], hdr[3]),
                })
            elif isinstance(msg, str):
                return ("json", json.loads(msg))
        except TimeoutError:
            return None
        except Exception:
            return None
        return None

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Wire-contract registry
# ---------------------------------------------------------------------------
# Single source of truth for the WebSocket control vocabulary: every command
# the client may send (client -> server) and every event the server may emit
# (server -> client). Projected into a transport-agnostic catalog served at
# ``GET /api/protocol`` and by the MCP ``describe_protocol`` tool, so a
# re-skinned / vibecoded UI -- or an MCP agent -- builds against this contract
# instead of reverse-engineering ``ws_adapter.py`` and ``web/engine/
# protocol.ts``. Mirrors the role ``acestep/streaming/knobs.py`` plays for
# the parameter surface.
#
# Authoritative, not just descriptive: ``tests/unit/test_wire_contract.py``
# AST-parses the dispatcher and asserts its handled message types are exactly
# ``COMMAND_NAMES``, and that every event the browser client handles appears
# in ``EVENT_NAMES``. So a new verb can't be added to either side without
# being registered here, or the suite fails.
#
# Out of scope here (these are framing, not part of the JSON command/event
# vocabulary): the phase-1 ``config`` handshake (the JSON config + binary PCM
# upload sent before streaming begins; see the ``SessionConfig`` TS interface)
# and the binary audio-slice stream (see ``SLICE_*`` constants above). The
# per-command ``params`` knob payload has its own manifest at ``/api/knobs``.

PROTOCOL_VERSION = 1


@dataclass
class FieldSpec:
    """One field of a command/event JSON payload."""
    name: str
    type: str                 # str | float | int | bool | dict | list | enum
    required: bool = False
    default: Any = None
    options: tuple = ()        # allowed values for enum fields
    nullable: bool = False     # explicit JSON null is a valid value
    description: str = ""


@dataclass
class CommandSpec:
    """One client -> server message type."""
    name: str
    fields: tuple = ()
    binary: bool = False           # a trailing binary audio frame follows
    binary_optional: bool = False  # binary present only in some variants
    # True = an EXTERNAL (MCP / control-bus) send is NOT applied; the server
    # echoes it back on ``echo_event`` for the session's own UI to mirror
    # and re-send through its smoothing tween. Commands without this flag
    # apply identically from any origin and ack via their normal events.
    # Must match the session's actual origin handling (StreamingSession:
    # only set_knobs and set_prompt_blend are origin-aware).
    origin_sensitive: bool = False
    echo_event: str = ""           # mirror event for EXTERNAL sends
    # Backend capability this command needs (a Capabilities field name,
    # acestep.streaming.generator_backend). On a backend whose mask lacks
    # it, the session rejects the command with the typed ``command_failed``
    # event instead of applying it (plan §3.4 — loud failure, never a
    # silent no-op). "" = universal command, every backend honors it.
    # Drift-guarded: every tagged command must have a matching
    # @requires_capability gate on its StreamingSession operation.
    requires: str = ""
    description: str = ""


@dataclass
class EventSpec:
    """One server -> client message type."""
    name: str
    fields: tuple = ()
    binary_follow: bool = False    # one or more binary frames follow the JSON
    description: str = ""


# Reusable framing note for the two binary upload commands.
_PCM_FRAME = (
    "JSON header followed by one binary PCM frame: ``<II`` (channels, samples) "
    "then interleaved float32 at 48 kHz."
)

COMMANDS: tuple = (
    CommandSpec(
        "params",
        fields=(
            FieldSpec("raw", "dict", required=True,
                      description="Knob name -> value map. The payload schema "
                                  "is the separate /api/knobs manifest; values "
                                  "are clamped/validated server-side."),
            FieldSpec("playback_pos", "float", default=0.0,
                      description="Playhead position in SECONDS (not a 0..1 "
                                  "ratio); used for time-keyed curve sampling."),
        ),
        origin_sensitive=True,
        echo_event="params_echo",
        description="Continuous parameter update; sent every tick (~125 Hz) "
                    "carrying the full knob dict.",
    ),
    CommandSpec(
        "loop_band",
        fields=(
            FieldSpec("start_sec", "float", nullable=True,
                      description="Loop start in seconds; null/degenerate clears."),
            FieldSpec("end_sec", "float", nullable=True,
                      description="Loop end in seconds; null/degenerate clears."),
        ),
        requires="loop_band",
        description="Arm / move / clear the playback loop band.",
    ),
    CommandSpec(
        "prompt",
        fields=(
            FieldSpec("tags", "str", required=True,
                      description="Prompt A (wire text; enabled-LoRA triggers "
                                  "are prepended client-side)."),
            FieldSpec("tags_b", "str",
                      description="Optional prompt B, cached for A/B blend."),
            FieldSpec("key", "str", description='Musical key, e.g. "C major".'),
            FieldSpec("time_signature", "str",
                      description='Meter numerator, e.g. "3"/"4"/"6".'),
        ),
        description="Re-encode the live prompt (text-encoder pass).",
    ),
    CommandSpec(
        "set_prompt_blend",
        fields=(FieldSpec("value", "float", required=True, default=0.0,
                          description="0.0 = A, 1.0 = B. Clamped to [0,1]."),),
        origin_sensitive=True,
        echo_event="prompt_blend_echo",
        description="Lerp cached prompt A/B conditioning. Cheap; no encode pass.",
    ),
    CommandSpec(
        "set_interp_method",
        fields=(
            FieldSpec("path", "enum", required=True,
                      options=("prompt", "timbre", "structure", "feedback"),
                      description="Which live blend to retune."),
            FieldSpec("method", "enum", required=True,
                      options=("slerp", "linear"),
                      description="Interpolation curve."),
        ),
        description="Switch a blend path between slerp (norm-preserving) and "
                    "linear.",
    ),
    CommandSpec(
        "set_depth",
        fields=(FieldSpec("value", "int", required=True,
                          description="Target ring depth; clamped to "
                                      "[1, max_pipeline_depth]."),),
        requires="depth",
        description="Live pipeline_depth retune. Echoed back as depth_applied.",
    ),
    CommandSpec(
        "enable_lora",
        fields=(
            FieldSpec("id", "str", required=True,
                      description="LoRA id/stem (see /api/loras)."),
            FieldSpec("strength", "float",
                      description="Target strength the refit lands at."),
        ),
        requires="lora",
        description="Enable a LoRA; allocates a lora_str_<id> knob.",
    ),
    CommandSpec(
        "disable_lora",
        fields=(FieldSpec("id", "str", required=True),),
        requires="lora",
        description="Disable a LoRA and drop its lora_str_<id> knob.",
    ),
    CommandSpec(
        "manual_slot_add",
        requires="steering",
        description="Allocate the next manual steering slot (LIFO); "
                    "allocates its four man_*_<N> knobs. Echoed back as "
                    "manual_slot_count on success AND refusal (at "
                    "manual_slot_cap).",
    ),
    CommandSpec(
        "manual_slot_pop",
        requires="steering",
        description="Remove the highest-numbered manual steering slot and "
                    "drop its man_*_<N> knobs (LIFO; interior deletion is "
                    "not supported). Echoed back as manual_slot_count on "
                    "success AND refusal (empty registry).",
    ),
    CommandSpec(
        "set_timbre_strength",
        fields=(FieldSpec("value", "float", required=True, default=1.0,
                          description="1.0 = full reference, 0.0 = silence "
                                      "baseline. Clamped to [0,1]."),),
        requires="timbre",
        description="Blend the cached (silence, full) timbre conditioning pair.",
    ),
    CommandSpec(
        "set_timbre_source",
        fields=(FieldSpec("name", "str", default="timbre",
                          description="Label echoed back in timbre_set."),),
        binary=True,
        requires="timbre",
        description="Upload audio as the timbre reference. " + _PCM_FRAME +
                    " Acked by timbre_set / timbre_failed.",
    ),
    CommandSpec(
        "set_timbre_fixture",
        fields=(FieldSpec("name", "str", required=True,
                          description="Fixture name (see /api/fixtures)."),),
        requires="timbre",
        description="Use a server-side fixture as the timbre reference (no "
                    "upload round-trip).",
    ),
    CommandSpec(
        "clear_timbre_source",
        requires="timbre",
        description="Drop the timbre override; fall back to self-timbre. Acked "
                    "by timbre_cleared.",
    ),
    CommandSpec(
        "set_structure_source",
        fields=(FieldSpec("name", "str", default="structure",
                          description="Label echoed back in structure_set."),),
        binary=True,
        requires="structure",
        description="Upload audio as the structure (semantic-hint) reference. "
                    + _PCM_FRAME + " Acked by structure_set / structure_failed.",
    ),
    CommandSpec(
        "set_structure_fixture",
        fields=(FieldSpec("name", "str", required=True,
                          description="Fixture name (see /api/fixtures)."),),
        requires="structure",
        description="Use a server-side fixture as the structure reference (no "
                    "upload round-trip).",
    ),
    CommandSpec(
        "clear_structure_source",
        requires="structure",
        description="Drop the structure override; restore the source's own "
                    "context_latent. Acked by structure_cleared.",
    ),
    CommandSpec(
        "swap_source",
        fields=(
            FieldSpec("tags", "str", description="Optional new prompt A."),
            FieldSpec("key", "str"),
            FieldSpec("time_signature", "str"),
            FieldSpec("fixture_name", "str",
                      description="Source label; for server-side loads, the "
                                  "fixture/upload name to read off the pod's "
                                  "disk."),
            FieldSpec("stem_source_mode", "enum",
                      options=("full", "vocals", "instruments"),
                      description="For uploads: which model-ripped stem feeds "
                                  "inference."),
            FieldSpec("use_server_source", "bool",
                      description="When true, the server loads the named source "
                                  "off its own disk and NO binary frame is "
                                  "sent."),
        ),
        binary=True,
        binary_optional=True,
        requires="swap",
        description="Replace the playback source in-flight. A binary PCM frame "
                    "follows UNLESS use_server_source is set. Acked by "
                    "swap_ready (+ binary buffer) / swap_failed.",
    ),
)

EVENTS: tuple = (
    EventSpec(
        "init_ack",
        fields=(
            FieldSpec("session_id", "str",
                      description="Server-minted session id, sent as soon as "
                                  "log context binds so client startup "
                                  "failures correlate with pod logs."),
            FieldSpec("client_id", "str", nullable=True,
                      description="The config client_id echoed back, or null "
                                  "when the client sent none."),
        ),
        description="Optional telemetry ack emitted after config parse but "
                    "before audio/model init. Sent ONLY when the config opts "
                    "in via telemetry_version, so old clients never see it.",
    ),
    EventSpec(
        "ready",
        fields=(
            FieldSpec("duration", "float", required=True),
            FieldSpec("channels", "int", required=True),
            FieldSpec("sample_rate", "int", required=True),
            FieldSpec("lora_catalog", "list"),
            FieldSpec("lora_dir", "str"),
            FieldSpec("bpm", "float", nullable=True),
            FieldSpec("key", "str", nullable=True),
            FieldSpec("time_signature", "str", nullable=True),
            FieldSpec("checkpoint", "str"),
            FieldSpec("checkpoint_scale", "str"),
            FieldSpec("pipeline_depth", "int"),
            FieldSpec("max_pipeline_depth", "int"),
            FieldSpec("lora_pending_enable", "list",
                      description="LoRA ids the server will auto-enable on the "
                                  "first tick (from the session's initial "
                                  "enable set); empty when none."),
            FieldSpec("session_id", "str",
                      description="Server-minted session id, echoed for "
                                  "client/analytics log correlation."),
            # Phase-2 contract surface (backend-seam plan §3.1–3.3).
            # Wire-optional: golden replay transcripts recorded before
            # Phase 2 lack them, and the client falls back to its
            # constants / /api/knobs when absent.
            FieldSpec("geometry", "dict",
                      description="Backend-declared audio geometry: "
                                  "{sample_rate, channels, chunk_rate_hz, "
                                  "duration_s|null}. chunk_rate_hz is the "
                                  "generation cadence (latent fps for "
                                  "diffusion, frame rate for AR models); "
                                  "duration_s null is reserved for endless "
                                  "streams."),
            FieldSpec("capabilities", "dict",
                      description="Backend capability mask: {capability: "
                                  "bool} over the Capabilities fields "
                                  "(swap, timbre, structure, lora, ...). "
                                  "Client panels and MCP tools gate on it; "
                                  "commands tagged with a matching "
                                  "`requires` fail with command_failed "
                                  "when the bit is false."),
            FieldSpec("knob_manifest", "dict",
                      description="Per-session knob manifest: the same "
                                  "{version, knobs} envelope GET /api/knobs "
                                  "serves, but backend-owned and session-"
                                  "resolved (SDE mode, enabled lora_str_<id> "
                                  "knobs). /api/knobs remains the static "
                                  "pre-session probe."),
            # Activation-steering surface. Wire-optional like the
            # Phase-2 fields above: absent on backends without the
            # steering capability, and the client hides the steering
            # tiles when steering_available isn't explicitly true.
            FieldSpec("manual_slot_count", "int",
                      description="Active manual steering slot count; "
                                  "drives the client's man_*_<N> row "
                                  "rendering. Updated live via the "
                                  "manual_slot_count event."),
            FieldSpec("manual_slot_cap", "int",
                      description="Server-imposed ceiling on manual "
                                  "steering slots; gates the client's "
                                  "+ button."),
            FieldSpec("steering_available", "bool",
                      description="True when the session's checkpoint has "
                                  "a reachable steering-vector bundle; "
                                  "false hides the steering surface (the "
                                  "steer_*/man_* knobs are absent from "
                                  "the manifest too)."),
        ),
        binary_follow=True,
        description="First JSON after the upload handshake, followed by the "
                    "binary float16 initial buffer.",
    ),
    EventSpec(
        "error",
        fields=(
            FieldSpec("code", "str"),
            FieldSpec("message", "str"),
            FieldSpec("build_command", "str"),
            FieldSpec("duration_s", "float",
                      description="Present only on the engine_not_built code: "
                                  "the source duration whose TRT profile is "
                                  "missing."),
        ),
        description="Structured init failure (sent during the handshake).",
    ),
    EventSpec(
        "params_update",
        fields=(FieldSpec("params", "dict", required=True,
                          description="Applied params + runtime telemetry "
                                      "(num_gens, tick_ms, dec_ms)."),),
        description="Periodic echo of the live param/telemetry snapshot.",
    ),
    EventSpec(
        "params_echo",
        fields=(FieldSpec("raw", "dict", required=True),),
        description="MCP-driven knob change echoed so the browser mirrors it; "
                    "the browser's own param updates do not echo.",
    ),
    EventSpec(
        "prompt_blend_echo",
        fields=(FieldSpec("value", "float", required=True),),
        description="MCP-driven prompt-blend echo (the dedicated blend slider "
                    "rides its own channel, not params).",
    ),
    EventSpec(
        "prompt_applied",
        fields=(FieldSpec("tags", "str"),),
        description="Ack that a prompt re-encode completed.",
    ),
    EventSpec(
        "lora_catalog",
        fields=(FieldSpec("catalog", "list", required=True),),
        description="Refreshed LoRA catalog after enable/disable.",
    ),
    EventSpec(
        "swap_ready",
        fields=(
            FieldSpec("duration", "float", required=True),
            FieldSpec("sample_rate", "int", required=True),
            FieldSpec("channels", "int", required=True),
            FieldSpec("bpm", "float", nullable=True),
            FieldSpec("key", "str", nullable=True),
            FieldSpec("time_signature", "str", nullable=True),
            FieldSpec("fixture_name", "str", nullable=True),
        ),
        binary_follow=True,
        description="Swap accepted; a binary float16 replacement buffer follows "
                    "the JSON.",
    ),
    EventSpec(
        "swap_failed",
        fields=(
            FieldSpec("error", "str"),
            FieldSpec("build_command", "str",
                      description="Present only when the swap failed on a "
                                  "missing TRT engine: the command to build "
                                  "the profile for the new source's duration."),
        ),
        description="Swap rejected (decode/load failure).",
    ),
    EventSpec(
        "stem_assets",
        fields=(
            FieldSpec("fixture_name", "str", required=True),
            FieldSpec("sample_rate", "int", required=True),
            FieldSpec("channels", "int", required=True),
            FieldSpec("frames", "int", required=True),
            FieldSpec("stems", "list", required=True,
                      description='Ordered subset of ("vocals","instruments").'),
            FieldSpec("source_mode", "enum",
                      options=("full", "vocals", "instruments")),
        ),
        binary_follow=True,
        description="Stem-overlay header; one binary float16 buffer per listed "
                    "stem follows, in order.",
    ),
    EventSpec(
        "stem_failed",
        fields=(FieldSpec("fixture_name", "str"), FieldSpec("error", "str")),
        description="Stem extraction/overlay failure.",
    ),
    EventSpec(
        "depth_applied",
        fields=(FieldSpec("value", "int", required=True,
                          description="The clamped applied depth."),),
        description="Ack for set_depth.",
    ),
    EventSpec(
        "manual_slot_count",
        fields=(FieldSpec("count", "int", required=True,
                          description="The live manual steering slot "
                                      "count after the command."),),
        description="Ack for manual_slot_add / manual_slot_pop — emitted on "
                    "success and refusal alike so the client's +/- UI "
                    "resyncs either way.",
    ),
    EventSpec(
        "timbre_set",
        fields=(FieldSpec("name", "str", required=True),
                FieldSpec("duration", "float", required=True)),
        description="Ack for set_timbre_source / set_timbre_fixture.",
    ),
    EventSpec("timbre_cleared", description="Ack for clear_timbre_source."),
    EventSpec(
        "timbre_failed",
        fields=(FieldSpec("error", "str"),),
        description="Failure ack for any set_timbre_* path.",
    ),
    EventSpec(
        "structure_set",
        fields=(FieldSpec("name", "str", required=True),
                FieldSpec("duration", "float", required=True)),
        description="Ack for set_structure_source / set_structure_fixture.",
    ),
    EventSpec("structure_cleared",
              description="Ack for clear_structure_source."),
    EventSpec(
        "structure_failed",
        fields=(FieldSpec("error", "str"),),
        description="Failure ack for any set_structure_* path, and the notice "
                    'emitted when a swap drops a structure override ("dropped '
                    'after swap: ...").',
    ),
    EventSpec(
        "command_failed",
        fields=(
            FieldSpec("command", "str", required=True,
                      description="The rejected command's wire name."),
            FieldSpec("requires", "str", required=True,
                      description="The Capabilities field the command needs "
                                  "and the session's backend doesn't "
                                  "declare."),
            FieldSpec("error", "str",
                      description="Human-readable reason."),
        ),
        description="A `requires`-tagged command was rejected because the "
                    "session's backend lacks the capability (loud failure, "
                    "never a silent no-op). The command was NOT applied. "
                    "With only the acestep backend registered this fires "
                    "solely for LoRA commands on a lora-disabled session.",
    ),
)

COMMAND_NAMES = frozenset(c.name for c in COMMANDS)
EVENT_NAMES = frozenset(e.name for e in EVENTS)


# ---------------------------------------------------------------------------
# Init-phase handshake vocabulary (pre-stream)
# ---------------------------------------------------------------------------
# These ride the connection BEFORE streaming begins and are dispatched
# differently from the per-tick commands above: the session-init ``config``
# JSON (projected from ``acestep.streaming.config.SessionConfig`` by
# :func:`config_catalog`) and the standalone ``upload_track`` sub-protocol that
# persists + encodes a track on the pod without starting a stream. Kept OUT of
# COMMANDS / EVENTS so the dispatcher drift guard (which matches the streaming
# ``mtype ==`` chain) stays exact; surfaced under ``handshake`` in
# :func:`wire_contract`.

HANDSHAKE_COMMANDS: tuple = (
    CommandSpec(
        "upload_track",
        fields=(
            FieldSpec("name", "str", default="upload",
                      description="Requested track label; deduped server-side."),
            FieldSpec("key", "str",
                      description="Optional key override; forces a re-encode "
                                  "instead of the content-dedup fast path."),
            FieldSpec("time_signature", "str",
                      description="Optional meter override; same effect as key."),
        ),
        binary=True,
        description="Init-phase: persist + encode an uploaded track on the pod "
                    "WITHOUT starting a stream (sent as the FIRST frame in place "
                    "of the session config). " + _PCM_FRAME + " Acked by "
                    "upload_ok / upload_failed, after which the socket closes.",
    ),
)

HANDSHAKE_EVENTS: tuple = (
    EventSpec(
        "upload_ok",
        fields=(
            FieldSpec("name", "str", required=True,
                      description="Final persisted track name (may differ from "
                                  "the requested name after dedup/uniquify)."),
            FieldSpec("bpm", "int"),
            FieldSpec("key", "str"),
            FieldSpec("time_signature", "str"),
            FieldSpec("duration_s", "float"),
            FieldSpec("samples", "int"),
        ),
        description="Init-phase ack: the uploaded track was persisted + encoded "
                    "(or an identical existing one was reused).",
    ),
    EventSpec(
        "upload_failed",
        fields=(FieldSpec("error", "str"),),
        description="Init-phase upload failure (decode/encode/persist error).",
    ),
)

HANDSHAKE_COMMAND_NAMES = frozenset(c.name for c in HANDSHAKE_COMMANDS)
HANDSHAKE_EVENT_NAMES = frozenset(e.name for e in HANDSHAKE_EVENTS)


def _field_payload(f: "FieldSpec") -> dict:
    out: dict = {"type": f.type, "required": f.required}
    if f.default is not None:
        out["default"] = f.default
    if f.options:
        out["options"] = list(f.options)
    if f.nullable:
        out["nullable"] = True
    if f.description:
        out["description"] = f.description
    return out


def _command_catalog_of(specs) -> dict:
    out: dict = {}
    for c in specs:
        entry: dict = {
            "fields": {f.name: _field_payload(f) for f in c.fields},
            "binary": c.binary,
            "binary_optional": c.binary_optional,
            "origin_sensitive": c.origin_sensitive,
            "description": c.description,
        }
        if c.echo_event:
            entry["echo_event"] = c.echo_event
        if c.requires:
            entry["requires"] = c.requires
        out[c.name] = entry
    return out


def _event_catalog_of(specs) -> dict:
    return {
        e.name: {
            "fields": {f.name: _field_payload(f) for f in e.fields},
            "binary_follow": e.binary_follow,
            "description": e.description,
        }
        for e in specs
    }


def command_catalog() -> dict:
    """Project COMMANDS into a transport-agnostic ``name -> spec`` catalog."""
    return _command_catalog_of(COMMANDS)


def event_catalog() -> dict:
    """Project EVENTS into a transport-agnostic ``name -> spec`` catalog."""
    return _event_catalog_of(EVENTS)


# Python type -> wire type for the SessionConfig projection.
_CONFIG_WIRE_TYPES = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    list: "list",
    dict: "dict",
}


def _project_config_type(tp) -> tuple:
    """Resolve one SessionConfig annotation to ``(wire_type, nullable)``.

    Unwraps ``X | None`` / ``Optional[X]`` into the inner type with
    ``nullable=True``. Raises ``TypeError`` on an annotation with no wire
    projection, so a novel field type fails loudly at codegen/serve time
    instead of silently landing in the contract as ``str``.
    """
    import types as _pytypes
    import typing as _typing

    origin = _typing.get_origin(tp)
    if origin is _typing.Union or origin is getattr(_pytypes, "UnionType", None):
        args = [a for a in _typing.get_args(tp) if a is not type(None)]
        if len(args) != 1:
            raise TypeError(
                f"SessionConfig union annotation {tp!r} is not a simple "
                f"Optional; no wire projection"
            )
        inner, _ = _project_config_type(args[0])
        return inner, True
    wire = _CONFIG_WIRE_TYPES.get(tp)
    if wire is None:
        raise TypeError(
            f"SessionConfig annotation {tp!r} has no wire projection; "
            f"extend _CONFIG_WIRE_TYPES / _project_config_type"
        )
    return wire, False


def config_catalog() -> dict:
    """Project the session-init handshake
    (:class:`acestep.streaming.config.SessionConfig`) into the same
    transport-agnostic ``name -> {type, required, nullable?, default?}``
    shape as the command/event catalogs.

    DERIVED from the dataclass via ``typing.get_type_hints`` (real resolved
    types, not annotation strings), so the config payload a UI sends at
    session start can't drift from what the server actually parses
    (``SessionConfig.from_dict``), and ``X | None`` fields project to
    ``nullable`` for free. Every field is wire-optional (the dataclass
    supplies defaults). The import is local so this module's top-level import
    stays torch-free / acestep-free; SessionConfig itself is both.
    """
    from dataclasses import MISSING
    from dataclasses import fields as _dc_fields
    from typing import get_type_hints

    from acestep.streaming.config import SessionConfig

    hints = get_type_hints(SessionConfig)
    out: dict = {}
    for f in _dc_fields(SessionConfig):
        wire_type, nullable = _project_config_type(hints[f.name])
        entry: dict = {"type": wire_type, "required": False}
        if nullable:
            entry["nullable"] = True
        if f.default is not MISSING and f.default is not None:
            entry["default"] = f.default
        elif f.default_factory is not MISSING:
            # Container fields (enabled_loras, lora_strengths, lora_paths)
            # default via factory; surface the produced value so the
            # contract shows their empty-container defaults too.
            entry["default"] = f.default_factory()
        out[f.name] = entry
    return out


def handshake_contract() -> dict:
    """The pre-stream handshake sub-protocol: the ``upload_track`` command and
    its ``upload_ok`` / ``upload_failed`` acks. Separate from the streaming
    commands/events because it's dispatched before a session exists."""
    return {
        "commands": _command_catalog_of(HANDSHAKE_COMMANDS),
        "events": _event_catalog_of(HANDSHAKE_EVENTS),
    }


def wire_contract() -> dict:
    """Full self-describing WS contract.

    ``{version, commands, events, config, handshake}``. Backs
    ``GET /api/protocol`` and the MCP ``describe_protocol`` tool. A consumer
    can build (and validate) every control message, decode every server event,
    assemble the session-init ``config`` payload, and run the pre-stream
    upload handshake from this alone — modulo the binary framing documented in
    each entry and the ``/api/knobs`` payload schema for ``params``.

    * ``commands`` / ``events`` — the per-tick streaming vocabulary.
    * ``config`` — the session-init payload, derived from SessionConfig.
    * ``handshake`` — the init-phase ``upload_track`` sub-protocol.
    """
    return {
        "version": PROTOCOL_VERSION,
        "commands": command_catalog(),
        "events": event_catalog(),
        "config": config_catalog(),
        "handshake": handshake_contract(),
    }


# ---------------------------------------------------------------------------
# Command-envelope validation
# ---------------------------------------------------------------------------

_COMMANDS_BY_NAME = {c.name: c for c in COMMANDS}
# Per-command {field name: FieldSpec}, precomputed once: coerce_command_payload
# runs on the 125 Hz params channel, so it must not rebuild this per call.
_COMMAND_FIELDS_BY_NAME = {
    c.name: {f.name: f for f in c.fields} for c in COMMANDS
}


def _coerce_command_field(cmd: str, field: "FieldSpec", val):
    """Coerce one command field by its declared type.

    Returns ``(value, error, keep)``; ``keep`` False means drop the field
    from the cleaned envelope (a type mismatch a prior value can't recover).
    Commands carry no numeric bounds (unlike knobs), so this only does type
    coercion + enum membership, never clamping.
    """
    t = field.type
    label = f"{cmd}.{field.name}"
    if val is None and field.nullable:
        return None, None, True  # explicit JSON null is part of the contract
    if t == "float":
        try:
            return float(val), None, True
        except (TypeError, ValueError):
            return None, f"{label}: {val!r} is not a number", False
    if t == "int":
        try:
            return int(round(float(val))), None, True
        except (TypeError, ValueError):
            return None, f"{label}: {val!r} is not an integer", False
    if t == "bool":
        if isinstance(val, bool):
            return val, None, True
        if isinstance(val, int) and val in (0, 1):
            return bool(val), None, True  # tolerate JSON 0/1
        return None, f"{label}: {val!r} is not a bool", False
    if t == "str":
        if isinstance(val, str):
            return val, None, True
        if val is None:
            return None, f"{label}: expected str, got None", False
        return str(val), None, True
    if t == "enum":
        if field.options and val not in field.options:
            return None, f"{label}: {val!r} not one of {list(field.options)}", False
        return val, None, True
    if t == "dict":
        if isinstance(val, dict):
            return val, None, True
        return None, f"{label}: expected object, got {type(val).__name__}", False
    if t == "list":
        if isinstance(val, list):
            return val, None, True
        return None, f"{label}: expected array, got {type(val).__name__}", False
    return val, None, True


def coerce_command_payload(name: str, payload: dict) -> tuple:
    """Validate a command envelope against its :class:`CommandSpec`.

    The wire-vocabulary analog of
    :func:`acestep.streaming.knobs.coerce_knob_values`: the single place that
    turns the command registry into enforcement, so every transport (the
    onboard MCP control bus today; the ws_adapter dispatcher in a later phase)
    validates inbound commands against one contract instead of re-deriving
    per-field checks inline. Returns ``(clean, errors)``:

    * ``clean`` — a NEW envelope safe to forward. Declared scalar fields are
      coerced to their type; ``enum`` fields are checked against ``options``;
      ``dict``/``list`` fields are type-checked. The ``type`` key and any field
      absent from the spec pass through untouched (forward-compat, plus
      free-form payloads like ``params.raw`` whose schema is the separate
      ``/api/knobs`` manifest, NOT this contract).
    * ``errors`` — human-readable strings for every missing required field and
      every value dropped on a type/enum mismatch.

    Discrete callers (the MCP tools) raise when ``errors`` is non-empty so the
    agent gets feedback. This function itself never raises.

    Deliberately divergent from the knob coercer
    (:func:`acestep.streaming.knobs.coerce_knob_values`): command fields
    carry no numeric bounds and DROP on type mismatch; knobs CLAMP
    out-of-range numerics into ``[min, max]``. Don't merge the two — the
    divergence is the contract.
    """
    spec = _COMMANDS_BY_NAME.get(name)
    if spec is None:
        return dict(payload), [f"unknown command {name!r}"]
    fields_by_name = _COMMAND_FIELDS_BY_NAME[name]
    errors: list = []
    for f in spec.fields:
        if f.required and f.name not in payload:
            errors.append(f"{name}.{f.name}: required field missing")
    clean: dict = {}
    for key, val in payload.items():
        if key == "type":
            clean[key] = val
            continue
        f = fields_by_name.get(key)
        if f is None:
            clean[key] = val  # unknown field: pass through verbatim
            continue
        coerced, err, keep = _coerce_command_field(name, f, val)
        if err:
            errors.append(err)
        if keep:
            clean[key] = coerced
    return clean, errors

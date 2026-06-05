"""MRT2 sidecar frame protocol — the single source of truth.

The Magenta RT 2 model runs out-of-process (JAX has no CUDA on native
Windows; the generation loop lives in a WSL venv or on a pod) behind a
deliberately tiny TCP protocol. This module defines it once; both ends
import it:

* the in-process :class:`~acestep.streaming.mrt2.backend.MRT2Backend`
  imports it normally, and
* ``scripts/mrt2_sidecar.py`` loads THIS FILE via
  ``importlib.util.spec_from_file_location`` — never ``import acestep``
  — because the sidecar venv has magenta_rt + JAX but no torch, and the
  ``acestep`` package import would pull the GPU stack.

So: stdlib only. No numpy, no acestep imports.

Framing (both directions, little-endian):

    u32 payload_length | u8 kind | payload

``kind == MSG_JSON``: payload is a UTF-8 JSON object with a ``"type"``
field. Control plane:

    backend -> sidecar: {"type": "hello"}
                        {"type": "prompt", "tags": str, "tags_b": str|null}
                        {"type": "blend", "value": float}          # A<->B style blend in [0,1]
                        {"type": "knobs", "temperature"?: float, "top_k"?: int,
                         "cfg_musiccoca"?: float, "cfg_notes"?: float,
                         "cfg_drums"?: float}
                        {"type": "credit", "frames": int}          # grant N more 40ms frames
                        {"type": "ping", "t": float}
    sidecar -> backend: {"type": "meta", "sample_rate": int, "channels": int,
                         "frame_samples": int, "model": str}
                        {"type": "pong", "t": float}
                        {"type": "err", "message": str}

``kind == MSG_AUDIO`` (sidecar -> backend only): payload is

    u64 frame_index | u16 num_frames | f32 PCM interleaved
                                       [num_frames * FRAME_SAMPLES * CHANNELS]

``frame_index`` is the index of the FIRST frame in the chunk, counting
every frame the sidecar has emitted since its process started (it does
NOT reset per connection — the backend anchors on the first chunk it
sees). Audio is final on first emit: the model is autoregressive and
never refines, so frames are append-only by construction.

Flow control is credit-based: the sidecar generates only while it holds
credit, the backend grants credit to keep the frontier a configured
lead ahead of the playhead. The model outruns real time (mrt2_small =
~1.7x RT on the dev 5090), so credit, not throughput, paces generation
— and the buffered lead stays small because for an append-only model
the buffered lead IS the knob-to-ear latency.
"""

import json
import struct

# Audio shape (matches Magenta RT 2's SpectroStream output and,
# fortuitously, DEMON's engine rate — no resample needed).
SAMPLE_RATE = 48000
CHANNELS = 2
FRAME_SAMPLES = 1920          # one model frame = 40 ms at 48 kHz
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE

# Message kinds.
MSG_JSON = 1
MSG_AUDIO = 2

_LEN = struct.Struct("<I")
_KIND = struct.Struct("<B")
AUDIO_HDR = struct.Struct("<QH")  # frame_index, num_frames

# Default sidecar address. WSL2 forwards localhost, so the Windows-side
# server reaches a sidecar bound inside WSL at the same address.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7531

# Hard ceiling on one message's payload (1 MiB covers ~34 frames of
# f32 stereo; chunks are a handful of frames). Protects both ends from
# a corrupt length prefix.
MAX_PAYLOAD = 1 << 20


def pack_msg(kind: int, payload: bytes) -> bytes:
    """One wire message: length prefix + kind byte + payload."""
    return _LEN.pack(len(payload) + 1) + _KIND.pack(kind) + payload


def pack_json(obj: dict) -> bytes:
    return pack_msg(MSG_JSON, json.dumps(obj).encode("utf-8"))


def pack_audio(frame_index: int, num_frames: int, pcm_bytes: bytes) -> bytes:
    return pack_msg(
        MSG_AUDIO, AUDIO_HDR.pack(frame_index, num_frames) + pcm_bytes,
    )


def recv_exact(sock, n: int) -> bytes:
    """Read exactly ``n`` bytes from a blocking socket, or raise
    ``ConnectionError`` on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("sidecar connection closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock) -> tuple:
    """Read one framed message. Returns ``(kind, payload_bytes)``."""
    (length,) = _LEN.unpack(recv_exact(sock, _LEN.size))
    if not 1 <= length <= MAX_PAYLOAD + 1:
        raise ConnectionError(f"bad frame length {length}")
    body = recv_exact(sock, length)
    return body[0], body[1:]


def unpack_json(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


def unpack_audio(payload: bytes) -> tuple:
    """Returns ``(frame_index, num_frames, pcm_bytes)``."""
    frame_index, num_frames = AUDIO_HDR.unpack_from(payload, 0)
    return frame_index, num_frames, payload[AUDIO_HDR.size:]

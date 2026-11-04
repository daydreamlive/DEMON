"""Per-subscriber binary-codec helpers for the WS transport.

Wire-format details hoisted out of the WS adapter so a future transport
(VST plugin, second browser, etc.) can reuse them or swap them for its
own encoding.

- :class:`SliceCodec` owns the per-subscriber zstd compressor and the
  ``client_mirror`` (the delta basis for this subscriber). Computes one
  binary slice frame (header + zstd-compressed float16 delta) from an
  :class:`~acestep.streaming.events.AudioReady` event and updates the
  mirror in place.
- :func:`send_stem_payload` serializes the post-init or post-swap stem
  bundle (one JSON header + one binary float16 frame per stem, in
  display order ``vocals`` then ``instruments``).

Neither helper acquires the WS send lock; callers do, so that JSON +
binary follow-ups for one logical event stay atomic.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch
import zstandard as zstd

from .protocol import (
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
    SLICE_HDR_FMT,
)


class SliceCodec:
    """Per-subscriber binary-slice serializer.

    One instance per WS subscriber. Construct with the initial source
    buffer (which becomes the first delta basis), then :meth:`encode`
    each AudioReady event into a single wire frame. Call
    :meth:`replace_mirror` on swap so subsequent deltas chase the new
    buffer.
    """

    def __init__(self, initial_mirror: np.ndarray, zstd_level: int = 1):
        # ``copy()`` because the caller's ``initial_buffer`` may be a
        # view into a session-owned array. The codec mutates this in
        # place on every encode.
        self._mirror = initial_mirror.copy()
        self._zctx = zstd.ZstdCompressor(level=zstd_level)

    @property
    def mirror(self) -> np.ndarray:
        """Current delta basis. Subscribers may inspect for diagnostics
        but must not mutate."""
        return self._mirror

    def replace_mirror(self, new_mirror: np.ndarray) -> None:
        """Wholesale replace the mirror buffer. Used on swap so the
        next slice's delta is computed against the buffer the client
        just crossfaded into."""
        self._mirror = new_mirror.copy()

    def encode(
        self,
        audio: np.ndarray,
        *,
        start_sample: int,
        channels: int,
        tick_ms: float,
        dec_ms: float,
        num_gens: int,
    ) -> bytes | None:
        """Compute one wire frame for an audio slice and update the
        mirror in place. Returns ``None`` if the slice is empty
        (``start_sample`` past the mirror's end)."""
        ss = int(start_sample)
        se = min(ss + len(audio), len(self._mirror))
        if se <= ss:
            return None
        region = audio[: se - ss]
        mirror_region = self._mirror[ss:se]
        # Delta = what server has now minus what client has
        delta = (region - mirror_region).astype(np.float16)
        compressed = self._zctx.compress(delta.tobytes())
        # Mirror our copy to the *reconstruction the client will hold*, not
        # the exact ``region``. The client applies ``mirror += float32(delta)``
        # with ``delta`` quantized to float16, so storing the exact region
        # here would leave our baseline off by the float16 rounding error.
        # Every subsequent delta is encoded against that baseline, so the
        # error never gets corrected — it accumulates. With the per-tick,
        # heavily-overlapping windowed slices (a 0.36 s region re-patched
        # dozens of times per second) that drift compounds into visible
        # ghosting — multiple decoded versions stacked on top of each other.
        # Encoding against the quantized reconstruction keeps server and
        # client byte-identical, so each delta corrects toward the truth.
        self._mirror[ss:se] = mirror_region + delta.astype(np.float32)
        hdr = struct.pack(
            SLICE_HDR_FMT,
            SLICE_FLAG_DELTA,
            ss, se - ss, channels,
            tick_ms, dec_ms, num_gens,
        )
        return hdr + compressed


def chunked_ws_send(ws, data: bytes, chunk_size: int = 262144) -> None:
    """Send a binary payload as a FRAGMENTED message in ~256 KiB pieces.

    websockets-sync holds ``protocol_mutex`` across the whole ``send()``
    (``send_data`` → ``socket.sendall``), and ``recv_events`` — the thread
    that reads EVERY inbound frame — needs that same mutex. A single
    multi-MB sendall therefore freezes all reads until the peer drains it,
    which deadlocks against a peer that is itself backpressured waiting
    for us to read (observed live: an 11 MB stem send vs a VST mid
    ``write_audio`` upload — both sides wedged until keepalive killed the
    session). An *iterable* payload sends as one fragmented message with
    the mutex released between fragments, so reads interleave and the
    deadlock cannot form. Fragmentation is invisible at the message layer
    (clients reassemble; payload bytes are identical).
    """
    if len(data) <= chunk_size:
        ws.send(data)
        return
    mv = memoryview(data)
    ws.send(mv[i:i + chunk_size] for i in range(0, len(data), chunk_size))


def send_stem_payload(
    ws,
    *,
    fixture_name: str | None,
    source_mode: str | None,
    stems: dict[str, torch.Tensor],
) -> None:
    """Serialize a ``stem_assets`` JSON frame + one binary float16
    follow-up per stem (in display order: vocals, instruments).

    Caller must hold the per-WS ``send_lock`` so the JSON header and
    its binary follow-ups don't interleave with other concurrent
    sends.
    """
    order = ["vocals", "instruments"]
    first = stems[order[0]]
    frames = int(first.shape[-1])
    channels = int(first.shape[0])
    ws.send(json.dumps({
        "type": "stem_assets",
        "fixture_name": fixture_name or "",
        "sample_rate": SAMPLE_RATE,
        "channels": channels,
        "frames": frames,
        "stems": order,
        # Empty string = overlay-only push (late background-rip
        # delivery): the client must NOT treat it as a source-mode
        # change. None keeps the legacy "full" default for the init /
        # swap paths.
        "source_mode": source_mode if source_mode is not None else "full",
    }))
    for name in order:
        arr = stems[name].detach().cpu().numpy().T.astype(np.float16)
        chunked_ws_send(ws, arr.tobytes())


def send_bounce_payload(
    ws,
    *,
    request_id: str,
    audio: np.ndarray,
    sample_rate: int,
    num_gens: int,
) -> None:
    """Serialize a ``bounce_ready`` JSON header + ONE binary frame: the
    whole window as interleaved ``[frames, channels]`` float16 — the
    same framing as the ``ready`` initial buffer and ``stem_assets``.

    Caller must hold the per-WS ``send_lock`` so the header and its
    binary follow-up can't interleave with slice frames.
    """
    pcm = np.ascontiguousarray(audio, dtype=np.float16)
    ws.send(json.dumps({
        "type": "bounce_ready",
        "request_id": request_id,
        "sample_rate": int(sample_rate),
        "channels": int(pcm.shape[1]),
        "frames": int(pcm.shape[0]),
        "num_gens": int(num_gens),
    }))
    chunked_ws_send(ws, pcm.tobytes())


def send_oneshot_payload(
    ws,
    *,
    request_id: str,
    pcm: np.ndarray,
    sample_rate: int,
    ids,
    seeds,
    streams,
    ms: dict,
    redo_of=None,
    latent_header=None,
    latents=None,
) -> None:
    """Serialize a ``oneshot_ready`` JSON header + ONE binary frame: int16
    little-endian ``[versions, streams, frames, channels]`` (version-major,
    then stream, each block interleaved ``[frames, channels]``).

    ``latent_header`` (when given) goes on the header as is. ``latents``
    (float16 ``[V, n_streams, C, L]``, when given) is announced on the
    header as ``latents: {dtype, shape, bytes}`` and sent as a SECOND
    binary frame after the PCM: raw little-endian float16, C-order.

    Caller must hold the per-WS ``send_lock`` so the header and its binary
    follow-up can't interleave with other frames.
    """
    a = np.ascontiguousarray(pcm, dtype="<i2")
    if a.ndim != 4:
        raise ValueError(f"oneshot pcm must be [V, S, N, C], got {a.shape}")
    lat = None
    if latents is not None:
        lat = np.ascontiguousarray(latents, dtype="<f2")
        if lat.ndim != 4:
            raise ValueError(f"oneshot latents must be [V, S, C, L], got {lat.shape}")
    head = {
        "type": "oneshot_ready",
        "request_id": request_id,
        "ids": [str(i) for i in ids],
        "seeds": [int(x) for x in seeds],
        "streams": [int(x) for x in streams],
        "sample_rate": int(sample_rate),
        "channels": int(a.shape[3]),
        "frames": int(a.shape[2]),
        "encoding": "pcm_s16le",
        "ms": {str(k): float(v) for k, v in (ms or {}).items()},
        "redo_of": redo_of,
    }
    if latent_header is not None:
        head["latent_header"] = {
            "n_streams": int(latent_header["n_streams"]),
            "channels": int(latent_header["channels"]),
            "frames": int(latent_header["frames"]),
            "seconds": float(latent_header["seconds"]),
        }
    if lat is not None:
        head["latents"] = {
            "dtype": "float16",
            "shape": [int(x) for x in lat.shape],
            "bytes": int(lat.nbytes),
        }
    ws.send(json.dumps(head))
    chunked_ws_send(ws, a.tobytes())
    if lat is not None:
        chunked_ws_send(ws, lat.tobytes())


def encode_oneshot_slice(
    pcm: np.ndarray, *, start_sample: int, zctx, dec_ms: float = 0.0,
    num_gens: int = 0,
) -> bytes:
    """One binary slice frame in the ring's codec for a one-shot window:
    the ring's header (``SLICE_HDR_FMT``: flags, start_sample, frames,
    channels, tick_ms, dec_ms, num_gens) + zstd of float16 ``[frames,
    channels]``. The ring's codec compresses only its DELTA form (a RAW
    frame is uncompressed, and every client parser decompresses only
    DELTA), so this is the DELTA form against a zero basis: the payload is
    the samples themselves (``x - 0`` is exact in float16), and any ring
    slice parser decodes it to them."""
    a = np.ascontiguousarray(pcm, dtype="<f2")
    if a.ndim != 2:
        raise ValueError(f"slice pcm must be [frames, channels], got {a.shape}")
    hdr = struct.pack(
        SLICE_HDR_FMT,
        SLICE_FLAG_DELTA,
        int(start_sample), int(a.shape[0]), int(a.shape[1]),
        0.0, float(dec_ms), int(num_gens),
    )
    return hdr + zctx.compress(a.tobytes())


def send_oneshot_slice(ws, *, event, zctx) -> None:
    """Serialize a ``oneshot_slice`` JSON header + ONE binary frame (the
    window as :func:`encode_oneshot_slice`, ``num_gens`` = the window index).
    Caller holds the per-WS ``send_lock``."""
    frame = encode_oneshot_slice(
        event.pcm, start_sample=event.start_frame, zctx=zctx,
        num_gens=event.index,
    )
    head = {
        "type": "oneshot_slice",
        "request_id": event.request_id,
        "version": str(event.version),
        "seed": int(event.seed),
        "index": int(event.index),
        "count": int(event.count),
        "start_frame": int(event.start_frame),
        "frames": int(event.frames),
        "channels": int(event.channels),
        "sample_rate": int(event.sample_rate),
        "encoding": "slice_f16_zstd",
    }
    peaks = getattr(event, "peaks", None)
    if peaks is not None:
        head["peaks"] = {"mix": float(peaks["mix"]), "streams": [float(v) for v in peaks["streams"]]}
    ws.send(json.dumps(head))
    chunked_ws_send(ws, frame)


def send_oneshot_done(ws, *, event) -> None:
    """Serialize a ``oneshot_done`` JSON header; with ``latents`` on it, ONE
    binary frame follows (raw little-endian float16 ``[1, n_streams, C,
    L]``, C-order, as ``oneshot_ready``'s second frame). Caller holds the
    per-WS ``send_lock``."""
    lat = None
    if event.latents is not None:
        lat = np.ascontiguousarray(event.latents, dtype="<f2")
        if lat.ndim != 4:
            raise ValueError(f"oneshot latents must be [1, S, C, L], got {lat.shape}")
    head = {
        "type": "oneshot_done",
        "request_id": event.request_id,
        "version": str(event.version),
        "seed": int(event.seed),
        "frames": int(event.frames),
        "channels": int(event.channels),
        "sample_rate": int(event.sample_rate),
        "windows": int(event.windows),
        "sha256": str(event.sha256),
        "ms": {str(k): float(v) for k, v in (event.ms or {}).items()},
    }
    h = event.latent_header
    if h is not None:
        head["latent_header"] = {
            "n_streams": int(h["n_streams"]),
            "channels": int(h["channels"]),
            "frames": int(h["frames"]),
            "seconds": float(h["seconds"]),
        }
    if lat is not None:
        head["latents"] = {
            "dtype": "float16",
            "shape": [int(x) for x in lat.shape],
            "bytes": int(lat.nbytes),
        }
    ws.send(json.dumps(head))
    if lat is not None:
        chunked_ws_send(ws, lat.tobytes())

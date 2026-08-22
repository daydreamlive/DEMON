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

import hashlib
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

    # Canvas chunking for the per-slice ``canvas_root`` (export forensics):
    # fine enough that re-hashing the chunks one slice touches is cheap,
    # coarse enough that the root recompute (one sha over ~170 digests)
    # is trivial.
    CANVAS_CHUNK = 16_384

    @staticmethod
    def _as_client_reconstruction(buf: np.ndarray) -> np.ndarray:
        """Source buffers travel to the client as float16 (ws_adapter's
        init/swap sends) and are upcast on arrival — so the client's base
        state is the f16-quantized source, NOT the exact f32 source. The
        mirror must track the client's reconstruction byte-for-byte (see
        the encode() note), including this base: without the round-trip
        every abs_sha256 differs from the client's buffer by the source's
        quantization error, and export forensics can never match."""
        return buf.astype(np.float16).astype(np.float32)

    def __init__(self, initial_mirror: np.ndarray, zstd_level: int = 1):
        # Quantized like the client's copy (see _as_client_reconstruction);
        # also a fresh array (never a view into a session-owned buffer —
        # the codec mutates this in place on every encode).
        self._mirror = self._as_client_reconstruction(initial_mirror)
        self._rebuild_canvas_hashes()
        self._zctx = zstd.ZstdCompressor(level=zstd_level)
        # Pod-side monotonic slice counter (spec 06 §3) and the last
        # encoded frame's slice-hash report (spec 06 §2.3), consumed by
        # the transport to emit a ``slice.pod_hash`` ledger event. ``None``
        # until the first non-empty frame is encoded.
        self._slice_seq = 0
        self.last_slice_hash: dict | None = None

    @property
    def mirror(self) -> np.ndarray:
        """Current delta basis. Subscribers may inspect for diagnostics
        but must not mutate."""
        return self._mirror

    def replace_mirror(self, new_mirror: np.ndarray) -> None:
        """Wholesale replace the mirror buffer. Used on swap so the
        next slice's delta is computed against the buffer the client
        just crossfaded into — which arrived over the wire as float16,
        so it is quantized here exactly like the init path."""
        self._mirror = self._as_client_reconstruction(new_mirror)
        self._rebuild_canvas_hashes()

    # ---- canvas root (export forensics, 06 §2.3) --------------------------
    #
    # After every slice the codec publishes a hash of the ENTIRE canvas as
    # the client then holds it: per-chunk SHA-256 digests folded into one
    # root. Clients copy their buffer for export under the same lock that
    # applies slices, so any exported file equals the canvas at some slice
    # boundary — and its recomputed root MUST appear among the receipted
    # ``canvas_root`` values. That gives a verifier 100% file coverage and
    # pins the export to a receipt timestamp, which per-region hashes of an
    # overlapping delta stream can never do (any later overlapping slice
    # invalidates a region snapshot).

    def _rebuild_canvas_hashes(self) -> None:
        ch = self.CANVAS_CHUNK
        m = self._mirror
        self._chunk_hashes = [
            hashlib.sha256(
                np.ascontiguousarray(m[i : i + ch], dtype=np.float32).tobytes()
            ).digest()
            for i in range(0, len(m), ch)
        ]

    def _refresh_canvas_hashes(self, lo: int, hi: int) -> None:
        ch = self.CANVAS_CHUNK
        m = self._mirror
        for idx in range(lo // ch, (max(hi, lo + 1) - 1) // ch + 1):
            start = idx * ch
            self._chunk_hashes[idx] = hashlib.sha256(
                np.ascontiguousarray(
                    m[start : start + ch], dtype=np.float32
                ).tobytes()
            ).digest()

    def _canvas_root(self) -> str:
        return hashlib.sha256(b"".join(self._chunk_hashes)).hexdigest()

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
            self.last_slice_hash = None
            return None
        region = audio[: se - ss]
        mirror_region = self._mirror[ss:se]
        # Delta = what server has now minus what client has
        delta = (region - mirror_region).astype(np.float16)
        delta_bytes = delta.tobytes()
        # Pod-side slice hash (spec 06 §2.3): SHA-256 over the uncompressed
        # interleaved float16 payload bytes — the exact bytes the client
        # gets back after zstd-decompressing this frame, so the pod and
        # client hashes compare directly for cross-checking.
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
        # ``abs_sha256`` (raw float32 bytes of this region) and
        # ``canvas_root`` (whole-canvas chunk fold) hash that same
        # reconstruction for export forensics (06 §2.3): a dragged float32
        # WAV carries these exact bytes.
        updated = mirror_region + delta.astype(np.float32)
        self._mirror[ss:se] = updated
        self._refresh_canvas_hashes(ss, se)
        self.last_slice_hash = {
            "sha256": hashlib.sha256(delta_bytes).hexdigest(),
            "abs_sha256": hashlib.sha256(
                np.ascontiguousarray(updated, dtype=np.float32).tobytes()
            ).hexdigest(),
            "canvas_root": self._canvas_root(),
            "canvas_chunk": self.CANVAS_CHUNK,
            "start_sample": ss,
            "num_samples": se - ss,
            "channels": int(channels),
            "slice_seq": self._slice_seq,
        }
        self._slice_seq += 1
        compressed = self._zctx.compress(delta_bytes)
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

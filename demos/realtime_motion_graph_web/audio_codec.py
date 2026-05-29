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
        self._mirror[ss:se] = region
        hdr = struct.pack(
            SLICE_HDR_FMT,
            SLICE_FLAG_DELTA,
            ss, se - ss, channels,
            tick_ms, dec_ms, num_gens,
        )
        return hdr + compressed


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
        "source_mode": source_mode or "full",
    }))
    for name in order:
        arr = stems[name].detach().cpu().numpy().T.astype(np.float16)
        ws.send(arr.tobytes())

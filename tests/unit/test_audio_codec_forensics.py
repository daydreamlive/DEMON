"""Export-forensics contract of the streaming codec (spec 06 §2.3).

``abs_sha256`` on every slice report must hash the bytes the CLIENT will
hold after applying the frame — that is what a pinned float32 drag-export
carries, and what scripts/verify-export.py (demon-provenance repo)
recomputes from the file. These tests simulate the client exactly:

  base  = float16 wire copy of the source, upcast to float32
          (ws_adapter's init/swap ``src.astype(np.float16).tobytes()``)
  apply = zstd-decompress the frame payload, ``buf += float32(f16 delta)``
          (RTMGWSClient.cpp / webui equivalent)

If the codec's mirror ever diverges from that reconstruction — as it did
when the mirror was seeded from the exact f32 source instead of the
quantized wire copy — every abs hash silently stops matching real exports.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.realtime_motion_graph_web.audio_codec import SliceCodec
from demos.realtime_motion_graph_web.protocol import SLICE_HDR_FMT

HDR_SIZE = struct.calcsize(SLICE_HDR_FMT)


def _client_apply(client: np.ndarray, frame: bytes) -> tuple[int, int]:
    """Apply one wire frame the way the plugin does; returns (start, end)."""
    flags, ss, n, ch, _tick, _dec, _gens = struct.unpack_from(
        SLICE_HDR_FMT, frame, 0,
    )
    assert flags == 1, "delta frames expected"
    raw = zstandard.ZstdDecompressor().decompress(
        frame[HDR_SIZE:], max_output_size=n * ch * 2,
    )
    delta = np.frombuffer(raw, dtype=np.float16).reshape(-1, ch)
    end = min(ss + n, len(client))
    client[ss:end] += delta[: end - ss].astype(np.float32)
    return ss, end


def test_abs_sha256_matches_client_reconstruction():
    rng = np.random.default_rng(3)
    src = (rng.standard_normal((48_000, 2)) * 0.2).astype(np.float32)

    codec = SliceCodec(src)
    # The client's base state: the f16 wire copy, upcast on arrival.
    client = src.astype(np.float16).astype(np.float32)

    # Overlapping re-patches, like live windowed slices.
    for start, n in [(0, 17_280), (8_640, 17_280), (17_280, 17_280),
                     (0, 17_280), (30_000, 18_000)]:
        audio = (rng.standard_normal((n, 2)) * 0.2).astype(np.float32)
        frame = codec.encode(
            audio, start_sample=start, channels=2,
            tick_ms=0.0, dec_ms=0.0, num_gens=1,
        )
        assert frame is not None
        report = dict(codec.last_slice_hash)

        ss, end = _client_apply(client, frame)
        got = hashlib.sha256(
            np.ascontiguousarray(client[ss:end], dtype="<f4").tobytes()
        ).hexdigest()
        assert got == report["abs_sha256"], (
            f"client reconstruction diverged at slice ss={ss}"
        )
        # And the wire hash still covers the payload bytes (unchanged
        # cross-check contract).
        raw = zstandard.ZstdDecompressor().decompress(
            frame[HDR_SIZE:], max_output_size=(end - ss) * 2 * 2,
        )
        assert hashlib.sha256(raw).hexdigest() == report["sha256"]


def test_replace_mirror_tracks_the_wire_copy_too():
    rng = np.random.default_rng(4)
    src = (rng.standard_normal((10_000, 2)) * 0.2).astype(np.float32)
    codec = SliceCodec(src)

    swapped = (rng.standard_normal((10_000, 2)) * 0.3).astype(np.float32)
    codec.replace_mirror(swapped)
    client = swapped.astype(np.float16).astype(np.float32)

    audio = (rng.standard_normal((4_096, 2)) * 0.2).astype(np.float32)
    frame = codec.encode(
        audio, start_sample=100, channels=2,
        tick_ms=0.0, dec_ms=0.0, num_gens=1,
    )
    report = dict(codec.last_slice_hash)
    ss, end = _client_apply(client, frame)
    got = hashlib.sha256(
        np.ascontiguousarray(client[ss:end], dtype="<f4").tobytes()
    ).hexdigest()
    assert got == report["abs_sha256"]

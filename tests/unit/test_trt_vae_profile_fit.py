"""TRT VAE profile-fit guard.

``_trt_vae_profile_fits`` decides whether a process-wide cached TRT VAE
engine can take a given input shape, so a caller with an eager VAE can
fall back instead of letting TRT reject the shape mid-encode (the
"120 s upload vs the live session's 60 s engine" hazard). Contract:

- True / False when the verdict is known from the cached engine's
  optimization profile,
- None when it can't be determined (engine not cached, engine slot
  empty, TRT API error) — callers treat None as "behave as before".

The engine cache is faked at the module seam; no TRT runtime needed.
"""

from __future__ import annotations

import os

import acestep.nodes.vae_nodes as vn

_PATH = "/tmp/fake_vae_encode.engine"


class _FakeEngine:
    def __init__(self, mn, opt, mx, error=None, extra_profiles=()):
        # Profile 0 plus any extra (mn, opt, mx) triples.
        self._profiles = [(mn, opt, mx), *extra_profiles]
        self._error = error

    @property
    def num_optimization_profiles(self):
        return len(self._profiles)

    def get_tensor_profile_shape(self, tensor_name, profile_index):
        if self._error is not None:
            raise self._error
        return self._profiles[profile_index]


def _install(monkeypatch, engine):
    monkeypatch.setattr(
        vn, "_trt_vae_cache",
        {os.path.abspath(_PATH): {"engine": engine}},
    )


def test_unknown_when_engine_not_cached(monkeypatch):
    monkeypatch.setattr(vn, "_trt_vae_cache", {})
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 48_000)) is None


def test_unknown_when_cache_entry_has_no_engine(monkeypatch):
    monkeypatch.setattr(
        vn, "_trt_vae_cache", {os.path.abspath(_PATH): {"engine": None}},
    )
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 48_000)) is None


def test_fits_when_shape_inside_profile(monkeypatch):
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
    ))
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 1_000_000)) is True
    # Boundary values are inside the profile.
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 48_000)) is True
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 2_880_000)) is True


def test_rejects_shape_outside_profile(monkeypatch):
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
    ))
    # The 120s-upload-vs-60s-engine case: too many samples.
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 5_760_000)) is False
    # Too few.
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 1_000)) is False
    # Wrong batch.
    assert vn._trt_vae_profile_fits(_PATH, "audio", (2, 2, 1_000_000)) is False


def test_rejects_rank_mismatch(monkeypatch):
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
    ))
    assert vn._trt_vae_profile_fits(_PATH, "audio", (2, 48_000)) is False


def test_any_profile_fitting_is_enough(monkeypatch):
    # Multi-profile engine: profile 0 is too small, profile 1 covers the
    # shape — the guard must not false-negative into the eager fallback
    # by only consulting profile 0.
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
        extra_profiles=[
            ((1, 2, 2_880_000), (1, 2, 5_760_000), (1, 2, 11_520_000)),
        ],
    ))
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 5_760_000)) is True
    # Outside every profile still rejects.
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 20_000_000)) is False


def test_unknown_on_trt_api_error(monkeypatch):
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
        error=RuntimeError("TRT API exploded"),
    ))
    assert vn._trt_vae_profile_fits(_PATH, "audio", (1, 2, 1_000_000)) is None


# ---------------------------------------------------------------------------
# Chunk-aware encode guard
# ---------------------------------------------------------------------------
#
# ``_trt_vae_encode_fits_or_chunkable`` is the encode node's variant of
# the fit guard: a samples dim above the profile max must NOT trigger
# the eager fallback, because _trt_vae_encode serves it via overlapping
# chunks — that is the very case the chunked encode was built for
# (>60 s upload vs the pinned 60 s engine). Only non-chunkable
# mismatches (rank, batch/channel, below the profile min) reject.


def test_chunkable_above_profile_max_uses_trt(monkeypatch):
    # 60 s engine (2_880_000 samples = 1500 frames), 120 s upload.
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
    ))
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (1, 2, 5_760_000)) is True


def test_chunk_guard_still_accepts_in_profile_shapes(monkeypatch):
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
    ))
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (1, 2, 1_000_000)) is True
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (1, 2, 2_880_000)) is True


def test_chunk_guard_rejects_non_chunkable_mismatches(monkeypatch):
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
    ))
    # Below the profile min: chunking can't help a too-short input.
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (1, 2, 1_000)) is False
    # Batch out of range.
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (2, 2, 5_760_000)) is False
    # Rank mismatch.
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (2, 48_000)) is False


def test_chunk_guard_rejects_engine_too_small_to_chunk(monkeypatch):
    # Max window of 128 frames leaves no core beyond the 2 x 64-frame
    # margins — _plan_encode_chunks would raise, so reject up front.
    too_small = 2 * vn._VAE_ENCODE_CHUNK_MARGIN_FRAMES * vn._VAE_SAMPLES_PER_FRAME
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, too_small), mx=(1, 2, too_small),
    ))
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (1, 2, too_small + 1)) is False


def test_chunk_guard_unknown_when_engine_not_cached(monkeypatch):
    monkeypatch.setattr(vn, "_trt_vae_cache", {})
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (1, 2, 5_760_000)) is None


def test_chunk_guard_unknown_on_trt_api_error(monkeypatch):
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
        error=RuntimeError("TRT API exploded"),
    ))
    assert vn._trt_vae_encode_fits_or_chunkable(_PATH, (1, 2, 5_760_000)) is None


# ---------------------------------------------------------------------------
# Cold-cache shape rejection → eager fallback (encode node)
# ---------------------------------------------------------------------------
#
# When the engine is NOT cached yet the fit guard returns None ("use
# TRT"); the engine then loads inside _trt_vae_encode and can still
# reject the shape. An eager-VAE handler must fall back instead of
# failing the upload with the original "rejected input shape" error.


from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


def _eager_handler(with_vae=True):
    @contextmanager
    def _load_model_context(name):
        yield

    return SimpleNamespace(
        vae=object() if with_vae else None,
        device="cpu",
        dtype=torch.float32,
        _load_model_context=_load_model_context,
        _encode_audio_to_latents=lambda wf: torch.zeros(
            (1, 8, 64), dtype=torch.float32,
        ),
    )


def _run_encode_node(monkeypatch, handler, trt_error):
    monkeypatch.setattr(vn, "_trt_available", lambda: True)
    monkeypatch.setattr(vn, "_find_best_vae_engine", lambda c: _PATH)
    monkeypatch.setattr(vn, "_trt_vae_cache", {})  # cold: fit guard → None

    def _reject(*_args, **_kwargs):
        raise trt_error

    monkeypatch.setattr(vn, "_trt_vae_encode", _reject)
    node = vn.VAEEncodeAudio()
    return node.execute(
        vae=SimpleNamespace(handler=handler),
        audio=SimpleNamespace(waveform=torch.zeros((2, 480))),
    )


def test_cold_cache_shape_rejection_falls_back_to_eager(monkeypatch):
    result = _run_encode_node(
        monkeypatch,
        _eager_handler(with_vae=True),
        RuntimeError("TRT VAE encode rejected input shape: (1, 2, 480)"),
    )
    assert result["latent"].tensor.shape == (1, 8, 64)  # eager path ran


def test_shape_rejection_without_eager_vae_still_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="rejected input shape"):
        _run_encode_node(
            monkeypatch,
            _eager_handler(with_vae=False),
            RuntimeError("TRT VAE encode rejected input shape: (1, 2, 480)"),
        )


def test_non_shape_trt_errors_still_raise(monkeypatch):
    with pytest.raises(RuntimeError, match="TRT VAE encode failed"):
        _run_encode_node(
            monkeypatch,
            _eager_handler(with_vae=True),
            RuntimeError("TRT VAE encode failed"),
        )


def test_warm_cache_long_upload_stays_on_trt_chunked_path(monkeypatch):
    # The PR-242 review case: cached 60 s engine, eager-VAE handler
    # (upload encoder session), >60 s upload. The guard must keep the
    # TRT path so _trt_vae_encode chunks it, instead of pulling the
    # eager VAE onto the GPU.
    monkeypatch.setattr(vn, "_trt_available", lambda: True)
    monkeypatch.setattr(vn, "_find_best_vae_engine", lambda c: _PATH)
    _install(monkeypatch, _FakeEngine(
        mn=(1, 2, 48_000), opt=(1, 2, 1_440_000), mx=(1, 2, 2_880_000),
    ))

    trt_called = []

    def _fake_trt_encode(waveform, path, device):
        trt_called.append(tuple(waveform.shape))
        return torch.zeros((1, 64, 3000), dtype=torch.float32)

    monkeypatch.setattr(vn, "_trt_vae_encode", _fake_trt_encode)
    result = vn.VAEEncodeAudio().execute(
        vae=SimpleNamespace(handler=_eager_handler(with_vae=True)),
        audio=SimpleNamespace(waveform=torch.zeros((2, 5_760_000))),
    )
    assert trt_called == [(1, 2, 5_760_000)]  # TRT ran, eager did not
    assert result["latent"].tensor.shape == (1, 3000, 64)

"""Source / conditioning resolver helpers (sidecar-aware).

Pure, transport-agnostic helpers for turning a (fixture-name, audio)
input pair into a ``PreparedSource`` plus the BPM / key / time-signature
needed by the text encoder.
"""

import struct

import numpy as np
import torch

from acestep.audio.key_detection import detect_key
from acestep.constants import VALID_TIME_SIGNATURES
from acestep.engine.obs import logger
from acestep.engine.session import PreparedSource, Session
from acestep.audio_clips import (
    audio_clip_sidecar,
    audio_clip_track_metadata,
    resolve_audio_clip,
)
from acestep.fixtures import FixtureSidecar, audio_fixture
from acestep.nodes.types import Audio, Latent


# Audio sample rate the ACE-Step v1.5 family is trained on. Duplicated
# from ``demos/realtime_motion_graph_web/protocol.py`` (and many other
# call sites — see tests/, scripts/) so this module stays free of demo
# imports.
SAMPLE_RATE = 48000


def _try_load_sidecar(
    fixture_name: str | None, *, samples: int, source_mode: str | None = "full",
) -> FixtureSidecar | None:
    """Look up a fixture sidecar; return None on miss / mismatch.

    Length check guards against runtime truncation that disagrees with
    what the sidecar was precomputed for (e.g. a smaller TRT profile
    cap kicking in). The caller falls back to live computation in that
    case so cached tensor shapes can't poison the streaming pipeline.

    Sidecars are not checkpoint-gated; the VAE and semantic
    tokenizer/detokenizer that produce the cached tensors are shared
    across the ACE-Step v1.5 family.
    """
    if not fixture_name:
        return None
    try:
        sc = audio_clip_sidecar(fixture_name, source_mode)
    except Exception as e:
        logger.warning(
            "sidecar_lookup_failed fixture={} error={}", fixture_name, e,
        )
        return None
    if sc is None:
        return None
    if sc.samples != samples:
        logger.warning(
            "sidecar_length_mismatch fixture={} sidecar_samples={} "
            "runtime_samples={}",
            fixture_name, sc.samples, samples,
        )
        return None
    return sc


def _decode_audio_msg(audio_msg: bytes) -> torch.Tensor:
    """Parse a binary audio frame into a [≤2, N] float32 tensor.

    Wire format (shared by the init handshake, ``swap_source``,
    ``set_timbre_source``, ``set_structure_source``): little-endian
    ``<II`` header carrying (channels, samples), followed by interleaved
    float32 PCM. Returns the waveform clipped to stereo (the model only
    consumes 2 channels).
    """
    ch, n = struct.unpack("<II", audio_msg[:8])
    arr = np.frombuffer(audio_msg[8:], dtype=np.float32).reshape(n, ch)
    return torch.from_numpy(arr.T.copy())[:2]


def _load_waveform_from_path(path: str) -> torch.Tensor:
    """Read an on-disk audio file into the exact shape
    ``_decode_audio_msg`` produces (``[≤2, N]`` float32 at
    ``SAMPLE_RATE``), resampling if the file isn't already at the backend
    rate."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    except Exception:
        import librosa
        mono, sr = librosa.load(path, sr=None, mono=True)
        data = np.stack([mono, mono], axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(
            data.T, orig_sr=sr, target_sr=SAMPLE_RATE
        ).T.astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return torch.from_numpy(np.ascontiguousarray(data.T, dtype=np.float32))[:2]


def _load_known_fixture_waveform(name: str) -> torch.Tensor:
    """Load a known fixture's audio from the pod's own fixture cache and
    return it in the exact shape ``_decode_audio_msg`` produces
    (``[≤2, N]`` float32 at ``SAMPLE_RATE``).

    The pod already serves this file at ``/fixtures/<name>``; for known
    fixtures the browser shouldn't have to download → decode → re-upload
    ~20 MB of PCM over the WebSocket (~11 s on the measured cold path).
    Same uniform-path output as the upload route, so every downstream
    consumer (sidecar resolve, echoed initial buffer, prepare_source
    fallback) is unchanged.
    """
    return _load_waveform_from_path(str(audio_fixture(name)))


def _load_clip_waveform(name: str) -> torch.Tensor:
    """Load an upload-or-fixture clip from the pod's disk by name.

    The swap analog of :func:`_load_known_fixture_waveform`: when the user
    picks a track that already lives on the server (a built-in fixture or
    a persisted user upload), there's no reason for the browser to decode
    it and re-upload the PCM only for the server to re-encode and re-rip
    stems. ``resolve_audio_clip`` checks the user-upload library first,
    then the fixture library, so the same name the dropdown shows lands
    on the right file. Output shape matches ``_decode_audio_msg`` so the
    swap path is identical to the client-supplied-PCM case from here on,
    and the sidecar / stem caches key off ``fixture_name`` exactly as
    before.
    """
    return _load_waveform_from_path(str(resolve_audio_clip(name)))


_VALID_TIME_SIG_STRS = frozenset(str(s) for s in VALID_TIME_SIGNATURES)


def _normalize_time_signature(value: object) -> str | None:
    """Coerce a wire-side time-signature value to one of
    ``VALID_TIME_SIGNATURES`` as a string. Returns ``None`` for
    unrecognized input (caller falls back to the sidecar / default
    instead of poisoning the encoder with junk)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        s = str(int(value))
        return s if s in _VALID_TIME_SIG_STRS else None
    if isinstance(value, str):
        s = value.strip()
        return s if s in _VALID_TIME_SIG_STRS else None
    return None


def _metadata_bpm(meta: dict) -> int | None:
    value = meta.get("bpm")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _metadata_str(meta: dict, key: str) -> str | None:
    value = meta.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _resolve_bpm_key_source(
    session: Session,
    *,
    audio_in: Audio,
    fixture_name: str | None,
    samples: int,
    key_override: str | None = None,
    time_signature_override: str | None = None,
) -> tuple[PreparedSource, int, str, str]:
    """Resolve (source, bpm, key, time_signature) for a (fixture, audio) pair.

    For known fixtures with a sidecar present (JSON+safetensors in the
    dataset or local staging dir, matching audio length), returns the
    cached source latent + context_latent and reads BPM, key, and
    time_signature from the sidecar JSON. Skips librosa beat tracking,
    CNN key detection, and ``Session.prepare_source`` — the
    prompt-independent half of the per-connect work.

    Conditioning is *not* cached (see fixtures.py). Callers run
    ``Session.encode_text`` against ``source.latent`` themselves; with
    the source latent already on GPU this is ~60ms warm.

    Falls through to live librosa + detect_key + prepare_source when:
      - ``fixture_name`` is None / unknown
      - sidecar files aren't in the dataset yet
      - audio-length truncation mismatch (e.g. operator's TRT profile
        cap is smaller than the natural fixture length)

    Clean v2 ``track.json`` metadata wins for BPM/key/time-signature when
    present because that file is the editable track-level authority.
    ``key_override`` and ``time_signature_override`` are the operator's
    manual choices coming from the swap_source path and are consulted on
    the live path only when track metadata does not pin those fields.
    """
    track_meta = audio_clip_track_metadata(fixture_name) if fixture_name else {}
    meta_bpm = _metadata_bpm(track_meta)
    meta_key = _metadata_str(track_meta, "key")
    meta_time_signature = _normalize_time_signature(track_meta.get("time_signature"))

    sc = _try_load_sidecar(fixture_name, samples=samples)

    if sc is not None:
        device = session.handler.device
        dtype = session.handler.dtype
        source = PreparedSource(
            latent=Latent(tensor=sc.latent.to(device, dtype).contiguous()),
            context_latent=Latent(tensor=sc.context_latent.to(device, dtype).contiguous()),
        )
        bpm = meta_bpm if meta_bpm is not None else sc.bpm
        # Track metadata is the editable source of truth; otherwise the
        # sidecar metadata beats client-supplied swap overrides. That
        # prevents a stale dropdown value from the previous track from
        # masking the new track's recorded key.
        key = meta_key or sc.key
        if key_override and key_override != sc.key:
            logger.info(
                "sidecar_override_ignored fixture={} field=key "
                "override={} sidecar={}",
                fixture_name, key_override, sc.key,
            )
        # Same precedence rule for time signature.
        time_signature = meta_time_signature or sc.time_signature
        if (
            time_signature_override
            and time_signature_override != sc.time_signature
        ):
            logger.info(
                "sidecar_override_ignored fixture={} field=time_signature "
                "override={} sidecar={}",
                fixture_name, time_signature_override, sc.time_signature,
            )
        logger.info(
            "sidecar_hit fixture={} bpm={} key={} time_signature={}",
            fixture_name, bpm, key, time_signature,
        )
        return source, bpm, key, time_signature

    # Live path: librosa BPM, CNN key detection, full prepare_source.
    # No automated time-signature detector today; the operator override
    # wins, otherwise we default to "4" (matches the model's most-
    # supported meter).
    import librosa
    logger.info("bpm_key_detect_start")
    mono_np = audio_in.waveform.mean(dim=0).numpy()
    if meta_bpm is not None:
        bpm = meta_bpm
    else:
        bpm_raw, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
        bpm = int(round(float(np.asarray(bpm_raw).flat[0])))
    key = meta_key or key_override or detect_key(mono_np, SAMPLE_RATE)
    time_signature = meta_time_signature or time_signature_override or "4"
    logger.info(
        "bpm_key_detected bpm={} key={} time_signature={}",
        bpm, key, time_signature,
    )

    logger.info("prepare_source_start")
    return session.prepare_source(audio_in), bpm, key, time_signature

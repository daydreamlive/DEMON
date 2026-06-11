"""User-uploaded audio library.

Mirrors the test-fixture library (``acestep.fixtures``) but is
operator-managed: the rtmg backend's WebSocket upload handler writes
audio files into :func:`acestep.paths.user_uploads_dir` and pairs each
with a precomputed sidecar (see :mod:`acestep.sidecars`). Subsequent
sessions that reference the same name as ``fixture_name`` hit the same
sidecar fast path test fixtures do — there's only one code path
through the pipeline regardless of how the audio landed on disk.

This module owns only the disk-layout half (listing + path lookup +
sidecar read). The write half is :mod:`acestep.sidecars`; callers run
:func:`acestep.sidecars.encode_and_save_sidecar` against
``user_uploads_dir()`` directly.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from acestep.paths import user_uploads_dir
from acestep.sidecars import POOL, AudioSidecar, load_sidecar, save_sidecar_pair
from acestep.track_assets import (
    STEM_MODES,
    load_track_metadata,
    read_stem_wavs,
    save_track_metadata,
    sidecar_paths,
    source_audio_path,
    source_sidecar_name,
    track_dir,
    write_track_wav,
    write_stem_wavs,
)

if TYPE_CHECKING:
    import torch
    from acestep.engine.session import PreparedSource

# Audio file extensions accepted as user uploads. Anything Web Audio
# decodes that ``soundfile`` can also read back server-side via
# libsndfile. The realtime demo's HTTP server and upload handler import
# this so the allowlist has a single source of truth.
USER_UPLOAD_EXTS: frozenset[str] = frozenset({
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
})


@dataclass(frozen=True)
class UserUploadPacket:
    name: str
    bpm: int
    key: str
    time_signature: str
    duration_s: float
    samples: int
    channels: int
    sample_rate: int


def enumerate_user_uploads() -> list[str]:
    """List filenames in ``user_uploads_dir()`` with a recognised audio extension.

    Returns names (not full paths), sorted. Empty if the directory
    doesn't exist — callers should treat that as "no uploads yet", not
    as an error. Sidecar files (``*.sidecar.json``,
    ``*.sidecar.safetensors``) are filtered out by the extension check
    (``.json`` and ``.safetensors`` are not in :data:`USER_UPLOAD_EXTS`).
    """
    d = user_uploads_dir()
    if not d.is_dir():
        return []

    # Legacy flat layout wrote per-stem WAVs as ``<name>.<mode>.wav`` next to
    # the source. Those are generated artifacts, not selectable tracks, so
    # exclude them from the listing (the clean v2 layout keeps stems under
    # ``<track>/stems/`` and never reaches the flat branch).
    stem_suffixes = tuple(f".{mode}.wav" for mode in STEM_MODES)

    names: set[str] = set()
    for p in d.iterdir():
        if p.is_dir() and (p / "track.json").is_file() and (p / "source.wav").is_file():
            meta = load_track_metadata(d, p.name)
            names.add(str(meta.get("source_name") or meta.get("display_name") or p.name))
        elif (
            p.is_file()
            and p.suffix.lower() in USER_UPLOAD_EXTS
            and not p.name.lower().endswith(stem_suffixes)
        ):
            names.add(p.name)
    return sorted(names)


def wipe_user_uploads() -> int:
    """Delete every entry under ``user_uploads_dir()``.

    The directory is per-pod scratch space, not per-user storage —
    uploads are durable on Tigris (see demon-public-demo's
    ``app/api/sessions/uploads/audio/presign``); the on-disk copy
    here is a fast-path cache for the current session only. Pods are
    rented from a pool and reused across users, so any file we leave
    behind would surface to the next renter via
    :func:`enumerate_user_uploads` and the ``/api/user_uploads``
    HTTP endpoint — that's the cross-user fixture leak the rtmg
    backend now closes by calling this between sessions.

    Returns the number of top-level entries removed. Tolerates the
    directory being absent (returns 0). Individual delete failures
    are logged to stdout and skipped — a stuck file on one user's
    upload must not prevent us from cleaning up the others.
    """
    d = user_uploads_dir()
    if not d.is_dir():
        return 0
    removed = 0
    for entry in d.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        except OSError as exc:
            sys.stdout.write(
                f"[user_uploads] wipe failed for {entry.name!r}: {exc}\n"
            )
            sys.stdout.flush()
    return removed


def user_upload_audio(name: str) -> Path:
    """Return the local path to a user-upload audio file.

    Raises :class:`FileNotFoundError` when the named upload isn't on
    disk; callers in :func:`acestep.audio_clips.resolve_audio_clip`
    treat this as "fall through to test fixtures".
    """
    root = user_uploads_dir()
    p = source_audio_path(root, name)
    if not p.is_file():
        p = root / name
    if not p.is_file():
        raise FileNotFoundError(
            f"user upload not found: {name!r} (looked in {p.parent})"
        )
    return p


def user_upload_sidecar(
    name: str,
    source_mode: str | None = "full",
) -> Optional[AudioSidecar]:
    """Load the user-upload sidecar bundle for ``name`` if available and fresh.

    Same staleness semantics as :func:`acestep.fixtures.fixture_sidecar`
    (format_version match), enforced by
    :func:`acestep.sidecars.load_sidecar`. No ``KNOWN_*`` gate — user
    uploads are identified by filename alone.
    """
    d = user_uploads_dir()
    # Single source of truth for the "full → name, variant → name.mode"
    # convention (also normalizes an unrecognised mode to "full"), so the
    # loaded tensors and the name we label them with can't disagree.
    sidecar_name = source_sidecar_name(name, source_mode)
    json_path, sf_path = sidecar_paths(d, name, source_mode)
    if not (json_path.is_file() and sf_path.is_file()):
        json_path = d / f"{sidecar_name}.sidecar.json"
        sf_path = d / f"{sidecar_name}.sidecar.safetensors"
    if not (json_path.is_file() and sf_path.is_file()):
        return None
    return load_sidecar(json_path, sf_path, name=sidecar_name)


def load_user_upload_stems(
    name: str,
    *,
    waveform: "torch.Tensor",
    sample_rate: int = 48_000,
) -> Optional[dict[str, "torch.Tensor"]]:
    """Load cached Mel-Band RoFormer stems for ``name`` when they match."""
    return read_stem_wavs(
        user_uploads_dir(),
        name,
        waveform=waveform,
        sample_rate=sample_rate,
    )


def unique_user_upload_name(requested_name: str) -> str:
    """Return a filesystem-safe, collision-free canonical WAV name."""
    cleaned = Path(str(requested_name or "upload")).name
    stem = Path(cleaned).stem.strip() or "upload"
    safe = "".join(ch if ch.isalnum() or ch in "._- ()" else "_" for ch in stem)
    safe = safe.strip(" .") or "upload"
    root = user_uploads_dir()
    candidate = f"{safe}.wav"
    if not track_dir(root, candidate).exists() and not (root / candidate).exists():
        return candidate
    i = 1
    while True:
        candidate = f"{safe} ({i}).wav"
        if not track_dir(root, candidate).exists() and not (root / candidate).exists():
            return candidate
        i += 1


def find_duplicate_upload(
    requested_name: str,
    *,
    waveform: "torch.Tensor",
    sample_rate: int,
    tolerance: float = 1e-3,
) -> Optional[UserUploadPacket]:
    """Return a packet for an existing upload already holding this audio.

    The realtime client's "serialize inputs" export embeds a track's PCM as
    a 16-bit WAV; re-importing it pushes that audio back through the upload
    pipeline under its ORIGINAL name. Without dedup the name-collision branch
    in :func:`unique_user_upload_name` mints a ``Foo (1).wav`` duplicate and
    re-runs the (expensive) VAE encode + stem rip for audio already on disk.

    Dedup is anchored on the requested name's canonical track dir (the export
    preserves the name) and confirmed by a *tolerant* content compare. We do
    NOT compare :func:`acestep.track_assets.waveform_fingerprint`: the 16-bit
    WAV round trip shifts samples by up to one LSB (~3e-5), which is far below
    ``tolerance`` here but enough to flip a few hundred of the fingerprint's
    quantized sample bins, so an exact-hash equality check misses on exactly
    this round trip. A genuinely different track sharing the name diffs by
    orders of magnitude more than ``tolerance`` and is correctly not reused.

    Returns ``None`` (caller proceeds to a fresh encode) when no same-named
    track exists, its metadata lacks the fields needed to rebuild a packet,
    its length differs by more than one latent pool, its source WAV can't be
    read, or the content compare exceeds ``tolerance``.
    """
    import torch
    import soundfile as sf

    root = user_uploads_dir()
    meta = load_track_metadata(root, requested_name)
    if not meta:
        return None
    src = source_audio_path(root, requested_name)
    if not src.is_file():
        return None
    sr = int(meta.get("sample_rate") or 0)
    samples = int(meta.get("samples") or 0)
    if sr != int(sample_rate) or samples <= 0:
        return None
    # A different-length clip of the same song (a different trim window) is a
    # distinct selection and must re-upload. Allow drift up to one pool since
    # pool alignment can land a sample or two apart.
    if abs(samples - int(waveform.shape[-1])) > POOL:
        return None
    try:
        data, file_sr = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception:
        return None
    if int(file_sr) != int(sample_rate):
        return None
    existing = torch.from_numpy(data.T.copy()).float().mean(dim=0)
    incoming = waveform.detach().cpu().float().mean(dim=0)
    n = min(int(existing.shape[-1]), int(incoming.shape[-1]))
    if n <= 0:
        return None
    if float((existing[:n] - incoming[:n]).abs().max()) > tolerance:
        return None
    name = str(meta.get("source_name") or meta.get("display_name") or requested_name)
    channels = int(meta.get("channels") or int(waveform.shape[0]))
    return UserUploadPacket(
        name=name,
        bpm=int(meta.get("bpm") or 0),
        key=str(meta.get("key") or ""),
        time_signature=str(meta.get("time_signature") or "4"),
        duration_s=samples / sr,
        samples=samples,
        channels=channels,
        sample_rate=sr,
    )


def persist_user_upload_packet(
    name: str,
    *,
    waveform: "torch.Tensor",
    stems: dict[str, "torch.Tensor"],
    sources: dict[str, "PreparedSource"],
    sample_rate: int,
    checkpoint: str,
    bpm: int,
    key: str,
    time_signature: str,
) -> UserUploadPacket:
    """Write the complete canonical user-track packet in one place."""
    root = user_uploads_dir()
    root.mkdir(parents=True, exist_ok=True)
    samples = int(waveform.shape[-1])
    channels = int(waveform.shape[0])
    duration_s = samples / int(sample_rate)
    written_root = track_dir(root, name)
    # Only clean up on failure if this call is what created the directory.
    # A pre-existing dir (concurrent upload, leftover from a prior run that
    # collided through the TOCTOU window after unique_user_upload_name) must
    # not be destroyed by our rollback.
    created_dir = not written_root.exists()
    try:
        write_track_wav(root, name, waveform=waveform, sample_rate=sample_rate)
        write_stem_wavs(root, name, stems=stems, sample_rate=sample_rate)
        for mode, source in sources.items():
            source_waveform = waveform if mode == "full" else stems[mode]
            json_path, sf_path = sidecar_paths(root, name, mode)
            save_sidecar_pair(
                json_path,
                sf_path,
                latent=source.latent.tensor,
                context_latent=source.context_latent.tensor,
                checkpoint=checkpoint,
                bpm=bpm,
                key=key,
                time_signature=time_signature,
                duration_s=int(source_waveform.shape[-1]) / int(sample_rate),
                samples=int(source_waveform.shape[-1]),
                sample_rate=int(sample_rate),
                channels=int(source_waveform.shape[0]),
            )
        # Metadata last: its stems/sidecars manifest is derived from the
        # files written above, so it only advertises assets that exist.
        save_track_metadata(
            root,
            name,
            waveform=waveform,
            sample_rate=sample_rate,
            bpm=bpm,
            key=key,
            time_signature=time_signature,
        )
    except Exception:
        if created_dir and written_root.exists():
            import shutil
            shutil.rmtree(written_root, ignore_errors=True)
        raise
    return UserUploadPacket(
        name=name,
        bpm=int(bpm),
        key=str(key),
        time_signature=str(time_signature),
        duration_s=duration_s,
        samples=samples,
        channels=channels,
        sample_rate=int(sample_rate),
    )

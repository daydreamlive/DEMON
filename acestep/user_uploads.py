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

from pathlib import Path
from typing import Optional

from acestep.paths import user_uploads_dir
from acestep.sidecars import AudioSidecar, load_sidecar

# Audio file extensions accepted as user uploads. Anything Web Audio
# decodes that ``soundfile`` can also read back server-side via
# libsndfile. The realtime demo's HTTP server and upload handler import
# this so the allowlist has a single source of truth.
USER_UPLOAD_EXTS: frozenset[str] = frozenset({
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
})


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
    return sorted(
        p.name for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in USER_UPLOAD_EXTS
    )


def user_upload_audio(name: str) -> Path:
    """Return the local path to a user-upload audio file.

    Raises :class:`FileNotFoundError` when the named upload isn't on
    disk; callers in :func:`acestep.audio_clips.resolve_audio_clip`
    treat this as "fall through to test fixtures".
    """
    p = user_uploads_dir() / name
    if not p.is_file():
        raise FileNotFoundError(
            f"user upload not found: {name!r} (looked in {p.parent})"
        )
    return p


def user_upload_sidecar(name: str) -> Optional[AudioSidecar]:
    """Load the user-upload sidecar bundle for ``name`` if available and fresh.

    Same staleness semantics as :func:`acestep.fixtures.fixture_sidecar`
    (format_version match), enforced by
    :func:`acestep.sidecars.load_sidecar`. No ``KNOWN_*`` gate — user
    uploads are identified by filename alone.
    """
    d = user_uploads_dir()
    json_path = d / f"{name}.sidecar.json"
    sf_path = d / f"{name}.sidecar.safetensors"
    if not (json_path.is_file() and sf_path.is_file()):
        return None
    return load_sidecar(json_path, sf_path, name=name)

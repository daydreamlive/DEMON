"""Unified lookup across the test-fixture and user-upload libraries.

Callers that don't care whether a name belongs to a test fixture or a
user upload (the rtmg HTTP audio-serving route, the backend's
``set_*_fixture`` fast-path helper, the session-init sidecar
fast-path) use this module.

User uploads win over test fixtures on a name collision: an operator
who chose to override a fixture by uploading a same-named file is
making a deliberate choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from acestep.fixtures import audio_fixture, fixture_sidecar
from acestep.sidecars import AudioSidecar
from acestep.user_uploads import user_upload_audio, user_upload_sidecar


def resolve_audio_clip(name: str) -> Path:
    """Return the local path to an audio clip.

    Checks the user-upload library first, then the test-fixture
    library. Raises :class:`KeyError` (matching
    :func:`acestep.fixtures.audio_fixture`'s contract) when the name
    resolves to neither.
    """
    try:
        return user_upload_audio(name)
    except FileNotFoundError:
        pass
    return audio_fixture(name)


def audio_clip_sidecar(name: str) -> Optional[AudioSidecar]:
    """Look up a sidecar by name, user uploads first then test fixtures.

    Sidecars are not checkpoint-gated; the VAE and semantic
    tokenizer/detokenizer that produce the cached tensors are shared
    across the ACE-Step v1.5 family.
    """
    sc = user_upload_sidecar(name)
    if sc is not None:
        return sc
    return fixture_sidecar(name)

"""Lazy auto-downloading audio fixtures.

Backed by the ``daydreamlive/demon-fixtures`` dataset repo on Hugging
Face. The first call to :func:`audio_fixture` for a given name
downloads the file into the shared HF cache
(``~/.cache/huggingface/hub/`` by default); subsequent calls hit the
cache and are effectively free.

Adding a new fixture is a two-step process:
  1. ``huggingface-cli upload daydreamlive/demon-fixtures <file> --repo-type dataset``
  2. Add the filename to :data:`KNOWN_FIXTURES`.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "daydreamlive/demon-fixtures"
REPO_TYPE = "dataset"

KNOWN_FIXTURES: frozenset[str] = frozenset({
    "inside_confusion_loop_60s_gsm.wav",
    "inside_confusion_loop_120s_gsm.wav",
    "low_fi_Gm_loop_60s_gnm.wav",
    "low_fi_loop_120s_gnm.wav",
    "prog_rock_loop_60s_enm.wav",
    "prog_rock_loop_120s_enm.wav",
    "thrash_metal_loop_60s_enm.wav",
    "thrash_metal_loop_120s_enm.wav",
})


def audio_fixture(name: str) -> Path:
    """Return a local :class:`Path` to the named fixture, downloading on cache miss.

    Raises :class:`KeyError` if ``name`` is not in :data:`KNOWN_FIXTURES`.
    Network errors propagate from :func:`huggingface_hub.hf_hub_download`.
    """
    if name not in KNOWN_FIXTURES:
        raise KeyError(
            f"unknown fixture {name!r}; add it to KNOWN_FIXTURES "
            f"in acestep/fixtures.py after uploading to {REPO_ID}"
        )
    return Path(hf_hub_download(repo_id=REPO_ID, filename=name, repo_type=REPO_TYPE))


def ensure_all() -> list[Path]:
    """Pre-warm every known fixture. Returns the local paths in sorted order."""
    return [audio_fixture(name) for name in sorted(KNOWN_FIXTURES)]

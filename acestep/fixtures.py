"""Lazy auto-downloading audio test fixtures.

Backed by the ``daydreamlive/demon-fixtures-v2`` dataset repo on Hugging
Face. The first call to :func:`audio_fixture` for a given name
downloads the file into :func:`acestep.paths.fixtures_dir` (under
``MODELS_DIR``); subsequent calls hit that managed directory and are
effectively free. HF's own cache stays as the under-the-hood
incremental store; the managed dir is what callers see.

Adding a new fixture is a two-step process:
  1. ``huggingface-cli upload daydreamlive/demon-fixtures-v2 <file> --repo-type dataset``
  2. Add the filename to :data:`KNOWN_FIXTURES`.

Each fixture optionally has a sidecar pair in the same dataset
(``<name>.sidecar.json`` + ``<name>.sidecar.safetensors``), used by the
realtime demo to skip the prompt-independent half of per-connect
preprocessing. The sidecar *format* lives in :mod:`acestep.sidecars`
and is shared with the user-upload library
(:mod:`acestep.user_uploads`); this module only owns the HF-specific
lookup half (``KNOWN_FIXTURES`` gate + HF download fallback for missing
local files).

Sidecars are produced by ``scripts/calibration/precompute_fixture_sidecars.py``
and uploaded to the dataset alongside the WAVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from acestep.paths import fixtures_dir
from acestep.sidecars import (
    AudioSidecar,
    SIDECAR_FORMAT_VERSION,
    load_sidecar,
)
from acestep.track_assets import (
    load_json_metadata,
    read_stem_wavs,
    sidecar_asset_name,
    source_audio_name,
    source_audio_path,
    source_sidecar_name,
    stem_audio_name,
    track_metadata_name,
)

# Re-export for backward compat with callers that imported these from
# acestep.fixtures back when this module owned the sidecar format.
# The canonical location is now acestep.sidecars.
__all__ = [
    "AudioSidecar",
    "KNOWN_FIXTURES",
    "LEGACY_REPO_ID",
    "REPO_ID",
    "REPO_TYPE",
    "SIDECAR_FORMAT_VERSION",
    "audio_fixture",
    "ensure_all",
    "fixture_sidecar",
    "fixture_stems",
    "fixture_track_metadata",
    "parse_key_from_filename",
]

REPO_ID = "daydreamlive/demon-fixtures-v2"
LEGACY_REPO_ID = "daydreamlive/demon-fixtures"
REPO_TYPE = "dataset"

KNOWN_FIXTURES: frozenset[str] = frozenset({
    "inside_confusion_loop_60s_gsm.wav",
    "low_fi_Gm_loop_60s_gnm.wav",
    "prog_rock_loop_60s_enm.wav",
    "thrash_metal_loop_60s_enm.wav",
})


def _hf_download_to_fixtures_dir(filename: str, *, allow_legacy: bool = True) -> Path:
    """``hf_hub_download`` into ``fixtures_dir()``.

    Matches the pattern used by :mod:`acestep.model_downloader`
    (``snapshot_download(local_dir=...)``): files materialize in the
    managed dir under ``MODELS_DIR`` instead of the user's HF cache. HF
    still uses its own cache as the under-the-hood incremental store —
    we just point ``local_dir`` at our managed root so the materialized
    copy is what callers see.
    """
    fixtures_dir().mkdir(parents=True, exist_ok=True)
    for repo_id in (REPO_ID, LEGACY_REPO_ID):
        if repo_id == LEGACY_REPO_ID and not allow_legacy:
            break
        try:
            return Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=str(fixtures_dir()),
            ))
        except (EntryNotFoundError, RepositoryNotFoundError):
            if repo_id == LEGACY_REPO_ID or not allow_legacy:
                raise
    raise EntryNotFoundError("not found")


def audio_fixture(name: str) -> Path:
    """Return a local :class:`Path` to the named fixture, downloading on cache miss.

    Raises :class:`KeyError` if ``name`` is not in :data:`KNOWN_FIXTURES`.
    Network errors propagate from :func:`huggingface_hub.hf_hub_download`.
    Downloads materialize into ``fixtures_dir()``.
    """
    if name not in KNOWN_FIXTURES:
        raise KeyError(
            f"unknown fixture {name!r}; add it to KNOWN_FIXTURES "
            f"in acestep/fixtures.py after uploading to {REPO_ID}"
        )
    local = source_audio_path(fixtures_dir(), name)
    if local.is_file():
        return local
    try:
        return _hf_download_to_fixtures_dir(source_audio_name(name), allow_legacy=False)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return _hf_download_to_fixtures_dir(name, allow_legacy=True)


def ensure_all() -> list[Path]:
    """Pre-warm every known fixture. Returns the local paths in sorted order."""
    return [audio_fixture(name) for name in sorted(KNOWN_FIXTURES)]


# ---------------------------------------------------------------------------
# Key abbreviation parsing
# ---------------------------------------------------------------------------

# Filenames carry the ground-truth key as a trailing token, since the CNN
# detector misclassifies enough of the test set to be unreliable. The
# convention is ``<note><modifier><mode>``:
#
#   note      a-g (lowercase)
#   modifier  s = sharp, n = natural, f = flat (optional in 2-letter form)
#   mode      m = minor, M = major
#
# ``gsm`` -> "G# minor", ``gnm`` -> "G minor", ``enm`` -> "E minor".
# This is a one-time bridge to seed sidecars; once a fixture has a
# sidecar JSON, that JSON is authoritative and the filename stops being
# consulted.

_NOTE_TO_PITCH = {
    "a": "A", "b": "B", "c": "C", "d": "D",
    "e": "E", "f": "F", "g": "G",
}
_MODIFIER_TO_ACCIDENTAL = {"s": "#", "n": "", "f": "b"}


def _parse_key_suffix(suffix: str) -> Optional[str]:
    """Parse a bare suffix like 'gsm' / 'enm' / 'cM'. Returns None on failure."""
    if not suffix:
        return None
    mode_ch = suffix[-1]
    if mode_ch == "m":
        mode = "minor"
    elif mode_ch == "M":
        mode = "major"
    else:
        return None
    body = suffix[:-1]
    if not body:
        return None
    note = _NOTE_TO_PITCH.get(body[0].lower())
    if note is None:
        return None
    if len(body) == 1:
        accidental = ""
    elif len(body) == 2:
        accidental = _MODIFIER_TO_ACCIDENTAL.get(body[1].lower())
        if accidental is None:
            return None
    else:
        return None
    return f"{note}{accidental} {mode}"


def parse_key_from_filename(name: str) -> Optional[str]:
    """Extract the ACE-Step key string from a fixture filename.

    Splits on the last underscore in the stem and parses the trailing
    token. ``inside_confusion_loop_60s_gsm.wav`` -> ``"G# minor"``.
    Returns ``None`` if the suffix isn't recognized.
    """
    stem = Path(name).stem
    suffix = stem.rsplit("_", 1)[-1] if "_" in stem else stem
    return _parse_key_suffix(suffix)


# ---------------------------------------------------------------------------
# Fixture-sidecar lookup
# ---------------------------------------------------------------------------

def _resolve_fixture_asset(name: str, *, allow_legacy: bool = True) -> Optional[Path]:
    """Locate a fixture sidecar file by name.

    ``fixtures_dir()`` wins over HF. Returns None on miss (caller falls
    back to live computation). The local-first ordering means
    precompute output can be tested without pushing to the HF dataset;
    once uploaded, fresh clones get the sidecars from HF on first use
    (materialized into ``fixtures_dir()`` alongside the WAVs).
    """
    local = fixtures_dir() / name
    if local.is_file():
        return local
    try:
        return _hf_download_to_fixtures_dir(name, allow_legacy=allow_legacy)
    except (EntryNotFoundError, RepositoryNotFoundError):
        # A missing file OR a not-yet-created v2 repo is an expected miss,
        # not an error: callers fall back to legacy assets / live compute.
        # Stay quiet so probing v2 on every fixture lookup (before the
        # dataset is populated) doesn't spam the logs.
        return None
    except Exception as exc:
        # Treat any other download error (network, permissions) the same
        # as a miss so the demo stays usable offline or before sidecars
        # have been uploaded. Log it so an unreachable HF / 401 doesn't
        # look identical to "no sidecar exists" in production traces.
        print(f"[fixture_sidecar] HF download failed for {name}: {exc}")
        return None


def _resolve_sidecar_file(name: str) -> Optional[Path]:
    return _resolve_fixture_asset(name, allow_legacy=True)


def fixture_track_metadata(name: str) -> dict:
    """Load editable fixture metadata, falling back to legacy sidecar JSON."""
    if name not in KNOWN_FIXTURES:
        return {}
    track_path = _resolve_fixture_asset(track_metadata_name(name), allow_legacy=False)
    if track_path is not None:
        return load_json_metadata(track_path)
    legacy_sidecar = _resolve_fixture_asset(f"{name}.sidecar.json", allow_legacy=True)
    return load_json_metadata(legacy_sidecar) if legacy_sidecar is not None else {}


def fixture_sidecar(
    name: str,
    source_mode: str | None = "full",
) -> Optional[AudioSidecar]:
    """Load the test-fixture sidecar bundle for ``name`` if available and fresh.

    Returns ``None`` (not an exception) on any of:
      - ``name`` not in :data:`KNOWN_FIXTURES`
      - sidecar JSON or safetensors not present in the dataset
      - format_version mismatch

    Sidecars are NOT gated on the runtime checkpoint: the VAE and the
    semantic tokenizer/detokenizer that produce the cached tensors are
    shared across the ACE-Step v1.5 family. The JSON's ``checkpoint``
    field is informational only.

    On every miss past ``name in KNOWN_FIXTURES`` we print the reason
    so the demo's logs make it obvious why the sidecar fast path didn't
    fire (silent ``None`` returns previously left operators staring at
    "Detecting BPM + key..." without any clue whether the sidecar files
    were missing or stale).
    """
    if name not in KNOWN_FIXTURES:
        return None

    json_path = _resolve_fixture_asset(sidecar_asset_name(name, source_mode, "json"), allow_legacy=False)
    if json_path is None:
        sidecar_name = source_sidecar_name(name, source_mode)
        json_path = _resolve_sidecar_file(f"{sidecar_name}.sidecar.json")
    if json_path is None:
        print(f"[fixture_sidecar] {name}: sidecar JSON not found (local dir + HF dataset)")
        return None
    sf_path = _resolve_fixture_asset(sidecar_asset_name(name, source_mode, "safetensors"), allow_legacy=False)
    if sf_path is None:
        sidecar_name = source_sidecar_name(name, source_mode)
        sf_path = _resolve_sidecar_file(f"{sidecar_name}.sidecar.safetensors")
    if sf_path is None:
        print(f"[fixture_sidecar] {name}: sidecar safetensors not found (local dir + HF dataset)")
        return None

    sidecar_name = source_sidecar_name(name, source_mode)
    return load_sidecar(json_path, sf_path, name=sidecar_name)


def fixture_stems(name: str, *, waveform, sample_rate: int) -> Optional[dict]:
    """Load fixture vocal/instrumental WAV stems from local/HF assets."""
    if name not in KNOWN_FIXTURES:
        return None
    for mode in ("vocals", "instruments"):
        if _resolve_sidecar_file(stem_audio_name(name, mode)) is None:
            return None
    metadata = fixture_track_metadata(name)
    return read_stem_wavs(
        fixtures_dir(),
        name,
        waveform=waveform,
        sample_rate=sample_rate,
        metadata=metadata,
    )


# Backward-compat alias. Old callers imported ``FixtureSidecar`` from
# this module before the type was generalised. The dataclass itself
# lives in :mod:`acestep.sidecars` as :class:`AudioSidecar` now.
FixtureSidecar = AudioSidecar

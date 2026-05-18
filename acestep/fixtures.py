"""Lazy auto-downloading audio fixtures.

Backed by the ``daydreamlive/demon-fixtures`` dataset repo on Hugging
Face. The first call to :func:`audio_fixture` for a given name
downloads the file into the shared HF cache
(``~/.cache/huggingface/hub/`` by default); subsequent calls hit the
cache and are effectively free.

Adding a new fixture is a two-step process:
  1. ``huggingface-cli upload daydreamlive/demon-fixtures <file> --repo-type dataset``
  2. Add the filename to :data:`KNOWN_FIXTURES`.

Each fixture optionally has a sidecar pair in the same dataset, used
by the realtime demo to skip the prompt-independent half of per-connect
preprocessing:

  ``<name>.sidecar.json``
      bpm, key, time_signature, duration metadata.
  ``<name>.sidecar.safetensors``
      pre-encoded source latent + semantic context_latent.

Conditioning (encode_text) is intentionally *not* cached: the demo's
blended-prompt UI typically drifts off any baked tags within seconds,
and encode_text is cheap enough (~60ms warm) that caching it is not
worth the complexity. See :func:`fixture_sidecar`. Sidecars are
produced by ``scripts/calibration/precompute_fixture_sidecars.py`` and uploaded
to the dataset alongside the WAVs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

REPO_ID = "daydreamlive/demon-fixtures"
REPO_TYPE = "dataset"

# Local staging dir for sidecars that haven't been pushed to HF yet.
# scripts/calibration/precompute_fixture_sidecars.py defaults its --out here, and
# fixture_sidecar() checks this first before falling through to the HF
# dataset. Override via DEMON_FIXTURE_SIDECARS_DIR.
_DEFAULT_LOCAL_SIDECAR_DIR = Path(__file__).resolve().parents[1] / "out" / "fixture_sidecars"


def _local_sidecar_dir() -> Path:
    override = os.environ.get("DEMON_FIXTURE_SIDECARS_DIR")
    return Path(override) if override else _DEFAULT_LOCAL_SIDECAR_DIR

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

# Sidecar schema version. Bump when the on-disk format changes in a way
# that prior sidecars can't satisfy. Loader refuses mismatches.
SIDECAR_FORMAT_VERSION = 2


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
# Sidecar loader
# ---------------------------------------------------------------------------

@dataclass
class FixtureSidecar:
    """Loaded sidecar bundle for a known fixture.

    Caches the deterministic, prompt-independent preprocessing the
    realtime demo would otherwise do on every connect: BPM (librosa),
    key (parsed from the filename suffix, since the CNN classifier
    misclassifies enough of the test set to be unreliable), and the
    source latent + semantic context_latent from
    ``Session.prepare_source``. Conditioning (encode_text) is *not*
    cached; the demo's blended-prompt UI means the client typically
    diverges from any baked tags within seconds of connecting, and
    encode_text is cheap enough (~60ms warm) that the cache savings
    don't justify the server-authoritative complication.

    Sidecars are NOT checkpoint-specific. The VAE that produces
    ``latent`` and the semantic tokenizer/detokenizer that produce
    ``context_latent`` are shared across the ACE-Step v1.5 family
    (turbo, xl-turbo, ...); only the DiT differs. ``produced_with``
    records which checkpoint happened to generate the file for
    provenance, but the loader does not gate on it.
    """

    name: str
    bpm: int
    key: str
    # Stringified meter numerator (matches the encoder boundary in
    # ``Session.encode_text``, which prepends ``- timesignature: <s>``
    # to the prompt). One of ``VALID_TIME_SIGNATURES`` (``"2"``, ``"3"``,
    # ``"4"``, ``"6"``); defaults to ``"4"`` when older sidecars don't
    # carry the field (loader uses ``meta.get(..., "4")``).
    time_signature: str
    duration_s: float
    samples: int
    sample_rate: int
    channels: int
    # Checkpoint the precompute script ran on. Informational only;
    # legacy ``checkpoint`` field is honored when ``produced_with`` is
    # missing. Empty string when neither is present.
    produced_with: str
    latent: torch.Tensor
    context_latent: torch.Tensor


def _resolve_sidecar_file(name: str) -> Optional[Path]:
    """Locate a sidecar file by name. Local staging dir wins over HF.

    Returns None on miss (caller falls back to live computation). The
    local-first ordering means precompute output can be tested without
    pushing to the HF dataset; once uploaded, fresh clones get the
    sidecars from HF on first use.
    """
    local = _local_sidecar_dir() / name
    if local.is_file():
        return local
    try:
        return Path(hf_hub_download(repo_id=REPO_ID, filename=name, repo_type=REPO_TYPE))
    except EntryNotFoundError:
        return None
    except Exception as exc:
        # Treat any other download error (network, permissions) the same
        # as a miss so the demo stays usable offline or before sidecars
        # have been uploaded. Log it so an unreachable HF / 401 doesn't
        # look identical to "no sidecar exists" in production traces.
        print(f"[fixture_sidecar] HF download failed for {name}: {exc}")
        return None


def fixture_sidecar(name: str) -> Optional[FixtureSidecar]:
    """Load the sidecar bundle for ``name`` if available and fresh.

    Returns ``None`` (not an exception) on any of:
      - ``name`` not in :data:`KNOWN_FIXTURES`
      - sidecar JSON or safetensors not present in the dataset
      - format_version mismatch

    Sidecars are NOT gated on the runtime checkpoint: the VAE and the
    semantic tokenizer/detokenizer that produce the cached tensors are
    shared across the ACE-Step v1.5 family. The JSON's ``produced_with``
    (legacy: ``checkpoint``) field is informational only.

    On every miss past ``name in KNOWN_FIXTURES`` we print the reason
    so the demo's logs make it obvious why the sidecar fast path didn't
    fire (silent ``None`` returns previously left operators staring at
    "Detecting BPM + key..." without any clue whether the sidecar files
    were missing or stale).
    """
    if name not in KNOWN_FIXTURES:
        return None

    json_path = _resolve_sidecar_file(f"{name}.sidecar.json")
    if json_path is None:
        print(f"[fixture_sidecar] {name}: sidecar JSON not found (local dir + HF dataset)")
        return None
    sf_path = _resolve_sidecar_file(f"{name}.sidecar.safetensors")
    if sf_path is None:
        print(f"[fixture_sidecar] {name}: sidecar safetensors not found (local dir + HF dataset)")
        return None

    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[fixture_sidecar] {name}: sidecar JSON unreadable ({exc})")
        return None
    fv = int(meta.get("format_version", 0))
    if fv != SIDECAR_FORMAT_VERSION:
        print(
            f"[fixture_sidecar] {name}: format_version mismatch "
            f"(sidecar={fv} vs loader={SIDECAR_FORMAT_VERSION}) — "
            f"re-run scripts/calibration/precompute_fixture_sidecars.py --force"
        )
        return None

    # Lazy import so the basic fixture path doesn't pull torch on import.
    from safetensors import safe_open

    try:
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            latent = f.get_tensor("latent")
            context_latent = f.get_tensor("context_latent")
    except Exception as exc:
        print(f"[fixture_sidecar] {name}: safetensors load failed ({exc})")
        return None

    # ``time_signature`` was added after the original sidecar format;
    # default to the model's standard ``"4"`` when older JSONs don't
    # carry it so existing dataset entries keep loading without a
    # format_version bump or a forced re-precompute.
    return FixtureSidecar(
        name=name,
        bpm=int(meta["bpm"]),
        key=str(meta["key"]),
        time_signature=str(meta.get("time_signature", "4")),
        duration_s=float(meta["duration_s"]),
        samples=int(meta["samples"]),
        sample_rate=int(meta["sample_rate"]),
        channels=int(meta["channels"]),
        produced_with=str(meta.get("produced_with") or meta.get("checkpoint") or ""),
        latent=latent,
        context_latent=context_latent,
    )

"""Shared sidecar format + I/O for precomputed audio clips.

A sidecar is a ``<name>.sidecar.json`` + ``<name>.sidecar.safetensors``
pair sitting next to an audio file. The JSON holds metadata (bpm, key,
time_signature, post-truncation duration / sample counts, checkpoint,
format_version); the safetensors holds the prompt-independent encoded
tensors (raw VAE source latent + semantic context_latent from
``Session.prepare_source``). The realtime demo loads the sidecar at
session-init to skip the per-connect ``prepare_source`` (saves 1-3s
warm, more cold).

Two libraries use this format with identical semantics:

  ``fixtures_dir()``       Test fixtures from the
                           ``daydreamlive/demon-fixtures-v2`` HF dataset.
                           Sidecars produced by
                           ``scripts/calibration/precompute_fixture_sidecars.py``.
  ``user_uploads_dir()``   User-uploaded audio. Sidecars produced by
                           the rtmg backend's upload handler.

The encoding step is the expensive part (a VAE forward pass over the
full source) and is identical for both libraries — the only thing that
differs is *where* the file lands on disk and *how* the operator-facing
metadata (bpm, key, time_signature) is resolved. Both pipelines share
:func:`encode_and_save_sidecar` for the encode + write half and
:func:`truncate_to_pool` for the pre-encode alignment.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Sidecar schema version. Bump when the on-disk format changes in a way
# that prior sidecars can't satisfy. :func:`load_sidecar` refuses
# mismatches so a stale on-disk file silently falls through to live
# computation instead of crashing the load.
SIDECAR_FORMAT_VERSION = 2

# Latent-frame alignment unit. The VAE produces one latent frame per
# 1920 input samples and the semantic extractor operates on groups of 5
# frames, so the source waveform has to be a multiple of 1920*5 = 9600
# samples for prepare_source to round-trip cleanly. The realtime demo
# enforces this on both the streaming upload path (handle_client) and
# the offline precompute path; this constant is the single source of
# truth.
POOL = 1920 * 5


@dataclass
class AudioSidecar:
    """Loaded sidecar bundle for an audio clip (fixture or user upload).

    Caches the deterministic, prompt-independent preprocessing the
    realtime demo would otherwise do on every connect: BPM (librosa),
    key (CNN classifier or filename suffix), and the source latent +
    semantic context_latent from ``Session.prepare_source``.
    Conditioning (encode_text) is *not* cached; the demo's
    blended-prompt UI means the client typically diverges from any
    baked tags within seconds of connecting, and encode_text is cheap
    enough (~60ms warm) that the cache savings don't justify the
    server-authoritative complication.
    """

    name: str
    bpm: int
    key: str
    # Stringified meter numerator (matches the encoder boundary in
    # ``Session.encode_text``, which prepends ``- timesignature: <s>``
    # to the prompt). One of ``VALID_TIME_SIGNATURES`` (``"2"``, ``"3"``,
    # ``"4"``, ``"6"``); defaults to ``"4"`` when older sidecars don't
    # carry the field.
    time_signature: str
    duration_s: float
    samples: int
    sample_rate: int
    channels: int
    checkpoint: str
    latent: torch.Tensor
    context_latent: torch.Tensor


def truncate_to_pool(waveform: torch.Tensor) -> torch.Tensor:
    """Stereo cap + mod-:data:`POOL`-sample drop.

    Does *not* apply any duration-based cap — callers that need to
    pre-fit a TRT profile (the runtime upload path) clamp the duration
    themselves before passing the waveform in. The precompute path is
    intentionally profile-agnostic and skips that clamp; sidecar
    staleness against a smaller profile is caught by the length check
    on the load side instead.
    """
    waveform = waveform[:2]
    rem = waveform.shape[-1] % POOL
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    return waveform


def _sidecar_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"{name}.sidecar.json", out_dir / f"{name}.sidecar.safetensors"


def load_sidecar(
    json_path: Path, sf_path: Path, *, name: str,
) -> Optional[AudioSidecar]:
    """Parse a JSON + safetensors pair into an :class:`AudioSidecar`.

    Returns ``None`` (not an exception) on any of:
      - JSON or safetensors unreadable / missing keys
      - ``format_version`` mismatch with :data:`SIDECAR_FORMAT_VERSION`

    The cached tensors are not gated on the runtime checkpoint: the VAE
    and the semantic tokenizer/detokenizer that produce them are shared
    across the ACE-Step v1.5 family. The JSON's ``checkpoint`` field is
    informational only.

    Caller falls back to live computation; same staleness semantics for
    fixtures and user uploads (only the file locations differ).
    """
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(meta.get("format_version", 0)) != SIDECAR_FORMAT_VERSION:
        return None

    # Lazy import so the read-only callers don't pull safetensors at
    # module import time. (The realtime backend imports this module
    # before it has loaded torch in some paths.)
    from safetensors import safe_open

    try:
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            latent = f.get_tensor("latent")
            context_latent = f.get_tensor("context_latent")
    except Exception:
        return None

    # ``time_signature`` was added after the original sidecar format;
    # default to the model's standard ``"4"`` when older JSONs don't
    # carry it so existing dataset entries keep loading without a
    # format_version bump or a forced re-precompute.
    try:
        return AudioSidecar(
            name=name,
            bpm=int(meta["bpm"]),
            key=str(meta["key"]),
            time_signature=str(meta.get("time_signature", "4")),
            duration_s=float(meta["duration_s"]),
            samples=int(meta["samples"]),
            sample_rate=int(meta["sample_rate"]),
            channels=int(meta["channels"]),
            checkpoint=str(meta.get("checkpoint", "")),
            latent=latent,
            context_latent=context_latent,
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_sidecar_pair(
    json_path: Path,
    sf_path: Path,
    *,
    latent: torch.Tensor,
    context_latent: torch.Tensor,
    checkpoint: str,
    bpm: int,
    key: str,
    time_signature: str,
    duration_s: float,
    samples: int,
    sample_rate: int,
    channels: int,
) -> None:
    """Write a sidecar JSON + safetensors pair for an audio clip.

    Atomic: each file is written to a sibling ``.tmp`` and then
    ``os.replace``-d into place. A partial write therefore reads as a
    miss in :func:`load_sidecar` (since the final filename never
    appears until the bytes are fully on disk), and there's no window
    where a half-written safetensors masks a complete JSON or vice
    versa.

    Tensors are detached to CPU before writing.
    """
    # Lazy import so the read-only path (load_sidecar) doesn't pull
    # safetensors.torch (and through it, the torch tensor save codepath)
    # at module import time.
    from safetensors.torch import save_file as safetensors_save

    json_path.parent.mkdir(parents=True, exist_ok=True)
    sf_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "format_version": SIDECAR_FORMAT_VERSION,
        "checkpoint": checkpoint,
        "bpm": int(bpm),
        "key": str(key),
        "time_signature": str(time_signature),
        "duration_s": float(duration_s),
        "samples": int(samples),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
    }
    tensors = {
        "latent": latent.detach().to("cpu").contiguous(),
        "context_latent": context_latent.detach().to("cpu").contiguous(),
    }
    # safetensors metadata is str->str only.
    sf_meta = {k: str(v) for k, v in meta.items()}

    sf_tmp = sf_path.with_suffix(sf_path.suffix + ".tmp")
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    safetensors_save(tensors, str(sf_tmp), metadata=sf_meta)
    json_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(sf_tmp, sf_path)
    os.replace(json_tmp, json_path)


def save_sidecar(
    out_dir: Path,
    name: str,
    *,
    latent: torch.Tensor,
    context_latent: torch.Tensor,
    checkpoint: str,
    bpm: int,
    key: str,
    time_signature: str,
    duration_s: float,
    samples: int,
    sample_rate: int,
    channels: int,
) -> None:
    """Write the legacy ``<name>.sidecar.*`` pair for an audio clip."""
    json_path, sf_path = _sidecar_paths(out_dir, name)
    save_sidecar_pair(
        json_path,
        sf_path,
        latent=latent,
        context_latent=context_latent,
        checkpoint=checkpoint,
        bpm=bpm,
        key=key,
        time_signature=time_signature,
        duration_s=duration_s,
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
    )


def encode_and_save_sidecar(
    session,
    *,
    out_dir: Path,
    name: str,
    json_path: Path | None = None,
    sf_path: Path | None = None,
    waveform: torch.Tensor,
    sample_rate: int,
    checkpoint: str,
    bpm: int,
    key: str,
    time_signature: str,
) -> AudioSidecar:
    """Run ``Session.prepare_source`` on ``waveform`` and write the sidecar.

    Caller is responsible for any pre-encode work (decode, profile
    duration cap, key/bpm resolution policy). This function is the
    shared half of the pipeline: VAE-encode the waveform, persist the
    tensors + metadata, and return the in-memory bundle so the caller
    can reuse it without a round-trip through :func:`load_sidecar`.

    ``waveform`` must already be ``[<=2, N]`` and pool-aligned (use
    :func:`truncate_to_pool` first); the function does *not* re-clamp.
    ``session`` is duck-typed against
    :meth:`acestep.engine.session.Session.prepare_source`; both eager
    and TRT sessions work.
    """
    # Lazy imports: callers of the read-only sidecar path don't need to
    # pay for these at import time. ``Audio`` pulls torch transitively
    # and ``time`` is std-lib but kept colocated for clarity.
    from acestep.nodes.types import Audio

    samples = int(waveform.shape[1])
    channels = int(waveform.shape[0])
    duration_s = samples / sample_rate

    audio_in = Audio(waveform=waveform, sample_rate=sample_rate)
    t0 = time.time()
    source = session.prepare_source(audio_in)
    elapsed = time.time() - t0

    if json_path is None or sf_path is None:
        json_path, sf_path = _sidecar_paths(out_dir, name)
    save_sidecar_pair(
        json_path,
        sf_path,
        latent=source.latent.tensor,
        context_latent=source.context_latent.tensor,
        checkpoint=checkpoint,
        bpm=bpm,
        key=key,
        time_signature=time_signature,
        duration_s=duration_s,
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
    )

    print(
        f"[sidecars] {name}: prepare_source {elapsed:.2f}s, "
        f"wrote {duration_s:.1f}s / {samples} samples / {channels}ch"
    )

    return AudioSidecar(
        name=name,
        bpm=int(bpm),
        key=str(key),
        time_signature=str(time_signature),
        duration_s=duration_s,
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        checkpoint=checkpoint,
        latent=source.latent.tensor,
        context_latent=source.context_latent.tensor,
    )

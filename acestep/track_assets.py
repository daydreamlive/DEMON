"""Shared clean v2 layout for fixture and user-upload track assets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

SOURCE_MODES = ("full", "vocals", "instruments")
STEM_MODES = ("vocals", "instruments")
TRACK_METADATA_VERSION = 2
SOURCE_WAV = "source.wav"
TRACK_JSON = "track.json"


def normalize_source_mode(mode: str | None) -> str:
    return mode if mode in SOURCE_MODES else "full"


def track_dir_name(name: str) -> str:
    cleaned = Path(str(name or "track")).name.strip()
    stem = Path(cleaned).stem if Path(cleaned).suffix else cleaned
    safe = "".join(ch if ch.isalnum() or ch in "._- ()" else "_" for ch in stem)
    return safe.strip(" .") or "track"


def track_dir(root: Path, name: str) -> Path:
    return root / track_dir_name(name)


def track_metadata_path(root: Path, name: str) -> Path:
    return track_dir(root, name) / TRACK_JSON


def source_audio_path(root: Path, name: str) -> Path:
    return track_dir(root, name) / SOURCE_WAV


def stems_dir(root: Path, name: str) -> Path:
    return track_dir(root, name) / "stems"


def sidecars_dir(root: Path, name: str) -> Path:
    return track_dir(root, name) / "sidecars"


def stem_audio_path(root: Path, name: str, mode: str) -> Path:
    if mode not in STEM_MODES:
        raise ValueError(f"unsupported stem mode: {mode!r}")
    return stems_dir(root, name) / f"{mode}.wav"


def sidecar_paths(root: Path, name: str, source_mode: str | None = "full") -> tuple[Path, Path]:
    mode = normalize_source_mode(source_mode)
    d = sidecars_dir(root, name)
    return d / f"{mode}.json", d / f"{mode}.safetensors"


def source_sidecar_name(name: str, source_mode: str | None = "full") -> str:
    mode = normalize_source_mode(source_mode)
    return name if mode == "full" else f"{name}.{mode}"


def track_metadata_name(name: str) -> str:
    return str(Path(track_dir_name(name)) / TRACK_JSON).replace("\\", "/")


def source_audio_name(name: str) -> str:
    return str(Path(track_dir_name(name)) / SOURCE_WAV).replace("\\", "/")


def stem_audio_name(name: str, mode: str) -> str:
    if mode not in STEM_MODES:
        raise ValueError(f"unsupported stem mode: {mode!r}")
    return str(Path(track_dir_name(name)) / "stems" / f"{mode}.wav").replace("\\", "/")


def sidecar_asset_name(name: str, source_mode: str | None = "full", suffix: str = "json") -> str:
    mode = normalize_source_mode(source_mode)
    return str(Path(track_dir_name(name)) / "sidecars" / f"{mode}.{suffix}").replace("\\", "/")


def waveform_fingerprint(waveform: "torch.Tensor") -> str:
    """Content fingerprint used to decide whether cached stems still match.

    Deliberately NOT a hash of the raw float32 bytes. The waveform hashed
    at persist time and the one presented at lookup time travel different
    decode/truncate routes (server-side fixture load vs client upload
    re-decode, pool-alignment, profile-cap truncation), so an exact-byte
    SHA would miss on essentially every real lookup and silently fall
    through to live extraction. Instead we:

      - mix to mono (channel layout is not identity),
      - decimate to a fixed grid so a few samples of length drift map to
        the same fingerprint while genuinely different audio does not,
      - quantize so sub-LSB float noise from a decode round-trip is swallowed.

    The result is stable across benign transforms but still distinguishes
    different tracks. Changing this function invalidates previously written
    fingerprints, which degrades gracefully to a one-time cache miss.
    """
    import torch

    arr = waveform.detach().cpu().float()
    mono = arr.mean(dim=0) if arr.dim() == 2 else arr.reshape(-1)
    n = int(mono.shape[-1])
    if n == 0:
        return "empty"
    grid = 4096
    if n > grid:
        idx = torch.linspace(0, n - 1, grid).round().long()
        mono = mono[idx]
    quantized = (mono.contiguous() * 10_000.0).round().to(torch.int32)
    return hashlib.sha256(quantized.numpy().tobytes()).hexdigest()


def load_json_metadata(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_track_metadata(root: Path, name: str) -> dict:
    p = track_metadata_path(root, name)
    if p.is_file():
        return load_json_metadata(p)
    legacy = root / f"{name}.track.json"
    if legacy.is_file():
        return load_json_metadata(legacy)
    return {}


def save_track_metadata(
    root: Path,
    name: str,
    *,
    waveform: "torch.Tensor",
    sample_rate: int,
    bpm: int | None = None,
    key: str | None = None,
    time_signature: str | None = None,
) -> None:
    p = track_metadata_path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    prior = load_track_metadata(root, name)
    meta = dict(prior)
    # The manifest reflects what is actually on disk, not what the layout
    # could hold. Callers write the asset files before metadata, so a
    # default precompute (full sidecar only, no stems) produces a track.json
    # that doesn't advertise stems/stem-sidecars it never wrote.
    stems_manifest = {
        mode: f"stems/{mode}.wav"
        for mode in STEM_MODES
        if stem_audio_path(root, name, mode).is_file()
    }
    sidecars_manifest = {
        mode: f"sidecars/{mode}.json"
        for mode in SOURCE_MODES
        if sidecar_paths(root, name, mode)[0].is_file()
    }
    meta.update({
        "format_version": TRACK_METADATA_VERSION,
        "display_name": meta.get("display_name") or name,
        "source_file": SOURCE_WAV,
        "source_name": name,
        "sample_rate": int(sample_rate),
        "samples": int(waveform.shape[-1]),
        "channels": int(waveform.shape[0]),
        "waveform_sha256": waveform_fingerprint(waveform),
        "stems": stems_manifest,
        "sidecars": sidecars_manifest,
    })
    if bpm is not None:
        meta["bpm"] = int(bpm)
    if key is not None:
        meta["key"] = str(key)
    if time_signature is not None:
        meta["time_signature"] = str(time_signature)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_stem_wavs(
    root: Path,
    name: str,
    *,
    stems: dict[str, "torch.Tensor"],
    sample_rate: int,
) -> None:
    import soundfile as sf

    for mode in STEM_MODES:
        if mode not in stems:
            raise ValueError(f"missing stem: {mode}")
        p = stem_audio_path(root, name, mode)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        wav = stems[mode].detach().cpu().float()
        sf.write(str(tmp), wav.numpy().T, int(sample_rate), format="WAV", subtype="FLOAT")
        os.replace(tmp, p)


def write_track_wav(
    root: Path,
    name: str,
    *,
    waveform: "torch.Tensor",
    sample_rate: int,
) -> None:
    import soundfile as sf

    p = source_audio_path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    wav = waveform.detach().cpu().float()
    sf.write(str(tmp), wav.numpy().T, int(sample_rate), format="WAV", subtype="FLOAT")
    os.replace(tmp, p)


def read_stem_wavs(
    root: Path,
    name: str,
    *,
    waveform: "torch.Tensor",
    sample_rate: int,
    metadata: Optional[dict] = None,
) -> Optional[dict[str, "torch.Tensor"]]:
    import soundfile as sf
    import torch

    meta = metadata if metadata is not None else load_track_metadata(root, name)
    expected_hash = meta.get("waveform_sha256")
    if not expected_hash or str(expected_hash) != waveform_fingerprint(waveform):
        return None
    stems: dict[str, torch.Tensor] = {}
    for mode in STEM_MODES:
        p = stem_audio_path(root, name, mode)
        if not p.is_file():
            legacy = root / f"{name}.{mode}.wav"
            p = legacy if legacy.is_file() else p
        if not p.is_file():
            return None
        try:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        except Exception:
            return None
        if int(sr) != int(sample_rate):
            return None
        t = torch.from_numpy(data.T.copy()).float()
        if t.shape[0] > waveform.shape[0]:
            t = t[:waveform.shape[0]]
        elif t.shape[0] < waveform.shape[0]:
            t = torch.cat([t, t[-1:].repeat(waveform.shape[0] - t.shape[0], 1)])
        if t.shape[-1] > waveform.shape[-1]:
            t = t[:, :waveform.shape[-1]]
        elif t.shape[-1] < waveform.shape[-1]:
            t = torch.nn.functional.pad(t, (0, waveform.shape[-1] - t.shape[-1]))
        stems[mode] = t.contiguous()
    return stems

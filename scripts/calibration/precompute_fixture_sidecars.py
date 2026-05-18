"""Precompute fixture sidecars for the realtime motion-graph demo.

For each fixture in :data:`acestep.fixtures.KNOWN_FIXTURES` this writes
a ``<name>.sidecar.json`` and ``<name>.sidecar.safetensors`` pair into
``--out`` (default ``out/fixture_sidecars``):

  JSON   bpm, key, time_signature, post-truncation duration / sample
         counts, sample rate, channels, checkpoint, format_version.

  Safetensors
         Tensors: latent, context_latent. Conditioning is *not* cached
         (see fixtures.py for rationale).

The script is idempotent: existing bpm / key / time_signature values
are preserved (so an operator override survives a re-run). Pass
``--force`` to overwrite from scratch.

Pipeline per fixture:
  1. Download the WAV (cache hit if already present).
  2. Apply the same audio-level truncation backend.py applies before
     prepare_source: stereo cap + drop the trailing samples below a
     1920*5-sample boundary. The TRT max-profile cap is intentionally
     NOT applied here so the precompute is profile-agnostic; the
     runtime only uses the cache when the live truncated length
     matches the recorded ``samples`` field.
  3. Resolve bpm / key / time_signature. Existing JSON wins;
     otherwise compute bpm via librosa, parse key from the filename
     suffix, and default time_signature to "4" (no automated detector
     today; operators can edit the JSON to override).
  4. ``Session.prepare_source`` -> raw VAE latent + semantic
     context_latent.
  5. Write the JSON and safetensors.

Run on a machine with the model checkpoint and a working CUDA build.
Eager backends are forced so this works without prebuilt TRT engines:

    uv run python -m scripts.precompute_fixture_sidecars
    uv run python -m scripts.precompute_fixture_sidecars --force
    uv run python -m scripts.precompute_fixture_sidecars --only \\
        inside_confusion_loop_60s_gsm.wav

Sidecars are uploaded to the daydreamlive/demon-fixtures HF dataset
in a separate step so the runtime can fetch them via hf_hub_download
alongside the WAVs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file as safetensors_save

from acestep.engine.session import Session
from acestep.constants import VALID_TIME_SIGNATURES
from acestep.fixtures import (
    KNOWN_FIXTURES,
    SIDECAR_FORMAT_VERSION,
    audio_fixture,
    parse_key_from_filename,
)
from acestep.nodes.types import Audio
from acestep.paths import checkpoints_dir

SAMPLE_RATE = 48000  # matches demos.realtime_motion_graph_web.protocol.SAMPLE_RATE
POOL = 1920 * 5  # 9600 samples = 5 latent frames at 25 fps; matches backend.py


def truncate_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Stereo cap + mod-9600-sample drop, mirroring backend.py.

    Does not apply the runtime's TRT-profile-based duration cap: the
    precompute is profile-agnostic and relies on a length check at
    load time to invalidate when truncation differs.
    """
    waveform = waveform[:2]
    rem = waveform.shape[-1] % POOL
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    return waveform


def _load_existing(json_path: Path) -> dict:
    if not json_path.is_file():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARNING: existing sidecar JSON unreadable, ignoring ({e})")
        return {}


def precompute_one(
    session: Session,
    name: str,
    *,
    out_dir: Path,
    checkpoint: str,
    force: bool,
) -> None:
    fixture_path = audio_fixture(name)

    audio_data, sr = sf.read(str(fixture_path), always_2d=True)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"{name}: unexpected sample rate {sr} (expected {SAMPLE_RATE})")
    waveform = torch.from_numpy(audio_data.T.copy()).float()
    waveform = truncate_audio(waveform)
    samples = int(waveform.shape[1])
    duration_s = samples / SAMPLE_RATE
    channels = int(waveform.shape[0])

    json_path = out_dir / f"{name}.sidecar.json"
    sf_path = out_dir / f"{name}.sidecar.safetensors"
    existing = {} if force else _load_existing(json_path)

    # bpm: prefer the existing JSON value (operator override) over a
    # fresh librosa run. librosa.beat_track is non-deterministic enough
    # that re-running shouldn't quietly clobber a value the operator
    # chose.
    if isinstance(existing.get("bpm"), (int, float)):
        bpm = int(existing["bpm"])
        bpm_source = "existing JSON"
    else:
        mono = waveform.mean(dim=0).numpy()
        bpm_raw, _ = librosa.beat.beat_track(y=mono, sr=SAMPLE_RATE)
        bpm = int(round(float(np.asarray(bpm_raw).flat[0])))
        bpm_source = "librosa"

    # key: prefer existing; else parse the filename suffix.
    if isinstance(existing.get("key"), str) and existing["key"]:
        key = existing["key"]
        key_source = "existing JSON"
    else:
        parsed = parse_key_from_filename(name)
        if parsed is None:
            raise RuntimeError(
                f"{name}: could not parse key from filename and no existing "
                f"JSON value to fall back to"
            )
        key = parsed
        key_source = "filename"

    # time_signature: prefer existing; else default to "4" (no detector
    # today; the model itself accepts "2"/"3"/"4"/"6", and most fixtures
    # are 4/4). Operator can edit the JSON to override before re-running
    # without --force.
    valid_ts = {str(s) for s in VALID_TIME_SIGNATURES}
    existing_ts = existing.get("time_signature")
    if isinstance(existing_ts, str) and existing_ts in valid_ts:
        time_signature = existing_ts
        ts_source = "existing JSON"
    elif isinstance(existing_ts, (int, float)) and str(int(existing_ts)) in valid_ts:
        time_signature = str(int(existing_ts))
        ts_source = "existing JSON"
    else:
        time_signature = "4"
        ts_source = "default"

    print(
        f"  bpm={bpm} ({bpm_source})  key={key!r} ({key_source})  "
        f"time_signature={time_signature!r} ({ts_source})  "
        f"dur={duration_s:.2f}s  samples={samples}"
    )

    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)
    t0 = time.time()
    source = session.prepare_source(audio_in)
    print(f"  prepare_source: {time.time() - t0:.2f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": SIDECAR_FORMAT_VERSION,
        "checkpoint": checkpoint,
        "bpm": bpm,
        "key": key,
        "time_signature": time_signature,
        "duration_s": duration_s,
        "samples": samples,
        "sample_rate": SAMPLE_RATE,
        "channels": channels,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    tensors = {
        "latent": source.latent.tensor.detach().to("cpu").contiguous(),
        "context_latent": source.context_latent.tensor.detach().to("cpu").contiguous(),
    }
    sf_meta = {k: str(v) for k, v in meta.items()}
    safetensors_save(tensors, str(sf_path), metadata=sf_meta)

    sizes = {k: tuple(v.shape) for k, v in tensors.items()}
    dtypes = {k: str(v.dtype) for k, v in tensors.items()}
    print(f"  wrote {json_path.name} + {sf_path.name}")
    print(f"    shapes: {sizes}  dtypes: {dtypes}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path,
        default=Path("out") / "fixture_sidecars",
        help="Output directory (default: out/fixture_sidecars)",
    )
    parser.add_argument(
        "--checkpoint", default="acestep-v15-turbo",
        help="DiT checkpoint name (used for staleness checks)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing JSON sidecars instead of preserving "
             "bpm/key/time_signature/tags",
    )
    parser.add_argument(
        "--only", action="append", default=[], metavar="NAME",
        help="Only process this fixture (repeatable). Default: all KNOWN_FIXTURES.",
    )
    args = parser.parse_args(argv)

    targets = sorted(args.only) if args.only else sorted(KNOWN_FIXTURES)
    unknown = [n for n in targets if n not in KNOWN_FIXTURES]
    if unknown:
        print(f"ERROR: unknown fixture(s): {unknown}", file=sys.stderr)
        return 2

    print(f"Loading session ({args.checkpoint}, eager backends)...")
    t0 = time.time()
    session = Session(
        project_root=str(checkpoints_dir()),
        config_path=args.checkpoint,
        decoder_backend="eager",
        vae_backend="eager",
    )
    print(f"  session loaded in {time.time() - t0:.1f}s")

    failures: list[tuple[str, str]] = []
    for name in targets:
        print(f"\n[{name}]")
        try:
            precompute_one(
                session, name,
                out_dir=args.out, checkpoint=args.checkpoint,
                force=args.force,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failures.append((name, str(e)))

    print(f"\nDone. {len(targets) - len(failures)}/{len(targets)} succeeded.")
    print(f"Sidecars in: {args.out.resolve()}")
    if failures:
        print(f"Failures:")
        for n, msg in failures:
            print(f"  {n}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

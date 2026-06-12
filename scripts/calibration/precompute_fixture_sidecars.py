"""Precompute fixture sidecars for the realtime motion-graph demo.

For each fixture in :data:`acestep.fixtures.KNOWN_FIXTURES` this writes
the clean v2 track directory under ``--out`` (default
:func:`acestep.paths.fixtures_dir`, i.e. ``MODELS_DIR/fixtures/``):

  track.json
         Editable bpm, key, time_signature, source/stem/sidecar manifest,
         post-truncation sample counts, sample rate, channels.

  sidecars/*.safetensors
         Tensors: latent, context_latent. Conditioning is *not* cached
         (see :mod:`acestep.sidecars` for rationale).

The script is idempotent: existing bpm / key / time_signature values
are preserved (so an operator override survives a re-run). Pass
``--force`` to overwrite from scratch.

Pipeline per fixture:
  1. Download the WAV (cache hit if already present).
  2. Apply the same pool-alignment truncation the realtime backend
     applies before prepare_source (see
     :func:`acestep.sidecars.truncate_to_pool`). The TRT max-profile
     cap is intentionally NOT applied here so the precompute is
     profile-agnostic; the runtime only uses the cache when the live
     truncated length matches the recorded ``samples`` field.
  3. Resolve bpm / key / time_signature. Existing JSON wins;
     otherwise compute bpm via librosa, parse key from the filename
     suffix, and default time_signature to "4" (no automated detector
     today; operators can edit the JSON to override).
  4. Hand off to :func:`acestep.sidecars.encode_and_save_sidecar`,
     which runs ``Session.prepare_source`` and writes both files
     atomically into ``sidecars/``. The user-upload path in the rtmg
     backend calls the same sidecar write logic, so both libraries
     produce byte-identical sidecar formats from byte-identical encode
     logic.

Run on a machine with the model checkpoint and a working CUDA build.
Eager backends are forced so this works without prebuilt TRT engines:

    uv run python -m scripts.calibration.precompute_fixture_sidecars
    uv run python -m scripts.calibration.precompute_fixture_sidecars --with-stems
    uv run python -m scripts.calibration.precompute_fixture_sidecars --force
    uv run python -m scripts.calibration.precompute_fixture_sidecars --only \\
        inside_confusion_loop_60s_gsm.wav

Sidecars and track assets are uploaded to the daydreamlive/demon-fixtures-v2
HF dataset in a separate step so the runtime can fetch them via
hf_hub_download alongside the WAVs.
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

from acestep.engine.session import Session
from acestep.constants import VALID_TIME_SIGNATURES
from acestep.fixtures import (
    KNOWN_FIXTURES,
    REPO_ID,
    audio_fixture,
    parse_key_from_filename,
)
from acestep.paths import checkpoints_dir, fixtures_dir
from acestep.sidecars import encode_and_save_sidecar, truncate_to_pool
from acestep.streaming.stems import extract_upload_stems
from acestep.track_assets import (
    save_track_metadata,
    sidecar_paths,
    source_sidecar_name,
    track_metadata_path,
    write_track_wav,
    write_stem_wavs,
)

SAMPLE_RATE = 48000  # matches demos.realtime_motion_graph_web.protocol.SAMPLE_RATE


def _load_existing(json_path: Path) -> dict:
    if not json_path.is_file():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARNING: existing sidecar JSON unreadable, ignoring ({e})")
        return {}


def _resolve_bpm_key_ts(
    name: str, waveform: torch.Tensor, existing: dict,
) -> tuple[int, str, str, tuple[str, str, str]]:
    """Pick bpm / key / time_signature for a fixture, preferring existing JSON.

    Returns ``(bpm, key, time_signature, (bpm_source, key_source, ts_source))``;
    the source tuple is for the log line. Raises :class:`RuntimeError`
    when key has no JSON value *and* the filename suffix doesn't parse
    (the only unrecoverable case — bpm and time_signature both have
    deterministic fallbacks).
    """
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

    return bpm, key, time_signature, (bpm_source, key_source, ts_source)


def precompute_one(
    session: Session,
    name: str,
    *,
    out_dir: Path,
    checkpoint: str,
    force: bool,
    with_stems: bool,
) -> None:
    fixture_path = audio_fixture(name)

    audio_data, sr = sf.read(str(fixture_path), always_2d=True)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"{name}: unexpected sample rate {sr} (expected {SAMPLE_RATE})")
    waveform = truncate_to_pool(torch.from_numpy(audio_data.T.copy()).float())

    json_path = track_metadata_path(out_dir, name)
    if not json_path.is_file():
        json_path = out_dir / f"{name}.sidecar.json"
    existing = {} if force else _load_existing(json_path)
    bpm, key, time_signature, sources = _resolve_bpm_key_ts(name, waveform, existing)
    bpm_source, key_source, ts_source = sources

    print(
        f"  bpm={bpm} ({bpm_source})  key={key!r} ({key_source})  "
        f"time_signature={time_signature!r} ({ts_source})  "
        f"dur={waveform.shape[1] / SAMPLE_RATE:.2f}s  samples={waveform.shape[1]}"
    )

    encode_and_save_sidecar(
        session,
        out_dir=out_dir,
        name=name,
        json_path=sidecar_paths(out_dir, name, "full")[0],
        sf_path=sidecar_paths(out_dir, name, "full")[1],
        waveform=waveform,
        sample_rate=SAMPLE_RATE,
        checkpoint=checkpoint,
        bpm=bpm,
        key=key,
        time_signature=time_signature,
    )
    write_track_wav(out_dir, name, waveform=waveform, sample_rate=SAMPLE_RATE)

    if with_stems:
        stems = extract_upload_stems(
            waveform=waveform,
            device=session.handler.device,
            backend_sample_rate=SAMPLE_RATE,
            model_context=session.handler,
        )
        write_stem_wavs(out_dir, name, stems=stems, sample_rate=SAMPLE_RATE)
        for mode in ("vocals", "instruments"):
            print(f"  precomputing {mode} source sidecar")
            encode_and_save_sidecar(
                session,
                out_dir=out_dir,
                name=source_sidecar_name(name, mode),
                json_path=sidecar_paths(out_dir, name, mode)[0],
                sf_path=sidecar_paths(out_dir, name, mode)[1],
                waveform=stems[mode],
                sample_rate=SAMPLE_RATE,
                checkpoint=checkpoint,
                bpm=bpm,
                key=key,
                time_signature=time_signature,
            )

    # Metadata last: its stems/sidecars manifest is derived from the files
    # written above, so a default (no --with-stems) run produces a track.json
    # that advertises only the full sidecar it actually wrote.
    save_track_metadata(
        out_dir,
        name,
        waveform=waveform,
        sample_rate=SAMPLE_RATE,
        bpm=bpm,
        key=key,
        time_signature=time_signature,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path,
        default=fixtures_dir(),
        help="Output directory (default: MODELS_DIR/fixtures, alongside the WAVs themselves)",
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
    parser.add_argument(
        "--with-stems", action="store_true",
        help="Also extract vocals/instruments WAVs and precompute sidecars for each stem source.",
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
                force=args.force, with_stems=args.with_stems,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failures.append((name, str(e)))

    print(f"\nDone. {len(targets) - len(failures)}/{len(targets)} succeeded.")
    print(f"Sidecars in: {args.out.resolve()}")
    print(f"HF upload target: {REPO_ID} (repo-type dataset)")
    if failures:
        print(f"Failures:")
        for n, msg in failures:
            print(f"  {n}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

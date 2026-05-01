#!/usr/bin/env python3
"""Swap TRT engines between 10.13 and 10.16 backup states.

For each decoder dir under trt_engines/, copy the matching .trt10.13.bak or
.trt10.16.bak (or alt_optlvl1) over the active .engine file. Same for VAE dirs.

Usage:
    python swap_engines.py 10.13
    python swap_engines.py 10.16
"""
import shutil
import sys
from pathlib import Path

ROOT = Path("C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines")
EXP_ENGINES = Path("C:/_dev/projects/ACE-Step-1.5_alt/_remote_scripts/_exp_engines")

# Decoder dirs and their backup file naming.
DECODER_DIRS = [
    "decoder_base_mixed_refit_b8_240s",
    "decoder_mixed_refit_b8_240s",
    "decoder_xl-turbo_bf16mix_dynbatch_b8_240s",
]

# VAE dirs.
VAE_DIRS = [
    "vae_decode_fp16_240s",
    "vae_encode_fp16_240s",
]

# Special cases: engines whose 10.13 backup lives in a different dir
SPECIAL_10_13 = {}

# For 10.16 VAE we use the optlvl1 builds we made in this session.
VAE_10_16_OPTLVL1 = {
    "vae_decode_fp16_240s": EXP_ENGINES / "vae_decode_fp16_optlvl1_240s.engine",
    "vae_encode_fp16_240s": EXP_ENGINES / "vae_encode_fp16_optlvl1_240s.engine",
}


def restore(target, source, label):
    if not source.exists():
        print(f"  MISSING source for {label}: {source}", flush=True)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    size_mb = target.stat().st_size / (1 << 20)
    print(f"  restored {label}: {target.name} ({size_mb:.1f} MB)", flush=True)
    return True


def main():
    target_version = sys.argv[1]
    assert target_version in ("10.13", "10.16"), f"unknown version {target_version}"

    print(f"Swapping all engines to TRT {target_version} state", flush=True)
    ok = True

    print("\n[decoders]", flush=True)
    for d in DECODER_DIRS:
        active = ROOT / d / f"{d}.engine"
        if target_version == "10.13":
            if d in SPECIAL_10_13:
                src = SPECIAL_10_13[d]
            else:
                src = ROOT / d / f"{d}.engine.trt10.13.bak"
        else:
            src = ROOT / d / f"{d}.engine.trt10.16.bak"
        if not restore(active, src, d):
            ok = False

    print("\n[VAE]", flush=True)
    for v in VAE_DIRS:
        active = ROOT / v / f"{v}.engine"
        if target_version == "10.13":
            src = ROOT / v / f"{v}.engine.trt10.13.bak"
        else:
            # Use the optlvl1 build we proved works on 10.16
            src = VAE_10_16_OPTLVL1[v]
        if not restore(active, src, v):
            ok = False

    print(f"\n{'OK' if ok else 'FAILED'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

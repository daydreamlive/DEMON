#!/usr/bin/env python3
"""Build a hybrid INT8 VAE decode engine.

Combines:
  - modelopt's high-quality per-tensor activation scales (from G2's QDQ ONNX)
  - TRT-native's aggressive int8/fp16 fusion (from the standard build pipeline)

Pipeline:
  1. Walk modelopt QDQ ONNX, extract per-tensor activation scales from each
     QuantizeLinear node's y_scale initializer. Per-channel (weight) scales
     are skipped: TRT folds those from the embedded Q/DQ pattern in the
     ONNX, not from the calibration cache.
  2. Parse the TRT-native cache (variant A's .calib) — it is the
     authoritative list of TRT internal tensor names + native scales.
  3. Empirical format check: for tensors with exact name match between
     modelopt and TRT, compare values to determine whether TRT stores raw
     scale factors (~1.0 ratio) or dynamic ranges (~127.0 ratio).
  4. Build a hybrid cache: modelopt scale where matched, TRT-native scale
     where not. Match strategy: exact > suffix > substring.
  5. Build a TRT engine using the hybrid cache. The existing
     build_vae_decode_int8_engine reads the cache via
     read_calibration_cache() and skips iterative calibration entirely.

Usage:
    # Dry-run: print analysis (overlap, format, coverage) and exit.
    uv run python scripts/build_vae_int8_hybrid.py --analyze-only

    # Write hybrid .calib but don't build the engine yet.
    uv run python scripts/build_vae_int8_hybrid.py --no-build

    # Full build.
    uv run python scripts/build_vae_int8_hybrid.py
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import onnx
from onnx import numpy_helper

from acestep.engine.trt.vae_decode_int8 import (
    VAEDecodeInt8Config,
    build_vae_decode_int8_engine,
)
from acestep.paths import models_dir, trt_engines_dir


# ------------------------------------------------------------------
# Cache file IO
# ------------------------------------------------------------------

def _hex_to_float(hx: str) -> float:
    """TRT calibration cache stores values as 4-byte big-endian IEEE 754
    floats encoded as 8 hex chars."""
    return struct.unpack(">f", bytes.fromhex(hx))[0]


def _float_to_hex(v: float) -> str:
    return struct.pack(">f", float(v)).hex()


def parse_trt_cache(path: Path) -> tuple[str, dict[str, str]]:
    """Return (header_line, {tensor_name: hex_value})."""
    lines = path.read_text().splitlines()
    header = lines[0]
    entries: dict[str, str] = {}
    for line in lines[1:]:
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, hx = line.partition(":")
        entries[name.strip()] = hx.strip()
    return header, entries


def write_trt_cache(path: Path, header: str, entries: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{name}: {hx}" for name, hx in entries.items())
    path.write_text(header + "\n" + body + "\n")


# ------------------------------------------------------------------
# Modelopt QDQ ONNX walking
# ------------------------------------------------------------------

def extract_modelopt_scales(qdq_onnx_path: Path) -> dict[str, float]:
    """Return {tensor_name: y_scale} for every per-tensor QuantizeLinear node.

    A QuantizeLinear node has inputs [x, y_scale, y_zero_point]. We take
    inputs[0] as the tensor being quantized and read y_scale's initializer.
    Per-channel weight quant has y_scale.size > 1 → skipped.
    """
    model = onnx.load(str(qdq_onnx_path))
    init_by_name: dict[str, np.ndarray] = {
        ini.name: numpy_helper.to_array(ini) for ini in model.graph.initializer
    }
    out: dict[str, float] = {}
    skipped_per_channel = 0
    skipped_no_init = 0
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        if len(node.input) < 2:
            continue
        x_name, scale_name = node.input[0], node.input[1]
        scale_arr = init_by_name.get(scale_name)
        if scale_arr is None:
            skipped_no_init += 1
            continue
        if scale_arr.size != 1:
            skipped_per_channel += 1
            continue
        out[x_name] = float(scale_arr.reshape(-1)[0])
    print(
        f"modelopt scales: {len(out)} per-tensor activations, "
        f"{skipped_per_channel} per-channel weight scales skipped, "
        f"{skipped_no_init} non-initializer scales skipped"
    )
    return out


# ------------------------------------------------------------------
# Empirical format check + matching
# ------------------------------------------------------------------

def determine_conversion_factor(
    modelopt_scales: dict[str, float],
    trt_entries: dict[str, str],
    *,
    n_show: int = 12,
) -> float:
    """Determine the multiplier to apply to modelopt y_scale.

    For every name in BOTH dicts, compute ratio = trt_decoded / modelopt_y_scale.
    Two expected clusters:
      ~1.0   → both store raw scale factors → factor = 1.0
      ~127.0 → TRT stores dynamic range = scale * 127 → factor = 127.0
    """
    overlap = [n for n in modelopt_scales if n in trt_entries]
    if not overlap:
        raise RuntimeError(
            "No exact name overlap between modelopt scales and TRT cache; "
            "cannot determine scale-format conversion factor empirically."
        )
    print(f"\n--- Empirical format check ---")
    print(f"  exact-match overlap: {len(overlap)} tensors")
    sample = overlap[:n_show]
    print(f"  showing first {len(sample)}:")
    print(f"    {'tensor':<60} {'modelopt':>12} {'TRT':>12} {'ratio':>10}")
    for n in sample:
        m = modelopt_scales[n]
        t = _hex_to_float(trt_entries[n])
        r = t / m if m != 0 else float("nan")
        print(f"    {n[:58]:<60} {m:>12.4e} {t:>12.4e} {r:>10.3f}")
    all_ratios = []
    for n in overlap:
        m = modelopt_scales[n]
        t = _hex_to_float(trt_entries[n])
        if m != 0 and np.isfinite(m):
            all_ratios.append(t / m)
    arr = np.array(all_ratios)
    median = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    print(f"\n  ratio distribution over all {len(arr)} overlapping tensors:")
    print(f"    median={median:.3f}  p25={p25:.3f}  p75={p75:.3f}")
    if abs(median - 1.0) < abs(median - 127.0):
        factor = 1.0
        print(f"  TRT cache stores raw scale factors -> factor=1.0")
    else:
        factor = 127.0
        print(f"  TRT cache stores dynamic ranges -> factor=127.0")
    return factor


def match_names(
    modelopt_scales: dict[str, float],
    trt_names: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """For each TRT tensor name, find the best matching modelopt name.

    Returns (mapping, strategy) where mapping[trt_name] = onnx_name and
    strategy[trt_name] is "exact" | "suffix" | "substring".
    """
    onnx_names = list(modelopt_scales.keys())
    onnx_set = set(onnx_names)
    mapping: dict[str, str] = {}
    strat: dict[str, str] = {}

    SUFFIXES_TO_STRIP = ["_output_0", "_output"]

    for trt_n in trt_names:
        if trt_n in onnx_set:
            mapping[trt_n] = trt_n
            strat[trt_n] = "exact"
            continue
        # suffix: names match modulo _output / _output_0 tail differences.
        suf_match = None
        for o in onnx_names:
            if o.endswith(trt_n) or trt_n.endswith(o):
                suf_match = o
                break
        if suf_match is not None:
            mapping[trt_n] = suf_match
            strat[trt_n] = "suffix"
            continue
        # substring: strip a known TRT-output suffix and search for a
        # unique containing onnx name.
        stripped = trt_n
        for sfx in SUFFIXES_TO_STRIP:
            if stripped.endswith(sfx):
                stripped = stripped[: -len(sfx)]
                break
        candidates = [o for o in onnx_names if stripped and stripped in o]
        if len(candidates) == 1:
            mapping[trt_n] = candidates[0]
            strat[trt_n] = "substring"
    return mapping, strat


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-name", default="vae_decode_int8_60s_hybrid")
    ap.add_argument("--qdq-onnx", default=None,
                    help="Default: G2's qdq.onnx alongside its engine")
    ap.add_argument("--source-cache", default=None,
                    help="Default: variant A's .calib (vae_decode_int8_60s.calib)")
    ap.add_argument("--onnx", default=None,
                    help="Default: <trt_engines>/_onnx_vae/vae_decode/vae_decode.onnx")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Print analysis and exit without writing files.")
    ap.add_argument("--no-build", action="store_true",
                    help="Write the hybrid cache but skip the TRT engine build.")
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--opt-frames", type=int, default=1500)
    ap.add_argument("--max-frames", type=int, default=1500)
    args = ap.parse_args()

    qdq_onnx = (
        Path(args.qdq_onnx) if args.qdq_onnx
        else (trt_engines_dir() / "vae_decode_int8_60s_modelopt_ent32"
              / "vae_decode_int8_60s_modelopt_ent32.qdq.onnx")
    )
    source_cache = (
        Path(args.source_cache) if args.source_cache
        else (trt_engines_dir() / "vae_decode_int8_60s"
              / "vae_decode_int8_60s.calib")
    )
    onnx_path = (
        Path(args.onnx) if args.onnx
        else trt_engines_dir() / "_onnx_vae" / "vae_decode" / "vae_decode.onnx"
    )

    if not qdq_onnx.exists():
        raise FileNotFoundError(
            f"QDQ ONNX not found: {qdq_onnx}\n"
            "Run scripts/build_vae_int8_modelopt.py with --keep-qdq-onnx first."
        )
    if not source_cache.exists():
        raise FileNotFoundError(f"TRT-native cache not found: {source_cache}")

    print(f"QDQ ONNX:      {qdq_onnx}")
    print(f"Source .calib: {source_cache}")

    print("\n[1/4] Extracting modelopt scales from QDQ ONNX...")
    modelopt_scales = extract_modelopt_scales(qdq_onnx)

    print("\n[2/4] Parsing TRT-native cache...")
    header, trt_entries = parse_trt_cache(source_cache)
    print(f"  header: {header}")
    print(f"  TRT entries: {len(trt_entries)}")

    print("\n[3/4] Determining scale format...")
    factor = determine_conversion_factor(modelopt_scales, trt_entries)

    print("\n[4/4] Matching tensor names and assembling hybrid cache...")
    mapping, strategy = match_names(modelopt_scales, list(trt_entries.keys()))
    n_total = len(trt_entries)
    n_matched = len(mapping)
    by_strat: dict[str, int] = {}
    for s in strategy.values():
        by_strat[s] = by_strat.get(s, 0) + 1
    print(f"  matches: {n_matched}/{n_total} ({n_matched/n_total*100:.1f}%)")
    for s in ["exact", "suffix", "substring"]:
        if s in by_strat:
            print(f"    {s:>10}: {by_strat[s]}")
    print(f"  unmatched (will keep TRT-native scale): {n_total - n_matched}")

    hybrid_entries = dict(trt_entries)
    n_replaced = 0
    n_skip_zero = 0
    for trt_n, onnx_n in mapping.items():
        m_scale = modelopt_scales[onnx_n]
        if m_scale <= 0 or not np.isfinite(m_scale):
            n_skip_zero += 1
            continue
        new_value = m_scale * factor
        hybrid_entries[trt_n] = _float_to_hex(new_value)
        n_replaced += 1
    print(f"  replaced: {n_replaced}  skipped(zero/nan): {n_skip_zero}")

    engine_dir = trt_engines_dir() / args.engine_name
    engine_path = engine_dir / f"{args.engine_name}.engine"
    hybrid_cache = engine_dir / f"{args.engine_name}.calib"
    sidecar = engine_dir / f"{args.engine_name}.match_log.txt"

    if args.analyze_only:
        print("\n--analyze-only set; exiting without writing files.")
        return

    print(f"\nWriting hybrid cache: {hybrid_cache}")
    write_trt_cache(hybrid_cache, header, hybrid_entries)

    print(f"Writing match log:    {sidecar}")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(f"# hybrid cache match log\n")
        f.write(f"# factor = {factor}\n")
        f.write(f"# replaced {n_replaced} / {n_total} TRT entries\n\n")
        for trt_n in trt_entries:
            if trt_n in mapping:
                onnx_n = mapping[trt_n]
                m = modelopt_scales[onnx_n]
                t_old = _hex_to_float(trt_entries[trt_n])
                t_new = m * factor
                f.write(
                    f"REPL[{strategy[trt_n]:>9}]  {trt_n}\n"
                    f"           onnx={onnx_n}\n"
                    f"           old={t_old:.4e}  new={t_new:.4e}  "
                    f"modelopt_y_scale={m:.4e}\n\n"
                )
            else:
                t_old = _hex_to_float(trt_entries[trt_n])
                f.write(f"KEEP                {trt_n}    {t_old:.4e}\n")

    if args.no_build:
        print("--no-build set; exiting without engine build.")
        return

    print(f"\nBuilding TRT engine: {engine_path}")
    config = VAEDecodeInt8Config(
        workspace_gb=args.workspace_gb,
        min_frames=args.min_frames,
        opt_frames=args.opt_frames,
        max_frames=args.max_frames,
        calibrator_kind="entropy",  # irrelevant: cache short-circuits get_batch
    )
    calib_dir = models_dir() / "calibration" / "vae_latents_60s"
    build_vae_decode_int8_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        calibration_dir=calib_dir,
        config=config,
        cache_path=hybrid_cache,
    )
    size_mb = engine_path.stat().st_size / 1e6
    print(f"\nDone: {engine_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

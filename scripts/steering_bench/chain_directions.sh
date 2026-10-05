#!/usr/bin/env bash
# P1/P3 chain: recapture at post_block_residual (MusicCaps + self-label), then build directions.
#   GPU=0 CAP=D:/steer-bench OUT=E:/Projects/steering-bench/2026-10-05/sa3 chain_directions.sh
# Steps (each skipped when its output exists): MusicCaps capture (5521, ~9 GPU-min, 3.3 GB),
# self-label capture (800 + audio, ~1.5-3 GPU-min, ~0.5 GB + ~0.7 GB audio), build both (CPU).
set -u
: "${OUT:?set OUT}"; CAP=${CAP:-D:/steer-bench}; GPU=${GPU:-0}; NSELF=${NSELF:-800}
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT PYTHONUTF8=1
P=${P:-/c/_dev/projects/DEMON/.venv/Scripts/python.exe}
B=scripts/steering_bench
mkdir -p "$OUT/logs"
[ -f "$CAP/resid_mc/meta.json" ] || $P $B/capture_resid.py --out "$CAP/resid_mc" > "$OUT/logs/capture_mc.log" 2>&1 \
  || { echo "FLAG: MusicCaps capture failed"; exit 1; }
[ -f "$CAP/resid_self/meta.json" ] || $P $B/capture_resid.py --self-label --n "$NSELF" --out "$CAP/resid_self" \
  > "$OUT/logs/capture_self.log" 2>&1 || { echo "FLAG: self-label capture failed"; exit 1; }
$P $B/build_directions.py --capture "$CAP/resid_mc" --out "$OUT" --per-step-pack > "$OUT/logs/build_mc.log" 2>&1 \
  || { echo "FLAG: build mc failed"; exit 1; }
$P $B/build_directions.py --capture "$CAP/resid_self" --self-label --out "$OUT" --per-step-pack \
  > "$OUT/logs/build_self.log" 2>&1 || { echo "FLAG: build self failed"; exit 1; }
echo "done: $OUT/resid_mc_summary.json $OUT/resid_self_summary.json"

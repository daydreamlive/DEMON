#!/usr/bin/env bash
# Per-block sweep of one concept's re-estimated directions (P2), one GPU, sequential.
#
#   CUDA GPU:  sweep_blocks.sh <concept> <gpu> <tag> <block> [<block> ...]
#   e.g.       OUT=E:/Projects/steering-bench/2026-10-05/sa3 sweep_blocks.sh bright 0 resid_mc 23 20 15 7 2
#
# Per block b: sa3_tada_run.py sweep --method pack --pack $OUT/packs_<tag>_b<bb> --sites pack
# (format-1 pack at block b; --method pack is the bench path that steers post_block_residual),
# 20 holdout prompts, --points 2 (alphas -a, -a/2, 0, a/2, a), a = K x alpha_scale_per_block[b]
# from $OUT/<tag>_summary.json (pooled within-class std of the projection, step mean), so every
# block is pushed by the same number of activation stds. Then LPAPS only (eval env) and the five
# descriptors (CPU) -> one row per block in $OUT/$SUB/summary.csv. PROD=1 (default) also runs
# the production pack at its own block with the same K (dir suffix _prod) as the baseline row.
#
# Env: OUT (bench root, required), K (default 4), SUB (default blocks_<tag>_<concept>),
#      PROD (1), PROD_PACKS (production packs dir, read-only), BATCH (20), TADA_ROOT, HF_HOME.
set -u
[ $# -ge 4 ] || { sed -n 2,16p "$0"; exit 2; }
C=$1; GPU=$2; TAG=$3; shift 3; BLOCKS="$*"
: "${OUT:?set OUT to the bench root (sa3_tada_run.py --out)}"
K=${K:-4}; SUB=${SUB:-blocks_${TAG}_${C}}; PROD=${PROD:-1}; BATCH=${BATCH:-20}
PROD_PACKS=${PROD_PACKS:-$HOME/.daydream-scope/models/demon/steering_packs/sa3/medium}
WT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT PYTHONUTF8=1
export HF_HOME=${HF_HOME:-D:/huggingface_cache} TADA_ROOT=${TADA_ROOT:-E:/Projects/tada-replication}
P=${P:-/c/_dev/projects/DEMON/.venv/Scripts/python.exe}
E=${E:-/e/Projects/tada-replication/evalenv/Scripts/python.exe}
R=scripts/tada/sa3_tada_run.py
S=scripts/tada/sa3_tada_score.py
B=scripts/steering_bench
LOG=$OUT/$SUB/logs; mkdir -p "$LOG"
$P $R --help 2>/dev/null | grep -q -- "--pack " || { echo "FLAG: $R has no --method pack (other agent's shim not in)"; exit 3; }
scale() { # scale <block>  -> K x alpha_scale_per_block[block]
  $P -c "import json,sys; d=json.load(open(sys.argv[1]))['concepts'][sys.argv[2]]; print(float(sys.argv[3])*d['alpha_scale_per_block'][int(sys.argv[4])])" \
    "$OUT/${TAG}_summary.json" "$C" "$K" "$1"
}
H="--holdout --n-prompts 20 --batch $BATCH --points 2 --concepts $C --eval-sub $SUB --out $OUT"
for b in $BLOCKS; do
  bb=$(printf %02d "$b"); a=$(scale "$b") || { echo "FLAG: no alpha scale for $C b$b"; exit 4; }
  echo "[$(date +%H:%M:%S)] $C b$bb alpha_max $a"
  $P $R sweep --method pack --pack "$OUT/packs_${TAG}_b$bb" --sites pack --alpha-max "$a" --suffix "_b$bb" $H \
    > "$LOG/sweep_b$bb.log" 2>&1 || { echo "FLAG: sweep b$bb failed (see $LOG/sweep_b$bb.log)"; exit 5; }
done
if [ "$PROD" = 1 ] && [ -f "$PROD_PACKS/$C.safetensors" ]; then
  pb=$($P -c "import json,sys; print(json.load(open(sys.argv[1]))['concepts'][sys.argv[2]]['pack_block'])" "$OUT/${TAG}_summary.json" "$C")
  a=$(scale "$pb")
  echo "[$(date +%H:%M:%S)] $C production pack b$pb alpha_max $a"
  $P $R sweep --method pack --pack "$PROD_PACKS/$C.safetensors" --sites pack --alpha-max "$a" --suffix _prod $H \
    > "$LOG/sweep_prod.log" 2>&1 || echo "FLAG: production sweep failed (see $LOG/sweep_prod.log)"
fi
echo "[$(date +%H:%M:%S)] LPAPS"
$E $S protocol --out "$OUT" --sub "$SUB" --lpaps-only > "$LOG/lpaps.log" 2>&1 || { echo "FLAG: LPAPS failed"; exit 6; }
echo "[$(date +%H:%M:%S)] descriptors"
$P $B/score_descriptors.py --dirs "$OUT/$SUB"/pack_pack_${C}_* --summary "$OUT/$SUB/summary.csv" \
  > "$LOG/descriptors.log" 2>&1 || { echo "FLAG: descriptor scoring failed"; exit 7; }
echo "[$(date +%H:%M:%S)] done: $OUT/$SUB/summary.csv"

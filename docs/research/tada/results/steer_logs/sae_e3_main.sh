#!/usr/bin/env bash
# SAE-SA3 E3 main chain: train -> score -> vectors -> held-out probes (+ one scorer beside) -> select.
cd /c/_dev/projects/DEMON-tada-sae-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sae-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
W=D:/tada-replication
O=$W/sae_eval
D="scripts/tada/sa3_sae.py"
S="scripts/tada/sa3_tada_score.py"
B="--blocks 3 5 6 7"
CS="tempo piano mood vocal_gender vocal_style violin guitar_electronic rock_genre electronic_music"
X="$B --work $W --sa3-out $O --steps 8"
until grep -q "CACHE_OK\|CACHE_FAILED" $L/sae_e3_cache.log; do sleep 30; done
grep -q CACHE_OK $L/sae_e3_cache.log || { echo "MAIN_FAILED cache"; exit 1; }
echo "train start $(date +%H:%M)"
$P $D train $X --epochs 10 --lr 3e-5 --json $W/sae_train_report.json > $L/sae_e3_train.log 2>&1 || { echo "MAIN_FAILED train"; exit 1; }
echo "score start $(date +%H:%M)"
$P $D score $X --concepts $CS --batch 25 > $L/sae_e3_score.log 2>&1 || { echo "MAIN_FAILED score"; exit 1; }
$P $D vectors $X --concepts $CS --forms perstep > $L/sae_e3_vectors.log 2>&1 || { echo "MAIN_FAILED vectors"; exit 1; }
echo "probe start $(date +%H:%M)"
SPID=
for c in $CS; do
  $P $D probe $X --concepts $c --forms perstep --batch 20 > $L/sae_e3_probe_$c.log 2>&1 || { echo "MAIN_FAILED probe $c"; exit 1; }
  [ -n "$SPID" ] && wait $SPID
  $E $S protocol --out $O --sub calib_sae --methods sae --concepts $c --skip-aesthetics > $L/sae_e3_score_calib_$c.log 2>&1 &
  SPID=$!
done
wait
$E $S auc --out $O --sub calib_sae > $L/sae_e3_auc_calib.log 2>&1 || { echo "MAIN_FAILED auc calib"; exit 1; }
$P $D select $X --concepts $CS --forms perstep --selection $W/sae_kc.json > $L/sae_e3_select.log 2>&1 || { echo "MAIN_FAILED select"; exit 1; }
$P $D calibrate $X --concepts $CS --forms perstep > $L/sae_e3_calibrate.log 2>&1 || { echo "MAIN_FAILED calibrate"; exit 1; }
echo "MAIN_OK $(date +%H:%M)"

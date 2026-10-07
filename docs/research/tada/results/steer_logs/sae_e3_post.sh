#!/usr/bin/env bash
# SAE-SA3 E3 after training: score -> vectors -> held-out probes (+1 scorer) -> select -> calibrate
# -> eval50 sweep (+1 scorer) -> swap refs -> auc -> report -> packs -> ear package. 7 concepts (no vocals).
cd /c/_dev/projects/DEMON-tada-sae-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sae-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
W=D:/tada-replication
O=$W/sae_eval
D="scripts/tada/sa3_sae.py"
S="scripts/tada/sa3_tada_score.py"
B="--blocks ${BLOCKS:-3 5 6 7}"
CS="tempo piano mood violin guitar_electronic rock_genre electronic_music"
X="$B --work $W --sa3-out $O --steps 8"
echo "score start $(date +%H:%M)"
$P $D score $X --concepts $CS --batch 25 > $L/sae_e3_score.log 2>&1 || { echo "POST_FAILED score"; exit 1; }
$P $D vectors $X --concepts $CS --forms perstep > $L/sae_e3_vectors.log 2>&1 || { echo "POST_FAILED vectors"; exit 1; }
echo "probe start $(date +%H:%M)"
SPID=
for c in $CS; do
  $P $D probe $X --concepts $c --forms perstep --batch 20 > $L/sae_e3_probe_$c.log 2>&1 || { echo "POST_FAILED probe $c"; exit 1; }
  [ -n "$SPID" ] && wait $SPID
  $E $S protocol --out $O --sub calib_sae --methods sae --concepts $c --skip-aesthetics > $L/sae_e3_score_calib_$c.log 2>&1 &
  SPID=$!
done
wait
$E $S auc --out $O --sub calib_sae > $L/sae_e3_auc_calib.log 2>&1 || { echo "POST_FAILED auc calib"; exit 1; }
$P $D select $X --concepts $CS --forms perstep --selection $W/sae_kc.json > $L/sae_e3_select.log 2>&1 || { echo "POST_FAILED select"; exit 1; }
$P $D calibrate $X --concepts $CS --forms perstep > $L/sae_e3_calibrate.log 2>&1 || { echo "POST_FAILED calibrate"; exit 1; }
echo "eval start $(date +%H:%M)"
Y="$X --selection $W/sae_kc.json"
SPID=
for c in $CS; do
  $P $D sweep $Y --concepts $c --forms perstep --sub eval_sae --site-name loc --n-prompts 50 --points 10 --batch 25 > $L/sae_e3_sweep_$c.log 2>&1 || { echo "POST_FAILED sweep $c"; exit 1; }
  [ -n "$SPID" ] && wait $SPID
  $E $S protocol --out $O --sub eval_sae --methods sae --concepts $c --skip-aesthetics > $L/sae_e3_score_eval_$c.log 2>&1 &
  SPID=$!
done
$P scripts/tada/sa3_tada_run.py swap --out $O --eval-sub eval_sae --concepts piano mood tempo --n-prompts 50 --batch 25 > $L/sae_e3_swap.log 2>&1 || echo "POST_FAILED swap"
wait
echo "gpu done $(date +%H:%M)"
$E $S auc --out $O --sub eval_sae > $L/sae_e3_auc_eval.log 2>&1 || { echo "POST_FAILED auc"; exit 1; }
$P $D report $Y --concepts $CS --sub eval_sae --site-name loc --json $W/sae_report.json > $L/sae_e3_report.log 2>&1 || echo "POST_FAILED report"
$P $D pack $Y --concepts $CS > $L/sae_e3_pack.log 2>&1 || echo "POST_FAILED pack"
$P $D listen $Y --concepts piano mood tempo --sub eval_sae --site-name loc > $L/sae_e3_listen.log 2>&1 || echo "POST_FAILED listen"
echo "POST_OK $(date +%H:%M)"

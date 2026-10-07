#!/usr/bin/env bash
# SAE-SA3 E3 eval chain: eval50 sweep at the selected k_c, PCI swap renders for the ear package,
# reference protocol scoring (one scorer beside the generator), AUC, report, packs, ear package.
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
X="$B --work $W --sa3-out $O --steps 8 --selection $W/sae_kc.json"
echo "eval start $(date +%H:%M)"
SPID=
for c in $CS; do
  $P $D sweep $X --concepts $c --forms perstep --sub eval_sae --site-name loc --n-prompts 50 --points 10 --batch 25 > $L/sae_e3_sweep_$c.log 2>&1 || { echo "EVAL_FAILED sweep $c"; exit 1; }
  [ -n "$SPID" ] && wait $SPID
  $E $S protocol --out $O --sub eval_sae --methods sae --concepts $c --skip-aesthetics > $L/sae_e3_score_eval_$c.log 2>&1 &
  SPID=$!
done
$P scripts/tada/sa3_tada_run.py swap --out $O --eval-sub eval_sae --concepts piano mood tempo --n-prompts 50 --batch 25 > $L/sae_e3_swap.log 2>&1 || echo "EVAL_FAILED swap"
echo "gpu generation done $(date +%H:%M)"
wait
$E $S auc --out $O --sub eval_sae > $L/sae_e3_auc_eval.log 2>&1 || { echo "EVAL_FAILED auc"; exit 1; }
$P $D report $X --concepts $CS --sub eval_sae --site-name loc --json $W/sae_report.json > $L/sae_e3_report.log 2>&1 || echo "EVAL_FAILED report"
$P $D pack $X --concepts $CS > $L/sae_e3_pack.log 2>&1 || echo "EVAL_FAILED pack"
$E -c "import sys" && $P $D listen $X --concepts piano mood tempo --sub eval_sae --site-name loc > $L/sae_e3_listen.log 2>&1 || echo "EVAL_FAILED listen"
echo "EVAL_OK $(date +%H:%M)"

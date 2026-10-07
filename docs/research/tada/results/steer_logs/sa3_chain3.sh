#!/usr/bin/env bash
# SA3 TADA chain 3 (replaces chain 2): eval on the first 50 benchmark prompts, 10 strengths per side.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
X="--eval-sub eval50 --n-prompts 50 --points 10 --batch 25 --loc 3 6 7 --ranges E:/Projects/tada-replication/sa3/ranges_eval50.json"
$P $R subset --eval-sub eval50 --n-prompts 50 > $L/tada_subset.log 2>&1 || echo "subset FAILED"
$E $S cutoff --sub eval50 --methods pci > $L/tada_cutoff.log 2>&1 || echo "cutoff FAILED"
$P $R calibrate --eval-sub eval50 > $L/tada_calibrate.log 2>&1 || echo "calibrate FAILED"
cat $L/tada_calibrate.log
$P $R sweep --method caa --sites loc all $X > $L/tada_sweep_caa.log 2>&1 || echo "sweep caa FAILED"
$E $S protocol --sub eval50 --methods caa > $L/tada_score_caa.log 2>&1 &
$P $R sweep --method caa --sites ablated --concepts piano tempo mood vocal_gender $X > $L/tada_sweep_caa_abl.log 2>&1 || echo "sweep caa abl FAILED"
$P $R sweep --method austeer --sites loc all $X > $L/tada_sweep_aus.log 2>&1 || echo "sweep aus FAILED"
wait
$E $S protocol --sub eval50 --methods caa austeer > $L/tada_score_aus.log 2>&1 || echo "score aus FAILED"
$E $S auc --sub eval50 > $L/tada_auc.log 2>&1 || echo "auc FAILED"
echo chain3 done

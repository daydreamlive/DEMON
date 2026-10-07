#!/usr/bin/env bash
# Distillation control: TADA CAA unchanged in method on SA3 medium-base (50 steps, cfg 7, cond-pass steering).
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
O=E:/Projects/tada-replication/sa3_base
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
CC="tempo piano mood"
B="--out $O --checkpoint medium-base --cfg 7 --steps 50 --concepts $CC"
H="--holdout --n-prompts 20 --batch 20 --loc 3 5 6 7"
$P $R caa $B --audio-tokens --vec-dir caa --batch 25 > $L/base_caa.log 2>&1 || echo "caa FAILED"
echo "caa $(date +%H:%M)"
$P $R pci $B $H --eval-sub eval --pci-ks 12 25 38 50 > $L/base_pci.log 2>&1 || echo "pci FAILED"
echo "pci $(date +%H:%M)"
$P $R probe --method caa --sites loc $B $H --vec-dir caa --calib-sub calib --alphas 4 8 16 32 64 -4 -8 -16 -32 -64 > $L/base_probe.log 2>&1 || echo "probe FAILED"
echo "probe $(date +%H:%M)"
$E $S cutoff --out $O --sub eval --methods pci > $L/base_cutoff.log 2>&1 || echo "cutoff FAILED"
$E $S protocol --out $O --sub calib --lpaps-only > $L/base_score_calib.log 2>&1 || echo "score calib FAILED"
$P $R calibrate --out $O --eval-sub eval --calib-sub calib --concepts $CC > $L/base_calibrate.log 2>&1 || echo "calibrate FAILED"
cat $L/base_calibrate.log
$P $R sweep --method caa --sites loc $B $H --vec-dir caa --eval-sub eval --points 14 --ranges $O/ranges_eval.json > $L/base_sweep.log 2>&1 || echo "sweep FAILED"
echo "swept $(date +%H:%M)"
$E $S protocol --out $O --sub eval > $L/base_score_a.log 2>&1 &
$E $S protocol --out $O --sub eval --reverse > $L/base_score_b.log 2>&1
wait
$E $S auc --out $O --sub eval > $L/base_auc.log 2>&1 || echo "auc FAILED"
cat $L/base_auc.log
echo "base done $(date +%H:%M)"

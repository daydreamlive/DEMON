#!/usr/bin/env bash
# SA3 TADA chain 2: AUSteer probes, scoring of probes + PCI, calibration, all sweeps, protocol, AUC.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
$P $R probe --method austeer --sites all loc --holdout --n-prompts 20 --loc 3 6 7 --batch 20 \
  --alphas 0.0625 0.125 0.25 0.5 1 2 4 -0.0625 -0.125 -0.25 -0.5 -1 -2 -4 > $L/tada_probe_aus.log 2>&1 || echo "probe aus FAILED"
$E $S protocol --sub calib --skip-aesthetics > $L/tada_score_calib.log 2>&1 || echo "score calib FAILED"
$E $S protocol --sub eval --methods pci --skip-aesthetics > $L/tada_score_pci.log 2>&1 || echo "score pci FAILED"
$P $R calibrate > $L/tada_calibrate.log 2>&1 || echo "calibrate FAILED"
cat $L/tada_calibrate.log
# sweeps, each followed by its scoring in the background so scoring overlaps generation
$P $R sweep --method caa --sites loc all ablated --loc 3 6 7 --batch 20 --ranges E:/Projects/tada-replication/sa3/ranges.json > $L/tada_sweep_caa.log 2>&1 || echo "sweep caa FAILED"
$E $S protocol --sub eval --methods caa > $L/tada_score_caa.log 2>&1 &
$P $R sweep --method austeer --sites loc all --loc 3 6 7 --batch 20 --ranges E:/Projects/tada-replication/sa3/ranges.json > $L/tada_sweep_aus.log 2>&1 || echo "sweep aus FAILED"
wait
$E $S protocol --sub eval --methods austeer > $L/tada_score_aus.log 2>&1 || echo "score aus FAILED"
$E $S auc --sub eval > $L/tada_auc.log 2>&1 || echo "auc FAILED"
echo chain2 done

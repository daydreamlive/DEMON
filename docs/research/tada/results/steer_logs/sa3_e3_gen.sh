#!/usr/bin/env bash
# E3 generator chain: AUSteer reselect at the paper-metric blocks, PCI-loc at those blocks, loc probes.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
R="scripts/tada/sa3_tada_run.py"
LOC="--loc 3 5 6 7"
C="--vec-dir caa_e3 --calib-sub calib_e3 --holdout --n-prompts 20 --batch 20 $LOC"
$P $R austeer --audio-tokens --vec-dir caa_e3 --batch 25 $LOC --force > $L/e3_austeer_loc.log 2>&1 || echo "austeer FAILED"
echo "austeer $(date +%H:%M)"
$P $R pci --eval-sub eval_e3 --n-prompts 50 --batch 25 --pci-sites loc $LOC > $L/e3_pci_loc.log 2>&1 || echo "pci loc FAILED"
echo "pci loc $(date +%H:%M)"
$P $R probe --method caa --sites loc $C --alphas 2 4 8 16 32 -2 -4 -8 -16 -32 > $L/e3_probe_caa_loc.log 2>&1 || echo "probe caa loc FAILED"
$P $R probe --method austeer --sites loc $C --alphas 0.25 0.5 1 2 4 -0.25 -0.5 -1 -2 -4 > $L/e3_probe_aus_loc.log 2>&1 || echo "probe aus loc FAILED"
$P $R probe --method caakv --sites loc $C --alphas 0.5 1 2 4 8 16 -0.5 -1 -2 -4 -8 -16 > $L/e3_probe_kv_loc.log 2>&1 || echo "probe kv loc FAILED"
echo "probed loc $(date +%H:%M)"

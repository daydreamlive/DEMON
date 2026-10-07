#!/usr/bin/env bash
# E3 step 3/5: calibration probes (holdout 20) for the corrected vectors at the all-blocks site.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
R="scripts/tada/sa3_tada_run.py"
C="--vec-dir caa_e3 --calib-sub calib_e3 --holdout --n-prompts 20 --batch 20 --loc 3 6 7"
$P $R probe --method caa --sites all $C --alphas 1 2 4 8 16 -1 -2 -4 -8 -16 > $L/e3_probe_caa_all.log 2>&1 || echo "probe caa all FAILED"
$P $R probe --method austeer --sites all $C --alphas 0.125 0.25 0.5 1 2 -0.125 -0.25 -0.5 -1 -2 > $L/e3_probe_aus_all.log 2>&1 || echo "probe aus all FAILED"
echo "probed all $(date +%H:%M)"

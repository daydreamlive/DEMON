#!/usr/bin/env bash
# SA3 TADA chain 1: calibration probes (holdout) then PCI (test), scoring overlapped.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
$P scripts/tada/sa3_tada_run.py probe --holdout --n-prompts 20 --loc 3 6 7 --batch 20 \
  --alphas 0.5 1 2 4 8 16 32 -0.5 -1 -2 -4 -8 -16 -32 > $L/tada_probe.log 2>&1 || echo "probe FAILED"
$P scripts/tada/sa3_tada_run.py pci --loc 3 6 7 --batch 20 > $L/tada_pci.log 2>&1 || echo "pci FAILED"
echo chain1 gen done

#!/usr/bin/env bash
# E3 step 1-2: MuQ eq.2 scoring of the patch audio at all three sites, then MuQ localization.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
$E scripts/tada/sa3_tada_score.py patch --metric muq --patch-dir patch > $L/e3_muq_patch_xattn_cond.log 2>&1 || echo "muq patch FAILED"
$P scripts/tada/sa3_tada_run.py localize --metric muq --patch-site xattn_cond > $L/e3_localize_muq_xattn_cond.log 2>&1 || echo "localize FAILED"
echo "muq xattn_cond $(date +%H:%M)"
$E scripts/tada/sa3_tada_score.py patch --metric muq --patch-dir patch_xattn_out > $L/e3_muq_patch_xattn_out.log 2>&1 || echo "muq xattn_out FAILED"
$P scripts/tada/sa3_tada_run.py localize --metric muq --patch-site xattn_out > $L/e3_localize_muq_xattn_out.log 2>&1
echo "muq xattn_out $(date +%H:%M)"
until grep -q "patched resid" $L/e2_chain.log; do sleep 60; done
$E scripts/tada/sa3_tada_score.py patch --metric muq --patch-dir patch_resid > $L/e3_muq_patch_resid.log 2>&1 || echo "muq resid FAILED"
$P scripts/tada/sa3_tada_run.py localize --metric muq --patch-site resid > $L/e3_localize_muq_resid.log 2>&1
echo "muq resid $(date +%H:%M)"

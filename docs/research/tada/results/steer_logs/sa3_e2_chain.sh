#!/usr/bin/env bash
# SA3 TADA E2: oracle (per-pair per-token / per-pair mean) at cross_attn_output loc blocks, then patching sweeps at xattn_out and resid.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
O=/e/Projects/tada-replication/sa3
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
CS="--concepts tempo piano mood"
for c in tempo piano mood; do for s in all loc; do
  mkdir -p $O/e2_oracle/pci_${s}_$c/protocol_results
  cp $O/eval50/pci_${s}_$c/protocol_results/*.csv $O/e2_oracle/pci_${s}_$c/protocol_results/
done; done
for m in oracle oraclemean; do
  $P $R probe --method $m --sites loc --holdout --n-prompts 25 --batch 25 --loc 3 6 7 $CS \
    --alphas 0.5 1 2 4 8 16 32 -0.5 -1 -2 -4 -8 -16 -32 > $L/e2_probe_$m.log 2>&1 || echo "probe $m FAILED"
done
echo "probed $(date +%H:%M)"
$E $S protocol --sub calib --methods oracle oraclemean --skip-aesthetics > $L/e2_score_calib.log 2>&1 || echo "score calib FAILED"
$P $R calibrate --eval-sub e2_oracle $CS > $L/e2_calibrate.log 2>&1 || echo "calibrate FAILED"
cat $L/e2_calibrate.log
echo "calibrated $(date +%H:%M)"
X="--eval-sub e2_oracle --n-prompts 50 --points 10 --batch 25 --loc 3 6 7 --sites loc --ranges $O/ranges_e2_oracle.json $CS"
$P $R sweep --method oracle $X > $L/e2_sweep_oracle.log 2>&1 || echo "sweep oracle FAILED"
$P $R sweep --method oraclemean $X > $L/e2_sweep_oraclemean.log 2>&1 || echo "sweep oraclemean FAILED"
echo "swept oracle $(date +%H:%M)"
( $E $S protocol --sub e2_oracle --methods oracle oraclemean > $L/e2_score_oracle.log 2>&1 && \
  $E $S auc --sub e2_oracle > $L/e2_auc_oracle.log 2>&1 || echo "score oracle FAILED"; echo "scored oracle $(date +%H:%M)" ) &
for site in xattn_out resid; do
  $P $R patch --patch-site $site --pairs 34 --seeds 8 --batch 16 > $L/e2_patch_$site.log 2>&1 || echo "patch $site FAILED"
  echo "patched $site $(date +%H:%M)"
  wait
  ( $E $S patch --patch-dir patch_$site > $L/e2_score_patch_$site.log 2>&1 && \
    $P $R localize --patch-site $site > $L/e2_localize_$site.log 2>&1 || echo "score patch $site FAILED"; echo "localized $site $(date +%H:%M)" ) &
done
wait
echo "e2 chain done $(date +%H:%M)"

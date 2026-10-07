#!/usr/bin/env bash
# E4b item 2: large-N CAA vectors (caa_e4) on ARC, CAA loc 3 5 6 7, tempo/piano/mood, 20 holdout prompts,
# cutoffs = eval_e3 PCI endpoints. Matched controls on the same 20 prompts: caa_e3 vectors and PCI-all.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
O=/e/Projects/tada-replication/sa3
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
CC="tempo piano mood"
H="--holdout --n-prompts 20 --batch 20 --loc 3 5 6 7 --concepts $CC"
for c in $CC; do for s in all loc; do
  mkdir -p $O/eval_e4/pci_${s}_$c/protocol_results $O/eval_e4_e3vec/pci_${s}_$c/protocol_results
  cp $O/eval_e3/pci_${s}_$c/protocol_results/lpaps_endpoints.csv $O/eval_e4/pci_${s}_$c/protocol_results/
  cp $O/eval_e3/pci_${s}_$c/protocol_results/lpaps_endpoints.csv $O/eval_e4_e3vec/pci_${s}_$c/protocol_results/
done; done
$P $R probe --method caa --sites loc $H --vec-dir caa_e4 --calib-sub calib_e4 --alphas 2 4 8 16 32 64 -2 -4 -8 -16 -32 -64 > $L/e4_probe.log 2>&1 || echo "probe FAILED"
$E $S protocol --sub calib_e4 --lpaps-only > $L/e4_score_calib.log 2>&1 || echo "score calib FAILED"
$P $R calibrate --eval-sub eval_e4 --calib-sub calib_e4 --concepts $CC > $L/e4_calibrate.log 2>&1 || echo "calibrate FAILED"
cat $L/e4_calibrate.log
echo "calibrated $(date +%H:%M)"
$P $R sweep --method caa --sites loc $H --vec-dir caa_e4 --eval-sub eval_e4 --points 14 --ranges $O/ranges_eval_e4.json > $L/e4_sweep.log 2>&1 || echo "sweep FAILED"
$E $S protocol --sub eval_e4 --labels caa_loc --skip-aesthetics > $L/e4_score_a.log 2>&1 &
# matched controls: caa_e3 vectors with their eval_e3 ranges, and the real prompt (PCI-all), same 20 prompts
$P $R sweep --method caa --sites loc $H --vec-dir caa_e3 --eval-sub eval_e4_e3vec --points 14 --ranges $O/ranges_eval_e3.json > $L/e4_sweep_e3vec.log 2>&1 || echo "sweep e3vec FAILED"
$P $R pci $H --eval-sub eval_e4_pci --pci-sites all > $L/e4_pci.log 2>&1 || echo "pci FAILED"
echo "swept $(date +%H:%M)"
$E $S protocol --sub eval_e4_e3vec --labels caa_loc --skip-aesthetics > $L/e4_score_b.log 2>&1 &
$E $S protocol --sub eval_e4_pci --labels pci_all --skip-aesthetics > $L/e4_score_c.log 2>&1
wait
for c in $CC; do
  for sub in eval_e4 eval_e4_e3vec; do
    cp $O/eval_e4_pci/pci_all_$c/protocol_results/*.csv $O/$sub/pci_all_$c/protocol_results/
    cp $O/eval_e3/pci_all_$c/protocol_results/lpaps_endpoints.csv $O/$sub/pci_all_$c/protocol_results/
  done
done
for sub in eval_e4 eval_e4_e3vec; do $E $S auc --sub $sub > $L/e4_auc_$sub.log 2>&1 || echo "auc $sub FAILED"; cat $L/e4_auc_$sub.log; done
echo "e4 done $(date +%H:%M)"

#!/usr/bin/env bash
# E5: final SA3 table. eval50 protocol (first 50 test prompts, seed 2115), 14 strengths/side, renorm off,
# cutoffs = existing eval_e3 PCI (PCI-all full protocol + PCI-loc 3,5,6,7 endpoints).
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
O=/e/Projects/tada-replication/sa3
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
C7="tempo piano mood violin guitar_electronic rock_genre electronic_music"
C3="tempo piano mood"
LOC="--loc 3 5 6 7"
for sub in eval_e5 eval_e5p; do for c in $C7; do for s in all loc; do
  mkdir -p $O/$sub/pci_${s}_$c/protocol_results
  cp $O/eval_e3/pci_${s}_$c/protocol_results/lpaps_endpoints.csv $O/$sub/pci_${s}_$c/protocol_results/
done; done; done
H="--holdout --n-prompts 20 --batch 20 $LOC"
$P $R probe --method caa --sites loc all $H --concepts $C7 --vec-dir caa_e5 --calib-sub calib_e5 --alphas 2 4 8 16 32 64 -2 -4 -8 -16 -32 -64 > $L/e5_probe.log 2>&1 || echo "probe FAILED"
$P $R probe --method caa --sites loc $H --concepts $C3 --vec-dir caa_e5p --calib-sub calib_e5p --alphas 2 4 8 16 32 64 -2 -4 -8 -16 -32 -64 > $L/e5p_probe.log 2>&1 || echo "probe p FAILED"
echo "probed $(date +%H:%M)"
$E $S protocol --sub calib_e5 --lpaps-only > $L/e5_score_calib.log 2>&1 || echo "score calib FAILED"
$E $S protocol --sub calib_e5p --lpaps-only > $L/e5p_score_calib.log 2>&1 || echo "score calib p FAILED"
$P $R calibrate --eval-sub eval_e5 --calib-sub calib_e5 --concepts $C7 > $L/e5_calibrate.log 2>&1 || echo "calibrate FAILED"
$P $R calibrate --eval-sub eval_e5p --calib-sub calib_e5p --concepts $C3 > $L/e5p_calibrate.log 2>&1 || echo "calibrate p FAILED"
cat $L/e5_calibrate.log $L/e5p_calibrate.log
echo "calibrated $(date +%H:%M)"
X="--n-prompts 50 --batch 25 --points 14 $LOC"
$P $R sweep --method caa --sites loc all --concepts $C7 --vec-dir caa_e5 --eval-sub eval_e5 $X --ranges $O/ranges_eval_e5.json > $L/e5_sweep.log 2>&1 || echo "sweep FAILED"
$E $S protocol --sub eval_e5 --labels caa_loc caa_all --skip-aesthetics > $L/e5_score_a.log 2>&1 &
$P $R sweep --method caa --sites loc --concepts $C3 --vec-dir caa_e5p --eval-sub eval_e5p $X --ranges $O/ranges_eval_e5p.json > $L/e5p_sweep.log 2>&1 || echo "sweep p FAILED"
$P $R swap --eval-sub eval_e5 --n-prompts 50 --batch 25 --concepts tempo piano > $L/e5_swap.log 2>&1 || echo "swap FAILED"
echo "swept $(date +%H:%M)"
$E $S protocol --sub eval_e5p --labels caa_loc --skip-aesthetics > $L/e5_score_b.log 2>&1
$E $S protocol --sub eval_e5 --labels caa_loc caa_all --skip-aesthetics --reverse > $L/e5_score_c.log 2>&1
# caa_e3 all-blocks row (rendered in E3, never scored)
$E $S protocol --sub eval_e3 --labels caa_all --concepts $C7 --skip-aesthetics > $L/e5_score_e3all.log 2>&1 || echo "score e3 all FAILED"
wait
for sub in eval_e5 eval_e5p; do for c in $C7; do
  [ -d $O/$sub/pci_all_$c ] && cp $O/eval_e3/pci_all_$c/protocol_results/*.csv $O/eval_e3/pci_all_$c/protocol_results/*.json $O/$sub/pci_all_$c/protocol_results/
done; done
for sub in eval_e5 eval_e5p eval_e3; do $E $S auc --sub $sub > $L/e5_auc_$sub.log 2>&1 || echo "auc $sub FAILED"; done
echo "e5 done $(date +%H:%M)"

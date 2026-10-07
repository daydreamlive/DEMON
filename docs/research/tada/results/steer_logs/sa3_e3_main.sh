#!/usr/bin/env bash
# E3 main chain: cutoffs, calibration, sweeps (corrected vectors, blocks 3 5 6 7), scoring, AUC.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
O=/e/Projects/tada-replication/sa3
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
LOC="--loc 3 5 6 7"
until grep -q "probed loc" $L/e3_gen_chain.log; do sleep 30; done
# K/V probe again with the real-token mask (--audio-tokens), replacing the gen chain's abs-sum-mask probe
$P $R probe --method caakv --sites loc --audio-tokens --vec-dir caa_e3 --calib-sub calib_e3 --holdout --n-prompts 20 --batch 20 $LOC \
  --alphas 0.5 1 2 4 8 16 -0.5 -1 -2 -4 -8 -16 --force > $L/e3_probe_kv_loc.log 2>&1 || echo "probe kv FAILED"
echo "kv probe $(date +%H:%M)"
$E $S cutoff --sub eval_e3 --methods pci > $L/e3_cutoff.log 2>&1 || echo "cutoff FAILED"
for c in tempo piano mood vocal_gender vocal_style violin guitar_electronic rock_genre electronic_music; do
  mkdir -p $O/eval_e3/pci_all_$c/protocol_results
  cp $O/eval50/pci_all_$c/protocol_results/lpaps_endpoints.csv $O/eval_e3/pci_all_$c/protocol_results/
done
$E $S protocol --sub calib_e3 --lpaps-only --force --labels caakv_loc > $L/e3_score_calib_kv.log 2>&1 || echo "score calib kv FAILED"
$E $S protocol --sub calib_e3 --lpaps-only > $L/e3_score_calib_loc.log 2>&1 || echo "score calib FAILED"
$P $R calibrate --eval-sub eval_e3 --calib-sub calib_e3 > $L/e3_calibrate.log 2>&1 || echo "calibrate FAILED"
cat $L/e3_calibrate.log
echo "calibrated $(date +%H:%M)"
X="--audio-tokens --vec-dir caa_e3 --eval-sub eval_e3 --n-prompts 50 --points 10 --batch 25 $LOC --ranges $O/ranges_eval_e3.json"
$P $R sweep --method caa --sites loc $X > $L/e3_sweep_caa_loc.log 2>&1 || echo "sweep caa loc FAILED"
echo "swept caa loc $(date +%H:%M)"
$E $S protocol --sub eval_e3 --labels caa_loc > $L/e3_score_a.log 2>&1 &
$P $R sweep --method austeer --sites loc $X > $L/e3_sweep_aus_loc.log 2>&1 || echo "sweep aus loc FAILED"
$P $R sweep --method caa --sites all $X > $L/e3_sweep_caa_all.log 2>&1 || echo "sweep caa all FAILED"
$P $R sweep --method caakv --sites loc $X > $L/e3_sweep_kv_loc.log 2>&1 || echo "sweep kv FAILED"
$P $R sweep --method caa --sites loc ${X/--eval-sub eval_e3/--eval-sub eval_e3_renorm} --concepts tempo --renorm --range-scale 4 > $L/e3_sweep_renorm.log 2>&1 || echo "sweep renorm FAILED"
echo "swept all $(date +%H:%M)"
wait
# PCI-all full protocol (eval50, same audio) must be done; copy its csvs so eval_e3 skips those dirs
while ps -ef | grep -v grep | grep -q "labels pci_all"; do sleep 30; done
for c in tempo piano mood vocal_gender vocal_style violin guitar_electronic rock_genre electronic_music; do
  cp $O/eval50/pci_all_$c/protocol_results/*.csv $O/eval50/pci_all_$c/protocol_results/*.json $O/eval_e3/pci_all_$c/protocol_results/
done
for s_ in all loc; do mkdir -p $O/eval_e3_renorm/pci_${s_}_tempo/protocol_results; done
$E $S protocol --sub eval_e3 > $L/e3_score_b.log 2>&1 &
$E $S protocol --sub eval_e3 --reverse > $L/e3_score_c.log 2>&1
wait
cp $O/eval_e3/pci_all_tempo/protocol_results/*.csv $O/eval_e3_renorm/pci_all_tempo/protocol_results/
cp $O/eval_e3/pci_loc_tempo/protocol_results/*.csv $O/eval_e3_renorm/pci_loc_tempo/protocol_results/
$E $S protocol --sub eval_e3_renorm --labels caa_loc > $L/e3_score_renorm.log 2>&1 || echo "score renorm FAILED"
$E $S auc --sub eval_e3 > $L/e3_auc.log 2>&1 || echo "auc FAILED"
$E $S auc --sub eval_e3_renorm > $L/e3_auc_renorm.log 2>&1 || echo "auc renorm FAILED"
echo "scored $(date +%H:%M)"

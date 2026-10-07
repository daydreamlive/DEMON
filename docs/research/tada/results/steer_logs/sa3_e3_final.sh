#!/usr/bin/env bash
# E3 final GPU chain (after SAE DONE/FLAG): localised rows at SITE2 blocks 0 1 3 5 6 7, 7 concepts (vocals excluded),
# K/V and renorm control, scoring, AUC, ear renders.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
O=/e/Projects/tada-replication/sa3
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
CC="tempo piano mood violin guitar_electronic rock_genre electronic_music"
LOC="--loc 0 1 3 5 6 7"
# keep the blocks 3 5 6 7 rows (sensitivity) in their own sub
mkdir -p $O/eval_e3_b3567
for d in $O/eval_e3/caa_loc_* $O/eval_e3/austeer_loc_* $O/eval_e3/caakv_loc_* $O/eval_e3/pci_loc_*; do mv "$d" $O/eval_e3_b3567/; done
for c in $CC vocal_gender vocal_style; do mkdir -p $O/eval_e3_b3567/pci_all_$c && cp -r $O/eval_e3/pci_all_$c/protocol_results $O/eval_e3_b3567/pci_all_$c/; done
# calibration sub for the new blocks; all-site probe curves carried over (same vectors)
mkdir -p $O/calib_e3b
for d in $O/calib_e3/caa_all_* $O/calib_e3/austeer_all_*; do mkdir -p $O/calib_e3b/$(basename $d) && cp -r $d/protocol_results $O/calib_e3b/$(basename $d)/; done
$P $R austeer --audio-tokens --vec-dir caa_e3 --batch 25 $LOC --force --concepts $CC > $L/e3f_austeer.log 2>&1 || echo "austeer FAILED"
$P $R pci --eval-sub eval_e3 --n-prompts 50 --batch 25 --pci-sites loc $LOC --concepts $CC > $L/e3f_pci_loc.log 2>&1 || echo "pci loc FAILED"
echo "pci loc $(date +%H:%M)"
C="--audio-tokens --vec-dir caa_e3 --calib-sub calib_e3b --holdout --n-prompts 20 --batch 20 $LOC --concepts $CC"
$P $R probe --method caa --sites loc $C --alphas 2 4 8 16 32 -2 -4 -8 -16 -32 > $L/e3f_probe_caa.log 2>&1 || echo "probe caa FAILED"
$P $R probe --method austeer --sites loc $C --alphas 0.25 0.5 1 2 4 -0.25 -0.5 -1 -2 -4 > $L/e3f_probe_aus.log 2>&1 || echo "probe aus FAILED"
$P $R probe --method caakv --sites loc $C --alphas 8 16 32 64 128 256 -8 -16 -32 -64 -128 -256 > $L/e3f_probe_kv.log 2>&1 || echo "probe kv FAILED"
echo "probed $(date +%H:%M)"
$E $S cutoff --sub eval_e3 --methods pci --concepts $CC > $L/e3f_cutoff.log 2>&1 || echo "cutoff FAILED"
$E $S protocol --sub calib_e3b --lpaps-only --labels caa_loc austeer_loc caakv_loc > $L/e3f_score_calib.log 2>&1 || echo "score calib FAILED"
$P $R calibrate --eval-sub eval_e3 --calib-sub calib_e3b --concepts $CC > $L/e3f_calibrate.log 2>&1 || echo "calibrate FAILED"
cat $L/e3f_calibrate.log
echo "calibrated $(date +%H:%M)"
X="--audio-tokens --vec-dir caa_e3 --eval-sub eval_e3 --n-prompts 50 --points 10 --batch 25 $LOC --ranges $O/ranges_eval_e3.json --concepts $CC"
$P $R sweep --method caa --sites loc $X > $L/e3f_sweep_caa.log 2>&1 || echo "sweep caa FAILED"
$E $S protocol --sub eval_e3 --concepts $CC > $L/e3f_score_a.log 2>&1 &
$P $R sweep --method austeer --sites loc $X > $L/e3f_sweep_aus.log 2>&1 || echo "sweep aus FAILED"
$P $R sweep --method caakv --sites loc $X > $L/e3f_sweep_kv.log 2>&1 || echo "sweep kv FAILED"
$P $R sweep --method caa --sites loc ${X/--eval-sub eval_e3/--eval-sub eval_e3_renorm} --concepts tempo --renorm --range-scale 4 > $L/e3f_sweep_renorm.log 2>&1 || echo "sweep renorm FAILED"
$P $R swap --eval-sub eval_e3 --n-prompts 50 --batch 25 --concepts piano mood tempo > $L/e3f_swap.log 2>&1 || echo "swap FAILED"
echo "swept $(date +%H:%M)"
wait
$E $S protocol --sub eval_e3 --concepts $CC > $L/e3f_score_b.log 2>&1 &
$E $S protocol --sub eval_e3 --concepts $CC --reverse > $L/e3f_score_c.log 2>&1
wait
for s_ in all loc; do mkdir -p $O/eval_e3_renorm/pci_${s_}_tempo; cp -r $O/eval_e3/pci_${s_}_tempo/protocol_results $O/eval_e3_renorm/pci_${s_}_tempo/; done
$E $S protocol --sub eval_e3_renorm --labels caa_loc > $L/e3f_score_renorm.log 2>&1 || echo "score renorm FAILED"
$E $S protocol --sub eval_e3_b3567 --labels pci_loc --concepts $CC > $L/e3f_score_b3567.log 2>&1 || echo "score b3567 FAILED"
for sub in eval_e3 eval_e3_renorm eval_e3_b3567; do $E $S auc --sub $sub > $L/e3f_auc_$sub.log 2>&1 || echo "auc $sub FAILED"; done
echo "scored $(date +%H:%M)"

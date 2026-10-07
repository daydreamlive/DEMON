#!/bin/bash
# Ship pass queue items (ship_uniform_rigor_runbook.md), run by ship_drive.sh in a driver slot $OUT.
#   ship_item.sh proto PACK OFF PCI_SRC   full protocol (probe, PCI-all, cutoff, calibrate, 7-point sweep, protocol,
#                                         auc) for one pack at EVAL_SEED+OFF -> $SH/results_s<OFF>/<PACK>; PCI_SRC (a
#                                         bench pack with the same descriptors, or -) reuses that pack's PCI-all
#                                         results of the same seed instead of rendering them
#   ship_item.sh fixed VID OFF            fixed-gain pass at the shipped gain -> $SH/fixed_s<OFF>/<VID>/<pack>/
# Audio is recycled through $OUT/pool (moved, never deleted), as last night's p2_proto.sh.
mode=$1
E2=${E2:-/dev/shm/steerbench/out/many_knobs_v2_eval}
source $E2/env2.sh
SH=$E2/ship; PK=$SH/packs; CATJ=$PK/pci_descriptors.json; POOL=$OUT/pool
S="$EV scripts/tada/sa3_tada_score.py"; SL="$DM $E2/slot.py"
mkdir -p $OUT/logs $POOL
run(){ local name=$1; shift; local lf=$OUT/logs/ship_${tag}_$name.log
  for att in 1 2; do "$@" > $lf 2>&1 && return 0; echo "$name attempt $att rc $?"; done
  echo "FAILED $tag at $name"; tail -20 $lf; cp $lf $RES/ 2>/dev/null; return 1; }
park(){ for d in "$@"; do for x in $OUT/$d/*; do [ -d "$x" ] && $SL drain $POOL $x; done; done
  local ST=$OUT/stale/$(date +%s%N)
  for e in $(for d in "$@"; do find $OUT/$d -mindepth 2 -maxdepth 2 -type d -empty 2>/dev/null; done); do
    mkdir -p $ST/$(dirname ${e#$OUT/}); mv $e $ST/${e#$OUT/}; done; }
R(){ $DM -c "import sys; sys.path[:0]=['scripts/tada','.']; sys.argv=['sa3_tada_run.py']+sys.argv[1:]; import sa3_tada_run as R; R.EVAL_SEED=R.EVAL_SEED+$off; sys.exit(R.main())" "$@"; }
t0=$(date +%s)
if [ "$mode" = proto ]; then
  c=$2; off=$3; src=${4:--}; tag=${c}_s$off
  RES=$SH/results_s$off/$c; mkdir -p $RES
  [ -e $RES/done ] && exit 0
  park calib eval
  # an earlier PCI dir of this concept in this slot is parked whole (it may hold another seed's results)
  if [ -d $OUT/eval/pci_all_$c ]; then P=$OUT/stale/pci_$(date +%s%N); mkdir -p $P; mv $OUT/eval/pci_all_$c $P/; fi
  PROBE="alpha_-128.0 alpha_-64.0 alpha_-32.0 alpha_-16.0 alpha_-8.0 alpha_-4.0 alpha_-2.0 alpha_0.0 alpha_2.0 alpha_4.0 alpha_8.0 alpha_16.0 alpha_32.0 alpha_64.0 alpha_128.0"
  PCI="alpha_-8.0 alpha_-6.0 alpha_-4.0 alpha_-2.0 alpha_0.0 alpha_2.0 alpha_4.0 alpha_6.0 alpha_8.0"
  $SL fill $POOL $OUT/calib/pack_pack_$c small $PROBE
  run probe R probe --method pack --pack $PK --concepts $c --holdout --n-prompts 20 --batch 20 --alphas -128 -64 -32 -16 -8 -4 -2 2 4 8 16 32 64 128 --calib-sub calib --out $OUT --force \
   && run probe_score $S protocol --out $OUT --sub calib --concepts $c --lpaps-only --force; rc=$?
  $SL drain $POOL $OUT/calib/pack_pack_$c; [ $rc -eq 0 ] || exit 1
  SRCD=""
  if [ "$src" != "-" ]; then
    for root in $SH/results_s$off $E2/results_s$off; do
      if [ -e $root/$src/eval/pci_all_$src/protocol_results/lpaps.csv ]; then SRCD=$root/$src/eval/pci_all_$src; break; fi
    done
  fi
  if [ -n "$SRCD" ]; then
    mkdir -p $OUT/eval/pci_all_$c; cp -r $SRCD/protocol_results $OUT/eval/pci_all_$c/
    echo "pci reused from $SRCD" > $RES/pci_reused.txt
  else
    $SL fill $POOL $OUT/eval/pci_all_$c large $PCI
    run pci R pci --pci-descriptors $CATJ --concepts $c --n-prompts 50 --batch 25 --pci-ks 2 4 6 8 --pci-sites all --eval-sub eval --out $OUT --force \
     && run pci_score $S protocol --out $OUT --sub eval --labels pci_all --concepts $c --catalogue $CATJ --force \
     && run cutoff $S cutoff --out $OUT --sub eval --concepts $c --force; rc=$?
    $SL drain $POOL $OUT/eval/pci_all_$c; [ $rc -eq 0 ] || exit 1
  fi
  run calibrate R calibrate --method pack --concepts $c --eval-sub eval --calib-sub calib --out $OUT || exit 1
  cp $OUT/ranges_eval.json $RES/ranges_eval_final_$c.json
  $SL fill $POOL $OUT/eval/pack_pack_$c large $($SL sweepnames $OUT/ranges_eval.json $c 7)
  run sweep R sweep --method pack --pack $PK --concepts $c --n-prompts 50 --batch 25 --points 7 --ranges $OUT/ranges_eval.json --eval-sub eval --out $OUT --force \
   && run protocol $S protocol --out $OUT --sub eval --labels pack_pack --concepts $c --catalogue $CATJ --force \
   && run auc $S auc --out $OUT --sub eval --concepts $c --catalogue $CATJ; rc=$?
  $SL drain $POOL $OUT/eval/pack_pack_$c
  for d in calib/pack_pack_$c eval/pci_all_$c eval/pack_pack_$c; do $SL results $OUT/$d $RES/$d; done
  cp $OUT/eval/auc.json $RES/auc.json 2>/dev/null; cp $OUT/logs/ship_${tag}_*.log $RES/ 2>/dev/null
  [ $rc -eq 0 ] || exit 1
  touch $RES/done; echo "done proto $c seed+$off $(( $(date +%s)-t0 )) s"
elif [ "$mode" = fixed ]; then
  vid=$2; off=$3; tag=fixed_${vid}_s$off
  RES=$SH/fixed_s$off/$vid; mkdir -p $RES
  [ -e $RES/done ] && exit 0
  park fixed
  $DM scripts/steering_bench/ship_tools.py fixedplan $vid $off > $OUT/ship_plan.txt || exit 1
  cp $OUT/ship_ranges.json $RES/ranges.json
  while read -r pack lo hi; do
    [ -z "$pack" ] && continue
    $SL fill $POOL $OUT/fixed/pack_pack_$pack large $($SL sweepnames $OUT/ship_ranges.json $pack 1)
    run sweep_$pack R sweep --method pack --pack $PK --concepts $pack --n-prompts 50 --batch 25 --points 1 --ranges $OUT/ship_ranges.json --eval-sub fixed --out $OUT --force \
     && run protocol_$pack $S protocol --out $OUT --sub fixed --labels pack_pack --concepts $pack --catalogue $CATJ --skip-aesthetics --force; rc=$?
    $SL drain $POOL $OUT/fixed/pack_pack_$pack
    $SL results $OUT/fixed/pack_pack_$pack $RES/$pack
    [ $rc -eq 0 ] || exit 1
  done < $OUT/ship_plan.txt
  touch $RES/done; echo "done fixed $vid seed+$off $(( $(date +%s)-t0 )) s"
else
  echo "unknown mode $mode"; exit 2
fi

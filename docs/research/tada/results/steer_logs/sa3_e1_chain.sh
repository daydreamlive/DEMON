#!/usr/bin/env bash
# SA3 TADA E1: steering guidance {3,5,7} (g1 = eval50, verified bit-identical), 30-step g1 control, renorm control.
cd /c/_dev/projects/DEMON-tada-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
O=/e/Projects/tada-replication/sa3
R="scripts/tada/sa3_tada_run.py"
S="scripts/tada/sa3_tada_score.py"
X="--n-prompts 50 --points 10 --batch 25 --loc 3 6 7 --ranges E:/Projects/tada-replication/sa3/ranges_eval50.json --method caa"
prep() {  # PCI cutoffs (8-step endpoints) shared with eval50
  for c in tempo piano mood; do for s in all loc; do
    mkdir -p $O/$1/pci_${s}_$c/protocol_results
    cp $O/eval50/pci_${s}_$c/protocol_results/*.csv $O/$1/pci_${s}_$c/protocol_results/
  done; done
}
score() { $E $S protocol --sub $1 --methods caa > $L/e1_score_$1.log 2>&1 && $E $S auc --sub $1 > $L/e1_auc_$1.log 2>&1 || echo "score $1 FAILED"; echo "scored $1 $(date +%H:%M)"; }
run() {  # sub, extra args
  local sub=$1; shift
  prep $sub
  $P $R sweep --sites loc all --concepts tempo piano mood --eval-sub $sub $X "$@" > $L/e1_sweep_$sub.log 2>&1 || echo "sweep $sub FAILED"
  echo "swept $sub $(date +%H:%M)"
}
run e1_g3 --guidance 3 --range-scale 0.3333333333; wait; score e1_g3 &
run e1_g5 --guidance 5 --range-scale 0.2; wait; score e1_g5 &
run e1_g7 --guidance 7 --range-scale 0.1428571429; wait; score e1_g7 &
run e1_renorm --renorm --range-scale 4 --concepts tempo; wait; score e1_renorm &
run e1_s30 --steps 30; wait; score e1_s30
echo "e1 chain done $(date +%H:%M)"

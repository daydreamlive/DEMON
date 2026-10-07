#!/bin/bash
# Ship pass, local side: hourly GPU accounting lines from the box into status_ship.md (append-only); when the box
# writes ship/DONE: copy packs + tables to E:, install the packs (old set moved to ~/.claude-trash), restart the
# SA3 steer demo backend on 1318 (Next on 6660 untouched), append the outcome, exit.
SSH="ssh -o ConnectTimeout=30 -i $HOME/.ssh/inside -p 10126 root@ssh1.vast.ai"
SCP="scp -q -o ConnectTimeout=30 -i $HOME/.ssh/inside -P 10126"
SH=/dev/shm/steerbench/out/many_knobs_v2_eval/ship
ST=/c/_dev/projects/DEMON/notes/steering_pr/status_ship.md
DST=/e/Projects/DEMON/steering-bench/many_knobs_v2/ship
INST=$HOME/.daydream-scope/models/demon/steering_packs/sa3/medium
seen=0
while true; do
  n=$($SSH "grep -c 'hourly GPU accounting\|FLAG\|final rc\|B2 mirror rc\|final set known' $SH/status_box.md 2>/dev/null" 2>/dev/null | tail -1)
  if [ -n "$n" ] && [ "$n" -gt "$seen" ] 2>/dev/null; then
    $SSH "grep 'hourly GPU accounting\|FLAG\|final rc\|B2 mirror rc\|final set known' $SH/status_box.md" 2>/dev/null \
      | grep '^- ' | tail -n $((n - seen)) >> $ST
    seen=$n
  fi
  if $SSH "test -e $SH/DONE" 2>/dev/null; then break; fi
  sleep 1800
done
mkdir -p $DST
$SSH "cd $SH && tar cf - --exclude=claims --exclude=packs --exclude='*.tmp' ." 2>/dev/null | tar xf - -C $DST
cp $DST/results_ship.md /c/_dev/projects/DEMON/notes/steering_pr/results_ship.md 2>/dev/null
NP=$(ls $DST/packs_final/sa3/medium/*.safetensors 2>/dev/null | wc -l)
echo "- $(date -u +%FT%TZ)(local) local copy to E:\\Projects\\DEMON\\steering-bench\\many_knobs_v2\\ship done ($NP packs); results_ship.md copied to notes/steering_pr" >> $ST
if [ "$NP" -gt 0 ]; then
  T=$HOME/.claude-trash/steering_packs_v2_2026-10-06
  mkdir -p $T
  [ -e $T/medium ] && T=$T/medium.$(date +%Y-%m-%dT%H-%M-%S) && mkdir -p $T
  mv $INST $T/ && mkdir -p $INST && cp $DST/packs_final/sa3/medium/*.safetensors $INST/
  echo "- $(date -u +%FT%TZ)(local) installed $(ls $INST | wc -l) packs to $INST; previous set moved to $T" >> $ST
  PID=$(netstat -ano | grep -E "127.0.0.1:1318 .*LISTENING" | awk '{print $NF}' | head -1)
  [ -n "$PID" ] && taskkill //PID $PID //F > /dev/null 2>&1
  sleep 5
  nohup bash /c/_dev/projects/DEMON/notes/family_merge/demo_logs/LAUNCH_sa3_steer.sh > /dev/null 2>&1 &
  for i in $(seq 1 30); do sleep 5; grep -q server_ready /c/_dev/projects/DEMON/notes/family_merge/demo_logs/sa3_steer.log 2>/dev/null && break; done
  R=$(grep -c server_ready /c/_dev/projects/DEMON/notes/family_merge/demo_logs/sa3_steer.log 2>/dev/null)
  echo "- $(date -u +%FT%TZ)(local) demo backend restarted (old PID ${PID:-none}); server_ready lines: $R; Next on 6660 untouched" >> $ST
fi

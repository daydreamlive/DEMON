#!/bin/bash
# usage: ship_drive.sh GPU J   claim driver for the ship pass: lowest-prio unclaimed line of $SH/queue.txt
# ("<prio> <id> <cmd...>"), claimed by mkdir $SH/claims/<id>; runs cmd with CUDA_VISIBLE_DEVICES=GPU,
# DRV=g<GPU>_<J>, OUT=$E2/slot_g<GPU>_<J> (last night's slots and audio pools, recycled in place).
g=$1; j=$2
export E2=${E2:-/dev/shm/steerbench/out/many_knobs_v2_eval}
SH=$E2/ship
export DRV=g${g}_$j OUT=$E2/slot_g${g}_$j CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT $SH/claims $SH/drivers
ST=$SH/drivers/$DRV.log
echo "$(date -u +%FT%TZ) up pid $$" >> $ST
idle0=""
while true; do
  [ -e $SH/FREEZE ] && { echo "$(date -u +%FT%TZ) freeze, exit" >> $ST; exit 0; }
  [ -e $SH/DONE ] && { echo "$(date -u +%FT%TZ) done, exit" >> $ST; exit 0; }
  item=""; cmd=""
  while read -r prio id rest; do
    [ -z "$id" ] && continue
    [ -e $SH/claims/$id ] && continue
    if mkdir $SH/claims/$id 2>/dev/null; then item=$id; cmd=$rest; break; fi
  done < <(sort -s -n -k1,1 $SH/queue.txt 2>/dev/null)
  if [ -z "$item" ]; then
    [ -z "$idle0" ] && { idle0=$(date +%s); echo "$(date -u +%FT%TZ) idle start" >> $ST; }
    sleep 20; continue
  fi
  [ -n "$idle0" ] && { echo "$(date -u +%FT%TZ) idle end $(( $(date +%s)-idle0 )) s" >> $ST; idle0=""; }
  t0=$(date +%s); echo "$(date -u +%FT%TZ) start $item" >> $ST
  bash -c "$cmd" > $SH/claims/$item/log 2>&1; rc=$?
  echo $rc > $SH/claims/$item/rc
  echo "$(date -u +%FT%TZ) end $item rc $rc $(( $(date +%s)-t0 )) s" >> $ST
done

#!/bin/bash
# B2 mirror of the ship pass (copy only: never deletes, never sync), then a one-way checksum check.
# Same credential route as /dev/shm/steerbench/b2_mirror.sh. Usage: bash b2_ship.sh [dry|run]
set -u
MODE="${1:-run}"
set -a; . /workspace/b2_creds.env; set +a
export RCLONE_CONFIG_B2M_TYPE=b2
export RCLONE_CONFIG_B2M_ACCOUNT="$B2_KEY_ID"
export RCLONE_CONFIG_B2M_KEY="$B2_APPLICATION_KEY"
DRY=""; [ "$MODE" = "run" ] || DRY="--dry-run"
SH=/dev/shm/steerbench/out/many_knobs_v2_eval/ship
MK=b2m:mucket/sa3/steering_bench/many_knobs_v2_ship_2026-10-06
rclone $DRY copy $SH $MK --transfers 8 --checksum --exclude "claims/**" --exclude "*.tmp" 2>&1 | tail -3
rc=${PIPESTATUS[0]}
if [ "$MODE" = run ]; then
  rclone check --one-way --checksum $SH $MK --exclude "claims/**" --exclude "*.tmp" --exclude "status_box.md" \
    --exclude "drivers/**" --exclude "loop.log" 2>&1 | tail -2
fi
echo "MIRROR_EXIT mode=$MODE rc=$rc $(date -u +%FT%TZ)"

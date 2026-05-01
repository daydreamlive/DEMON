#!/usr/bin/env bash
# Update /root/ace-step to latest main and launch the realtime_motion_graph
# server on 0.0.0.0:8765 in a detached tmux session.
set -euo pipefail

export PATH="$PATH:/root/.local/bin:/root/.cargo/bin"

REPO=/root/ace-step
SESSION=motion_graph
LOG=/root/motion_graph_server.log
# acestep.paths.checkpoints_dir() defaults to ~/.daydream-scope/models/rtmg/checkpoints.
# The models on this box live at /root/ace-step/{checkpoints,trt_engines}, so point
# ACESTEP_MODELS_DIR at /root/ace-step to make paths.py resolve correctly.
export ACESTEP_MODELS_DIR=/root/ace-step

cd "$REPO"

echo "=== Pre-pull state ==="
git log --oneline -1
git status --short || true

# The only 'local modification' is CRLF/LF whitespace on a file the refactor deletes.
# Stash it out of the way so pull can proceed, then drop the stash.
if ! git diff --quiet -- demos/realtime_motion_graph_server.py 2>/dev/null; then
    echo "=== Stashing whitespace-only local mod ==="
    git stash push -u -m "pre-pull-crlf-$(date +%s)" -- demos/realtime_motion_graph_server.py || true
fi

echo "=== Pulling origin main ==="
git pull --ff-only origin main

echo "=== Dropping stash (whitespace-only, file deleted by refactor) ==="
git stash drop 2>/dev/null || true

echo "=== Post-pull state ==="
git log --oneline -3
ls demos/realtime_motion_graph/ || true

echo "=== uv sync ==="
uv sync 2>&1 | tail -30

echo "=== Killing any existing motion_graph tmux session / port 8765 holders ==="
tmux kill-session -t "$SESSION" 2>/dev/null || true
# fuser may not be installed; try lsof then ss
if command -v fuser >/dev/null; then
    fuser -k 8765/tcp 2>/dev/null || true
fi

echo "=== Launching server in tmux session '$SESSION' ==="
: > "$LOG"
tmux new-session -d -s "$SESSION" "cd $REPO && ACESTEP_MODELS_DIR=/root/ace-step uv run python -u -m demos.realtime_motion_graph.server --host 0.0.0.0 --port 8765 2>&1 | tee -a $LOG"

echo "=== Tmux sessions ==="
tmux ls

echo "=== Waiting for server to bind port 8765 (up to 120s) ==="
for i in $(seq 1 120); do
    if ss -ltn 2>/dev/null | grep -q ':8765'; then
        echo "Port 8765 is listening (after ${i}s)"
        break
    fi
    sleep 1
done

echo "=== Final ss output ==="
ss -ltn 2>/dev/null | grep -E '8765|State' || true

echo "=== Last 40 lines of server log ==="
tail -40 "$LOG" || true

echo "=== DONE ==="

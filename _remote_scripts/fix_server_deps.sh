#!/usr/bin/env bash
# Install all server-side runtime deps that `uv sync` stripped out, restart the
# motion_graph tmux session with ACESTEP_MODELS_DIR set, tail the log.
set -euo pipefail

export PATH="$PATH:/root/.local/bin:/root/.cargo/bin"

echo "=== apt: libportaudio2 (needed by sounddevice) ==="
apt-get install -y libportaudio2 2>&1 | tail -5 || true

echo "=== uv pip install missing server deps ==="
cd /root/ace-step
/root/.local/bin/uv pip install \
    librosa \
    sounddevice \
    soxr \
    zstandard \
    opencv-python-headless \
    pygame \
    mido \
    python-rtmidi \
    2>&1 | tail -20

echo "=== import smoke test ==="
/root/ace-step/.venv/bin/python - <<'PY'
import sys
mods = ["librosa", "sounddevice", "soxr", "zstandard", "numpy", "websockets"]
for m in mods:
    try:
        __import__(m)
        print(f"  ok  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
        sys.exit(1)
PY

echo "=== restart tmux motion_graph ==="
tmux kill-session -t motion_graph 2>/dev/null || true
: > /root/motion_graph_server.log
tmux new-session -d -s motion_graph \
    "cd /root/ace-step && ACESTEP_MODELS_DIR=/root/ace-step /root/.local/bin/uv run python -u -m demos.realtime_motion_graph.server --host 0.0.0.0 --port 8765 2>&1 | tee -a /root/motion_graph_server.log"

echo "=== wait for listening ==="
for i in $(seq 1 60); do
    if grep -q 'Listening' /root/motion_graph_server.log 2>/dev/null; then
        echo "Server listening (after ${i}s)"
        break
    fi
    sleep 1
done

echo "=== log tail ==="
tail -30 /root/motion_graph_server.log
echo "=== DONE ==="

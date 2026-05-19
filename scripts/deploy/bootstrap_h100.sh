#!/bin/bash
# Runs on the H100 vast.ai box. Idempotent.
# Bootstraps the repo, installs uv + deps, downloads models, builds TRT engines.
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN not set (non-fatal; public models should still download)"
fi

REPO_DIR=/root/acestep
MODELS_DIR=/root/.daydream-scope/models/demon

# ---------- system deps ----------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends git git-lfs curl rsync build-essential ca-certificates openssh-client
git lfs install --system

# ---------- uv ----------
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"
uv --version

# ---------- github deploy key ----------
chmod 600 /root/.ssh/deploy_key
cat > /root/.ssh/config <<'SSHCONF'
Host github.com
    IdentityFile /root/.ssh/deploy_key
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
SSHCONF
chmod 600 /root/.ssh/config
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null || true

# ---------- clone repo ----------
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone git@github.com:daydreamlive/DEMON.git "$REPO_DIR"
else
    echo "Repo already present at $REPO_DIR"
    (cd "$REPO_DIR" && git fetch && git reset --hard origin/main)
fi

cd "$REPO_DIR"
git log -1 --oneline

# ---------- uv sync ----------
uv sync --frozen || uv sync
# Verify critical imports
uv run python -c "import torch, tensorrt, transformers, diffusers, websockets, zstandard; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('TRT', tensorrt.__version__)"

echo "=== BOOTSTRAP DONE ==="

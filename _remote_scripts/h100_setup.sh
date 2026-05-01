#!/bin/bash
# H100 bootstrap: uv sync, download HF models, prepare for engine build.
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/acestep

echo "=== uv sync (torch 2.9.1+cu128, TRT 10.13, flash-attn) ==="
uv sync 2>&1 | tail -40

echo
echo "=== sanity: torch / cuda / TRT versions ==="
uv run python -c "
import torch, tensorrt
print(f'torch: {torch.__version__}')
print(f'cuda: {torch.version.cuda}')
print(f'gpu: {torch.cuda.get_device_name(0)}')
print(f'gpu sm: {torch.cuda.get_device_capability(0)}')
print(f'tensorrt: {tensorrt.__version__}')
"

echo
echo "=== download HF models (main + xl-turbo) ==="
mkdir -p /root/.daydream-scope/models/rtmg/checkpoints
uv run python -c "
import os
os.environ.setdefault('HF_HOME','/root/.cache/huggingface')
from huggingface_hub import snapshot_download
print('main model: ACE-Step/Ace-Step1.5')
snapshot_download('ACE-Step/Ace-Step1.5',
    local_dir='/root/.daydream-scope/models/rtmg/checkpoints',
    allow_patterns=['vae/*','Qwen3-Embedding-0.6B/*','acestep-v15-turbo/*','acestep-5Hz-lm-1.7B/*'])
print('xl-turbo: ACE-Step/acestep-v15-xl-turbo')
snapshot_download('ACE-Step/acestep-v15-xl-turbo',
    local_dir='/root/.daydream-scope/models/rtmg/checkpoints/acestep-v15-xl-turbo')
print('done')
"

echo
echo "=== check disk and gpu free ==="
df -h /
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv

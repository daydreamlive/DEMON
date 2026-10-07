#!/usr/bin/env bash
# SAE-SA3 E3: cache cross_attn outputs at blocks 3 5 6 7 (audio tokens, all 8 steps) on D:.
cd /c/_dev/projects/DEMON-tada-sae-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sae-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
$P -c "import acestep;print(acestep.__file__)" > $L/sae_e3_cache.log 2>&1
$P scripts/tada/sa3_sae.py cache --blocks 3 5 6 7 --steps 8 --every 1 --batch 16 --work D:/tada-replication --json D:/tada-replication/sae_cache_report.json >> $L/sae_e3_cache.log 2>&1 && echo CACHE_OK >> $L/sae_e3_cache.log || echo CACHE_FAILED >> $L/sae_e3_cache.log

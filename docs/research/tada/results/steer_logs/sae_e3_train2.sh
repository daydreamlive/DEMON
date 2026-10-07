#!/usr/bin/env bash
# SAE-SA3 E3 training, revised: blocks 6 7 paper grid; then k=128 (m 4 8 16) for 5 6 7, then 3.
cd /c/_dev/projects/DEMON-tada-sae-sa3 || exit 1
export PYTHONPATH=/c/_dev/projects/DEMON-tada-sae-sa3 PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
L=/c/_dev/projects/DEMON/notes/family_merge/steer_logs
W=D:/tada-replication
D="scripts/tada/sa3_sae.py"
echo "train2 start $(date +%H:%M)"
$P $D train --blocks 6 7 --work $W --epochs 10 --lr 3e-5 > $L/sae_e3_train_67.log 2>&1 || { echo "TRAIN_FAILED 67"; exit 1; }
echo "k128 start $(date +%H:%M)"
$P $D train --blocks 5 6 7 3 --work $W --epochs 10 --lr 3e-5 --ms 4 8 16 --ks 128 > $L/sae_e3_train_k128.log 2>&1 || { echo "TRAIN_FAILED k128"; exit 1; }
echo "TRAIN_OK $(date +%H:%M)"

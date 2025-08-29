#!/bin/bash
# 
# OpenPI ACRLPD Fold Box Training Script
# 启动fold_box任务的强化学习训练
#

set -e  # Exit on any error

echo "Starting OpenPI ACRLPD Fold Box Training"
echo "========================================"
#关闭遗留进程，定期清理
pkill -f "spawn_main" || true
# Set environment variables for GPU and HuggingFace cache  
# 增加XLA内存分配以充分利用GPU资源
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95  # 增加至0.9到0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # 禁用预分配优化（启用有可能会报错）
#export XLA_PYTHON_CLIENT_ALLOCATOR="platform" #禁用更高效
export HF_HOME="/era-ai/lm/dataset/huggingface"
export HF_CACHE_HOME="/era-ai/lm/dataset/huggingface_cache"
export HF_DATASETS_CACHE="/era-ai/lm/dataset/huggingface_cache/datasets"
export HF_LEROBOT_HOME="/era-ai/lm/dataset/huggingface_cache/lerobot"
export WANDB_BASE_URL=https://api.bandw.top
# Training parameters
CONFIG="rl_fold_box"
#EXP_NAME="fold_box_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="fold_box_rl2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AC_TRAINING_DIR="$(dirname "$SCRIPT_DIR")"

echo "Working directory: $AC_TRAINING_DIR"
echo "Configuration: $CONFIG"
echo "Experiment name: $EXP_NAME"
echo "GPU memory fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "HF cache home: $HF_LEROBOT_HOME"

# Change to ac_training directory
cd "$AC_TRAINING_DIR"

# Check if dataset exists
if [ ! -d "/era-ai/lm/dataset/huggingface_cache/lerobot/fold_box_unified" ]; then
    echo "ERROR: Dataset not found: fold_box_unified"
    echo "Please run the data conversion pipeline first to create the dataset"
    exit 1
fi

echo "Dataset found: fold_box_unified"
echo ""

# Create log directory if it doesn't exist
mkdir -p logs

# Start training with nohup background execution
LOG_FILE="logs/train_${EXP_NAME}.log"
echo "Starting training in background..."
echo "Command: nohup /era-ai/conda_envs/openpi/bin/uv run python scripts/train_acrlpd_pi0_v2.py --config $CONFIG --exp_name $EXP_NAME --overwrite --no_wandb"
echo "Log file: $LOG_FILE"
echo ""

nohup /era-ai/conda_envs/openpi/bin/uv run python scripts/train_acrlpd_pi0_v2.py \
    --config "$CONFIG" \
    --exp_name "$EXP_NAME" \
    --overwrite \
    --debug-transforms \
    --no_wandb \
    > "$LOG_FILE" 2>&1 &
# --resume
TRAIN_PID=$!
echo "Training started in background with PID: $TRAIN_PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Kill training with: kill $TRAIN_PID"
echo "Check results in: ./checkpoints/$EXP_NAME/"
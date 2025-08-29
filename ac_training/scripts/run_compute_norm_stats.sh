#!/bin/bash
# ACRLPD Normalization Statistics Computation Script
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_HOME="/era-ai/lm/dataset/huggingface"
export HF_CACHE_HOME="/era-ai/lm/dataset/huggingface_cache"
export HF_DATASETS_CACHE="/era-ai/lm/dataset/huggingface_cache/datasets"
export HF_LEROBOT_HOME="/era-ai/lm/dataset/huggingface_cache/lerobot"

# Configuration
REPO_ID="fold_box_unified"
OUTPUT_DIR="/tmp/acrlpd_norm_stats/fold_box_unified"
MAX_SAMPLES=10000  # 增加样本数以覆盖更多数据
BATCH_SIZE=128
ACTION_HORIZON=20  # Q-chunking使用20帧action horizon

# Run computation
/era-ai/conda_envs/openpi/bin/uv run python -m ac_training.data.compute_acrlpd_norm_stats \
    --repo-id "${REPO_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples ${MAX_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --action-horizon ${ACTION_HORIZON} \
    --skip-problematic-episodes \
    --verbose
#!/bin/bash
# Q-Chunking V2 正式训练脚本
# 基于改进的分批动态缓存池和ACT backbone可训练版本
export HF_ENDPOINT=https://hf-mirror.com

set -e

echo "Q-Chunking ACT模式训练 V2"
echo "========================"
echo "模式: ACT直接优化 + RL增强"

# 基本参数设置（不在qc_config.py中管理的参数）
DATASET_DIR="/era-ai/lm/dataset/lmc/aloha_pp"
PRETRAIN_CKPT="/dev/shm/lmc/aloha-devel/train_dir/ACT_aloha_pp_20250711_134313/policy_best.ckpt"
GPU_IDS="0,1,2,3,4,5,6,7"
BATCH_SIZE_PER_GPU=40  # ACT模式建议较小batch size   48
EPISODES_PER_EPOCH=64   # ACT模式建议较少episodes   64
STEPS_PER_EPOCH=2500     # ACT模式建议较少steps 500
USE_WANDB=false
RUN_BACKGROUND=false
ACTOR_ARCHITECTURE="act_direct"  # 默认使用ACT直接模式
UPDATE_ACT_DIRECTLY=true         # ACT模式下直接优化ACT参数
RESUME_CKPT=""                  # Resume模式的checkpoint路径
OVERWRITE_MODE=true            # Overwrite模式标志
# 注意：其他算法参数(alpha, qc_lr, num_epochs等)全部在qc_config.py中管理，脚本不重复设置

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --pretrain_ckpt)
            PRETRAIN_CKPT="$2" 
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE_PER_GPU="$2"
            shift 2
            ;;
        --episodes_per_epoch)
            EPISODES_PER_EPOCH="$2"
            shift 2
            ;;
        --steps_per_epoch)
            STEPS_PER_EPOCH="$2"
            shift 2
            ;;
        --update_act_backbone)
            UPDATE_ACT_BACKBONE=true
            shift
            ;;
        --act_mode)
            ACTOR_ARCHITECTURE="act_direct"
            UPDATE_ACT_DIRECTLY=true
            echo "启用ACT直接优化模式"
            shift
            ;;
        --flow_mode)
            ACTOR_ARCHITECTURE="flow_dual"
            UPDATE_ACT_DIRECTLY=false
            echo "启用Flow双网络模式"
            shift
            ;;
        --update_act_directly)
            UPDATE_ACT_DIRECTLY=true
            shift
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --resume)
            RESUME_CKPT="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE_MODE=true
            shift
            ;;
        --help)
            echo "Q-Chunking V2训练脚本"
            echo "参数："
            echo "  --dataset_dir PATH        数据集路径" 
            echo "  --pretrain_ckpt PATH      预训练ACT模型路径（必须）"
            echo "  --gpu_ids IDS             GPU ID列表，用逗号分隔（如: 0,1,2,3）"
            echo "  --batch_size N            每GPU批次大小"
            echo "  --episodes_per_epoch N    每个Epoch的episodes数量"
            echo "  --steps_per_epoch N       每个Epoch的固定步数"
            echo "  --update_act_backbone     更新ACT backbone参数（Flow模式）"
            echo "  --act_mode               使用ACT直接模式（直接优化整个ACT模型）"
            echo "  --flow_mode              使用Flow双网络模式（标准ACFQL，默认）"
            echo "  --update_act_directly    ACT模式下直接优化ACT参数"
            echo "  --use_wandb              启用WandB记录"
            echo "  --resume PATH            从指定checkpoint恢复训练（resume模式）"
            echo "  --overwrite              从头开始训练，忽略现有checkpoint（overwrite模式）"
            echo ""
            echo "模式说明："
            echo "  Flow模式: 标准ACFQL with BC Flow Actor + Onestep Actor + 蒸馏"
            echo "  ACT模式:  简化的直接ACT优化 + RL增强，更实用"
            echo ""
            echo "使用示例："
            echo "  # 从头训练（overwrite模式）"
            echo "  $0 --pretrain_ckpt /path/to/act.ckpt --overwrite"
            echo ""
            echo "  # 接续训练（resume模式）"
            echo "  $0 --pretrain_ckpt /path/to/act.ckpt --resume /path/to/qc_checkpoint.pth"
            echo ""
            echo "注意：算法参数(alpha, qc_lr, num_epochs等)在qc_config.py中管理"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查Python环境
python_cmd="python"
if command -v python3 &> /dev/null; then
    python_cmd="python3"
fi

echo "使用Python: $python_cmd"

# 验证CUDA环境
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到NVIDIA GPU驱动"
    exit 1
fi

echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 设置CUDA可见设备（用户手动指定）
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
echo "设置GPU设备: $GPU_IDS"
echo "GPU数量: $NUM_GPUS"

# 验证数据集
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    exit 1
fi

pos_count=$(find $DATASET_DIR/score_5 -name "*.hdf5" 2>/dev/null | wc -l)
neg_count=$(find $DATASET_DIR/score_1 -name "*.hdf5" 2>/dev/null | wc -l)

echo "数据集统计:"
echo "  路径: $DATASET_DIR" 
echo "  正样本: $pos_count episodes"
echo "  负样本: $neg_count episodes"

if [ $pos_count -eq 0 ] || [ $neg_count -eq 0 ]; then
    echo "错误: 正样本或负样本数量为0"
    exit 1
fi

# 验证训练模式参数（必须选择一种）
if [ -n "$RESUME_CKPT" ] && [ "$OVERWRITE_MODE" = true ]; then
    echo "错误: 不能同时指定 --resume 和 --overwrite"
    exit 1
fi

if [ -z "$RESUME_CKPT" ] && [ "$OVERWRITE_MODE" = false ]; then
    echo "错误: 必须选择训练模式："
    echo "  --resume /path/to/checkpoint.pth  (接续训练)"
    echo "  --overwrite                       (从头训练)"
    exit 1
fi

# 验证预训练ACT模型（两种模式都需要）
if [ -z "$PRETRAIN_CKPT" ]; then
    echo "错误: 必须指定预训练ACT模型路径"
    echo "使用: --pretrain_ckpt /path/to/act.ckpt"
    exit 1
fi

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "错误: 预训练ACT模型文件不存在: $PRETRAIN_CKPT"
    exit 1
fi

echo "使用预训练ACT模型: $PRETRAIN_CKPT"

# Resume模式验证
if [ -n "$RESUME_CKPT" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "错误: Resume checkpoint文件不存在: $RESUME_CKPT"
        exit 1
    fi
    echo "训练模式: Resume - 从QC checkpoint恢复: $RESUME_CKPT"
fi

# Overwrite模式验证
if [ "$OVERWRITE_MODE" = true ]; then
    echo "训练模式: Overwrite - 从头开始QC训练"
fi

# 验证并读取ACT配置
echo "正在验证预训练模型配置..."
$python_cmd test_config_reader.py --pretrain_path "$(dirname "$PRETRAIN_CKPT")" --quiet
if [ $? -ne 0 ]; then
    echo "警告: 无法读取预训练模型配置，将使用默认参数"
fi

# 配置系统说明
echo "配置系统: 参数从qc_config.py读取，可通过命令行覆盖"
echo "架构模式: $ACTOR_ARCHITECTURE"
echo "ACT直接优化: $UPDATE_ACT_DIRECTLY"
echo "显示当前配置:"
$python_cmd -c "from qc_config import print_config; print_config()"

# 计算总batch size
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
echo "总批次大小: $TOTAL_BATCH_SIZE ($NUM_GPUS x $BATCH_SIZE_PER_GPU)"

# 数据量计算和内存估算（V3简化版）
TOTAL_SAMPLES=$(($EPISODES_PER_EPOCH * 50))  # 每个episode最多50个采样点
echo "数据统计:"
echo "  每Epoch episodes: $EPISODES_PER_EPOCH"
echo "  每Epoch采样点: $TOTAL_SAMPLES"
echo "  每Epoch步数: $STEPS_PER_EPOCH (固定)"
echo "  数据利用率: 每个episode随机采样，每Epoch重新随机化"

# 内存估算 (每个episode约50个采样点 × 预估11MB/采样点)
MEMORY_PER_EPOCH=$(($EPISODES_PER_EPOCH * 50 * 11 / 1024))  # GB  
TOTAL_MEMORY=$(($MEMORY_PER_EPOCH * $NUM_GPUS))
echo "预估GPU内存需求: ~${TOTAL_MEMORY}GB (比V2节省约75%内存)"

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./checkpoints/checkpoints_qc_v2_${TIMESTAMP}"
LOG_DIR="./log"
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR
echo "输出目录: $OUTPUT_DIR"
echo "日志目录: $LOG_DIR"

# 架构模式参数
ARCHITECTURE_FLAGS=""
ACT_BACKBONE_FLAG=""

# 设置架构模式
if [ "$ACTOR_ARCHITECTURE" = "act_direct" ]; then
    ARCHITECTURE_FLAGS="--actor_architecture act_direct"
    if [ "$UPDATE_ACT_DIRECTLY" = true ]; then
        ARCHITECTURE_FLAGS="$ARCHITECTURE_FLAGS --update_act_directly"
        echo "ACT直接模式: 启用ACT参数直接优化"
    else
        echo "ACT直接模式: 仅使用ACT作为Actor，参数冻结"
    fi
elif [ "$ACTOR_ARCHITECTURE" = "flow_dual" ]; then
    ARCHITECTURE_FLAGS="--actor_architecture flow_dual"
    echo "Flow双网络模式: 标准ACFQL架构"
    
    # Flow模式下的ACT backbone参数
    if [ "$UPDATE_ACT_BACKBONE" = true ]; then
        ACT_BACKBONE_FLAG="--update_act_backbone"
        echo "启用ACT backbone参数更新（用于Critic编码器）"
    fi
fi

# WandB参数
WANDB_FLAG=""
if [ "$USE_WANDB" = true ]; then
    WANDB_FLAG="--use_wandb"
    echo "启用WandB记录"
fi

# 训练模式参数
MODE_FLAGS=""
if [ -n "$RESUME_CKPT" ]; then
    MODE_FLAGS="--resume $RESUME_CKPT"
    echo "Resume模式: 从checkpoint恢复训练"
elif [ "$OVERWRITE_MODE" = true ]; then
    MODE_FLAGS="--overwrite"
    echo "Overwrite模式: 从头开始训练"
fi

# 构建训练命令
if [ $NUM_GPUS -gt 1 ]; then
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 train_qc_v2.py"
    echo "启用分布式训练: $NUM_GPUS GPUs"
else
    TRAIN_CMD="$python_cmd train_qc_v2.py"
fi

# 完整命令 - 两种模式都需要预训练ACT模型和数据集
TRAIN_CMD="$TRAIN_CMD \
    --dataset_dir $DATASET_DIR \
    --task_name aloha_pp_qc_v2 \
    --pretrain_ckpt $PRETRAIN_CKPT \
    --ckpt_dir $OUTPUT_DIR \
    --seed 42 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --episodes_per_epoch $EPISODES_PER_EPOCH \
    --steps_per_epoch $STEPS_PER_EPOCH \
    $ARCHITECTURE_FLAGS \
    $ACT_BACKBONE_FLAG \
    $WANDB_FLAG \
    $MODE_FLAGS"


echo "开始训练..."
echo "命令: $TRAIN_CMD"
echo ""

# 设置日志文件
LOG_FILE="${LOG_DIR}/qc_v2_training_${TIMESTAMP}.log"

# 保存训练命令和完整启动脚本
echo "$TRAIN_CMD" > $OUTPUT_DIR/train_command_v2.sh
chmod +x $OUTPUT_DIR/train_command_v2.sh
echo "训练命令已保存: $OUTPUT_DIR/train_command_v2.sh"

# 创建完整的nohup启动脚本
NOHUP_SCRIPT="${OUTPUT_DIR}/start_training.sh"
cat > $NOHUP_SCRIPT << EOF
#!/bin/bash
# Q-Chunking V2 后台训练启动脚本
# 生成时间: $(date)
# GPU设备: $GPU_IDS
# 预训练模型: $PRETRAIN_CKPT
# 数据集: $DATASET_DIR
# 架构模式: $ACTOR_ARCHITECTURE
# ACT直接优化: $UPDATE_ACT_DIRECTLY

cd "$(pwd)"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "开始Q-Chunking V2训练 - \$(date)"
echo "日志文件: $LOG_FILE"
echo "检查点目录: $OUTPUT_DIR"
echo ""

# 执行训练命令
$TRAIN_CMD

echo ""
echo "训练结束 - \$(date)"
EOF

chmod +x $NOHUP_SCRIPT

echo ""
echo "启动nohup后台训练..."
echo "训练日志: $LOG_FILE"
echo "检查点目录: $OUTPUT_DIR"

# 启动后台训练（stderr重定向到stdout）
nohup bash $NOHUP_SCRIPT > $LOG_FILE 2>&1 &
TRAIN_PID=$!

echo ""
echo "训练已在后台启动:"
echo "  进程ID: $TRAIN_PID"
echo "  日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE              # 查看训练日志"
echo "  ps aux | grep $TRAIN_PID       # 检查进程状态"
echo "  kill $TRAIN_PID                # 停止训练"
echo ""
echo "训练将在后台继续运行，可以安全关闭当前终端。"

# 等待几秒确保进程启动
sleep 3
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "训练进程启动成功 (PID: $TRAIN_PID)"
else
    echo "训练进程启动失败，请检查日志: $LOG_FILE"
    exit 1
fi
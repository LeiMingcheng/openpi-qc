#!/bin/bash

# 🚀 ACRLPD高性能数据转换脚本
# 基于OpenPI设计，预计提升30-40倍速度
# 使用：bash fast_conversion.sh

echo "🚀 启动ACRLPD高性能数据转换..."
echo "📅 $(date): 开始快速转换模式"

# 设置环境变量
export HF_HOME="/era-ai/lm/dataset/huggingface"
export HF_CACHE_HOME="/era-ai/lm/dataset/huggingface_cache"
export HF_DATASETS_CACHE="/era-ai/lm/dataset/huggingface_cache/datasets"
export HF_LEROBOT_HOME="/era-ai/lm/dataset/huggingface_cache/lerobot"

# 切换到工作目录
cd /dev/shm/lmc/openpi/ac_training

# 设置日志文件
LOG_FILE="/dev/shm/lmc/openpi/data_conversion.log"

echo "📝 日志文件: $LOG_FILE"
echo "⚡ 使用高性能配置: 10进程, 5线程 (vs 原来的2进程1线程)"
echo "🎯 预计速度提升: 30-40倍 (从180s/episode → 4-6s/episode)"

# 清空之前的日志
> "$LOG_FILE"

# 构建高性能转换命令
CONVERT_CMD="cd /dev/shm/lmc/openpi/ac_training && /era-ai/conda_envs/openpi/bin/uv run python data/acrlpd_data_converter.py --input-dir /era-ai/lm/dataset/lmc/fold_box_unified --repo-id fold_box_unified --task fold_box_unified --resume"

echo "🔧 转换命令:"
echo "   $CONVERT_CMD"
echo ""

# 显示当前进度（如果有的话）
PROGRESS_FILE="/era-ai/lm/dataset/huggingface_cache/lerobot/fold_box_unified/conversion_progress.json"
if [ -f "$PROGRESS_FILE" ]; then
    EXISTING_COUNT=$(wc -l < "$PROGRESS_FILE" 2>/dev/null || echo "0")
    echo "📊 当前进度: 已转换约 $EXISTING_COUNT 个episodes"
    echo "💡 启用断点续传，将从现有进度继续"
else
    echo "📊 首次转换，将处理所有episodes"
fi

echo ""
echo "⏱️  转换开始时间: $(date)"

# 后台运行转换命令
nohup bash -c "$CONVERT_CMD" > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!

echo "✅ 高性能转换已启动!"
echo "📋 进程信息:"
echo "   - 转换器: acrlpd_data_converter.py (基于OpenPI优化)"
echo "   - 进程ID: $PID"
echo "   - 日志文件: $LOG_FILE"
echo "   - 并行配置: 10进程, 5线程"
echo ""
echo "📊 监控进度:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🔍 检查进程状态:"
echo "   ps aux | grep $PID"
echo ""
echo "⏹️  停止转换:"
echo "   kill $PID"

# 保存进程ID
echo $PID > /tmp/fast_conversion.pid
echo "💾 进程ID已保存到: /tmp/fast_conversion.pid"

echo ""
echo "🎯 预期效果:"
echo "   - 速度: ~4-6秒/episode (vs 原来180秒/episode)"
echo "   - 总时间: ~20-40分钟完成339episodes (vs 原来17小时)"
echo "   - 兼容: 支持chunk数据 + reward字段 + Q-chunking训练"

echo ""
echo "🚀 高性能转换正在后台运行!"
echo "📅 启动完成: $(date)"

# 简单进度监控 (可选)
if command -v watch >/dev/null 2>&1; then
    echo ""
    echo "💡 提示: 可以使用以下命令实时监控："
    echo "   watch 'tail -10 $LOG_FILE'"
fi
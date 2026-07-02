!/bin/bash

# =============================================================================
# 配置部分 - 请根据您的实际路径修改以下变量
# =============================================================================
CONDA_BASE="/root/anaconda3"                    # conda 安装基础路径
ENV_NAME="nora_train"                          # 环境名称
WORK_DIR="/path/to/directory/training"
CONFIG_FILE="/path/to/directory/training/accelerator_config.yaml"

# =============================================================================
# 脚本主体
# =============================================================================

# 检查参数
if [ $# -lt 2 ]; then
  echo "Usage: $0 <data_mix> <model_path> [additional_args...]"
  echo "Example: $0 libero_10_no_noops /path/to/directory/checkpoint/extracted_model/task_agnostic/steps_10000"
  echo "Example: $0 custom_dataset /path/to/model --per_device_batch_size 8"
  exit 1
fi

DATA_MIX="$1"
MODEL_PATH="$2"
shift 2
ADDITIONAL_ARGS="$@"

# 从 model_path 的最后一级目录中提取数字，例如 steps_10000 -> 10000
MODEL_BASENAME="$(basename "$MODEL_PATH")"
CHECKPOINT_NUM="$(echo "$MODEL_BASENAME" | grep -oE '[0-9]+' | tail -n1)"

if [ -z "$CHECKPOINT_NUM" ]; then
  echo "Warning: 未在模型路径最后一级目录名中发现数字: $MODEL_BASENAME，将使用 'unknown'"
  CHECKPOINT_NUM="unknown"
fi

OUTPUT_DIR="/path/to/directory/checkpoint/TAA_fractal_stage1/$DATA_MIX/checkpoint_$CHECKPOINT_NUM"

echo "=========================================="
echo "Starting training with dataset: $DATA_MIX"
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
if [ ! -z "$ADDITIONAL_ARGS" ]; then
  echo "Additional arguments: $ADDITIONAL_ARGS"
fi
echo "=========================================="

# 切换到工作目录
cd "$WORK_DIR" || {
  echo "Error: Cannot change to work directory $WORK_DIR"
  exit 1
}

# 解析 Python/Accelerate
PYTHON_BIN="$(command -v python)"
ACCELERATE_BIN="$(command -v accelerate)"
if [ -n "$ACCELERATE_BIN" ]; then
  LAUNCH_CMD="$ACCELERATE_BIN launch"
else
  LAUNCH_CMD="$PYTHON_BIN -m accelerate.commands.launch"
fi

export WANDB_API_KEY=''
export WANDB_MODE=offline

# WANDB offline 时跳过登录
if [ "$WANDB_MODE" = "offline" ]; then
  echo "WANDB_MODE=offline, skip wandb login"
else
  if command -v wandb >/dev/null 2>&1; then
    wandb login
  fi
fi

# 创建输出目录（如不存在）
mkdir -p "$OUTPUT_DIR"

# 启动训练
/root/anaconda3/envs/nora_train/bin/python -m accelerate.commands.launch \
  --config_file="$CONFIG_FILE" \
  train_random_action.py \
  --data_mix "$DATA_MIX" \
  --output_dir "$OUTPUT_DIR" \
  --model_path "$MODEL_PATH" \
  $ADDITIONAL_ARGS

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
  echo "=========================================="
  echo "Training completed successfully!"
  echo "=========================================="
else
  echo "=========================================="
  echo "Training failed with exit code: $?"
  echo "=========================================="
  exit 1
fi

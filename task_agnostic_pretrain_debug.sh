#!/bin/bash

# 切换到工作目录
cd /path/to/directory

# 初始化 conda 环境（重要！）
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate nora_train

export WANDB_API_KEY=''
export WANDB_MODE=offline

wandb login

# 安装 accelerate（第一次运行时需要安装）
# pip install accelerate

# 启动训练
accelerate launch train_random_action.py --config_file='/path/to/directory/training/accelerator_config.yaml'
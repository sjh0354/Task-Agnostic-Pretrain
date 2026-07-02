#!/bin/bash

# ================= 配置区域 =================
# 模型名称
export MODEL_NAME="nora"
# Action Ensemble
export ACTION_ENSEMBLE_TEMP=-0.8

# 定义每个GPU上的并发数 (如果显存不够，改小这里，比如 4)
MAX_JOBS_PER_GPU=8
# 可用GPU
GPU_IDS=(0 1 2 3 4 5 6 7)

# 基础路径 (请确保这是准确的)
ROOT_SEARCH_PATH="/path/to/your/ckpt"

# 指定实验目录 (对应你 ls 看到的文件夹名)
EXPERIMENT_DIRS=("folder_name")

# ================= 核心处理函数 =================
run_single_ckpt() {
    # --- 这里重新定义 TASKS，因为数组无法 export 到子进程 ---
    local TASKS=("bridge.sh") 
    
    local ckpt_path=$1
    local device_id=$2
    
    # 激活对应显卡
    export CUDA_VISIBLE_DEVICES=$device_id
    
    # 路径解析
    # 假设路径是 .../TAA_stage2_origin/checkpoint_10000/step_10000
    # dirname(ckpt_path) -> .../checkpoint_10000
    # dirname(dirname(...)) -> .../TAA_stage2_origin
    
    exp_name=$(basename $(dirname $(dirname $ckpt_path)))
    ckpt_folder=$(basename $(dirname $ckpt_path))
    step_folder=$(basename $ckpt_path)
    
    # 构建结果目录
    logging_dir="results/${exp_name}/${ckpt_folder}/${step_folder}${ACTION_ENSEMBLE_TEMP}"
    log_subdir="${logging_dir}/logs"
    
    mkdir -p "$logging_dir"
    mkdir -p "$log_subdir"

    # 打印调试信息，确认进来了
    echo "▶ [GPU ${device_id}] Start: ${exp_name}/${ckpt_folder}/${step_folder}"

    # 1. 运行评测任务
    for task in "${TASKS[@]}"; do
        task_name=$(basename "$task" .sh)
        log_file="${log_subdir}/${task_name}.log"
        
        echo "   -> Running $task ..."
        echo "=== Started at $(date) ===" > "$log_file"
        
        # 执行脚本，如果 scripts/ 下没有这个文件，这里会报错
        if [ -f "scripts/$task" ]; then
            bash scripts/$task "$ckpt_path" "$MODEL_NAME" "$ACTION_ENSEMBLE_TEMP" "$logging_dir" "$device_id" \
                >> "$log_file" 2>&1
        else
            echo "Error: scripts/$task not found!" >> "$log_file"
            echo "❌ Error: scripts/$task missing"
        fi
            
        echo -e "\n=== Finished at $(date) ===" >> "$log_file"
    done

    # 2. 计算 Metrics
    # echo "   -> Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root "$logging_dir" \
        >> "$logging_dir/total.metrics" \
        2>> "${log_subdir}/calc_metrics.log"
        
    echo "✅ [GPU ${device_id}] Done: ${step_folder}"
}

export -f run_single_ckpt
# 导出普通变量 (数组不能 export)
export MODEL_NAME
export ACTION_ENSEMBLE_TEMP

# ================= 主流程 =================

echo "📦 Checking environment..."
# 检查 scripts 目录是否存在
if [ ! -d "scripts" ]; then
    echo "❌ Error: Current directory does not have a 'scripts' folder."
    exit 1
fi

echo "🔍 Scanning checkpoints..."
ALL_CKPTS=()

# 扫描逻辑：TAA_stage2_origin -> checkpoint_xxx -> step_xxx
for exp in "${EXPERIMENT_DIRS[@]}"; do
    search_path="${ROOT_SEARCH_PATH}/${exp}"
    if [ -d "$search_path" ]; then
        # 在 exp 目录下找所有的 step_*
        # -path "*/checkpoint_*/step_*" 确保层级正确
        while IFS= read -r line; do
            ALL_CKPTS+=("$line")
        done < <(find "$search_path" -type d -name "step_*" | sort)
    else
        echo "⚠️ Warning: Directory $search_path not found, skipping."
    fi
done

TOTAL_CKPTS=${#ALL_CKPTS[@]}
echo "📊 Found $TOTAL_CKPTS checkpoints."

if [ $TOTAL_CKPTS -eq 0 ]; then
    echo "❌ Error: No checkpoints found!"
    echo "   Check path: $ROOT_SEARCH_PATH"
    exit 1
fi

# 建立临时任务列表
rm -rf .queue_tmp
mkdir -p .queue_tmp

for ((i=0; i<TOTAL_CKPTS; i++)); do
    ckpt="${ALL_CKPTS[$i]}"
    gpu_idx=$((i % ${#GPU_IDS[@]}))
    target_gpu=${GPU_IDS[$gpu_idx]}
    echo "$ckpt" >> ".queue_tmp/gpu_${target_gpu}.list"
done

echo "🚀 Launching parallel tasks..."

pids=()
for gpu in "${GPU_IDS[@]}"; do
    list_file=".queue_tmp/gpu_${gpu}.list"
    if [ -f "$list_file" ]; then
        count=$(wc -l < "$list_file")
        echo "   -> GPU $gpu: $count tasks"
        # 启动后台进程
        (
            cat "$list_file" | xargs -n 1 -P "$MAX_JOBS_PER_GPU" -I {} bash -c "run_single_ckpt '{}' '$gpu'"
        ) &
        pids+=($!)
    fi
done

echo "⏳ Waiting for tasks to finish..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

rm -rf .queue_tmp
echo "🎉 All Done."
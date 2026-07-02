import numpy as np
import json
import os

# ================= 配置区域 =================
# 给你的数据集起个名字，这个就是后续代码里的 unnorm_key
DATASET_NAME = "my_custom_dataset" 
OUTPUT_FILE = "norm_stats.json"

# 这里模拟加载你的数据
# 你需要修改这就部分代码，把你的真实动作数据读取进来
def load_all_actions():
    """
    TODO: 修改这里以读取你的真实数据
    目标：返回一个 shape 为 (N, 7) 的 numpy 数组
    其中 N 是所有 episode 的总步数之和，7 是动作维度 (x,y,z, rx,ry,rz, gripper)
    """
    print("正在加载数据... (请确保你修改了 load_all_actions 函数)")
    
    # --- 示例：如果你是 .npy 文件 ---
    # actions = np.load("path/to/your/all_actions.npy")
    
    # --- 示例：如果你是很多 .h5 文件 ---
    # all_actions = []
    # for file in h5_files:
    #     data = h5py.File(file, 'r')
    #     all_actions.append(data['action'][:]) # 假设 key 叫 'action'
    # actions = np.concatenate(all_actions, axis=0)
    
    # --- 临时模拟数据 (不要直接用这个！) ---
    # 假设有 1000 条数据，7维
    # 这里的模拟数据范围：位置 -1~1, 旋转 -3.14~3.14, 夹爪 0/1
    actions = np.random.randn(1000, 7) 
    
    return actions

# ================= 计算逻辑 =================
def compute_stats():
    # 1. 获取所有动作
    actions = load_all_actions()
    print(f"数据加载完毕，总步数: {actions.shape[0]}, 动作维度: {actions.shape[1]}")

    # 2. 计算统计值
    # Nora/OpenVLA 核心依赖 q01 (1%) 和 q99 (99%)
    q01 = np.quantile(actions, 0.01, axis=0)
    q99 = np.quantile(actions, 0.99, axis=0)
    
    # 计算其他常用值（虽然推理时不一定用，但为了格式完整建议保留）
    max_val = np.max(actions, axis=0)
    min_val = np.min(actions, axis=0)
    mean_val = np.mean(actions, axis=0)
    std_val = np.std(actions, axis=0)

    # 3. 构造 JSON 结构
    # 结构必须是: { "dataset_name": { "action": { "q01": [...], "q99": [...] } } }
    stats = {
        DATASET_NAME: {
            "action": {
                "q01": q01.tolist(),
                "q99": q99.tolist(),
                "max": max_val.tolist(),
                "min": min_val.tolist(),
                "mean": mean_val.tolist(),
                "std": std_val.tolist(),
                # mask 通常设为全 1，表示所有维度都进行归一化
                "mask": [1.0] * actions.shape[1] 
            }
        }
    }

    # 4. 保存
    with open(OUTPUT_FILE, "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"统计完成！文件已保存为: {OUTPUT_FILE}")
    print(f"Key 名称为: {DATASET_NAME}")
    print("请检查 q01 和 q99 的数值是否符合你的物理直觉。")

if __name__ == "__main__":
    compute_stats()
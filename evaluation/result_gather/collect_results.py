import os
import glob
import pandas as pd
import re

# 1. 定义需要遍历的根目录列表
root_paths = [
    "path1","path2"
]

# 2. 定义 Partial 和 Entire 的字段
partial_keys = [
    'put_spoon_on_tablecloth/matching_partial',
    'put_carrot_on_plate/matching_partial',
    'stack_green_block_on_yellow_block/matching_partial',
    'put_eggplant_in_basket/matching_partial'
]

entire_keys = [
    'put_spoon_on_tablecloth/matching_entire',
    'put_carrot_on_plate/matching_entire',
    'stack_green_block_on_yellow_block/matching_entire',
    'put_eggplant_in_basket/matching_entire'
]

# 3. 映射关系
key_to_name = {
    'put_spoon_on_tablecloth/matching_partial': 'Spoon on cloth partial',
    'put_spoon_on_tablecloth/matching_entire': 'Spoon on cloth entire',
    'put_carrot_on_plate/matching_partial': 'Carrot on plate partial',
    'put_carrot_on_plate/matching_entire': 'Carrot on plate entire',
    'stack_green_block_on_yellow_block/matching_partial': 'Stack partial',
    'stack_green_block_on_yellow_block/matching_entire': 'Stack entire',
    'put_eggplant_in_basket/matching_partial': 'Eggplant in basket partial',
    'put_eggplant_in_basket/matching_entire': 'Eggplant in basket entire'
}

def extract_number(text):
    match = re.search(r'\d+', text)
    return int(match.group()) if match else 0

def format_percent(val):
    try:
        return f"{float(val):.2%}"
    except (ValueError, TypeError):
        return "N/A"

def process_results():
    all_records = []
    print("开始扫描并计算...")
    
    for root_path in root_paths:
        experiment_name = os.path.basename(root_path.rstrip('/'))
        pattern = os.path.join(root_path, "checkpoint_*", "step_*", "results.csv")
        csv_files = glob.glob(pattern)
        
        for file_path in csv_files:
            try:
                parts = file_path.split(os.sep)
                step_dir = parts[-2]
                ckpt_dir = parts[-3]
                
                ckpt_num = extract_number(ckpt_dir)
                step_num = extract_number(step_dir)
                
                stage1_label = f"{ckpt_num // 1000}k"
                stage2_label = f"{step_num // 1000}k step"
                
                df = pd.read_csv(file_path)
                if df.empty: continue
                
                row_data = df.iloc[0]
                
                # 计算均分
                partial_vals = [float(row_data.get(k, 0.0)) for k in partial_keys]
                entire_vals = [float(row_data.get(k, 0.0)) for k in entire_keys]
                
                avg_partial = sum(partial_vals) / len(partial_vals) if partial_vals else 0.0
                avg_entire = sum(entire_vals) / len(entire_vals) if entire_vals else 0.0
                all_vals = partial_vals + entire_vals
                avg_all = sum(all_vals) / len(all_vals) if all_vals else 0.0
                
                record = {
                    "Experiment": experiment_name,
                    "stage 1": stage1_label,
                    "stage 2": stage2_label,
                    "_sort_ckpt": ckpt_num,
                    "_sort_step": step_num
                }
                
                for k, v in key_to_name.items():
                    record[v] = format_percent(row_data.get(k, 0.0))
                
                record["Avg-partial"] = format_percent(avg_partial)
                record["Avg-entire"] = format_percent(avg_entire)
                record["Avg-all"] = format_percent(avg_all)
                
                all_records.append(record)
            except Exception as e:
                pass

    if not all_records:
        print("未找到数据")
        return

    # 1. 先创建 DataFrame 并进行标准排序
    df = pd.DataFrame(all_records)
    df_sorted = df.sort_values(by=["Experiment", "_sort_ckpt", "_sort_step"])
    
    # 2. 【核心修改】遍历排序后的数据，插入空行
    final_rows = []
    prev_key = None # 用于记录上一行的 (Experiment, stage 1) 组合
    
    # 想要的列顺序
    base_cols = ["Experiment", "stage 1", "stage 2"]
    metric_cols = list(key_to_name.values())
    avg_cols = ["Avg-partial", "Avg-entire", "Avg-all"]
    final_cols = base_cols + metric_cols + avg_cols

    for _, row in df_sorted.iterrows():
        # 当前行的 key：由 实验名 和 stage1 组成
        # 这样无论是换了实验，还是同一个实验内换了 stage 1 (如 20k -> 40k)，都会触发
        current_key = (row['Experiment'], row['stage 1'])
        
        # 如果不是第一行，且 Key 发生了变化，就插一行空行
        if prev_key is not None and current_key != prev_key:
            empty_row = {col: "" for col in final_cols} # 生成全空字典
            final_rows.append(empty_row)
            
        final_rows.append(row.to_dict())
        prev_key = current_key

    # 3. 生成最终 DataFrame
    df_final = pd.DataFrame(final_rows)
    
    # 确保列顺序正确
    df_final = df_final[final_cols]
    
    # 保存
    output_file = "aggregated_results_with_avg.csv"
    df_final.to_csv(output_file, index=False)
    
    print(f"统计完成！已保存至: {output_file}")
    print("预览前15行（包含空行效果）:")
    # 打印预览时将 NaN 显示为空白，方便查看效果
    print(df_final.head(15).fillna("").to_string(index=False))

if __name__ == "__main__":
    process_results()
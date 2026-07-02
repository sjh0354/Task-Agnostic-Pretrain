from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
import torch
import numpy as np

model_path = "/path/to/model/Qwen2.5-VL-3B-Instruct"

# 使用正确的模型类加载
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

num_new_tokens = 2048
new_tokens = [f"<robot_action_{i}>" for i in range(num_new_tokens)]

# 添加新token前记录原始信息
original_vocab_size = len(tokenizer)
print(f"原始词汇表大小: {original_vocab_size}")

# 添加新token
num_added = tokenizer.add_tokens(new_tokens)
new_vocab_size = len(tokenizer)
print(f"成功添加 {num_added} 个新token (共 {new_vocab_size} 个token)")

# 调整嵌入层前记录原始lm_head权重
if hasattr(model, 'lm_head'):
    print("记录原始lm_head权重...")
    original_lm_head_weight = model.lm_head.weight.data.clone()
    if model.lm_head.bias is not None:
        original_lm_head_bias = model.lm_head.bias.data.clone()

# 调整嵌入层
print("调整嵌入层大小...")
model.resize_token_embeddings(new_vocab_size)

# 更新模型配置
model.config.vocab_size = new_vocab_size
print(f"更新后模型配置词汇表大小: {model.config.vocab_size}")

# 手动修复lm_head权重（如果需要）
if hasattr(model, 'lm_head'):
    print("修复lm_head权重...")
    current_weight = model.lm_head.weight.data
    
    # 检查是否需要修复
    if current_weight.size(0) > original_vocab_size + num_added:
        # 计算需要复制的原始部分大小
        copy_size = min(original_vocab_size, current_weight.size(0) - num_added)
        
        # 复制原始权重
        model.lm_head.weight.data[:copy_size] = original_lm_head_weight[:copy_size]
        
        # 对新token部分进行特殊初始化
        model.lm_head.weight.data[copy_size:copy_size+num_added] = torch.normal(
            mean=0.0,
            std=model.config.initializer_range,
            size=(num_added, model.config.hidden_size),
            device=model.device,
            dtype=model.dtype
        )
        
        # 如果有bias
        if model.lm_head.bias is not None:
            model.lm_head.bias.data[:copy_size] = original_lm_head_bias[:copy_size]
            model.lm_head.bias.data[copy_size:copy_size+num_added] = 0.0
    
    print(f"lm_head权重形状: {model.lm_head.weight.shape}")

save_path = "./Qwen2.5-VL-3B-added-action-tokens"

# 保存模型和tokenizer
model.save_pretrained(
    save_path,
    safe_serialization=True,
    max_shard_size="2GB"
)
tokenizer.save_pretrained(save_path)

# 验证
print("\n最终验证:")
print(f"Tokenizer词汇量: {len(tokenizer)}")
print(f"模型配置vocab_size: {model.config.vocab_size}")
print(f"输入嵌入层大小: {model.get_input_embeddings().weight.size(0)}")
print(f"输出嵌入层大小: {model.get_output_embeddings().weight.size(0)}")

# 测试新token
test_token = "<robot_action_0>"
test_id = tokenizer.convert_tokens_to_ids(test_token)
print(f"测试token '{test_token}' 的ID: {test_id}")

if test_id < original_vocab_size:
    print(f"警告: 测试token ID {test_id} 在原始词汇表范围内!")
else:
    print(f"测试token ID 有效: {test_id} (应在 {original_vocab_size} 到 {new_vocab_size-1} 之间)")
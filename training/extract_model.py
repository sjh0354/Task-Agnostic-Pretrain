import os
import torch
import json
import shutil
import logging
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformers import Qwen2_5_VLForConditionalGeneration

'''
python extract_model.py     --checkpoint_path "/inspire/hdd/global_user/gongjingjing-25039/jhshi/nora/checkpoint/libero_libero_spatial_no_noops/libero_spatial_no_noops/steps_90000"     --base_model_path "/inspire/hdd/global_user/gongjingjing-25039/jhshi/nora/download_models/Qwen2.5-VL-3B-added-action-tokens"     --output_path "/inspire/hdd/global_user/gongjingjing-25039/jhshi/nora/checkpoint/extracted_model/libero_spatial/steps_90000"   
'''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_model_directly(
    checkpoint_path: str,
    base_model_path: str,
    output_path: str
):
    """
    直接从checkpoint文件中提取模型权重，绕过accelerate的load_state
    """
    
    checkpoint_path = Path(checkpoint_path)
    base_model_path = Path(base_model_path)
    output_path = Path(output_path)
    
    logger.info(f"Checkpoint directory: {checkpoint_path}")
    logger.info(f"Base model directory: {base_model_path}")
    logger.info(f"Output directory: {output_path}")
    
    # 检查checkpoint目录内容
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info("Contents of checkpoint directory:")
    for item in checkpoint_path.iterdir():
        logger.info(f"  {item.name}")
    
    # 寻找模型权重文件
    model_file = None
    potential_model_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "model_0.safetensors",  # 可能的分片文件
    ]
    
    for filename in potential_model_files:
        file_path = checkpoint_path / filename
        if file_path.exists():
            model_file = file_path
            logger.info(f"Found model file: {model_file}")
            break
    
    if model_file is None:
        # 查找所有可能的权重文件
        safetensor_files = list(checkpoint_path.glob("*.safetensors"))
        bin_files = list(checkpoint_path.glob("*.bin"))
        
        logger.info(f"Found .safetensors files: {[f.name for f in safetensor_files]}")
        logger.info(f"Found .bin files: {[f.name for f in bin_files]}")
        
        if safetensor_files:
            model_file = safetensor_files[0]  # 使用第一个找到的文件
        elif bin_files:
            model_file = bin_files[0]
        else:
            raise FileNotFoundError("No model weight files found in checkpoint")
    
    logger.info(f"Using model file: {model_file}")
    
    # 加载模型权重
    try:
        if model_file.suffix == '.safetensors':
            logger.info("Loading safetensors file...")
            state_dict = load_file(str(model_file))
        elif model_file.suffix == '.bin':
            logger.info("Loading PyTorch bin file...")
            state_dict = torch.load(str(model_file), map_location='cpu')
        else:
            raise ValueError(f"Unsupported file format: {model_file.suffix}")
        
        logger.info(f"Loaded {len(state_dict)} parameters from checkpoint")
        
        # 打印一些权重信息用于验证
        for i, (key, value) in enumerate(state_dict.items()):
            logger.info(f"  {key}: {value.shape}")
            if i >= 5:  # 只显示前几个
                logger.info(f"  ... and {len(state_dict) - 6} more parameters")
                break
                
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存提取的模型权重
    output_model_file = output_path / "model.safetensors"
    logger.info(f"Saving extracted weights to: {output_model_file}")
    save_file(state_dict, str(output_model_file))
    
    # 复制配置文件
    config_files = [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"
    ]
    
    logger.info("Copying configuration files...")
    for config_file in config_files:
        src = base_model_path / config_file
        dst = output_path / config_file
        
        if src.exists():
            shutil.copy2(str(src), str(dst))
            logger.info(f"  Copied: {config_file}")
        else:
            logger.warning(f"  Not found: {config_file}")
    
    # 创建简化的model index文件
    index_data = {
        "metadata": {"total_size": sum(p.numel() * p.element_size() for p in state_dict.values())},
        "weight_map": {key: "model.safetensors" for key in state_dict.keys()}
    }
    
    index_file = output_path / "model.safetensors.index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Model extraction completed successfully!")
    logger.info(f"Output saved to: {output_path}")
    
    return str(output_path)

def verify_extracted_model(model_path: str, base_model_path: str):
    """验证提取的模型是否可以正常加载"""
    try:
        logger.info(f"Verifying extracted model: {model_path}")
        
        # 尝试加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        
        logger.info("✓ Model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model config: {model.config}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load extracted model: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract model weights directly from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                      help="Path to checkpoint directory")
    parser.add_argument("--base_model_path", type=str, required=True,
                      help="Path to base model directory")
    parser.add_argument("--output_path", type=str, required=True,
                      help="Output directory to save extracted model")
    parser.add_argument("--verify", action="store_true",
                      help="Verify the extracted model can be loaded")
    
    args = parser.parse_args()
    
    # 提取模型
    output_path = extract_model_directly(
        checkpoint_path=args.checkpoint_path,
        base_model_path=args.base_model_path,
        output_path=args.output_path
    )
    
    # 验证模型（如果请求）
    if args.verify:
        verify_extracted_model(output_path, args.base_model_path)

if __name__ == "__main__":
    main()
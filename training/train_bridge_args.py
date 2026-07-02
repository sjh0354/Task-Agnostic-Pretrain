import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from transformers import SchedulerType, get_scheduler
from datasets import RLDSDataset, RLDSBatchTransform
from torch.utils.data import Subset
from qwen_vl_utils import process_vision_info
import math
import numpy as np
from tqdm import tqdm
import wandb

# --- 1. Configuration ---
class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 16,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 1,
        num_warmup_steps: int = 1500,
        max_train_steps: int = 100000,
        output_dir: str = './checkpoints',
        resume_from_checkpoint: str = '',
        load_model_weights: Optional[str] = None,
        data_root_dir: str = "/path/to/your/data",
        data_mix: str = "bridge_orig", #"libero_10_no_noops", ## For this, please check out the data mix in /training/datasets/rlds/oxe/mixtures.py
        resize_resolution: tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 32_000,
        wandb_project_name: str = "Nora VLA",
        checkpoint_save_frequency: int = 10000,
        logging_frequency: int = 100,
        gradient_clipping: Optional[float] = None, # Add gradient clipping option
        model_path: Optional[str] = "/path/to/your/model",
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint ## This is used to continue a training by loadinng the optimizer states, model weights etc ...
        self.load_model_weights = load_model_weights ## This is the path to a pretrained model weights if you want to finetune the model.
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping
        self.model_path = model_path

    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        config = cls()
        
        # Override default values with command line arguments
        if args.data_mix:
            config.data_mix = args.data_mix
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.data_root_dir:
            config.data_root_dir = args.data_root_dir
        if args.per_device_batch_size:
            config.per_device_batch_size = args.per_device_batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        if args.max_train_steps:
            config.max_train_steps = args.max_train_steps
        if args.resume_from_checkpoint:
            config.resume_from_checkpoint = args.resume_from_checkpoint
        if args.load_model_weights:
            config.load_model_weights = args.load_model_weights
        if args.wandb_project_name:
            config.wandb_project_name = args.wandb_project_name
        if args.gradient_clipping:
            config.gradient_clipping = args.gradient_clipping
        if args.model_path:
            config.model_path = args.model_path      
        return config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train VLA model with configurable datasets")
    
    # Dataset related arguments
    parser.add_argument(
        "--data_mix", 
        type=str, 
        default=None,
        help="Dataset mix name (e.g., 'libero_10_no_noops', 'your_dataset_name')"
    )
    parser.add_argument(
        "--data_root_dir", 
        type=str, 
        default=None,
        help="Root directory containing the datasets"
    )
    
    # Training related arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save checkpoints and outputs"
    )
    parser.add_argument(
        "--per_device_batch_size", 
        type=int, 
        default=None,
        help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_train_steps", 
        type=int, 
        default=None,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--load_model_weights", 
        type=str, 
        default=None,
        help="Path to pretrained model weights"
    )
    parser.add_argument(
        "--wandb_project_name", 
        type=str, 
        default=None,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--gradient_clipping", 
        type=float, 
        default=None,
        help="Gradient clipping value"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path or HF Hub id for base model (Qwen2.5-VL) to load via from_pretrained"
    )
    
    return parser.parse_args()

class StepTracker:
    def __init__(self):
        self.completed_steps = 0
        self.total_loss_since_ckpt = 0.0
    
    def state_dict(self):
        return {
            "completed_steps": self.completed_steps,
            "total_loss_since_ckpt": self.total_loss_since_ckpt,
        }
    
    def load_state_dict(self, state):
        self.completed_steps = int(state.get("completed_steps", 0))
        self.total_loss_since_ckpt = float(state.get("total_loss_since_ckpt", 0.0))
        
        
# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_dataset(config: TrainingConfig, processor: AutoProcessor, is_train: bool = True) -> RLDSDataset:
    """Loads and prepares the RLDS dataset."""
    return RLDSDataset(
        data_root_dir=Path(config.data_root_dir),
        data_mix=config.data_mix,
        batch_transform=RLDSBatchTransform(),
        resize_resolution=config.resize_resolution,
        shuffle_buffer_size=config.shuffle_buffer_size if is_train else None,
        train=is_train,
    )

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format.
    Action token 0 is mapped to the string <robot_action_0>  ... and so on
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def process_example(example: Dict[str, Any], fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    """Processes a single example from the dataset."""
    pixel_values = example['image']
    action = example['action']
    lang = example['lang']
    fast_tokens = fast_tokenizer(action)
    vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pixel_values},
                {"type": "text", "text": lang},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": vlm_action},
            ],
        },
    ]
    return messages

def collate_fn(examples, processor, fast_tokenizer):
    messages = [process_example(example, fast_tokenizer) for example in examples]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    batch_input = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    action_token_min = 151665
    action_token_max = 153712
    labels = batch_input['input_ids'].clone()
    # For each sequence in the batch, find the first occurrence of an action token.
    for i in range(labels.size(0)):
        seq = labels[i]
        # Create a mask for tokens within the action token range.
        mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
        nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
        if nonzero_indices.numel() > 0:
            first_action_index = nonzero_indices[0].item()
            # Mask out all tokens before the first action token.
            seq[:first_action_index] = -100
        else:
            # If no action token is found, mask the entire sequence.
            seq[:] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
    batch_input['labels'] = labels
    return batch_input

# --- 3. Model Initialization ---
def load_model_and_processor(config: TrainingConfig, accelerator: Accelerator) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    processor = AutoProcessor.from_pretrained('/path/to/model')
    processor.tokenizer.padding_side = 'left'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    fast_tokenizer = AutoProcessor.from_pretrained(
        "/path/to/model/pi_fast", trust_remote_code=True
    )

    if config.load_model_weights:
        tensors = {}
        from safetensors import safe_open
        with safe_open(config.load_model_weights, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        accelerator.print("Pretrained weights loaded.")

    return model, processor, fast_tokenizer

# --- 4. Training Loop ---
def train(config: TrainingConfig):
    """Main training loop."""
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    accelerator.dataloader_config.dispatch_batches = False
    
    set_seed(42, device_specific=True)
    
    # Initialize the logger AFTER accelerator is created
    logger = get_logger(__name__)
    
    logger.info(accelerator.state, main_process_only=False)

    # Log the configuration
    if accelerator.is_main_process:
        logger.info("***** Training Configuration *****")
        logger.info(f"  Model path: {config.model_path}")
        logger.info(f"  Dataset: {config.data_mix}")
        logger.info(f"  Data root directory: {config.data_root_dir}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Batch size: {config.per_device_batch_size}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Max steps: {config.max_train_steps}")

    # Initialize Weights and Biases
    if accelerator.is_main_process:
        # Include dataset name in wandb run name
        run_name = f"{config.data_mix}_{config.max_train_steps}steps"
        wandb.init(
            project=config.wandb_project_name,
            name=run_name,
            config=vars(config)  # Log all config parameters
        )

    # Load model and processor
    model, processor, fast_tokenizer = load_model_and_processor(config, accelerator)

    # Load and prepare dataset
    with accelerator.main_process_first():
        train_dataset = load_and_prepare_dataset(config, processor, is_train=True)

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: collate_fn(examples, processor, fast_tokenizer)
    )
    
    skip_batches = 1000 
    train_dataloader = accelerator.skip_first_batches(train_dataloader, skip_batches)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    # Initialize learning rate scheduler
    max_train_steps = config.max_train_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=max_train_steps
    )
    
    tracker = StepTracker()

    accelerator.register_for_checkpointing(lr_scheduler)
    accelerator.register_for_checkpointing(tracker)

    # Prepare everything with Accelerator
    accelerator.even_batches=False
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )


    # Resume from checkpoint if provided
    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {config.resume_from_checkpoint}")

    # Training loop
    # Right now we assume single node training. I did not test on multi node training.
    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Dataset: {config.data_mix}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num steps = {config.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # tracker.completed_steps = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    # total_loss = 0.0
    progress_bar.update(tracker.completed_steps)

    while tracker.completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss

                # 只在同步梯度时累积损失（即完成一个完整的优化步骤时）
                if accelerator.sync_gradients:
                    tracker.total_loss_since_ckpt += loss.detach().float()
                
                accelerator.backward(loss)
                
                if config.gradient_clipping is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    tracker.completed_steps += 1
                
                optimizer.step()
                lr_scheduler.step()

                # Logging
                if tracker.completed_steps % config.logging_frequency == 0:
                    if accelerator.is_main_process:
                        total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm**0.5
                        lr = lr_scheduler.get_last_lr()[0]
                        logger.info(f"Step {tracker.completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}")
                        result = {
                            "train_loss": loss.item(),
                            "grad_norm": total_norm,
                            "learning_rate": lr,
                            "dataset": config.data_mix,  # Include dataset name in logs
                        }
                        wandb.log({
                            "train_loss": loss.item(), 
                            "learning_rate": lr,
                            "grad_norm": total_norm,
                            "dataset": config.data_mix
                        }, step=tracker.completed_steps)

                # Checkpointing
                if tracker.completed_steps % config.checkpoint_save_frequency == 0 and tracker.completed_steps > 0:
                    if accelerator.is_main_process:
                        # Create dataset-specific checkpoint directory
                        checkpoint_dir = os.path.join(config.output_dir, config.data_mix, f"steps_{tracker.completed_steps}")
                        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
                        accelerator.save_state(checkpoint_dir)

                        # 计算平均损失
                        avg_loss = tracker.total_loss_since_ckpt / config.checkpoint_save_frequency
                        
                        summary_data = {
                            "steps": int(tracker.completed_steps), 
                            "train_loss": float(avg_loss),
                            "dataset": config.data_mix
                        }
                        
                        # Save summary to dataset-specific directory
                        summary_dir = os.path.join(config.output_dir, config.data_mix)
                        os.makedirs(summary_dir, exist_ok=True)
                        with open(os.path.join(summary_dir, "summary.jsonl"), "a") as f:
                            f.write(json.dumps(summary_data) + "\n")
                        
                        logger.info(f"Checkpoint saved at step {tracker.completed_steps} for dataset {config.data_mix}")
                        tracker.total_loss_since_ckpt = 0.0

                if tracker.completed_steps >= max_train_steps:
                    break

    # Save final checkpoint
    if accelerator.is_main_process:
        final_checkpoint_dir = os.path.join(config.output_dir, config.data_mix, f"steps_{tracker.completed_steps}")
        os.makedirs(os.path.dirname(final_checkpoint_dir), exist_ok=True)
        accelerator.save_state(final_checkpoint_dir)
        logger.info(f"Training finished. Final checkpoint saved at {final_checkpoint_dir}")
        wandb.finish()

def main():
    # Parse command line arguments
    args = parse_args()
    
    print(args,flush=True)
    
    # Initialize training configuration from arguments
    config = TrainingConfig.from_args(args)
    
    # Set up basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Use standard Python logging for initial configuration logging
    # (before Accelerator is initialized)
    std_logger = logging.getLogger(__name__)
    std_logger.info("Starting training script...")
    std_logger.info(f"Dataset: {config.data_mix}")
    std_logger.info(f"Model Path: {config.model_path}")
    std_logger.info(f"Output directory: {config.output_dir}")
    
    if dist.is_initialized():
        print(f"Current backend: {dist.get_backend()}")

    # Run the training
    train(config)

if __name__ == "__main__":
    main()
    
'''
# 基本用法 - 指定数据集
accelerate launch --config_file your_config.yaml train_script.py --data_mix your_dataset_name

# 完整示例 - 指定多个参数
accelerate launch --config_file your_config.yaml train_script.py \
    --data_mix libero_10_no_noops \
    --output_dir /path/to/output \
    --per_device_batch_size 8 \
    --learning_rate 1e-5 \
    --max_train_steps 50000 \
    --wandb_project_name "My VLA Experiments"

# 从检查点恢复训练
accelerate launch --config_file your_config.yaml train_script.py \
    --data_mix custom_dataset \
    --resume_from_checkpoint /path/to/checkpoint

# 加载预训练权重进行微调
accelerate launch --config_file your_config.yaml train_script.py \
    --data_mix new_dataset \
    --load_model_weights /path/to/pretrained/weights.safetensors'''
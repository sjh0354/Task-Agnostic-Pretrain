import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import pickle
import time
import gc

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.distributed as dist

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_scheduler

from datasets import EpisodicRLDSDataset, RLDSBatchTransform
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import wandb

logger = get_logger(__name__)

# ===================== 1. 配置类 & 参数解析 =====================
class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 16,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 1,
        num_warmup_steps: int = 1000,
        max_train_steps: int = 100000,
        output_dir: str = './checkpoints',
        resume_from_checkpoint: str = '',
        load_model_weights: Optional[str] = None,
        data_root_dir="/path/to/your/data",
        data_mix="bridge_orig",
        resize_resolution: tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 32_000,
        wandb_project_name: str = "Nora VLA",
        checkpoint_save_frequency: int = 10000,
        logging_frequency: int = 100,
        gradient_clipping: Optional[float] = None,
        # 模型路径
        model_path: str = '/path/to/your/model',
        processor_path: str = '/path/to/your/model',
        fast_tokenizer_path: str = '/path/to/your/tokenizer',
        # 连续帧参数
        use_consecutive_frames: bool = True,
        consecutive_frames_seed: int = 4396,
        max_episodes_in_memory: int = 500,
        data_cache_dir: str = "./cache",
        target_max_samples: int = 5000000, # 硬截断限制
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.load_model_weights = load_model_weights
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping
        
        self.model_path = model_path
        self.processor_path = processor_path
        self.fast_tokenizer_path = fast_tokenizer_path
        
        self.use_consecutive_frames = use_consecutive_frames
        self.consecutive_frames_seed = consecutive_frames_seed
        self.max_episodes_in_memory = max_episodes_in_memory
        self.data_cache_dir = data_cache_dir
        self.target_max_samples = target_max_samples

    @classmethod
    def from_args(cls, args):
        config = cls()
        if args.data_mix: config.data_mix = args.data_mix
        if args.output_dir: config.output_dir = args.output_dir
        if args.data_root_dir: config.data_root_dir = args.data_root_dir
        if args.per_device_batch_size: config.per_device_batch_size = args.per_device_batch_size
        if args.learning_rate: config.learning_rate = args.learning_rate
        if args.max_train_steps: config.max_train_steps = args.max_train_steps
        if args.resume_from_checkpoint: config.resume_from_checkpoint = args.resume_from_checkpoint
        if args.wandb_project_name: config.wandb_project_name = args.wandb_project_name
        if args.gradient_clipping: config.gradient_clipping = args.gradient_clipping
        if args.model_path: config.model_path = args.model_path
        if args.data_cache_dir: config.data_cache_dir = args.data_cache_dir
        return config

def parse_args():
    parser = argparse.ArgumentParser(description="Train VLA model")
    parser.add_argument("--data_mix", type=str, default=None)
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--gradient_clipping", type=float, default=None)
    parser.add_argument("--data_cache_dir", type=str, default=None)
    return parser.parse_args()

class StepTracker:
    def __init__(self):
        self.completed_steps = 0
    def state_dict(self): return {"completed_steps": self.completed_steps}
    def load_state_dict(self, state): self.completed_steps = int(state.get("completed_steps", 0))

# ===================== 2. 数据集类定义 (修复 NameError) =====================

# --- 1. 最终用于训练的 Dataset 包装类 (之前报错缺少这个) ---
class PreprocessedConsecutiveFrameDataset(Dataset):
    def __init__(self, consecutive_frames):
        # 这里已经是处理好的列表，直接使用
        self.consecutive_frames = consecutive_frames
        logger.info(f"Initialized dataset with {len(self.consecutive_frames)} samples")

    def __len__(self):
        return len(self.consecutive_frames)

    def __getitem__(self, idx):
        frame_pair = self.consecutive_frames[idx]
        return {
            'image_current': frame_pair['current']['image'],
            'image_next': frame_pair['next']['image'],
            'action': frame_pair['current']['action'],
            'lang': frame_pair['current']['lang'],
            'dataset_name': frame_pair['current']['dataset_name']
        }

# --- 2. 负责原始数据读取和处理的类 (优化了内存和速度) ---
class MemoryOptimizedConsecutiveFrameDataset(Dataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        resize_resolution: tuple[int, int],
        shuffle_buffer_size: int = 32_000,
        train: bool = True,
        seed: int = 3407,
        max_episodes_in_memory: int = 500,
        target_max_samples: int = 300000, # 新增：硬截断限制
    ):
        self.shuffle_buffer_size = shuffle_buffer_size
        random.seed(seed)
        np.random.seed(seed)
        
        # 关键优化：生成阶段将 buffer 设为 1，加快启动速度，防止卡死
        creation_buffer_size = 1
        
        episodic_dataset = EpisodicRLDSDataset(
            data_root_dir=data_root_dir,
            data_mix=data_mix,
            batch_transform=RLDSBatchTransform(),
            resize_resolution=resize_resolution,
            shuffle_buffer_size=creation_buffer_size, 
            train=train,
        )
        
        # 开始处理并截断
        self.consecutive_frames = self._process_in_batches(
            episodic_dataset, 
            max_episodes_in_memory,
            target_max_samples
        )
        
        self._shuffle()

    def _process_in_batches(self, episodic_dataset, max_eps, target_max_samples):
        all_pairs = []
        batch = []
        
        logger.info(f"Start processing data. Target limit: {target_max_samples}")
        pbar = tqdm(total=target_max_samples, desc="Processing Frames")
        
        for idx, episode in enumerate(episodic_dataset):
            if len(all_pairs) >= target_max_samples:
                break
                
            batch.append((idx, episode))
            
            # 攒够一波就处理
            if len(batch) >= max_eps:
                new_pairs = self._extract_pairs(batch, len(all_pairs))
                
                # 检查是否溢出
                needed = target_max_samples - len(all_pairs)
                if len(new_pairs) > needed:
                    new_pairs = new_pairs[:needed]
                
                all_pairs.extend(new_pairs)
                pbar.update(len(new_pairs))
                
                batch.clear()
                gc.collect() # 强制回收
        
        # 处理尾巴
        if batch and len(all_pairs) < target_max_samples:
            new_pairs = self._extract_pairs(batch, len(all_pairs))
            needed = target_max_samples - len(all_pairs)
            if len(new_pairs) > needed:
                new_pairs = new_pairs[:needed]
            all_pairs.extend(new_pairs)
            pbar.update(len(new_pairs))
            
        pbar.close()
        logger.info(f"Finished processing. Total pairs: {len(all_pairs)}")
        return all_pairs

    def _extract_pairs(self, batch, current_count):
        pairs = []
        for ep_idx, episode in batch:
            if len(episode) > 1:
                for i in range(len(episode) - 1):
                    pairs.append({
                        'current': episode[i],
                        'next': episode[i + 1],
                        'episode_idx': ep_idx,
                        'frame_idx': i,
                        'original_order': current_count + len(pairs),
                    })
        return pairs

    def _shuffle(self):
        logger.info("Shuffling dataset...")
        if self.shuffle_buffer_size >= len(self.consecutive_frames):
            random.shuffle(self.consecutive_frames)
        else:
            buffer = []
            shuffled = []
            for p in self.consecutive_frames:
                buffer.append(p)
                if len(buffer) >= self.shuffle_buffer_size:
                    shuffled.append(buffer.pop(random.randint(0, len(buffer) - 1)))
            random.shuffle(buffer)
            shuffled.extend(buffer)
            self.consecutive_frames = shuffled

# ===================== 3. 缓存管理与加载 =====================

def _create_and_cache_dataset(config: TrainingConfig, cache_file: Path):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用优化后的类生成数据
    dataset = MemoryOptimizedConsecutiveFrameDataset(
        Path(config.data_root_dir),
        config.data_mix,
        config.resize_resolution,
        config.shuffle_buffer_size,
        seed=config.consecutive_frames_seed,
        max_episodes_in_memory=config.max_episodes_in_memory,
        target_max_samples=config.target_max_samples # 传递截断参数
    )
    
    logger.info(f"Saving {len(dataset.consecutive_frames)} frames to pickle cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset.consecutive_frames, f)
    
    return PreprocessedConsecutiveFrameDataset(dataset.consecutive_frames)

def load_and_prepare_dataset(config: TrainingConfig, accelerator: Accelerator):
    cache_file = Path(config.data_cache_dir) / f"frames_cache_{config.data_mix}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. 主进程负责生成并写入缓存
    with accelerator.main_process_first():
        if cache_file.exists():
            pass 
        else:
            logger.info(f"Cache not found, creating dataset for {config.data_mix}...")
            _create_and_cache_dataset(config, cache_file)
    
    # 2. 所有进程加载缓存
    logger.info("Loading cached data from pickle...")
    with open(cache_file, 'rb') as f:
        frames = pickle.load(f)
    
    dataset = PreprocessedConsecutiveFrameDataset(frames)
    return dataset

# ===================== 4. 数据处理 (Collate) =====================
def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    return ''.join([f"<robot_action_{token}>" for token in tokens])
def process_example_consecutive_frames(example: Dict[str, Any], fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    image_current = example['image_current']
    image_next = example['image_next']
    action = example['action']
    
    # === 修复开始 ===
    # 这里的 fast_tokenizer 需要 numpy array 才能检查 .ndim
    # 如果它是 list，强制转回 numpy
    if isinstance(action, list):
        action = np.array(action)
    elif isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    # 如果已经是 numpy array，保持不动
    # === 修复结束 ===
        
    fast_tokens = fast_tokenizer(action)
    vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])
    
    return [
        {"role": "user", "content": [
            {"type": "image", "image": image_current},
            {"type": "image", "image": image_next},
            {"type": "text", "text": ""},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": vlm_action},
        ]}
    ]

def collate_fn(examples, processor, fast_tokenizer):
    messages = [process_example_consecutive_frames(ex, fast_tokenizer) for ex in examples]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    
    batch_input = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    action_token_min, action_token_max = 151665, 153712
    labels = batch_input['input_ids'].clone()
    for i in range(labels.size(0)):
        seq = labels[i]
        mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
        nz = torch.nonzero(mask_seq, as_tuple=False)
        if nz.numel() > 0:
            seq[:nz[0].item()] = -100
        else:
            seq[:] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch_input['labels'] = labels
    return batch_input

# ===================== 5. 模型加载 =====================
def load_model_and_processor(config: TrainingConfig):
    logger.info(f"Loading processor from {config.processor_path}")
    processor = AutoProcessor.from_pretrained(config.processor_path)
    processor.tokenizer.padding_side = 'left'
    
    logger.info(f"Loading model from {config.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    # 显存优化
    model.gradient_checkpointing_enable()
    
    logger.info(f"Loading fast tokenizer from {config.fast_tokenizer_path}")
    fast_tokenizer = AutoProcessor.from_pretrained(config.fast_tokenizer_path, trust_remote_code=True)
    return model, processor, fast_tokenizer

# ===================== 6. 训练流程 =====================
def train(config: TrainingConfig):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    
    set_seed(config.consecutive_frames_seed, device_specific=True)
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        wandb.init(project=config.wandb_project_name, config=vars(config))

    model, processor, fast_tokenizer = load_model_and_processor(config)
    dataset = load_and_prepare_dataset(config, accelerator)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.per_device_batch_size,
        collate_fn=lambda ex: collate_fn(ex, processor, fast_tokenizer),
        num_workers=0,
        pin_memory=True,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer,
                                 num_warmup_steps=config.num_warmup_steps,
                                 num_training_steps=config.max_train_steps)
    
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    tracker = StepTracker()
    accelerator.register_for_checkpointing(tracker)

    if config.resume_from_checkpoint:
        if os.path.isdir(config.resume_from_checkpoint):
            accelerator.load_state(config.resume_from_checkpoint)
            logger.info(f"Resumed from checkpoint: {config.resume_from_checkpoint}")

    progress_bar = tqdm(range(config.max_train_steps), disable=not accelerator.is_local_main_process, initial=tracker.completed_steps)
    
    while tracker.completed_steps < config.max_train_steps:
        model.train()
        for batch in dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                if config.gradient_clipping:
                    accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                
                optimizer.step()
                lr_scheduler.step()
                
                if accelerator.sync_gradients:
                    tracker.completed_steps += 1
                    progress_bar.update(1)
                    
                    if tracker.completed_steps % config.logging_frequency == 0 and accelerator.is_main_process:
                        logger.info(f"Step {tracker.completed_steps}, Loss: {loss.item():.4f}")
                        wandb.log({"train_loss": loss.item()}, step=tracker.completed_steps)
                        
                    if tracker.completed_steps % config.checkpoint_save_frequency == 0:
                        if accelerator.is_main_process:
                            accelerator.save_state(os.path.join(config.output_dir, f"steps_{tracker.completed_steps}"))
            # ====== 新增：强制清理本轮循环的内存 ======
            # 删除大张量引用
            del outputs
            del loss
            del batch 
            
            # 定期手动触发 Python 垃圾回收 (比如每 50 步)
            if tracker.completed_steps % 50 == 0:
                gc.collect()

            if tracker.completed_steps >= config.max_train_steps:
                break

    if accelerator.is_main_process:
        accelerator.save_state(os.path.join(config.output_dir, f"steps_{tracker.completed_steps}_final"))
        wandb.finish()

def main():
    args = parse_args()
    config = TrainingConfig.from_args(args)
    
    logging.basicConfig(level=logging.INFO)
    if dist.is_initialized():
        print(f"Current backend: {dist.get_backend()}")
        
    train(config)

if __name__ == "__main__":
    main()
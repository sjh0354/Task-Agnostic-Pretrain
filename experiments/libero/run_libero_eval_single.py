"""
run_libero_eval_single.py
Runs a model in a LIBERO simulation environment for a single task and episode.
Usage:
    python experiments/robot/libero/run_libero_eval_single.py \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --task_id <TASK_ID> \
        --episode_id <EPISODE_ID> \
        --output_dir <OUTPUT_DIR>
"""
import os
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import draccus
import numpy as np
from libero.libero import benchmark
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from nora_utils import Nora

@dataclass
class GenerateConfig:
    # fmt: off
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/inspire/hdd/global_user/gongjingjing-25039/sywang/nora_icl/checkpoint/extracted_model/libero_spatial/steps_80000"
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    task_id: int = 0                                 # Task ID to evaluate
    episode_id: int = 0                              # Episode ID to evaluate
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    output_dir: str = "./episode_results"            # Output directory for individual episode results
    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on

@draccus.wrap()
def eval_libero_single(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    
    # Load model
    model = Nora(model_path=cfg.pretrained_checkpoint)
    model.fast_tokenizer.time_horizon = 6
    
    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-qwen-task{cfg.task_id}-ep{cfg.episode_id}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    
    # Get task
    task = task_suite.get_task(cfg.task_id)
    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    # Initialize LIBERO environment and task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    
    print(f"\nTask: {task_description}")
    log_file.write(f"\nTask: {task_description}\n")
    
    episode_start_time = time.time()
    
    # Reset environment
    env.reset()
    # Set initial states
    obs = env.set_init_state(initial_states[cfg.episode_id])
    
    # Setup
    t = 0
    replay_images = []
    if cfg.task_suite_name == "libero_spatial":
        max_steps = 300
    elif cfg.task_suite_name == "libero_object":
        max_steps = 300
    elif cfg.task_suite_name == "libero_goal":
        max_steps = 300
    elif cfg.task_suite_name == "libero_10":
        max_steps = 300
    elif cfg.task_suite_name == "libero_90":
        max_steps = 300
    
    print(f"Starting episode {cfg.episode_id}...")
    log_file.write(f"Starting episode {cfg.episode_id}...\n")
    
    done = False
    while t < max_steps + cfg.num_steps_wait:
        try:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            
            # Get preprocessed image
            img = get_libero_image(obs)
            # Save preprocessed image for replay video
            replay_images.append(img)
            
            # Prepare observations dict
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            
            # Query model to get action
            action = model.inference(observation["full_image"], task_description, cfg.task_suite_name)
            if action is None:
                raise ValueError("Model inference returned None.")
            
            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)
            
            # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)
            
            # Execute action in environment
            action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, action[..., -1])
            
            for i in range(len(action[:-1])):
                obs, reward, done, info = env.step(action[i].tolist())
            
            if done:
                break
            t += 1
            
        except Exception as e:
            print(f"Caught exception: {e}")
            log_file.write(f"Caught exception: {e}\n")
            break
    
    episode_end_time = time.time()
    episode_duration = episode_end_time - episode_start_time
    
    # Save a replay video of the episode
    save_rollout_video(
        replay_images, 0, success=done, task_description=task_description, log_file=log_file
    )
    
    # Log results
    print(f"Task {cfg.task_id} Episode {cfg.episode_id} Success: {done} Time: {episode_duration:.2f}s")
    log_file.write(f"Task {cfg.task_id} Episode {cfg.episode_id} Success: {done} Time: {episode_duration:.2f}s\n")
    log_file.flush()
    
    # Save result json
    result_json = {
        "task_suite": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "episode_id": cfg.episode_id,
        "success": str(done),
        "duration_sec": episode_duration,
    }
    
    # Save local log file
    log_file.close()
    
    # Save result to output directory
    result_path = os.path.join(cfg.output_dir, cfg.task_suite_name)
    os.makedirs(result_path, exist_ok=True)
    result_file = os.path.join(
        result_path, f"gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 0)}_task{cfg.task_id}_ep{cfg.episode_id}_results.json"
    )
    with open(result_file, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"Saved result to {result_file}")

if __name__ == "__main__":
    eval_libero_single()
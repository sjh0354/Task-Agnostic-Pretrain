import sys
import os

# 添加 SimplerEnv 的路径
simpler_env_path = "/inspire/hdd/global_user/gongjingjing-25039/jhshi/SimplerEnv"
if simpler_env_path not in sys.path:
    sys.path.insert(0, simpler_env_path)

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
import logging
# from sapien import disable_renderer

# disable_renderer()  # <-- 添加这一行跳过渲染器


logging.basicConfig(level=logging.DEBUG)

env_name = "PutEggplantInBasketScene-v0"

kwargs = {
    "obs_mode":"rgbd", 
    "robot": "widowx_sink_camera_setup",
    "sim_freq": 500,
    "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    "control_freq": 5,
    "max_episode_steps": 120,
    "scene_name": "bridge_table_1_v2",
    "camera_cfgs": {"add_segmentation": True},
    "rgb_overlay_path": "/inspire/hdd/global_user/gongjingjing-25039/jhshi/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
}

additional_env_build_kwargs = {}

print("🔧 Start building ManiSkill2 env...")
env = build_maniskill2_env(
    env_name,
    **additional_env_build_kwargs,
    **kwargs,
)
print("✅ Env built successfully:", env)

obs = env.reset()
print("📷 First observation keys:", obs.keys() if isinstance(obs, dict) else type(obs))

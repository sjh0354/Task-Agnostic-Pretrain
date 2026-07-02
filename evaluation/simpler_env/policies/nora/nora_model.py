import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
import cv2 as cv
from transforms3d.euler import euler2axangle
import matplotlib.pyplot as plt

# 这里默认使用 Nora 官方仓库中 bridge 版本的 Nora 类
# 如果您使用的是另一个路径或 LIBERO 版本，请改成相应的导入路径
from .nora_utils import Nora as NoraCore


class NoraInference:
    def __init__(
        self,
        saved_model_path: str = "declare-lab/nora",
        fast_tokenizer_id: str = "/inspire/hdd/global_user/gongjingjing-25039/jhshi/nora/download_models/fast",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: Tuple[int, int] = (224, 224),
        action_scale: float = 1.0,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        与 OpenVLAInference 接口对应的 Nora 推理类。
        - saved_model_path: Nora 模型权重（HF repo 或本地路径），默认 declare-lab/nora
        - fast_tokenizer_id: Nora 解码动作的专用 tokenizer（HF repo 或本地路径）
        - unnorm_key: 用于动作反归一化的 key。若不指定，将根据 policy_setup 设置默认值
        - policy_setup: ["widowx_bridge", "google_robot"] 两种模式，和原 OpenVLAInference 保持一致
        - image_size: 输入图像尺寸，默认(224,224)
        - action_scale: 动作缩放因子
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if policy_setup == "widowx_bridge":
            # 与 OpenVLA 中一致
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            # 与 OpenVLA 中一致
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported. Use 'widowx_bridge' or 'google_robot'."
            )

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.image_size = tuple(image_size)
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        # 初始化 Nora 主体
        self.nora = NoraCore(
            model_id=saved_model_path,
            fast_tokenizer_id=fast_tokenizer_id,
            device=device,
            torch_dtype=torch_dtype,
        )

        # 内部状态（与 OpenVLAInference 对齐）
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.num_image_history = 0

    def reset(self, task_description: str) -> None:
        # 重置任务和粘滞夹爪逻辑
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)
        return image

    @torch.inference_mode()
    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> Tuple[dict, dict]:
        """
        输入:
            image: np.ndarray, (H, W, 3), uint8
            task_description: Optional[str], 任务描述；若与之前不同则自动 reset
        输出:
            raw_action: dict; Nora 模型的原始连续动作输出，包含:
                - "world_vector": np.ndarray(3,)
                - "rotation_delta": np.ndarray(3,)  (Euler 角增量: roll, pitch, yaw)
                - "open_gripper": np.ndarray(1,)    (范围 [0,1], 1=open, 0=close)
            action: dict; 送入环境的动作，包含:
                - 'world_vector': np.ndarray(3,), 末端位姿 xyz 平移
                - 'rot_axangle': np.ndarray(3,), 末端朝向（轴角，axis*angle）
                - 'gripper': np.ndarray(1,), 夹爪动作（widowx_bridge: {-1,+1}；google_robot: 相对开合）
                - 'terminate_episode': np.ndarray(1,), 是否终止当前 episode（恒为 0）
        """
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8, "Input image must be uint8."
        image = self._resize_image(image)
        pil_image = Image.fromarray(image)

        # 调用 Nora 推理（返回去归一化后的 7 维动作）
        # 形状可能是 (time_horizon, 7) 或 (7,), 这里统一取第一个时间步
        nora_out = self.nora.inference(pil_image, self.task_description, unnorm_key=self.unnorm_key)

        if isinstance(nora_out, np.ndarray):
            if nora_out.ndim == 2 and nora_out.shape[-1] == 7:
                action_vec = nora_out[0]
            elif nora_out.ndim == 1 and nora_out.shape[0] == 7:
                action_vec = nora_out
            else:
                raise ValueError(f"Unexpected Nora output shape: {nora_out.shape}")
        else:
            # 安全检查
            action_vec = np.array(nora_out)
            if action_vec.ndim == 2 and action_vec.shape[-1] == 7:
                action_vec = action_vec[0]
            elif action_vec.ndim == 1 and action_vec.shape[0] == 7:
                pass
            else:
                raise ValueError(f"Unexpected Nora output type/shape: {type(nora_out)} / {action_vec.shape}")

        # 7 维: [dx, dy, dz, droll, dpitch, dyaw, open_gripper]
        world_vector = np.array(action_vec[:3], dtype=np.float32)
        rotation_delta = np.array(action_vec[3:6], dtype=np.float32)
        open_gripper = np.array(action_vec[6:7], dtype=np.float32)  # [0,1]

        raw_action = {
            "world_vector": world_vector.copy(),
            "rotation_delta": rotation_delta.copy(),
            "open_gripper": open_gripper.copy(),
        }

        # 构造最终动作
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale

        # Euler -> axis-angle
        roll, pitch, yaw = raw_action["rotation_delta"].astype(np.float64)
        axis, angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = (axis * angle).astype(np.float64) * self.action_scale

        # 夹爪逻辑
        if self.policy_setup == "google_robot":
            # 相对动作 + 粘滞逻辑（与 OpenVLAInference 一致）
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0.0], dtype=np.float32)
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action.astype(np.float32)

        elif self.policy_setup == "widowx_bridge":
            # Nora 的 open_gripper ∈ [0,1]，这里二值化到 {-1,+1}（与 OpenVLA 对齐：>0.5 视为 open）
            action["gripper"] = (2.0 * (raw_action["open_gripper"] > 0.5) - 1.0).astype(np.float32)

        action["terminate_episode"] = np.array([0.0], dtype=np.float32)
        return raw_action, action

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[dict], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        """
        与 openvla_model.py 中的可视化保持一致：
        - 将多张图拼成长条
        - 绘制每一维动作随时间的曲线
        """
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # 将 raw_action dict 列表打包为 (T,7)
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )

        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
        plt.close(fig)


# 使用要点与差异说明
# - 替换方式：和您 openvla_model.py 的用法一致，把类名换成 NoraInference 即可，其余调用（reset、step、可视化）保持不变。
# - 参数：
#   - saved_model_path 默认 "declare-lab/nora"
#   - fast_tokenizer_id 默认 "physical-intelligence/fast"
#   - policy_setup 支持 "widowx_bridge"（默认）与 "google_robot"
#   - image_size 默认 (224,224)
#   - unnorm_key 默认根据 policy_setup 自动选择（widowx_bridge -> "bridge_orig", google_robot -> "fractal20220817_data"）
# - 多步动作：如果 Nora 的 time_horizon > 1，当前实现只取第 1 步。如需一次返回多步并在环境内执行，可参考您 Bridge 代码里的执行逻辑进行扩展。
# - 夹爪一致性：widowx_bridge 下输出 {-1,+1}；google_robot 下按 OpenVLA 的相对+粘滞逻辑实现。您也可以根据具体环境接口调整阈值或方向。
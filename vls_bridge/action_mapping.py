from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ActionMappingConfig:
    action_mode: str = "joint_delta"
    input_normalized: bool = True
    joint_delta_scale: float = 0.05
    gripper_mode: str = "absolute"
    gripper_delta_scale: float = 10.0


class ActionMapper:
    def __init__(self, cfg: ActionMappingConfig):
        self.cfg = cfg

    def _denormalize(self, values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return low + (values + 1.0) * 0.5 * (high - low)

    def map_first_action(
        self,
        policy_action: np.ndarray,
        current_ctrl: np.ndarray,
        ctrl_low: Optional[np.ndarray],
        ctrl_high: Optional[np.ndarray],
    ) -> np.ndarray:
        mapped = current_ctrl.copy()
        cmd = np.asarray(policy_action, dtype=np.float32).copy()
        ctrl_dim = mapped.shape[0]
        joint_dim = min(6, ctrl_dim)

        if cmd.shape[0] < joint_dim:
            raise ValueError(f"Policy action dim {cmd.shape[0]} is smaller than required joint dim {joint_dim}.")

        joint_cmd = cmd[:joint_dim]
        gripper_cmd = cmd[joint_dim] if cmd.shape[0] > joint_dim and ctrl_dim > joint_dim else None

        if self.cfg.action_mode == "joint_absolute":
            if self.cfg.input_normalized and ctrl_low is not None and ctrl_high is not None:
                mapped[:joint_dim] = self._denormalize(joint_cmd, ctrl_low[:joint_dim], ctrl_high[:joint_dim])
            else:
                mapped[:joint_dim] = joint_cmd
        elif self.cfg.action_mode == "joint_delta":
            delta = joint_cmd * self.cfg.joint_delta_scale if self.cfg.input_normalized else joint_cmd
            mapped[:joint_dim] = current_ctrl[:joint_dim] + delta
        else:
            raise ValueError(f"Unsupported action_mode: {self.cfg.action_mode}")

        if gripper_cmd is not None:
            if self.cfg.gripper_mode == "absolute":
                if self.cfg.input_normalized and ctrl_low is not None and ctrl_high is not None:
                    mapped[joint_dim] = self._denormalize(
                        np.array([gripper_cmd], dtype=np.float32),
                        np.array([ctrl_low[joint_dim]], dtype=np.float32),
                        np.array([ctrl_high[joint_dim]], dtype=np.float32),
                    )[0]
                else:
                    mapped[joint_dim] = gripper_cmd
            elif self.cfg.gripper_mode == "delta":
                delta = gripper_cmd * self.cfg.gripper_delta_scale if self.cfg.input_normalized else gripper_cmd
                mapped[joint_dim] = current_ctrl[joint_dim] + delta
            else:
                raise ValueError(f"Unsupported gripper_mode: {self.cfg.gripper_mode}")

        if ctrl_low is not None and ctrl_high is not None:
            mapped = np.clip(mapped, ctrl_low, ctrl_high)
        return mapped


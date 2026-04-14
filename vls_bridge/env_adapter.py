from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from env.ur5_grasp_env import UR5GraspEnv


@dataclass
class EnvStepResult:
    obs: Dict
    reward: float
    done: bool
    info: Dict


class MujocoEnvAdapter:
    def __init__(self, env: Optional[UR5GraspEnv] = None, *, sim_hz: int = 500, show_gui: bool = False,
                 camera_id: int = 0):
        self.env = env or UR5GraspEnv(sim_hz=sim_hz, show_gui=show_gui, camera_id=camera_id)
        self._step_count = 0

    def reset(self) -> Dict:
        self.env.reset()
        self._step_count = 0
        return self.get_obs()

    def step(self, action: np.ndarray) -> EnvStepResult:
        self.env.step(action)
        self._step_count += 1
        obs = self.get_obs()
        return EnvStepResult(obs=obs, reward=0.0, done=False, info={"step_count": self._step_count})

    def get_obs(self) -> Dict:
        obs = self.env.get_obs()
        return {
            "proprio": np.concatenate([obs["joint_q"], obs["joint_dq"]]).astype(np.float32),
            "rgb": obs["rgb"],
            "depth": obs["depth"],
            "action": obs["action"].astype(np.float32),
        }

    def get_camera_image(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.env.get_camera_image()

    def close(self) -> None:
        self.env.close()


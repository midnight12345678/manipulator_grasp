from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np

from vls_bridge.config import VLSConfig
from vls_bridge.env_adapter import MujocoEnvAdapter
from vls_bridge.policy_adapter import DiffusionPolicyAdapter, PiPolicyAdapter
from vls_bridge.task_runner import VLSRunner


class RandomPolicy:
    """Drop-in placeholder. Replace with your pretrained diffusion/Pi model wrapper."""

    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def __call__(self, obs: Dict[str, Any], horizon: int, batch_size: int) -> np.ndarray:
        del obs
        return np.random.uniform(-1.0, 1.0, size=(batch_size, horizon, self.action_dim)).astype(np.float32)


def build_policy_adapter(policy_type: str):
    base_policy = RandomPolicy()
    if policy_type.lower() == "pi":
        return PiPolicyAdapter(base_policy)
    return DiffusionPolicyAdapter(base_policy)


def main():
    parser = argparse.ArgumentParser(description="Run VLS-style inference-time steering on UR5 MuJoCo env.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON/YAML config file.")
    args = parser.parse_args()

    cfg = VLSConfig.from_path(args.config)
    env = MujocoEnvAdapter(
        sim_hz=cfg.runtime.sim_hz,
        show_gui=cfg.runtime.show_gui,
        camera_id=cfg.runtime.camera_id,
    )
    policy = build_policy_adapter(cfg.policy_type)
    runner = VLSRunner(env, policy, cfg.runtime, cfg.guidance)

    try:
        result = runner.run_episode()
        print(f"Episode done. steps={result['steps']} guidance={result['guidance']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

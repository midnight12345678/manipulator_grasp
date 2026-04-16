from __future__ import annotations

import argparse

from vls_bridge.config import VLSConfig
from vls_bridge.env_adapter import MujocoEnvAdapter
from vls_bridge.policy_adapter import DiffusionPolicyAdapter, PiPolicyAdapter, build_policy_callable
from vls_bridge.task_runner import VLSRunner


def build_policy_adapter(cfg: VLSConfig):
    policy_cfg = cfg.policy
    base_policy = build_policy_callable(
        backend=policy_cfg.backend,
        action_dim=policy_cfg.action_dim,
        checkpoint_path=policy_cfg.checkpoint_path,
        factory=policy_cfg.factory,
        device=policy_cfg.device,
        obs_keys=policy_cfg.obs_keys,
        extra_kwargs=policy_cfg.extra_kwargs,
    )
    if policy_cfg.policy_type.lower() == "pi":
        return PiPolicyAdapter(base_policy)
    return DiffusionPolicyAdapter(base_policy)


def main():
    parser = argparse.ArgumentParser(description="Run VLS-style inference-time steering on UR5 MuJoCo env.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON/YAML config file.")
    parser.add_argument("--instruction", type=str, default=None, help="Override instruction from config.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config.")
    args = parser.parse_args()

    cfg = VLSConfig.from_path(args.config)
    if args.instruction is not None:
        cfg.runtime.instruction = args.instruction
    if args.seed is not None:
        cfg.runtime.seed = args.seed
    env = MujocoEnvAdapter(
        sim_hz=cfg.runtime.sim_hz,
        show_gui=cfg.runtime.show_gui,
        camera_id=cfg.runtime.camera_id,
    )
    policy = build_policy_adapter(cfg)
    runner = VLSRunner(env, policy, cfg.runtime, cfg.guidance, cfg.action_mapping)

    try:
        result = runner.run_episode()
        print(f"Episode done. steps={result['steps']} seed={result['seed']} guidance={result['guidance']}")
        if cfg.runtime.save_rollout_path:
            print(f"Rollout saved to: {cfg.runtime.save_rollout_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

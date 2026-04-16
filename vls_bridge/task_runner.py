from __future__ import annotations

import importlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .action_mapping import ActionMapper, ActionMappingConfig
from .config import GuidanceConfig, RuntimeConfig
from .env_adapter import MujocoEnvAdapter
from .policy_adapter import PolicyAdapter
from .steering import feynman_kac_resample, gradient_refinement, rbf_diversity_bonus

# Default heuristic weights; keep in one place for easy tuning.
GRIPPER_CLOSE_WEIGHT = 1e-3
SMOOTHNESS_PENALTY_WEIGHT = 0.1
GUIDE_SCALE_MULTIPLIER = 1e-3


@dataclass
class GuidanceContext:
    instruction: str
    guidance: Dict[str, Any]
    obs: Dict[str, Any]


class SimpleGuidanceProvider:
    """Pluggable VLM guidance provider; replace `query` with your VLM pipeline."""

    def query(self, instruction: str, rgb: np.ndarray, depth: np.ndarray) -> Dict[str, Any]:
        h, w = depth.shape[:2]
        y, x = np.unravel_index(np.argmin(depth), depth.shape)
        return {
            "target_pixel": np.array([x / max(w - 1, 1), y / max(h - 1, 1)], dtype=np.float32),
            "target_depth": float(depth[y, x]),
            "instruction": instruction,
        }


def _import_from_path(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"Factory path must be 'module.submodule:callable', got: {path}")
    module_name, symbol_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    if not hasattr(module, symbol_name):
        raise ValueError(f"Symbol '{symbol_name}' not found in module '{module_name}'.")
    return getattr(module, symbol_name)


def build_guidance_provider(guidance_cfg: GuidanceConfig) -> SimpleGuidanceProvider:
    if not guidance_cfg.provider_factory:
        return SimpleGuidanceProvider()
    factory = _import_from_path(guidance_cfg.provider_factory)
    provider = factory(**guidance_cfg.provider_kwargs)
    if not hasattr(provider, "query"):
        raise TypeError("Guidance provider must expose a callable `query` method.")
    return provider


class VLSRunner:
    def __init__(
        self,
        env: MujocoEnvAdapter,
        policy: PolicyAdapter,
        runtime_cfg: RuntimeConfig,
        guidance_cfg: GuidanceConfig,
        action_mapping_cfg: Optional[ActionMappingConfig] = None,
        guidance_provider: Optional[SimpleGuidanceProvider] = None,
    ):
        self.env = env
        self.policy = policy
        self.runtime_cfg = runtime_cfg
        self.guidance_cfg = guidance_cfg
        self.action_mapper = ActionMapper(action_mapping_cfg or ActionMappingConfig())
        self.guidance_provider = guidance_provider or build_guidance_provider(guidance_cfg)
        self.rng = np.random.default_rng(runtime_cfg.seed)

    @staticmethod
    def _score_actions(action_sequences: np.ndarray, context: GuidanceContext) -> np.ndarray:
        """Score candidate action sequences and return one scalar score per sequence.

        Note: gripper signal is assumed to be the last action dimension.
        """
        if action_sequences.shape[-1] < 1:
            raise ValueError("Expected action_sequences[..., A] with A >= 1.")
        target_depth = float(context.guidance.get("target_depth", 0.0))
        gripper = action_sequences[:, :, -1]
        gripper_signal_mean = np.mean(gripper, axis=1) * GRIPPER_CLOSE_WEIGHT
        smooth_penalty = np.mean(np.linalg.norm(np.diff(action_sequences, axis=1), axis=-1), axis=1)
        depth_term = -abs(target_depth)
        return gripper_signal_mean - SMOOTHNESS_PENALTY_WEIGHT * smooth_penalty + depth_term

    def _guided_actions(self, obs: Dict[str, Any], guidance: Dict[str, Any]) -> np.ndarray:
        gcfg = self.guidance_cfg
        policy_obs = {**obs, "instruction": self.runtime_cfg.instruction}
        candidates = self.policy.sample_action_sequences(policy_obs, gcfg.action_horizon, gcfg.sample_batch_size)
        context = GuidanceContext(self.runtime_cfg.instruction, guidance, obs)

        if gcfg.use_guidance:
            candidates = gradient_refinement(
                candidates,
                lambda arr, _: self._score_actions(arr, context),
                {"guidance": guidance},
                guide_scale=gcfg.guide_scale * GUIDE_SCALE_MULTIPLIER,
                mcmc_steps=gcfg.mcmc_steps,
                rng=self.rng,
            )

        reward = self._score_actions(candidates, context)
        reward = reward + gcfg.diversity_scale * rbf_diversity_bonus(candidates, sigma=max(gcfg.temperature, 1e-3))
        stabilized = reward - np.max(reward)
        weights = np.exp(stabilized / max(gcfg.temperature, 1e-3))
        resampled = feynman_kac_resample(candidates, weights, self.rng)
        best = np.argmax(self._score_actions(resampled, context))
        return resampled[best]

    def _save_rollout(self, actions: np.ndarray, guidance: Dict[str, Any]) -> None:
        if not self.runtime_cfg.save_rollout_path:
            return
        output_path = Path(self.runtime_cfg.save_rollout_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "instruction": self.runtime_cfg.instruction,
            "seed": self.runtime_cfg.seed,
            "guidance": self._to_jsonable(guidance),
        }
        np.savez_compressed(
            output_path,
            actions=np.asarray(actions, dtype=np.float32),
        )
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): VLSRunner._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [VLSRunner._to_jsonable(v) for v in value]
        return value

    def run_episode(self) -> Dict[str, Any]:
        obs = self.env.reset()
        rgb, depth = self.env.get_camera_image()
        guidance = self.guidance_provider.query(self.runtime_cfg.instruction, rgb, depth)

        executed = []
        for _ in range(self.runtime_cfg.episode_steps):
            plan = self._guided_actions(obs, guidance)
            ctrl_low, ctrl_high = self.env.get_action_bounds()
            action = self.action_mapper.map_first_action(plan[0], obs["action"], ctrl_low, ctrl_high)
            result = self.env.step(action)
            obs = result.obs
            executed.append(action.copy())

        actions = np.asarray(executed, dtype=np.float32)
        self._save_rollout(actions, guidance)
        return {"steps": len(executed), "actions": actions, "guidance": guidance, "seed": self.runtime_cfg.seed}

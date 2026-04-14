from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .action_mapping import ActionMappingConfig


@dataclass
class GuidanceConfig:
    use_guidance: bool = True
    guide_scale: float = 40.0
    diversity_scale: float = 10.0
    sample_batch_size: int = 20
    action_horizon: int = 14
    mcmc_steps: int = 4
    temperature: float = 1.0


@dataclass
class RuntimeConfig:
    instruction: str = "grasp the target object"
    episode_steps: int = 250
    show_gui: bool = False
    camera_id: int = 0
    sim_hz: int = 500


@dataclass
class PolicyConfig:
    policy_type: str = "diffusion"
    backend: str = "random"
    checkpoint_path: Optional[str] = None
    factory: Optional[str] = None
    device: str = "cpu"
    action_dim: int = 7
    obs_keys: Optional[list] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VLSConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    action_mapping: ActionMappingConfig = field(default_factory=ActionMappingConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VLSConfig":
        runtime = RuntimeConfig(**data.get("runtime", {}))
        guidance = GuidanceConfig(**data.get("guidance", {}))
        raw_policy = data.get("policy", {})
        if not raw_policy:
            raw_policy = {
                "policy_type": data.get("policy_type", "diffusion"),
                **data.get("policy_kwargs", {}),
            }
        policy = PolicyConfig(**raw_policy)
        action_mapping = ActionMappingConfig(**data.get("action_mapping", {}))
        return VLSConfig(
            runtime=runtime,
            guidance=guidance,
            policy=policy,
            action_mapping=action_mapping,
        )

    @staticmethod
    def from_path(path: str) -> "VLSConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ImportError("PyYAML is required for YAML configs. Install with `pip install pyyaml`.") from exc
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return VLSConfig.from_dict(data or {})

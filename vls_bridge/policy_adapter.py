from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class PolicyAdapter(ABC):
    @abstractmethod
    def sample_action_sequences(self, obs: Dict[str, Any], horizon: int, batch_size: int) -> np.ndarray:
        """Return actions with shape [batch_size, horizon, action_dim]."""


class CallablePolicyAdapter(PolicyAdapter):
    def __init__(self, policy: Any):
        self.policy = policy

    def _call_policy(self, obs: Dict[str, Any], horizon: int, batch_size: int) -> np.ndarray:
        if hasattr(self.policy, "sample_action_sequences"):
            actions = self.policy.sample_action_sequences(obs, horizon, batch_size)
        elif callable(self.policy):
            actions = self.policy(obs=obs, horizon=horizon, batch_size=batch_size)
        else:
            raise TypeError("Policy must be callable or expose sample_action_sequences.")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError(f"Expected policy output [B, H, A], got shape {actions.shape}.")
        return actions

    def sample_action_sequences(self, obs: Dict[str, Any], horizon: int, batch_size: int) -> np.ndarray:
        return self._call_policy(obs, horizon, batch_size)


class DiffusionPolicyAdapter(CallablePolicyAdapter):
    """Adapter for pretrained diffusion-policy style models."""


class PiPolicyAdapter(CallablePolicyAdapter):
    """Adapter for Pi-family policies."""


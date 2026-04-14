from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

import numpy as np


def _import_from_path(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"Factory path must be 'module.submodule:callable', got: {path}")
    module_name, symbol_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    if not hasattr(module, symbol_name):
        raise ValueError(f"Symbol '{symbol_name}' not found in module '{module_name}'.")
    return getattr(module, symbol_name)


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
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3:
            raise ValueError(f"Expected policy output [B, H, A], got shape {actions.shape}.")
        if actions.shape[0] != batch_size:
            raise ValueError(f"Expected batch_size={batch_size}, got {actions.shape[0]}.")
        if actions.shape[1] < horizon:
            pad = np.repeat(actions[:, -1:, :], horizon - actions.shape[1], axis=1)
            actions = np.concatenate([actions, pad], axis=1)
        return actions[:, :horizon, :]

    def sample_action_sequences(self, obs: Dict[str, Any], horizon: int, batch_size: int) -> np.ndarray:
        return self._call_policy(obs, horizon, batch_size)


class RandomPolicy:
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def __call__(self, obs: Dict[str, Any], horizon: int, batch_size: int) -> np.ndarray:
        del obs
        return np.random.uniform(-1.0, 1.0, size=(batch_size, horizon, self.action_dim)).astype(np.float32)


class TorchScriptPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        obs_keys: Optional[Iterable[str]] = None,
        default_action_dim: int = 7,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Torch backend requested but torch is not installed.") from exc

        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for backend='torchscript'.")
        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()
        self.obs_keys = list(obs_keys) if obs_keys is not None else ["proprio", "action"]
        self.default_action_dim = default_action_dim

    def _build_obs_tensor(self, obs: Dict[str, Any], batch_size: int) -> "Any":
        chunks = []
        for key in self.obs_keys:
            if key not in obs:
                continue
            value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            chunks.append(value)
        if not chunks:
            chunks = [np.zeros((self.default_action_dim,), dtype=np.float32)]
        vec = np.concatenate(chunks, axis=0)
        tiled = np.tile(vec[None, :], (batch_size, 1))
        return self.torch.from_numpy(tiled).to(self.device)

    def __call__(self, obs: Dict[str, Any], horizon: int, batch_size: int) -> np.ndarray:
        with self.torch.no_grad():
            obs_tensor = self._build_obs_tensor(obs, batch_size)
            output = self.model(obs_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = output.detach().cpu().numpy().astype(np.float32)
            if output.ndim == 2:
                output = output[:, None, :]
            if output.ndim == 3 and output.shape[1] == 1 and horizon > 1:
                output = np.repeat(output, horizon, axis=1)
            return output


def build_policy_callable(
    backend: str,
    *,
    action_dim: int = 7,
    checkpoint_path: Optional[str] = None,
    factory: Optional[str] = None,
    device: str = "cpu",
    obs_keys: Optional[Iterable[str]] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    extra_kwargs = extra_kwargs or {}
    backend = backend.lower()
    if backend == "random":
        return RandomPolicy(action_dim=action_dim)
    if backend == "torchscript":
        return TorchScriptPolicy(
            checkpoint_path=checkpoint_path,
            device=device,
            obs_keys=obs_keys,
            default_action_dim=action_dim,
        )
    if backend == "factory":
        if not factory:
            raise ValueError("backend='factory' requires policy.factory in config.")
        fn = _import_from_path(factory)
        return fn(
            checkpoint_path=checkpoint_path,
            action_dim=action_dim,
            device=device,
            obs_keys=list(obs_keys) if obs_keys is not None else None,
            **extra_kwargs,
        )
    raise ValueError(f"Unsupported policy backend: {backend}")


class DiffusionPolicyAdapter(CallablePolicyAdapter):
    """Semantic adapter for diffusion-style policies using the unified callable contract."""
    pass


class PiPolicyAdapter(CallablePolicyAdapter):
    """Semantic adapter for Pi-family policies using the unified callable contract."""
    pass

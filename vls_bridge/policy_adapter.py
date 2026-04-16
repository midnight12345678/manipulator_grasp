from __future__ import annotations

import importlib
from contextlib import nullcontext
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


class LeRobotPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        task: str = "",
        image_key: Optional[str] = None,
        policy_name: str = "pi05",
        dtype: str = "auto",
        use_autocast: bool = True,
        state_dim: Optional[int] = None,
    ):
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be set to a HF repo id or local path for backend='lerobot'.")
        try:
            import torch
            from lerobot.policies.factory import get_policy_class
            from lerobot.processor import PolicyProcessorPipeline
            from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
        except ImportError as exc:
            raise ImportError("backend='lerobot' requires installing lerobot and torch.") from exc

        self.torch = torch
        self.ACTION = ACTION
        self.OBS_IMAGES = OBS_IMAGES
        self.OBS_STATE = OBS_STATE
        self.task = task
        self.image_key_override = image_key
        self.device = torch.device(device)
        self.inference_dtype = self._resolve_dtype(dtype)
        self.autocast_enabled = bool(use_autocast) and self.device.type == "cuda"
        policy_cls = get_policy_class(policy_name)
        self.policy = self._load_policy(policy_cls=policy_cls, checkpoint_path=checkpoint_path)
        self.policy.eval()
        if hasattr(self.policy, "to"):
            try:
                if self.device.type == "cuda":
                    self.policy.to(device=self.device, dtype=self.inference_dtype)
                else:
                    self.policy.to(self.device)
            except (TypeError, RuntimeError, ValueError):
                self.policy.to(self.device)
                self.inference_dtype = torch.float32
                self.autocast_enabled = False

        self.preprocessor = None
        self.postprocessor = None
        try:
            self.preprocessor = PolicyProcessorPipeline.from_pretrained(
                checkpoint_path, config_filename="preprocessor_config.json"
            )
        except (FileNotFoundError, OSError):
            self.preprocessor = None
        try:
            self.postprocessor = PolicyProcessorPipeline.from_pretrained(
                checkpoint_path, config_filename="postprocessor_config.json"
            )
        except (FileNotFoundError, OSError):
            self.postprocessor = None

        cfg_input = getattr(self.policy.config, "input_features", {}) or {}
        image_keys = [k for k in cfg_input.keys() if k.startswith(f"{OBS_IMAGES}.")]
        # Pi0.5 policy输入约定为 `observation.state`，这里固定使用标准键，避免不同配置分支造成不一致。
        self.state_key = self.OBS_STATE
        self.resolved_image_key = image_key or (image_keys[0] if image_keys else f"{self.OBS_IMAGES}.main")
        self.expected_state_dim = state_dim if state_dim is not None else self._infer_state_dim(cfg_input, self.state_key)

    @staticmethod
    def _load_policy(policy_cls: Any, checkpoint_path: str) -> Any:
        if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
            raise ValueError("checkpoint_path must be a non-empty string.")
        try:
            return policy_cls.from_pretrained(checkpoint_path)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            raise RuntimeError(
                "Failed to load LeRobot policy checkpoint via official from_pretrained interface. "
                "Please ensure checkpoint_path points to a compatible official checkpoint."
            ) from exc

    def _resolve_dtype(self, dtype_name: str):
        dtype_str = str(dtype_name or "auto").lower()
        if dtype_str == "auto":
            if self.device.type == "cuda":
                try:
                    major, _ = self.torch.cuda.get_device_capability(self.device)
                except (RuntimeError, AssertionError, AttributeError, ValueError, TypeError):
                    major = 0
                return self.torch.bfloat16 if major >= 8 else self.torch.float16
            return self.torch.float32
        mapping = {
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
        }
        if dtype_str not in mapping:
            raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: auto/fp32/fp16/bf16.")
        return mapping[dtype_str]

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _to_1d_float(value: Any) -> np.ndarray:
        if value is None:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(value, dtype=np.float32).reshape(-1)

    @staticmethod
    def _extract_feature_dim(spec: Any) -> Optional[int]:
        if spec is None:
            return None
        if isinstance(spec, (int, np.integer)):
            return int(spec)
        if isinstance(spec, dict):
            for key in ("shape", "sizes", "size", "dim", "dimension"):
                if key in spec:
                    dim = LeRobotPolicy._extract_feature_dim(spec[key])
                    if dim is not None:
                        return dim
            return None
        if hasattr(spec, "shape"):
            dim = LeRobotPolicy._extract_feature_dim(getattr(spec, "shape"))
            if dim is not None:
                return dim
        if isinstance(spec, (list, tuple)):
            dims = [int(v) for v in spec if isinstance(v, (int, np.integer))]
            return dims[-1] if dims else None
        return None

    @classmethod
    def _infer_state_dim(cls, input_features: Dict[str, Any], state_key: str) -> Optional[int]:
        if not isinstance(input_features, dict) or state_key not in input_features:
            return None
        return cls._extract_feature_dim(input_features[state_key])

    def _build_state_vector(self, obs: Dict[str, Any]) -> np.ndarray:
        joint_q = self._to_1d_float(obs.get("joint_q"))
        joint_dq = self._to_1d_float(obs.get("joint_dq"))
        action = self._to_1d_float(obs.get("action"))
        proprio = self._to_1d_float(obs.get("proprio"))

        if joint_q.size > 0:
            # Use physically meaningful state ordering first: arm joints + gripper command/state proxy.
            core_parts = [joint_q]
            if action.size > joint_q.size:
                core_parts.append(action[joint_q.size:joint_q.size + 1])
            core_state = np.concatenate(core_parts, axis=0)
            extended_parts = [core_state]
            if joint_dq.size > 0:
                extended_parts.append(joint_dq)
            if action.size > 0:
                extended_parts.append(action)
            extended_state = np.concatenate(extended_parts, axis=0)
        elif proprio.size > 0:
            core_state = proprio
            extended_state = np.concatenate([proprio, action], axis=0) if action.size > 0 else proprio
        else:
            raise ValueError("LeRobot PI05 backend requires robot state in obs['joint_q'] or obs['proprio'].")

        if self.expected_state_dim is None:
            return core_state.astype(np.float32)
        if self.expected_state_dim <= core_state.shape[0]:
            return core_state[:self.expected_state_dim].astype(np.float32)
        if self.expected_state_dim <= extended_state.shape[0]:
            return extended_state[:self.expected_state_dim].astype(np.float32)
        padded = np.zeros((self.expected_state_dim,), dtype=np.float32)
        padded[:extended_state.shape[0]] = extended_state
        return padded

    def _build_raw_batch(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        state_vec = self._build_state_vector(obs)
        batch: Dict[str, Any] = {self.state_key: state_vec}
        if "rgb" in obs and obs["rgb"] is not None:
            batch[self.resolved_image_key] = np.asarray(obs["rgb"])
        task_text = str(obs.get("instruction", self.task))
        if task_text:
            batch["task"] = task_text
        return batch

    def _predict_chunk(self, obs: Dict[str, Any], horizon: int) -> np.ndarray:
        raw_batch = self._build_raw_batch(obs)
        policy_batch = self.preprocessor(raw_batch) if self.preprocessor is not None else raw_batch
        use_amp = self.autocast_enabled and self.inference_dtype in (self.torch.float16, self.torch.bfloat16)
        amp_context = self.torch.cuda.amp.autocast(dtype=self.inference_dtype) if use_amp else nullcontext()
        with self.torch.inference_mode():
            with amp_context:
                if hasattr(self.policy, "predict_action_chunk"):
                    actions = self.policy.predict_action_chunk(policy_batch)
                else:
                    action = self.policy.select_action(policy_batch)
                    action_np = self._to_numpy(action)
                    if action_np.ndim == 1:
                        action_np = action_np[None, :]
                    return np.repeat(action_np[:, None, :], horizon, axis=1)

        if self.postprocessor is not None:
            processed = self.postprocessor(actions)
            if isinstance(processed, dict) and self.ACTION in processed:
                actions = processed[self.ACTION]
            else:
                actions = processed

        actions_np = self._to_numpy(actions)
        if actions_np.ndim == 2:
            actions_np = actions_np[:, None, :]
        if actions_np.shape[1] < horizon:
            pad = np.repeat(actions_np[:, -1:, :], horizon - actions_np.shape[1], axis=1)
            actions_np = np.concatenate([actions_np, pad], axis=1)
        return actions_np[:, :horizon, :]

    def __call__(self, obs: Dict[str, Any], horizon: int, batch_size: int) -> np.ndarray:
        chunks = []
        for _ in range(batch_size):
            chunk = self._predict_chunk(obs, horizon)
            chunks.append(chunk[0])
        return np.stack(chunks, axis=0).astype(np.float32)


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
    if backend in {"lerobot", "lerobot_pi05"}:
        return LeRobotPolicy(
            checkpoint_path=checkpoint_path or "lerobot/pi05_base",
            device=device,
            task=extra_kwargs.get("task", ""),
            image_key=extra_kwargs.get("image_key"),
            policy_name=extra_kwargs.get("policy_name", "pi05"),
            dtype=extra_kwargs.get("dtype", "auto"),
            use_autocast=extra_kwargs.get("use_autocast", True),
            state_dim=extra_kwargs.get("state_dim"),
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


# Deprecated alias kept for backward compatibility with older configs/imports.
LeRobotPI05Policy = LeRobotPolicy

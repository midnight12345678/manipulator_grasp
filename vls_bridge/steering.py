from __future__ import annotations

from typing import Callable, Dict

import numpy as np


RewardFn = Callable[[np.ndarray, Dict], np.ndarray]
# RBF kernel parameters for diversity regularization.
RBF_VARIANCE_SCALE = 2.0
RBF_MIN_VARIANCE = 1e-8
# Numerical floor for particle weights before normalization.
MIN_WEIGHT = 1e-12


def rbf_diversity_bonus(action_sequences: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    flat = action_sequences.reshape(action_sequences.shape[0], -1)
    sq_norm = np.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=-1)
    kernel = np.exp(-sq_norm / max(RBF_VARIANCE_SCALE * sigma ** 2, RBF_MIN_VARIANCE))
    return 1.0 - np.mean(kernel, axis=1)


def feynman_kac_resample(action_sequences: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = np.maximum(weights, MIN_WEIGHT)
    weights = weights / np.sum(weights)
    idx = rng.choice(len(action_sequences), size=len(action_sequences), p=weights, replace=True)
    return action_sequences[idx]


def finite_diff_gradient(actions: np.ndarray, reward_fn: RewardFn, context: Dict, eps: float = 1e-3) -> np.ndarray:
    """Approximate d(reward)/d(actions) for one action sequence with vectorized finite differences."""
    base = reward_fn(actions[None, ...], context)[0]
    t, a = actions.shape
    total = t * a
    perturbed = np.repeat(actions[None, ...], total, axis=0)
    t_idx = np.repeat(np.arange(t), a)
    a_idx = np.tile(np.arange(a), t)
    perturbed[np.arange(total), t_idx, a_idx] += eps
    values = reward_fn(perturbed, context)
    return ((values - base) / eps).reshape(t, a)


def gradient_refinement(
    action_sequences: np.ndarray,
    reward_fn: RewardFn,
    context: Dict,
    *,
    guide_scale: float,
    mcmc_steps: int,
    noise_scale: float = 1e-2,
) -> np.ndarray:
    refined = action_sequences.copy()
    for i in range(refined.shape[0]):
        for _ in range(max(mcmc_steps, 1)):
            grad = finite_diff_gradient(refined[i], reward_fn, context)
            refined[i] = refined[i] + guide_scale * grad + np.random.randn(*grad.shape) * noise_scale
    return refined

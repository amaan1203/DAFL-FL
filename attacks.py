"""
attacks.py  —  Adversarial attacks for federated learning benchmarking.

Supports four attack types from fl-sandbox:
  - scaling   : scales gradient delta by random factor s ∈ [min, max]
  - signflip  : scales gradient delta by −s (reverses sign)
  - label_flip: data-time — flips labels y → (y+1) % num_classes during training
  - backdoor  : data-time — injects pixel/square trigger + relabels to target class
"""

import torch
import numpy as np
from typing import Dict, Any, Optional


# ------------------------------------------------------------------
# Update-time attacks  (applied to delta after local training)
# ------------------------------------------------------------------

def apply_update_attack(
    delta_dict: Dict[str, torch.Tensor],
    attack: str,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    """Apply a model-update-time attack to a parameter delta dict.

    Parameters
    ----------
    delta_dict : dict
        { param_name -> tensor } — the local delta (local - global).
    attack : str
        One of: 'scaling', 'signflip'.
    params : dict
        Attack-specific hyperparameters.
    rng : np.random.Generator
        RNG for randomness.

    Returns
    -------
    dict
        Modified delta dict.
    """
    if attack == "scaling":
        s = float(rng.uniform(params.get("min", 3.0), params.get("max", 10.0)))
        return {k: v * s for k, v in delta_dict.items()}

    if attack == "signflip":
        s = float(rng.uniform(params.get("min", 3.0), params.get("max", 10.0)))
        return {k: v * (-s) for k, v in delta_dict.items()}

    return delta_dict  # unknown attack — pass through unchanged


# ------------------------------------------------------------------
# Data-time attack helpers  (called inside the training loop)
# ------------------------------------------------------------------

def maybe_label_flip(
    y: torch.Tensor,
    num_classes: int,
    prob: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Randomly flip labels: y → (y + 1) % num_classes.

    Parameters
    ----------
    y : torch.Tensor  (long)
    num_classes : int
    prob : float
        Probability that any given sample gets its label flipped.
    rng : np.random.Generator, optional
    """
    if prob <= 0:
        return y
    if rng is not None and rng.random() > prob:
        return y
    return (y + 1) % num_classes


def add_backdoor_trigger(
    x: torch.Tensor,
    trigger: str = "corner_pixel",
    value: float = 1.0,
) -> torch.Tensor:
    """Inject a backdoor trigger pattern into a batch of images.

    Parameters
    ----------
    x : torch.Tensor  shape (B, C, H, W) or (B, features) for non-image
    trigger : str
        'corner_pixel' — sets bottom-right pixel to `value`.
        'square'       — sets 4×4 bottom-right block to `value`.
    value : float
    """
    x2 = x.clone()
    if x.dim() < 3:
        # Non-image (e.g. tabular): just perturb the last feature
        x2[:, -1] = value
        return x2

    if trigger == "corner_pixel":
        x2[..., -1, -1] = value
    elif trigger == "square":
        x2[..., -4:, -4:] = value
    else:
        x2[..., -1, -1] = value
    return x2


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def pick_attack(attack_pool: list, rng: np.random.Generator) -> Optional[str]:
    """Randomly pick one attack from a pool. Returns None if pool is empty."""
    if not attack_pool:
        return None
    idx = rng.integers(0, len(attack_pool))
    return attack_pool[idx]


# Default hyperparameters for each attack
DEFAULT_ATTACK_PARAMS = {
    "scaling": {"min": 3.0, "max": 10.0},
    "signflip": {"min": 3.0, "max": 10.0},
    "label_flip": {"prob": 1.0},
    "backdoor": {"poison_frac": 0.3, "trigger": "corner_pixel", "trigger_value": 1.0, "target": 0},
}

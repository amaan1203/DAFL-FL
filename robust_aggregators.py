"""
robust_aggregators.py  —  Robust aggregation methods for federated learning.

Implements:
  - fedavg        : plain weighted FedAvg (baseline)
  - krum          : Krum — select the single best update
  - multi_krum    : Multi-Krum — average the top-m best updates
  - trimmed_mean  : coordinate-wise trimmed mean
  - sss           : Shamir's Secret Sharing aggregator (our contribution,
                    implemented in secure_aggregator.py — this module only
                    provides the post-reconstruction weighted average)

All aggregators take a list of delta dicts  { param_name -> tensor }
and return one aggregated delta dict.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple


# ------------------------------------------------------------------
# Helper: flatten / unflatten state dicts to vectors
# ------------------------------------------------------------------

def _state_dict_to_vec(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in d.values()])


def _vec_to_state_dict(
    vec: torch.Tensor,
    ref: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = {}
    idx = 0
    for k, v in ref.items():
        n = v.numel()
        out[k] = vec[idx: idx + n].reshape(v.shape).to(v.dtype)
        idx += n
    return out


# ------------------------------------------------------------------
# Pairwise distance matrix  (squared Euclidean)
# ------------------------------------------------------------------

def _pairwise_dist2(mat: torch.Tensor) -> torch.Tensor:
    """mat: (n, d) → returns (n, n) squared distances."""
    norms = (mat ** 2).sum(dim=1, keepdim=True)
    d2 = norms + norms.t() - 2.0 * (mat @ mat.t())
    return torch.clamp(d2, min=0.0)


# ------------------------------------------------------------------
# Aggregation functions
# ------------------------------------------------------------------

def fedavg(
    deltas: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Weighted FedAvg.

    Parameters
    ----------
    deltas : list of delta dicts
    weights : optional per-client weights (unnormalized); uniform if None.
    """
    if weights is None:
        weights = [1.0] * len(deltas)

    w_sum = sum(weights)
    keys = list(deltas[0].keys())
    out = {k: torch.zeros_like(deltas[0][k], dtype=torch.float32) for k in keys}

    for delta, w in zip(deltas, weights):
        for k in keys:
            out[k] += (w / w_sum) * delta[k].float()

    return out


def krum(
    deltas: List[Dict[str, torch.Tensor]],
    f: int,
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Krum: select the update with minimum sum of distances to closest n-f-2 neighbours.

    Parameters
    ----------
    deltas : list of delta dicts
    f : int — assumed number of malicious clients
    weights : ignored (Krum selects, not averages)
    """
    n = len(deltas)
    if n <= 3:
        # Krum is undefined for n <= 3; fall back to FedAvg
        return fedavg(deltas, weights)
    mat = torch.stack([_state_dict_to_vec(d) for d in deltas])  # (n, d)
    d2 = _pairwise_dist2(mat)
    # Cap f so that m = n - f - 2 is always >= 2 (Krum requires at least 2 neighbours)
    f_capped = min(f, n - 4)  # worst-case: m = n - (n-4) - 2 = 2
    f_capped = max(0, f_capped)
    m = max(2, n - f_capped - 2)
    scores = []
    for i in range(n):
        vals, _ = torch.sort(d2[i])
        score = vals[1: 1 + m].sum()   # skip self (0)
        scores.append(score)
    scores = torch.stack(scores)
    best = int(torch.argmin(scores).item())
    return {k: deltas[best][k].clone() for k in deltas[best]}


def multi_krum(
    deltas: List[Dict[str, torch.Tensor]],
    f: int,
    m_select: Optional[int] = None,
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Multi-Krum: average the m_select best updates by Krum score.

    Parameters
    ----------
    deltas : list of delta dicts
    f : int — assumed number of malicious clients
    m_select : number of updates to select; defaults to n - f
    weights : optional weights for the selected updates
    """
    n = len(deltas)
    if n <= 3:
        # Krum is undefined for n <= 3; fall back to FedAvg
        return fedavg(deltas, weights)
    mat = torch.stack([_state_dict_to_vec(d) for d in deltas])
    d2 = _pairwise_dist2(mat)
    # Cap f so that m_neighbour = n - f - 2 is always >= 2
    f_capped = min(f, n - 4)
    f_capped = max(0, f_capped)
    m_neighbour = max(2, n - f_capped - 2)
    scores = []
    for i in range(n):
        vals, _ = torch.sort(d2[i])
        scores.append(vals[1: 1 + m_neighbour].sum())
    scores = torch.stack(scores)
    order = torch.argsort(scores)
    m_sel = m_select if m_select is not None else max(1, n - f)
    m_sel = min(m_sel, n)  # guard: never select more than available
    selected = [int(i.item()) for i in order[:m_sel]]

    selected_deltas = [deltas[i] for i in selected]
    selected_weights = [weights[i] for i in selected] if weights else None
    return fedavg(selected_deltas, selected_weights)


def trimmed_mean(
    deltas: List[Dict[str, torch.Tensor]],
    beta: float = 0.1,
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise trimmed mean.

    Parameters
    ----------
    deltas : list of delta dicts
    beta : float — fraction to trim from each end (0 ≤ beta < 0.5)
    weights : ignored (trimmed mean doesn't weight)
    """
    mat = torch.stack([_state_dict_to_vec(d) for d in deltas])  # (n, d)
    n = mat.shape[0]
    k = int(beta * n)

    sorted_mat, _ = torch.sort(mat, dim=0)
    if n - 2 * k > 0:
        trimmed = sorted_mat[k: n - k]
    else:
        trimmed = sorted_mat
    out_vec = trimmed.mean(dim=0)

    return _vec_to_state_dict(out_vec, deltas[0])


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

AGGREGATOR_NAMES = ["fedavg", "krum", "multi_krum", "trimmed_mean", "sss"]


def aggregate(
    agg_name: str,
    deltas: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    f: int = 1,
    m_select: Optional[int] = None,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Unified aggregation dispatcher.

    Parameters
    ----------
    agg_name : str
        One of: 'fedavg', 'krum', 'multi_krum', 'trimmed_mean'.
        'sss' is handled by SecureAggregator directly.
    deltas : list[dict]
        Per-client parameter delta dicts.
    weights : list[float], optional
        Per-client weights (for fedavg / multi_krum).
    f : int
        Assumed number of malicious clients (for Krum variants).
    m_select : int, optional
        Number of update to select in Multi-Krum.
    beta : float
        Trim fraction for Trimmed Mean.

    Returns
    -------
    dict — aggregated delta
    """
    if agg_name == "fedavg":
        return fedavg(deltas, weights)
    elif agg_name == "krum":
        return krum(deltas, f, weights)
    elif agg_name == "multi_krum":
        return multi_krum(deltas, f, m_select, weights)
    elif agg_name == "trimmed_mean":
        return trimmed_mean(deltas, beta, weights)
    else:
        raise ValueError(f"Unknown aggregator '{agg_name}'. Options: {AGGREGATOR_NAMES}")

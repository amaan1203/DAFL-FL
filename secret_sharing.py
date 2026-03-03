import random
import math
from typing import List, Tuple, Set, Optional
from collections import defaultdict
import itertools
import numpy as np

# --- FIXED PARAMETERS ---
# SCALE_FACTOR converts float parameter deltas to integers for SSS.
SCALE_FACTOR = 10**6

# FIELD_SIZE is a large prime for the finite field.
FIELD_SIZE = 2**31 - 1  # 2147483647 — Mersenne prime, safely larger than any scaled delta


def polynom(x, coefficients):
    """Evaluate polynomial at point x over the finite field."""
    point = 0
    for coefficient_index, coefficient_value in enumerate(coefficients[::-1]):
        point += x ** coefficient_index * coefficient_value
    return point % FIELD_SIZE


def coeff(t: int, secret: int) -> List[int]:
    """Generate polynomial coefficients for secret sharing.
    
    The secret must be a non-negative integer in [0, FIELD_SIZE).
    For negative secrets, the caller must convert them to the field representation
    before calling this function.
    """
    t = int(t)
    coefficients = [random.randrange(0, FIELD_SIZE) for _ in range(t - 1)]
    coefficients.append(secret % FIELD_SIZE)  # ensure within field
    return coefficients


def generate_shares(n: int, m: int, secret: int) -> List[Tuple[int, int]]:
    """Generate n shares for a secret with threshold m.
    
    Parameters
    ----------
    n : int
        Total number of shares to generate.
    m : int
        Minimum number of shares required to reconstruct the secret (threshold).
    secret : int
        The integer secret to share. May be negative.
    
    Returns
    -------
    List[Tuple[int, int]]
        List of (x, y) share tuples.
    """
    # Encode negative secrets into the field by wrapping: s_field = s % FIELD_SIZE
    secret_field = secret % FIELD_SIZE
    coefficients = coeff(m, secret_field)
    shares = []
    used_x_values = set()

    for _ in range(n):
        while True:
            x = random.randrange(1, FIELD_SIZE)
            if x not in used_x_values:
                used_x_values.add(x)
                break
        y = polynom(x, coefficients) % FIELD_SIZE
        shares.append((x, y))
    return shares


def mod_inverse(n, modulus):
    """
    Computes the modular multiplicative inverse of n modulo modulus
    using the Extended Euclidean Algorithm.
    """
    n = n % modulus
    t, new_t = 0, 1
    r, new_r = modulus, n
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError(f"n={n} is not invertible modulo {modulus}")
    if t < 0:
        t = t + modulus
    return t


def reconstruct_secret(shares: List[Tuple[int, int]]) -> int:
    """
    Reconstructs the secret from shares using Lagrange interpolation
    with modular arithmetic in the finite field.
    
    Returns the secret as a signed integer (handling wrap-around for negatives).
    """
    sums = 0
    for j, share_j in enumerate(shares):
        xj, yj = share_j
        prod = 1
        for i, share_i in enumerate(shares):
            xi, _ = share_i
            if i != j:
                numerator = xi
                denominator = xi - xj
                inv_denominator = mod_inverse(denominator, FIELD_SIZE)
                term = (numerator * inv_denominator) % FIELD_SIZE
                prod = (prod * term) % FIELD_SIZE

        term = (yj * prod) % FIELD_SIZE
        sums = (sums + term) % FIELD_SIZE

    # --- CRITICAL FIX: Handle negative secrets ---
    # Negative integers were encoded as (secret % FIELD_SIZE) before sharing.
    # After reconstruction we need to undo this: if the result is in the upper
    # half of the field, it represents a negative number.
    if sums > FIELD_SIZE // 2:
        sums = sums - FIELD_SIZE

    return sums


class SecretSharingBuffer:
    """Buffer to collect shares from multiple uploaders and reconstruct
    the secret parameters belonging to a single owner client."""

    def __init__(self, owner_id, threshold):
        """
        Parameters
        ----------
        owner_id : int
            The client whose parameters this buffer will reconstruct.
        threshold : int
            Minimum number of distinct shares needed to reconstruct.
        """
        self.owner_id = owner_id
        self.threshold = threshold
        # Structure: { param_name -> { pos -> { uploader_id -> (x, y) share } } }
        self.shares_collection = defaultdict(lambda: defaultdict(dict))

    def add_shares(self, uploader_id: int, param_shares: dict):
        """Add shares contributed by one uploader client.

        Parameters
        ----------
        uploader_id : int
            The client sending the shares (not the owner — the sender).
        param_shares : dict
            { param_name -> [ {'pos': pos, 'share': (x, y)}, ... ] }
        """
        for param_name, param_data in param_shares.items():
            for pos_info in param_data:
                pos = pos_info['pos']
                share = pos_info['share']
                # Key by uploader_id so each client contributes at most 1 share per position
                self.shares_collection[param_name][pos][uploader_id] = share

    def has_enough_shares(self) -> bool:
        """Returns True if at least one parameter has enough shares to reconstruct."""
        for param_name, positions_data in self.shares_collection.items():
            for pos, client_shares in positions_data.items():
                if len(client_shares) >= self.threshold:
                    return True
        return False

    def reconstruct_parameters(self) -> dict:
        """Reconstruct all parameters from collected shares.

        Returns
        -------
        dict
            { param_name -> { pos -> float_value } }
            Empty dict if reconstruction fails.
        """
        reconstructed_params = {}

        for param_name, positions_data in self.shares_collection.items():
            param_values = {}

            for pos, client_shares in positions_data.items():
                n_available = len(client_shares)

                if n_available < self.threshold:
                    # Not enough shares for this position — skip
                    continue

                # Use exactly `threshold` shares for reconstruction
                # (using more is fine but slower; using threshold is sufficient)
                selected_uploaders = list(client_shares.keys())[:self.threshold]
                shares_for_reconstruction = [client_shares[uid] for uid in selected_uploaders]

                try:
                    reconstructed_int = reconstruct_secret(shares_for_reconstruction)
                    # Scale back to float
                    reconstructed_float = reconstructed_int / SCALE_FACTOR
                    param_values[pos] = reconstructed_float
                except Exception as e:
                    print(f"[Buffer {self.owner_id}] ERROR reconstructing {param_name}@{pos}: {e}")
                    continue

            if param_values:
                reconstructed_params[param_name] = param_values

        return reconstructed_params

    # ------------------------------------------------------------------
    # Byzantine-Robust reconstruction via majority-vote
    # ------------------------------------------------------------------

    def robust_reconstruct_parameters(
        self,
        n_trials: int = 100,
        consensus_tol: float = 1e-3,
        min_consensus_fraction: float = 0.3,
    ):
        """Reconstruct parameters using majority-vote over random share subsets.

        For each scalar position we draw `n_trials` random subsets of size
        `threshold` from the available n shares.  Each subset produces one
        candidate value.  We then find the largest cluster of candidates that
        mutually agree within `consensus_tol`.  Uploaders that **never** appear
        in any consensus-cluster subset are returned as suspected malicious.

        Returns
        -------
        (reconstructed_params, suspected_uploader_ids)
            reconstructed_params : dict  { param_name -> { pos -> float_value } }
            suspected_uploader_ids : set[int]
        """
        reconstructed_params = {}
        uploader_ever_in_consensus = defaultdict(bool)
        uploader_ever_seen = set()

        for param_name, positions_data in self.shares_collection.items():
            param_values = {}

            # We will find the "trusted subset" of uploaders using the first position (pos 0),
            # since a malicious client's update poisons all parameters simultaneously.
            trusted_uploaders = None

            for pos, client_shares in positions_data.items():
                uploader_ids = list(client_shares.keys())
                n_available  = len(uploader_ids)

                for uid in uploader_ids:
                    uploader_ever_seen.add(uid)

                if n_available < self.threshold:
                    continue

                # No redundancy — must trust all shares (plain reconstruct)
                if n_available == self.threshold:
                    try:
                        val = reconstruct_secret(
                            [client_shares[uid] for uid in uploader_ids]
                        ) / SCALE_FACTOR
                        param_values[pos] = val
                        for uid in uploader_ids:
                            uploader_ever_in_consensus[uid] = True
                    except Exception:
                        pass
                    continue
                
                # ---- Redundancy case ----
                
                # If we already found a trusted subset for this parameter, just use it
                if trusted_uploaders is not None:
                    try:
                        val = reconstruct_secret(
                            [client_shares[uid] for uid in trusted_uploaders]
                        ) / SCALE_FACTOR
                        param_values[pos] = val
                    except Exception:
                        pass
                    continue

                # Otherwise (this is pos=0 usually), run the full majority-vote discovery
                max_possible  = math.comb(n_available, self.threshold)
                actual_trials = min(n_trials, max_possible)

                # Generate subsets
                if max_possible <= n_trials:
                    subsets = list(itertools.combinations(uploader_ids, self.threshold))
                else:
                    rng_local = random.Random(f"{self.owner_id}_{pos}")
                    seen_subs = set()
                    subsets   = []
                    for _ in range(actual_trials * 8):
                        if len(subsets) >= actual_trials:
                            break
                        s = tuple(sorted(rng_local.sample(uploader_ids, self.threshold)))
                        if s not in seen_subs:
                            seen_subs.add(s)
                            subsets.append(s)

                # Reconstruct each subset
                candidates       = []   # float values
                subset_uploaders = []   # which uploader_ids were in each subset

                for subset in subsets:
                    try:
                        val = reconstruct_secret(
                            [client_shares[uid] for uid in subset]
                        ) / SCALE_FACTOR
                        candidates.append(val)
                        subset_uploaders.append(list(subset))
                    except Exception:
                        pass

                if not candidates:
                    continue

                # ---- Find majority consensus cluster ----
                arr = np.array(candidates, dtype=np.float64)
                n_cands = len(arr)
                min_cluster = max(2, int(min_consensus_fraction * n_cands))

                # For every candidate, count how many others agree within tolerance
                cluster_sizes = np.array([
                    np.sum(np.abs(arr - ref) <= consensus_tol)
                    for ref in arr
                ])
                best_size = int(cluster_sizes.max())

                # Require (a) absolute minimum, (b) clear plurality gap over runner-up
                runner_up_size = int(np.partition(cluster_sizes, -2)[-2]) if n_cands > 1 else 0
                gap_ok = best_size > runner_up_size or best_size >= min_cluster

                if best_size < min_cluster or not gap_ok:
                    # No clear majority — reconstruction excluded
                    continue

                best_indices = np.where(np.abs(arr - arr[np.argmax(cluster_sizes)]) <= consensus_tol)[0].tolist()

                # Consensus value = mean of majority cluster
                param_values[pos] = float(np.mean(arr[best_indices]))

                # Propagate consensus membership to uploaders
                consensus_set = set(best_indices)
                
                # We pick the FIRST subset that was part of the consensus cluster
                # to be our "trusted subset" for the rest of the parameters.
                first_consensus_idx = best_indices[0]
                trusted_uploaders = subset_uploaders[first_consensus_idx]
                
                for idx in consensus_set:
                    for uid in subset_uploaders[idx]:
                        uploader_ever_in_consensus[uid] = True

            if param_values:
                reconstructed_params[param_name] = param_values

        # Uploaders never in any consensus subset → suspects
        suspected = {
            uid for uid in uploader_ever_seen
            if not uploader_ever_in_consensus.get(uid, False)
        }
        if suspected:
            print(f"[Buffer {self.owner_id}] ⚠️  Detected suspicious uploaders: {suspected}")

        return reconstructed_params, suspected
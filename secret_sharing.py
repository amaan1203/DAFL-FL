import random
import math
from typing import List, Tuple, Set, Optional, Union
from collections import defaultdict
import itertools
import numpy as np

# --- FIXED PARAMETERS ---
# SCALE_FACTOR converts float parameter deltas to integers for SSS.
SCALE_FACTOR = 10**6  # 1e6 for high precision (e.g. 1e-4 -> 100)
                       # FIELD_SIZE = 2^31-1 ≈ 2.1e9 → max safe |delta| = 2147483 (vs 2147 before).

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


def coeff_vectorized(t: int, secrets_field: np.ndarray) -> np.ndarray:
    """Generate polynomial coefficients for a batch of secrets.
    secrets_field: an array of shape (N,) containing secrets encoded in the field.
    returns: an array of shape (N, t), where the last column is the secret.
    """
    N = secrets_field.shape[0]
    t = int(t)
    # Generate random coefficients in [0, FIELD_SIZE)
    coeffs = np.random.randint(0, FIELD_SIZE, size=(N, t - 1), dtype=np.int64)
    secrets_col = secrets_field.reshape(N, 1).astype(np.int64)
    return np.hstack((coeffs, secrets_col))


def polynom_vectorized(x: int, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate polynomials for N secrets at point x.
    coeffs: (N, t) array
    x: integer point
    returns: (N,) array of y values
    """
    N, t = coeffs.shape
    # Powers of x: [x^(t-1), x^(t-2), ..., x^0]
    powers = np.power(x, np.arange(t - 1, -1, -1, dtype=np.int64)) % FIELD_SIZE
    
    y = np.zeros(N, dtype=np.int64)
    for i in range(t):
        term = (coeffs[:, i] * powers[i]) % FIELD_SIZE
        y = (y + term) % FIELD_SIZE
    return y


def generate_shares_vectorized(n: int, m: int, secrets: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """Generate n shares for a batch of secrets with threshold m.
    
    Parameters
    ----------
    n : int
        Total number of shares to generate (one for each participant).
    m : int
        Threshold.
    secrets : np.ndarray
        Array of secrets of shape (N,).
        
    Returns
    -------
    Tuple[List[int], np.ndarray]
        List of x_values (len n)
        Array of y_values of shape (n, N) where each row corresponds to shares for an x_value.
    """
    # Encode negative secrets into the field
    secrets_field = secrets % FIELD_SIZE
    coeffs = coeff_vectorized(m, secrets_field)
    
    x_values = []
    y_values_list = []
    
    used_x_values = set()
    for _ in range(n):
        while True:
            x = np.random.randint(1, FIELD_SIZE)
            if x not in used_x_values:
                used_x_values.add(x)
                break
        
        y = polynom_vectorized(x, coeffs)
        x_values.append(int(x))
        y_values_list.append(y)
        
    # Return x_values and a stacked array of y_values (n, N)
    return x_values, np.vstack(y_values_list)


# ------------------------------------------------------------------
# PACKED SHAMIR'S SECRET SHARING (PSSS)
# ------------------------------------------------------------------

def coeff_packed_vectorized(t: int, l: int, secrets_field: np.ndarray) -> np.ndarray:
    """Generate polynomial coefficients for a batch of PACKED secrets.
    secrets_field: an array of shape (N, l) containing secrets encoded in the field.
    Returns: an array of shape (N, t + l - 1).
    The polynomial is P(x) = sum_i(c_i * x^i).
    To pack `l` secrets, we need the polynomial to evaluate to the secrets at specific points.
    For simplicity and efficiency in the packed domain, we'll embed the `l` secrets
    into the lowest `l` coefficients. Wait, standard packed SSS evaluates P(e_i) = s_i.
    However, for federated learning where we simply need additive homomorphism, we can
    directly set the lowest `l` coefficients to the secrets!
    P(x) = s_1 + s_2*x + ... + s_l*x^{l-1} + r_1*x^l + ... + r_{t-1}*x^{t+l-2}
    This requires `t` shares to reconstruct `l` secrets securely from `t+l-1` degree polynomial.
    """
    N = secrets_field.shape[0]
    degree = t + l - 1
    
    # Generate random high-degree coefficients: r_1 to r_{t-1}
    # Make absolutely sure type is int64 object to avoid any np default 32-bit overflow
    random_coeffs = np.random.randint(0, FIELD_SIZE, size=(N, t - 1), dtype=np.int64)
    
    # Lower coefficients are the secrets: s_1 to s_l
    secrets_coeffs = secrets_field.astype(np.int64)
    
    # Custom polynom_vectorized expects [c_degree, c_{degree-1}, ..., c_1, c_0]
    # So that it evaluates y = c_degree * x^degree + ... + c_1 * x + c_0
    # Our secrets are c_0 to c_{l-1}. So secrets are at the END of the array.
    # Therefore we stack [random_coeffs, secrets_coeffs[:, ::-1]]
    # Wait, secrets_coeffs is shape (N, l). Example row: [s0, s1, s2].
    # Reversed: [s2, s1, s0].
    # Stacked with random: [r1, r0, s2, s1, s0].
    # Degree evaluates: c_4 x^4 + c_3 x^3 + c_2 x^2 + c_1 x + c_0.
    # So c_0 = s0, c_1 = s1, c_2 = s2. This is perfectly correct!
    return np.hstack((random_coeffs, secrets_coeffs[:, ::-1]))


def generate_packed_shares_vectorized(n: int, t: int, l: int, secrets: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """Generate n packed shares for a batch of secrets.
    
    Parameters
    ----------
    n : Total number of shares to generate.
    t : Threshold of malicious users to tolerate (degrees of freedom).
    l : Number of secrets to pack into one polynomial.
    secrets : np.ndarray of shape (N, l).
    
    Returns
    -------
    Tuple[List[int], np.ndarray]
        x_values (len n)
        y_values of shape (n, N) where each row is an array of size N evaluating the polynomials.
    """
    secrets_field = secrets % FIELD_SIZE
    coeffs = coeff_packed_vectorized(t, l, secrets_field)
    
    x_values = []
    y_values_list = []
    
    used_x_values = set()
    for _ in range(n):
        while True:
            x = int(np.random.randint(1, FIELD_SIZE))
            # Prevent evaluating at x=0 (that directly reveals secrets in constant term!)
            if x not in used_x_values and x != 0:
                used_x_values.add(x)
                break
        
        # We must use Horner's method with explicit Python large integers
        # because numpy's int64 overflows during (coeff * x^power) mod p
        # coeffs shape is (N, degree + 1). We need y of shape (N,)
        
        y_vec = np.zeros(secrets_field.shape[0], dtype=np.int64)
        for i in range(coeffs.shape[1]):
            # Use int64 for speed; product fits in 64 bits as FIELD_SIZE < 2^31
            y_vec = (y_vec * x + coeffs[:, i]) % FIELD_SIZE
            
        x_values.append(int(x))
        y_values_list.append(y_vec.astype(np.int64))
        
    return x_values, np.vstack(y_values_list)


def reconstruct_packed_secret(shares: List[Tuple[int, int]], l: int) -> List[int]:
    """
    Reconstruct `l` packed secrets from `t + l - 1` shares.
    We need to find the lowest `l` coefficients of the interpolated polynomial.
    
    Lagrange interpolation to find P(x), then extract lowest `l` coefficients.
    P(x) = sum_j [ y_j * prod_{i != j} (x - x_i)/(x_j - x_i) ]
    Since we need the coefficients, we have to expand the algebraic product,
    or we can evaluate the derivative, but the easiest way over a finite field
    when we only need low degree (or all) coefficients is to solve the Vandermonde system,
    or iteratively build the polynomial.
    """
    # Instead of full polynomial multiplication, we can compute the secrets by evaluating
    # at specific x if we packed them via evaluation points (e.g., P(-i) = s_i).
    # But we packed them as coefficients! Thus we need to find the coefficients of P(x).
    
    # Let's find the coefficients of the polynomial passing through `shares`.
    # Interpolate polynom:
    # poly = 0
    # for j:
    #   basis_j = y_j * prod_{i!=j} (x - x_i) * (x_j - x_i)^-1
    #   poly += basis_j
    
    k = len(shares)
    
    # We will build polynomials as lists of coefficients: [c_0, c_1, ..., c_d]
    # representing c_0 + c_1*x + c_2*x^2 + ...
    def poly_mul(p1, p2):
        res = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            if c1 == 0: continue
            for j, c2 in enumerate(p2):
                res[i+j] = (res[i+j] + c1 * c2) % FIELD_SIZE
        return res
        
    def poly_add(p1, p2):
        max_len = max(len(p1), len(p2))
        res = [0] * max_len
        for i in range(max_len):
            v1 = p1[i] if i < len(p1) else 0
            v2 = p2[i] if i < len(p2) else 0
            res[i] = (v1 + v2) % FIELD_SIZE
        return res
        
    def poly_scalar_mul(p, scalar):
        return [(c * scalar) % FIELD_SIZE for c in p]

    final_poly = []
    
    for j, (xj, yj) in enumerate(shares):
        # build prod_{i!=j} (x - xi) / (xj - xi)
        basis = [1]
        denominator = 1
        for i, (xi, _) in enumerate(shares):
            if i != j:
                # (x - xi) is mathematically represented as [-xi, 1] (i.e., -xi + x)
                term = [(-xi) % FIELD_SIZE, 1]
                basis = poly_mul(basis, term)
                denominator = (denominator * (xj - xi)) % FIELD_SIZE
                
        inv_den = mod_inverse(denominator, FIELD_SIZE)
        
        if isinstance(yj, np.ndarray):
            coeff = (yj.astype(np.int64) * inv_den) % FIELD_SIZE
        else:
            coeff = (yj * inv_den) % FIELD_SIZE
        
        basis = poly_scalar_mul(basis, coeff)
        final_poly = poly_add(final_poly, basis)
        
    # We literally just need `final_poly[:l]`!
    secrets = final_poly[:l]
    
    is_array = any(isinstance(s, np.ndarray) for s in secrets)
    
    if is_array:
        while len(secrets) < l:
            secrets.append(np.zeros_like(secrets[0]))
            
        secrets_arr = np.array(secrets)
        secrets_arr = np.where(secrets_arr > FIELD_SIZE // 2, secrets_arr - FIELD_SIZE, secrets_arr)
        # Returns flattened values: block0_s0, block0_s1, ..., block1_s0...
        return secrets_arr.T.flatten()
    else:
        # Critical Fix for Negative Secrets
        decoded_secrets = []
        for s in secrets:
            # PSSS uses integers in the field size, anything in the upper half is negative
            if s > FIELD_SIZE // 2:
                s -= FIELD_SIZE
            decoded_secrets.append(int(s))
            
        # Pad if we have less coefficients than l
        while len(decoded_secrets) < l:
            decoded_secrets.append(0)
            
        return decoded_secrets


def reconstruct_secret_vectorized(shares: List[Tuple[int, np.ndarray]], FIELD_SIZE: int) -> np.ndarray:
    """
    Vectorized Lagrange interpolation to reconstruct multiple secrets from shares at same x-points.
    shares: list of (x, y_vector) where y_vector is (N,)
    returns: (N,) array of reconstructed secrets
    """
    n_shares = len(shares)
    x_vals = [s[0] for s in shares]
    y_vectors = np.stack([s[1] for s in shares])  # (n_shares, N)
    
    N = y_vectors.shape[1]
    sums = np.zeros(N, dtype=np.int64)
    
    for j in range(n_shares):
        xj = x_vals[j]
        yj = y_vectors[j]
        
        # Compute Lagrange basis polynomial L_j(0)
        prod = 1
        for i in range(n_shares):
            if i == j:
                continue
            xi = x_vals[i]
            # L_j(0) = product_{i != j} (0 - xi) / (xj - xi) = product xi / (xi - xj)
            num = xi
            den = (xi - xj) % FIELD_SIZE
            inv_den = mod_inverse(den, FIELD_SIZE)
            term = (num * inv_den) % FIELD_SIZE
            prod = (prod * term) % FIELD_SIZE
            
        term_j = (yj.astype(np.int64) * prod) % FIELD_SIZE
        sums = (sums + term_j) % FIELD_SIZE
        
    # Standard signed conversion
    sums = np.where(sums > FIELD_SIZE // 2, sums - FIELD_SIZE, sums)
    return sums


def reconstruct_packed_secret_vectorized(
    shares: List[Tuple[int, np.ndarray]], l: int, FIELD_SIZE: int
) -> np.ndarray:
    """
    Vectorized reconstruction for PSSS over blocks.
    shares: list of (x, y_vector) where y_vector is (n_blocks,)
    returns: (n_blocks * l,) flattened array of secrets
    """
    n_shares = len(shares)
    x_vals = [s[0] for s in shares]
    y_vectors = np.stack([s[1] for s in shares])  # (n_shares, n_blocks)
    n_blocks = y_vectors.shape[1]
    
    # In PSSS, we need to solve for the first l coefficients of the polynomial.
    # This involves the Vandermonde-style system or Lagrange basis at multiple points.
    # In this encoding, secrets are the coefficients P_0...P_{l-1}.
    # We recover the full polynomial coefficients by solving the Vandermonde system V*c = y.
    # V[i, j] = x_i^j.
    
    k = n_shares # Number of available shares
    V = np.ones((k, k), dtype=np.int64)
    for i in range(k):
        xi = int(x_vals[i])
        for j in range(1, k):
            V[i, j] = (V[i, j-1] * xi) % FIELD_SIZE
            
    # Solve V * c = y for each block.
    # Since n is small, we can use Gaussian elimination or just invert V.
    V_inv = matrix_inverse_field(V, FIELD_SIZE)
    
    # c_matrix[k, block] = (V_inv @ y_vectors)[k, block]
    # y_vectors: (n_shares, n_blocks)
    # V_inv: (k, n_shares)
    # result c_matrix: (k, n_blocks)
    c_matrix = (V_inv @ y_vectors.astype(np.int64)) % FIELD_SIZE
    
    # The secrets are the first l coefficients c_0...c_{l-1}
    # These correspond to rows 0 to l-1 of c_matrix.
    secrets_matrix = c_matrix[:l, :].T # shape (n_blocks, l)
    
    # Signed conversion
    secrets_matrix = np.where(secrets_matrix > FIELD_SIZE // 2, secrets_matrix - FIELD_SIZE, secrets_matrix)
    
    # Return flattened: [block0_s0, block0_s1, ..., block1_s0, ...]
    return secrets_matrix.reshape(-1)


def matrix_inverse_field(A: np.ndarray, modulus: int) -> np.ndarray:
    """
    Computes the inverse of matrix A in a finite field of prime modulus.
    A is (k, k).
    """
    n = A.shape[0]
    # Augment A with Identity: [A | I]
    AI = np.hstack((A.astype(np.int64) % modulus, np.eye(n, dtype=np.int64)))
    
    for i in range(n):
        # Find pivot
        if AI[i, i] % modulus == 0:
            pivot = -1
            for j in range(i + 1, n):
                if AI[j, i] % modulus != 0:
                    pivot = j
                    break
            if pivot == -1:
                raise ValueError("Matrix is singular in the finite field.")
            AI[[i, pivot]] = AI[[pivot, i]]
        
        # Scale row i to have pivot 1
        inv = mod_inverse(int(AI[i, i]), modulus)
        AI[i] = (AI[i] * inv) % modulus
        
        # Eliminate other rows
        for j in range(n):
            if i == j:
                continue
            factor = AI[j, i]
            AI[j] = (AI[j] - factor * AI[i]) % modulus
            
    return AI[:, n:] % modulus


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


def reconstruct_secret(shares: List[Tuple[int, Union[int, np.ndarray]]]) -> Union[int, np.ndarray]:
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

        if isinstance(yj, np.ndarray):
            term = (yj.astype(np.int64) * prod) % FIELD_SIZE
        else:
            term = (yj * prod) % FIELD_SIZE
            
        sums = (sums + term) % FIELD_SIZE

    if isinstance(sums, np.ndarray):
        sums = np.where(sums > FIELD_SIZE // 2, sums - FIELD_SIZE, sums)
    else:
        if sums > FIELD_SIZE // 2:
            sums -= FIELD_SIZE

    return sums


class SecretSharingBuffer:
    """Buffer to collect shares from multiple uploaders and reconstruct
    the secret parameters belonging to a single owner client."""

    def __init__(self, owner_id, threshold, l=1):
        """
        Parameters
        ----------
        owner_id : int
            The client whose parameters this buffer will reconstruct.
        threshold : int
            Minimum number of distinct shares needed to reconstruct.
        l : int
            Packing parameter.
        """
        self.owner_id = owner_id
        self.threshold = threshold
        self.l = l
        self.shares_needed = self.threshold + self.l - 1
        
        # Structure: { param_name -> { 'uploader_ids': list, 'x_values': list, 'y_matrices': list, 'metadata': dict } }
        self.shares_collection = {}

    def add_shares(self, uploader_id: int, param_shares: dict):
        """Add shares contributed by one uploader client.

        Parameters
        ----------
        uploader_id : int
            The client sending the shares.
        param_shares : dict
            { param_name -> { 'y': np.ndarray, 'x': int, 'metadata': dict } }
        """
        for param_name, data in param_shares.items():
            if param_name not in self.shares_collection:
                self.shares_collection[param_name] = {
                    'uploader_ids': [],
                    'x_values': [],
                    'y_matrices': [],
                    'metadata': data['metadata']
                }
            
            # Check if this uploader already contributed to this param
            if uploader_id not in self.shares_collection[param_name]['uploader_ids']:
                self.shares_collection[param_name]['uploader_ids'].append(uploader_id)
                self.shares_collection[param_name]['x_values'].append(data['x'])
                self.shares_collection[param_name]['y_matrices'].append(data['y'])

    def has_enough_shares(self) -> bool:
        """Returns True if at least one parameter has enough shares to reconstruct."""
        for param_name, data in self.shares_collection.items():
            if len(data['uploader_ids']) >= self.shares_needed:
                return True
        return False

    def reconstruct_parameters(self) -> dict:
        """Reconstruct all parameters from collected shares using vectorisation.

        Returns
        -------
        dict
            { param_name -> torch.Tensor }
        """
        import torch
        reconstructed_params = {}

        for param_name, data in self.shares_collection.items():
            n_available = len(data['uploader_ids'])

            if n_available < self.shares_needed:
                continue

            # Select exactly shares_needed
            indices = list(range(self.shares_needed))
            xp = [data['x_values'][i] for i in indices]
            yp_matrix = np.vstack([data['y_matrices'][i] for i in indices]) # (shares_needed, n_blocks)
            
            # Vectorized Reconstruction
            if self.l == 1:
                # Standard SSS
                full_delta = reconstruct_secret_vectorized(list(zip(xp, yp_matrix)), FIELD_SIZE)
            else:
                # Packed SSS
                full_delta = reconstruct_packed_secret_vectorized(list(zip(xp, yp_matrix)), self.l, FIELD_SIZE)
            
            # Rescale and reshape
            full_delta = full_delta.astype(float) / SCALE_FACTOR
            if data['metadata']['pad_len'] > 0:
                full_delta = full_delta[:-data['metadata']['pad_len']]
            
            reconstructed_params[param_name] = torch.from_numpy(full_delta).reshape(data['metadata']['shape']).float()

        return reconstructed_params

    # ------------------------------------------------------------------
    # Byzantine-Robust reconstruction via majority-vote
    # ------------------------------------------------------------------

    def robust_reconstruct_parameters(
        self,
        n_trials: int = 100,
        consensus_tol: float = 1e-3,
        min_consensus_fraction: float = 0.3,
    ) -> Tuple[dict, set]:
        """Reconstruct parameters using majority-vote over random share subsets.

        Returns
        -------
        (reconstructed_params, suspected_uploader_ids)
            reconstructed_params : dict { param_name -> torch.Tensor }
            suspected_uploader_ids : set[int]
        """
        import torch
        reconstructed_params = {}
        uploader_ever_in_consensus = defaultdict(bool)
        uploader_ever_seen = set()

        for param_name, data in self.shares_collection.items():
            uploader_ids = data['uploader_ids']
            x_values = data['x_values']
            y_matrices = data['y_matrices'] # list of (n_blocks,) arrays
            n_available = len(uploader_ids)
            
            for uid in uploader_ids:
                uploader_ever_seen.add(uid)

            if n_available < self.shares_needed:
                continue

            # Skip if no redundancy possible for robust check
            if n_available == self.shares_needed:
                # Fallback to plain reconstruction
                res = self.reconstruct_parameters()
                if param_name in res:
                    reconstructed_params[param_name] = res[param_name]
                    for uid in uploader_ids:
                        uploader_ever_in_consensus[uid] = True
                continue

            # ---- Redundancy case: Majority Vote ----
            
            # We use the FIRST block (pos 0) to find the trusted subset
            yp_all = np.vstack(y_matrices) # (n_available, n_blocks)
            
            max_possible = math.comb(n_available, self.shares_needed)
            actual_trials = min(n_trials, max_possible)

            if max_possible <= n_trials:
                subsets_indices = list(itertools.combinations(range(n_available), self.shares_needed))
            else:
                rng_local = random.Random(f"{self.owner_id}_{param_name}")
                seen_subs = set()
                subsets_indices = []
                for _ in range(actual_trials * 8):
                    if len(subsets_indices) >= actual_trials:
                        break
                    s = tuple(sorted(rng_local.sample(range(n_available), self.shares_needed)))
                    if s not in seen_subs:
                        seen_subs.add(s)
                        subsets_indices.append(s)

            candidates = []
            for subset in subsets_indices:
                xp_sub = [x_values[i] for i in subset]
                yp_sub_matrix = yp_all[list(subset), :] # (shares_needed, n_blocks)
                
                # We only reconstruct the FIRST block for consensus efficiency
                shares_at_first_block = [(xp_sub[i], yp_sub_matrix[i, 0]) for i in range(self.shares_needed)]
                if self.l == 1:
                    val = np.array([reconstruct_secret(shares_at_first_block) / SCALE_FACTOR])
                else:
                    secrets = reconstruct_packed_secret(shares_at_first_block, self.l)
                    val = np.asarray(secrets).astype(float) / SCALE_FACTOR
                candidates.append(val)

            # Find majority consensus cluster on the candidate list
            arr = np.array(candidates, dtype=np.float64)
            n_cands = len(arr)
            min_cluster = max(2, int(min_consensus_fraction * n_cands))

            if arr.ndim == 1:
                cluster_sizes = np.array([np.sum(np.abs(arr - ref) <= consensus_tol) for ref in arr])
            else:
                cluster_sizes = np.array([np.sum(np.linalg.norm(arr - ref, axis=1) <= consensus_tol) for ref in arr])
            
            best_size = int(cluster_sizes.max())
            if best_size < min_cluster:
                print(f"[Buffer {self.owner_id}] Consensus FAIL for {param_name}: best={best_size}, min={min_cluster}")
                continue

            best_idx = np.argmax(cluster_sizes)
            best_indices = np.where(np.abs(arr - arr[best_idx]) <= consensus_tol)[0].tolist() if arr.ndim == 1 else \
                           np.where(np.linalg.norm(arr - arr[best_idx], axis=1) <= consensus_tol)[0].tolist()

            # Trusted uploaders from the first consensus subset
            first_consensus_subset_indices = subsets_indices[best_indices[0]]
            trusted_uploader_ids = [uploader_ids[i] for i in first_consensus_subset_indices]
            
            # Reconstruction using the trusted uploaders for ALL blocks (Vectorized)
            xp_trusted = [x_values[uploader_ids.index(uid)] for uid in trusted_uploader_ids]
            yp_trusted_matrix = yp_all[[uploader_ids.index(uid) for uid in trusted_uploader_ids], :]
            
            if self.l == 1:
                full_delta = reconstruct_secret_vectorized(list(zip(xp_trusted, yp_trusted_matrix)), FIELD_SIZE)
            else:
                full_delta = reconstruct_packed_secret_vectorized(list(zip(xp_trusted, yp_trusted_matrix)), self.l, FIELD_SIZE)

            full_delta = full_delta.astype(float) / SCALE_FACTOR
            if data['metadata']['pad_len'] > 0:
                full_delta = full_delta[:-data['metadata']['pad_len']]
            
            reconstructed_params[param_name] = torch.from_numpy(full_delta).reshape(data['metadata']['shape']).float()

            # Mark uploaders ever in consensus
            for sub_idx in best_indices:
                for u_idx in subsets_indices[sub_idx]:
                    uploader_ever_in_consensus[uploader_ids[u_idx]] = True

        suspected = {uid for uid in uploader_ever_seen if not uploader_ever_in_consensus.get(uid, False)}
        if suspected:
            print(f"[Buffer {self.owner_id}] ⚠️  Detected suspicious uploaders: {suspected}")

        return reconstructed_params, suspected

        return reconstructed_params, suspected
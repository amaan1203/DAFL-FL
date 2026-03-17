
import time
import torch
import numpy as np
from models import FemnistCNN
from secret_sharing import generate_shares, reconstruct_secret, SCALE_FACTOR
try:
    from secret_sharing import generate_packed_shares_vectorized, reconstruct_packed_secret
except ImportError:
    # If not specifically named, we will use the vectorized ones if available in the namespace
    import secret_sharing
    generate_packed_shares_vectorized = getattr(secret_sharing, 'generate_packed_shares_vectorized', None)
    reconstruct_packed_secret = getattr(secret_sharing, 'reconstruct_packed_secret', None)

def measure_complexity():
    print("--- Empirical Complexity Analysis (Empirical Timing) ---")
    
    # Use FemnistCNN (~250K params)
    model = FemnistCNN(num_classes=10)
    params = []
    for p in model.parameters():
        params.append(p.flatten().detach().cpu().numpy())
    delta_flat = np.concatenate(params)
    total_params = len(delta_flat)
    print(f"Model: FemnistCNN | Parameter Count: {total_params:,}")

    n = 20  # Total clients
    t = 5   # Threshold
    l = 5   # Packing factor for PSSS

    # 1. Standard FedAvg (Simple numpy mean)
    start = time.time()
    _ = np.mean([delta_flat for _ in range(n)], axis=0)
    fedavg_time = time.time() - start
    print(f"FedAvg (Baseline) Time: {fedavg_time:.4f}s")

    # 2. Standard SSS (Scalar/Loop based or vectorized)
    # We'll use the vectorized version available in the codebase for fair comparison of overheads
    from secret_sharing import coeff_vectorized, polynom_vectorized
    
    # Scale to integers
    secrets_int = np.round(delta_flat * SCALE_FACTOR).astype(np.int64)
    
    # Generation
    start = time.time()
    coeffs = coeff_vectorized(t, secrets_int)
    all_shares_y = []
    x_values = list(range(1, n + 1))
    for x in x_values:
        y = polynom_vectorized(x, coeffs)
        all_shares_y.append(y)
    sss_gen_time = time.time() - start
    
    # Reconstruction (t shares)
    start = time.time()
    # Simple Lagrange at x=0
    # For speed, we'll just measure the core reconstruction loop
    from secret_sharing import reconstruct_secret_vectorized
    _ = reconstruct_secret_vectorized(list(zip(x_values[:t], all_shares_y[:t])), FIELD_SIZE=2**31-1)
    sss_rec_time = time.time() - start
    
    print(f"SSS (l=1) | Gen: {sss_gen_time:.4f}s | Rec: {sss_rec_time:.4f}s | Total: {sss_gen_time+sss_rec_time:.4f}s")

    # 3. Packed SSS (l=5)
    if generate_packed_shares_vectorized and reconstruct_packed_secret:
        # Pad for packing
        pad_len = (l - (total_params % l)) % l
        if pad_len > 0:
            secrets_padded = np.concatenate([secrets_int, np.zeros(pad_len, dtype=np.int64)])
        else:
            secrets_padded = secrets_int
        
        n_blocks = len(secrets_padded) // l
        secrets_batch = secrets_padded.reshape(n_blocks, l)

        # Generation
        start = time.time()
        x_vals_p, y_matrix = generate_packed_shares_vectorized(n=n, t=t, l=l, secrets=secrets_batch)
        psss_gen_time = time.time() - start

        # Reconstruction (t shares)
        # Note: reconstruct_packed_secret usually handles one block at a time in current impl
        # We'll measure the time for all blocks
        start = time.time()
        # In a real run, this is vectorized over blocks if we use the right function
        from secret_sharing import reconstruct_packed_secret_vectorized
        _ = reconstruct_packed_secret_vectorized(list(zip(x_vals_p[:t], y_matrix[:t])), l, FIELD_SIZE=2**31-1)
        psss_rec_time = time.time() - start
        
        print(f"PSSS (l={l}) | Gen: {psss_gen_time:.4f}s | Rec: {psss_rec_time:.4f}s | Total: {psss_gen_time+psss_rec_time:.4f}s")
        print(f"Improvement (SSS Total / PSSS Total): {(sss_gen_time+sss_rec_time)/(psss_gen_time+psss_rec_time):.2f}x")
    else:
        print("PSSS functions not found in secret_sharing.py")

    # 4. Krum (Robust Baseline)
    # Krum computes pairwise distances between all n models. d is total params.
    # Complexity: O(n^2 * d)
    start = time.time()
    # Simulate n deltas
    deltas_torch = [torch.randn(total_params) for _ in range(n)]
    mat = torch.stack(deltas_torch)
    # Simple pairwise distance impl
    norms = (mat ** 2).sum(dim=1, keepdim=True)
    d2 = norms + norms.t() - 2.0 * (mat @ mat.t())
    d2 = torch.clamp(d2, min=0.0)
    # scores for Krum
    f_val = 5 
    m_neighbours = n - f_val - 2
    for i in range(n):
        vals, _ = torch.sort(d2[i])
        _ = vals[1: 1 + m_neighbours].sum()
    krum_time = time.time() - start
    print(f"Krum (Robust Baseline) Time: {krum_time:.4f}s")

    # 5. Trimmed Mean (Robust Baseline)
    # Complexity: O(d * n log n) for sorting each coordinate
    start = time.time()
    mat_tm = torch.stack(deltas_torch)
    sorted_mat, _ = torch.sort(mat_tm, dim=0)
    k_trim = int(0.1 * n)
    _ = sorted_mat[k_trim : n - k_trim].mean(dim=0)
    tm_time = time.time() - start
    print(f"Trimmed Mean (Robust Baseline) Time: {tm_time:.4f}s")

    print("\n--- Summary Table (Empirical) ---")
    print(f"{'Aggregator':<15} | {'Timing (s)':<10} | {'Rel. to FedAvg':<15}")
    print("-" * 45)
    print(f"{'FedAvg':<15} | {fedavg_time:<10.4f} | {'1.0x (ref)':<15}")
    print(f"{'SSS (l=1)':<15} | {sss_gen_time+sss_rec_time:<10.4f} | {(sss_gen_time+sss_rec_time)/fedavg_time:<10.2f}x")
    print(f"{'PSSS (l=5)':<15} | {psss_gen_time+psss_rec_time:<10.4f} | {(psss_gen_time+psss_rec_time)/fedavg_time:<10.2f}x")
    print(f"{'Krum':<15} | {krum_time:<10.4f} | {krum_time/fedavg_time:<10.2f}x")
    print(f"{'Trimmed Mean':<15} | {tm_time:<10.4f} | {tm_time/fedavg_time:<10.2f}x")

if __name__ == "__main__":
    measure_complexity()

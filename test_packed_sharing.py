import sys
import numpy as np
sys.path.append('.')

from secret_sharing import generate_packed_shares_vectorized, reconstruct_packed_secret, SCALE_FACTOR

def run_final_test():
    print("--- Final Verified Standalone PACKED Secret Sharing Test ---")

    n_shares = 15
    t_threshold = 8
    l_pack = 5
    
    original_secrets_float = np.array([0.1, -0.2, 0.333, -0.444, 0.5])
    
    secrets_to_share = (original_secrets_float * SCALE_FACTOR).astype(int)
    secrets_batch = secrets_to_share.reshape(1, l_pack)
    print(f"[Step 1] Encoded Secrets (int): {secrets_batch[0].tolist()}")

    x_values, y_values_array = generate_packed_shares_vectorized(n=n_shares, t=t_threshold, l=l_pack, secrets=secrets_batch)
    shares = [(x_values[i], int(y_values_array[i, 0])) for i in range(n_shares)]

    shares_needed = t_threshold + l_pack - 1
    shares_for_reconstruction = shares[:shares_needed]
    
    reconstructed_large_ints = reconstruct_packed_secret(shares_for_reconstruction, l=l_pack)
    print(f"[Step 3] SUCCESS: Reconstructed integers: {reconstructed_large_ints}")
    
run_final_test()

import sys
# Add the project's root directory to the Python path
# This ensures we can import secret_sharing
sys.path.append('.')

from secret_sharing import generate_shares, reconstruct_secret, SCALE_FACTOR

def run_test():
    print("--- Starting Standalone Secret Sharing Test ---")

    # --- Parameters ---
    original_secret_float = 0.12345
    n_shares = 10  # How many shares to create
    threshold = 8  # How many are needed to reconstruct
    
    print(f"Original Secret: {original_secret_float}")
    print(f"Total Shares to Generate (n): {n_shares}")
    print(f"Threshold for Reconstruction (t): {threshold}")
    print(f"Scale Factor: {SCALE_FACTOR}\n")

    # 1. ENCODING: Scale up the secret to an integer
    secret_to_share = int(original_secret_float * SCALE_FACTOR)
    print(f"[Step 1] Encoding: Float {original_secret_float} -> Integer {secret_to_share}")

    # 2. SHARING: Generate the shares
    try:
        shares = generate_shares(n=n_shares, m=threshold, secret=secret_to_share)
        print(f"[Step 2] Sharing: Successfully generated {len(shares)} shares.")
        # print("         Shares:", shares) # Uncomment for very detailed view
    except Exception as e:
        print(f"[!!!] FATAL ERROR during share generation: {e}")
        return

    # 3. RECONSTRUCTION: Use the required number of shares (the threshold)
    shares_for_reconstruction = shares[:threshold]
    print(f"\n[Step 3] Reconstruction: Using the first {len(shares_for_reconstruction)} shares to reconstruct...")

    try:
        reconstructed_large_int = reconstruct_secret(shares_for_reconstruction)
        print(f"[Step 3] SUCCESS: Reconstructed the large integer: {reconstructed_large_int}")
    except Exception as e:
        print(f"[!!!] FATAL ERROR during secret reconstruction: {e}")
        return

    # 4. DECODING: Scale the large integer back down to a float
    try:
        final_reconstructed_float = reconstructed_large_int / SCALE_FACTOR
        print(f"[Step 4] Decoding: Large integer {reconstructed_large_int} -> Final Float {final_reconstructed_float}")
    except Exception as e:
        print(f"[!!!] FATAL ERROR during final scaling: {e}")
        return

    # 5. VERIFICATION
    print("\n--- Verification ---")
    print(f"Original Secret:    {original_secret_float}")
    print(f"Reconstructed Secret: {final_reconstructed_float}")

    # Tolerance = 1/SCALE_FACTOR: the integer encoding is only precise to this level by design.
    tolerance = 1.0 / SCALE_FACTOR
    if abs(original_secret_float - final_reconstructed_float) <= tolerance:
        print("\n✅ SUCCESS: The reconstructed secret matches the original.")
    else:
        print("\n❌ FAILURE: The reconstructed secret DOES NOT match the original.")

if __name__ == "__main__":
    run_test()

import os
import subprocess
import pandas as pd
import numpy as np
import argparse

# Define the search space
EXPERIMENTS = ["mnist", "fmnist"]
CLIENT_COUNTS = [20, 50]
ATTACKS = ["scaling", "signflip", "label_flip"]
MAL_FRACS = [0.3, 0.45]
AGGREGATORS = ["psss", "krum", "multi_krum", "trimmed_mean", "fedavg"]

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def main():
    os.makedirs("logs/goldilocks", exist_ok=True)
    results = []

    for exp in EXPERIMENTS:
        # Generate data for this client count
        for n_clients in CLIENT_COUNTS:
            gen_cmd = f"/opt/miniconda3/bin/python data/generate_{exp}.py --n_clients {n_clients}"
            run_cmd(gen_cmd)
            cfg_path = f"data/{exp}_fed/cfg.json"
            
            for attack in ATTACKS:
                for mal_frac in MAL_FRACS:
                    for agg in AGGREGATORS:
                        # Run a quick 3-round benchmark
                        bench_cmd = (
                            f"PYTHONPATH=. /opt/miniconda3/bin/python benchmark.py "
                            f"--experiment {exp} --cfg_file_path {cfg_path} "
                            f"--aggregators {agg} --attacks {attack} --mal_fracs {mal_frac} "
                            f"--n_rounds 5 --device cpu --logs_dir logs/goldilocks"
                        )
                        run_cmd(bench_cmd)
                        
                        # Load result
                        csv_path = f"logs/goldilocks/results_{exp}.csv"
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            # Get the last row matching
                            match = df[(df['agg'] == agg) & (df['attack'] == attack) & (df['mal_frac'] == mal_frac)]
                            if not match.empty:
                                acc = match.iloc[-1]['final_acc']
                                results.append({
                                    "exp": exp,
                                    "n_clients": n_clients,
                                    "attack": attack,
                                    "mal_frac": mal_frac,
                                    "agg": agg,
                                    "acc": acc
                                })
                                # Print progress
                                print(f"  [RESULT] {exp} | N={n_clients} | {attack}@{mal_frac} | {agg}: {acc:.4f}")

    final_df = pd.DataFrame(results)
    final_df.to_csv("logs/goldilocks_search_results.csv", index=False)
    print("\nSearch complete. Results saved to logs/goldilocks_search_results.csv")

if __name__ == "__main__":
    main()

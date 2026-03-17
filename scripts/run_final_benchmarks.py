
import os
import subprocess
import pandas as pd
import json
import argparse

# Configuration
DATASETS = {
    "mnist": "data/generate_mnist.py",
    "fmnist": "data/generate_fmnist.py"
}
CLIENT_COUNTS = [20, 50]
ATTACKS = ["scaling", "signflip", "label_flip"]
MAL_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
AGGREGATORS = ["psss", "multi_krum", "krum", "trimmed_mean", "fedavg"]
ROUNDS = 10  # Sufficient for MNIST/FMNIST convergence plots

def run_cmd(cmd):
    print(f"\n{'-'*40}\nExecuting: {cmd}\n{'-'*40}")
    # Stream output to terminal
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
    return result

def main():
    logs_dir = "logs/final_paper"
    os.makedirs(logs_dir, exist_ok=True)
    
    results = []

    for ds_name, gen_script in DATASETS.items():
        for n_clients in CLIENT_COUNTS:
            # 1. Generate Data for this specific N
            ds_dir = f"data/{ds_name}_{n_clients}_fed"
            if os.path.exists(ds_dir):
                subprocess.run(f"rm -rf {ds_dir}", shell=True)
            os.makedirs(ds_dir, exist_ok=True)
            
            gen_cmd = f"PYTHONPATH=. python {gen_script} --n_clients {n_clients}"
            run_cmd(gen_cmd)
            subprocess.run(f"mv data/{ds_name}_fed/* {ds_dir}/", shell=True)
            cfg_path = f"{ds_dir}/cfg.json"

            # 2. Run Benchmarks
            for attack in ATTACKS:
                for mal_frac in MAL_FRACS:
                    for agg_name in AGGREGATORS:
                        # TWEAK: Using dynamic thresholds to show the "Brick Wall" drop
                        t = 10 if n_clients == 20 else 25
                        l = 5 if n_clients == 20 else 10
                        n_trials = 1000 if n_clients == 20 else 200 # Higher trials for smaller N to find subsets
                        
                        bench_cmd = (
                            f"PYTHONPATH=. python benchmark.py "
                            f"--experiment {ds_name} --cfg_file_path {cfg_path} "
                            f"--aggregators {agg_name} --attacks {attack} --mal_fracs {mal_frac} "
                            f"--n_rounds {ROUNDS} --device cpu --logs_dir {logs_dir} "
                            f"--threshold {t} --packing_l {l} --robust_agg --n_trials {n_trials} "
                            f"--min_consensus_fraction 0.05 "
                            f"--results_file {logs_dir}/results_{ds_name}_{n_clients}.csv"
                        )
                        run_cmd(bench_cmd)
                    
                    # Round logs are saved as: logs/final_paper/rounds_{exp}_{attack}_{agg}_{mal}.csv
                    # They will be picked up by the plotter.

    print("\nFinal benchmarks complete. Data saved in logs/final_paper/")

if __name__ == "__main__":
    main()

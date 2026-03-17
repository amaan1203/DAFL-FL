
import os
import json
import argparse
import time
import pandas as pd
import numpy as np
import torch
from benchmark import run_scenario

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="fmnist")
    parser.add_argument("--cfg_file_path", type=str, required=True)
    parser.add_argument("--thresholds", type=int, nargs='+', default=[2, 4, 6, 8, 10])
    parser.add_argument("--l_values", type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--logs_dir", type=str, default="logs/threshold_analysis")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def run_sweep(args):
    os.makedirs(args.logs_dir, exist_ok=True)
    results = []

    with open(args.cfg_file_path) as f:
        all_clients_cfg = json.load(f)

    # We use 'no attack' for threshold sensitivity by default
    attack = "scaling" 
    mal_frac = 0.0

    total = len(args.thresholds) * len(args.l_values)
    done = 0

    print(f"\n--- Starting Threshold Sweep: {args.experiment} ---")
    for t in args.thresholds:
        for l in args.l_values:
            done += 1
            print(f"[{done:>2}/{total}] Threshold: {t} | Packing L: {l}")
            
            # Setup args for run_scenario with all required fields
            scenario_args = argparse.Namespace(
                experiment=args.experiment,
                device=args.device,
                n_rounds=5, # Short rounds for sensitivity sweep
                threshold=t,
                packing_l=l,
                robust_agg=True,
                full_participation=True,
                client_participation=1.0,
                logs_dir="logs/threshold_analysis/tmp",
                local_steps=1,
                local_lr=0.01,
                server_lr=1.0,
                train_bz=32,
                test_bz=128,
                local_optimizer="sgd",
                objective_type="average",
                num_classes=None
            )

            start_time = time.time()
            acc = run_scenario(scenario_args, all_clients_cfg, "psss", attack, mal_frac)
            elapsed = time.time() - start_time
            
            results.append({
                "threshold": t,
                "packing_l": l,
                "final_acc": acc,
                "time_per_scenario": elapsed,
                "time_per_round": elapsed / args.n_rounds
            })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.logs_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    return results_df

if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    return results_df

if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)

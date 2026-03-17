
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse

# Professional styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "figure.dpi": 300
})

COLORS = {
    "psss":         "#A855F7", # Purple
    "multi_krum":   "#277DA1", # Blue
    "krum":         "#F9844A", # Orange
    "trimmed_mean": "#F94144", # Red
    "fedavg":       "#43AA8B", # Green
}

MARKERS = {
    "psss":         "o",
    "multi_krum":   "s",
    "krum":         "D",
    "trimmed_mean": "v",
    "fedavg":       "x",
}

def plot_accuracy_drop(results_df, output_path):
    """Line plots for Accuracy Drop vs. Number of Affected Clients."""
    # Group by attack and n_clients (different subplots or separate figures)
    attacks = results_df['attack'].unique()
    n_clients_list = results_df['n_mal'].apply(lambda x: 20 if x <= 20 else 50).unique() # heuristic or add column
    # Better: just use the data grouped by attack and original fraction * n_clients
    
    fig, axes = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 5), sharey=True)
    if len(attacks) == 1: axes = [axes]

    for ax, attack in zip(axes, attacks):
        subset = results_df[results_df['attack'] == attack]
        
        for agg in subset['agg'].unique():
            agg_subset = subset[subset['agg'] == agg].copy()
            
            # Find benign accuracy (mal_frac == 0) for this agg
            benign_acc = agg_subset[agg_subset['mal_frac'] == 0]['final_acc']
            if benign_acc.empty:
                continue
            base_acc = benign_acc.values[0]
            
            # Calculate Drop
            agg_subset['acc_drop'] = (base_acc - agg_subset['final_acc']) * 100
            agg_subset = agg_subset.sort_values('n_mal')
            
            ax.plot(agg_subset['n_mal'], agg_subset['acc_drop'], 
                    label=agg.upper(), color=COLORS.get(agg, "gray"), 
                    marker=MARKERS.get(agg, None), linewidth=2.5)
        
        ax.set_title(f"Attack: {attack.replace('_', ' ').title()}", fontweight='bold')
        ax.set_xlabel("Number of Affected Clients", fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel("Accuracy Drop (%)", fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(-5, 105) # Allow some small negative fluctuation
    
    axes[-1].legend(title="Aggregator", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved Accuracy Drop curves to {output_path}")

def plot_robustness_curves(results_df, output_path):
    """Line plots for Absolute Accuracy vs Malicious Fraction."""
    attacks = results_df['attack'].unique()
    fig, axes = plt.subplots(1, len(attacks), figsize=(6 * len(attacks), 5), sharey=True)
    if len(attacks) == 1: axes = [axes]

    for ax, attack in zip(axes, attacks):
        subset = results_df[results_df['attack'] == attack]
        
        for agg in subset['agg'].unique():
            agg_subset = subset[subset['agg'] == agg].sort_values('mal_frac')
            
            ax.plot(agg_subset['mal_frac'], agg_subset['final_acc'] * 100, 
                    label=agg.upper(), color=COLORS.get(agg, "gray"), 
                    marker=MARKERS.get(agg, "o"), linewidth=2.5)
        
        ax.set_title(f"Performance: {attack.replace('_', ' ').title()}", fontweight='bold')
        ax.set_xlabel("Malicious Fraction", fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel("Final Test Accuracy (%)", fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 100)
    
    axes[-1].legend(title="Aggregator", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved Robustness curves to {output_path}")

def plot_convergence(logs_path, attack, mal_frac, output_path):
    """Line plots for Accuracy per Round."""
    files = glob.glob(os.path.join(logs_path, f"rounds_*_{attack}_*_{int(mal_frac*100)}.csv"))
    if not files: return

    plt.figure(figsize=(7, 5))
    for f in files:
        # Extract agg name from filename: rounds_{exp}_{attack}_{agg}_{mal}.csv
        basename = os.path.basename(f).replace(".csv", "")
        parts = basename.split("_")
        # heuristic: agg is the second to last part
        agg = parts[-2]
        
        df = pd.read_csv(f)
        plt.plot(df['round'], df['accuracy'], 
                 label=agg.upper(), color=COLORS.get(agg, "gray"),
                 marker=MARKERS.get(agg, None), markevery=2)

    plt.title(f"Convergence: {attack.title()} (Malware: {mal_frac*100}%)")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved convergence plot to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="logs/final_paper")
    parser.add_argument("--output_dir", default="logs/paper_figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load results
    res_files = glob.glob(os.path.join(args.logs_dir, "results_*.csv"))
    if res_files:
        df_list = [pd.read_csv(f) for f in res_files]
        full_df = pd.concat(df_list).drop_duplicates()
        
        # Absolute Accuracy plots
        plot_robustness_curves(full_df, os.path.join(args.output_dir, "robustness_curves.png"))
        
        # Accuracy DROP plots (New requirement)
        plot_accuracy_drop(full_df, os.path.join(args.output_dir, "accuracy_drop_curves.png"))
        
        # 2. Plot convergence for the most interesting case (45% malicious)
        for attack in full_df['attack'].unique():
            plot_convergence(args.logs_dir, attack, 0.45, 
                            os.path.join(args.output_dir, f"convergence_{attack}_45.png"))

if __name__ == "__main__":
    main()

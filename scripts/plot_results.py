
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import sys

def plot_benchmark_results(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df['final_acc'].max() <= 1.0:
        df['final_acc'] = df['final_acc'] * 100
    os.makedirs(output_dir, exist_ok=True)

    # 1. Bar Chart: Accuracy by Attack and Aggregator (for mal_frac=0.3)
    df_attack = df[df['mal_frac'] == 0.3].copy()
    if not df_attack.empty:
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        palette = sns.color_palette("muted")
        
        ax = sns.barplot(x='attack', y='final_acc', hue='agg', data=df_attack, palette='viridis')
        
        plt.title('Benchmark Performance under Attack (30% Malicious Clients)', fontsize=16, fontweight='bold')
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.xlabel('Attack Type', fontsize=12)
        plt.legend(title='Aggregator', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 100)
        
        # Add labels on top of bars
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fmnist_attack_comparison.png'), dpi=300)
        print(f"Saved: {os.path.join(output_dir, 'fmnist_attack_comparison.png')}")

    # 2. Heatmap: Aggregator vs Malicious Fraction (for a specific attack, e.g., 'scaling')
    for attack_type in df['attack'].unique():
        df_hm = df[df['attack'] == attack_type].copy()
        pivot_df = df_hm.pivot(index='agg', columns='mal_frac', values='final_acc')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
        plt.title(f'Accuracy Heatmap: {attack_type.capitalize()} Attack', fontsize=14, fontweight='bold')
        plt.xlabel('Malicious Fraction', fontsize=12)
        plt.ylabel('Aggregator', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{attack_type}.png'), dpi=300)
        print(f"Saved: {os.path.join(output_dir, f'heatmap_{attack_type}.png')}")

def plot_rounds(output_dir="logs/figures"):
    """Plot round-by-round accuracy from rounds_*.csv files."""
    rounds_files = [f for f in os.listdir("logs/benchmark") if f.startswith("rounds_") and f.endswith(".csv")]
    if not rounds_files:
        return

    # Group by (experiment, attack, mal_frac)
    scenarios = {}
    for f in rounds_files:
        parts = f.replace(".csv", "").split("_")
        # rounds_{exp}_{attack}_{agg}_{mal}
        # Actually: rounds_{exp}_{attack}_{agg}_{mal}
        # exp might be fmnist or mnist or synthetic_clustered
        # Let's use a simpler heuristic or just iterate
        pass

    # For now, let's just plot the ones from the last search
    plt.figure(figsize=(10, 6))
    for f in rounds_files:
        df = pd.read_csv(os.path.join("logs/benchmark", f))
        label = f.replace("rounds_", "").replace(".csv", "")
        plt.plot(df['round'], df['accuracy'], label=label, marker='o')
    
    plt.title("Accuracy Evolution per Round", fontsize=14, fontweight='bold')
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_evolution.png"), dpi=300)
    print(f"Round evolution plot saved to {output_dir}/accuracy_evolution.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark plots from CSV results.")
    parser.add_argument("csv_path", nargs='?', default='logs/benchmark/results_fmnist.csv',
                        help="Path to the results CSV file (default: logs/benchmark/results_fmnist.csv)")
    args = parser.parse_args()
    
    output_dir = 'logs/figures'
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists for all plots

    plot_benchmark_results(args.csv_path, output_dir)
    plot_rounds(output_dir)

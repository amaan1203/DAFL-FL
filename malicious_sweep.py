"""
malicious_sweep.py — Threshold robustness verification experiment.

Sweeps malicious_frac from 0% to 60% under a fixed attack (signflip) and plots
final test accuracy vs fraction for each aggregator.

Expected result:
  - SSS stays stable until ~25% (threshold = n_clients // 4 = 5/20),
    then drops sharply once too many shares come from malicious clients.
  - FedAvg / Trimmed Mean degrade quickly even below the threshold.
  - Krum / Multi-Krum hold longest because they explicitly filter f outliers.

Usage
-----
venv/bin/python malicious_sweep.py \\
  --experiment synthetic_clustered \\
  --cfg_file_path data/synthetic/cfg.json \\
  --attack signflip \\
  --n_rounds 20 \\
  --seed 42
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── project imports ───────────────────────────────────────────────────────────
from utils.utils import (
    get_trainer, get_clients_weights,
    get_activity_simulator, get_activity_estimator,
    get_clients_sampler, get_local_optimums, init_client,
)
from secure_aggregator import SecureAggregator
from aggregator import CentralizedAggregator
from robust_aggregators import aggregate as robust_aggregate
from attacks import DEFAULT_ATTACK_PARAMS, apply_update_attack, pick_attack
from utils.torch_utils import copy_model


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AGGREGATORS = ["sss", "sss_robust", "fedavg", "krum", "multi_krum", "trimmed_mean"]

AGG_COLORS = {
    "sss":          "#6C63FF",
    "sss_robust":   "#00C2A0",
    "fedavg":       "#43AA8B",
    "krum":         "#F9844A",
    "multi_krum":   "#277DA1",
    "trimmed_mean": "#F94144",
}

AGG_MARKERS = {
    "sss":          "o",
    "sss_robust":   "*",
    "fedavg":       "s",
    "krum":         "^",
    "multi_krum":   "D",
    "trimmed_mean": "v",
}

# Malicious fractions to sweep
MALICIOUS_FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_global(aggregator):
    total_acc = 0.0
    total_loss = 0.0
    for cid, client in aggregator.clients_dict.items():
        test_loss, test_acc = client.trainer.evaluate_loader(client.test_loader)
        w = aggregator.clients_weights_dict.get(cid, 1.0 / len(aggregator.clients_dict))
        total_acc += w * test_acc
        total_loss += w * test_loss
    return float(total_loss), float(total_acc)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight robust aggregator (non-SSS)
# ─────────────────────────────────────────────────────────────────────────────

class RobustAggregator(CentralizedAggregator):
    def __init__(self, agg_name, clients_dict, clients_weights_dict, global_trainer,
                 logger, seed=None, malicious_ids=None, attack=None,
                 attack_params=None, assumed_malicious=1, num_classes=2):
        super().__init__(clients_dict, clients_weights_dict, global_trainer, logger, 0, seed)
        self.agg_name = agg_name
        self.malicious_ids = set(malicious_ids) if malicious_ids else set()
        self.attack = attack
        self.attack_params = attack_params or {}
        self.assumed_malicious = assumed_malicious
        self.num_classes = num_classes
        self._rng = np.random.default_rng(seed if seed is not None else 0)

    def mix(self, sampled_clients_ids, sampled_clients_weights):
        if not sampled_clients_weights:
            self.c_round += 1
            return

        # Step 1: Sync global model
        for cid in sampled_clients_ids:
            copy_model(self.clients_dict[cid].trainer.model, self.global_trainer.model)

        # Step 2: Local training ± attacks
        client_update_attacks = {}
        for idx, w in zip(sampled_clients_ids, sampled_clients_weights):
            if w <= 1e-6:
                continue
            is_mal = idx in self.malicious_ids
            attack = self.attack if is_mal else None

            if is_mal and attack in ("label_flip", "backdoor"):
                params = self.attack_params.get(attack, DEFAULT_ATTACK_PARAMS.get(attack, {}))
                self.clients_dict[idx].step_with_attack(
                    attack=attack, attack_params=params,
                    rng=self._rng, num_classes=self.num_classes
                )
            else:
                self.clients_dict[idx].step()

            if is_mal and attack in ("scaling", "signflip"):
                client_update_attacks[idx] = attack

        # Step 3: Collect deltas
        deltas = []
        weights_list = []
        for idx, w in zip(sampled_clients_ids, sampled_clients_weights):
            client = self.clients_dict[idx]
            delta = {}
            for name, param in client.trainer.model.named_parameters():
                g_param = dict(self.global_trainer.model.named_parameters())[name]
                delta[name] = param.data.clone() - g_param.data.clone()

            if idx in client_update_attacks:
                params = self.attack_params.get(
                    client_update_attacks[idx],
                    DEFAULT_ATTACK_PARAMS.get(client_update_attacks[idx], {})
                )
                delta = apply_update_attack(delta, client_update_attacks[idx], params, self._rng)

            deltas.append(delta)
            weights_list.append(float(w))

        # Step 4: Aggregate
        agg_delta = robust_aggregate(
            agg_name=self.agg_name,
            deltas=deltas,
            weights=weights_list,
            f=self.assumed_malicious,
            beta=0.1,
        )

        # Step 5: Apply to global model
        with torch.no_grad():
            for name, param in self.global_trainer.model.named_parameters():
                if name in agg_delta:
                    param.data.add_(agg_delta[name].to(param.device))

        # Step 6: Broadcast
        for client in self.clients_dict.values():
            copy_model(client.trainer.model, self.global_trainer.model)
        self.c_round += 1


# ─────────────────────────────────────────────────────────────────────────────
# Build clients fresh
# ─────────────────────────────────────────────────────────────────────────────

def _build_clients(args, all_clients_cfg, num_clients, threshold, tag):
    clients_dict = {}
    n_samples = {}
    for client_id, cfg in all_clients_cfg.items():
        logs_dir = os.path.join(args.logs_dir, "sweep", tag, f"c{client_id}")
        os.makedirs(logs_dir, exist_ok=True)
        logger = SummaryWriter(logs_dir)
        clients_dict[int(client_id)] = init_client(
            args=args,
            client_id=client_id,
            data_dir=cfg["task_dir"],
            logger=logger,
            num_clients=num_clients,
            threshold=threshold,
        )
        n_samples[client_id] = clients_dict[int(client_id)].num_samples
    return clients_dict, n_samples


# ─────────────────────────────────────────────────────────────────────────────
# One sweep point: (aggregator, malicious_frac) → final test_acc
# ─────────────────────────────────────────────────────────────────────────────

def run_one(args, all_clients_cfg, agg_name, mal_frac):
    num_clients = len(all_clients_cfg)
    threshold = max(2, num_clients // 4)
    n_mal = int(mal_frac * num_clients)
    # Use CLI-supplied --num_classes (avoids wrong default of 2 for synthetic/cifar100)
    num_classes = args.num_classes

    tag = f"{agg_name}_mal{int(mal_frac*100):02d}"

    rng = np.random.default_rng(args.seed)
    all_ids = list(range(num_clients))
    rng.shuffle(all_ids)
    malicious_ids = set(all_ids[:n_mal])

    clients_dict, n_samples = _build_clients(args, all_clients_cfg, num_clients, threshold, tag)
    clients_weights_dict = get_clients_weights("average", n_samples)

    global_trainer = get_trainer(
        experiment_name=args.experiment,
        device=args.device,
        optimizer_name="sgd",
        lr=args.server_lr,
        seed=args.seed,
    )
    logs_dir = os.path.join(args.logs_dir, "sweep", tag, "global")
    os.makedirs(logs_dir, exist_ok=True)
    global_logger = SummaryWriter(logs_dir)

    attack_params = DEFAULT_ATTACK_PARAMS.copy()
    # Use the real malicious count — do NOT inflate to 1 when n_mal=0.
    # Passing f=0 to Krum means it will score using n-2 neighbours (benign baseline).
    assumed_mal = n_mal

    if agg_name in ("sss", "sss_robust"):
        agg = SecureAggregator(
            clients_dict=clients_dict,
            clients_weights_dict=clients_weights_dict,
            global_trainer=global_trainer,
            logger=global_logger,
            threshold=threshold,
            verbose=0,
            seed=args.seed,
            malicious_ids=malicious_ids,
            attack_pool=[args.attack],
            attack_params=attack_params,
            assumed_malicious=assumed_mal,
            num_classes=num_classes,
            # Robust detection is ON for sss_robust, OFF for plain sss
            use_robust_detection=(agg_name == "sss_robust"),
            n_trials=100,
            consensus_tol=1e-3,
            min_consensus_fraction=0.5,
        )
    else:
        agg = RobustAggregator(
            agg_name=agg_name,
            clients_dict=clients_dict,
            clients_weights_dict=clients_weights_dict,
            global_trainer=global_trainer,
            logger=global_logger,
            seed=args.seed,
            malicious_ids=malicious_ids,
            attack=args.attack,
            attack_params=attack_params,
            assumed_malicious=assumed_mal,
            num_classes=num_classes,
        )

    # Build clients sampler
    activity_rng = np.random.default_rng(args.seed)
    activity_sim = get_activity_simulator(all_clients_cfg, activity_rng)
    est_rng = np.random.default_rng(args.seed)
    activity_est = get_activity_estimator("oracle", all_clients_cfg, est_rng)
    sampler_rng = np.random.default_rng(args.seed)
    clients_sampler = get_clients_sampler(
        sampler_type="unbiased",
        activity_simulator=activity_sim,
        activity_estimator=activity_est,
        clients_weights_dict=clients_weights_dict,
        clients_optimums_dict=get_local_optimums(clients_dict),
        smoothness_param=0.0,
        tolerance=0.0,
        time_horizon=args.n_rounds,
        fast_n_clients_per_round=10,
        adafed_full_participation=False,
        bias_const=1.0,
        rng=sampler_rng,
    )

    per_round_acc = []
    for _ in range(args.n_rounds):
        active = clients_sampler.get_active_clients()
        sampled_ids, sampled_wts = clients_sampler.sample(active_clients=active, loss_dict=None)
        try:
            agg.mix(sampled_ids, sampled_wts)
        except Exception as e:
            import traceback
            print(f"    [WARN] mix() failed:")
            print(traceback.format_exc())
            print(f"    → final_acc = 0.0000")
            break
        _, test_acc = _evaluate_global(agg)
        per_round_acc.append(test_acc)

    final_acc = per_round_acc[-1] if per_round_acc else 0.0
    return final_acc, per_round_acc


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_sweep_plot(sweep_results, threshold_frac, args, out_path):
    """Plot final test accuracy vs malicious fraction for each aggregator."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for agg_name in AGGREGATORS:
        fracs = [r["mal_frac"] for r in sweep_results if r["agg"] == agg_name]
        accs  = [r["final_acc"] for r in sweep_results if r["agg"] == agg_name]
        ax.plot(
            [f * 100 for f in fracs], accs,
            label=agg_name,
            color=AGG_COLORS.get(agg_name, "gray"),
            marker=AGG_MARKERS.get(agg_name, "o"),
            markersize=7,
            linewidth=2.0,
        )

    # Mark threshold line
    ax.axvline(x=threshold_frac * 100, color="crimson", linestyle="--", linewidth=1.5,
               label=f"SSS threshold ({int(threshold_frac*100)}%)")
    ax.fill_betweenx([0, 1], 0, threshold_frac * 100, alpha=0.05, color="green",
                     label="Below threshold (safe zone)")
    ax.fill_betweenx([0, 1], threshold_frac * 100, 100, alpha=0.05, color="red",
                     label="Above threshold (unsafe zone)")

    ax.set_xlabel("Malicious Client Fraction (%)", fontsize=12)
    ax.set_ylabel("Final Test Accuracy (round 30)", fontsize=12)
    attack_label = args.attack.replace("_", " ").title()
    ax.set_title(
        f"Accuracy vs Malicious Fraction — {attack_label} Attack\n"
        f"Dataset: {args.experiment}  |  {args.n_rounds} rounds  |  Threshold: {int(threshold_frac*100)}% of clients",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlim(-1, 63)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Sweep] Plot saved → {out_path}")


def make_curves_plot(all_round_results, threshold_frac, args, out_path):
    """Per-round accuracy curves for key malicious fractions (0%, threshold, threshold+10%)."""
    key_fracs = [0.0, threshold_frac, min(threshold_frac + 0.15, 0.60)]
    key_fracs = sorted(set([round(f, 2) for f in key_fracs]))

    n = len(key_fracs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, frac in zip(axes, key_fracs):
        for agg_name in AGGREGATORS:
            key = (agg_name, round(frac, 2))
            rounds_data = all_round_results.get(key, [])
            if not rounds_data:
                continue
            ax.plot(
                range(1, len(rounds_data) + 1), rounds_data,
                label=agg_name,
                color=AGG_COLORS.get(agg_name, "gray"),
                linewidth=2.0,
            )
        label = f"{int(frac*100)}% malicious"
        if abs(frac - threshold_frac) < 0.01:
            label += " (= SSS threshold)"
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Test Accuracy", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    attack_label = args.attack.replace("_", " ").title()
    plt.suptitle(
        f"Per-Round Learning Curves — {attack_label} Attack\n(Dataset: {args.experiment})",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Sweep] Curves plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Malicious Fraction Sweep")
    p.add_argument("--experiment",    default="synthetic_clustered")
    p.add_argument("--cfg_file_path", required=True)
    p.add_argument("--attack",        default="signflip",
                   help="Attack for malicious clients: scaling | signflip | label_flip")
    p.add_argument("--n_rounds",  type=int,   default=20)
    p.add_argument("--local_lr",  type=float, default=0.01)
    p.add_argument("--server_lr", type=float, default=1.0)
    p.add_argument("--train_bz",  type=int,   default=32)
    p.add_argument("--test_bz",   type=int,   default=128)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--logs_dir",  default="logs/sweep")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--local_steps",  type=int, default=1)
    p.add_argument("--local_optimizer", default="sgd")
    p.add_argument("--objective_type",  default="average")
    p.add_argument("--num_classes", type=int, default=10,
                   help="Number of output classes in the task (e.g. 10 for MNIST/CIFAR-10, "
                        "100 for CIFAR-100, 2 for synthetic binary). Used by label_flip attack.")
    p.add_argument("--results_file", default=None,
                   help="CSV to save sweep results. Defaults to <logs_dir>/sweep_results.csv")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    torch.manual_seed(args.seed)

    with open(args.cfg_file_path) as f:
        all_clients_cfg = json.load(f)

    num_clients = len(all_clients_cfg)
    threshold = max(2, num_clients // 4)
    threshold_frac = threshold / num_clients

    print(f"\n{'='*60}")
    print(f"  Malicious Fraction Sweep")
    print(f"  Dataset      : {args.experiment}")
    print(f"  Attack       : {args.attack}")
    print(f"  Clients      : {num_clients}")
    print(f"  SSS threshold: {threshold} clients = {threshold_frac*100:.1f}%")
    print(f"  Rounds       : {args.n_rounds}")
    print(f"  Fracs tested : {[int(f*100) for f in MALICIOUS_FRACS]}%")
    print(f"{'='*60}\n")

    os.makedirs(args.logs_dir, exist_ok=True)

    sweep_results = []       # {agg, mal_frac, final_acc}
    all_round_results = {}   # {(agg, mal_frac): [acc_round0, acc_round1, ...]}

    total = len(AGGREGATORS) * len(MALICIOUS_FRACS)
    done = 0

    for mal_frac in MALICIOUS_FRACS:
        for agg_name in AGGREGATORS:
            done += 1
            n_mal = int(mal_frac * num_clients)
            print(f"[{done:>3}/{total}] {agg_name:15s} | {int(mal_frac*100):3d}% malicious ({n_mal}/{num_clients} clients)")
            try:
                final_acc, round_accs = run_one(args, all_clients_cfg, agg_name, mal_frac)
                sweep_results.append({
                    "dataset":   args.experiment,
                    "attack":    args.attack,
                    "agg":       agg_name,
                    "mal_frac":  mal_frac,
                    "n_mal":     n_mal,
                    "threshold": threshold,
                    "threshold_frac": threshold_frac,
                    "final_acc": f"{final_acc:.6f}",
                })
                all_round_results[(agg_name, round(mal_frac, 2))] = round_accs
                print(f"    → final_acc = {final_acc:.4f}")
            except Exception as e:
                import traceback
                print(f"    [WARN] mix() failed:")
                print(traceback.format_exc())
                print(f"    → final_acc = 0.0000")
                # The original instruction had a syntax error here:
                # print(f"    → final_acc = 0.0000")traceback.print_exc()
                # Corrected to be on a new line for valid Python syntax.

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = args.results_file if args.results_file else os.path.join(args.logs_dir, "sweep_results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    if sweep_results:
        with open(csv_path, "a" if file_exists else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(sweep_results)
        print(f"\n[Sweep] Results saved → {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    sweep_plot_path = os.path.join(
        args.logs_dir, f"sweep_accuracy_vs_frac_{args.experiment}_{args.attack}.png"
    )
    make_sweep_plot(sweep_results, threshold_frac, args, sweep_plot_path)

    curves_plot_path = os.path.join(
        args.logs_dir, f"sweep_curves_{args.experiment}_{args.attack}.png"
    )
    make_curves_plot(all_round_results, threshold_frac, args, curves_plot_path)

    print("\n[Sweep] Done!")

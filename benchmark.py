"""
benchmark.py — Multi-scenario federated learning benchmark.

Runs every combination of (aggregator × attack scenario) and produces:
  - logs/benchmark/results.csv   — per-round accuracy for all configurations
  - logs/benchmark/benchmark_plot.png — accuracy comparison chart

Usage
-----
venv/bin/python benchmark.py \\
  --experiment synthetic_clustered \\
  --cfg_file_path data/synthetic/cfg.json \\
  --n_rounds 30 \\
  --local_lr 0.01 \\
  --train_bz 32 \\
  --test_bz 128 \\
  --seed 42

Note: run from /Users/amaansiddiqui/Desktop/Our-Fed/
"""

import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
from utils.utils import (
    get_loader, get_trainer, get_clients_weights,
    get_activity_simulator, get_activity_estimator,
    get_clients_sampler, get_local_optimums, init_client,
)
from secure_aggregator import SecureAggregator
from aggregator import CentralizedAggregator, NoCommunicationAggregator
from robust_aggregators import aggregate as robust_aggregate
from attacks import DEFAULT_ATTACK_PARAMS
from utils.torch_utils import copy_model


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark scenarios
# ─────────────────────────────────────────────────────────────────────────────

ATTACK_SCENARIOS = {
    "no_attack":   {"malicious_frac": 0.0, "attack_pool": []},
    "scaling":     {"malicious_frac": 0.3, "attack_pool": ["scaling"]},
    "signflip":    {"malicious_frac": 0.3, "attack_pool": ["signflip"]},
    "label_flip":  {"malicious_frac": 0.3, "attack_pool": ["label_flip"]},
}

AGGREGATORS = ["sss", "fedavg", "krum", "multi_krum", "trimmed_mean"]

# Colours for the plot
COLORS = {
    "sss":          "#6C63FF",
    "fedavg":       "#43AA8B",
    "krum":         "#F9844A",
    "multi_krum":   "#277DA1",
    "trimmed_mean": "#F94144",
}
LINE_STYLES = {
    "no_attack":  "-",
    "scaling":    "--",
    "signflip":   "-.",
    "label_flip": ":",
}


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight non-SSS aggregator that uses robust_aggregators.py
# ─────────────────────────────────────────────────────────────────────────────

class RobustCentralizedAggregator(CentralizedAggregator):
    """FedAvg / Krum / Multi-Krum / Trimmed Mean aggregator with attack injection."""

    def __init__(self, agg_name, clients_dict, clients_weights_dict, global_trainer,
                 logger, verbose=0, seed=None,
                 malicious_ids=None, attack_pool=None, attack_params=None,
                 assumed_malicious=1, num_classes=2, trimmed_beta=0.1,
                 multi_krum_m=None):
        super().__init__(clients_dict, clients_weights_dict, global_trainer,
                         logger, verbose, seed)
        self.agg_name = agg_name
        self.malicious_ids = set(malicious_ids) if malicious_ids else set()
        self.attack_pool = list(attack_pool) if attack_pool else []
        self.attack_params = attack_params or {}
        self.assumed_malicious = assumed_malicious
        self.num_classes = num_classes
        self.trimmed_beta = trimmed_beta
        self.multi_krum_m = multi_krum_m
        self._rng = np.random.default_rng(seed if seed is not None else 0)

    def mix(self, sampled_clients_ids, sampled_clients_weights):
        from attacks import pick_attack, apply_update_attack, DEFAULT_ATTACK_PARAMS

        if not sampled_clients_weights:
            self.c_round += 1
            return

        weights_tensor = torch.tensor(sampled_clients_weights, dtype=torch.float32)

        # Step 1: Sync global model
        for cid in sampled_clients_ids:
            copy_model(self.clients_dict[cid].trainer.model, self.global_trainer.model)

        # Step 2: Local training ± attacks
        client_update_attacks = {}
        for idx, w in zip(sampled_clients_ids, weights_tensor):
            if w <= 1e-6:
                continue
            is_mal = idx in self.malicious_ids
            attack = pick_attack(self.attack_pool, self._rng) if is_mal else None

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
        client_weights_list = []
        for idx, w in zip(sampled_clients_ids, weights_tensor):
            client = self.clients_dict[idx]
            delta = {}
            for name, param in client.trainer.model.named_parameters():
                global_param = dict(self.global_trainer.model.named_parameters())[name]
                delta[name] = param.data.clone() - global_param.data.clone()

            # Apply update-time attacks
            if idx in client_update_attacks:
                params = self.attack_params.get(
                    client_update_attacks[idx],
                    DEFAULT_ATTACK_PARAMS.get(client_update_attacks[idx], {})
                )
                delta = apply_update_attack(delta, client_update_attacks[idx], params, self._rng)

            deltas.append(delta)
            client_weights_list.append(float(w))

        # Step 4: Aggregate
        agg_delta = robust_aggregate(
            agg_name=self.agg_name,
            deltas=deltas,
            weights=client_weights_list,
            f=self.assumed_malicious,
            m_select=self.multi_krum_m,
            beta=self.trimmed_beta,
        )

        # Step 5: Apply to global model
        with torch.no_grad():
            for name, param in self.global_trainer.model.named_parameters():
                if name in agg_delta:
                    param.data.add_(agg_delta[name].to(param.device))

        # Step 6: Broadcast to all clients
        self.update_clients()
        self.c_round += 1

    def update_clients(self):
        for client in self.clients_dict.values():
            copy_model(client.trainer.model, self.global_trainer.model)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_global(aggregator):
    """Return weighted global test accuracy across all clients."""
    total_acc = 0.0
    total_loss = 0.0
    for cid, client in aggregator.clients_dict.items():
        _, test_metric = client.trainer.evaluate_loader(client.test_loader)
        test_loss, _ = client.trainer.evaluate_loader(client.test_loader)
        w = aggregator.clients_weights_dict.get(cid, 1.0 / len(aggregator.clients_dict))
        total_acc += w * test_metric
        total_loss += w * test_loss
    return float(total_loss), float(total_acc)


def _build_clients(args, all_clients_cfg, num_clients, threshold, out_prefix):
    """Initialise all clients fresh."""
    clients_dict = {}
    n_samples = {}
    for client_id, cfg in all_clients_cfg.items():
        logs_dir = os.path.join(args.logs_dir, out_prefix, f"client_{client_id}")
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


def _build_aggregator(agg_name, args, clients_dict, clients_weights_dict,
                      global_trainer, logger, malicious_ids, attack_pool,
                      attack_params, threshold):
    """Factory — returns correct aggregator for each agg_name."""
    # Infer num_classes from experiment name
    num_classes = 10 if args.experiment in ("mnist", "cifar10", "cifar100") else 2
    common = dict(
        clients_dict=clients_dict,
        clients_weights_dict=clients_weights_dict,
        global_trainer=global_trainer,
        logger=logger,
        verbose=0,
        seed=args.seed,
        malicious_ids=malicious_ids,
        attack_pool=attack_pool,
        attack_params=attack_params,
        assumed_malicious=max(1, int(0.3 * len(clients_dict))),
        num_classes=num_classes,
    )
    if agg_name == "sss":
        return SecureAggregator(threshold=threshold, **common)
    else:
        return RobustCentralizedAggregator(
            agg_name=agg_name,
            trimmed_beta=0.1,
            multi_krum_m=None,
            **common,
        )


# ─────────────────────────────────────────────────────────────────────────────
# One scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(args, all_clients_cfg, agg_name, scenario_name, scenario_cfg):
    """Run one (aggregator, attack_scenario) combination for n_rounds.

    Returns list of (round, train_loss, test_acc) tuples.
    """
    label = f"{agg_name}_{scenario_name}"
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")

    num_clients = len(all_clients_cfg)
    threshold = max(2, num_clients // 4)  # sensible default
    mal_frac = scenario_cfg["malicious_frac"]
    attack_pool = scenario_cfg["attack_pool"]

    # Designate malicious clients
    rng = np.random.default_rng(args.seed)
    all_ids = list(range(num_clients))
    rng.shuffle(all_ids)
    n_mal = int(mal_frac * num_clients)
    malicious_ids = set(all_ids[:n_mal])

    # Build fresh clients
    clients_dict, n_samples = _build_clients(
        args, all_clients_cfg, num_clients, threshold, label
    )
    clients_weights_dict = get_clients_weights(
        objective_type="average",
        n_samples_per_client=n_samples,
    )

    # Build global trainer
    global_trainer = get_trainer(
        experiment_name=args.experiment,
        device=args.device,
        optimizer_name="sgd",
        lr=args.server_lr,
        seed=args.seed,
    )

    # Global logger
    global_logs_dir = os.path.join(args.logs_dir, label, "global")
    os.makedirs(global_logs_dir, exist_ok=True)
    global_logger = SummaryWriter(global_logs_dir)

    # Build aggregator
    aggregator = _build_aggregator(
        agg_name=agg_name, args=args,
        clients_dict=clients_dict,
        clients_weights_dict=clients_weights_dict,
        global_trainer=global_trainer,
        logger=global_logger,
        malicious_ids=malicious_ids,
        attack_pool=attack_pool,
        attack_params=DEFAULT_ATTACK_PARAMS,
        threshold=threshold,
    )

    # Activity simulator / sampler (unbiased for benchmark)
    activity_rng = np.random.default_rng(args.seed)
    activity_sim = get_activity_simulator(all_clients_cfg, activity_rng)
    estimator_rng = np.random.default_rng(args.seed)
    activity_est = get_activity_estimator("oracle", all_clients_cfg, estimator_rng)
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

    history = []
    for rnd in tqdm(range(args.n_rounds), desc=label, leave=False):
        active = clients_sampler.get_active_clients()
        sampled_ids, sampled_wts = clients_sampler.sample(
            active_clients=active, loss_dict=None
        )
        aggregator.mix(sampled_ids, sampled_wts)
        test_loss, test_acc = _evaluate_global(aggregator)
        history.append({
            "dataset": args.experiment,
            "round": rnd + 1,
            "agg": agg_name,
            "scenario": scenario_name,
            "label": label,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

    print(f"  Final test_acc = {history[-1]['test_acc']:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(all_results, out_path):
    fig, axes = plt.subplots(1, len(ATTACK_SCENARIOS), figsize=(6 * len(ATTACK_SCENARIOS), 5),
                              sharey=True)
    if len(ATTACK_SCENARIOS) == 1:
        axes = [axes]

    for ax, (scenario_name, _) in zip(axes, ATTACK_SCENARIOS.items()):
        for agg_name in AGGREGATORS:
            label = f"{agg_name}_{scenario_name}"
            rows = [r for r in all_results if r["label"] == label]
            if not rows:
                continue
            rounds = [r["round"] for r in rows]
            accs   = [r["test_acc"] for r in rows]
            ax.plot(rounds, accs,
                    label=agg_name,
                    color=COLORS.get(agg_name, "gray"),
                    linewidth=2.0)

        ax.set_title(scenario_name.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Test Accuracy", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle("Federated Learning Benchmark\nAggregators × Attack Scenarios",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Benchmark] Plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FL Benchmark Runner")
    p.add_argument("--experiment",    default="synthetic_clustered",
                   help="Experiment name: synthetic_clustered | mnist")
    p.add_argument("--cfg_file_path", required=True,
                   help="Path to the cfg.json for this dataset")
    p.add_argument("--n_rounds",      type=int,   default=30)
    p.add_argument("--local_steps",   type=int,   default=1)
    p.add_argument("--local_lr",      type=float, default=0.01)
    p.add_argument("--server_lr",     type=float, default=1.0)
    p.add_argument("--train_bz",      type=int,   default=32)
    p.add_argument("--test_bz",       type=int,   default=128)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--logs_dir",      default="logs/benchmark")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--local_optimizer", default="sgd")
    p.add_argument("--objective_type",  default="average")
    p.add_argument("--results_file",
                   default=None,
                   help="Path to combined CSV to APPEND results to. "
                        "If not given, saved as <logs_dir>/results.csv")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    torch.manual_seed(args.seed)

    # Validate that experiment and config file match to prevent shape mismatch crashes
    if "mnist" in args.experiment and "synthetic" in args.cfg_file_path:
        raise ValueError("CRITICAL ERROR: You specified --experiment mnist but gave the synthetic data configuration (--cfg_file_path data/synthetic/cfg.json).\n"
                         "Please run with the correct MNIST config: --cfg_file_path data/mnist_fed/cfg.json")
    if "synthetic" in args.experiment and "mnist" in args.cfg_file_path:
        raise ValueError("CRITICAL ERROR: Data configuration mismatch! Use --cfg_file_path data/synthetic/cfg.json")

    with open(args.cfg_file_path) as f:
        all_clients_cfg = json.load(f)

    os.makedirs(args.logs_dir, exist_ok=True)

    all_results = []
    total = len(AGGREGATORS) * len(ATTACK_SCENARIOS)
    done = 0

    for scenario_name, scenario_cfg in ATTACK_SCENARIOS.items():
        for agg_name in AGGREGATORS:
            done += 1
            print(f"\n[Benchmark {done}/{total}]")
            try:
                history = run_scenario(args, all_clients_cfg,
                                       agg_name, scenario_name, scenario_cfg)
                all_results.extend(history)
            except Exception as e:
                print(f"  ERROR in {agg_name}_{scenario_name}: {e}")
                import traceback; traceback.print_exc()

    # Save CSV — either append to a shared results file or write standalone
    per_run_csv = args.results_file if args.results_file else os.path.join(args.logs_dir, "results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(per_run_csv)), exist_ok=True)
    file_exists = os.path.isfile(per_run_csv)
    if all_results:
        with open(per_run_csv, "a" if file_exists else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_results)
        print(f"[Benchmark] Results appended/saved → {per_run_csv}")

    # Save plot (per-run, named by experiment)
    plot_path = os.path.join(args.logs_dir, f"benchmark_plot_{args.experiment}.png")
    make_plot(all_results, plot_path)

    print("\n[Benchmark] Done!")

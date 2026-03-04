"""
benchmark.py — Multi-scenario federated learning benchmark.

Runs every combination of (aggregator × attack scenario × malicious_fraction) and produces:
  - logs/benchmark/results.csv   — final accuracy for each configuration
  - logs/benchmark/benchmark_plot.png — accuracy vs fraction chart

Usage
-----
venv/bin/python benchmark.py \
  --experiment synthetic_clustered \
  --cfg_file_path data/synthetic/cfg.json \
  --n_rounds 5 \
  --local_lr 0.01 \
  --seed 42
"""

import os
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


ATTACKS = ["scaling", "signflip", "label_flip"]
AGGREGATORS = ["sss", "fedavg", "krum", "multi_krum", "trimmed_mean"]
MALICIOUS_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

COLORS = {
    "sss":          "#6C63FF",
    "fedavg":       "#43AA8B",
    "krum":         "#F9844A",
    "multi_krum":   "#277DA1",
    "trimmed_mean": "#F94144",
}


class RobustCentralizedAggregator(CentralizedAggregator):
    def __init__(self, agg_name, clients_dict, clients_weights_dict, global_trainer,
                 logger, verbose=0, seed=None,
                 malicious_ids=None, attack=None, attack_params=None,
                 assumed_malicious=1, num_classes=2, trimmed_beta=0.1,
                 multi_krum_m=None):
        super().__init__(clients_dict, clients_weights_dict, global_trainer,
                         logger, verbose, seed)
        self.agg_name = agg_name
        self.malicious_ids = set(malicious_ids) if malicious_ids else set()
        self.attack = attack
        self.attack_params = attack_params or {}
        self.assumed_malicious = assumed_malicious
        self.num_classes = num_classes
        self.trimmed_beta = trimmed_beta
        self.multi_krum_m = multi_krum_m
        self._rng = np.random.default_rng(seed if seed is not None else 0)

    def mix(self, sampled_clients_ids, sampled_clients_weights):
        if not sampled_clients_weights:
            self.c_round += 1
            return

        weights_tensor = torch.tensor(sampled_clients_weights, dtype=torch.float32)

        for cid in sampled_clients_ids:
            copy_model(self.clients_dict[cid].trainer.model, self.global_trainer.model)

        client_update_attacks = {}
        for idx, w in zip(sampled_clients_ids, weights_tensor):
            if w <= 1e-6:
                continue
            is_mal = idx in self.malicious_ids
            atk = self.attack if is_mal else None

            if is_mal and atk in ("label_flip", "backdoor"):
                params = self.attack_params.get(atk, DEFAULT_ATTACK_PARAMS.get(atk, {}))
                self.clients_dict[idx].step_with_attack(
                    attack=atk, attack_params=params,
                    rng=self._rng, num_classes=self.num_classes
                )
            else:
                self.clients_dict[idx].step()

            if is_mal and atk in ("scaling", "signflip"):
                client_update_attacks[idx] = atk

        deltas = []
        client_weights_list = []
        for idx, w in zip(sampled_clients_ids, weights_tensor):
            client = self.clients_dict[idx]
            delta = {}
            for name, param in client.trainer.model.named_parameters():
                global_param = dict(self.global_trainer.model.named_parameters())[name]
                delta[name] = param.data.clone() - global_param.data.clone()

            if idx in client_update_attacks:
                atk = client_update_attacks[idx]
                params = self.attack_params.get(atk, DEFAULT_ATTACK_PARAMS.get(atk, {}))
                delta = apply_update_attack(delta, atk, params, self._rng)

            deltas.append(delta)
            client_weights_list.append(float(w))

        agg_delta = robust_aggregate(
            agg_name=self.agg_name,
            deltas=deltas,
            weights=client_weights_list,
            f=self.assumed_malicious,
            m_select=self.multi_krum_m,
            beta=self.trimmed_beta,
        )

        with torch.no_grad():
            for name, param in self.global_trainer.model.named_parameters():
                if name in agg_delta:
                    param.data.add_(agg_delta[name].to(param.device))

        self.update_clients()
        self.c_round += 1

    def update_clients(self):
        for client in self.clients_dict.values():
            copy_model(client.trainer.model, self.global_trainer.model)


def _evaluate_global(aggregator):
    total_acc = 0.0
    for cid, client in aggregator.clients_dict.items():
        _, test_metric = client.trainer.evaluate_loader(client.test_loader)
        w = aggregator.clients_weights_dict.get(cid, 1.0 / len(aggregator.clients_dict))
        total_acc += w * test_metric
    return float(total_acc)


def _build_clients(args, all_clients_cfg, num_clients, threshold, out_prefix):
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
    # Clear memory to prevent crashes with too many runs
    import gc
    gc.collect()
    return clients_dict, n_samples


def _cleanup_scenario(clients_dict, global_logger):
    """Close tensorboard loggers to prevent thread exhaustion."""
    if global_logger:
        global_logger.close()
    for client in clients_dict.values():
        if hasattr(client, 'logger') and client.logger:
            client.logger.close()
        # Fallback if logger is on the trainer
        elif hasattr(client, 'trainer') and hasattr(client.trainer, 'logger') and client.trainer.logger:
            client.trainer.logger.close()


def run_scenario(args, all_clients_cfg, agg_name, attack, mal_frac):
    label = f"{agg_name}_{attack}_{int(mal_frac*100)}"
    num_clients = len(all_clients_cfg)
    threshold = max(2, num_clients // 4)
    n_mal = int(mal_frac * num_clients)

    rng = np.random.default_rng(args.seed)
    all_ids = list(range(num_clients))
    rng.shuffle(all_ids)
    malicious_ids = set(all_ids[:n_mal])

    clients_dict, n_samples = _build_clients(args, all_clients_cfg, num_clients, threshold, label)
    clients_weights_dict = get_clients_weights("average", n_samples)

    global_trainer = get_trainer(
        experiment_name=args.experiment,
        device=args.device,
        optimizer_name="sgd",
        lr=args.server_lr,
        seed=args.seed,
    )

    global_logs_dir = os.path.join(args.logs_dir, label, "global")
    os.makedirs(global_logs_dir, exist_ok=True)
    global_logger = SummaryWriter(global_logs_dir)

    num_classes = args.num_classes if args.num_classes is not None else (10 if args.experiment in ("mnist", "cifar10") else 2)

    if agg_name == "sss":
        aggregator = SecureAggregator(
            clients_dict=clients_dict,
            clients_weights_dict=clients_weights_dict,
            global_trainer=global_trainer,
            logger=global_logger,
            threshold=threshold,
            verbose=0,
            seed=args.seed,
            malicious_ids=malicious_ids,
            attack_pool=[attack] if attack else [],
            attack_params=DEFAULT_ATTACK_PARAMS.copy(),
            assumed_malicious=n_mal,
            num_classes=num_classes,
        )
    else:
        aggregator = RobustCentralizedAggregator(
            agg_name=agg_name,
            clients_dict=clients_dict,
            clients_weights_dict=clients_weights_dict,
            global_trainer=global_trainer,
            logger=global_logger,
            malicious_ids=malicious_ids,
            attack=attack,
            attack_params=DEFAULT_ATTACK_PARAMS.copy(),
            assumed_malicious=n_mal,
            num_classes=num_classes,
        )

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

    for rnd in range(args.n_rounds):
        active = clients_sampler.get_active_clients()
        sampled_ids, sampled_wts = clients_sampler.sample(active_clients=active, loss_dict=None)
        try:
            aggregator.mix(sampled_ids, sampled_wts)
        except Exception as e:
            print(f"Aggregator {agg_name} encountered an error: {e}")
            _cleanup_scenario(clients_dict, global_logger)
            return 0.0

    final_acc = _evaluate_global(aggregator)
    _cleanup_scenario(clients_dict, global_logger)
    return final_acc


def make_plot(all_results, args, out_path):
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(5 * len(ATTACKS), 5), sharey=True)
    if len(ATTACKS) == 1:
        axes = [axes]

    num_clients = 20  # Approx, will use threshold line from dataset directly but static line is fine
    
    for ax, attack in zip(axes, ATTACKS):
        for agg_name in AGGREGATORS:
            rows = [r for r in all_results if r["attack"] == attack and r["agg"] == agg_name]
            if not rows:
                continue
            fracs = [r["mal_frac"] for r in rows]
            accs  = [float(r["final_acc"]) for r in rows]

            ax.plot(fracs, accs, label=agg_name.upper(),
                    color=COLORS.get(agg_name, "gray"), linewidth=2.0, marker="o")

        ax.set_title(attack.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Fraction of Affected Clients", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Final Accuracy", fontsize=11)
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle(f"Benchmark: Final Accuracy vs Compromised Clients ({args.experiment})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Benchmark] Plot saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment",    default="synthetic_clustered")
    p.add_argument("--cfg_file_path", required=True)
    p.add_argument("--n_rounds",      type=int,   default=5)
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
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--results_file", default=None)
    return p.parse_args()


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    torch.manual_seed(args.seed)

    if "mnist" in args.experiment and "synthetic" in args.cfg_file_path:
        raise ValueError("Config mismatch.")
    if "synthetic" in args.experiment and "mnist" in args.cfg_file_path:
        raise ValueError("Config mismatch.")

    with open(args.cfg_file_path) as f:
        all_clients_cfg = json.load(f)

    os.makedirs(args.logs_dir, exist_ok=True)
    
    csv_file = args.results_file if args.results_file else os.path.join(args.logs_dir, f"results_{args.experiment}.csv")
    
    all_results = []
    total = len(ATTACKS) * len(MALICIOUS_FRACS) * len(AGGREGATORS)
    done = 0

    print(f"\n--- Starting Benchmark: {args.experiment} ---")
    for attack in ATTACKS:
        for mal_frac in MALICIOUS_FRACS:
            for agg_name in AGGREGATORS:
                done += 1
                n_mal = int(mal_frac * len(all_clients_cfg))
                print(f"[{done:>3}/{total}] Attack: {attack:10s} | Frac: {mal_frac:.1f} ({n_mal}) | Agg: {agg_name}")
                acc = run_scenario(args, all_clients_cfg, agg_name, attack, mal_frac)
                all_results.append({
                    "dataset": args.experiment,
                    "attack": attack,
                    "agg": agg_name,
                    "mal_frac": mal_frac,
                    "n_mal": n_mal,
                    "final_acc": acc
                })

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"[Benchmark] Results saved to {csv_file}")
    
    plot_path = os.path.join(args.logs_dir, f"benchmark_plot_{args.experiment}.png")
    make_plot(all_results, args, plot_path)
    print("[Benchmark] Complete!")

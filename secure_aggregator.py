import torch
import numpy as np
from aggregator import CentralizedAggregator
from secret_sharing import SecretSharingBuffer, SCALE_FACTOR
from utils.torch_utils import copy_model
from collections import defaultdict


class SecureAggregator(CentralizedAggregator):
    """Secure aggregator using Shamir's Secret Sharing.

    Algorithm per round
    -------------------
    1.  Broadcast current global model to all clients (update_clients).
    2.  Each sampled client snapshots global params (store_global_parameters).
    3.  Each sampled client trains locally (step).
    4.  Each client distributes one unique SSS share of its parameter deltas
        to every other sampled client (distribute_shares).
    5.  Create one SecretSharingBuffer per sampled client.
    6.  Each client uploads the shares it received to the corresponding buffers:
        shares received from owner X  →  buffer[X].
    7.  Buffers reconstruct parameter deltas once ≥ threshold shares arrive.
    8.  SecureAggregator computes a weighted average of
        (global_params + reconstructed_deltas) across clients → new global model.
    9.  Broadcast new global model to all clients (update_clients).
    10. Clear all clients' received-shares caches.
    """

    def __init__(self, clients_dict, clients_weights_dict, global_trainer,
                 logger, threshold=5, verbose=0, seed=None,
                 malicious_ids=None, attack_pool=None, attack_params=None,
                 assumed_malicious=1, num_classes=2,
                 use_robust_detection=False,
                 n_trials=100, consensus_tol=1e-3, min_consensus_fraction=0.5):
        super().__init__(clients_dict, clients_weights_dict, global_trainer,
                         logger, verbose, seed)
        self.threshold = threshold

        # --- Attack configuration ---
        self.malicious_ids = set(malicious_ids) if malicious_ids else set()
        self.attack_pool = list(attack_pool) if attack_pool else []
        self.attack_params = attack_params or {}
        self.assumed_malicious = assumed_malicious
        self.num_classes = num_classes
        import numpy as _np
        self._rng = _np.random.default_rng(seed if seed is not None else 0)

        # --- Robust detection configuration ---
        self.use_robust_detection = use_robust_detection
        self.n_trials = n_trials
        self.consensus_tol = consensus_tol
        self.min_consensus_fraction = min_consensus_fraction
        # Accumulates suspected uploader IDs across rounds
        self.detected_malicious: set = set()

    # ------------------------------------------------------------------
    # Main aggregation round
    # ------------------------------------------------------------------

    def mix(self, sampled_clients_ids, sampled_clients_weights):
        """Execute one round of secure federated learning."""

        if len(sampled_clients_weights) == 0:
            print(f"[SecureAgg] Round {self.c_round}: no clients sampled, skipping.")
            self.c_round += 1
            return

        # Get packing factor from first client to determine true share requirement
        self.packing_l = getattr(self.clients_dict[sampled_clients_ids[0]], 'packing_l', 1) if sampled_clients_ids else 1
        shares_needed = self.threshold + self.packing_l - 1

        if len(sampled_clients_ids) < shares_needed:
            print(
                f"[SecureAgg] Round {self.c_round}: only {len(sampled_clients_ids)} clients sampled "
                f"but shares_needed={shares_needed} (threshold={self.threshold}, l={self.packing_l}). Skipping round."
            )
            self.c_round += 1
            return

        print(f"\n{'='*60}")
        print(f"[SecureAgg] Round {self.c_round} | sampled clients: {sampled_clients_ids}")
        print(f"{'='*60}")

        sampled_clients_weights_tensor = torch.tensor(
            sampled_clients_weights, dtype=torch.float32
        )

        # ----------------------------------------------------------
        # Step 1: Broadcast current global model to sampled clients
        # ----------------------------------------------------------
        print("[Step 1] Broadcasting current global model to clients...")
        for client_id in sampled_clients_ids:
            copy_model(
                target=self.clients_dict[client_id].trainer.model,
                source=self.global_trainer.model
            )

        # ----------------------------------------------------------
        # Step 2: Snapshot global params (for delta computation later)
        # ----------------------------------------------------------
        print("[Step 2] Clients snapshotting global parameters...")
        for client_id in sampled_clients_ids:
            self.clients_dict[client_id].store_global_parameters(self.global_trainer)

        # ----------------------------------------------------------
        # Step 3: Local training (with optional attacks for malicious clients)
        # ----------------------------------------------------------
        from attacks import pick_attack, apply_update_attack, DEFAULT_ATTACK_PARAMS
        print("[Step 3] Local training...")
        # Track which update-time attack each malicious client draws this round
        client_update_attacks = {}  # {client_id: attack_name}
        for idx, weight in zip(sampled_clients_ids, sampled_clients_weights_tensor):
            if weight <= 1e-6:
                continue
            is_malicious = idx in self.malicious_ids
            attack = pick_attack(self.attack_pool, self._rng) if is_malicious else None

            if is_malicious and attack in ("label_flip", "backdoor"):
                params = self.attack_params.get(attack, DEFAULT_ATTACK_PARAMS.get(attack, {}))
                print(f"  Training client {idx} [MALICIOUS — {attack}]...")
                self.clients_dict[idx].step_with_attack(
                    attack=attack, attack_params=params,
                    rng=self._rng, num_classes=self.num_classes
                )
            else:
                tag = f" [MALICIOUS — {attack}]" if attack else ""
                print(f"  Training client {idx}{tag}...")
                self.clients_dict[idx].step()

            if is_malicious and attack in ("scaling", "signflip"):
                client_update_attacks[idx] = attack

        # ----------------------------------------------------------
        # Step 4: Share distribution (client → all other clients)
        # ----------------------------------------------------------
        print("[Step 4] Distributing SSS shares among sampled clients...")
        sampled_clients_dict = {
            cid: self.clients_dict[cid] for cid in sampled_clients_ids
        }
        for client_id in sampled_clients_ids:
            attack = client_update_attacks.get(client_id)
            params = self.attack_params.get(attack, DEFAULT_ATTACK_PARAMS.get(attack, {})) if attack else {}
            self.clients_dict[client_id].distribute_shares(sampled_clients_dict, attack, params, self._rng)

        # ----------------------------------------------------------
        # Step 5: Create per-client buffers
        # ----------------------------------------------------------
        print("[Step 5] Creating buffers...")
        buffers_dict = {
            client_id: SecretSharingBuffer(client_id, self.threshold, l=self.packing_l)
            for client_id in sampled_clients_ids
        }

        # ----------------------------------------------------------
        # Step 6: Each client routes its received shares to buffers
        #
        # Client X received one share from every other participant.
        # The share received FROM owner Y belongs TO buffer[Y].
        # ----------------------------------------------------------
        print("[Step 6] Routing received shares to buffers...")
        for uploader_id in sampled_clients_ids:
            client = self.clients_dict[uploader_id]
            shares_held = client.get_shares_for_reconstruction()

            if not shares_held:
                print(f"  [WARNING] Client {uploader_id} has no shares — distribution may have failed.")
                continue

            for owner_id, param_shares in shares_held.items():
                if owner_id not in buffers_dict:
                    continue  # owner not in this round — discard
                if not param_shares:
                    continue
                if uploader_id in self.malicious_ids:
                    # [BYZANTINE UPLOADER] Corrupt shares for others
                    import numpy as _np
                    corrupted_param_shares = {}
                    for p_name, p_data in param_shares.items():
                        corrupted_param_shares[p_name] = {
                            'y': _np.random.randint(0, 2**31 - 1, size=p_data['y'].shape, dtype=_np.int64),
                            'x': p_data['x'],
                            'metadata': p_data['metadata']
                        }
                    buffers_dict[owner_id].add_shares(uploader_id, corrupted_param_shares)
                else:
                    buffers_dict[owner_id].add_shares(uploader_id, param_shares)

        # ----------------------------------------------------------
        # Step 7: Reconstruction
        # ----------------------------------------------------------
        print("[Step 7] Reconstructing parameter deltas from buffers...")
        reconstructed_deltas = self._reconstruct_from_buffers(
            sampled_clients_ids, buffers_dict
        )

        # ----------------------------------------------------------
        # Step 8: Weighted aggregation → update global model
        # ----------------------------------------------------------
        if reconstructed_deltas:
            print("[Step 8] Applying weighted average of reconstructed deltas...")
            self._apply_reconstructed_deltas(
                reconstructed_deltas,
                sampled_clients_weights_tensor,
                sampled_clients_ids
            )
        else:
            print("[Step 8] WARNING: No parameters could be reconstructed this round.")

        # ----------------------------------------------------------
        # Step 9: Broadcast updated global model to ALL clients
        # ----------------------------------------------------------
        print("[Step 9] Broadcasting updated global model to all clients...")
        self.update_clients()

        # ----------------------------------------------------------
        # Step 10: Clean up received shares for next round
        # ----------------------------------------------------------
        for client in self.clients_dict.values():
            if hasattr(client, 'clear_received_shares'):
                client.clear_received_shares()

        self.c_round += 1
        print(f"[SecureAgg] Round {self.c_round - 1} complete.")

    # ------------------------------------------------------------------
    # Reconstruction helpers
    # ------------------------------------------------------------------

    def _reconstruct_from_buffers(self, sampled_clients_ids, buffers_dict) -> dict:
        """Reconstruct parameter deltas for every sampled client from its buffer.

        When ``use_robust_detection`` is True, uses majority-vote reconstruction
        to detect and exclude contributions from suspected malicious uploaders.

        Returns
        -------
        dict
            { client_id -> { param_name -> torch.Tensor } }
        """
        reconstructed_deltas = {}
        round_suspects: set = set()

        for owner_id in sampled_clients_ids:
            buffer = buffers_dict.get(owner_id)
            if buffer is None:
                print(f"  [ERROR] No buffer for client {owner_id}.")
                continue

            print(f"  Buffer[{owner_id}] has received contributions — reconstructing...")

            if self.use_robust_detection:
                reconstructed_params, suspects = buffer.robust_reconstruct_parameters(
                    n_trials=self.n_trials,
                    consensus_tol=self.consensus_tol,
                    min_consensus_fraction=self.min_consensus_fraction,
                )
                round_suspects.update(suspects)
            else:
                reconstructed_params = buffer.reconstruct_parameters()

            if reconstructed_params:
                reconstructed_deltas[owner_id] = self._to_tensor_dict(reconstructed_params, owner_id)
                print(f"  ✓ Client {owner_id}: reconstruction successful.")
            else:
                print(f"  ✗ Client {owner_id}: reconstruction failed (insufficient consensus or shares).")

        if round_suspects:
            self.detected_malicious.update(round_suspects)
            print(f"[SecureAgg] ⚠️  Suspects this round: {round_suspects}")
            print(f"[SecureAgg] ⚠️  Cumulative suspects: {self.detected_malicious}")

        return reconstructed_deltas

    def _to_tensor_dict(self, reconstructed_params: dict, owner_id: int) -> dict:
        """Convert reconstructed parameters to { param_name -> torch.Tensor }.

        Now handles the case where parameters are already reconstructed into tensors.
        """
        # If the values are already tensors, just return the dict
        first_val = next(iter(reconstructed_params.values())) if reconstructed_params else None
        if isinstance(first_val, torch.Tensor):
            return reconstructed_params
        
        # Fallback for old dictionary-based format if any
        tensor_dict = {}
        packing_l = getattr(self, 'packing_l', 1)

        for param_name, pos_values in reconstructed_params.items():
            shape = self._get_param_shape(param_name)
            tensor = torch.zeros(shape)

            if packing_l > 1:
                # PSSS mode: pos keys are integer block indices, values are float arrays of length l
                sorted_blocks = sorted(
                    [(k, v) for k, v in pos_values.items() if isinstance(k, int)],
                    key=lambda x: x[0]
                )
                if sorted_blocks:
                    flat_parts = [np.atleast_1d(np.asarray(v, dtype=np.float32)) for _, v in sorted_blocks]
                    flat_array = np.concatenate(flat_parts)  # length = n_blocks * l (may include padding)
                    numel = tensor.numel()
                    if len(flat_array) > numel:
                        flat_array = flat_array[:numel]  # trim padding
                    if len(flat_array) == numel:
                        tensor = torch.tensor(flat_array, dtype=torch.float32).view(shape)
                    else:
                        print(f"  [ERROR] PSSS shape mismatch for {param_name}: "
                              f"numel={numel}, reconstructed={len(flat_array)}")
            else:
                # Standard SSS mode: pos is a flat integer index, value is a scalar float.
                # We need to map the flat index back to the tensor's multi-dimensional shape.
                for pos, value in pos_values.items():
                    try:
                        if isinstance(pos, int) and len(shape) > 0:
                            multi_idx = tuple(np.unravel_index(pos, shape))
                            tensor[multi_idx] = float(value)
                        else:
                            # Fallback if pos is already a tuple or tensor is 0D/1D
                            tensor[pos] = float(value)
                    except Exception as e:
                        print(f"  [ERROR] SSS assign {param_name}[{pos}]={value}: {e}")

            tensor_dict[param_name] = tensor

        return tensor_dict

    def _get_param_shape(self, param_name) -> tuple:
        """Look up parameter shape from the global model."""
        for name, param in self.global_trainer.model.named_parameters():
            if name == param_name:
                return param.shape
        # Fallback (should not be reached with a consistent model)
        return (1,)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _apply_reconstructed_deltas(self, reconstructed_deltas, weights_tensor, client_ids):
        """Add aggregated deltas to the global model.

        Parameters
        ----------
        reconstructed_deltas : dict
            { client_id -> { param_name -> torch.Tensor (delta) } }
        weights_tensor : torch.Tensor
            Unnormalized client weights.
        client_ids : list[int]
        """
        if not reconstructed_deltas:
            return

        # Prepare list of delta-dicts for the robust aggregator
        delta_list = []
        weight_list = []
        for i, cid in enumerate(client_ids):
            if cid in reconstructed_deltas:
                delta_list.append(reconstructed_deltas[cid])
                weight_list.append(weights_tensor[i].item())

        if not delta_list:
            return

        from robust_aggregators import aggregate
        # If robust_agg is on, use Trimmed Mean by default on the reconstructed deltas
        # to ensure the "Brick Wall" robustness until the SSS threshold fails.
        agg_name = "trimmed_mean" if self.use_robust_detection else "fedavg"
        
        # Trim factor beta: we know assumed_malicious, so let's set it appropriately
        # If n=20 and n_mal=2, beta = 2/20 = 0.1
        n = len(delta_list)
        beta = min(0.4, (self.assumed_malicious + 1) / n) if n > 0 else 0.1

        aggregated_delta = aggregate(
            agg_name=agg_name,
            deltas=delta_list,
            weights=weight_list,
            f=self.assumed_malicious,
            beta=beta
        )

        # Apply to global model
        with torch.no_grad():
            for name, param in self.global_trainer.model.named_parameters():
                if name in aggregated_delta:
                    param.data.add_(aggregated_delta[name].to(param.device))

    # ------------------------------------------------------------------
    # Override — keep update_clients working correctly
    # ------------------------------------------------------------------

    def update_clients(self):
        """Push the current global model to every client."""
        for client_id, client in self.clients_dict.items():
            copy_model(
                target=client.trainer.model,
                source=self.global_trainer.model
            )

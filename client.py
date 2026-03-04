import numpy as np
from collections import defaultdict
from secret_sharing import generate_shares, SCALE_FACTOR
import torch


class SecureClient:
    """Client with Shamir's Secret Sharing capabilities for secure federated learning.

    After local training, each client:
      1. Computes parameter deltas (local - global).
      2. Generates n secret shares of each delta using SSS.
      3. Distributes one unique share per parameter to every other participant.
      4. Receives shares from every other participant and stores them.

    During aggregation, the aggregator routes each received share to the
    buffer corresponding to the share's original owner for reconstruction.
    """

    def __init__(
            self,
            client_id,
            local_steps,
            logger,
            trainer=None,
            train_loader=None,
            val_loader=None,
            test_loader=None,
            num_clients=20,
            threshold=5):

        self.client_id = client_id
        self.trainer = trainer
        self.device = self.trainer.device
        self.num_clients = num_clients
        self.threshold = threshold
        self.previous_global_params = None

        # received_shares: { owner_client_id -> { param_name -> [{'pos': pos, 'share': (x,y)}, ...] } }
        self.received_shares = defaultdict(dict)

        self.logger = logger
        self.counter = 0
        self.local_steps = local_steps

        if train_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.num_samples = len(self.train_loader.dataset)
            self.train_iterator = iter(self.train_loader)
            self.is_ready = True
        else:
            self.is_ready = False
            self.metadata = dict()

    # ------------------------------------------------------------------
    # Core training interface
    # ------------------------------------------------------------------

    def step(self, by_batch=False):
        """Perform one local training step (benign).

        Parameters
        ----------
        by_batch : bool
            If True, train on a single batch; otherwise train for local_steps epochs.
        """
        self.counter += 1

        if by_batch:
            batch = self.get_next_batch()
            self.trainer.fit_batch(batch=batch)
        else:
            self.trainer.fit_epochs(
                loader=self.train_loader,
                n_epochs=self.local_steps
            )

    def step_with_attack(self, attack: str, attack_params: dict, rng, num_classes: int = 2):
        """Perform local training with a data-time attack (label_flip or backdoor).

        For update-time attacks (scaling, signflip), call `step()` as usual, then
        apply `attacks.apply_update_attack()` on the returned delta externally.

        Parameters
        ----------
        attack : str
            'label_flip' or 'backdoor' (data-time attacks).
        attack_params : dict
            Attack hyperparameters.
        rng : np.random.Generator
        num_classes : int
        """
        from attacks import maybe_label_flip, add_backdoor_trigger
        self.counter += 1
        self.trainer.model.train()

        poison_frac = float(attack_params.get("poison_frac", 0.3))
        trigger = attack_params.get("trigger", "corner_pixel")
        trigger_value = float(attack_params.get("trigger_value", 1.0))
        target_label = int(attack_params.get("target", 0))
        label_prob = float(attack_params.get("prob", 1.0))

        for _ in range(self.local_steps):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if attack == "label_flip":
                    y = maybe_label_flip(y, num_classes, label_prob, rng)

                elif attack == "backdoor":
                    import torch
                    mask = torch.rand(x.size(0), device=self.device) < poison_frac
                    if mask.any():
                        x[mask] = add_backdoor_trigger(x[mask], trigger, trigger_value)
                        y[mask] = target_label % num_classes

                self.trainer.optimizer.zero_grad()
                outs = self.trainer.model(x.float())
                # BCE (binary): output shape (N,1) → squeeze to (N,) and use float labels
                # CrossEntropy (multi-class): output shape (N,C) → keep and use long labels
                if outs.dim() > 1 and outs.shape[1] == 1:
                    outs = outs.squeeze(1)
                    y_loss = y.float()
                else:
                    y_loss = y.long()
                loss = self.trainer.criterion(outs, y_loss)
                loss.backward()
                self.trainer.optimizer.step()

    def get_next_batch(self):
        """Return the next batch from the data iterator."""
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            batch = next(self.train_iterator)
        return batch

    def write_logs(self, counter=None):
        """Evaluate and log train/test metrics."""
        if counter is None:
            counter = self.counter

        train_loss, train_metric = self.trainer.evaluate_loader(self.train_loader)
        test_loss, test_metric = self.trainer.evaluate_loader(self.test_loader)

        self.logger.add_scalar("Train/Loss", train_loss, counter)
        self.logger.add_scalar("Train/Metric", train_metric, counter)
        self.logger.add_scalar("Test/Loss", test_loss, counter)
        self.logger.add_scalar("Test/Metric", test_metric, counter)
        self.logger.flush()

        return train_loss, train_metric, test_loss, test_metric

    # ------------------------------------------------------------------
    # Secret sharing protocol
    # ------------------------------------------------------------------

    def store_global_parameters(self, global_trainer):
        """Snapshot the current global model parameters for delta computation."""
        self.previous_global_params = {}
        for name, param in global_trainer.model.named_parameters():
            self.previous_global_params[name] = param.data.clone()

    def get_parameter_deltas(self) -> dict:
        """Compute delta = local_params - global_params after local training.

        Returns
        -------
        dict
            { param_name -> tensor of deltas }
        """
        if self.previous_global_params is None:
            # First round: treat full local parameters as the contribution
            return {
                name: param.data.clone()
                for name, param in self.trainer.model.named_parameters()
            }

        deltas = {}
        for name, param in self.trainer.model.named_parameters():
            if name in self.previous_global_params:
                deltas[name] = param.data - self.previous_global_params[name]
            else:
                deltas[name] = param.data.clone()
        return deltas

    def create_parameter_shares(self, participating_clients_ids: list) -> dict:
        """Generate one unique SSS share per participant for each parameter delta using vectorization.

        Parameters
        ----------
        participating_clients_ids : list[int]
            IDs of all clients participating in this round (including self).

        Returns
        -------
        dict
            { recipient_client_id -> { param_name -> [{'pos': pos, 'share': (x,y)}, ...] } }
        """
        from secret_sharing import generate_shares_vectorized

        deltas = self.get_parameter_deltas()

        if not deltas:
            print(f"[Client {self.client_id}] WARNING: parameter deltas are empty.")

        n_participants = len(participating_clients_ids)
        participant_map = {i: cid for i, cid in enumerate(participating_clients_ids)}

        shares_for_clients = {cid: {} for cid in participating_clients_ids}

        for param_name, delta_tensor in deltas.items():
            shares_for_clients_param = {cid: [] for cid in participating_clients_ids}

            # Flatten the tensor to vectorise over all elements at once
            shape = delta_tensor.shape
            delta_flat = delta_tensor.flatten().detach().cpu().numpy()

            # Convert to integers by scaling safely
            secrets_arr = np.round(delta_flat * SCALE_FACTOR).astype(np.int64)

            # Generate shares for the entire tensor in one C++ vectorized call
            # x_values is a list of len `n_participants`
            # y_values_matrix is (n_participants, N)
            x_values, y_values_matrix = generate_shares_vectorized(n_participants, self.threshold, secrets_arr)

            # Map the flat indices back to their multi-dimensional coordinates
            # This is necessary because the aggregator's buffer reconstructs param-by-param
            all_positions = list(np.ndindex(shape))

            for i in range(n_participants):
                recipient_id = participant_map[i]
                x_val = x_values[i]
                y_vals = y_values_matrix[i]
                
                # Bundle the pos with its (x, y) share
                shares_list = [
                    {'pos': pos, 'share': (x_val, int(y))}
                    for pos, y in zip(all_positions, y_vals)
                ]
                
                shares_for_clients_param[recipient_id] = shares_list

            for cid in participating_clients_ids:
                shares_for_clients[cid][param_name] = shares_for_clients_param[cid]

        return shares_for_clients

    def distribute_shares(self, participating_clients_dict: dict):
        """Generate shares and deliver them to every participating client.

        Self receives its own share directly (stored in received_shares).
        All others receive their share via receive_shares().

        Parameters
        ----------
        participating_clients_dict : dict[int, SecureClient]
            All clients participating in this round, keyed by client_id.
        """
        participating_clients_ids = list(participating_clients_dict.keys())
        all_shares = self.create_parameter_shares(participating_clients_ids)

        print(f"[Client {self.client_id}] Distributing shares to {len(participating_clients_ids)} clients...")
        for recipient_id, param_shares in all_shares.items():
            if recipient_id == self.client_id:
                # Store own share locally (self-share)
                self.received_shares[self.client_id] = param_shares
            else:
                participating_clients_dict[recipient_id].receive_shares(self.client_id, param_shares)

    def receive_shares(self, sender_id: int, param_shares: dict):
        """Receive and store shares sent by another client.

        Parameters
        ----------
        sender_id : int
            The client_id of the client who generated these shares
            (i.e., the owner of the secret parameters being shared).
        param_shares : dict
            { param_name -> [{'pos': pos, 'share': (x,y)}, ...] }
        """
        # sender_id IS the owner: Client A distributes shares of *its own* params,
        # so the shares stored here are keyed by A's id.
        self.received_shares[sender_id] = param_shares

    def get_shares_for_reconstruction(self) -> dict:
        """Return all received shares, keyed by the owner client's id.

        Returns
        -------
        dict
            { owner_client_id -> { param_name -> [{'pos': pos, 'share': (x,y)}, ...] } }
        """
        return dict(self.received_shares)

    def clear_received_shares(self):
        """Reset the received shares store. Call after each round."""
        self.received_shares.clear()
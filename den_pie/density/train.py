import logging
import os
import time
import tempfile
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..util.logger import separator


class DensityTraining:
    """Training loop for density field inference (encoder + flow).

    Uses DataLoader-based batching (not path-based like existing Training).
    Supports 3 training stages: 'encoder' (MSE pretrain), 'flow' (cached
    embeddings), 'both' (end-to-end).

    When use_sbi is True, uses SBI's density_estimator instead of separate
    encoder + flow modules.
    """

    def __init__(self, params: dict, encoder: nn.Module, flow: nn.Module,
                 data: dict, device: torch.device,
                 density_estimator=None, trial=None):
        self.params = params
        self.cfg = params['density']['train']
        self.data = data
        self.device = device
        self.save_dir = params['name'] + '/'
        self.n_params = len(params['density']['data']['active_params'])

        # SBI mode
        self.use_sbi = params['density']['flow'].get('use_sbi', False)

        if self.use_sbi:
            self.density_estimator = density_estimator.to(device)
            self.encoder = None
            self.flow = None
        else:
            self.encoder = encoder.to(device)
            self.flow = flow.to(device)
            self.density_estimator = None

        # Optuna trial for pruning (optional)
        self.trial = trial

        # Training config
        self.epochs = self.cfg['epochs']
        self.batch_size = self.cfg['batch_size']
        self.lr = self.cfg['lr']
        self.grad_accum_steps = self.cfg.get('gradient_accumulation_steps', 1)
        self.grad_clip = self.cfg.get('grad_clip', None)
        self.patience = self.cfg.get('patience', None)

    def _make_loader(self, x, y, shuffle=True):
        num_workers = self.params['density']['data'].get('num_workers', 4)
        if isinstance(x, torch.utils.data.Dataset):
            ds = x
        else:
            ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=(num_workers > 0),
                          drop_last=False)

    # ------------------------------------------------------------------ #
    #  Optimizer setup
    # ------------------------------------------------------------------ #

    def setup_optimizer_and_scheduler(self, mode: str):
        weight_decay = self.cfg.get('weight_decay', 0.0)
        optimizer_name = self.cfg.get('optimizer', 'AdamW')

        if self.use_sbi:
            params_to_opt = self._get_sbi_params(mode)
        else:
            if mode == 'encoder':
                params_to_opt = self.encoder.parameters()
            elif mode == 'flow':
                params_to_opt = self.flow.parameters()
            elif mode == 'both':
                params_to_opt = list(self.encoder.parameters()) + list(self.flow.parameters())
            else:
                raise ValueError(f"Unknown train mode: {mode}")

        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(params_to_opt, lr=self.lr, weight_decay=weight_decay)
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(params_to_opt, lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler_type = self.cfg.get('scheduler', 'ReduceLROnPlateau')
        scheduler_params = self.cfg.get('scheduler_params', {})
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_params)
        elif scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        logging.info(f"Optimizer: {optimizer_name}, lr={self.lr}, scheduler={scheduler_type}")

    def _get_sbi_params(self, mode: str):
        """Get parameters to optimize for SBI mode."""
        de = self.density_estimator
        if mode == 'encoder':
            # Only optimize the embedding net (encoder inside density estimator)
            return de.net._embedding_net.parameters()
        elif mode == 'flow':
            # Freeze embedding net, optimize transform + distribution
            for p in de.net._embedding_net.parameters():
                p.requires_grad = False
            params_to_opt = (list(de.net._transform.parameters())
                             + list(de.net._distribution.parameters()))
            return params_to_opt
        elif mode == 'both':
            # Unfreeze everything
            for p in de.parameters():
                p.requires_grad = True
            return de.parameters()
        else:
            raise ValueError(f"Unknown train mode: {mode}")

    # ------------------------------------------------------------------ #
    #  Cache embeddings
    # ------------------------------------------------------------------ #

    def _cache_embeddings(self, x, batch_size=64):
        """Run encoder on all data and cache embeddings on CPU.

        Accepts either a preloaded tensor (legacy) or a
        torch.utils.data.Dataset (new lazy path).
        """
        if self.use_sbi:
            encoder = self.density_estimator.net._embedding_net
        else:
            encoder = self.encoder
        encoder.eval()

        if isinstance(x, torch.utils.data.Dataset):
            num_workers = self.params['density']['data'].get('num_workers', 4)
            loader = DataLoader(x, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=(num_workers > 0))
            def batches():
                for xb, _ in loader:
                    yield xb.to(self.device, non_blocking=True)
        else:
            def batches():
                for i in range(0, len(x), batch_size):
                    yield x[i:i + batch_size].to(self.device)

        embeddings = []
        with torch.no_grad():
            for batch in batches():
                emb = encoder(batch).cpu()
                embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

    # ------------------------------------------------------------------ #
    #  Non-SBI training epochs (original)
    # ------------------------------------------------------------------ #

    def train_epoch_encoder(self, loader):
        """Pretrain encoder with MSE loss."""
        self.encoder.train()
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            # Encoder outputs summary_dim, but we train with MSE against params
            # Need a temporary head for this; use first n_params dims
            pred = self.encoder(x_batch)[:, :self.n_params]
            loss = loss_fn(pred, y_batch) / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_epoch_flow(self, loader):
        """Train flow with cached embeddings (encoder frozen)."""
        self.flow.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, (emb_batch, y_batch) in enumerate(loader):
            emb_batch = emb_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            log_prob = self.flow(y_batch, context=emb_batch)
            loss = -log_prob.mean() / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_epoch_both(self, loader):
        """End-to-end training: encoder + flow."""
        self.encoder.train()
        self.flow.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            context = self.encoder(x_batch)
            log_prob = self.flow(y_batch, context=context)
            loss = -log_prob.mean() / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip:
                    all_params = list(self.encoder.parameters()) + list(self.flow.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------ #
    #  SBI training epochs
    # ------------------------------------------------------------------ #

    def train_epoch_encoder_sbi(self, loader):
        """Pretrain encoder (SBI embedding net) with MSE loss."""
        embedding_net = self.density_estimator.net._embedding_net
        embedding_net.train()
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            pred = embedding_net(x_batch)[:, :self.n_params]
            loss = loss_fn(pred, y_batch) / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(embedding_net.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_epoch_flow_sbi(self, loader):
        """Train SBI flow with cached embeddings (embedding net frozen)."""
        self.density_estimator.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, (emb_batch, y_batch) in enumerate(loader):
            emb_batch = emb_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            # Use SBI internals to compute flow loss with cached embeddings
            noise, logabsdet = self.density_estimator.net._transform(
                y_batch, context=emb_batch)
            log_prob = self.density_estimator.net._distribution.log_prob(
                noise, context=emb_batch)
            loss = -(log_prob + logabsdet).mean() / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip:
                    params_to_clip = (list(self.density_estimator.net._transform.parameters())
                                      + list(self.density_estimator.net._distribution.parameters()))
                    torch.nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_epoch_both_sbi(self, loader):
        """End-to-end SBI training via density_estimator.loss()."""
        self.density_estimator.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            losses = self.density_estimator.loss(y_batch, condition=x_batch)
            loss = losses.mean() / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.density_estimator.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def val_epoch(self, loader, mode):
        """Validation epoch for any mode."""
        if self.use_sbi:
            self.density_estimator.eval()
        else:
            self.encoder.eval()
            self.flow.eval()

        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            if mode == 'encoder':
                if self.use_sbi:
                    pred = self.density_estimator.net._embedding_net(x_batch)[:, :self.n_params]
                else:
                    pred = self.encoder(x_batch)[:, :self.n_params]
                loss = nn.functional.mse_loss(pred, y_batch)

            elif mode == 'flow':
                # x_batch is cached embeddings
                if self.use_sbi:
                    noise, logabsdet = self.density_estimator.net._transform(
                        y_batch, context=x_batch)
                    log_prob = self.density_estimator.net._distribution.log_prob(
                        noise, context=x_batch)
                    loss = -(log_prob + logabsdet).mean()
                else:
                    log_prob = self.flow(y_batch, context=x_batch)
                    loss = -log_prob.mean()

            elif mode == 'both':
                if self.use_sbi:
                    losses = self.density_estimator.loss(y_batch, condition=x_batch)
                    loss = losses.mean()
                else:
                    context = self.encoder(x_batch)
                    log_prob = self.flow(y_batch, context=context)
                    loss = -log_prob.mean()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path, epoch, val_loss):
        """Atomic checkpoint save."""
        if self.use_sbi:
            state = {
                'epoch': epoch,
                'val_loss': val_loss,
                'density_estimator_state_dict': self.density_estimator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
        else:
            state = {
                'epoch': epoch,
                'val_loss': val_loss,
                'encoder_state_dict': self.encoder.state_dict(),
                'flow_state_dict': self.flow.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        try:
            torch.save(state, tmp_path)
            shutil.move(tmp_path, path)
        except:
            os.close(tmp_fd)
            raise

    def plot_loss(self, trn_loss, val_loss, mode):
        loss_dir = self.save_dir + 'loss/'
        os.makedirs(loss_dir, exist_ok=True)
        plt.figure()
        plt.plot(trn_loss, label='Training')
        plt.plot(val_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(loss_dir + f'loss_{mode}.pdf')
        plt.close()
        np.save(loss_dir + f'loss_{mode}.npy', [trn_loss, val_loss])

    # ------------------------------------------------------------------ #
    #  Main training loop
    # ------------------------------------------------------------------ #

    def train_loop(self):
        mode = self.cfg['train_network']
        model_dir = self.save_dir + f'models/{mode}/'
        if self.trial is None:
            os.makedirs(model_dir, exist_ok=True)

        self.setup_optimizer_and_scheduler(mode)

        # Prepare data loaders depending on mode
        if mode == 'flow':
            logging.info("Caching encoder embeddings for flow-only training...")
            emb_train = self._cache_embeddings(self.data['x_train'])
            emb_val = self._cache_embeddings(self.data['x_val'])
            trn_loader = self._make_loader(emb_train, self.data['y_train'])
            val_loader = self._make_loader(emb_val, self.data['y_val'], shuffle=False)
        else:
            trn_loader = self._make_loader(self.data['x_train'], self.data['y_train'])
            val_loader = self._make_loader(self.data['x_val'], self.data['y_val'], shuffle=False)

        trn_losses = []
        val_losses = []
        best_val_loss = float('inf')
        ema_val_loss = None
        ema_alpha = 0.1
        patience_counter = 0

        sbi_tag = " [SBI]" if self.use_sbi else ""
        separator()
        logging.info(f"Training '{mode}'{sbi_tag} for {self.epochs} epochs")
        separator()

        start_time = time.time()
        for epoch in range(self.epochs):
            # Train
            if self.use_sbi:
                if mode == 'encoder':
                    trn_loss = self.train_epoch_encoder_sbi(trn_loader)
                elif mode == 'flow':
                    trn_loss = self.train_epoch_flow_sbi(trn_loader)
                elif mode == 'both':
                    trn_loss = self.train_epoch_both_sbi(trn_loader)
            else:
                if mode == 'encoder':
                    trn_loss = self.train_epoch_encoder(trn_loader)
                elif mode == 'flow':
                    trn_loss = self.train_epoch_flow(trn_loader)
                elif mode == 'both':
                    trn_loss = self.train_epoch_both(trn_loader)

            # Validate
            val_loss = self.val_epoch(val_loader, mode)

            trn_losses.append(trn_loss)
            val_losses.append(val_loss)

            # EMA smoothed val loss for early stopping
            if ema_val_loss is None:
                ema_val_loss = val_loss
            else:
                ema_val_loss = ema_alpha * val_loss + (1 - ema_alpha) * ema_val_loss

            # Scheduler step
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{self.epochs}  "
                         f"trn_loss={trn_loss:.6f}  val_loss={val_loss:.6f}  "
                         f"ema_val={ema_val_loss:.6f}  lr={current_lr:.2e}")

            # Optuna pruning
            if self.trial is not None:
                import optuna
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.trial is None:
                    self.save_checkpoint(model_dir + 'best.pth', epoch, val_loss)
                logging.info(f"  -> new best val loss: {val_loss:.6f}")
            else:
                patience_counter += 1

            if self.trial is None:
                # Periodic save
                if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                    self.save_checkpoint(model_dir + f'epoch_{epoch + 1}.pth', epoch, val_loss)

                # Plot
                self.plot_loss(trn_losses, val_losses, mode)

            # Early stopping
            if self.patience and patience_counter >= self.patience:
                logging.info(f"Early stopping at epoch {epoch + 1} (patience={self.patience})")
                break

        # Save final
        if self.trial is None:
            self.save_checkpoint(model_dir + 'final.pth', epoch, val_loss)
        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
        logging.info(f"Total training time: {h}h {m}m {s}s")
        separator()
        logging.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
        return best_val_loss

    def main(self):
        return self.train_loop()

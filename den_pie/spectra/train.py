"""Single-stage trainer for the spectra SBI pipeline.

Mirrors the conventions of density/train.py (atomic checkpoints, loss-curve
plot, EMA-smoothed val loss, early stopping) and the substance of
SBI_Pk's TrainSBINSF (divergence check, ReduceLROnPlateau).
"""
import logging
import math
import os
import shutil
import tempfile
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..util.logger import separator


class SpectraTraining:
    """Train an sbi density estimator on cached P+B feature vectors."""

    def __init__(self, params: dict, density_estimator, data: dict,
                 device: torch.device):
        self.params = params
        self.cfg = params['spectra']['train']
        self.data = data
        self.device = device
        self.save_dir = params['name'] + '/'

        self.density_estimator = density_estimator.to(device)

        self.epochs = self.cfg['epochs']
        self.batch_size = self.cfg['batch_size']
        self.lr = self.cfg['lr']
        self.patience = self.cfg.get('patience', None)
        self.clip_grad = self.cfg.get('clip_grad', None)
        self.grad_accum_steps = self.cfg.get('gradient_accumulation_steps', 1)
        self.weight_decay = self.cfg.get('weight_decay', 0.0)

        self.optimizer = optim.Adam(
            self.density_estimator.parameters(),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=self.cfg.get('scheduler_factor', 0.5),
            patience=self.cfg.get('scheduler_patience', 10),
            min_lr=self.cfg.get('scheduler_min_lr', 1e-6),
            threshold=1e-4,
        )

    def _make_loader(self, x, y, shuffle):
        ds = TensorDataset(x, y)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            pin_memory=True, num_workers=0, drop_last=False,
        )

    def _forward_loss(self, x_batch, y_batch):
        losses = self.density_estimator.loss(y_batch, condition=x_batch)
        return losses.mean()

    def _save_checkpoint(self, path, epoch, val_loss):
        state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'density_estimator_state_dict': self.density_estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        try:
            torch.save(state, tmp_path)
            shutil.move(tmp_path, path)
        except Exception:
            os.close(tmp_fd)
            raise

    def _plot_loss(self, trn_loss, val_loss):
        loss_dir = self.save_dir + 'loss/'
        os.makedirs(loss_dir, exist_ok=True)
        plt.figure()
        plt.plot(trn_loss, label='Training')
        plt.plot(val_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('NLL')
        plt.legend()
        plt.savefig(loss_dir + 'loss_density_estimator.pdf')
        plt.close()
        np.save(loss_dir + 'loss_density_estimator.npy', [trn_loss, val_loss])

    def main(self):
        model_dir = self.save_dir + 'models/density_estimator/'
        os.makedirs(model_dir, exist_ok=True)

        trn_loader = self._make_loader(
            self.data['x_train'], self.data['y_train'], shuffle=True)
        val_loader = self._make_loader(
            self.data['x_val'], self.data['y_val'], shuffle=False)

        trn_losses, val_losses = [], []
        best_val_loss = float('inf')
        ema_val_loss = None
        ema_alpha = 0.3
        patience_counter = 0

        separator()
        logging.info(
            f"spectra-train: {self.epochs} epochs, batch_size={self.batch_size}, "
            f"lr={self.lr}, feature_dim={self.data['feature_dim']}"
        )
        separator()

        start_time = time.time()
        for epoch in range(self.epochs):
            t0 = time.time()
            self.density_estimator.train()
            self.optimizer.zero_grad(set_to_none=True)

            epoch_loss = 0.0
            n_batches = 0
            for step, (x_batch, y_batch) in enumerate(trn_loader):
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                loss = self._forward_loss(x_batch, y_batch) / self.grad_accum_steps
                loss.backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    if self.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.density_estimator.parameters(), self.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * self.grad_accum_steps
                n_batches += 1
            trn_loss = epoch_loss / max(n_batches, 1)

            # Validation
            self.density_estimator.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    val_loss += self._forward_loss(x_batch, y_batch).item()
                    n_val_batches += 1
            val_loss /= max(n_val_batches, 1)

            ema_val_loss = (val_loss if ema_val_loss is None
                            else ema_alpha * val_loss
                                 + (1 - ema_alpha) * ema_val_loss)
            self.scheduler.step(ema_val_loss)

            trn_losses.append(trn_loss)
            val_losses.append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            sec = time.time() - t0
            logging.info(
                f"Epoch {epoch + 1}/{self.epochs}  "
                f"trn_loss={trn_loss:.4f}  val_loss={val_loss:.4f}  "
                f"ema_val={ema_val_loss:.4f}  lr={current_lr:.2e}  "
                f"{sec:.1f}s"
            )

            # Best checkpoint (on EMA val loss).
            if ema_val_loss < best_val_loss:
                best_val_loss = ema_val_loss
                patience_counter = 0
                self._save_checkpoint(
                    model_dir + 'best.pth', epoch, ema_val_loss)
                logging.info(f"  -> new best ema_val_loss: {ema_val_loss:.4f}")
            else:
                patience_counter += 1

            # Periodic save + loss plot.
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                self._save_checkpoint(
                    model_dir + f'epoch_{epoch + 1}.pth', epoch, val_loss)
            self._plot_loss(trn_losses, val_losses)

            # Divergence check.
            if not math.isfinite(val_loss) or (
                    epoch >= 5 and val_loss > best_val_loss + 1000):
                logging.warning(
                    f"DIVERGENCE at epoch {epoch + 1}, val_loss={val_loss:.4f}"
                )
                break

            # Early stopping.
            if self.patience and patience_counter >= self.patience:
                logging.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(patience={self.patience})"
                )
                break

        self._save_checkpoint(model_dir + 'final.pth', epoch, val_loss)
        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
        separator()
        logging.info(f"Training time: {h}h {m}m {s}s")
        logging.info(f"Best ema_val_loss: {best_val_loss:.4f}")
        separator()
        return best_val_loss

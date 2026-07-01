"""Embedding networks for the spectra pipeline.

The embedding sits between the (already-vectorized) P+B feature input and the
normalizing flow. Built via a factory so different branches (mlp, pca, none)
plug in without touching the trainer.
"""
from __future__ import annotations  # lazy annotations (e.g. `torch.Tensor | None`)

import logging

import torch
import torch.nn as nn


class MLPEmbedding(nn.Module):
    """Compress the P+B feature vector to a small embedding before the flow.

    Trained end-to-end with the density estimator under the NLL loss.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

    def forward(self, x):
        return self.net(x)


class _Identity(nn.Module):
    """nn.Identity with an `embedding_dim` attribute matching the input."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.embedding_dim = input_dim
        self.input_dim = input_dim
        # sbi's check_net_device calls next(net.parameters()); give it one.
        self._device_marker = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return x


class PCAEmbedding(nn.Module):
    """Frozen PCA projection fit on training features.

    Stores per-feature mean and the top-k principal components as buffers so
    they travel with the model checkpoint. No learnable parameters — the flow
    sees a fixed linear compression of the (already z-scored) feature vector.
    """

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        if embedding_dim > input_dim:
            raise ValueError(
                f"PCA embedding_dim={embedding_dim} > input_dim={input_dim}"
            )
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("components", torch.zeros(embedding_dim, input_dim))
        self.register_buffer("explained_variance_ratio", torch.zeros(embedding_dim))
        # sbi's check_net_device calls next(net.parameters()); give it one.
        self._device_marker = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._fitted = False

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"PCA fit expects (N, {self.input_dim}); got {tuple(x.shape)}"
            )
        x = x.detach().to(torch.float32).cpu()
        mean = x.mean(dim=0)
        centered = x - mean
        # economy SVD: centered = U S Vh, components are rows of Vh
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[: self.embedding_dim]
        var = (s ** 2) / max(centered.shape[0] - 1, 1)
        total = var.sum().clamp_min(1e-30)
        var_top = var[: self.embedding_dim]
        ratio = (var_top / total).to(torch.float32)
        # Whiten: divide each component by its PC std so the embedding output
        # has ~unit variance per dim on the training set. Without this, the
        # MAF conditioning sees wildly varying per-dim scales (eigenvalues
        # span orders of magnitude) and trains poorly.
        component_std = var_top.clamp_min(1e-8).sqrt()
        components = components / component_std.unsqueeze(1)
        self.mean.copy_(mean.to(self.mean.dtype))
        self.components.copy_(components.to(self.components.dtype))
        self.explained_variance_ratio.copy_(ratio.to(self.explained_variance_ratio.dtype))
        self._fitted = True

    def forward(self, x):
        return (x - self.mean) @ self.components.T


def build_embedding(
    params: dict,
    input_dim: int,
    x_train: torch.Tensor | None = None,
) -> nn.Module:
    """Factory: dispatches on params['spectra']['embedding']['type'].

    If the `embedding` block is absent, or `type` is one of {'none', 'identity',
    None}, the raw feature vector conditions the flow directly.

    `x_train` is only consulted for `type: pca` (used to fit the projection).
    """
    cfg = params['spectra'].get('embedding') or {}
    kind = cfg.get('type', 'none')
    if kind in (None, 'none', 'identity'):
        logging.info(f"No embedding (pass-through): input_dim={input_dim}")
        return _Identity(input_dim)
    if kind == 'mlp':
        emb = MLPEmbedding(
            input_dim=input_dim,
            embedding_dim=cfg.get('embedding_dim', 64),
            hidden=cfg.get('hidden', 256),
        )
        n = sum(p.numel() for p in emb.parameters())
        logging.info(
            f"MLPEmbedding: input_dim={input_dim}, "
            f"embedding_dim={emb.embedding_dim}, params={n:,}"
        )
        return emb
    if kind == 'pca':
        embedding_dim = cfg.get('embedding_dim', 64)
        emb = PCAEmbedding(input_dim=input_dim, embedding_dim=embedding_dim)
        if x_train is None:
            logging.warning(
                "PCAEmbedding built without x_train; buffers left at zero. "
                "Expecting checkpoint load to populate them."
            )
        else:
            emb.fit(x_train)
            cum = float(emb.explained_variance_ratio.sum().item())
            logging.info(
                f"PCAEmbedding: input_dim={input_dim}, "
                f"embedding_dim={embedding_dim}, "
                f"explained_variance={cum:.4f} (cumulative over {embedding_dim} comps)"
            )
        return emb
    raise ValueError(f"Unknown embedding type: {kind}")

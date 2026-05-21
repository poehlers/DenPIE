"""Embedding networks for the spectra pipeline.

The embedding sits between the (already-vectorized) P+B feature input and the
normalizing flow. Built via a factory so a future PCA branch can be added
without touching the trainer.
"""
import logging

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

    def forward(self, x):
        return x


def build_embedding(params: dict, input_dim: int) -> nn.Module:
    """Factory: dispatches on params['spectra']['embedding']['type'].

    If the `embedding` block is absent, or `type` is one of {'none', 'identity',
    None}, the raw feature vector conditions the flow directly.
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
    raise ValueError(f"Unknown embedding type: {kind}")

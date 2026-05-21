"""Build the sbi density estimator for the spectra pipeline.

Thin wrapper around `sbi.neural_nets.posterior_nn` configured for vector-valued
context (the P+B feature vector or its embedding).
"""
import logging

import torch
import torch.nn as nn


def build_sbi_density_estimator(params: dict, embedding_net: nn.Module,
                                x_sample: torch.Tensor, y_sample: torch.Tensor):
    """Build an sbi density estimator wrapping `embedding_net` as embedding_net.

    Args:
        params: full parameter dict.
        embedding_net: MLPEmbedding / Identity / (future) PCAEmbedding.
        x_sample: small batch of training inputs for shape inference, shape
            (>=2, n_features).
        y_sample: small batch of training labels for shape inference, shape
            (>=2, n_active_params).
    """
    from sbi.neural_nets import posterior_nn

    cfg = params['spectra']['flow']
    flow_type = cfg['type']
    if flow_type not in ('maf', 'nsf'):
        raise ValueError(
            f"Unsupported flow type '{flow_type}'. Use 'maf' or 'nsf'."
        )

    build_fn = posterior_nn(
        model=flow_type,
        z_score_theta='independent',
        z_score_x=None,
        hidden_features=cfg.get('hidden_features', 512),
        num_transforms=cfg.get('num_transforms', 15),
        num_bins=cfg.get('num_bins', 8),
        embedding_net=embedding_net,
    )
    density_estimator = build_fn(y_sample, x_sample)
    n_total = sum(p.numel() for p in density_estimator.parameters())
    logging.info(
        f"SBI density estimator ({flow_type}): parameters={n_total:,}"
    )
    return density_estimator

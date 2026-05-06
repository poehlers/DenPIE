import logging
import math

import torch
import torch.nn as nn
import numpy as np

def build_sbi_density_estimator(params: dict, encoder: nn.Module,
                                x_sample: torch.Tensor, y_sample: torch.Tensor):
    """Build SBI density estimator wrapping encoder as embedding_net.

    Args:
        params: full parameter dict
        encoder: pretrained or freshly built encoder (becomes embedding_net)
        x_sample: small batch of training inputs for shape inference, e.g. (2, 1, 128, 128, 128)
        y_sample: small batch of training labels for shape inference, e.g. (2, n_params)
    Returns:
        density_estimator: SBI neural network module
    """
    from sbi.neural_nets import posterior_nn

    cfg = params['density']['flow']
    flow_type = cfg['type']
    if flow_type == 'freia':
        raise ValueError(
            "SBI does not support FrEIA flows. Set use_sbi: false or use nsf/maf.")

    build_fn = posterior_nn(
        model=flow_type,
        z_score_theta='independent',
        z_score_x=None,
        hidden_features=cfg.get('hidden_features', 128),
        num_transforms=cfg.get('num_transforms', 10),
        num_bins=cfg.get('num_bins', 8),
        embedding_net=encoder,
    )
    density_estimator = build_fn(y_sample, x_sample)
    n_total = sum(p.numel() for p in density_estimator.parameters())
    logging.info(f"SBI density estimator ({flow_type}): parameters={n_total:,}")
    return density_estimator


def build_sbi_posterior(density_estimator, active_params, device: torch.device):
    """Build DirectPosterior from a trained SBI density estimator.

    Args:
        density_estimator: trained SBI density estimator
        active_params: list of active parameter names, used to construct the
            per-parameter composite prior in physical units.
        device: torch device
    Returns:
        DirectPosterior object with .sample_batched() and .log_prob()
    """
    from sbi.inference.posteriors.direct_posterior import DirectPosterior
    from .priors import build_composite_prior

    prior = build_composite_prior(active_params, device)
    return DirectPosterior(density_estimator, prior, device=str(device))


def build_flow(params: dict) -> nn.Module:
    """Factory: dispatches on params['density']['flow']['type']."""
    cfg = params['density']['flow']
    encoder_cfg = params['density']['encoder']
    n_params = len(params['density']['data']['active_params'])
    summary_dim = encoder_cfg['summary_dim']
    flow_type = cfg['type']

    if flow_type == 'nsf':
        return NSFFlow(
            n_params=n_params,
            context_dim=summary_dim,
            num_transforms=cfg.get('num_transforms', 10),
            hidden_features=cfg.get('hidden_features', 128),
            num_bins=cfg.get('num_bins', 8),
            dropout=cfg.get('dropout', 0.1),
        )
    elif flow_type == 'maf':
        return MAFFlow(
            n_params=n_params,
            context_dim=summary_dim,
            num_transforms=cfg.get('num_transforms', 10),
            hidden_features=cfg.get('hidden_features', 128),
            dropout=cfg.get('dropout', 0.1),
            use_batch_norm=cfg.get('use_batch_norm', False),
        )
    elif flow_type == 'freia':
        return FrEIAFlowWrapper(
            n_params=n_params,
            cond_dim=summary_dim,
            n_blocks=cfg.get('n_blocks', 8),
            n_nodes=cfg.get('n_nodes', 256),
        )
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")


class NSFFlow(nn.Module):
    """Neural Spline Flow using nflows library.

    Ported from ViT_NSF's SBINSF. Uses MaskedPiecewiseRationalQuadratic
    AutoregressiveTransform + ReversePermutation.

    Interface:
        forward(theta, context) -> log_prob  (shape: B)
        sample(num_samples, context) -> samples  (shape: B, num_samples, n_params)
    """

    def __init__(self, n_params, context_dim, num_transforms=10,
                 hidden_features=128, num_bins=8, dropout=0.1):
        super().__init__()
        self.n_params = n_params

        from nflows.flows import Flow
        from nflows.distributions import StandardNormal
        from nflows.transforms import (
            CompositeTransform,
            ReversePermutation,
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )

        transforms = []
        for _ in range(num_transforms):
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=n_params,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=3.0,
                    dropout_probability=dropout,
                )
            )
            transforms.append(ReversePermutation(n_params))

        self.flow = Flow(CompositeTransform(transforms), StandardNormal([n_params]))
        n_total = sum(p.numel() for p in self.flow.parameters())
        logging.info(f"NSFFlow: n_params={n_params}, context_dim={context_dim}, "
                     f"transforms={num_transforms}, bins={num_bins}, parameters={n_total:,}")

    def forward(self, theta, context=None):
        """Compute log probability. Returns shape (B,)."""
        return self.flow.log_prob(theta, context=context)

    def sample(self, num_samples, context=None):
        """Sample from the flow. Returns shape (B, num_samples, n_params)."""
        return self.flow.sample(num_samples, context=context)


class MAFFlow(nn.Module):
    """Masked Autoregressive Flow using nflows library.

    Uses MaskedAffineAutoregressiveTransform + ReversePermutation.
    Simpler affine transforms (vs NSF's spline transforms), faster to
    train but less expressive per layer.

    Interface:
        forward(theta, context) -> log_prob  (shape: B)
        sample(num_samples, context) -> samples  (shape: B, num_samples, n_params)
    """

    def __init__(self, n_params, context_dim, num_transforms=10,
                 hidden_features=128, dropout=0.1, use_batch_norm=False):
        super().__init__()
        self.n_params = n_params

        from nflows.flows import Flow
        from nflows.distributions import StandardNormal
        from nflows.transforms import (
            CompositeTransform,
            ReversePermutation,
            MaskedAffineAutoregressiveTransform,
            BatchNorm,
        )

        transforms = []
        for _ in range(num_transforms):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=n_params,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                    dropout_probability=dropout,
                )
            )
            if use_batch_norm:
                transforms.append(BatchNorm(n_params))
            transforms.append(ReversePermutation(n_params))

        self.flow = Flow(CompositeTransform(transforms), StandardNormal([n_params]))
        n_total = sum(p.numel() for p in self.flow.parameters())
        logging.info(f"MAFFlow: n_params={n_params}, context_dim={context_dim}, "
                     f"transforms={num_transforms}, batch_norm={use_batch_norm}, "
                     f"parameters={n_total:,}")

    def forward(self, theta, context=None):
        """Compute log probability. Returns shape (B,)."""
        return self.flow.log_prob(theta, context=context)

    def sample(self, num_samples, context=None):
        """Sample from the flow. Returns shape (B, num_samples, n_params)."""
        return self.flow.sample(num_samples, context=context)


class FrEIAFlowWrapper(nn.Module):
    """Wraps existing FrEIA ConditionalInvertibleBlock pattern.

    Adapts FrEIA's (z, jac) = flow(theta, c=[cond]) interface to the
    same forward/sample API as NSFFlow.

    Interface:
        forward(theta, context) -> log_prob  (shape: B)
        sample(num_samples, context) -> samples  (shape: B, num_samples, n_params)
    """

    def __init__(self, n_params, cond_dim, n_blocks=8, n_nodes=256):
        super().__init__()
        import FrEIA.framework as Ff
        import FrEIA.modules as Fm

        self.n_params = n_params
        self._log_2pi = math.log(2 * math.pi)

        def subnet_fc(dims_in, dims_out):
            return nn.Sequential(
                nn.Linear(dims_in, n_nodes),
                nn.ReLU(),
                nn.Linear(n_nodes, dims_out),
            )

        permute_soft = n_params > 1
        self.inn = Ff.SequenceINN(n_params)
        for _ in range(n_blocks):
            self.inn.append(
                Fm.AllInOneBlock,
                cond=0,
                cond_shape=(cond_dim,),
                subnet_constructor=subnet_fc,
                permute_soft=permute_soft,
            )

        n_total = sum(p.numel() for p in self.inn.parameters())
        logging.info(f"FrEIAFlowWrapper: n_params={n_params}, cond_dim={cond_dim}, "
                     f"blocks={n_blocks}, nodes={n_nodes}, parameters={n_total:,}")

    def forward(self, theta, context=None):
        """Compute log probability. Returns shape (B,)."""
        z, jac = self.inn(theta, c=[context])
        log_prob = -0.5 * torch.sum(z ** 2, dim=1) + jac - 0.5 * self.n_params * self._log_2pi
        return log_prob

    def sample(self, num_samples, context=None):
        """Sample from the flow. Returns shape (B, num_samples, n_params).

        context: (B, context_dim)
        """
        B = context.shape[0]
        context_expanded = context.unsqueeze(1).expand(-1, num_samples, -1).reshape(B * num_samples, -1)
        z = torch.randn(B * num_samples, self.n_params, device=context.device)
        samples, _ = self.inn(z, c=[context_expanded], rev=True)
        return samples.reshape(B, num_samples, self.n_params)

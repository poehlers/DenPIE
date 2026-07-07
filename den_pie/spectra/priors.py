"""Prior construction for the spectra pipeline.

Mixed uniform/normal priors via sbi.utils.MultipleIndependent. Reads the
per-parameter prior tuples from the dataset registry and filters by
`active_params`.
"""
import torch

from .registry import DATASET_REGISTRY


def build_sbi_prior(dataset_key: str, active_params: list, device):
    """Build a sbi MultipleIndependent prior over the active parameters.

    Args:
        dataset_key: key into DATASET_REGISTRY.
        active_params: ordered list of parameter names; must be a subset of
            the dataset's `param_names`.
        device: torch device (or str).
    """
    from sbi.utils import BoxUniform, MultipleIndependent

    ds = DATASET_REGISTRY[dataset_key]
    name_to_prior = dict(zip(ds['param_names'], ds['param_priors']))
    dev = str(device) if not isinstance(device, str) else device

    dists = []
    for p in active_params:
        if p not in name_to_prior:
            raise KeyError(
                f"Parameter '{p}' not in dataset '{dataset_key}' "
                f"(known: {ds['param_names']})"
            )
        kind, a, b = name_to_prior[p]
        if kind == 'uniform':
            dists.append(BoxUniform(
                low=torch.tensor([a], device=dev),
                high=torch.tensor([b], device=dev),
                device=dev,
            ))
        elif kind == 'normal':
            dists.append(torch.distributions.Normal(
                torch.tensor([a], device=dev),
                torch.tensor([b], device=dev),
            ))
        else:
            raise ValueError(f"Unknown prior type '{kind}' for {p}")

    return MultipleIndependent(dists, validate_args=False, device=dev)


def build_sbi_posterior(density_estimator, dataset_key, active_params, device):
    """Build DirectPosterior from a trained spectra density estimator."""
    from sbi.inference.posteriors.direct_posterior import DirectPosterior

    prior = build_sbi_prior(dataset_key, active_params, device)
    return DirectPosterior(density_estimator, prior, device=str(device))

"""Parameter definitions and prior construction for density mode."""
import torch


DENSITY_PARAMS = {
    'Omega_b': {
        'name': 'Omega_b', 'latex': r'$\Omega_b$', 'latex_bare': r'\Omega_b',
        'range': [0.04, 0.06],
        'prior': {'type': 'uniform'},
    },
    'h': {
        'name': 'h', 'latex': r'$h$', 'latex_bare': r'h',
        'range': [0.6, 0.8],
        'prior': {'type': 'uniform'},
    },
    'n_s': {
        'name': 'n_s', 'latex': r'$n_s$', 'latex_bare': r'n_s',
        'range': [0.9, 1.05],
        'prior': {'type': 'uniform'},
    },
    'Omega_m': {
        'name': 'Omega_m', 'latex': r'$\Omega_m$', 'latex_bare': r'\Omega_m',
        'range': [0.1, 0.5],
        'prior': {'type': 'uniform'},
    },
    'sigma_8': {
        'name': 'sigma_8', 'latex': r'$\sigma_8$', 'latex_bare': r'\sigma_8',
        'range': [0.6, 1.0],
        'prior': {'type': 'uniform'},
    },
    'b1': {
        'name': 'b1', 'latex': r'$b_1$', 'latex_bare': r'b_1',
        'range': [-1.1, 2.9],
        'prior': {'type': 'normal', 'mean': 1.0, 'std': 0.5},
    },
    'b2': {
        'name': 'b2', 'latex': r'$b_2$', 'latex_bare': r'b_2',
        'range': [-7.2, 8.2],
        'prior': {'type': 'normal', 'mean': 0.0, 'std': 2.0},
    },
    'bs2': {
        'name': 'bs2', 'latex': r'$b_{s^2}$', 'latex_bare': r'b_{s^2}',
        'range': [-8.0, 7.8],
        'prior': {'type': 'normal', 'mean': 0.0, 'std': 2.0},
    },
    'bn2': {
        'name': 'bn2', 'latex': r'$b_{\nabla^2}$', 'latex_bare': r'b_{\nabla^2}',
        'range': [-6.7, 6.9],
        'prior': {'type': 'normal', 'mean': 0.0, 'std': 2.0},
    },
}


def build_composite_prior(active_params, device):
    """Build a per-parameter composite prior in physical units.

    Uniform-prior params use their `range` as the BoxUniform support;
    normal-prior params use Normal(mean, std).
    """
    from sbi.utils import BoxUniform, MultipleIndependent

    dists = []
    dev = str(device) if not isinstance(device, str) else device
    for p in active_params:
        pdef = DENSITY_PARAMS[p]
        prior = pdef['prior']
        if prior['type'] == 'uniform':
            lo, hi = pdef['range']
            dists.append(BoxUniform(
                low=torch.tensor([lo], device=dev),
                high=torch.tensor([hi], device=dev),
                device=dev,
            ))
        elif prior['type'] == 'normal':
            mean = torch.tensor([prior['mean']], device=dev)
            std = torch.tensor([prior['std']], device=dev)
            dists.append(torch.distributions.Normal(mean, std))
        else:
            raise ValueError(f"Unknown prior type for {p}: {prior['type']}")

    return MultipleIndependent(dists, validate_args=False, device=dev)

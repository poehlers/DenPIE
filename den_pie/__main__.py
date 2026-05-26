import argparse
import os
import socket
import logging
import torch

from .util.parse import parse, setup_dir, find_run, log_yaml
from .util.logger import init_logger, separator

def main():
    """
    Entry point of the program.
    Parses command line arguments and executes the corresponding function based on the subcommand.

    Usage:
        python -m den_pie density-train  <paramcard> [--verbose]
        python -m den_pie density-plot   <paramcard> [--verbose]
        python -m den_pie density-search <paramcard> [--verbose]
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    density_train_parser = subparsers.add_parser("density-train")
    density_train_parser.add_argument("paramcard")
    density_train_parser.add_argument("--verbose", action="store_true")
    density_train_parser.set_defaults(func=density_train)

    density_plot_parser = subparsers.add_parser("density-plot")
    density_plot_parser.add_argument("paramcard")
    density_plot_parser.add_argument("--verbose", action="store_true")
    density_plot_parser.set_defaults(func=density_plot)

    density_search_parser = subparsers.add_parser("density-search")
    density_search_parser.add_argument("paramcard")
    density_search_parser.add_argument("--verbose", action="store_true")
    density_search_parser.set_defaults(func=density_search)

    spectra_train_parser = subparsers.add_parser("spectra-train")
    spectra_train_parser.add_argument("paramcard")
    spectra_train_parser.add_argument("--verbose", action="store_true")
    spectra_train_parser.set_defaults(func=spectra_train)

    spectra_plot_parser = subparsers.add_parser("spectra-plot")
    spectra_plot_parser.add_argument("paramcard")
    spectra_plot_parser.add_argument("--verbose", action="store_true")
    spectra_plot_parser.set_defaults(func=spectra_plot)

    args = parser.parse_args()
    args.func(args)


def density_init(params):
    """Initialize density-mode components: encoder, optional flow, data loader, device."""
    from .density.encoder import build_encoder
    from .density.data import FLIDataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Load data first (SBI needs a data sample for shape inference)
    data_loader = FLIDataLoader(params)
    data = data_loader.data

    use_sbi = params['density']['flow'].get('use_sbi', False)

    encoder = build_encoder(params)

    # Load pretrained encoder weights if specified
    enc_cfg = params['density']['encoder']
    if enc_cfg.get('load', False) and enc_cfg.get('model_location'):
        ckpt = torch.load(enc_cfg['model_location'], map_location=device)
        if 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        elif 'density_estimator_state_dict' in ckpt:
            # Extract encoder weights from SBI density_estimator checkpoint
            # SBI wraps encoder as net._embedding_net; extract matching keys
            de_state = ckpt['density_estimator_state_dict']
            prefix = 'net._embedding_net.'
            enc_state = {k[len(prefix):]: v for k, v in de_state.items()
                         if k.startswith(prefix)}
            encoder.load_state_dict(enc_state)
        else:
            raise KeyError(f"Checkpoint has neither 'encoder_state_dict' nor "
                           f"'density_estimator_state_dict': {list(ckpt.keys())}")
        logging.info(f"Loaded encoder from {enc_cfg['model_location']}")

    if use_sbi:
        from .density.flow import build_sbi_density_estimator
        logging.info("Building SBI density estimator...")
        x_sample = torch.stack(
            [data['x_train'][i][0] for i in range(2)], dim=0
        ).to(device)
        y_sample = data['y_train'].to(device)
        density_estimator = build_sbi_density_estimator(
            params, encoder, x_sample, y_sample)

        # Load pretrained density_estimator if specified
        flow_cfg = params['density']['flow']
        if flow_cfg.get('load', False) and flow_cfg.get('model_location'):
            ckpt = torch.load(flow_cfg['model_location'], map_location=device)
            if 'density_estimator_state_dict' in ckpt:
                density_estimator.load_state_dict(ckpt['density_estimator_state_dict'])
            logging.info(f"Loaded SBI density estimator from {flow_cfg['model_location']}")

        return encoder, None, data, device, density_estimator
    else:
        from .density.flow import build_flow
        flow = build_flow(params)
        flow_cfg = params['density']['flow']
        if flow_cfg.get('load', False) and flow_cfg.get('model_location'):
            ckpt = torch.load(flow_cfg['model_location'], map_location=device)
            flow.load_state_dict(ckpt['flow_state_dict'])
            logging.info(f"Loaded flow from {flow_cfg['model_location']}")

        return encoder, flow, data, device, None

def density_train(args: argparse.Namespace) -> None:
    """Train density field inference model."""
    from .density.train import DensityTraining

    params = parse(args.paramcard)
    run_name = setup_dir(args.paramcard)
    params['name'] = run_name
    init_logger(fn=run_name, verbose=args.verbose)
    logging.info(f'{socket.gethostname()}: density-train starting')
    log_yaml(params, source=args.paramcard)
    separator()

    encoder, flow, data, device, density_estimator = density_init(params)
    DensityTraining(params, encoder, flow, data, device,
                    density_estimator=density_estimator).main()

def density_plot(args: argparse.Namespace) -> None:
    """Evaluate and plot density field inference model."""
    from .density.eval import DensityPlotting
    import sys

    params = parse(args.paramcard)
    run_name, run_exists = find_run(params['name'])
    if not run_exists:
        logging.error("No run found with the same name")
        sys.exit(1)
    plot_dir = run_name + '/plots/'
    os.makedirs(plot_dir, exist_ok=True)
    params['name'] = run_name
    params.setdefault('density', {}).setdefault('plot', {})
    params['density']['plot']['plot_dir'] = plot_dir
    init_logger(fn=plot_dir, verbose=args.verbose)
    logging.info(f'{socket.gethostname()}: density-plot starting')

    encoder, flow, data, device, density_estimator = density_init(params)
    DensityPlotting(params, encoder, flow, data, device,
                    density_estimator=density_estimator).main()

def density_search(args: argparse.Namespace) -> None:
    """Run Optuna hyperparameter search for density mode."""
    from .density.optuna_search import run_search

    init_logger(fn='output/optuna_search', verbose=args.verbose)
    logging.info('density-search starting')
    run_search(args.paramcard)


def spectra_init(params: dict):
    """Initialize spectra-mode components: data, embedding, density estimator."""
    from .spectra.data import SpectraDataLoader
    from .spectra.embedding import build_embedding
    from .spectra.flow import build_sbi_density_estimator

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    data_loader = SpectraDataLoader(params)
    data = data_loader.data

    embedding_net = build_embedding(
        params, input_dim=data['feature_dim'], x_train=data['x_train']
    )

    # SBI needs sample tensors for shape inference.
    x_sample = data['x_train'][:2].to(device)
    y_sample = data['y_train'][:2].to(device)
    density_estimator = build_sbi_density_estimator(
        params, embedding_net, x_sample, y_sample)

    flow_cfg = params['spectra']['flow']
    if flow_cfg.get('load', False) and flow_cfg.get('model_location'):
        ckpt = torch.load(flow_cfg['model_location'], map_location=device)
        density_estimator.load_state_dict(ckpt['density_estimator_state_dict'])
        logging.info(
            f"Loaded spectra density estimator from {flow_cfg['model_location']}"
        )

    return data, device, density_estimator


def spectra_train(args: argparse.Namespace) -> None:
    """Train the spectra-mode SBI density estimator."""
    from .spectra.train import SpectraTraining

    params = parse(args.paramcard)
    run_name = setup_dir(args.paramcard)
    params['name'] = run_name
    init_logger(fn=run_name, verbose=args.verbose)
    logging.info(f'{socket.gethostname()}: spectra-train starting')
    log_yaml(params, source=args.paramcard)
    separator()

    data, device, density_estimator = spectra_init(params)
    SpectraTraining(params, density_estimator, data, device).main()


def spectra_plot(args: argparse.Namespace) -> None:
    """Evaluate and plot the spectra-mode SBI model."""
    from .spectra.eval import SpectraPlotting
    import sys

    params = parse(args.paramcard)
    run_name, run_exists = find_run(params['name'])
    if not run_exists:
        logging.error("No run found with the same name")
        sys.exit(1)
    plot_dir = run_name + '/plots/'
    os.makedirs(plot_dir, exist_ok=True)
    params['name'] = run_name
    params.setdefault('spectra', {}).setdefault('plot', {})
    params['spectra']['plot']['plot_dir'] = plot_dir
    init_logger(fn=plot_dir, verbose=args.verbose)
    logging.info(f'{socket.gethostname()}: spectra-plot starting')

    # If the user didn't set flow.load explicitly, auto-load best checkpoint.
    flow_cfg = params['spectra']['flow']
    if not flow_cfg.get('load', False):
        best = os.path.join(run_name, 'models/density_estimator/best.pth')
        if os.path.isfile(best):
            flow_cfg['load'] = True
            flow_cfg['model_location'] = best
            logging.info(f"Auto-loading best checkpoint: {best}")

    data, device, density_estimator = spectra_init(params)
    SpectraPlotting(params, density_estimator, data, device).main()


if __name__ == "__main__":
    main()

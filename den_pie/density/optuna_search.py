import copy
import gc
import logging
import os
from datetime import datetime

import optuna
import torch
import yaml


def build_trial_params(trial, search_cfg):
    """Construct a full params dict from Optuna trial suggestions.

    Fixed architecture: cnn3d encoder + maf flow (SBI).
    Searches over training hyperparameters only.

    Args:
        trial: optuna.Trial
        search_cfg: dict loaded from optuna_search.yaml

    Returns:
        Complete params dict matching the density training YAML structure.
    """
    space = search_cfg.get('search_space', {})

    # Fixed architecture
    encoder_type = 'cnn3d'
    flow_type = 'maf'
    use_sbi = True

    # Hyperparameters to search
    summary_dim = trial.suggest_categorical(
        'summary_dim', space.get('summary_dim', [64, 128, 256, 512]))

    lr_cfg = space.get('lr', {'low': 1e-6, 'high': 1e-3, 'log': True})
    lr = trial.suggest_float('lr', lr_cfg['low'], lr_cfg['high'], log=lr_cfg.get('log', True))

    batch_size = trial.suggest_categorical(
        'batch_size', space.get('batch_size', [8, 16, 32]))

    wd_cfg = space.get('weight_decay', {'low': 1e-4, 'high': 1e-1, 'log': True})
    weight_decay = trial.suggest_float(
        'weight_decay', wd_cfg['low'], wd_cfg['high'], log=wd_cfg.get('log', True))

    dropout_cfg = space.get('dropout', {'low': 0.0, 'high': 0.3})
    dropout = trial.suggest_float('dropout', dropout_cfg['low'], dropout_cfg['high'])

    gc_cfg = space.get('grad_clip', {'low': 50.0, 'high': 500.0})
    grad_clip = trial.suggest_float('grad_clip', gc_cfg['low'], gc_cfg['high'])

    encoder_cfg = {
        'type': encoder_type,
        'summary_dim': summary_dim,
        'use_checkpointing': True,
        'proj_dropout': 0.0,
        'load': False,
        'model_location': '',
    }

    # MAF-specific hyperparameters
    nt_cfg = space.get('num_transforms', {'low': 4, 'high': 16})
    num_transforms = trial.suggest_int(
        'num_transforms', nt_cfg['low'], nt_cfg['high'])
    hidden_features = trial.suggest_categorical(
        'hidden_features', space.get('hidden_features', [64, 128, 256]))

    flow_cfg = {
        'type': flow_type,
        'use_sbi': use_sbi,
        'dropout': dropout,
        'num_transforms': num_transforms,
        'hidden_features': hidden_features,
        'load': False,
        'model_location': '',
    }

    search_epochs = search_cfg.get('search_epochs', 150)

    train_cfg = {
        'train_network': 'encoder',  # overridden per stage
        'epochs': search_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'optimizer': 'AdamW',
        'weight_decay': weight_decay,
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_params': {
            'factor': 0.3,
            'patience': 10,
            'min_lr': 1e-7,
        },
        'gradient_accumulation_steps': 2,
        'grad_clip': grad_clip,
        'patience': search_cfg.get('search_patience', 20),
    }

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_name = f"output/optuna_trial_{trial.number}_{now}"

    params = {
        'name': trial_name,
        'density': {
            'encoder': encoder_cfg,
            'flow': flow_cfg,
            'data': search_cfg['data'],
            'train': train_cfg,
        },
    }
    return params


def run_3stage_trial(params, data, trial):
    """Run 3-stage training pipeline, returning stage-3 best val loss.

    Models are kept in memory between stages (no disk checkpoint chaining).

    Args:
        params: full params dict from build_trial_params
        data: pre-loaded data dict from FLIDataLoader
        trial: optuna.Trial (used for pruning in stage 3 only)

    Returns:
        best_val_loss from stage 3 (end-to-end)
    """
    from .encoder import build_encoder
    from .flow import build_sbi_density_estimator
    from .train import DensityTraining

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = build_encoder(params)
    x_sample = torch.stack(
        [data['x_train'][i][0] for i in range(2)], dim=0
    ).to(device)
    y_sample = data['y_train'].to(device)
    density_estimator = build_sbi_density_estimator(
        params, encoder, x_sample, y_sample)
    flow = None

    stages = ['encoder', 'flow', 'both']
    best_val_loss = float('inf')

    for stage in stages:
        stage_params = copy.deepcopy(params)
        stage_params['density']['train']['train_network'] = stage

        # Lower learning rate for fine-tuning stage
        if stage == 'both':
            stage_params['density']['train']['lr'] = params['density']['train']['lr'] * 0.1

        # Use a dummy name (no disk I/O needed during search)
        stage_dir = params['name'] + f'_{stage}/'
        stage_params['name'] = stage_dir

        # Pass trial to all stages to skip checkpoint saving;
        # pruning only triggers in stage 3 (report() called there)
        trainer = DensityTraining(
            stage_params, encoder, flow, data, device,
            density_estimator=density_estimator,
            trial=trial,
        )
        best_val_loss = trainer.main()

    return best_val_loss


def objective(trial, search_cfg, data):
    """Optuna objective function.

    Args:
        trial: optuna.Trial
        search_cfg: search configuration dict
        data: pre-loaded data dict

    Returns:
        best validation loss from stage 3
    """
    params = build_trial_params(trial, search_cfg)

    logging.info(f"=== Trial {trial.number} ===")
    logging.info(f"  encoder={params['density']['encoder']['type']}, "
                 f"flow={params['density']['flow']['type']}, "
                 f"sbi={params['density']['flow']['use_sbi']}")
    logging.info(f"  summary_dim={params['density']['encoder']['summary_dim']}, "
                 f"lr={params['density']['train']['lr']:.2e}, "
                 f"batch_size={params['density']['train']['batch_size']}")

    try:
        best_val_loss = run_3stage_trial(params, data, trial)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logging.warning(f"Trial {trial.number} failed: {e}", exc_info=True)
        raise optuna.TrialPruned()
    finally:
        # Clean up GPU memory between trials
        torch.cuda.empty_cache()
        gc.collect()

    logging.info(f"Trial {trial.number} finished: best_val_loss={best_val_loss:.6f}")
    return best_val_loss


def export_best_yaml(study, search_cfg, output_path):
    """Export the best trial's configuration as a full 3-stage YAML set.

    Args:
        study: completed optuna.Study
        search_cfg: original search config (for data section)
        output_path: directory to write YAML files
    """
    best = study.best_trial
    p = best.params
    os.makedirs(output_path, exist_ok=True)

    encoder_cfg = {
        'type': 'cnn3d',
        'summary_dim': p['summary_dim'],
        'use_checkpointing': True,
        'proj_dropout': 0.0,
        'load': False,
        'model_location': '',
    }

    flow_cfg = {
        'type': 'maf',
        'use_sbi': True,
        'dropout': p['dropout'],
        'num_transforms': p['num_transforms'],
        'hidden_features': p['hidden_features'],
        'load': False,
        'model_location': '',
    }

    base_name = "optuna_best_cnn3d_maf"

    stages = [
        ('stage1_encoder', 'encoder', False, False),
        ('stage2_flow', 'flow', True, False),
        ('stage3_finetune', 'both', True, True),
    ]

    for stage_name, train_network, load_enc, load_flow in stages:
        stage_encoder = copy.deepcopy(encoder_cfg)
        stage_flow = copy.deepcopy(flow_cfg)

        stage_encoder['load'] = load_enc
        if load_enc:
            stage_encoder['model_location'] = '<SET_PATH_TO_STAGE1_ENCODER_BEST>'
        stage_flow['load'] = load_flow
        if load_flow:
            stage_flow['model_location'] = '<SET_PATH_TO_STAGE2_FLOW_BEST>'

        lr = p['lr']
        if train_network == 'both':
            lr *= 0.1

        cfg = {
            'name': f"{base_name}_{stage_name}",
            'density': {
                'encoder': stage_encoder,
                'flow': stage_flow,
                'data': search_cfg['data'],
                'train': {
                    'train_network': train_network,
                    'epochs': 500,
                    'batch_size': p['batch_size'],
                    'lr': lr,
                    'optimizer': 'AdamW',
                    'weight_decay': p['weight_decay'],
                    'scheduler': 'ReduceLROnPlateau',
                    'scheduler_params': {
                        'factor': 0.3,
                        'patience': 10,
                        'min_lr': 1e-7,
                    },
                    'gradient_accumulation_steps': 2,
                    'grad_clip': p['grad_clip'],
                    'patience': 25,
                },
            },
        }

        fname = os.path.join(output_path, f"{base_name}_{stage_name}.yaml")
        with open(fname, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        logging.info(f"Exported: {fname}")


def run_search(config_path):
    """Main entry point for Optuna search.

    Args:
        config_path: path to optuna_search.yaml
    """
    from .data import FLIDataLoader

    with open(config_path, 'r') as f:
        search_cfg = yaml.safe_load(f)

    study_name = search_cfg.get('study_name', 'density_hpo')
    storage = search_cfg.get('storage', 'sqlite:///output/optuna_study.db')
    n_trials = search_cfg.get('n_trials', 50)

    # Ensure output directory for SQLite exists
    db_path = storage.replace('sqlite:///', '')
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)

    # Load data once (expensive, reused across all trials)
    logging.info("Loading data (shared across all trials)...")
    data_params = {'density': {'data': search_cfg['data']}}
    data_loader = FLIDataLoader(data_params)
    data = data_loader.data
    logging.info("Data loaded.")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
        ),
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, search_cfg, data),
        n_trials=n_trials,
    )

    # Report results
    logging.info("=" * 60)
    logging.info("Optuna search complete.")
    logging.info(f"  Best trial: {study.best_trial.number}")
    logging.info(f"  Best val loss: {study.best_value:.6f}")
    logging.info(f"  Best params:")
    for k, v in study.best_params.items():
        logging.info(f"    {k}: {v}")

    # Export best config as YAML files for full retraining
    export_dir = 'params/optuna_best/'
    export_best_yaml(study, search_cfg, export_dir)
    logging.info(f"Best trial configs exported to {export_dir}")
    logging.info("Update model_location paths in stage2/stage3 YAMLs before retraining.")

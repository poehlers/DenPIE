# 21cm-PIE: Project Structure & Usage Guide

## Overview

**21cm-PIE** (Parameter Inference Engine) performs simulation-based inference (SBI)
on 21cm cosmology density fields. It learns a posterior over cosmological and bias
parameters from 3D field data using a two-component model:

- **Encoder** — 3D CNN (`cnn3d`) or Swin Transformer (`swin3d`) that compresses
  a 3D field into a summary statistic.
- **Flow** — Normalizing flow (`maf`, `nsf`, or `freia`) that maps the summary
  statistic to a posterior over parameters.

Training is split into three stages:
1. **Stage 1 (encoder)** — train the encoder alone.
2. **Stage 2 (flow)** — freeze the encoder, train the flow.
3. **Stage 3 (finetune)** — jointly fine-tune encoder + flow end-to-end.

---

## Directory layout

```
den_pie/
├── params/                         # YAML configuration files
│   ├── optuna_search.yaml          # Hyperparameter search config
│   ├── optuna_best/                # Best-trial configs exported after a search
│   │   ├── optuna_best_cnn3d_maf_stage1_encoder.yaml
│   │   ├── optuna_best_cnn3d_maf_stage2_flow.yaml
│   │   └── optuna_best_cnn3d_maf_stage3_finetune.yaml
│   ├── density_bias_train_stage1_encoder_maf.yaml   # Manual (non-HPO) runs
│   ├── density_bias_train_stage2_flow_maf.yaml
│   ├── density_bias_train_stage3_finetune_maf.yaml
│   └── ...                         # Other flow/encoder variant configs
│
├── output/                         # All training artefacts (auto-created)
│   └── <run_name>_<timestamp>/
│       ├── models/
│       │   ├── encoder/
│       │   │   ├── best.pth        # Best checkpoint (lowest val loss)
│       │   │   ├── final.pth
│       │   │   └── epoch_N.pth
│       │   └── flow/               # (or "both/" after stage 3)
│       │       └── best.pth
│       ├── loss/                   # Loss curves (.npy + .pdf)
│       ├── plots/                  # Posterior plots (after density-plot)
│       ├── log                     # Training log
│       └── <paramcard>.yaml        # Copy of the config used for this run
│
├── den_pie/                        # Python package
│   ├── __main__.py                 # CLI entry point (subcommands listed below)
│   ├── density/
│   │   ├── data.py                 # FLIDataLoader — loads field + parameter data
│   │   ├── encoder.py              # build_encoder() — cnn3d / swin3d
│   │   ├── flow.py                 # build_flow() / build_sbi_density_estimator()
│   │   ├── priors.py               # build_composite_prior() — parameter priors
│   │   ├── train.py                # DensityTraining — runs one stage
│   │   ├── eval.py                 # DensityPlotting — posterior corner plots
│   │   └── optuna_search.py        # run_search() — Optuna HPO loop
│   └── util/
│       ├── parse.py                # YAML parsing, output-dir helpers
│       └── logger.py               # Logging setup
│
├── run_density_bias_3stage_maf.sh  # SLURM job: full 3-stage MAF run
├── run_density_bias_3stage_nsf.sh  # SLURM job: full 3-stage NSF run
├── run_optuna_search.sh            # SLURM job: Optuna HPO search
├── run_density_bias_plot_only.sh   # SLURM job: plotting only
└── setup.py
```

---

## CLI subcommands

```
python -m den_pie <subcommand> <paramcard.yaml> [--verbose]
```

| Subcommand        | What it does                                      |
|-------------------|---------------------------------------------------|
| `density-train`   | Train one stage (encoder / flow / both)           |
| `density-plot`    | Evaluate a trained model and produce corner plots |
| `density-search`  | Run Optuna hyperparameter search                  |

---

## Running the best trial from an Optuna search

### Step 1 — Run the HPO search

```bash
sbatch run_optuna_search.sh
```

This reads `params/optuna_search.yaml`, runs `n_trials` Optuna trials, and at the
end exports three YAML files to `params/optuna_best/`:

```
params/optuna_best/optuna_best_cnn3d_maf_stage1_encoder.yaml
params/optuna_best/optuna_best_cnn3d_maf_stage2_flow.yaml
params/optuna_best/optuna_best_cnn3d_maf_stage3_finetune.yaml
```

The SQLite study database is written to `output/optuna_density_bias.db`.

### Step 2 — Inspect the best trial (optional)

```python
import optuna
study = optuna.load_study(
    study_name="density_bias_hpo",
    storage="sqlite:///output/optuna_density_bias.db",
)
print("Best trial:", study.best_trial.number)
print("Best value:", study.best_value)
print("Best params:", study.best_params)
```

### Step 3 — Fix the placeholder model paths in stage 2 and 3

The exported stage 2 and 3 YAMLs contain placeholder strings that must be
replaced with the actual checkpoint paths produced by the previous stage.

**`params/optuna_best/optuna_best_cnn3d_maf_stage2_flow.yaml`** — set
`density.encoder.model_location` to the `best.pth` from the stage 1 output run:

```yaml
density:
  encoder:
    load: true
    model_location: 'output/optuna_best_cnn3d_maf_stage1_encoder_<timestamp>/models/encoder/best.pth'
```

**`params/optuna_best/optuna_best_cnn3d_maf_stage3_finetune.yaml`** — set both
`encoder.model_location` (stage 1 best) and `flow.model_location` (stage 2 best):

```yaml
density:
  encoder:
    load: true
    model_location: 'output/optuna_best_cnn3d_maf_stage1_encoder_<timestamp>/models/encoder/best.pth'
  flow:
    load: true
    model_location: 'output/optuna_best_cnn3d_maf_stage2_flow_<timestamp>/models/flow/best.pth'
```

> **Tip:** Each run creates a timestamped directory **and** a `_latest` symlink, e.g.
> `output/optuna_best_cnn3d_maf_stage1_encoder_latest/`. You can use the symlink
> path in the YAML so you never need to update timestamps manually.

### Step 4 — Train the three stages in sequence

```bash
# Activate the environment first (adjust to your venv)
source ~/21cmvenv/bin/activate

# Stage 1 — encoder only
python -m den_pie density-train \
    params/optuna_best/optuna_best_cnn3d_maf_stage1_encoder.yaml --verbose

# Stage 2 — flow only (encoder frozen)
python -m den_pie density-train \
    params/optuna_best/optuna_best_cnn3d_maf_stage2_flow.yaml --verbose

# Stage 3 — joint fine-tuning
python -m den_pie density-train \
    params/optuna_best/optuna_best_cnn3d_maf_stage3_finetune.yaml --verbose
```

Or submit as a single SLURM job by copying `run_density_bias_3stage_maf.sh` and
replacing the three `density-train` param paths with the `optuna_best/` variants.

### Step 5 — Evaluate and plot

After stage 3 completes, run the evaluator. It looks up the run directory by the
`name` field in the YAML:

```bash
python -m den_pie density-plot \
    params/optuna_best/optuna_best_cnn3d_maf_stage3_finetune.yaml --verbose
```

Plots are written to `output/optuna_best_cnn3d_maf_stage3_finetune_<timestamp>/plots/`.

---

## Output directory naming

Every `density-train` run creates two entries in `output/`:

| Path | Contents |
|------|----------|
| `output/<name>_<YYYYMMDD_HHMMSS>/` | Timestamped, permanent copy |
| `output/<name>_latest/`            | Symlink to the most recent run with that name |

Checkpoints inside a run directory follow the pattern:

```
models/
  encoder/best.pth       # saved whenever val loss improves
  encoder/final.pth      # saved at end of training
  encoder/epoch_N.pth    # periodic snapshots
  flow/best.pth          # (stage 2 and 3)
  both/...               # (stage 3 joint checkpoint)
```

---

## YAML parameter reference (density mode)

```yaml
name: <run_name>          # used as output dir prefix

density:
  encoder:
    type: cnn3d           # cnn3d | swin3d
    summary_dim: 256      # output dimensionality of the encoder
    use_checkpointing: true
    proj_dropout: 0.0
    load: false           # set true to resume from a checkpoint
    model_location: ''    # path to .pth checkpoint (if load: true)

  flow:
    type: maf             # maf | nsf | freia
    use_sbi: true         # use sbi library wrapper (recommended)
    num_transforms: 6
    hidden_features: 128
    dropout: 0.3
    load: false
    model_location: ''

  data:
    data_path: /path/to/samples/
    fiducial_path: /path/to/fiducial/
    field_key: delta_biased
    param_keys: [cosmo_params, bias_params]
    all_param_names: [Omega_m, sigma_8, b1, b2, bs2, bn2]
    active_params: [Omega_m, sigma_8, b1, b2, bs2, bn2]
    val_split: 0.1
    test_split: 0.1
    n_realizations: null  # null = use all available
    log_transform: true

  train:
    train_network: encoder   # encoder | flow | both
    epochs: 500
    batch_size: 32
    lr: 1.0e-5
    optimizer: AdamW
    weight_decay: 0.01
    scheduler: ReduceLROnPlateau
    scheduler_params:
      factor: 0.3
      patience: 10
      min_lr: 1.0e-7
    gradient_accumulation_steps: 2
    grad_clip: 400
    patience: 25             # early stopping patience
```

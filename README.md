<h2 align="center">den_pie — Simulation-Based Inference on Dark Matter Overdensity Fields</h2>

<!-- <p align="center">
<a href="https://arxiv.org/abs/2401.04174"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2401.04174-b31b1b.svg"></a>
</p> -->

`den_pie` is a machine-learning tool for simulation-based inference (SBI) of cosmological and bias parameters from 3D dark matter overdensity fields. It pairs a 3D encoder (CNN or Swin Transformer) with a normalizing flow (MAF, NSF, or FrEIA) and trains the two in three stages: encoder alone, flow alone, then jointly fine-tuned end-to-end.

<img src="animation/animation.gif" width="600" height="600" alt="Animation">

## Installation

```sh
# clone the repository
git clone https://github.com/<TODO-org>/<TODO-repo>
# install in editable mode
cd den_pie
pip install --editable .
```

## Usage

The full reference (CLI subcommands, YAML schema, output layout, and the recommended Optuna-driven workflow) lives in [`STRUCTURE.md`](STRUCTURE.md). A quick tour:

**Hyperparameter search** — runs Optuna and exports the best-trial configs into `params/optuna_best/`:
```
python -m den_pie density-search params/optuna_search.yaml --verbose
```

**Train the three stages** (encoder → flow → joint fine-tune):
```
python -m den_pie density-train params/optuna_best/optuna_best_cnn3d_maf_stage1_encoder.yaml --verbose
python -m den_pie density-train params/optuna_best/optuna_best_cnn3d_maf_stage2_flow.yaml --verbose
python -m den_pie density-train params/optuna_best/optuna_best_cnn3d_maf_stage3_finetune.yaml --verbose
```

**Evaluate** — produces posterior corner plots in `output/<run>/plots/`:
```
python -m den_pie density-plot params/optuna_best/optuna_best_cnn3d_maf_stage3_finetune.yaml --verbose
```

Manual (non-HPO) param cards for the three flow variants live in `params/density_bias_train_stage{1,2,3}_{maf,nsf,freia}.yaml`. SLURM submission scripts for HPC systems are at the project root (`run_*.sh`).

## Acknowledgements

This package builds on the original 21cmPIE-INN work. If you use it, please cite:

```
@article{Schosser:2024aic,
    author = "Schosser, Benedikt and Heneka, Caroline and Plehn, Tilman",
    title = "{Optimal, fast, and robust inference of reionization-era cosmology with the 21cmPIE-INN}",
    eprint = "2401.04174",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "1",
    year = "2024"
}
```

When using the 3D CNN encoder please also cite:

```
@ARTICLE{2022arXiv220107587N,
       author = {{Neutsch}, S. and {Heneka}, C. and {Br{\"u}ggen}, M.},
        title = "{Inferring Astrophysics and Dark Matter Properties from 21cm Tomography using Deep Learning}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = jan,
          eid = {arXiv:2201.07587},
        pages = {arXiv:2201.07587},
archivePrefix = {arXiv},
       eprint = {2201.07587},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220107587N},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

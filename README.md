<h2 align="center">den_pie — Forward model, Fisher forecasts & simulation-based inference for cosmology</h2>

<p align="center">
<a href="https://poehlers.github.io/DenPIE/"><img alt="Docs" src="https://img.shields.io/badge/docs-poehlers.github.io%2FDenPIE-blue.svg"></a>
<a href="https://arxiv.org/abs/2401.04174"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2401.04174-b31b1b.svg"></a>
</p>

`den_pie` is a field-level cosmology toolkit that unifies three things in one package:

- a **differentiable JAX forward model** (`den_pie.forward`) mapping cosmological + Lagrangian-bias parameters to density fields and their summaries ($P_{0,2,4}(k)$ + $B_0$), built on DiscoDJ + BFast + Falcon;
- a **Fisher-forecast library** (`den_pie.fisher`) that differentiates the forward model for Gaussian parameter constraints; and
- **neural simulation-based inference** (`den_pie.density`, `den_pie.spectra`): 3D field encoders (CNN / Swin / ResNet) and $P$+$B$ summary embeddings paired with normalizing flows (MAF / NSF / FrEIA) via the `sbi` library — with a built-in SBI-vs-Fisher corner overlay.

📖 **Documentation & the full pipeline diagram: https://poehlers.github.io/DenPIE/**

This repository merges the SBI pipeline (`den_pie`) with the forward-model + Fisher code from `joint_fli_sbi`; see [`NOTICE`](NOTICE) for attribution.

## Installation

den_pie spans two GPU stacks (PyTorch for SBI, JAX for the forward model + Fisher) that coexist in one environment:

```sh
git clone https://github.com/poehlers/DenPIE
cd DenPIE
# unified env — build on a GPU node with FFTW + a compiler (see docs/installation)
bash scripts/make_unified_venv.sh /path/to/denpievenv
source /path/to/denpievenv/bin/activate
```

For just the SBI half: `pip install -e .` · add the forward/Fisher half: `pip install -e ".[forward]"`.

## Usage

The full reference (CLI, YAML schema, methodology) is on the [documentation site](https://poehlers.github.io/DenPIE/); [`STRUCTURE.md`](STRUCTURE.md) covers the density Optuna workflow in depth. Quick tour:

```sh
# forward model — generate SBI training data (Falcon)
python -m den_pie forward-sample --config-name config_files/config_base.yml --run-dir RUN_DIR
# Fisher forecast
python -m den_pie fisher-forecast --config config_files/fisher/fisher_5param_P024_B0_emp.yml
# neural SBI (P+B summaries, or the raw 3D field)
python -m den_pie spectra-train  params/spectra_train_fli_bias_SN_PCA.yaml
python -m den_pie density-train   params/optuna_best/optuna_best_cnn3d_maf_stage3_finetune.yaml
# overlay the SBI posterior on the Fisher forecast
python -m den_pie fisher-compare  <fisher_run> <sbi_run> --out sbi_vs_fisher.png
```

Smoke-test the install with `bash scripts/smoke_test.sh`. SLURM submission scripts live in `scripts/`.

## Acknowledgements

The SBI pipeline builds on the original 21cmPIE-INN work; the forward model and
Fisher library were ported from [`joint_fli_sbi`](https://github.com/oleg-savchenko/joint_fli_sbi)
(O. Savchenko *et al.*) — see [`NOTICE`](NOTICE). If you use this package, please cite:

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

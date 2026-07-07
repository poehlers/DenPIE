# Forward model

The forward model ({mod}`den_pie.forward`) is a differentiable, JAX-based map
from cosmological (and Lagrangian-bias) parameters to density fields and their
summary statistics. It is expressed as a set of [Falcon](https://github.com/)
nodes so the same code both (a) generates large suites of simulations for SBI
training and (b) is differentiated by the {doc}`Fisher pipeline <fisher>`.

```{image} _static/forward_model.png
:alt: forward-model architecture
:width: 620px
:align: center
```

## Stages

1. **Linear power spectrum** — `PowerSpectrum` turns a cosmology
   $\boldsymbol{\theta} = (\Omega_m, \Omega_b, h, n_s, \sigma_8)$ into $P_L(k)$
   via DiscoDJ.
2. **Initial conditions** — `InitialConditions` colours white noise by
   $\sqrt{P_L(k)}$ to make $\delta_{\rm ic}$. Seeds are derived from the number
   of existing samples on disk, so re-running appends reproducible,
   non-colliding chunks.
3. **Gravity** — `ForwardModel` evolves $\delta_{\rm ic}$ with a 2LPT + PM
   solver (optionally applying redshift-space distortions along `los_axis`),
   producing the evolved dark-matter field $\delta_{\rm dm}$.
4. **Bias** — `ForwardModelBiasPk` / `bias.lagrangian_bias_weights` apply a
   Lagrangian bias expansion
   $w(\boldsymbol{q}) = 1 + b_1\delta + b_2(\delta^2-\langle\delta^2\rangle)
   + b_{s^2}(s^2-\langle s^2\rangle) + b_{\nabla^2}\nabla^2\delta$
   and scatter the reweighted particles to the biased tracer $\delta_{\rm g}$.
5. **Summaries** — `PkMultipoles` and `BkMonopole` (BFast) compute
   $P_{0,2,4}(k)$ and $B_0(k_1,k_2,k_3)$ for the `ic` and `fin` fields.

Global constants (`BOX_PARAMS`, `SIM_PARAMS`, `_FIDUCIAL_COSMO`) live in
`den_pie/forward/model.py`.

## Summary-statistic contract

Every per-simulation `.npz` produced by the forward model carries:

- `pk_l_<ic|fin>` — shape `(4, n_k)`: row 0 = $k$-bin centres, rows 1/2/3 =
  $P_0 / P_2 / P_4$;
- `bk_0_<ic|fin>` — shape `(n_tri, 4)`: columns 0–2 = $(k_1,k_2,k_3)$, column 3
  = $B_0$;
- `cosmo_params`, `bias_params` — the truth parameters.

This is exactly the format the SBI loaders ({mod}`den_pie.spectra`) consume, so
the producer and consumer halves interoperate without conversion.

## Generating training data

Data generation is driven by the Falcon CLI through a thin den_pie wrapper:

```bash
python -m den_pie forward-sample \
    --config-name config_files/config_base.yml --run-dir RUN_DIR
```

Falcon dataset graphs live under `config_files/` (e.g. `config_base.yml`,
`config_5param.yml`, `config_bias_*.yml`, and `config_fiducial_*` companions).
Each node's `_target_` points at `den_pie.forward.*`. Samples accumulate under
`RUN_DIR/samples_dir/prior/`; re-running appends more.

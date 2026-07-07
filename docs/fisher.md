# Fisher forecasting

{mod}`den_pie.fisher` differentiates the {doc}`forward model <forward_model>`
with respect to the parameters to build the Fisher information matrix and produce
Gaussian parameter forecasts. It reuses the *same* differentiable pipeline that
generates the SBI training data, so the forecast and the neural posterior are
grounded in identical physics.

## Data-vector Jacobian

For a fixed set of white-noise seeds, the pipeline
$\boldsymbol{\theta}\mapsto \boldsymbol{d}(\boldsymbol{\theta})$ (a data vector
of $P_\ell(k)$ and/or $B_0$) is differentiated three ways:

- **`autodiff`** (default) — reverse-mode `jax.jacrev`. DiscoDJ's N-body has a
  custom VJP but no JVP, and the cosmology pytree path yields NaN cotangents, so
  the cosmology fed to the N-body is wrapped in `stop_gradient`: the cosmology
  gradient flows only through $\delta_{\rm ic}$ (the growth-factor channel is
  omitted). See {doc}`methodology/fisher_jacrev_fix`.
- **`jacfwd`** — forward-mode AD with `jax_enable_x64=True`; *growth-complete*
  (the correct method), enabled automatically for the 5-parameter config.
- **`fd`** — central finite differences with per-parameter steps; also
  growth-complete, useful as a cross-check.

`compute_jacobian` averages over `n_seeds` white-noise realisations. Seed
averaging biases the Fisher slightly *upward* ($\sim 1/n_{\rm seeds}$), so the
default is `n_seeds=20` and values below 10 warn.

## Covariance

- `diag` — analytic block-diagonal Gaussian: $2P^2/N_{\rm modes}$ (or the
  Grieb et al. multipole covariance), Scoccimarro/Sefusatti for $B_0$, zero
  $P$–$B$ cross.
- `euclid_cov` — empirical sample covariance over `n_cov_seeds` fiducial
  realisations, Hartlap-corrected; captures $P$–$B$, $k$–$k'$, $\ell$–$\ell'$ and
  non-Gaussian terms.
- `block_with_pb_cross_empirical` — analytic $P$–$P$/$B$–$B$ blocks with an
  empirical $P$–$B$ cross block; cheaper than full `euclid_cov`.

## Assembly

The total Fisher matrix is $F = J^{\mathsf T} C^{-1} J + F_{\rm prior}$, with the
prior Fisher from `prior_fisher_matrix` (uniform → $12/(hi-lo)^2$, normal →
$1/\sigma^2$). The parameter covariance is $F^{-1}$; $1\sigma$ constraints are
$\sqrt{\operatorname{diag}(F^{-1})}$.

## Running a forecast

```bash
# Generic driver (all flags): den_pie fisher-forecast -h
python -m den_pie fisher-forecast --config config_files/fisher/fisher_5param_P024_B0_emp.yml

# Or ad hoc, e.g. a quick 2-parameter run:
python -m den_pie fisher-forecast --n-seeds 20 --n-bins 30 --output-dir outputs/fisher_fin
```

Fisher run configs live under `config_files/fisher/`. Each run writes a
timestamped directory containing `fisher_results.npz` (`F_data`, `F_prior`,
`F_total`, `fisher_cov` = $F^{-1}$, `jacobian`, per-seed Jacobians, `Pk_fid`,
`k_bins`, `sigma_sq`, `N_modes`, the fiducial values, and `param_names`), the
input config copied as `source_config.yml`, and a Fisher corner plot.

```{image} _static/fisher_corner_example.png
:alt: example Fisher corner (tiny res=16 smoke run)
:width: 460px
:align: center
```

The standalone drivers `den_pie/fisher/fisher_2param.py` and `fisher_5param.py`
target the pure-matter datasets, and `den_pie/fisher/verification/` holds
cross-checks (jacrev vs finite difference, Grieb covariance, `n_seeds` sweeps).

See {doc}`methodology/fisher_methodology` for the full derivation.

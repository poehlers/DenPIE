# Seeds in the Fisher pipeline

Quick reference for the different random-seed streams used by
`core/fisher.py` and `core/run_fisher.py`.

## TL;DR

| Stream | CLI flag | Count | What it controls | Used by |
|--------|----------|-------|------------------|---------|
| Jacobian seeds | `--n-seeds` | typically 10–30 | Per-seed white-noise IC field; Jacobian is averaged over seeds for variance reduction | All runners |
| Covariance seeds | `--n-cov-seeds` | 100–1000 | Independent fiducial-cosmology samples used to build an empirical data-vector covariance | `--cov-type {euclid_cov, block_with_pb_cross_empirical}` only |
| Shot-noise seeds | (none — derived) | 1 per Jacobian seed (and 1 per cov seed) | Additive Gaussian shot noise `ε ~ N(0, 1/(n̄ V_cell))` injected per voxel after the bias scatter | `--n-g-density` active |

All three streams are produced from `jax.random.PRNGKey(seed_int)`; they
never collide because:

- Jacobian seeds use `PRNGKey(seed)` for `seed in 0..n_seeds-1`.
- Covariance seeds use `PRNGKey(seed + 10_000)` for `seed in 0..n_cov_seeds-1`
  (`seed_offset=10_000` in `compute_empirical_covariance`).
- Shot-noise sub-streams are produced via `jax.random.split(key, 2|3)` of
  the parent key — so they share a parent with the IC noise but are
  cryptographically independent of it.

## Jacobian seeds — `--n-seeds`

Each Jacobian seed = one Gaussian white-noise realisation `noise` of shape
`(res,)*dim` that drives the IC field
`δ_ic = irfftn(rfftn(noise) · √(P_lin · norm))`.

At every seed we compute the data-vector Jacobian (autodiff or FD) and the
fiducial data vector, then average across seeds to suppress per-realisation
noise. The per-seed Jacobians are also saved (`jacobians_per_seed` in the
output `.npz`) so you can inspect convergence.

**Why averaging matters.** Seed-averaged Jacobian noise biases the Fisher
*upward* roughly as `1/n_seeds`, so `n_seeds < 10` produces artificially
tight constraints. Default 20 is a reasonable balance; thesis runs should
use ≥ 20.

**Where in the code:** the `for seed in range(n_seeds)` loops in
`compute_jacobian` (`fisher.py:~1514`) and `run_fisher_forecast_bias_PB`
(`fisher.py:~2973`).

## Covariance seeds — `--n-cov-seeds`

Only used when `--cov-type` is `euclid_cov` or
`block_with_pb_cross_empirical`.

Each cov-seed is a *fresh* independent white-noise realisation at the
*fiducial* parameters. The pipeline runs forward, the data vector is
collected, and after all `n_cov_seeds` samples are in we compute the sample
covariance

```
C_emp = (1/(N-1)) Σ_s (d_s - d̄)(d_s - d̄)^T
```

and apply the Hartlap factor `(N - n_data - 2)/(N - 1)` to `inv(C_emp)` so
the inverse covariance is unbiased.

**Constraint:** `n_cov_seeds > n_data + 2` (the runner raises a `ValueError`
otherwise). For comfort aim for `n_cov_seeds ≥ n_data + 50`.

**Why a separate stream from `--n-seeds`?** Jacobian seeds and cov seeds
serve different statistical purposes:
- Jacobian seeds reduce stochastic noise on `∂d/∂θ` (gradient estimator).
- Cov seeds estimate `<(d − <d>)(d − <d>)^T>` (covariance estimator).
You need many more samples for a stable matrix inverse than for a vector
mean, hence the much higher default (`200` vs `20`).

**Where in the code:** `compute_empirical_covariance` and
`empirical_pb_cross_block` in `fisher.py`.

## Shot-noise seeds — derived from `jax.random.split`

When `--n-g-density n_g` is set, every white-noise field is paired with one
or two extra Gaussian fields that get added to the post-bias-scatter
density contrast:

```
δ_galaxy = δ_biased + (1 / √(n̄ · V_cell)) · sn_noise
```

Key splits (see `run_fisher_forecast_bias_PB` and
`compute_empirical_covariance`):

- `--field fin|ic` (single field, SN on): `k_ic, k_sn = split(PRNGKey(seed), 2)`
- `--field both` (two fields, SN on both): `k_ic, k_sn_ic, k_sn_fin = split(PRNGKey(seed), 3)`

The shot-noise fields are held fixed across the ±ε FD evaluations within
one seed (consistent finite differences), and are regenerated per seed for
ensemble averaging.

**Why this matters for Fisher.** `sn_noise` is θ-independent, so it does
not contribute to `∂d/∂θ`. It *does* inflate the data-vector variance,
which the empirical covariance picks up automatically; the analytic `diag`
covariance picks it up implicitly because P_fid (measured) already includes
the shot-noise plateau.

## Reproducibility

The Fisher run is fully reproducible from `(n_seeds, n_cov_seeds, --res,
--boxsize, --pb-step, --n-g-density)` because every PRNG key is derived
deterministically from a Python integer. No system-time entropy. Two runs
with the same flags produce identical Fisher matrices (modulo JAX device
non-determinism at very low precision).

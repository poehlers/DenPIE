# Verification scripts

Each script corresponds to a numbered step in
`docs/fisher_too_tight_fixes.md` § "Suggested verification protocol".

| # | Script | Needs | Runtime |
| --- | --- | --- | --- |
| 1 | `plot_fisher_sigma_compare.py` | two or more `fisher_results.npz` | seconds, CPU |
| 2 | `plot_nseeds_sweep.py` | one `fisher_results.npz` with `jacobians_per_seed` | seconds, CPU |
| 3 | `validate_grieb_cov.py` | GPU; 100 fid-cosmo forward passes | ~1h on A100 |
| 4 | `compare_jacrev_vs_fd.py` | GPU; 1 jacrev + ~12 forward passes | ~30 min on A100 |

GPU jobs have SLURM wrappers at the repo root:
- `my_py_gpu_job_verify_grieb_cov.sh`
- `my_py_gpu_job_verify_jacrev_vs_fd.sh`

## 1) Marginal-sigma comparison

```
python core/verification/plot_fisher_sigma_compare.py \
    "n_seeds=3 (old):outputs/fisher_bias_P_B_rsd_SN_legacy/fisher_results.npz" \
    "n_seeds=20 (new):outputs/fisher_bias_P_B_rsd_SN/fisher_results.npz" \
    --out outputs/verification/sigma_compare_old_vs_new.png
```

Reads `fisher_cov` (data+prior) from each `.npz`, plots a grouped bar chart
with ratio annotations. Use `--data-only` to plot the data-only Fisher.

The "new" run loosens σ on every parameter — the post-prior σ(Ω_m) should
move from the old `~0.004` (artificially tight) to a larger value.

## 2) n_seeds plateau

```
python core/verification/plot_nseeds_sweep.py \
    outputs/fisher_bias_P_B_rsd_SN/fisher_results.npz \
    --seeds 3 5 10 20 40 \
    --out outputs/verification/nseeds_sweep.png
```

Uses `jacobians_per_seed` saved in the `.npz` to *recompute* the Fisher
matrix at each requested subset size (covariance held fixed at the
all-seeds value). Two subplots: σ/σ(largest) for data-only and
data+prior cases. Plateau within ~5 % from `n_seeds=20 → 40` is the
convergence criterion.

Recomputation uses the same code paths as `fisher_bias_PB`
(`multipole_covariance_grieb`, `bispectrum_covariance`).

## 3) Empirical Cov(P_l, P_l') vs Grieb+2016

```
sbatch my_py_gpu_job_verify_grieb_cov.sh
```

Runs 100 independent noise seeds at fid cosmology through the RSD
multipole pipeline (`_differentiable_pipeline_fin_bias_PB_rsd_multi`),
collects `P_l(k)` per seed, computes:

- empirical `Cov(P_l, P_l')(k)` over the seed ensemble
- analytic Grieb cov from the seed-averaged P_l

Plots a `(n_l × n_l)` grid: diagonals show empirical vs Grieb variance,
off-diagonals show the Pearson correlation coefficient. Saves raw arrays
to a parallel `.npz` for further inspection.

If the empirical Var(P_l) and ρ(P_l, P_l') match Grieb's prediction
across k (the off-diagonal correlations should be ~0.2–0.4 for
Kaiser-strength RSD), the Grieb covariance is validated.

## 4) jacrev vs central FD derivative

```
sbatch my_py_gpu_job_verify_jacrev_vs_fd.sh
```

For a single fid noise seed, computes `∂P_l(k)/∂θ` two ways:

- jacrev — the pipeline as wired into Fisher (uses
  `stop_gradient(cosmo)` through the N-body)
- central finite differences — full chain rule including the cosmology
  dependence of the N-body integrator

Plots derivatives on the same axes plus a separate ratio plot
(`jacrev / FD`). Expected:

- σ_8: ratio ≈ 1 at all k (σ_8 doesn't enter D(a) ratios)
- Ω_m, h: ratio deviates from 1, increasingly so at higher k (the
  stop_gradient bias). The deviation pattern tells you how much the
  Fisher derivative is systematically wrong.

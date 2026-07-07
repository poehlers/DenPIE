# Fisher forecasts for the thesis

This document records every Fisher forecast produced for the thesis: what each run
computes, which SBI result it complements, and the resulting parameter constraints.
All runs are **on the final (evolved) field only** and use **forward-mode autodiff
(`jacfwd`)** for the parameter Jacobian.

---

## Methodology

### Final field only
Every Fisher run summarises the **final, redshift-space field** `delta_fin` (matter) or
`delta_biased` (biased tracer), never the initial conditions. Rationale: the only thing we
actually *observe* is the evolved field at z=0. A Fisher forecast bounds the information
content available to an idealised analysis of that observable, so restricting to the final
field makes the Fisher and the (final-field) SBI directly comparable: both see the same
information. The SBI networks may additionally ingest the IC field for diagnostics, but the
fair Fisher↔SBI comparison is on the final field.

### Data vector and empirical covariance
The data vector is the joint summary of the final field:

    d(theta) = [P_0(k), P_2(k), P_4(k), B_0(k1,k2,k3)]      (RSD multipoles + bispectrum)

(or `[P_0]` only for the base P0 test). Shot noise (DESI-like, n_g=1e-3 (h/Mpc)^3) is
injected per voxel where the matching dataset has it.

The **full empirical covariance** is the sample covariance over `n_cov_seeds` independent
fiducial realisations,

    C = (1/(N-1)) * sum_i (d_i - d_bar)(d_i - d_bar)^T ,

inverted with the Hartlap correction `(N - n_data - 2)/(N - 1)` applied to `C^{-1}`. This is
the **same estimator as `docs/Analyse.ipynb`** (`np.cov` + Hartlap on the inverse); the only
difference is that shot noise is forward-modelled into the ensemble rather than analytically
subtracted. Analytical ("Gaussian") covariance runs use the Grieb+2016 multipole P-block +
Scoccimarro bispectrum diagonal instead, for the empirical-vs-analytical comparison plots.

### The derivative method: why `jacfwd` (forward-mode AD + float64)

The Fisher matrix is `F = J^T C^{-1} J` with `J = d d/d theta`. How `J` is computed matters
for the cosmological parameters, because of how DiscoDJ's N-body is differentiated:

- **Reverse-mode AD (`jacrev`, the old default)** routes the N-body through DiscoDJ's
  hand-written `jax.custom_vjp` adjoint (`discodj/nbody/nbody_scan_functions.py`). That
  backward rule declares the ODE `terms` (the growth factor D(a) and time-stepping — i.e. the
  cosmology) as `nondiff_argnums` and **discards their cotangent**. So the gradient of the
  final field w.r.t. cosmology *through gravitational growth* is dropped; only the
  `cosmo -> P_lin -> delta_ic` path survives. The pipeline makes this explicit with
  `stop_gradient(cosmo)` to avoid relying on a NaN cotangent.

- **Forward-mode AD (`jacfwd`)** with `requires_jacfwd=True` flips the N-body to a plain
  `jax.lax.scan` (no custom_vjp override) and lets cosmology flow through, so it captures the
  **full** cosmo dependence including growth. Forward mode is also cheap here (cost scales with
  the number of parameters, 2–6, not the data-vector length).

**The catch and the fix.** Naive `jacfwd` returned `NaN` for the Omega_m derivative. Bisection
localised it to DiscoDJ's growth-factor ODE (`cosmology.py: compute_unnormed_growth`), which
overflows in float32 in the *tangent* (the ODE has `a**-3` and `exp(y)` terms). DiscoDJ has an
explicit float64 branch for exactly this ODE. **Enabling `jax_enable_x64=True` makes the
tangent finite.** This is cheap: DiscoDJ keeps the heavy N-body/FFT arrays in float32 via
`dtype_num=32`, so only the tiny 1-D growth ODE upgrades to float64; the covariance forward
passes stay float32. (A second bug surfaced for the bias pipeline — the float64 bias weights
hit the float32 particle-mesh scatter — fixed by casting the weights to the position dtype, a
differentiable no-op that preserves the tangent.)

**Validation (the gate).** At res=32, 1 seed, the 5-param Jacobian was computed three ways and
compared column-by-column (relative L2):

| param   | rel(jacfwd, FD) | rel(jacrev, FD) |
|---------|-----------------|-----------------|
| Omega_m | **0.0038**      | 0.1097          |
| Omega_b | 0.0008          | 0.0008          |
| h       | 0.0027          | 0.0027          |
| n_s     | 0.0013          | 0.0013          |
| sigma8  | 0.0008          | 0.0008          |

Finite differences (FD) re-runs the full pipeline at theta±eps and is therefore the
growth-complete ground truth. `jacfwd` matches FD to **0.4% on Omega_m** while keeping autodiff
precision on every other parameter; reverse-mode is **11% low on Omega_m** — precisely the
dropped growth term. Conclusion: **`jacfwd`+x64 is the correct method** and is used for all runs
below. (FD is an equally-correct but slower fallback; reverse-mode `jacrev` is biased for
Omega_m and is retained only for backward-compatibility / cross-checks.)

Implementation: `compute_jacobian_forward` in `core/fisher.py`; `forward_mode=True` flag on
each pipeline; `--derivative-method jacfwd` (auto-enables x64) in `core/run_fisher.py`,
`core/fisher_2param.py`, `core/fisher_5param.py`.

---

## The runs

Fiducial cosmology (Planck-2018 / Quijote): Omega_m=0.3175, Omega_b=0.0490, h=0.6711,
n_s=0.9624, sigma8=0.8340. All res=128, boxsize=1000 Mpc/h, final field, los_axis=2.

### Base tests (2-param: Omega_m, sigma_8) — empirical vs analytical + SBI

| ID | Summary | Covariance | Config | Thesis plot |
|----|---------|-----------|--------|-------------|
| B1 | P0      | empirical (euclid_cov) | `fisher_2param_P0_emp.yml` | `fisher_cov_comp_2param_w_sbi.png` |
| B2 | P0      | analytical (diag)      | `fisher_2param_P0_gauss.yml` | `fisher_cov_comp_2param_w_sbi.png` |
| B3 | P024+B0 | empirical, **P-B cross != 0** | `fisher_2param_P024_B0_emp.yml` | `fisher_cov_comp_w_SBI_P024_B0_nonzeroPBcross.png` |
| B4 | P024+B0 | empirical, **P-B cross = 0** | `fisher_2param_P024_B0_emp_nocross.yml` | `fisher_cov_comp_w_SBI_P024_B0_zeroPBcross.png` |
| B5 | P024+B0 | analytical, **P-B cross = 0** (diag) | `fisher_2param_P024_B0_gauss.yml` | `..._zeroPBcross.png` |
| B6 | P024+B0 | analytical blocks + **P-B cross != 0** (empirical cross) | `fisher_2param_P024_B0_blockcross.yml` | `..._nonzeroPBcross.png` |

The four P024+B0 covariances form a 2x2 grid {empirical, analytical} x {cross=0, cross!=0}:

|            | cross = 0 | cross != 0 |
|------------|-----------|------------|
| empirical  | B4        | B3         |
| analytical | B5 (diag) | B6 (analytic P/B blocks + empirical cross) |

Note: there is **no analytic formula** for the P-B cross — it is exactly zero at Gaussian
order (a disconnected 5-point moment). So "analytical with a non-zero cross" (B6) necessarily
estimates *only that one cross block* empirically, keeping the analytic Grieb P-P and
Scoccimarro B-B blocks. The two comparison plots then pair like-with-like:
`zeroPBcross` = B4 vs B5; `nonzeroPBcross` = B3 vs B6. (B3-vs-B4 isolates the cross impact on
the empirical side; B5-vs-B6 the same on the analytical side.)

SBI counterparts: 2-param SBI on the final field with the matching summary
(`spectra_fli_2param_*`).

### Main results (best Fisher per case, complementing the density-field SBI)

| ID | Params | Dataset / fiducial | Config | Complements SBI |
|----|--------|--------------------|--------|-----------------|
| M1 | Omega_m, sigma_8, b1, b2, bs2, bn2 | `fli_bias_SN`, zero bias fiducial (b1=1, b2=bs2=bn2=0) | `fisher_bias_PB_rsd_SN_jacfwd.yml` | `sbi_vs_cnn_bias.png` (bias) |
| M2 | Omega_m, Omega_b, h, n_s, sigma_8 | `fli_5param` (+SN) | `fisher_5param_P024_B0_emp.yml` | 5-param SBI |
| M3 | Omega_m, sigma_8, b1, b2, bs2, bn2 | `fli_bias_nonzero`, **nonzero** fiducial (b1=b2=bs2=bn2=1) | `fisher_bias_nonzero_jacfwd.yml` | `density_bias_..._both_nonzero_20260618_113901` |

All M-runs: P024+B0, full empirical covariance, n_seeds=20, n_cov_seeds=2400, shot noise
n_g=1e-3, jacfwd. M3 uses the `--fid-b1/b2/bs2/bn2` override to set the nonzero bias fiducial
matching `config_fiducial_bias_nonzero.yml`.

---

## Results

> Marginal 1-sigma constraints (DATA+PRIOR) and figure of merit. Filled in as jobs complete.
> Output dirs under `outputs/`. Smoke tests (res=32) confirmed each driver end-to-end before
> the res=128 production runs.

### Reference (pre-jacfwd, for comparison)
- **M2 5-param, FD** (`fisher_5param_P024_B0_emp_20260619_162950`, growth-complete FD ground
  truth): Omega_m 3.83%, Omega_b 14.20%, h 9.82%, n_s 5.68%, sigma8 0.45%.
- **M1 bias, old reverse-mode autodiff** (`..._20260529_140056`, Omega_m growth omitted):
  Omega_m ±0.0070 (2.19%), sigma8 ±0.0119 (1.43%), b1 ±0.0288, b2 ±0.0141, bs2 ±0.0509,
  bn2 ±0.3761. The jacfwd M1 below supersedes this (corrected Omega_m).

### jacfwd production runs (res=128, 2026-06-19)

**Base tests (sigma(Omega_m), sigma(sigma_8), FoM):**

| ID | Output dir | Hartlap | sigma(Omega_m) | sigma(sigma_8) | FoM |
|----|-----------|---------|----------------|----------------|-----|
| B1 P0 emp        | `fisher_2param_P0_emp_20260619_223556`        | 0.977 | 0.00666 | 0.00295 | 5.30e4 |
| B2 P0 gauss      | `fisher_2param_P0_gauss_20260619_223554`      | n/a   | 0.00720 | 0.00241 | 8.03e4 |
| B3 P024+B0 emp (cross kept) | `fisher_2param_P024_B0_emp_20260619_223556` | 0.846 | 0.00503 | 0.00241 | 8.27e4 |
| B4 P024+B0 emp (cross=0)    | _rerunning (job 24042242; bugfix)_ | | | | |
| B5 P024+B0 gauss | `fisher_2param_P024_B0_gauss_20260619_223555` | n/a   | 0.00151 | 0.00122 | 1.25e6 |

Note B5 (diag/analytic) is much tighter than B3 (empirical) — the analytic Gaussian covariance
underestimates the true (empirical) covariance for P024+B0; that gap is the point of the
empirical-vs-analytical comparison. B3-vs-B4 isolates the P-B cross-covariance contribution.

**Main results — 1-sigma marginals (DATA+PRIOR):**

M1 — bias, **zero** fiducial (`fisher_bias_PB_rsd_SN_jacfwd_20260619_223556`, Hartlap 0.552):
| Omega_m | sigma_8 | b1 | b2 | bs2 | bn2 |
|---------|---------|----|----|-----|-----|
| ±0.0070 (2.22%) | ±0.0119 (1.42%) | ±0.0299 | ±0.0147 | ±0.0520 | ±0.3698 |

M2 — 5-param (`fisher_5param_P024_B0_emp_20260619_223556`, Hartlap 0.846, FoM 8.14e9):
| Omega_m | Omega_b | h | n_s | sigma_8 |
|---------|---------|---|-----|---------|
| ±0.01217 (3.83%) | ±0.00696 (14.2%) | ±0.0659 (9.8%) | ±0.0546 (5.7%) | ±0.00375 (0.45%) |
→ matches the FD ground-truth reference to the printed precision (jacfwd validated at res=128).

M3 — bias, **nonzero** fiducial b=(1,1,1,1) (`fisher_bias_nonzero_jacfwd_20260619_223556`, Hartlap 0.552):
| Omega_m | sigma_8 | b1 | b2 | bs2 | bn2 |
|---------|---------|----|----|-----|-----|
| ±0.0079 (2.50%) | ±0.0099 (1.19%) | ±0.0565 (5.65%) | ±0.0638 (6.38%) | ±0.1686 (16.9%) | ±1.526 (152.6%) |

_SLURM job IDs: B1–M3 = 24032707–24032714; B4 rerun = 24042242 (submitted 2026-06-19)._

### Known fix applied mid-run
`euclid_cov_no_pb_cross` (B4) initially failed at the cross-zeroing step: `np.asarray()` of a
JAX array returns a read-only view, so the in-place `cov_emp[...] = 0` raised. Fixed by using
`np.array()` (writable copy) in `fisher_2param.py` and `fisher_5param.py`; B4 rerun.

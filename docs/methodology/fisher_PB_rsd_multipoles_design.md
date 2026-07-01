# Fisher P+B with redshift-space P-multipoles

## What this adds

A new mode of `run_fisher_forecast_bias_PB`, selected by the CLI flag
`--pk-multipoles`, that swaps the real-space monopole P(k) summary on
the P side for the **redshift-space P_0, P_2, P_4 multipoles** while
keeping the bispectrum side as a monopole. Two BFast calls per
field-block (`powerspectrum(..., multipole_axis=2)` for the multipoles,
`bispectrum(..., only_B=True)` for the monopole bispectrum); both are
pure-JAX and AD-traceable, so `jax.jacrev` propagates cotangents
end-to-end through the same pipeline used by the existing monopole
forecast.

Reference for the multipole estimator: `BFast.core.powerspectrum.powerspectrum`
in `BFast/core/powerspectrum.py:140-188`. Reference for the SBI side:
`/home/poehlers/cosmo_thesis/SBI_Pk/src/main_sbi.py:660-700`.

Field modes supported with `--pk-multipoles`:
- `--field fin`: multipoles on the post-N-body redshift-space biased
  field. Output `outputs/fisher_bias_P_B_rsd/`.
- `--field both`: multipoles on **both** the Lagrangian field
  (P_2/P_4 ≈ 0 by isotropy — no RSD on δ_L; kept for shape uniformity)
  **and** the post-N-body redshift-space field. Data-vector layout
  `[P0_ic, P2_ic, P4_ic, B_ic, P0_fin, P2_fin, P4_fin, B_fin]`. Output
  `outputs/fisher_bias_P_B_rsd_both/`.
- `--field ic` is rejected at the CLI: the Lagrangian δ_L is isotropic,
  so P_2 = P_4 ≡ 0 there and multipoles add no information.

LOS axis is hard-coded to `2` (z), matching `model_bias.py`. The CLI
exposes only the on/off flag; if a non-z LOS is needed later, lift
`radial_dim=2` to a parameter in the two new pipelines.

## Where the changes live

- `core/fisher.py`:
  - `_bfast_pk_multi_bk_safe` — JAX-AD-safe `(P_0, P_2, P_4, B)` from a
    real- or redshift-space density-contrast field. Two BFast calls.
  - `_differentiable_pipeline_fin_bias_PB_rsd_multi` — clones the
    existing FIN P+B pipeline, captures canonical momenta from
    `dj.run_nbody`, and scatters with `in_redshift_space=True, vel=P_flat,
    a=sp["a_end"], radial_dim=2, deconvolve=True`.
  - `_differentiable_pipeline_both_bias_PB_rsd_multi` — joint version,
    one shared N-body run; IC summary on δ_L (no RSD), FIN summary on
    δ_RSD.
  - `run_fisher_forecast_bias_PB` — new `use_multipoles` parameter,
    extended pipeline + output-dir dispatch, generalised
    slicing/covariance loop over multipole channels, extended `.npz`
    save.
- `core/run_fisher.py` — new `--pk-multipoles` flag, validation, threads
  `use_multipoles=args.pk_multipoles` into the driver.
- `my_py_gpu_job_fisher_bias_PB_rsd.sh`,
  `my_py_gpu_job_fisher_bias_PB_rsd_both.sh` — SLURM jobs for the two
  new modes.

## Covariance: what we assume

Diagonal Gaussian, separately per multipole and per IC/FIN block:
```
sigma^2(P_l, k) = (2l + 1) * 2 * P_0(k)^2 / N_modes(k),    l in {0, 2, 4}
sigma^2(B_T)   = analytic Scoccimarro-2000 / Sefusatti-2006   (uses P_0)
```
This is the leading-order Gaussian-field formula in the Grieb+2016
limit. It uses the **monopole** P_0 inside the multipole variance —
standard in the literature, and the same approximation class as the
existing P↔B and IC↔FIN block-diagonal assumptions in this codebase.

What it neglects, explicitly:
- **Off-diagonal multipole covariance** (P_0–P_2, P_0–P_4, P_2–P_4
  cross). Real RSD multipoles are correlated; treating them as
  independent will over-state the constraint power, especially on
  RSD-sensitive parameters.
- **Trispectrum / non-Gaussian** corrections to the multipole
  variance.
- **Window-function / AP-effect** distortions (we assume a periodic
  plane-parallel box, consistent with the rest of the pipeline).

The driver prints two warnings when both `--field both` and
`--pk-multipoles` are active: one for the IC↔FIN block-diagonal
covariance, one for the diagonal multipole covariance.

## Why the IC side is kept symmetric

For `--field both --pk-multipoles`, the IC block emits
`[P0_ic, P2_ic, P4_ic, B_ic]` even though the Lagrangian field has no
peculiar velocities and no preferred axis, so P_2 / P_4 are ≈ 0 (up
to discretisation noise) and contribute essentially zero Fisher
information. Keeping them in the data vector is purely for shape
uniformity — the slicing loop in the driver and the `.npz` save use
the same per-block layout in every mode. The corresponding rows of
the Jacobian are also ~0, so the Fisher matrix is unchanged whether
we drop them or not.

## Expected effect on marginals

Compared to `outputs/fisher_bias_P_B/` (real-space monopole) at the
same n_seeds and binning:
- `Omega_m`, `sigma8`: should tighten — RSD breaks the
  `(b1·sigma8, f·sigma8)` degeneracy via the P_2/P_0 ratio.
- `b1`: typically slightly tighter (RSD adds an independent constraint
  on `b1·f`).
- `b2, bs2, bn2`: roughly unchanged or marginally tighter — these are
  driven mostly by the bispectrum, which has the same triangle
  configurations in both modes.
- `_rsd_both` should tighten further than `_rsd` (more data; same
  block-diagonal optimism caveat).

If the multipole forecast does **not** tighten Omega_m / sigma8
relative to the monopole baseline, the most likely culprits are:
(a) momenta `P` not propagating through the scatter call (check
that the new pipelines capture `P_mom` from `run_nbody`);
(b) `radial_dim` mismatch (we use 2; DiscoDJ defaults agree);
(c) `a=sp["a_end"]` not matching the snapshot redshift used by the
N-body integrator.

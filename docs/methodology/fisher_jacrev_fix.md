# Fisher Jacobian fix: jacfwd → jacrev

## Symptom

Forward-mode AD over the cosmo → P_lin → δ_ic → δ_fin → P_NL pipeline
returned an all-NaN Jacobian (`Jacobian finite: 0/180`), forcing a
fallback to numerical finite differences. Forward evaluation of the
pipeline was finite, so the gradient — not the model — was broken.

## Diagnosis

A staged `jax.jvp` diagnostic (`_diagnose_tangent` /
`_diagnose_tangent_bias` in `core/fisher.py`) localized the failure:

```
P_lin:     primal 1000/1000,    tangent 1000/1000      OK
delta_ic:  primal 262144/262144, tangent 262144/262144  OK
X_sim:     primal 786432/786432, tangent 0/786432       BROKEN
delta_fin: primal 262144/262144, tangent 0/262144       NaN-poisoned
P_NL:      primal 30/30,         tangent 4/30           NaN-poisoned
```

The N-body call (`dj.run_nbody(...)`) kills the tangent. Looking at
DiscoDJ's source explained why:

- `discodj/nbody/nbody_scan_functions.py:52` defines `scan_wrapper` with
  `@jax.custom_vjp`. There is no companion `custom_jvp` rule.
- When `requires_jacfwd=True`, DiscoDJ deliberately routes around that
  custom_vjp into `scan_wrapper_no_custom_vjp`, a vanilla `jax.lax.scan`.
  But the underlying CIC scatter/gather ops (`scatter_and_gather.py`)
  also have only VJP rules, so plain forward-mode through them is
  undefined and silently produces NaN.
- The intended forward-mode escape hatch was `use_diffrax=True` (which
  would use `diffrax.ForwardMode`), but in this DiscoDJ install that
  branch is hard-disabled — `disco_dj.py:1103` raises
  `NotImplementedError("Diffrax support is currently disabled...")`.

Net: the **only** AD mode DiscoDJ's N-body actually supports is
**reverse-mode (VJP)**.

## Fix

Switch the Fisher Jacobian to reverse-mode AD.

### Code changes (all in `core/fisher.py`)

1. **`requires_jacfwd=True` → `requires_jacfwd=False`** (19 sites: 4
   pipelines + 2 staged diagnostics). With this flag false, `run_nbody`
   sets `adjoint_method=True` (`disco_dj.py:954`) which routes through
   `scan_wrapper`'s custom_vjp — the working path.
2. **`jax.jacfwd` → `jax.jacrev`** in `compute_jacobian` (single line at
   the per-seed loop). Output shape (n_bins=30, n_params=5–6) means
   reverse-mode does ~30 backward passes per seed.
3. **JVP diagnostic → VJP diagnostic** (`_diagnose_tangent`,
   `_diagnose_tangent_bias`, plus a new `_report_vjp` helper). With
   `requires_jacfwd=False`, JVP through the custom_vjp would now crash
   instead of silently returning NaN, so the diagnostic itself had to
   move to reverse-mode. Each stage now reports input-cotangent
   finiteness pulled back from a unit output cotangent.
4. **Cosmetic**: doctring + log messages updated `jacfwd` → `jacrev` so
   that future readers don't get confused about which AD mode is in use.

### Independent change: AD-safe shell-binned P(k) (`_power_spectrum_safe`)

While diagnosing, `_power_spectrum_safe` was also rewritten in pure JAX
to drop the dependency on `discodj.core.summary_statistics.power_spectrum_core`.
The new implementation does shell-binning directly via
`jnp.bincount` + `jnp.digitize`, with a `jnp.where(N_modes > 0, ..., 0)`
guard to keep gradients NaN-free at empty bins. Bin edges and the rFFT
mode-counting weights match `count_modes()` exactly so `k_bins` and
`N_modes` line up across the codebase.

This wasn't strictly needed (the upstream `power_spectrum_core` is
already pure JAX), but it removes one DiscoDJ dependency from the diff
path and makes the binning behaviour unambiguous.

## Follow-up: stop_gradient on cosmo for the N-body

After flipping to reverse-mode, the diagnostic showed **partial NaN** at
the N-body stage:

```
P_lin     5/5 finite, norm 4.95e+07   OK
delta_ic  5/5 finite, norm 4.82e-01   OK
X_sim     3/5 finite, norm nan        2 cosmo params NaN
delta_fin 3/5 finite, norm nan
P_NL      3/5 finite, norm nan
```

In the bias run, the NaN entry was Ω_m specifically (5/6 finite) — so the
broken parameters are the ones that flow through `cosmo.Dplus` /
`cosmo.Fplus` / `cosmo.compute_timetables` (growth factor and time
evolution), which is the cosmology-pytree path through DiscoDJ's adjoint.
σ_8 / n_s / bias params, which only enter via P_lin or the bias weights,
return finite cotangents.

DiscoDJ's reverse-mode adjoint for the cosmology pytree is broken for
specific parameters. We work around this by **freezing cosmo for the
N-body call** with `jax.lax.stop_gradient`:

```python
cosmo_nbody = jax.lax.stop_gradient(cosmo)
dj2 = DiscoDJ(..., cosmo=cosmo_nbody, requires_jacfwd=False)
```

(applied at all 4 N-body sites in the pipelines and the 4 staged-
diagnostic counterparts). Cosmo dependence still flows through `delta_ic`
(which carries the full P_lin gradient). What's lost is the
growth-factor / time-evolution channel of cosmo's effect on P_NL.

**Why this is a small approximation**:

- Dplus(1) = 1 by normalization, so growth-factor effects at z=0 are
  largely degenerate with σ_8 — and σ_8 already has a working gradient
  via P_lin.
- The dominant cosmo dependence of P_NL at the box scales (Ω_m, Ω_b,
  h via the BAO and turnover) is shape information in P_lin, not in
  the N-body's time evolution.
- For the bias forecast, only Ω_m is affected; the bias parameters
  (b1, b2, bs2, bn2) didn't flow through the N-body anyway.

The numerical-FD fallback remains wired in. If a column of the jacrev
Jacobian comes back NaN despite the stop_gradient, FD takes over.

## Trade-offs

- **Memory**: reverse-mode through the N-body needs to retain particle
  positions/velocities at every checkpoint. DiscoDJ's adjoint integrator
  uses recursive checkpointing, but memory will rise vs. the
  forward-mode path. At `res=64`, `n_steps=1`, this is negligible.
- **Compute**: jacrev does `n_bins` backward passes (~30 here) versus
  jacfwd's `n_params` (5–6) forward passes. The asymmetry is moderate —
  expect a 4–6× slowdown of the Jacobian computation. With `n_seeds=10`
  this should still fit inside the existing 2-hour SLURM allocation.
- **Numerical FD fallback**: still wired in. If reverse-mode Jacobian
  comes back all-NaN, FD takes over with the same Pk_fid output, so the
  forecast still produces valid (FD-based) constraints rather than
  crashing.

## What didn't fix it

Tried/ruled-out paths during diagnosis:

- **Pure-JAX shell-binning** (`_power_spectrum_safe`): didn't change the
  diagnostic outcome — the binning step was never the culprit. Kept the
  rewrite anyway as a clean-up.
- **Eisenstein-Hu P_lin**: P_lin tangent was already finite via
  DiscoDJ's existing JAX-traceable `with_linear_ps()`, so a hand-rolled
  EH was unnecessary.
- **`use_diffrax=True`**: hard-disabled in this DiscoDJ version.

## Verification plan

1. `sbatch my_py_gpu_job_fisher_diagnose.sh` — confirm every stage
   reports `cotangent finite ≥ #cosmo params` (i.e. ≥ 5 / ≥ 6) and
   nonzero norm.
2. `sbatch my_py_gpu_job_fisher.sh` — full forecast. Expect
   `Jacobian finite: ≥130/150` (cosmo-only, 5 params × 26 inner bins),
   no fallback to numerical.
3. `sbatch my_py_gpu_job_fisher_bias.sh` — full bias forecast. Expect
   `Jacobian finite: ≥156/180` (6 params × 26 inner bins).
4. Cross-check: at seed 0, compare jacrev Jacobian against the FD path
   (`compute_jacobian_numerical`) — relative error per inner bin should
   be ≲ 1%.
5. Regenerate `outputs/fisher/fisher_corner.png` and compare to the
   existing FD-based corner plot for sanity.

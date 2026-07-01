# Project Overview: Forward Model, Dataset Generation & Fisher Forecasting

> **Purpose of this document.** Give another reader (human or LLM) full
> comprehension of how this repository turns cosmological parameters into
> mock observations, how those observations are generated *en masse* as
> training datasets, and how the same physics pipeline is differentiated to
> produce Fisher forecasts. It is written to be self-contained: you should
> not need to read the source to understand the data flow, only to change it.

---

## 1. What this project is

The repo (`joint_fli_sbi`) is a thesis codebase for **joint Field-Level
Inference (FLI) and Simulation-Based Inference (SBI)** of cosmological
parameters from the large-scale structure of the Universe.

Two things are built on top of **one shared forward model**:

1. **Dataset generation** — run the forward model many times over parameters
   drawn from a prior, producing `(θ, observable)` pairs. These train an
   SBI posterior estimator (a normalizing flow, via the **Falcon** framework).
2. **Fisher forecasting** — differentiate the *same* forward model w.r.t. the
   parameters with JAX autodiff, build the Fisher information matrix, and
   produce forecasted parameter constraints (error ellipses). This is the
   analytic/Gaussian baseline against which the SBI posteriors are compared.

The N-body engine underneath everything is **DiscoDJ** (a differentiable JAX
particle-mesh code), installed at
`/gpfs/home6/poehlers/cosmo_env/lib/python3.11/site-packages/discodj/`.

### Parameters

Two parameter "modes" appear throughout:

- **5-param cosmology** (full): `[Ω_m, Ω_b, h, n_s, σ_8]` — canonical order
  defined by `_COSMO_PARAM_ORDER` in `core/model/model.py`.
- **2-param cosmology + bias** (the "bias" track): cosmology reduced to
  `[Ω_m, σ_8]` (with `Ω_b, h, n_s` fixed to fiducial), plus a **Lagrangian
  bias** vector `[b1, b2, bs2, bn2]`. Combined parameter vector is the 6-tuple
  `[Ω_m, σ_8, b1, b2, bs2, bn2]`.

**Fiducial cosmology** (`_FIDUCIAL_COSMO`, Planck 2018 = Quijote):
`Ω_m=0.3175, Ω_b=0.0490, h=0.6711, n_s=0.9624, σ_8=0.8340`.
**Fiducial bias** (`_FIDUCIAL_BIAS`): `b1=1, b2=0, bs2=0, bn2=0` (unbiased
matter).

> **DiscoDJ quirk:** its `cosmo` dict uses `Omega_c` (CDM only), not `Omega_m`.
> Conversion happens in `_cosmo_dict()`: `Omega_c = Omega_m − Omega_b`. That
> helper also handles parameter ordering and re-insertion of fixed params.

---

## 2. The shared forward model (the physics pipeline)

The forward map is a 4-stage chain from parameters to a density field, then to
a summary statistic:

```
cosmo_params ──► P_lin(k) ──► delta_ic ──► delta_fin ──► summary (P(k), B, ...)
                (linear PS)   (Gaussian   (non-linear    (power spectrum /
                              random IC)   N-body field)   bispectrum)
```

### Stage 1 — Linear power spectrum `P_lin(k)`
DiscoDJ builds the linear matter power spectrum at z=0 from the cosmology:
```python
dj = DiscoDJ(dim, res, boxsize, cosmo=_cosmo_dict(params)).with_timetables().with_linear_ps()
pk_lin = dj._pk_table["Pk"];  k_pk = dj._pk_table["k"]
```
The `k` grid depends only on box geometry, not cosmology.

### Stage 2 — Initial conditions `delta_ic` (Gaussian random field)
A white-noise field is colored by `√P_lin(k)`:
```python
k_grid   = get_fourier_grid([res]*dim, boxsize)["|k|"]
Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
norm_fac  = (res / boxsize) ** dim
delta_ic  = irfftn( rfftn(white_noise) * sqrt(Pk_interp * norm_fac) )
```
This reproduces DiscoDJ's `grf.py` real-space IC formula: the `inv_laplace`
kernel (`1/-k²`) inside `delta_ini` cancels the `-k²` from the potential, so
`delta_ic = irfftn(rfftn(noise) · √(P·norm))`. At a=1, `D_plus(1)=1` (DiscoDJ
normalizes P at z=0), so `delta_ic` is already the z=0 linear field.

### Stage 3 — N-body evolution `delta_fin`
A particle-mesh N-body run advects particles from z=3 to z=0:
```python
dj2 = DiscoDJ(..., cosmo=cosmo).with_timetables()
dj2 = dj2.with_external_ics(delta=delta_ic).with_lpt(n_order=2, grad_kernel_order=0)
X_sim, P_mom, _ = dj2.run_nbody(stepper="bullfrog", method="pm", ...)
delta_fin = dj2.get_delta_from_pos(X_sim, worder=2, antialias=1, deconvolve=True)
```
Key N-body settings (`SIM_PARAMS` in `model.py`): `a_ini=0.25 (z=3)`,
`a_end=1.0 (z=0)`, `stepper="bullfrog"`, `method="pm"`, `res_pm = 2×res`
(up-sampled mesh), `n_steps=1`, second-order LPT ICs (`nlpt_order_ics=2`),
CIC mass assignment (`worder=2`).

### Stage 4 — Summary statistic
The density field is reduced to one or more summaries:
- **P(k)** — binned power spectrum (the default observable).
- **P_0, P_2, P_4** — redshift-space multipoles (monopole/quadrupole/hexadecapole).
- **B(k1,k2,k3)** — bispectrum (via the **BFast** library).

### Optional physics layers (apply at Stage 3/4)

**Redshift-space distortions (RSD).** Instead of `get_delta_from_pos`, particles
are scattered with `in_redshift_space=True`, displaced along the line of sight
(`los_axis=2`, the z-axis) by their peculiar velocity `v_los/(aH)` derived from
the canonical momenta `P_mom` returned by `run_nbody`.

**Lagrangian bias (galaxy/tracer field).** Defined in `core/model/bias.py`
(`lagrangian_bias_weights`). Each particle gets a weight on the Lagrangian grid:
```
w(q) = 1 + b1·δ(q) + b2·[δ²(q)−⟨δ²⟩] + bs2·[s²(q)−⟨s²⟩] + bn2·∇²δ(q)
```
where `s²` is the tidal-shear-squared (built from the tidal tensor
`T_ij = (k_i k_j/k²)·δ̂`) and `∇²δ` is the Laplacian (`-k²·δ̂`). The weights
are computed from `delta_ic` (Lagrangian grid coincides with the initial mesh,
so no interpolation needed), then **scattered onto the Eulerian mesh** using the
post-N-body particle positions:
```python
weights = lagrangian_bias_weights(delta_ic, k_vecs, b1, b2, bs2, bn2)
n_biased = dj2.compute_field_quantity_from_particles(pos=X_sim, quantity=weights.reshape(-1), ...)
delta_biased = n_biased / n_biased.mean() - 1.0
```
The **IC-side** ("ic") variant skips N-body and measures the summary directly on
`delta_L = w(q) − 1` (the Lagrangian bias expansion field).

**Shot noise (Poisson Gaussian limit, arXiv:2504.20130v2).** Optional per-voxel
Gaussian noise `ε ~ N(0, 1/N̄_g)`, `N̄_g = n_g · V_cell`, `V_cell=(boxsize/res)³`.
Controlled by `n_g_density` (typical DESI-like value `1e-3`).

---

## 3. Dataset generation (the SBI / Falcon track)

Datasets are generated by running the forward model as a **Falcon graph** of
"nodes". Falcon is a DAG-based simulation/inference framework; each node is a
Python class with a `simulate_batch(batch_size, *parent_arrays) -> np.ndarray`
contract. The graph is declared in a YAML config under `config_files/`.

### Node classes (`core/model/`)

| Falcon `_target_`               | Class (file)                       | Maps |
|---------------------------------|------------------------------------|------|
| `core.model.PowerSpectrum`      | `PowerSpectrum` (model.py)         | `cosmo_params → P_lin(k)` |
| `core.model.InitialConditions`  | `InitialConditions` (model.py)     | `P_lin → delta_ic` (seeded GRF) |
| `core.model.ForwardModel`       | `ForwardModel` (model.py)          | `(cosmo, delta_ic) → delta_fin` (N-body, optional RSD + shot noise) |
| `core.model.ForwardModelPk`     | `ForwardModelPk` (model.py)        | as above, returns measured P(k) (Pylians) |
| `core.model.ForwardModelBiasPk` | `ForwardModelBiasPk` (model_bias.py)| biased-tracer field/PS |
| `core.model.PkMultipoles`       | `PkMultipoles` (summaries.py)      | `field → [P_0, P_2, P_4]` (BFast) |
| `core.model.BkMonopole`         | `BkMonopole` (summaries.py)        | `field → B_0` (BFast) |

These are re-exported from `core/model/__init__.py` so the `_target_` paths
resolve.

### How a node works (interface contract)

- All parent arrays arrive as **numpy float32**, already batched (axis 0 =
  `batch_size`); the return must be numpy float32 with the same leading axis.
- **JAX must not be imported at module level** (it breaks Ray actor
  serialization) — each node imports JAX inside its methods.
- Inside a node, the single-sample kernel is `jax.jit`-compiled and
  `jax.vmap`-ed over the batch. `ForwardModel` additionally splits the batch
  into `sub_batch_size` chunks via `jax.lax.map` to bound GPU memory
  (`sub_batch_size=8` fits an A100 at res=64; configs often use 1).

### Seeding (reproducibility & uniqueness)

`InitialConditions` derives each sample's RNG seed from the **count of existing
prior samples on disk** (`run_dir/samples_dir/prior/*.npz`), so successive
"chunks" never collide and the dataset is reproducible. Shot-noise RNG in
`ForwardModel` uses the same counting scheme but offset by
`_SN_SEED_NAMESPACE = 0x5000_0000` so it is decorrelated from the IC seeds.

### Example graph (`config_files/config_base.yml`)

The 2-param base config wires:
```
cosmo_params ─► pk ─► delta_ic ─┬─► delta_fin ─┬─► pk_l_fin (P0,P2,P4)
                                │              └─► bk_0_fin (B0)
                                ├─► pk_l_ic (P0,P2,P4)
                                └─► bk_0_ic (B0)
```
- `cosmo_params`: `Hypercube` prior, `uniform[0.10,0.50]` for Ω_m and
  `uniform[0.60,1.00]` for σ_8. Ω_b, h, n_s pinned to fiducial.
- `delta_fin`: unbiased matter mapped into redshift space (`apply_rsd: true`,
  `los_axis: 2`) with DESI-like shot noise (`n_g_density: 1.0e-3`) baked in.
- BFast multipole + bispectrum summaries stored for both the IC and final fields.
- The `cosmo_params` node also carries the SBI **estimator** spec (a
  normalizing-flow `Flow`/`nsf` with an embedding network) — uncomment the
  `evidence:` / `observed:` keys to switch from forward-sampling to inference.

### Running dataset generation

```bash
RUN_DIR="outputs/base_run_$(date +%y%m%d_%H%M%S)"
falcon sample prior --config-name config_files/config_base.yml --run-dir "$RUN_DIR"
# Accumulate more samples by re-running (Falcon appends; seeds auto-advance):
for i in $(seq 1 $N_CHUNKS); do
  falcon sample prior --config-name config_files/config_base.yml --run-dir "$RUN_DIR"
done
# Train the posterior estimator (after uncommenting evidence/observed):
falcon train --config-name config_files/config_base.yml --run-dir "$RUN_DIR"
```
On the cluster this is wrapped in SLURM scripts `my_py_gpu_job_*.sh`. There are
many config/job variants (`config_5param*`, `config_bias_*`, `config_fiducial_*`,
`config_*_pk`, `config_*_LR`, etc.) — each fixes a different parameter
subset, box size, resolution, summary, or prior. The `config_fiducial_*` set
generates **companion datasets** that pin the cosmology via a degenerate
`uniform[fid,fid]` prior so a varying-param set's full observable distribution
can be matched at the fiducial point.

---

## 4. Fisher forecasting (the autodiff track)

Code lives in `core/fisher.py` (library) and `core/run_fisher.py` (CLI). The
key idea: the forward model above is re-implemented as **single differentiable
functions** `params → summary`, and `jax.jacrev` gives exact Jacobians
`dP(k)/dθ` that build the Fisher matrix.

### 4.1 The differentiable pipelines

Each is a standalone function reproducing the physics pipeline of §2 with the
white noise held **fixed** (generated outside, not differentiated → the map is
deterministic and differentiable). Naming convention `_differentiable_pipeline_<field>[_bias][_PB][_rsd_multi]`:

| Function | Params → output |
|----------|-----------------|
| `_differentiable_pipeline_fin`        | cosmo → P(delta_fin) |
| `_differentiable_pipeline_ic`         | cosmo → P(delta_ic) |
| `_differentiable_pipeline_fin_bias`   | [cosmo+bias] → P(delta_biased) |
| `_differentiable_pipeline_ic_bias`    | [cosmo+bias] → P(delta_L = w−1) |
| `_differentiable_pipeline_fin_bias_PB`| [cosmo+bias] → [P(k), B(k1,k2,k3)] (BFast) |
| `_differentiable_pipeline_ic_bias_PB` | [cosmo+bias] → [P, B] of delta_L |
| `_differentiable_pipeline_both_bias_PB`| → [P_ic, B_ic, P_fin, B_fin] (one N-body, both summaries) |
| `*_rsd_multi`                          | RSD multipole variants: → [P_0, P_2, P_4, B] |

### 4.2 The crucial autodiff subtlety (read this)

DiscoDJ's N-body has a **custom VJP (reverse-mode) rule but no JVP rule**, so
forward-mode AD (`jacfwd`) through it returns all-NaN. **Reverse-mode
(`jax.jacrev`) is used** because it pulls cotangents back through the working
VJP path.

Even so, DiscoDJ's reverse-mode adjoint produces **NaN cotangents for the
cosmology pytree path** through `D_plus`/`F_plus`/time-tables. The fix:
the cosmology fed to the N-body is wrapped in `jax.lax.stop_gradient`:
```python
cosmo_nbody = jax.lax.stop_gradient(cosmo)
```
So the cosmology gradient flows **only through `delta_ic`** (which already
carries the full `P_lin → IC` derivative); the growth-factor/time-evolution
channel through the N-body is treated as a fixed approximation. Bias gradients
flow through the Lagrangian weights, which are pure JAX and fully
differentiable. (See `docs/fisher_jacrev_fix.md` for the history.)

A **finite-difference fallback** (`compute_jacobian_numerical`, central
differences with per-parameter steps `_FINITE_DIFF_STEPS{,_BIAS}`) kicks in if
jacrev returns all-NaN, and can be forced for the bias-PB path with
`--derivative-method fd` (full chain rule, no stop_gradient bias; 1+2·n_params
forward passes per seed). A JVP/VJP **diagnostic** (`--diagnose`,
`_diagnose_tangent[_bias]`) pulls a unit cotangent through every stage and
reports where finiteness drops — useful to locate where a gradient dies.

### 4.3 Seed averaging

Per the design, the Jacobian is **averaged over `n_seeds` white-noise
realizations** (`compute_jacobian`). Each seed: one fresh
`jax.random.normal(PRNGKey(seed))` noise field, one jacrev call.

> ⚠️ **Seed-count caveat (documented in the CLI help):** seed-averaged Jacobian
> noise biases the Fisher *upward* by ~1/n_seeds, so `n_seeds < 10` yields
> artificially tight constraints. Default is 20.

### 4.4 Covariance

The data covariance `C` of the summary vector. Options (`--cov-type`):

- **`diag`** (default): block-diagonal Gaussian.
  - P(k): `σ²(k) = 2 P(k)² / N_modes(k)` (`gaussian_covariance`; `count_modes`
    replicates DiscoDJ's log-binned mode counting, rFFT interior modes counted
    twice).
  - Multipoles: full Grieb+2016 Gaussian multipole covariance
    (`multipole_covariance_grieb`, includes ℓ–ℓ' cross terms — the diagonal
    limit biases Fisher upward under Kaiser RSD).
  - Bispectrum: Scoccimarro/Sefusatti diagonal variance
    (`bispectrum_covariance`).
  - P–B cross block = 0.
- **`euclid_cov`**: full **empirical** covariance estimated from `--n-cov-seeds`
  fiducial realizations (`compute_empirical_covariance`), **Hartlap-corrected**.
  Captures P–B cross, k–k', ℓ–ℓ', IC–FIN cross, and non-Gaussian terms. Requires
  `n_cov_seeds > n_data + 2` for invertibility.
- **`block_with_pb_cross_empirical`** (joint-PB only): analytic P–P (Grieb) and
  B–B (Scoccimarro) blocks, but an empirical P–B cross block — cheaper than full
  `euclid_cov`.

Invalid bins (zero modes, non-finite P or Jacobian) are masked out before the
Fisher contraction.

### 4.5 The Fisher matrix

For the diagonal case (`fisher_matrix`):
```
F_ij = Σ_a (dP(k_a)/dθ_i) · (1/σ²_a) · (dP(k_a)/dθ_j)
```
For a full covariance (`fisher_matrix_full`): `F = Jᵀ C⁻¹ J`.

Then:
- **Prior Fisher** `F_prior` (`prior_fisher_matrix`) adds the prior information:
  uniform priors contribute via their variance, normal priors via `1/σ²`.
  Prior specs are `_PRIOR_SPEC` (cosmo: all uniform) and `_PRIOR_SPEC_BIAS`
  (cosmo uniform, bias params normal: b1~N(1,0.5), b2/bs2/bn2~N(0,2)).
- **Total** `F_total = F_data + F_prior`.
- **Parameter covariance** = `inv(F_total)`; marginal 1σ errors =
  `sqrt(diag(inv(F_total)))`.

### 4.6 Outputs

`run_fisher_forecast[_bias][_PB]` writes to a timestamped `output_dir`:
- `fisher_results.npz` — `F_data`, `F_prior`, `F_total`, `fisher_cov`
  (=inv F_total), `fisher_cov_data`, `jacobian`, per-seed Jacobians, `Pk_fid`,
  `k_bins`, `sigma_sq`, `N_modes`, `cosmo_fid`, `param_names`.
- A **corner plot** of the Fisher error ellipses (`plot_fisher_corner`),
  optionally overlaid with SBI posterior samples loaded from `--sbi-samples`
  (a `.npz` with key `samples` or a Falcon run directory). This overlay is how
  the Fisher forecast is compared against the SBI posterior.

### 4.7 Running a Fisher forecast

```bash
# Plain 5-param P(delta_fin):
python core/run_fisher.py --n-seeds 20 --n-bins 30 --output-dir outputs/fisher/

# IC field instead of final field:
python core/run_fisher.py --field ic

# Bias track, joint P+B with RSD multipoles, empirical covariance:
python core/run_fisher.py --bias --joint-pb --pk-multipoles \
    --field both --cov-type euclid_cov --n-cov-seeds 300 --n-g-density 1e-3

# Overlay an SBI posterior:
python core/run_fisher.py --sbi-samples outputs/5param_run/sbi_samples.npz
```
Or via config: `python core/run_fisher.py --config config_files/fisher/<x>.yml`
(YAML keys = argparse dests with dashes→underscores; the config is copied into
the run dir as `source_config.yml`). The SLURM wrapper is
`my_py_gpu_job_fisher.sh` (config `config_files/fisher/fisher.yml`), which
tees stdout into `RUN_DIR/run.log` and snapshots SLURM logs.

Important CLI flags (`run_fisher.py`):

| Flag | Meaning |
|------|---------|
| `--n-seeds` | white-noise seeds for Jacobian averaging (default 20; <10 biases tight) |
| `--n-bins` | number of P(k) bins (default 30) |
| `--bias` | bias track (cosmo+bias params) |
| `--field {fin,ic,both}` | which field's summary to forecast |
| `--joint-pb` | joint P(k)+B(k1,k2,k3) via BFast (bias-only) |
| `--pk-multipoles` | redshift-space P_0/P_2/P_4 (RSD, los=z) |
| `--n-g-density` | inject shot noise; suffixes output dir with `_SN` |
| `--cov-type` | `diag` / `euclid_cov` / `block_with_pb_cross_empirical` |
| `--derivative-method {autodiff,fd}` | jacrev vs finite differences |
| `--jacrev-chunk-size` | memory-bounded jacrev (`_chunked_jacrev`) for large data vectors |
| `--diagnose` | run only the per-stage tangent diagnostic and exit |

---

## 5. How the two tracks connect

```
                       ┌─────────────────────────────┐
   cosmo_params (θ) ──►│   SHARED FORWARD PHYSICS    │──► observable summary
                       │  P_lin → δ_ic → δ_fin → S    │
                       └─────────────────────────────┘
                          │                       │
        many draws from   │                       │  differentiate w.r.t. θ
        prior (Falcon)    ▼                       ▼  (jax.jacrev, fixed noise)
                 (θ, S) training pairs        dS/dθ  ──► Fisher F = Jᵀ C⁻¹ J
                          │                       │
                          ▼                       ▼
                  SBI posterior estimator   forecast ellipses inv(F)
                  (normalizing flow)               │
                          └──────── overlaid on the corner plot ◄──┘
```

- **Same physics, two derivatives of use:** the dataset track *samples* the
  forward map; the Fisher track *differentiates* it. They must agree on box
  geometry, `SIM_PARAMS`, fiducial cosmology, RSD/shot-noise/bias settings to
  be comparable.
- **The Fisher forecast is the Gaussian/linear-response baseline**; the SBI
  posterior is the full non-Gaussian answer. Comparing them (the overlay in
  `plot_fisher_corner`) is a central goal of the thesis.

---

## 6. File map (where to look)

| Path | Role |
|------|------|
| `core/model/model.py` | Full 5-param forward model nodes + `BOX_PARAMS`, `SIM_PARAMS`, `_FIDUCIAL_COSMO`, `_cosmo_dict` |
| `core/model/model_minimal.py` | Fixed-cosmology forward model |
| `core/model/model_bias.py` | Biased-tracer forward model node (`ForwardModelBias`/`...Pk`) |
| `core/model/bias.py` | Pure-JAX Lagrangian bias operators (`lagrangian_bias_weights`, tidal shear, Laplacian) |
| `core/model/summaries.py` | BFast summary nodes (`PkMultipoles`, `BkMonopole`) |
| `core/model/__init__.py` | Re-exports so Falcon `_target_` paths resolve |
| `core/fisher.py` | Differentiable pipelines, Jacobian, covariances, Fisher assembly, plotting |
| `core/fisher_2param.py` | Fisher forecast specialized for (Ω_m, σ_8) |
| `core/run_fisher.py` | Fisher CLI (argparse + YAML config) |
| `core/utils.py` | Plotting / physics helpers (Pylians & CLASS P(k), etc.) |
| `config_files/*.yml` | Falcon graph configs (datasets) + `fisher/*.yml` (Fisher runs) |
| `my_py_gpu_job_*.sh` | SLURM wrappers (one per config/run variant) |
| `docs/` | Design notes: `fisher_methodology.md`, `fisher_jacrev_fix.md`, `biased_pipeline.md`, `fisher_PB_*_design.md`, `fisher_too_tight_fixes.md`, `seeds_overview.md` |
| `core/verification/` | Cross-checks: jacrev-vs-FD, n-seeds sweep, Grieb-cov validation, σ comparison |

---

## 7. Gotchas / non-obvious facts (collected)

1. **`Omega_c` not `Omega_m`** in DiscoDJ cosmo dicts (`Omega_c = Omega_m − Omega_b`).
2. **Forward-mode AD is broken** through DiscoDJ's N-body (no JVP) → always use
   `jax.jacrev`.
3. **Cosmology is `stop_gradient`'d through the N-body**; the cosmo gradient
   only flows via `delta_ic`. This is an approximation, deliberate.
4. **White noise is fixed outside** the differentiated function — required for
   the map to be deterministic/differentiable.
5. **`n_seeds < 10` biases the Fisher too tight** (~1/n_seeds upward bias).
6. **Default DiscoDJ P(k) binning is logarithmic**; `count_modes` and
   `_power_spectrum_safe` must match it (rFFT interior modes counted twice).
7. **JAX is not imported at module level** in node files — Ray serialization.
8. **Seeds derive from on-disk sample counts** so dataset chunks are unique and
   reproducible; shot-noise seeds are namespaced apart from IC seeds.
9. **`--field ic` with `--bias`** measures the Lagrangian bias expansion field
   `P(w(q)−1)`, *not* the linear matter IC.
10. **Fiducial companion datasets** use a degenerate `uniform[fid,fid]` prior to
    pin cosmology while keeping the full observable pipeline.
```

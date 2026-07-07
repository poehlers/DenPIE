# Biased-Tracer Forward Model — End-to-End Pipeline

This document describes the full pipeline that turns cosmological + bias
parameters into a biased tracer density field (and, optionally, its measured
power spectrum). The pipeline is orchestrated by [Falcon] as a DAG of
simulator nodes; this doc walks each node in physical order and points to
the relevant code.

[Falcon]: https://github.com/auto-differentiation/falcon

---

## 1. Overview

```
cosmo_params  ─────────────────────────────────┐
     │                                         ▼
     ▼                                  delta_biased  ◄── bias_params
    pk ──► delta_ic ───────────────────────────┘
```

The graph has five nodes (see `config_files/config_bias_base.yml`):

| Node           | `_target_`                                  | Parents                                   |
| -------------- | ------------------------------------------- | ----------------------------------------- |
| `cosmo_params` | `falcon.priors.Hypercube`                   | —                                         |
| `bias_params`  | `falcon.priors.Hypercube`                   | —                                         |
| `pk`           | `core.model.PowerSpectrum`                  | `cosmo_params`                            |
| `delta_ic`     | `core.model.InitialConditions`              | `pk`                                      |
| `delta_biased` | `core.model.model_bias.ForwardModelBias`    | `cosmo_params, delta_ic, bias_params`     |

The biased field is produced by evolving a Gaussian-random IC through a PM
N-body, weighting each Lagrangian particle by a 2nd-order Lagrangian bias
expansion, and scattering the weighted particles back to a mesh (optionally
in redshift space).

---

## 2. Stage-by-stage walkthrough

### 2.1 Cosmological parameters

Defined in `config_files/config_bias_base.yml:33-39`. The base config samples
only the two parameters that drive most of the LSS signal, with the rest
pinned to the Planck 2018 / Quijote fiducial:

| Parameter | Status     | Prior / value                                |
| --------- | ---------- | -------------------------------------------- |
| Ω_m       | sampled    | Uniform(0.10, 0.50)                          |
| σ_8       | sampled    | Uniform(0.60, 1.00)                          |
| Ω_b       | fixed      | 0.0490                                       |
| h         | fixed      | 0.6711                                       |
| n_s       | fixed      | 0.9624                                       |

The sampled vector has shape `(batch_size, D)` with `D = 5 − len(fixed_cosmo_params)`.
`core/model/model.py:_FIDUCIAL_COSMO` and `_cosmo_dict()` translate the sampled
+ fixed parts into the DiscoDJ cosmology dict (note: DiscoDJ uses
`Omega_c = Omega_m − Omega_b`, handled inside `_cosmo_dict`).

### 2.2 Bias parameters

Sampled independently from cosmology (`config_bias_base.yml:66-74`):

| Parameter | Meaning                       | Prior            |
| --------- | ----------------------------- | ---------------- |
| b₁        | Linear bias                   | Normal(1.0, 0.5) |
| b₂        | Quadratic bias                | Normal(0.0, 2.0) |
| bs²       | Tidal-shear bias              | Normal(0.0, 2.0) |
| bn²       | Laplacian bias (∇²δ)          | Normal(0.0, 2.0) |

Shape passed to the simulator: `(batch_size, 4)` in the order
`[b₁, b₂, bs², bn²]`.

### 2.3 Linear matter power spectrum

`core.model.PowerSpectrum` (instantiated in `config_bias_base.yml:100-112`)
calls DiscoDJ's transfer-function machinery to produce the linear P(k) for the
sampled cosmology. The fixed parameters `Ω_b, h, n_s` are injected via
`fixed_cosmo_params`. Box geometry: `dim=3, res=128, boxsize=1000.0 Mpc/h`.

### 2.4 Initial conditions

`core.model.InitialConditions` (config lines 114-125) generates a
Gaussian-random density field consistent with the linear P(k):

```
delta_ic = irfftn( rfftn(white_noise) * sqrt(P(k) · norm_fac) )
```

White noise comes from `jax.random.normal`, with a deterministic seed offset
derived from the number of samples already in `run_dir`. This makes chunked
runs append cleanly without seed collisions (see the chunking loop in
`my_py_gpu_job_fiducial_bias.sh`).

### 2.5 N-body evolution

Implemented in `core/model/model_bias.py:_simulate_impl` (lines 129–181). For
each sample in the batch:

1. Build a DiscoDJ instance at the sampled cosmology
   (`model_bias.py:135-138`).
2. Inject the externally-supplied `delta_ic` via
   `dj.with_external_ics(...)` (line 140).
3. Apply 2nd-order LPT for the initial kick (`with_lpt(n_order=2)`, line 141).
4. Run the PM N-body to `a_end = 1.0` (line 146):

   ```python
   X_sim, P, _ = dj.run_nbody(**run_params, use_diffrax=False)
   ```

   `X_sim : (res, res, res, 3)` are the Eulerian particle positions and
   `P` are the canonical momenta (used for RSD only).

Integration settings (`sim_params` in config, lines 144-160):

| Key                  | Value     | Notes                                |
| -------------------- | --------- | ------------------------------------ |
| `a_ini` → `a_end`    | 0.25 → 1.0| `z ≈ 3 → 0`                          |
| `stepper`            | bullfrog  | symplectic                           |
| `method`             | pm        | particle-mesh                        |
| `res_pm`             | 256       | 2 × `res` (force resolution)         |
| `nlpt_order_ics`     | 2         | 2-LPT initial kick                   |
| `n_steps`            | 1         | single PM step                       |
| `worder`             | 2         | CIC kernel order for the final scatter |

### 2.6 Lagrangian bias expansion

At every Lagrangian grid point `q`, the per-particle weight is

```
w(q) = 1 + b₁·δ(q) + b₂·[δ²(q) − ⟨δ²⟩] + bs²·[s²(q) − ⟨s²⟩] + bn²·∇²δ(q)
```

with all fields evaluated on the **initial** density field `delta_ic`. Because
the Lagrangian grid is the regular initial mesh, the weight of particle
`(i,j,k)` is just `w[i,j,k]` — no interpolation needed.

Implemented in `core/model/bias.py`:

- `lagrangian_bias_weights(delta_ic, k_vecs, b1, b2, bs2, bn2)` — line 89,
  the top-level function used by the model.
- `_tidal_shear_sq(delta_k, k_vecs)` — line 34. Computes
  `Φ̂ = −δ̂/k² = inv_laplace_kernel · δ̂`, then
  `T_ij = IRFFT(gᵢ · gⱼ · Φ̂)` with `gᵢ = i·kᵢ`, and finally

  ```
  s² = T_xx² + T_yy² + T_zz² + 2·(T_xy² + T_xz² + T_yz²) − (1/3)·δ²
  ```

- `_laplacian(delta_k, k_vecs)` — line 70. Returns
  `IRFFT(−k² · δ̂)`.

The `k_vecs` (sparse 1-D Fourier wavenumber grids) are pre-computed once in
`ForwardModelBias.__init__` (model_bias.py:116-120) since they depend only on
box geometry, and reused across all batched evaluations.

The bias call sits at `model_bias.py:159-162`. Setting
`b₁=1, b₂=bs²=bn²=0` recovers unbiased matter (`weights = 1 + δ_ic`).

### 2.7 Scatter to the Eulerian mesh

After bias weighting, the particles are scattered to a mesh
(`model_bias.py:167-178`):

```python
n_biased = dj.compute_field_quantity_from_particles(
    pos=X_sim,
    quantity=weights.reshape(-1),
    normalize_by_density=False,
    in_redshift_space=self.apply_rsd,
    vel=P_flat,
    a=sp["a_end"] if self.apply_rsd else None,
    radial_dim=self.los_axis,
    worder=sp["worder"],   # 2 → CIC
    antialias=1,
    deconvolve=True,
)
```

The resulting raw weighted count `n_biased(x)` is normalised to a density
contrast (`model_bias.py:181`):

```
δ_biased(x) = n_biased(x) / ⟨n_biased⟩ − 1
```

### 2.8 Optional Kaiser redshift-space distortions

When `apply_rsd=True` (default in `config_bias_base.yml:138`), particles are
displaced by `v_los / H` along `los_axis` *before* the scatter. The momenta
returned by DiscoDJ are in grid shape `(res, res, res, 3)`; they must be
pre-flattened to `(res³, 3)` so that `vel[:, radial_dim]` indexes the particle
axis rather than a spatial axis (see the comment block at
`model_bias.py:150-155`):

```python
P_flat = P.reshape(-1, 3) if self.apply_rsd else None
```

The line-of-sight direction is configured via `los_axis ∈ {0, 1, 2}`.

---

## 3. P(k) variant

`ForwardModelBiasPk` (`model_bias.py:252-301`) inherits from
`ForwardModelBias` and changes the output to the *measured* 3-D power
spectrum:

```
pk_measured : (batch_size, 2, N_bins) float32
    [:, 0, :] = k   in h/Mpc
    [:, 1, :] = P(k) in (Mpc/h)³
```

The density field is computed transiently (line 285) and never stored by
Falcon; P(k) is measured per-sample with Pylians via
`core.utils.pylians_pk` (line 288 / 292).

Use `config_files/config_bias_pk.yml` to switch the `delta_biased` node from
the raw-field simulator to the P(k) simulator.

---

## 4. Batching & GPU memory

- The kernel `_simulate_impl` is `jax.vmap`'d over the batch axis
  (`model_bias.py:123`).
- The batch is further chopped into sub-batches of size `sub_batch_size`
  using `jax.lax.map` (lines 187-204), which executes them sequentially on
  the GPU. This caps peak VRAM.
- **Constraint:** `sample.prior.n` (in the config) must be divisible by
  `sub_batch_size`. `sub_batch_size=1` is always safe; larger values give
  better GPU utilisation if memory permits (≤ 16 on an A100 at `res=128`).

---

## 5. Config knob reference

All knobs live in `config_files/config_bias_base.yml`.

| Section / key                                       | Purpose                                                 |
| --------------------------------------------------- | ------------------------------------------------------- |
| `sample.prior.n`                                    | Samples per `falcon sample prior` call                  |
| `graph.cosmo_params.simulator.priors`               | Cosmology prior bounds                                  |
| `graph.bias_params.simulator.priors`                | Bias prior means/stds                                   |
| `graph.pk` / `graph.delta_ic` (`dim, res, boxsize`) | Box geometry (must match `delta_biased`)                |
| `graph.delta_biased.simulator.fixed_cosmo_params`   | Cosmological parameters pinned to fiducial             |
| `graph.delta_biased.simulator.sim_params`           | N-body integration settings (see §2.5)                  |
| `graph.delta_biased.simulator.apply_rsd`            | Toggle Kaiser RSD                                       |
| `graph.delta_biased.simulator.los_axis`             | Line-of-sight axis (0=x, 1=y, 2=z)                      |
| `graph.delta_biased.simulator.sub_batch_size`       | GPU sub-batch (must divide `sample.prior.n`)            |

---

## 6. How to run

### Prior sampling

```bash
RUN_DIR=outputs/bias_run_$(date +%y%m%d_%H%M%S)
falcon sample prior \
    --config-name config_files/config_bias_base.yml \
    --run-dir "${RUN_DIR}"
```

### Chunked accumulation

`my_py_gpu_job_fiducial_bias.sh` loops 10 calls of 100 samples each into a
shared `run-dir`, giving 1000 prior samples; the `InitialConditions` node
offsets its RNG seed by the existing sample count so chunks don't collide.

### Posterior training

1. In `config_bias_base.yml`, uncomment the `evidence: [delta_biased]` line(s)
   on the node(s) you want to infer (`cosmo_params`, `bias_params`, and/or
   `delta_ic`).
2. Uncomment `observed: "./data/observed_biased.npz['delta_biased']"` (line 131)
   and point it at the observation.
3. Run:

   ```bash
   falcon train --config-name config_files/config_bias_base.yml --run-dir "${RUN_DIR}"
   ```

---

## 7. Outputs

After a run, `${run_dir}/graph/` contains one subdirectory per node:

```
graph/
├── cosmo_params/   # prior samples + trained flow estimator
├── bias_params/    # prior samples + trained flow estimator
├── pk/             # linear P(k) per sample
├── delta_ic/       # Gaussian-random IC fields
├── delta_biased/   # final biased δ fields (float32, (B, res, res, res))
└── driver/         # Falcon orchestration metadata
```

The biased fields are stored as `.npz` arrays under
`delta_biased/.../prior/` and can be loaded directly with `numpy.load`.

---

## 8. File map

| File                                                   | Role                                              |
| ------------------------------------------------------ | ------------------------------------------------- |
| `core/model/model_bias.py`                             | `ForwardModelBias`, `ForwardModelBiasPk`          |
| `core/model/bias.py`                                   | Pure-JAX Lagrangian bias operators                |
| `core/model/model.py`                                  | `PowerSpectrum`, `InitialConditions`, fiducial constants |
| `core/utils.py`                                        | `pylians_pk` and plotting helpers                 |
| `config_files/config_bias_base.yml`                    | Base biased-field config                          |
| `config_files/config_bias_pk.yml`                      | Same pipeline, P(k) output                        |
| `my_py_gpu_job_fiducial_bias.sh`                       | SLURM entry point (prior sampling, chunked)       |

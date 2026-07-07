# Fisher Forecasting Methodology

This document describes the Fisher information matrix formalism as implemented in this project for two cases:

1. **Case 1** — Cosmological parameters only, observable = nonlinear matter power spectrum $P_\mathrm{NL}(k)$
2. **Case 2** — Joint cosmological + bias parameters, observable = biased tracer power spectrum $P_\mathrm{biased}(k)$

Both cases differentiate through a full particle-mesh N-body simulation using JAX automatic differentiation.


---

## 1. Introduction

The Fisher information matrix provides a lower bound on the variance of any unbiased estimator of model parameters (the Cramer-Rao bound). In cosmology, Fisher forecasts are used to predict how well a given observable (e.g. the matter power spectrum) can constrain cosmological parameters, without needing to run a full Bayesian inference pipeline.

In this project the Fisher matrix is computed by differentiating through a **fully differentiable N-body simulation** (DiscoDJ) using JAX forward-mode automatic differentiation (`jax.jacfwd`). This yields exact Jacobians of the observable with respect to the parameters, which are then combined with a Gaussian covariance model to obtain the Fisher matrix.


---

## 2. General Fisher Formalism

### 2.1 Definition

For a data vector $\mathbf{d}$ with likelihood $\mathcal{L}(\mathbf{d} | \boldsymbol{\theta})$, the Fisher information matrix is

$$
F_{ij} = -\left\langle \frac{\partial^2 \ln \mathcal{L}}{\partial \theta_i \, \partial \theta_j} \right\rangle
$$

where the average is taken over realisations of the data at fixed parameters $\boldsymbol{\theta}$.

### 2.2 Gaussian Likelihood

For a Gaussian likelihood with parameter-independent covariance $\mathbf{C}$, the Fisher matrix reduces to

$$
F_{ij} = \frac{\partial \boldsymbol{\mu}^T}{\partial \theta_i} \, \mathbf{C}^{-1} \, \frac{\partial \boldsymbol{\mu}}{\partial \theta_j}
$$

where $\boldsymbol{\mu}(\boldsymbol{\theta})$ is the mean data vector (the model prediction). When the covariance is diagonal with entries $\sigma_a^2$, this simplifies to a sum over data bins:

$$
F_{ij} = \sum_a \frac{1}{\sigma_a^2} \, \frac{\partial \mu_a}{\partial \theta_i} \, \frac{\partial \mu_a}{\partial \theta_j}
$$

### 2.3 Cramer-Rao Bound

The covariance of any unbiased estimator $\hat{\boldsymbol{\theta}}$ satisfies

$$
\mathrm{Cov}(\hat{\boldsymbol{\theta}}) \geq \mathbf{F}^{-1}
$$

The marginalised 1-$\sigma$ uncertainty on parameter $\theta_i$ is therefore bounded by

$$
\sigma(\theta_i) \geq \sqrt{(\mathbf{F}^{-1})_{ii}}
$$


---

## 3. Observable: The Power Spectrum

### 3.1 Gaussian Covariance

For a density field in a periodic box of volume $V = L^3$, the Gaussian covariance of the binned power spectrum is

$$
\sigma^2(k_a) = \frac{2 \, P(k_a)^2}{N_\mathrm{modes}(k_a)}
$$

where $N_\mathrm{modes}(k_a)$ is the number of independent Fourier modes in bin $a$. This is the leading-order covariance; it neglects the connected (non-Gaussian) trispectrum contribution, which is subdominant on large scales.

### 3.2 Mode Counting

Modes are counted on the discrete Fourier grid of the simulation box. For a grid with resolution $N_\mathrm{res}$ and box size $L$:

- Fundamental mode: $k_\mathrm{min} = 2\pi / L$
- Nyquist frequency: $k_\mathrm{max} = \pi N_\mathrm{res} / L$

Because the density field is real, its Fourier transform has Hermitian symmetry $\hat{\delta}(\mathbf{k}) = \hat{\delta}^*(-\mathbf{k})$. The real FFT (rFFT) stores only half the modes along the last axis ($N_\mathrm{res}/2 + 1$ elements). Interior modes along this axis (not $k = 0$ or $k = k_\mathrm{Nyq}$) represent two independent modes each (+$\mathbf{k}$ and $-\mathbf{k}$), so they are counted with a factor of 2.

Implementation (`core/fisher.py:count_modes`):

```python
rfft_factor = jnp.ones(rfft_shape, dtype=jnp.int32)
rfft_factor = rfft_factor.at[..., 1:-1].set(2)  # interior modes count double
N_modes = jnp.bincount(dig, weights=rfft_factor)
```

### 3.3 Logarithmic Binning

$P(k)$ is binned in $n_\mathrm{bins}$ logarithmically-spaced bins between $k_\mathrm{min}$ and $k_\mathrm{max}$, matching the default binning in DiscoDJ's `power_spectrum` function. Bin edges are computed as:

$$
\Delta \log_{10} k = \frac{\log_{10} k_\mathrm{max} - \log_{10} k_\mathrm{min}}{n_\mathrm{bins} - 1}
$$

$$
k_\mathrm{edges} = \mathrm{geomspace}\!\left(k_\mathrm{min} \cdot 10^{-\Delta\log k / 2}, \; k_\mathrm{max} \cdot 10^{+\Delta\log k / 2}, \; n_\mathrm{bins} + 1\right)
$$

Effective bin centres are mode-count-weighted averages of $|\mathbf{k}|$ within each bin.

### 3.4 Fisher Matrix in Terms of P(k)

Combining Sections 2.2 and 3.1, the Fisher matrix for the binned power spectrum is

$$
\boxed{F_{ij} = \sum_a \frac{N_\mathrm{modes}(k_a)}{2 \, P(k_a)^2} \, \frac{\partial P(k_a)}{\partial \theta_i} \, \frac{\partial P(k_a)}{\partial \theta_j}}
$$

This is implemented as a matrix product:

```python
weighted = jacobian / sigma_sq[:, None]   # (n_bins, N_params)
F = weighted.T @ jacobian                 # (N_params, N_params)
```


---

## 4. Case 1: Cosmological Parameters Only

### 4.1 Parameter Space

The 5 sampled cosmological parameters are:

| Parameter    | Symbol       | Fiducial (Planck 2018) | Prior range       |
|:-------------|:-------------|:-----------------------|:------------------|
| Total matter | $\Omega_m$   | 0.3175                 | [0.10, 0.50]      |
| Baryons      | $\Omega_b$   | 0.0490                 | [0.03, 0.07]      |
| Hubble       | $h$          | 0.6711                 | [0.50, 0.90]      |
| Spectral index | $n_s$      | 0.9624                 | [0.80, 1.20]      |
| Amplitude    | $\sigma_8$   | 0.8340                 | [0.60, 1.00]      |

Note: DiscoDJ uses $\Omega_c$ (CDM density) internally, related via $\Omega_c = \Omega_m - \Omega_b$.

Any subset of these parameters may be sampled while the rest are held fixed (via `fixed_cosmo_params`).

### 4.2 The Differentiable Pipeline

The observable $P_\mathrm{NL}(k)$ is computed through a 4-stage differentiable pipeline:

$$
\boldsymbol{\theta} \xrightarrow{\text{CLASS}} P_\mathrm{lin}(k) \xrightarrow{\text{IC gen}} \delta_\mathrm{ic}(\mathbf{x}) \xrightarrow{\text{N-body}} \delta_\mathrm{fin}(\mathbf{x}) \xrightarrow{\text{binning}} P_\mathrm{NL}(k)
$$

#### Stage 1: Linear Power Spectrum

Cosmological parameters $\boldsymbol{\theta}$ are passed to DiscoDJ, which calls the CLASS Boltzmann solver to compute the linear matter power spectrum $P_\mathrm{lin}(k)$ at $z = 0$.

#### Stage 2: Initial Conditions

Gaussian random initial conditions are generated from $P_\mathrm{lin}(k)$ and a fixed white-noise field $\xi(\mathbf{x})$:

$$
\hat{\delta}_\mathrm{ic}(\mathbf{k}) = \hat{\xi}(\mathbf{k}) \cdot \sqrt{P_\mathrm{lin}(|\mathbf{k}|) \cdot \left(\frac{N_\mathrm{res}}{L}\right)^3}
$$

$$
\delta_\mathrm{ic}(\mathbf{x}) = \mathrm{IFFT}\!\left[\hat{\delta}_\mathrm{ic}(\mathbf{k})\right]
$$

The normalization factor $(N_\mathrm{res}/L)^3$ converts from continuous to discrete Fourier conventions. The white noise $\xi$ is drawn from `jax.random.normal` with a fixed seed and is **not** differentiated through --- it is held constant while derivatives are taken with respect to $\boldsymbol{\theta}$.

**Why the formula is just $\sqrt{P \cdot \mathrm{norm}}$ and not the full GRF expression:** DiscoDJ's internal IC generation applies an inverse Laplacian kernel $\hat{\phi} = -\hat{\delta}/k^2$ followed by $\delta_\mathrm{ini} = \mathrm{IFFT}[-k^2 \cdot \hat{\phi}]$. The $-k^2$ and $-1/k^2$ cancel, leaving only $\hat{\xi} \cdot \sqrt{P \cdot \mathrm{norm}}$. At $a = 1$, the growth factor $D_+(1) = 1$ in DiscoDJ's normalization, so $\delta_\mathrm{ic} = \delta_\mathrm{ini}$ directly.

#### Stage 3: N-body Simulation

The initial density field is evolved to $z = 0$ using a particle-mesh (PM) N-body solver:

1. **2LPT initialization** ($n_\mathrm{order} = 2$): particles are displaced from the Lagrangian grid using second-order Lagrangian perturbation theory
2. **PM evolution**: the BullFrog stepper integrates from $a_\mathrm{ini} = 0.25$ ($z = 3$) to $a_\mathrm{end} = 1.0$ ($z = 0$) in a single time step on a $2\times$ upsampled force mesh ($N_\mathrm{PM} = 2 N_\mathrm{res}$)
3. **Mass assignment**: final particle positions are deposited onto the density grid using a CIC window ($w_\mathrm{order} = 2$) with anti-aliasing and deconvolution

The output is the nonlinear density contrast $\delta_\mathrm{fin}(\mathbf{x})$.

#### Stage 4: Power Spectrum Measurement

The binned power spectrum is measured from $\delta_\mathrm{fin}$ using logarithmic binning:

$$
P_\mathrm{NL}(k_a) = \frac{1}{N_\mathrm{modes}(k_a)} \sum_{|\mathbf{k}| \in \text{bin } a} |\hat{\delta}_\mathrm{fin}(\mathbf{k})|^2 \cdot \left(\frac{L}{N_\mathrm{res}}\right)^3
$$

### 4.3 Jacobian Computation

The Jacobian $J_{ai} = \partial P_\mathrm{NL}(k_a) / \partial \theta_i$ is computed using `jax.jacfwd` (forward-mode AD). Forward mode is natural here because the number of parameters ($N_\mathrm{params} \leq 5$) is small relative to the number of output bins ($n_\mathrm{bins} = 30$), so $N_\mathrm{params}$ forward passes suffice.

The full pipeline is treated as a single function $f: \mathbb{R}^{N_\mathrm{params}} \to \mathbb{R}^{n_\mathrm{bins}}$, with white noise as a non-differentiated auxiliary input:

```python
J = jax.jacfwd(pipeline_fn)(cosmo_fid, noise)  # shape (n_bins, N_params)
```

### 4.4 Seed Averaging

Because the power spectrum of a single realization is noisy (cosmic variance), the Jacobian is averaged over $N_\mathrm{seeds}$ independent white-noise realizations:

$$
\bar{J}_{ai} = \frac{1}{N_\mathrm{seeds}} \sum_{s=1}^{N_\mathrm{seeds}} J_{ai}^{(s)}
$$

Each seed uses `jax.random.PRNGKey(s)` to generate independent noise. The fiducial $P_\mathrm{NL}(k)$ is similarly averaged. Typical values: $N_\mathrm{seeds} = 10$.


---

## 5. Case 2: Joint Cosmological + Bias Parameters

### 5.1 Motivation

In practice, galaxies are biased tracers of the underlying matter field. The observed galaxy density field $\delta_g$ is not equal to the matter density $\delta_m$ but is related through a bias model. To forecast constraints from galaxy surveys, one must jointly constrain cosmological parameters and bias parameters.

### 5.2 Parameter Space

In the bias case, only 2 cosmological parameters are sampled (the others are fixed to Planck 2018):

| Parameter    | Symbol              | Fiducial | Prior range       |
|:-------------|:--------------------|:---------|:------------------|
| Total matter | $\Omega_m$          | 0.3175   | [0.10, 0.50]      |
| Amplitude    | $\sigma_8$          | 0.8340   | [0.60, 1.00]      |
| Linear bias  | $b_1$               | 1.0      | [-0.5, 2.5]       |
| Quadratic    | $b_2$               | 0.0      | [-4.0, 4.0]       |
| Tidal shear  | $b_{s^2}$           | 0.0      | [-4.0, 4.0]       |
| Laplacian    | $b_{\nabla^2}$      | 0.0      | [-4.0, 4.0]       |

Fixed: $\Omega_b = 0.049$, $h = 0.6711$, $n_s = 0.9624$.

The combined parameter vector is $\boldsymbol{\theta} = (\Omega_m, \sigma_8, b_1, b_2, b_{s^2}, b_{\nabla^2})$.

### 5.3 The Extended Pipeline

The biased-tracer pipeline extends Case 1 with a Lagrangian bias weighting step:

$$
\boldsymbol{\theta} \xrightarrow{\text{CLASS}} P_\mathrm{lin}(k) \xrightarrow{\text{IC gen}} \delta_\mathrm{ic} \xrightarrow{\text{N-body}} \mathbf{X}_\mathrm{fin} \xrightarrow[\text{+ bias weights}]{\text{scatter}} \delta_\mathrm{biased} \xrightarrow{\text{binning}} P_\mathrm{biased}(k)
$$

The key difference from Case 1: instead of depositing unit-weight particles to get $\delta_\mathrm{fin}$, each particle carries a bias weight $w(\mathbf{q})$ computed on the Lagrangian grid.

#### Steps:
1. **Stages 1-3** are identical to Case 1 (with only $\Omega_m, \sigma_8$ varying the cosmology)
2. **Stage 3b**: Compute Lagrangian bias weights $w(\mathbf{q})$ from $\delta_\mathrm{ic}$ and bias parameters (Section 6)
3. **Stage 3c**: Scatter bias-weighted particles onto the Eulerian mesh:

$$
n_\mathrm{biased}(\mathbf{x}) = \sum_p w(\mathbf{q}_p) \, W(\mathbf{x} - \mathbf{X}_p)
$$

where $W$ is the CIC window function and $\mathbf{X}_p$ is the final (Eulerian) position of particle $p$.

4. **Normalisation**: Convert to overdensity:

$$
\delta_\mathrm{biased}(\mathbf{x}) = \frac{n_\mathrm{biased}(\mathbf{x})}{\langle n_\mathrm{biased} \rangle} - 1
$$

5. **Stage 4**: Measure $P_\mathrm{biased}(k)$ from $\delta_\mathrm{biased}$ (same binning as Case 1)


---

## 6. The Lagrangian Bias Model

### 6.1 Second-Order Lagrangian Bias Expansion

Following Modi et al. (2020), the bias weight at Lagrangian position $\mathbf{q}$ is:

$$
\boxed{w(\mathbf{q}) = 1 + b_1 \, \delta(\mathbf{q}) + b_2 \left[\delta^2(\mathbf{q}) - \langle\delta^2\rangle\right] + b_{s^2} \left[s^2(\mathbf{q}) - \langle s^2 \rangle\right] + b_{\nabla^2} \, \nabla^2\delta(\mathbf{q})}$$

All fields are evaluated on the initial (Lagrangian) grid, i.e. $\delta = \delta_\mathrm{ic}$.

**Physical meaning of each term:**

- $b_1 \, \delta$: Linear bias --- galaxies form preferentially in overdense regions
- $b_2 \, (\delta^2 - \langle\delta^2\rangle)$: Quadratic density bias --- sensitivity to the square of the density, mean-subtracted to ensure $\langle w \rangle$ is unaffected
- $b_{s^2} \, (s^2 - \langle s^2 \rangle)$: Tidal shear bias --- sensitivity to the tidal environment (anisotropic collapse)
- $b_{\nabla^2} \, \nabla^2\delta$: Higher-derivative (Laplacian) bias --- scale-dependent correction, already mean-zero by construction ($\langle\nabla^2\delta\rangle = 0$ for a mean-zero field)

### 6.2 Tidal Tensor and Shear Squared

The tidal (shear) tensor is defined as the traceless part of the Hessian of the gravitational potential:

$$
T_{ij}(\mathbf{q}) = \left(\frac{\partial_i \partial_j}{\nabla^2} - \frac{1}{3}\delta_{ij}^{(\mathrm{K})}\right) \delta(\mathbf{q})
$$

In Fourier space, this is computed via the gravitational potential $\hat{\Phi}(\mathbf{k}) = -\hat{\delta}(\mathbf{k}) / k^2$:

$$
\hat{T}_{ij}(\mathbf{k}) = \frac{k_i \, k_j}{k^2} \, \hat{\delta}(\mathbf{k})
$$

The tidal shear squared is then:

$$
s^2(\mathbf{q}) = T_{ij} T_{ij} - \frac{1}{3} \delta^2 = \sum_{i \leq j} c_{ij} \, T_{ij}^2 - \frac{1}{3} \delta^2
$$

where $c_{ij} = 1$ for diagonal components and $c_{ij} = 2$ for off-diagonal components (accounting for symmetry $T_{ij} = T_{ji}$).

**Implementation** (`core/model/bias.py:_tidal_shear_sq`): The 6 independent components of $T_{ij}$ are computed in Fourier space using DiscoDJ's gradient kernels $\hat{g}_i = i k_i$ and inverse Laplacian kernel:

$$
\hat{T}_{ij}(\mathbf{k}) = \hat{g}_i \cdot \hat{g}_j \cdot \hat{\Phi}(\mathbf{k})
$$

Since $\hat{g}_i \hat{g}_j = (ik_i)(ik_j) = -k_i k_j$ (real-valued), the result is real and Hermitian-symmetric, as required.

### 6.3 Laplacian Term

The Laplacian of the density field in Fourier space:

$$
\widehat{\nabla^2\delta}(\mathbf{k}) = -k^2 \, \hat{\delta}(\mathbf{k})
$$

where $k^2 = k_x^2 + k_y^2 + k_z^2$. This is computed directly via `core/model/bias.py:_laplacian`.


---

## 7. Key Differences Between Case 1 and Case 2

### 7.1 Parameter Space

| Aspect              | Case 1 (matter)                   | Case 2 (biased tracer)                    |
|:--------------------|:----------------------------------|:------------------------------------------|
| Sampled params      | Up to 5 ($\Omega_m, \Omega_b, h, n_s, \sigma_8$) | 6 ($\Omega_m, \sigma_8, b_1, b_2, b_{s^2}, b_{\nabla^2}$) |
| Fixed params        | None (or a user-specified subset) | $\Omega_b, h, n_s$                        |
| Observable          | $P_\mathrm{NL}(k)$               | $P_\mathrm{biased}(k)$                    |
| Fisher matrix size  | Up to $5 \times 5$                | $6 \times 6$                              |

### 7.2 Jacobian Structure

In Case 1, the full Jacobian $\partial P_\mathrm{NL}(k) / \partial \theta_i$ has derivatives that flow through the entire pipeline (CLASS $\to$ ICs $\to$ N-body $\to$ P(k)).

In Case 2, the Jacobian has a block structure. Derivatives with respect to cosmological parameters ($\Omega_m, \sigma_8$) flow through the N-body pipeline, while derivatives with respect to bias parameters ($b_1, b_2, b_{s^2}, b_{\nabla^2}$) only affect the bias weighting step. Specifically:

- $\partial P_\mathrm{biased}/\partial \Omega_m$ and $\partial P_\mathrm{biased}/\partial \sigma_8$: depend on how $P_\mathrm{lin}(k)$ changes, which alters both $\delta_\mathrm{ic}$ (and hence the N-body evolution + bias weights) and the particle trajectories
- $\partial P_\mathrm{biased}/\partial b_i$: only enter through the bias weight formula $w(\mathbf{q})$; the N-body trajectories $\mathbf{X}_p$ are independent of the bias parameters

### 7.3 Degeneracies

Case 2 introduces new degeneracies between cosmological and bias parameters:

- **$\sigma_8$--$b_1$ degeneracy**: On large scales, $P_\mathrm{biased}(k) \approx b_1^2 \, P_\mathrm{NL}(k) \propto b_1^2 \, \sigma_8^2$. The Fisher matrix will reflect this near-degeneracy as a strong off-diagonal element.
- **$b_2$--$b_{s^2}$ degeneracy**: Both quadratic operators contribute similarly shaped corrections to the power spectrum, making them difficult to separate from $P(k)$ alone.
- **$b_{\nabla^2}$ scale dependence**: The Laplacian bias contributes a $k^2$-dependent correction, providing leverage at high $k$ but potentially degenerate with nonlinear evolution effects.


---

## 8. Implementation Details

### 8.1 White Noise Fixing

The white-noise field $\xi(\mathbf{x})$ is generated **outside** the differentiated function and passed in as an auxiliary argument. This ensures:

1. The pipeline is **deterministic** for a given noise realization (no stochastic operations inside the differentiated function)
2. JAX does not attempt to differentiate through the random number generator
3. Derivatives isolate the cosmology/bias dependence, not sampling noise

```python
noise = jax.random.normal(jax.random.PRNGKey(seed), shape=(res,) * dim)
J = jax.jacfwd(pipeline_fn)(cosmo_fid, noise)  # noise not differentiated
```

### 8.2 JVP Diagnostic

Before computing the full Jacobian, a diagnostic function (`_diagnose_tangent`) runs `jax.jvp` (Jacobian-vector product) through each pipeline stage individually. This pinpoints exactly where tangent propagation might break (produce NaN or zero tangents):

1. **Stage 1**: $\boldsymbol{\theta} \to P_\mathrm{lin}(k)$ --- tests CLASS differentiability
2. **Stage 2**: $\boldsymbol{\theta} \to \delta_\mathrm{ic}$ --- tests IC generation + FFTs
3. **Stage 3**: $\boldsymbol{\theta} \to P_\mathrm{NL}(k)$ --- tests the full pipeline including N-body

At each stage, the number of finite primal/tangent values and the tangent norm are printed.

### 8.3 Finite-Difference Fallback

If `jax.jacfwd` produces an entirely NaN Jacobian (e.g. due to non-differentiable operations in some DiscoDJ code path), the code falls back to **central finite differences**:

$$
\frac{\partial P(k_a)}{\partial \theta_i} \approx \frac{P(k_a; \theta_i + h_i) - P(k_a; \theta_i - h_i)}{2 h_i}
$$

Step sizes are calibrated per parameter:

| Parameter    | Step $h_i$ |
|:-------------|:-----------|
| $\Omega_m$   | 0.01       |
| $\Omega_b$   | 0.002      |
| $h$          | 0.02       |
| $n_s$        | 0.02       |
| $\sigma_8$   | 0.015      |

For the bias case, bias parameter steps are $h = 0.05$ ($b_1$) or $h = 0.1$ ($b_2, b_{s^2}, b_{\nabla^2}$).

The cost is $1 + 2 N_\mathrm{params}$ forward passes per seed (compared to $N_\mathrm{params}$ for `jacfwd`).

### 8.4 Bin Validation

Before computing the Fisher matrix, bins are filtered to remove invalid entries:

1. **Zero modes**: bins with $N_\mathrm{modes} = 0$ (no Fourier modes fall in this bin)
2. **Non-finite P(k)**: bins where the fiducial power spectrum is NaN or Inf
3. **Non-positive P(k)**: bins with $P(k) \leq 0$ (which would make $\sigma^2$ undefined)
4. **Non-finite Jacobian**: bins where any partial derivative is NaN

The Fisher matrix is computed only over the valid bins. The number of dropped bins is reported.

### 8.5 `requires_jacfwd=True`

All DiscoDJ instances in the Fisher pipeline are constructed with `requires_jacfwd=True`. This flag tells DiscoDJ to use JAX-compatible code paths internally (avoiding operations that would break tangent propagation), enabling automatic differentiation through the full N-body solver.

### 8.6 Box and Simulation Parameters

Default configuration used for the Fisher forecast:

- **Box**: $L = 1000 \; \mathrm{Mpc}/h$, $N_\mathrm{res} = 64$ grid cells per side, 3D
- **N-body**: BullFrog stepper, PM method, 1 time step from $z = 3$ to $z = 0$
- **Force mesh**: $N_\mathrm{PM} = 128$ ($2\times$ upsampled)
- **Mass assignment**: CIC ($w_\mathrm{order} = 2$), anti-aliased, deconvolved
- **ICs**: 2LPT initialization
- **Binning**: $n_\mathrm{bins} = 30$ logarithmic bins

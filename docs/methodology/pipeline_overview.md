# Pipeline Overview (for context hand-off)

This is a one-page summary of my master-thesis project so another assistant can
give me the right statistics / machine-learning background. It describes *what*
the pipeline does and *which methods* it uses — not the code details.

## The scientific goal

I do **cosmological parameter inference**. Given a simulated piece of the
universe, I want the posterior distribution over a handful of parameters:

- **Cosmological:** `Omega_m` (matter density), `sigma_8` (clustering amplitude).
- **Galaxy/halo bias:** `b1, b2, bs2, bn2` (how tracers trace the underlying
  matter density — a perturbative bias expansion).

So the target is a posterior `p(theta | data)` over ~2–6 parameters.

## The core method: Simulation-Based Inference (SBI)

The likelihood `p(data | theta)` is intractable, but I *can* simulate data for
any `theta`. So I use **neural posterior estimation (NPE / SBI)** via the
[`sbi`](https://github.com/sbi-dev/sbi) library:

1. Run many simulations, each with parameters `theta` drawn from a **prior**.
   Priors are mixed uniform / normal per parameter (`MultipleIndependent`).
2. Train a **normalizing flow** as a conditional density estimator
   `q(theta | x)` that approximates the true posterior.
3. Query the trained flow at the observed/fiducial data to get the posterior.

Flow architectures I use: **MAF** (Masked Autoregressive Flow), **NSF**
(Neural Spline Flow), and FrEIA-based INNs. The flow is conditioned on a learned
or fixed **embedding / summary statistic** of the data.

## Two parallel data representations

The project has two pipelines that infer the *same* parameters from the *same*
underlying simulations, but ingest the data differently:

### 1. Density-field pipeline (`den_pie/density/`)
Input is the **raw 3D density field** (a 3D grid). A neural **encoder**
compresses the field into a summary vector before the flow:

- Encoder: 3D CNN (`cnn3d`) or Swin Transformer (`swin3d`).
- Trained in **three stages**: (1) encoder alone, (2) freeze encoder + train
  flow, (3) joint end-to-end fine-tune.
- Hyperparameters tuned with **Optuna**.

This is the "learn the summary statistic from scratch" approach.

### 2. Spectra pipeline (`den_pie/spectra/`) — current active work
Instead of the raw field, the input is a **hand-crafted summary statistic**: the
standard cosmological N-point functions measured from each sim:

- **Power-spectrum multipoles** `P_0, P_2, P_4` (monopole/quadrupole/hexadecapole).
- **Bispectrum monopole** `B_0`.
- Measured on initial-conditions (`delta_ic`) and/or evolved (`delta_biased`) fields.
- Features are transformed for stability (`log` for `P_0`, `arcsinh` for the
  rest) and normalized.

The embedding before the flow is lightweight here: `none` (raw features), a
small **MLP**, or a **frozen PCA** projection fit on the training set
(dimensionality reduction).

This pipeline asks: how much cosmological information is in the summary
statistics alone, and does a learned field encoder beat them?

## Evaluation / diagnostics

Posteriors are validated with standard SBI calibration checks:

- **Corner plots** — marginal + joint posteriors vs. true values.
- **SBC** (Simulation-Based Calibration) — rank statistics; should be uniform
  if the posterior is well-calibrated.
- **TARP** (Tests of Accuracy with Random Points) — coverage test.

## Where statistics/ML background would help me

- Normalizing flows (MAF vs. NSF), conditional density estimation, NPE/SNPE.
- SBI calibration theory: SBC rank uniformity, TARP/expected coverage, what
  over/under-dispersed posteriors look like.
- PCA as a fixed embedding vs. a learned (MLP/CNN) embedding — bias/variance,
  information loss, when each wins.
- Priors and posterior interpretation for a small number of correlated params.
- Cosmology side: power spectrum multipoles, bispectrum, perturbative bias.

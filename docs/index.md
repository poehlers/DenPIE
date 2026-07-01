# den_pie

**Field-level forward modelling, Fisher forecasting, and simulation-based
inference (SBI) for cosmology — in one package.**

den_pie combines the two halves of a cosmological field-level analysis:

- a **differentiable JAX forward model** ({mod}`den_pie.forward`) mapping
  cosmological (and Lagrangian-bias) parameters to initial and evolved density
  fields and their summary statistics — power-spectrum multipoles
  $P_{0,2,4}(k)$ and the bispectrum monopole $B_0$ — built on
  DiscoDJ + BFast + Falcon;
- a **Fisher-forecast library** ({mod}`den_pie.fisher`) that differentiates that
  forward model to produce Gaussian parameter constraints; and
- a **PyTorch neural SBI** stack ({mod}`den_pie.density`, {mod}`den_pie.spectra`)
  that trains normalizing-flow posteriors on the same simulations — and can be
  overlaid directly against the Fisher forecast.

```{image} _static/forward_model.png
:alt: den_pie forward model + Fisher + SBI pipeline
:width: 640px
:align: center
```

## The pipeline at a glance

From a prior over parameters $\boldsymbol{\theta}$, DiscoDJ builds the linear
power spectrum $P_L(k)$ and an initial density field $\delta_{\rm ic}$; a
2LPT + particle-mesh solver evolves it; particles are then either scattered with
uniform weight (dark matter $\delta_{\rm dm}$) or reweighted by a Lagrangian
bias expansion $w(\boldsymbol{q})$ (biased tracer $\delta_{\rm g}$). The fields
feed two compression routes — a CNN/ViT field encoder and BFast summaries
$P_{0,2,4}(k) + B_0$ — which drive either a **Fisher forecast** or a **neural
SBI** posterior $p(\boldsymbol{\theta}\mid\boldsymbol{d})$. Because both routes
share the same simulations, summary format, and parameter conventions, the SBI
posterior and the Fisher ellipses can be plotted on the same corner.

## Two halves, one environment

The forward model + Fisher run on JAX/DiscoDJ/BFast; the SBI stack runs on
PyTorch/`sbi`. They coexist in a single environment (see
{doc}`installation`). Historically these lived in two repositories
(`den_pie` for SBI and `joint_fli_sbi` for the forward model + Fisher); this
package is their merger — see the `NOTICE` file for attribution.

```{toctree}
:maxdepth: 2
:caption: Guide

installation
forward_model
fisher
inference
comparison
cli
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
methodology/index
```

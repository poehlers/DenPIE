# SBI vs Fisher

Because the {doc}`neural posterior <inference>` and the
{doc}`Fisher forecast <fisher>` share the same simulations, summary statistics,
and parameter conventions, they can be drawn on a single corner — a direct check
of how close the (Gaussian) Fisher approximation is to the full SBI posterior.

```{image} _static/sbi_vs_fisher_example.png
:alt: SBI posterior overlaid on a Fisher forecast
:width: 640px
:align: center
```

*(Illustrative overlay: Fisher 68/95% ellipses and Gaussian marginals in blue,
an SBI posterior sample set in red, fiducial values in orange.)*

## Running

```bash
python -m den_pie fisher-compare  <fisher_run>  <sbi_run>  [--out corner.png]
```

- `<fisher_run>` — a Fisher run directory (or a `fisher_results.npz` file).
- `<sbi_run>` — a den_pie SBI run directory (the noise-reduced
  `fiducial_mean.npz` posterior is preferred automatically) or a specific
  posterior `.npz`.

{func}`den_pie.fisher.compare.compare_corner` loads the Fisher covariance and
fiducial values, loads the SBI samples, and draws Fisher ellipses + Gaussian
marginals against the SBI histogram marginals + smoothed 2-D contours. LaTeX
labels and prior bounds are reused from `den_pie.density.priors.DENSITY_PARAMS`
so the figure matches den_pie's own corner plots.

The module is deliberately lightweight — numpy + matplotlib + scipy, **no
JAX** — so the comparison runs wherever an SBI run produced samples.

## Parameter-name reconciliation

The Fisher half names the amplitude parameter `sigma8`; the SBI half and the
dataset registry name it `sigma_8`. All other names already agree. The shim in
{mod}`den_pie.util.params` reconciles them at this boundary (and only here):

```python
from den_pie.util.params import to_den_pie, align_to
to_den_pie("sigma8")                              # -> "sigma_8"
align_to(["Omega_m", "sigma8"], ["Omega_m", "sigma_8"])  # -> [0, 1]
```

Pass `--sbi-names` to `fisher-compare` if your SBI sample columns are in a
different order than the Fisher parameters; otherwise identical ordering is
assumed (true for the registry datasets).

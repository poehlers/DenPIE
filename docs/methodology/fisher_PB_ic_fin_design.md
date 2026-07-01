# Joint P+B Fisher: IC, FIN, and combined-field design notes

Short companion doc for the upcoming change that extends
`run_fisher_forecast_bias_PB` to support `--field ic`, `--field fin`,
and a new `--field both` mode.

## What changes

| Mode | Field(s) entering the data vector | N-body? | Output dir |
|------|------------------------------------|---------|------------|
| `--field fin` *(existing)* | `δ_biased` (Eulerian, post-N-body) | yes | `outputs/fisher_bias_P_B` |
| `--field ic` *(new)* | `δ_L = w(q) − 1` (biased Lagrangian) | no | `outputs/fisher_bias_P_B_ic` |
| `--field both` *(new)* | `concat([P_ic, B_ic, P_fin, B_fin])` | yes (once) | `outputs/fisher_bias_P_B_both` |

All three modes use the same BFast precompute (`bin_edges`, `B_info`,
`B_norm`) and the same 6×6 parameter space `(Ω_m, σ8, b1, b2, bs2, bn2)`.
The `both` mode runs the bias N-body **once** per pipeline call and
emits two summaries from it (one before scattering, one after), so its
wall-clock is roughly `fin + 2× BFast` rather than `fin + ic`.

## Why `--field ic` = biased Lagrangian, not raw `δ_ic`

The `Analyse.ipynb` notebook used as a reference defines `pre` as the
**raw, unbiased** Quijote IC snapshot. We deliberately **do not** copy
that convention here:

- The existing P-only `--field ic` already means the **biased** field
  `δ_L = w(q) − 1`. Reusing the same name with a different meaning
  would silently break consistency with `outputs/fisher_bias/` and any
  downstream comparison.
- Bias parameters `b1, b2, bs2, bn2` enter `w(q)` but not raw `δ_ic`.
  An "unbiased IC" Fisher block would have **zero Jacobian** for the
  four bias parameters — i.e. it would only constrain `(Ω_m, σ8)` and
  contribute nothing to bias-parameter constraints. That's not the
  forecast question we're asking ("how much extra info does the IC
  field add to the bias forecast?").
- The notebook's `pre` is the RAW Quijote snapshot at `z=0` with shot
  noise subtracted, not a model field at all — its analogue in our
  forward model would be the unbiased Lagrangian density. Skipping
  bias on the IC side would also break symmetry with `--field fin`,
  where the bias weights are applied.

So `--field ic` keeps its existing semantics: BFast P+B on
`δ_L = w(q) − 1`, with all six parameters live.

## Why deliver all three modes (separate + joint)

Asked to choose between separate-only, joint-only, or both. Picked
both because each answers a distinct question:

- **`--field ic` alone** — quantifies the constraint power of the
  biased Lagrangian field by itself, on the same six parameters. Cheap
  (no N-body), so it's the natural "what does the IC bring on its own?"
  baseline.
- **`--field fin` alone** — already exists; the existing
  Eulerian-only joint P+B baseline.
- **`--field both`** — the actual scientific quantity: how tight do the
  bias parameters get when the *full* observation (Lagrangian + Eulerian
  P and B) is exploited jointly? The `both` Fisher is what feeds into
  any forecast comparison vs. the SBI joint posterior.

Producing only the joint matrix would hide where the constraining
power comes from; producing only the per-field matrices would skip the
question we actually want answered. The marginal cost of all three is
small because BFast precompute is shared and the IC pipeline is
N-body-free.

## Block-diagonal IC↔FIN covariance: known optimism

In `--field both` we set the IC↔FIN cross-covariance block to zero.
This is **not** physically correct — `δ_biased` is generated from
`δ_ic` by the N-body operator, so the two fields are correlated.
Keeping the cross block forces a sample-based covariance from many
sims with Hartlap correction (à la `Analyse.ipynb`), which is out of
scope for this change.

Practical consequence: the marginal sigmas reported by `--field both`
will be **tighter than the truth** (correlated data treated as
independent → over-counted information). This is the *same* class of
approximation already in use for the P↔B cross-block within a single
field, just lifted one level up. The forecast is still useful for
relative comparisons (e.g. "does b2 gain more from IC or from FIN?"),
but the absolute marginals should be quoted with the caveat.

The driver will print a one-line warning to console when `--field
both` is selected so this isn't forgotten when reading numbers off the
output.

## Expected ordering of constraints

For each bias parameter `θ ∈ {b1, b2, bs2, bn2}` we expect the
marginal `σ(θ)` to follow:

```
σ_P_only(fin)  ≥  σ_P+B(fin)  ≥  σ_P+B(both)   (and similarly for ic)
```

The bispectrum-sensitive parameters (`b2`, `bs2`, `bn2`) should tighten
most going from P-only to P+B; cosmology parameters (`Ω_m`, `σ8`)
should tighten less. If a `both` marginal is *not* tighter than the
matching `fin` marginal, the IC pipeline or covariance wiring is
broken — that's the smoke-test failure mode to watch for.

## Files touched

- `core/fisher.py` — two new pipelines (`_differentiable_pipeline_ic_bias_PB`,
  `_differentiable_pipeline_both_bias_PB`); `run_fisher_forecast_bias_PB`
  gains a `field` parameter and dispatches pipeline + covariance + default
  output dir.
- `core/run_fisher.py` — `--field` choices extended to `{fin, ic, both}`;
  validation block relaxed; `field` threaded through.
- `my_py_gpu_job_fisher_bias_PB_ic.sh`, `my_py_gpu_job_fisher_bias_PB_both.sh`
  — new SLURM jobs.
- `my_py_gpu_job_fisher_bias_PB.sh` — unchanged (still `--field fin`).

Existing baselines `outputs/fisher_bias/` (P-only) and
`outputs/fisher_bias_P_B/` (FIN P+B) are not touched and remain the
comparison reference.

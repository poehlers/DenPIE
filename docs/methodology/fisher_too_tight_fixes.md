# Why the Fisher constraints were too tight (and what we changed)

This note documents a critical review of the joint P+B (+RSD multipoles +
shot-noise) Fisher pipeline in `core/fisher.py` and the two changes made
in response. References for the comparison: arXiv:2305.07018 (method),
arXiv:2504.20130 (parameter setup), Grieb+2016 MNRAS 457, 1577.

## What we changed

### 1. `n_seeds` default bumped from 3/10 → 20

**Files.** `core/run_fisher.py:22`,
`my_py_gpu_job_fisher_bias_PB_rsd_SN.sh:33`,
`my_py_gpu_job_fisher_bias_PB_rsd_SN_both.sh:33`.

**Why.** The Jacobian is averaged over `n_seeds` independent white-noise
realisations, `J̄ = (1/N) Σ J_s`. With finite N the average still carries
sample variance `Var(J̄) ∝ 1/N`. Because the Fisher

```
F = J̄ᵀ C⁻¹ J̄
```

is a *positive-definite quadratic* in `J̄`, taking the expectation gives

```
E[F] = F_true + Tr(C⁻¹ · Var(J̄))
```

i.e. the Fisher is biased **upward** and the marginal sigmas downward
(*artificially tight*) by an amount that scales like `1/n_seeds`. At
`n_seeds=3` (the previous joint-PB default) the per-bin Jacobian noise is
O(20–50 %) of the signal, so the bias was a real fraction of the reported
sigmas. The reference fNL paper averaged derivatives over **12,500**
Quijote sims — three is not in the right ballpark.

`n_seeds=20` brings the per-parameter Fisher-noise bias under ~5 % per
component, while doubling to 40 should now change marginals by <5 %
(use that as the empirical convergence check).

### 2. P-multipole covariance: diagonal Grieb-limit → full Grieb+2016 3×3 block

**File.** `core/fisher.py`: new function `multipole_covariance_grieb`
(near `bispectrum_covariance`), `fisher_bias_PB` rewritten to build
Fisher block-by-block.

**Before.** For each k bin the code used the *weak-anisotropy diagonal*

```
σ²(P_l)(k) ≈ (2l+1) · 2 · P_0(k)² / N_modes(k)        [diagonal in l, no cross]
```

This is the limit `P_2 = P_4 = 0` of the Grieb formula and **ignores all
P_l-P_l' cross-covariance**.

**After.** The full Grieb+2016 (MNRAS 457, 1577, eq A.6) formula

```
Cov(P_l(k), P_l'(k)) = (2 / N_modes(k)) · (2l+1)(2l'+1)/2
                       · ∫_{-1}^{1} L_l(μ) L_l'(μ) · [P(k,μ)]² dμ
```

with `P(k,μ) = P_0(k) + P_2(k) L_2(μ) + P_4(k) L_4(μ)`, computed via
64-point Gauss-Legendre quadrature in JAX. The result is a `(n_k, n_l,
n_l)` tensor with one dense 3×3 block per k; the Fisher contraction
becomes

```
F^P_ij = Σ_k Σ_{a,b}  J[a,k,i] · [C(k)⁻¹]_{a,b} · J[b,k,j]
```

implemented as a single `jnp.einsum('aki,kab,bkj->ij', J, C_inv, J)`.

**Why this was the dominant tightness bias** (the in-tree numerical
check confirms): for a Kaiser-strength RSD field with `P_2/P_0 ≈ 0.3`
and `P_4/P_0 ≈ 0.05`,

| Quantity | Old diagonal | Grieb | Ratio |
| --- | --- | --- | --- |
| `Var(P_0)` | `2 P_0²/N` | `2.04 P_0²/N` | ×1.02 |
| `Var(P_2)` | `10 P_0²/N` | `12.4 P_0²/N` | ×1.24 |
| `Cov(P_0, P_2)` | **0** | `1.27 P_0²/N` | — |

The previous block-diagonal recipe was both *underestimating* the
diagonal multipole variances and *ignoring a ~25 % correlation* between
P_0 and P_2 errors. Both effects double-count multipole information →
artificially tight σ(Ω_m), σ(σ_8).

The Grieb path also covers the monopole-only case (`use_multipoles=False`):
the formula reduces to `2 P_0²/N_modes` for `l=l'=0`, so the same code
path handles both modes without an `if`.

### Save outputs

`fisher_results.npz` now additionally contains, per field block:

- `cov_P_grieb_blocks`: shape `(n_blocks, n_k, n_l, n_l)`
- `valid_k_P_blocks`: shape `(n_blocks, n_k)` (k bins where *all* used
  multipoles are valid)
- `multipole_ls`: which `l` values are in each block

The legacy `sigma_sq`, `sigma_sq_P`, `sigma_sq_B` keys still hold the
**diagnostic** diagonal entries (they are no longer the values actually
used to build the Fisher when `use_multipoles=True`).

## What we *verified* and left as-is

These were on the suspect list before reading the upstream code; both
turned out to be **correct**.

### Bispectrum variance prefactor (`bispectrum_covariance`, fisher.py:1656)

The code uses

```
σ²(B)(k₁,k₂,k₃) = s · (2π)⁶ / V_box · P(k₁) P(k₂) P(k₃) / V_T,
V_T = 8π² · k₁ k₂ k₃ · dk₁ dk₂ dk₃,  s ∈ {6, 2, 1}.
```

This matches BFast's bispectrum normalisation: tracing
`flivenv/lib/python3.13/site-packages/BFast/core/bispectrum.py`, `Bk` is
the standard `<δ̃_c δ̃_c δ̃_c> = V · B` bispectrum in `(Mpc/h)⁶`. For
that normalisation the Gaussian variance of the bin-averaged estimator is

```
Var(B) = s · V · P³ / N_T,   N_T = V² · V_B / (2π)⁶,
       = s · (2π)⁶ / V_box · P³ / V_B   ✓
```

— exactly the code. The `V_f² · V = (2π)⁶ / V_box` is one factor of `V_f`
*from the estimator definition* and one *from the Gaussian Wick
contraction*; the (2π)³-vs-(2π)⁶ confusion that motivated the original
worry was a misread of the Sefusatti formula.

### IC normalisation (`fisher.py:282` comment, DiscoDJ)

`delta_ic` is generated as the **z=0** linear field, then passed to
`with_external_ics(delta=...)`. That helper computes
`fphi = rfft(get_phi_from_delta(delta))` (`disco_dj.py:540`) and stores
it; the LPT then rescales internally when evaluating at `a_ini`. This
matches the internal `with_ics → generate_grf` convention. There is
**no** D(z=0)²/D(z_init)² over-amplification, so the IC has the right
linear amplitude and the N-body starts in the right regime.

## What we *did not* fix but flagged

These are documented in the in-tree review plan and worth tracking; they
do not bias toward tighter constraints in obvious ways, but they make
the forecast either *physically wrong* or *unrealistically optimistic*:

- **P-B cross covariance set to zero** (`fisher.py:2659-2664`). Real
  non-Gaussian Cov(P, B) at trans-nonlinear scales is 20–40 % of the
  diagonal in Quijote N-body covariances (Hahn+ 2020). Block-diagonal
  recipe is documented as optimistic.

- **IC-FIN block-diagonal covariance** (`fisher.py:2785-2787`). For
  `--field both` IC and FIN summaries share *the same realisation* of
  the initial conditions, so Cov(IC, FIN) is large; setting it to zero
  double-counts shared information.

- **`stop_gradient(cosmo)` through the N-body** (`fisher.py:300,431,544,
  644,988,1099`). For σ_8 this is exact; for Ω_m and h the missing
  dD/dθ piece *typically reduces* |dP/dθ| → looser constraints, so
  this is not a tightness bug but is a correctness bug. Cross-check
  jacrev vs central-difference derivatives at fid.

- **`n_steps=1` PM step** (`core/model/model.py:74`) and **no FoG
  damping** in the RSD pipeline (`fisher.py:600-689`). Together these
  give an unrealistically clean Kaiser quadrupole at small scales,
  inflating ∂P_2/∂θ. Add a Lorentzian FoG with a marginalised σ_v.

- **k_max = π/cellsize (Nyquist)** (`fisher.py:110,1606`). Going right
  up to Nyquist after CIC+deconvolve over-weights unphysical small
  scales. Typical P+B forecasts cut at k_max ≈ 0.2–0.3 h/Mpc.

## Suggested verification protocol

1. Re-run `my_py_gpu_job_fisher_bias_PB_rsd_SN.sh` with the new defaults
   (`n_seeds=20`, Grieb covariance). Compare σ(Ω_m), σ(σ_8) against the
   old run.
2. Sweep `n_seeds ∈ {10, 20, 40}` and check the marginal sigmas
   plateau within ~5 % from 20 → 40.
3. Empirically validate the Grieb multipole cov: generate 100+ noise
   seeds at fid cosmo, measure Cov(P_l(k), P_l'(k)), compare to
   `cov_P_grieb_blocks` saved in the npz. The off-diagonal entries
   should agree.
4. Cross-check jacrev vs central-finite-difference derivative on
   `--field fin --pk-multipoles`. Discrepancy at high k for Ω_m is the
   stop_gradient bias.

## Post-fix measurement (run 22921318, May 19 2026)

Output: `outputs/fisher_bias_P_B_rsd_both_SN_20260519_182011/`.
Config: `res=64`, `boxsize=1000 Mpc/h`, `field=both`, multipoles + RSD,
shot noise at `n_g=1e-3`, `n_seeds=20`, Grieb cov.

```
                                σ(data only)   σ(data + N(0,1/σ²_prior))
  Omega_m  (fid 0.3175)             0.00457        0.00433
  sigma8   (fid 0.8340)             0.00680        0.00671
  b1       (fid 1.0000)             0.01312        0.01232
  b2       (fid 0.0000)             0.00598        0.00598
  bs2      (fid 0.0000)             0.02430        0.02428
  bn2      (fid 0.0000)             1.05016        0.92924
```

### How much each fix actually moved the needle

Both fixes can be isolated from the *post*-fix npz by replaying the saved
`jacobians_per_seed` and recomputing the Fisher under different choices.

**(A3) n_seeds sweep** (`core/verification/plot_nseeds_sweep.py`, Grieb
cov held fixed):

```
  n_seeds   σ(Ω_m) (D+P)   σ(σ_8)
  1         0.00357        0.00598
  3         0.00408        0.00643
  5         0.00424        0.00662
  10        0.00422        0.00663
  15        0.00432        0.00670
  20        0.00433        0.00671
```

σ(Ω_m) grows **+21 %** from 1 → 20 seeds (Jensen-style upward Fisher
bias). The legacy `n_seeds=3` default sits **5.9 %** below the converged
σ(Ω_m); the plateau is reached by n=15 (n=15 → 20 changes σ by <1 %).

So **A3 was real but smaller than initially estimated** (a few %, not
tens of %), and at the old `n_seeds=3` default it was already partially
self-averaged.

**(A1) Diagonal vs full Grieb covariance** at n_seeds=20:

```
                σ_diag (D+P)  σ_Grieb (D+P)    Δ
  Omega_m       0.00434       0.00433        -0.19 %
  sigma8        0.00670       0.00671        +0.07 %
  b1            0.01232       0.01232        +0.06 %
  bn2           0.93064       0.92936        -0.14 %
```

The diagonal vs Grieb cov makes **<0.2 %** difference. The earlier
estimate of "1.5–3× tightening" was based on a P-only thought
experiment; once the bispectrum is in the data vector it dominates the
Fisher (see breakdown below), so changes to the P-multipole covariance
are washed out.

### Why the Grieb fix is theoretically right but numerically inert here

P-only vs B-only Fisher (post-fix, n_seeds=20):

```
            σ (P-only, D+P)   σ (B-only, D+P)   σ (P+B, D+P)
  Ω_m       0.00911           0.00536           0.00433
  σ_8       0.02274           0.02335           0.00671
  b1        0.02623           0.06982           0.01232
  b2        0.1988            0.00614           0.00598
  bs2       1.06              0.02520           0.02428
```

The bispectrum gives 2× tighter σ(Ω_m) than P-multipoles, ~30× tighter
σ(b_s2), ~30× tighter σ(b_2). The joint P+B σ tracks the bispectrum
column on most parameters. The multipole-cov fix can only redistribute
information *inside* the (small) P contribution, and that's mostly in
the IC block where RSD is not applied anyway:

```
  IC block (no RSD):  P_2/P_0 ∈ [-0.05, +0.05]  (noise-level)
  FIN block (Kaiser): P_2/P_0 ∈ [+0.26, +0.35]  (Kaiser-strength)
```

In the IC block the P_0–P_2 cross-cov term goes ~as `(P_2/P_0)² P_0²/N`
≈ 0.25 % of the diagonal — invisible. In the FIN block it's ~10 %, but
the FIN block contributes a small fraction of the total Fisher.

The Grieb implementation is still the **right** formula to ship — it
costs nothing and removes a published-paper-comparable shortcut — but
it should not be marketed as the fix for the "too tight" symptom.

### So what *is* still making the constraints tight?

After A3 + A1, the σ are unchanged at the digit (relative to the
already-running config). The likely real explanations for the tight
constraints are now the **un-fixed** items, in rough order of
suspected impact:

1. **No FoG damping + n_steps=1** (B2, B3). Together these produce an
   unrealistically clean Kaiser P_2 at small scales, inflating
   `∂P_2/∂(Ω_m, σ_8)`. The reference paper (2305.07018) stays in real
   space precisely to avoid having to model FoG.
2. **k_max = π/cellsize (Nyquist)** (B4). 10 P-bins running to Nyquist
   on a `res=64` grid is k_max ≈ 0.20 h/Mpc, which is actually within
   sensible DESI-style EFT cuts — *probably not* the main culprit, but
   verify.
3. **P-B and IC-FIN cross-covariance set to zero** (A2). Both biases
   are *optimistic*: real Cov(P,B) is 20–40 % of the diagonal at
   trans-nonlinear scales (Hahn+ 2020); Cov(IC, FIN) is large because
   they share the same noise realisation. Quantifying the impact
   requires an N-body covariance ensemble.
4. **`stop_gradient(cosmo)`** through the N-body (B1). Bias is in the
   *opposite* direction (looser constraints) for Ω_m, so this isn't a
   tightness bug — but it *is* a correctness bug; check via the
   jacrev-vs-FD plot.
5. **Sample-variance limit from a single `res=64` box.** `(Gpc/h)³` is
   a respectable volume, but the number of independent (k, μ) modes at
   `res=64` is small enough that the Fisher only sees a few hundred
   effective data points. Bumping to `res=128` halves the per-mode
   variance and doubles `k_max` — likely the single biggest realistic
   action.

### Recommended next step

Submit the two GPU verification jobs to nail down (4):

```
sbatch my_py_gpu_job_verify_grieb_cov.sh        # A1 empirical check (B3+B4 indirect)
sbatch my_py_gpu_job_verify_jacrev_vs_fd.sh     # B1 check (Ω_m derivative bias)
```

If both pass, the next coding change to consider is a Lorentzian FoG
damping term with a marginalised σ_v in `_apply_rsd_*` (B3 fix), and
optionally bumping `n_steps` to 5–10 to get a less idealised δ_fin.
The Quijote-reference comparison (target: σ from 2305.07018) will only
be meaningful once those two are in.

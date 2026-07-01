# Handling the power-spectrum–bispectrum (P–B) cross covariance

For the joint data vector `d = [P_0, P_2, P_4, B_0]`, the covariance has an
off-diagonal block coupling the power spectrum to the bispectrum:

```
        [  C_PP (Grieb)      C_PB         ]
C   =   [                                 ]
        [  C_PB^T            C_BB (Scocc.) ]
```

## Where we stand

| cov_type | C_PB treatment |
|---|---|
| `diag` | **C_PB = 0** (Gaussian/Wick: the connected 5-point term vanishes at leading order). |
| `euclid_cov` | **np.cov** of the whole vector — captures C_PB, but the whole matrix is sample-noisy and pays a large Hartlap penalty. |
| `euclid_cov_no_pb_cross` | **np.cov** P-P and B-B blocks (identical to `euclid_cov`) but **C_PB ≡ 0** — the clean on/off toggle used to *measure* the cross effect (see "Chosen study" below). |
| `block_with_pb_cross_empirical` | analytic `C_PP` (Grieb) + analytic diagonal `C_BB` (Scoccimarro) + **empirical** `C_PB` from `empirical_pb_cross_block`. |

**The problem with the empirical cross:** `C_PB` is a large `(n_P × n_tri)` block
(e.g. ~141×330) whose *true* signal is small and sparse, so almost every entry is
pure estimator noise. That noise (i) inflates the Hartlap penalty and (ii) can
de-condition `C⁻¹`. The options below are about taming it.

> ### Chosen study: empirical cross **on vs off** (`euclid_cov_no_pb_cross`)
> Rather than build a denoised/analytic cross, we measure how much the cross
> matters with a clean apples-to-apples toggle: take the **full empirical**
> covariance and zero **only** the P–B off-diagonal block, keeping the empirical
> P-P and B-B blocks **identical** to `euclid_cov`. Then
> `euclid_cov` (cross **on**) vs `euclid_cov_no_pb_cross` (cross **off**) isolates
> the P–B cross with nothing else changed. This is the Flöss-style empirical
> setup (`np.cov(PB.T)`), implemented in `core/fisher_2param.py` and run on the
> **P024+B0** data vector (`fisher_2param_P024_B0_emp` vs
> `fisher_2param_P024_B0_emp_nocross`). See the Findings section.
>
> **Note — why not the analytic Sefusatti cross (Option 2):** its leading
> Eq. (20) term couples **only the monopole P₀ to B₀**
> (`2k_f³/V_P(k_i)·P(k_i)·B(△)` on matching legs); it does **not** capture
> P₂×B₀ or P₄×B₀. Since our data vector is the full RSD multipole set, that
> leading term is too partial to stand in for the cross, so we did not implement
> it and instead study the cross empirically (above).

---

## Option 1 — Exploit the sparsity structure (cheap, do this first)

The leading non-vanishing P–B cross is a connected 5-point term ∝ the fiducial
bispectrum, and it is **non-zero only when the power-spectrum bin `k_i` coincides
with one of the triangle legs** `{k_1, k_2, k_3}`. Every other entry of `C_PB`
*should be exactly zero* and is currently just Monte-Carlo noise.

**Action:** mask the empirical estimate to that support — zero every entry where
`k_i ∉ {k_1, k_2, k_3}` — before inverting. Denoises the block dramatically and
shrinks the effective dimension the Hartlap factor sees. Strictly improves the
existing `block_with_pb_cross_empirical`.

- **Cost:** trivial (a boolean mask).
- **Bias:** none beyond what's already assumed (it only removes noise from
  entries that are zero in theory).

## Option 2 — Analytic cross from Sefusatti et al. 2006 (no sims, fully consistent)

**Sefusatti, Crocce, Pueblas & Scoccimarro 2006** (PRD 74, 023522,
astro-ph/0604505) derive the *joint* P+B covariance, **including the analytic
P–B cross block** — the abstract explicitly notes "covariance properties of the
power spectrum and bispectrum including the effects of beat coupling that lead to
interesting cross-correlations." So the closed form we want is already in the
literature. Its structure is:

```
Cov[P(k_i), B(k_1,k_2,k_3)]  =
      (1/N(k_i)) * Σ_{a∈{1,2,3}} δ_{k_i,k_a} * [prefactor] * P(k_i) * B(k_1,k_2,k_3)   # leading, ∝ bispectrum
    + trispectrum term  T(k_i,-k_i,k_1,k_2,k_3)                                          # higher-order
    + beat-coupling (finite-volume / super-survey) term                                 # Sefusatti+2006
```

i.e. the **leading term is ∝ the fiducial bispectrum** and is switched on only
when the P-bin `k_i` coincides with one of the triangle legs `{k_1,k_2,k_3}`
(same sparsity as Option 1), plus a trispectrum piece and the beat-coupling
cross-correlation. Keeping just the leading (∝ B) term already gives a covariance
that is 100% analytic (no Hartlap, no sims) yet captures the dominant non-Gaussian
cross — a clean middle ground between `diag` and the empirical block, and it lets
the thesis show the progression **zero cross → analytic cross → empirical cross**
and demonstrate convergence. (Oddo et al. 2021 / Gualdi et al. 2021 give the same
leading term in an explicit estimator-ready form.)

- **Cost:** one new helper, no sims.
- **Bias:** if only the leading ∝B term is kept, misses the trispectrum +
  beat-coupling pieces (which Sefusatti+2006 also provide if needed).

## Option 3 — Shrinkage estimator (if keeping sims)

Combine the noisy empirical block with a low-variance target:

```
C_PB_shrink = α · C_PB_target + (1 − α) · C_PB_empirical
```

with `C_PB_target` = either `0` or the analytic form of Option 2, and `α` from
Ledoit–Wolf (or a quick FoM-stability scan). Trades a little bias for a large
variance reduction and a much milder Hartlap correction.

- **Cost:** moderate (shrinkage coefficient).
- **Bias:** tunable via α; fallback if the analytic cross under-captures the
  measured one.

---

## Recommendation

1. **Do Option 1 unconditionally** — one-line mask on the existing
   `empirical_pb_cross_block` output; strictly improves the runs already
   submitted (`*_blockcross`).
2. **Add Option 2 as a new analytic `cov_type`** (e.g. `block_pb_analytic`) so
   the thesis covers *zero → analytic → empirical* cross and shows convergence.
3. **Keep Option 3 in reserve** if the analytic cross visibly under-predicts the
   empirical one.

## Where each slots into the code

- **Option 1:** in `core/fisher_2param.py`, in the
  `block_with_pb_cross_empirical` branch where the dense block covariance is
  assembled — apply the leg-matching mask to `cov_pb` before placing it in the
  off-diagonal block. (Same idea reusable in
  `fisher.py:empirical_pb_cross_block`.)
- **Option 2:** new helper `bispectrum_pk_cross_covariance` next to
  `bispectrum_covariance` in `core/fisher.py`, returning the analytic
  `(n_P × n_tri)` cross; wire it into a new `cov_type` in `fisher_2param.py`.
- **Option 3:** a thin wrapper combining the Option-2 target with the empirical
  block via a shrinkage coefficient.

See also [`fisher_2param_cov_overview.md`](fisher_2param_cov_overview.md) for the
full covariance/run map.

---

## Findings (P024+B0, res=128, shot noise)

Comparison of the (Ω_m, σ_8) Fisher constraint with the empirical P–B cross
**on** vs **off**, plus the analytic `diag` vs `blockcross` pair for reference.

| Run (`cov_type`) | C_PB | σ(Ω_m) | σ(σ_8) | corr | FoM |
|---|---|---|---|---|---|
| `fisher_2param_P024_B0_emp` (`euclid_cov`) | empirical (on) | _pending_ | _pending_ | _pending_ | _pending_ |
| `fisher_2param_P024_B0_emp_nocross` (`euclid_cov_no_pb_cross`) | **0 (off)** | _pending_ | _pending_ | _pending_ | _pending_ |
| `fisher_2param_P024_B0_gauss` (`diag`) | 0 (analytic blocks) | _pending_ | _pending_ | _pending_ | _pending_ |
| `fisher_2param_P024_B0_blockcross` | empirical cross, analytic blocks | _pending_ | _pending_ | _pending_ | _pending_ |

_To be filled once `fisher_2param_P024_B0_emp` (job 23926064) and
`fisher_2param_P024_B0_emp_nocross` (job 23930520) finish; the on-vs-off FoM ratio
is the headline number for "how much the P–B cross matters."_

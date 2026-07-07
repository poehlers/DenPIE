# 2-parameter (Ω_m, σ_8) Fisher + SBI: covariance overview

Final-field (`fli_2param`) observable, **res=128**, **shot noise on** (`n_g_density=1e-3`),
redshift space (los=2). Fixed: Ω_b, h, n_s (Planck-2018). Code: `core/fisher_2param.py`
(reuses primitives from `core/fisher.py`).

---

## 1. The covariance "calculations" — which name is which

| Your name | What it actually is | Function in code | Applies to | Sample-based? |
|---|---|---|---|---|
| **Grieb** | Grieb+2016 analytic **Gaussian** covariance of the redshift-space power-spectrum **multipoles** P₀/P₂/P₄ — includes the P_ℓ–P_ℓ' cross terms (per-k 3×3 block). | `multipole_covariance_grieb` | P-block (P₀,P₂,P₄) | No (analytic) |
| **Floss** | The **bispectrum** side. B₀ is *measured* with BFast (T. Floss, github.com/tsfloss/BFast); its **analytic covariance** is the diagonal **Scoccimarro-2000 / Sefusatti-2006** Gaussian form σ²(B) ∝ P(k₁)P(k₂)P(k₃)/V_T. | `bispectrum_covariance` | B-block (B₀) | No (analytic) |
| **np.cov** | **Empirical** sample covariance over many fiducial seeds (centred data matrix → sample cov), made unbiased on inversion with the **Hartlap** factor. | `compute_empirical_covariance` | whole data vector | Yes (needs n_cov_seeds > n_data+4) |
| _empirical cross_ | Same sample-covariance idea but **only the P–B off-diagonal block** is estimated from seeds; the P–P and B–B blocks stay analytic (Grieb / Scoccimarro). | `empirical_pb_cross_block` | P↔B cross only | Yes |

> ⚠️ **"Franco" is *not* a covariance.** "Franco-Abellan 2024" = Franco Abellán et al.
> 2024, *"Fast likelihood-free inference in the LSS Stage IV era"*
> (**arXiv:2403.14750**) — a **simulation-based-inference (MNRE)** paper; it does
> **not** define or estimate a data covariance. In this codebase it is referenced
> only for the **finite-difference step convention** for the derivatives
> (`_resolve_fd_steps`, relative FD step `eps=1e-2`), i.e. how the *Jacobian* ∂d/∂θ
> is taken — not how the data covariance C is built. And the 2-param runs compute
> the Jacobian by **autodiff (`jacrev`)**, so the Franco-Abellan step rule does not
> even enter here. (Note: the SBI side of this project uses den_pie's NPE+MAF flow,
> a different method from 2403.14750's MNRE.)

---

## 2. The three `cov_type` switches (combinations of the above)

The Fisher matrix is always `F = Jᵀ C⁻¹ J + F_prior`. Only **C** changes:

| `cov_type` (label) | P–P block | B–B block | P–B cross | Hartlap? |
|---|---|---|---|---|
| **`diag`** ("gauss") | **Grieb** | **Floss/Scoccimarro** (diag) | 0 | no (fully analytic) |
| **`euclid_cov`** ("emp") | **np.cov** | **np.cov** | **np.cov** | yes |
| **`euclid_cov_no_pb_cross`** ("emp_nocross") | **np.cov** | **np.cov** | **0** (zeroed) | yes |
| **`block_with_pb_cross_empirical`** ("blockcross") | **Grieb** | **Floss/Scoccimarro** (diag) | **empirical cross** | yes (cross/joint-block) |

- `diag` is the optimistic, sim-free analytic Gaussian forecast.
- `euclid_cov` is the fully empirical "truth" (captures non-Gaussian + all cross terms), but needs many seeds.
- `block_with_pb_cross_empirical` keeps the cheap analytic diagonals but restores the non-Gaussian **P–B** correlation from sims.

---

## 3. Fisher runs (8) — cov_type × data vector

All final field, res=128, shot noise, `pb_step=3`, `jacrev_chunk_size=32`, autodiff Jacobian.
Configs in `config_files/fisher/`, jobs `my_py_gpu_job_<name>.sh`, outputs `outputs/<name>_<timestamp>/`.

| Run name | Data vector | Covariance (in plain names) | `cov_type` | n_cov_seeds |
|---|---|---|---|---|
| `fisher_2param_P0_gauss`           | P₀                | Grieb (monopole limit)                       | diag       | – (analytic) |
| `fisher_2param_P0_emp`             | P₀                | np.cov                                        | euclid_cov | 600 |
| `fisher_2param_P024_gauss`         | P₀,P₂,P₄          | Grieb                                         | diag       | – (analytic) |
| `fisher_2param_P024_emp`           | P₀,P₂,P₄          | np.cov                                        | euclid_cov | 1200 |
| `fisher_2param_P024_B0_gauss`      | P₀,P₂,P₄,B₀       | Grieb + Floss/Scoccimarro                     | diag       | – (analytic) |
| `fisher_2param_P024_B0_emp`        | P₀,P₂,P₄,B₀       | np.cov                                        | euclid_cov | 2400 |
| `fisher_2param_P024_B0_emp_nocross`| P₀,P₂,P₄,B₀       | np.cov, **P–B cross zeroed** (isolates the cross) | euclid_cov_no_pb_cross | 2400 |
| `fisher_2param_P024_B0_blockcross` | P₀,P₂,P₄,B₀       | Grieb + Floss/Scoccimarro + empirical P–B cross | block_with_pb_cross_empirical | 2400 |

---

## 4. SBI runs (14) — the "compression" comparison (den_pie)

SBI learns the likelihood directly, so there is **no covariance**; the analogous axis is the
**data compression (embedding)** in front of the normalizing flow. Dataset `fli_2param`
(already shot-noised), `k_max=0.3`, `maf` flow. Params in `den_pie/params/`,
jobs `den_pie/scripts/run_<name>.sh`, outputs `den_pie/output/<name>_<timestamp>/`.

Compression types: **none** (Identity), **mlp** (learned MLP), **pca** (linear PCA).
Applied to P024 and P024+B0 only (P₀ is too small to compress); `embedding_dim` = 64 for
P024 (input 141/188), 128 for P024+B0 (input 471/518).

| Field config | P₀ | P₀P₂P₄ | P₀P₂P₄+B₀ |
|---|---|---|---|
| **fin** (delta_fin only) | `fin_P0_none` | `fin_P024_{none,mlp,pca}` | `fin_P024_B0_{none,mlp,pca}` |
| **icfin** (delta_ic P₀ + delta_fin) | `icfin_P0_none` | `icfin_P024_{none,mlp,pca}` | `icfin_P024_B0_{none,mlp,pca}` |

(Run name prefix: `spectra_2param_…`; param file `spectra_train_fli_2param_…yaml`.)

> **Note — 20260616 batch crashed; rerun on 20260617.** The first SBI batch
> (`*_20260616_*`) all died at ~epoch 35: `den_pie/spectra/train.py` saved a
> 167 MB `epoch_N.pth` every 10 epochs and never pruned them, so 14 concurrent
> jobs exhausted the `/gpfs/home6` quota and `torch.save` failed
> (`OSError: Disk quota exceeded`). Those plots are from a 35-epoch model and are
> invalid. **Fix:** `train.py` now keeps only `best.pth` (best EMA-val
> checkpoint); the periodic `epoch_N.pth` and `final.pth` saves were removed.
> The 14 runs were rerun (`*_20260617_*`) and `combined_corner_plot.py` repointed
> at the new dirs.

---

## 5. How to read the comparison

- **Across cov_type (Fisher):** `diag` (Grieb/Floss) vs `euclid_cov` (np.cov) vs
  `blockcross` shows how much the Gaussian-analytic assumption over-tightens the
  constraint, and how much of the gap is the non-Gaussian P–B cross.
- **Across data vector:** P₀ → +multipoles → +B₀ shows the information added by RSD
  multipoles and the bispectrum.
- **Fisher vs SBI:** for each data vector, the SBI posterior (best compression) is the
  realistic benchmark the Fisher ellipses approximate.

---

## 6. Correspondence to Flöss & Meerburg (arXiv:2305.07018) P+B analysis

The joint P+B Fisher in Flöss & Meerburg 2023 (*"Improving constraints on primordial
non-Gaussianity using neural network based reconstruction"*; notebook
`docs/Analyse.ipynb`) builds its covariance **empirically**:

```python
Cov_P_pre  = np.cov(Pk_fiducial_pre.T)      # power spectrum
Cov_B_pre  = np.cov(Bk_fiducial_pre.T)      # bispectrum
Cov_PB_pre = np.cov(PB_fiducial_pre.T)      # JOINT P+B — P–B cross captured inside np.cov
hartlap    = (N_sims - N_bins - 2) / (N_sims - 1)
Cov_PB_pre_Inv_wHartlap = np.linalg.inv(Cov_PB_pre) * hartlap
Fish_PB    = dPBdf.dot(Cov_PB_pre_Inv_wHartlap).dot(dPBdf.T)
```

i.e. `np.cov` over many fiducial sims + the Hartlap factor — **not** the analytic
Grieb/Scoccimarro/Sefusatti forms. The mapping to this project:

| Ingredient | Flöss 2305.07018 / `Analyse.ipynb` | This project |
|---|---|---|
| **B₀ measurement** (estimator) | BFast (`tsfloss/BFast`) | same BFast wrappers → analogous |
| **P+B covariance** | `np.cov(PB.T)` + Hartlap (empirical) | **`euclid_cov`** (incl. empirical P–B cross) → analogous |
| **Hartlap factor** | `(N−n−2)/(N−1)` | identical formula in `euclid_cov` / `blockcross` |
| **Analytic Gaussian cov** (Grieb / Scoccimarro / Sefusatti) | **not used** | `diag` / `blockcross` blocks |

**Takeaways:**
- "Floss" enters this project in two analogous ways: the **BFast bispectrum estimator**
  (reused) and the **empirical `np.cov` + Hartlap** covariance (= `euclid_cov`).
- The analytic **Scoccimarro2000 / Sefusatti2006** bispectrum covariance is *theory*
  that Flöss does **not** use; it lives only in this project's `diag` (and as the
  analytic B-block of `blockcross`).
- Therefore **`euclid_cov` is the direct apples-to-apples analog** of the Flöss P+B
  covariance, and the **`diag` vs `euclid_cov`** comparison quantifies exactly what the
  analytic-Gaussian approximation costs relative to the Flöss-style empirical covariance.

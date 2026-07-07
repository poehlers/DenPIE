"""Sweep n_seeds and plot marginal sigmas — verification step 2.

Reuses the per-seed Jacobians saved in ``fisher_results.npz`` and *rebuilds*
the Fisher matrix at several values of ``n_seeds_use`` by averaging the
first N saved Jacobians. The covariance is held fixed at the all-seeds
value, so the sweep isolates the Jacobian-averaging bias.

Plateau within ~5% from n_seeds=20 → 40 confirms the bias has converged.

Usage
-----
    python core/verification/plot_nseeds_sweep.py \
        outputs/fisher_bias_P_B_rsd_SN/fisher_results.npz \
        --seeds 3 5 10 20 40 \
        --out outputs/verification/nseeds_sweep.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the production cov/Fisher helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import jax.numpy as jnp
from den_pie.fisher.fisher import (
    multipole_covariance_grieb,
    bispectrum_covariance,
    prior_fisher_matrix,
    _PRIOR_SPEC_BIAS,
    _PRIOR_SPEC,
)


_LATEX = {
    "Omega_m": r"$\sigma(\Omega_m)$",
    "sigma8":  r"$\sigma(\sigma_8)$",
    "b1":      r"$\sigma(b_1)$",
    "b2":      r"$\sigma(b_2)$",
    "bs2":     r"$\sigma(b_{s^2})$",
    "bn2":     r"$\sigma(b_{\nabla^2})$",
}


def _scalar(x):
    return x.item() if hasattr(x, "item") else x


def _detect_field_and_blocks(d):
    """Return (field, n_blocks, n_l, multipole_ls, has_grieb)."""
    field = str(d["field"]) if "field" in d.files else "fin"
    use_multi = bool(_scalar(d["use_multipoles"])) if "use_multipoles" in d.files else False
    multipole_ls = tuple(int(l) for l in d["multipole_ls"]) if "multipole_ls" in d.files else \
        ((0, 2, 4) if use_multi else (0,))
    n_l = len(multipole_ls)
    n_blocks = 2 if field == "both" else 1
    has_grieb = "cov_P_grieb_blocks" in d.files
    return field, n_blocks, n_l, multipole_ls, has_grieb


def _per_block_Pl(d, b, multipole_ls, field):
    """Return (P0, P2, P4) for the b-th block; zeros for missing l."""
    if field == "both":
        suffix = "_ic_fid" if b == 0 else "_fin_fid"
    else:
        suffix = "_fid"
    P0 = np.asarray(d.get(f"Pk0{suffix}", d.get("Pk_fid")), dtype=float)
    n_k = P0.size
    P2 = np.asarray(d[f"Pk2{suffix}"], dtype=float) if f"Pk2{suffix}" in d.files else np.zeros(n_k)
    P4 = np.asarray(d[f"Pk4{suffix}"], dtype=float) if f"Pk4{suffix}" in d.files else np.zeros(n_k)
    if 2 not in multipole_ls:
        P2 = np.zeros(n_k)
    if 4 not in multipole_ls:
        P4 = np.zeros(n_k)
    return P0, P2, P4


def _per_block_Bk(d, b, field):
    if field == "both":
        return np.asarray(d["Bk_ic_fid"] if b == 0 else d["Bk_fin_fid"], dtype=float)
    return np.asarray(d["Bk_fid"], dtype=float)


def _compute_fisher_from_subset(d, jac_subset_mean):
    """Replicate fisher_bias_PB Fisher assembly from a saved npz + subset-averaged Jacobian."""
    field, n_blocks, n_l, multipole_ls, has_grieb = _detect_field_and_blocks(d)
    n_P_bins = int(_scalar(d["n_P_bins"]))
    n_triangles = int(_scalar(d["n_triangles"]))
    block_size = n_l * n_P_bins + n_triangles

    N_modes = jnp.asarray(d["N_modes"])
    k_bins_P = jnp.asarray(d["k_bins"])
    bin_widths_P = jnp.asarray(d["bin_widths_P"])
    triangle_indices = jnp.asarray(d["triangle_indices"])
    bin_edges = np.asarray(d["bin_edges"])
    # V_box: we don't have it directly; reconstruct from k_F = bin_edges[0]*2pi/(?) -- skip.
    # bispectrum_covariance signature: (Pk, triangle_indices, k_bins, bin_widths, V_box)
    # We need V_box. Recover from N_modes/k bins is tricky -- instead read from
    # the per-block sigma_sq_B saved in the npz (when available) by inverting the formula.
    # Simpler: deduce from the BFast bin_edges (units of k_F) and the saved k_bins:
    #   k_F = k_bins / bin_centre_in_k_F  -> bin_centre = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bin_centre_kF = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    k_F = float(np.asarray(d["k_bins"]).flatten()[0] / bin_centre_kF.flatten()[0])
    boxsize = 2.0 * np.pi / k_F
    dim = 3
    V_box = boxsize ** dim

    F_data = jnp.zeros((jac_subset_mean.shape[-1], jac_subset_mean.shape[-1]),
                       dtype=jnp.float32)
    eye_l = jnp.eye(n_l, dtype=jnp.float32)

    for b in range(n_blocks):
        off = b * block_size
        # Reshape P Jacobian: (n_l, n_P_bins, n_params)
        J_P = jnp.stack([
            jac_subset_mean[off + li * n_P_bins:off + (li + 1) * n_P_bins]
            for li in range(n_l)
        ], axis=0)
        J_B = jac_subset_mean[off + n_l * n_P_bins:off + block_size]

        # Grieb cov (n_P_bins, n_l, n_l)
        if has_grieb:
            cov_P = jnp.asarray(d["cov_P_grieb_blocks"][b])
        else:
            P0, P2, P4 = _per_block_Pl(d, b, multipole_ls, field)
            cov_P = multipole_covariance_grieb(
                jnp.asarray(P0), jnp.asarray(P2), jnp.asarray(P4), N_modes,
                multipole_ls=multipole_ls,
            )
        cov_safe = jnp.where(
            (N_modes > 0)[:, None, None], cov_P, eye_l[None, :, :],
        )
        # B diagonal cov
        P0_for_B, _, _ = _per_block_Pl(d, b, multipole_ls, field)
        sigma_sq_B = bispectrum_covariance(
            jnp.asarray(P0_for_B), triangle_indices, k_bins_P, bin_widths_P, V_box,
        )
        # Validity (drop k with N_modes==0; drop triangles with bad sigma_sq_B)
        valid_k = (N_modes > 0)
        cov_inv = jnp.linalg.inv(cov_safe)
        J_P_safe = jnp.where(valid_k[None, :, None], J_P, jnp.zeros_like(J_P))
        F_P = jnp.einsum('aki,kab,bkj->ij', J_P_safe, cov_inv, J_P_safe)

        valid_B = (sigma_sq_B > 0) & jnp.isfinite(sigma_sq_B)
        sigma_sq_safe = jnp.where(valid_B, sigma_sq_B, jnp.ones_like(sigma_sq_B))
        J_B_safe = jnp.where(valid_B[:, None], J_B, jnp.zeros_like(J_B))
        F_B = (J_B_safe / sigma_sq_safe[:, None]).T @ J_B_safe

        F_data = F_data + F_P + F_B

    return np.asarray(F_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="fisher_results.npz with jacobians_per_seed")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[3, 5, 10, 20, 40],
                    help="n_seeds values to sweep over")
    ap.add_argument("--out", default="outputs/verification/nseeds_sweep.png")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    J_per_seed = np.asarray(d["jacobians_per_seed"], dtype=np.float32)  # (n_seeds, n_data, n_params)
    N_total = J_per_seed.shape[0]
    print(f"Loaded {args.npz}")
    print(f"  jacobians_per_seed shape: {J_per_seed.shape}")
    print(f"  n_seeds available: {N_total}")

    valid_seeds = [s for s in args.seeds if s <= N_total]
    if not valid_seeds:
        raise SystemExit(f"All requested n_seeds > available {N_total}")
    skipped = [s for s in args.seeds if s > N_total]
    if skipped:
        print(f"  WARNING: requested n_seeds {skipped} exceed available; skipped.")

    names = [str(n) for n in d["param_names"]]
    fid = np.asarray(d["fid_values"], dtype=float)

    # Detect param-set: bias if 6 params, full-cosmo if 5
    if len(names) == 6 and "b1" in names:
        prior_spec = _PRIOR_SPEC_BIAS
    else:
        prior_spec = _PRIOR_SPEC
    F_prior = prior_fisher_matrix(names, prior_spec)

    sigmas_data = np.zeros((len(valid_seeds), len(names)))
    sigmas_post = np.zeros((len(valid_seeds), len(names)))
    for i, n in enumerate(valid_seeds):
        J = jnp.asarray(J_per_seed[:n].mean(axis=0))
        F_data = _compute_fisher_from_subset(d, J)
        # Marginal sigmas (data only and data+prior)
        try:
            cov_d = np.linalg.inv(F_data)
            sigmas_data[i] = np.sqrt(np.diag(cov_d))
        except np.linalg.LinAlgError:
            sigmas_data[i] = np.nan
        cov_p = np.linalg.inv(F_data + F_prior)
        sigmas_post[i] = np.sqrt(np.diag(cov_p))
        print(f"  n_seeds={n:3d}: "
              + ", ".join(f"σ({nm})={s:.4g}" for nm, s in zip(names, sigmas_post[i])))

    # --- Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    seeds_arr = np.asarray(valid_seeds)
    for ax, sig_arr, title in [
        (axes[0], sigmas_data, "Data only"),
        (axes[1], sigmas_post, "Data + prior"),
    ]:
        for j, name in enumerate(names):
            # Normalise by σ at largest n_seeds for readability
            ref = sig_arr[-1, j]
            ax.plot(seeds_arr, sig_arr[:, j] / ref, "o-",
                    label=_LATEX.get(name, name))
        ax.axhline(1.05, color="gray", ls=":", lw=0.8)
        ax.axhline(1.00, color="gray", ls="-", lw=0.5)
        ax.axhline(0.95, color="gray", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("n_seeds used")
        ax.set_ylabel(f"σ / σ(n_seeds={seeds_arr[-1]})")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Marginal-sigma convergence vs n_seeds (Jacobian noise bias)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nSaved {args.out}")

    # Print plateau summary
    print("\nFractional change σ(n=largest) vs σ(n=20):")
    if 20 in valid_seeds:
        i20 = valid_seeds.index(20)
        for j, name in enumerate(names):
            ref = sigmas_post[-1, j]
            d20 = sigmas_post[i20, j]
            print(f"  {name}: {(d20 - ref) / ref * 100:+.2f}%")


if __name__ == "__main__":
    main()

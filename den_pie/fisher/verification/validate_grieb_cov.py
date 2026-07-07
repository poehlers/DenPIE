"""Empirical Cov(P_l(k), P_l'(k)) vs Grieb+2016 prediction.

Verification step 3 of `docs/fisher_too_tight_fixes.md`: run N independent
noise seeds at the fiducial cosmology through the RSD-multipole pipeline,
measure the empirical multipole covariance, and compare to the analytic
Grieb cov.

This is a forward-only validation (no jacrev) so each seed is cheap. Run as
a SLURM job for n_seeds >= 100.

Usage
-----
    python core/verification/validate_grieb_cov.py \
        --n-seeds 100 --pb-step 2 --n-g-density 1.0e-3 \
        --field fin --use-multipoles \
        --out outputs/verification/grieb_cov_validation.png
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import jax
import jax.numpy as jnp

from den_pie.forward.model import _FIDUCIAL_COSMO, BOX_PARAMS, SIM_PARAMS
from den_pie.fisher.fisher import (
    _BIAS_COSMO_KEYS, _BIAS_PARAM_ORDER, _FIDUCIAL_BIAS,
    _precompute_k_vecs, _bfast_precompute,
    _differentiable_pipeline_fin_bias_PB_rsd_multi,
    _differentiable_pipeline_fin_bias_PB,
    multipole_covariance_grieb,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--pb-step", type=int, default=2)
    ap.add_argument("--n-g-density", type=float, default=None)
    ap.add_argument("--use-multipoles", action="store_true",
                    help="If set, use RSD P_0/P_2/P_4 + B pipeline; otherwise monopole+B")
    ap.add_argument("--field", default="fin", choices=["fin"])
    ap.add_argument("--out", default="outputs/verification/grieb_cov_validation.png")
    args = ap.parse_args()

    box_params = dict(BOX_PARAMS)
    sim_params = dict(SIM_PARAMS)
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]

    fixed_cosmo_params = {
        "Omega_b": _FIDUCIAL_COSMO["Omega_b"],
        "h": _FIDUCIAL_COSMO["h"],
        "n_s": _FIDUCIAL_COSMO["n_s"],
    }
    sampled_keys = _BIAS_COSMO_KEYS + _BIAS_PARAM_ORDER
    fid_dict = {
        "Omega_m": _FIDUCIAL_COSMO["Omega_c"] + _FIDUCIAL_COSMO["Omega_b"],
        "sigma8":  _FIDUCIAL_COSMO["sigma8"],
    }
    fid_dict.update(_FIDUCIAL_BIAS)
    fid_values = jnp.array([fid_dict[k] for k in sampled_keys], dtype=jnp.float32)

    bin_edges = jnp.arange(1, res // 3 + 1, args.pb_step, dtype=jnp.int32)
    n_P_bins = int(bin_edges.shape[0] - 1)

    print("Pre-computing k_vecs + BFast triangles ...", flush=True)
    k_vecs = _precompute_k_vecs(box_params)
    B_info, B_norm = _bfast_precompute(
        boxsize=boxsize, bin_edges=bin_edges, res=res, dim=dim, mas_order=0,
    )
    n_triangles = int(B_info["triangle_indices"].shape[0])
    N_modes = jnp.asarray(B_norm["Pk"])
    k_F = 2.0 * jnp.pi / boxsize
    k_bins_P = 0.5 * (bin_edges[1:] + bin_edges[:-1]) * k_F

    multipole_ls = (0, 2, 4) if args.use_multipoles else (0,)
    n_l = len(multipole_ls)
    device = jax.devices()[0]

    if args.use_multipoles:
        pipeline = _differentiable_pipeline_fin_bias_PB_rsd_multi
    else:
        pipeline = _differentiable_pipeline_fin_bias_PB

    sn_active = args.n_g_density is not None

    def _eval(noise, sn_noise=None):
        if sn_active:
            return pipeline(
                fid_values, noise, k_vecs, bin_edges, B_info, B_norm,
                box_params=box_params, sim_params=sim_params,
                fixed_cosmo_params=fixed_cosmo_params, device=device,
                sn_noise=sn_noise, n_g_density=args.n_g_density,
            )
        return pipeline(
            fid_values, noise, k_vecs, bin_edges, B_info, B_norm,
            box_params=box_params, sim_params=sim_params,
            fixed_cosmo_params=fixed_cosmo_params, device=device,
        )

    # JIT for speed
    eval_jit = jax.jit(_eval)

    P_per_seed = np.zeros((args.n_seeds, n_l, n_P_bins), dtype=np.float32)
    t0 = time.time()
    for s in range(args.n_seeds):
        key = jax.random.PRNGKey(s)
        if sn_active:
            k_ic, k_sn = jax.random.split(key)
            noise    = jax.random.normal(k_ic, shape=(res,) * dim)
            sn_noise = jax.random.normal(k_sn, shape=(res,) * dim)
            data = eval_jit(noise, sn_noise)
        else:
            noise = jax.random.normal(key, shape=(res,) * dim)
            data = eval_jit(noise)
        # Layout: [P_0(k_0)..P_0(k_n), P_2..., P_4..., B(t_0)..B(t_T)]
        for li in range(n_l):
            P_per_seed[s, li] = np.asarray(data[li * n_P_bins:(li + 1) * n_P_bins])
        if (s + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  seed {s+1}/{args.n_seeds}  elapsed {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. {args.n_seeds} forward passes in {elapsed:.1f}s "
          f"({elapsed / args.n_seeds:.2f}s/seed)")

    # Empirical covariance per k-bin: cov_emp[k, a, b]
    # = (1/(N-1)) sum_s (P[s, a, k] - <P[:, a, k]>) (P[s, b, k] - <P[:, b, k]>)
    mean = P_per_seed.mean(axis=0)                # (n_l, n_P_bins)
    dev = P_per_seed - mean[None, :, :]            # (N, n_l, n_P_bins)
    cov_emp = np.einsum('san,sbn->nab', dev, dev) / (args.n_seeds - 1)  # (n_P_bins, n_l, n_l)

    # Analytic Grieb cov using the seed-averaged P_l as Pk_fid
    Pk0 = jnp.asarray(mean[0])
    Pk2 = jnp.asarray(mean[1]) if n_l > 1 else jnp.zeros_like(Pk0)
    Pk4 = jnp.asarray(mean[2]) if n_l > 2 else jnp.zeros_like(Pk0)
    cov_grieb = np.asarray(multipole_covariance_grieb(
        Pk0, Pk2, Pk4, N_modes, multipole_ls=multipole_ls,
    ))                                              # (n_P_bins, n_l, n_l)

    # --- Plot
    k_np = np.asarray(k_bins_P)
    fig, axes = plt.subplots(n_l, n_l, figsize=(3 + 2.5 * n_l, 2.5 * n_l + 1),
                              sharex=True)
    if n_l == 1:
        axes = np.array([[axes]])
    for a in range(n_l):
        for b in range(n_l):
            ax = axes[a, b]
            if a == b:
                ax.plot(k_np, cov_emp[:, a, b], "o-", color="C0",
                        label="empirical")
                ax.plot(k_np, cov_grieb[:, a, b], "x--", color="C1",
                        label="Grieb")
                ax.set_yscale("log")
                ax.set_ylabel(f"Var(P_{multipole_ls[a]})")
            else:
                # Correlation coefficient: ρ = Cov / sqrt(Var_a * Var_b)
                var_a_e = cov_emp[:, a, a]
                var_b_e = cov_emp[:, b, b]
                var_a_g = cov_grieb[:, a, a]
                var_b_g = cov_grieb[:, b, b]
                rho_e = cov_emp[:, a, b] / np.sqrt(np.abs(var_a_e * var_b_e))
                rho_g = cov_grieb[:, a, b] / np.sqrt(np.abs(var_a_g * var_b_g))
                ax.plot(k_np, rho_e, "o-", color="C0", label="empirical")
                ax.plot(k_np, rho_g, "x--", color="C1", label="Grieb")
                ax.axhline(0, color="k", lw=0.5)
                ax.set_ylabel(f"ρ(P_{multipole_ls[a]}, P_{multipole_ls[b]})")
                ax.set_ylim(-0.2, 1.05)
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if a == n_l - 1:
                ax.set_xlabel("k [h/Mpc]")
            if a == 0 and b == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Empirical vs Grieb+2016 multipole cov "
                 f"({args.n_seeds} seeds, "
                 f"{'SN' if sn_active else 'no SN'})", y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")

    # Also save raw arrays for further inspection
    npz_out = args.out.replace(".png", ".npz")
    np.savez(
        npz_out,
        k_bins=np.asarray(k_bins_P),
        N_modes=np.asarray(N_modes),
        P_per_seed=P_per_seed,
        cov_emp=cov_emp,
        cov_grieb=cov_grieb,
        multipole_ls=np.asarray(multipole_ls, dtype=np.int32),
        n_seeds=np.int32(args.n_seeds),
        n_g_density=np.asarray(args.n_g_density if sn_active else np.nan),
    )
    print(f"Saved {npz_out}")

    # Print summary statistics
    print("\nVar(P_l) ratio (empirical / Grieb), per multipole, average over k:")
    for li, ell in enumerate(multipole_ls):
        ratio = cov_emp[:, li, li] / cov_grieb[:, li, li]
        print(f"  l={ell}: mean={np.nanmean(ratio):.3f} median={np.nanmedian(ratio):.3f}")
    if n_l > 1:
        print("\nOff-diagonal Pearson correlation Cov(0, 2)/sqrt(Var(0)Var(2)):")
        rho_e_02 = cov_emp[:, 0, 1] / np.sqrt(cov_emp[:, 0, 0] * cov_emp[:, 1, 1])
        rho_g_02 = cov_grieb[:, 0, 1] / np.sqrt(cov_grieb[:, 0, 0] * cov_grieb[:, 1, 1])
        print(f"  empirical: mean={np.nanmean(rho_e_02):.3f}")
        print(f"  Grieb:     mean={np.nanmean(rho_g_02):.3f}")


if __name__ == "__main__":
    main()

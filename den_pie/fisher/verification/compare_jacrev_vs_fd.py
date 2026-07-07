"""jacrev vs central finite-difference derivative comparison.

Verification step 4 of `docs/fisher_too_tight_fixes.md`: confirm whether the
``jax.lax.stop_gradient(cosmo)`` through the N-body biases the derivative
of P_l(k) w.r.t. cosmology. For σ_8 the two methods should agree exactly
(σ_8 doesn't enter D(a) ratios); for Ω_m and h they should diverge at
high k by the missing dD/dθ contribution.

Run as SLURM job — one fid pass + 2*N_params perturbed passes + 1 jacrev.

Usage
-----
    python core/verification/compare_jacrev_vs_fd.py \
        --pb-step 2 --n-g-density 1.0e-3 \
        --use-multipoles \
        --params Omega_m sigma8 \
        --out outputs/verification/jacrev_vs_fd.png
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
)

# Reasonable central-difference steps (~1% of fiducial)
_FD_STEPS = {
    "Omega_m": 0.005,
    "sigma8":  0.008,
    "b1":      0.05,
    "b2":      0.10,
    "bs2":     0.10,
    "bn2":     0.10,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb-step", type=int, default=2)
    ap.add_argument("--n-g-density", type=float, default=None)
    ap.add_argument("--use-multipoles", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="Single white-noise seed to use for the comparison")
    ap.add_argument("--params", nargs="+",
                    default=["Omega_m", "sigma8"],
                    help="Subset of params to plot (must be in sampled_keys)")
    ap.add_argument("--out", default="outputs/verification/jacrev_vs_fd.png")
    args = ap.parse_args()

    box_params = dict(BOX_PARAMS)
    sim_params = dict(SIM_PARAMS)
    dim, res, boxsize = box_params["dim"], box_params["res"], box_params["boxsize"]

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
    k_F = 2.0 * jnp.pi / boxsize
    k_bins_P = np.asarray(0.5 * (bin_edges[1:] + bin_edges[:-1]) * k_F)

    multipole_ls = (0, 2, 4) if args.use_multipoles else (0,)
    n_l = len(multipole_ls)
    device = jax.devices()[0]

    sn_active = args.n_g_density is not None
    if args.use_multipoles:
        pipeline = _differentiable_pipeline_fin_bias_PB_rsd_multi
    else:
        pipeline = _differentiable_pipeline_fin_bias_PB

    # Fix the noise: same seed for jacrev AND finite differences (consistency)
    key = jax.random.PRNGKey(args.seed)
    if sn_active:
        k_ic, k_sn = jax.random.split(key)
        noise    = jax.random.normal(k_ic, shape=(res,) * dim)
        sn_noise = jax.random.normal(k_sn, shape=(res,) * dim)
        def pipe(params):
            return pipeline(
                params, noise, k_vecs, bin_edges, B_info, B_norm,
                box_params=box_params, sim_params=sim_params,
                fixed_cosmo_params=fixed_cosmo_params, device=device,
                sn_noise=sn_noise, n_g_density=args.n_g_density,
            )
    else:
        noise = jax.random.normal(key, shape=(res,) * dim)
        def pipe(params):
            return pipeline(
                params, noise, k_vecs, bin_edges, B_info, B_norm,
                box_params=box_params, sim_params=sim_params,
                fixed_cosmo_params=fixed_cosmo_params, device=device,
            )

    pipe_jit = jax.jit(pipe)

    # 1) jacrev derivative — shape (n_data, n_params)
    print("\nComputing jacrev derivative ...", flush=True)
    t0 = time.time()
    J_rev = jax.jacrev(pipe_jit)(fid_values)
    J_rev.block_until_ready()
    print(f"  jacrev done in {time.time() - t0:.1f}s, shape {J_rev.shape}")
    J_rev = np.asarray(J_rev)

    # 2) Central FD for each requested param
    print("\nComputing central FD derivative ...", flush=True)
    J_fd = np.zeros_like(J_rev)
    fid_np = np.asarray(fid_values)
    for j, name in enumerate(sampled_keys):
        if name not in args.params:
            continue
        step = _FD_STEPS.get(name, 0.01)
        plus  = fid_np.copy()
        minus = fid_np.copy()
        plus[j]  = fid_np[j] + step
        minus[j] = fid_np[j] - step
        t0 = time.time()
        Pk_plus  = np.asarray(pipe_jit(jnp.asarray(plus,  dtype=jnp.float32)))
        Pk_minus = np.asarray(pipe_jit(jnp.asarray(minus, dtype=jnp.float32)))
        J_fd[:, j] = (Pk_plus - Pk_minus) / (2.0 * step)
        print(f"  d/d{name} (step={step}) done in {time.time() - t0:.1f}s")

    # --- Plot per requested param
    requested_idx = [sampled_keys.index(p) for p in args.params]
    n_params_plot = len(requested_idx)
    fig, axes = plt.subplots(n_l, n_params_plot,
                              figsize=(4.5 * n_params_plot, 3.0 * n_l + 1),
                              sharex=True, squeeze=False)
    for li, ell in enumerate(multipole_ls):
        for pi, j in enumerate(requested_idx):
            ax = axes[li, pi]
            seg = slice(li * n_P_bins, (li + 1) * n_P_bins)
            J_r = J_rev[seg, j]
            J_f = J_fd[seg, j]
            ax.plot(k_bins_P, J_r, "o-", color="C0", label="jacrev (stop_gradient cosmo)")
            ax.plot(k_bins_P, J_f, "x--", color="C1", label="central FD (full chain rule)")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=max(1e-6 * np.max(np.abs(J_f)), 1e-30))
            ax.set_xlabel("k [h/Mpc]")
            ax.set_ylabel(f"∂P_{ell}/∂{args.params[pi]}")
            ax.grid(alpha=0.3)
            if li == 0 and pi == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Jacobian: jacrev (with cosmo stop_gradient) vs central FD")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")

    # Also: ratio plot
    fig2, axes2 = plt.subplots(n_l, n_params_plot,
                               figsize=(4.5 * n_params_plot, 2.5 * n_l + 1),
                               sharex=True, squeeze=False)
    for li, ell in enumerate(multipole_ls):
        for pi, j in enumerate(requested_idx):
            ax = axes2[li, pi]
            seg = slice(li * n_P_bins, (li + 1) * n_P_bins)
            J_r = J_rev[seg, j]
            J_f = J_fd[seg, j]
            mask = np.abs(J_f) > 1e-12
            ratio = np.where(mask, J_r / np.where(mask, J_f, 1), np.nan)
            ax.plot(k_bins_P, ratio, "o-", color="C2")
            ax.axhline(1.0, color="k", lw=0.8, ls="-")
            ax.set_xscale("log")
            ax.set_xlabel("k [h/Mpc]")
            ax.set_ylabel(f"jacrev / FD  (∂P_{ell}/∂{args.params[pi]})")
            ax.grid(alpha=0.3)
            ax.set_ylim(0.5, 1.5)
    fig2.suptitle("Ratio jacrev / FD — deviation from 1 = stop_gradient bias")
    fig2.tight_layout()
    out_ratio = args.out.replace(".png", "_ratio.png")
    fig2.savefig(out_ratio, dpi=140, bbox_inches="tight")
    print(f"Saved {out_ratio}")

    # Save raw data
    out_npz = args.out.replace(".png", ".npz")
    np.savez(
        out_npz,
        J_jacrev=J_rev, J_fd=J_fd,
        k_bins=k_bins_P,
        multipole_ls=np.asarray(multipole_ls, dtype=np.int32),
        param_names=np.array(sampled_keys),
        fid_values=fid_np,
    )
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()

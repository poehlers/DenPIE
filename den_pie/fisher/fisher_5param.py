#!/usr/bin/env python
"""Fisher forecast for the 5 cosmological params on the `fli_5param` dataset.

Standalone counterpart to ``core/fisher.py`` / ``core/fisher_2param.py``,
specialised to the 5-parameter **pure-matter** model behind the ``fli_5param``
dataset. The final field is mapped into redshift space (RSD, los=2) and carries
DESI-like Gaussian shot noise, and the data vector is the joint summary

    d(theta) = [P_0(k), P_2(k), P_4(k), B_0(k1,k2,k3)]   of delta_fin,

with theta = [Omega_m, Omega_b, h, n_s, sigma_8] (nothing fixed). This is built
to complement the SBI run
``den_pie/output/spectra_fli_5param_icP0_finP024_B0_none_*`` but on the **final
field only** and **with shot noise** (the SBI counterpart with shot noise is
generated separately by the user).

This is exactly the matter RSD-multipole pipeline of ``fisher_2param.py`` with
all five cosmological parameters sampled instead of two; the Lagrangian bias
weights are 1 (uniform mass = matter). We reuse the proven numerical primitives
from ``fisher.py`` (AD-safe BFast wrappers, empirical covariance + Hartlap,
Fisher assembly, finite-difference Jacobian) and only swap in the 5-parameter
constants + driver here.

Derivatives default to **central finite differences** (``derivative_method="fd"``):
DiscoDJ's N-body has no working JVP/VJP for the cosmology pytree, so the autodiff
path ``stop_gradient``'s ``cosmo`` through the N-body and the cosmo gradient only
flows via P_lin -> delta_ic. For a *pure-cosmology* forecast that omission matters
(notably the Omega_m growth/displacement dependence), so we re-run the full
pipeline at theta +/- eps and capture the complete dependence. Steps come from
``fisher._FINITE_DIFF_STEPS``.

Covariance is **empirical** (sample covariance over many fiducial seeds,
Hartlap-corrected), matching the methodology in ``docs/Analyse.ipynb``
(``cov = (centred.T @ centred)/(N-1)`` with the Hartlap factor on the inverse).

The script writes ``fisher_results.npz`` (Fisher matrix, covariance, marginal
sigmas, Jacobian, fiducial data vector, block index spans) so the Fisher
constraint can be overlaid against an SBI posterior downstream.

Run from the repo root (so ``core/`` lands on sys.path[0]):

    python core/fisher_5param.py --config config_files/fisher/fisher_5param_P024_B0_emp.yml
    python core/fisher_5param.py --res 32 --pb-step 4 --n-seeds 1 --n-cov-seeds 400
"""

import argparse
import os
import shutil
import time
from datetime import datetime

import numpy as np


# Fiducial cosmology (Planck-2018 / Quijote), matching model.py _FIDUCIAL_COSMO.
# All five parameters are sampled (nothing fixed).
_FIDUCIAL_5PARAM = {
    "Omega_m": 0.3175, "Omega_b": 0.0490, "h": 0.6711,
    "n_s": 0.9624, "sigma8": 0.8340,
}
_FIXED_COSMO = {}  # nothing fixed -> _cosmo_dict unpacks all 5 params
_SAMPLED_KEYS = ["Omega_m", "Omega_b", "h", "n_s", "sigma8"]

# Uniform priors from config_files/config_5param.yml. Used for the prior Fisher
# term and as plot bounds.
_PRIOR_SPEC = {
    "Omega_m": ("uniform", 0.10, 0.50),
    "Omega_b": ("uniform", 0.03, 0.07),
    "h":       ("uniform", 0.50, 0.90),
    "n_s":     ("uniform", 0.80, 1.20),
    "sigma8":  ("uniform", 0.60, 1.00),
}
_PRIOR_BOUNDS = {k: (v[1], v[2]) for k, v in _PRIOR_SPEC.items()}

# LaTeX labels for the Fisher-only corner plot.
_LATEX = {
    "Omega_m": r"$\Omega_m$", "Omega_b": r"$\Omega_b$", "h": r"$h$",
    "n_s": r"$n_s$", "sigma8": r"$\sigma_8$",
}


def _pipeline_5param(
    params,
    white_noise,
    sn_noise,
    bin_edges,
    B_info,
    B_norm,
    box_params,
    sim_params,
    fixed_cosmo_params,
    n_g_density,
    device=None,
    pk_multipoles=True,
    joint_pb=True,
    forward_mode=False,
):
    """Differentiable matter pipeline: 5 cosmo params -> selected summary.

    Identical to ``fisher_2param._pipeline_2param`` (uniform particle weights =
    pure matter, RSD los=2, optional Gaussian shot noise, BFast [P0,P2,P4,B0])
    except that all five cosmological parameters are consumed: with
    ``fixed_cosmo_params`` empty, ``_cosmo_dict`` unpacks the full
    [Omega_m, Omega_b, h, n_s, sigma8] vector.

    ``forward_mode`` switches the cosmo-gradient handling for ``jax.jacfwd``:

    - False (default, for reverse-mode jacrev / FD): the N-body uses
      ``requires_jacfwd=False`` (custom_vjp adjoint) and ``cosmo`` is
      ``stop_gradient``'d through it, so the cosmo gradient only flows via
      P_lin -> delta_ic (the N-body growth dependence is dropped).
    - True (for forward-mode jacfwd): the N-body uses ``requires_jacfwd=True``
      (plain ``jax.lax.scan`` + custom_jvp scatter, both forward-safe) and
      ``cosmo`` is NOT stop_gradient'd, so jacfwd captures the *full* cosmo
      dependence including the N-body growth/displacement.
    """
    import jax
    import jax.numpy as jnp
    from discodj import DiscoDJ
    from discodj.core.grids import get_fourier_grid
    from den_pie.fisher.fisher import _cosmo_dict, _add_shot_noise, _bfast_pk_multi_bk_safe

    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]
    sp = sim_params

    if device is None:
        device = jax.devices()[0]

    req_jf = bool(forward_mode)
    cosmo = _cosmo_dict(params, fixed=fixed_cosmo_params)

    # Step 1: linear power spectrum P_lin(k)
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                  requires_jacfwd=req_jf)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: generate delta_ic from the fixed white-noise field
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: N-body. In reverse/FD mode cosmo is stop_gradient'd (the custom_vjp
    # adjoint drops the cosmo cotangent anyway); in forward_mode cosmo flows
    # through so jacfwd captures the full growth dependence.
    cosmo_nbody = cosmo if forward_mode else jax.lax.stop_gradient(cosmo)
    dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                  cosmo=cosmo_nbody, requires_jacfwd=req_jf)
    dj2 = dj2.with_timetables()
    dj2 = dj2.with_external_ics(delta=delta_ic)
    dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

    _skip = {"nlpt_order_ics", "worder"}
    run_params = {k: v for k, v in sp.items() if k not in _skip}

    # Capture canonical momenta (P_mom) for the RSD displacement.
    X_sim, P_mom, _ = dj2.run_nbody(**run_params, use_diffrax=False)
    P_flat = P_mom.reshape(-1, 3)

    # Step 3b: scatter onto the Eulerian mesh in redshift space with UNIFORM
    # weights (matter, no bias); los_axis = 2 (z), particles displaced by
    # v_los/(aH).
    n_field = dj2.compute_field_quantity_from_particles(
        pos=X_sim,
        quantity=jnp.ones((res ** dim,), dtype=X_sim.dtype),
        normalize_by_density=False,
        in_redshift_space=True,
        vel=P_flat,
        a=sp["a_end"],
        radial_dim=2,
        worder=sp["worder"],
        antialias=1,
        deconvolve=True,
    )
    delta = n_field / n_field.mean() - 1.0

    # Optional Gaussian shot noise (Poisson-Gaussian limit; no-op if n_g is None).
    delta = _add_shot_noise(delta, sn_noise, n_g_density, boxsize, res)

    # Step 4: joint P_l(k) + B_0(k1,k2,k3) via BFast (mas_order=0: pre-deconvolved).
    Pk0, Pk2, Pk4, Bk = _bfast_pk_multi_bk_safe(
        delta, boxsize=boxsize, bin_edges=bin_edges,
        B_info=B_info, B_norm=B_norm, multipole_axis=2, mas_order=0,
    )
    blocks = [Pk0]
    if pk_multipoles:
        blocks += [Pk2, Pk4]
    if joint_pb:
        blocks.append(Bk)
    return jnp.concatenate(blocks)


def _block_spans(n_P_bins, n_triangles, pk_multipoles, joint_pb):
    """[start, stop) index spans of each included block in the data vector."""
    spans = [[0, n_P_bins]]                          # P_0 always present
    if pk_multipoles:
        spans += [[n_P_bins, 2 * n_P_bins],          # P_2
                  [2 * n_P_bins, 3 * n_P_bins]]       # P_4
    if joint_pb:
        off = (3 if pk_multipoles else 1) * n_P_bins
        spans.append([off, off + n_triangles])        # B_0
    return np.array(spans)


def run_fisher_5param(
    n_seeds=10,
    n_cov_seeds=2400,
    pb_step=3,
    n_g_density=1.0e-3,
    output_dir=None,
    box_params=None,
    sim_params=None,
    derivative_method="fd",
    jacrev_chunk_size=None,
    fd_step_kind="absolute",
    fd_relative_eps=1e-2,
    pk_multipoles=True,
    joint_pb=True,
    cov_type="euclid_cov",
    config_path=None,
    resolved_args=None,
):
    """Run the 5-parameter cosmology Fisher forecast on [P_0,P_2,P_4,B_0].

    ``derivative_method``:
    - ``"fd"`` (default): central finite differences, 1 + 2*5 = 11 forward
      passes per seed. Captures the *full* cosmology dependence (including the
      N-body growth factors), unlike the autodiff path which stop_gradients
      ``cosmo`` through the N-body and would mis-estimate Omega_m.
    - ``"autodiff"``: ``jax.jacrev`` (one reverse pass per output element).
      Memory-heavy at res=128 -- use ``jacrev_chunk_size`` to bound it. Only the
      P_lin -> delta_ic channel carries cosmo gradient (growth omitted).
    """
    import jax
    import jax.numpy as jnp
    from den_pie.fisher.fisher import (
        _bfast_precompute, compute_jacobian, compute_jacobian_numerical,
        compute_jacobian_forward,
        compute_empirical_covariance, empirical_pb_cross_block,
        fisher_matrix_full, prior_fisher_matrix,
        gaussian_covariance, multipole_covariance_grieb, bispectrum_covariance,
        plot_fisher_corner, _log, _FINITE_DIFF_STEPS,
    )

    res = box_params["res"]
    boxsize = box_params["boxsize"]
    dim = box_params["dim"]

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/fisher_5param_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    # --- Constant (field-independent) BFast precompute, hoisted out of jacrev ---
    bin_edges = jnp.arange(1, res // 3 + 1, pb_step)
    B_info, B_norm = _bfast_precompute(boxsize, bin_edges, res, dim=dim, mas_order=0)
    n_P_bins = int(len(bin_edges) - 1)
    n_triangles = int(np.asarray(B_info["triangle_centers"]).shape[0])
    n_P_blocks = 3 if pk_multipoles else 1          # P_0 (+ P_2, P_4)
    n_bk = n_triangles if joint_pb else 0           # B_0
    n_data = n_P_blocks * n_P_bins + n_bk

    _log(f"Data vector: {n_P_bins} P-bins x {n_P_blocks} multipole(s)"
         f"{f' + {n_triangles} triangles' if joint_pb else ''} = {n_data} "
         f"(pk_multipoles={pk_multipoles}, joint_pb={joint_pb})")
    _log(f"Empirical covariance needs n_cov_seeds > n_data + 2 = {n_data + 2} "
         f"(have {n_cov_seeds})")
    if n_cov_seeds <= n_data + 4:
        _log("WARNING: n_cov_seeds <= n_data + 4 -> Hartlap factor is small / "
             "non-positive. Increase --n-cov-seeds or coarsen --pb-step.")

    fid_values = jnp.array([_FIDUCIAL_5PARAM[k] for k in _SAMPLED_KEYS],
                           dtype=jnp.float32)
    n_params = len(_SAMPLED_KEYS)

    # Close the pipeline over the run constants -> (params, noise, sn_noise).
    def pipeline_fn(p, noise, sn_noise):
        return _pipeline_5param(
            p, noise, sn_noise, bin_edges, B_info, B_norm,
            box_params, sim_params, _FIXED_COSMO, n_g_density,
            pk_multipoles=pk_multipoles, joint_pb=joint_pb,
        )

    # --- Jacobian d d(theta)/d theta, seed-averaged ---
    if derivative_method == "fd":
        _log(f"Computing Jacobian via central finite differences over "
             f"{n_seeds} seeds (step_kind={fd_step_kind}) ...")
        steps = jnp.array([_FINITE_DIFF_STEPS[k] for k in _SAMPLED_KEYS],
                          dtype=jnp.float32)
        J_mean, J_all, data_fid, _kb = compute_jacobian_numerical(
            fid_values, pipeline_fn, steps, _SAMPLED_KEYS,
            n_seeds=n_seeds, n_bins=30, box_params=box_params,
            extra_noise_per_seed=True,
            step_kind=fd_step_kind, relative_eps=fd_relative_eps,
        )
    elif derivative_method == "autodiff":
        _log(f"Computing Jacobian via jax.jacrev over {n_seeds} seeds ...")
        J_mean, J_all, data_fid, _kb = compute_jacobian(
            fid_values, pipeline_fn,
            n_seeds=n_seeds, n_bins=30, box_params=box_params,
            extra_noise_per_seed=True, jacrev_chunk_size=jacrev_chunk_size,
        )
    elif derivative_method == "jacfwd":
        _log(f"Computing Jacobian via jax.jacfwd (forward mode, x64) over "
             f"{n_seeds} seeds -- captures full cosmo dependence incl. growth ...")
        def pipeline_fn_fwd(p, noise, sn_noise):
            return _pipeline_5param(
                p, noise, sn_noise, bin_edges, B_info, B_norm,
                box_params, sim_params, _FIXED_COSMO, n_g_density,
                pk_multipoles=pk_multipoles, joint_pb=joint_pb, forward_mode=True,
            )
        J_mean, J_all, data_fid, _kb = compute_jacobian_forward(
            fid_values, pipeline_fn_fwd,
            n_seeds=n_seeds, n_bins=30, box_params=box_params,
            extra_noise_per_seed=True,
        )
    else:
        raise ValueError(f"derivative_method must be 'fd', 'autodiff', or "
                         f"'jacfwd', got {derivative_method!r}")
    J_mean = np.asarray(J_mean, dtype=np.float64)     # (n_data, 5)
    data_fid = np.asarray(data_fid, dtype=np.float64)  # (n_data,)

    # --- Data covariance (cov_type branches) ---
    # Exactly one field block (field=fin), so this mirrors fisher_2param.py:
    #   diag       : analytic Gaussian (Grieb P_l + Scoccimarro B), no extra sims.
    #   euclid_cov : full empirical (Hartlap-corrected) cov of the whole vector.
    #   block_with_pb_cross_empirical : analytic P/B blocks + empirical P-B cross.
    n_pk = 3 if pk_multipoles else 1
    multipole_ls = (0, 2, 4) if pk_multipoles else (0,)
    n_P_total = n_pk * n_P_bins
    k_F = 2.0 * np.pi / boxsize
    be = np.asarray(bin_edges, dtype=np.float64)
    k_bins_P = 0.5 * (be[1:] + be[:-1]) * k_F
    bin_widths_P = (be[1:] - be[:-1]) * k_F
    V_box = float(boxsize) ** dim
    triangle_indices = np.asarray(B_info["triangle_indices"])
    N_modes = np.asarray(B_norm["Pk"], dtype=np.float64)

    P0_fid = data_fid[0:n_P_bins]
    P2_fid = data_fid[n_P_bins:2 * n_P_bins] if pk_multipoles else np.zeros_like(P0_fid)
    P4_fid = data_fid[2 * n_P_bins:3 * n_P_bins] if pk_multipoles else np.zeros_like(P0_fid)

    def _analytic_PP():
        grieb = np.asarray(multipole_covariance_grieb(
            jnp.asarray(P0_fid), jnp.asarray(P2_fid), jnp.asarray(P4_fid),
            jnp.asarray(N_modes), multipole_ls=multipole_ls,
        ))  # (n_k, n_l, n_l)
        cov_PP = np.zeros((n_P_total, n_P_total), dtype=np.float64)
        for la in range(n_pk):
            for lb in range(n_pk):
                sub = cov_PP[la*n_P_bins:(la+1)*n_P_bins, lb*n_P_bins:(lb+1)*n_P_bins]
                np.fill_diagonal(sub, grieb[:, la, lb])
        return cov_PP

    def _analytic_BB_diag():
        return np.asarray(bispectrum_covariance(
            jnp.asarray(P0_fid), jnp.asarray(triangle_indices),
            jnp.asarray(k_bins_P), jnp.asarray(bin_widths_P), V_box,
        ), dtype=np.float64)  # (n_triangles,)

    cov_emp = np.zeros((n_data, n_data), dtype=np.float64)
    mean_emp = data_fid.copy()
    hartlap = np.nan
    if cov_type == "diag":
        _log("=== Analytic Gaussian covariance (Grieb P_l + Scoccimarro B) ===")
        cov_emp[:n_P_total, :n_P_total] = _analytic_PP()
        if joint_pb:
            np.fill_diagonal(cov_emp[n_P_total:, n_P_total:], _analytic_BB_diag())
    elif cov_type in ("euclid_cov", "euclid_cov_no_pb_cross"):
        _log(f"=== Full empirical covariance over {n_cov_seeds} fiducial seeds "
             f"(cov_type={cov_type}) ===")
        cov_full, mean_full, hartlap = compute_empirical_covariance(
            pipeline_fn, fid_values, n_cov_seeds, box_params,
            extra_noise_per_seed=True,
        )
        cov_emp = np.array(cov_full, dtype=np.float64)  # writable copy (np.asarray of a JAX array is read-only)
        mean_emp = np.asarray(mean_full, dtype=np.float64)
        _log(f"Hartlap factor: {hartlap:.4f}")
        if cov_type == "euclid_cov_no_pb_cross":
            if not joint_pb:
                raise ValueError("cov_type='euclid_cov_no_pb_cross' requires "
                                 "joint_pb=True (needs a B_0 block to decouple).")
            cov_emp[:n_P_total, n_P_total:] = 0.0
            cov_emp[n_P_total:, :n_P_total] = 0.0
            _log("  P-B cross block zeroed (empirical P-P and B-B retained).")
    elif cov_type == "block_with_pb_cross_empirical":
        if not joint_pb:
            raise ValueError("cov_type='block_with_pb_cross_empirical' requires "
                             "joint_pb=True (needs a B_0 block).")
        _log(f"=== Block analytic + empirical P-B cross "
             f"({n_cov_seeds} fiducial seeds) ===")
        cov_pb = np.asarray(empirical_pb_cross_block(
            pipeline_fn, fid_values, n_cov_seeds, box_params,
            n_P_total=n_P_total, n_triangles=n_triangles,
            block_size=n_data, n_blocks=1, extra_noise_per_seed=True,
        )[0], dtype=np.float64)  # (n_P_total, n_triangles)
        cov_emp[:n_P_total, :n_P_total] = _analytic_PP()
        np.fill_diagonal(cov_emp[n_P_total:, n_P_total:], _analytic_BB_diag())
        cov_emp[:n_P_total, n_P_total:] = cov_pb
        cov_emp[n_P_total:, :n_P_total] = cov_pb.T
        _log(f"  |cov_PB|_max = {np.max(np.abs(cov_pb)):.3e}")
    else:
        raise ValueError(
            f"cov_type={cov_type!r} not supported (expected one of: "
            f"diag, euclid_cov, euclid_cov_no_pb_cross, "
            f"block_with_pb_cross_empirical)."
        )

    # --- Mask invalid data-vector entries, invert, assemble Fisher ---
    diag_cov = np.diag(cov_emp)
    valid = (np.isfinite(diag_cov) & (diag_cov > 0)
             & np.isfinite(data_fid) & np.all(np.isfinite(J_mean), axis=1))
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        raise RuntimeError("All data-vector entries invalid -- cannot build Fisher.")
    _log(f"Valid data-vector entries: {valid_idx.size}/{n_data}")
    cov_valid = cov_emp[np.ix_(valid_idx, valid_idx)]

    if cov_type == "block_with_pb_cross_empirical":
        n_block = int(cov_valid.shape[0])
        hartlap = float((n_cov_seeds - n_block - 2) / (n_cov_seeds - 1))
        if hartlap <= 0.0:
            raise ValueError(
                f"Hartlap factor non-positive (n_cov_seeds={n_cov_seeds}, "
                f"block size={n_block}); increase --n-cov-seeds.")
        _log(f"Hartlap factor (block dim {n_block}): {hartlap:.4f}")
    h = hartlap if np.isfinite(hartlap) else 1.0
    inv_cov_valid = h * np.linalg.inv(cov_valid)
    inv_cov = np.zeros((n_data, n_data), dtype=np.float64)
    inv_cov[np.ix_(valid_idx, valid_idx)] = inv_cov_valid
    J_valid = J_mean[valid_idx]

    # --- Fisher matrix: F = J^T C^{-1} J + F_prior ---
    F_data = np.asarray(fisher_matrix_full(J_valid, inv_cov_valid), dtype=np.float64)
    F_prior = prior_fisher_matrix(_SAMPLED_KEYS, _PRIOR_SPEC)
    F_total = F_data + F_prior
    fisher_cov = np.linalg.inv(F_total)

    sigmas = np.sqrt(np.diag(fisher_cov))
    fom = 1.0 / np.sqrt(np.linalg.det(fisher_cov))
    Dinv = np.diag(1.0 / sigmas)
    corr = Dinv @ fisher_cov @ Dinv          # full correlation matrix

    _log("=== Fisher forecast (5-param cosmology) ===")
    for i, k in enumerate(_SAMPLED_KEYS):
        fid = _FIDUCIAL_5PARAM[k]
        pct = 100.0 * sigmas[i] / abs(fid) if fid != 0 else float("nan")
        _log(f"  sigma({k:8s}) = {sigmas[i]:.5f}   (fiducial {fid}, {pct:.2f}%)")
    _log(f"  figure of merit = {fom:.3e}")

    # --- Save everything needed to draw the Fisher ellipses downstream ---
    results_path = os.path.join(output_dir, "fisher_results.npz")
    np.savez(
        results_path,
        F_data=F_data, F_prior=F_prior, F_total=F_total,
        fisher_cov=fisher_cov,
        jacobian=J_mean, jacobians_per_seed=np.asarray(J_all, dtype=np.float64),
        cov_emp=cov_emp, inv_cov=inv_cov, hartlap=hartlap,
        cov_type=cov_type,
        mean_emp=mean_emp, data_fid=data_fid,
        sigmas=sigmas, correlation=corr, figure_of_merit=fom,
        bin_edges=np.asarray(bin_edges),
        triangle_centers=np.asarray(B_info["triangle_centers"]),
        n_P_bins=n_P_bins, n_triangles=n_triangles, n_data=n_data,
        block_spans=_block_spans(n_P_bins, n_triangles, pk_multipoles, joint_pb),
        param_names=np.array(_SAMPLED_KEYS),
        fiducial_vals=np.array([_FIDUCIAL_5PARAM[k] for k in _SAMPLED_KEYS]),
        n_seeds=n_seeds, n_cov_seeds=n_cov_seeds, pb_step=pb_step,
        derivative_method=derivative_method,
        n_g_density=(n_g_density if n_g_density is not None else np.nan),
        res=res, boxsize=boxsize,
    )
    _log(f"Saved {results_path}")

    # Provenance: copy source config + dump resolved args.
    if config_path is not None and os.path.isfile(config_path):
        shutil.copy(config_path, os.path.join(output_dir, "source_config.yml"))
    if resolved_args is not None:
        import yaml
        with open(os.path.join(output_dir, "run_config.yml"), "w") as f:
            yaml.safe_dump({k: v for k, v in resolved_args.items()}, f,
                           default_flow_style=False, sort_keys=True)

    # Quick Fisher-only sanity corner (no SBI overlay; comparison done by user).
    try:
        plot_fisher_corner(
            fisher_cov=fisher_cov,
            param_names=[_LATEX[k] for k in _SAMPLED_KEYS],
            fiducial_vals=np.array([_FIDUCIAL_5PARAM[k] for k in _SAMPLED_KEYS]),
            output_dir=output_dir,
            prior_bounds=[_PRIOR_BOUNDS[k] for k in _SAMPLED_KEYS],
        )
    except Exception as e:  # plotting must never sink a finished forecast
        _log(f"WARNING: Fisher-only corner plot failed ({type(e).__name__}: {e})")

    return {
        "F_data": F_data, "F_prior": F_prior, "F_total": F_total,
        "fisher_cov": fisher_cov, "sigmas": sigmas, "correlation": corr,
        "figure_of_merit": fom, "output_dir": output_dir,
    }


def _build_parser():
    p = argparse.ArgumentParser(
        description="Fisher forecast for the 5 cosmological params "
                    "[Omega_m, Omega_b, h, n_s, sigma_8] on the fli_5param "
                    "observable [P_0, P_2, P_4, B_0] (final field, RSD + shot noise).",
    )
    p.add_argument("--config", type=str, default=None,
                   help="YAML file of defaults (keys = argparse dests with "
                        "dashes->underscores). CLI flags override it.")
    p.add_argument("--n-seeds", type=int, default=10,
                   help="White-noise seeds for Jacobian averaging (default 10).")
    p.add_argument("--n-cov-seeds", type=int, default=2400,
                   help="Fiducial seeds for the empirical covariance. Must "
                        "exceed n_data + 2 (ideally + 4). Default 2400.")
    p.add_argument("--pb-step", type=int, default=3,
                   help="Stride for BFast bin_edges = arange(1, res//3+1, step) "
                        "in units of k_F. Larger = coarser/faster, smaller "
                        "n_data. Default 3.")
    p.add_argument("--n-g-density", type=float, default=1.0e-3,
                   help="Gaussian shot-noise density (h/Mpc)^3 (default 1e-3, "
                        "DESI-like). Ignored if --no-shot-noise.")
    p.add_argument("--no-shot-noise", action="store_true",
                   help="Disable shot noise (sets n_g_density=None).")
    p.add_argument("--field", type=str, default="fin",
                   help="Field to summarise. Only 'fin' (final redshift-space "
                        "matter field) is supported by this pipeline.")
    p.add_argument("--pk-multipoles", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include P_2 and P_4 alongside P_0. --no-pk-multipoles "
                        "keeps only the monopole P_0. Default True.")
    p.add_argument("--joint-pb", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Append the bispectrum B_0 to the data vector. "
                        "--no-joint-pb drops it. Default True.")
    p.add_argument("--derivative-method", type=str, default="fd",
                   choices=["fd", "autodiff", "jacfwd"],
                   help="Jacobian method: 'fd' (central finite differences; "
                        "captures full cosmo dependence incl. growth), 'autodiff' "
                        "(jax.jacrev; cosmo stop_gradient'd, growth omitted), or "
                        "'jacfwd' (forward-mode AD + x64; full cosmo dependence "
                        "incl. growth at autodiff precision -- the correct method).")
    p.add_argument("--fd-step-kind", type=str, default="absolute",
                   choices=["absolute", "relative"],
                   help="FD step convention: 'absolute' (per-param dict, default) "
                        "or 'relative' (eps*|fid|). Only used with --derivative-method fd.")
    p.add_argument("--fd-relative-eps", type=float, default=1e-2,
                   help="Relative FD step amplitude when --fd-step-kind relative "
                        "(default 1e-2).")
    p.add_argument("--cov-type", type=str, default="euclid_cov",
                   choices=["diag", "euclid_cov", "euclid_cov_no_pb_cross",
                            "block_with_pb_cross_empirical"],
                   help="Data covariance model (default 'euclid_cov', full "
                        "empirical Hartlap-corrected).")
    p.add_argument("--res", type=int, default=128,
                   help="Resolution (default 128, matching fli_5param).")
    p.add_argument("--boxsize", type=float, default=1000.0,
                   help="Box size in Mpc/h (default 1000).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output dir (default outputs/fisher_5param_<timestamp>).")
    p.add_argument("--jacrev-chunk-size", type=int, default=None,
                   help="If set, chunk jax.jacrev over output basis vectors "
                        "(memory-bounded). Only used with --derivative-method autodiff.")
    return p


def _load_config(parser, config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        parser.error(f"--config {config_path}: expected a YAML mapping, got "
                     f"{type(cfg).__name__}")
    known = {a.dest for a in parser._actions}
    unknown = sorted(set(cfg) - known)
    if unknown:
        parser.error(f"--config {config_path}: unknown keys {unknown}. "
                     f"Allowed: {sorted(known - {'help', 'config'})}")
    parser.set_defaults(**cfg)
    return cfg


def main():
    parser = _build_parser()
    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        _load_config(parser, pre_args.config)
    args = parser.parse_args()

    if args.pb_step < 1:
        parser.error("--pb-step must be >= 1")
    if args.field != "fin":
        parser.error("--field: only 'fin' is supported (the pipeline always "
                     "builds the final redshift-space matter field).")
    if not args.no_shot_noise and args.n_g_density is not None \
            and args.n_g_density <= 0.0:
        parser.error("--n-g-density must be positive")
    if args.cov_type in ("block_with_pb_cross_empirical",
                         "euclid_cov_no_pb_cross") and not args.joint_pb:
        parser.error(f"--cov-type {args.cov_type} requires --joint-pb "
                     "(needs a B_0 block in the data vector).")

    # jacfwd needs float64 for the DiscoDJ growth ODE (float32 -> NaN tangent).
    # Must be set before JAX initialises (i.e. before importing model.model).
    if args.derivative_method == "jacfwd":
        import jax
        jax.config.update("jax_enable_x64", True)
        print("  [jacfwd] enabled jax_enable_x64=True (growth ODE needs float64).")

    # Import after parsing (JAX init is slow); this also sets CUDA_VISIBLE_DEVICES.
    from den_pie.forward.model import BOX_PARAMS, SIM_PARAMS

    box_params = dict(BOX_PARAMS)
    sim_params = dict(SIM_PARAMS)
    box_params["res"] = args.res
    box_params["boxsize"] = args.boxsize
    sim_params["res_pm"] = 2 * args.res

    n_g_density = None if args.no_shot_noise else args.n_g_density

    print("=== Fisher 5-param run config ===")
    if args.config is not None:
        print(f"  (loaded from --config {args.config})")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  n_g_density (effective): {n_g_density}")
    print(f"  box_params: {box_params}")
    print(f"  sim_params: {sim_params}")
    print("=================================")

    t0 = time.time()
    run_fisher_5param(
        n_seeds=args.n_seeds,
        n_cov_seeds=args.n_cov_seeds,
        pb_step=args.pb_step,
        n_g_density=n_g_density,
        output_dir=args.output_dir,
        box_params=box_params,
        sim_params=sim_params,
        derivative_method=args.derivative_method,
        jacrev_chunk_size=args.jacrev_chunk_size,
        fd_step_kind=args.fd_step_kind,
        fd_relative_eps=args.fd_relative_eps,
        pk_multipoles=args.pk_multipoles,
        joint_pb=args.joint_pb,
        cov_type=args.cov_type,
        config_path=args.config,
        resolved_args={k: v for k, v in vars(args).items() if k != "config"},
    )
    elapsed = time.time() - t0
    print(f"Total Fisher computation time: {int(elapsed // 3600)}h "
          f"{int(elapsed % 3600 // 60)}m {int(elapsed % 60)}s ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

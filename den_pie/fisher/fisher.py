"""
Fisher forecast from DiscoDJ via JAX autodiff.

Computes the Fisher information matrix for the non-linear power spectrum P_NL(k)
w.r.t. cosmological parameters [Omega_m, Omega_b, h, n_s, sigma_8] by
differentiating through the full N-body pipeline:

    cosmo_params -> P_lin(k) -> delta_ic -> delta_fin -> P_NL(k)

Primary method: jax.jacrev (reverse-mode AD) for exact Jacobians.
DiscoDJ's N-body has a custom_vjp (reverse-mode rule) but no JVP rule, so
forward-mode AD through it returns all-NaN. Reverse-mode pulls cotangents
back through the working VJP path.
Fallback: central finite differences if jacrev produces all NaN.
Includes a JVP diagnostic to pinpoint where tangent propagation breaks.

White noise is generated outside the differentiated function and held fixed,
making the pipeline deterministic and fully differentiable.

The Fisher matrix is:
    F_ij = sum_a (dP(k_a)/dtheta_i) * (1/sigma^2_a) * (dP(k_a)/dtheta_j)
with Gaussian covariance sigma^2(k_a) = 2 P(k_a)^2 / N_modes(k_a).
"""

import os
import shutil
import time as _time
from datetime import datetime
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import jax
import jax.numpy as jnp
from discodj import DiscoDJ
from discodj.core.summary_statistics import power_spectrum, power_spectrum_core
from discodj.core.grids import get_fourier_grid

from den_pie.forward.model import (
    _cosmo_dict, _FIDUCIAL_COSMO, _COSMO_PARAM_ORDER,
    BOX_PARAMS, SIM_PARAMS,
)
from den_pie.forward.bias import lagrangian_bias_weights


def _log(msg):
    """Print ``msg`` with a leading ``[HH:MM:SS]`` timestamp."""
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Default prior bounds (Quijote Latin Hypercube / config_5param.yml ranges)
_PRIOR_BOUNDS = {
    "Omega_m": (0.10, 0.50),
    "Omega_b": (0.03, 0.07),
    "h":       (0.50, 0.90),
    "n_s":     (0.80, 1.20),
    "sigma8":  (0.60, 1.00),
}


def _add_shot_noise(field, sn_noise, n_g_density, boxsize, res):
    """Add Gaussian shot noise to a density-contrast field.

    Poisson Gaussian limit (arXiv:2504.20130v2):

        delta_g(x) = delta_biased(x) + epsilon(x),
        epsilon ~ N(0, 1/N̄_g)  per voxel,
        N̄_g    = n_g_density · V_cell,  V_cell = (boxsize / res)**3.

    No-op if either ``sn_noise`` or ``n_g_density`` is None (so the SN feature
    is fully optional and existing call-sites keep working unchanged).

    ``sn_noise`` is a standard-normal JAX array (held fixed per seed in the
    Fisher driver) so the Jacobian w.r.t. params is unaffected.
    """
    if sn_noise is None or n_g_density is None:
        return field
    V_cell  = (float(boxsize) / int(res)) ** 3
    sigma_n = 1.0 / jnp.sqrt(jnp.asarray(n_g_density, dtype=field.dtype) * V_cell)
    return field + sigma_n * sn_noise


def _power_spectrum_safe(field, cellsize, bins=30, dtype_num=32):
    """JAX-native shell-binned P(k) from a real-space field.

    Replaces DiscoDJ's ``power_spectrum_core``. Computes |FFT|^2 via
    ``(z * conj(z)).real`` (AD-safe) and bins it in log-spaced k shells
    using ``jnp.bincount`` + ``jnp.digitize``. Empty bins return 0 with
    NaN-free gradients via the standard ``jnp.where(N>0, ..., default)``
    recipe.

    Returns ``(k_bins, Pk, None)`` to match the legacy 3-tuple signature.

    Conventions (must match :func:`count_modes` so k_bins / N_modes line up):
    - Log-spaced edges from ``2π/L`` to ``π/cellsize``.
    - rFFT layout: interior of last axis counted twice.
    - Normalization ``(L^dim) / (res^dim)^2`` matching
      ``discodj/core/summary_statistics.py:power_spectrum_core`` line 72.
    """
    dtype = jnp.float64 if dtype_num == 64 else jnp.float32
    field = jnp.asarray(field, dtype=dtype)
    dim = field.ndim
    shape = field.shape
    res = shape[0]
    boxsize = res * cellsize

    fk = jnp.fft.rfftn(field)
    Pgrid = (fk * jnp.conj(fk)).real  # AD-safe |FFT|^2

    rfft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    kmag = get_fourier_grid(
        list(shape), boxsize, dtype_num=dtype_num, with_jax=True, full=False,
    )["|k|"].reshape(-1)

    rfft_factor = jnp.ones(rfft_shape, dtype=dtype)
    rfft_factor = rfft_factor.at[..., 1:-1].set(2.0)
    rfft_factor = rfft_factor.reshape(-1)

    kmin = 2 * jnp.pi / boxsize
    kmax = jnp.pi / cellsize
    dlogk = (jnp.log10(kmax) - jnp.log10(kmin)) / (bins - 1)
    kedges = jnp.geomspace(
        kmin * 10 ** (-dlogk / 2), kmax * 10 ** (dlogk / 2),
        bins + 1, dtype=dtype,
    )
    dig = jnp.digitize(kmag, kedges)

    norm = (boxsize ** dim) / (res ** dim) ** 2
    Pgrid_flat = Pgrid.reshape(-1) * norm

    N_modes = jnp.bincount(dig, length=bins + 2, weights=rfft_factor)
    P_sum = jnp.bincount(dig, length=bins + 2, weights=rfft_factor * Pgrid_flat)
    k_sum = jnp.bincount(dig, length=bins + 2, weights=rfft_factor * kmag.astype(dtype))

    safe_N = jnp.where(N_modes > 0, N_modes, jnp.ones_like(N_modes))
    Pk = jnp.where(N_modes > 0, P_sum / safe_N, jnp.zeros_like(P_sum))
    k_bins = jnp.where(N_modes > 0, k_sum / safe_N, jnp.zeros_like(k_sum))

    return k_bins[1:-1], Pk[1:-1], None


def _bfast_precompute(boxsize, bin_edges, res, dim=3, mas_order=0):
    """One-time pre-computation for BFast joint P+B Fisher.

    Returns ``(B_info, B_norm)`` — both independent of the input field, so
    they can be hoisted outside the differentiated pipeline:

    - ``B_info`` is the triangle list (centres + bin-index triplets) for the
      given ``bin_edges`` (linear, in units of k_F = 2π/L).
    - ``B_norm`` is the BFast normalisation dict (``compute_norm=True``).
      ``B_norm["Pk"]`` is the per-bin mode count (``N_modes``) and
      ``B_norm["Bk"] * res**dim`` is the per-triangle mode-triplet count
      (``N_triangles``).

    The field passed to ``compute_norm=True`` is replaced by an all-ones
    Fourier field inside BFast, so any same-shape dummy works.
    """
    from BFast.core.bispectrum import (
        bispectrum as _bfast_bispectrum,
        get_triangles as _bfast_get_triangles,
    )
    B_info = _bfast_get_triangles(bin_edges, open_triangles=True)
    dummy = jnp.zeros((res,) * dim, dtype=jnp.float32)
    B_norm = _bfast_bispectrum(
        dummy, boxsize=float(boxsize), bin_edges=bin_edges,
        triangle_centers=B_info["triangle_centers"],
        triangle_indices=B_info["triangle_indices"],
        mas_order=mas_order, fast=True, only_B=False, compute_norm=True,
    )
    return B_info, B_norm


def _bfast_pk_bk_safe(field, boxsize, bin_edges, B_info, B_norm, mas_order=0):
    """JAX-AD-safe joint P(k) + B(k1,k2,k3) from a real-space field via BFast.

    Differentiable end-to-end: BFast's pure-JAX FFT pipeline (with custom
    VJPs on the rFFTs) propagates cotangents through both summary stats.

    ``B_info`` and ``B_norm`` are the constant outputs of ``_bfast_precompute``
    — passed in to avoid recomputing the (field-independent) normalisation
    inside every per-seed Jacobian call.

    ``mas_order=0`` because the input field is already CIC-deconvolved by
    DiscoDJ's ``compute_field_quantity_from_particles(..., deconvolve=True)``.

    Returns
    -------
    Pk : jnp.array, shape (n_P_bins,)         Power spectrum on bin_edges
    Bk : jnp.array, shape (n_triangles,)      Bispectrum on B_info["triangle_indices"]
    """
    from BFast.core.bispectrum import bispectrum as _bfast_bispectrum
    out = _bfast_bispectrum(
        field, boxsize=float(boxsize), bin_edges=bin_edges,
        triangle_centers=B_info["triangle_centers"],
        triangle_indices=B_info["triangle_indices"],
        mas_order=mas_order, fast=True, only_B=False, compute_norm=False,
    )
    Pk = out["Pk"] / B_norm["Pk"]
    Bk = out["Bk"] / B_norm["Bk"]
    return Pk, Bk


def _bfast_pk_multi_bk_safe(field, boxsize, bin_edges, B_info, B_norm,
                            multipole_axis=2, mas_order=0):
    """JAX-AD-safe [P_0(k), P_2(k), P_4(k), B(k1,k2,k3)] from a real- or
    redshift-space density-contrast field.

    Two BFast calls (one Pk-multipoles, one Bk-monopole) since BFast's
    bispectrum helper has no multipole_axis arg. Both are pure-JAX and
    AD-traceable through BFast's custom-VJP rFFTs.

    ``BFast.core.powerspectrum.powerspectrum`` already normalises
    internally (``Pk_unnorm / counts * V_cell``); only the bispectrum
    needs the ``B_norm["Bk"]`` division (matches ``_bfast_pk_bk_safe``).

    Returns
    -------
    Pk0, Pk2, Pk4 : jnp.array, shape (n_P_bins,)
    Bk            : jnp.array, shape (n_triangles,)
    """
    from BFast.core.powerspectrum import powerspectrum as _bfast_pk
    from BFast.core.bispectrum    import bispectrum    as _bfast_bk
    pk_out = _bfast_pk(
        field, boxsize=float(boxsize), bin_edges=bin_edges,
        mas_order=mas_order, multipole_axis=multipole_axis,
    )
    bk_out = _bfast_bk(
        field, boxsize=float(boxsize), bin_edges=bin_edges,
        triangle_centers=B_info["triangle_centers"],
        triangle_indices=B_info["triangle_indices"],
        mas_order=mas_order, fast=True, only_B=True, compute_norm=False,
    )
    return pk_out["Pk0"], pk_out["Pk2"], pk_out["Pk4"], bk_out["Bk"] / B_norm["Bk"]


# ---------------------------------------------------------------------------
# 1. Differentiable pipeline: cosmo_params -> P_NL(k)
# ---------------------------------------------------------------------------

def _differentiable_pipeline_fin(
    cosmo_params,
    white_noise,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    n_bins=30,
    device=None,
):
    """Single differentiable function: cosmo_params -> P_NL(k).

    Parameters
    ----------
    cosmo_params : jnp.array, shape (N_sampled,)
        Cosmological parameters (all 5 or a subset).
    white_noise : jnp.array, shape (res, res, res)
        Fixed white-noise realization (generated outside, not differentiated).
    box_params : dict
        Box geometry (dim, res, boxsize).
    sim_params : dict
        Simulation parameters.
    fixed_cosmo_params : dict or None
        Fixed cosmological parameters (if any).
    n_bins : int
        Number of P(k) bins.
    device : jax device or None
        Device for N-body. If None, uses first available.

    Returns
    -------
    Pk_nl : jnp.array, shape (n_bins,)
        Binned non-linear power spectrum.
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]
    sp = sim_params

    if device is None:
        device = jax.devices()[0]

    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum from cosmological parameters
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=False)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic from white noise + P_lin
    # Replicate grf.py real-space formula with inv_laplace_kernel cancelling -k^2:
    #   fphi = rfftn(noise) * sqrt(Pk * norm) * inv_laplace_kernel
    #   delta_ini = irfftn(-k^2 * fphi) = irfftn(rfftn(noise) * sqrt(Pk * norm))
    # At a=1, Dplus(1)=1 (DiscoDJ normalises P(k) at z=0), so delta_ic = delta_ini.
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: N-body simulation -> delta_fin
    # NOTE: cosmo is stop_gradient'd here. DiscoDJ's reverse-mode adjoint
    # produces NaN cotangents for the cosmology-pytree path through Dplus/
    # Fplus / time tables. We keep the cosmo dependence through delta_ic
    # (which already carries the full P_lin gradient) and treat the
    # growth-factor / time-evolution channel as a fixed approximation.
    cosmo_nbody = jax.lax.stop_gradient(cosmo)
    dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                   cosmo=cosmo_nbody, requires_jacfwd=False)
    dj2 = dj2.with_timetables()
    dj2 = dj2.with_external_ics(delta=delta_ic)
    dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

    _skip = {"nlpt_order_ics", "worder"}
    run_params = {k: v for k, v in sp.items() if k not in _skip}

    X_sim, _, _ = dj2.run_nbody(**run_params, use_diffrax=False)
    delta_fin = dj2.get_delta_from_pos(
        X_sim, res=res, worder=sp["worder"], antialias=1, deconvolve=True,
    )

    # Step 4: Measure P(k) from delta_fin (AD-safe)
    cellsize = boxsize / res
    _k_bins, Pk_nl, _ = _power_spectrum_safe(
        delta_fin, cellsize=cellsize, bins=n_bins, dtype_num=32,
    )
    return Pk_nl


# ---------------------------------------------------------------------------
# 1a. Bias pipeline constants + differentiable function
# ---------------------------------------------------------------------------

_FIDUCIAL_BIAS = {"b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0}
_BIAS_PARAM_ORDER = ["b1", "b2", "bs2", "bn2"]
_PRIOR_BOUNDS_BIAS = {
    "Omega_m": (0.10, 0.50),
    "sigma8":  (0.60, 1.00),
    "b1": (-0.5, 2.5),
    "b2": (-4.0, 4.0),
    "bs2": (-4.0, 4.0),
    "bn2": (-4.0, 4.0),
}
_PRIOR_SPEC_BIAS = {
    "Omega_m": ("uniform", 0.10, 0.50),
    "sigma8":  ("uniform", 0.60, 1.00),
    "b1":      ("normal",  1.0,  0.5),
    "b2":      ("normal",  0.0,  2.0),
    "bs2":     ("normal",  0.0,  2.0),
    "bn2":     ("normal",  0.0,  2.0),
}
_PRIOR_SPEC = {
    "Omega_m": ("uniform", 0.10, 0.50),
    "Omega_b": ("uniform", 0.03, 0.07),
    "h":       ("uniform", 0.50, 0.90),
    "n_s":     ("uniform", 0.80, 1.20),
    "sigma8":  ("uniform", 0.60, 1.00),
}
_FINITE_DIFF_STEPS_BIAS = {
    "Omega_m": 0.01,
    "sigma8": 0.015,
    "b1": 0.05,
    "b2": 0.1,
    "bs2": 0.1,
    "bn2": 0.1,
}


def _differentiable_pipeline_fin_bias(
    params,
    white_noise,
    k_vecs,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    n_bins=30,
    device=None,
    sn_noise=None,
    n_g_density=None,
):
    """Differentiable pipeline for biased tracers: params -> P_biased(k).

    Parameters
    ----------
    params : jnp.array, shape (6,)
        Combined parameters [Omega_m, sigma8, b1, b2, bs2, bn2].
    white_noise : jnp.array, shape (res, res, res)
        Fixed white-noise realization.
    k_vecs : list of 3 sparse JAX arrays
        Pre-computed Fourier wavenumber vectors from DiscoDJ.k_vecs.
    box_params : dict
    sim_params : dict
    fixed_cosmo_params : dict or None
        Fixed cosmological parameters (Omega_b, h, n_s).
    n_bins : int
    device : jax device or None

    Returns
    -------
    Pk_biased : jnp.array, shape (n_bins,)
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]
    sp = sim_params

    if device is None:
        device = jax.devices()[0]

    # Unpack: first 2 are cosmo, last 4 are bias
    cosmo_params = params[:2]
    b1, b2, bs2, bn2 = params[2], params[3], params[4], params[5]

    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=False)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: N-body simulation
    # See note in _differentiable_pipeline_fin: cosmo is stop_gradient'd
    # for the N-body call to bypass DiscoDJ's NaN-producing reverse-mode
    # path through cosmo.Dplus / Fplus / time tables. Cosmo gradient still
    # flows via delta_ic (P_lin → IC), and bias gradients flow via weights.
    cosmo_nbody = jax.lax.stop_gradient(cosmo)
    dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                   cosmo=cosmo_nbody, requires_jacfwd=False)
    dj2 = dj2.with_timetables()
    dj2 = dj2.with_external_ics(delta=delta_ic)
    dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

    _skip = {"nlpt_order_ics", "worder"}
    run_params = {k: v for k, v in sp.items() if k not in _skip}

    X_sim, _, _ = dj2.run_nbody(**run_params, use_diffrax=False)

    # Step 3b: Compute Lagrangian bias weights
    weights = lagrangian_bias_weights(
        delta_ic, k_vecs, b1=b1, b2=b2, bs2=bs2, bn2=bn2,
    )

    # Step 3c: Scatter bias-weighted particles onto Eulerian mesh
    n_biased = dj2.compute_field_quantity_from_particles(
        pos=X_sim,
        quantity=weights.reshape(-1).astype(X_sim.dtype),
        normalize_by_density=False,
        in_redshift_space=False,
        worder=sp["worder"],
        antialias=1,
        deconvolve=True,
    )
    delta_biased = n_biased / n_biased.mean() - 1.0

    # Optional shot noise (Poisson Gaussian limit; see _add_shot_noise).
    delta_biased = _add_shot_noise(
        delta_biased, sn_noise, n_g_density, boxsize, res,
    )

    # Step 4: Measure P(k) from delta_biased (AD-safe)
    cellsize = boxsize / res
    _k_bins, Pk_biased, _ = _power_spectrum_safe(
        delta_biased, cellsize=cellsize, bins=n_bins, dtype_num=32,
    )
    return Pk_biased


def _differentiable_pipeline_fin_bias_PB(
    params,
    white_noise,
    k_vecs,
    bin_edges,
    B_info,
    B_norm,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    device=None,
    sn_noise=None,
    n_g_density=None,
    forward_mode=False,
):
    """Differentiable pipeline for biased tracers: params -> [P(k), B(k1,k2,k3)].

    Identical to ``_differentiable_pipeline_fin_bias`` through Eulerian
    ``delta_biased``; the final step is replaced with a BFast joint P+B
    summary, returning a concatenated data vector for the joint Fisher
    forecast.

    Parameters
    ----------
    params : jnp.array, shape (6,)
        [Omega_m, sigma8, b1, b2, bs2, bn2].
    white_noise : jnp.array, shape (res, res, res)
    k_vecs : list of 3 sparse JAX arrays
    bin_edges : jnp.array
        BFast k-bin edges in units of k_F = 2π/L (linear; integer-valued).
    B_info, B_norm : dict
        Pre-computed via ``_bfast_precompute`` (field-independent constants).
    box_params, sim_params : dict
    fixed_cosmo_params : dict or None
    device : jax device or None

    Returns
    -------
    data : jnp.array, shape (n_P_bins + n_triangles,)
        Concatenated [Pk_biased, Bk_biased].
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]
    sp = sim_params

    if device is None:
        device = jax.devices()[0]

    cosmo_params = params[:2]
    b1, b2, bs2, bn2 = params[2], params[3], params[4], params[5]

    req_jf = bool(forward_mode)
    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=req_jf)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: N-body. Reverse/FD: cosmo stop_gradient'd (custom_vjp drops it);
    # forward_mode: cosmo flows so jacfwd captures the full growth dependence.
    cosmo_nbody = cosmo if forward_mode else jax.lax.stop_gradient(cosmo)
    dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                   cosmo=cosmo_nbody, requires_jacfwd=req_jf)
    dj2 = dj2.with_timetables()
    dj2 = dj2.with_external_ics(delta=delta_ic)
    dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

    _skip = {"nlpt_order_ics", "worder"}
    run_params = {k: v for k, v in sp.items() if k not in _skip}

    X_sim, _, _ = dj2.run_nbody(**run_params, use_diffrax=False)

    # Step 3b: Lagrangian bias weights
    weights = lagrangian_bias_weights(
        delta_ic, k_vecs, b1=b1, b2=b2, bs2=bs2, bn2=bn2,
    )

    # Step 3c: Scatter onto Eulerian mesh (CIC + deconvolve)
    n_biased = dj2.compute_field_quantity_from_particles(
        pos=X_sim,
        quantity=weights.reshape(-1).astype(X_sim.dtype),
        normalize_by_density=False,
        in_redshift_space=False,
        worder=sp["worder"],
        antialias=1,
        deconvolve=True,
    )
    delta_biased = n_biased / n_biased.mean() - 1.0

    # Optional shot noise (Poisson Gaussian limit; see _add_shot_noise).
    delta_biased = _add_shot_noise(
        delta_biased, sn_noise, n_g_density, boxsize, res,
    )

    # Step 4: Joint P(k) + B(k1,k2,k3) via BFast (mas_order=0: already deconvolved)
    Pk, Bk = _bfast_pk_bk_safe(
        delta_biased, boxsize=boxsize, bin_edges=bin_edges,
        B_info=B_info, B_norm=B_norm, mas_order=0,
    )
    return jnp.concatenate([Pk, Bk])


def _differentiable_pipeline_fin_bias_PB_rsd_multi(
    params,
    white_noise,
    k_vecs,
    bin_edges,
    B_info,
    B_norm,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    device=None,
    sn_noise=None,
    n_g_density=None,
    forward_mode=False,
):
    """RSD multipole variant: params -> [P_0, P_2, P_4, B] of delta_RSD.

    ``forward_mode`` (for jax.jacfwd): when True, build the N-body with
    ``requires_jacfwd=True`` and do NOT stop_gradient ``cosmo``, so forward-mode
    AD captures the full cosmo dependence incl. N-body growth (requires x64).

    Identical to ``_differentiable_pipeline_fin_bias_PB`` through the
    Lagrangian bias weights; the scatter call enables redshift-space
    distortions (vel = canonical momenta from run_nbody, los_axis = 2)
    and the summary is the joint Pk-multipoles + Bk-monopole estimator.

    Returns
    -------
    data : jnp.array, shape (3 * n_P_bins + n_triangles,)
        Concatenated [Pk0, Pk2, Pk4, Bk] of the redshift-space biased
        Eulerian field.
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]
    sp = sim_params

    if device is None:
        device = jax.devices()[0]

    cosmo_params = params[:2]
    b1, b2, bs2, bn2 = params[2], params[3], params[4], params[5]

    req_jf = bool(forward_mode)
    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=req_jf)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: N-body. Reverse/FD: cosmo stop_gradient'd (custom_vjp drops it);
    # forward_mode: cosmo flows so jacfwd captures the full growth dependence.
    cosmo_nbody = cosmo if forward_mode else jax.lax.stop_gradient(cosmo)
    dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                   cosmo=cosmo_nbody, requires_jacfwd=req_jf)
    dj2 = dj2.with_timetables()
    dj2 = dj2.with_external_ics(delta=delta_ic)
    dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

    _skip = {"nlpt_order_ics", "worder"}
    run_params = {k: v for k, v in sp.items() if k not in _skip}

    # Capture canonical momenta (P_mom) for RSD displacement
    X_sim, P_mom, _ = dj2.run_nbody(**run_params, use_diffrax=False)
    P_flat = P_mom.reshape(-1, 3)

    # Step 3b: Lagrangian bias weights
    weights = lagrangian_bias_weights(
        delta_ic, k_vecs, b1=b1, b2=b2, bs2=bs2, bn2=bn2,
    )

    # Step 3c: Scatter onto Eulerian mesh in redshift space
    # los_axis hard-coded to 2 (z); particles displaced by v_los/(aH).
    n_biased = dj2.compute_field_quantity_from_particles(
        pos=X_sim,
        quantity=weights.reshape(-1).astype(X_sim.dtype),
        normalize_by_density=False,
        in_redshift_space=True,
        vel=P_flat,
        a=sp["a_end"],
        radial_dim=2,
        worder=sp["worder"],
        antialias=1,
        deconvolve=True,
    )
    delta_biased = n_biased / n_biased.mean() - 1.0

    # Optional shot noise (Poisson Gaussian limit; see _add_shot_noise).
    delta_biased = _add_shot_noise(
        delta_biased, sn_noise, n_g_density, boxsize, res,
    )

    # Step 4: Joint P_l(k) + B(k1,k2,k3) via BFast (mas_order=0: already deconvolved)
    Pk0, Pk2, Pk4, Bk = _bfast_pk_multi_bk_safe(
        delta_biased, boxsize=boxsize, bin_edges=bin_edges,
        B_info=B_info, B_norm=B_norm, multipole_axis=2, mas_order=0,
    )
    return jnp.concatenate([Pk0, Pk2, Pk4, Bk])


# ---------------------------------------------------------------------------
# 1b. IC pipeline (non-biased): cosmo_params -> P(delta_ic)
# ---------------------------------------------------------------------------

def _differentiable_pipeline_ic(
    cosmo_params,
    white_noise,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    n_bins=30,
    device=None,
):
    """Differentiable pipeline: cosmo_params -> P(delta_ic).

    Stages 1-2 of the fin pipeline (linear P(k) and IC generation from white
    noise), then measures P(delta_ic). No N-body advection.

    Parameters
    ----------
    cosmo_params : jnp.array, shape (N_sampled,)
    white_noise : jnp.array, shape (res, res, res)
    box_params, sim_params : dict
    fixed_cosmo_params : dict or None
    n_bins : int
    device : jax device or None
        Unused for IC (no N-body); kept in signature for call-site symmetry
        with the fin pipelines.

    Returns
    -------
    Pk_ic : jnp.array, shape (n_bins,)
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]

    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=False)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: Measure P(delta_ic) (AD-safe)
    cellsize = boxsize / res
    _k_bins, Pk_ic, _ = _power_spectrum_safe(
        delta_ic, cellsize=cellsize, bins=n_bins, dtype_num=32,
    )
    return Pk_ic


# ---------------------------------------------------------------------------
# 1c. IC pipeline (biased): params -> P(Lagrangian bias expansion field)
# ---------------------------------------------------------------------------

def _differentiable_pipeline_ic_bias(
    params,
    white_noise,
    k_vecs,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    n_bins=30,
    device=None,
    sn_noise=None,
    n_g_density=None,
):
    """Differentiable pipeline for the Lagrangian biased IC field:
    params -> P(delta_L), where

        delta_L(q) = w(q) - 1
                   = b1·delta_ic(q)
                   + b2 ·[delta_ic(q)^2 - <delta_ic^2>]
                   + bs2·[s^2(q)        - <s^2>]
                   + bn2·(nabla^2 delta_ic)(q)

    is the Lagrangian bias expansion field (no N-body advection). The weight
    w(q) is produced by ``lagrangian_bias_weights`` (core.model.bias).

    Parameters
    ----------
    params : jnp.array, shape (6,)
        [Omega_m, sigma8, b1, b2, bs2, bn2].
    white_noise : jnp.array, shape (res, res, res)
    k_vecs : list of 3 sparse JAX arrays
        Pre-computed from DiscoDJ.k_vecs.
    box_params, sim_params : dict
    fixed_cosmo_params : dict or None
    n_bins : int
    device : jax device or None
        Unused for IC.

    Returns
    -------
    Pk_L : jnp.array, shape (n_bins,)
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]

    # Unpack: first 2 are cosmo, last 4 are bias
    cosmo_params = params[:2]
    b1, b2, bs2, bn2 = params[2], params[3], params[4], params[5]

    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=False)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: Lagrangian bias expansion field (on the Lagrangian grid)
    weights = lagrangian_bias_weights(
        delta_ic, k_vecs, b1=b1, b2=b2, bs2=bs2, bn2=bn2,
    )
    delta_L = weights - 1.0

    # Optional shot noise applied at the Lagrangian level (treats delta_L like
    # the observed galaxy field; same prescription as on delta_biased so the
    # noise model is consistent across --field {fin,ic,both}).
    delta_L = _add_shot_noise(delta_L, sn_noise, n_g_density, boxsize, res)

    # Step 4: Measure P(delta_L) (AD-safe)
    cellsize = boxsize / res
    _k_bins, Pk_L, _ = _power_spectrum_safe(
        delta_L, cellsize=cellsize, bins=n_bins, dtype_num=32,
    )
    return Pk_L


def _differentiable_pipeline_ic_bias_PB(
    params,
    white_noise,
    k_vecs,
    bin_edges,
    B_info,
    B_norm,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    device=None,
    sn_noise=None,
    n_g_density=None,
):
    """Differentiable pipeline for the Lagrangian biased IC field:
    params -> [P(delta_L), B(delta_L)].

    Identical to ``_differentiable_pipeline_ic_bias`` through delta_L; the
    final ``_power_spectrum_safe`` call is replaced with a BFast joint P+B
    summary, returning a concatenated data vector for the joint Fisher
    forecast. No N-body advection.

    Returns
    -------
    data : jnp.array, shape (n_P_bins + n_triangles,)
        Concatenated [Pk_L, Bk_L].
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]

    cosmo_params = params[:2]
    b1, b2, bs2, bn2 = params[2], params[3], params[4], params[5]

    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=False)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: Lagrangian bias expansion field on the Lagrangian grid
    weights = lagrangian_bias_weights(
        delta_ic, k_vecs, b1=b1, b2=b2, bs2=bs2, bn2=bn2,
    )
    delta_L = weights - 1.0

    # Optional shot noise (same prescription as on delta_biased).
    delta_L = _add_shot_noise(delta_L, sn_noise, n_g_density, boxsize, res)

    # Step 4: Joint P(k) + B(k1,k2,k3) via BFast (mas_order=0: pure FFT field)
    Pk, Bk = _bfast_pk_bk_safe(
        delta_L, boxsize=boxsize, bin_edges=bin_edges,
        B_info=B_info, B_norm=B_norm, mas_order=0,
    )
    return jnp.concatenate([Pk, Bk])


def _differentiable_pipeline_both_bias_PB(
    params,
    white_noise,
    k_vecs,
    bin_edges,
    B_info,
    B_norm,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    device=None,
    sn_noise_ic=None,
    sn_noise_fin=None,
    n_g_density=None,
):
    """Differentiable joint pipeline: params -> [P_ic, B_ic, P_fin, B_fin].

    Runs the bias N-body once and emits both the Lagrangian-biased
    summary (on delta_L = w(q)-1) and the Eulerian-biased summary (on
    the post-N-body, CIC-deconvolved delta_biased). The two BFast calls
    share bin_edges/B_info/B_norm.

    Returns
    -------
    data : jnp.array, shape (2 * (n_P_bins + n_triangles),)
        Concatenated [Pk_ic, Bk_ic, Pk_fin, Bk_fin].
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]
    sp = sim_params

    if device is None:
        device = jax.devices()[0]

    cosmo_params = params[:2]
    b1, b2, bs2, bn2 = params[2], params[3], params[4], params[5]

    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=False)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: Lagrangian bias weights — shared between IC and FIN summaries
    weights = lagrangian_bias_weights(
        delta_ic, k_vecs, b1=b1, b2=b2, bs2=bs2, bn2=bn2,
    )

    # Step 4: IC-side BFast P+B on delta_L = w(q) - 1
    delta_L = weights - 1.0
    delta_L = _add_shot_noise(delta_L, sn_noise_ic, n_g_density, boxsize, res)
    Pk_ic, Bk_ic = _bfast_pk_bk_safe(
        delta_L, boxsize=boxsize, bin_edges=bin_edges,
        B_info=B_info, B_norm=B_norm, mas_order=0,
    )

    # Step 5: N-body (cosmo stop_gradient'd — see _differentiable_pipeline_fin)
    cosmo_nbody = jax.lax.stop_gradient(cosmo)
    dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                   cosmo=cosmo_nbody, requires_jacfwd=False)
    dj2 = dj2.with_timetables()
    dj2 = dj2.with_external_ics(delta=delta_ic)
    dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

    _skip = {"nlpt_order_ics", "worder"}
    run_params = {k: v for k, v in sp.items() if k not in _skip}

    X_sim, _, _ = dj2.run_nbody(**run_params, use_diffrax=False)

    # Step 6: Scatter onto Eulerian mesh (CIC + deconvolve)
    n_biased = dj2.compute_field_quantity_from_particles(
        pos=X_sim,
        quantity=weights.reshape(-1).astype(X_sim.dtype),
        normalize_by_density=False,
        in_redshift_space=False,
        worder=sp["worder"],
        antialias=1,
        deconvolve=True,
    )
    delta_biased = n_biased / n_biased.mean() - 1.0
    delta_biased = _add_shot_noise(
        delta_biased, sn_noise_fin, n_g_density, boxsize, res,
    )

    # Step 7: FIN-side BFast P+B on delta_biased
    Pk_fin, Bk_fin = _bfast_pk_bk_safe(
        delta_biased, boxsize=boxsize, bin_edges=bin_edges,
        B_info=B_info, B_norm=B_norm, mas_order=0,
    )

    return jnp.concatenate([Pk_ic, Bk_ic, Pk_fin, Bk_fin])


def _differentiable_pipeline_both_bias_PB_rsd_multi(
    params,
    white_noise,
    k_vecs,
    bin_edges,
    B_info,
    B_norm,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    device=None,
    sn_noise_ic=None,
    sn_noise_fin=None,
    n_g_density=None,
):
    """RSD multipole variant of the joint IC+FIN pipeline.

    IC side: multipole estimator on delta_L = w(q)-1 (no RSD on the
    Lagrangian field; Pk2_ic / Pk4_ic are ~ 0 by isotropy and are
    returned for shape uniformity with the FIN side).

    FIN side: N-body with captured momenta, redshift-space scatter
    (los_axis = 2), then multipole estimator on delta_biased.

    Returns
    -------
    data : jnp.array, shape (2 * (3 * n_P_bins + n_triangles),)
        Concatenated [P0_ic, P2_ic, P4_ic, B_ic, P0_fin, P2_fin, P4_fin, B_fin].
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]
    sp = sim_params

    if device is None:
        device = jax.devices()[0]

    cosmo_params = params[:2]
    b1, b2, bs2, bn2 = params[2], params[3], params[4], params[5]

    cosmo = _cosmo_dict(cosmo_params, fixed=fixed_cosmo_params)

    # Step 1: Linear power spectrum
    dj1 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                   requires_jacfwd=False)
    dj1 = dj1.with_timetables().with_linear_ps()
    pk_lin = dj1._pk_table["Pk"]
    k_pk = dj1._pk_table["k"]

    # Step 2: Generate delta_ic
    k_grid = get_fourier_grid(
        [res] * dim, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"]
    Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
    norm_fac = (res / boxsize) ** dim
    delta_ic = jnp.fft.irfftn(
        jnp.fft.rfftn(white_noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
    )

    # Step 3: Lagrangian bias weights — shared between IC and FIN summaries
    weights = lagrangian_bias_weights(
        delta_ic, k_vecs, b1=b1, b2=b2, bs2=bs2, bn2=bn2,
    )

    # Step 4: IC-side multipole P+B on delta_L = w(q) - 1
    # P_2 / P_4 of the Lagrangian field are ~0 by isotropy; returned for
    # shape uniformity with the FIN side.
    delta_L = weights - 1.0
    delta_L = _add_shot_noise(delta_L, sn_noise_ic, n_g_density, boxsize, res)
    Pk0_ic, Pk2_ic, Pk4_ic, Bk_ic = _bfast_pk_multi_bk_safe(
        delta_L, boxsize=boxsize, bin_edges=bin_edges,
        B_info=B_info, B_norm=B_norm, multipole_axis=2, mas_order=0,
    )

    # Step 5: N-body (cosmo stop_gradient'd) — capture momenta for RSD
    cosmo_nbody = jax.lax.stop_gradient(cosmo)
    dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                   cosmo=cosmo_nbody, requires_jacfwd=False)
    dj2 = dj2.with_timetables()
    dj2 = dj2.with_external_ics(delta=delta_ic)
    dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

    _skip = {"nlpt_order_ics", "worder"}
    run_params = {k: v for k, v in sp.items() if k not in _skip}

    X_sim, P_mom, _ = dj2.run_nbody(**run_params, use_diffrax=False)
    P_flat = P_mom.reshape(-1, 3)

    # Step 6: Scatter in redshift space (los = z)
    n_biased = dj2.compute_field_quantity_from_particles(
        pos=X_sim,
        quantity=weights.reshape(-1).astype(X_sim.dtype),
        normalize_by_density=False,
        in_redshift_space=True,
        vel=P_flat,
        a=sp["a_end"],
        radial_dim=2,
        worder=sp["worder"],
        antialias=1,
        deconvolve=True,
    )
    delta_biased = n_biased / n_biased.mean() - 1.0
    delta_biased = _add_shot_noise(
        delta_biased, sn_noise_fin, n_g_density, boxsize, res,
    )

    # Step 7: FIN-side multipole P+B on delta_RSD
    Pk0_fin, Pk2_fin, Pk4_fin, Bk_fin = _bfast_pk_multi_bk_safe(
        delta_biased, boxsize=boxsize, bin_edges=bin_edges,
        B_info=B_info, B_norm=B_norm, multipole_axis=2, mas_order=0,
    )

    return jnp.concatenate([
        Pk0_ic, Pk2_ic, Pk4_ic, Bk_ic,
        Pk0_fin, Pk2_fin, Pk4_fin, Bk_fin,
    ])


# ---------------------------------------------------------------------------
# 1c-bis. Memory-bounded jacrev (chunk the output-basis vmap)
# ---------------------------------------------------------------------------

def _chunked_jacrev(fn, chunk_size):
    """Memory-bounded equivalent of ``jax.jacrev(fn)``.

    ``jax.jacrev`` builds the Jacobian as ``vmap(pullback)(I_n)`` where ``n``
    is the output size. With large output vectors (e.g. joint P+B at
    res=128: n_data ~ 2150) the vmapped backward pass through BFast's
    bispectrum scan blows past A100 memory.

    This helper traces ``jax.vjp`` once and then sweeps the output-basis
    cotangents through ``jax.lax.map(batch_size=chunk_size)``, which runs
    ``vmap`` over chunks sequentially. Result is bit-identical to
    ``jax.jacrev`` but peak intermediate memory scales with ``chunk_size``
    instead of the full output size.
    """
    def jac(x):
        y, vjp_fn = jax.vjp(fn, x)
        n_out = y.size
        basis = jnp.eye(n_out, dtype=y.dtype).reshape((n_out,) + y.shape)
        J = jax.lax.map(
            lambda cot: vjp_fn(cot)[0], basis, batch_size=chunk_size,
        )
        return J
    return jac


# ---------------------------------------------------------------------------
# 1d. VJP diagnostic: pinpoint where cotangent propagation breaks
# ---------------------------------------------------------------------------

def _report_vjp(name, primal, fn, x):
    """Pull a unit cotangent through ``fn`` and report finiteness at the input.

    ``primal`` is the input array (size = parameter count). ``fn(x)`` returns
    the stage output. We compute jax.vjp, pass jnp.ones_like(out) as the
    output cotangent, and check the cotangent at the input.

    Forward-mode (jvp) crashes through DiscoDJ's custom_vjp N-body, so we
    use reverse-mode here to match the path used by jax.jacrev.
    """
    out, vjp_fn = jax.vjp(fn, x)
    cot = vjp_fn(jnp.ones_like(out))[0]
    print(f"  {name}: primal finite={int(jnp.sum(jnp.isfinite(out)))}/{out.size}, "
          f"input cotangent finite={int(jnp.sum(jnp.isfinite(cot)))}/{cot.size}, "
          f"cotangent norm={float(jnp.linalg.norm(cot)):.3e}")


def _diagnose_tangent(cosmo_fid, box_params=BOX_PARAMS, sim_params=SIM_PARAMS,
                      fixed_cosmo_params=None, n_bins=30):
    """Pull a unit output-cotangent through each pipeline stage via jax.vjp.

    Reports per-stage:
    - primal finiteness (sanity check on forward eval)
    - input-cotangent finiteness (where does reverse-mode AD break?)

    The first stage where the cotangent's finite count drops below the
    parameter count is where the gradient is killed.
    """
    dim, res, boxsize = box_params["dim"], box_params["res"], box_params["boxsize"]
    sp = sim_params
    noise = jax.random.normal(jax.random.PRNGKey(0), shape=(res,) * dim)

    device = jax.devices()[0]

    # Stage 1: cosmo → P_lin
    def stage_plin(params):
        cosmo = _cosmo_dict(params, fixed=fixed_cosmo_params)
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        return dj._pk_table["Pk"]

    _report_vjp("P_lin", cosmo_fid, stage_plin, cosmo_fid)

    # Stage 2: cosmo → delta_ic
    def stage_ic(params):
        cosmo = _cosmo_dict(params, fixed=fixed_cosmo_params)
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        pk_lin = dj._pk_table["Pk"]
        k_pk = dj._pk_table["k"]
        k_grid = get_fourier_grid([res]*dim, boxsize, dtype_num=32,
                                  with_jax=True, full=False)["|k|"]
        Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
        norm_fac = (res / boxsize) ** dim
        return jnp.fft.irfftn(
            jnp.fft.rfftn(noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
        )

    _report_vjp("delta_ic", cosmo_fid, stage_ic, cosmo_fid)

    # Stage 3: cosmo → X_sim (particle positions after N-body)
    def stage_nbody(params):
        cosmo = _cosmo_dict(params, fixed=fixed_cosmo_params)
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        pk_lin = dj._pk_table["Pk"]
        k_pk = dj._pk_table["k"]
        k_grid = get_fourier_grid([res]*dim, boxsize, dtype_num=32,
                                  with_jax=True, full=False)["|k|"]
        Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
        norm_fac = (res / boxsize) ** dim
        delta_ic = jnp.fft.irfftn(
            jnp.fft.rfftn(noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
        )
        dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                       cosmo=jax.lax.stop_gradient(cosmo), requires_jacfwd=False)
        dj2 = dj2.with_timetables()
        dj2 = dj2.with_external_ics(delta=delta_ic)
        dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)
        _skip = {"nlpt_order_ics", "worder"}
        run_params = {k: v for k, v in sp.items() if k not in _skip}
        X_sim, _, _ = dj2.run_nbody(**run_params, use_diffrax=False)
        return X_sim.reshape(-1)

    _report_vjp("X_sim (N-body)", cosmo_fid, stage_nbody, cosmo_fid)

    # Stage 4: cosmo → delta_fin (density field after CIC deposit)
    def stage_delta_fin(params):
        cosmo = _cosmo_dict(params, fixed=fixed_cosmo_params)
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        pk_lin = dj._pk_table["Pk"]
        k_pk = dj._pk_table["k"]
        k_grid = get_fourier_grid([res]*dim, boxsize, dtype_num=32,
                                  with_jax=True, full=False)["|k|"]
        Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
        norm_fac = (res / boxsize) ** dim
        delta_ic = jnp.fft.irfftn(
            jnp.fft.rfftn(noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
        )
        dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                       cosmo=jax.lax.stop_gradient(cosmo), requires_jacfwd=False)
        dj2 = dj2.with_timetables()
        dj2 = dj2.with_external_ics(delta=delta_ic)
        dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)
        _skip = {"nlpt_order_ics", "worder"}
        run_params = {k: v for k, v in sp.items() if k not in _skip}
        X_sim, _, _ = dj2.run_nbody(**run_params, use_diffrax=False)
        return dj2.get_delta_from_pos(
            X_sim, res=res, worder=sp["worder"], antialias=1, deconvolve=True,
        )

    _report_vjp("delta_fin", cosmo_fid, stage_delta_fin, cosmo_fid)

    # Stage 5: full pipeline cosmo → P_NL(k)
    def stage_full(params):
        return _differentiable_pipeline_fin(
            params, noise, box_params=box_params, sim_params=sim_params,
            fixed_cosmo_params=fixed_cosmo_params, n_bins=n_bins)

    _report_vjp("P_NL", cosmo_fid, stage_full, cosmo_fid)


def _diagnose_tangent_bias(fid_values, k_vecs,
                           box_params=BOX_PARAMS, sim_params=SIM_PARAMS,
                           fixed_cosmo_params=None, n_bins=30):
    """VJP diagnostic for the biased pipeline: params = [Ω_m, σ_8, b1, b2, bs2, bn2].

    Pulls a unit cotangent through each stage and reports cotangent
    finiteness at the (params,) input. Stages where bias doesn't enter
    will have zero cotangent for the bias entries — that's expected.
    """
    dim, res, boxsize = box_params["dim"], box_params["res"], box_params["boxsize"]
    sp = sim_params
    noise = jax.random.normal(jax.random.PRNGKey(0), shape=(res,) * dim)

    device = jax.devices()[0]

    def _cosmo_from_params(params):
        return _cosmo_dict(params[:2], fixed=fixed_cosmo_params)

    # Stage 1: params -> P_lin
    def stage_plin(params):
        cosmo = _cosmo_from_params(params)
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        return dj._pk_table["Pk"]

    _report_vjp("P_lin", fid_values, stage_plin, fid_values)

    # Stage 2: params -> delta_ic
    def stage_ic(params):
        cosmo = _cosmo_from_params(params)
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        pk_lin = dj._pk_table["Pk"]
        k_pk = dj._pk_table["k"]
        k_grid = get_fourier_grid([res]*dim, boxsize, dtype_num=32,
                                  with_jax=True, full=False)["|k|"]
        Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
        norm_fac = (res / boxsize) ** dim
        return jnp.fft.irfftn(
            jnp.fft.rfftn(noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
        )

    _report_vjp("delta_ic", fid_values, stage_ic, fid_values)

    # Stage 3: params -> X_sim (cosmo only flows through N-body; bias doesn't enter)
    def stage_nbody(params):
        cosmo = _cosmo_from_params(params)
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        pk_lin = dj._pk_table["Pk"]
        k_pk = dj._pk_table["k"]
        k_grid = get_fourier_grid([res]*dim, boxsize, dtype_num=32,
                                  with_jax=True, full=False)["|k|"]
        Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
        norm_fac = (res / boxsize) ** dim
        delta_ic = jnp.fft.irfftn(
            jnp.fft.rfftn(noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
        )
        dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                       cosmo=jax.lax.stop_gradient(cosmo), requires_jacfwd=False)
        dj2 = dj2.with_timetables()
        dj2 = dj2.with_external_ics(delta=delta_ic)
        dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)
        _skip = {"nlpt_order_ics", "worder"}
        run_params = {k: v for k, v in sp.items() if k not in _skip}
        X_sim, _, _ = dj2.run_nbody(**run_params, use_diffrax=False)
        return X_sim.reshape(-1)

    _report_vjp("X_sim (N-body)", fid_values, stage_nbody, fid_values)

    # Stage 4: params -> bias weights (b1,b2,bs2,bn2 enter here, cosmo via delta_ic)
    def stage_weights(params):
        cosmo = _cosmo_from_params(params)
        b1_, b2_, bs2_, bn2_ = params[2], params[3], params[4], params[5]
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        pk_lin = dj._pk_table["Pk"]
        k_pk = dj._pk_table["k"]
        k_grid = get_fourier_grid([res]*dim, boxsize, dtype_num=32,
                                  with_jax=True, full=False)["|k|"]
        Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
        norm_fac = (res / boxsize) ** dim
        delta_ic = jnp.fft.irfftn(
            jnp.fft.rfftn(noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
        )
        weights = lagrangian_bias_weights(
            delta_ic, k_vecs, b1=b1_, b2=b2_, bs2=bs2_, bn2=bn2_,
        )
        return weights.reshape(-1)

    _report_vjp("weights", fid_values, stage_weights, fid_values)

    # Stage 5: params -> delta_biased (CIC scatter of bias-weighted particles)
    def stage_delta_biased(params):
        cosmo = _cosmo_from_params(params)
        b1_, b2_, bs2_, bn2_ = params[2], params[3], params[4], params[5]
        dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu",
                     cosmo=cosmo, requires_jacfwd=False)
        dj = dj.with_timetables().with_linear_ps()
        pk_lin = dj._pk_table["Pk"]
        k_pk = dj._pk_table["k"]
        k_grid = get_fourier_grid([res]*dim, boxsize, dtype_num=32,
                                  with_jax=True, full=False)["|k|"]
        Pk_interp = jnp.interp(k_grid, k_pk, pk_lin)
        norm_fac = (res / boxsize) ** dim
        delta_ic = jnp.fft.irfftn(
            jnp.fft.rfftn(noise) * jnp.sqrt(jnp.maximum(Pk_interp * norm_fac, 1e-30))
        )
        dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                       cosmo=jax.lax.stop_gradient(cosmo), requires_jacfwd=False)
        dj2 = dj2.with_timetables()
        dj2 = dj2.with_external_ics(delta=delta_ic)
        dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)
        _skip = {"nlpt_order_ics", "worder"}
        run_params = {k: v for k, v in sp.items() if k not in _skip}
        X_sim, _, _ = dj2.run_nbody(**run_params, use_diffrax=False)
        weights = lagrangian_bias_weights(
            delta_ic, k_vecs, b1=b1_, b2=b2_, bs2=bs2_, bn2=bn2_,
        )
        n_biased = dj2.compute_field_quantity_from_particles(
            pos=X_sim, quantity=weights.reshape(-1).astype(X_sim.dtype),
            normalize_by_density=False, in_redshift_space=False,
            worder=sp["worder"], antialias=1, deconvolve=True,
        )
        return n_biased / n_biased.mean() - 1.0

    _report_vjp("delta_biased", fid_values, stage_delta_biased, fid_values)

    # Stage 6: full bias pipeline -> P_biased(k)
    def stage_full(params):
        return _differentiable_pipeline_fin_bias(
            params, noise, k_vecs,
            box_params=box_params, sim_params=sim_params,
            fixed_cosmo_params=fixed_cosmo_params, n_bins=n_bins,
        )

    _report_vjp("P_biased", fid_values, stage_full, fid_values)


# ---------------------------------------------------------------------------
# 2. Compute Jacobian dP(k)/dtheta via jacrev, averaged over noise seeds
# ---------------------------------------------------------------------------

def compute_jacobian(
    fid_values,
    pipeline_fn,
    n_seeds=10,
    n_bins=30,
    box_params=BOX_PARAMS,
    extra_noise_per_seed=False,
    jacrev_chunk_size=None,
):
    """Compute the Jacobian dP(k)/dtheta averaged over noise seeds.

    Parameters
    ----------
    fid_values : jnp.array, shape (N_params,)
        Fiducial parameter values (cosmo, or cosmo+bias for bias pipelines).
    pipeline_fn : callable (params, noise) -> Pk
        Differentiable pipeline. Closes over box/sim/fixed/device/n_bins.
        When ``extra_noise_per_seed=True``, the signature is
        ``(params, noise, sn_noise) -> Pk`` and a second independent
        noise field is generated per seed (used for shot-noise injection).
    n_seeds : int
        Number of white-noise realisations to average over.
    n_bins : int
        Number of P(k) bins.
    box_params : dict
        Used to generate the white-noise field and the k bin centres.
    extra_noise_per_seed : bool
        If True, generate a second independent Gaussian noise field per seed
        and pass it as a third positional arg to ``pipeline_fn``. The base
        ``noise`` is generated from a fresh ``PRNGKey(seed)`` split for
        consistency with the SN-disabled path being a bitwise no-op when
        ``pipeline_fn`` ignores ``sn_noise``.

    Returns
    -------
    jacobian_mean : jnp.array, shape (n_bins, N_params)
    jacobians_all : jnp.array, shape (n_seeds, n_bins, N_params)
    Pk_fid : jnp.array, shape (n_bins,)
        Fiducial P(k) (mean over seeds).
    k_bins : jnp.array, shape (n_bins,)
    """
    dim = box_params["dim"]
    res = box_params["res"]

    # k_bins depend only on box geometry + n_bins, not on the field
    _, k_bins = count_modes(n_bins, box_params=box_params)

    if jacrev_chunk_size is not None:
        print(f"  (chunked jacrev: chunk_size={jacrev_chunk_size})", flush=True)
        def _jac(fn):
            return _chunked_jacrev(fn, jacrev_chunk_size)
    else:
        _jac = jax.jacrev

    # Compute per-seed Jacobians and fiducial P(k)
    jacobians = []
    Pk_fids = []
    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds} ...", flush=True)
        if extra_noise_per_seed:
            key = jax.random.PRNGKey(seed)
            k_ic, k_sn = jax.random.split(key)
            noise    = jax.random.normal(k_ic, shape=(res,) * dim)
            sn_noise = jax.random.normal(k_sn, shape=(res,) * dim)
            J = _jac(lambda p: pipeline_fn(p, noise, sn_noise))(fid_values)
            Pk_fids.append(pipeline_fn(fid_values, noise, sn_noise))
        else:
            noise = jax.random.normal(jax.random.PRNGKey(seed), shape=(res,) * dim)
            J = _jac(lambda p: pipeline_fn(p, noise))(fid_values)
            Pk_fids.append(pipeline_fn(fid_values, noise))
        jacobians.append(J)

    jacobians_all = jnp.stack(jacobians, axis=0)     # (n_seeds, n_bins, N_params)
    jacobian_mean = jnp.mean(jacobians_all, axis=0)   # (n_bins, N_params)
    Pk_fid = jnp.mean(jnp.stack(Pk_fids), axis=0)     # (n_bins,)

    return jacobian_mean, jacobians_all, Pk_fid, k_bins


# ---------------------------------------------------------------------------
# 2b. Numerical Jacobian via central finite differences (fallback)
# ---------------------------------------------------------------------------

_FINITE_DIFF_STEPS = {
    "Omega_m": 0.01, "Omega_b": 0.002, "h": 0.02,
    "n_s": 0.02, "sigma8": 0.015,
}


_FD_ZERO_THRESHOLD = 1e-6


def _resolve_fd_steps(fid_values, steps, step_kind, relative_eps):
    """Compute effective FD steps from absolute-step dict + kind.

    For ``step_kind="absolute"`` the input ``steps`` is returned unchanged
    (reproduces prior behaviour bit-for-bit).

    For ``step_kind="relative"`` we use a *gated* floor:

    - If ``|fid_i| >= _FD_ZERO_THRESHOLD``: ``effective_i = relative_eps * |fid_i|``
      (pure relative -- matches Franco-Abellan 2024; the absolute dict is
      ignored so the user's ``--fd-relative-eps`` actually controls the step).
    - If ``|fid_i| <  _FD_ZERO_THRESHOLD``: ``effective_i = steps_i`` (pure
      relative would give ~0; fall back on the absolute dict so we get a
      well-defined finite difference for params with fiducial = 0 -- e.g.
      ``b2``, ``bs2``, ``bn2``).

    This is the intended Franco-Abellan-style behaviour: relative everywhere
    except where it degenerates, in which case we use the user's prior
    absolute step.
    """
    if step_kind == "absolute":
        return jnp.asarray(steps, dtype=fid_values.dtype)
    if step_kind == "relative":
        abs_steps = jnp.asarray(steps, dtype=fid_values.dtype)
        rel_steps = float(relative_eps) * jnp.abs(fid_values)
        is_zero = jnp.abs(fid_values) < _FD_ZERO_THRESHOLD
        return jnp.where(is_zero, abs_steps, rel_steps)
    raise ValueError(
        f"step_kind must be 'absolute' or 'relative', got {step_kind!r}"
    )


def compute_jacobian_numerical(
    fid_values,
    pipeline_fn,
    steps,
    param_names,
    n_seeds=10,
    n_bins=30,
    box_params=BOX_PARAMS,
    extra_noise_per_seed=False,
    step_kind="absolute",
    relative_eps=1e-2,
):
    """Compute Jacobian dP(k)/dtheta via central finite differences.

    Drop-in replacement for compute_jacobian when jacrev produces NaN.
    Per seed: 1 fiducial + 2*N_params forward passes.

    Parameters
    ----------
    fid_values : jnp.array, shape (N_params,)
    pipeline_fn : callable (params, noise) -> Pk
        When ``extra_noise_per_seed=True``, signature is
        ``(params, noise, sn_noise) -> Pk`` (sn_noise held fixed across the
        ±ε evaluations within a seed so the finite difference is consistent).
    steps : jnp.array, shape (N_params,)
        Absolute central-difference step per parameter (also used as the
        relative-step floor when ``step_kind="relative"``).
    param_names : list of str
        Parameter names, used only for print labels.
    n_seeds, n_bins, box_params : same as compute_jacobian.
    extra_noise_per_seed : bool
    step_kind : {"absolute", "relative"}
        "absolute" (default) uses ``steps`` verbatim, reproducing prior
        behaviour bit-for-bit. "relative" uses
        ``step_i = max(relative_eps * |fid_values_i|, steps_i)`` --
        matches the Franco-Abellan 2024 convention, with the absolute floor
        protecting params with fiducial ~ 0.
    relative_eps : float
        Relative-step amplitude when ``step_kind="relative"``. Default 1e-2.

    Returns
    -------
    jacobian_mean, jacobians_all, Pk_fid, k_bins : same as compute_jacobian.
    """
    dim = box_params["dim"]
    res = box_params["res"]

    n_params = len(fid_values)

    _, k_bins = count_modes(n_bins, box_params=box_params)

    effective_steps = _resolve_fd_steps(
        fid_values, steps, step_kind, relative_eps,
    )
    if step_kind == "relative":
        print(f"  FD step_kind=relative (eps={relative_eps}); effective steps: "
              + ", ".join(
                  f"{n}={float(s):.4g}"
                  for n, s in zip(param_names, effective_steps.tolist())
              ))

    jacobians = []
    Pk_fids = []

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds} (numerical) ...", flush=True)
        if extra_noise_per_seed:
            key = jax.random.PRNGKey(seed)
            k_ic, k_sn = jax.random.split(key)
            noise    = jax.random.normal(k_ic, shape=(res,) * dim)
            sn_noise = jax.random.normal(k_sn, shape=(res,) * dim)
            _call = lambda p: pipeline_fn(p, noise, sn_noise)
        else:
            noise = jax.random.normal(jax.random.PRNGKey(seed), shape=(res,) * dim)
            _call = lambda p: pipeline_fn(p, noise)

        # Fiducial
        Pk_0 = _call(fid_values)
        Pk_fids.append(Pk_0)

        # Data-vector length from the actual pipeline output (supports the joint
        # [P_0, P_2, P_4, B_0] vector, not just the legacy 30-bin single P(k)).
        n_out = int(Pk_0.shape[0])

        # Central differences per parameter
        J = jnp.zeros((n_out, n_params))
        for i in range(n_params):
            step_i = effective_steps[i]
            delta = jnp.zeros(n_params).at[i].set(step_i)
            Pk_plus  = _call(fid_values + delta)
            Pk_minus = _call(fid_values - delta)
            J = J.at[:, i].set((Pk_plus - Pk_minus) / (2 * step_i))
            n_fin = int(jnp.sum(jnp.isfinite(J[:, i])))
            print(f"    d/d{param_names[i]}: finite={n_fin}/{n_out}", flush=True)

        jacobians.append(J)

    jacobians_all = jnp.stack(jacobians, axis=0)
    jacobian_mean = jnp.mean(jacobians_all, axis=0)
    Pk_fid = jnp.mean(jnp.stack(Pk_fids), axis=0)

    return jacobian_mean, jacobians_all, Pk_fid, k_bins


def compute_jacobian_forward(
    fid_values,
    pipeline_fn,
    n_seeds=10,
    n_bins=30,
    box_params=BOX_PARAMS,
    extra_noise_per_seed=False,
):
    """Forward-mode (jax.jacfwd) Jacobian d(data)/d(theta), seed-averaged.

    Unlike reverse-mode (which routes the N-body through DiscoDJ's custom_vjp
    adjoint that drops the cosmology cotangent), forward mode runs the N-body
    via requires_jacfwd=True (plain scan + custom_jvp scatter) with cosmo NOT
    stop_gradient'd, so it captures the FULL cosmo dependence including growth.

    Requirements (caller-enforced):
    - ``jax_enable_x64`` must be True: the DiscoDJ growth ODE overflows in
      float32 and yields a NaN tangent for d/dOmega_m; float64 fixes it.
    - ``pipeline_fn`` must be built with ``forward_mode=True``.

    ``fid_values`` is promoted to float64 (cosmo-tangent precision); the
    white-noise fields are float32 to match DiscoDJ's float32 mesh (avoids a
    float64->float32 scatter cast). Cost ~ n_params forward passes per seed.
    """
    dim = box_params["dim"]
    res = box_params["res"]
    _, k_bins = count_modes(n_bins, box_params=box_params)
    fid64 = jnp.asarray(fid_values, dtype=jnp.float64)
    need_two_sn = extra_noise_per_seed == "both"

    jacobians = []
    data_fids = []
    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds} (jacfwd) ...", flush=True)
        key = jax.random.PRNGKey(seed)
        if need_two_sn:
            k_ic, k_sn_ic, k_sn_fin = jax.random.split(key, 3)
            noise        = jax.random.normal(k_ic,     shape=(res,) * dim, dtype=jnp.float32)
            sn_noise_ic  = jax.random.normal(k_sn_ic,  shape=(res,) * dim, dtype=jnp.float32)
            sn_noise_fin = jax.random.normal(k_sn_fin, shape=(res,) * dim, dtype=jnp.float32)
            _call = lambda p: pipeline_fn(p, noise, sn_noise_ic, sn_noise_fin)
        elif extra_noise_per_seed:
            k_ic, k_sn = jax.random.split(key)
            noise    = jax.random.normal(k_ic, shape=(res,) * dim, dtype=jnp.float32)
            sn_noise = jax.random.normal(k_sn, shape=(res,) * dim, dtype=jnp.float32)
            _call = lambda p: pipeline_fn(p, noise, sn_noise)
        else:
            noise = jax.random.normal(key, shape=(res,) * dim, dtype=jnp.float32)
            _call = lambda p: pipeline_fn(p, noise)
        jacobians.append(jax.jacfwd(_call)(fid64))
        data_fids.append(_call(fid64))

    jacobians_all = jnp.stack(jacobians, axis=0)      # (n_seeds, n_data, N_params)
    jacobian_mean = jnp.mean(jacobians_all, axis=0)   # (n_data, N_params)
    data_fid = jnp.mean(jnp.stack(data_fids), axis=0)  # (n_data,)
    return jacobian_mean, jacobians_all, data_fid, k_bins


# ---------------------------------------------------------------------------
# 3. Gaussian covariance: sigma^2(k_a) = 2 P(k_a)^2 / N_modes(k_a)
# ---------------------------------------------------------------------------

def count_modes(n_bins, box_params=BOX_PARAMS):
    """Count the number of independent Fourier modes per k bin.

    Replicates the binning logic from discodj.core.summary_statistics.power_spectrum_core.

    Parameters
    ----------
    n_bins : int
        Number of P(k) bins (must match what was used in the pipeline).
    box_params : dict

    Returns
    -------
    N_modes : jnp.array, shape (n_bins,)
    k_bins : jnp.array, shape (n_bins,)
    """
    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]
    cellsize = boxsize / res

    shape = (res,) * dim
    kmin = 2 * jnp.pi / boxsize
    kmax = jnp.pi / cellsize

    # Logarithmic binning (default in power_spectrum)
    dlogk = (jnp.log10(kmax) - jnp.log10(kmin)) / (n_bins - 1)
    kedges = jnp.geomspace(
        kmin * 10 ** (-dlogk / 2), kmax * 10 ** (dlogk / 2),
        n_bins + 1, dtype=jnp.float32,
    )

    # k magnitude grid (rFFT layout)
    kmag = get_fourier_grid(
        shape, boxsize, dtype_num=32, with_jax=True, full=False,
    )["|k|"].reshape(-1)

    dig = jnp.digitize(kmag, kedges)

    # rfft_factor: modes in the interior of the last dimension count twice
    # (they represent both +k and -k), except for k=0 and k=Nyquist.
    rfft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    rfft_factor = jnp.ones(rfft_shape, dtype=jnp.int32)
    rfft_factor = rfft_factor.at[..., 1:-1].set(2)
    rfft_factor = rfft_factor.reshape(-1)

    N_modes = jnp.bincount(dig, length=n_bins + 2, weights=rfft_factor.astype(jnp.float32))
    N_modes = N_modes[1:-1]  # trim overflow bins

    # Effective k bin centres
    k_bins = (
        jnp.bincount(dig, length=n_bins + 2, weights=kmag * rfft_factor.astype(jnp.float32))
        / jnp.bincount(dig, length=n_bins + 2, weights=rfft_factor.astype(jnp.float32))
    )[1:-1]

    return N_modes, k_bins


def gaussian_covariance(Pk_fid, N_modes):
    """Gaussian covariance: sigma^2(k_a) = 2 P(k_a)^2 / N_modes(k_a).

    Parameters
    ----------
    Pk_fid : array, shape (n_bins,)
    N_modes : array, shape (n_bins,)

    Returns
    -------
    sigma_sq : array, shape (n_bins,)
    """
    return 2.0 * Pk_fid ** 2 / N_modes


def bispectrum_covariance(Pk, triangle_indices, k_bins, bin_widths, V_box):
    """Diagonal Gaussian variance of B(k1,k2,k3) per triangle bin.

    Scoccimarro-2000 / Sefusatti-2006 form for a Gaussian density field on
    a periodic box of volume ``V_box``:

        sigma^2(B(k1,k2,k3)) = s * (2*pi)^6 / V_box * P(k1) P(k2) P(k3) / V_T
        V_T = 8 * pi^2 * k1 * k2 * k3 * dk1 * dk2 * dk3   (continuum)

    with symmetry factor ``s = 6`` (equilateral), ``2`` (isoceles), ``1``
    (scalene). This is dimensionally consistent: ``B`` has units (Mpc/h)^6,
    so ``sigma^2(B)`` has units (Mpc/h)^12. ``V_box * P^3 / V_T`` carries
    those units (with ``V_T`` in (h/Mpc)^6).

    Parameters
    ----------
    Pk : array, shape (n_P_bins,)
        Fiducial power spectrum on ``bin_edges``.
    triangle_indices : array, shape (n_triangles, 3)
        Bin indices into Pk for each (k1, k2, k3) -- from
        ``BFast.core.bispectrum.get_triangles``.
    k_bins : array, shape (n_P_bins,)
        Physical k centres in h/Mpc (= bin centre * k_F).
    bin_widths : array, shape (n_P_bins,)
        Physical bin widths in h/Mpc (= bin width * k_F).
    V_box : float
        Box volume in (Mpc/h)^3.

    Returns
    -------
    sigma_sq_B : array, shape (n_triangles,)
    """
    i = triangle_indices[:, 0]
    j = triangle_indices[:, 1]
    k = triangle_indices[:, 2]
    eq = (i == j) & (j == k)
    iso = ((i == j) | (j == k) | (i == k)) & ~eq
    s = jnp.where(eq, 6.0, jnp.where(iso, 2.0, 1.0))
    P_prod = Pk[i] * Pk[j] * Pk[k]
    k_prod = k_bins[i] * k_bins[j] * k_bins[k]
    dk_prod = bin_widths[i] * bin_widths[j] * bin_widths[k]
    V_T = 8.0 * jnp.pi ** 2 * k_prod * dk_prod
    return s * (2.0 * jnp.pi) ** 6 / V_box * P_prod / V_T


def multipole_covariance_grieb(Pk0, Pk2, Pk4, N_modes,
                               multipole_ls=(0, 2, 4), n_mu=64):
    """Full Gaussian covariance of redshift-space P-multipoles (Grieb+2016).

    Cov(P_l(k), P_l'(k)) = (2 / N_modes(k)) * (2l+1)(2l'+1)/2
                           * integral_{-1}^{1} L_l(mu) L_l'(mu) P(k,mu)^2 dmu

    with P(k,mu) = P_0(k) + P_2(k) L_2(mu) + P_4(k) L_4(mu) (Legendre expansion).

    Reference: Grieb, Sanchez, Salazar-Albornoz & dalla Vecchia 2016,
               MNRAS 457, 1577 (arXiv:1509.04293), eq A.6.

    Replaces the diagonal Grieb-limit form
    ``sigma^2(P_l) = (2l+1) * 2 P_0^2 / N_modes`` which assumes P_2 = P_4 = 0
    and zero P_l-P_l' cross-covariance. For Kaiser-strength RSD the off-diagonal
    (0,2), (2,4) cross terms are O(P_0^2 / N_modes), so the diagonal form
    biases the Fisher *upward* (artificially tight marginals).

    Parameters
    ----------
    Pk0, Pk2, Pk4 : array, shape (n_k,)
        Fiducial monopole, quadrupole, hexadecapole on the same k grid.
        For monopole-only (``multipole_ls=(0,)``), Pk2/Pk4 may be zero.
    N_modes : array, shape (n_k,)
        Number of independent rfft modes per k bin (interior modes count twice).
    multipole_ls : tuple of ints, default (0, 2, 4)
        Multipole orders to include. Must be a subset of (0, 2, 4).
    n_mu : int, default 64
        Number of Gauss-Legendre quadrature points for the mu-integral.

    Returns
    -------
    cov : array, shape (n_k, n_l, n_l)
        ``cov[k, a, b] = Cov(P_{l_a}(k), P_{l_b}(k))`` with l_a, l_b drawn
        from ``multipole_ls`` in the given order.
    """
    from numpy.polynomial.legendre import leggauss
    mu_np, w_np = leggauss(int(n_mu))
    mu = jnp.asarray(mu_np, dtype=Pk0.dtype)
    w = jnp.asarray(w_np, dtype=Pk0.dtype)

    L0 = jnp.ones_like(mu)
    L2 = 0.5 * (3.0 * mu ** 2 - 1.0)
    L4 = (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0
    L_table = {0: L0, 2: L2, 4: L4}
    for ell in multipole_ls:
        if ell not in L_table:
            raise ValueError(
                f"multipole_covariance_grieb only supports l in (0,2,4), got {ell}"
            )
    L = jnp.stack([L_table[ell] for ell in multipole_ls], axis=0)  # (n_l, n_mu)

    Pkmu = (
        Pk0[:, None] * L0[None, :]
        + Pk2[:, None] * L2[None, :]
        + Pk4[:, None] * L4[None, :]
    )  # (n_k, n_mu)
    Pkmu2 = Pkmu ** 2

    # I[k, a, b] = sum_mu w_mu L_la(mu) L_lb(mu) Pkmu2[k, mu]
    I = jnp.einsum('m,am,bm,km->kab', w, L, L, Pkmu2)

    ls = jnp.asarray(multipole_ls, dtype=Pk0.dtype)
    prefactor = (2.0 * ls[:, None] + 1.0) * (2.0 * ls[None, :] + 1.0) / 2.0
    safe_N = jnp.where(N_modes > 0, N_modes, jnp.ones_like(N_modes))
    cov = (2.0 / safe_N[:, None, None]) * prefactor[None, :, :] * I
    return cov


# ---------------------------------------------------------------------------
# 3b. Empirical data covariance + non-Gaussian P-B cross block
# ---------------------------------------------------------------------------

def compute_empirical_covariance(
    pipeline_fn,
    fid_values,
    n_cov_seeds,
    box_params,
    extra_noise_per_seed=False,
    seed_offset=10_000,
):
    """Sample covariance of the pipeline output at the fiducial parameters.

    Generates ``n_cov_seeds`` independent white-noise realisations at fixed
    fiducial parameters, stacks the pipeline output into a
    ``(n_cov_seeds, n_data)`` matrix, and returns the sample mean + sample
    covariance. Captures every coupling that lives inside the data vector:
    P-B cross, k-k' cross, multipole l-l' cross, IC-FIN cross (when
    ``field=both``), and non-Gaussian terms (super-sample, connected
    trispectrum, ...).

    Parameters
    ----------
    pipeline_fn : callable
        ``(params, noise[, sn_noise[, sn_noise_fin]]) -> data_vector``.
        Same signature variants the Jacobian helpers consume.
    fid_values : array, shape (N_params,)
        Held fixed across all seeds.
    n_cov_seeds : int
        Number of fiducial seeds. Must satisfy
        ``n_cov_seeds > n_data + 2`` for the sample covariance to be
        invertible (and ``> n_data + 4`` for a finite Hartlap factor).
    box_params : dict
        Used only for the white-noise shape.
    extra_noise_per_seed : bool
        See ``compute_jacobian``: if True, also draw an independent
        ``sn_noise`` field and pass it positionally. For the ``--field both``
        joint pipeline a third ``sn_noise_fin`` field is drawn.
    seed_offset : int
        Added to ``seed`` indices so the empirical-cov seeds do not collide
        with the Jacobian seeds 0..n_seeds-1. Default 10_000.

    Returns
    -------
    cov_emp : jnp.array, shape (n_data, n_data)
    mean_emp : jnp.array, shape (n_data,)
    hartlap_factor : float
        ``(n_cov_seeds - n_data - 2) / (n_cov_seeds - 1)`` -- multiply onto
        ``inv(cov_emp)`` for an unbiased inverse covariance (Hartlap 2007).

    Raises
    ------
    ValueError
        If ``n_cov_seeds <= n_data + 2`` (sample cov not invertible / Hartlap
        factor non-positive). Hint: increase ``--n-cov-seeds`` or coarsen
        ``--pb-step`` to shrink the data vector.
    """
    dim = box_params["dim"]
    res = box_params["res"]
    need_two_sn = extra_noise_per_seed == "both"

    samples = []
    for seed in range(n_cov_seeds):
        key = jax.random.PRNGKey(seed + seed_offset)
        if need_two_sn:
            k_ic, k_sn_ic, k_sn_fin = jax.random.split(key, 3)
            noise        = jax.random.normal(k_ic,     shape=(res,) * dim, dtype=jnp.float32)
            sn_noise_ic  = jax.random.normal(k_sn_ic,  shape=(res,) * dim, dtype=jnp.float32)
            sn_noise_fin = jax.random.normal(k_sn_fin, shape=(res,) * dim, dtype=jnp.float32)
            d = pipeline_fn(fid_values, noise, sn_noise_ic, sn_noise_fin)
        elif extra_noise_per_seed:
            k_ic, k_sn = jax.random.split(key)
            noise    = jax.random.normal(k_ic, shape=(res,) * dim, dtype=jnp.float32)
            sn_noise = jax.random.normal(k_sn, shape=(res,) * dim, dtype=jnp.float32)
            d = pipeline_fn(fid_values, noise, sn_noise)
        else:
            noise = jax.random.normal(key, shape=(res,) * dim, dtype=jnp.float32)
            d = pipeline_fn(fid_values, noise)
        samples.append(d)
        if (seed + 1) % max(n_cov_seeds // 10, 1) == 0:
            print(f"    [cov] seed {seed + 1}/{n_cov_seeds}", flush=True)

    D = jnp.stack(samples, axis=0)            # (n_cov_seeds, n_data)
    n_data = int(D.shape[1])
    if n_cov_seeds <= n_data + 2:
        raise ValueError(
            f"n_cov_seeds={n_cov_seeds} <= n_data+2={n_data + 2}; sample "
            f"covariance is not invertible. Increase --n-cov-seeds or coarsen "
            f"--pb-step to shrink the data vector."
        )
    mean_emp = jnp.mean(D, axis=0)
    centred  = D - mean_emp[None, :]
    cov_emp  = (centred.T @ centred) / (n_cov_seeds - 1)
    hartlap_factor = float((n_cov_seeds - n_data - 2) / (n_cov_seeds - 1))
    return cov_emp, mean_emp, hartlap_factor


def empirical_pb_cross_block(
    pipeline_fn,
    fid_values,
    n_cov_seeds,
    box_params,
    n_P_total,
    n_triangles,
    block_size,
    n_blocks=1,
    extra_noise_per_seed=False,
    seed_offset=10_000,
):
    """Sample-estimate the P-B cross-covariance block(s) only.

    Cheaper than :func:`compute_empirical_covariance` when the user wants the
    P-B cross term but is happy keeping analytic Gaussian forms for the P-P
    (Grieb) and B-B (Scoccimarro) diagonal blocks. We only need to estimate
    the off-diagonal ``(n_P_total, n_triangles)`` block per field-block (one
    for ``field in {"fin","ic"}``, two for ``field="both"`` with no IC-FIN
    cross by construction).

    Strictly-Gaussian P-B cross is zero (5-point moment vanishes by Wick);
    the non-zero cross is a non-Gaussian term (proportional to the fiducial
    bispectrum). Estimating it from seeds captures this without needing a
    closed-form Sefusatti/Chan-Blot expression.

    Parameters
    ----------
    pipeline_fn, fid_values, n_cov_seeds, box_params, extra_noise_per_seed,
    seed_offset : same as :func:`compute_empirical_covariance`.
    n_P_total : int
        Per-block P-side length (``n_pk_per_field * n_P_bins``).
    n_triangles : int
        Per-block B-side length.
    block_size : int
        ``n_P_total + n_triangles``: the per-field-block stride into the
        full data vector.
    n_blocks : int
        1 for ``field in {"fin","ic"}``, 2 for ``field="both"``.

    Returns
    -------
    cov_pb_blocks : list of jnp.array, length ``n_blocks``
        Each entry has shape ``(n_P_total, n_triangles)``: the cross block
        Cov[P_l_a(k_i), B(k_1,k_2,k_3)] estimated from the sample.
    """
    cov_full, _mean, _hartlap = compute_empirical_covariance(
        pipeline_fn, fid_values, n_cov_seeds, box_params,
        extra_noise_per_seed=extra_noise_per_seed, seed_offset=seed_offset,
    )
    blocks = []
    for b in range(n_blocks):
        off = b * block_size
        p_slice = slice(off, off + n_P_total)
        b_slice = slice(off + n_P_total, off + n_P_total + n_triangles)
        blocks.append(cov_full[p_slice, b_slice])
    return blocks


# ---------------------------------------------------------------------------
# 4. Fisher matrix
# ---------------------------------------------------------------------------

def fisher_matrix(jacobian, sigma_sq):
    """Compute the Fisher information matrix.

    F_ij = sum_a (dP_a/dtheta_i) * (1/sigma^2_a) * (dP_a/dtheta_j)

    Parameters
    ----------
    jacobian : array, shape (n_bins, N_params)
    sigma_sq : array, shape (n_bins,)

    Returns
    -------
    F : array, shape (N_params, N_params)
    """
    weighted = jacobian / sigma_sq[:, None]  # (n_bins, N_params)
    return weighted.T @ jacobian              # (N_params, N_params)


def fisher_matrix_full(jacobian, inv_cov):
    """Fisher matrix from a dense inverse data covariance: F = J^T C^{-1} J.

    Generalisation of :func:`fisher_matrix` for non-diagonal data
    covariances (empirical, or block-dense with off-diagonal cross terms).

    Parameters
    ----------
    jacobian : array, shape (n_data, N_params)
    inv_cov : array, shape (n_data, n_data)

    Returns
    -------
    F : array, shape (N_params, N_params)
    """
    return jacobian.T @ inv_cov @ jacobian


def prior_fisher_matrix(sampled_keys, prior_spec):
    """Build a diagonal prior Fisher matrix from prior specifications.

    Parameters
    ----------
    sampled_keys : list of str
        Parameter names in order.
    prior_spec : dict
        Maps param name -> ("uniform", lo, hi) or ("normal", mu, sigma).

    Returns
    -------
    F_prior : np.ndarray, shape (N, N)
    """
    N = len(sampled_keys)
    F_prior = np.zeros((N, N))
    for i, key in enumerate(sampled_keys):
        kind, *args = prior_spec[key]
        if kind == "uniform":
            lo, hi = args
            F_prior[i, i] = 12.0 / (hi - lo) ** 2
        elif kind == "normal":
            _mu, sigma = args
            F_prior[i, i] = 1.0 / sigma ** 2
        else:
            raise ValueError(f"Unknown prior type '{kind}' for {key}")
    return F_prior


# ---------------------------------------------------------------------------
# 5. Corner plot: Fisher ellipses
# ---------------------------------------------------------------------------

def _confidence_ellipse(mean_x, mean_y, cov_2x2, confidence, **kwargs):
    """Return a matplotlib Ellipse patch for a 2D Gaussian confidence region.

    Parameters
    ----------
    mean_x, mean_y : float
        Centre of the ellipse.
    cov_2x2 : array, shape (2, 2)
        2×2 covariance sub-matrix.
    confidence : float
        Confidence level (e.g. 0.682 for 1σ, 0.954 for 2σ).
    **kwargs
        Forwarded to `matplotlib.patches.Ellipse`.
    """
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2

    eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
    # Semi-axis lengths scaled by chi2 quantile for 2 dof
    scale = np.sqrt(chi2.ppf(confidence, df=2))
    width = 2 * scale * np.sqrt(eigenvalues[0])
    height = 2 * scale * np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    return Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        **kwargs,
    )


def _plot_1d_marginal(ax, mean, sigma, x_range, color, label=None,
                      fill_alpha=0.15, n_pts=300):
    """Plot an analytical 1D Gaussian marginal on *ax*."""
    from scipy.stats import norm

    x = np.linspace(x_range[0], x_range[1], n_pts)
    y = norm.pdf(x, loc=mean, scale=sigma)
    ax.plot(x, y, color=color, lw=1.5, label=label)
    ax.fill_between(x, y, alpha=fill_alpha, color=color)


def plot_fisher_corner(
    fisher_cov,
    param_names,
    fiducial_vals,
    output_dir,
    prior_bounds=None,
):
    """Corner plot with analytical Fisher ellipses (2D) and Gaussian curves (1D).

    Parameters
    ----------
    fisher_cov : array, shape (N, N)
        Fisher covariance (F^{-1}).
    param_names : list of str
        LaTeX-formatted parameter names.
    fiducial_vals : array, shape (N,)
        Fiducial parameter values.
    output_dir : str
        Directory to save outputs.
    prior_bounds : list of (lo, hi) or None
        Prior bounds per parameter; drawn as red dashed lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba

    fisher_cov = np.asarray(fisher_cov)
    fiducial_vals = np.asarray(fiducial_vals)
    N = len(param_names)

    # Plot ranges: Fisher ±4σ (ellipses are ~2σ, so they fit comfortably).
    sigmas = np.sqrt(np.diag(fisher_cov))
    ranges = [(fiducial_vals[i] - 4 * sigmas[i],
                fiducial_vals[i] + 4 * sigmas[i]) for i in range(N)]

    fisher_color = "C0"

    # --- Create figure manually ---
    fig, axes_2d = plt.subplots(N, N, figsize=(2.5 * N, 2.5 * N))
    if N == 1:
        axes_2d = np.array([[axes_2d]])

    # Upper triangle: hide
    for i in range(N):
        for j in range(N):
            if j > i:
                axes_2d[i, j].set_visible(False)

    # --- Diagonal: 1D Gaussian marginals ---
    for i in range(N):
        ax = axes_2d[i, i]
        sigma_i = np.sqrt(fisher_cov[i, i])
        _plot_1d_marginal(ax, fiducial_vals[i], sigma_i, ranges[i],
                          color=fisher_color)
        ax.set_xlim(ranges[i])
        ax.set_yticks([])
        # Fiducial truth line
        ax.axvline(fiducial_vals[i], color="C1", lw=1.2, zorder=5)

    # --- Lower triangle: 2D confidence ellipses ---
    for i in range(N):
        for j in range(i):
            ax = axes_2d[i, j]
            sub_cov = fisher_cov[np.ix_([j, i], [j, i])]
            r, g, b, _ = to_rgba(fisher_color)
            for conf, alpha in [(0.954, 0.15), (0.682, 0.35)]:
                ell = _confidence_ellipse(
                    fiducial_vals[j], fiducial_vals[i], sub_cov, conf,
                    facecolor=(r, g, b, alpha),
                    edgecolor=fisher_color,
                    lw=1.2,
                )
                ax.add_patch(ell)
            ax.set_xlim(ranges[j])
            ax.set_ylim(ranges[i])
            # Fiducial crosshair
            ax.axvline(fiducial_vals[j], color="C1", lw=0.8, alpha=0.5)
            ax.axhline(fiducial_vals[i], color="C1", lw=0.8, alpha=0.5)

    # --- Labels: bottom row x-labels, left column y-labels ---
    for i in range(N):
        axes_2d[N - 1, i].set_xlabel(param_names[i])
        if i > 0:
            axes_2d[i, 0].set_ylabel(param_names[i])

    # --- Draw prior bounds as red dashed lines ---
    if prior_bounds is not None:
        for i in range(N):
            for j in range(N):
                if i >= j:
                    ax = axes_2d[i, j]
                    lo_j, hi_j = prior_bounds[j]
                    if i == j:
                        ax.axvline(lo_j, color="red", ls="--", alpha=0.3, lw=0.8)
                        ax.axvline(hi_j, color="red", ls="--", alpha=0.3, lw=0.8)
                    else:
                        lo_i, hi_i = prior_bounds[i]
                        ax.axvline(lo_j, color="red", ls="--", alpha=0.3, lw=0.8)
                        ax.axvline(hi_j, color="red", ls="--", alpha=0.3, lw=0.8)
                        ax.axhline(lo_i, color="red", ls="--", alpha=0.3, lw=0.8)
                        ax.axhline(hi_i, color="red", ls="--", alpha=0.3, lw=0.8)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], color=fisher_color, lw=2, label="Fisher forecast"),
        Line2D([0], [0], color="C1", lw=2, label="Fiducial values"),
        Line2D([0], [0], color="red", ls="--", alpha=0.7, lw=1.5, label="Prior bounds"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.7)

    fig.suptitle("Fisher forecast", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fisher_corner.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {plot_path}")


# ---------------------------------------------------------------------------
# 6. Top-level driver
# ---------------------------------------------------------------------------

def run_diagnose(
    n_bins=30,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    field="fin",
):
    """Run the per-stage JVP tangent diagnostic for the cosmo-only pipeline.

    Reports for each stage of cosmo → P_lin → δ_ic → X_sim → δ_fin → P_NL
    whether the forward-mode tangent stays finite. The first stage where
    ``tangent_finite < primal_finite`` (or norm is NaN/0) is the broken
    one.
    """
    if field != "fin":
        print(f"  --diagnose currently only meaningful for field=fin "
              f"(got '{field}'); skipping.")
        return

    if fixed_cosmo_params:
        sampled_keys = [k for k in _COSMO_PARAM_ORDER if k not in fixed_cosmo_params]
    else:
        sampled_keys = list(_COSMO_PARAM_ORDER)

    fid_dict = {
        "Omega_m": _FIDUCIAL_COSMO["Omega_c"] + _FIDUCIAL_COSMO["Omega_b"],
        "Omega_b": _FIDUCIAL_COSMO["Omega_b"],
        "h": _FIDUCIAL_COSMO["h"],
        "n_s": _FIDUCIAL_COSMO["n_s"],
        "sigma8": _FIDUCIAL_COSMO["sigma8"],
    }
    cosmo_fid = jnp.array([fid_dict[k] for k in sampled_keys], dtype=jnp.float32)
    print(f"Fiducial params: {dict(zip(sampled_keys, cosmo_fid.tolist()))}")
    print("\n=== JVP tangent diagnostic (cosmo-only pipeline) ===")
    _diagnose_tangent(cosmo_fid, box_params, sim_params, fixed_cosmo_params, n_bins)


def run_diagnose_bias(
    n_bins=30,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    field="fin",
):
    """Run the per-stage JVP tangent diagnostic for the biased tracer pipeline."""
    if field != "fin":
        print(f"  --diagnose currently only meaningful for field=fin "
              f"(got '{field}'); skipping.")
        return

    fixed_cosmo_params = {
        "Omega_b": _FIDUCIAL_COSMO["Omega_b"],
        "h": _FIDUCIAL_COSMO["h"],
        "n_s": _FIDUCIAL_COSMO["n_s"],
    }
    sampled_keys = _BIAS_COSMO_KEYS + _BIAS_PARAM_ORDER
    fid_dict = {
        "Omega_m": _FIDUCIAL_COSMO["Omega_c"] + _FIDUCIAL_COSMO["Omega_b"],
        "sigma8": _FIDUCIAL_COSMO["sigma8"],
    }
    fid_dict.update(_FIDUCIAL_BIAS)
    fid_values = jnp.array([fid_dict[k] for k in sampled_keys], dtype=jnp.float32)
    print(f"Fiducial params: {dict(zip(sampled_keys, fid_values.tolist()))}")
    print(f"Fixed cosmo: {fixed_cosmo_params}")

    print("\nPre-computing k_vecs ...")
    k_vecs = _precompute_k_vecs(box_params)

    print("\n=== JVP tangent diagnostic (biased tracer pipeline) ===")
    _diagnose_tangent_bias(fid_values, k_vecs, box_params, sim_params,
                           fixed_cosmo_params, n_bins)


def run_fisher_forecast(
    n_seeds=10,
    n_bins=30,
    sbi_samples_path=None,
    output_dir="outputs/fisher_fin",
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    fixed_cosmo_params=None,
    field="fin",
    jacrev_chunk_size=None,
    cov_type="diag",
    n_cov_seeds=200,
    fd_step_kind="absolute",
    fd_relative_eps=1e-2,
):
    """Run the full Fisher forecast pipeline.

    ``field`` selects the observable:
    - "fin" → P(delta_fin), i.e. the non-linear power spectrum after N-body.
    - "ic"  → P(delta_ic),  i.e. the initial linear power spectrum on the grid.

    Parameters
    ----------
    n_seeds : int
        Number of white-noise seeds for Jacobian averaging.
    n_bins : int
        Number of P(k) bins.
    sbi_samples_path : str or None
        Path to .npz with SBI posterior samples (key "samples", shape (n, 5)),
        or a Falcon run directory.
    output_dir : str
        Directory for output files.
    box_params, sim_params : dict
    fixed_cosmo_params : dict or None
    """
    output_dir = output_dir.rstrip("/") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Fiducial cosmology as a JAX array
    if fixed_cosmo_params:
        sampled_keys = [k for k in _COSMO_PARAM_ORDER if k not in fixed_cosmo_params]
    else:
        sampled_keys = list(_COSMO_PARAM_ORDER)

    fid_dict = {
        "Omega_m": _FIDUCIAL_COSMO["Omega_c"] + _FIDUCIAL_COSMO["Omega_b"],
        "Omega_b": _FIDUCIAL_COSMO["Omega_b"],
        "h": _FIDUCIAL_COSMO["h"],
        "n_s": _FIDUCIAL_COSMO["n_s"],
        "sigma8": _FIDUCIAL_COSMO["sigma8"],
    }
    cosmo_fid = jnp.array([fid_dict[k] for k in sampled_keys], dtype=jnp.float32)

    print(f"Fiducial params: {dict(zip(sampled_keys, cosmo_fid.tolist()))}")
    print(f"n_seeds={n_seeds}, n_bins={n_bins}")
    print(f"field={field}")

    # --- Build pipeline_fn for the chosen field ---
    if field == "fin":
        _pipeline = _differentiable_pipeline_fin
    elif field == "ic":
        _pipeline = _differentiable_pipeline_ic
    else:
        raise ValueError(f"Unknown field '{field}' (expected 'fin' or 'ic')")

    device = jax.devices()[0]

    def pipeline_fn(params, noise):
        return _pipeline(
            params, noise,
            box_params=box_params,
            sim_params=sim_params,
            fixed_cosmo_params=fixed_cosmo_params,
            n_bins=n_bins,
            device=device,
        )

    # --- JVP diagnostic (one-time, lightweight). Only meaningful for fin. ---
    if field == "fin":
        print("\n=== JVP tangent diagnostic ===")
        _diagnose_tangent(cosmo_fid, box_params, sim_params, fixed_cosmo_params, n_bins)

    # --- Jacobian ---
    print("\n=== Computing Jacobian via jacrev ===")
    jacobian, jacobians_all, Pk_fid, k_bins = compute_jacobian(
        cosmo_fid,
        pipeline_fn,
        n_seeds=n_seeds,
        n_bins=n_bins,
        box_params=box_params,
        jacrev_chunk_size=jacrev_chunk_size,
    )
    print(f"  Jacobian shape: {jacobian.shape}")
    n_finite = int(jnp.sum(jnp.isfinite(jacobian)))
    print(f"  Jacobian finite: {n_finite}/{jacobian.size}")
    print(f"  Pk_fid finite: {int(jnp.sum(jnp.isfinite(Pk_fid)))}/{Pk_fid.size}")

    # --- Fallback to numerical if jacfwd produced all NaN ---
    if n_finite == 0:
        print("\n  jacrev Jacobian all NaN — falling back to numerical differentiation")
        steps = jnp.array([_FINITE_DIFF_STEPS[k] for k in sampled_keys],
                          dtype=jnp.float32)
        jacobian, jacobians_all, Pk_fid, k_bins = compute_jacobian_numerical(
            cosmo_fid,
            pipeline_fn,
            steps,
            sampled_keys,
            n_seeds=n_seeds,
            n_bins=n_bins,
            box_params=box_params,
            step_kind=fd_step_kind,
            relative_eps=fd_relative_eps,
        )
        print(f"  Numerical Jacobian shape: {jacobian.shape}")
        print(f"  Numerical Jacobian finite: {int(jnp.sum(jnp.isfinite(jacobian)))}/{jacobian.size}")
        print(f"  Pk_fid finite: {int(jnp.sum(jnp.isfinite(Pk_fid)))}/{Pk_fid.size}")

    # --- Covariance ---
    print(f"\n=== Computing covariance (cov_type={cov_type}) ===")
    N_modes, k_bins_check = count_modes(n_bins, box_params=box_params)
    sigma_sq = gaussian_covariance(Pk_fid, N_modes)  # always computed (diagnostic)
    print(f"  N_modes range: [{float(N_modes.min()):.0f}, {float(N_modes.max()):.0f}]")

    # Mask out invalid bins: zero modes, non-finite Pk, non-finite Jacobian
    valid = (
        (N_modes > 0)
        & jnp.isfinite(Pk_fid)
        & (Pk_fid > 0)
        & jnp.all(jnp.isfinite(jacobian), axis=1)
    )
    n_valid = int(jnp.sum(valid))
    n_total = len(N_modes)
    print(f"  Valid bins: {n_valid}/{n_total}")
    if n_valid < n_total:
        print(f"  Dropping {n_total - n_valid} bin(s) "
              f"(zero modes: {int(jnp.sum(N_modes == 0))}, "
              f"non-finite Pk: {int(jnp.sum(~jnp.isfinite(Pk_fid)))}, "
              f"non-finite Jacobian: {int(jnp.sum(~jnp.all(jnp.isfinite(jacobian), axis=1)))})")
    if n_valid == 0:
        raise RuntimeError("All bins invalid — Jacobian is entirely NaN "
                           "(both jacrev and numerical fallback failed).")
    jacobian = jacobian[valid]
    Pk_fid = Pk_fid[valid]
    k_bins = k_bins[valid]
    N_modes = N_modes[valid]
    sigma_sq = sigma_sq[valid]

    # --- Fisher matrix (data only) ---
    print("\n=== Computing Fisher matrix ===")
    if cov_type == "diag":
        F_data = fisher_matrix(jacobian, sigma_sq)
    elif cov_type == "euclid_cov":
        print(f"  Building empirical covariance from n_cov_seeds={n_cov_seeds} "
              f"fiducial seeds ...")
        cov_emp, _mean, hartlap = compute_empirical_covariance(
            pipeline_fn, cosmo_fid, n_cov_seeds, box_params,
            extra_noise_per_seed=False,
        )
        cov_emp_valid = cov_emp[jnp.ix_(valid, valid)]
        inv_cov = hartlap * jnp.linalg.inv(cov_emp_valid)
        print(f"  Hartlap factor: {hartlap:.4f}")
        F_data = fisher_matrix_full(jacobian, inv_cov)
    else:
        raise ValueError(
            f"cov_type={cov_type!r} is not supported by run_fisher_forecast "
            f"(only 'diag' and 'euclid_cov'; 'block_with_pb_cross_empirical' "
            f"requires --joint-pb)."
        )
    print(f"  Data Fisher matrix:\n{np.asarray(F_data)}")

    fisher_cov_data = np.linalg.inv(np.asarray(F_data))
    sigmas_data = np.sqrt(np.diag(fisher_cov_data))
    print(f"\n  Marginal 1-sigma constraints (DATA ONLY):")
    for name, sig, fid in zip(sampled_keys, sigmas_data, cosmo_fid):
        print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f} "
              f"({100 * float(sig / fid):.2f}%)")

    # --- Prior Fisher matrix ---
    prior_spec = {k: _PRIOR_SPEC[k] for k in sampled_keys}
    F_prior = prior_fisher_matrix(sampled_keys, prior_spec)
    print(f"\n  Prior Fisher diagonal: {np.diag(F_prior)}")

    # --- Total Fisher = data + prior ---
    F_total = np.asarray(F_data) + F_prior
    fisher_cov = np.linalg.inv(F_total)
    marginal_sigmas = np.sqrt(np.diag(fisher_cov))
    print(f"\n  Marginal 1-sigma constraints (DATA + PRIOR):")
    for name, sig, fid in zip(sampled_keys, marginal_sigmas, cosmo_fid):
        print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f} "
              f"({100 * float(sig / fid):.2f}%)")

    # --- Save results ---
    results_path = os.path.join(output_dir, "fisher_results.npz")
    np.savez(
        results_path,
        F_data=np.asarray(F_data),
        F_prior=F_prior,
        F_total=F_total,
        fisher_cov=np.asarray(fisher_cov),
        fisher_cov_data=fisher_cov_data,
        jacobian=np.asarray(jacobian),
        jacobians_per_seed=np.asarray(jacobians_all),
        Pk_fid=np.asarray(Pk_fid),
        k_bins=np.asarray(k_bins),
        sigma_sq=np.asarray(sigma_sq),
        N_modes=np.asarray(N_modes),
        cosmo_fid=np.asarray(cosmo_fid),
        param_names=np.array(sampled_keys),
    )
    print(f"\n  Saved {results_path}")

    # --- Corner plot ---
    # LaTeX-style names for plots
    _latex = {
        "Omega_m": r"$\Omega_m$",
        "Omega_b": r"$\Omega_b$",
        "h": r"$h$",
        "n_s": r"$n_s$",
        "sigma8": r"$\sigma_8$",
    }
    param_labels = [_latex.get(k, k) for k in sampled_keys]

    # Load SBI samples if available
    if sbi_samples_path is not None:
        sbi_samples, _ = _load_sbi_samples(sbi_samples_path, sampled_keys)
        if sbi_samples is not None:
            sbi_save_path = os.path.join(output_dir, "sbi_samples.npz")
            np.savez(sbi_save_path, samples=sbi_samples, param_names=np.array(sampled_keys))
            print(f"  Saved {sbi_save_path}")

    # Prior bounds for the sampled parameters
    prior_bounds = [_PRIOR_BOUNDS[k] for k in sampled_keys]

    plot_fisher_corner(
        fisher_cov=np.asarray(fisher_cov),
        param_names=param_labels,
        fiducial_vals=np.asarray(cosmo_fid),
        output_dir=output_dir,
        prior_bounds=prior_bounds,
    )

    print("\nDone.")
    return {
        "F_data": F_data,
        "F_prior": F_prior,
        "F_total": F_total,
        "fisher_cov": fisher_cov,
        "fisher_cov_data": fisher_cov_data,
        "jacobian": jacobian,
        "Pk_fid": Pk_fid,
        "k_bins": k_bins,
        "sigma_sq": sigma_sq,
        "N_modes": N_modes,
    }


def _load_sbi_samples(path, sampled_keys):
    """Load SBI posterior samples from .npz file or Falcon run directory.

    Parameters
    ----------
    path : str
        Path to .npz file with key "samples" (shape (n, N_params)),
        or a Falcon run directory containing a ``samples_dir/{posterior,prior}``
        subtree of per-sample .npz files.
    sampled_keys : list of str
        Parameter names in the order used.

    Returns
    -------
    samples : np.ndarray, shape (n, N_params) or None
    label : str or None
        Legend label for the corner plot ("SBI posterior" or "Falcon prior"),
        or None if no samples were loaded.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        if "samples" in data:
            return data["samples"], "SBI posterior"
        # Try to find any array with the right second dimension
        for key in data.files:
            arr = data[key]
            if arr.ndim == 2 and arr.shape[1] == len(sampled_keys):
                print(f"  Using key '{key}' from {path}")
                return arr, "SBI posterior"
        print(f"  Warning: no suitable samples found in {path}")
        return None, None

    # Falcon run directory layout: <path>/samples_dir/{posterior,prior}/*.npz.
    samples_root = os.path.join(path, "samples_dir")
    if not os.path.isdir(samples_root):
        print(f"  Warning: no samples_dir in {path}")
        return None, None

    # Each .npz stores params as "cosmo_params" (shape (2,)) and, for bias runs,
    # "bias_params" (shape (4,)).  Concatenate in order.
    keys_to_stack = ["cosmo_params"]
    if len(sampled_keys) == 6:
        keys_to_stack.append("bias_params")

    import glob
    for kind in ("posterior", "prior"):
        kind_dir = os.path.join(samples_root, kind)
        if not os.path.isdir(kind_dir):
            continue
        files = sorted(glob.glob(os.path.join(kind_dir, "*.npz")))
        if not files:
            continue

        # Read ONLY the param keys from each file — avoid pulling the 128³
        # delta fields that share the same .npz (~16 MB each).
        rows = []
        skipped = 0
        for f in files:
            try:
                with np.load(f) as d:
                    rows.append(np.concatenate(
                        [np.asarray(d[k]).ravel() for k in keys_to_stack]
                    ))
            except (OSError, EOFError, ValueError, KeyError) as e:
                skipped += 1
                print(f"  Warning: skipping unreadable {os.path.basename(f)} "
                      f"({type(e).__name__})")
        if skipped:
            print(f"  Skipped {skipped}/{len(files)} corrupted samples")
        if not rows:
            print(f"  Warning: no readable samples in {kind_dir}")
            continue
        samples = np.stack(rows)
        if samples.shape[1] != len(sampled_keys):
            print(f"  Warning: {kind} samples shape {samples.shape} doesn't "
                  f"match {len(sampled_keys)} sampled keys")
            return None, None

        label = "SBI posterior" if kind == "posterior" else "Falcon prior"
        print(f"  Loaded {samples.shape[0]} {kind} samples from {kind_dir}")
        return samples, label

    print(f"  Warning: no posterior or prior samples found under {samples_root}")
    return None, None


# ---------------------------------------------------------------------------
# 7. Bias Fisher forecast driver
# ---------------------------------------------------------------------------

_BIAS_COSMO_KEYS = ["Omega_m", "sigma8"]


def _precompute_k_vecs(box_params):
    """Pre-compute k_vecs once from a dummy DiscoDJ instance."""
    _dj = DiscoDJ(
        dim=box_params["dim"], res=box_params["res"],
        boxsize=box_params["boxsize"],
        device="cpu", cosmo=_FIDUCIAL_COSMO,
    )
    return _dj.k_vecs


def run_fisher_forecast_bias(
    n_seeds=10,
    n_bins=30,
    sbi_samples_path=None,
    output_dir="outputs/fisher_fin_bias",
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    field="fin",
    n_g_density=None,
    jacrev_chunk_size=None,
    cov_type="diag",
    n_cov_seeds=200,
    fd_step_kind="absolute",
    fd_relative_eps=1e-2,
):
    """Run the Fisher forecast for the biased tracer pipeline.

    Joint constraints on [Omega_m, sigma8, b1, b2, bs2, bn2].
    Other cosmological parameters (Omega_b, h, n_s) are fixed to Planck 2018.

    ``field`` selects the observable:
    - "fin" → P(delta_biased), Eulerian biased tracer after N-body.
    - "ic"  → P(delta_L), P of the Lagrangian bias expansion w(q)-1 on the IC grid.

    Parameters
    ----------
    n_seeds : int
        Number of white-noise seeds for Jacobian averaging.
    n_bins : int
        Number of P(k) bins.
    sbi_samples_path : str or None
        Path to SBI posterior samples (.npz or Falcon run dir).
    output_dir : str
    box_params, sim_params : dict
    field : {"fin", "ic"}
    n_g_density : float or None
        If set, inject Gaussian shot noise N(0, 1/(n_g · V_cell)) per voxel
        into the field summed by the pipeline (Poisson Gaussian limit,
        arXiv:2504.20130v2). Output dir is suffixed with ``_SN`` so SN and
        no-SN runs do not clobber each other.

    Notes
    -----
    Why the covariance code is left unchanged when SN is enabled: the
    Gaussian formula ``σ²(P) = 2 P_fid² / N_modes`` uses the *measured*
    fiducial P(k); when SN is injected, P_fid already includes the 1/n̄_g
    plateau, so the variance is automatically inflated. The Jacobian
    ∂P/∂θ is unchanged because the noise is θ-independent.
    """
    if n_g_density is not None and "_SN" not in output_dir:
        output_dir = output_dir.rstrip("/") + "_SN"
    output_dir = output_dir.rstrip("/") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Fixed cosmo params: everything except Omega_m and sigma8
    fixed_cosmo_params = {
        "Omega_b": _FIDUCIAL_COSMO["Omega_b"],
        "h": _FIDUCIAL_COSMO["h"],
        "n_s": _FIDUCIAL_COSMO["n_s"],
    }

    # Combined param order: [Omega_m, sigma8, b1, b2, bs2, bn2]
    sampled_keys = _BIAS_COSMO_KEYS + _BIAS_PARAM_ORDER

    fid_dict = {
        "Omega_m": _FIDUCIAL_COSMO["Omega_c"] + _FIDUCIAL_COSMO["Omega_b"],
        "sigma8": _FIDUCIAL_COSMO["sigma8"],
    }
    fid_dict.update(_FIDUCIAL_BIAS)

    fid_values = jnp.array([fid_dict[k] for k in sampled_keys], dtype=jnp.float32)

    print(f"Fiducial params: {dict(zip(sampled_keys, fid_values.tolist()))}")
    print(f"Fixed cosmo: {fixed_cosmo_params}")
    print(f"n_seeds={n_seeds}, n_bins={n_bins}")
    print(f"field={field}")

    # Pre-compute k_vecs once
    print("\nPre-computing k_vecs ...")
    k_vecs = _precompute_k_vecs(box_params)

    device = jax.devices()[0]

    # --- Build pipeline_fn for the chosen field ---
    if field == "fin":
        _pipeline = _differentiable_pipeline_fin_bias
    elif field == "ic":
        _pipeline = _differentiable_pipeline_ic_bias
    else:
        raise ValueError(f"Unknown field '{field}' (expected 'fin' or 'ic')")

    sn_active = n_g_density is not None
    if sn_active:
        print(f"\nShot noise: n_g_density = {n_g_density:.3e} (h/Mpc)^-3")
        cell = box_params["boxsize"] / box_params["res"]
        print(f"  → N̄ per voxel ≈ {n_g_density * cell**3:.4f}, "
              f"σ_n ≈ {1.0 / np.sqrt(n_g_density * cell**3):.4f}")

        def pipeline_fn(params, noise, sn_noise):
            return _pipeline(
                params, noise, k_vecs,
                box_params=box_params, sim_params=sim_params,
                fixed_cosmo_params=fixed_cosmo_params,
                n_bins=n_bins, device=device,
                sn_noise=sn_noise, n_g_density=n_g_density,
            )
    else:
        def pipeline_fn(params, noise):
            return _pipeline(
                params, noise, k_vecs,
                box_params=box_params, sim_params=sim_params,
                fixed_cosmo_params=fixed_cosmo_params,
                n_bins=n_bins, device=device,
            )

    # --- Jacobian via jacrev ---
    print("\n=== Computing Jacobian via jacrev ===")
    jacobian, jacobians_all, Pk_fid, k_bins = compute_jacobian(
        fid_values,
        pipeline_fn,
        n_seeds=n_seeds,
        n_bins=n_bins,
        box_params=box_params,
        extra_noise_per_seed=sn_active,
        jacrev_chunk_size=jacrev_chunk_size,
    )

    print(f"  Jacobian shape: {jacobian.shape}")
    n_finite = int(jnp.sum(jnp.isfinite(jacobian)))
    print(f"  Jacobian finite: {n_finite}/{jacobian.size}")

    # --- Fallback to numerical if jacfwd produced all NaN ---
    if n_finite == 0:
        print("\n  jacrev Jacobian all NaN — falling back to numerical differentiation")
        steps = jnp.array([_FINITE_DIFF_STEPS_BIAS[k] for k in sampled_keys],
                          dtype=jnp.float32)
        jacobian, jacobians_all, Pk_fid, k_bins = compute_jacobian_numerical(
            fid_values,
            pipeline_fn,
            steps,
            sampled_keys,
            n_seeds=n_seeds,
            n_bins=n_bins,
            box_params=box_params,
            extra_noise_per_seed=sn_active,
            step_kind=fd_step_kind,
            relative_eps=fd_relative_eps,
        )
        print(f"  Numerical Jacobian shape: {jacobian.shape}")
        print(f"  Numerical Jacobian finite: "
              f"{int(jnp.sum(jnp.isfinite(jacobian)))}/{jacobian.size}")

    # --- Covariance ---
    print(f"\n=== Computing covariance (cov_type={cov_type}) ===")
    N_modes, _ = count_modes(n_bins, box_params=box_params)
    sigma_sq = gaussian_covariance(Pk_fid, N_modes)
    print(f"  N_modes range: [{float(N_modes.min()):.0f}, {float(N_modes.max()):.0f}]")

    # Mask out invalid bins
    valid = (
        (N_modes > 0)
        & jnp.isfinite(Pk_fid)
        & (Pk_fid > 0)
        & jnp.all(jnp.isfinite(jacobian), axis=1)
    )
    n_valid = int(jnp.sum(valid))
    n_total = len(N_modes)
    print(f"  Valid bins: {n_valid}/{n_total}")
    if n_valid == 0:
        raise RuntimeError("All bins invalid — cannot compute Fisher matrix.")
    jacobian = jacobian[valid]
    Pk_fid = Pk_fid[valid]
    k_bins = k_bins[valid]
    N_modes = N_modes[valid]
    sigma_sq = sigma_sq[valid]

    # --- Fisher matrix (data only) ---
    print("\n=== Computing Fisher matrix ===")
    if cov_type == "diag":
        F_data = fisher_matrix(jacobian, sigma_sq)
    elif cov_type == "euclid_cov":
        print(f"  Building empirical covariance from n_cov_seeds={n_cov_seeds} "
              f"fiducial seeds ...")
        cov_emp, _mean, hartlap = compute_empirical_covariance(
            pipeline_fn, fid_values, n_cov_seeds, box_params,
            extra_noise_per_seed=sn_active,
        )
        cov_emp_valid = cov_emp[jnp.ix_(valid, valid)]
        inv_cov = hartlap * jnp.linalg.inv(cov_emp_valid)
        print(f"  Hartlap factor: {hartlap:.4f}")
        F_data = fisher_matrix_full(jacobian, inv_cov)
    else:
        raise ValueError(
            f"cov_type={cov_type!r} is not supported by run_fisher_forecast_bias "
            f"(only 'diag' and 'euclid_cov'; 'block_with_pb_cross_empirical' "
            f"requires --joint-pb)."
        )
    print(f"  Data Fisher matrix:\n{np.asarray(F_data)}")

    fisher_cov_data = np.linalg.inv(np.asarray(F_data))
    sigmas_data = np.sqrt(np.diag(fisher_cov_data))
    print(f"\n  Marginal 1-sigma constraints (DATA ONLY):")
    for name, sig, fid in zip(sampled_keys, sigmas_data, fid_values):
        if float(fid) != 0:
            print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f} "
                  f"({100 * float(sig / fid):.2f}%)")
        else:
            print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f}")

    # --- Prior Fisher matrix ---
    F_prior = prior_fisher_matrix(sampled_keys, _PRIOR_SPEC_BIAS)
    print(f"\n  Prior Fisher diagonal: {np.diag(F_prior)}")

    # --- Total Fisher = data + prior ---
    F_total = np.asarray(F_data) + F_prior
    fisher_cov = np.linalg.inv(F_total)
    marginal_sigmas = np.sqrt(np.diag(fisher_cov))
    print(f"\n  Marginal 1-sigma constraints (DATA + PRIOR):")
    for name, sig, fid in zip(sampled_keys, marginal_sigmas, fid_values):
        if float(fid) != 0:
            print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f} "
                  f"({100 * float(sig / fid):.2f}%)")
        else:
            print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f}")

    # --- Save results ---
    results_path = os.path.join(output_dir, "fisher_results.npz")
    np.savez(
        results_path,
        F_data=np.asarray(F_data),
        F_prior=F_prior,
        F_total=F_total,
        fisher_cov=np.asarray(fisher_cov),
        fisher_cov_data=fisher_cov_data,
        jacobian=np.asarray(jacobian),
        jacobians_per_seed=np.asarray(jacobians_all),
        Pk_fid=np.asarray(Pk_fid),
        k_bins=np.asarray(k_bins),
        sigma_sq=np.asarray(sigma_sq),
        N_modes=np.asarray(N_modes),
        fid_values=np.asarray(fid_values),
        param_names=np.array(sampled_keys),
        n_g_density=np.asarray(n_g_density if n_g_density is not None else np.nan),
    )
    print(f"\n  Saved {results_path}")

    # --- Corner plot ---
    _latex = {
        "Omega_m": r"$\Omega_m$",
        "sigma8": r"$\sigma_8$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "bs2": r"$b_{s^2}$",
        "bn2": r"$b_{\nabla^2}$",
    }
    param_labels = [_latex.get(k, k) for k in sampled_keys]

    if sbi_samples_path is not None:
        sbi_samples, _ = _load_sbi_samples(sbi_samples_path, sampled_keys)
        if sbi_samples is not None:
            sbi_save_path = os.path.join(output_dir, "sbi_samples.npz")
            np.savez(sbi_save_path, samples=sbi_samples,
                     param_names=np.array(sampled_keys))
            print(f"  Saved {sbi_save_path}")

    prior_bounds = [_PRIOR_BOUNDS_BIAS[k] for k in sampled_keys]

    plot_fisher_corner(
        fisher_cov=np.asarray(fisher_cov),
        param_names=param_labels,
        fiducial_vals=np.asarray(fid_values),
        output_dir=output_dir,
        prior_bounds=prior_bounds,
    )

    print("\nDone.")
    return {
        "F_data": F_data,
        "F_prior": F_prior,
        "F_total": F_total,
        "fisher_cov": fisher_cov,
        "fisher_cov_data": fisher_cov_data,
        "jacobian": jacobian,
        "Pk_fid": Pk_fid,
        "k_bins": k_bins,
        "sigma_sq": sigma_sq,
        "N_modes": N_modes,
    }


# ---------------------------------------------------------------------------
# 8. Joint P(k) + B(k1,k2,k3) Fisher forecast (biased tracer, BFast)
# ---------------------------------------------------------------------------

def run_fisher_forecast_bias_PB(
    n_seeds=10,
    bin_edges_step=1,
    sbi_samples_path=None,
    output_dir=None,
    box_params=BOX_PARAMS,
    sim_params=SIM_PARAMS,
    field="fin",
    use_multipoles=False,
    n_g_density=None,
    jacrev_chunk_size=None,
    derivative_method="autodiff",
    config_path=None,
    resolved_args=None,
    cov_type="diag",
    n_cov_seeds=200,
    fd_step_kind="absolute",
    fd_relative_eps=1e-2,
    fiducial_bias=None,
):
    """Joint P(k) + B(k1,k2,k3) Fisher forecast for the biased tracer pipeline.

    Constraints on [Omega_m, sigma8, b1, b2, bs2, bn2] from a concatenated
    data vector measured via the BFast library
    (https://github.com/tsfloss/BFast).

    The ``field`` parameter selects which real-space field(s) feed the
    BFast joint summary:

    - ``"fin"`` : ``D = [P(k), B(k1,k2,k3)]`` on the post-N-body Eulerian
      biased field ``delta_biased``. (Existing default.)
    - ``"ic"``  : ``D = [P(k), B(k1,k2,k3)]`` on the Lagrangian biased
      field ``delta_L = w(q) - 1`` (no N-body).
    - ``"both"``: ``D = [P_ic, B_ic, P_fin, B_fin]`` -- both summaries,
      one N-body run shared between them. Block-diagonal covariance with
      no IC-FIN cross term (optimistic; flagged in console output).

    Covariance is block-diagonal Gaussian within each field:
    - P-P : ``2 P^2 / N_modes``
    - B-B : analytic Scoccimarro-2000 ``s_t * (2 pi)^6 / V_box * P^3 / V_T``.
    - P-B cross : zero (Gaussian-field assumption).
    For ``field="both"`` the IC and FIN blocks are independently treated
    with this same recipe (no IC<->FIN cross block).

    When ``use_multipoles=True`` the P-side summary is replaced by the
    redshift-space P_0/P_2/P_4 multipoles (LOS = z, hard-coded). This
    requires ``field`` in ``{"fin", "both"}``. The bispectrum stays as a
    monopole. Multipole covariance is diagonal Gaussian:
    ``sigma^2(P_l, k) = (2l+1) * 2 * P_0(k)^2 / N_modes(k)``
    (Grieb+2016 limit; neglects P_l-P_l' cross terms).

    Parameters
    ----------
    n_seeds : int
        White-noise seeds for Jacobian averaging.
    bin_edges_step : int
        Stride for BFast linear k-bins ``jnp.arange(1, res//3+1, step)``.
    sbi_samples_path : str or None
        SBI posterior samples to overplot.
    output_dir : str or None
        If None, defaults per (field, use_multipoles):
        ``outputs/fisher_bias_P_B[_ic|_both][_rsd[_both]]``.
    box_params, sim_params : dict
    field : {"fin", "ic", "both"}
    use_multipoles : bool
        If True, use redshift-space P_0/P_2/P_4 + B monopole. Requires
        ``field`` in ``{"fin", "both"}``.
    jacrev_chunk_size : int or None
        If set, compute the per-seed Jacobian via :func:`_chunked_jacrev`
        with this chunk size (memory-bounded sequential vmap over the
        output basis). Use when vanilla ``jax.jacrev`` OOMs on large data
        vectors (e.g. res>=128 with ``field=both`` + multipoles). Default
        None preserves the original ``jax.jacrev`` path.
    """
    if derivative_method not in ("autodiff", "fd", "jacfwd"):
        raise ValueError(
            f"derivative_method must be 'autodiff', 'fd', or 'jacfwd', "
            f"got {derivative_method!r}"
        )
    if derivative_method == "jacfwd" and field != "fin":
        raise ValueError(
            f"derivative_method='jacfwd' is only wired for field='fin', "
            f"got field={field!r}."
        )
    output_dir_was_none = output_dir is None
    if output_dir is None:
        output_dir = {
            ("fin",  False): "outputs/fisher_bias_P_B",
            ("ic",   False): "outputs/fisher_bias_P_B_ic",
            ("both", False): "outputs/fisher_bias_P_B_both",
            ("fin",  True):  "outputs/fisher_bias_P_B_rsd",
            ("both", True):  "outputs/fisher_bias_P_B_rsd_both",
        }[(field, use_multipoles)]
        if n_g_density is not None and "_SN" not in output_dir:
            output_dir = output_dir.rstrip("/") + "_SN"
        output_dir = output_dir.rstrip("/") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Copy source config + dump resolved args for provenance. Both files
    # land next to fisher_results.npz so the run dir is self-describing.
    if config_path is not None and os.path.isfile(config_path):
        shutil.copy2(config_path, os.path.join(output_dir, "source_config.yml"))
    if resolved_args is not None:
        try:
            import yaml as _yaml
            with open(os.path.join(output_dir, "run_config.yml"), "w") as _f:
                _yaml.safe_dump(
                    {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                     for k, v in resolved_args.items()},
                    _f, default_flow_style=False, sort_keys=True,
                )
        except Exception as _exc:
            print(f"WARNING: failed to write run_config.yml: {_exc}")

    fixed_cosmo_params = {
        "Omega_b": _FIDUCIAL_COSMO["Omega_b"],
        "h": _FIDUCIAL_COSMO["h"],
        "n_s": _FIDUCIAL_COSMO["n_s"],
    }

    sampled_keys = _BIAS_COSMO_KEYS + _BIAS_PARAM_ORDER

    fid_dict = {
        "Omega_m": _FIDUCIAL_COSMO["Omega_c"] + _FIDUCIAL_COSMO["Omega_b"],
        "sigma8": _FIDUCIAL_COSMO["sigma8"],
    }
    fid_dict.update(_FIDUCIAL_BIAS if fiducial_bias is None else fiducial_bias)
    fid_values = jnp.array([fid_dict[k] for k in sampled_keys], dtype=jnp.float32)

    print(f"Fiducial params: {dict(zip(sampled_keys, fid_values.tolist()))}")
    print(f"Fixed cosmo: {fixed_cosmo_params}")
    print(f"n_seeds={n_seeds}")

    dim = box_params["dim"]
    res = box_params["res"]
    boxsize = box_params["boxsize"]

    bin_edges = jnp.arange(1, res // 3 + 1, bin_edges_step, dtype=jnp.int32)
    n_P_bins = int(bin_edges.shape[0] - 1)
    print(f"\nBFast bin_edges (units of k_F={2*np.pi/boxsize:.4f}): "
          f"{bin_edges.tolist()}")
    print(f"  n_P_bins={n_P_bins}")

    print("\nPre-computing k_vecs ...")
    k_vecs = _precompute_k_vecs(box_params)

    print("Pre-computing BFast triangles + normalisation ...")
    B_info, B_norm = _bfast_precompute(
        boxsize=boxsize, bin_edges=bin_edges, res=res, dim=dim, mas_order=0,
    )
    triangle_indices = B_info["triangle_indices"]
    triangle_centers = B_info["triangle_centers"]
    n_triangles = int(triangle_indices.shape[0])
    print(f"  n_triangles={n_triangles}")
    print(f"  data vector size = n_P_bins + n_triangles = "
          f"{n_P_bins + n_triangles}")

    # B_norm["Pk"] is N_modes(k) per P-bin (compute_norm of unit field).
    # The bispectrum variance does NOT use BFast's compute_norm("Bk") -- that's
    # an estimator normalisation, not a triangle count. Instead we apply the
    # analytic Scoccimarro-2000 / Sefusatti-2006 Gaussian formula in
    # bispectrum_covariance, parametrised by k centres + bin widths.
    N_modes = jnp.asarray(B_norm["Pk"])
    print(f"  N_modes per P-bin range: "
          f"[{float(N_modes.min()):.0f}, {float(N_modes.max()):.0f}]")

    device = jax.devices()[0]

    if use_multipoles:
        _pipeline_map = {
            "fin":  _differentiable_pipeline_fin_bias_PB_rsd_multi,
            "both": _differentiable_pipeline_both_bias_PB_rsd_multi,
        }
        if field not in _pipeline_map:
            raise ValueError(
                f"use_multipoles=True requires field in {{'fin','both'}}, got {field!r}"
            )
        _pipeline = _pipeline_map[field]
        multipole_ls = (0, 2, 4)
        print("\nINFO: pk_multipoles enabled (RSD; los_axis=2; data vector = "
              "[P_0, P_2, P_4, B_monopole]).")
        print("INFO: full Grieb+2016 multipole covariance "
              "(eq A6) — per-k 3x3 P_l-P_l' block built from P_0,P_2,P_4 "
              "via 64-point Gauss-Legendre quadrature.")
    else:
        _pipeline_map = {
            "fin":  _differentiable_pipeline_fin_bias_PB,
            "ic":   _differentiable_pipeline_ic_bias_PB,
            "both": _differentiable_pipeline_both_bias_PB,
        }
        if field not in _pipeline_map:
            raise ValueError(
                f"field must be one of fin/ic/both, got {field!r}"
            )
        _pipeline = _pipeline_map[field]
        multipole_ls = (0,)
    if field == "both":
        print("\nWARNING: --field both uses block-diagonal IC<->FIN "
              "covariance (no cross terms); reported marginals are "
              "optimistic.")

    sn_active = n_g_density is not None
    is_both   = (field == "both")
    if sn_active:
        cell = float(boxsize) / int(res)
        print(f"\nShot noise: n_g_density = {n_g_density:.3e} (h/Mpc)^-3")
        print(f"  → N̄ per voxel ≈ {n_g_density * cell**3:.4f}, "
              f"σ_n ≈ {1.0 / np.sqrt(n_g_density * cell**3):.4f}")
        print("  → Output dir suffixed with '_SN'.")
        print("  → Covariance is left at its measured-P_fid form; the shot-noise "
              "plateau is automatically inside P_fid (see docstring).")

    if sn_active and is_both:
        def pipeline_fn(params, noise, sn_noise_ic, sn_noise_fin):
            return _pipeline(
                params, noise, k_vecs, bin_edges, B_info, B_norm,
                box_params=box_params, sim_params=sim_params,
                fixed_cosmo_params=fixed_cosmo_params, device=device,
                sn_noise_ic=sn_noise_ic, sn_noise_fin=sn_noise_fin,
                n_g_density=n_g_density,
            )
    elif sn_active:
        def pipeline_fn(params, noise, sn_noise):
            return _pipeline(
                params, noise, k_vecs, bin_edges, B_info, B_norm,
                box_params=box_params, sim_params=sim_params,
                fixed_cosmo_params=fixed_cosmo_params, device=device,
                sn_noise=sn_noise, n_g_density=n_g_density,
            )
    else:
        def pipeline_fn(params, noise):
            return _pipeline(
                params, noise, k_vecs, bin_edges, B_info, B_norm,
                box_params=box_params, sim_params=sim_params,
                fixed_cosmo_params=fixed_cosmo_params, device=device,
            )

    n_pk_per_field = len(multipole_ls)
    n_blocks = 2 if field == "both" else 1
    block_size = n_pk_per_field * n_P_bins + n_triangles
    n_data = n_blocks * block_size
    n_params = int(fid_values.shape[0])
    if derivative_method == "jacfwd":
        # Forward-mode AD (+ x64): captures the FULL cosmo dependence incl. the
        # N-body growth that the reverse-mode custom_vjp adjoint drops. Wired for
        # field='fin' only (the pipeline gets forward_mode=True -> requires_jacfwd
        # + no stop_gradient on cosmo). Reuses compute_jacobian_forward.
        if is_both:
            raise ValueError("derivative_method='jacfwd' is only wired for "
                             "field='fin' (not 'both').")
        _log("=== Computing Jacobian via jax.jacfwd (forward mode, x64) -- "
             "full cosmo dependence incl. growth ===")
        if sn_active:
            def pipeline_fn_fwd(p, noise, sn_noise):
                return _pipeline(
                    p, noise, k_vecs, bin_edges, B_info, B_norm,
                    box_params=box_params, sim_params=sim_params,
                    fixed_cosmo_params=fixed_cosmo_params, device=device,
                    sn_noise=sn_noise, n_g_density=n_g_density, forward_mode=True,
                )
            _extra = True
        else:
            def pipeline_fn_fwd(p, noise):
                return _pipeline(
                    p, noise, k_vecs, bin_edges, B_info, B_norm,
                    box_params=box_params, sim_params=sim_params,
                    fixed_cosmo_params=fixed_cosmo_params, device=device,
                    forward_mode=True,
                )
            _extra = False
        jacobian, jacobians_all, data_fid, _kb = compute_jacobian_forward(
            fid_values, pipeline_fn_fwd, n_seeds=n_seeds, n_bins=30,
            box_params=box_params, extra_noise_per_seed=_extra,
        )
    else:
        if derivative_method == "autodiff":
            if jacrev_chunk_size is not None:
                _log(f"=== Computing Jacobian via chunked jacrev "
                     f"(per seed: ~{n_data} backward passes, "
                     f"chunk_size={jacrev_chunk_size}) ===")
                def _jac(fn):
                    return _chunked_jacrev(fn, jacrev_chunk_size)
            else:
                _log(f"=== Computing Jacobian via jacrev "
                     f"(per seed: ~{n_data} backward passes) ===")
                _jac = jax.jacrev
        else:
            # Central finite differences: 1 + 2*N_params forward passes per seed.
            # Reuses _FINITE_DIFF_STEPS_BIAS for the per-parameter step sizes.
            # No stop_gradient(cosmo) shortcut: full chain rule via numerical
            # perturbation of fid_values fed to pipeline_fn(p, noise, ...).
            fd_steps_arr = jnp.array(
                [_FINITE_DIFF_STEPS_BIAS[k] for k in sampled_keys],
                dtype=fid_values.dtype,
            )
            fd_steps_arr = _resolve_fd_steps(
                fid_values, fd_steps_arr, fd_step_kind, fd_relative_eps,
            )
            _log(f"=== Computing Jacobian via central finite differences "
                 f"(per seed: {1 + 2 * n_params} forward passes, "
                 f"step_kind={fd_step_kind}) ===")
            print(f"  FD steps: " + ", ".join(
                f"{k}={float(s):.4g}" for k, s in zip(sampled_keys, fd_steps_arr.tolist())
            ))
        jacobians = []
        data_fids = []
        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds} ...", flush=True)
            if sn_active and is_both:
                key = jax.random.PRNGKey(seed)
                k_ic, k_sn_ic, k_sn_fin = jax.random.split(key, 3)
                noise        = jax.random.normal(k_ic,     shape=(res,) * dim, dtype=jnp.float32)
                sn_noise_ic  = jax.random.normal(k_sn_ic,  shape=(res,) * dim, dtype=jnp.float32)
                sn_noise_fin = jax.random.normal(k_sn_fin, shape=(res,) * dim, dtype=jnp.float32)
                _call = lambda p: pipeline_fn(p, noise, sn_noise_ic, sn_noise_fin)
            elif sn_active:
                key = jax.random.PRNGKey(seed)
                k_ic, k_sn = jax.random.split(key)
                noise    = jax.random.normal(k_ic, shape=(res,) * dim, dtype=jnp.float32)
                sn_noise = jax.random.normal(k_sn, shape=(res,) * dim, dtype=jnp.float32)
                _call = lambda p: pipeline_fn(p, noise, sn_noise)
            else:
                noise = jax.random.normal(jax.random.PRNGKey(seed), shape=(res,) * dim, dtype=jnp.float32)
                _call = lambda p: pipeline_fn(p, noise)

            if derivative_method == "autodiff":
                J = _jac(_call)(fid_values)
            else:
                cols = []
                for j in range(n_params):
                    step = fd_steps_arr[j]
                    delta = jnp.zeros_like(fid_values).at[j].set(step)
                    cols.append(
                        (_call(fid_values + delta) - _call(fid_values - delta))
                        / (2.0 * step)
                    )
                J = jnp.stack(cols, axis=-1)
            data_fids.append(_call(fid_values))
            jacobians.append(J)

        jacobians_all = jnp.stack(jacobians, axis=0)
        jacobian = jnp.mean(jacobians_all, axis=0)
        data_fid = jnp.mean(jnp.stack(data_fids), axis=0)

    print(f"  Jacobian shape: {jacobian.shape}")
    n_finite = int(jnp.sum(jnp.isfinite(jacobian)))
    print(f"  Jacobian finite: {n_finite}/{jacobian.size}")

    _log("=== Computing joint Gaussian covariance (block-diagonal) ===")
    k_F = 2.0 * jnp.pi / boxsize
    k_bins_P = 0.5 * (bin_edges[1:] + bin_edges[:-1]) * k_F
    bin_widths_P = (bin_edges[1:] - bin_edges[:-1]) * k_F
    V_box = float(boxsize) ** dim

    # Slice fiducial data vector into per-field pieces. Layout:
    # - field=fin/ic, monopole : [P, B]                          -> 1 block
    # - field=both,   monopole : [P_ic, B_ic, P_fin, B_fin]      -> 2 blocks
    # - field=fin,   multipole : [P0, P2, P4, B]                 -> 1 block
    # - field=both,  multipole : [P0_ic, P2_ic, P4_ic, B_ic,
    #                            P0_fin, P2_fin, P4_fin, B_fin]  -> 2 blocks
    #
    # P-side covariance:
    # - monopole only            : diagonal sigma^2(P_0) = 2 P_0^2 / N_modes
    # - multipoles (P_0,P_2,P_4) : full Grieb+2016 (n_k, n_l, n_l) block
    #                              with P_l-P_l' cross terms (correct Gaussian
    #                              cov for Kaiser-RSD multipole estimators).
    block_labels = ["ic", "fin"] if field == "both" else [field]
    Pk_fid_blocks = []          # monopole P0 per block (legacy alias)
    Pk_fid_multi_blocks = []    # [{0: P0, 2: P2, 4: P4}, ...] when use_multipoles
    Bk_fid_blocks = []
    sigma_sq_blocks = []        # diagonal entries only -- kept for diagnostics
    valid_blocks = []
    # For use_multipoles=True the actual Fisher uses the dense Grieb block
    # below; the (2l+1) * 2 P_0^2 / N_modes diagonal is recorded only as a
    # legacy diagnostic.
    cov_P_grieb_blocks = []     # (n_k, n_l, n_l) per block
    valid_k_P_blocks = []       # (n_k,) per block, joint validity across l
    jacobian_P_blocks = []      # (n_l, n_k, n_params) per block
    jacobian_B_blocks = []      # (n_triangles, n_params) per block
    sigma_sq_B_blocks = []      # (n_triangles,) per block
    for b, label in enumerate(block_labels):
        offset = b * block_size

        # --- P side: one entry per multipole (monopole only when
        # use_multipoles=False).
        Pk0_b = data_fid[offset:offset + n_P_bins]
        pk_dict_b = {}
        block_p_sigmas = []
        block_p_valids = []
        for li, ell in enumerate(multipole_ls):
            pk_offset = offset + li * n_P_bins
            Pk_l_b = data_fid[pk_offset:pk_offset + n_P_bins]
            # Diagnostic-only diagonal (matches the old Grieb-limit form):
            sigma_sq_Pl_b = (2 * ell + 1) * gaussian_covariance(Pk0_b, N_modes)
            if ell == 0:
                valid_Pl_b = (
                    (N_modes > 0) & jnp.isfinite(Pk_l_b) & (Pk_l_b > 0)
                    & jnp.all(
                        jnp.isfinite(jacobian[pk_offset:pk_offset + n_P_bins]),
                        axis=1,
                    )
                )
            else:
                valid_Pl_b = (
                    (N_modes > 0) & jnp.isfinite(Pk_l_b)
                    & jnp.all(
                        jnp.isfinite(jacobian[pk_offset:pk_offset + n_P_bins]),
                        axis=1,
                    )
                )
            pk_dict_b[ell] = Pk_l_b
            block_p_sigmas.append(sigma_sq_Pl_b)
            block_p_valids.append(valid_Pl_b)
            tag = f"P{ell}" if use_multipoles else "P"
            print(f"  [{label}] sigma_sq_{tag} diag range: "
                  f"[{float(sigma_sq_Pl_b.min()):.3e}, "
                  f"{float(sigma_sq_Pl_b.max()):.3e}]")

        # --- Full Grieb+2016 multipole cov (n_k, n_l, n_l) when use_multipoles
        # (also returns the trivial (n_k, 1, 1) form for the monopole-only case,
        # which equals 2 P_0^2 / N_modes, so we use it uniformly).
        Pk0_for_grieb = pk_dict_b.get(0, Pk0_b)
        Pk2_for_grieb = pk_dict_b.get(2, jnp.zeros_like(Pk0_b))
        Pk4_for_grieb = pk_dict_b.get(4, jnp.zeros_like(Pk0_b))
        cov_P_grieb_b = multipole_covariance_grieb(
            Pk0_for_grieb, Pk2_for_grieb, Pk4_for_grieb, N_modes,
            multipole_ls=multipole_ls,
        )
        cov_P_grieb_blocks.append(cov_P_grieb_b)
        # Joint k-validity: drop k bin if ANY multipole or its Jacobian is bad
        valid_k_P = block_p_valids[0]
        for v in block_p_valids[1:]:
            valid_k_P = valid_k_P & v
        valid_k_P_blocks.append(valid_k_P)
        # Stack the per-block P Jacobian as (n_l, n_k, n_params)
        J_P_block = jnp.stack([
            jacobian[offset + li * n_P_bins:offset + (li + 1) * n_P_bins]
            for li in range(n_pk_per_field)
        ], axis=0)
        jacobian_P_blocks.append(J_P_block)

        # --- B side (monopole; uses P0 in variance formula)
        bk_offset = offset + n_pk_per_field * n_P_bins
        Bk_b = data_fid[bk_offset:bk_offset + n_triangles]
        sigma_sq_B_b = bispectrum_covariance(
            Pk0_b, triangle_indices, k_bins_P, bin_widths_P, V_box,
        )
        valid_B_b = (
            jnp.isfinite(Bk_b) & jnp.isfinite(sigma_sq_B_b)
            & (sigma_sq_B_b > 0)
            & jnp.all(
                jnp.isfinite(jacobian[bk_offset:bk_offset + n_triangles]),
                axis=1,
            )
        )
        print(f"  [{label}] sigma_sq_B range: "
              f"[{float(jnp.nanmin(sigma_sq_B_b)):.3e}, "
              f"{float(jnp.nanmax(sigma_sq_B_b)):.3e}]")
        jacobian_B_blocks.append(jacobian[bk_offset:bk_offset + n_triangles])
        sigma_sq_B_blocks.append(sigma_sq_B_b)

        # Append diagnostics in data-vector order: P0[, P2, P4], B
        sigma_sq_blocks.extend(block_p_sigmas)
        sigma_sq_blocks.append(sigma_sq_B_b)
        valid_blocks.extend(block_p_valids)
        valid_blocks.append(valid_B_b)

        Pk_fid_blocks.append(Pk0_b)
        Bk_fid_blocks.append(Bk_b)
        if use_multipoles:
            Pk_fid_multi_blocks.append(pk_dict_b)

        valid_summary = ", ".join(
            f"{('P' + str(ell) if use_multipoles else 'P')} "
            f"{int(jnp.sum(block_p_valids[li]))}/{n_P_bins}"
            for li, ell in enumerate(multipole_ls)
        )
        print(f"  [{label}] valid: {valid_summary} (joint k: "
              f"{int(jnp.sum(valid_k_P))}/{n_P_bins}), "
              f"B {int(jnp.sum(valid_B_b))}/{n_triangles}")

    sigma_sq = jnp.concatenate(sigma_sq_blocks)
    valid = jnp.concatenate(valid_blocks)
    n_valid = int(jnp.sum(valid))
    print(f"  Total valid bins: {n_valid}/{n_data}")
    if n_valid == 0:
        raise RuntimeError("All bins invalid -- cannot compute Fisher matrix.")

    # Back-compat aliases for the FIN-only / IC-only case so the existing
    # downstream save/plot code keeps working. In multipole mode Pk_fid
    # aliases to the monopole P0.
    if field == "both":
        Pk_fid = Pk_fid_blocks[1]   # FIN P0 (legacy alias)
        Bk_fid = Bk_fid_blocks[1]   # FIN B
    else:
        Pk_fid = Pk_fid_blocks[0]
        Bk_fid = Bk_fid_blocks[0]
    # Last block's P0 sigma² and B sigma² (legacy aliases sigma_sq_P / sigma_sq_B).
    sigma_sq_B = sigma_sq_blocks[-1]
    sigma_sq_P = sigma_sq_blocks[-(n_pk_per_field + 1)]

    # --- Fisher: cov_type branches.
    #
    # diag (current default): dense multipole P block + diagonal B block,
    #     summed over (independent) IC and FIN field blocks. IC<->FIN
    #     remains zero (documented optimistic).
    # block_with_pb_cross_empirical: per-block dense (n_P_total + n_triangles)^2
    #     covariance with Grieb P-P, Scoccimarro B-B diag, AND empirical P-B
    #     cross block. IC<->FIN still zero.
    # euclid_cov: full empirical covariance of the *entire* data vector --
    #     unifies P-B cross, k-k', l-l', IC<->FIN, and non-Gaussian terms in
    #     one matrix.
    n_params = jacobian.shape[-1]

    # Helper: build a JAX validity mask for one field-block in the
    # data-vector order [P_0, ..., P_{l_max}, B].
    def _block_valid(b):
        start = b * (n_pk_per_field + 1)
        end = (b + 1) * (n_pk_per_field + 1)
        return jnp.concatenate(valid_blocks[start:end])

    if cov_type == "diag":
        _log("=== Computing Fisher matrix (Grieb+2016 multipole cov) ===")
        F_data = jnp.zeros((n_params, n_params), dtype=jacobian.dtype)
        eye_l = jnp.eye(n_pk_per_field, dtype=jacobian.dtype)
        for b in range(n_blocks):
            # P contribution: per-k 3x3 (or 1x1 for monopole-only) dense block.
            valid_k = valid_k_P_blocks[b]
            cov_safe = jnp.where(
                valid_k[:, None, None], cov_P_grieb_blocks[b], eye_l[None, :, :],
            )
            J_P_safe = jnp.where(
                valid_k[None, :, None], jacobian_P_blocks[b],
                jnp.zeros_like(jacobian_P_blocks[b]),
            )
            cov_inv = jnp.linalg.inv(cov_safe)               # (n_k, n_l, n_l)
            F_P_b = jnp.einsum(
                'aki,kab,bkj->ij', J_P_safe, cov_inv, J_P_safe,
            )
            # B contribution: diagonal.
            valid_B = valid_blocks[(b + 1) * (n_pk_per_field + 1) - 1]
            sigma_sq_B_b = sigma_sq_B_blocks[b]
            J_B = jacobian_B_blocks[b]
            sigma_sq_B_safe = jnp.where(
                valid_B, sigma_sq_B_b, jnp.ones_like(sigma_sq_B_b),
            )
            J_B_safe = jnp.where(valid_B[:, None], J_B, jnp.zeros_like(J_B))
            F_B_b = (J_B_safe / sigma_sq_B_safe[:, None]).T @ J_B_safe
            F_data = F_data + F_P_b + F_B_b
        F_data = np.asarray(F_data)

    elif cov_type in ("euclid_cov", "block_with_pb_cross_empirical"):
        # Determine the extra-noise mode for compute_empirical_covariance:
        # False / True / "both" matches the three pipeline_fn signatures
        # used during Jacobian computation.
        if sn_active and is_both:
            cov_extra_noise = "both"
        elif sn_active:
            cov_extra_noise = True
        else:
            cov_extra_noise = False

        n_P_total = n_pk_per_field * n_P_bins

        if cov_type == "euclid_cov":
            _log(f"=== Computing Fisher matrix (full empirical cov, "
                 f"n_cov_seeds={n_cov_seeds}) ===")
            cov_emp, _mean, hartlap = compute_empirical_covariance(
                pipeline_fn, fid_values, n_cov_seeds, box_params,
                extra_noise_per_seed=cov_extra_noise,
            )
            print(f"  Empirical cov shape: {tuple(cov_emp.shape)}; "
                  f"Hartlap factor: {hartlap:.4f}")
            valid_full = jnp.concatenate(valid_blocks)
            valid_idx = jnp.where(valid_full)[0]
            cov_valid = cov_emp[jnp.ix_(valid_idx, valid_idx)]
            inv_cov_valid = hartlap * jnp.linalg.inv(cov_valid)
            J_valid = jacobian[valid_idx]
            F_data = np.asarray(fisher_matrix_full(J_valid, inv_cov_valid))

        else:  # block_with_pb_cross_empirical
            _log(f"=== Computing Fisher matrix (block + empirical P-B cross, "
                 f"n_cov_seeds={n_cov_seeds}) ===")
            cov_pb_blocks = empirical_pb_cross_block(
                pipeline_fn, fid_values, n_cov_seeds, box_params,
                n_P_total=n_P_total, n_triangles=n_triangles,
                block_size=block_size, n_blocks=n_blocks,
                extra_noise_per_seed=cov_extra_noise,
            )
            for b in range(n_blocks):
                cov_pb_b = cov_pb_blocks[b]
                pb_norm = float(jnp.max(jnp.abs(cov_pb_b)))
                print(f"  [block {b}] |cov_PB|_max = {pb_norm:.3e}")

            F_data = jnp.zeros((n_params, n_params), dtype=jacobian.dtype)
            for b in range(n_blocks):
                # Build dense per-block covariance of shape
                # (n_P_total + n_triangles, n_P_total + n_triangles):
                #   [[ Cov_PP (Grieb)   Cov_PB (empirical) ],
                #    [ Cov_PB^T          diag(sigma_sq_B)   ]]
                grieb_np = np.asarray(cov_P_grieb_blocks[b])  # (n_k, n_l, n_l)
                cov_PP = np.zeros((n_P_total, n_P_total), dtype=np.float64)
                n_k = n_P_bins
                for la in range(n_pk_per_field):
                    for lb in range(n_pk_per_field):
                        sub = cov_PP[la*n_k:(la+1)*n_k, lb*n_k:(lb+1)*n_k]
                        np.fill_diagonal(sub, grieb_np[:, la, lb])
                        cov_PP[la*n_k:(la+1)*n_k, lb*n_k:(lb+1)*n_k] = sub

                cov_BB = np.diag(np.asarray(sigma_sq_B_blocks[b], dtype=np.float64))
                cov_PB_np = np.asarray(cov_pb_b, dtype=np.float64)
                size = n_P_total + n_triangles
                cov_block = np.zeros((size, size), dtype=np.float64)
                cov_block[:n_P_total, :n_P_total] = cov_PP
                cov_block[:n_P_total, n_P_total:] = cov_PB_np
                cov_block[n_P_total:, :n_P_total] = cov_PB_np.T
                cov_block[n_P_total:, n_P_total:] = cov_BB

                # Mask rows/cols of invalid data-vector entries
                valid_b = np.asarray(_block_valid(b))
                valid_idx_b = np.where(valid_b)[0]
                cov_block_valid = cov_block[np.ix_(valid_idx_b, valid_idx_b)]
                # Symmetric Hartlap on the partial-empirical block: the P-P
                # and B-B blocks are analytic, so no Hartlap on them; we
                # apply Hartlap only on cov_pb_b. Inversion of the joint
                # block however mixes Hartlap implicitly. Pragmatic choice:
                # use a Hartlap factor computed at the cross-block dimension
                # (n_cov_seeds - n_block - 2)/(n_cov_seeds - 1) -- gives the
                # correct unbiased inverse of the joint block at leading
                # order. (Approximation; documented.)
                n_block = int(cov_block_valid.shape[0])
                hartlap_b = float((n_cov_seeds - n_block - 2) / (n_cov_seeds - 1))
                if hartlap_b <= 0.0:
                    raise ValueError(
                        f"Hartlap factor non-positive (n_cov_seeds={n_cov_seeds}, "
                        f"block size={n_block}); increase --n-cov-seeds."
                    )
                inv_cov_b = hartlap_b * np.linalg.inv(cov_block_valid)

                # Jacobian rows for this field-block, in data-vector order:
                off = b * block_size
                J_b_full = jacobian[off:off + block_size]
                J_b_valid = np.asarray(J_b_full)[valid_idx_b]
                F_data = F_data + (J_b_valid.T @ inv_cov_b @ J_b_valid)
            F_data = np.asarray(F_data)

    else:
        raise ValueError(
            f"cov_type={cov_type!r} is not supported "
            f"(expected one of: diag, euclid_cov, block_with_pb_cross_empirical)."
        )
    print(f"  Data Fisher matrix:\n{F_data}")

    fisher_cov_data = np.linalg.inv(np.asarray(F_data))
    sigmas_data = np.sqrt(np.diag(fisher_cov_data))
    print(f"\n  Marginal 1-sigma constraints (DATA ONLY):")
    for name, sig, fid in zip(sampled_keys, sigmas_data, fid_values):
        if float(fid) != 0:
            print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f} "
                  f"({100 * float(sig / fid):.2f}%)")
        else:
            print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f}")

    F_prior = prior_fisher_matrix(sampled_keys, _PRIOR_SPEC_BIAS)
    print(f"\n  Prior Fisher diagonal: {np.diag(F_prior)}")

    F_total = np.asarray(F_data) + F_prior
    fisher_cov = np.linalg.inv(F_total)
    marginal_sigmas = np.sqrt(np.diag(fisher_cov))
    print(f"\n  Marginal 1-sigma constraints (DATA + PRIOR):")
    for name, sig, fid in zip(sampled_keys, marginal_sigmas, fid_values):
        if float(fid) != 0:
            print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f} "
                  f"({100 * float(sig / fid):.2f}%)")
        else:
            print(f"    {name}: {float(fid):.4f} +/- {float(sig):.4f}")

    results_path = os.path.join(output_dir, "fisher_results.npz")
    save_kwargs = dict(
        F_data=np.asarray(F_data),
        F_prior=F_prior,
        F_total=F_total,
        fisher_cov=np.asarray(fisher_cov),
        fisher_cov_data=fisher_cov_data,
        jacobian=np.asarray(jacobian),
        jacobians_per_seed=np.asarray(jacobians_all),
        Pk_fid=np.asarray(Pk_fid),
        Bk_fid=np.asarray(Bk_fid),
        k_bins=np.asarray(k_bins_P),
        bin_edges=np.asarray(bin_edges),
        triangle_centers=np.asarray(triangle_centers),
        triangle_indices=np.asarray(triangle_indices),
        sigma_sq=np.asarray(sigma_sq),
        sigma_sq_P=np.asarray(sigma_sq_P),
        sigma_sq_B=np.asarray(sigma_sq_B),
        N_modes=np.asarray(N_modes),
        bin_widths_P=np.asarray(bin_widths_P),
        valid=np.asarray(valid),
        n_P_bins=np.int32(n_P_bins),
        n_triangles=np.int32(n_triangles),
        fid_values=np.asarray(fid_values),
        param_names=np.array(sampled_keys),
        field=np.array(field),
        data_fid=np.asarray(data_fid),
    )
    if field == "both":
        save_kwargs["Pk_ic_fid"] = np.asarray(Pk_fid_blocks[0])
        save_kwargs["Bk_ic_fid"] = np.asarray(Bk_fid_blocks[0])
        save_kwargs["Pk_fin_fid"] = np.asarray(Pk_fid_blocks[1])
        save_kwargs["Bk_fin_fid"] = np.asarray(Bk_fid_blocks[1])
    save_kwargs["use_multipoles"] = np.array(use_multipoles)
    save_kwargs["n_g_density"] = np.asarray(
        n_g_density if n_g_density is not None else np.nan, dtype=np.float64
    )
    save_kwargs["derivative_method"] = np.array(derivative_method)
    # Full Grieb+2016 multipole covariance, per field block (n_k, n_l, n_l).
    save_kwargs["cov_P_grieb_blocks"] = np.stack(
        [np.asarray(c) for c in cov_P_grieb_blocks], axis=0,
    )
    save_kwargs["valid_k_P_blocks"] = np.stack(
        [np.asarray(v) for v in valid_k_P_blocks], axis=0,
    )
    save_kwargs["multipole_ls"] = np.asarray(multipole_ls, dtype=np.int32)
    if use_multipoles:
        if field == "fin":
            save_kwargs["Pk0_fid"] = np.asarray(Pk_fid_multi_blocks[0][0])
            save_kwargs["Pk2_fid"] = np.asarray(Pk_fid_multi_blocks[0][2])
            save_kwargs["Pk4_fid"] = np.asarray(Pk_fid_multi_blocks[0][4])
        elif field == "both":
            save_kwargs["Pk0_ic_fid"] = np.asarray(Pk_fid_multi_blocks[0][0])
            save_kwargs["Pk2_ic_fid"] = np.asarray(Pk_fid_multi_blocks[0][2])
            save_kwargs["Pk4_ic_fid"] = np.asarray(Pk_fid_multi_blocks[0][4])
            save_kwargs["Pk0_fin_fid"] = np.asarray(Pk_fid_multi_blocks[1][0])
            save_kwargs["Pk2_fin_fid"] = np.asarray(Pk_fid_multi_blocks[1][2])
            save_kwargs["Pk4_fin_fid"] = np.asarray(Pk_fid_multi_blocks[1][4])
    np.savez(results_path, **save_kwargs)
    print(f"\n  Saved {results_path}")

    _latex = {
        "Omega_m": r"$\Omega_m$",
        "sigma8": r"$\sigma_8$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "bs2": r"$b_{s^2}$",
        "bn2": r"$b_{\nabla^2}$",
    }
    param_labels = [_latex.get(k, k) for k in sampled_keys]

    if sbi_samples_path is not None:
        sbi_samples, _ = _load_sbi_samples(sbi_samples_path, sampled_keys)
        if sbi_samples is not None:
            sbi_save_path = os.path.join(output_dir, "sbi_samples.npz")
            np.savez(sbi_save_path, samples=sbi_samples,
                     param_names=np.array(sampled_keys))
            print(f"  Saved {sbi_save_path}")

    prior_bounds = [_PRIOR_BOUNDS_BIAS[k] for k in sampled_keys]

    plot_fisher_corner(
        fisher_cov=np.asarray(fisher_cov),
        param_names=param_labels,
        fiducial_vals=np.asarray(fid_values),
        output_dir=output_dir,
        prior_bounds=prior_bounds,
    )

    print("\nDone.")
    return {
        "F_data": F_data,
        "F_prior": F_prior,
        "F_total": F_total,
        "fisher_cov": fisher_cov,
        "fisher_cov_data": fisher_cov_data,
        "jacobian": jacobian,
        "Pk_fid": Pk_fid,
        "Bk_fid": Bk_fid,
        "sigma_sq": sigma_sq,
        "N_modes": N_modes,
    }

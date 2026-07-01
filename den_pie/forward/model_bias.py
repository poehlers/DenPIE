"""Biased tracer forward model: N-body + second-order Lagrangian bias + optional RSD.

Pipeline node (parents: cosmo_params, delta_ic, bias_params):

    cosmo_params  ─────────────────────────────────┐
         │                                         ▼
         ▼                                  delta_fin ◄── bias_params
        pk ──► delta_ic ───────────────────────────┘

Physics
-------
Particles are evolved from the initial Lagrangian grid to Eulerian positions
by a PM N-body simulation.  Each particle at Lagrangian position q carries a
bias weight

    w(q) = 1 + b₁·δ(q) + b₂·[δ²(q)−⟨δ²⟩] + bs²·[s²(q)−⟨s²⟩] + bn²·∇²δ(q)

computed from the linear IC field.  The bias-weighted particles are scattered
onto a mesh to form the biased density contrast:

    1 + δ_biased(x) = [Σ_q w(q) · W(x−X(q))] / ⟨ Σ_q w(q) · W(x−X(q)) ⟩

Optionally, particles are displaced along the line-of-sight by their peculiar
velocity before scattering (Kaiser RSD, apply_rsd=True).

k_vecs (Fourier wavenumber grid) are pre-computed once in __init__ from the
box geometry and shared across all bias field evaluations, avoiding redundant
computation during batched simulation.

Usage
-----
Generate prior samples:

    RUN_DIR="outputs/bias_run_$(date +%y%m%d_%H%M%S)"
    falcon sample prior --config-name config_files/config_bias.yml --run-dir "${RUN_DIR}"

Chunk over multiple calls to accumulate more samples (Falcon appends each run):

    for i in $(seq 1 $N_CHUNKS); do
        falcon sample prior --config-name config_files/config_bias.yml --run-dir "${RUN_DIR}"
    done

Train the posterior estimator (uncomment evidence/observed in config_files/config_bias.yml first):

    falcon train --config-name config_files/config_bias.yml --run-dir "${RUN_DIR}"

New-Falcon interface contract:
  - simulate_batch(batch_size, *parent_arrays) -> np.ndarray
  - All parent arrays are numpy float32, already batched (first dim = batch_size)
  - Return value must be numpy float32 with first dim = batch_size
  - JAX must NOT be imported at module level (breaks Ray actor serialisation)
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
from discodj import DiscoDJ
import falcon

from .model import SIM_PARAMS, BOX_PARAMS, _FIDUCIAL_COSMO, _cosmo_dict
from .bias import lagrangian_bias_weights

# Seed namespace for shot-noise RNG, decorrelating it from IC seeds.
_SN_SEED_NAMESPACE = 0x5000_0000


class ForwardModelBias:
    """N-body simulation + Lagrangian bias + optional RSD at variable cosmology.

    Inputs (parent order: cosmo_params, delta_ic, bias_params)
    -----------------------------------------------------------
    cosmo_params : (batch_size, D) float32
        Sampled cosmological parameters in canonical order [Ω_m, Ω_b, h, n_s, σ_8],
        containing only the parameters NOT present in ``fixed_cosmo_params``.
        D = 5 when ``fixed_cosmo_params`` is empty (default); D < 5 otherwise.
    delta_ic     : (batch_size, res, res, res) float32
    bias_params  : (batch_size, 4) float32   [b₁, b₂, bs², bn²]

    Output
    ------
    delta_fin : (batch_size, res, res, res) float32
        Bias-weighted density contrast at z = 0 (or in redshift space).

    Memory management
    -----------------
    The batch is split into sub-batches of `sub_batch_size` via jax.lax.map,
    which runs them sequentially on the GPU.  sub_batch_size=1 is always safe;
    increase it for better GPU utilisation if VRAM permits.  Requirement:
    ``sample.prior.n`` in config must be divisible by ``sub_batch_size``.
    """

    def __init__(
        self,
        dim:                int             = BOX_PARAMS["dim"],
        res:                int             = BOX_PARAMS["res"],
        boxsize:            float           = BOX_PARAMS["boxsize"],
        sim_params:         dict            = SIM_PARAMS,
        sub_batch_size:     int             = 1,
        apply_rsd:          bool            = False,
        los_axis:           int             = 2,
        fixed_cosmo_params: dict            = {},
        n_g_density:        Optional[float] = None,
        run_dir:            str             = "",
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.dim                 = dim
        self.res                 = res
        self.boxsize             = boxsize
        self.sim_params          = dict(sim_params)
        self.sub_batch_size      = sub_batch_size
        self.apply_rsd           = apply_rsd
        self.los_axis            = los_axis
        self._fixed_cosmo_params = dict(fixed_cosmo_params)
        self.n_g_density         = n_g_density   # galaxies per (h/Mpc)^3
        self._run_dir            = run_dir

        import jax
        devices     = jax.devices()
        self.device = devices[0]   # GPU if available, else CPU

        # Pre-compute k_vecs once from box geometry (independent of cosmology).
        # k_vecs are pure functions of res and boxsize, so any cosmology gives
        # the same result.
        _dj = DiscoDJ(
            dim=dim, res=res, boxsize=boxsize,
            device="cpu", cosmo=_FIDUCIAL_COSMO,
        )
        self._k_vecs = _dj.k_vecs   # list of 3 sparse 1-D JAX arrays

        # vmap over (delta_ic, cosmo_params, bias_params) simultaneously.
        self._batched_simulate = jax.vmap(self._simulate_impl, in_axes=(0, 0, 0))

    # ------------------------------------------------------------------
    # JAX kernel
    # ------------------------------------------------------------------

    def _simulate_impl(self, delta_ic, cosmo_params, bias_params):
        """Single-sample: run N-body, compute bias weights, scatter."""
        sp = self.sim_params
        b1, b2, bs2, bn2 = bias_params   # unpack (4,) JAX array

        # 1. Build DiscoDJ instance and run PM N-body at the given cosmology.
        dj = DiscoDJ(
            dim=self.dim, res=self.res, boxsize=self.boxsize,
            device=self.device, cosmo=_cosmo_dict(cosmo_params, fixed=self._fixed_cosmo_params),
        )
        dj = dj.with_timetables()
        dj = dj.with_external_ics(delta=delta_ic)
        dj = dj.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

        _skip = {"nlpt_order_ics", "worder"}
        run_params = {k: v for k, v in sp.items() if k not in _skip}

        X_sim, P, _ = dj.run_nbody(**run_params, use_diffrax=False)
        # X_sim : (res, res, res, 3) — Eulerian particle positions
        # P     : (res, res, res, 3) — canonical momenta (used for RSD)
        #
        # DiscoDJ's compute_field_quantity_from_particles auto-flattens `pos`
        # via ensure_flat_shape(), but does NOT flatten `vel`.  Passing P in
        # grid shape (res,res,res,3) causes vel[:,radial_dim] to index axis 1
        # instead of the particle axis, giving shape (res,res,3) instead of
        # (res³,).  Pre-flatten P so vel[:,radial_dim] → (res³,) as expected.
        P_flat = P.reshape(-1, 3) if self.apply_rsd else None

        # 2. Compute Lagrangian bias weights at the initial grid positions q.
        # Since q = regular grid, weights[i,j,k] is the weight of particle (i,j,k).
        weights = lagrangian_bias_weights(
            delta_ic, self._k_vecs,
            b1=b1, b2=b2, bs2=bs2, bn2=bn2,
        )   # (res, res, res)

        # 3. Scatter bias-weighted particles onto the Eulerian mesh.
        # normalize_by_density=False → return raw weighted number count n_biased(x).
        # in_redshift_space=True → particles are displaced by v_los/H before scatter.
        n_biased = dj.compute_field_quantity_from_particles(
            pos=X_sim,
            quantity=weights.reshape(-1),   # (N_particles,) — returns (res, res, res)
            normalize_by_density=False,
            in_redshift_space=self.apply_rsd,
            vel=P_flat,
            a=sp["a_end"] if self.apply_rsd else None,
            radial_dim=self.los_axis,
            worder=sp["worder"],
            antialias=1,
            deconvolve=True,
        )   # (res, res, res)

        # 4. Normalise to density contrast: δ_biased = n_biased/<n_biased> − 1
        return n_biased / n_biased.mean() - 1.0

    # ------------------------------------------------------------------
    # Sub-batching (jax.lax.map for GPU memory control)
    # ------------------------------------------------------------------

    def _sub_batched_simulate(
        self, delta_ic_batch, cosmo_params_batch, bias_params_batch
    ):
        import jax

        B   = delta_ic_batch.shape[0]
        sb  = self.sub_batch_size
        nsb = B // sb   # B must be divisible by sb

        delta_ic_sub     = delta_ic_batch.reshape((nsb, sb) + delta_ic_batch.shape[1:])
        cosmo_params_sub = cosmo_params_batch.reshape((nsb, sb) + cosmo_params_batch.shape[1:])
        bias_params_sub  = bias_params_batch.reshape((nsb, sb) + bias_params_batch.shape[1:])

        results = jax.lax.map(
            lambda x: self._batched_simulate(x[0], x[1], x[2]),
            (delta_ic_sub, cosmo_params_sub, bias_params_sub),
        )   # (nsb, sb, res, res, res)
        return results.reshape(delta_ic_batch.shape)

    # ------------------------------------------------------------------
    # New-Falcon interface
    # ------------------------------------------------------------------

    def simulate_batch(
        self,
        batch_size:   int,
        cosmo_params: np.ndarray,
        delta_ic:     np.ndarray,
        bias_params:  np.ndarray,
    ) -> np.ndarray:
        """Simulate the biased tracer density field for a batch.

        Parameters
        ----------
        cosmo_params : (batch_size, D) float32
            Sampled params in canonical order; D < 5 when fixed_cosmo_params is set.
        delta_ic     : (batch_size, res, res, res) float32
        bias_params  : (batch_size, 4) float32   [b₁, b₂, bs², bn²]

        Returns
        -------
        delta_fin : (batch_size, res, res, res) float32
        """
        import jax.numpy as jnp

        cosmo_jax = jnp.array(cosmo_params, dtype=jnp.float32)
        delta_jax = jnp.array(delta_ic,     dtype=jnp.float32)
        bias_jax  = jnp.array(bias_params,  dtype=jnp.float32)

        # np.array (not np.asarray) — forces a writable copy. np.asarray
        # on a JAX array can return a read-only DLPack view, which would
        # break the in-place result[i] += ... shot-noise injection below.
        result = np.array(
            self._sub_batched_simulate(delta_jax, cosmo_jax, bias_jax),
            dtype=np.float32,
        )

        # ── Optional shot noise (Poisson Gaussian limit, arXiv:2504.20130v2) ──
        # δ_g ← δ_biased + ε,  ε ~ N(0, 1/N̄_g)  per voxel
        # with N̄_g = n̄_g · V_cell, V_cell = (boxsize / res)^3.
        # Seed is deterministic: same file-counting pattern as
        # InitialConditions (model.py:289-296), namespaced with
        # _SN_SEED_NAMESPACE so it never collides with IC RNG.
        if self.n_g_density:
            V_cell  = (self.boxsize / self.res) ** 3
            N_bar   = float(self.n_g_density) * V_cell
            sigma_n = 1.0 / np.sqrt(N_bar)
            if self._run_dir:
                prior_dir = Path(self._run_dir) / "samples_dir" / "prior"
                seed_offset = (
                    len(list(prior_dir.glob("*.npz"))) if prior_dir.exists() else 0
                )
            else:
                seed_offset = 0
            for i in range(batch_size):
                rng = np.random.default_rng(_SN_SEED_NAMESPACE + seed_offset + i)
                result[i] += rng.normal(
                    0.0, sigma_n, result[i].shape
                ).astype(np.float32)
            falcon.log({
                "shot_noise_n_g_density": float(self.n_g_density),
                "shot_noise_N_bar_per_voxel": float(N_bar),
                "shot_noise_sigma": float(sigma_n),
            })

        falcon.log({
            "delta_fin_mean": float(result.mean()),
            "delta_fin_std":  float(result.std()),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node: pk_measured  (N-body + bias → P(k) via Pylians)
# ─────────────────────────────────────────────────────────────────────────────

class ForwardModelBiasPk(ForwardModelBias):
    """ForwardModelBias that outputs the measured P(k) instead of the raw density field.

    Runs the PM N-body + Lagrangian bias simulation via the parent class, then
    measures the 3-D power spectrum with Pylians.  The density field is computed
    transiently and never stored by Falcon.

    Inputs (same as ForwardModelBias)
    ------
    cosmo_params : (batch_size, D) float32
    delta_ic     : (batch_size, res, res, res) float32
    bias_params  : (batch_size, 4) float32   [b₁, b₂, bs², bn²]

    Output
    ------
    pk_measured : (batch_size, 2, N_bins) float32
        pk_measured[:, 0, :] = k   [h Mpc⁻¹]
        pk_measured[:, 1, :] = P(k) [(h⁻¹ Mpc)³]
    """

    def __init__(self, mas: str = "None", pk_threads: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mas = mas
        self.pk_threads = pk_threads

    def simulate_batch(
        self,
        batch_size: int,
        cosmo_params: np.ndarray,
        delta_ic: np.ndarray,
        bias_params: np.ndarray,
    ) -> np.ndarray:
        # 1. Run N-body + bias via parent class
        delta_biased = super().simulate_batch(batch_size, cosmo_params, delta_ic, bias_params)

        # 2. Measure P(k) for each sample with Pylians
        from .utils import pylians_pk

        results = []
        for i in range(batch_size):
            k, pk = pylians_pk(delta_biased[i], self.boxsize, self.mas, threads=self.pk_threads)
            results.append(np.stack([k, pk], axis=0))  # (2, N_bins)

        result = np.array(results, dtype=np.float32)  # (batch_size, 2, N_bins)

        falcon.log({
            "pk_measured_mean": float(result[:, 1, :].mean()),
            "pk_measured_max":  float(result[:, 1, :].max()),
        })
        return result

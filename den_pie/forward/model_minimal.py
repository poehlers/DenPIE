"""
Minimal fixed-cosmology forward model using DiscoDJ.

Pipeline (2-node DAG):

    delta_ic ──► delta_fin

The cosmology is fixed to the Planck 2018 fiducial (also adopted in the Quijote
simulations) throughout.  P(k) is computed exactly once at
InitialConditions.__init__ time and reused for every sample, so there is no
redundant DiscoDJ timetable or power-spectrum computation per batch call.

Fiducial cosmology — DiscoDJ parameter convention
--------------------------------------------------
The fiducial cosmology is given in terms of Omega_m (total matter density), but
DiscoDJ uses Omega_c (CDM only):
    Omega_c = Omega_m - Omega_b = 0.3175 - 0.0490 = 0.2685

Usage
-----
Generate prior samples (forward pass only):

    RUN_DIR="outputs/minimal_run_$(date +%y%m%d_%H%M%S)"
    falcon sample prior --config-name config_files/config_minimal.yml --run-dir "${RUN_DIR}"

Chunk over multiple calls to accumulate more samples (Falcon appends each run):

    for i in $(seq 1 $N_CHUNKS); do
        falcon sample prior --config-name config_files/config_minimal.yml --run-dir "${RUN_DIR}"
    done

Train the posterior estimator (uncomment evidence/observed in config_files/config_minimal.yml first):

    falcon train --config-name config_files/config_minimal.yml --run-dir "${RUN_DIR}"

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

from .bias import lagrangian_bias_weights

# Seed namespace for shot-noise RNG, decorrelating it from IC seeds.
_SN_SEED_NAMESPACE = 0x5000_0000

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Planck 2018 fiducial cosmology, also adopted in the Quijote simulations.
# DiscoDJ uses Omega_c (CDM only) = Omega_m - Omega_b.
_FIDUCIAL_COSMO = {
    "Omega_c": 0.3175 - 0.0490,   # = 0.2685  (Omega_m=0.3175, Omega_b=0.0490)
    "Omega_b": 0.0490,
    "h":       0.6711,
    "n_s":     0.9624,
    "sigma8":  0.8340,
}

BOX_PARAMS: dict = {
    "dim":     3,
    "res":     64,
    "boxsize": 1000.0,   # Mpc/h
}

SIM_PARAMS: dict = {
    "a_ini":                1.0 / (1.0 + 3.0),   # z = 3  →  scale factor ≈ 0.25
    "a_end":                1.0 / (1.0 + 0.0),   # z = 0  (present day)
    "stepper":              "bullfrog",
    "method":               "pm",
    "res_pm":               2 * BOX_PARAMS["res"],  # 128  (2× up-sampled PM mesh)
    "time_var":             "D",
    "alpha":                1.5,
    "theta":                0.5,
    "antialias":            0,
    "grad_kernel_order":    4,
    "laplace_kernel_order": 0,
    "n_resample":           1,
    "n_steps":              1,
    "deconvolve":           False,
    "nlpt_order_ics":       2,    # used in dj.with_lpt(n_order=...)
    "worder":               2,    # used in dj.get_delta_from_pos(worder=...)
}


# ─────────────────────────────────────────────────────────────────────────────
# Node: pk  (root node — no Falcon parents, fiducial P(k) only)
# ─────────────────────────────────────────────────────────────────────────────

class FixedPowerSpectrum:
    """Output the fiducial P(k) as a root node (no Falcon parents).

    P(k) is computed once at __init__ time using the fiducial cosmology.
    simulate_batch returns the same fiducial P(k) repeated batch_size times
    so each .npz stores it consistently with the varying-cosmology format.

    Output
    ------
    pk : (batch_size, N_k) float32
    """

    def __init__(
        self,
        dim:     int   = BOX_PARAMS["dim"],
        res:     int   = BOX_PARAMS["res"],
        boxsize: float = BOX_PARAMS["boxsize"],
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        _dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=_FIDUCIAL_COSMO)
        _dj = _dj.with_timetables().with_linear_ps()
        self._pk = np.asarray(_dj._pk_table["Pk"], dtype=np.float32)  # (N_k,)

    def simulate_batch(self, batch_size: int) -> np.ndarray:
        """Return fiducial P(k) repeated batch_size times.

        Returns
        -------
        pk : (batch_size, N_k) float32
        """
        return np.tile(self._pk, (batch_size, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Node: delta_ic  (root node — no Falcon parents)
# ─────────────────────────────────────────────────────────────────────────────

class InitialConditions:
    """Sample Gaussian-random linear initial conditions at fixed fiducial cosmology.

    P(k) and the k grid are computed once at __init__ time from the fiducial
    cosmology and cached for the lifetime of this actor.  Each call to
    simulate_batch draws fresh random seeds so successive batches are independent.

    This node has no Falcon parents — it is the root of the minimal DAG.

    Output
    ------
    delta_ic : (batch_size, res, res, res) float32  — linear density field at z = 0
    """

    def __init__(
        self,
        dim:            int   = BOX_PARAMS["dim"],
        res:            int   = BOX_PARAMS["res"],
        boxsize:        float = BOX_PARAMS["boxsize"],
        sub_batch_size: int   = 8,
        run_dir:        str   = "",
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.dim            = dim
        self.res            = res
        self.boxsize        = boxsize
        self.sub_batch_size = sub_batch_size
        self._run_dir       = run_dir

        # ── Compute P(k) once from fiducial cosmology ──────────────────────
        import jax
        import jax.numpy as jnp

        _dj = DiscoDJ(
            dim=dim, res=res, boxsize=boxsize,
            device="cpu", cosmo=_FIDUCIAL_COSMO,
        )
        _dj = _dj.with_timetables().with_linear_ps()

        # Cache k and Pk as JAX arrays — never recomputed again.
        self._k  = jnp.array(_dj._pk_table["k"])    # (N_k,)
        self._pk = jnp.array(_dj._pk_table["Pk"])   # (N_k,)

        # vmap over seeds only; k and Pk are broadcast (in_axes=None).
        self._jit_sample_ics     = jax.jit(self._sample_ics_impl)
        self._batched_sample_ics = jax.vmap(
            self._jit_sample_ics, in_axes=(None, None, 0)
        )

    # ------------------------------------------------------------------
    # JAX kernel
    # ------------------------------------------------------------------

    def _sample_ics_impl(self, k, pk, seed):
        """Return the linear density field for a single seed (JAX-traceable)."""
        dj = DiscoDJ(dim=self.dim, res=self.res, boxsize=self.boxsize, device="cpu")
        dj = dj.with_timetables()
        dj._pk_table["k"]  = k
        dj._pk_table["Pk"] = pk
        dj = dj.with_ics(seed=seed)
        return dj.get_delta_linear(a=1)

    # ------------------------------------------------------------------
    # Sub-batching (jax.lax.map over sub-batches for memory control)
    # ------------------------------------------------------------------

    def _sub_batched_sample(self, seeds):
        import jax

        B   = seeds.shape[0]
        sb  = self.sub_batch_size
        nsb = B // sb

        seeds_sub  = seeds.reshape((nsb, sb))
        results = jax.lax.map(
            lambda s: self._batched_sample_ics(self._k, self._pk, s),
            seeds_sub,
        )                                        # (nsb, sb, res, res, res)
        return results.reshape((B,) + results.shape[2:])

    # ------------------------------------------------------------------
    # New-Falcon interface  (no parent arguments — root node)
    # ------------------------------------------------------------------

    def simulate_batch(self, batch_size: int) -> np.ndarray:
        """Draw random ICs for a batch.  No parent inputs (root node).

        Returns
        -------
        delta_ic : (batch_size, res, res, res) float32
        """
        import jax.numpy as jnp
        from pathlib import Path

        # Seed = simulation index: count existing prior samples on disk so that
        # every chunk starts where the last one left off, guaranteeing uniqueness.
        if self._run_dir:
            prior_dir = Path(self._run_dir) / "samples_dir" / "prior"
            seed_offset = len(list(prior_dir.glob("*.npz"))) if prior_dir.exists() else 0
        else:
            seed_offset = 0
        seeds  = jnp.array(np.arange(seed_offset, seed_offset + batch_size, dtype=np.int32))
        result = np.asarray(self._sub_batched_sample(seeds), dtype=np.float32)

        falcon.log({
            "delta_ic_mean": float(result.mean()),
            "delta_ic_std":  float(result.std()),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node: delta_fin  (one parent: delta_ic)
# ─────────────────────────────────────────────────────────────────────────────

class ForwardModel:
    """Run a Particle-Mesh N-body simulation from ICs to z = 0 at fixed cosmology.

    Cosmology is fixed to the Planck 2018 fiducial — no per-sample cosmology overhead.

    Inputs
    ------
    delta_ic : (batch_size, res, res, res) float32

    Output
    ------
    delta_fin : (batch_size, res, res, res) float32  — non-linear density field at z = 0

    Memory management
    -----------------
    The batch is split into sub-batches of `sub_batch_size` via jax.lax.map,
    which runs them sequentially.  sub_batch_size=8 fits on an A100 at res=64.
    """

    def __init__(
        self,
        dim:            int   = BOX_PARAMS["dim"],
        res:            int   = BOX_PARAMS["res"],
        boxsize:        float = BOX_PARAMS["boxsize"],
        sim_params:     dict  = SIM_PARAMS,
        sub_batch_size: int   = 8,
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.dim            = dim
        self.res            = res
        self.boxsize        = boxsize
        self.sim_params     = dict(sim_params)
        self.sub_batch_size = sub_batch_size
        self._cosmo         = _FIDUCIAL_COSMO   # compile-time constant inside JAX kernels

        import jax
        devices     = jax.devices()
        self.device = devices[0]   # GPU if available, else CPU

        # vmap over delta_ic only (cosmo is fixed, not a traced argument).
        self._batched_simulate = jax.vmap(self._simulate_impl, in_axes=(0,))

    # ------------------------------------------------------------------
    # JAX kernel
    # ------------------------------------------------------------------

    def _simulate_impl(self, delta_ic):
        """Run one N-body simulation at fixed cosmology; return delta_fin."""
        sp = self.sim_params

        dj = DiscoDJ(
            dim=self.dim, res=self.res, boxsize=self.boxsize,
            device=self.device, cosmo=self._cosmo,
        )
        dj = dj.with_timetables()
        dj = dj.with_external_ics(delta=delta_ic)
        dj = dj.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

        _skip = {"nlpt_order_ics", "worder"}
        run_params = {k: v for k, v in sp.items() if k not in _skip}

        X_sim, _, _ = dj.run_nbody(**run_params, use_diffrax=False)
        delta_fin = dj.get_delta_from_pos(
            X_sim,
            res=dj.res,
            worder=sp["worder"],
            antialias=1,
            deconvolve=True,
        )
        return delta_fin

    # ------------------------------------------------------------------
    # Sub-batching
    # ------------------------------------------------------------------

    def _sub_batched_simulate(self, delta_ic_batch):
        import jax

        B   = delta_ic_batch.shape[0]
        sb  = self.sub_batch_size
        nsb = B // sb

        delta_ic_sub = delta_ic_batch.reshape((nsb, sb) + delta_ic_batch.shape[1:])
        results = jax.lax.map(
            lambda x: self._batched_simulate(x),
            delta_ic_sub,
        )                                        # (nsb, sb, res, res, res)
        return results.reshape(delta_ic_batch.shape)

    # ------------------------------------------------------------------
    # New-Falcon interface
    # ------------------------------------------------------------------

    def simulate_batch(self, batch_size: int, delta_ic: np.ndarray) -> np.ndarray:
        """Simulate the final density field for a batch of initial conditions.

        Parameters
        ----------
        delta_ic : (batch_size, res, res, res) float32

        Returns
        -------
        delta_fin : (batch_size, res, res, res) float32
        """
        import jax.numpy as jnp

        delta_ic_jax    = jnp.array(delta_ic, dtype=jnp.float32)
        delta_fin_batch = self._sub_batched_simulate(delta_ic_jax)
        result = np.asarray(delta_fin_batch, dtype=np.float32)

        falcon.log({
            "delta_fin_mean": float(result.mean()),
            "delta_fin_std":  float(result.std()),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node: delta_biased  (one parent: delta_ic, fixed cosmo + fixed bias)
# ─────────────────────────────────────────────────────────────────────────────

class FixedForwardModelBias:
    """N-body + Lagrangian bias + optional RSD at fixed cosmology and fixed bias.

    Used to generate fiducial "observed" simulations where only the random ICs
    vary.  Cosmology is fixed to the Planck 2018 fiducial; bias parameters are
    fixed at construction time.

    Inputs
    ------
    delta_ic : (batch_size, res, res, res) float32

    Output
    ------
    delta_biased : (batch_size, res, res, res) float32
    """

    def __init__(
        self,
        dim:                  int             = BOX_PARAMS["dim"],
        res:                  int             = BOX_PARAMS["res"],
        boxsize:              float           = BOX_PARAMS["boxsize"],
        sim_params:           dict            = SIM_PARAMS,
        sub_batch_size:       int             = 1,
        apply_rsd:            bool            = False,
        los_axis:             int             = 2,
        fiducial_bias_params: dict            = {"b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0},
        n_g_density:          Optional[float] = None,
        run_dir:              str             = "",
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.dim            = dim
        self.res            = res
        self.boxsize        = boxsize
        self.sim_params     = dict(sim_params)
        self.sub_batch_size = sub_batch_size
        self.apply_rsd      = apply_rsd
        self.los_axis       = los_axis
        self._cosmo         = _FIDUCIAL_COSMO
        self.n_g_density    = n_g_density   # galaxies per (h/Mpc)^3
        self._run_dir       = run_dir

        self._b1  = fiducial_bias_params["b1"]
        self._b2  = fiducial_bias_params["b2"]
        self._bs2 = fiducial_bias_params["bs2"]
        self._bn2 = fiducial_bias_params["bn2"]

        import jax
        devices     = jax.devices()
        self.device = devices[0]

        _dj = DiscoDJ(
            dim=dim, res=res, boxsize=boxsize,
            device="cpu", cosmo=_FIDUCIAL_COSMO,
        )
        self._k_vecs = _dj.k_vecs

        self._batched_simulate = jax.vmap(self._simulate_impl, in_axes=(0,))

    # ------------------------------------------------------------------
    # JAX kernel
    # ------------------------------------------------------------------

    def _simulate_impl(self, delta_ic):
        """Single-sample: run N-body at fixed cosmo, apply fixed bias, scatter."""
        sp = self.sim_params

        dj = DiscoDJ(
            dim=self.dim, res=self.res, boxsize=self.boxsize,
            device=self.device, cosmo=self._cosmo,
        )
        dj = dj.with_timetables()
        dj = dj.with_external_ics(delta=delta_ic)
        dj = dj.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

        _skip = {"nlpt_order_ics", "worder"}
        run_params = {k: v for k, v in sp.items() if k not in _skip}

        X_sim, P, _ = dj.run_nbody(**run_params, use_diffrax=False)
        P_flat = P.reshape(-1, 3) if self.apply_rsd else None

        weights = lagrangian_bias_weights(
            delta_ic, self._k_vecs,
            b1=self._b1, b2=self._b2, bs2=self._bs2, bn2=self._bn2,
        )

        n_biased = dj.compute_field_quantity_from_particles(
            pos=X_sim,
            quantity=weights.reshape(-1),
            normalize_by_density=False,
            in_redshift_space=self.apply_rsd,
            vel=P_flat,
            a=sp["a_end"] if self.apply_rsd else None,
            radial_dim=self.los_axis,
            worder=sp["worder"],
            antialias=1,
            deconvolve=True,
        )

        return n_biased / n_biased.mean() - 1.0

    # ------------------------------------------------------------------
    # Sub-batching
    # ------------------------------------------------------------------

    def _sub_batched_simulate(self, delta_ic_batch):
        import jax

        B   = delta_ic_batch.shape[0]
        sb  = self.sub_batch_size
        nsb = B // sb

        delta_ic_sub = delta_ic_batch.reshape((nsb, sb) + delta_ic_batch.shape[1:])
        results = jax.lax.map(
            lambda x: self._batched_simulate(x),
            delta_ic_sub,
        )
        return results.reshape(delta_ic_batch.shape)

    # ------------------------------------------------------------------
    # New-Falcon interface
    # ------------------------------------------------------------------

    def simulate_batch(self, batch_size: int, delta_ic: np.ndarray) -> np.ndarray:
        """Simulate the biased tracer density field for a batch of ICs.

        Parameters
        ----------
        delta_ic : (batch_size, res, res, res) float32

        Returns
        -------
        delta_biased : (batch_size, res, res, res) float32
        """
        import jax.numpy as jnp

        delta_jax = jnp.array(delta_ic, dtype=jnp.float32)
        # np.array (not np.asarray) — forces a writable copy. np.asarray
        # on a JAX array can return a read-only DLPack view, which would
        # break the in-place result[i] += ... shot-noise injection below.
        result = np.array(
            self._sub_batched_simulate(delta_jax),
            dtype=np.float32,
        )

        # ── Optional shot noise (Poisson Gaussian limit, arXiv:2504.20130v2) ──
        # δ_g ← δ_biased + ε,  ε ~ N(0, 1/N̄_g)  per voxel
        # with N̄_g = n̄_g · V_cell, V_cell = (boxsize / res)^3.
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
            "delta_biased_mean": float(result.mean()),
            "delta_biased_std":  float(result.std()),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node: pk_measured  (fixed-cosmology N-body → P(k) via Pylians)
# ─────────────────────────────────────────────────────────────────────────────

class ForwardModelPk(ForwardModel):
    """Fixed-cosmology ForwardModel that outputs P(k) instead of the density field.

    Runs the PM N-body simulation via the parent class, then measures the 3-D
    power spectrum with Pylians.  The density field is computed transiently and
    never stored by Falcon.

    Inputs (same as ForwardModel)
    ------
    delta_ic : (batch_size, res, res, res) float32

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

    def simulate_batch(self, batch_size: int, delta_ic: np.ndarray) -> np.ndarray:
        # 1. Run N-body via parent class
        delta_fin = super().simulate_batch(batch_size, delta_ic)

        # 2. Measure P(k) for each sample with Pylians
        from .utils import pylians_pk

        results = []
        for i in range(batch_size):
            k, pk = pylians_pk(delta_fin[i], self.boxsize, self.mas, threads=self.pk_threads)
            results.append(np.stack([k, pk], axis=0))  # (2, N_bins)

        result = np.array(results, dtype=np.float32)  # (batch_size, 2, N_bins)

        falcon.log({
            "pk_measured_mean": float(result[:, 1, :].mean()),
            "pk_measured_max":  float(result[:, 1, :].max()),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node: pk_measured  (fixed-cosmology + fixed-bias N-body → P(k) via Pylians)
# ─────────────────────────────────────────────────────────────────────────────

class FixedForwardModelBiasPk(FixedForwardModelBias):
    """Fixed-cosmology + fixed-bias model that outputs P(k) instead of the density field.

    Runs the PM N-body + Lagrangian bias simulation via the parent class, then
    measures the 3-D power spectrum with Pylians.  The density field is computed
    transiently and never stored by Falcon.

    Inputs (same as FixedForwardModelBias)
    ------
    delta_ic : (batch_size, res, res, res) float32

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

    def simulate_batch(self, batch_size: int, delta_ic: np.ndarray) -> np.ndarray:
        # 1. Run N-body + bias via parent class
        delta_biased = super().simulate_batch(batch_size, delta_ic)

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

"""
Forward model for cosmological N-body simulations using DiscoDJ.

Pipeline (4-node DAG):

    cosmo_params  ──────────────────────────────┐
         │                                      ▼
         ▼                                ForwardModel
    PowerSpectrum ──► InitialConditions ──────┘
         (pk)              (delta_ic)          (delta_fin)

Parameters (Ω_m, Ω_b, h, n_s, σ_8) are sampled from uniform priors defined in
config_files/config.yml via falcon.priors.Hypercube.  The three classes in this
file cover the remaining nodes: pk, delta_ic, delta_fin.

By default no noise is stored — apply it on-the-fly in the training dataloader:
    x_obs = delta_fin + sigma * torch.randn_like(delta_fin)
ForwardModel can, however, optionally map the final field into redshift space
(apply_rsd=True) and bake in Gaussian shot noise (n_g_density set), in which case
delta_fin is stored already in redshift space with shot noise included.

Usage
-----
Generate prior samples (forward pass only):

    RUN_DIR="outputs/forward_run_$(date +%y%m%d_%H%M%S)"
    falcon sample prior --config-name config_files/config.yml --run-dir "${RUN_DIR}"

Chunk over multiple calls to accumulate more samples (Falcon appends each run):

    for i in $(seq 1 $N_CHUNKS); do
        falcon sample prior --config-name config_files/config.yml --run-dir "${RUN_DIR}"
    done

Train the posterior estimator (uncomment evidence/observed in config_files/config.yml first):

    falcon train --config-name config_files/config.yml --run-dir "${RUN_DIR}"

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

# Seed namespace for shot-noise RNG, decorrelating it from IC seeds.
_SN_SEED_NAMESPACE = 0x5000_0000

# ─────────────────────────────────────────────────────────────────────────────
# Shared simulation parameters
# ─────────────────────────────────────────────────────────────────────────────

BOX_PARAMS: dict = {
    "dim":     3,
    "res":     64,
    "boxsize": 1000.0,   # Mpc/h
}

# Parameters forwarded to DiscoDJ.run_nbody (plus nlpt_order_ics / worder
# which are handled separately in _simulate_impl).
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

# Planck 2018 fiducial cosmology (also adopted in the Quijote simulations).
# Used only to pre-compute the k grid in InitialConditions.__init__
# (k values depend solely on box geometry, not on cosmology).
_FIDUCIAL_COSMO: dict = {
    "Omega_c": 0.3175 - 0.0490,   # = 0.2685  (Omega_m=0.3175, Omega_b=0.0490)
    "Omega_b": 0.0490,
    "h":       0.6711,
    "n_s":     0.9624,
    "sigma8":  0.8340,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the DiscoDJ cosmo dict from a sampled parameter vector
# ─────────────────────────────────────────────────────────────────────────────

_COSMO_PARAM_ORDER = ["Omega_m", "Omega_b", "h", "n_s", "sigma8"]

def _cosmo_dict(cosmo_params, fixed=None):
    """Convert sampled cosmo_params (+ optional fixed params) → DiscoDJ cosmo dict.

    Parameters
    ----------
    cosmo_params : 1-D JAX array
        Sampled parameters in canonical order [Ω_m, Ω_b, h, n_s, σ_8],
        containing only the parameters NOT present in ``fixed``.
        Pass all 5 when ``fixed`` is None (default, full-parameter model).
    fixed : dict, optional
        Fixed parameter values keyed by canonical name (e.g.
        ``{"Omega_b": 0.049, "h": 0.6711, "n_s": 0.9624}``).
        ``fixed`` is a Python dict and is resolved at JAX trace time, so
        it is safe to use inside jit / vmap.
    """
    if not fixed:
        Om, Ob, h, ns, s8 = cosmo_params
    else:
        sampled_keys = [k for k in _COSMO_PARAM_ORDER if k not in fixed]
        vals = dict(fixed)
        for i, k in enumerate(sampled_keys):
            vals[k] = cosmo_params[i]
        Om, Ob, h, ns, s8 = (vals[k] for k in _COSMO_PARAM_ORDER)
    return {"h": h, "Omega_b": Ob, "Omega_c": Om - Ob, "n_s": ns, "sigma8": s8}


# ─────────────────────────────────────────────────────────────────────────────
# Node: pk
# ─────────────────────────────────────────────────────────────────────────────

class PowerSpectrum:
    """Compute the linear matter power spectrum P(k) for a batch of cosmologies.

    Inputs
    ------
    cosmo_params : (batch_size, N) float32  [Ω_m, Ω_b, h, n_s, σ_8] or a subset
        N = 5 when all parameters are sampled; N < 5 when some are fixed via
        the ``fixed_cosmo_params`` constructor argument.

    Output
    ------
    pk : (batch_size, N_k) float32   — P(k) evaluated on DiscoDJ's internal k grid.

    Note: only P(k) is returned (not k itself).  The k grid is a pure function of
    box geometry and is pre-computed once inside InitialConditions.__init__.
    """

    def __init__(
        self,
        dim:                int   = BOX_PARAMS["dim"],
        res:                int   = BOX_PARAMS["res"],
        boxsize:            float = BOX_PARAMS["boxsize"],
        fixed_cosmo_params: dict  = {},
    ) -> None:
        # GPU visibility must be set before JAX initialises.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.dim                  = dim
        self.res                  = res
        self.boxsize              = boxsize
        self._fixed_cosmo_params  = dict(fixed_cosmo_params)

        # Compile the single-sample kernel and vectorise over the batch axis.
        import jax
        self._jit_sample_pk     = jax.jit(self._sample_pk_impl)
        self._batched_sample_pk = jax.vmap(self._jit_sample_pk)

    # ------------------------------------------------------------------
    # JAX kernel (jit + vmap compatible — no Python side-effects)
    # ------------------------------------------------------------------

    def _sample_pk_impl(self, cosmo_params):
        """Return P(k) for a single cosmology (JAX array, shape (N_k,))."""
        dj = DiscoDJ(
            dim=self.dim, res=self.res, boxsize=self.boxsize,
            device="cpu", cosmo=_cosmo_dict(cosmo_params, fixed=self._fixed_cosmo_params),
        )
        dj = dj.with_timetables()
        dj = dj.with_linear_ps()
        return dj._pk_table["Pk"]

    # ------------------------------------------------------------------
    # New-Falcon interface
    # ------------------------------------------------------------------

    def simulate_batch(self, batch_size: int, cosmo_params: np.ndarray) -> np.ndarray:
        """Compute P(k) for a batch of cosmological parameters.

        Parameters
        ----------
        cosmo_params : (batch_size, 5) float32

        Returns
        -------
        pk : (batch_size, N_k) float32
        """
        import jax.numpy as jnp

        cosmo_jax = jnp.array(cosmo_params, dtype=jnp.float32)
        pk_batch  = self._batched_sample_pk(cosmo_jax)           # (B, N_k)
        result    = np.asarray(pk_batch, dtype=np.float32)

        falcon.log({
            "pk_mean": float(result.mean()),
            "pk_max":  float(result.max()),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node: delta_ic
# ─────────────────────────────────────────────────────────────────────────────

class InitialConditions:
    """Generate Gaussian-random linear initial conditions from P(k).

    Inputs
    ------
    pk : (batch_size, N_k) float32   — P(k) from the PowerSpectrum node

    Output
    ------
    delta_ic : (batch_size, res, res, res) float32  — linear density field at z = 0
               (scale factor a = 1, before non-linear evolution)

    Each sample uses a different random integer seed so that successive calls to
    simulate_batch always produce independent realisations.
    """

    def __init__(
        self,
        dim:     int   = BOX_PARAMS["dim"],
        res:     int   = BOX_PARAMS["res"],
        boxsize: float = BOX_PARAMS["boxsize"],
        run_dir: str   = "",
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.dim      = dim
        self.res      = res
        self.boxsize  = boxsize
        self._run_dir = run_dir

        # Pre-compute the k grid once using the fiducial cosmology.
        # k depends only on box geometry, so any cosmology gives the same result.
        import jax
        import jax.numpy as jnp

        _dj = DiscoDJ(
            dim=dim, res=res, boxsize=boxsize,
            device="cpu", cosmo=_FIDUCIAL_COSMO,
        )
        _dj = _dj.with_timetables()
        _dj = _dj.with_linear_ps()
        self._k = jnp.array(_dj._pk_table["k"])   # (N_k,) — cached

        # vmap over (pk, seed) pairs; k is shared across the batch (in_axes=None).
        self._jit_sample_ics     = jax.jit(self._sample_ics_impl)
        self._batched_sample_ics = jax.vmap(
            self._jit_sample_ics, in_axes=(None, 0, 0)   # k: broadcast, pk+seed: batched
        )

    # ------------------------------------------------------------------
    # JAX kernel
    # ------------------------------------------------------------------

    def _sample_ics_impl(self, k, pk, seed):
        """Return the linear density field for a single (pk, seed) pair."""
        dj = DiscoDJ(dim=self.dim, res=self.res, boxsize=self.boxsize, device="cpu")
        dj = dj.with_timetables()
        dj._pk_table["k"]  = k
        dj._pk_table["Pk"] = pk
        dj = dj.with_ics(seed=seed)
        return dj.get_delta_linear(a=1)

    # ------------------------------------------------------------------
    # New-Falcon interface
    # ------------------------------------------------------------------

    def simulate_batch(self, batch_size: int, pk: np.ndarray) -> np.ndarray:
        """Draw random initial conditions for a batch of power spectra.

        Parameters
        ----------
        pk : (batch_size, N_k) float32

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
        pk_jax = jnp.array(pk, dtype=jnp.float32)

        delta_ic_batch = self._batched_sample_ics(self._k, pk_jax, seeds)  # (B, res³)
        result = np.asarray(delta_ic_batch, dtype=np.float32)

        falcon.log({
            "delta_ic_mean": float(result.mean()),
            "delta_ic_std":  float(result.std()),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node: delta_fin
# ─────────────────────────────────────────────────────────────────────────────

class ForwardModel:
    """Run a Particle-Mesh N-body simulation from initial conditions to z = 0.

    Inputs (in parent order, matching config.yml parents: [cosmo_params, delta_ic])
    ------
    cosmo_params : (batch_size, N) float32   [Ω_m, Ω_b, h, n_s, σ_8] or a subset
        N = 5 when all parameters are sampled; N < 5 when some are fixed via
        the ``fixed_cosmo_params`` constructor argument.
    delta_ic     : (batch_size, res, res, res) float32

    Output
    ------
    delta_fin : (batch_size, res, res, res) float32  — non-linear density field at z = 0

    Memory management
    -----------------
    GPU memory limits the vmap batch size.  The batch is split into sub-batches
    of `sub_batch_size` using jax.lax.map, which runs them sequentially on the
    GPU.  sub_batch_size=8 fits comfortably on an A100 at res=64.
    """

    def __init__(
        self,
        dim:                int             = BOX_PARAMS["dim"],
        res:                int             = BOX_PARAMS["res"],
        boxsize:            float           = BOX_PARAMS["boxsize"],
        sim_params:         dict            = SIM_PARAMS,
        sub_batch_size:     int             = 8,
        fixed_cosmo_params: dict            = {},
        apply_rsd:          bool            = False,
        los_axis:           int             = 2,
        n_g_density:        Optional[float] = None,
        run_dir:            str             = "",
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.dim                 = dim
        self.res                 = res
        self.boxsize             = boxsize
        self.sim_params          = dict(sim_params)
        self.sub_batch_size      = sub_batch_size
        self._fixed_cosmo_params = dict(fixed_cosmo_params)
        self.apply_rsd           = apply_rsd
        self.los_axis            = los_axis
        self.n_g_density         = n_g_density   # galaxies per (h/Mpc)^3
        self._run_dir            = run_dir

        import jax
        devices     = jax.devices()
        self.device = devices[0]   # GPU if available, else CPU

        # vmap a single simulation over the sub-batch axis.
        self._batched_simulate = jax.vmap(self._simulate_impl, in_axes=(0, 0))

    # ------------------------------------------------------------------
    # JAX kernel
    # ------------------------------------------------------------------

    def _simulate_impl(self, delta_ic, cosmo_params):
        """Run one N-body simulation; return the final density field."""
        sp = self.sim_params

        dj = DiscoDJ(
            dim=self.dim, res=self.res, boxsize=self.boxsize,
            device=self.device, cosmo=_cosmo_dict(cosmo_params, fixed=self._fixed_cosmo_params),
        )
        dj = dj.with_timetables()
        dj = dj.with_external_ics(delta=delta_ic)
        dj = dj.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)

        # Collect only the keys that run_nbody accepts.
        _skip = {"nlpt_order_ics", "worder"}
        run_params = {k: v for k, v in sp.items() if k not in _skip}

        X_sim, P, _ = dj.run_nbody(**run_params, use_diffrax=False)

        if self.apply_rsd:
            # Redshift space: displace particles along the line of sight by their
            # peculiar velocity before scattering (Kaiser RSD).  Uniform weights
            # (=1) give the unbiased matter field.  P must be pre-flattened so
            # vel[:, radial_dim] indexes the particle axis (see model_bias.py).
            import jax.numpy as jnp
            P_flat = P.reshape(-1, 3)
            n_field = dj.compute_field_quantity_from_particles(
                pos=X_sim,
                quantity=jnp.ones((self.res ** self.dim,), dtype=X_sim.dtype),
                normalize_by_density=False,
                in_redshift_space=True,
                vel=P_flat,
                a=sp["a_end"],
                radial_dim=self.los_axis,
                worder=sp["worder"],
                antialias=1,
                deconvolve=True,
            )
            return n_field / n_field.mean() - 1.0

        delta_fin = dj.get_delta_from_pos(
            X_sim,
            res=dj.res,
            worder=sp["worder"],
            antialias=1,
            deconvolve=True,
        )
        return delta_fin

    def _sub_batched_simulate(self, delta_ic_batch, cosmo_params_batch):
        """Split the batch into sub-batches and run them sequentially via lax.map."""
        import jax

        B   = delta_ic_batch.shape[0]
        sb  = self.sub_batch_size
        nsb = B // sb   # number of sub-batches (B must be divisible by sb)

        delta_ic_sub     = delta_ic_batch.reshape((nsb, sb) + delta_ic_batch.shape[1:])
        cosmo_params_sub = cosmo_params_batch.reshape((nsb, sb) + cosmo_params_batch.shape[1:])

        results = jax.lax.map(
            lambda x: self._batched_simulate(x[0], x[1]),
            (delta_ic_sub, cosmo_params_sub),
        )
        return results.reshape(delta_ic_batch.shape)

    # ------------------------------------------------------------------
    # New-Falcon interface
    # ------------------------------------------------------------------

    def simulate_batch(
        self,
        batch_size: int,
        cosmo_params: np.ndarray,
        delta_ic: np.ndarray,
    ) -> np.ndarray:
        """Simulate the final density field for a batch of (cosmology, IC) pairs.

        Parameters
        ----------
        cosmo_params : (batch_size, 5) float32
        delta_ic     : (batch_size, res, res, res) float32

        Returns
        -------
        delta_fin : (batch_size, res, res, res) float32
        """
        import jax.numpy as jnp

        cosmo_jax    = jnp.array(cosmo_params, dtype=jnp.float32)
        delta_ic_jax = jnp.array(delta_ic,     dtype=jnp.float32)

        # np.array (not np.asarray) — forces a writable copy. np.asarray on a JAX
        # array can return a read-only view, which would break the in-place
        # result[i] += ... shot-noise injection below.
        result = np.array(
            self._sub_batched_simulate(delta_ic_jax, cosmo_jax),
            dtype=np.float32,
        )

        # ── Optional shot noise (Poisson Gaussian limit, arXiv:2504.20130v2) ──
        # δ ← δ + ε,  ε ~ N(0, 1/N̄_g) per voxel, N̄_g = n̄_g · V_cell,
        # V_cell = (boxsize / res)^3.  Seed is deterministic via the same
        # file-counting pattern as InitialConditions, namespaced with
        # _SN_SEED_NAMESPACE so it never collides with the IC RNG.
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
                result[i] += rng.normal(0.0, sigma_n, result[i].shape).astype(np.float32)
            falcon.log({
                "shot_noise_n_g_density":     float(self.n_g_density),
                "shot_noise_N_bar_per_voxel": float(N_bar),
                "shot_noise_sigma":           float(sigma_n),
            })

        falcon.log({
            "delta_fin_mean": float(result.mean()),
            "delta_fin_std":  float(result.std()),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node: pk_measured  (N-body → P(k) via Pylians)
# ─────────────────────────────────────────────────────────────────────────────

class ForwardModelPk(ForwardModel):
    """ForwardModel that outputs the measured P(k) instead of the raw density field.

    Runs the PM N-body simulation via the parent class, then measures the 3-D
    power spectrum with Pylians.  The density field is computed transiently and
    never stored by Falcon.

    Inputs (same as ForwardModel)
    ------
    cosmo_params : (batch_size, N) float32
    delta_ic     : (batch_size, res, res, res) float32

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
    ) -> np.ndarray:
        # 1. Run N-body via parent class
        delta_fin = super().simulate_batch(batch_size, cosmo_params, delta_ic)

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

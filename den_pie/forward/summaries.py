"""Summary-statistic simulator nodes for the biased-tracer Falcon pipeline.

These nodes consume a parent density field and emit P_l multipoles or the
bispectrum monopole via the BFast library, using the exact same calls
SBI_Pk (/home/poehlers/cosmo_thesis/SBI_Pk/src/main_sbi.py) makes when
building its cache, so the outputs are byte-identical to what SBI_Pk would
otherwise compute itself.

New-Falcon interface contract:
  - simulate_batch(batch_size, field) -> np.ndarray
  - field is the parent array, already batched (first dim = batch_size)
  - Return value must be numpy float32 with first dim = batch_size
  - JAX must NOT be imported at module level (breaks Ray actor serialisation)
"""

import os
import numpy as np
import falcon


class PkMultipoles:
    """BFast P_l(k) for l in {0, 2, 4} of a single parent density field.

    Output
    ------
    pk_l : (batch_size, 4, n_k) float32
        Channel 0 = k (h/Mpc), channels 1..3 = P_0, P_2, P_4 [(h^-1 Mpc)^3].

    Notes
    -----
    Bin edges match SBI_Pk's convention (main_sbi.py:626-629):
        pk_edges = jnp.arange(1, res // 2 + 1)   # integer k_F up to Nyquist

    mas_order=0 is correct for both delta_ic (no MAS filter, gridded GRF)
    and delta_biased (already CIC-deconvolved by DiscoDJ).
    """

    def __init__(
        self,
        boxsize:   float,
        res:       int,
        los_axis:  int = 2,
        mas_order: int = 0,
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.boxsize   = boxsize
        self.res       = res
        self.los_axis  = los_axis
        self.mas_order = mas_order

    def simulate_batch(self, batch_size: int, field: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp
        import BFast

        edges = jnp.arange(1, self.res // 2 + 1)

        out = []
        for i in range(batch_size):
            cube = jnp.asarray(np.ascontiguousarray(field[i].astype(np.float32)))
            r = BFast.Pk(
                cube, self.boxsize, edges,
                mas_order=self.mas_order,
                multipole_axis=self.los_axis,
                jit=True,
            )
            out.append(np.stack([
                np.asarray(r["k"],   dtype=np.float32),
                np.asarray(r["Pk0"], dtype=np.float32),
                np.asarray(r["Pk2"], dtype=np.float32),
                np.asarray(r["Pk4"], dtype=np.float32),
            ], axis=0))

        result = np.asarray(out, dtype=np.float32)   # (B, 4, n_k)
        falcon.log({
            "pk0_mean": float(result[:, 1, :].mean()),
            "pk0_max":  float(result[:, 1, :].max()),
        })
        return result


class BkMonopole:
    """BFast B_0 of a single parent density field on open triangles.

    Output
    ------
    bk_0 : (batch_size, n_tri, 4) float32
        Per triangle: [k1, k2, k3, B_0]  (k's are k-bin centres in h/Mpc).

    Notes
    -----
    Bin edges match SBI_Pk's convention (main_sbi.py:632-635):
        bk_edges = jnp.arange(1, res // 3 + 1, dk_kF)   # in units of k_F

    `open_triangles=True` returns triangle_centers alongside Bk.
    """

    def __init__(
        self,
        boxsize:   float,
        res:       int,
        dk_kF:     int = 3,
        mas_order: int = 0,
    ) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.boxsize   = boxsize
        self.res       = res
        self.dk_kF     = int(dk_kF)
        self.mas_order = mas_order

    def simulate_batch(self, batch_size: int, field: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp
        import BFast

        edges = jnp.arange(1, self.res // 3 + 1, self.dk_kF)

        out = []
        for i in range(batch_size):
            cube = jnp.asarray(np.ascontiguousarray(field[i].astype(np.float32)))
            b = BFast.Bk(
                cube, self.boxsize, edges,
                mas_order=self.mas_order,
                open_triangles=True, fast=True,
                only_B=True, jit=True,
            )
            tri = np.asarray(b["triangle_centers"], dtype=np.float32)   # (n_tri, 3)
            Bk  = np.asarray(b["Bk"],               dtype=np.float32)   # (n_tri,)
            out.append(np.concatenate([tri, Bk[:, None]], axis=1))      # (n_tri, 4)

        result = np.asarray(out, dtype=np.float32)   # (B, n_tri, 4)
        falcon.log({
            "bk0_mean": float(result[:, :, 3].mean()),
            "bk0_absmax": float(np.abs(result[:, :, 3]).max()),
        })
        return result

"""Pure JAX field operators for second-order Lagrangian bias (Modi+2020).

Used by ``core.model.model_bias.ForwardModelBias`` to compute per-particle
bias weights on the Lagrangian grid.  All functions are JAX-traceable
(jit / vmap compatible) and free of DiscoDJ instances or Falcon imports.
They operate on Fourier-space arrays and sparse k-vectors of the kind
produced by ``DiscoDJ.k_vecs``.

Physics
-------
Bias weight per particle at Lagrangian position q:

    w(q) = 1 + b₁·δ(q) + b₂·[δ²(q)−⟨δ²⟩] + bs²·[s²(q)−⟨s²⟩] + bn²·∇²δ(q)

All fields are evaluated on the initial regular Lagrangian grid.  Because the
Lagrangian grid coincides with the regular initial mesh, no CIC interpolation
is required: the weight for particle (i,j,k) is simply ``w[i,j,k]``.

Tidal shear squared:

    T̂_ij(k) = (k_i k_j / k²) · δ̂(k)   (dimensionless tidal tensor)
    s²(q)   = Tr(T²) − (1/3)·δ²
            = T_xx² + T_yy² + T_zz² + 2·(T_xy² + T_xz² + T_yz²) − (1/3)·δ²

Laplacian:

    ∇̂²δ(k) = −k² · δ̂(k)
"""

import jax.numpy as jnp
from discodj.core.kernels import gradient_kernel, inv_laplace_kernel


def _tidal_shear_sq(delta_k, k_vecs):
    """Compute the tidal shear squared s²(q) from the Fourier-space density.

    Parameters
    ----------
    delta_k : complex array, shape (res, res, res//2+1)
        Fourier-space density contrast, i.e. ``jnp.fft.rfftn(delta_ic)``.
    k_vecs : list of 3 sparse JAX arrays
        From ``DiscoDJ.k_vecs``; broadcast over the 3-D Fourier mesh.

    Returns
    -------
    s2 : float array, shape (res, res, res)
    """
    # Gravitational potential: Φ̂(k) = -δ̂(k) / k²  =  inv_laplace_kernel · δ̂
    phi_k = delta_k * inv_laplace_kernel(k_vecs)   # shape (res, res, res//2+1)

    # Gradient kernels: g_i = i·k_i  (purely imaginary for real k_i).
    # g_i · g_j = (i·k_i)(i·k_j) = −k_i·k_j  — real valued, Hermitian-preserving.
    # T_ij = IRFFT(g_i · g_j · Φ̂)
    g = [gradient_kernel(k_vecs, i) for i in range(3)]   # [i·kx, i·ky, i·kz]

    T_xx = jnp.fft.irfftn(g[0] * g[0] * phi_k)
    T_yy = jnp.fft.irfftn(g[1] * g[1] * phi_k)
    T_zz = jnp.fft.irfftn(g[2] * g[2] * phi_k)
    T_xy = jnp.fft.irfftn(g[0] * g[1] * phi_k)
    T_xz = jnp.fft.irfftn(g[0] * g[2] * phi_k)
    T_yz = jnp.fft.irfftn(g[1] * g[2] * phi_k)

    delta = jnp.fft.irfftn(delta_k)

    return (T_xx**2 + T_yy**2 + T_zz**2
            + 2.0 * (T_xy**2 + T_xz**2 + T_yz**2)
            - (1.0 / 3.0) * delta**2)


def _laplacian(delta_k, k_vecs):
    """Compute the real-space Laplacian ∇²δ(q) from the Fourier-space density.

    ∇̂²δ(k) = −k² · δ̂(k)

    Parameters
    ----------
    delta_k : complex array, shape (res, res, res//2+1)
        Fourier-space density contrast.
    k_vecs : list of 3 sparse JAX arrays

    Returns
    -------
    lap_delta : float array, shape (res, res, res)
    """
    k_sq = sum(k**2 for k in k_vecs)   # (res, 1, 1) + (1, res, 1) + (1, 1, res//2+1)
    return jnp.fft.irfftn(-k_sq * delta_k)


def lagrangian_bias_weights(delta_ic, k_vecs, b1, b2, bs2, bn2):
    """Compute second-order Lagrangian bias weights on the initial Lagrangian grid.

    w(q) = 1 + b₁·δ(q) + b₂·[δ²(q)−⟨δ²⟩] + bs²·[s²(q)−⟨s²⟩] + bn²·∇²δ(q)

    All fields are evaluated at Lagrangian grid positions q.  Because q
    coincides with the regular initial grid, the weight for particle (i,j,k)
    is simply ``w[i,j,k]``.

    Parameters
    ----------
    delta_ic : float array, shape (res, res, res)
        Linear IC density field (real space).
    k_vecs : list of 3 sparse JAX arrays
        From ``DiscoDJ.k_vecs``; pre-computed once in ``ForwardModelBias.__init__``.
    b1, b2, bs2, bn2 : float scalars (or 0-d JAX arrays)
        Linear, quadratic, tidal-shear, and Laplacian bias parameters.

    Returns
    -------
    weights : float array, shape (res, res, res)
        Per-particle Lagrangian bias weights.  Unbiased matter corresponds to
        b1=1, b2=bs2=bn2=0, which gives weights=1+delta_ic (mean ≈ 1).
    """
    delta_k = jnp.fft.rfftn(delta_ic)

    s2     = _tidal_shear_sq(delta_k, k_vecs)
    lap    = _laplacian(delta_k, k_vecs)
    delta2 = delta_ic**2

    return (1.0
            + b1  * delta_ic
            + b2  * (delta2 - delta2.mean())
            + bs2 * (s2     - s2.mean())
            + bn2 * lap)

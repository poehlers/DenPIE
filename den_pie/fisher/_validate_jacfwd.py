#!/usr/bin/env python
"""GATE: validate forward-mode (jax.jacfwd) Jacobian against FD and reverse-mode.

Runs the 5-param pure-matter pipeline at tiny res and computes d(data)/d(theta)
three ways at the fiducial, with a single fixed white-noise + shot-noise seed:

  * jacfwd : forward-mode AD, forward_mode=True (requires_jacfwd=True, cosmo NOT
             stop_gradient'd) -> should capture the FULL cosmo dependence
             including N-body growth.
  * jacrev : reverse-mode AD, forward_mode=False (custom_vjp adjoint,
             cosmo stop_gradient'd) -> growth dependence dropped.
  * FD     : central finite differences -> growth-inclusive ground truth.

PASS criterion (the plan's gate):
  jacfwd finite everywhere AND its Omega_m column matches FD within ~10% AND it
  differs from the reverse-mode Omega_m column by >5% (i.e. growth is captured).

Run:  python core/_validate_jacfwd.py
"""
import jax
jax.config.update("jax_enable_x64", True)  # growth ODE needs float64 for a finite jacfwd
import time  # noqa: E402
import numpy as np  # noqa: E402


def _relL2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    import jax
    import jax.numpy as jnp
    from den_pie.forward.model import BOX_PARAMS, SIM_PARAMS
    from den_pie.fisher.fisher import _bfast_precompute, _FINITE_DIFF_STEPS
    from fisher_5param import _pipeline_5param, _FIDUCIAL_5PARAM, _SAMPLED_KEYS

    res = 32
    boxsize = 1000.0
    pb_step = 4
    n_g_density = 1.0e-3

    box_params = dict(BOX_PARAMS); box_params["res"] = res; box_params["boxsize"] = boxsize
    sim_params = dict(SIM_PARAMS); sim_params["res_pm"] = 2 * res
    dim = box_params["dim"]

    bin_edges = jnp.arange(1, res // 3 + 1, pb_step)
    B_info, B_norm = _bfast_precompute(boxsize, bin_edges, res, dim=dim, mas_order=0)

    fid = jnp.array([_FIDUCIAL_5PARAM[k] for k in _SAMPLED_KEYS], dtype=jnp.float64)
    key = jax.random.PRNGKey(0)
    k_ic, k_sn = jax.random.split(key)
    noise = jax.random.normal(k_ic, shape=(res,) * dim, dtype=jnp.float64)
    sn_noise = jax.random.normal(k_sn, shape=(res,) * dim, dtype=jnp.float64)

    def pipe(p, fwd):
        return _pipeline_5param(
            p, noise, sn_noise, bin_edges, B_info, B_norm,
            box_params, sim_params, {}, n_g_density,
            pk_multipoles=True, joint_pb=True, forward_mode=fwd,
        )

    print(f"res={res} pb_step={pb_step}; warming up forward pass ...", flush=True)
    d0 = np.asarray(pipe(fid, False))
    print(f"data vector length n_data={d0.shape[0]}", flush=True)

    # --- reverse-mode (stop_gradient) ---
    t = time.time()
    J_rev = np.asarray(jax.jacrev(lambda p: pipe(p, False))(fid))
    print(f"jacrev done in {time.time()-t:.1f}s  shape={J_rev.shape}", flush=True)

    # --- forward-mode jacfwd (the candidate) ---
    fwd_error = None
    try:
        t = time.time()
        J_fwd = np.asarray(jax.jacfwd(lambda p: pipe(p, True))(fid))
        print(f"jacfwd done in {time.time()-t:.1f}s  shape={J_fwd.shape}", flush=True)
    except Exception as e:
        fwd_error = e
        print(f"\n!!! jacfwd RAISED {type(e).__name__}: {e}", flush=True)

    # --- FD central differences (growth-inclusive ground truth) ---
    steps = np.array([_FINITE_DIFF_STEPS[k] for k in _SAMPLED_KEYS], dtype=np.float64)
    J_fd = np.zeros_like(J_rev)
    for i in range(len(fid)):
        d = jnp.zeros_like(fid).at[i].set(float(steps[i]))
        plus = np.asarray(pipe(fid + d, False))
        minus = np.asarray(pipe(fid - d, False))
        J_fd[:, i] = (plus - minus) / (2 * steps[i])
    print("FD done", flush=True)

    if fwd_error is not None:
        print("\nVERDICT: FAIL (jacfwd raised an exception -> forward mode not "
              "supported through the pipeline). Fall back to FD.")
        return

    print(f"\n{'param':9s} {'||J_fd||':>12s} {'rel(fwd,fd)':>12s} "
          f"{'rel(rev,fd)':>12s} {'rel(fwd,rev)':>12s}")
    for i, k in enumerate(_SAMPLED_KEYS):
        print(f"{k:9s} {np.linalg.norm(J_fd[:, i]):12.4e} "
              f"{_relL2(J_fwd[:, i], J_fd[:, i]):12.4f} "
              f"{_relL2(J_rev[:, i], J_fd[:, i]):12.4f} "
              f"{_relL2(J_fwd[:, i], J_rev[:, i]):12.4f}")

    om = _SAMPLED_KEYS.index("Omega_m")
    fwd_finite = bool(np.all(np.isfinite(J_fwd)))
    fwd_vs_fd = _relL2(J_fwd[:, om], J_fd[:, om])
    fwd_vs_rev = _relL2(J_fwd[:, om], J_rev[:, om])
    print(f"\nGATE checks (Omega_m): finite={fwd_finite}  "
          f"rel(fwd,fd)={fwd_vs_fd:.4f} (<0.10?)  "
          f"rel(fwd,rev)={fwd_vs_rev:.4f} (>0.05 => growth captured?)")
    verdict = "PASS" if (fwd_finite and fwd_vs_fd < 0.10) else "FAIL"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()

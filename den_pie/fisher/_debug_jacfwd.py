#!/usr/bin/env python
"""Bisect where forward-mode (jacfwd) first produces NaN in the 5-param pipeline.

Builds the pipeline stage by stage with requires_jacfwd=True and cosmo NOT
stop_gradient'd, and jacfwd's a scalar reduction of each stage's output. Prints
the per-stage Jacobian (w.r.t. the 5 cosmo params) and whether it is finite, so
we can localise the first NaN.

Run:  python core/_debug_jacfwd.py
"""
import numpy as np


def main():
    import jax
    import jax.numpy as jnp
    from discodj import DiscoDJ
    from discodj.core.grids import get_fourier_grid
    from den_pie.fisher.fisher import _cosmo_dict
    from fisher_5param import _FIDUCIAL_5PARAM, _SAMPLED_KEYS
    from den_pie.forward.model import BOX_PARAMS, SIM_PARAMS

    res = 16
    boxsize = 1000.0
    dim = 3
    box = dict(BOX_PARAMS); box["res"] = res; box["boxsize"] = boxsize
    sp = dict(SIM_PARAMS); sp["res_pm"] = 2 * res
    fid = jnp.array([_FIDUCIAL_5PARAM[k] for k in _SAMPLED_KEYS], dtype=jnp.float32)
    noise = jax.random.normal(jax.random.PRNGKey(0), shape=(res,) * dim)
    device = jax.devices()[0]

    def _dj1(p):
        cosmo = _cosmo_dict(p, fixed={})
        return DiscoDJ(dim=dim, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                       requires_jacfwd=True).with_timetables()

    def s_growth(p):
        dj = _dj1(p)
        return jnp.asarray(dj.cosmo.Dplus(sp["a_end"])).reshape(())

    def s_plin(p):
        dj = _dj1(p).with_linear_ps()
        return jnp.sum(dj._pk_table["Pk"])

    def _delta_ic(p):
        dj1 = _dj1(p).with_linear_ps()
        pk = dj1._pk_table["Pk"]; k = dj1._pk_table["k"]
        kgrid = get_fourier_grid([res] * dim, boxsize, dtype_num=32,
                                 with_jax=True, full=False)["|k|"]
        Pk = jnp.interp(kgrid, k, pk)
        nf = (res / boxsize) ** dim
        return jnp.fft.irfftn(jnp.fft.rfftn(noise)
                              * jnp.sqrt(jnp.maximum(Pk * nf, 1e-30)))

    def s_ic(p):
        return jnp.sum(_delta_ic(p) ** 2)

    def _xsim(p):
        cosmo = _cosmo_dict(p, fixed={})
        delta_ic = _delta_ic(p)
        dj2 = DiscoDJ(dim=dim, res=res, boxsize=boxsize, device=device,
                      cosmo=cosmo, requires_jacfwd=True).with_timetables()
        dj2 = dj2.with_external_ics(delta=delta_ic)
        dj2 = dj2.with_lpt(n_order=sp["nlpt_order_ics"], grad_kernel_order=0)
        _skip = {"nlpt_order_ics", "worder"}
        run_params = {kk: v for kk, v in sp.items() if kk not in _skip}
        X_sim, P_mom, _ = dj2.run_nbody(**run_params, use_diffrax=False)
        return X_sim, P_mom, dj2

    def s_nbody(p):
        X_sim, P_mom, _ = _xsim(p)
        return jnp.sum(X_sim ** 2) + jnp.sum(P_mom ** 2)

    def s_field(p):
        X_sim, P_mom, dj2 = _xsim(p)
        P_flat = P_mom.reshape(-1, 3)
        n_field = dj2.compute_field_quantity_from_particles(
            pos=X_sim, quantity=jnp.ones((res ** dim,), dtype=X_sim.dtype),
            normalize_by_density=False, in_redshift_space=True, vel=P_flat,
            a=sp["a_end"], radial_dim=2, worder=sp["worder"], antialias=1,
            deconvolve=True)
        delta = n_field / n_field.mean() - 1.0
        return jnp.sum(delta ** 2)

    stages = [("Dplus(growth)", s_growth), ("P_lin", s_plin),
              ("delta_ic", s_ic), ("nbody X,P", s_nbody), ("final field", s_field)]
    for name, fn in stages:
        try:
            g = np.asarray(jax.jacfwd(fn)(fid))
            fin = bool(np.all(np.isfinite(g)))
            print(f"{name:16s} jacfwd={np.array2string(g, precision=3)}  finite={fin}",
                  flush=True)
            if not fin:
                print(f"  --> FIRST NaN at stage '{name}'. Stopping.", flush=True)
                break
        except Exception as e:
            print(f"{name:16s} ERROR {type(e).__name__}: {e}", flush=True)
            break


if __name__ == "__main__":
    main()

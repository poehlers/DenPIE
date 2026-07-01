#!/usr/bin/env python
"""Test whether enabling float64 makes d(Dplus)/d(Omega_m) finite under jacfwd.

The growth ODE in cosmology.py has an explicit x64 branch (line ~341). The
jacfwd NaN was seen in float32; a NaN only in the *tangent* of an ODE with
a**-3 / exp(y) terms is the signature of float32 overflow. Test x64.

Run:  python core/_debug_growth_x64.py
"""
import jax
jax.config.update("jax_enable_x64", True)  # MUST be before any jnp work
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def main():
    from discodj import DiscoDJ
    from den_pie.fisher.fisher import _cosmo_dict
    from fisher_5param import _FIDUCIAL_5PARAM, _SAMPLED_KEYS
    from den_pie.forward.model import SIM_PARAMS

    res = 16
    boxsize = 1000.0
    sp = dict(SIM_PARAMS)
    fid = jnp.array([_FIDUCIAL_5PARAM[k] for k in _SAMPLED_KEYS], dtype=jnp.float64)

    def s_growth(p):
        cosmo = _cosmo_dict(p, fixed={})
        dj = DiscoDJ(dim=3, res=res, boxsize=boxsize, device="cpu", cosmo=cosmo,
                     requires_jacfwd=True).with_timetables()
        return jnp.asarray(dj.cosmo.Dplus(sp["a_end"])).reshape(())

    val = float(s_growth(fid))
    g = np.asarray(jax.jacfwd(s_growth)(fid))
    print(f"x64 enabled. Dplus(a_end)={val:.6f}")
    print(f"d(Dplus)/d[Om,Ob,h,ns,s8] = {np.array2string(g, precision=5)}")
    print(f"finite = {bool(np.all(np.isfinite(g)))}")


if __name__ == "__main__":
    main()

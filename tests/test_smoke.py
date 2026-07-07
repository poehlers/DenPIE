"""End-to-end smoke test for the merged den_pie (forward model + Fisher + SBI).

Fast on CPU. Each test skips cleanly if its stack is absent, so the suite is
informative in any of the three relevant environments:

  * the unified env  -> everything runs (this is the gating check),
  * flivenv          -> JAX forward model + Fisher + the overlay,
  * 21cmvenv         -> PyTorch SBI side.

Run:
    pytest tests/test_smoke.py -v
The tiny end-to-end Fisher forecast (~2 min on CPU) is opt-in:
    DENPIE_RUNSLOW=1 pytest tests/test_smoke.py -v
"""
import os

import numpy as np
import pytest


def _runslow():
    return os.environ.get("DENPIE_RUNSLOW", "") not in ("", "0", "false", "False")


# --------------------------- parameter-name shim ---------------------------

def test_params_shim():
    from den_pie.util.params import (
        to_den_pie, to_fisher, names_to_den_pie, align_to,
    )
    assert to_den_pie("sigma8") == "sigma_8"
    assert to_fisher("sigma_8") == "sigma8"
    assert to_den_pie("Omega_m") == "Omega_m"          # already agree
    # Fisher order vs den_pie order: same params, different sigma spelling.
    assert align_to(["Omega_m", "sigma8"], ["Omega_m", "sigma_8"]) == [0, 1]
    assert align_to(["sigma8", "Omega_m"], ["Omega_m", "sigma_8"]) == [1, 0]
    assert names_to_den_pie(["Omega_m", "sigma8"]) == ["Omega_m", "sigma_8"]


# ----------------------------- import coverage -----------------------------

@pytest.mark.parametrize("mod", ["den_pie.util.params", "den_pie.fisher.compare"])
def test_import_light(mod):
    """numpy-only modules must import in every environment."""
    __import__(mod)


def test_import_torch_side():
    pytest.importorskip("torch")
    pytest.importorskip("sbi")
    import den_pie.spectra.train   # noqa: F401
    import den_pie.density.train   # noqa: F401


def test_import_jax_side():
    pytest.importorskip("jax")
    pytest.importorskip("discodj")
    import den_pie.forward          # noqa: F401  (re-exports trigger jax/discodj/falcon/BFast)
    import den_pie.fisher.fisher    # noqa: F401  (3845-LOC lib; pulls den_pie.forward.*)


def test_unified_imports_together():
    """Headline check: BOTH stacks import in a single process (unified env)."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    pytest.importorskip("sbi")
    pytest.importorskip("discodj")
    import den_pie.forward            # noqa: F401
    import den_pie.fisher.fisher      # noqa: F401
    import den_pie.spectra.train      # noqa: F401
    assert torch.__version__ and jax.__version__


# -------------------- SBI <-> Fisher overlay (fast, no JAX) -----------------

def test_sbi_vs_fisher_overlay(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    pytest.importorskip("torch")          # compare reuses DENSITY_PARAMS for labels
    from den_pie.fisher.compare import compare_corner

    names = ["Omega_m", "sigma8"]         # Fisher spelling (sigma8)
    fid = np.array([0.3175, 0.834])
    cov = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])
    fdir = tmp_path / "fisher_run"
    fdir.mkdir()
    np.savez(fdir / "fisher_results.npz", fisher_cov=cov, cosmo_fid=fid,
             param_names=np.array(names))

    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(fid, cov, size=4000).astype("float32")
    spath = tmp_path / "sbi.npz"
    np.savez(spath, samples=samples, label=fid)

    out = tmp_path / "corner.png"
    compare_corner(str(fdir), str(spath), str(out))
    assert out.exists() and out.stat().st_size > 0


# ----------------------- tiny SBI flow + one step ---------------------------

def test_sbi_flow_one_step():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sbi")
    import torch.nn as nn
    from den_pie.spectra.flow import build_sbi_density_estimator

    params = {"spectra": {"flow": {"type": "maf", "hidden_features": 16,
                                    "num_transforms": 2}}}
    n_feat, n_param = 6, 2
    x = torch.randn(32, n_feat)     # summary features (condition)
    y = torch.randn(32, n_param)    # parameters (theta)
    de = build_sbi_density_estimator(params, nn.Identity(), x, y)
    loss = de.loss(y, condition=x).mean()   # same call as den_pie training loops
    loss.backward()
    assert torch.isfinite(loss)


# ----------------- tiny end-to-end Fisher forecast (slow) -------------------

@pytest.mark.skipif(not _runslow(), reason="slow; set DENPIE_RUNSLOW=1 to run")
def test_fisher_forecast_tiny(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("discodj")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import glob
    from den_pie.fisher.fisher import run_fisher_forecast, BOX_PARAMS, SIM_PARAMS

    box = dict(BOX_PARAMS); box["res"] = 16
    sim = dict(SIM_PARAMS); sim["res_pm"] = 32
    out = tmp_path / "fisher"
    run_fisher_forecast(n_seeds=1, n_bins=6, sbi_samples_path=None,
                        output_dir=str(out), box_params=box, sim_params=sim,
                        field="fin")
    hits = glob.glob(str(out) + "*/fisher_results.npz")
    assert hits, "no fisher_results.npz produced"
    with np.load(hits[0], allow_pickle=True) as d:
        cov = np.asarray(d["fisher_cov"])
    assert cov.shape == (5, 5) and np.isfinite(cov).all()

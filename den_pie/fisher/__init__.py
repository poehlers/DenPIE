"""den_pie.fisher — Fisher-matrix forecasting for the FLI forward model.

Ported from joint_fli_sbi (https://github.com/oleg-savchenko/joint_fli_sbi);
see the top-level NOTICE for attribution. The Fisher pipeline differentiates the
JAX forward model in :mod:`den_pie.forward` w.r.t. cosmological (+ bias)
parameters, builds the Fisher information matrix, and produces parameter
forecasts that can be overlaid against den_pie's SBI posteriors.

Entry points
------------
``run_fisher.main`` — generic Fisher CLI (exposed as ``den_pie fisher-forecast``)
``fisher_2param``   — standalone (Omega_m, sigma8) driver
``fisher_5param``   — standalone 5-cosmo-param driver

The heavy submodules import JAX / DiscoDJ / BFast at module load, so nothing is
imported here at package-import time; import the submodule you need explicitly.
"""

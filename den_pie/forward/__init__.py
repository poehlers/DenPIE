"""
den_pie.forward — forward-model simulators (ported from joint_fli_sbi/core/model).

Sub-modules
-----------
model         — variable-cosmology pipeline (PowerSpectrum, InitialConditions, ForwardModel)
                  supports ``fixed_cosmo_params`` to pin any subset of parameters
model_minimal — fixed-cosmology pipeline    (InitialConditions, ForwardModel)
model_bias    — biased tracer pipeline      (ForwardModelBias)
bias          — pure JAX Lagrangian bias operators (lagrangian_bias_weights, ...)
summaries     — BFast-backed summary-statistic nodes (PkMultipoles, BkMonopole)

Re-exports from ``model`` are provided so that the Falcon ``_target_`` paths
``den_pie.forward.PowerSpectrum``, ``den_pie.forward.InitialConditions``, and
``den_pie.forward.ForwardModel`` resolve, and so that
``from den_pie.forward import SIM_PARAMS, ...`` (used in model_bias) works.
"""

from .model import (
    BOX_PARAMS,
    SIM_PARAMS,
    _FIDUCIAL_COSMO,
    _cosmo_dict,
    PowerSpectrum,
    InitialConditions,
    ForwardModel,
    ForwardModelPk,
)
from .model_bias import ForwardModelBiasPk
from .summaries import PkMultipoles, BkMonopole

__all__ = [
    "BOX_PARAMS",
    "SIM_PARAMS",
    "_FIDUCIAL_COSMO",
    "_cosmo_dict",
    "PowerSpectrum",
    "InitialConditions",
    "ForwardModel",
    "ForwardModelPk",
    "ForwardModelBiasPk",
    "PkMultipoles",
    "BkMonopole",
]

"""Parameter-name reconciliation between the Fisher and SBI halves of den_pie.

The JAX Fisher pipeline (:mod:`den_pie.fisher`, ported from joint_fli_sbi) names
the amplitude parameter ``sigma8``, while the PyTorch SBI pipeline
(:mod:`den_pie.spectra`, :mod:`den_pie.density`) and its dataset registry name it
``sigma_8``. All other cosmological names (``Omega_m``, ``Omega_b``, ``h``,
``n_s``) and bias names (``b1``, ``b2``, ``bs2``, ``bn2``) already agree.

This module is a small, dependency-free shim used at the SBI<->Fisher boundary
(e.g. the ``fisher-compare`` overlay) so parameter vectors line up by name
regardless of which half produced them. It deliberately renames nothing inside
either pipeline; extend ``_TO_DEN_PIE`` if more aliases ever appear.
"""
from __future__ import annotations

# Canonical den_pie (SBI-side) spelling for each known Fisher-side alias.
_TO_DEN_PIE = {
    "sigma8": "sigma_8",
}
# Reverse map: canonical Fisher-side spelling for each den_pie name.
_TO_FISHER = {v: k for k, v in _TO_DEN_PIE.items()}


def to_den_pie(name: str) -> str:
    """Map a single Fisher-side parameter name to its den_pie (SBI) spelling."""
    return _TO_DEN_PIE.get(name, name)


def to_fisher(name: str) -> str:
    """Map a single den_pie (SBI) parameter name to its Fisher spelling."""
    return _TO_FISHER.get(name, name)


def names_to_den_pie(names):
    """Map an iterable of Fisher-side names to den_pie spellings (order kept)."""
    return [to_den_pie(n) for n in names]


def names_to_fisher(names):
    """Map an iterable of den_pie names to Fisher spellings (order kept)."""
    return [to_fisher(n) for n in names]


def align_to(names, reference):
    """Indices that reorder ``names`` to match ``reference`` by parameter.

    Names are compared after normalising to den_pie spelling, so ``sigma8`` and
    ``sigma_8`` are treated as the same parameter. Raises ``KeyError`` if a
    reference name is absent from ``names``.

    Example: to overlay a Fisher covariance (parameter order A) onto an SBI
    posterior (parameter order B), use
    ``perm = align_to(fisher_names, sbi_names)`` and index rows/cols of the
    covariance with ``perm``.
    """
    canon = [to_den_pie(n) for n in names]
    lookup = {n: i for i, n in enumerate(canon)}
    return [lookup[to_den_pie(r)] for r in reference]

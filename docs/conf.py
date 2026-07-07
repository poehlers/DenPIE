"""Sphinx configuration for the den_pie documentation site.

Builds a GitHub Pages site presenting the full field-level pipeline: the JAX
forward model, Fisher forecasting, and PyTorch simulation-based inference.

Heavy scientific dependencies (torch/jax/discodj/BFast/Pylians/classy/sbi/...)
are mocked *only when not importable*, so:
  * a local build inside the unified env documents the real objects, and
  * a lightweight CI build (numpy/scipy/matplotlib + sphinx only) still succeeds.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # repo root, so `import den_pie` works

# -- Project ----------------------------------------------------------------
project = "den_pie"
author = "Pieter Oehlers"
copyright = "2026, Pieter Oehlers"
release = "2.0.0"

# -- General ----------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# Mock heavy deps that aren't importable in the current build environment.
_HEAVY = [
    "torch", "torchvision", "torchaudio",
    "jax", "jaxlib", "jaxtyping",
    "discodj", "BFast", "falcon", "falcon_sbi",
    "Pylians", "Pk_library", "classy",
    "sbi", "nflows", "FrEIA", "getdist", "optuna", "tarp",
]
autodoc_mock_imports = [m for m in _HEAVY if importlib.util.find_spec(m) is None]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_figures/**"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- HTML output ------------------------------------------------------------
html_theme = "furo"
html_title = "den_pie"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/poehlers/DenPIE",
    "source_branch": "main",
    "source_directory": "docs/",
}

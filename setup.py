from setuptools import setup, find_packages

HTTPS_GITHUB_URL = "https://github.com/poehlers/DenPIE"

with open("README.md", "r") as fh:
    long_description = fh.read()

# Base install = the PyTorch SBI inference half (den_pie/density, den_pie/spectra).
# The JAX forward-model + Fisher half (den_pie/forward, den_pie/fisher) is an
# extra because it pulls a heavy, separately-built scientific stack. A single
# env can hold BOTH — see scripts/make_unified_venv.sh and requirements-unified.txt.
requirements = [
    "numpy", "scipy", "pyyaml", "matplotlib",
    "torch", "torchvision", "FrEIA", "nflows", "sbi", "getdist", "optuna",
]

# JAX forward model + Fisher. Best installed via scripts/make_unified_venv.sh
# (git pins + the CUDA-12 jax wheels); listed here for completeness.
forward_requires = [
    "jax[cuda12]==0.9.1", "jaxtyping",
    "discodj @ git+https://github.com/cosmo-sims/DISCO-DJ.git@e066802913293b590372c3e5031a694e8819cfa6",
    "BFast @ git+https://github.com/tsfloss/BFast.git@5edeb5c7d4395ca67ddab7b14d5f781c10332ec1",
    "Pylians @ git+https://github.com/franciscovillaescusa/Pylians3.git@5bfaf0006a80a2aa2f0f33f309d68b7ac3172b2d",
    "classy==3.3.4.0", "falcon-sbi==0.3.0",
]

docs_requires = ["sphinx>=7", "furo", "myst-parser", "sphinx-copybutton", "pymupdf"]

setup(
    name="den_pie",
    version="2.0.0",
    author="Pieter Oehlers",
    author_email="pieteroehlers@gmail.com",
    description="Field-level forward model, Fisher forecasts, and simulation-based "
                "inference for cosmology (SBI pipeline + the joint_fli_sbi forward "
                "model and Fisher, merged). See NOTICE for attribution.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    extras_require={
        "forward": forward_requires,
        "fisher": forward_requires,       # same JAX stack
        "docs": docs_requires,
        "all": forward_requires + docs_requires,
    },
    entry_points={"console_scripts": ["den_pie=den_pie.__main__:main"]},
)

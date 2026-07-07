#!/usr/bin/env bash
# Build a SINGLE virtual environment that runs BOTH halves of den_pie:
#   * JAX forward model + Fisher  (den_pie/forward, den_pie/fisher):
#         jax, discodj, BFast, Pylians, classy, falcon-sbi
#   * PyTorch SBI inference        (den_pie/density, den_pie/spectra):
#         torch, torchvision, sbi, nflows, FrEIA, getdist, optuna
#
# Why one env is safe: the existing flivenv and 21cmvenv are BOTH Python 3.13.1
# and pull the SAME nvidia-cu12 CUDA 12.8 wheels; flivenv already runs torch
# 2.10.0 and jax 0.9.1 together. This script reproduces that union with the
# exact pins captured from those venvs (2026-06-30).
#
# Usage:
#   bash scripts/make_unified_venv.sh [TARGET_VENV_DIR]
#
# Notes:
#   * Default target is $HOME/denpievenv. A full torch+jax+CUDA env is several
#     GB — point TARGET_VENV_DIR at a roomy filesystem and MIND YOUR HOME QUOTA.
#   * Build on a GPU node (gpu_a100) so the CUDA wheels match the runtime.
#   * discodj/BFast/Pylians build from git; classy compiles CLASS — expect a few
#     minutes of compilation and have a C/C++ toolchain + FFTW available
#     (module load 2025 OpenMPI FFTW, as in the SLURM run scripts).
set -euo pipefail

VENV="${1:-$HOME/denpievenv}"
PYBIN="${PYBIN:-python3.13}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Creating venv at: $VENV  (python: $PYBIN)"
"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Pin the GPU stacks for EVERY pip install below so transitive deps can't drift
# them (jax/jaxlib must stay 0.9.1 to match the CUDA plugins + DiscoDJ/BFast).
export PIP_CONSTRAINT="${REPO_ROOT}/constraints-unified.txt"

echo "==> [1/4] PyTorch (CUDA 12.8 build) + torchvision/torchaudio"
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

echo "==> [2/4] JAX (CUDA 12)"
pip install "jax[cuda12]==0.9.1" jaxtyping==0.3.9

echo "==> [3/4] Differentiable forward-model stack (exact git pins from flivenv)"
pip install \
    "discodj @ git+https://github.com/cosmo-sims/DISCO-DJ.git@e066802913293b590372c3e5031a694e8819cfa6" \
    "BFast @ git+https://github.com/tsfloss/BFast.git@5edeb5c7d4395ca67ddab7b14d5f781c10332ec1" \
    "Pylians @ git+https://github.com/franciscovillaescusa/Pylians3.git@5bfaf0006a80a2aa2f0f33f309d68b7ac3172b2d" \
    classy==3.3.4.0 falcon-sbi==0.3.0

echo "==> [4/4] PyTorch SBI inference stack + den_pie (editable, all extras)"
pip install sbi==0.26.1 nflows==0.14 FrEIA==0.2 getdist==1.7.6 optuna==4.8.0 tarp pyyaml
pip install -e "${REPO_ROOT}[docs]"

# JAX must not pre-grab the whole GPU when torch is also resident in-process.
# Non-fatal: some home-dir ACLs make `activate` read-only, so don't abort on it.
chmod u+w "$VENV/bin/activate" 2>/dev/null || true
echo 'export XLA_PYTHON_CLIENT_PREALLOCATE=false' >> "$VENV/bin/activate" 2>/dev/null \
    || echo "  (note: could not append to activate; set XLA_PYTHON_CLIENT_PREALLOCATE=false in your run scripts)"

echo
echo "==> Unified env ready: $VENV"
echo "    Activate with:  source $VENV/bin/activate"
echo "    Sanity check:   python -c 'import torch, jax, den_pie.forward, den_pie.fisher.fisher, den_pie.spectra.train; print(\"unified OK\")'"
echo "    Exact lock:     pip freeze > requirements-lock.txt"

#!/usr/bin/env bash
#SBATCH --job-name=denpie-venv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --output=/home/poehlers/cosmo_thesis/den_pie/output/logs/%x-%j-%N_slurm.out
#SBATCH --error=/home/poehlers/cosmo_thesis/den_pie/output/logs/R-%x.%j.err
#
# Build the UNIFIED den_pie environment (PyTorch SBI + JAX forward model/Fisher)
# and smoke-test it, on a GPU node with the compile toolchain loaded.
#
#   sbatch scripts/build_unified_env.sh [VENV_DIR]     # default: $HOME/denpievenv
# or, on a node you already hold interactively:
#   bash   scripts/build_unified_env.sh [VENV_DIR]
#
# NOTE: several GB — pass a roomy VENV_DIR (e.g. /projects/prjs1926/denpievenv)
# if $HOME is tight. First build compiles Pylians/CLASS/DiscoDJ/BFast (~10-30 min).
set -euo pipefail

REPO=/home/poehlers/cosmo_thesis/den_pie
VENV="${1:-$HOME/denpievenv}"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load OpenMPI/5.0.7-GCC-14.2.0
module load FFTW/3.3.10-GCC-14.2.0

mkdir -p "$REPO/output/logs"
cd "$REPO"

# 1. Build the venv: torch + jax + discodj/BFast/Pylians/classy/falcon + sbi + den_pie.
export PYBIN=python3
bash scripts/make_unified_venv.sh "$VENV"

# 2. Activate and verify both stacks import in one process.
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import torch, jax, den_pie.forward, den_pie.fisher.fisher, den_pie.spectra.train; print('unified OK')"

# 3. Full smoke test (incl. a tiny end-to-end Fisher forecast). pytest is a
#    test-only dep, not part of the runtime env, so install it here.
pip install -q pytest
DENPIE_RUNSLOW=1 bash scripts/smoke_test.sh

echo
echo "=========================================================="
echo " Unified env ready:  $VENV"
echo " Use it in run scripts with:  source $VENV/bin/activate"
echo "=========================================================="

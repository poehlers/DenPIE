# Installation

den_pie spans two GPU stacks:

| Half | Package(s) | Key dependencies |
|------|------------|------------------|
| Forward model + Fisher | {mod}`den_pie.forward`, {mod}`den_pie.fisher` | JAX (CUDA 12), DiscoDJ, BFast, Pylians, CLASS, Falcon |
| Neural SBI | {mod}`den_pie.density`, {mod}`den_pie.spectra` | PyTorch (CUDA 12), `sbi`, `nflows`, `FrEIA`, `getdist` |

Both are Python 3.13 and share the **same** NVIDIA CUDA 12.8 wheels, so a
**single virtual environment holds both** — PyTorch and JAX run side by side in
one process.

## Unified environment (recommended)

```bash
# Build on a GPU node with a C/C++ toolchain + FFTW available, e.g.
#   module load 2025 OpenMPI FFTW
bash scripts/make_unified_venv.sh /path/to/denpievenv
source /path/to/denpievenv/bin/activate
```

The script installs, in order: PyTorch (CUDA 12.8 index), JAX (CUDA 12), the
differentiable forward-model stack (DiscoDJ/BFast/Pylians pinned by git commit,
CLASS, Falcon), the SBI stack, and finally den_pie itself (`pip install -e .`).
The exact top-level pins also live in `requirements-unified.txt`.

:::{note}
DiscoDJ, BFast, and Pylians build from source; CLASS compiles too. Build on a
node that has FFTW and a compiler (the SLURM run scripts `module load 2025
OpenMPI FFTW`). A full torch + jax + CUDA environment is several GB — point the
target at a filesystem with room and mind home quotas.
:::

Sanity check:

```bash
python -c "import torch, jax, den_pie.forward, den_pie.fisher.fisher, den_pie.spectra.train; print('unified OK')"
bash scripts/smoke_test.sh                     # fast checks
DENPIE_RUNSLOW=1 bash scripts/smoke_test.sh    # + a tiny end-to-end Fisher forecast (~2 min CPU)
```

## Partial installs

If you only need one half:

```bash
pip install -e .            # SBI stack only (torch/sbi/nflows/FrEIA/getdist/optuna)
pip install -e ".[forward]" # add the JAX forward-model + Fisher stack
pip install -e ".[docs]"    # docs toolchain (sphinx, furo, myst-parser, pymupdf)
```

Because `den_pie/__init__.py` is empty and every submodule imports its heavy
dependencies lazily, `import den_pie.spectra` never pulls JAX and
`import den_pie.forward` never pulls PyTorch — so a half-install is usable.

## GPU memory coexistence

When both stacks are live in one process, keep JAX from pre-grabbing the whole
GPU (which would starve PyTorch):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

`make_unified_venv.sh` appends this to the venv's `activate` script.

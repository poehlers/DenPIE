#!/usr/bin/env bash
# Smoke-test the merged den_pie (forward model + Fisher + SBI).
#
# Run inside the UNIFIED env (scripts/make_unified_venv.sh) for full coverage;
# in flivenv or 21cmvenv the tests for the absent stack skip cleanly.
#
#   bash scripts/smoke_test.sh                    # fast checks only
#   DENPIE_RUNSLOW=1 bash scripts/smoke_test.sh   # + tiny end-to-end Fisher (~2 min CPU)
#
# Extra args are forwarded to pytest, e.g.:
#   bash scripts/smoke_test.sh -k overlay
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep JAX on CPU and don't let XLA pre-grab the GPU when torch is also resident.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec python -m pytest tests/test_smoke.py -v "$@"

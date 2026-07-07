#!/usr/bin/env python3
"""
Forward-model sample plotting script.

Loads one sample produced by `falcon sample prior`, computes and saves:
  1. Field slices       — IC / final  (xy, xz, yz planes)
  2. Power spectra      — 2×2 grid: P(k) panels (top) + Pylians/DiscoDJ ratio panels (bottom)
  3. Cross-correlation  — C(k) = P_cross / sqrt(P_IC * P_fin) between IC and final fields
  4. 1-pt PDF           — histogram + skewness + excess kurtosis for IC and final
  5. Bispectrum         — reduced Q(theta) for IC and final at two triangle configs

Usage
-----
    python core/plot_samples.py \
        --run-dir outputs/forward_run_20260225_194536 \
        --index 0

All plots are written to <run-dir>/simulation_plots/
"""

import argparse
import logging
import os
import sys

# Must be set before JAX is imported (triggered transitively via discodj).
# Suppresses CUDA plugin initialisation errors on CPU-only nodes.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# Silence the JAX CUDA-plugin initialisation error logged at ERROR level when
# running on a CPU-only node.  The plugin still tries initialize() before
# JAX_PLATFORMS=cpu takes effect; this suppresses the resulting log noise.
logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)

import numpy as np
import matplotlib as mpl

# Allow running as `python core/plot_samples.py` from the project root
sys.path.insert(0, os.path.dirname(__file__))
from den_pie.forward.utils import (
    load_sample, _cosmo_dict, recover_k_grid,
    plot_slices, plot_power_spectra, plot_cross_correlation,
    plot_1pt_pdf, plot_bispectrum,
)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults — must match the config file (see config_files/)
# ─────────────────────────────────────────────────────────────────────────────
_BOX_SIZE = 1000.0   # Mpc/h
_RES      = 64


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Plot diagnostics for one sample from a falcon sample prior run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--run-dir', required=True,
                   help='Run directory produced by falcon sample prior')
    p.add_argument('--index', type=int, default=0,
                   help='Sample index (0 → 000000.npz, 1 → 000001.npz, …)')
    p.add_argument('--mas', type=str, default=None,
                   help='Mass-assignment scheme for Pylians (e.g. PCS). None = no correction')
    p.add_argument('--out-dir', type=str, default=None,
                   help='Output directory for plots (default: <run-dir>/simulation_plots)')
    p.add_argument('--box-size', type=float, default=_BOX_SIZE,
                   help='Box size in Mpc/h')
    p.add_argument('--res', type=int, default=_RES,
                   help='Grid resolution')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    mpl.rcdefaults()
    mpl.rcParams['figure.facecolor'] = 'white'

    out_dir = args.out_dir if args.out_dir else os.path.join(args.run_dir, 'simulation_plots')
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(args.run_dir.rstrip("/"))
    tag = f'{basename}_sample_{args.index:06d}'

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f'Loading sample {args.index} from {args.run_dir} ...')
    data         = load_sample(args.run_dir, args.index)
    cosmo_params = data['cosmo_params']   # (5,)
    pk_stored    = data['pk']             # (N_k,)  linear P(k) used for ICs
    delta_ic     = data['delta_ic']       # (res, res, res)
    delta_fin    = data['delta_fin']      # (res, res, res)

    Om, Ob, h, ns, s8 = cosmo_params
    print(f'  Ωm={Om:.4f}  Ωb={Ob:.4f}  h={h:.4f}  ns={ns:.4f}  σ8={s8:.4f}')
    print(f'  delta_ic  : mean={delta_ic.mean():.4f}  std={delta_ic.std():.4f}')
    print(f'  delta_fin : mean={delta_fin.mean():.4f}  std={delta_fin.std():.4f}')

    cd = _cosmo_dict(cosmo_params)

    # ── Build parameter subtitle (only include keys present in the file) ───────
    _parts = []
    if 'cosmo_params' in data:
        Om, Ob, h, ns, s8 = data['cosmo_params']
        _parts.append(f'Ωm={Om:.3f}  Ωb={Ob:.3f}  h={h:.3f}  ns={ns:.3f}  σ8={s8:.3f}')
    if 'bias_params' in data:
        b1, b2, bs2, bn2 = data['bias_params']
        _parts.append(f'b₁={b1:.3f}  b₂={b2:.3f}  bs²={bs2:.3f}  bn²={bn2:.3f}')
    param_str = '     '.join(_parts)

    # ── Recover k grid (needed to pair with pk_stored) ────────────────────────
    print('Recovering DiscoDJ k grid ...')
    k_stored = recover_k_grid(cd, args.res, args.box_size)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('1/5  Field slices ...')
    plot_slices(delta_ic, delta_fin, args.box_size, out_dir, tag, subtitle=param_str)

    print('2/5  Power spectra + ratios ...')
    b1 = float(data['bias_params'][0]) if 'bias_params' in data else None
    plot_power_spectra(delta_ic, delta_fin, pk_stored, k_stored, cd,
                       args.box_size, args.res, out_dir, tag, args.mas,
                       subtitle=param_str, b1=b1)

    print('3/5  Cross-correlation C(k) ...')
    plot_cross_correlation(delta_ic, delta_fin, args.box_size, args.res, out_dir, tag, args.mas,
                           subtitle=param_str)

    print('4/5  1-point PDFs ...')
    plot_1pt_pdf(delta_ic, delta_fin, out_dir, tag, subtitle=param_str)

    print('5/5  Bispectra (may take ~30 s) ...')
    plot_bispectrum(delta_ic, delta_fin, args.box_size, out_dir, tag, args.mas, subtitle=param_str)

    print(f'\nDone. All plots saved to {out_dir}/')


if __name__ == '__main__':
    main()

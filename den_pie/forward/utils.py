"""
Shared utilities for forward-model sample analysis.

Provides data loading, physics helpers (DiscoDJ k-grid recovery, CLASS P(k),
Pylians P(k) and bispectrum), and the four diagnostic plot routines used by
plot_samples.py.

Contents
--------
Data loading
    load_sample       — load a .npz sample from a falcon run directory
    _cosmo_dict       — convert cosmo_params array → DiscoDJ/CLASS dict

Physics helpers
    recover_k_grid    — recover the k array DiscoDJ used for the ICs
    pk_class          — linear P(k) from CLASS at a given redshift
    pylians_pk        — 3-D power spectrum via Pylians
    pylians_bk        — reduced bispectrum Q(theta) via Pylians

Plot routines
    plot_slices           — Fig 1: field slices (IC / final)
    plot_power_spectra    — Fig 2: 2×2 grid — P(k) panels (top) + ratio panels (bottom)
    plot_cross_correlation— Fig 3: IC–final cross-correlation coefficient C(k)
    plot_1pt_pdf          — Fig 4: 1-point PDF with moments
    plot_bispectrum       — Fig 5: reduced Q(theta) at two triangle configs
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
from classy import Class
from discodj import DiscoDJ
from scipy.stats import skew, kurtosis

# ─────────────────────────────────────────────────────────────────────────────
# Bispectrum triangle configurations (equilateral & squeezed)
# ─────────────────────────────────────────────────────────────────────────────
_BK_CONFIGS = [
    {'k1': 0.10, 'k2': 0.10, 'label': r'$k_1 = k_2 = 0.1\;h/\mathrm{Mpc}$ (equilateral)'},
    {'k1': 0.05, 'k2': 0.10, 'label': r'$k_1 = 0.05,\;k_2 = 0.1\;h/\mathrm{Mpc}$ (squeezed)'},
]
_BK_THETA = np.linspace(0, np.pi, 25)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sample(run_dir, index):
    path = os.path.join(run_dir, 'samples_dir', 'prior', f'{index:06d}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Sample not found: {path}')
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _cosmo_dict(cosmo_params):
    Om, Ob, h, ns, s8 = cosmo_params.tolist()
    return {'h': h, 'Omega_b': Ob, 'Omega_c': Om - Ob, 'n_s': ns, 'sigma8': s8}


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def recover_k_grid(cosmo_dict, res, box_size):
    """Return the k array DiscoDJ used to evaluate P(k)."""
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    dj = DiscoDJ(dim=3, res=res, boxsize=box_size, device='cpu', cosmo=cosmo_dict)
    dj = dj.with_timetables().with_linear_ps()
    return np.asarray(dj._pk_table['k'])


def pk_class(cosmo_dict, k_arr, z=0.0):
    """Linear P(k) from CLASS at redshift z, in (Mpc/h)^3.

    cosmo_dict uses DiscoDJ keys (Omega_c, Omega_b, h, n_s, sigma8).
    CLASS uses Omega_cdm instead of Omega_c.
    YHe is fixed to the BBN-concordance value to avoid interpolation errors
    at the edges of the prior (high Omega_b * h^2).
    """
    h = cosmo_dict['h']
    # Translate DiscoDJ key names → CLASS key names
    params = {k: v for k, v in cosmo_dict.items() if k != 'Omega_c'}
    if 'Omega_c' in cosmo_dict:
        params['Omega_cdm'] = cosmo_dict['Omega_c']
    params.update({'output': 'mPk',
                   'P_k_max_h/Mpc': float(k_arr.max()) * 1.1,
                   'z_max_pk': float(z),
                   'YHe': 0.2454})
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    pk = h**3 * np.array([cosmo.pk_lin(h * ki, z) for ki in k_arr])
    cosmo.struct_cleanup()
    cosmo.empty()
    return pk


def pylians_pk(field, box_size, mas, threads=4):
    field_c = np.ascontiguousarray(field.astype(np.float32))
    Pk = PKL.Pk(field_c, box_size, axis=0, MAS=mas, threads=threads, verbose=False)
    return Pk.k3D, Pk.Pk[:, 0]


def pylians_cross_pk(field1, field2, box_size, mas):
    """Cross-correlation coefficient C(k) = P_cross / sqrt(P_11 * P_22) via Pylians.

    Uses PKL.XPk: Pk[:,0,0] auto(f1), Pk[:,0,1] auto(f2), XPk[:,0,0] cross(f1,f2).
    """
    f1 = np.ascontiguousarray(field1.astype(np.float32))
    f2 = np.ascontiguousarray(field2.astype(np.float32))
    XPk = PKL.XPk([f1, f2], box_size, axis=0, MAS=[mas, mas], threads=1)
    P_cross = XPk.XPk[:, 0, 0]
    C = P_cross / np.sqrt(XPk.Pk[:, 0, 0] * XPk.Pk[:, 0, 1])
    return XPk.k3D, C


def pylians_bk(field, box_size, k1, k2, theta, mas, threads=4):
    field_c = np.ascontiguousarray(field.astype(np.float32))
    BBk = PKL.Bk(field_c, box_size, k1, k2, theta, mas, threads=threads)
    return BBk.Q


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: field slices
# ─────────────────────────────────────────────────────────────────────────────

def plot_slices(delta_ic, delta_fin, box_size, out_dir, tag, subtitle=''):
    mid = delta_ic.shape[0] // 2
    rows = [
        (delta_ic,  r'$\delta_\mathrm{IC}$  (linear, $z=0$)'),
        (delta_fin, r'$\delta_\mathrm{fin}$ ($z=0$)'),
    ]
    col_fns    = [lambda f, m=mid: f[m, :, :],
                  lambda f, m=mid: f[:, m, :],
                  lambda f, m=mid: f[:, :, m]]
    col_labels = ['xy-plane', 'xz-plane', 'yz-plane']

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)

    for row_idx, (field, row_label) in enumerate(rows):
        vmax = float(np.percentile(np.abs(field), 99))
        for col_idx, (sl_fn, sl_lbl) in enumerate(zip(col_fns, col_labels)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(sl_fn(field), origin='lower', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax,
                           extent=[0, box_size, 0, box_size])
            ax.set_xlabel(r'$[h^{-1}\,\mathrm{Mpc}]$', fontsize=8)
            ax.set_ylabel(r'$[h^{-1}\,\mathrm{Mpc}]$', fontsize=8)
            ax.set_title(f'{row_label}   {sl_lbl}', fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.88, pad=0.02)

    fig.suptitle(f'Field slices  —  sample {tag}' + (f'\n{subtitle}' if subtitle else ''),
                 fontsize=13)
    _save(fig, out_dir, f'1_slices_{tag}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: power spectra + ratios (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_spectra(delta_ic, delta_fin,
                       pk_stored, k_stored,
                       cosmo_dict, box_size, res,
                       out_dir, tag, mas, subtitle='', b1=None):
    k_Nq = np.pi * res / box_size

    k_ic,  pk_ic  = pylians_pk(delta_ic,  box_size, mas)
    k_fin, pk_fin = pylians_pk(delta_fin, box_size, mas)

    # When b1 is provided, divide the final-field P(k) by (1+b1)² so it can be
    # compared directly with the matter power spectrum.
    if b1 is not None:
        pk_fin = pk_fin / (1 + b1)**2
        fin_label = rf'$P_\mathrm{{fin}}(k)\,/\,(1+b_1)^2$  ($b_1={b1:.3f}$)'
    else:
        fin_label = r'$\delta_\mathrm{fin}$ (non-linear)'

    # CLASS linear theory at z=0
    k_cl  = np.logspace(np.log10(max(k_ic[0] * 0.8, 1e-3)), np.log10(k_Nq * 1.1), 300)
    pk_cl = pk_class(cosmo_dict, k_cl, z=0.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    panels = [
        (k_ic,  pk_ic,  r'$\delta_\mathrm{IC}$ (linear)',  'tab:blue'),
        (k_fin, pk_fin, fin_label,                          'tab:orange'),
    ]

    # ── Top row: P(k) ────────────────────────────────────────────────────────
    for col, (k_pyl, pk_pyl, field_label, color) in enumerate(panels):
        ax = axes[0, col]
        ax.loglog(k_pyl,    pk_pyl,    marker='.', ms=3, lw=1,    color=color,
                  label=f'Pylians  {field_label}')
        ax.loglog(k_stored, pk_stored, lw=1.5, ls='--', color='tab:green', alpha=0.85,
                  label='DiscoDJ theory (input to ICs)')
        ax.loglog(k_cl,     pk_cl,     lw=1,   ls=':',  color='k', alpha=0.65,
                  label='CLASS linear  $z=0$')
        ax.axvline(k_Nq, color='r', ls='--', lw=0.8, alpha=0.5, label=r'$k_\mathrm{Nyq}$')
        pk_pos = pk_pyl[pk_pyl > 0]
        ax.set_xlim(k_pyl[0] * 0.8, k_Nq * 1.2)
        ax.set_ylim(pk_pos.min() * 0.3, pk_pos.max() * 3)
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', fontsize=12)
        ax.set_ylabel(r'$P(k)\;[(h^{-1}\,\mathrm{Mpc})^3]$', fontsize=12)
        ax.set_title(f'Power spectrum  {field_label}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(which='both', alpha=0.15)

    # ── Bottom row: ratios Pylians / DiscoDJ theory ───────────────────────────
    for col, (k_pyl, pk_pyl, field_label, color) in enumerate(panels):
        ax = axes[1, col]
        pk_th = np.interp(k_pyl, k_stored, pk_stored)
        ratio = pk_pyl / np.where(pk_th > 0, pk_th, np.nan)
        ax.semilogx(k_pyl, ratio, marker='.', ms=3, lw=1, color=color,
                    label='Pylians / DiscoDJ theory')
        ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.7, label='Ratio = 1')
        ax.axvline(k_Nq, color='r', ls='--', lw=0.8, alpha=0.5, label=r'$k_\mathrm{Nyq}$')
        ax.set_xlim(k_pyl[0] * 0.8, k_Nq * 1.2)
        ax.set_ylim(0.0, 2.0)
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', fontsize=12)
        ax.set_ylabel(r'$P_\mathrm{Pylians}(k)\;/\;P_\mathrm{theory}(k)$', fontsize=12)
        ax.set_title(f'Ratio  {field_label}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(which='both', alpha=0.15)

    fig.suptitle(f'Power spectra  —  sample {tag}' + (f'\n{subtitle}' if subtitle else ''),
                 fontsize=13)
    _save(fig, out_dir, f'2_power_spectra_{tag}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: IC–final cross-correlation coefficient C(k)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_correlation(delta_ic, delta_fin, box_size, res, out_dir, tag, mas, subtitle=''):
    k_Nq = np.pi * res / box_size
    k, C = pylians_cross_pk(delta_ic, delta_fin, box_size, mas)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.semilogx(k, C, marker='.', ms=3, lw=1, color='tab:purple',
                label=r'$C(k) = P_\times / \sqrt{P_\mathrm{IC}\,P_\mathrm{fin}}$')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.7, label='$C = 1$')
    ax.axvline(k_Nq, color='r', ls='--', lw=0.8, alpha=0.5, label=r'$k_\mathrm{Nyq}$')
    ax.set_xlim(k[0] * 0.8, k_Nq * 1.2)
    ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', fontsize=12)
    ax.set_ylabel(r'$C(k)$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(which='both', alpha=0.15)
    fig.suptitle(r'IC–final cross-correlation  —  sample ' + tag
                 + (f'\n{subtitle}' if subtitle else ''), fontsize=13)
    _save(fig, out_dir, f'3_cross_correlation_{tag}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: 1-point PDFs
# ─────────────────────────────────────────────────────────────────────────────

def plot_1pt_pdf(delta_ic, delta_fin, out_dir, tag, subtitle=''):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for ax, (field, label, color) in zip(axes, [
        (delta_ic,  r'$\delta_\mathrm{IC}$',  'tab:blue'),
        (delta_fin, r'$\delta_\mathrm{fin}$', 'tab:orange'),
    ]):
        flat = field.ravel()
        sk   = skew(flat)
        ku   = kurtosis(flat, fisher=True)   # excess kurtosis
        lo, hi = np.percentile(flat, [0.5, 99.5])

        ax.hist(flat, bins=120, range=(lo, hi), density=True,
                color=color, alpha=0.75, label=label)
        ax.set_xlabel(r'$\delta$', fontsize=12)
        ax.set_ylabel('PDF', fontsize=12)
        ax.set_title(f'{label}\n'
                     f'mean={flat.mean():.3f},  std={flat.std():.3f}\n'
                     f'skew={sk:.3f},  excess kurtosis={ku:.3f}', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.15)

    fig.suptitle(f'1-point PDF  —  sample {tag}' + (f'\n{subtitle}' if subtitle else ''),
                 fontsize=13)
    _save(fig, out_dir, f'4_1pt_pdf_{tag}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: reduced bispectrum Q(theta)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bispectrum(delta_ic, delta_fin, box_size, out_dir, tag, mas, subtitle=''):
    theta     = _BK_THETA
    theta_deg = np.degrees(theta)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for col, (field, field_label) in enumerate([
        (delta_ic,  r'$\delta_\mathrm{IC}$  (Gaussian — expect $Q \approx 0$)'),
        (delta_fin, r'$\delta_\mathrm{fin}$ (non-linear)'),
    ]):
        for row, cfg in enumerate(_BK_CONFIGS):
            ax = axes[row, col]
            Q = pylians_bk(field, box_size, cfg['k1'], cfg['k2'], theta, mas)
            ax.plot(theta_deg, Q, marker='.', ms=3, lw=1.2)
            ax.axhline(0.0, color='k', ls='--', lw=0.6, alpha=0.5)
            ax.set_xlabel(r'$\theta$ [deg]', fontsize=11)
            ax.set_ylabel(r'$Q(\theta)$', fontsize=11)
            ax.set_title(f'{field_label}\n{cfg["label"]}', fontsize=9)
            ax.grid(alpha=0.15)

    fig.suptitle(f'Reduced bispectrum  —  sample {tag}' + (f'\n{subtitle}' if subtitle else ''),
                 fontsize=13)
    _save(fig, out_dir, f'5_bispectrum_{tag}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

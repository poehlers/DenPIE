"""Overlay a den_pie SBI posterior on a Fisher forecast corner.

Loads a Fisher run's ``fisher_results.npz`` (Gaussian forecast covariance,
fiducial values, parameter names — written by
:func:`den_pie.fisher.fisher.run_fisher_forecast` and its bias/PB siblings)
together with a den_pie SBI posterior sample set, and draws a corner plot
comparing the two: Fisher 68/95% ellipses + Gaussian marginals against the SBI
posterior's histogram marginals + smoothed 2-D contours.

The module is deliberately light — numpy + matplotlib (+ scipy) only, no JAX /
DiscoDJ — so the comparison runs wherever a den_pie SBI run produced posterior
samples. Parameter columns are matched by *name* (``sigma8`` <-> ``sigma_8`` via
:mod:`den_pie.util.params`), and LaTeX labels / prior ranges are reused from
:data:`den_pie.density.priors.DENSITY_PARAMS` so the figure matches den_pie's
own corner plots.
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np

from den_pie.util.params import to_den_pie

# Fiducial values are saved as ``cosmo_fid`` by the pure-cosmology driver and as
# ``fid_values`` by the bias / joint-PB drivers; ``fiducial_vals`` is accepted
# as a defensive fallback.
_FISHER_FIDUCIAL_KEYS = ("cosmo_fid", "fid_values", "fiducial_vals")


def load_fisher_results(path):
    """Load a Fisher forecast from ``fisher_results.npz`` (file or run dir).

    Returns ``(cov, names, fiducial)`` with ``names`` normalised to den_pie
    spelling (``sigma8`` -> ``sigma_8``).
    """
    if os.path.isdir(path):
        path = os.path.join(path, "fisher_results.npz")
    with np.load(path, allow_pickle=True) as d:
        cov = np.asarray(d["fisher_cov"], dtype=float)
        names = [to_den_pie(str(n)) for n in np.asarray(d["param_names"]).ravel()]
        fid = None
        for k in _FISHER_FIDUCIAL_KEYS:
            if k in d.files:
                fid = np.asarray(d[k], dtype=float).ravel()
                break
    if fid is None:
        raise KeyError(f"{path}: no fiducial values under any of {_FISHER_FIDUCIAL_KEYS}")
    if not (cov.shape[0] == cov.shape[1] == len(names) == len(fid)):
        raise ValueError(
            f"{path}: inconsistent shapes cov{cov.shape}, "
            f"{len(names)} names, {len(fid)} fiducial values"
        )
    return cov, names, fid


def _resolve_sbi_npz(path):
    """Resolve a den_pie SBI sample ``.npz`` from a file or run directory."""
    if path.endswith(".npz"):
        return path
    if os.path.isdir(path):
        # Prefer the noise-reduced fiducial-mean posterior (comparable to Fisher).
        preferred = [
            os.path.join(path, "plots", "fiducial_samples", "fiducial_mean.npz"),
            os.path.join(path, "fiducial_samples", "fiducial_mean.npz"),
        ]
        for p in preferred:
            if os.path.isfile(p):
                return p
        for sub in ("plots/fiducial_samples", "fiducial_samples",
                    "plots/test_samples", "test_samples", "plots"):
            hits = sorted(glob.glob(os.path.join(path, sub, "*.npz")))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"No SBI sample .npz found at {path}")


def load_sbi_samples(path, key="samples"):
    """Load den_pie SBI posterior samples (shape ``(n, N_params)``).

    ``path`` is an ``.npz`` written by den_pie (``samples`` key, optional
    ``label``) or a den_pie run directory (the fiducial-mean posterior is
    preferred, then any fiducial / test sample file).
    """
    npz = _resolve_sbi_npz(path)
    with np.load(npz, allow_pickle=True) as d:
        if key not in d.files:
            cand = [k for k in d.files if np.asarray(d[k]).ndim == 2]
            if not cand:
                raise KeyError(f"{npz}: no '{key}' array and no 2-D array present")
            key = cand[0]
        samples = np.asarray(d[key], dtype=float)
    logging.info("Loaded SBI samples %s from %s", samples.shape, npz)
    return samples


def _labels_and_bounds(names):
    """LaTeX labels + (lo, hi) prior bounds for ``names`` via DENSITY_PARAMS."""
    from den_pie.density.priors import DENSITY_PARAMS  # local: avoids torch at import
    labels, bounds = [], []
    for n in names:
        d = DENSITY_PARAMS.get(n)
        labels.append(d["latex"] if d else n)
        bounds.append(tuple(d["range"]) if d else None)
    return labels, bounds


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _add_ellipse(ax, mx, my, cov2, conf, color, alpha):
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2

    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(chi2.ppf(conf, df=2))  # 2-dof credible region
    width, height = 2 * scale * np.sqrt(np.maximum(vals, 0.0))
    r, g, b, _ = to_rgba(color)
    ax.add_patch(Ellipse((mx, my), width, height, angle=angle,
                         facecolor=(r, g, b, alpha), edgecolor=color, lw=1.2))


def _sbi_contours(ax, x, y, xr, yr, color, smooth):
    H, xe, ye = np.histogram2d(x, y, bins=48, range=[xr, yr])
    if smooth and smooth > 0:
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, smooth)
        except Exception:
            pass
    H = H.T
    if H.max() <= 0:
        return
    # Iso-density levels enclosing 95.4% then 68.2% of the probability mass.
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    levels = []
    for frac in (0.954, 0.682):
        idx = min(int(np.searchsorted(csum, frac)), len(flat) - 1)
        levels.append(flat[idx])
    levels = sorted(set(levels))
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    if levels:
        ax.contour(xc, yc, H, levels=levels, colors=color, linewidths=1.3)


def compare_corner(fisher_path, sbi_path, output_path,
                   sbi_names=None, sbi_label="SBI posterior",
                   smooth=1.0, show_prior=True):
    """Draw a Fisher-vs-SBI comparison corner and save it to ``output_path``.

    Parameters
    ----------
    fisher_path : str
        ``fisher_results.npz`` or a Fisher run directory.
    sbi_path : str
        den_pie posterior ``.npz`` or a den_pie SBI run directory.
    output_path : str
        Where to write the figure (``.png``/``.pdf``).
    sbi_names : list of str, optional
        Parameter names of the SBI sample *columns*. If given, the columns are
        reordered to match the Fisher parameter order; otherwise the columns are
        assumed to already be in Fisher order (true for den_pie registry order).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cov, names, fid = load_fisher_results(fisher_path)
    samples = load_sbi_samples(sbi_path)
    N = len(names)

    if samples.shape[1] != N:
        raise ValueError(
            f"SBI samples have {samples.shape[1]} parameter columns but the "
            f"Fisher forecast has {N} ({names})."
        )
    if sbi_names is not None:
        from den_pie.util.params import align_to
        perm = align_to([to_den_pie(n) for n in sbi_names], names)
        samples = samples[:, perm]

    labels, bounds = _labels_and_bounds(names)
    sig = np.sqrt(np.diag(cov))
    ranges = []
    for i in range(N):
        lo = min(fid[i] - 4 * sig[i], np.percentile(samples[:, i], 0.3))
        hi = max(fid[i] + 4 * sig[i], np.percentile(samples[:, i], 99.7))
        ranges.append((lo, hi))

    fcol, scol, tcol = "C0", "C3", "C1"  # fisher / sbi / truth
    fig, ax = plt.subplots(N, N, figsize=(2.5 * N, 2.5 * N))
    ax = np.atleast_2d(ax)
    for i in range(N):
        for j in range(N):
            if j > i:
                ax[i, j].set_visible(False)

    for i in range(N):  # diagonal marginals
        a = ax[i, i]
        xs = np.linspace(*ranges[i], 256)
        a.plot(xs, _gauss(xs, fid[i], sig[i]), color=fcol, lw=1.5)
        a.hist(samples[:, i], bins=48, range=ranges[i], density=True,
               histtype="step", color=scol, lw=1.5)
        a.axvline(fid[i], color=tcol, lw=1.2)
        if show_prior and bounds[i]:
            for b in bounds[i]:
                a.axvline(b, color="red", ls="--", lw=0.8, alpha=0.3)
        a.set_xlim(ranges[i])
        a.set_yticks([])

    for i in range(N):  # lower-triangle joint distributions
        for j in range(i):
            a = ax[i, j]
            sub = cov[np.ix_([j, i], [j, i])]
            for conf, alpha in [(0.954, 0.15), (0.682, 0.35)]:
                _add_ellipse(a, fid[j], fid[i], sub, conf, fcol, alpha)
            _sbi_contours(a, samples[:, j], samples[:, i], ranges[j], ranges[i],
                          scol, smooth)
            a.axvline(fid[j], color=tcol, lw=0.8, alpha=0.6)
            a.axhline(fid[i], color=tcol, lw=0.8, alpha=0.6)
            a.set_xlim(ranges[j])
            a.set_ylim(ranges[i])

    for i in range(N):
        ax[N - 1, i].set_xlabel(labels[i])
        if i > 0:
            ax[i, 0].set_ylabel(labels[i])

    handles = [
        Line2D([0], [0], color=fcol, lw=2, label="Fisher forecast"),
        Line2D([0], [0], color=scol, lw=2, label=sbi_label),
        Line2D([0], [0], color=tcol, lw=2, label="Fiducial"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.7)
    fig.suptitle("SBI posterior vs Fisher forecast", fontsize=14)
    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved SBI-vs-Fisher corner -> %s", output_path)
    return output_path

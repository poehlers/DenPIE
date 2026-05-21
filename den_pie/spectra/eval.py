"""Evaluation and diagnostic plots for the spectra pipeline.

Three diagnostics: corner plot, TARP (full + per-parameter marginal), and
SBC rank plot. All saved into the run's plots/ directory.
"""
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import norm as scipy_norm

import torch
from getdist import plots, MCSamples

from ..util.logger import separator
from ..density.priors import DENSITY_PARAMS
from .priors import build_sbi_posterior


class SpectraPlotting:
    """Posterior corner + TARP + SBC diagnostics for the spectra pipeline."""

    def __init__(self, params: dict, density_estimator, data: dict,
                 device: torch.device):
        self.params = params
        self.data_cfg = params['spectra']['data']
        self.eval_cfg = params['spectra'].get('eval', {})
        self.plot_cfg = params['spectra'].get('plot', {})
        self.data = data
        self.device = device

        self.output_dir = self.plot_cfg.get('plot_dir', 'plots/')
        if not self.output_dir.endswith('/'):
            self.output_dir += '/'
        os.makedirs(self.output_dir, exist_ok=True)

        self.active_params = self.data_cfg['active_params']
        self.n_params = len(self.active_params)
        self.param_defs = [DENSITY_PARAMS[p] for p in self.active_params]
        self.has_normal_prior = any(
            p['prior']['type'] == 'normal' for p in self.param_defs
        )
        self.param_ranges = [p['range'] for p in self.param_defs]

        self.density_estimator = density_estimator.to(device)
        self.density_estimator.eval()
        self.posterior = build_sbi_posterior(
            self.density_estimator,
            self.data_cfg['dataset'], self.active_params, device,
        )

    # ------------------------------------------------------------------ #
    #  Sampling helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _sample_batch(self, x_batch: torch.Tensor, num_samples: int) -> np.ndarray:
        """Returns (B, num_samples, n_params) numpy."""
        samples = self.posterior.sample_batched(
            (num_samples,), x=x_batch.to(self.device)
        ).detach().cpu()
        return samples.permute(1, 0, 2).numpy()

    @torch.no_grad()
    def _sample_posteriors(self, x: torch.Tensor, num_samples: int,
                           batch_size: int = 16):
        """Returns (n, num_samples, n_params) numpy."""
        n = x.shape[0]
        out = np.zeros((n, num_samples, self.n_params))
        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size]
            out[i:i + xb.shape[0]] = self._sample_batch(xb, num_samples)
            if (i // batch_size) % 10 == 0:
                logging.info(
                    f"  sampled posteriors: {min(i + batch_size, n)}/{n}"
                )
        return out

    # ------------------------------------------------------------------ #
    #  Corner plot
    # ------------------------------------------------------------------ #

    def _plot_corner(self, label, samples):
        size = self.plot_cfg.get('fontsize', 16)
        color = self.plot_cfg.get('color', 'teal')
        names = [p['latex'] for p in self.param_defs]
        labs = [p['latex_bare'] for p in self.param_defs]

        samp_mc = MCSamples(samples=samples, names=names, labels=labs)
        g = plots.get_subplot_plotter()
        g.settings.legend_fontsize = size
        g.settings.axes_fontsize = size
        g.settings.axes_labelsize = size
        g.settings.linewidth = 2
        g.triangle_plot([samp_mc], filled=True, legend_loc='upper right',
                        colors=[color], contour_colors=[color])

        # Truth markers.
        for i in range(self.n_params):
            g.subplots[i, i].axes.axvline(label[i], color='k', ls='--', lw=2)
        for i in range(self.n_params):
            for j in range(i):
                g.subplots[i, j].axes.scatter(
                    label[j], label[i], color='k', marker='x', s=80)

        # Prior overlay + axis limits.
        priors = [p['prior'] for p in self.param_defs]
        ranges = self.param_ranges

        def _bounds(i):
            pi = priors[i]
            if pi['type'] == 'normal':
                return (pi['mean'] - 4 * pi['std'], pi['mean'] + 4 * pi['std'])
            return (ranges[i][0], ranges[i][1])

        for i in range(self.n_params):
            ax = g.subplots[i, i]
            pi = priors[i]
            if pi['type'] == 'uniform':
                ax.axvline(ranges[i][0], color='grey', ls=':', lw=1)
                ax.axvline(ranges[i][1], color='grey', ls=':', lw=1)
            else:
                xlim = ax.get_xlim()
                xs = np.linspace(xlim[0], xlim[1], 400)
                pdf = scipy_norm.pdf(xs, pi['mean'], pi['std'])
                peak = pdf.max()
                if peak > 0:
                    ax.plot(xs, pdf / peak, color='grey', ls='--', lw=1)
            lo_i, hi_i = _bounds(i)
            g.subplots[i, i].set_xlim(lo_i, hi_i)
            for j in range(i):
                lo_j, hi_j = _bounds(j)
                g.subplots[i, j].set_xlim(lo_j, hi_j)
                g.subplots[i, j].set_ylim(lo_i, hi_i)

        fig = g.fig
        post_patch = mpatches.Patch(color=color, label='Posterior')
        true_line = mlines.Line2D([], [], color='k', marker='x', ls='--',
                                  lw=2, markersize=10, label='True')
        handles = [post_patch, true_line]
        if self.has_normal_prior:
            handles.append(mlines.Line2D(
                [], [], color='grey', ls='--', lw=1, label='Prior'))
        fig.legend(handles=handles, bbox_to_anchor=(0.98, 0.98), fontsize=size)
        return fig

    def corner_plot(self):
        """One corner plot per requested test simulation."""
        n_plots = int(self.eval_cfg.get('n_corner_plots', 5))
        sample_size = int(self.eval_cfg.get('num_samples', 1000))

        x_test = self.data['x_test']
        y_test = self.data['y_test']
        if x_test.shape[0] == 0:
            logging.warning("No test data, skipping corner plot")
            return

        rng = np.random.default_rng(0)
        indices = rng.choice(x_test.shape[0],
                             size=min(n_plots, x_test.shape[0]),
                             replace=False)

        save_dir = self.output_dir + 'test_samples/'
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Making {len(indices)} corner plots...")

        with PdfPages(self.output_dir + 'corner.pdf') as pdf:
            for idx in indices:
                x_i = x_test[idx:idx + 1]
                samples = self._sample_batch(x_i, sample_size)[0]
                label = y_test[idx].cpu().numpy()
                np.savez(save_dir + f'test_{idx}.npz',
                         label=label, samples=samples)
                fig = self._plot_corner(label, samples)
                pdf.savefig(fig)
                plt.close(fig)
                logging.info(f"  corner plot for test sim {idx}")
        separator()

    # ------------------------------------------------------------------ #
    #  TARP coverage
    # ------------------------------------------------------------------ #

    def tarp_plot(self):
        if not self.eval_cfg.get('do_tarp', True):
            return
        try:
            from tarp import get_tarp_coverage
        except ImportError:
            logging.warning("tarp not installed, skipping TARP coverage")
            return

        sample_size = int(self.eval_cfg.get('num_tarp_samples', 1000))
        x_test = self.data['x_test']
        y_test = self.data['y_test']
        if x_test.shape[0] == 0:
            logging.warning("No test data, skipping TARP")
            return

        logging.info(f"Computing TARP coverage ({x_test.shape[0]} sims, "
                     f"{sample_size} posterior samples each)...")
        all_samples = self._sample_posteriors(x_test, sample_size)
        all_labels = y_test.cpu().numpy()
        samples_tarp = all_samples.transpose(1, 0, 2)  # (n_samples, n_sims, n_params)

        # Joint TARP.
        ecp, alpha = get_tarp_coverage(
            samples_tarp, all_labels,
            references='random', metric='euclidean', norm=True, bootstrap=False,
        )
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k', label='Ideal')
        ax.plot(alpha, ecp, label='TARP')
        ax.legend()
        ax.set_xlabel('Credibility Level')
        ax.set_ylabel('Expected Coverage')
        plt.tight_layout()
        plt.savefig(self.output_dir + 'tarp_joint.pdf', dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Per-parameter marginal TARP.
        fig, axes = plt.subplots(
            1, self.n_params,
            figsize=(3.5 * self.n_params, 3.5), sharey=True)
        if self.n_params == 1:
            axes = [axes]
        for p in range(self.n_params):
            sp = samples_tarp[:, :, p][:, :, None]
            yp = all_labels[:, p][:, None]
            ecp_p, alpha_p = get_tarp_coverage(
                sp, yp, references='random', metric='euclidean',
                norm=True, bootstrap=False,
            )
            ax = axes[p]
            ax.plot([0, 1], [0, 1], ls='--', color='k', lw=1)
            ax.plot(alpha_p, ecp_p, lw=2)
            ax.set_title(self.param_defs[p]['latex'])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Credibility level')
            if p == 0:
                ax.set_ylabel('Empirical coverage')
        plt.suptitle('TARP marginal coverage', y=1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir + 'tarp_marginal.pdf', dpi=300,
                    bbox_inches='tight')
        plt.close()
        logging.info("TARP plots saved")
        separator()

    # ------------------------------------------------------------------ #
    #  SBC rank
    # ------------------------------------------------------------------ #

    def sbc_plot(self):
        if not self.eval_cfg.get('do_sbc', True):
            return
        try:
            from sbi.diagnostics import run_sbc
            from sbi.analysis import sbc_rank_plot
        except ImportError:
            logging.warning("sbi diagnostics unavailable, skipping SBC")
            return

        n_sbc = int(self.eval_cfg.get('num_sbc_sims', 100))
        n_post = int(self.eval_cfg.get('num_sbc_samples', 1000))

        x_test = self.data['x_test']
        y_test = self.data['y_test']
        n_sbc = min(n_sbc, x_test.shape[0])
        if n_sbc == 0:
            logging.warning("No test data, skipping SBC")
            return

        logging.info(
            f"Running SBC ({n_sbc} sims, {n_post} posterior samples each)..."
        )
        theta = y_test[:n_sbc]
        xs = x_test[:n_sbc]
        ranks, _ = run_sbc(theta, xs, self.posterior,
                           num_posterior_samples=n_post)
        fig, _ = sbc_rank_plot(
            ranks, num_posterior_samples=n_post,
            parameter_labels=[p['latex'] for p in self.param_defs],
        )
        fig.savefig(self.output_dir + 'sbc_ranks.pdf', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
        logging.info("SBC rank plot saved")
        separator()

    # ------------------------------------------------------------------ #
    #  Entry
    # ------------------------------------------------------------------ #

    def main(self):
        if self.eval_cfg.get('do_corner', True):
            self.corner_plot()
        if self.eval_cfg.get('do_tarp', True):
            self.tarp_plot()
        if self.eval_cfg.get('do_sbc', True):
            self.sbc_plot()
        separator()
        logging.info('Done plotting')
        separator()

import logging
import os
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import binom

import torch
import torch.nn as nn
from getdist import plots, MCSamples
from scipy.stats import norm as scipy_norm

from ..util.logger import separator
from .priors import DENSITY_PARAMS


class DensityPlotting:
    """Evaluation and plotting for density field inference.

    When use_sbi is True, uses SBI's DirectPosterior for sampling and
    provides access to SBI's native SBC diagnostics.
    """

    def __init__(self, params: dict, encoder: nn.Module, flow: nn.Module,
                 data: dict, device: torch.device,
                 density_estimator=None):
        self.params = params
        self.plot_cfg = params['density'].get('plot', params.get('plot', {}))
        self.data_cfg = params['density']['data']
        self.data = data
        self.device = device
        self.output_dir = params['density'].get('plot', {}).get('plot_dir',
                          params.get('plot', {}).get('plot_dir', 'plots/'))
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

        # SBI mode
        self.use_sbi = params['density']['flow'].get('use_sbi', False)

        if self.use_sbi:
            from .flow import build_sbi_posterior
            self.density_estimator = density_estimator.to(device)
            self.density_estimator.eval()
            self.posterior = build_sbi_posterior(
                self.density_estimator, self.active_params, device)
            self.encoder = self.density_estimator.net._embedding_net
            self.flow = None
        else:
            self.encoder = encoder.to(device)
            self.flow = flow.to(device)
            self.density_estimator = None
            self.posterior = None
            self.encoder.eval()
            self.flow.eval()

    # ------------------------------------------------------------------ #
    #  Sampling helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _sample_sbi(self, x_batch: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample from SBI posterior. Returns (B, num_samples, n_params)."""
        # sample_batched returns (num_samples, B, n_params)
        samples = self.posterior.sample_batched(
            (num_samples,), x=x_batch.to(self.device)
        ).detach().cpu()
        # Transpose to (B, num_samples, n_params)
        return samples.permute(1, 0, 2)

    @torch.no_grad()
    def _sample_nflows(self, x_batch: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample from nflows flow. Returns (B, num_samples, n_params)."""
        context = self.encoder(x_batch.to(self.device))
        return self.flow.sample(num_samples, context=context).cpu()

    @torch.no_grad()
    def sample_posteriors(self, x_test: torch.Tensor, y_test: torch.Tensor,
                          num_samples: int = 10000, batch_size: int = 16):
        """Sample posteriors for all test data.

        Returns:
            all_samples: (n_sims, num_samples, n_params) numpy, physical units
            all_labels: (n_sims, n_params) numpy, physical units
        """
        n_sims = x_test.shape[0]
        all_samples = np.zeros((n_sims, num_samples, self.n_params))
        all_labels = y_test.cpu().numpy()

        for i in range(0, n_sims, batch_size):
            x_batch = x_test[i:i + batch_size]

            if self.use_sbi:
                samples = self._sample_sbi(x_batch, num_samples)
            else:
                samples = self._sample_nflows(x_batch, num_samples)

            B = samples.shape[0]
            all_samples[i:i + B] = samples.numpy()

            if (i // batch_size) % 10 == 0:
                logging.info(f"  Sampled posteriors: {min(i + B, n_sims)}/{n_sims}")

        return all_samples, all_labels

    def calc_statistics(self, sample_size: int = 10000,
                        cal_error: bool = True,
                        rank_stat: bool = True,
                        mean_metrics: bool = True):
        """Calculate test statistics."""
        logging.info('Calculating statistics...')
        x_test = self.data['x_test']
        y_test = self.data['y_test']

        all_samples, all_labels = self.sample_posteriors(x_test, y_test, sample_size)
        n_sims = all_samples.shape[0]

        names = [p['latex'] for p in self.param_defs]
        labs = [p['latex_bare'] for p in self.param_defs]

        mean = np.zeros((n_sims, self.n_params))
        lower = np.zeros((n_sims, self.n_params))
        upper = np.zeros((n_sims, self.n_params))
        rank = np.zeros((n_sims, self.n_params)) if rank_stat else None

        if cal_error:
            bin_alpha = 100
            alpha = np.linspace(0.01, 0.99, bin_alpha)
            alpha_0 = np.zeros((n_sims, bin_alpha, self.n_params))

        for i in range(n_sims):
            samples_i = all_samples[i]  # (num_samples, n_params)

            if rank_stat:
                for j in range(self.n_params):
                    rank[i, j] = (samples_i[:, j] < all_labels[i, j]).sum()

            if cal_error:
                for a_idx, a in enumerate(alpha):
                    for p in range(self.n_params):
                        try:
                            samp_1d = MCSamples(samples=samples_i[:, p:p + 1],
                                                settings={'smooth_scale_1D': 0.5})
                            grid = samp_1d.get1DDensityGridData(0)
                            low, up = grid.getLimits([a])[0:2]
                            if low < all_labels[i, p] < up:
                                alpha_0[i, a_idx, p] = 1
                        except (IndexError, Exception):
                            pass

            samp_mc = MCSamples(samples=samples_i, names=names, labels=labs,
                                settings={'smooth_scale_2D': 0.5, 'smooth_scale_1D': 0.5})
            stats = samp_mc.getMargeStats()
            for p, pname in enumerate(names):
                mean[i, p] = stats.parWithName(pname).mean
                mean_stat = stats.parWithName(pname)
                if mean_stat.limits:
                    lower[i, p] = mean_stat.limits[0].lower
                    upper[i, p] = mean_stat.limits[0].upper

            if i % 100 == 0:
                logging.info(f"  Stats: {i}/{n_sims}")

        results = {
            'label': all_labels,
            'mean': mean,
            'lower': lower,
            'upper': upper,
        }

        if rank_stat:
            results['rank'] = rank

        if cal_error:
            alpha_0_bar = np.mean(alpha_0, axis=0)
            cal_err = np.mean(np.absolute(alpha_0_bar.T - alpha), axis=1)
            results['cal_err'] = cal_err

        if mean_metrics:
            r2_mean = np.zeros(self.n_params)
            nrmse = np.zeros(self.n_params)
            for p in range(self.n_params):
                avg = np.mean(all_labels[:, p])
                ss_res = np.sum((all_labels[:, p] - mean[:, p]) ** 2)
                ss_tot = np.sum((all_labels[:, p] - avg) ** 2)
                r2_mean[p] = 1 - ss_res / ss_tot
                rmse = np.sqrt(np.mean((mean[:, p] - all_labels[:, p]) ** 2))
                nrmse[p] = rmse / (np.max(all_labels[:, p]) - np.min(all_labels[:, p]))
            results['r2_mean'] = r2_mean
            results['nrmse'] = nrmse
            for p, pdef in enumerate(self.param_defs):
                logging.info(f"  {pdef['name']}: R2={r2_mean[p]:.4f}, NRMSE={nrmse[p]:.4f}")

        np.savez(self.output_dir + 'statistics.npz', **results)
        logging.info('Saved statistics')
        separator()

    def plot_calibration(self):
        """Plot predicted vs true with 68% CL error bars."""
        logging.info('Making calibration plots')
        data = np.load(self.output_dir + 'statistics.npz')
        label = data['label']
        mean = data['mean']
        lower = data['lower']
        upper = data['upper']
        error_low = np.abs(mean - lower)
        error_up = np.abs(upper - mean)
        size = self.plot_cfg.get('fontsize', 16)

        with PdfPages(self.output_dir + 'calibration.pdf') as pdf:
            for p, pdef in enumerate(self.param_defs):
                plt.figure(figsize=(9, 6))
                plt.errorbar(label[:, p], mean[:, p],
                             yerr=(error_low[:, p], error_up[:, p]),
                             fmt='.', markersize=3, color='darkred',
                             alpha=1, lw=1., ecolor='lightsteelblue')
                plt.plot(label[:, p], label[:, p], color='k', zorder=100, lw=1)
                plt.title(pdef['latex'], fontsize=size)
                plt.xlabel('True', fontsize=size)
                plt.ylabel('Posterior (68% CL)', fontsize=size)
                plt.xticks(fontsize=size)
                plt.yticks(fontsize=size)
                pdf.savefig()
                plt.close()
        separator()

    def plot_rank_statistic(self):
        """Plot SBC rank histograms with binomial confidence bands."""
        logging.info('Making rank statistic plots')
        data = np.load(self.output_dir + 'statistics.npz')
        label = data['label']
        rank = data['rank']
        size = self.plot_cfg.get('fontsize', 16)
        bins = 15
        sample_size = int(np.max(rank))
        ranges = np.linspace(-500, sample_size + 500, bins)
        avg = label.shape[0] / bins
        low, up = binom.interval(0.99, label.shape[0], 1 / bins)

        with PdfPages(self.output_dir + 'rank_statistic.pdf') as pdf:
            for p, pdef in enumerate(self.param_defs):
                plt.figure(figsize=(9, 6))
                plt.hist(rank[:, p], bins=bins, ec='teal', histtype='step')
                plt.fill_between(ranges, low, up, color='k', alpha=0.2)
                plt.plot(ranges, np.ones(bins) * avg, color='k')
                plt.title(pdef['latex'], fontsize=size)
                plt.xlabel('Rank statistic', fontsize=size)
                plt.xticks(fontsize=size)
                plt.yticks(fontsize=0)
                pdf.savefig()
                plt.close()
        separator()

    def plot_corner(self, label: np.ndarray, samples: np.ndarray):
        """Make a corner plot for one test sim."""
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
        g.settings.line_labels = False
        g.triangle_plot([samp_mc], filled=True, legend_loc='upper right',
                        colors=[color], contour_colors=[color])

        for i in range(self.n_params):
            ax = g.subplots[i, i].axes
            ax.axvline(label[i], color='k', ls='--', lw=2)
        n = 0
        m = 1
        while n < self.n_params - 1:
            ax = g.subplots[m, n].axes
            ax.scatter(label[n], label[m], color='k', marker='x', s=100)
            m += 1
            if m == self.n_params:
                n += 1
                m = n + 1

        # Prior overlay: uniform => dotted bounds, normal => bell curve / ellipses
        ranges = self.param_ranges
        priors = [p['prior'] for p in self.param_defs]

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
                    pdf = pdf / peak
                ax.plot(xs, pdf, color='grey', ls='--', lw=1)

            for j in range(i):
                ax_ij = g.subplots[i, j]
                pj = priors[j]
                if pj['type'] == 'uniform':
                    ax_ij.axvline(ranges[j][0], color='grey', ls=':', lw=1)
                    ax_ij.axvline(ranges[j][1], color='grey', ls=':', lw=1)
                else:
                    for k in (1, 2):
                        ax_ij.axvline(pj['mean'] - k * pj['std'],
                                      color='grey', ls='--', lw=1)
                        ax_ij.axvline(pj['mean'] + k * pj['std'],
                                      color='grey', ls='--', lw=1)
                if pi['type'] == 'uniform':
                    ax_ij.axhline(ranges[i][0], color='grey', ls=':', lw=1)
                    ax_ij.axhline(ranges[i][1], color='grey', ls=':', lw=1)
                else:
                    for k in (1, 2):
                        ax_ij.axhline(pi['mean'] - k * pi['std'],
                                      color='grey', ls='--', lw=1)
                        ax_ij.axhline(pi['mean'] + k * pi['std'],
                                      color='grey', ls='--', lw=1)

        # Set axis limits: uniform priors use the prior range; normal priors
        # use a hard ±4σ window around the mean.
        def _bounds(i):
            pi = priors[i]
            if pi['type'] == 'normal':
                return (pi['mean'] - 4 * pi['std'],
                        pi['mean'] + 4 * pi['std'])
            return (ranges[i][0], ranges[i][1])

        for i in range(self.n_params):
            lo_i, hi_i = _bounds(i)
            g.subplots[i, i].set_xlim(lo_i, hi_i)
            for j in range(i):
                lo_j, hi_j = _bounds(j)
                g.subplots[i, j].set_xlim(lo_j, hi_j)
                g.subplots[i, j].set_ylim(lo_i, hi_i)

        fig = g.fig
        post_patch = mpatches.Patch(color=color, label='Posterior')
        true_line = mlines.Line2D([], [], color='k', marker='x', ls='--', lw=2,
                                  markersize=10, label='True')
        handles = [post_patch, true_line]
        if self.has_normal_prior:
            handles.append(mlines.Line2D([], [], color='grey', ls='--', lw=1,
                                         label=r'Prior (1$\sigma$, 2$\sigma$)'))
        fig.legend(handles=handles, bbox_to_anchor=(0.98, 0.98), fontsize=size)
        return fig

    def _sample_for_corner(self, x_i: torch.Tensor, sample_size: int) -> np.ndarray:
        """Sample posterior for a single observation and rescale. Returns (sample_size, n_params)."""
        with torch.no_grad():
            if self.use_sbi:
                samples = self._sample_sbi(x_i, sample_size)
            else:
                samples = self._sample_nflows(x_i, sample_size)
        return samples.numpy()[0]

    def plot_many_corners(self):
        """Generate corner plots for test sims."""
        corner_cfg = self.plot_cfg.get('corner_plot', {})
        if not corner_cfg.get('do_it', True):
            return

        n_plots = corner_cfg.get('n_plots', 5)
        sample_size = corner_cfg.get('sample_size', 10000)

        x_test = self.data['x_test']
        y_test = self.data['y_test']
        n_test = x_test.shape[0]

        indices = np.random.choice(n_test, min(n_plots, n_test), replace=False)

        os.makedirs(self.output_dir + 'fiducial_samples', exist_ok=True)
        logging.info(f'Making {len(indices)} corner plots')

        figures = []
        for idx in indices:
            x_i = x_test[idx:idx + 1]
            samples_np = self._sample_for_corner(x_i, sample_size)
            label_np = y_test[idx].numpy()

            np.savez(self.output_dir + f'fiducial_samples/fiducial_{idx}.npz',
                     label=label_np, samples=samples_np)
            logging.info(f'  Corner plot for test sim {idx}')
            figures.append(self.plot_corner(label_np, samples_np))

        with PdfPages(self.output_dir + 'corner.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        separator()

    def plot_fiducial_corners(self):
        """Generate corner plots for fiducial (Planck 2018) simulations."""
        corner_cfg = self.plot_cfg.get('corner_plot', {})
        if not corner_cfg.get('fiducial', True):
            return

        norm_fid_data = self.data.get('norm_fid_data')
        if norm_fid_data is None:
            logging.warning('No fiducial data loaded, skipping fiducial corners')
            return

        n_plots = corner_cfg.get('n_plots', 5)
        sample_size = corner_cfg.get('sample_size', 10000)

        FIDUCIAL_COSMO = {
            'Omega_m': 0.3175, 'Omega_b': 0.0490,
            'h': 0.6711, 'n_s': 0.9624, 'sigma_8': 0.8340,
            'b1': 1.0, 'b2': 0.0, 'bs2': 0.0, 'bn2': 0.0,
        }
        fiducial_label = np.array([FIDUCIAL_COSMO[p] for p in self.active_params])

        n_fid = norm_fid_data.shape[0]
        n_plots = min(n_plots, n_fid)

        save_dir = self.output_dir + 'fiducial_samples'
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f'Making {n_plots} fiducial corner plots')

        figures = []
        for idx in range(n_plots):
            x_i = norm_fid_data[idx:idx + 1]
            samples_np = self._sample_for_corner(x_i, sample_size)

            np.savez(f'{save_dir}/fiducial_fid_{idx}.npz',
                     label=fiducial_label, samples=samples_np)
            logging.info(f'  Fiducial corner plot {idx}')
            figures.append(self.plot_corner(fiducial_label, samples_np))

        with PdfPages(self.output_dir + 'corner_fiducial.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        separator()

    def tarp_coverage(self):
        """TARP diagnostic for posterior calibration."""
        try:
            from tarp import get_tarp_coverage
        except ImportError:
            logging.warning("tarp not installed, skipping TARP coverage")
            return

        logging.info('Computing TARP coverage...')
        x_test = self.data['x_test']
        y_test = self.data['y_test']
        sample_size = self.plot_cfg.get('tarp_samples', 5000)

        all_samples, all_labels = self.sample_posteriors(
            x_test, y_test, num_samples=sample_size)

        # TARP expects: samples (n_samples, n_sims, n_params), y (n_sims, n_params)
        samples_tarp = all_samples.transpose(1, 0, 2)  # (num_samples, n_sims, n_params)

        # Joint TARP
        ecp, alpha = get_tarp_coverage(samples_tarp, all_labels,
                                       references="random", metric="euclidean",
                                       norm=True, bootstrap=False)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k', label='Ideal')
        ax.plot(alpha, ecp, label='TARP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")
        plt.tight_layout()
        plt.savefig(self.output_dir + 'tarp_joint.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Marginal TARP per parameter
        fig, axes = plt.subplots(1, self.n_params, figsize=(3.5 * self.n_params, 3.5), sharey=True)
        if self.n_params == 1:
            axes = [axes]
        for p in range(self.n_params):
            samples_p = samples_tarp[:, :, p][:, :, None]
            y_p = all_labels[:, p][:, None]
            ecp_p, alpha_p = get_tarp_coverage(samples_p, y_p,
                                               references="random", metric="euclidean",
                                               norm=True, bootstrap=False)
            ax = axes[p]
            ax.plot([0, 1], [0, 1], ls='--', color='k', lw=1)
            ax.plot(alpha_p, ecp_p, lw=2)
            ax.set_title(self.param_defs[p]['latex'])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Credibility level")
            if p == 0:
                ax.set_ylabel("Empirical coverage")

        plt.suptitle("TARP marginal coverage", y=1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir + 'tarp_marginal.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info('TARP plots saved')
        separator()

    @torch.no_grad()
    def check_latent(self):
        """Check latent space Gaussianity."""
        logging.info('Making latent space plots')
        x_test = self.data['x_test']
        y_test = self.data['y_test']
        size = self.plot_cfg.get('fontsize', 16)

        # Encode test data
        if self.use_sbi:
            encoder = self.density_estimator.net._embedding_net
        else:
            encoder = self.encoder

        embeddings = []
        for i in range(0, len(x_test), 32):
            x_batch = x_test[i:i + 32].to(self.device)
            embeddings.append(encoder(x_batch))
        context = torch.cat(embeddings, dim=0)
        y_test_dev = y_test.to(self.device)

        # Extract latent z
        if self.use_sbi:
            z = self.density_estimator.net._transform(y_test_dev, context=context)[0]
            z = z.cpu().numpy()
        elif hasattr(self.flow, 'inn'):
            # FrEIA flow
            z, _ = self.flow.inn(y_test_dev, c=[context])
            z = z.cpu().numpy()
        elif hasattr(self.flow, 'flow'):
            # nflows: transform_to_noise gives z
            z = self.flow.flow._transform(y_test_dev, context=context)[0]
            z = z.cpu().numpy()
        else:
            logging.warning("Cannot extract latent space for this flow type")
            return

        names = [f"z_{i}" for i in range(self.n_params)]
        labs = [f"z_{i}" for i in range(self.n_params)]
        r = np.random.randn(100000, self.n_params)
        samp_gaussian = MCSamples(samples=r, names=names, labels=labs)
        samp_mc = MCSamples(samples=z, names=names, labels=labs)
        g = plots.get_subplot_plotter()
        g.settings.legend_fontsize = size
        g.settings.axes_fontsize = size
        g.settings.axes_labelsize = size
        g.settings.linewidth = 2
        color = ['darkred', 'k']
        g.triangle_plot([samp_mc, samp_gaussian], filled=[True, False],
                        legend_labels=['Latent Distribution', 'Gaussian'],
                        legend_loc='upper right', colors=color, contour_colors=color)
        plt.savefig(self.output_dir + 'latent.pdf')
        plt.close()
        separator()

    def sbi_sbc_diagnostics(self):
        """Run SBI's native SBC (Simulation-Based Calibration) diagnostics."""
        if not self.use_sbi:
            return

        try:
            from sbi.diagnostics import run_sbc
            from sbi.analysis import sbc_rank_plot
        except ImportError:
            logging.warning("sbi diagnostics not available, skipping SBI SBC")
            return

        sbc_cfg = self.plot_cfg.get('sbi_sbc', {})
        if isinstance(sbc_cfg, bool):
            if not sbc_cfg:
                return
            sbc_cfg = {}

        num_sbc_samples = sbc_cfg.get('num_posterior_samples', 1000)
        n_sbc_sims = sbc_cfg.get('n_sims', min(100, self.data['x_test'].shape[0]))

        logging.info(f'Running SBI SBC diagnostics ({n_sbc_sims} sims, '
                     f'{num_sbc_samples} posterior samples each)...')

        sbc_theta = self.data['y_test'][:n_sbc_sims]
        sbc_x = self.data['x_test'][:n_sbc_sims]

        ranks, dap_samples = run_sbc(
            sbc_theta, sbc_x, self.posterior,
            num_posterior_samples=num_sbc_samples,
        )

        fig, axes = sbc_rank_plot(
            ranks, num_posterior_samples=num_sbc_samples,
            parameter_labels=[p['latex'] for p in self.param_defs],
        )
        fig.savefig(self.output_dir + 'sbi_sbc_ranks.pdf',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info('SBI SBC rank plot saved')
        separator()

    def main(self):
        plot_cfg = self.plot_cfg

        # Fast plots first (no dependency on calc_statistics)
        if plot_cfg.get('corner_plot', {}).get('do_it', True):
            self.plot_many_corners()
            self.plot_fiducial_corners()
        if plot_cfg.get('check_latent', True):
            self.check_latent()
        if plot_cfg.get('tarp', False):
            self.tarp_coverage()

        # SBI native SBC diagnostics
        if self.use_sbi and plot_cfg.get('sbi_sbc', False):
            self.sbi_sbc_diagnostics()

        # Slow statistics + dependent plots
        if plot_cfg.get('calc_statistics', {}).get('do_it', True):
            stat_cfg = plot_cfg.get('calc_statistics', {})
            self.calc_statistics(
                sample_size=stat_cfg.get('sample_size', 10000),
                cal_error=stat_cfg.get('cal_error', True),
                rank_stat=stat_cfg.get('rank_stat', True),
                mean_metrics=stat_cfg.get('mean_metrics', True),
            )
        if plot_cfg.get('plot_calibration', True):
            self.plot_calibration()
        if plot_cfg.get('plot_rank_stat', True):
            self.plot_rank_statistic()

        separator()
        logging.info('Done plotting')
        separator()

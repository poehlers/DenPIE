#!/usr/bin/env python
"""CLI script for running the Fisher forecast.

Usage
-----
    python core/run_fisher.py --n-seeds 10 --n-bins 30 --output-dir outputs/fisher/
    python core/run_fisher.py --sbi-samples outputs/5param_run/sbi_samples.npz
    python core/run_fisher.py --sbi-samples /projects/prjs1926/data/fli_data/fli_5param
"""

import argparse
import os
import sys
import time
from datetime import datetime


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Fisher forecast from DiscoDJ via JAX autodiff",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file with default values for the flags below. "
             "Keys must match argparse dests (dashes -> underscores). "
             "CLI flags passed explicitly override the config; missing keys "
             "fall through to the argparse default. The file is copied into "
             "the run's output directory as 'source_config.yml'.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=20,
        help="Number of white-noise seeds for Jacobian averaging (default: 20). "
             "Seed-averaged Jacobian noise biases the Fisher *upward* ~1/n_seeds, "
             "so n_seeds<10 produces artificially tight constraints.",
    )
    parser.add_argument(
        "--n-bins", type=int, default=30,
        help="Number of P(k) bins (default: 30)",
    )
    parser.add_argument(
        "--sbi-samples", type=str, default=None,
        help="Path to .npz file with SBI posterior samples or Falcon run directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/fisher_<field>[_bias])",
    )
    parser.add_argument(
        "--res", type=int, default=None,
        help="Override resolution (default: from BOX_PARAMS = 64)",
    )
    parser.add_argument(
        "--boxsize", type=float, default=None,
        help="Override box size in Mpc/h (default: from BOX_PARAMS = 1000)",
    )
    parser.add_argument(
        "--bias", action="store_true",
        help="Run biased tracer Fisher forecast (joint cosmo + bias params)",
    )
    parser.add_argument(
        "--field", type=str, choices=["fin", "ic", "both"], default="fin",
        help="Field to forecast on: 'fin' = P(delta_fin) after N-body, "
             "'ic' = P(delta_ic) on the Lagrangian grid (no N-body), "
             "'both' = joint [P_ic, B_ic, P_fin, B_fin] data vector "
             "(only valid with --bias --joint-pb). Combined with --bias, "
             "'ic' measures the Lagrangian bias expansion field "
             "P(w(q)-1). Default: fin.",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Run only the JVP tangent diagnostic (per-stage finiteness "
             "of the forward-mode AD pipeline) and exit. No Fisher matrix "
             "is computed.",
    )
    parser.add_argument(
        "--joint-pb", action="store_true",
        help="Joint P(k)+B(k1,k2,k3) Fisher forecast via BFast. "
             "Requires --bias. Uses linear k_F-spaced bin_edges (independent "
             "of --n-bins). Output defaults to outputs/fisher_bias_P_B.",
    )
    parser.add_argument(
        "--pb-step", type=int, default=1,
        help="Stride for BFast bin_edges = arange(1, res//3+1, step) in "
             "units of k_F. Larger step = coarser binning, fewer triangles, "
             "faster jacrev. Default: 1.",
    )
    parser.add_argument(
        "--pk-multipoles", action="store_true",
        help="Use redshift-space P-multipoles (P_0, P_2, P_4) instead of "
             "the real-space P monopole on the P side. Bispectrum stays "
             "monopole. Requires --bias --joint-pb and --field fin or both. "
             "Hard-codes los_axis=2 (z-axis) and applies RSD to particles "
             "before scattering. Output defaults to "
             "outputs/fisher_bias_P_B_rsd[_both].",
    )
    parser.add_argument(
        "--n-g-density", type=float, default=None,
        help="If set, inject Gaussian shot noise N(0, 1/(n_g * V_cell)) per "
             "voxel into the field summarised by the pipeline (delta_biased "
             "for --field fin, delta_L for --field ic, both for --field both). "
             "Matches arXiv:2504.20130v2 (typical: 1.0e-3 for DESI-like). "
             "Output dir is suffixed with '_SN'. Requires --bias.",
    )
    parser.add_argument(
        "--jacrev-chunk-size", type=int, default=None,
        help="If set, compute jax.jacrev one chunk of `chunk_size` output "
             "basis vectors at a time via jax.lax.map (memory-bounded). "
             "Use when vanilla jacrev OOMs with large data vectors (e.g. "
             "--field both --pk-multipoles at res>=128). Default: None "
             "(vanilla jacrev). Ignored when --derivative-method fd.",
    )
    parser.add_argument(
        "--derivative-method", type=str, choices=["autodiff", "fd", "jacfwd"],
        default="autodiff",
        help="How to compute the data-vector Jacobian in fisher_bias_PB. "
             "'autodiff' uses jax.jacrev with stop_gradient(cosmo) through "
             "the N-body (1 fwd + ~n_data bwd per seed; Omega_m growth omitted). "
             "'fd' uses central finite differences with per-parameter step sizes "
             "from _FINITE_DIFF_STEPS_BIAS (1 + 2*n_params = 13 forwards per "
             "seed; growth-complete). 'jacfwd' uses forward-mode AD + x64 "
             "(growth-complete at autodiff precision -- the correct method; "
             "field='fin' only). Default: autodiff.",
    )
    parser.add_argument(
        "--cov-type", type=str,
        choices=["diag", "euclid_cov", "block_with_pb_cross_empirical"],
        default="diag",
        help="Data covariance model. 'diag' (default): block-diagonal "
             "Gaussian (2P^2/N_modes or Grieb multipole) + Scoccimarro B + "
             "zero P-B cross. 'euclid_cov': empirical from --n-cov-seeds "
             "fiducial seeds, Hartlap-corrected; captures P-B cross, k-k' "
             "cross, multipole l-l' cross, IC-FIN cross, and non-Gaussian "
             "terms. 'block_with_pb_cross_empirical' (joint-PB only): "
             "keeps analytic P-P (Grieb) and B-B (Scoccimarro) but adds an "
             "empirical P-B cross block estimated from --n-cov-seeds seeds; "
             "cheaper than full euclid_cov.",
    )
    parser.add_argument(
        "--n-cov-seeds", type=int, default=200,
        help="Number of fiducial seeds for --cov-type euclid_cov / "
             "block_with_pb_cross_empirical. Must exceed data vector length "
             "+ 2 for an invertible sample covariance. Default: 200.",
    )
    parser.add_argument(
        "--fd-step-kind", type=str, choices=["absolute", "relative"],
        default="absolute",
        help="Central-FD step convention. 'absolute' (default): uses "
             "absolute steps from _FINITE_DIFF_STEPS{,_BIAS} (reproduces "
             "existing FD outputs bit-for-bit). 'relative': "
             "step_i = max(--fd-relative-eps * |theta_i|, abs_step_i) "
             "matching the Franco-Abellan 2024 notebook; the absolute floor "
             "protects params with fiducial=0 (b2, bs2, bn2).",
    )
    parser.add_argument(
        "--fd-relative-eps", type=float, default=1e-2,
        help="Relative-step amplitude when --fd-step-kind relative. "
             "Default: 1e-2 (matches Franco-Abellan 2024).",
    )
    # Bias fiducial override (defaults reproduce the hardcoded _FIDUCIAL_BIAS:
    # b1=1, b2=bs2=bn2=0). Set b2=bs2=bn2=1 for the fli_bias_nonzero dataset.
    parser.add_argument("--fid-b1", type=float, default=1.0,
                        help="Fiducial b1 for the bias forecast (default 1.0).")
    parser.add_argument("--fid-b2", type=float, default=0.0,
                        help="Fiducial b2 (default 0.0; use 1.0 for fli_bias_nonzero).")
    parser.add_argument("--fid-bs2", type=float, default=0.0,
                        help="Fiducial bs2 (default 0.0; use 1.0 for fli_bias_nonzero).")
    parser.add_argument("--fid-bn2", type=float, default=0.0,
                        help="Fiducial bn2 (default 0.0; use 1.0 for fli_bias_nonzero).")
    return parser


def _load_config(parser, config_path):
    """Load a YAML config and apply it as argparse defaults.

    YAML keys must match argparse dests (dashes -> underscores). Unknown
    keys raise; missing keys fall through to the existing argparse
    defaults. CLI flags passed explicitly still override because
    argparse applies CLI > set_defaults > add_argument(default=...).
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        parser.error(
            f"--config {config_path}: expected a YAML mapping at top level, "
            f"got {type(cfg).__name__}"
        )
    known_dests = {a.dest for a in parser._actions}
    unknown = sorted(set(cfg) - known_dests)
    if unknown:
        parser.error(
            f"--config {config_path}: unknown keys {unknown}. "
            f"Allowed keys: {sorted(known_dests - {'help', 'config'})}"
        )
    parser.set_defaults(**cfg)
    return cfg


def main(argv=None):
    """Run a Fisher forecast.

    ``argv`` defaults to ``sys.argv[1:]`` (standalone use:
    ``python -m den_pie.fisher.run_fisher ...``). The den_pie CLI passes the
    arguments after ``fisher-forecast`` through here, so
    ``den_pie fisher-forecast --config ...`` and
    ``python core/run_fisher.py --config ...`` behave identically.
    """
    parser = _build_parser()

    # First pass: detect --config without enforcing required args.
    pre_args, _ = parser.parse_known_args(argv)
    loaded_cfg = None
    if pre_args.config is not None:
        loaded_cfg = _load_config(parser, pre_args.config)

    args = parser.parse_args(argv)

    if args.joint_pb and not args.bias:
        parser.error("--joint-pb requires --bias (joint P+B is bias-only)")
    if args.joint_pb and args.diagnose:
        parser.error("--joint-pb and --diagnose are mutually exclusive")
    if args.field == "both" and not args.joint_pb:
        parser.error("--field both requires --joint-pb")
    if args.pk_multipoles and not args.joint_pb:
        parser.error("--pk-multipoles requires --joint-pb")
    if args.pk_multipoles and args.field not in ("fin", "both"):
        parser.error("--pk-multipoles requires --field fin or --field both")
    if args.n_g_density is not None:
        if not args.bias:
            parser.error("--n-g-density requires --bias")
        if args.n_g_density <= 0.0:
            parser.error("--n-g-density must be positive")
    if args.derivative_method == "fd" and args.jacrev_chunk_size is not None:
        print("WARNING: --jacrev-chunk-size is ignored when "
              "--derivative-method fd (FD does only forward passes).")
    if args.cov_type == "block_with_pb_cross_empirical" and not args.joint_pb:
        parser.error("--cov-type block_with_pb_cross_empirical requires "
                     "--joint-pb (no bispectrum block to cross-couple with).")
    if args.cov_type in ("euclid_cov", "block_with_pb_cross_empirical") \
            and args.n_cov_seeds <= 2:
        parser.error("--n-cov-seeds must be > 2; the runner additionally "
                     "checks n_cov_seeds > n_data + 2 once n_data is known.")
    if args.fd_relative_eps <= 0.0:
        parser.error("--fd-relative-eps must be positive.")
    if args.derivative_method == "jacfwd" and args.field != "fin":
        parser.error("--derivative-method jacfwd is only wired for --field fin.")

    # jacfwd needs float64 for the DiscoDJ growth ODE (float32 -> NaN tangent).
    # Must be set before JAX initialises (i.e. before importing fisher/model).
    if args.derivative_method == "jacfwd":
        import jax
        jax.config.update("jax_enable_x64", True)
        print("  [jacfwd] enabled jax_enable_x64=True (growth ODE needs float64).")

    # Import after parsing (JAX init is slow)
    from den_pie.fisher.fisher import (
        run_fisher_forecast, run_fisher_forecast_bias,
        run_fisher_forecast_bias_PB,
        run_diagnose, run_diagnose_bias,
        BOX_PARAMS, SIM_PARAMS,
    )

    box_params = dict(BOX_PARAMS)
    sim_params = dict(SIM_PARAMS)

    if args.res is not None:
        box_params["res"] = args.res
        sim_params["res_pm"] = 2 * args.res
    if args.boxsize is not None:
        box_params["boxsize"] = args.boxsize

    # Default output dir.
    # For --bias --joint-pb, leave None and let run_fisher_forecast_bias_PB
    # pick a per-field default (outputs/fisher_bias_P_B[_ic|_both]).
    if args.output_dir is None:
        if args.bias and args.joint_pb:
            output_dir = None
        else:
            suffix = "_bias" if args.bias else ""
            output_dir = f"outputs/fisher_{args.field}{suffix}"
    else:
        output_dir = args.output_dir

    print("=== Fisher run config ===")
    if args.config is not None:
        print(f"  (loaded from --config {args.config})")
    for _k, _v in vars(args).items():
        print(f"  {_k}: {_v}")
    print(f"  output_dir: {output_dir}")
    print(f"  box_params: {box_params}")
    print(f"  sim_params: {sim_params}")
    print("=========================")

    print(f"=== Python start: {datetime.now().isoformat(timespec='seconds')} ===")
    t0 = time.time()
    if args.diagnose:
        if args.bias:
            run_diagnose_bias(
                n_bins=args.n_bins,
                box_params=box_params,
                sim_params=sim_params,
                field=args.field,
            )
        else:
            run_diagnose(
                n_bins=args.n_bins,
                box_params=box_params,
                sim_params=sim_params,
                field=args.field,
            )
    elif args.bias and args.joint_pb:
        run_fisher_forecast_bias_PB(
            n_seeds=args.n_seeds,
            bin_edges_step=args.pb_step,
            sbi_samples_path=args.sbi_samples,
            output_dir=output_dir,
            box_params=box_params,
            sim_params=sim_params,
            field=args.field,
            use_multipoles=args.pk_multipoles,
            n_g_density=args.n_g_density,
            jacrev_chunk_size=args.jacrev_chunk_size,
            derivative_method=args.derivative_method,
            config_path=args.config,
            resolved_args={k: v for k, v in vars(args).items() if k != "config"},
            cov_type=args.cov_type,
            n_cov_seeds=args.n_cov_seeds,
            fd_step_kind=args.fd_step_kind,
            fd_relative_eps=args.fd_relative_eps,
            fiducial_bias={"b1": args.fid_b1, "b2": args.fid_b2,
                           "bs2": args.fid_bs2, "bn2": args.fid_bn2},
        )
    elif args.bias:
        run_fisher_forecast_bias(
            n_seeds=args.n_seeds,
            n_bins=args.n_bins,
            sbi_samples_path=args.sbi_samples,
            output_dir=output_dir,
            box_params=box_params,
            sim_params=sim_params,
            field=args.field,
            n_g_density=args.n_g_density,
            jacrev_chunk_size=args.jacrev_chunk_size,
            cov_type=args.cov_type,
            n_cov_seeds=args.n_cov_seeds,
            fd_step_kind=args.fd_step_kind,
            fd_relative_eps=args.fd_relative_eps,
        )
    else:
        run_fisher_forecast(
            n_seeds=args.n_seeds,
            n_bins=args.n_bins,
            sbi_samples_path=args.sbi_samples,
            output_dir=output_dir,
            box_params=box_params,
            sim_params=sim_params,
            field=args.field,
            jacrev_chunk_size=args.jacrev_chunk_size,
            cov_type=args.cov_type,
            n_cov_seeds=args.n_cov_seeds,
            fd_step_kind=args.fd_step_kind,
            fd_relative_eps=args.fd_relative_eps,
        )
    elapsed = time.time() - t0
    h = int(elapsed // 3600)
    m = int(elapsed % 3600 // 60)
    s = int(elapsed % 60)
    print(f"Total Fisher computation time: {h}h {m}m {s}s ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

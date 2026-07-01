"""Bar-chart comparison of marginal sigmas across Fisher runs.

Verification step 1 of `docs/fisher_too_tight_fixes.md`: compare σ(Ω_m),
σ(σ_8), ... before and after the Grieb-covariance + n_seeds fixes.

Usage
-----
    python core/verification/plot_fisher_sigma_compare.py \
        old:outputs/fisher_bias_P_B_rsd_SN_legacy/fisher_results.npz \
        new:outputs/fisher_bias_P_B_rsd_SN/fisher_results.npz \
        --out outputs/verification/sigma_compare.png

Any number of `LABEL:PATH` positional args are accepted.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_LATEX = {
    "Omega_m": r"$\sigma(\Omega_m)$",
    "sigma8":  r"$\sigma(\sigma_8)$",
    "b1":      r"$\sigma(b_1)$",
    "b2":      r"$\sigma(b_2)$",
    "bs2":     r"$\sigma(b_{s^2})$",
    "bn2":     r"$\sigma(b_{\nabla^2})$",
}


def load_run(path):
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["param_names"]]
    fid = np.asarray(d["fid_values"], dtype=float)
    cov = np.asarray(d["fisher_cov"], dtype=float)        # data + prior
    cov_data = np.asarray(d["fisher_cov_data"], dtype=float)
    return {
        "names":     names,
        "fid":       fid,
        "sigmas":    np.sqrt(np.diag(cov)),
        "sigmas_data": np.sqrt(np.diag(cov_data)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+",
                    help="LABEL:PATH pairs to compare")
    ap.add_argument("--out", default="outputs/verification/sigma_compare.png")
    ap.add_argument("--data-only", action="store_true",
                    help="Plot sigmas before prior addition")
    args = ap.parse_args()

    runs = []
    for e in args.entries:
        if ":" not in e:
            raise SystemExit(f"Expected LABEL:PATH, got {e!r}")
        label, path = e.split(":", 1)
        runs.append((label, load_run(path)))

    names = runs[0][1]["names"]
    for label, r in runs[1:]:
        if r["names"] != names:
            raise SystemExit(f"Param name mismatch in {label}: "
                             f"{r['names']} vs {names}")

    sig_key = "sigmas_data" if args.data_only else "sigmas"
    n_p = len(names)
    n_r = len(runs)
    x = np.arange(n_p)
    width = 0.8 / n_r

    fig, ax = plt.subplots(figsize=(2 + 1.4 * n_p, 4.5))
    for i, (label, r) in enumerate(runs):
        ax.bar(x + (i - (n_r - 1) / 2) * width, r[sig_key], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([_LATEX.get(n, n) for n in names], rotation=0)
    ax.set_ylabel("1-sigma marginal" + (" (data only)" if args.data_only else ""))
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Fisher marginal sigmas — comparison")

    # Annotate ratios vs first run
    base = runs[0][1][sig_key]
    for i, (label, r) in enumerate(runs):
        for j, sig in enumerate(r[sig_key]):
            ratio = sig / base[j]
            ax.text(x[j] + (i - (n_r - 1) / 2) * width, sig,
                    f"{ratio:.2f}×", ha="center", va="bottom",
                    fontsize=7, rotation=90)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")

    # Also print the table
    print("\nMarginal sigmas " + ("(data only)" if args.data_only else "(data+prior)"))
    hdr = ["param"] + [r[0] for r in runs]
    print("\t".join(hdr))
    for j, name in enumerate(names):
        row = [name] + [f"{r[1][sig_key][j]:.4g}" for r in runs]
        print("\t".join(row))


if __name__ == "__main__":
    main()

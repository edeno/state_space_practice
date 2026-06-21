"""Run the Laplace-EKF vs exact-Polya-Gamma coupling cross-check and plot it.

Sweeps coupling strength on identical simulated data, fits both estimators, and
saves a figure comparing their credible-interval coverage, posterior-mean bias,
interval width, and mutual disagreement vs coupling magnitude.

Run from the repo root (needs the `coupling` extra):

    uv run --extra coupling python experiments/coupling_ekf_vs_pg/run_crosscheck.py

Outputs (next to this script): crosscheck.png and crosscheck_table.txt.
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from state_space_practice.coupling_crosscheck import (  # noqa: E402
    aggregate,
    run_crosscheck,
)
from state_space_practice.coupling_model import CouplingModelParams  # noqa: E402

OUT_DIR = Path(__file__).parent


def base_params() -> CouplingModelParams:
    """3 neurons, 2 bands (band 0 couplable, band 1 control), ~5 Hz base rate."""
    return CouplingModelParams(
        osc_frequencies=jnp.array([6.0, 10.0]),
        osc_decay=jnp.array([0.99, 0.99]),
        process_noise_var=jnp.array([1 - 0.99**2, 1 - 0.99**2]),
        beta_real=jnp.array([[2.0, 0.0], [0.0, 0.0], [-2.0, 0.0]]),
        beta_imag=jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
        baseline=jnp.full((3,), float(np.log(0.05 / 0.95))),
        dt=1e-3,
    )


def main() -> None:
    scales = [0.05, 0.1, 0.2, 0.4, 1.0]
    records = run_crosscheck(
        base_params(),
        scales=scales,
        n_time=4000,
        n_replicates=4,
        pg_n_iter=400,
        pg_burn_in=200,
    )
    agg = aggregate(records)
    mags = sorted(agg)

    # --- table ---
    lines = [
        f"{'mag':>5} {'method':>4} {'cover95':>8} {'abs_bias':>9} "
        f"{'ci_width':>9} {'det_auc':>8} {'phaseMAE':>9} {'ekf-pg':>7}"
    ]
    for mag in mags:
        cell = agg[mag]
        for method in ("ekf", "pg"):
            s = cell[method]
            diff = f"{cell['ekf_pg_mean_maxdiff']:.3f}" if method == "ekf" else ""
            lines.append(
                f"{mag:>5.2f} {method:>4} {s['coverage95']:>8.2f} {s['abs_bias']:>9.3f} "
                f"{s['ci_width']:>9.3f} {s['detection_auc']:>8.2f} {s['phase_mae']:>9.3f} {diff:>7}"
            )
    table = "\n".join(lines)
    (OUT_DIR / "crosscheck_table.txt").write_text(table + "\n")
    print(table)

    # --- figure (Wong colorblind-safe palette) ---
    ekf_c, pg_c = "#0072B2", "#D55E00"
    metrics = [
        ("coverage95", "95% CI coverage", 0.95),
        ("abs_bias", "mean |bias| (coupled)", None),
        ("ci_width", "mean 95% CI width", None),
        ("detection_auc", "detection ROC AUC", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3), constrained_layout=True)
    for ax, (key, title, ref) in zip(axes, metrics):
        for method, color in (("ekf", ekf_c), ("pg", pg_c)):
            ax.plot(
                mags,
                [agg[m][method][key] for m in mags],
                "o-",
                color=color,
                label="Laplace-EKF" if method == "ekf" else "exact PG",
            )
        if ref is not None:
            ax.axhline(ref, ls="--", color="gray", lw=1, label="nominal")
        ax.set_xscale("log")
        ax.set_xlabel("coupling magnitude")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Laplace-EKF vs exact Polya-Gamma coupling posterior (identical data)",
        fontsize=11,
    )
    fig.savefig(OUT_DIR / "crosscheck.png", dpi=150)
    print(f"\nwrote {OUT_DIR / 'crosscheck.png'}")


if __name__ == "__main__":
    main()

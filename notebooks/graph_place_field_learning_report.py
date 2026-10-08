"""Render scientific figures and acceptance summaries from frozen validation data.

    uv run --no-sync python notebooks/graph_place_field_learning_report.py

Missing final reports are left pending. Pilot data cannot satisfy acceptance.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def require_final(report):
    seeds = report["protocol"]["seeds"]
    if seeds != list(range(100, 120)):
        raise ValueError("Acceptance requires frozen independent seeds 100--119.")


def reference_figure(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for seed in (0, 1, 2):
        rows = [r for r in report["records"] if r["seed"] == seed and r["q"] > 0]
        q = [r["q"] for r in rows]
        axes[0].semilogx(
            q,
            [r["sequential_evidence"] - r["dense_evidence"] for r in rows],
            "o-",
            label=f"Sequential, seed {seed}",
        )
        axes[0].semilogx(
            q,
            [r["joint_evidence"] - r["dense_evidence"] for r in rows],
            "k.",
            label="Joint Laplace" if seed == 0 else None,
        )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(
        xlabel="Coefficient variance q per transition",
        ylabel="Evidence error (nats)",
        title="Small spatial profiles versus dense oracle",
    )
    axes[0].legend(fontsize=8)
    orders = report["static_order_counterexample"]
    x = np.arange(2)
    axes[1].bar(
        x - 0.17, [r["sequential_evidence"] for r in orders], 0.34, label="Sequential"
    )
    axes[1].bar(
        x + 0.17, [r["joint_evidence"] for r in orders], 0.34, label="Joint Laplace"
    )
    axes[1].set(
        xticks=x,
        xticklabels=["Counts [8, 0]", "Counts [0, 8]"],
        ylabel="Log evidence (nats)",
        title="Static evidence must be order invariant",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(output / "reference_evidence.svg")
    plt.close(fig)


def recovery_summary(report, output):
    require_final(report)
    conditions = [
        case for case, cfg in report["protocol"]["cases"].items() if cfg["informative"]
    ]
    fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
    rows = []
    for i, case in enumerate(conditions):
        for offset, method, color in (
            (-0.15, "known", "#0072B2"),
            (0.15, "joint", "#D55E00"),
        ):
            selected = [
                r
                for r in report["records"]
                if r["case"] == case and r["method"] == method
            ]
            values = np.array([r["session_mean_field_ratio"] for r in selected])
            ax.scatter(
                np.full(len(values), i + offset),
                values,
                color=color,
                alpha=0.65,
                s=18,
                label=method if i == 0 else None,
            )
            ax.plot(
                [i + offset - 0.1, i + offset + 0.1],
                [np.median(values)] * 2,
                color="black",
                linewidth=2,
            )
            stats = report["summary"][case][method]
            rows.append(
                dict(
                    case=case,
                    method=method,
                    **stats,
                    field_ratio_quantiles=np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
                )
            )
    ax.axhspan(0.5, 2.0, color="grey", alpha=0.15, label="Median acceptance band")
    ax.axhline(1.0, color="black", linewidth=0.6)
    ax.set(
        xticks=np.arange(len(conditions)),
        xticklabels=[c.replace("_", " ") for c in conditions],
        ylabel="Estimated / generating field-increment variance",
        title="20 independent sessions per informative condition (3 neurons/session)",
    )
    ax.legend(fontsize=8)
    fig.savefig(output / "recovery.svg")
    plt.close(fig)
    gates = [r["informative_recovery_gate"] for r in rows]
    # Preserve controls and all distributions in the original report.
    controls = {
        case: stats
        for case, stats in report["summary"].items()
        if case not in conditions
    }
    return dict(gate=all(gates), conditions=rows, controls=controls)


def prediction_summary(report, output):
    require_final(report)
    cases = list(report["summary"])
    fig, ax = plt.subplots(figsize=(10, 4.3), constrained_layout=True)
    rows = []
    for offset, comparator, color in (
        (-0.2, "static", "#0072B2"),
        (0.0, "tuned", "#D55E00"),
        (0.2, "windowed", "#009E73"),
    ):
        stats = [report["summary"][c][comparator] for c in cases]
        means = np.array([s["mean_bits_gain"] for s in stats])
        ci = np.array([s["interval"] for s in stats]).T
        ax.errorbar(
            np.arange(len(cases)) + offset,
            means,
            yerr=np.vstack((means - ci[0], ci[1] - means)),
            fmt="o",
            capsize=3,
            color=color,
            label=comparator,
        )
    for case in cases:
        selected = [r for r in report["records"] if r["case"] == case]
        rows.append(
            dict(
                case=case,
                **report["summary"][case],
                mean_field_rmse=float(
                    np.mean([r["auto"]["field_log_rate_rmse"] for r in selected])
                ),
                mean_coverage_95=float(
                    np.mean(
                        [r["auto"]["observed_log_rate_95_coverage"] for r in selected]
                    )
                ),
                median_seconds=float(
                    np.median([r["diagnostics"]["seconds"] for r in selected])
                ),
            )
        )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axhline(
        -0.02,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        label="Non-inferiority margin (tuned; stable/static)",
    )
    ax.set(
        xticks=np.arange(len(cases)),
        xticklabels=[c.replace("_", " ") for c in cases],
        ylabel="Automatic fit gain (bits per test spike)",
        title="Retrospective held-out predictions: paired-session 95% intervals",
    )
    ax.legend(fontsize=8)
    fig.savefig(output / "prediction_gains.svg")
    plt.close(fig)
    return dict(
        gate=all(r["tuned_gate"] and r["static_gate"] for r in rows), conditions=rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/validation/graph-place-field-learning"),
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    reference = read(output / "joint_reference.json")
    matched = read(output / "matched.json")
    application = read(output / "application.json")
    optimizer = read(
        output.parent / "graph-place-field-estimation" / "optimizer_joint.json"
    )
    summary = dict(
        optimizer=None,
        matched=None,
        application=None,
        accepted=False,
        limitation="Synthetic retrospective reconstruction; neither biological identification nor exact Bayesian calibration.",
    )
    if reference:
        reference_figure(reference, output)
    if matched:
        summary["matched"] = recovery_summary(matched, output)
    if application:
        summary["application"] = prediction_summary(application, output)
    if optimizer:
        summary["optimizer"] = optimizer["summary"]["profile_lbfgs"]
    if matched and application and optimizer:
        summary["accepted"] = (
            summary["matched"]["gate"]
            and summary["application"]["gate"]
            and summary["optimizer"]["n_objective_gap_at_most_0_001"] == 60
            and summary["optimizer"]["n_gradient_converged"] == 60
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

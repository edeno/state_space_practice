"""Compare sequential-baseline and parallel joint Laplace benchmark runs.

Pairs ``<prefix>_T<n_time>_N<n_neurons>.{json,npz}`` files written by
``joint_laplace_parallel_benchmark.py`` in two directories, asserts that the
outputs agree, and prints a timing/memory table (markdown).

Usage::

    python scripts/benchmarks/compare_joint_laplace_outputs.py BASELINE_DIR PARALLEL_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def compare_outputs(baseline: Path, comparison: Path) -> None:
    """Assert evidence, smoothed means and (when present) fitted results agree."""
    a, b = np.load(baseline), np.load(comparison)
    np.testing.assert_allclose(b["log_evidence"], a["log_evidence"], rtol=1e-10)
    np.testing.assert_allclose(
        b["mean_first_neuron"], a["mean_first_neuron"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_array_equal(b["converged"], a["converged"])
    if "fit_log_likelihood" in a.files:
        np.testing.assert_allclose(
            b["fit_log_likelihood"], a["fit_log_likelihood"], rtol=0, atol=1e-6
        )
        # L-BFGS stops within its gradient tolerance; objectives that differ in
        # the last bit give slightly different drift scales at equal evidence.
        np.testing.assert_allclose(
            b["fit_drift_scale"], a["fit_drift_scale"], rtol=1e-3, atol=1e-12
        )


def main() -> None:
    baseline_dir, comparison_dir = map(Path, sys.argv[1:3])
    rows = []
    for baseline_json in sorted(baseline_dir.glob("*_T*_N*.json")):
        comparison_json = comparison_dir / baseline_json.name
        if not comparison_json.exists():
            continue
        compare_outputs(
            baseline_json.with_suffix(".npz"), comparison_json.with_suffix(".npz")
        )
        a = json.loads(baseline_json.read_text())
        b = json.loads(comparison_json.read_text())
        assert not a["parallel"] and b["parallel"], baseline_json.name
        seq, par = a["value_and_grad"]["min_s"], b["value_and_grad"]["min_s"]
        gib = [(r["peak_bytes_in_use"] or 0) / 2**30 for r in (a, b)]
        fit = ""
        if "fit" in a and "fit" in b:
            fit = f" fit {a['fit']['wall_s']:.0f} -> {b['fit']['wall_s']:.0f} s"
        rows.append(
            f"| {a['n_time']} | {a['n_neurons']} | {a['device_kind']} | "
            f"{seq:.3f} | {par:.3f} | {seq / par:.1f}x | {gib[0]:.2f} | "
            f"{gib[1]:.2f} |{fit}"
        )
    if not rows:
        raise SystemExit("No matching baseline/comparison pairs found.")
    print(
        "| T | N | device | seq value+grad s | par value+grad s | speedup "
        "| seq peak GiB | par peak GiB |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))
    print(f"\nAll {len(rows)} pairs agree within tolerance.")


if __name__ == "__main__":
    main()

"""Compare Newton budgets and actual SGD drift-scale fits with fixed references.

Run from the repository root with the pinned spatial/test extras installed::

    MPLCONFIGDIR=/private/tmp/ssp-matplotlib LOKY_MAX_CPU_COUNT=4 \
        uv run --no-sync python notebooks/graph_place_field_estimation_validation.py \
        --section newton --output-dir docs/validation/graph-place-field-estimation
    MPLCONFIGDIR=/private/tmp/ssp-matplotlib LOKY_MAX_CPU_COUNT=4 \
        uv run --no-sync python notebooks/graph_place_field_estimation_validation.py \
        --section sgd --output-dir docs/validation/graph-place-field-estimation

Newton comparisons reuse the independent analytic-field benchmark, with identical
data and hyperparameters at each budget. SGD fits reuse the scalar datasets from
the independent integral study. Its oracle and all other parameters are fixed:
this tests optimization of the implemented objective, not joint spatial learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from graph_place_field_validation import (  # noqa: E402
    CASES,
    Environment,
    infer_graph,
    metrics,
    simulate_session,
    software_versions,
)
from scipy.optimize import minimize_scalar  # noqa: E402

from state_space_practice.graph_place_field import GraphPlaceFieldModel  # noqa: E402


def newton_experiment(seeds: list[int]) -> dict:
    env = Environment.from_samples(np.linspace(0, 100, 2001)[:, None], bin_size=2.5)
    pipelines = dict(defaults=(None, 1e-3, True), fixed=(8, 1e-2, False))
    budgets = dict(defaults=(1, 5), fixed=(1, 3, 5, 10))
    # Warm each compiled rank/budget on separate pilot data before timing.
    pilot = simulate_session(env, "stable", 0)
    for name, (rank, q, defaults) in pipelines.items():
        for n in budgets[name]:
            infer_graph(env, pilot, rank, q, defaults=defaults, max_newton_iter=n)
    records = []
    for case in CASES:
        for seed in seeds:
            session = simulate_session(env, case, seed)
            for name, (rank, q, defaults) in pipelines.items():
                # Alternate order to limit systematic timing-order effects.
                order = budgets[name] if seed % 2 == 0 else budgets[name][::-1]
                for n in order:
                    start = perf_counter()
                    prediction = infer_graph(
                        env, session, rank, q, defaults=defaults, max_newton_iter=n
                    )
                    elapsed = perf_counter() - start
                    records.append(
                        dict(
                            case=case,
                            seed=seed,
                            pipeline=name,
                            max_newton_iter=n,
                            pipeline_seconds=elapsed,
                            **metrics(session, prediction),
                        )
                    )
            print(f"Newton comparison {case} seed {seed} complete", flush=True)
    summary = {}
    for name in pipelines:
        summary[name] = {}
        for case in CASES:
            summary[name][case] = {}
            for n in budgets[name]:
                rows = [
                    r
                    for r in records
                    if r["pipeline"] == name
                    and r["case"] == case
                    and r["max_newton_iter"] == n
                ]
                summary[name][case][str(n)] = {
                    key: float(np.mean([r[key] for r in rows]))
                    for key in (
                        "pipeline_seconds",
                        "test_log_score",
                        "field_log_rate_rmse",
                        "observed_log_rate_95_coverage",
                    )
                }
    return dict(
        software=software_versions(),
        protocol=dict(
            seeds=seeds,
            cases=CASES,
            budgets=budgets,
            data="Same 400-second blocked-holdout analytic-field sessions at every budget",
            fixed="rank 8, q 0.01, tau2 100, kappa2 1; no EM updates",
            defaults="Full rank, q 0.001, normal initial-mean/amplitude EM updates, 100-iteration cap",
            timing="Setup, fit, posterior prediction and validation score; excludes final-test metrics; compilation warmed on independent pilot seed 0; alternating budget order",
            limitation="Exploratory paired subset of the previous audit; no hyperparameter selection or causal forecasting",
        ),
        records=records,
        summary=summary,
    )


def sgd_experiment(reference_path: Path, seeds: list[int]) -> dict:
    reference = json.loads(reference_path.read_text())["q_profile"]
    source = {row["seed"]: row for row in reference["datasets"]}
    if not set(seeds) <= source.keys():
        raise ValueError(
            "SGD seeds must have independent reference profiles in the input report."
        )
    env = Environment.from_samples(np.linspace(0, 10, 201)[:, None], bin_size=2.0)
    model = GraphPlaceFieldModel(
        env,
        dt=reference["dt"],
        rank=1,
        update_drift_scale=True,
        update_amplitude=False,
        update_kappa2=False,
        update_init_mean=False,
        max_firing_rate_hz=1e50,
    )
    phi = float(model.basis.eigvecs[0, 0])
    model.tau2 = reference["prior_variance"] / phi**2
    model.init_mean = jnp.array([[reference["prior_mean"] / phi]])
    times = np.arange(reference["n_time"]) * model.dt
    trajectory = np.repeat(env.bin_centers[:1], len(times), axis=0)
    starts, budgets = (0.001, 0.03, 0.3), (200, 1000)
    log_grid = np.linspace(np.log(1e-4), np.log(1.0), 81)

    # Data are arguments, so all sessions share the compiled direct objective.
    @jax.jit
    def value_and_gradient(log_q, z, spikes, mask):
        def loss(x):
            return model._sgd_loss_fn(
                {"drift_scale": jnp.atleast_1d(jnp.exp(x) / phi**2)}, z, spikes, mask
            )

        return jax.value_and_grad(loss)(log_q)

    records, profiles = [], []
    for seed in seeds:
        row = source[seed]
        positions = trajectory.copy()
        positions[~np.asarray(row["valid"])] = 1e6
        counts = np.asarray(row["counts"])
        z, spikes, mask = model._design_and_spikes(times, positions, counts)

        def objective(log_q, z=z, spikes=spikes, mask=mask):
            return float(value_and_gradient(jnp.asarray(log_q), z, spikes, mask)[0])

        losses = np.array([objective(x) for x in log_grid])
        candidates = [
            (float(x), float(y)) for x, y in zip(log_grid, losses, strict=True)
        ]
        for i in range(1, len(log_grid) - 1):
            if losses[i] <= min(losses[i - 1], losses[i + 1]):
                result = minimize_scalar(
                    objective,
                    bounds=(log_grid[i - 1], log_grid[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-8},
                )
                candidates.append((float(result.x), float(result.fun)))
        best_log_q, best_loss = min(candidates, key=lambda p: p[1])
        exact = row["profile"]
        exact_q = max(exact, key=lambda p: p["reference_log_evidence"])["q"]
        approximate_on_exact_grid = [objective(np.log(p["q"])) for p in exact]
        profiles.append(
            dict(
                seed=seed,
                profile_q=np.exp(log_grid).tolist(),
                profile_log_evidence=(-losses).tolist(),
                refined_q=float(np.exp(best_log_q)),
                refined_log_evidence=-best_loss,
                boundary=best_log_q in (float(log_grid[0]), float(log_grid[-1])),
                exact_grid_q=exact_q,
                approximate_on_exact_grid_log_evidence=(
                    -np.asarray(approximate_on_exact_grid)
                ).tolist(),
                exact_grid_log_evidence=[p["reference_log_evidence"] for p in exact],
            )
        )
        for initial_q in starts:
            for n in budgets:
                model.drift_scale = jnp.array([initial_q / phi**2])
                start = perf_counter()
                history = model.fit_sgd(
                    times,
                    positions,
                    counts,
                    num_steps=n,
                    warm_start=False,
                    verbose=False,
                )
                elapsed = perf_counter() - start
                learned_q = float(model.drift_scale[0]) * phi**2
                loss, gradient = value_and_gradient(jnp.log(learned_q), z, spikes, mask)
                records.append(
                    dict(
                        seed=seed,
                        initial_q=initial_q,
                        steps=n,
                        learned_q=learned_q,
                        log_evidence=-float(loss),
                        first_log_evidence=history[0],
                        log_evidence_gap_to_profile=float(loss) - best_loss,
                        log_q_gradient=float(gradient),
                        fit_seconds=elapsed,
                        outside_profile_bounds=not (1e-4 <= learned_q <= 1.0),
                    )
                )
        print(f"SGD drift comparison seed {seed} complete", flush=True)
    summary = {}
    for n in budgets:
        rows = [r for r in records if r["steps"] == n]
        summary[str(n)] = dict(
            n_fits=len(rows),
            n_objective_gap_at_most_0_001=sum(
                r["log_evidence_gap_to_profile"] <= 0.001 for r in rows
            ),
            n_outside_profile_bounds=sum(r["outside_profile_bounds"] for r in rows),
            maximum_log_evidence_gap=max(
                r["log_evidence_gap_to_profile"] for r in rows
            ),
            median_log_evidence_gap=float(
                np.median([r["log_evidence_gap_to_profile"] for r in rows])
            ),
            median_fit_seconds=float(np.median([r["fit_seconds"] for r in rows])),
        )
    return dict(
        software=software_versions(),
        protocol=dict(
            reference_file=str(reference_path),
            seeds=seeds,
            initial_q=starts,
            steps=budgets,
            newton_steps=model.max_newton_iter,
            parameter_units="q is scalar log-rate variance per transition, phi^2 times coefficient-space drift_scale",
            fixed_parameters="Known initial mean/variance and kappa2; only per-neuron q is optimized",
            objective="Approximate masked forward-filter evidence, full-sequence Adam(0.01), shared SGD mixin",
            direct_search="81 log-spaced scales on [0.0001,1], all sampled local maxima refined; includes endpoints",
            optimization_check="Objective gap <=0.001 nat against direct search; outside-bound fits reported",
            timing="Descriptive wall-clock including first-call compilation; not a performance comparison",
            limitation="Short scalar, model-matched datasets with known nuisance parameters; no full spatial joint-learning claim",
        ),
        profiles=profiles,
        records=records,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("newton", "sgd"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs/validation/graph-place-field-math/results.json"),
    )
    args = parser.parse_args()
    seeds = (
        args.seeds
        if args.seeds is not None
        else ([10, 11] if args.section == "newton" else list(range(30, 50)))
    )
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        parser.error("Use distinct nonnegative seeds.")
    result = (
        newton_experiment(seeds)
        if args.section == "newton"
        else sgd_experiment(args.reference, seeds)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.section}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

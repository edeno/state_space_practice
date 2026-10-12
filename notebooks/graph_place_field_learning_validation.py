"""Frozen recovery and held-out prediction protocol for automatic graph fits.

Pilot seeds are 0/1. Final seeds are 100--119, with no changes to settings after
opening their results. Matched simulations test covariance-scale recovery;
analytic sessions test retrospective prediction under model mismatch.

Run both sections (spatial/test extras and x64 required)::

    uv run --no-sync python notebooks/graph_place_field_learning_validation.py \
        --section matched --output-dir docs/validation/graph-place-field-learning
    uv run --no-sync python notebooks/graph_place_field_learning_validation.py \
        --section application --output-dir docs/validation/graph-place-field-learning
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
    DT,
    Environment,
    Prediction,
    infer_windowed_map,
    metrics,
    pointwise_score,
    simulate_session,
    software_versions,
)

from state_space_practice.graph_place_field import GraphPlaceFieldModel  # noqa: E402

MATCHED_CASES = {
    "linear_static": dict(
        branch=False, n_time=2000, baseline=5.0, q=0.0, informative=False
    ),
    "linear_slow": dict(
        branch=False, n_time=2000, baseline=5.0, q=0.0005, informative=True
    ),
    "linear_fast": dict(
        branch=False, n_time=2000, baseline=5.0, q=0.002, informative=True
    ),
    "branching_fast": dict(
        branch=True, n_time=3000, baseline=5.0, q=0.002, informative=True
    ),
    "sparse_short": dict(
        branch=False, n_time=600, baseline=1.0, q=0.002, informative=False
    ),
}


def interval(values: list[float]) -> list[float]:
    rng = np.random.default_rng(20261008)
    array = np.asarray(values)
    means = array[rng.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def environment(branch: bool) -> Environment:
    if not branch:
        return Environment.from_samples(np.linspace(0, 10, 201)[:, None], bin_size=2.0)
    samples = np.vstack(
        (
            np.column_stack((np.linspace(-10, 10, 401), np.zeros(401))),
            np.column_stack((np.zeros(201), np.linspace(0, 10, 201))),
        )
    )
    return Environment.from_samples(samples, bin_size=2.0)


def matched_experiment(seeds: list[int]) -> dict:
    records = []
    for case, settings in MATCHED_CASES.items():
        env = environment(settings["branch"])
        rank = 8 if settings["branch"] else 4
        template = GraphPlaceFieldModel(env, DT, rank=rank, inference_method="joint")
        phi = np.asarray(template.basis.eigvecs)
        shape = np.asarray(template._spectral_shape_current())
        factor = shape.sum() / env.n_bins
        n_time = settings["n_time"]
        n_cells = 3
        times = np.arange(n_time) * DT
        means = np.zeros((n_cells, rank))
        for component in range(template.basis.n_components):
            row = int(np.flatnonzero(phi[:, component])[0])
            means[:, component] = (
                np.log(settings["baseline"] * np.array([0.7, 1.0, 1.3]))
                / phi[row, component]
            )
        true_q = settings["q"] * np.array([0.7, 1.0, 1.5])
        for seed in seeds:
            movement_rng, latent_rng, count_rng = [
                np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
            ]
            if settings["branch"]:
                phase = (np.arange(n_time) * 0.15) % 20
                radius = np.minimum(phase, 20 - phase)
                cycles = (np.arange(n_time) * 0.15 // 20).astype(int)
                arms = movement_rng.integers(0, 3, cycles.max() + 1)
                directions = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])[
                    arms[cycles]
                ]
                positions = directions * radius[:, None]
            else:
                positions = (
                    5
                    + 4.9
                    * np.sin(
                        np.arange(n_time) * 0.1 + movement_rng.uniform(0, 2 * np.pi)
                    )
                )[:, None]
            ids = np.asarray(
                env.bin_sequence(times, positions, dedup=False, outside_value=-1)
            )
            if np.any(ids < 0):
                raise RuntimeError("Matched movement left its environment.")
            states = np.empty((n_cells, n_time, rank))
            states[:, 0] = means + latent_rng.normal(size=(n_cells, rank)) * np.sqrt(
                shape
            )
            states[:, 1:] = states[:, 0, None] + np.cumsum(
                latent_rng.normal(size=(n_cells, n_time - 1, rank))
                * np.sqrt(true_q[:, None, None] * shape),
                axis=1,
            )
            eta = np.einsum("tr,ctr->tc", phi[ids], states)
            counts = count_rng.poisson(DT * np.exp(eta)).astype(float)
            observed = (np.floor(times).astype(int) + seed) % 10 != 7
            fit_positions = positions.copy()
            fit_positions[~observed] = 1e6
            fitting_counts = np.where(observed[:, None], counts, 0.0)
            for method in ("known", "joint"):
                model = GraphPlaceFieldModel(
                    env,
                    DT,
                    rank=rank,
                    update_drift_scale=True,
                    update_amplitude=method == "joint",
                    update_kappa2=method == "joint",
                    update_init_mean=method == "joint",
                    inference_method="joint",
                    max_firing_rate_hz=1e50,
                )
                if method == "known":
                    model.init_mean = jnp.asarray(means)
                start = perf_counter()
                model.fit_mle(
                    times, fit_positions, fitting_counts, warm_start=method != "known"
                )
                elapsed = perf_counter() - start
                field = np.asarray(model.smoother_mean) @ phi.T
                truth = states @ phi.T
                rows = times >= min(40.0, n_time * DT / 4)
                field_var = np.einsum(
                    "br,ctrs,bs->ctb", phi, np.asarray(model.smoother_cov), phi
                )
                rec = dict(
                    case=case,
                    seed=seed,
                    method=method,
                    rank=rank,
                    n_bins=env.n_bins,
                    true_q=true_q.tolist(),
                    learned_q=np.asarray(model.drift_scale).tolist(),
                    true_field_variance=(true_q * factor).tolist(),
                    learned_field_variance=model.field_drift_scale_.tolist(),
                    kappa2=model.kappa2,
                    tau2=model.tau2,
                    converged=model.converged_,
                    gradient_norm=model.optimizer_result_.gradient_norm,
                    seconds=elapsed,
                    field_rmse=float(
                        np.sqrt(np.mean((field[:, rows] - truth[:, rows]) ** 2))
                    ),
                    spikes=counts.sum(axis=0).tolist(),
                    coverage_90=float(
                        np.mean(
                            np.abs(field[:, rows] - truth[:, rows])
                            <= 1.6448536269514722
                            * np.sqrt(np.maximum(field_var[:, rows], 0))
                        )
                    ),
                    parameter_bound_hits=list(model.parameter_bound_hits_),
                    profiles=[
                        dict(
                            best_field_variance=p.scale,
                            flat=p.flat,
                            upper_bound=p.at_upper_bound,
                            within_two_nats=[
                                float(np.min(p.scales[p.losses <= p.loss + 2])),
                                float(np.max(p.scales[p.losses <= p.loss + 2])),
                            ],
                        )
                        for p in model.drift_profiles_
                    ],
                )
                if settings["q"]:
                    rec["session_mean_field_ratio"] = float(
                        np.mean(model.field_drift_scale_ / (true_q * factor))
                    )
                    rec["session_mean_coefficient_ratio"] = float(
                        np.mean(np.asarray(model.drift_scale) / true_q)
                    )
                records.append(rec)
            print(f"Matched {case} seed {seed} complete", flush=True)
    summary = {}
    for case, settings in MATCHED_CASES.items():
        summary[case] = {}
        for method in ("known", "joint"):
            rows = [r for r in records if r["case"] == case and r["method"] == method]
            summary[case][method] = dict(
                n_sessions=len(rows),
                n_converged=sum(r["converged"] for r in rows),
                mean_rmse=float(np.mean([r["field_rmse"] for r in rows])),
                mean_coverage_90=float(np.mean([r["coverage_90"] for r in rows])),
                zero_fraction=float(
                    np.mean([q == 0 for r in rows for q in r["learned_q"]])
                ),
                median_seconds=float(np.median([r["seconds"] for r in rows])),
                n_parameter_bound_hits=sum(
                    bool(r["parameter_bound_hits"]) for r in rows
                ),
            )
            if settings["q"]:
                ratios = [r["session_mean_field_ratio"] for r in rows]
                coefficient_ratios = [r["session_mean_coefficient_ratio"] for r in rows]
                median = float(np.median(ratios))
                summary[case][method].update(
                    median_field_ratio=median,
                    mean_field_ratio_interval=interval(ratios),
                    median_coefficient_ratio=float(np.median(coefficient_ratios)),
                    informative_recovery_gate=(0.5 <= median <= 2)
                    if settings["informative"]
                    else None,
                )
    return dict(
        software=software_versions(),
        protocol=dict(
            seeds=seeds,
            cases=MATCHED_CASES,
            dt=DT,
            n_neurons=3,
            prior_tau2=1.0,
            prior_kappa2=1.0,
            observation_mask="One-second blocks, 10% unobserved; full transition grid",
            target="Known shape: coefficient q; joint shape: mean field increment variance. Raw q and kappa reported separately.",
            optimizer="Public profile-initialized L-BFGS, component baseline means, field-variance coordinates, package defaults",
            generating_model="Every retained mode, including null modes, has Gaussian prior/random-walk increments; independent Poisson counts",
            criterion="Median session mean variance ratio 0.5--2 in informative positive cases; sparse/short reported without recovery gate",
        ),
        records=records,
        summary=summary,
    )


def application_prediction(
    env: Environment, session, q: float | None
) -> tuple[Prediction, dict]:
    model = GraphPlaceFieldModel(
        env,
        DT,
        rank=8,
        init_drift_scale=0.001 if q is None else q,
        update_drift_scale=q is None,
        inference_method="joint",
    )
    positions = session.trajectory.copy()
    positions[~session.train] = 1e6
    counts = np.where(session.train, session.spikes, 0.0)
    start = perf_counter()
    model.fit_mle(session.times, positions, counts)
    elapsed = perf_counter() - start
    phi = np.asarray(model.basis.eigvecs)[session.bin_ids]
    variance = np.einsum("tr,trs,ts->t", phi, np.asarray(model.smoother_cov[0]), phi)
    prediction = Prediction(
        model.predict_log_rate_trajectory(),
        variance,
        dict(
            q=float(model.drift_scale[0]),
            field_variance=float(model.field_drift_scale_[0]),
            kappa2=model.kappa2,
            tau2=model.tau2,
            method="automatic" if q is None else "fixed q",
        ),
        0.0,
    )
    prediction.validation_log_score = pointwise_score(
        session, prediction, session.validation
    )
    return prediction, dict(
        seconds=elapsed,
        converged=model.converged_,
        gradient_norm=model.optimizer_result_.gradient_norm,
    )


def application_experiment(seeds: list[int]) -> dict:
    env = Environment.from_samples(np.linspace(0, 100, 2001)[:, None], bin_size=2.5)
    records = []
    for case in CASES:
        for seed in seeds:
            session = simulate_session(env, case, seed)
            automatic, diagnostics = application_prediction(env, session, None)
            candidates = [
                application_prediction(env, session, q)[0]
                for q in (0.0, 0.001, 0.01, 0.1)
            ]
            selected = max(candidates, key=lambda p: p.validation_log_score)
            windows = [
                infer_windowed_map(env, session, t, s)
                for t in (10.0, 30.0)
                for s in (5.0, 10.0)
            ]
            windowed = max(windows, key=lambda p: p.validation_log_score)
            predictions = dict(
                auto=automatic, static=candidates[0], tuned=selected, windowed=windowed
            )
            result = {name: metrics(session, p) for name, p in predictions.items()}
            test_spikes = session.spikes[session.test & (session.times >= 40)].sum()
            for name in ("static", "tuned", "windowed"):
                result["auto"][f"bits_over_{name}"] = (
                    result["auto"]["test_log_score"] - result[name]["test_log_score"]
                ) / (max(test_spikes, 1) * np.log(2))
            records.append(
                dict(case=case, seed=seed, diagnostics=diagnostics, **result)
            )
            print(f"Application {case} seed {seed} complete", flush=True)
    summary = {}
    for case in CASES:
        rows = [r for r in records if r["case"] == case]
        summary[case] = dict(
            n_sessions=len(rows),
            n_converged=sum(r["diagnostics"]["converged"] for r in rows),
        )
        for comparator in ("static", "tuned", "windowed"):
            values = [r["auto"][f"bits_over_{comparator}"] for r in rows]
            bounds = interval(values)
            summary[case][comparator] = dict(
                mean_bits_gain=float(np.mean(values)), interval=bounds
            )
        summary[case]["tuned_gate"] = summary[case]["tuned"]["interval"][0] > -0.02
        summary[case]["static_gate"] = summary[case]["static"]["interval"][0] > (
            -0.02 if case == "stable" else 0.0
        )
    return dict(
        software=software_versions(),
        protocol=dict(
            seeds=seeds,
            cases=CASES,
            rank=8,
            dt=DT,
            n_time=4000,
            nuisance_policy="Both automatic and fixed-q candidates fit kappa2/tau2 and component baseline prior means",
            fixed_q_grid=(0.0, 0.001, 0.01, 0.1),
            selection="Fixed q and window widths selected on validation only",
            task="Retrospective missing-block predictions, marginal Poisson-lognormal scores, 60/20/20 split, 40s burn-in",
            criterion="Lower paired-session 95% interval >-0.02 bits/spike vs tuned; stable vs static >-0.02, moving vs static >0",
            optimizer="Public profile-initialized L-BFGS and joint Laplace inference, package defaults",
        ),
        records=records,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("matched", "application"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 120)))
    args = parser.parse_args()
    if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct nonnegative seeds.")
    result = (
        matched_experiment(args.seeds)
        if args.section == "matched"
        else application_experiment(args.seeds)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.section}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

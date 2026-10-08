"""Independent, blocked-holdout validation of graph place-field inference.

Run from the repository root (the spatial and test extras must be installed)::

    MPLCONFIGDIR=/private/tmp/ssp-matplotlib LOKY_MAX_CPU_COUNT=4 \
        uv run --no-sync python notebooks/graph_place_field_validation.py \
        --output-dir /private/tmp/graph-place-field-validation

The generator never uses a graph eigenbasis or spectral random walk. It moves an
analytic Gaussian firing field along a linear track, with continuous animal
movement, pauses, low rates, and stable / smooth / abrupt-remapping conditions.
Whole one-second blocks are split 60/20/20 into fitting, validation, and test data.
Both counts and positions in held-out blocks are excluded from fitting; the full
time grid is retained. Rank / drift and the static penalty are selected using
validation counts only. Final test counts are opened once after selection.

This measures retrospective interpolation of missing observations. A smoother
can use neighboring training observations on either side of a held-out block;
this is deliberately NOT a claim about causal forecasting or real recordings.
Pointwise Poisson-lognormal scores integrate posterior uncertainty with Gaussian
quadrature. They are marginal scores, not a joint held-out likelihood. Sessions
are the independent units for uncertainty summaries, not individual time bins.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from neurospatial import Environment  # noqa: E402
from numpy.polynomial.hermite import hermgauss  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402

from state_space_practice.graph_place_field import (  # noqa: E402
    GraphPlaceFieldModel,
    build_graph_basis,
    fit_static_graph_glm,
    spectral_precision,
)

DT = 0.1
N_TIME = 4000
BURN_IN_SECONDS = 40.0
RANKS = (8, 16)
DRIFT_SCALES = (1e-10, 1e-3, 1e-2, 1e-1)
STATIC_AMPLITUDES = (1.0, 100.0)
MAP_TIME_WIDTHS = (10.0, 30.0)
MAP_SPACE_WIDTHS = (5.0, 10.0)
CASES = ("stable", "smooth", "remap", "sparse")
_GH_X, _GH_W = hermgauss(256)


@dataclass
class Session:
    times: np.ndarray
    trajectory: np.ndarray
    spikes: np.ndarray
    bin_ids: np.ndarray
    truth: np.ndarray
    observation_truth: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass
class Prediction:
    mean: np.ndarray  # full log-rate field, (time, bin)
    observed_variance: np.ndarray  # at the actual animal position
    config: dict
    validation_log_score: float


def simulate_session(env: Environment, case: str, seed: int) -> Session:
    """Analytic rate truth independent of the fitted model's parameterization."""
    movement_rng, spike_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)
    ]
    times = np.arange(N_TIME) * DT
    speed = np.empty(N_TIME)
    speed[0] = 18.0
    for t in range(1, N_TIME):
        speed[t] = np.clip(
            0.95 * speed[t - 1] + 0.05 * 18 + movement_rng.normal(0, 1), 8, 30
        )
    speed[(times % 30) < 2] = 0.0  # repeated pauses make occupancy nonuniform
    phase = (movement_rng.uniform(0, 200) + np.cumsum(speed) * DT) % 200
    position = np.minimum(phase, 200 - phase)  # reflected laps, no teleportation
    trajectory = position[:, None]
    if case == "stable":
        center = np.full(N_TIME, 50.0)
    elif case == "remap":
        center = np.where(times < N_TIME * DT / 2, 30.0, 70.0)
    else:
        center = 50 + 25 * np.sin(2 * np.pi * times / 180)
    peak = 3.0 if case == "sparse" else 12.0
    background = 0.2 if case == "sparse" else 0.5

    def log_rate(x: np.ndarray) -> np.ndarray:
        return np.log(
            background + peak * np.exp(-0.5 * ((x - center[:, None]) / 12) ** 2)
        )

    truth = log_rate(np.asarray(env.bin_centers)[:, 0][None, :])
    observation_truth = log_rate(position[:, None])[:, 0]
    spikes = spike_rng.poisson(DT * np.exp(observation_truth)).astype(float)
    bin_ids = np.asarray(
        env.bin_sequence(times, trajectory, dedup=False, outside_value=-1)
    )
    if np.any(bin_ids < 0):
        raise RuntimeError("Simulation left the fixed track environment.")
    # Assignment is determined without looking at counts or field truth. All rows
    # in a one-second block share an assignment; inference must bridge missing blocks.
    block_role = (np.floor(times).astype(int) + seed) % 5
    train = np.isin(block_role, [0, 1, 3])
    return Session(
        times,
        trajectory,
        spikes,
        bin_ids,
        truth,
        observation_truth,
        train,
        block_role == 2,
        block_role == 4,
    )


def poisson_lognormal_log_score(
    counts: np.ndarray, mean: np.ndarray, variance: np.ndarray
) -> np.ndarray:
    """Log E[Poisson(y; dt * exp(eta))], eta ~ Normal(mean, variance)."""
    log_mu = (
        np.log(DT)
        + mean[:, None]
        + np.sqrt(2 * np.maximum(variance, 0))[:, None] * _GH_X
    )
    integrand = counts[:, None] * log_mu - np.exp(log_mu) - gammaln(counts[:, None] + 1)
    return logsumexp(integrand + np.log(_GH_W), axis=1) - 0.5 * np.log(np.pi)


def pointwise_score(
    session: Session, prediction: Prediction, rows: np.ndarray
) -> float:
    idx = np.flatnonzero(rows & (session.times >= BURN_IN_SECONDS))
    mean = prediction.mean[idx, session.bin_ids[idx]]
    return float(
        poisson_lognormal_log_score(
            session.spikes[idx], mean, prediction.observed_variance[idx]
        ).sum()
    )


def check_predictive_scorer() -> dict:
    """Compare quadrature scores with an independent adaptive integrator."""
    errors = []
    for count in (0, 1, 3, 5):
        for rate in (1.0, 10.0, 30.0):
            for variance in (0.01, 0.2, 1.0, 3.0):
                mean = np.log(rate)
                actual = float(
                    poisson_lognormal_log_score(
                        np.array([count]), np.array([mean]), np.array([variance])
                    )[0]
                )

                def integrand(
                    z: float,
                    count: int = count,
                    mean: float = mean,
                    variance: float = variance,
                ) -> float:
                    log_mu = np.log(DT) + mean + np.sqrt(variance) * z
                    return float(
                        np.exp(
                            count * log_mu
                            - np.exp(log_mu)
                            - gammaln(count + 1)
                            - 0.5 * z * z
                        )
                        / np.sqrt(2 * np.pi)
                    )

                expected = np.log(quad(integrand, -12, 12, epsabs=1e-12)[0])
                errors.append(abs(actual - expected))
    maximum_error = float(max(errors))
    if maximum_error >= 1e-5:
        raise RuntimeError(
            f"Predictive quadrature is inaccurate: error={maximum_error}."
        )
    # Zero variance must reproduce the Poisson density used by the map baseline.
    actual = poisson_lognormal_log_score(
        np.arange(6), np.full(6, np.log(10.0)), np.zeros(6)
    )
    np.testing.assert_allclose(actual, -1 - gammaln(np.arange(6) + 1), atol=1e-12)
    return dict(
        adaptive_integration_cases=len(errors),
        maximum_log_score_error=maximum_error,
        zero_variance_poisson_limit=True,
    )


def software_versions() -> dict:
    return dict(
        python=python_version(),
        jax=jax.__version__,
        numpy=np.__version__,
        scipy=version("scipy"),
        neurospatial=version("neurospatial"),
    )


def infer_graph(
    env: Environment,
    session: Session,
    rank: int | None,
    q: float,
    *,
    defaults: bool = False,
    max_em_iter: int = 100,
    max_newton_iter: int | None = None,
) -> Prediction:
    settings = (
        {}
        if defaults
        else dict(
            tau2=100.0,
            update_amplitude=False,
            update_init_mean=False,
            update_kappa2=False,
        )
    )
    if max_newton_iter is not None:
        settings["max_newton_iter"] = max_newton_iter
    model = GraphPlaceFieldModel(
        env,
        dt=DT,
        rank=rank,
        init_drift_scale=q,
        inference_method="sequential",
        update_drift_scale=False,
        **settings,
    )
    masked_trajectory = session.trajectory.copy()
    masked_trajectory[~session.train] = 1e6
    # Zero held-out counts as well as masking positions: even an accidental helper
    # that ignores the position mask cannot consume the evaluation spikes.
    fitting_spikes = np.where(session.train, session.spikes, 0.0)
    model.fit_em(
        session.times,
        masked_trajectory,
        fitting_spikes,
        max_iter=max_em_iter if defaults else 1,
        verbose=False,
    )
    phi_observed = np.asarray(model.basis.eigvecs)[session.bin_ids]
    covariance = np.asarray(model.smoother_cov[0])
    variance = np.einsum("tr,trs,ts->t", phi_observed, covariance, phi_observed)
    prediction = Prediction(
        model.predict_log_rate_trajectory(),
        variance,
        dict(
            rank=model.rank,
            q=q,
            max_newton_iter=model.max_newton_iter,
            tau2=model.tau2,
            fitting="default EM" if defaults else "fixed parameters",
            em_iteration_budget=max_em_iter if defaults else 1,
            accepted_e_steps=len(model.log_likelihoods),
            converged=model.converged_,
        ),
        0.0,
    )
    prediction.validation_log_score = pointwise_score(
        session, prediction, session.validation
    )
    return prediction


def infer_static(
    env: Environment, session: Session, rank: int, tau2: float
) -> Prediction:
    basis = build_graph_basis(env, rank=rank)
    counts = np.bincount(
        session.bin_ids[session.train],
        weights=session.spikes[session.train],
        minlength=env.n_bins,
    )
    occupancy = np.bincount(session.bin_ids[session.train], minlength=env.n_bins) * DT
    weights, covariance = fit_static_graph_glm(
        counts,
        occupancy,
        basis.eigvecs,
        spectral_precision(basis.eigvals, tau2, 1.0),
        max_iter=50,
    )
    phi = np.asarray(basis.eigvecs)
    variance = np.einsum("br,rs,bs->b", phi, np.asarray(covariance), phi)[
        session.bin_ids
    ]
    prediction = Prediction(
        np.broadcast_to(phi @ np.asarray(weights), session.truth.shape),
        variance,
        dict(rank=rank, tau2=tau2),
        0.0,
    )
    prediction.validation_log_score = pointwise_score(
        session, prediction, session.validation
    )
    return prediction


def infer_windowed_map(
    env: Environment, session: Session, time_width: float, space_width: float
) -> Prediction:
    """Simple competing method: smooth training counts / training exposure.

    Ten-second histogram cells are smoothed in time and space. A weak fixed
    0.1-second pseudo-exposure shrinks empty neighborhoods to the training mean
    rate. This is an empirical rate estimate with a plug-in Poisson predictive
    density, not a posterior uncertainty model. All widths are selected on the
    same validation blocks as the graph models.
    """
    time_bin_width = 10.0
    time_centers = np.arange(time_bin_width / 2, N_TIME * DT, time_bin_width)
    time_ids = (session.times[session.train] / time_bin_width).astype(int)
    shape = (len(time_centers), env.n_bins)
    counts, exposure = np.zeros(shape), np.zeros(shape)
    np.add.at(
        counts,
        (time_ids, session.bin_ids[session.train]),
        session.spikes[session.train],
    )
    np.add.at(exposure, (time_ids, session.bin_ids[session.train]), DT)
    sigma = (time_width / time_bin_width, space_width / 2.5)
    smoothed_counts = gaussian_filter(counts, sigma=sigma)
    smoothed_exposure = gaussian_filter(exposure, sigma=sigma)
    global_rate = session.spikes[session.train].sum() / (session.train.sum() * DT)
    # Scale the pseudo-exposure to the normalized Gaussian kernel's weights.
    prior_exposure = 0.1 / (2 * np.pi * sigma[0] * sigma[1])
    rate = (smoothed_counts + prior_exposure * global_rate) / (
        smoothed_exposure + prior_exposure
    )
    interpolated = np.stack(
        [np.interp(session.times, time_centers, rate[:, b]) for b in range(env.n_bins)],
        axis=1,
    )
    prediction = Prediction(
        np.log(np.maximum(interpolated, 1e-12)),
        np.zeros(N_TIME),
        dict(
            time_width_seconds=time_width,
            space_width_cm=space_width,
            predictive_density="plug-in Poisson",
        ),
        0.0,
    )
    prediction.validation_log_score = pointwise_score(
        session, prediction, session.validation
    )
    return prediction


def metrics(session: Session, prediction: Prediction) -> dict:
    rows = session.test & (session.times >= BURN_IN_SECONDS)
    mean = prediction.mean[rows]
    truth = session.truth[rows]
    error = mean - truth
    obs_mean = mean[np.arange(rows.sum()), session.bin_ids[rows]]
    obs_error = obs_mean - session.observation_truth[rows]
    binned_obs_error = obs_mean - truth[np.arange(rows.sum()), session.bin_ids[rows]]
    sd = np.sqrt(np.maximum(prediction.observed_variance[rows], 0))
    has_interval = prediction.config.get("predictive_density") != "plug-in Poisson"

    def spatial_drift(field: np.ndarray) -> np.ndarray:
        centered = field - field.mean(axis=1, keepdims=True)
        return centered - centered.mean(axis=0, keepdims=True)

    true_drift = spatial_drift(truth)
    estimated_drift = spatial_drift(mean)
    truth_rms = float(np.sqrt(np.mean(true_drift**2)))
    drift_error = float(np.sqrt(np.mean((estimated_drift - true_drift) ** 2)))
    return dict(
        config=prediction.config,
        validation_log_score=prediction.validation_log_score,
        test_log_score=pointwise_score(session, prediction, session.test),
        field_log_rate_rmse=float(np.sqrt(np.mean(error**2))),
        observed_log_rate_rmse=float(np.sqrt(np.mean(obs_error**2))),
        drift_nrmse=drift_error / truth_rms if truth_rms > 1e-10 else None,
        estimated_drift_rms=float(np.sqrt(np.mean(estimated_drift**2))),
        observed_log_rate_95_coverage=float(np.mean(np.abs(obs_error) <= 1.96 * sd))
        if has_interval
        else None,
        binned_log_rate_95_coverage=float(
            np.mean(np.abs(binned_obs_error) <= 1.96 * sd)
        )
        if has_interval
        else None,
        mean_log_rate_95_interval_width=float(np.mean(2 * 1.96 * sd))
        if has_interval
        else None,
    )


def evaluate(
    env: Environment, session: Session, *, max_newton_iter: int | None = None
) -> tuple[dict, dict[str, Prediction]]:
    graph = [
        infer_graph(env, session, rank, q, max_newton_iter=max_newton_iter)
        for rank in RANKS
        for q in DRIFT_SCALES
    ]
    static = [
        infer_static(env, session, rank, tau)
        for rank in RANKS
        for tau in STATIC_AMPLITUDES
    ]
    windowed = [
        infer_windowed_map(env, session, time_width, space_width)
        for time_width in MAP_TIME_WIDTHS
        for space_width in MAP_SPACE_WIDTHS
    ]
    selected = max(graph, key=lambda p: p.validation_log_score)
    selected_static = max(static, key=lambda p: p.validation_log_score)
    selected_windowed = max(windowed, key=lambda p: p.validation_log_score)
    defaults = infer_graph(
        env, session, None, 1e-3, defaults=True, max_newton_iter=max_newton_iter
    )
    predictions = dict(
        selected=selected,
        defaults=defaults,
        static=selected_static,
        windowed=selected_windowed,
    )
    # Only after all selections are complete do we evaluate final test counts.
    result = {
        name: metrics(session, prediction) for name, prediction in predictions.items()
    }
    test_rows = session.test & (session.times >= BURN_IN_SECONDS)
    test_spikes = int(session.spikes[test_rows].sum())
    for name in ("selected", "defaults"):
        for baseline in ("static", "windowed"):
            result[name][f"test_bits_per_spike_over_{baseline}"] = (
                result[name]["test_log_score"] - result[baseline]["test_log_score"]
            ) / (max(test_spikes, 1) * np.log(2))
    # The oracle is a reference score at the known continuous-position rate, never
    # an input to fitting or selection. It need not win on every realized draw.
    expected = DT * np.exp(session.observation_truth[test_rows])
    result["oracle_test_log_score"] = float(
        np.sum(
            session.spikes[test_rows] * np.log(expected)
            - expected
            - gammaln(session.spikes[test_rows] + 1)
        )
    )
    result["test_spikes"] = test_spikes
    result["total_spikes"] = int(session.spikes.sum())
    result["train_visited_bins"] = int(np.unique(session.bin_ids[session.train]).size)
    result["graph_validation_grid"] = [
        dict(**p.config, validation_log_score=p.validation_log_score) for p in graph
    ]
    result["static_validation_grid"] = [
        dict(**p.config, validation_log_score=p.validation_log_score) for p in static
    ]
    result["windowed_validation_grid"] = [
        dict(**p.config, validation_log_score=p.validation_log_score) for p in windowed
    ]
    return result, predictions


def summarize(records: list[dict]) -> dict:
    """Paired descriptive uncertainty: resample independent sessions, not rows."""
    rng = np.random.default_rng(20261006)
    output = {}
    for case in CASES:
        rows = [r for r in records if r["case"] == case]
        if not rows:
            continue
        output[case] = {}
        for model in ("selected", "defaults"):
            output[case][model] = dict(
                n_sessions=len(rows),
                mean_field_rmse=float(
                    np.mean([r[model]["field_log_rate_rmse"] for r in rows])
                ),
                mean_observed_95_coverage=float(
                    np.mean([r[model]["observed_log_rate_95_coverage"] for r in rows])
                ),
                mean_estimated_drift_rms=float(
                    np.mean([r[model]["estimated_drift_rms"] for r in rows])
                ),
            )
            if model == "defaults":
                output[case][model]["converged_sessions"] = sum(
                    r[model]["config"]["converged"] for r in rows
                )
                output[case][model]["mean_accepted_e_steps"] = float(
                    np.mean([r[model]["config"]["accepted_e_steps"] for r in rows])
                )
            for baseline in ("static", "windowed"):
                gains = np.array(
                    [r[model][f"test_bits_per_spike_over_{baseline}"] for r in rows]
                )
                boot = rng.choice(gains, size=(10000, len(gains)), replace=True).mean(
                    axis=1
                )
                output[case][model][f"mean_test_bits_per_spike_over_{baseline}"] = (
                    float(gains.mean())
                )
                output[case][model][
                    f"descriptive_bootstrap_95_interval_over_{baseline}"
                ] = np.quantile(boot, [0.025, 0.975]).tolist()
                output[case][model][f"sessions_beating_{baseline}"] = int(
                    (gains > 0).sum()
                )
        output[case]["static_mean_field_rmse"] = float(
            np.mean([r["static"]["field_log_rate_rmse"] for r in rows])
        )
        output[case]["windowed_mean_field_rmse"] = float(
            np.mean([r["windowed"]["field_log_rate_rmse"] for r in rows])
        )
    return output


def plot_examples(examples: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(examples),
        4,
        figsize=(15, 3 * len(examples)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, (case, (session, predictions)) in enumerate(examples.items()):
        for column, (name, field) in enumerate(
            (
                ("truth", session.truth),
                ("selected", predictions["selected"].mean),
                ("static", predictions["static"].mean),
                ("windowed map", predictions["windowed"].mean),
            )
        ):
            ax = axes[row, column]
            im = ax.imshow(
                np.exp(field).T,
                origin="lower",
                aspect="auto",
                extent=(0, N_TIME * DT, 0, 100),
                vmin=0,
                vmax=12.5,
                cmap="magma",
            )
            ax.set_title(f"{case}: {name}")
            if column == 0:
                ax.set_ylabel("Position (cm)")
            ax.set_xlabel("Time (s)")
    fig.subplots_adjust(hspace=0.42, wspace=0.22)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Firing rate (Hz)", fraction=0.025)
    fig.suptitle(
        "Independent analytic fields; selected model uses training blocks only"
    )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument(
        "--max-newton-iter",
        type=int,
        help="Override the constructor's Newton budget (use 1 to reproduce the original audit)",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers.")
    if len(set(args.cases)) != len(args.cases):
        parser.error("Cases must be distinct.")
    if args.max_newton_iter is not None and args.max_newton_iter < 1:
        parser.error("Newton budget must be positive.")
    scorer_checks = check_predictive_scorer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = Environment.from_samples(np.linspace(0, 100, 2001)[:, None], bin_size=2.5)
    records, examples = [], {}
    start = perf_counter()
    for case in args.cases:
        for seed in args.seeds:
            session = simulate_session(env, case, seed)
            result, predictions = evaluate(
                env, session, max_newton_iter=args.max_newton_iter
            )
            records.append(dict(case=case, seed=seed, **result))
            if seed == args.seeds[0]:
                examples[case] = session, predictions
            print(
                f"{case} seed={seed}: selected={result['selected']['test_bits_per_spike_over_static']:+.3f}, defaults={result['defaults']['test_bits_per_spike_over_static']:+.3f} bits/spike over static",
                flush=True,
            )
    report = dict(
        software=software_versions(),
        scorer_checks=scorer_checks,
        protocol=dict(
            dt=DT,
            n_time=N_TIME,
            burn_in_seconds=BURN_IN_SECONDS,
            n_bins=env.n_bins,
            ranks=RANKS,
            drift_scales=DRIFT_SCALES,
            static_amplitudes=STATIC_AMPLITUDES,
            map_time_widths_seconds=MAP_TIME_WIDTHS,
            map_space_widths_cm=MAP_SPACE_WIDTHS,
            cases=args.cases,
            seeds=args.seeds,
            split="one-second blocks, 60% train / 20% validation / 20% test",
            task="retrospective interpolation, pointwise Poisson-lognormal predictive score",
            selection="validation counts only; no truth or final test counts",
            quadrature_nodes=256,
            default_em_iteration_budget=100,
            newton_iteration_override=args.max_newton_iter,
            uncertainty_coverage_models=("selected", "defaults", "static"),
            limitation="exploratory synthetic sessions on a linear track; bootstrap intervals are descriptive",
        ),
        summary=summarize(records),
        records=records,
        runtime_seconds=perf_counter() - start,
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot_examples(examples, args.output_dir / "fields.png")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

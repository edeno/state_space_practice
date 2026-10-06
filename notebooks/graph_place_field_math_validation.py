"""Quantify graph Poisson inference approximation against a converged grid oracle.

Run from the repository root with spatial/test extras and x64 enabled::

    MPLCONFIGDIR=/private/tmp/ssp-matplotlib LOKY_MAX_CPU_COUNT=4 \
      uv run --no-sync python notebooks/graph_place_field_math_validation.py \
      --output-dir docs/validation/graph-place-field-math

The reference uses NumPy/SciPy Bayes integration and is independently checked in
test_graph_place_field_math.py against adaptive integration and 2D Gauss-Hermite
integration in noise coordinates. Every result here must also pass grid refinement
and domain expansion. Fixed true parameters isolate approximation error; no EM
fitting, hyperparameter selection, or inference helper is used in the reference.

The one-mode model has kappa2=1 (null-mode S=1) and a constant graph eigenvector phi.
Reported scalar states
are log rates eta=phi*w, so tau_eta^2=phi^2*tau_w^2 and q_eta=phi^2*q_w. The
likelihood-profile experiment uses multiple model-matched datasets, rather than
requiring a single realization's evidence to peak at its generating q. It is
diagnostic evidence about this scalar model, not a general identifiability proof.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from neurospatial import Environment  # noqa: E402

from state_space_practice.graph_place_field import GraphPlaceFieldModel  # noqa: E402
from state_space_practice.tests.graph_math_reference import (  # noqa: E402
    GridPosterior,
    interval_mass,
    poisson_grid_posterior,
)

PROFILE_Q = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
TRUE_Q = 0.03
NEWTON_STEPS = (1, 25)


@dataclass
class Problem:
    mean: float
    variance: float
    q: float
    dt: float
    counts: np.ndarray
    valid: np.ndarray


@dataclass
class Approximation:
    filtered_mean: np.ndarray
    filtered_variance: np.ndarray
    smoothed_mean: np.ndarray
    smoothed_variance: np.ndarray
    lag_covariance: np.ndarray
    log_evidence: float


def converged_reference(p: Problem) -> tuple[GridPosterior, dict]:
    """Refuse an unconverged reference instead of forgiving its errors."""
    settings = dict(
        initial_mean=p.mean,
        initial_variance=p.variance,
        process_variance=p.q,
        design=np.ones(len(p.counts)),
        counts=p.counts,
        valid=p.valid,
        dt=p.dt,
    )
    coarse = poisson_grid_posterior(**settings, n_grid=2001, radius_sd=20)
    fine = poisson_grid_posterior(**settings, n_grid=4001, radius_sd=20)
    expanded = poisson_grid_posterior(**settings, n_grid=4801, radius_sd=24)
    longer_tails = poisson_grid_posterior(
        **settings, n_grid=4001, radius_sd=20, transition_radius_sd=16
    )
    fields = (
        "filtered_mean",
        "filtered_variance",
        "smoothed_mean",
        "smoothed_variance",
        "lag_covariance",
    )
    maximum_moment_error, maximum_evidence_error = 0.0, 0.0
    for other in (coarse, expanded, longer_tails):
        for field in fields:
            delta = np.abs(getattr(other, field) - getattr(fine, field))
            maximum_moment_error = max(
                maximum_moment_error, float(delta.max()) if delta.size else 0.0
            )
        maximum_evidence_error = max(
            maximum_evidence_error, abs(other.log_evidence - fine.log_evidence)
        )
    if maximum_moment_error > 1e-6 or maximum_evidence_error > 1e-6:
        raise RuntimeError(
            f"Reference did not converge: moments={maximum_moment_error}, logZ={maximum_evidence_error}."
        )
    return fine, dict(
        grid_points=(2001, 4001, 4801),
        domain_radius_sd=(20, 20, 24),
        transition_radius_sd=(12, 16),
        maximum_moment_difference=maximum_moment_error,
        maximum_log_evidence_difference=maximum_evidence_error,
        maximum_transition_mass_error=fine.maximum_transition_mass_error,
        maximum_boundary_mass=fine.maximum_boundary_mass,
    )


def graph_approximation(env: Environment, p: Problem, iterations: int) -> Approximation:
    # A high numerical ceiling isolates the intended unclipped Poisson model.
    # This is a mathematical approximation study, not the application defaults.
    model = GraphPlaceFieldModel(
        env,
        dt=p.dt,
        rank=1,
        max_newton_iter=iterations,
        max_firing_rate_hz=1e50,
        update_amplitude=False,
        update_init_mean=False,
    )
    phi = float(model.basis.eigvecs[0, 0])
    model.tau2 = p.variance / phi**2
    model.init_mean = jnp.array([[p.mean / phi]])
    model.drift_scale = jnp.array([p.q / phi**2])
    times = np.arange(len(p.counts)) * p.dt
    trajectory = np.repeat(env.bin_centers[:1], len(times), axis=0)
    trajectory[~p.valid] = 1e6
    z, counts, mask = model._design_and_spikes(times, trajectory, p.counts)
    log_evidence = model._e_step(z, counts, mask)
    filtered_mean = np.asarray(model.filtered_mean[0, :, 0]) * phi
    if np.max(filtered_mean + np.log(p.dt)) >= model._max_log_count:
        raise RuntimeError(
            "Rate clipping was active; the unclipped reference cannot validate this run."
        )
    return Approximation(
        filtered_mean,
        np.asarray(model.filtered_cov[0, :, 0, 0]) * phi**2,
        np.asarray(model.smoother_mean[0, :, 0]) * phi,
        np.asarray(model.smoother_cov[0, :, 0, 0]) * phi**2,
        np.asarray(model.smoother_cross_cov[0, :, 0, 0]) * phi**2,
        log_evidence,
    )


def approximation_errors(reference: GridPosterior, approximate: Approximation) -> dict:
    sd = np.sqrt(approximate.smoothed_variance)
    mass = interval_mass(
        reference,
        approximate.smoothed_mean - 1.96 * sd,
        approximate.smoothed_mean + 1.96 * sd,
    )
    return dict(
        maximum_filtered_mean_error_in_reference_sd=float(
            np.max(
                np.abs(approximate.filtered_mean - reference.filtered_mean)
                / np.sqrt(reference.filtered_variance)
            )
        ),
        maximum_smoothed_mean_error_in_reference_sd=float(
            np.max(
                np.abs(approximate.smoothed_mean - reference.smoothed_mean)
                / np.sqrt(reference.smoothed_variance)
            )
        ),
        maximum_relative_smoothed_variance_error=float(
            np.max(
                np.abs(approximate.smoothed_variance / reference.smoothed_variance - 1)
            )
        ),
        maximum_lag_covariance_error=float(
            np.max(np.abs(approximate.lag_covariance - reference.lag_covariance))
        )
        if len(reference.lag_covariance)
        else 0.0,
        log_evidence_error=approximate.log_evidence - reference.log_evidence,
        mean_reference_mass_inside_nominal_95_interval=float(mass.mean()),
        minimum_reference_mass_inside_nominal_95_interval=float(mass.min()),
    )


def declared_cases() -> dict[str, Problem]:
    return dict(
        benign=Problem(
            np.log(25),
            0.03,
            0.006,
            0.02,
            np.array([1, 0, 1, 2, 0, 1], dtype=float),
            np.array([True, True, False, True, False, True]),
        ),
        sparse=Problem(
            np.log(2),
            0.7,
            0.06,
            0.1,
            np.array([0, 1, 0, 0, 2, 0, 1, 0], dtype=float),
            np.array([False, True, True, False, True, True, True, False]),
        ),
        surprising_count=Problem(0.0, 1.0, 0.0, 0.1, np.array([8.0]), np.array([True])),
    )


def profile_experiment(env: Environment, seeds: list[int]) -> dict:
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n_time, dt, variance, mean = 24, 0.1, 0.08, float(np.log(15))
        truth = np.empty(n_time)
        truth[0] = rng.normal(mean, np.sqrt(variance))
        truth[1:] = truth[0] + np.cumsum(rng.normal(0, np.sqrt(TRUE_Q), n_time - 1))
        counts = rng.poisson(dt * np.exp(truth)).astype(float)
        valid = np.ones(n_time, dtype=bool)
        valid[[3, 9, 10, 17]] = False
        profile = []
        for q in PROFILE_Q:
            p = Problem(mean, variance, q, dt, counts, valid)
            reference, convergence = converged_reference(p)
            approximations = {
                str(iterations): graph_approximation(env, p, iterations)
                for iterations in NEWTON_STEPS
            }
            profile.append(
                dict(
                    q=q,
                    reference_log_evidence=reference.log_evidence,
                    approximation_log_evidence={
                        key: value.log_evidence for key, value in approximations.items()
                    },
                    approximation_errors={
                        key: approximation_errors(reference, value)
                        for key, value in approximations.items()
                    },
                    convergence=convergence,
                )
            )
        results.append(
            dict(
                seed=seed,
                counts=counts.tolist(),
                truth_log_rate=truth.tolist(),
                valid=valid.tolist(),
                profile=profile,
            )
        )
        print(f"q profile seed {seed} complete", flush=True)
    exact = np.array(
        [
            [point["reference_log_evidence"] for point in row["profile"]]
            for row in results
        ]
    )
    approx = {
        str(n): np.array(
            [
                [
                    point["approximation_log_evidence"][str(n)]
                    for point in row["profile"]
                ]
                for row in results
            ]
        )
        for n in NEWTON_STEPS
    }
    summary = dict(
        q=PROFILE_Q,
        reference_mean_log_evidence=exact.mean(axis=0).tolist(),
        reference_grid_maximizer=PROFILE_Q[int(exact.mean(axis=0).argmax())],
        approximation_mean_log_evidence={
            key: value.mean(axis=0).tolist() for key, value in approx.items()
        },
        approximation_grid_maximizer={
            key: PROFILE_Q[int(value.mean(axis=0).argmax())]
            for key, value in approx.items()
        },
        mean_absolute_evidence_error={
            key: np.abs(value - exact).mean(axis=0).tolist()
            for key, value in approx.items()
        },
        mean_reference_mass_at_generating_q={
            str(n): float(
                np.mean(
                    [
                        point["approximation_errors"][str(n)][
                            "mean_reference_mass_inside_nominal_95_interval"
                        ]
                        for row in results
                        for point in row["profile"]
                        if point["q"] == TRUE_Q
                    ]
                )
            )
            for n in NEWTON_STEPS
        },
    )
    return dict(
        generating_q=TRUE_Q,
        prior_mean=mean,
        prior_variance=variance,
        dt=dt,
        n_time=n_time,
        n_datasets=len(seeds),
        summary=summary,
        datasets=results,
    )


def plot_report(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    summary = report["q_profile"]["summary"]
    q = np.array(summary["q"])
    exact = np.array(summary["reference_mean_log_evidence"])
    axes[0].semilogx(q, exact - exact.max(), marker="o", label="Grid reference")
    for steps in NEWTON_STEPS:
        values = np.array(summary["approximation_mean_log_evidence"][str(steps)])
        axes[0].semilogx(
            q,
            values - values.max(),
            marker="o",
            label=f"{steps} Newton step" + ("s" if steps > 1 else ""),
        )
    axes[0].axvline(TRUE_Q, color="black", linestyle=":", label="Generating q")
    axes[0].set(
        xlabel="Log-rate variance per step",
        ylabel="Mean log evidence minus method maximum",
        title="Evidence profile across independent datasets",
    )
    axes[0].legend()
    cases = list(report["cases"])
    for index, steps in enumerate(NEWTON_STEPS):
        mass = [
            report["cases"][case]["approximation_errors"][str(steps)][
                "mean_reference_mass_inside_nominal_95_interval"
            ]
            for case in cases
        ]
        axes[1].bar(
            np.arange(len(cases)) + index * 0.35,
            mass,
            width=0.35,
            label=f"{steps} Newton step" + ("s" if steps > 1 else ""),
        )
    axes[1].axhline(0.95, color="black", linestyle=":")
    axes[1].set_xticks(np.arange(len(cases)) + 0.175, cases)
    axes[1].set(
        ylabel="Grid-reference posterior mass",
        ylim=(0, 1.03),
        title="Mass inside nominal 95% intervals",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30, 50)))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Use distinct nonnegative seeds.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = Environment.from_samples(np.linspace(0, 10, 201)[:, None], bin_size=2.0)
    cases = {}
    for name, problem in declared_cases().items():
        reference, convergence = converged_reference(problem)
        errors = {
            str(n): approximation_errors(
                reference, graph_approximation(env, problem, n)
            )
            for n in NEWTON_STEPS
        }
        cases[name] = dict(
            problem=dict(
                initial_mean=problem.mean,
                initial_variance=problem.variance,
                process_variance=problem.q,
                dt=problem.dt,
                counts=problem.counts.tolist(),
                valid=problem.valid.tolist(),
            ),
            convergence=convergence,
            reference_log_evidence=reference.log_evidence,
            reference_smoothed_mean=reference.smoothed_mean.tolist(),
            reference_smoothed_variance=reference.smoothed_variance.tolist(),
            approximation_errors=errors,
        )
        print(name, json.dumps(errors), flush=True)
    profile = profile_experiment(env, args.seeds)
    report = dict(
        software=dict(
            python=python_version(),
            jax=jax.__version__,
            numpy=np.__version__,
            scipy=version("scipy"),
            neurospatial=version("neurospatial"),
        ),
        protocol=dict(
            reference="Uniform-grid Bayes filtering/backward integration using direct Gaussian convolution",
            grid_convergence_tolerance=1e-6,
            scalar_coordinate="eta=phi*w; q and prior variance are expressed in log-rate units",
            numerical_rate_guard_hz=1e50,
            parameters="Fixed known values, with no warm start from observed counts and no EM updates",
            newton_steps=NEWTON_STEPS,
            seeds=args.seeds,
            limitation="Scalar constant-mode examples; posterior integration is numerical, and evidence maxima are descriptive grid maxima",
        ),
        cases=cases,
        q_profile=profile,
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot_report(report, args.output_dir / "math.png")
    print(json.dumps(profile["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

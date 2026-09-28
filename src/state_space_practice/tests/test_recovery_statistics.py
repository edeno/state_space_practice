# ruff: noqa: E402
"""Parameter recovery as statistics: error must shrink with more data.

A single-seed "fit and compare against a threshold" test cannot distinguish a
consistent estimator from a lucky seed.  Each test here fits a model on
``len(SEEDS)`` independent simulations at a short and a long sequence length
(4-8x longer) and asserts that

* the mean parameter error over seeds drops by a clear factor with more data
  (a consistent estimator's error shrinks roughly as ``1/sqrt(T)``), and
* the error drops for a majority of the seeds individually,

reporting every per-seed error in the assertion message.  EM starts from a
mildly perturbed truth so the tests measure estimation error (not the basin
of attraction, which ``test_warm_init.py`` and ``test_scenario_recovery.py``
cover).

Errors are computed on identifiable quantities.  The oscillator latent state
is only defined up to a rotation of each oscillator's 2-d coordinates whenever
the loading onto it is free (COM's ``H``, every point-process model's spike
weights), so those comparisons use rotation-invariant summaries (per-block
Gram matrices, coupling magnitudes, eigenvalues).
"""

import jax

jax.config.update("jax_enable_x64", True)

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.tests import recovery_helpers as rh

SEEDS = (0, 1, 2)
STICKY_Z = jnp.array([[0.98, 0.02], [0.02, 0.98]])


def _relative_error(estimate, truth) -> float:
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def _assert_error_shrinks(errors: dict, label: str, factor: float) -> None:
    """``errors[n_time]`` is the per-seed error list for that length."""
    short_len, long_len = sorted(errors)
    short = np.asarray(errors[short_len])
    long = np.asarray(errors[long_len])
    table = (
        f"{label}: per-seed error at T={short_len}: {np.round(short, 4).tolist()}, "
        f"at T={long_len}: {np.round(long, 4).tolist()}"
    )
    assert np.all(np.isfinite(short)) and np.all(np.isfinite(long)), table
    assert long.mean() < factor * short.mean(), (
        f"mean error did not shrink by {factor}x with more data. " + table
    )
    assert np.sum(long < short) >= 2, "error grew for most seeds. " + table


# ---------------------------------------------------------------------------
# Gaussian oscillator models (COM / CNM / DIM)
# ---------------------------------------------------------------------------


def _com_error(model, truth) -> float:
    """Rotation-invariant H error: per-state, per-oscillator Gram blocks."""
    num = den = 0.0
    for j in range(2):
        for k in range(2):
            block = slice(2 * k, 2 * k + 2)
            est = np.asarray(model.measurement_matrix[:, block, j])
            ref = np.asarray(truth.measurement_matrix[:, block, j])
            num += np.sum((est @ est.T - ref @ ref.T) ** 2)
            den += np.sum((ref @ ref.T) ** 2)
    return float(np.sqrt(num / den))


def _perturb_gaussian(kind: str, model, rng: np.random.Generator) -> None:
    if kind == "COM":
        H = np.asarray(model.measurement_matrix)
        model.measurement_matrix = jnp.asarray(
            H * (1.0 + 0.2 * rng.normal(size=H.shape))
        )
    elif kind == "CNM":
        model.process_variance = model.process_variance * 1.2
        model.coupling_strength = model.coupling_strength * 0.7
        model._initialize_process_covariance()
    else:
        model.coupling_strength = model.coupling_strength * 0.7
        model.damping_coef = model.damping_coef * 0.99
        model._rebuild_stable_transition_matrix()
    model.measurement_cov = model.measurement_cov * 1.3


def _gaussian_fit_error(kind: str, seed: int, n_time: int, max_iter: int) -> float:
    truth = rh.oscillator_model_at_truth(kind, 2, switching=True)
    truth.discrete_transition_matrix = STICKY_Z
    _, y, _ = rh.simulate_from_oscillator_model(
        truth, n_time, np.random.default_rng(seed)
    )
    model = rh.oscillator_model_at_truth(kind, 2, switching=True)
    model.discrete_transition_matrix = STICKY_Z
    _perturb_gaussian(kind, model, np.random.default_rng(100 + seed))
    model.fit(jnp.asarray(y), skip_init=True, max_iter=max_iter, tol=1e-8)
    if kind == "COM":
        return _com_error(model, truth)
    if kind == "CNM":
        return _relative_error(model.process_cov, truth.process_cov)
    return _relative_error(
        model.continuous_transition_matrix, truth.continuous_transition_matrix
    )


# Observed mean error (T=300 -> 1200) in exploration with 30 EM iterations:
# COM 0.17 -> 0.071 (H Gram blocks), CNM 0.28 -> 0.17 (Q), DIM 0.094 -> 0.056
# (A).  COM's H converges slowly at T=1200 (12 iterations: 0.14 -> 0.11), so
# it keeps 30 iterations.
@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "factor", "max_iter"),
    [("COM", 0.7, 30), ("CNM", 0.85, 15), ("DIM", 0.8, 15)],
)
def test_gaussian_oscillator_error_shrinks_with_data(kind, factor, max_iter, caplog):
    caplog.set_level(logging.ERROR)
    errors = {
        n_time: [_gaussian_fit_error(kind, seed, n_time, max_iter) for seed in SEEDS]
        for n_time in (300, 1200)
    }
    _assert_error_shrinks(errors, kind, factor)


# ---------------------------------------------------------------------------
# Point-process oscillator models and the switching spike-oscillator model
# ---------------------------------------------------------------------------

_PP_NEURONS = 15
_PP_STICKY_Z = jnp.array([[0.99, 0.01], [0.01, 0.99]])


def _pp_model(kind: str, **kwargs):
    """Point-process oscillator model whose parameters are the simulation truth."""
    from state_space_practice.oscillator_utils import (
        construct_common_oscillator_transition_matrix,
    )
    from state_space_practice.point_process_models import (
        CommonOscillatorPointProcessModel,
        CorrelatedNoisePointProcessModel,
        DirectedInfluencePointProcessModel,
    )
    from state_space_practice.switching_point_process import (
        SpikeObsParams,
        SwitchingSpikeOscillatorModel,
    )

    common = {
        "n_neurons": _PP_NEURONS,
        "n_discrete_states": 2,
        "sampling_freq": 100.0,
        "dt": 0.01,
    }
    if kind == "COM-PP":
        model = CommonOscillatorPointProcessModel(
            n_oscillators=1,
            freqs=jnp.array([8.0]),
            damping_coef=jnp.array([0.97]),
            process_variance=jnp.array([0.05]),
            **common,
            **kwargs,
        )
    elif kind == "CNM-PP":
        model = CorrelatedNoisePointProcessModel(
            n_oscillators=2,
            freqs=jnp.array([6.0, 11.0]),
            damping_coef=jnp.array([0.97, 0.95]),
            process_variance=jnp.array([[0.05, 0.02], [0.04, 0.02]]),
            phase_difference=jnp.zeros((2, 2, 2)).at[0, 1, 0].set(0.5),
            coupling_strength=jnp.zeros((2, 2, 2)).at[0, 1, 0].set(0.03),
            **common,
            **kwargs,
        )
    elif kind == "DIM-PP":
        model = DirectedInfluencePointProcessModel(
            n_oscillators=2,
            freqs=jnp.array([6.0, 11.0]),
            damping_coef=jnp.array([0.97, 0.95]),
            process_variance=jnp.array([0.05, 0.03]),
            phase_difference=jnp.zeros((2, 2, 2))
            .at[1, 0, 0]
            .set(0.4)
            .at[0, 1, 1]
            .set(-0.4),
            coupling_strength=jnp.zeros((2, 2, 2))
            .at[1, 0, 0]
            .set(0.3)
            .at[0, 1, 1]
            .set(0.3),
            **common,
            **kwargs,
        )
    else:
        model = SwitchingSpikeOscillatorModel(n_oscillators=1, **common, **kwargs)
    model._initialize_parameters(jax.random.PRNGKey(0))
    if kind == "SSO":
        model.continuous_transition_matrix = jnp.stack(
            [
                construct_common_oscillator_transition_matrix(
                    freqs=jnp.array([8.0]),
                    damping_coef=jnp.array([d]),
                    sampling_freq=100.0,
                )
                for d in (0.98, 0.85)
            ],
            axis=-1,
        )
        model.process_cov = jnp.stack([0.05 * jnp.eye(2), 0.2 * jnp.eye(2)], axis=-1)
    n_latent = model.n_latent
    weights = jax.random.normal(jax.random.PRNGKey(3), (_PP_NEURONS, n_latent, 2)) * 0.5
    baseline = jnp.full((_PP_NEURONS, 2), 3.5)
    if kind == "COM-PP":  # COM: the observation model is what switches
        baseline = baseline.at[:, 1].set(2.8)
    else:
        weights = jnp.repeat(weights[..., :1], 2, axis=-1)
    model.spike_params = SpikeObsParams(baseline=baseline, weights=weights)
    model.discrete_transition_matrix = _PP_STICKY_Z
    model.init_mean = jnp.zeros((n_latent, 2))
    model.init_cov = jnp.stack([0.5 * jnp.eye(n_latent)] * 2, axis=-1)
    return model


def _pp_summary(kind: str, model) -> dict:
    """Identifiable, rotation-invariant parameter summaries per model.

    Returns ``{"consistent": ..., "biased": ...}``: the first part is expected
    to converge to the truth; the second (process-noise scale of CNM-PP and
    the spike-oscillator model) carries a documented approximation bias and
    is only bounded.
    """
    if kind == "COM-PP":
        W = np.asarray(model.spike_params.weights)
        grams = [(W[..., j] @ W[..., j].T).ravel() for j in range(2)]
        return {
            "consistent": np.concatenate(
                [*grams, np.asarray(model.spike_params.baseline).ravel()]
            )
        }
    if kind == "CNM-PP":
        return {
            "consistent": np.abs(np.asarray(model.coupling_strength)[0, 1]),
            "biased": np.asarray(model.process_variance).ravel(),
        }
    if kind == "DIM-PP":
        coupling = np.abs(np.asarray(model.coupling_strength))
        return {
            "consistent": np.concatenate(
                [np.asarray(model.damping_coef), coupling[1, 0], coupling[0, 1]]
            )
        }
    A = np.asarray(model.continuous_transition_matrix)
    Q = np.asarray(model.process_cov)
    eig = [np.linalg.eigvals(A[..., j]) for j in range(2)]
    return {
        "consistent": np.concatenate(
            [[np.abs(e).max() for e in eig], [np.abs(np.angle(e)).max() for e in eig]]
        ),
        "biased": np.concatenate([np.linalg.eigvalsh(Q[..., j]) for j in range(2)]),
    }


def _pp_fit_error(kind: str, seed: int, n_time: int) -> dict:
    from state_space_practice.simulate.simulate_switching_spikes import (
        simulate_switching_spike_oscillator,
    )
    from state_space_practice.switching_point_process import (
        QRegularizationConfig,
        SpikeObsParams,
    )

    truth = _pp_model(kind)
    spikes, _, _ = simulate_switching_spike_oscillator(
        n_time=n_time,
        transition_matrices=truth.continuous_transition_matrix,
        process_covs=truth.process_cov,
        discrete_transition_matrix=_PP_STICKY_Z,
        spike_weights=truth.spike_params.weights,
        spike_baseline=truth.spike_params.baseline,
        dt=truth.dt,
        key=jax.random.PRNGKey(seed),
        init_mean=jnp.zeros(truth.n_latent),
        init_cov=0.5 * jnp.eye(truth.n_latent),
    )
    rng = np.random.default_rng(100 + seed)
    if kind == "COM-PP":
        # COM-PP's dynamics are fixed by design: recover the spike GLMs.
        model = _pp_model(kind)
        sp = model.spike_params
        model.spike_params = SpikeObsParams(
            baseline=sp.baseline + 0.1 * rng.normal(size=sp.baseline.shape),
            weights=sp.weights * (1.0 + 0.2 * rng.normal(size=sp.weights.shape)),
        )
    else:
        # Recover the dynamics with the spike GLM known: with free weights the
        # latent scale trades off against Q (and the weight ridge biases it).
        kwargs = {"update_spike_params": False, "max_newton_iter": 3}
        if kind == "CNM-PP":
            kwargs["q_regularization"] = QRegularizationConfig(enabled=False)
        if kind == "DIM-PP":
            kwargs["use_reparameterized_mstep"] = True
        model = _pp_model(kind, **kwargs)
        if kind == "CNM-PP":
            model.process_variance = model.process_variance * 1.05
            model._initialize_process_covariance()
        elif kind == "DIM-PP":
            model.coupling_strength = model.coupling_strength * 0.95
            model._rebuild_stable_transition_matrix()
        else:
            model.process_cov = model.process_cov * 1.05
    model.fit(spikes, skip_init=True, max_iter=6, tol=1e-8)
    estimate, reference = _pp_summary(kind, model), _pp_summary(kind, truth)
    return {k: _relative_error(estimate[k], reference[k]) for k in reference}


# Consistent parts: COM-PP spike-GLM Gram blocks + baselines, DIM-PP damping +
# coupling, CNM-PP coupling magnitude, SSO damping + frequency.  In
# exploration (T=200 -> 2400, 3 seeds) the CNM-PP coupling error went
# 0.20 -> 0.077 and the SSO damping/frequency error 0.059 -> 0.016.
# The process-noise scale (CNM-PP variances, SSO Q eigenvalues) stays at a
# relative error of ~0.05-0.1 at every length -- the approximate (Laplace-EKF
# + GPB) E-step misassigns bins between the discrete states and contaminates
# the low-noise state's Q -- so it is only bounded.  These fits use three
# Newton steps per Laplace update: with the default single Fisher-scoring
# step, SSO's low-noise Q is ~2x too large (0.09-0.10 vs 0.05) even at
# T=3200.
@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "factor", "biased_bound"),
    [
        ("COM-PP", 0.75, None),
        ("DIM-PP", 0.75, None),
        ("CNM-PP", 0.75, 0.25),
        ("SSO", 0.75, 0.15),
    ],
)
def test_point_process_oscillator_error_shrinks_with_data(
    kind, factor, biased_bound, caplog
):
    caplog.set_level(logging.ERROR)
    fits = {
        n_time: [_pp_fit_error(kind, seed, n_time) for seed in SEEDS]
        for n_time in (300, 2400)
    }
    _assert_error_shrinks(
        {n: [f["consistent"] for f in rows] for n, rows in fits.items()},
        kind,
        factor,
    )
    if biased_bound is not None:
        biased = [f["biased"] for f in fits[2400]]
        assert max(biased) < biased_bound, (
            f"{kind} process-noise error at T=2400 {np.round(biased, 4).tolist()}"
        )
        # Guard: the bias the comment documents is real (not silently fixed).
        assert min(biased) > 0.01, np.round(biased, 4).tolist()

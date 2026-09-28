# ruff: noqa: E402
"""Golden-value regression test for the EM ``fit`` loops.

Every model below is fitted by EM on small, fixed-seed synthetic data, and the
full log-likelihood history (hence also the iteration count) plus a few
fitted-parameter summaries are compared against recorded values.  The point is
to catch refactors of the EM loops and their E/M-step internals that
*silently* shift the numerics: a correct refactor reproduces these numbers to
near round-off.

Configurations are chosen so that EM runs its ordinary path (no rollback, no
non-finite log-likelihood); the rollback branches are covered by behavioural
tests elsewhere.  The values were recorded with float64 on Linux x86-64 with
jax/jaxlib 0.10.2.  The tolerances (``LL_TOLERANCES`` and
``parameter_tolerance``) are set from the deviations observed when the same
code runs on macOS arm64.

History
-------
The first pins were recorded from ``master`` at ``f9e1d8f`` (macOS arm64).
They were re-recorded after the numerical-review fixes
that intentionally change EM numerics; the cases that moved, and why:

* ``point_process_glm``, ``place_field_*``: the EM update of the initial
  state uses the smoothed ``x_0`` (one RTS step behind the smoother output)
  instead of the ``x_1`` moments; the Laplace-EKF log-likelihood takes both
  log-determinants from the factors the update used (one jitter policy); Q
  comes from the centred residual form.  LL histories moved by 1e-7 to 2e-6
  relative, the last smoothed mean by up to 2e-2 relative.
* ``directed_influence``: the switching M-step estimates R at the fixed H the
  model installs instead of at the unconstrained H*, so the iteration-1
  log-likelihood moved by 9% and the fitted transition entries changed.
* ``common_oscillator_pp``, ``directed_influence_pp``,
  ``switching_spike_oscillator``: the shared Laplace update's jitter policy
  (spike weights up to 2e-5 relative, LL up to 6e-10 relative).
* ``common_oscillator``, ``covariate_choice``: round-off only (< 1e-10
  relative) from the residual-form covariance updates and jitted M-steps.
* ``smith_learning``, ``multinomial_choice``: unchanged.

They were re-recorded a second time after the
verification round (exact oracles, calibration and M-step stationarity tests)
found and fixed further bugs; every case moved:

* all cases: the Cholesky shift in every solve is scale-relative instead of
  an absolute ``1e-9`` (place-field ``Q ~ 1e-6`` moved 0.8%, most others at
  ``1e-8`` to ``1e-5`` relative).
* ``point_process_glm``, ``place_field_*``: the ``x_0 -> x_1`` transition is
  counted in the ``A`` / ``Q`` statistics (exact EM).
* ``smith_learning``: the log-likelihood is the Laplace evidence instead of
  the plug-in binomial likelihood (a different quantity; the history changed
  by 2e-3 relative), the initial-state variance M-step is the exact maximiser
  and the Newton mode-finding is line-searched.
* ``common_oscillator``, ``directed_influence``: warm init now seeds each
  state's parameters (the first log-likelihood is far higher; the fits end at
  a different local optimum).
* ``directed_influence_pp``: the joint BFGS restarts on line-search failure
  and the projected M-step never lowers the objective.
* ``switching_spike_oscillator``: ``Q`` is estimated at the projected ``A``.
* ``multinomial_choice``, ``covariate_choice``: the softmax Laplace update
  takes 10 line-searched Newton steps instead of 3 undamped ones.

A third re-record (same commit series) followed the change of the default
``max_newton_iter`` from 1 to 3 in the point-process filters, ``PlaceFieldModel``,
the point-process oscillator models and ``SwitchingSpikeOscillatorModel``; it
moved ``point_process_glm``, the two single-step place-field cases and the three
point-process oscillator cases (log-likelihood by up to 1e-3 relative).

A fourth re-record followed two round-off-level changes: the round-off slack
in the point-process Fisher line search and the softmax Laplace evidence taking
its log-determinants from the update's own factors (point-process, place-field
and choice cases moved by at most 3e-10 relative in log-likelihood).

Regenerating the expected values
--------------------------------
Only do this for an *intended* change in numerics.  Running this module as a
script prints ``EXPECTED`` for whichever ``state_space_practice`` it imports.
From the checkout that contains the intended change, run::

    uv run --no-sync python src/state_space_practice/tests/test_em_golden_regression.py

(prefix ``PYTHONPATH=<tree>/src`` to record from a different source tree).
Paste the output over ``EXPECTED`` and add an entry to the history above that
names the change and which cases moved, by how much.  Record on Linux x86-64,
the platform the tolerances were measured against, or re-measure them.
"""

import logging
import warnings
from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.covariate_choice import CovariateChoiceModel
from state_space_practice.multinomial_choice import MultinomialChoiceModel
from state_space_practice.oscillator_models import (
    CommonOscillatorModel,
    DirectedInfluenceModel,
)
from state_space_practice.place_field_model import PlaceFieldModel
from state_space_practice.point_process_kalman import PointProcessModel
from state_space_practice.point_process_models import (
    CommonOscillatorPointProcessModel,
    DirectedInfluencePointProcessModel,
)
from state_space_practice.smith_learning_algorithm import SmithLearningModel
from state_space_practice.switching_point_process import (
    SwitchingSpikeOscillatorModel,
)

FitResult = dict[str, list[float]]


# --- Synthetic data (self-contained so that only the EM code is pinned) ---


def _binary_learning_curve() -> np.ndarray:
    """Bernoulli outcomes of a learner going from ~12% to ~88% correct.

    Returns
    -------
    np.ndarray, shape (120,)
    """
    rng = np.random.default_rng(0)
    p_correct = 1.0 / (1.0 + np.exp(-np.linspace(-2.0, 2.0, 120)))
    return rng.binomial(1, p_correct).astype(float)


def _glm_design_and_spikes() -> tuple[np.ndarray, np.ndarray]:
    """Poisson counts from a 3-dim random-walk GLM weight.

    Returns
    -------
    design_matrix : np.ndarray, shape (300, 3)
    spike_counts : np.ndarray, shape (300,)
    """
    rng = np.random.default_rng(1)
    n_time, n_dims = 300, 3
    design_matrix = rng.normal(size=(n_time, n_dims)) * 0.5
    weights = np.cumsum(rng.normal(size=(n_time, n_dims)) * 0.05, axis=0)
    rate_hz = np.exp(1.0 + np.sum(design_matrix * weights, axis=1))
    return design_matrix, rng.poisson(rate_hz * 0.02).astype(float)


def _place_field_data(n_neurons: int) -> tuple[np.ndarray, np.ndarray]:
    """2D random-walk trajectory with Gaussian place-field Poisson spiking.

    Returns
    -------
    position : np.ndarray, shape (400, 2)
    spikes : np.ndarray, shape (400,) if ``n_neurons == 1``
        else (400, n_neurons)
    """
    rng = np.random.default_rng(2)
    n_time, dt = 400, 0.02
    steps = rng.normal(scale=3.0, size=(n_time, 2))
    position = np.clip(50.0 + np.cumsum(steps, axis=0), 0.0, 100.0)
    centers = rng.uniform(30.0, 70.0, size=(n_neurons, 2))
    sq_dist = np.sum((position[:, None, :] - centers[None]) ** 2, axis=-1)
    rate_hz = 1.0 + 30.0 * np.exp(-sq_dist / (2.0 * 15.0**2))
    spikes = rng.poisson(rate_hz * dt).astype(np.int64)
    return position, spikes[:, 0] if n_neurons == 1 else spikes


def _gaussian_oscillator_observations(n_sources: int) -> jax.Array:
    """White-noise observations, shape (200, n_sources)."""
    return jax.random.normal(jax.random.PRNGKey(0), (200, n_sources))


def _oscillator_spikes() -> jax.Array:
    """Poisson(0.5) counts, shape (80, 4)."""
    return jax.random.poisson(jax.random.PRNGKey(0), 0.5, shape=(80, 4)).astype(float)


def _choice_data() -> tuple[jax.Array, jax.Array]:
    """Uniform 3-option choices and 2 covariates over 150 trials.

    Returns
    -------
    choices : jax.Array, shape (150,)
    covariates : jax.Array, shape (150, 2)
    """
    rng = np.random.default_rng(55)
    choices = jnp.asarray(rng.integers(0, 3, 150))
    covariates = jnp.asarray(rng.normal(size=(150, 2)))
    return choices, covariates


# --- One fit per configuration ---

_TWO_OSCILLATORS = dict(
    n_oscillators=2,
    n_discrete_states=2,
    sampling_freq=100.0,
    freqs=jnp.array([8.0, 12.0]),
    damping_coef=jnp.array([0.95, 0.95]),
    process_variance=jnp.array([0.1, 0.1]),
)
_DIRECTED_COUPLING = dict(
    phase_difference=jnp.zeros((2, 2, 2)).at[0, 1, 1].set(0.5),
    coupling_strength=jnp.zeros((2, 2, 2)).at[0, 1, 1].set(0.02),
)


_MAX_SUMMARY_SIZE = 8


def _result(log_likelihoods: list[float], **summaries: object) -> FitResult:
    """Collect the LL history and fitted-parameter summaries as float lists.

    Each summary is flattened and thinned by a constant stride to at most
    ``_MAX_SUMMARY_SIZE`` evenly spaced entries, keeping the recorded values
    compact while still touching every block of a large array.
    """
    out = {"log_likelihoods": [float(ll) for ll in log_likelihoods]}
    for name, value in summaries.items():
        flat = np.asarray(value, dtype=float).ravel()
        stride = -(-flat.size // _MAX_SUMMARY_SIZE)
        out[name] = [float(v) for v in flat[::stride]]
    return out


def _fit_smith() -> FitResult:
    model = SmithLearningModel(max_possible_correct=1)
    lls = model.fit(jnp.asarray(_binary_learning_curve()), max_iter=10)
    return _result(
        lls,
        sigma_epsilon=model.sigma_epsilon,
        init_learning=[model.init_learning_state, model.init_learning_variance],
        learning_state_mode=model.smoothed_learning_state_mode,
    )


def _fit_point_process() -> FitResult:
    design_matrix, spikes = _glm_design_and_spikes()
    model = PointProcessModel(n_state_dims=3, dt=0.02)
    lls = model.fit(jnp.asarray(design_matrix), jnp.asarray(spikes), max_iter=5)
    return _result(
        lls,
        transition_diag=jnp.diag(model.transition_matrix),
        process_cov_diag=jnp.diag(model.process_cov),
        last_smoother_mean=model.smoother_mean[-1],
    )


def _fit_place_field(n_neurons: int, **model_kwargs: object) -> FitResult:
    position, spikes = _place_field_data(n_neurons)
    model = PlaceFieldModel(dt=0.02, n_interior_knots=3, **model_kwargs)
    lls = model.fit(position, spikes, max_iter=4, verbose=False)
    return _result(
        lls,
        process_cov_diag=jnp.diag(model.process_cov),
        transition_diag=jnp.diag(model.transition_matrix),
        last_smoother_mean=model.smoother_mean[-1],
    )


def _fit_common_oscillator() -> FitResult:
    model = CommonOscillatorModel(
        **_TWO_OSCILLATORS, n_sources=3, measurement_variance=0.05
    )
    lls = model.fit(
        _gaussian_oscillator_observations(3),
        max_iter=5,
        key=jax.random.PRNGKey(1),
    )
    return _result(
        lls,
        discrete_transition=model.discrete_transition_matrix,
        measurement_matrix=model.measurement_matrix,
    )


def _fit_directed_influence() -> FitResult:
    model = DirectedInfluenceModel(
        **_TWO_OSCILLATORS, **_DIRECTED_COUPLING, measurement_variance=0.05
    )
    lls = model.fit(
        _gaussian_oscillator_observations(2),
        max_iter=5,
        key=jax.random.PRNGKey(1),
    )
    return _result(
        lls,
        discrete_transition=model.discrete_transition_matrix,
        continuous_transition=model.continuous_transition_matrix,
    )


def _fit_common_oscillator_pp() -> FitResult:
    model = CommonOscillatorPointProcessModel(
        n_oscillators=1,
        n_neurons=4,
        n_discrete_states=2,
        sampling_freq=100.0,
        dt=0.01,
        freqs=jnp.array([8.0]),
        damping_coef=jnp.array([0.95]),
        process_variance=jnp.array([0.1]),
    )
    lls = model.fit(_oscillator_spikes(), max_iter=3, key=jax.random.PRNGKey(42))
    return _result(
        lls,
        discrete_transition=model.discrete_transition_matrix,
        spike_weights=model.spike_params.weights,
        spike_baseline=model.spike_params.baseline,
    )


def _fit_directed_influence_pp() -> FitResult:
    # The reparameterized M-step (use_reparameterized_mstep=True) is
    # deliberately not pinned: EM with it amplifies round-off chaotically.  On
    # a 300 x 5 spike train, two implementations whose first log-likelihoods
    # agree to 2e-12 differ by 1.2e-3 by the third iteration, so no tight
    # golden value is meaningful for it.
    model = DirectedInfluencePointProcessModel(
        n_oscillators=2,
        n_neurons=4,
        n_discrete_states=2,
        sampling_freq=100.0,
        dt=0.01,
        freqs=jnp.array([8.0, 12.0]),
        damping_coef=jnp.array([0.95, 0.95]),
        process_variance=jnp.array([0.1, 0.1]),
        phase_difference=jnp.zeros((2, 2, 2)),
        coupling_strength=jnp.zeros((2, 2, 2)),
    )
    lls = model.fit(_oscillator_spikes(), max_iter=3, key=jax.random.PRNGKey(42))
    return _result(
        lls,
        continuous_transition=model.continuous_transition_matrix,
        spike_weights=model.spike_params.weights,
    )


def _fit_switching_spike_oscillator() -> FitResult:
    model = SwitchingSpikeOscillatorModel(
        n_oscillators=1,
        n_neurons=4,
        n_discrete_states=2,
        sampling_freq=100.0,
        dt=0.01,
    )
    lls = model.fit(_oscillator_spikes(), max_iter=3, key=jax.random.PRNGKey(42))
    return _result(
        lls,
        continuous_transition=model.continuous_transition_matrix,
        spike_weights=model.spike_params.weights,
    )


def _fit_multinomial_choice() -> FitResult:
    choices, _ = _choice_data()
    model = MultinomialChoiceModel(n_options=3)
    lls = model.fit(choices, max_iter=8)
    return _result(
        lls,
        inverse_temperature_and_noise=[
            model.inverse_temperature,
            model.process_noise,
        ],
        smoothed_values=model.smoothed_option_values_,
    )


def _fit_covariate_choice() -> FitResult:
    choices, covariates = _choice_data()
    model = CovariateChoiceModel(n_options=3, n_covariates=2)
    lls = model.fit(choices, covariates=covariates, max_iter=8)
    return _result(
        lls,
        inverse_temperature_and_noise=[
            model.inverse_temperature,
            model.process_noise,
        ],
        input_gain=model.input_gain_,
    )


CASES: dict[str, Callable[[], FitResult]] = {
    "smith_learning": _fit_smith,
    "point_process_glm": _fit_point_process,
    "place_field_1_neuron_dense": lambda: _fit_place_field(1),
    "place_field_2_neuron_block_update_A": lambda: _fit_place_field(
        2, update_transition_matrix=True
    ),
    "place_field_2_neuron_block_newton3": lambda: _fit_place_field(
        2, max_newton_iter=3
    ),
    "common_oscillator": _fit_common_oscillator,
    "directed_influence": _fit_directed_influence,
    "common_oscillator_pp": _fit_common_oscillator_pp,
    "directed_influence_pp": _fit_directed_influence_pp,
    "switching_spike_oscillator": _fit_switching_spike_oscillator,
    "multinomial_choice": _fit_multinomial_choice,
    "covariate_choice": _fit_covariate_choice,
}

# Per-case (rtol, atol) for the log-likelihood histories.  Each rtol is
# >= ~100x the largest relative deviation observed between the Linux x86-64
# pins and the same code on macOS arm64 (noted inline), so a different CPU or
# XLA reduction order passes.  Most cases, including those whose E-step takes
# Newton / Laplace steps, reproduce to ~1e-14 and get rtol=1e-10.  The
# switching point-process oscillator cases amplify round-off through their
# Laplace updates and numerically optimised M-steps and get looser rtols.
# Changes whose effect is itself near round-off can pass: the point-process
# Fisher line-search slack moved ``point_process_glm`` by 8.9e-11 relative, and
# on ``common_oscillator_pp`` (sparse spikes, Newton converges in one step)
# ``max_newton_iter=1`` instead of 3 moves the history by only 2.4e-9.
_EXACT = (1e-10, 1e-14)
LL_TOLERANCES: dict[str, tuple[float, float]] = {
    "smith_learning": _EXACT,  # observed 1.8e-16 rel
    "point_process_glm": _EXACT,  # observed 1.0e-14 rel
    "place_field_1_neuron_dense": _EXACT,  # observed 1.5e-15 rel
    "place_field_2_neuron_block_update_A": _EXACT,  # observed 5.4e-15 rel
    "place_field_2_neuron_block_newton3": _EXACT,  # observed 9.2e-16 rel
    "common_oscillator": _EXACT,  # observed 1.3e-16 rel
    "directed_influence": _EXACT,  # observed 4.5e-16 rel
    "common_oscillator_pp": (1e-8, 1e-14),  # observed 9.5e-11 rel
    "directed_influence_pp": (5e-8, 1e-14),  # observed 1.2e-10 rel
    "switching_spike_oscillator": (1e-9, 1e-14),  # observed 1.2e-12 rel
    "multinomial_choice": _EXACT,  # observed 1.7e-16 rel
    "covariate_choice": _EXACT,  # observed 3.5e-16 rel
}

# Fitted-parameter summaries.  Each element must satisfy
#
#     |actual - expected| <= PARAMETER_RTOL * |expected| + k * max|expected|
#
# with the maximum taken over that summary and k = PARAMETER_SCALE_ATOL[case].
# The relative term covers entries of the summary's own magnitude.  The scale
# term covers entries near zero, whose round-off is set by the magnitude of
# the quantities they are computed with rather than by their own value: in
# ``common_oscillator_pp`` every spike weight comes out of the same Newton
# solves and moves by 0.7e-9 to 6e-9 across platforms (<= 2.3e-7 of the
# largest weight), which is 3.1e-6 relative on the smallest weight (2.3e-4).
# Its k = 5e-6 leaves ~20x headroom over that and still fails a 1% change in
# ``spike_weight_l2`` (which moves the weights by 1.2e-5 of their scale).
# Every other summary element reproduces to <= 5.3e-8 relative (>= ~20x
# headroom under PARAMETER_RTOL), so the default k only admits entries at or
# very near zero.
PARAMETER_RTOL = 1e-6
PARAMETER_SCALE_ATOL: dict[str, float] = {case: 1e-8 for case in CASES} | {
    "common_oscillator_pp": 5e-6,  # observed 2.3e-7 of the largest weight
}


def parameter_tolerance(case: str, expected: list[float]) -> tuple[float, float]:
    """(rtol, atol) for one recorded fitted-parameter summary of ``case``.

    Parameters
    ----------
    case : str
        Key of ``CASES``.
    expected : list of float
        The recorded summary.

    Returns
    -------
    rtol : float
        ``PARAMETER_RTOL``.
    atol : float
        ``PARAMETER_SCALE_ATOL[case]`` times the summary's largest magnitude,
        at least 1e-14 so that all-zero summaries admit round-off.
    """
    scale = float(np.max(np.abs(expected)))
    return PARAMETER_RTOL, max(PARAMETER_SCALE_ATOL[case] * scale, 1e-14)


EXPECTED: dict[str, FitResult] = {
    "smith_learning": {
        "log_likelihoods": [
            -79.57945306816822,
            -78.94923697992077,
            -78.55662985468761,
            -78.28982597458777,
            -78.09683342120394,
            -77.95055047961128,
            -77.8350746971021,
            -77.73758486357944,
            -77.65403442889277,
            -77.58207824674852,
            -77.51981195350182,
        ],
        "sigma_epsilon": [0.20552478794221526],
        "init_learning": [-1.0251597717914256, 1e-08],
        "learning_state_mode": [
            -1.0636127689749522,
            -1.1830564944830462,
            -1.048034617982454,
            -0.585669013529004,
            -0.11649960556878435,
            0.07946863036931866,
            -0.07958266670088346,
            0.7622347856809302,
        ],
    },
    "point_process_glm": {
        "log_likelihoods": [
            -56.58051712182312,
            -54.8097661798567,
            -54.447173129146094,
            -54.27654737253754,
            -54.17299962323734,
            -54.101125262825036,
        ],
        "transition_diag": [0.9994630540796986, 0.999489234177392, 0.9997521176802389],
        "process_cov_diag": [
            9.996780454921847e-05,
            9.995362542280617e-05,
            9.994191760732463e-05,
        ],
        "last_smoother_mean": [
            -0.010336089583922084,
            -0.4634775400800419,
            -1.2662387481936537,
        ],
    },
    "place_field_1_neuron_dense": {
        "log_likelihoods": [
            -117.75063387557441,
            -115.2505306664526,
            -114.25023225517646,
            -113.66468911220213,
            -113.2165785619284,
        ],
        "process_cov_diag": [
            1.000000079329766e-06,
            1.0000000791876573e-06,
            1.0000000573029412e-06,
            9.999999650034397e-07,
            9.999998825804824e-07,
            9.999999743826038e-07,
            1.000000079329766e-06,
            1.000000079329766e-06,
        ],
        "transition_diag": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "last_smoother_mean": [
            0.0,
            -0.009819664323868998,
            0.4784177157404418,
            3.899140752550965,
            0.5471103497077312,
            1.5408188875690907,
            0.0,
            0.0,
        ],
    },
    "place_field_2_neuron_block_update_A": {
        "log_likelihoods": [
            -165.80706212383333,
            -159.84698312578564,
            -156.93070822562862,
            -155.23629023758463,
            -154.04913206106042,
        ],
        "process_cov_diag": [
            1.000000079613983e-06,
            1.000000073361207e-06,
            1.000000079613983e-06,
            1.0000000682452992e-06,
            1.000000079613983e-06,
            1.0000000766297036e-06,
            1.000000079613983e-06,
            1.0000001569210326e-06,
        ],
        "transition_diag": [
            0.9999999999959603,
            0.9999999945222663,
            0.9999999999959603,
            0.9999999743718494,
            0.9999999999959603,
            0.9999999968708866,
            0.9999999999959603,
            0.9999999681188678,
        ],
        "last_smoother_mean": [
            0.0,
            0.46777327360882287,
            0.0,
            -0.942159521165275,
            0.0,
            0.13281246595920893,
            0.0,
            2.0910348333162605,
        ],
    },
    "place_field_2_neuron_block_newton3": {
        "log_likelihoods": [
            -165.80706212383333,
            -159.84747998762353,
            -156.93146001197863,
            -155.23718467547394,
            -154.0501137400275,
        ],
        "process_cov_diag": [
            1.000000079329766e-06,
            1.00000007307699e-06,
            1.000000079329766e-06,
            1.0000000742138583e-06,
            1.000000079329766e-06,
            1.0000000773402463e-06,
            1.000000079329766e-06,
            1.0000002008325737e-06,
        ],
        "transition_diag": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "last_smoother_mean": [
            0.0,
            0.46776727993377737,
            0.0,
            -0.942118533034253,
            0.0,
            0.13282299063191216,
            0.0,
            2.0908804557593363,
        ],
    },
    "common_oscillator": {
        "log_likelihoods": [
            -863.4294471570297,
            -855.9016045558803,
            -854.2091348016164,
            -853.2128792174877,
            -852.6605005561846,
            -852.3371921538061,
        ],
        "discrete_transition": [
            0.9705270389600903,
            0.029472961039909616,
            0.0038944551091833015,
            0.9961055448908167,
        ],
        "measurement_matrix": [
            0.23087295296818605,
            0.20664252209262265,
            0.1099310396325969,
            0.05023137857120453,
            0.1702710577206667,
            -0.11600566779793342,
            -0.08677823609767334,
            0.07998065308600343,
        ],
    },
    "directed_influence": {
        "log_likelihoods": [
            -765.0012549431083,
            -601.8670529366505,
            -581.8496731602861,
            -579.8503534157039,
            -579.3506358719297,
            -579.023144094457,
        ],
        "discrete_transition": [
            0.9939742725415738,
            0.006025727458426146,
            0.019557510303373223,
            0.9804424896966267,
        ],
        "continuous_transition": [
            0.41373089058138396,
            -0.07797922580905468,
            0.422766538480847,
            -0.09393022382643132,
            -0.0017224912339403793,
            0.4181623474285132,
            -0.07722568151607806,
            0.5823955846037171,
        ],
    },
    "common_oscillator_pp": {
        "log_likelihoods": [
            -273.72126437892206,
            -271.97071205113537,
            -271.28916338001227,
            -269.90221213371524,
        ],
        "discrete_transition": [
            0.994192802888215,
            0.005807197111785088,
            0.12591881823106085,
            0.8740811817689391,
        ],
        "spike_weights": [
            -0.0002340023758586622,
            -0.003502327300746238,
            -0.027166346677827764,
            -0.006240025066072157,
            0.01069110663773097,
            -0.007447269631915379,
            0.005971863480478094,
            0.01790918297317234,
        ],
        "spike_baseline": [
            3.5028519965290372,
            1.8115525295027157,
            3.8164665533942035,
            3.435367169154613,
            3.804326538035624,
            3.2036497131748267,
            4.0499983497913865,
            4.062300094005257,
        ],
    },
    "directed_influence_pp": {
        "log_likelihoods": [
            -273.9050067840272,
            -271.9368476640412,
            -271.4831414466713,
            -270.6349879276118,
        ],
        "continuous_transition": [
            0.834713491505655,
            0.0001319374296103501,
            0.45840672300507546,
            -0.0001969426249478829,
            -0.00015194584446566765,
            0.6939957218002814,
            0.00032282887682074643,
            0.6514200050667776,
        ],
        "spike_weights": [
            0.05161435420678284,
            0.05313050572797891,
            0.03874692121981605,
            -0.04575862238296673,
            0.029986582152752947,
            -0.04365413485140413,
            -0.059803842245911204,
            0.030999217273827154,
        ],
    },
    "switching_spike_oscillator": {
        "log_likelihoods": [
            -662.1640134701505,
            -270.7479424004044,
            -269.5808323399934,
            -267.75400748495855,
        ],
        "continuous_transition": [
            0.8558929578612207,
            0.8558740085003571,
            -0.2763314243280173,
            -0.274237844330679,
            0.2763314243280173,
            0.274237844330679,
            0.8558929578612207,
            0.8558740085003571,
        ],
        "spike_weights": [
            -0.05883960486753495,
            -0.039594863411840514,
            -0.032890281291532465,
            -0.03302855340124357,
            -0.07357040171692353,
            0.006483662767825229,
            0.015581842726741926,
            -0.01386843516265752,
        ],
    },
    "multinomial_choice": {
        "log_likelihoods": [
            -168.09077881750065,
            -164.9590408024478,
            -164.95900742976474,
        ],
        "inverse_temperature_and_noise": [0.1, 0.00993616276760951],
        "smoothed_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "covariate_choice": {
        "log_likelihoods": [
            -168.09077881750065,
            -164.95390630873698,
            -164.95304035756067,
        ],
        "inverse_temperature_and_noise": [0.1, 0.009935072930874672],
        "input_gain": [
            0.0005955247264239305,
            -0.001298151051085048,
            -0.00026503778060830155,
            0.0007788807460037201,
        ],
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_em_matches_recorded_values(case: str) -> None:
    expected = EXPECTED[case]
    actual = CASES[case]()

    assert set(actual) == set(expected)
    assert len(actual["log_likelihoods"]) == len(expected["log_likelihoods"]), (
        "number of EM iterations changed"
    )
    for name, expected_value in expected.items():
        rtol, atol = (
            LL_TOLERANCES[case]
            if name == "log_likelihoods"
            else parameter_tolerance(case, expected_value)
        )
        np.testing.assert_allclose(
            actual[name], expected_value, rtol=rtol, atol=atol, err_msg=name
        )


if __name__ == "__main__":
    import pprint

    import state_space_practice

    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    print(f"# recorded from {state_space_practice.__file__}")
    print("EXPECTED: dict[str, FitResult] = ", end="")
    pprint.pprint({case: fit() for case, fit in CASES.items()}, sort_dicts=False)

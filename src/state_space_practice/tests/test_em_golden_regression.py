# ruff: noqa: E402
"""Golden-value regression test for the EM ``fit`` loops.

Every model below is fitted by EM on small, fixed-seed synthetic data, and the
full log-likelihood history (hence also the iteration count) plus a few
fitted-parameter summaries are compared against values recorded from the
reference implementation at commit
``6fc3ef269f96543fbbcd1f0b65d5ad17fee4acb3``.
The point is to catch refactors of the EM loops and their E/M-step internals
that *silently* shift the numerics: a correct refactor reproduces these numbers
to near round-off.

Configurations are chosen so that EM runs its ordinary path (no rollback, no
non-finite log-likelihood); the rollback branches are covered by behavioural
tests elsewhere.  The values were recorded with float64 on Linux x86-64 with
jax/jaxlib 0.10.2.  Log-likelihood histories reproduce across platforms to
~1e-13 relative, so they get tight tolerances; the fitted-parameter summaries
include tiny quantities (process noise ~1e-6, spike weights ~1e-4) in which EM
amplifies platform round-off to ~1e-8 relative, so they get a looser floor.

History
-------
The first pins were recorded from ``master`` at ``f9e1d8f`` (macOS arm64).
They were re-recorded at the commit above after the numerical-review fixes
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

They were re-recorded a second time at the commit above after the
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

Regenerating the expected values
--------------------------------
Only do this for an *intended* change in numerics.  Running this module as a
script prints ``EXPECTED`` for whichever ``state_space_practice`` it imports,
so point ``PYTHONPATH`` at the source tree you want to record from::

    git worktree add /tmp/ssp_ref <commit>
    PYTHONPATH=/tmp/ssp_ref/src uv run --no-sync python \
        src/state_space_practice/tests/test_em_golden_regression.py

and paste the output over ``EXPECTED`` (and update the commit hash and the
history above).
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

# Per-case tolerances for the log-likelihood histories.  Cases that pass
# through no Laplace update or block-covariance machinery reproduce the
# reference to ~1e-13 relative on another platform; they get rtol=1e-10, which
# still admits round-off-level reordering but nothing an algorithmic change
# could hide in.  The point-process and block place-field cases accumulate
# round-off through Newton iterations; their tolerances are >= ~10x the largest
# cross-platform deviation observed, noted inline.
_EXACT = (1e-10, 1e-14)
LL_TOLERANCES: dict[str, tuple[float, float]] = {
    "smith_learning": _EXACT,
    "point_process_glm": _EXACT,
    "place_field_1_neuron_dense": (1e-9, 1e-14),
    "place_field_2_neuron_block_update_A": (2e-8, 1e-14),  # observed 1.4e-9 rel
    "place_field_2_neuron_block_newton3": (2e-8, 1e-14),  # observed 1.1e-9 rel
    "common_oscillator": _EXACT,
    "directed_influence": _EXACT,
    "common_oscillator_pp": (1e-8, 1e-14),  # observed 1.6e-10 rel
    "directed_influence_pp": (5e-8, 1e-14),  # observed 3.9e-9 rel
    "switching_spike_oscillator": (1e-9, 1e-14),  # observed 1.9e-11 rel
    "multinomial_choice": _EXACT,
    "covariate_choice": _EXACT,
}
# Fitted-parameter summaries: EM amplifies platform round-off most in the
# smallest parameters (process noise ~1e-6 moved 1.4e-8 relative between macOS
# arm64 and Linux x86-64 on *unchanged* code; spike weights ~1e-4 up to 7e-9).
# An algorithmic change moves them by >= 1e-5 relative, so 1e-6 keeps the
# signal while tolerating a different CPU / XLA reduction order.  The absolute
# floor covers summaries that are exactly zero.
PARAMETER_TOLERANCE = (1e-6, 1e-14)

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
            -56.58051711678264,
            -54.80976617698238,
            -54.4471731272842,
            -54.27654737310783,
            -54.172999624342516,
            -54.10112526349191,
        ],
        "transition_diag": [0.9994630540798528, 0.9994892341772292, 0.999752117680606],
        "process_cov_diag": [
            9.996780454932499e-05,
            9.99536254229601e-05,
            9.994191760722985e-05,
        ],
        "last_smoother_mean": [
            -0.010336086771678065,
            -0.46347753753460896,
            -1.2662387515616698,
        ],
    },
    "place_field_1_neuron_dense": {
        "log_likelihoods": [
            -117.75063387446063,
            -115.25053065806298,
            -114.25023224886743,
            -113.66468910590642,
            -113.2165785565966,
        ],
        "process_cov_diag": [
            1.000000079329766e-06,
            1.0000000803245257e-06,
            1.0000000563081812e-06,
            9.999999647192227e-07,
            9.999998827225908e-07,
            9.999999746668208e-07,
            1.000000079329766e-06,
            1.000000079329766e-06,
        ],
        "transition_diag": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "last_smoother_mean": [
            0.0,
            -0.009819664336416123,
            0.4784177155460623,
            3.8991407537020124,
            0.5471103492424446,
            1.540818894062041,
            0.0,
            0.0,
        ],
    },
    "place_field_2_neuron_block_update_A": {
        "log_likelihoods": [
            -165.8070621228866,
            -159.8469831261885,
            -156.93070822410272,
            -155.236290231165,
            -154.04913205157933,
        ],
        "process_cov_diag": [
            1.000000079613983e-06,
            1.0000000727927727e-06,
            1.000000079613983e-06,
            1.0000000682452992e-06,
            1.000000079613983e-06,
            1.000000077766572e-06,
            1.000000079613983e-06,
            1.000000156494707e-06,
        ],
        "transition_diag": [
            0.9999999999959603,
            0.9999999945222665,
            0.9999999999959603,
            0.9999999743718486,
            0.9999999999959603,
            0.9999999968708857,
            0.9999999999959603,
            0.9999999681188657,
        ],
        "last_smoother_mean": [
            0.0,
            0.4677732731509472,
            0.0,
            -0.9421595211743581,
            0.0,
            0.13281246554573065,
            0.0,
            2.091034833531335,
        ],
    },
    "place_field_2_neuron_block_newton3": {
        "log_likelihoods": [
            -165.8070621228866,
            -159.84747998600267,
            -156.93146000617037,
            -155.23718466545924,
            -154.05011373705355,
        ],
        "process_cov_diag": [
            1.000000079329766e-06,
            1.00000007307699e-06,
            1.000000079329766e-06,
            1.000000073361207e-06,
            1.000000079329766e-06,
            1.000000078335006e-06,
            1.000000079329766e-06,
            1.000000200690465e-06,
        ],
        "transition_diag": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "last_smoother_mean": [
            0.0,
            0.46776727951840114,
            0.0,
            -0.9421185329625135,
            0.0,
            0.1328229902386753,
            0.0,
            2.090880456560306,
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
            -271.9707120507043,
            -271.28916338928894,
            -269.9022122095804,
        ],
        "discrete_transition": [
            0.9941928029049868,
            0.005807197095013221,
            0.12591881839773883,
            0.8740811816022612,
        ],
        "spike_weights": [
            -0.00023400789916628938,
            -0.0035023136385016807,
            -0.027166329212501897,
            -0.006240011230099387,
            0.010691097124737543,
            -0.007447262541390552,
            0.0059718655385002095,
            0.017909166062841028,
        ],
        "spike_baseline": [
            3.5028519966595755,
            1.8115525247559356,
            3.816466554054244,
            3.4353671672739914,
            3.804326538306651,
            3.203649712418013,
            4.049998349898891,
            4.062300096095867,
        ],
    },
    "directed_influence_pp": {
        "log_likelihoods": [
            -273.9050067840272,
            -271.9368476523573,
            -271.4831414382042,
            -270.63498786428784,
        ],
        "continuous_transition": [
            0.8347134917381902,
            0.00013193744749430533,
            0.45840672312399006,
            -0.00019694261872440624,
            -0.00015194583157676498,
            0.693995721983612,
            0.000322828904803015,
            0.6514200049770722,
        ],
        "spike_weights": [
            0.05161436020216339,
            0.05313050715175827,
            0.03874692487687411,
            -0.04575862574847337,
            0.029986583905982266,
            -0.04365413477432386,
            -0.059803847137009536,
            0.030999213010219587,
        ],
    },
    "switching_spike_oscillator": {
        "log_likelihoods": [
            -662.1640134653549,
            -270.7479423725282,
            -269.5808322597622,
            -267.75400739704395,
        ],
        "continuous_transition": [
            0.8558929611485477,
            0.855874008291248,
            -0.2763314254979694,
            -0.274237844158445,
            0.2763314254979694,
            0.274237844158445,
            0.8558929611485477,
            0.855874008291248,
        ],
        "spike_weights": [
            -0.05883960362262944,
            -0.0395948671399391,
            -0.032890280132122314,
            -0.033028556149410936,
            -0.07357040358908393,
            0.006483659372507976,
            0.015581843637562836,
            -0.01386843561016831,
        ],
    },
    "multinomial_choice": {
        "log_likelihoods": [
            -168.09077878407945,
            -164.95904080196226,
            -164.959007429279,
        ],
        "inverse_temperature_and_noise": [0.1, 0.00993616276760951],
        "smoothed_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "covariate_choice": {
        "log_likelihoods": [
            -168.09077878407945,
            -164.95390630825122,
            -164.95304035707505,
        ],
        "inverse_temperature_and_noise": [0.1, 0.0099350729308741],
        "input_gain": [
            0.0005955247264239374,
            -0.0012981510510850524,
            -0.00026503778060831364,
            0.000778880746003721,
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
            LL_TOLERANCES[case] if name == "log_likelihoods" else PARAMETER_TOLERANCE
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

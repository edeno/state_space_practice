# ruff: noqa: E402
"""Golden-value regression test for the EM ``fit`` loops.

Every model below is fitted by EM on small, fixed-seed synthetic data, and the
full log-likelihood history (hence also the iteration count) plus a few
fitted-parameter summaries are compared against values recorded from the
reference implementation at commit
``06cc7de01cf7b6435e9fb744e102688fff481269``.
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
            -79.69933218133612,
            -78.99575513298883,
            -78.47779111769137,
            -78.14728232599646,
            -77.94989307877549,
            -77.83441527346407,
            -77.76550067621883,
            -77.72345406431378,
            -77.69527958987813,
            -77.67521945277451,
            -77.65922564741237,
        ],
        "sigma_epsilon": [0.20620096333061894],
        "init_learning": [-1.4050861069963385, 0.12758878770117518],
        "learning_state_mode": [
            -1.4277736321081707,
            -1.2817049760533763,
            -1.0720653537112688,
            -0.590079265867241,
            -0.11669528230069952,
            0.07984331154355515,
            -0.08094448013343314,
            0.763274549369248,
        ],
    },
    "point_process_glm": {
        "log_likelihoods": [
            -56.64088192154854,
            -54.805898572605784,
            -54.444689388489785,
            -54.27481133323117,
            -54.17167947574161,
            -54.10005514855322,
        ],
        "transition_diag": [0.9994653194644283, 0.9994819325301134, 0.9997512474032848],
        "process_cov_diag": [
            9.997781482057636e-05,
            9.996291950978701e-05,
            9.995049230577501e-05,
        ],
        "last_smoother_mean": [
            -0.009998089022387465,
            -0.4629534292203509,
            -1.2668993739887677,
        ],
    },
    "place_field_1_neuron_dense": {
        "log_likelihoods": [
            -117.7417577466171,
            -115.24256698230296,
            -114.24280563943415,
            -113.6552474010005,
            -113.2053132730756,
        ],
        "process_cov_diag": [
            1.0079999917974862e-06,
            1.0079999920824156e-06,
            1.007999981540027e-06,
            1.007999896132434e-06,
            1.0079998165658941e-06,
            1.007999914225452e-06,
            1.0079999917974862e-06,
            1.0079999917974862e-06,
        ],
        "transition_diag": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "last_smoother_mean": [
            0.0,
            -0.009840578049769195,
            0.47726905281482296,
            3.901285150686082,
            0.5465328349384494,
            1.5454521733535929,
            0.0,
            0.0,
        ],
    },
    "place_field_2_neuron_block_update_A": {
        "log_likelihoods": [
            -165.77703492169934,
            -159.79658765650126,
            -156.86390759826725,
            -155.15865926718595,
            -153.97364717963447,
        ],
        "process_cov_diag": [
            1.0079999917974862e-06,
            1.0079999872386153e-06,
            1.0079999917974862e-06,
            1.0079999852441094e-06,
            1.0079999917974862e-06,
            1.0079999896605154e-06,
            1.0079999917974862e-06,
            1.0080000757091996e-06,
        ],
        "transition_diag": [
            0.9999999959209992,
            0.9999999904190654,
            0.9999999959209992,
            0.9999999699588251,
            0.9999999959209992,
            0.9999999927831443,
            0.9999999959209992,
            0.9999999644355377,
        ],
        "last_smoother_mean": [
            0.0,
            0.4676535444902131,
            0.0,
            -0.9418900569850257,
            0.0,
            0.13158539281013507,
            0.0,
            2.0927273008614247,
        ],
    },
    "place_field_2_neuron_block_newton3": {
        "log_likelihoods": [
            -165.80707556042552,
            -159.84749421692482,
            -156.93147561226226,
            -155.2372018255782,
            -154.0501325501947,
        ],
        "process_cov_diag": [
            1.0079999917974862e-06,
            1.0079999876660095e-06,
            1.0079999917974862e-06,
            1.0079999905153036e-06,
            1.0079999917974862e-06,
            1.0079999912276272e-06,
            1.0079999917974862e-06,
            1.0080001187335417e-06,
        ],
        "transition_diag": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "last_smoother_mean": [
            0.0,
            0.4677672078330044,
            0.0,
            -0.94211862478966,
            0.0,
            0.13282291088896941,
            0.0,
            2.0908807407549372,
        ],
    },
    "common_oscillator": {
        "log_likelihoods": [
            -5264.806444212983,
            -864.2181355875056,
            -853.1167260334241,
            -852.5712856936526,
            -852.3750063247894,
            -852.2950663588033,
        ],
        "discrete_transition": [
            0.8959358823789606,
            0.10406411762103951,
            0.22651637096668603,
            0.773483629033314,
        ],
        "measurement_matrix": [
            0.15446390370366214,
            0.13551357748177464,
            0.07632379400232135,
            -0.02375522804590668,
            0.10518977536810814,
            -0.05270179574890487,
            -0.053239997432183904,
            -0.032358135139554496,
        ],
    },
    "directed_influence": {
        "log_likelihoods": [
            -1633.3124019174236,
            -763.1296588944158,
            -601.875381559935,
            -581.8818724600944,
            -579.903965013694,
            -579.4172564989066,
        ],
        "discrete_transition": [
            0.980991830510689,
            0.01900816948931121,
            0.0062569774947468975,
            0.993743022505253,
        ],
        "continuous_transition": [
            0.40567262522152325,
            -0.11736659817797383,
            0.42922370146958544,
            -0.08917959002344307,
            0.005448984338919801,
            0.4750437777706677,
            -0.009904623306323489,
            0.5954558229702361,
        ],
    },
    "common_oscillator_pp": {
        "log_likelihoods": [
            -273.7212644609051,
            -271.9707121294263,
            -271.28916352747956,
            -269.90221283996783,
        ],
        "discrete_transition": [
            0.9941928027320925,
            0.0058071972679073465,
            0.12591881692853113,
            0.8740811830714689,
        ],
        "spike_weights": [
            -0.00023399832686438134,
            -0.0035023236999313805,
            -0.027166341268409683,
            -0.006240028548839944,
            0.010691106607746383,
            -0.00744726618753062,
            0.0059718589063385156,
            0.017909177700624414,
        ],
        "spike_baseline": [
            3.5028519962908318,
            1.8115525776260524,
            3.8164665504067097,
            3.435367207017822,
            3.8043265364310423,
            3.2036497426726593,
            4.049998349334377,
            4.062300098331773,
        ],
    },
    "directed_influence_pp": {
        "log_likelihoods": [
            -273.9050069538668,
            -271.93685759718795,
            -271.4769540534215,
            -270.61518987520145,
        ],
        "continuous_transition": [
            0.8350089265299702,
            0.00033829897314898653,
            0.4584941930657065,
            -0.0002160976130016461,
            -6.487180937817834e-05,
            0.6946559760392366,
            0.000572681357630095,
            0.6520250993600448,
        ],
        "spike_weights": [
            0.0518686568528939,
            0.05287618140270316,
            0.0388717920873224,
            -0.0459248454666966,
            0.030104089678413767,
            -0.043785134861137436,
            -0.06000366638040976,
            0.031111820135246795,
        ],
    },
    "switching_spike_oscillator": {
        "log_likelihoods": [
            -662.1638576943454,
            -270.74796874727446,
            -269.5811025980952,
            -267.7544885508181,
        ],
        "continuous_transition": [
            0.8558958377181372,
            0.8558619788087026,
            -0.2763338371046401,
            -0.2742348125651018,
            0.2763338371046401,
            0.2742348125651018,
            0.8558958377181372,
            0.8558619788087026,
        ],
        "spike_weights": [
            -0.05884250803288802,
            -0.0395964560838335,
            -0.03289377589460647,
            -0.0330278938213854,
            -0.07357521989857148,
            0.0064857064621266095,
            0.015582256237362934,
            -0.013868133807748872,
        ],
    },
    "multinomial_choice": {
        "log_likelihoods": [-168.09077828766885, -164.9617450966434, -164.961710738157],
        "inverse_temperature_and_noise": [0.10081306187557834, 0.009936097272214669],
        "smoothed_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "covariate_choice": {
        "log_likelihoods": [
            -168.09077828766885,
            -164.95657530249645,
            -164.9556969789486,
        ],
        "inverse_temperature_and_noise": [0.10081306187557834, 0.009935007000345365],
        "input_gain": [
            0.0005960370207988287,
            -0.001299500560891564,
            -0.00026530545893527663,
            0.0007794742331517103,
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

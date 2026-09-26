# ruff: noqa: E402
"""Golden-value regression test for the EM ``fit`` loops.

Every model below is fitted by EM on small, fixed-seed synthetic data, and the
full log-likelihood history (hence also the iteration count) plus a few
fitted-parameter summaries are compared against values recorded from the
reference implementation at commit
``f9e1d8f18691b0a2998807465eefc9707cac9445``.  The point is to catch
refactors of the EM loops and their E/M-step internals that *silently* shift
the numerics: a correct refactor reproduces these numbers to near round-off.

Configurations are chosen so that EM runs its ordinary path (no rollback, no
non-finite log-likelihood); the rollback branches are covered by behavioural
tests elsewhere.  The values were recorded with float64 on macOS arm64 with
jax/jaxlib 0.10.2; the tightest tolerances below assume a comparable
platform, since a different CPU or XLA version may reorder reductions.

Regenerating the expected values
--------------------------------
Only do this for an *intended* change in numerics.  Running this module as a
script prints ``EXPECTED`` for whichever ``state_space_practice`` it imports,
so point ``PYTHONPATH`` at the source tree you want to record from::

    git worktree add /tmp/ssp_ref <commit>
    PYTHONPATH=/tmp/ssp_ref/src uv run --no-sync python \
        src/state_space_practice/tests/test_em_golden_regression.py

and paste the output over ``EXPECTED`` (and update the commit hash above).
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

# (rtol, atol) per case; atol is a floor for entries at or near zero.
# Cases whose E/M steps perform the same floating-point operations as the
# reference reproduce it bit-for-bit; they get rtol=1e-10, which still admits
# round-off-level reordering but nothing an algorithmic change could hide in.
# The block place-field and point-process oscillator cases differ from the
# reference at round-off level, which EM amplifies over iterations (most in
# the smallest fitted parameters).  Their tolerances are >= ~10x the largest
# deviation observed against the reference, noted inline.
_EXACT = (1e-10, 1e-14)
TOLERANCES: dict[str, tuple[float, float]] = {
    "smith_learning": _EXACT,
    "point_process_glm": _EXACT,
    "place_field_1_neuron_dense": _EXACT,
    "place_field_2_neuron_block_update_A": (2e-8, 1e-14),  # observed 1.4e-9 rel
    "place_field_2_neuron_block_newton3": (2e-8, 1e-14),  # observed 1.1e-9 rel
    "common_oscillator": _EXACT,
    "directed_influence": _EXACT,
    # Observed: LL 1.6e-10 rel; spike weights (|w| >= 2e-4) up to 2.6e-8 abs,
    # i.e. 1.4e-6 relative on the smallest weight, hence an absolute floor.
    "common_oscillator_pp": (1e-8, 3e-7),
    "directed_influence_pp": (5e-8, 1e-14),  # observed 3.9e-9 rel
    "switching_spike_oscillator": (1e-9, 1e-14),  # observed 1.9e-11 rel
    "multinomial_choice": _EXACT,
    "covariate_choice": _EXACT,
}

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
            -56.64088098125764,
            -54.8057648297898,
            -54.44459305349927,
            -54.27476619254216,
            -54.17169517584386,
            -54.10014329128558,
        ],
        "transition_diag": [0.9994653248040615, 0.9994819400860228, 0.9997512528533303],
        "process_cov_diag": [
            9.997783174530033e-05,
            9.996293578175032e-05,
            9.995051059825807e-05,
        ],
        "last_smoother_mean": [
            -0.010173495850860657,
            -0.4629744956852703,
            -1.2666545382419248,
        ],
    },
    "place_field_1_neuron_dense": {
        "log_likelihoods": [
            -117.74174845375951,
            -115.24255533611213,
            -114.24279500742355,
            -113.6552360037551,
            -113.20530092265507,
        ],
        "process_cov_diag": [
            1.007999991370092e-06,
            1.0079999925098095e-06,
            1.0079999822523507e-06,
            1.007999905178943e-06,
            1.0079998159960354e-06,
            1.0079999142966843e-06,
            1.007999991370092e-06,
            1.007999991370092e-06,
        ],
        "transition_diag": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "last_smoother_mean": [
            0.0,
            -0.009840604605700562,
            0.47726785623895873,
            3.9012865551360614,
            0.5465316847150868,
            1.545454374706094,
            0.0,
            0.0,
        ],
    },
    "place_field_2_neuron_block_update_A": {
        "log_likelihoods": [
            -165.77701776679186,
            -159.79656401559905,
            -156.8638824712707,
            -155.1586313095854,
            -153.9736164328118,
        ],
        "process_cov_diag": [
            1.0080798126369258e-06,
            1.0080798087903787e-06,
            1.0080798126369258e-06,
            1.0080798085054493e-06,
            1.0080798126369258e-06,
            1.0080798093602375e-06,
            1.0080798126369258e-06,
            1.0080798974034276e-06,
        ],
        "transition_diag": [
            0.9999999959210049,
            0.9999999904188883,
            0.9999999959210049,
            0.9999999699581358,
            0.9999999959210049,
            0.999999992783058,
            0.9999999959210049,
            0.9999999644345146,
        ],
        "last_smoother_mean": [
            0.0,
            0.46765294338225766,
            0.0,
            -0.9418921728087534,
            0.0,
            0.13158460848876016,
            0.0,
            2.0927314038448124,
        ],
    },
    "place_field_2_neuron_block_newton3": {
        "log_likelihoods": [
            -165.8070583932336,
            -159.8474705858473,
            -156.9314517282477,
            -155.23717657612875,
            -154.05010589987342,
        ],
        # Recorded after the per-neuron block fix, not on the reference commit:
        # there the block path smoothed every neuron with neuron 0's Q once the
        # per-neuron Q differed by less than the equal-blocks tolerance, which
        # put Q ~1e-13 off the dense path. These values agree with
        # ``fit(..., force_dense=True)`` to ~1e-15.
        "process_cov_diag": [
            1.007999991370092e-06,
            1.0079999888057273e-06,
            1.007999991370092e-06,
            1.0079999906577684e-06,
            1.007999991370092e-06,
            1.0079999912276272e-06,
            1.007999991370092e-06,
            1.0080001194458653e-06,
        ],
        "transition_diag": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "last_smoother_mean": [
            0.0,
            0.4677666700856715,
            0.0,
            -0.9421205734848691,
            0.0,
            0.1328221837317423,
            0.0,
            2.0908842973533863,
        ],
    },
    "common_oscillator": {
        "log_likelihoods": [
            -5264.806444212982,
            -864.2181355874648,
            -853.1167260334222,
            -852.571285693658,
            -852.3750063247974,
            -852.2950663588105,
        ],
        "discrete_transition": [
            0.8959358823790271,
            0.10406411762097277,
            0.22651637096681385,
            0.7734836290331861,
        ],
        "measurement_matrix": [
            0.15446390370214075,
            0.13551357747820197,
            0.076323794001411,
            -0.02375522804510817,
            0.10518977536741621,
            -0.052701795747433534,
            -0.053239997431782364,
            -0.03235813513988948,
        ],
    },
    "directed_influence": {
        "log_likelihoods": [
            -1633.3124019174234,
            -835.8268231277262,
            -628.6617696500973,
            -585.038772844064,
            -580.0969178462706,
            -579.3717476885153,
        ],
        "discrete_transition": [
            0.985620521873509,
            0.014379478126490928,
            0.008554729785406856,
            0.9914452702145933,
        ],
        "continuous_transition": [
            0.38303967568227926,
            -0.13101985504730881,
            0.4428077751066417,
            -0.09147403864527288,
            0.011021095508626119,
            0.4753400570645071,
            -0.0010238554411398288,
            0.6031348776285204,
        ],
    },
    "common_oscillator_pp": {
        "log_likelihoods": [
            -273.72126438291116,
            -271.97071205761534,
            -271.289163462526,
            -269.9022127510272,
        ],
        "discrete_transition": [
            0.9941928026628999,
            0.0058071973371001005,
            0.1259188166557504,
            0.8740811833442496,
        ],
        "spike_weights": [
            -0.0002340020967738842,
            -0.0035023250459563678,
            -0.027166344093586203,
            -0.0062400241960830554,
            0.010691105219450008,
            -0.007447268368016869,
            0.005971861899730394,
            0.01790920234504152,
        ],
        "spike_baseline": [
            3.5028519961378266,
            1.8115525959233865,
            3.816466549744381,
            3.435367214947459,
            3.804326536385313,
            3.203649745664705,
            4.0499983730774325,
            4.06230009972351,
        ],
    },
    "directed_influence_pp": {
        "log_likelihoods": [
            -273.9050068017913,
            -271.9368574435317,
            -271.4769538980737,
            -270.6151897202182,
        ],
        "continuous_transition": [
            0.8350089264805329,
            0.00033829897289660795,
            0.45849419303725486,
            -0.0002160976134380212,
            -6.487181007478574e-05,
            0.6946559760097927,
            0.0005726813568868674,
            0.6520250993319063,
        ],
        "spike_weights": [
            0.051868656923750325,
            0.05287618141034226,
            0.03887179234543186,
            -0.0459248456080469,
            0.030104089707579527,
            -0.043785134857723625,
            -0.0600036664643555,
            0.031111820133567707,
        ],
    },
    "switching_spike_oscillator": {
        "log_likelihoods": [
            -662.1638572716206,
            -270.7479686808796,
            -269.58110250333215,
            -267.7544884944406,
        ],
        "continuous_transition": [
            0.8558958216501339,
            0.8558619903743119,
            -0.27633383181032545,
            -0.27423481678345407,
            0.27633383181032545,
            0.27423481678345407,
            0.8558958216501339,
            0.8558619903743119,
        ],
        "spike_weights": [
            -0.058842508175110914,
            -0.039596459088451996,
            -0.032893774865250434,
            -0.0330278939203562,
            -0.07357521803228662,
            0.006485704444619204,
            0.01558225506749641,
            -0.01386813086944359,
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
            -164.95569697894862,
        ],
        "inverse_temperature_and_noise": [0.10081306187557834, 0.009935007000345747],
        "input_gain": [
            0.0005960370207988313,
            -0.0012995005608915603,
            -0.000265305458935273,
            0.0007794742331517101,
        ],
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_em_matches_recorded_values(case: str) -> None:
    expected = EXPECTED[case]
    rtol, atol = TOLERANCES[case]
    actual = CASES[case]()

    assert set(actual) == set(expected)
    assert len(actual["log_likelihoods"]) == len(expected["log_likelihoods"]), (
        "number of EM iterations changed"
    )
    for name, expected_value in expected.items():
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

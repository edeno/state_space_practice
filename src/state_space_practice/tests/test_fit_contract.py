"""Contracts every fitted model honors, across the model families.

- A fit that fails -- a non-finite log-likelihood at the starting parameters,
  or settings rejected before the fit starts -- leaves the model reading as
  unfitted (or untouched), never serving an earlier fit's outputs next to
  newly bound data, or non-finite ones.
- Every fit (EM ``fit`` and ``fit_sgd``) records its own results:
  ``log_likelihood_history_`` is the returned list, ``n_iter_`` counts EM
  iterations (``None`` after SGD), and ``log_likelihood_`` is the marginal
  log-likelihood at the stored parameters -- never a penalized training
  objective and never the other fitter's value.
- Refitting a model on same-shaped data reuses its compiled SGD step.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from state_space_practice.contingency_belief import (
    ContingencyBeliefModel,
    _smooth_with_filter,
)
from state_space_practice.covariate_choice import CovariateChoiceModel
from state_space_practice.exceptions import NonFiniteLikelihoodError, NotFittedError
from state_space_practice.fitted_state import is_set
from state_space_practice.hamiltonian_joint import JointHamiltonianModel
from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel
from state_space_practice.hamiltonian_spikes import HamiltonianSpikeModel
from state_space_practice.hamiltonian_switching import SwitchingHamiltonianJointModel
from state_space_practice.multinomial_choice import MultinomialChoiceModel
from state_space_practice.oscillator_models import (
    CommonOscillatorModel,
    CorrelatedNoiseModel,
    DirectedInfluenceModel,
)
from state_space_practice.oscillator_regularization import OscillatorPenaltyConfig
from state_space_practice.place_field_model import PlaceFieldModel
from state_space_practice.point_process_kalman import PointProcessModel
from state_space_practice.point_process_models import (
    CommonOscillatorPointProcessModel,
    CorrelatedNoisePointProcessModel,
    DirectedInfluencePointProcessModel,
)
from state_space_practice.simulate.scenarios import (
    simulate_cnm_pp_scenario,
    simulate_cnm_scenario,
    simulate_com_pp_scenario,
    simulate_com_scenario,
    simulate_dim_pp_scenario,
    simulate_dim_scenario,
)
from state_space_practice.simulate_data import simulate_2d_moving_place_field
from state_space_practice.smith_learning_algorithm import (
    SmithLearningModel,
    simulate_learning_data,
)
from state_space_practice.switching_choice import SwitchingChoiceModel
from state_space_practice.switching_point_process import SwitchingSpikeOscillatorModel
from state_space_practice.temporal_rate_gp import TemporalRateGP
from state_space_practice.tests.model_state import (
    assert_model_state_unchanged,
    assert_snapshots_equal,
    snapshot_model_state,
)

_RNG = np.random.default_rng(0)
_CHOICES_50 = _RNG.integers(0, 3, size=50)
_CHOICES_80 = _RNG.integers(0, 3, size=80)
_REWARDS_80 = _RNG.integers(0, 2, size=80)


def _assert_fit_results_cleared(model) -> None:
    assert not is_set(model, "log_likelihood_")
    assert not is_set(model, "log_likelihood_history_")
    assert not is_set(model, "converged_")
    assert model.n_iter_ is None


# --- choice / belief models: hand-rolled EM ------------------------------


def _multinomial():
    return MultinomialChoiceModel(n_options=3)


def _covariate():
    return CovariateChoiceModel(n_options=3, n_covariates=1)


def _switching():
    return SwitchingChoiceModel(n_options=3, n_discrete_states=2)


def _contingency():
    return ContingencyBeliefModel(n_states=2, n_options=3)


def _fit(model, choices):
    n = len(choices)
    if isinstance(model, CovariateChoiceModel):
        return model.fit(choices, covariates=np.zeros((n, 1)), max_iter=2)
    if isinstance(model, ContingencyBeliefModel):
        return model.fit(choices, _REWARDS_80[:n], max_iter=2)
    return model.fit(choices, max_iter=2)


def _poison(model) -> None:
    """Make the next E-step's log-likelihood non-finite."""
    if isinstance(model, SwitchingChoiceModel):
        model.process_noises_ = jnp.full_like(model.process_noises_, jnp.nan)
    elif isinstance(model, ContingencyBeliefModel):
        model.reward_probs_ = jnp.full_like(model.reward_probs_, jnp.nan)
    else:
        model.inverse_temperature = float("nan")


_CHOICE_MODELS = [_multinomial, _covariate, _switching, _contingency]


@pytest.mark.slow
@pytest.mark.parametrize("make_model", _CHOICE_MODELS)
def test_nonfinite_refit_leaves_no_stale_results(make_model):
    """A refit whose first E-step is non-finite raises and leaves the model
    unfitted -- not holding the previous fit's results for other data."""
    model = make_model()
    _fit(model, _CHOICES_50)
    assert model.is_fitted  # guard: there is an earlier fit to go stale

    _poison(model)
    with pytest.raises(NonFiniteLikelihoodError):
        _fit(model, _CHOICES_80)

    assert not model.is_fitted
    _assert_fit_results_cleared(model)


@pytest.mark.slow
@pytest.mark.parametrize("make_model", _CHOICE_MODELS)
def test_em_rejects_max_iter_below_one_before_binding_data(make_model):
    model = make_model()
    _fit(model, _CHOICES_50)
    before = snapshot_model_state(model)

    with pytest.raises(ValueError, match="max_iter"):
        if isinstance(model, ContingencyBeliefModel):
            model.fit(_CHOICES_80, _REWARDS_80, max_iter=0)
        elif isinstance(model, CovariateChoiceModel):
            model.fit(_CHOICES_80, covariates=np.zeros((80, 1)), max_iter=0)
        else:
            model.fit(_CHOICES_80, max_iter=0)

    assert_model_state_unchanged(model, before)


def test_fit_sgd_from_unusable_start_raises_and_clears():
    """fit_sgd whose stored parameters give a non-finite log-likelihood raises
    instead of returning normally with log_likelihood_ = NaN."""
    model = _multinomial()
    model.fit_sgd(_CHOICES_50, num_steps=2)
    assert model.is_fitted  # guard

    model.inverse_temperature = 1e300
    with pytest.raises(NonFiniteLikelihoodError):
        model.fit_sgd(_CHOICES_80, num_steps=2)

    assert not model.is_fitted
    _assert_fit_results_cleared(model)


# --- switching point-process models: run_em with the "raise" policy -------


def _com_pp():
    return CommonOscillatorPointProcessModel(
        n_oscillators=2,
        n_neurons=5,
        n_discrete_states=2,
        sampling_freq=100.0,
        dt=0.01,
        freqs=jnp.array([8.0, 12.0]),
        damping_coef=jnp.array([0.95, 0.95]),
        process_variance=jnp.array([0.1, 0.1]),
    )


_SPIKES = jax.random.poisson(jax.random.PRNGKey(1), 0.05, (60, 5))


def _nonfinite_e_step(model, monkeypatch) -> None:
    e_step = model._e_step

    def bad_e_step(spikes):
        e_step(spikes)  # installs (finite or not) posteriors, as a real failure would
        return jnp.nan

    monkeypatch.setattr(model, "_e_step", bad_e_step)


@pytest.mark.slow
def test_switching_pp_nonfinite_first_e_step_clears_outputs(monkeypatch):
    model = _com_pp()
    _nonfinite_e_step(model, monkeypatch)

    with pytest.raises(NonFiniteLikelihoodError):
        model.fit(_SPIKES, max_iter=2, key=jax.random.PRNGKey(0))

    assert not is_set(model, "smoother_discrete_state_prob")
    with pytest.raises(NotFittedError):
        model.decode()
    _assert_fit_results_cleared(model)


@pytest.mark.slow
def test_switching_pp_all_restarts_failing_raises_nonfinite_error(monkeypatch):
    model = _com_pp()
    _nonfinite_e_step(model, monkeypatch)

    with pytest.raises(NonFiniteLikelihoodError, match="All 2 restarts"):
        model.fit(_SPIKES, max_iter=2, key=jax.random.PRNGKey(0), n_restarts=2)

    assert not is_set(model, "smoother_discrete_state_prob")
    _assert_fit_results_cleared(model)


@pytest.mark.slow
def test_switching_pp_restarts_report_the_best_restart(monkeypatch):
    """With several restarts, every recorded fit result -- history, n_iter_,
    converged_ and log_likelihood_ -- belongs to the kept (best) restart, not
    to whichever restart ran last."""
    model = _com_pp()
    fit_single = model._fit_single
    restarts: list[tuple[list[float], bool]] = []

    def observed_fit_single(*args, **kwargs):
        lls = fit_single(*args, **kwargs)
        # Give the restarts distinct convergence flags, so reporting another
        # restart's flag shows whatever the real fits concluded.
        model.converged_ = len(restarts) % 2 == 0
        restarts.append((list(lls), model.converged_))
        return lls

    monkeypatch.setattr(model, "_fit_single", observed_fit_single)
    # Key chosen so the first of the two restarts ends best (guarded below).
    lls = model.fit(_SPIKES, max_iter=3, key=jax.random.PRNGKey(5), n_restarts=2)

    final_lls = [restart_lls[-1] for restart_lls, _ in restarts]
    best = int(np.argmax(final_lls))
    # Guard: the best restart is not the last one, and the two restarts end
    # at different log-likelihoods, so last-restart results would show.
    assert best != len(restarts) - 1
    assert not np.isclose(final_lls[best], final_lls[-1])

    best_lls, best_converged = restarts[best]
    assert lls == best_lls
    assert model.log_likelihood_history_ == best_lls
    assert model.n_iter_ == len(best_lls)
    assert model.converged_ is best_converged
    # The best state is restored and re-scored, which reproduces its final LL.
    assert model.log_likelihood_ == pytest.approx(final_lls[best], rel=1e-10)
    assert model.log_likelihood_ == pytest.approx(
        float(model._e_step(_SPIKES)), rel=1e-12
    )


# --- shared model cases --------------------------------------------------


@dataclass(frozen=True)
class FitCase:
    """A fresh model, a small dataset for it, and how to fit and score it.

    Attributes
    ----------
    model : Any
        The fresh model.
    data : tuple
        Positional data arguments of ``fit`` / ``fit_sgd``, time (or trial)
        axis first.
    score : Callable or None
        ``score(model, data, data_kwargs)`` re-evaluates the marginal
        log-likelihood of ``data`` at the model's stored parameters with the
        model's own filter / E-step (no penalty term).
    data_kwargs : dict
        Keyword data / settings both fitters take (covariates, ``key``).
    sgd_kwargs, em_kwargs : dict
        Keyword arguments only ``fit_sgd`` / only ``fit`` takes.
    em_ll_is_last_history : bool
        After EM, ``log_likelihood_`` is the last history entry (the run_em
        models and the choice models whose final E-step joins the history).
    resumes : bool
        A repeat ``fit_sgd`` (with ``repeat_kwargs``) continues from the
        stored parameters instead of re-initializing them.
    repeat_kwargs : dict
        Extra ``fit_sgd`` arguments for the repeat-fit step-cache test.
    binding_guard : Callable or None
        Asserts that binding this case's data would change the fresh model
        beyond recording its length (so the fresh-model rejection test
        checks more than that).
    """

    model: Any
    data: tuple[Any, ...]
    score: Callable[[Any, tuple[Any, ...], dict[str, Any]], float] | None
    data_kwargs: dict[str, Any] = field(default_factory=dict)
    sgd_kwargs: dict[str, Any] = field(default_factory=dict)
    em_kwargs: dict[str, Any] = field(default_factory=dict)
    em_ll_is_last_history: bool = True
    resumes: bool = True
    repeat_kwargs: dict[str, Any] = field(default_factory=dict)
    binding_guard: Callable[[Any], None] | None = None

    @property
    def n_time(self) -> int:
        return int(np.shape(self.data[0])[0])

    def truncated(self, n_time: int) -> "FitCase":
        """The same model with every time-indexed input cut to ``n_time``."""

        def cut(value: Any) -> Any:
            if isinstance(value, (jax.Array, np.ndarray)) and value.ndim >= 1:
                if value.shape[0] == self.n_time:
                    return value[:n_time]
            return value

        def cut_all(kwargs: dict[str, Any]) -> dict[str, Any]:
            return {k: cut(v) for k, v in kwargs.items()}

        return replace(
            self,
            data=tuple(cut(v) for v in self.data),
            data_kwargs=cut_all(self.data_kwargs),
            sgd_kwargs=cut_all(self.sgd_kwargs),
            em_kwargs=cut_all(self.em_kwargs),
        )

    def fit(self, **kwargs: Any) -> list[float]:
        return self.model.fit(
            *self.data, **self.data_kwargs, **self.em_kwargs, **kwargs
        )

    def fit_sgd(self, **kwargs: Any) -> list[float]:
        return self.model.fit_sgd(
            *self.data, **self.data_kwargs, **self.sgd_kwargs, **kwargs
        )

    def marginal_log_likelihood(self) -> float:
        assert self.score is not None
        return float(self.score(self.model, self.data, self.data_kwargs))


_KEY = jax.random.PRNGKey(0)


def _score_e_step(model: Any, data: tuple[Any, ...], _: dict[str, Any]) -> float:
    return float(model._e_step(*(jnp.asarray(d) for d in data)))


def _choices(data: tuple[Any, ...]) -> jax.Array:
    return jnp.asarray(data[0], dtype=jnp.int32)


def _score_choice_smoother(model: Any, data: tuple[Any, ...], _: Any) -> float:
    return float(model._run_smoother(_choices(data)).marginal_log_likelihood)


def _score_switching_choice(model: Any, data: tuple[Any, ...], _: Any) -> float:
    result = model._run_filter(_choices(data), model._covariates, model._obs_covariates)
    return float(result.marginal_log_likelihood)


def _contingency_smoother(model: Any, data: tuple[Any, ...]) -> Any:
    choices = _choices(data)
    rewards = jnp.asarray(data[1], dtype=jnp.int32)
    return _smooth_with_filter(
        **model._smoother_kwargs(
            choices,
            rewards,
            model._transition_design_matrix,
            model._obs_design_matrix,
        )
    )


def _score_contingency(model: Any, data: tuple[Any, ...], _: Any) -> float:
    return float(_contingency_smoother(model, data).log_likelihood)


def _score_hamiltonian(model: Any, data: tuple[Any, ...], _: Any) -> float:
    marginal_lls = model.filter(*data, model._build_param_spec()[0])[-1]
    return float(jnp.sum(marginal_lls))


def _oscillator_case(name: str) -> FitCase:
    if name == "COM":
        scenario = simulate_com_scenario(n_time=50, seed=42)
        p = scenario["params"]
        model = CommonOscillatorModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            n_sources=p["n_sources"],
            sampling_freq=p["sampling_freq"],
            freqs=p["freqs"],
            damping_coef=p["damping"],
            process_variance=p["process_variance"],
            measurement_variance=p["measurement_variance"],
        )
    else:
        cls, simulate = {
            "CNM": (CorrelatedNoiseModel, simulate_cnm_scenario),
            "DIM": (DirectedInfluenceModel, simulate_dim_scenario),
        }[name]
        scenario = simulate(n_time=50, seed=42)
        p = scenario["params"]
        model = cls(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            freqs=p["freqs"],
            damping_coef=p["damping"],
            process_variance=p["process_variance"],
            measurement_variance=p["measurement_variance"],
            phase_difference=p["phase_difference"],
            coupling_strength=p["coupling_strength"],
        )
    # fit_sgd re-initializes from ``key`` unless skip_init=True.
    return FitCase(
        model,
        (jnp.asarray(scenario["obs"]),),
        _score_e_step,
        data_kwargs={"key": _KEY},
        resumes=False,
    )


def _point_process_case(name: str, **model_kwargs: Any) -> FitCase:
    simulate = {
        "COM-PP": simulate_com_pp_scenario,
        "CNM-PP": simulate_cnm_pp_scenario,
        "DIM-PP": simulate_dim_pp_scenario,
    }[name]
    scenario = simulate(n_time=50, seed=42)
    p = scenario["params"]
    kwargs = {
        "n_oscillators": p["n_oscillators"],
        "n_neurons": p["n_neurons"],
        "n_discrete_states": p["n_discrete_states"],
        "sampling_freq": p["sampling_freq"],
        "dt": p["dt"],
        "freqs": p["freqs"],
        "damping_coef": p["damping"],
        "process_variance": p["process_variance"],
        **model_kwargs,
    }
    if name == "COM-PP":
        model = CommonOscillatorPointProcessModel(**kwargs)
    else:
        cls = (
            CorrelatedNoisePointProcessModel
            if name == "CNM-PP"
            else DirectedInfluencePointProcessModel
        )
        model = cls(
            **kwargs,
            phase_difference=p["phase_difference"],
            coupling_strength=p["coupling_strength"],
        )
    return FitCase(
        model, (scenario["spikes"],), _score_e_step, data_kwargs={"key": _KEY}
    )


def _switching_spike_oscillator_case() -> FitCase:
    model = SwitchingSpikeOscillatorModel(
        n_oscillators=2,
        n_neurons=4,
        n_discrete_states=2,
        sampling_freq=100.0,
        dt=0.01,
    )
    spikes = jax.random.poisson(jax.random.PRNGKey(42), jnp.full((50, 4), 0.5))
    return FitCase(model, (spikes,), _score_e_step, data_kwargs={"key": _KEY})


def _point_process_kalman_case() -> FitCase:
    key = jax.random.PRNGKey(42)
    n_time, n_neurons, n_state = 100, 3, 2
    weights = jax.random.normal(key, (n_neurons, n_state)) * 0.1
    design_matrix = jnp.tile(weights, (n_time, 1, 1))
    spikes = jax.random.poisson(key, jnp.full((n_time, n_neurons), 0.01))
    return FitCase(
        PointProcessModel(n_state, 0.001), (design_matrix, spikes), _score_e_step
    )


def _place_field_case() -> FitCase:
    sim = simulate_2d_moving_place_field(
        total_time=10.0,
        dt=0.020,
        arena_size=80.0,
        peak_rate=25.0,
        background_rate=1.0,
        n_interior_knots=3,
        rng=np.random.default_rng(42),
    )
    return FitCase(
        PlaceFieldModel(dt=sim["dt"], n_interior_knots=3),
        (sim["position"], sim["spikes"]),
        lambda model, data, _: model.score(*data),
        # One optimizer instance: the compiled-step cache keys on it.
        sgd_kwargs={"optimizer": optax.adam(1e-3)},
        em_kwargs={"verbose": False},
        # fit_sgd warm-starts (re-initializes) by default; without it each
        # call resumes from the trained parameters.
        repeat_kwargs={"warm_start": False},
    )


def _graph_place_field_case() -> FitCase:
    neurospatial = pytest.importorskip("neurospatial")
    from state_space_practice.graph_place_field import GraphPlaceFieldModel

    rng = np.random.default_rng(42)
    env = neurospatial.Environment.from_samples(
        rng.uniform(0, 20, (400, 2)), bin_size=5.0
    )
    times = np.arange(40, dtype=float) * 0.02
    trajectory = env.bin_centers[rng.integers(env.n_bins, size=40)]
    spikes = rng.poisson(0.2, size=40).astype(float)
    return FitCase(
        GraphPlaceFieldModel(env, dt=0.02, rank=4),
        (times, trajectory, spikes),
        lambda model, data, _: model.score(*data),
        sgd_kwargs={"optimizer": optax.adam(1e-3)},
        em_kwargs={"verbose": False},
        repeat_kwargs={"warm_start": False},
    )


def _smith_case() -> FitCase:
    outcomes, _ = simulate_learning_data(
        n_trials=100, prob_success_init=0.3, prob_success_final=0.9, seed=42
    )
    return FitCase(SmithLearningModel(sigma_epsilon=0.1), (outcomes,), _score_e_step)


def _multinomial_case() -> FitCase:
    choices = np.random.default_rng(0).integers(0, 3, size=60)
    return FitCase(
        MultinomialChoiceModel(n_options=3),
        (choices,),
        _score_choice_smoother,
        # Without convergence the final E-step after the last M-step is
        # scored but not added to the history.
        em_ll_is_last_history=False,
    )


def _covariate_case() -> FitCase:
    rng = np.random.default_rng(1)
    n_trials = 60
    model = CovariateChoiceModel(
        n_options=3,
        n_covariates=2,
        n_obs_covariates=2,
        init_decay=0.9,
        learn_decay=True,
    )
    return FitCase(
        model,
        (rng.integers(0, 3, size=n_trials),),
        _score_choice_smoother,
        data_kwargs={
            "covariates": rng.standard_normal((n_trials, 2)),
            "obs_covariates": rng.standard_normal((n_trials, 2)),
        },
    )


def _switching_choice_case() -> FitCase:
    rng = np.random.default_rng(7)
    n_trials = 40
    model = SwitchingChoiceModel(
        n_options=3, n_discrete_states=2, n_covariates=2, n_obs_covariates=1
    )
    data = (
        rng.integers(0, 3, size=n_trials),
        rng.normal(size=(n_trials, 2)),
        rng.normal(size=(n_trials, 1)),
    )
    # Without convergence log_likelihood_ is the final filter's, after the
    # last M-step; the history ends before it.
    return FitCase(model, data, _score_switching_choice, em_ll_is_last_history=False)


def _stay_obs_design(choices: np.ndarray, n_options: int = 3) -> np.ndarray:
    """Previous-choice indicator, shape (n_trials, n_options)."""
    obs = np.zeros((len(choices), n_options))
    obs[np.arange(1, len(choices)), choices[:-1]] = 1.0
    return obs


def _contingency_case(with_covariates: bool = False) -> FitCase:
    rng = np.random.default_rng(3)
    n_trials = 60
    choices = rng.integers(0, 3, size=n_trials)
    rewards = rng.integers(0, 2, size=n_trials)
    if not with_covariates:
        return FitCase(
            ContingencyBeliefModel(n_states=2, n_options=3),
            (choices, rewards),
            _score_contingency,
        )
    covariates = np.zeros((n_trials, 1))
    covariates[n_trials // 2, 0] = 1.0

    def pads_coefficients(model: Any) -> None:
        # These covariates add a coefficient row, so binding them pads
        # transition_coefficients_.
        assert model.transition_coefficients_.shape[0] == 1

    return FitCase(
        ContingencyBeliefModel(n_states=2, n_options=3, n_obs_covariates=3),
        (choices, rewards),
        _score_contingency,
        data_kwargs={"transition_covariates": covariates},
        # EM has no observation covariates; fit_sgd learns their weights.
        sgd_kwargs={"obs_design_matrix": _stay_obs_design(choices)},
        binding_guard=pads_coefficients,
    )


def _temporal_rate_gp_case() -> FitCase:
    n_time, dt = 24, 0.1
    times = jnp.arange(n_time) * dt
    true_rate = jnp.exp(0.7 * jnp.sin(2.0 * jnp.pi * times / 1.5))
    counts = jax.random.poisson(jax.random.PRNGKey(0), true_rate * dt).astype(float)
    return FitCase(TemporalRateGP(dt=dt), (counts,), None)


def _hamiltonian_case(name: str) -> FitCase:
    k_lfp, k_spikes = jax.random.split(jax.random.PRNGKey(42))
    n_time = 50
    if name == "LFP":
        model = HamiltonianLFPModel(
            n_sources=4,
            n_oscillators=1,
            hidden_dims=[16],
            seed=0,
            sampling_freq=1000.0,
        )
        data = (jax.random.normal(k_lfp, (n_time, 4)),)
    elif name == "spikes":
        model = HamiltonianSpikeModel(
            n_sources=8,
            n_oscillators=1,
            hidden_dims=[16],
            seed=0,
            sampling_freq=1000.0,
        )
        data = (jax.random.poisson(k_spikes, jnp.full((n_time, 8), 0.5)),)
    elif name == "joint":
        model = JointHamiltonianModel(
            n_lfp_sources=4,
            n_spike_sources=8,
            n_oscillators=1,
            hidden_dims=[16],
            seed=0,
            sampling_freq=1000.0,
        )
        data = (
            jax.random.normal(k_lfp, (n_time, 4)),
            jax.random.poisson(k_spikes, jnp.full((n_time, 8), 0.5)),
        )
    else:
        model = SwitchingHamiltonianJointModel(
            n_oscillators=1,
            n_discrete_states=2,
            n_lfp_sources=2,
            n_spike_sources=3,
            sampling_freq=100.0,
            hidden_dims=[8, 8],
            seed=0,
        )
        data = (
            jax.random.normal(k_lfp, (20, 2)),
            jax.random.poisson(k_spikes, 0.5, (20, 3)).astype(float),
        )
    return FitCase(model, data, _score_hamiltonian)


#: Every concrete model class with an EM ``fit`` and ``fit_sgd``.
_EM_AND_SGD_CASES: dict[str, Callable[[], FitCase]] = {
    "CommonOscillatorModel": lambda: _oscillator_case("COM"),
    "CorrelatedNoiseModel": lambda: _oscillator_case("CNM"),
    "DirectedInfluenceModel": lambda: _oscillator_case("DIM"),
    "CommonOscillatorPointProcessModel": lambda: _point_process_case("COM-PP"),
    "CorrelatedNoisePointProcessModel": lambda: _point_process_case("CNM-PP"),
    "DirectedInfluencePointProcessModel": lambda: _point_process_case("DIM-PP"),
    "SwitchingSpikeOscillatorModel": _switching_spike_oscillator_case,
    "PointProcessModel": _point_process_kalman_case,
    "PlaceFieldModel": _place_field_case,
    "GraphPlaceFieldModel": _graph_place_field_case,
    "SmithLearningModel": _smith_case,
    "MultinomialChoiceModel": _multinomial_case,
    "CovariateChoiceModel": _covariate_case,
    "SwitchingChoiceModel": _switching_choice_case,
    "ContingencyBeliefModel": _contingency_case,
    "ContingencyBeliefModel-covariates": lambda: _contingency_case(True),
}

#: Every concrete model class with ``fit_sgd`` (and some setting variants).
_SGD_CASES: dict[str, Callable[[], FitCase]] = {
    **_EM_AND_SGD_CASES,
    "TemporalRateGP": _temporal_rate_gp_case,
    "HamiltonianLFPModel": lambda: _hamiltonian_case("LFP"),
    "HamiltonianSpikeModel": lambda: _hamiltonian_case("spikes"),
    "JointHamiltonianModel": lambda: _hamiltonian_case("joint"),
    "SwitchingHamiltonianJointModel": lambda: _hamiltonian_case("switching"),
}

#: The oscillator and switching point-process families: a single
#: ``(n_time, n_channels)`` data array whose channel count is validated.
_OSCILLATOR_FAMILY = [
    "CommonOscillatorModel",
    "CorrelatedNoiseModel",
    "DirectedInfluenceModel",
    "CommonOscillatorPointProcessModel",
    "CorrelatedNoisePointProcessModel",
    "DirectedInfluencePointProcessModel",
    "SwitchingSpikeOscillatorModel",
]


# --- fit results follow the latest fitter ---------------------------------


def _assert_em_results(case: FitCase, lls: list[float]) -> float:
    model = case.model
    assert model.log_likelihood_history_ == lls
    assert model.n_iter_ == len(lls)
    assert model.converged_ in (True, False)
    if case.em_ll_is_last_history:
        assert model.log_likelihood_ == lls[-1]
    assert model.log_likelihood_ == pytest.approx(
        case.marginal_log_likelihood(), rel=1e-9
    )
    return model.log_likelihood_


def _assert_sgd_results(case: FitCase, lls: list[float]) -> float:
    model = case.model
    assert model.log_likelihood_history_ == lls
    assert model.n_iter_ is None  # n_iter_ counts EM iterations only
    assert model.converged_ is False  # no convergence_tol: SGD cannot converge
    assert model.log_likelihood_ == pytest.approx(
        case.marginal_log_likelihood(), rel=1e-9
    )
    return model.log_likelihood_


@pytest.mark.slow
@pytest.mark.parametrize("case_id", list(_EM_AND_SGD_CASES))
def test_fit_results_follow_the_latest_fitter(case_id: str) -> None:
    """fit -> fit_sgd -> fit: each records its own history, n_iter_,
    converged_ and the marginal log-likelihood at its stored parameters, so
    switching fitters never leaves the other fitter's values behind."""
    case = _EM_AND_SGD_CASES[case_id]()

    em_ll = _assert_em_results(case, case.fit(max_iter=2))
    sgd_ll = _assert_sgd_results(case, case.fit_sgd(num_steps=3))
    # Guard: the fitters end at different values, so a stale one would show.
    assert sgd_ll != em_ll
    refit_ll = _assert_em_results(case, case.fit(max_iter=2))
    assert refit_ll != sgd_ll


_PENALIZED_SGD_CASES: dict[str, Callable[[], FitCase]] = {
    "DirectedInfluenceModel-edge_l1": lambda: replace(
        _oscillator_case("DIM"),
        sgd_kwargs={"connectivity_penalty": OscillatorPenaltyConfig(edge_l1=5.0)},
    ),
    "DirectedInfluencePointProcessModel-edge_l1": lambda: replace(
        _point_process_case("DIM-PP"),
        sgd_kwargs={"connectivity_penalty": OscillatorPenaltyConfig(edge_l1=5.0)},
    ),
    "CommonOscillatorPointProcessModel-spike_weight_l2": lambda: _point_process_case(
        "COM-PP", spike_weight_l2=10.0
    ),
    **{
        f"{case_id}-l2_reg": (
            lambda case_id=case_id: replace(
                _SGD_CASES[case_id](), sgd_kwargs={"l2_reg": 10.0}
            )
        )
        for case_id in (
            "HamiltonianLFPModel",
            "HamiltonianSpikeModel",
            "JointHamiltonianModel",
            "SwitchingHamiltonianJointModel",
        )
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("case_id", list(_PENALIZED_SGD_CASES))
def test_penalized_fit_sgd_records_the_marginal_log_likelihood(case_id: str) -> None:
    """log_likelihood_ after a penalized fit_sgd is the marginal log-likelihood
    at the fitted parameters, not the penalized training objective recorded in
    log_likelihood_history_."""
    case = _PENALIZED_SGD_CASES[case_id]()

    objective = case.fit_sgd(num_steps=3)

    marginal_ll = case.marginal_log_likelihood()
    # Guard: the penalty makes the objective differ from the likelihood.
    assert not np.isclose(objective[-1], marginal_ll)
    assert case.model.log_likelihood_ == pytest.approx(marginal_ll, rel=1e-12)
    assert case.model.log_likelihood_history_ == objective


def _assert_trees_equal(stored: Any, fresh: Any) -> None:
    stored_leaves, stored_def = jax.tree_util.tree_flatten(stored)
    fresh_leaves, fresh_def = jax.tree_util.tree_flatten(fresh)
    assert stored_def == fresh_def
    for s, f in zip(stored_leaves, fresh_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(s), np.asarray(f))


def _stored_and_fresh_posteriors(case: FitCase) -> tuple[Any, Any]:
    """The model's stored E-step outputs and a fresh run at its parameters."""
    model = case.model
    if isinstance(model, SwitchingChoiceModel):
        fresh_filter = model._run_filter(
            _choices(case.data), model._covariates, model._obs_covariates
        )
        fresh_smoother = model._run_smoother(fresh_filter)
        stored = (
            model._filter_result,
            model.smoothed_discrete_probs_,
            model._smoother_state_cond_means,
            model._smoother_state_cond_covs,
        )
        fresh = (
            fresh_filter,
            fresh_smoother.smoother_discrete_state_prob,
            fresh_smoother.state_cond_smoother_means,
            fresh_smoother.state_cond_smoother_covs,
        )
        return stored, fresh
    if isinstance(model, ContingencyBeliefModel):
        fresh = _contingency_smoother(model, case.data)
        stored = (
            model._smoother_result,
            model.smoothed_state_posterior_,
            model.state_posterior_,
        )
        return stored, (fresh, fresh.smoothed_state_prob, fresh.filtered_state_prob)
    return model._smoother_result, model._run_smoother(_choices(case.data))


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_id",
    [
        "MultinomialChoiceModel",
        "CovariateChoiceModel",
        "SwitchingChoiceModel",
        "ContingencyBeliefModel",
    ],
)
def test_converged_em_keeps_the_e_step_at_the_final_parameters(case_id: str) -> None:
    """On convergence no M-step follows the last E-step, so EM reuses it: the
    stored posteriors and log_likelihood_ are exactly a fresh E-step's at the
    final parameters."""
    case = _EM_AND_SGD_CASES[case_id]()

    # Any change passes so huge a tolerance: EM converges at its second E-step.
    lls = case.fit(max_iter=10, tolerance=1e12)

    assert case.model.converged_ is True
    # Guard: one M-step ran between the two E-steps and moved the parameters,
    # so posteriors from before it would differ from a fresh run.
    assert len(lls) == 2 and lls[0] != lls[1]
    assert case.model.log_likelihood_ == lls[-1]
    assert case.model.log_likelihood_ == case.marginal_log_likelihood()
    _assert_trees_equal(*_stored_and_fresh_posteriors(case))


# --- rejected fit_sgd leaves the model unchanged ---------------------------


@pytest.mark.parametrize("case_id", list(_SGD_CASES))
def test_rejected_fit_sgd_leaves_fresh_model_unchanged(case_id: str) -> None:
    """Invalid settings fail before the data is bound or parameters are
    initialized: every rejection comes before the first mutation."""
    case = _SGD_CASES[case_id]()
    if case.binding_guard is not None:
        case.binding_guard(case.model)
    before = snapshot_model_state(case.model)

    with pytest.raises(ValueError, match="num_steps"):
        case.fit_sgd(num_steps=-1)

    assert_model_state_unchanged(case.model, before)


def _prefit(case: FitCase, method: str) -> None:
    if method == "fit":
        case.fit(max_iter=2)
    else:
        case.fit_sgd(num_steps=2)
    assert is_set(case.model, "log_likelihood_")  # guard: a fit to preserve


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case_id", "prefit"),
    [(case_id, "fit_sgd") for case_id in _SGD_CASES]
    + [(case_id, "fit") for case_id in _EM_AND_SGD_CASES],
)
def test_rejected_fit_sgd_leaves_fitted_model_unchanged(
    case_id: str, prefit: str
) -> None:
    """A rejected fit_sgd on a fitted model keeps the previous fit: no data
    rebinding, no re-initialization or warm start over fitted parameters."""
    case = _SGD_CASES[case_id]()
    _prefit(case, prefit)
    before = snapshot_model_state(case.model)

    # Shorter (otherwise valid) data, so recording its length would show.
    with pytest.raises(ValueError, match="num_steps"):
        case.truncated(case.n_time - case.n_time // 3).fit_sgd(num_steps=-1)

    assert_model_state_unchanged(case.model, before)


@pytest.mark.slow
@pytest.mark.parametrize("case_id", _OSCILLATOR_FAMILY)
def test_fit_sgd_rejecting_data_shape_leaves_fitted_model_unchanged(
    case_id: str,
) -> None:
    case = _SGD_CASES[case_id]()
    _prefit(case, "fit_sgd")
    # Guard: the fit left posteriors that a re-initialization would drop.
    assert is_set(case.model, "smoother_discrete_state_prob")
    before = snapshot_model_state(case.model)

    with pytest.raises(ValueError):
        replace(case, data=(case.data[0][:, :-1],)).fit_sgd(num_steps=2)

    assert_model_state_unchanged(case.model, before)


def test_fit_sgd_missing_obs_design_matrix_leaves_model_unchanged() -> None:
    case = _SGD_CASES["ContingencyBeliefModel-covariates"]()
    before = snapshot_model_state(case.model)

    with pytest.raises(ValueError, match="obs_design_matrix"):
        replace(case, sgd_kwargs={}).fit_sgd(num_steps=2)

    assert_model_state_unchanged(case.model, before)


# --- repeat fit_sgd reuses the compiled step -------------------------------


def _trained_parameters(model: Any) -> list[np.ndarray]:
    params, spec = model._build_param_spec()
    return [
        np.asarray(leaf)
        for name in sorted(params)
        if spec[name].trainable
        for leaf in jax.tree_util.tree_leaves(params[name])
    ]


def _parameters_changed(before: list[np.ndarray], after: list[np.ndarray]) -> bool:
    return any(not np.array_equal(b, a) for b, a in zip(before, after, strict=True))


def _repeat_fit_sgd(
    case: FitCase, builds: list[None], calls: list[dict[str, Any]]
) -> tuple[list[int], list[list[np.ndarray]], list[list[float]]]:
    """Run one ``fit_sgd`` per entry of ``calls`` (its extra keyword arguments).

    Returns the SGD steps each call compiled (``builds`` is the
    ``sgd_step_builds`` fixture), the trained parameters after each call and
    each call's history.
    """
    per_call, trained, histories = [], [], []
    for call_kwargs in calls:
        n_before = len(builds)
        histories.append(case.fit_sgd(num_steps=2, **call_kwargs))
        per_call.append(len(builds) - n_before)
        trained.append(_trained_parameters(case.model))
    return per_call, trained, histories


@pytest.mark.slow
@pytest.mark.parametrize("case_id", list(_SGD_CASES))
def test_repeat_fit_sgd_reuses_compiled_step(
    case_id: str, sgd_step_builds: list[None]
) -> None:
    """Refitting a model on same-shaped data reuses its compiled SGD step.

    Every fit rewrites the trained attributes, so a loss that read them while
    tracing (instead of taking them from the optimized parameters) would miss
    the model's step cache and recompile on every later call. The first call
    must compile, which also shows the counter sees the builds.
    """
    case = _SGD_CASES[case_id]()
    builds, trained, histories = _repeat_fit_sgd(
        case, sgd_step_builds, [case.repeat_kwargs] * 3
    )
    assert builds == [1, 0, 0]
    if case.resumes:
        # Guard: each refit started from, and changed, the trained parameters
        # the step cache must not depend on.
        assert _parameters_changed(trained[0], trained[1])
        assert _parameters_changed(trained[1], trained[2])
        # The reused step continues optimizing from the stored parameters.
        assert histories[2][-1] > histories[0][0]


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_id",
    ["CommonOscillatorModel", "CorrelatedNoiseModel", "DirectedInfluenceModel"],
)
def test_repeat_fit_sgd_resuming_from_fitted_parameters(
    case_id: str, sgd_step_builds: list[None]
) -> None:
    # skip_init=True continues from the previous fit's parameters instead of
    # re-initializing them, so every trained attribute has changed.
    case = _SGD_CASES[case_id]()
    resume = {"skip_init": True}
    builds, trained, _ = _repeat_fit_sgd(case, sgd_step_builds, [{}, resume, resume])
    assert builds == [1, 0, 0]
    assert _parameters_changed(trained[0], trained[1])  # guard
    assert _parameters_changed(trained[1], trained[2])


# --- PRNG keys: typed (jax.random.key) and legacy (PRNGKey) both accepted ----


@pytest.mark.slow
@pytest.mark.parametrize("method", ["fit", "fit_sgd"])
def test_typed_and_legacy_keys_give_identical_fits(method):
    """The library uses typed keys internally but accepts legacy uint32 keys;
    both encode the same stream, so a fit is identical either way."""
    fitted = []
    for key in (jax.random.key(3), jax.random.PRNGKey(3)):
        model = _com_pp()
        if method == "fit":
            model.fit(_SPIKES, max_iter=2, key=key)
        else:
            model.fit_sgd(_SPIKES, key=key, num_steps=2)
        fitted.append(snapshot_model_state(model))
    assert_snapshots_equal(*fitted)

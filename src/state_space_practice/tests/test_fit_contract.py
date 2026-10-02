"""Contracts every fitted model honors when a fit cannot produce a result.

A fit that fails -- a non-finite log-likelihood at the starting parameters, or
settings rejected before the fit starts -- must leave the model reading as
unfitted (or untouched), never serving an earlier fit's outputs next to newly
bound data, or non-finite ones.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.contingency_belief import ContingencyBeliefModel
from state_space_practice.covariate_choice import CovariateChoiceModel
from state_space_practice.exceptions import NonFiniteLikelihoodError, NotFittedError
from state_space_practice.fitted_state import is_set
from state_space_practice.multinomial_choice import MultinomialChoiceModel
from state_space_practice.point_process_models import (
    CommonOscillatorPointProcessModel,
)
from state_space_practice.switching_choice import SwitchingChoiceModel
from state_space_practice.tests.model_state import (
    assert_model_state_unchanged,
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

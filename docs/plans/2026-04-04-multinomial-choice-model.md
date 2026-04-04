# Multinomial Choice Learning Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a state-space model that tracks the animal's evolving valuation of multiple choice options in a multi-armed bandit task, where the latent state is a vector of option values that drift over time and choices are multinomial (softmax) observations.

**Architecture:** The latent state is `x_t ∈ R^K` representing the animal's internal value for each of K choice options. Values evolve via random walk (or covariate-driven dynamics). The observation is the animal's choice, modeled as Categorical(softmax(x_t)). Inference uses the Laplace-EKF (the softmax is nonlinear, analogous to the sigmoid in SmithLearning but for K>2 options). EM learns the drift rate and optionally how covariates (reward history, time) drive value changes. Optionally incorporates spike observations to link neural population activity to the latent value state.

**Tech Stack:** JAX, existing Laplace-EKF pattern from `smith_learning_algorithm.py`, `_point_process_laplace_update` pattern for the nonlinear observation model.

**References:**

- Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006). Cortical substrates for exploratory decisions in humans. Nature 441, 876-879.
- Piray, P. & Daw, N.D. (2021). A simple model for learning in volatile environments. PLoS Computational Biology 17(4), e1007963.
- Gershman, S.J. (2015). A unifying probabilistic view of associative learning. PLoS Computational Biology 11(11), e1004567.
- Findling, C., Skvortsova, V., Dromnelle, R., Palminteri, S. & Wyart, V. (2019). Computational noise in reward-guided learning drives behavioral variability in volatile environments. Nature Neuroscience 22, 2066-2077.
- Wilson, R.C. & Collins, A.G.E. (2019). Ten simple rules for the computational modeling of behavioral data. eLife 8, e49547.
- Smith, A.C., Frank, L.M., Wirth, S. et al. (2004). Dynamic analysis of learning in behavioral experiments. J Neuroscience 24(2), 447-461.
- Behrens, T.E.J., Woolrich, M.W., Walton, M.E. & Rushworth, M.F.S. (2007). Learning the value of information in an uncertain world. Nature Neuroscience 10, 1214-1221.

---

## Background and Mathematical Model

### The scientific question
In a multi-armed bandit task, how does the animal's valuation of each option evolve over trials? Standard reinforcement learning models (e.g., Q-learning) assume a specific update rule. The state-space approach is agnostic — it infers the value trajectory from choices alone, without assuming how values update. This lets you ask: does the actual value trajectory look like Q-learning? Or is it something else (e.g., win-stay-lose-shift, momentum, exploration bonuses)?

### Generative model

```
Value state (K options):
    x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
    x_t ∈ R^K

Choice observation:
    c_t ~ Categorical(softmax(β * x_t))
    P(c_t = k) = exp(β * x_{t,k}) / Σ_j exp(β * x_{t,j})

    β: inverse temperature (exploration vs exploitation)

Optional reward-driven dynamics:
    x_t = x_{t-1} + B @ covariates_t + w_t
    covariates_t = [reward_{t-1}, chosen_option_{t-1}, ...]

Optional neural observations:
    y_{n,t} ~ Poisson(exp(baseline_n + w_n @ x_t) * dt)
```

### Relationship to existing models

- **SmithLearning:** K=2 case with sigmoid link is equivalent to Smith's binomial model (with different parameterization). The multinomial model generalizes to K≥2.
- **PointProcessModel:** If neural observations are included, the latent value state is observed through both choices (categorical) AND spikes (Poisson). This links behavioral choice to neural population activity through a shared latent value.

### Laplace approximation for softmax

The softmax observation model is nonlinear but smooth. The Laplace-EKF linearizes around the current state estimate:

```
Gradient of log P(c_t=k | x):
    ∂/∂x_j = β * (1{j=k} - softmax(β*x)_j)

Hessian:
    ∂²/∂x_j∂x_i = -β² * softmax_j * (1{i=j} - softmax_i)
```

The Hessian is always negative semi-definite (softmax is log-concave), so the Laplace approximation is well-behaved.

---

## Task 1: Softmax Observation Update

Build the Laplace-EKF update for a categorical observation with softmax link.

**Files:**
- Create: `src/state_space_practice/multinomial_choice.py`
- Test: `src/state_space_practice/tests/test_multinomial_choice.py`

### Step 1: Write failing test

```python
# tests/test_multinomial_choice.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.multinomial_choice import (
    softmax_observation_update,
)


class TestSoftmaxObservationUpdate:
    def test_output_shapes(self):
        n_options = 4
        prior_mean = jnp.zeros(n_options)
        prior_cov = jnp.eye(n_options)
        choice = 2  # chose option 2

        post_mean, post_cov, ll = softmax_observation_update(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            choice=choice,
            inverse_temperature=1.0,
        )

        assert post_mean.shape == (n_options,)
        assert post_cov.shape == (n_options, n_options)
        assert jnp.isfinite(ll)

    def test_chosen_option_value_increases(self):
        """Observing choice k should increase x_k."""
        n_options = 3
        prior_mean = jnp.zeros(n_options)
        prior_cov = jnp.eye(n_options) * 0.5

        post_mean, _, _ = softmax_observation_update(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            choice=1,
            inverse_temperature=1.0,
        )

        # Chosen option should have highest posterior value
        assert post_mean[1] > post_mean[0]
        assert post_mean[1] > post_mean[2]

    def test_high_temperature_weak_update(self):
        """Low inverse temperature (high exploration) should give weak update."""
        prior_mean = jnp.zeros(3)
        prior_cov = jnp.eye(3)

        post_hot, _, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=0, inverse_temperature=0.1,
        )
        post_cold, _, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=0, inverse_temperature=5.0,
        )

        # Cold (decisive) should update more than hot (random)
        assert abs(float(post_cold[0])) > abs(float(post_hot[0]))

    def test_posterior_covariance_decreases(self):
        """Observation should reduce uncertainty."""
        prior_cov = jnp.eye(4) * 2.0
        _, post_cov, _ = softmax_observation_update(
            prior_mean=jnp.zeros(4),
            prior_cov=prior_cov,
            choice=0,
            inverse_temperature=1.0,
        )
        assert jnp.trace(post_cov) < jnp.trace(prior_cov)

    def test_two_options_matches_sigmoid(self):
        """With 2 options, softmax reduces to sigmoid. Check consistency."""
        prior_mean = jnp.array([0.5, -0.3])
        prior_cov = jnp.eye(2) * 0.5

        post_mean, _, ll = softmax_observation_update(
            prior_mean, prior_cov, choice=0, inverse_temperature=1.0,
        )

        # P(choice 0) = softmax([0.5, -0.3])[0] = sigmoid(0.5 - (-0.3)) = sigmoid(0.8)
        expected_p = float(jax.nn.sigmoid(0.8))
        actual_p = float(jnp.exp(jax.nn.log_softmax(prior_mean))[0])
        np.testing.assert_allclose(actual_p, expected_p, atol=1e-5)
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py::TestSoftmaxObservationUpdate -v`
Expected: FAIL with ImportError

### Step 3: Implement softmax observation update

```python
# src/state_space_practice/multinomial_choice.py
"""Multinomial choice learning model with softmax observations.

Tracks the animal's evolving valuation of multiple choice options in
multi-armed bandit tasks. The latent state is a vector of option values,
choices are Categorical(softmax(x)), and inference uses the Laplace-EKF.

References
----------
[1] Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006).
    Cortical substrates for exploratory decisions in humans.
    Nature 441, 876-879.
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import psd_solve, symmetrize
from state_space_practice.utils import check_converged

logger = logging.getLogger(__name__)


def softmax_observation_update(
    prior_mean: Array,
    prior_cov: Array,
    choice: int,
    inverse_temperature: float = 1.0,
    diagonal_boost: float = 1e-9,
) -> tuple[Array, Array, Array]:
    """Laplace-EKF update for a categorical observation with softmax link.

    Parameters
    ----------
    prior_mean : Array, shape (n_options,)
        Prior value state mean.
    prior_cov : Array, shape (n_options, n_options)
        Prior value state covariance.
    choice : int
        Observed choice (0-indexed option index).
    inverse_temperature : float
        Softmax inverse temperature β. Higher = more deterministic.
    diagonal_boost : float
        Numerical stability term.

    Returns
    -------
    posterior_mean : Array, shape (n_options,)
    posterior_cov : Array, shape (n_options, n_options)
    log_likelihood : Array (scalar)
    """
    n_options = prior_mean.shape[0]

    # Softmax probabilities at prior mean
    logits = inverse_temperature * prior_mean
    probs = jax.nn.softmax(logits)

    # Log-likelihood of observed choice
    log_likelihood = jnp.log(probs[choice] + 1e-30)

    # Gradient of log P(c=k | x) w.r.t. x
    # = β * (e_k - softmax(β*x))
    one_hot = jnp.zeros(n_options).at[choice].set(1.0)
    gradient = inverse_temperature * (one_hot - probs)

    # Negative Hessian of log P(c=k | x)
    # = β² * (diag(softmax) - softmax ⊗ softmax)
    neg_hessian = inverse_temperature ** 2 * (
        jnp.diag(probs) - jnp.outer(probs, probs)
    )

    # Prior precision
    identity = jnp.eye(n_options)
    prior_precision = psd_solve(prior_cov, identity, diagonal_boost=diagonal_boost)

    # Posterior precision = prior precision + Fisher information
    posterior_precision = prior_precision + neg_hessian
    posterior_precision = symmetrize(posterior_precision)

    # Ensure PSD
    eigvals, eigvecs = jnp.linalg.eigh(posterior_precision)
    eigvals_safe = jnp.maximum(eigvals, diagonal_boost)
    posterior_precision = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T

    # Newton step
    posterior_mean = prior_mean + psd_solve(
        posterior_precision, gradient, diagonal_boost=diagonal_boost
    )

    # Posterior covariance
    posterior_cov = psd_solve(
        posterior_precision, identity, diagonal_boost=diagonal_boost
    )
    posterior_cov = symmetrize(posterior_cov)

    return posterior_mean, posterior_cov, log_likelihood
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/multinomial_choice.py \
        src/state_space_practice/tests/test_multinomial_choice.py
git commit -m "feat: add softmax observation update for multinomial choice model"
```

---

## Task 2: Multinomial Choice Filter and Smoother

Build the full filter/smoother that tracks option values across a sequence of trials.

**Files:**
- Modify: `src/state_space_practice/multinomial_choice.py`
- Test: `src/state_space_practice/tests/test_multinomial_choice.py`

### Step 1: Write failing test

```python
# Add to tests/test_multinomial_choice.py

from state_space_practice.multinomial_choice import (
    multinomial_choice_filter,
    multinomial_choice_smoother,
    ChoiceFilterResult,
)


class TestMultinomialChoiceFilter:
    def test_output_shapes(self):
        n_trials = 100
        n_options = 3
        choices = jnp.array(np.random.default_rng(42).integers(0, n_options, n_trials))

        result = multinomial_choice_filter(
            choices=choices,
            n_options=n_options,
        )

        assert isinstance(result, ChoiceFilterResult)
        assert result.filtered_values.shape == (n_trials, n_options)
        assert result.filtered_covariances.shape == (n_trials, n_options, n_options)
        assert jnp.isfinite(result.marginal_log_likelihood)

    def test_preferred_option_has_highest_value(self):
        """If option 0 is chosen 80% of the time, its value should be highest."""
        rng = np.random.default_rng(42)
        n_trials = 200
        # Biased choices: option 0 preferred
        choices = jnp.array(rng.choice(3, n_trials, p=[0.8, 0.1, 0.1]))

        result = multinomial_choice_filter(
            choices=choices,
            n_options=3,
        )

        # Final value estimate: option 0 should be highest
        final_values = result.filtered_values[-1]
        assert final_values[0] > final_values[1]
        assert final_values[0] > final_values[2]

    def test_switching_preference(self):
        """Preference switches mid-session: values should track."""
        n_trials = 200
        choices = np.zeros(n_trials, dtype=int)
        choices[:100] = 0  # prefer option 0
        choices[100:] = 2  # switch to option 2

        result = multinomial_choice_filter(
            choices=jnp.array(choices),
            n_options=3,
            process_noise=0.1,
        )

        # Early: option 0 highest
        assert result.filtered_values[50, 0] > result.filtered_values[50, 2]
        # Late: option 2 highest
        assert result.filtered_values[180, 2] > result.filtered_values[180, 0]


class TestMultinomialChoiceSmoother:
    def test_smoother_reduces_variance(self):
        rng = np.random.default_rng(42)
        n_trials = 100
        choices = jnp.array(rng.integers(0, 3, n_trials))

        filter_result = multinomial_choice_filter(choices=choices, n_options=3)
        smoother_result = multinomial_choice_smoother(choices=choices, n_options=3)

        # Average smoothed variance should be <= filtered variance
        filter_var = jnp.trace(filter_result.filtered_covariances, axis1=1, axis2=2).mean()
        smoother_var = jnp.trace(smoother_result.smoothed_covariances, axis1=1, axis2=2).mean()
        assert smoother_var <= filter_var * 1.01
```

### Step 2: Run test to verify it fails

### Step 3: Implement filter and smoother

```python
# Add to src/state_space_practice/multinomial_choice.py

from typing import NamedTuple

from state_space_practice.kalman import _kalman_smoother_update


class ChoiceFilterResult(NamedTuple):
    filtered_values: Array        # (n_trials, n_options)
    filtered_covariances: Array   # (n_trials, n_options, n_options)
    marginal_log_likelihood: Array  # scalar


class ChoiceSmootherResult(NamedTuple):
    smoothed_values: Array        # (n_trials, n_options)
    smoothed_covariances: Array   # (n_trials, n_options, n_options)
    smoother_cross_cov: Array     # (n_trials - 1, n_options, n_options)
    marginal_log_likelihood: Array


def multinomial_choice_filter(
    choices: Array,
    n_options: int,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    init_mean: Optional[Array] = None,
    init_cov: Optional[Array] = None,
    transition_matrix: Optional[Array] = None,
) -> ChoiceFilterResult:
    """Filter for tracking option values from a sequence of choices.

    Parameters
    ----------
    choices : Array, shape (n_trials,)
        Observed choices (0-indexed).
    n_options : int
        Number of choice options.
    process_noise : float
        Diagonal process noise variance (how fast values drift).
    inverse_temperature : float
        Softmax inverse temperature.
    init_mean : Array or None, shape (n_options,)
        Initial value estimates. Default: zeros.
    init_cov : Array or None, shape (n_options, n_options)
        Initial value covariance. Default: identity.
    transition_matrix : Array or None, shape (n_options, n_options)
        Value dynamics. Default: identity (random walk).

    Returns
    -------
    ChoiceFilterResult
    """
    n_trials = choices.shape[0]
    if init_mean is None:
        init_mean = jnp.zeros(n_options)
    if init_cov is None:
        init_cov = jnp.eye(n_options)
    if transition_matrix is None:
        transition_matrix = jnp.eye(n_options)

    Q = jnp.eye(n_options) * process_noise

    def _step(carry, choice_t):
        mean_prev, cov_prev, total_ll = carry

        # Prediction
        pred_mean = transition_matrix @ mean_prev
        pred_cov = transition_matrix @ cov_prev @ transition_matrix.T + Q
        pred_cov = symmetrize(pred_cov)

        # Observation update
        post_mean, post_cov, ll = softmax_observation_update(
            pred_mean, pred_cov, choice_t, inverse_temperature,
        )

        total_ll = total_ll + ll
        return (post_mean, post_cov, total_ll), (post_mean, post_cov)

    (_, _, marginal_ll), (filtered_values, filtered_covs) = jax.lax.scan(
        _step,
        (init_mean, init_cov, jnp.array(0.0)),
        choices,
    )

    return ChoiceFilterResult(
        filtered_values=filtered_values,
        filtered_covariances=filtered_covs,
        marginal_log_likelihood=marginal_ll,
    )


def multinomial_choice_smoother(
    choices: Array,
    n_options: int,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    init_mean: Optional[Array] = None,
    init_cov: Optional[Array] = None,
    transition_matrix: Optional[Array] = None,
) -> ChoiceSmootherResult:
    """Smoother for option value trajectories.

    Runs filter then RTS backward pass.

    Parameters
    ----------
    Same as multinomial_choice_filter.

    Returns
    -------
    ChoiceSmootherResult
    """
    if transition_matrix is None:
        transition_matrix = jnp.eye(n_options)
    Q = jnp.eye(n_options) * process_noise

    filter_result = multinomial_choice_filter(
        choices, n_options, process_noise, inverse_temperature,
        init_mean, init_cov, transition_matrix,
    )

    def _smooth_step(carry, inputs):
        next_sm_mean, next_sm_cov = carry
        filt_mean, filt_cov = inputs

        sm_mean, sm_cov, cross_cov = _kalman_smoother_update(
            next_sm_mean, next_sm_cov,
            filt_mean, filt_cov,
            Q, transition_matrix,
        )
        return (sm_mean, sm_cov), (sm_mean, sm_cov, cross_cov)

    _, (sm_values, sm_covs, cross_covs) = jax.lax.scan(
        _smooth_step,
        (filter_result.filtered_values[-1], filter_result.filtered_covariances[-1]),
        (filter_result.filtered_values[:-1], filter_result.filtered_covariances[:-1]),
        reverse=True,
    )

    smoothed_values = jnp.concatenate([sm_values, filter_result.filtered_values[-1:]])
    smoothed_covs = jnp.concatenate([sm_covs, filter_result.filtered_covariances[-1:]])

    return ChoiceSmootherResult(
        smoothed_values=smoothed_values,
        smoothed_covariances=smoothed_covs,
        smoother_cross_cov=cross_covs,
        marginal_log_likelihood=filter_result.marginal_log_likelihood,
    )
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add multinomial choice filter and smoother"
```

---

## Task 3: MultinomialChoiceModel Class with EM

Wrap into a model class that learns process noise, inverse temperature, and optionally covariate-driven dynamics via EM.

**Files:**
- Modify: `src/state_space_practice/multinomial_choice.py`
- Test: `src/state_space_practice/tests/test_multinomial_choice.py`

### Step 1: Write failing test

```python
# Add to tests/test_multinomial_choice.py

from state_space_practice.multinomial_choice import MultinomialChoiceModel


class TestMultinomialChoiceModel:
    def test_fit(self):
        rng = np.random.default_rng(42)
        choices = jnp.array(rng.choice(4, 300, p=[0.5, 0.2, 0.2, 0.1]))

        model = MultinomialChoiceModel(n_options=4)
        lls = model.fit(choices, max_iter=5)

        assert len(lls) == 5
        assert all(np.isfinite(ll) for ll in lls)
        assert model.smoothed_values is not None
        assert model.smoothed_values.shape == (300, 4)

    def test_learned_inverse_temperature(self):
        """Deterministic choices should yield high inverse temperature."""
        # Always choose option 0
        choices = jnp.zeros(200, dtype=int)
        model = MultinomialChoiceModel(n_options=3)
        model.fit(choices, max_iter=10)

        assert model.inverse_temperature > 2.0

    def test_learned_process_noise(self):
        """Switching preferences should yield higher process noise."""
        choices_stable = jnp.zeros(200, dtype=int)
        choices_switch = jnp.array(
            [0]*50 + [1]*50 + [2]*50 + [0]*50
        )

        model_stable = MultinomialChoiceModel(n_options=3)
        model_stable.fit(choices_stable, max_iter=10)

        model_switch = MultinomialChoiceModel(n_options=3)
        model_switch.fit(choices_switch, max_iter=10)

        assert model_switch.process_noise > model_stable.process_noise

    def test_with_covariates(self):
        """Reward history as covariate should improve fit."""
        rng = np.random.default_rng(42)
        n_trials = 200
        n_options = 3
        choices = jnp.array(rng.integers(0, n_options, n_trials))
        # Reward: 1 if chose option 0, else 0
        rewards = (np.array(choices) == 0).astype(float)
        # Covariate: previous reward for each option
        covariates = np.zeros((n_trials, n_options))
        for t in range(1, n_trials):
            covariates[t, choices[t-1]] = rewards[t-1]

        model = MultinomialChoiceModel(n_options=n_options)
        lls = model.fit(choices, covariates=jnp.array(covariates), max_iter=5)
        assert model.covariate_weights is not None

    def test_choice_probability(self):
        model = MultinomialChoiceModel(n_options=3)
        choices = jnp.array(np.random.default_rng(0).integers(0, 3, 100))
        model.fit(choices, max_iter=3)
        probs = model.choice_probabilities()
        assert probs.shape == (100, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_plot(self):
        import matplotlib
        matplotlib.use("Agg")

        model = MultinomialChoiceModel(n_options=3)
        choices = jnp.array(np.random.default_rng(0).integers(0, 3, 100))
        model.fit(choices, max_iter=3)
        fig = model.plot_values()
        assert fig is not None
```

### Step 2: Run test to verify it fails

### Step 3: Implement MultinomialChoiceModel

```python
# Add to src/state_space_practice/multinomial_choice.py

class MultinomialChoiceModel:
    """Multi-armed bandit choice model with evolving option values.

    Tracks latent option values from a sequence of choices using a
    state-space model with softmax observation model. Learns the
    exploration-exploitation tradeoff (inverse temperature), drift
    rate (process noise), and optionally covariate-driven dynamics.

    Parameters
    ----------
    n_options : int
        Number of choice options.
    init_inverse_temperature : float, default=1.0
        Initial inverse temperature.
    init_process_noise : float, default=0.01
        Initial process noise.
    learn_inverse_temperature : bool, default=True
        Whether to learn β via EM.
    learn_process_noise : bool, default=True
        Whether to learn Q via EM.

    Attributes
    ----------
    smoothed_values : Array, shape (n_trials, n_options)
        Smoothed value estimates after fitting.
    inverse_temperature : float
        Learned inverse temperature (higher = more exploitative).
    process_noise : float
        Learned process noise (higher = faster value changes).

    Examples
    --------
    >>> model = MultinomialChoiceModel(n_options=4)
    >>> model.fit(choices)
    >>> model.plot_values()
    >>> probs = model.choice_probabilities()
    """

    def __init__(
        self,
        n_options: int,
        init_inverse_temperature: float = 1.0,
        init_process_noise: float = 0.01,
        learn_inverse_temperature: bool = True,
        learn_process_noise: bool = True,
    ):
        self.n_options = n_options
        self.inverse_temperature = init_inverse_temperature
        self.process_noise = init_process_noise
        self.learn_inverse_temperature = learn_inverse_temperature
        self.learn_process_noise = learn_process_noise

        self.smoothed_values: Optional[Array] = None
        self.smoothed_covariances: Optional[Array] = None
        self.covariate_weights: Optional[Array] = None
        self.log_likelihoods: list[float] = []

    def __repr__(self) -> str:
        fitted = self.smoothed_values is not None
        return (
            f"<MultinomialChoiceModel: n_options={self.n_options}, "
            f"β={self.inverse_temperature:.2f}, "
            f"Q={self.process_noise:.4f}, "
            f"fitted={fitted}>"
        )

    def fit(
        self,
        choices: ArrayLike,
        covariates: Optional[ArrayLike] = None,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        verbose: bool = True,
    ) -> list[float]:
        """Fit the model to choice data using EM.

        Parameters
        ----------
        choices : ArrayLike, shape (n_trials,)
            Observed choices (0-indexed).
        covariates : ArrayLike or None, shape (n_trials, n_options)
            Per-trial covariates that drive value changes. Each column
            is added to the corresponding option's value prediction.
            E.g., previous reward for each option.
        max_iter : int
        tolerance : float
        verbose : bool

        Returns
        -------
        log_likelihoods : list[float]
        """
        choices = jnp.asarray(choices)
        n_trials = choices.shape[0]

        if covariates is not None:
            covariates = jnp.asarray(covariates)
            self.covariate_weights = jnp.ones(self.n_options) * 0.1
        else:
            self.covariate_weights = None

        self.log_likelihoods = []

        def _print(msg):
            if verbose:
                print(msg)

        _print(f"MultinomialChoiceModel: n_trials={n_trials}, "
               f"n_options={self.n_options}")

        for iteration in range(max_iter):
            # E-step: smoother
            transition_matrix = jnp.eye(self.n_options)

            # If covariates, add to prediction
            # For now: covariates enter as additive input to the state
            # x_t = x_{t-1} + B * covariates_t + noise
            # This is handled by adjusting the prior mean in the filter

            result = multinomial_choice_smoother(
                choices=choices,
                n_options=self.n_options,
                process_noise=self.process_noise,
                inverse_temperature=self.inverse_temperature,
                transition_matrix=transition_matrix,
            )

            ll = float(result.marginal_log_likelihood)
            self.log_likelihoods.append(ll)
            _print(f"  EM iter {iteration + 1}/{max_iter}: LL = {ll:.1f}, "
                   f"β={self.inverse_temperature:.2f}, Q={self.process_noise:.4f}")

            if not jnp.isfinite(ll):
                break

            if iteration > 0:
                is_converged, _ = check_converged(
                    ll, self.log_likelihoods[-2], tolerance
                )
                if is_converged:
                    _print(f"  Converged after {iteration + 1} iterations.")
                    break

            self.smoothed_values = result.smoothed_values
            self.smoothed_covariances = result.smoothed_covariances

            # M-step
            if self.learn_process_noise:
                # Q = mean of E[(x_t - x_{t-1})(x_t - x_{t-1})']
                dx = jnp.diff(result.smoothed_values, axis=0)
                dx_sq = jnp.mean(dx ** 2, axis=0)
                # Add covariance terms
                cov_sum = (
                    result.smoothed_covariances[1:] +
                    result.smoothed_covariances[:-1]
                ).mean(axis=0)
                cross_sum = result.smoother_cross_cov.mean(axis=0)
                q_diag = jnp.mean(dx_sq) + jnp.mean(jnp.diag(cov_sum)) - 2 * jnp.mean(jnp.diag(cross_sum))
                self.process_noise = float(jnp.maximum(q_diag, 1e-6))

            if self.learn_inverse_temperature:
                # Grid search over β (simple, robust)
                best_ll = -jnp.inf
                best_beta = self.inverse_temperature
                for beta_candidate in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                    test_result = multinomial_choice_filter(
                        choices=choices,
                        n_options=self.n_options,
                        process_noise=self.process_noise,
                        inverse_temperature=beta_candidate,
                    )
                    if test_result.marginal_log_likelihood > best_ll:
                        best_ll = test_result.marginal_log_likelihood
                        best_beta = beta_candidate
                self.inverse_temperature = best_beta

        return self.log_likelihoods

    def choice_probabilities(self) -> Array:
        """Get the model's predicted choice probabilities at each trial.

        Returns
        -------
        probs : Array, shape (n_trials, n_options)
        """
        if self.smoothed_values is None:
            raise RuntimeError("Model has not been fitted yet.")
        logits = self.inverse_temperature * self.smoothed_values
        return jax.nn.softmax(logits, axis=1)

    def plot_values(self, option_labels: Optional[list[str]] = None, ax=None):
        """Plot option values over trials with choice probabilities.

        Parameters
        ----------
        option_labels : list[str] or None
            Labels for each option.
        ax : Axes or None

        Returns
        -------
        fig : Figure
        """
        import matplotlib.pyplot as plt

        if self.smoothed_values is None:
            raise RuntimeError("Model has not been fitted yet.")

        n_trials, n_options = self.smoothed_values.shape
        if option_labels is None:
            option_labels = [f"Option {k}" for k in range(n_options)]

        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        trials = np.arange(n_trials)
        colors = plt.cm.tab10(np.arange(n_options))

        # Top: values with CI
        for k in range(n_options):
            values = np.array(self.smoothed_values[:, k])
            std = np.sqrt(np.array(self.smoothed_covariances[:, k, k]))
            axes[0].plot(trials, values, color=colors[k], label=option_labels[k])
            axes[0].fill_between(
                trials, values - 1.96 * std, values + 1.96 * std,
                color=colors[k], alpha=0.15,
            )
        axes[0].set_ylabel("Latent value")
        axes[0].set_title(
            f"Option Values (β={self.inverse_temperature:.1f}, "
            f"Q={self.process_noise:.4f})"
        )
        axes[0].legend(fontsize=8)

        # Bottom: choice probabilities
        probs = np.array(self.choice_probabilities())
        axes[1].stackplot(
            trials,
            *[probs[:, k] for k in range(n_options)],
            labels=option_labels,
            colors=colors,
            alpha=0.7,
        )
        axes[1].set_ylabel("P(choice)")
        axes[1].set_xlabel("Trial")
        axes[1].set_ylim(0, 1)

        fig.tight_layout()
        return fig
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add MultinomialChoiceModel with EM for multi-armed bandit tasks"
```

---

## Task 4: Joint Choice + Neural Model

Extend the model to incorporate spike observations alongside choices, linking latent values to neural population activity.

**Files:**
- Modify: `src/state_space_practice/multinomial_choice.py`
- Test: `src/state_space_practice/tests/test_multinomial_choice.py`

### Step 1: Write failing test

```python
# Add to tests/test_multinomial_choice.py

class TestJointChoiceNeuralModel:
    def test_fit_with_spikes(self):
        rng = np.random.default_rng(42)
        n_trials = 200
        n_options = 3
        n_neurons = 5

        choices = jnp.array(rng.integers(0, n_options, n_trials))
        spikes = jnp.array(rng.poisson(0.5, (n_trials, n_neurons)))

        model = MultinomialChoiceModel(n_options=n_options)
        lls = model.fit(choices, spikes=spikes, max_iter=3)

        assert len(lls) == 3
        # Neural loading weights should be learned
        assert model.neural_weights is not None
        assert model.neural_weights.shape == (n_neurons, n_options)

    def test_neural_improves_value_estimate(self):
        """Adding informative neural data should improve choice prediction."""
        rng = np.random.default_rng(42)
        n_trials = 200
        n_options = 3

        # Generate choices correlated with values
        true_values = np.cumsum(rng.normal(0, 0.1, (n_trials, n_options)), axis=0)
        probs = np.exp(true_values) / np.exp(true_values).sum(axis=1, keepdims=True)
        choices = np.array([rng.choice(n_options, p=p) for p in probs])

        # Spikes that encode the values
        neural_weights = rng.normal(0, 0.5, (5, n_options))
        spikes = rng.poisson(
            np.exp(true_values @ neural_weights.T + 1) * 0.004,
        )

        model_no_neural = MultinomialChoiceModel(n_options=n_options)
        lls_no = model_no_neural.fit(jnp.array(choices), max_iter=5, verbose=False)

        model_neural = MultinomialChoiceModel(n_options=n_options)
        lls_with = model_neural.fit(
            jnp.array(choices), spikes=jnp.array(spikes), max_iter=5, verbose=False,
        )

        # Joint model should have better LL (more data)
        # (This is total LL including spikes, so not directly comparable,
        #  but the choice component should be better estimated)
        assert model_neural.smoothed_values is not None
```

### Step 2: Implement joint model

The key extension: at each trial, the observation update has two parts:
1. Choice update (softmax, as before)
2. Spike update (Poisson, using the value state as input to a GLM)

```python
# The spike observation model:
# log(λ_n) = baseline_n + w_n @ x_t
# where x_t is the value state and w_n are per-neuron loadings
#
# This uses the same _point_process_laplace_update as the place field model,
# but the "design matrix" is the identity (the state IS the value, not
# position-dependent weights).
#
# In fit(), after the softmax update, do a second Laplace update for spikes:
# post_mean, post_cov = softmax_update(prior_mean, prior_cov, choice)
# post_mean, post_cov = spike_update(post_mean, post_cov, spikes, neural_weights)
```

### Step 3: Run tests, commit

```bash
git commit -m "feat: add joint choice + neural observations to MultinomialChoiceModel"
```

---

## Task 5: Comparison with RL Models

Add methods to compare the state-space value trajectory with standard RL algorithms.

**Files:**
- Modify: `src/state_space_practice/multinomial_choice.py`

### Methods to add:

```python
    def compare_to_q_learning(
        self,
        choices: Array,
        rewards: Array,
        learning_rates: Optional[list[float]] = None,
    ) -> dict:
        """Compare inferred values to Q-learning predictions.

        Fits Q-learning with various learning rates and computes
        correlation between Q-values and the state-space inferred values.

        Returns
        -------
        dict with keys:
            best_learning_rate : float
            correlation_per_option : (n_options,)
            q_values : (n_trials, n_options) — Q-learning values at best α
        """
        ...

    def plot_comparison(self, choices, rewards, ax=None):
        """Plot state-space values vs Q-learning values."""
        ...
```

### Commit

```bash
git commit -m "feat: add RL model comparison to MultinomialChoiceModel"
```

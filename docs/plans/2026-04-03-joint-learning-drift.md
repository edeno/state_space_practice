# Joint Learning + Representational Drift Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a model that jointly infers brain state, behavioral learning, and place field drift — linking them through a shared switching discrete state so that learning rate and drift rate are both state-dependent.

**Architecture:** A factored state-space model with three components sharing a discrete switching state: (1) a binomial learning model for behavioral performance (Smith-style), (2) per-neuron place field weight models (PlaceFieldModel-style), and (3) a discrete state that modulates the process noise of both. The E-step uses a structured variational approach: infer discrete states from all observations jointly, then run separate smoothers for learning state and spatial weights conditional on the discrete state. The M-step learns per-state process noise (learning rate and drift rate), spike parameters, and discrete transition probabilities.

**Tech Stack:** JAX, patsy (spline basis), existing `smith_learning_filter`/`smith_learning_smoother`, `stochastic_point_process_smoother`, `switching_kalman_filter` infrastructure.

**References:**

- Smith, A.C., Frank, L.M., Wirth, S. et al. (2004). Dynamic analysis of learning in behavioral experiments. J Neuroscience 24(2), 447-461.
- Ziv, Y., Burns, L.D., Cocker, E.D. et al. (2013). Long-term dynamics of CA1 hippocampal place codes. Nature Neuroscience 16, 264-266.
- Karlsson, M.P. & Frank, L.M. (2009). Awake replay of remote experiences in the hippocampus. Nature Neuroscience 12(7), 913-918.
- Linderman, S.W., Johnson, M.J., Miller, A.C. et al. (2017). Bayesian learning and inference in recurrent switching linear dynamical systems. AISTATS.
- Lever, C., Wills, T., Cacucci, F., Burgess, N. & O'Keefe, J. (2002). Long-term plasticity in hippocampal place-cell representation of environmental geometry. Nature 416, 90-94.
- Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004). Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering. Neural Computation 16, 971-998.

---

## Background and Mathematical Model

### The scientific question
During learning, hippocampal place fields remap. Is this remapping driven by the same brain state that drives learning? The model discovers discrete brain states from neural+behavioral data jointly, and estimates whether learning rate and representational drift rate co-vary across states.

### Generative model

```
Discrete state:
    s_t ~ Categorical(Z @ e_{s_{t-1}})         # Markov switching

Learning state (scalar):
    theta_t = theta_{t-1} + eta_t               # eta_t ~ N(0, q_learn^{s_t})
    y_t^{behav} ~ Binomial(n_t, sigmoid(theta_t))

Spatial weights (per neuron n, n_basis-dimensional):
    x_{n,t} = x_{n,t-1} + eps_{n,t}            # eps_{n,t} ~ N(0, Q_drift^{s_t})
    y_{n,t}^{spike} ~ Poisson(exp(Z_t @ x_{n,t}) * dt)
```

Key parameters to learn:
- `q_learn^{s}`: per-state learning process noise (scalar per state)
- `Q_drift^{s}`: per-state drift process noise (diagonal matrix per state)
- `Z`: discrete transition matrix
- Per-neuron initial conditions for spatial weights

### Factored E-step

The full latent state `[s_t, theta_t, x_{1,t}, ..., x_{N,t}]` is too large for joint inference. We exploit conditional independence:

1. **Discrete state sweep**: Given current estimates of theta and x, compute `P(s_t | all data)` using forward-backward on the discrete state chain, where emission likelihoods come from the behavioral and neural observation models.
2. **Learning smoother**: Given `P(s_t)`, run a state-dependent Smith learning smoother for `theta_t` (uses per-state `q_learn^{s_t}`).
3. **Place field smoothers**: Given `P(s_t)`, run per-neuron point-process smoothers for `x_{n,t}` (uses per-state `Q_drift^{s_t}`), independently across neurons.
4. Iterate steps 1-3 until the discrete state posteriors stabilize (inner loop, typically 2-3 iterations).

---

## Task 1: State-Dependent Smith Learning Smoother

Build a variant of the Smith learning filter/smoother that accepts per-state process noise and discrete state probabilities.

**Files:**
- Create: `src/state_space_practice/state_dependent_learning.py`
- Reference: `src/state_space_practice/smith_learning_algorithm.py` (lines 165-450 for filter/smoother)
- Test: `src/state_space_practice/tests/test_state_dependent_learning.py`

### Step 1: Write failing test for state-dependent learning filter

```python
# tests/test_state_dependent_learning.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.state_dependent_learning import (
    state_dependent_learning_filter,
)


class TestStateDependentLearningFilter:
    def test_single_state_matches_standard_filter(self):
        """With one discrete state, should match standard Smith filter."""
        from state_space_practice.smith_learning_algorithm import (
            smith_learning_filter,
        )

        rng = np.random.default_rng(42)
        n_time = 100
        n_correct = rng.binomial(10, 0.7, n_time)

        # Single state: prob always 1.0
        discrete_state_prob = jnp.ones((n_time, 1))
        q_per_state = jnp.array([0.01])

        # Standard filter
        std_result = smith_learning_filter(
            init_mean=0.0,
            init_variance=1.0,
            n_correct_responses=jnp.array(n_correct),
            max_possible_correct=10,
            sigma_epsilon=0.1,
            prob_correct_by_chance=0.5,
        )

        # State-dependent filter with one state
        sd_result = state_dependent_learning_filter(
            init_mean=0.0,
            init_variance=1.0,
            n_correct_responses=jnp.array(n_correct),
            max_possible_correct=10,
            sigma_epsilon_per_state=q_per_state,
            discrete_state_prob=discrete_state_prob,
            prob_correct_by_chance=0.5,
        )

        np.testing.assert_allclose(
            sd_result.filtered_mean, std_result[0], atol=0.01
        )

    def test_two_states_different_noise(self):
        """Two states with different process noise should produce different
        smoothness in different time regions."""
        n_time = 200
        discrete_state_prob = jnp.zeros((n_time, 2))
        # First half: state 0 (low noise), second half: state 1 (high noise)
        discrete_state_prob = discrete_state_prob.at[:100, 0].set(1.0)
        discrete_state_prob = discrete_state_prob.at[100:, 1].set(1.0)

        q_per_state = jnp.array([0.001, 0.1])  # low vs high learning rate

        n_correct = jnp.ones(n_time, dtype=int) * 5  # constant performance

        result = state_dependent_learning_filter(
            init_mean=0.0,
            init_variance=1.0,
            n_correct_responses=n_correct,
            max_possible_correct=10,
            sigma_epsilon_per_state=q_per_state,
            discrete_state_prob=discrete_state_prob,
            prob_correct_by_chance=0.5,
        )

        # Check result is finite
        assert jnp.all(jnp.isfinite(result.filtered_mean))
        # In high-noise state, filtered variance should be larger
        var_low = result.filtered_variance[:100].mean()
        var_high = result.filtered_variance[100:].mean()
        assert var_high > var_low

    def test_output_shapes(self):
        """Check output shapes match input."""
        n_time = 50
        n_correct = jnp.ones(n_time, dtype=int) * 3
        discrete_state_prob = jnp.ones((n_time, 2)) * 0.5
        q_per_state = jnp.array([0.01, 0.05])

        result = state_dependent_learning_filter(
            init_mean=0.0,
            init_variance=1.0,
            n_correct_responses=n_correct,
            max_possible_correct=10,
            sigma_epsilon_per_state=q_per_state,
            discrete_state_prob=discrete_state_prob,
            prob_correct_by_chance=0.5,
        )

        assert result.filtered_mean.shape == (n_time,)
        assert result.filtered_variance.shape == (n_time,)
        assert result.marginal_log_likelihood.shape == ()
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_state_dependent_learning.py -v`
Expected: FAIL with ImportError

### Step 3: Implement state-dependent learning filter

The key modification to the standard Smith filter: the process noise at each time step is a weighted average of per-state process noises, weighted by `P(s_t)`.

```python
# src/state_space_practice/state_dependent_learning.py
"""State-dependent learning model.

Extends the Smith learning algorithm with discrete-state-dependent process
noise, allowing learning rate to vary across brain states.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class LearningFilterResult(NamedTuple):
    filtered_mean: Array       # (n_time,)
    filtered_variance: Array   # (n_time,)
    marginal_log_likelihood: Array  # scalar


class LearningSmoothResult(NamedTuple):
    smoothed_mean: Array       # (n_time,)
    smoothed_variance: Array   # (n_time,)
    smoother_gain: Array       # (n_time - 1,)


def state_dependent_learning_filter(
    init_mean: float,
    init_variance: float,
    n_correct_responses: Array,
    max_possible_correct: int,
    sigma_epsilon_per_state: Array,
    discrete_state_prob: Array,
    prob_correct_by_chance: float = 0.5,
) -> LearningFilterResult:
    """Smith learning filter with state-dependent process noise.

    Parameters
    ----------
    init_mean : float
        Initial learning state.
    init_variance : float
        Initial learning state variance.
    n_correct_responses : Array, shape (n_time,)
        Number correct per trial.
    max_possible_correct : int
        Maximum possible correct per trial.
    sigma_epsilon_per_state : Array, shape (n_states,)
        Process noise (sigma_epsilon) for each discrete state.
    discrete_state_prob : Array, shape (n_time, n_states)
        Probability of each discrete state at each time.
    prob_correct_by_chance : float
        Chance-level performance.

    Returns
    -------
    LearningFilterResult
    """
    # Effective process noise at each time: weighted average
    q_per_state = sigma_epsilon_per_state ** 2
    effective_q = discrete_state_prob @ q_per_state  # (n_time,)

    mu = jnp.log(prob_correct_by_chance / (1.0 - prob_correct_by_chance))

    def _step(carry, inputs):
        mean_prev, var_prev, total_ll = carry
        n_correct, q_t = inputs

        # One-step prediction
        pred_mean = mean_prev
        pred_var = var_prev + q_t

        # Observation update (Laplace approximation for binomial)
        p = jax.nn.sigmoid(mu + pred_mean)
        expected = max_possible_correct * p
        obs_var = max_possible_correct * p * (1 - p)

        # Kalman-like update
        innovation = n_correct - expected
        S = obs_var + 1e-10  # observation variance in linearized space
        # Derivative of sigmoid: dp/dx = p*(1-p)
        H = max_possible_correct * p * (1 - p)  # d(expected)/d(state)
        K = pred_var * H / (H * pred_var * H + S)

        post_mean = pred_mean + K * innovation
        post_var = pred_var - K * H * pred_var
        post_var = jnp.maximum(post_var, 1e-10)

        # Log-likelihood contribution
        ll = jax.scipy.stats.binom.logpmf(
            n_correct, max_possible_correct, jnp.clip(p, 1e-10, 1 - 1e-10)
        )
        total_ll = total_ll + ll

        return (post_mean, post_var, total_ll), (post_mean, post_var)

    init_carry = (jnp.float64(init_mean), jnp.float64(init_variance), jnp.float64(0.0))
    (_, _, marginal_ll), (filtered_mean, filtered_variance) = jax.lax.scan(
        _step, init_carry, (n_correct_responses, effective_q)
    )

    return LearningFilterResult(
        filtered_mean=filtered_mean,
        filtered_variance=filtered_variance,
        marginal_log_likelihood=marginal_ll,
    )


def state_dependent_learning_smoother(
    filtered_mean: Array,
    filtered_variance: Array,
    sigma_epsilon_per_state: Array,
    discrete_state_prob: Array,
) -> LearningSmoothResult:
    """RTS smoother for state-dependent learning model.

    Parameters
    ----------
    filtered_mean : Array, shape (n_time,)
    filtered_variance : Array, shape (n_time,)
    sigma_epsilon_per_state : Array, shape (n_states,)
    discrete_state_prob : Array, shape (n_time, n_states)

    Returns
    -------
    LearningSmoothResult
    """
    q_per_state = sigma_epsilon_per_state ** 2
    effective_q = discrete_state_prob @ q_per_state

    def _step(carry, inputs):
        next_sm_mean, next_sm_var = carry
        filt_mean, filt_var, q_t = inputs

        pred_var = filt_var + q_t
        gain = filt_var / jnp.maximum(pred_var, 1e-10)

        sm_mean = filt_mean + gain * (next_sm_mean - filt_mean)
        sm_var = filt_var + gain ** 2 * (next_sm_var - pred_var)
        sm_var = jnp.maximum(sm_var, 1e-10)

        return (sm_mean, sm_var), (sm_mean, sm_var, gain)

    init_carry = (filtered_mean[-1], filtered_variance[-1])
    _, (smoothed_mean, smoothed_variance, smoother_gain) = jax.lax.scan(
        _step,
        init_carry,
        (filtered_mean[:-1], filtered_variance[:-1], effective_q[:-1]),
        reverse=True,
    )

    # Append final time step (smoother = filter at last step)
    smoothed_mean = jnp.concatenate([smoothed_mean, filtered_mean[-1:]])
    smoothed_variance = jnp.concatenate([smoothed_variance, filtered_variance[-1:]])

    return LearningSmoothResult(
        smoothed_mean=smoothed_mean,
        smoothed_variance=smoothed_variance,
        smoother_gain=smoother_gain,
    )
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_state_dependent_learning.py -v`
Expected: PASS (all 3 tests)

### Step 5: Commit

```bash
git add src/state_space_practice/state_dependent_learning.py \
        src/state_space_practice/tests/test_state_dependent_learning.py
git commit -m "feat: add state-dependent learning filter/smoother"
```

---

## Task 2: Discrete State Inference from Joint Observations

Build the forward-backward algorithm for the discrete state, with emission likelihoods computed from both behavioral and neural observations.

**Files:**
- Create: `src/state_space_practice/joint_discrete_state.py`
- Reference: `src/state_space_practice/switching_kalman.py` (forward-backward code)
- Test: `src/state_space_practice/tests/test_joint_discrete_state.py`

### Step 1: Write failing test

```python
# tests/test_joint_discrete_state.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.joint_discrete_state import (
    forward_backward_joint,
)


class TestForwardBackwardJoint:
    def test_output_shapes(self):
        n_time = 100
        n_states = 2
        # Emission log-likelihoods from two observation sources
        behav_ll = jnp.zeros((n_time, n_states))
        neural_ll = jnp.zeros((n_time, n_states))
        transition_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])
        init_prob = jnp.array([0.5, 0.5])

        gamma, xi, marginal_ll = forward_backward_joint(
            behav_emission_ll=behav_ll,
            neural_emission_ll=neural_ll,
            transition_matrix=transition_matrix,
            init_state_prob=init_prob,
        )

        assert gamma.shape == (n_time, n_states)
        assert xi.shape == (n_time - 1, n_states, n_states)
        assert marginal_ll.shape == ()

    def test_posterior_sums_to_one(self):
        n_time = 50
        n_states = 3
        rng = np.random.default_rng(42)
        behav_ll = jnp.array(rng.normal(0, 1, (n_time, n_states)))
        neural_ll = jnp.array(rng.normal(0, 1, (n_time, n_states)))
        transition_matrix = jnp.eye(n_states) * 0.9 + 0.1 / n_states
        init_prob = jnp.ones(n_states) / n_states

        gamma, _, _ = forward_backward_joint(
            behav_emission_ll=behav_ll,
            neural_emission_ll=neural_ll,
            transition_matrix=transition_matrix,
            init_state_prob=init_prob,
        )

        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-5)

    def test_strong_signal_in_one_source(self):
        """If behavioral signal strongly favors state 0 in first half,
        posterior should reflect that even with uninformative neural data."""
        n_time = 100
        behav_ll = jnp.zeros((n_time, 2))
        behav_ll = behav_ll.at[:50, 0].set(10.0)  # strong evidence for state 0
        behav_ll = behav_ll.at[50:, 1].set(10.0)   # strong evidence for state 1
        neural_ll = jnp.zeros((n_time, 2))  # uninformative

        transition_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])
        init_prob = jnp.array([0.5, 0.5])

        gamma, _, _ = forward_backward_joint(
            behav_emission_ll=behav_ll,
            neural_emission_ll=neural_ll,
            transition_matrix=transition_matrix,
            init_state_prob=init_prob,
        )

        # First half should be mostly state 0
        assert gamma[:40, 0].mean() > 0.9
        # Second half should be mostly state 1
        assert gamma[60:, 1].mean() > 0.9
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_discrete_state.py -v`
Expected: FAIL with ImportError

### Step 3: Implement forward-backward for joint observations

```python
# src/state_space_practice/joint_discrete_state.py
"""Forward-backward algorithm for discrete states with multiple observation sources.

Computes posterior state probabilities P(s_t | all observations) when the
emission likelihood is the product of likelihoods from independent observation
channels (behavioral performance + neural spiking).
"""

import jax
import jax.numpy as jnp
from jax import Array


def forward_backward_joint(
    behav_emission_ll: Array,
    neural_emission_ll: Array,
    transition_matrix: Array,
    init_state_prob: Array,
) -> tuple[Array, Array, Array]:
    """Forward-backward algorithm with joint emission likelihoods.

    Parameters
    ----------
    behav_emission_ll : Array, shape (n_time, n_states)
        Log-likelihood of behavioral observation given each state.
    neural_emission_ll : Array, shape (n_time, n_states)
        Log-likelihood of neural observations given each state.
    transition_matrix : Array, shape (n_states, n_states)
        Discrete state transition matrix. Row i is P(s_t | s_{t-1}=i).
    init_state_prob : Array, shape (n_states,)
        Initial state probabilities.

    Returns
    -------
    gamma : Array, shape (n_time, n_states)
        Posterior state probabilities P(s_t | all data).
    xi : Array, shape (n_time - 1, n_states, n_states)
        Pairwise posterior P(s_t=i, s_{t+1}=j | all data).
    marginal_log_likelihood : Array
        Total log-likelihood of the observations (scalar).
    """
    # Total emission log-likelihood is sum of independent sources
    total_emission_ll = behav_emission_ll + neural_emission_ll

    # Forward pass (in log-space for numerical stability)
    def _forward_step(carry, emission_ll_t):
        log_alpha_prev, total_ll = carry
        # Prediction: sum over previous states
        log_pred = jax.nn.logsumexp(
            log_alpha_prev[:, None] + jnp.log(transition_matrix + 1e-30),
            axis=0,
        )
        # Update with emission
        log_alpha = log_pred + emission_ll_t
        # Normalize for numerical stability
        log_norm = jax.nn.logsumexp(log_alpha)
        log_alpha = log_alpha - log_norm
        total_ll = total_ll + log_norm

        return (log_alpha, total_ll), log_alpha

    log_alpha_0 = jnp.log(init_state_prob + 1e-30) + total_emission_ll[0]
    log_norm_0 = jax.nn.logsumexp(log_alpha_0)
    log_alpha_0 = log_alpha_0 - log_norm_0

    (_, marginal_ll), log_alphas_rest = jax.lax.scan(
        _forward_step,
        (log_alpha_0, log_norm_0),
        total_emission_ll[1:],
    )
    log_alphas = jnp.concatenate([log_alpha_0[None], log_alphas_rest])

    # Backward pass
    def _backward_step(log_beta_next, emission_ll_t):
        log_msg = emission_ll_t[None, :] + log_beta_next[None, :]
        log_beta = jax.nn.logsumexp(
            jnp.log(transition_matrix + 1e-30) + log_msg, axis=1
        )
        log_beta = log_beta - jax.nn.logsumexp(log_beta)
        return log_beta, log_beta

    n_states = init_state_prob.shape[0]
    log_beta_T = jnp.zeros(n_states)
    _, log_betas_rest = jax.lax.scan(
        _backward_step,
        log_beta_T,
        total_emission_ll[1:],
        reverse=True,
    )
    log_betas = jnp.concatenate([log_betas_rest, log_beta_T[None]])

    # Posterior: gamma ∝ alpha * beta
    log_gamma = log_alphas + log_betas
    log_gamma = log_gamma - jax.nn.logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)

    # Pairwise posterior xi
    def _compute_xi(log_alpha_t, log_beta_next, emission_ll_next):
        log_xi = (
            log_alpha_t[:, None]
            + jnp.log(transition_matrix + 1e-30)
            + emission_ll_next[None, :]
            + log_beta_next[None, :]
        )
        log_xi = log_xi - jax.nn.logsumexp(log_xi)
        return jnp.exp(log_xi)

    xi = jax.vmap(_compute_xi)(
        log_alphas[:-1], log_betas[1:], total_emission_ll[1:]
    )

    return gamma, xi, marginal_ll
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_discrete_state.py -v`
Expected: PASS (all 3 tests)

### Step 5: Commit

```bash
git add src/state_space_practice/joint_discrete_state.py \
        src/state_space_practice/tests/test_joint_discrete_state.py
git commit -m "feat: add forward-backward for joint behavioral+neural discrete states"
```

---

## Task 3: State-Dependent Place Field Smoother

Build a place field smoother that uses per-state process noise, weighted by discrete state probabilities. This is the drift component.

**Files:**
- Modify: `src/state_space_practice/place_field_model.py`
- Test: `src/state_space_practice/tests/test_state_dependent_drift.py`

### Step 1: Write failing test

```python
# tests/test_state_dependent_drift.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.place_field_model import (
    build_2d_spline_basis,
    evaluate_basis,
)
from state_space_practice.state_dependent_drift import (
    state_dependent_place_field_smoother,
)


class TestStateDependentPlaceFieldSmoother:
    @pytest.fixture
    def simple_data(self):
        rng = np.random.default_rng(42)
        n_time = 500
        position = rng.uniform(0, 100, (n_time, 2))
        design_matrix, basis_info = build_2d_spline_basis(
            position, n_interior_knots=3
        )
        n_basis = basis_info["n_basis"]
        spikes = rng.poisson(0.01, n_time)
        return {
            "design_matrix": jnp.asarray(design_matrix),
            "spikes": jnp.asarray(spikes),
            "n_basis": n_basis,
            "n_time": n_time,
        }

    def test_output_shapes(self, simple_data):
        n_time = simple_data["n_time"]
        n_basis = simple_data["n_basis"]
        n_states = 2

        discrete_state_prob = jnp.ones((n_time, n_states)) * 0.5
        Q_per_state = jnp.stack(
            [jnp.eye(n_basis) * 1e-5, jnp.eye(n_basis) * 1e-3]
        )  # (n_states, n_basis, n_basis)

        result = state_dependent_place_field_smoother(
            init_mean=jnp.zeros(n_basis),
            init_cov=jnp.eye(n_basis),
            design_matrix=simple_data["design_matrix"],
            spikes=simple_data["spikes"],
            dt=0.004,
            transition_matrix=jnp.eye(n_basis),
            Q_per_state=Q_per_state,
            discrete_state_prob=discrete_state_prob,
        )

        assert result.smoother_mean.shape == (n_time, n_basis)
        assert result.smoother_cov.shape == (n_time, n_basis, n_basis)
        assert jnp.isfinite(result.marginal_log_likelihood)

    def test_single_state_matches_standard(self, simple_data):
        """With one state, should match standard smoother."""
        from state_space_practice.point_process_kalman import (
            log_conditional_intensity,
            stochastic_point_process_smoother,
        )

        n_basis = simple_data["n_basis"]
        n_time = simple_data["n_time"]
        Q = jnp.eye(n_basis) * 1e-5

        # Standard smoother
        std_sm, std_sc, _, std_ll = stochastic_point_process_smoother(
            init_mean_params=jnp.zeros(n_basis),
            init_covariance_params=jnp.eye(n_basis),
            design_matrix=simple_data["design_matrix"],
            spike_indicator=simple_data["spikes"],
            dt=0.004,
            transition_matrix=jnp.eye(n_basis),
            process_cov=Q,
            log_conditional_intensity=log_conditional_intensity,
        )

        # State-dependent with one state
        discrete_state_prob = jnp.ones((n_time, 1))
        result = state_dependent_place_field_smoother(
            init_mean=jnp.zeros(n_basis),
            init_cov=jnp.eye(n_basis),
            design_matrix=simple_data["design_matrix"],
            spikes=simple_data["spikes"],
            dt=0.004,
            transition_matrix=jnp.eye(n_basis),
            Q_per_state=Q[None],  # (1, n_basis, n_basis)
            discrete_state_prob=discrete_state_prob,
        )

        np.testing.assert_allclose(
            result.smoother_mean, std_sm, atol=1e-4
        )
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_state_dependent_drift.py -v`
Expected: FAIL with ImportError

### Step 3: Implement state-dependent place field smoother

```python
# src/state_space_practice/state_dependent_drift.py
"""State-dependent place field smoother.

Runs the point-process Laplace-EKF filter/smoother with process noise
that varies over time according to the discrete state posterior.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.point_process_kalman import (
    _point_process_laplace_update,
    log_conditional_intensity,
)
from state_space_practice.kalman import _kalman_smoother_update, symmetrize


class PlaceFieldSmootherResult(NamedTuple):
    smoother_mean: Array         # (n_time, n_basis)
    smoother_cov: Array          # (n_time, n_basis, n_basis)
    smoother_cross_cov: Array    # (n_time - 1, n_basis, n_basis)
    marginal_log_likelihood: Array  # scalar


def state_dependent_place_field_smoother(
    init_mean: Array,
    init_cov: Array,
    design_matrix: Array,
    spikes: Array,
    dt: float,
    transition_matrix: Array,
    Q_per_state: Array,
    discrete_state_prob: Array,
) -> PlaceFieldSmootherResult:
    """Point-process smoother with state-dependent process noise.

    Parameters
    ----------
    init_mean : Array, shape (n_basis,)
    init_cov : Array, shape (n_basis, n_basis)
    design_matrix : Array, shape (n_time, n_basis)
    spikes : Array, shape (n_time,)
    dt : float
    transition_matrix : Array, shape (n_basis, n_basis)
    Q_per_state : Array, shape (n_states, n_basis, n_basis)
    discrete_state_prob : Array, shape (n_time, n_states)

    Returns
    -------
    PlaceFieldSmootherResult
    """
    spikes = jnp.atleast_2d(spikes.T).T  # ensure (n_time, 1) for multi-neuron API
    if spikes.ndim == 1:
        spikes = spikes[:, None]

    # Effective Q at each time step
    # Q_t = sum_s P(s_t = s) * Q^{s}
    effective_Q = jnp.einsum("ts,sij->tij", discrete_state_prob, Q_per_state)

    # Pre-compute gradient/Hessian functions
    def _log_intensity(dm_t, x):
        return jnp.atleast_1d(log_conditional_intensity(dm_t, x))

    _grad = jax.jacfwd(_log_intensity, argnums=1)
    _hess = jax.jacfwd(_grad, argnums=1)

    # Forward filter
    def _filter_step(carry, inputs):
        mean_prev, cov_prev, total_ll = carry
        dm_t, spike_t, Q_t = inputs

        one_step_mean = transition_matrix @ mean_prev
        one_step_cov = transition_matrix @ cov_prev @ transition_matrix.T + Q_t
        one_step_cov = symmetrize(one_step_cov)

        def log_int(x):
            return _log_intensity(dm_t, x)

        def grad_log_int(x):
            return _grad(dm_t, x)

        def hess_log_int(x):
            return _hess(dm_t, x)

        post_mean, post_cov, ll = _point_process_laplace_update(
            one_step_mean, one_step_cov, spike_t, dt,
            log_int,
            grad_log_intensity_func=grad_log_int,
            hess_log_intensity_func=hess_log_int,
        )

        total_ll = total_ll + ll
        return (post_mean, post_cov, total_ll), (post_mean, post_cov)

    init_carry = (init_mean, init_cov, jnp.array(0.0))
    (_, _, marginal_ll), (filtered_mean, filtered_cov) = jax.lax.scan(
        _filter_step,
        init_carry,
        (design_matrix, spikes, effective_Q),
    )

    # Backward smoother (RTS)
    def _smoother_step(carry, inputs):
        next_sm_mean, next_sm_cov = carry
        filt_mean, filt_cov, Q_t = inputs

        # Use effective Q for this time step's prediction
        process_cov = Q_t

        sm_mean, sm_cov, cross_cov = _kalman_smoother_update(
            next_sm_mean, next_sm_cov,
            filt_mean, filt_cov,
            process_cov, transition_matrix,
        )
        return (sm_mean, sm_cov), (sm_mean, sm_cov, cross_cov)

    _, (sm_mean, sm_cov, cross_cov) = jax.lax.scan(
        _smoother_step,
        (filtered_mean[-1], filtered_cov[-1]),
        (filtered_mean[:-1], filtered_cov[:-1], effective_Q[:-1]),
        reverse=True,
    )

    smoother_mean = jnp.concatenate([sm_mean, filtered_mean[-1:]])
    smoother_cov = jnp.concatenate([sm_cov, filtered_cov[-1:]])

    return PlaceFieldSmootherResult(
        smoother_mean=smoother_mean,
        smoother_cov=smoother_cov,
        smoother_cross_cov=cross_cov,
        marginal_log_likelihood=marginal_ll,
    )
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_state_dependent_drift.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/state_dependent_drift.py \
        src/state_space_practice/tests/test_state_dependent_drift.py
git commit -m "feat: add state-dependent place field smoother"
```

---

## Task 4: JointLearningDriftModel Class

Assemble the three components (state-dependent learning, discrete state inference, state-dependent drift) into a unified model class.

**Files:**
- Create: `src/state_space_practice/joint_learning_drift_model.py`
- Test: `src/state_space_practice/tests/test_joint_learning_drift_model.py`

### Step 1: Write failing test

```python
# tests/test_joint_learning_drift_model.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.joint_learning_drift_model import (
    JointLearningDriftModel,
)


class TestJointLearningDriftModel:
    @pytest.fixture
    def simulated_data(self):
        """Simulate data with two states: low-learning/low-drift and
        high-learning/high-drift."""
        rng = np.random.default_rng(42)
        n_time = 1000
        dt = 0.004

        # True discrete state: switches every 500 steps
        true_state = np.zeros(n_time, dtype=int)
        true_state[500:] = 1

        # Position: lawnmower
        position = rng.uniform(0, 100, (n_time, 2))

        # Behavior: binomial with state-dependent learning
        n_correct = rng.binomial(10, 0.6, n_time)
        n_correct[500:] = rng.binomial(10, 0.8, 500)  # better in state 1

        # Spikes: Poisson with low rate
        spikes = rng.poisson(0.02, n_time)

        return {
            "position": position,
            "spikes": spikes,
            "n_correct": n_correct,
            "max_possible_correct": 10,
            "n_time": n_time,
            "dt": dt,
        }

    def test_init(self):
        model = JointLearningDriftModel(
            dt=0.004,
            n_discrete_states=2,
            n_interior_knots=3,
        )
        assert model.n_discrete_states == 2

    def test_fit_runs(self, simulated_data):
        model = JointLearningDriftModel(
            dt=simulated_data["dt"],
            n_discrete_states=2,
            n_interior_knots=3,
        )
        lls = model.fit(
            position=simulated_data["position"],
            spikes=simulated_data["spikes"],
            n_correct=simulated_data["n_correct"],
            max_possible_correct=simulated_data["max_possible_correct"],
            max_iter=3,
            verbose=False,
        )
        assert len(lls) == 3
        assert all(np.isfinite(ll) for ll in lls)

    def test_discrete_state_posterior_shape(self, simulated_data):
        model = JointLearningDriftModel(
            dt=simulated_data["dt"],
            n_discrete_states=2,
            n_interior_knots=3,
        )
        model.fit(
            position=simulated_data["position"],
            spikes=simulated_data["spikes"],
            n_correct=simulated_data["n_correct"],
            max_possible_correct=simulated_data["max_possible_correct"],
            max_iter=2,
            verbose=False,
        )
        assert model.discrete_state_prob.shape == (
            simulated_data["n_time"],
            2,
        )
        np.testing.assert_allclose(
            model.discrete_state_prob.sum(axis=1), 1.0, atol=1e-5
        )

    def test_learned_parameters(self, simulated_data):
        model = JointLearningDriftModel(
            dt=simulated_data["dt"],
            n_discrete_states=2,
            n_interior_knots=3,
        )
        model.fit(
            position=simulated_data["position"],
            spikes=simulated_data["spikes"],
            n_correct=simulated_data["n_correct"],
            max_possible_correct=simulated_data["max_possible_correct"],
            max_iter=3,
            verbose=False,
        )
        # Per-state learning rates should be different
        assert model.sigma_epsilon_per_state.shape == (2,)
        # Per-state drift noise should be different
        assert model.Q_drift_per_state.shape[0] == 2
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_learning_drift_model.py -v`
Expected: FAIL with ImportError

### Step 3: Implement JointLearningDriftModel

```python
# src/state_space_practice/joint_learning_drift_model.py
"""Joint learning and representational drift model.

Simultaneously infers brain state, behavioral learning, and place field
drift through a shared discrete switching state.
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.joint_discrete_state import forward_backward_joint
from state_space_practice.place_field_model import (
    build_2d_spline_basis,
    evaluate_basis,
)
from state_space_practice.state_dependent_drift import (
    state_dependent_place_field_smoother,
)
from state_space_practice.state_dependent_learning import (
    state_dependent_learning_filter,
    state_dependent_learning_smoother,
)
from state_space_practice.utils import check_converged

logger = logging.getLogger(__name__)


class JointLearningDriftModel:
    """Joint model linking brain state, behavioral learning, and place field drift.

    Discrete brain states modulate both:
    - Learning rate (process noise on behavioral learning state)
    - Drift rate (process noise on place field weights)

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    n_discrete_states : int
        Number of discrete brain states.
    n_interior_knots : int, default=3
        Spline knots per spatial dimension.
    prob_correct_by_chance : float, default=0.5
        Chance-level performance for behavioral task.
    """

    def __init__(
        self,
        dt: float,
        n_discrete_states: int = 2,
        n_interior_knots: int = 3,
        prob_correct_by_chance: float = 0.5,
    ):
        self.dt = dt
        self.n_discrete_states = n_discrete_states
        self.n_interior_knots = n_interior_knots
        self.prob_correct_by_chance = prob_correct_by_chance

        # Learned parameters
        self.sigma_epsilon_per_state: Optional[Array] = None
        self.Q_drift_per_state: Optional[Array] = None
        self.discrete_transition_matrix: Optional[Array] = None
        self.discrete_state_prob: Optional[Array] = None

        # Smoother outputs
        self.learning_state_mean: Optional[Array] = None
        self.learning_state_variance: Optional[Array] = None
        self.smoother_mean: Optional[Array] = None
        self.smoother_cov: Optional[Array] = None
        self.basis_info: Optional[dict] = None
        self.log_likelihoods: list[float] = []

    def _initialize_parameters(self, n_basis: int) -> None:
        K = self.n_discrete_states
        self.sigma_epsilon_per_state = jnp.full(K, 0.05)
        self.Q_drift_per_state = jnp.stack(
            [jnp.eye(n_basis) * 1e-5 * (k + 1) for k in range(K)]
        )
        self.discrete_transition_matrix = (
            jnp.eye(K) * 0.95 + (1 - 0.95) / K
        )
        init_prob = jnp.ones(K) / K
        self._init_state_prob = init_prob

    def _compute_behavioral_emission_ll(
        self,
        n_correct: Array,
        max_possible_correct: int,
    ) -> Array:
        """Compute P(y_t^behav | s_t) for each state using the current
        learning state estimate."""
        n_time = n_correct.shape[0]
        K = self.n_discrete_states
        ll = jnp.zeros((n_time, K))

        for k in range(K):
            # Run learning filter with this state's noise
            single_state_prob = jnp.zeros((n_time, K)).at[:, k].set(1.0)
            result = state_dependent_learning_filter(
                init_mean=0.0,
                init_variance=1.0,
                n_correct_responses=n_correct,
                max_possible_correct=max_possible_correct,
                sigma_epsilon_per_state=self.sigma_epsilon_per_state,
                discrete_state_prob=single_state_prob,
                prob_correct_by_chance=self.prob_correct_by_chance,
            )
            # Per-time-step behavioral LL under this state's parameters
            p = jax.nn.sigmoid(result.filtered_mean)
            p = jnp.clip(p, 1e-10, 1 - 1e-10)
            ll = ll.at[:, k].set(
                jax.scipy.stats.binom.logpmf(
                    n_correct, max_possible_correct, p
                )
            )

        return ll

    def _compute_neural_emission_ll(
        self,
        design_matrix: Array,
        spikes: Array,
    ) -> Array:
        """Compute P(y_t^neural | s_t) for each state, based on how well
        the current spatial weights explain the spikes under each state's Q."""
        # For simplicity, use uniform neural emission initially.
        # After first iteration, use the per-state smoothed weights
        # to compute Poisson log-likelihood.
        n_time = spikes.shape[0]
        K = self.n_discrete_states

        if self.smoother_mean is None:
            return jnp.zeros((n_time, K))

        ll = jnp.zeros((n_time, K))
        weights = self.smoother_mean
        log_rate = jnp.sum(design_matrix * weights, axis=1)
        rate = jnp.exp(log_rate) * self.dt

        # All states share the same rate estimate for now
        per_time_ll = jax.scipy.stats.poisson.logpmf(spikes, rate)
        ll = jnp.broadcast_to(per_time_ll[:, None], (n_time, K))

        return ll

    def fit(
        self,
        position: np.ndarray,
        spikes: ArrayLike,
        n_correct: ArrayLike,
        max_possible_correct: int,
        max_iter: int = 10,
        tolerance: float = 1e-4,
        n_inner_iter: int = 3,
        verbose: bool = True,
    ) -> list[float]:
        """Fit the joint model using structured EM.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
        spikes : ArrayLike, shape (n_time,)
        n_correct : ArrayLike, shape (n_time,)
            Number correct per trial.
        max_possible_correct : int
        max_iter : int
        tolerance : float
        n_inner_iter : int
            Inner iterations for discrete state convergence.
        verbose : bool

        Returns
        -------
        log_likelihoods : list[float]
        """
        position = np.asarray(position)
        spikes = jnp.asarray(spikes)
        n_correct = jnp.asarray(n_correct)
        n_time = len(spikes)

        # Build spatial basis
        design_matrix_np, self.basis_info = build_2d_spline_basis(
            position, n_interior_knots=self.n_interior_knots
        )
        design_matrix = jnp.asarray(design_matrix_np)
        n_basis = self.basis_info["n_basis"]

        self._initialize_parameters(n_basis)
        self.discrete_state_prob = jnp.ones(
            (n_time, self.n_discrete_states)
        ) / self.n_discrete_states

        self.log_likelihoods = []

        def _print(msg):
            if verbose:
                print(msg)

        _print(f"JointLearningDriftModel: n_time={n_time}, "
               f"n_basis={n_basis}, n_states={self.n_discrete_states}")

        for iteration in range(max_iter):
            total_ll = 0.0

            # --- Inner loop: alternate discrete state and continuous states ---
            for inner in range(n_inner_iter):
                # 1. Discrete state inference
                behav_ll = self._compute_behavioral_emission_ll(
                    n_correct, max_possible_correct
                )
                neural_ll = self._compute_neural_emission_ll(
                    design_matrix, spikes
                )
                self.discrete_state_prob, _, discrete_ll = forward_backward_joint(
                    behav_emission_ll=behav_ll,
                    neural_emission_ll=neural_ll,
                    transition_matrix=self.discrete_transition_matrix,
                    init_state_prob=self._init_state_prob,
                )

                # 2. Learning smoother
                learn_filter = state_dependent_learning_filter(
                    init_mean=0.0,
                    init_variance=1.0,
                    n_correct_responses=n_correct,
                    max_possible_correct=max_possible_correct,
                    sigma_epsilon_per_state=self.sigma_epsilon_per_state,
                    discrete_state_prob=self.discrete_state_prob,
                    prob_correct_by_chance=self.prob_correct_by_chance,
                )
                learn_smooth = state_dependent_learning_smoother(
                    learn_filter.filtered_mean,
                    learn_filter.filtered_variance,
                    self.sigma_epsilon_per_state,
                    self.discrete_state_prob,
                )
                self.learning_state_mean = learn_smooth.smoothed_mean
                self.learning_state_variance = learn_smooth.smoothed_variance

                # 3. Place field smoother
                pf_result = state_dependent_place_field_smoother(
                    init_mean=jnp.zeros(n_basis),
                    init_cov=jnp.eye(n_basis),
                    design_matrix=design_matrix,
                    spikes=spikes,
                    dt=self.dt,
                    transition_matrix=jnp.eye(n_basis),
                    Q_per_state=self.Q_drift_per_state,
                    discrete_state_prob=self.discrete_state_prob,
                )
                self.smoother_mean = pf_result.smoother_mean
                self.smoother_cov = pf_result.smoother_cov

            total_ll = float(
                learn_filter.marginal_log_likelihood
                + pf_result.marginal_log_likelihood
                + discrete_ll
            )
            self.log_likelihoods.append(total_ll)
            _print(f"  EM iter {iteration + 1}/{max_iter}: LL = {total_ll:.1f}")

            if not jnp.isfinite(total_ll):
                break

            if iteration > 0:
                is_converged, _ = check_converged(
                    total_ll, self.log_likelihoods[-2], tolerance
                )
                if is_converged:
                    _print(f"  Converged after {iteration + 1} iterations.")
                    break

            # --- M-step ---
            # Update per-state learning noise
            # q^s = E_P(s_t=s) [(theta_t - theta_{t-1})^2]
            dtheta = jnp.diff(self.learning_state_mean)
            dtheta_sq = dtheta ** 2 + jnp.diff(self.learning_state_variance)
            for k in range(self.n_discrete_states):
                weights_k = self.discrete_state_prob[1:, k]
                q_k = jnp.sum(weights_k * dtheta_sq) / jnp.sum(weights_k)
                self.sigma_epsilon_per_state = (
                    self.sigma_epsilon_per_state.at[k].set(jnp.sqrt(jnp.maximum(q_k, 1e-10)))
                )

            # Update per-state drift noise (diagonal)
            sm = self.smoother_mean
            dx = jnp.diff(sm, axis=0)  # (n_time-1, n_basis)
            dx_sq = dx ** 2  # approximate; ignoring covariance terms for speed
            for k in range(self.n_discrete_states):
                weights_k = self.discrete_state_prob[1:, k]
                q_diag_k = (
                    jnp.sum(weights_k[:, None] * dx_sq, axis=0)
                    / jnp.sum(weights_k)
                )
                q_diag_k = jnp.maximum(q_diag_k, 1e-10)
                self.Q_drift_per_state = (
                    self.Q_drift_per_state.at[k].set(jnp.diag(q_diag_k))
                )

            # Update discrete transition matrix
            xi = jnp.zeros(
                (self.n_discrete_states, self.n_discrete_states)
            )
            # Use discrete_state_prob to estimate transitions
            for k in range(self.n_discrete_states):
                for j in range(self.n_discrete_states):
                    xi = xi.at[k, j].set(
                        jnp.sum(
                            self.discrete_state_prob[:-1, k]
                            * self.discrete_state_prob[1:, j]
                        )
                    )
            row_sums = xi.sum(axis=1, keepdims=True)
            self.discrete_transition_matrix = xi / jnp.maximum(row_sums, 1e-10)

        return self.log_likelihoods
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_learning_drift_model.py -v`
Expected: PASS (all 4 tests)

### Step 5: Commit

```bash
git add src/state_space_practice/joint_learning_drift_model.py \
        src/state_space_practice/tests/test_joint_learning_drift_model.py
git commit -m "feat: add JointLearningDriftModel linking brain state, learning, and drift"
```

---

## Task 5: Convenience Methods and Plotting

Add user-facing methods for analysis and visualization.

**Files:**
- Modify: `src/state_space_practice/joint_learning_drift_model.py`
- Test: add to `src/state_space_practice/tests/test_joint_learning_drift_model.py`

### Step 1: Add methods to JointLearningDriftModel

Add these methods to the class:

```python
    def state_summary(self) -> dict:
        """Summarize what each discrete state represents."""
        return {
            "learning_noise_per_state": np.array(self.sigma_epsilon_per_state),
            "drift_noise_per_state": np.array(
                [float(jnp.diag(self.Q_drift_per_state[k]).mean())
                 for k in range(self.n_discrete_states)]
            ),
            "state_occupancy": np.array(
                self.discrete_state_prob.mean(axis=0)
            ),
            "transition_matrix": np.array(self.discrete_transition_matrix),
        }

    def plot_states(self, ax=None):
        """Plot discrete state posterior over time with learning curve overlay."""
        import matplotlib.pyplot as plt

        if self.discrete_state_prob is None:
            raise RuntimeError("Model has not been fitted yet.")

        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        else:
            axes = ax
            fig = axes[0].figure

        # Top: discrete state posterior
        n_time = self.discrete_state_prob.shape[0]
        t = np.arange(n_time) * self.dt
        for k in range(self.n_discrete_states):
            axes[0].fill_between(
                t,
                self.discrete_state_prob[:, k],
                alpha=0.5,
                label=f"State {k}",
            )
        axes[0].set_ylabel("P(state)")
        axes[0].legend()
        axes[0].set_title("Discrete Brain State Posterior")

        # Bottom: learning curve
        if self.learning_state_mean is not None:
            p = jax.nn.sigmoid(np.array(self.learning_state_mean))
            axes[1].plot(t, p, "k-")
            axes[1].set_ylabel("P(correct)")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_title("Learning Curve")

        fig.tight_layout()
        return fig
```

### Step 2: Write test for new methods

```python
    def test_state_summary(self, simulated_data):
        model = JointLearningDriftModel(
            dt=simulated_data["dt"],
            n_discrete_states=2,
            n_interior_knots=3,
        )
        model.fit(
            position=simulated_data["position"],
            spikes=simulated_data["spikes"],
            n_correct=simulated_data["n_correct"],
            max_possible_correct=simulated_data["max_possible_correct"],
            max_iter=2,
            verbose=False,
        )
        summary = model.state_summary()
        assert "learning_noise_per_state" in summary
        assert len(summary["learning_noise_per_state"]) == 2
        assert summary["state_occupancy"].sum() == pytest.approx(1.0, abs=0.01)
```

### Step 3: Run tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_learning_drift_model.py -v`
Expected: PASS

### Step 4: Commit

```bash
git add src/state_space_practice/joint_learning_drift_model.py \
        src/state_space_practice/tests/test_joint_learning_drift_model.py
git commit -m "feat: add convenience methods and plotting to JointLearningDriftModel"
```

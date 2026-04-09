"""Contingency-belief / latent task-state model.

An input-output HMM that infers hidden contingency or rule states from
bandit behavior (choices and rewards). The hidden state represents what
the animal believes the world rules are — not option values, not
behavioral strategy.

Model:
    s_t ~ Categorical(softmax(eta[s_{t-1}, :] + Gamma[s_{t-1}, :, :] @ h_t))
    r_t | s_t, a_t ~ Bernoulli(rho[s_t, a_t])
    a_t | s_t ~ Categorical(softmax(beta * V[s_t, :] + Theta @ z_t))

where:
    s_t: latent contingency/task state
    a_t: observed choice
    r_t: observed reward (0/1)
    h_t: transition covariates (e.g., session reset, surprise)
    z_t: choice-bias covariates (e.g., stay bias)

References
----------
[1] Smith, A.C. et al. (2004). Dynamic analysis of learning in
    behavioral experiments. J Neuroscience 24(2), 447-461.
"""

import logging
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.optimize import minimize as scipy_minimize

from state_space_practice.sgd_fitting import SGDFittableMixin

logger = logging.getLogger(__name__)


class ContingencyBeliefResult(NamedTuple):
    """Result container for contingency belief filter/smoother."""

    state_posterior: Array  # (n_trials, n_states)
    log_likelihood: Array  # scalar


def centered_log_softmax(logits: Array) -> Array:
    """Centered log-softmax: last state is the reference category.

    Parameters
    ----------
    logits : Array, shape (..., n_states - 1)
        Unconstrained logits for states 0..S-2.

    Returns
    -------
    Array, shape (..., n_states)
        Log-probabilities for all S states.
    """
    full_logits = jnp.concatenate(
        [logits, jnp.zeros_like(logits[..., :1])], axis=-1
    )
    return jax.nn.log_softmax(full_logits, axis=-1)


def centered_softmax(logits: Array) -> Array:
    """Centered softmax: last state is the reference category.

    Parameters
    ----------
    logits : Array, shape (..., n_states - 1)

    Returns
    -------
    Array, shape (..., n_states)
    """
    return jnp.exp(centered_log_softmax(logits))


def centered_softmax_inverse(probs: Array) -> Array:
    """Inverse of centered softmax: extract logits relative to last state.

    Parameters
    ----------
    probs : Array, shape (..., n_states)
        Probability vectors summing to 1.

    Returns
    -------
    Array, shape (..., n_states - 1)
        Logits relative to last state.
    """
    eps = 1e-10
    safe = jnp.clip(probs, eps, None)
    return jnp.log(safe[..., :-1]) - jnp.log(safe[..., -1:])


def transition_logits_to_matrix(logits: Array) -> Array:
    """Convert centered logits to a row-stochastic transition matrix.

    Parameters
    ----------
    logits : Array, shape (n_states, n_states - 1)
        Centered transition logits (last state is reference).

    Returns
    -------
    Array, shape (n_states, n_states)
        Row-stochastic transition matrix.
    """
    return centered_softmax(logits)


def compute_transition_matrix_from_design(
    design_row: Array,
    coefficients: Array,
) -> Array:
    """Compute one transition matrix from a design matrix row.

    Parameters
    ----------
    design_row : Array, shape (n_coefficients,)
        One row of the design matrix.
    coefficients : Array, shape (n_coefficients, n_states, n_states - 1)
        Regression coefficients for each from-state.

    Returns
    -------
    Array, shape (n_states, n_states)
        Row-stochastic transition matrix.
    """
    # For each from-state i, compute logits: design_row @ coefficients[:, i, :]
    # shape (n_states, n_states - 1)
    logits = jnp.einsum("k,kij->ij", design_row, coefficients)
    return centered_softmax(logits)


def compute_input_output_transition_matrix(
    baseline_logits: Array,
    transition_weights: Array,
    covariates_t: Array,
) -> Array:
    """Compute time-varying transition matrix from covariates.

    Uses centered softmax parameterization (last state is reference).

    Parameters
    ----------
    baseline_logits : Array, shape (n_states, n_states - 1)
        Baseline transition logits (centered softmax).
    transition_weights : Array, shape (n_states, n_states - 1, d_h)
        Covariate weights for transitions.
    covariates_t : Array, shape (d_h,)
        Transition covariates at time t.

    Returns
    -------
    Array, shape (n_states, n_states)
        Row-stochastic transition matrix at time t.
    """
    logits = baseline_logits + jnp.einsum("ijk,k->ij", transition_weights, covariates_t)
    return centered_softmax(logits)


def get_transition_prior(
    concentration: float,
    stickiness: float,
    n_states: int,
) -> Array:
    """Create Dirichlet prior parameters for transition matrix rows.

    Parameters
    ----------
    concentration : float
        Base concentration (uniform part of the Dirichlet).
    stickiness : float
        Extra concentration on the diagonal (self-transitions).
    n_states : int

    Returns
    -------
    Array, shape (n_states, n_states)
        Dirichlet alpha parameters (>= 1).
    """
    alpha = concentration * jnp.ones((n_states, n_states))
    alpha = alpha + stickiness * jnp.eye(n_states)
    return jnp.maximum(alpha, 1.0)


@jax.jit
def dirichlet_neg_log_likelihood(
    coefficients_flat: Array,
    design_matrix: Array,
    response: Array,
    alpha: Array,
    l2_penalty: float = 1e-5,
) -> Array:
    """Negative expected complete log-likelihood for transition M-step.

    Combines multinomial cross-entropy with Dirichlet prior.

    Parameters
    ----------
    coefficients_flat : Array, shape (n_coefficients * (n_states - 1),)
        Flattened regression coefficients for one from-state row.
    design_matrix : Array, shape (n_samples, n_coefficients)
    response : Array, shape (n_samples, n_states)
        Expected counts (from joint distribution) for this from-state.
    alpha : Array, shape (n_states,)
        Dirichlet prior for this row.
    l2_penalty : float
        L2 on non-intercept coefficients.

    Returns
    -------
    Array, shape ()
    """
    n_coefficients = design_matrix.shape[1]
    coefficients = coefficients_flat.reshape((n_coefficients, -1))
    log_probs = centered_log_softmax(design_matrix @ coefficients)

    n_samples = response.shape[0]
    prior = alpha - 1.0
    neg_ll = -jnp.sum((response + prior) * log_probs) / n_samples

    # L2 on non-intercept coefficients
    l2_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)
    return neg_ll + l2_term


_dirichlet_gradient = jax.grad(dirichlet_neg_log_likelihood)
_dirichlet_hessian = jax.hessian(dirichlet_neg_log_likelihood)


def compute_reward_log_likelihood(
    reward_t: ArrayLike,
    choice_t: ArrayLike,
    reward_probs: Array,
) -> Array:
    """Compute log P(reward | state, choice) for each state.

    Parameters
    ----------
    reward_t : int or Array
        Reward outcome (0 or 1).
    choice_t : int or Array
        Chosen option index.
    reward_probs : Array, shape (n_states, n_options)
        P(reward=1 | state, choice) for each state-option pair.

    Returns
    -------
    Array, shape (n_states,)
        Log-likelihood of the reward under each state.
    """
    p = reward_probs[:, choice_t]  # (n_states,)
    eps = 1e-10
    p = jnp.clip(p, eps, 1.0 - eps)
    return jnp.where(
        reward_t == 1,
        jnp.log(p),
        jnp.log(1.0 - p),
    )


def compute_choice_log_likelihood(
    choice_t: ArrayLike,
    state_values: Array,
    inverse_temperature: float,
) -> Array:
    """Compute log P(choice | state) for each state.

    Parameters
    ----------
    choice_t : int or Array
        Chosen option index.
    state_values : Array, shape (n_states, n_options)
        Value preferences per state-option pair.
    inverse_temperature : float
        Softmax temperature.

    Returns
    -------
    Array, shape (n_states,)
        Log-likelihood of the choice under each state.
    """
    logits = inverse_temperature * state_values  # (n_states, n_options)
    log_probs = jax.nn.log_softmax(logits, axis=1)  # (n_states, n_options)
    return log_probs[:, choice_t]


def contingency_belief_filter(
    choices: ArrayLike,
    rewards: ArrayLike,
    n_states: int,
    n_options: int,
    reward_probs: Array,
    state_values: Array,
    inverse_temperature: float = 1.0,
    transition_logits: Optional[Array] = None,
    transition_covariates: Optional[Array] = None,
    transition_weights: Optional[Array] = None,
    init_state_prob: Optional[Array] = None,
) -> ContingencyBeliefResult:
    """Forward filter for the contingency-belief HMM.

    Computes P(s_t | a_{1:t}, r_{1:t}) for each trial t.

    Parameters
    ----------
    choices : ArrayLike, shape (n_trials,)
        Observed choices (0-indexed).
    rewards : ArrayLike, shape (n_trials,)
        Observed rewards (0 or 1).
    n_states : int
        Number of latent contingency states.
    n_options : int
        Number of choice options.
    reward_probs : Array, shape (n_states, n_options)
        P(reward=1 | state, choice).
    state_values : Array, shape (n_states, n_options)
        Choice value preferences per state.
    inverse_temperature : float
        Softmax temperature for choice policy.
    transition_logits : Array or None, shape (n_states, n_states)
        Baseline transition logits. Default: zeros (uniform).
    transition_covariates : Array or None, shape (n_trials, d_h)
        Time-varying transition covariates.
    transition_weights : Array or None, shape (n_states, n_states, d_h)
        Weights for covariate-driven transitions.
    init_state_prob : Array or None, shape (n_states,)
        Initial state distribution. Default: uniform.

    Returns
    -------
    ContingencyBeliefResult
        state_posterior: shape (n_trials, n_states)
        log_likelihood: scalar
    """
    choices = jnp.asarray(choices, dtype=jnp.int32)
    rewards = jnp.asarray(rewards, dtype=jnp.int32)
    n_trials = choices.shape[0]

    if transition_logits is None:
        transition_logits = jnp.zeros((n_states, n_states))
    if init_state_prob is None:
        init_state_prob = jnp.ones(n_states) / n_states

    has_covariates = (
        transition_covariates is not None and transition_weights is not None
    )
    if not has_covariates:
        transition_covariates = jnp.zeros((n_trials, 1))
        transition_weights = jnp.zeros((n_states, n_states - 1, 1))

    def _update(predicted, choice_t, reward_t):
        """Bayes update: predicted → posterior given observations."""
        reward_ll = compute_reward_log_likelihood(
            reward_t, choice_t, reward_probs
        )
        choice_ll = compute_choice_log_likelihood(
            choice_t, state_values, inverse_temperature
        )
        log_obs = reward_ll + choice_ll
        log_joint = jnp.log(jnp.maximum(predicted, 1e-30)) + log_obs
        log_norm = jax.nn.logsumexp(log_joint)
        posterior = jnp.exp(log_joint - log_norm)
        return posterior, log_norm

    def _step(carry, trial_data):
        prev_belief, accum_ll = carry
        choice_t, reward_t, h_t = trial_data

        # Predict: P(s_t) = sum_i T(i→j|h_t) * P(s_{t-1}=i)
        trans = compute_input_output_transition_matrix(
            transition_logits, transition_weights, h_t
        )
        predicted = trans.T @ prev_belief

        posterior, log_norm = _update(predicted, choice_t, reward_t)
        return (posterior, accum_ll + log_norm), posterior

    # t=0: use init_state_prob directly as prior (no transition applied)
    posterior_0, log_norm_0 = _update(
        init_state_prob, choices[0], rewards[0]
    )

    # t=1:T: scan with transitions
    init_carry = (posterior_0, log_norm_0)
    remaining_inputs = (choices[1:], rewards[1:], transition_covariates[1:])
    (_, total_ll), posteriors_rest = jax.lax.scan(
        _step, init_carry, remaining_inputs
    )

    posteriors = jnp.concatenate([posterior_0[None], posteriors_rest], axis=0)

    return ContingencyBeliefResult(
        state_posterior=posteriors,
        log_likelihood=total_ll,
    )


class SmootherResult(NamedTuple):
    """Result container for contingency belief smoother."""

    smoothed_state_prob: Array  # (n_trials, n_states)
    pairwise_state_prob: Array  # (n_trials-1, n_states, n_states)
    log_likelihood: Array  # scalar


def contingency_belief_smoother(
    choices: ArrayLike,
    rewards: ArrayLike,
    n_states: int,
    n_options: int,
    reward_probs: Array,
    state_values: Array,
    inverse_temperature: float = 1.0,
    transition_logits: Optional[Array] = None,
    transition_covariates: Optional[Array] = None,
    transition_weights: Optional[Array] = None,
    init_state_prob: Optional[Array] = None,
) -> SmootherResult:
    """Forward-backward smoother for the contingency-belief HMM.

    Computes P(s_t | a_{1:T}, r_{1:T}) and P(s_{t-1}, s_t | data).

    Parameters
    ----------
    Same as contingency_belief_filter.

    Returns
    -------
    SmootherResult
        smoothed_state_prob: shape (n_trials, n_states)
        pairwise_state_prob: shape (n_trials-1, n_states, n_states)
        log_likelihood: scalar
    """
    choices = jnp.asarray(choices, dtype=jnp.int32)
    rewards = jnp.asarray(rewards, dtype=jnp.int32)
    n_trials = choices.shape[0]

    if transition_logits is None:
        transition_logits = jnp.zeros((n_states, n_states))
    if init_state_prob is None:
        init_state_prob = jnp.ones(n_states) / n_states

    has_covariates = (
        transition_covariates is not None and transition_weights is not None
    )
    if not has_covariates:
        transition_covariates = jnp.zeros((n_trials, 1))
        transition_weights = jnp.zeros((n_states, n_states - 1, 1))

    # --- Forward pass: store filter beliefs and per-step info ---
    def _fwd_update(predicted, choice_t, reward_t):
        reward_ll = compute_reward_log_likelihood(
            reward_t, choice_t, reward_probs
        )
        choice_ll = compute_choice_log_likelihood(
            choice_t, state_values, inverse_temperature
        )
        log_obs = reward_ll + choice_ll
        log_joint = jnp.log(jnp.maximum(predicted, 1e-30)) + log_obs
        log_norm = jax.nn.logsumexp(log_joint)
        posterior = jnp.exp(log_joint - log_norm)
        return posterior, log_norm

    def _forward_step(carry, trial_data):
        prev_belief, accum_ll = carry
        choice_t, reward_t, h_t = trial_data

        trans = compute_input_output_transition_matrix(
            transition_logits, transition_weights, h_t
        )
        predicted = trans.T @ prev_belief
        posterior, log_norm = _fwd_update(predicted, choice_t, reward_t)

        return (posterior, accum_ll + log_norm), (posterior, predicted, trans)

    # t=0: use init_state_prob directly
    posterior_0, log_norm_0 = _fwd_update(
        init_state_prob, choices[0], rewards[0]
    )
    # Dummy transition for t=0 (not used by backward pass)
    dummy_trans = centered_softmax(transition_logits)

    # t=1:T
    init_carry = (posterior_0, log_norm_0)
    remaining_inputs = (choices[1:], rewards[1:], transition_covariates[1:])
    (_, total_ll), (filt_rest, pred_rest, trans_rest) = (
        jax.lax.scan(_forward_step, init_carry, remaining_inputs)
    )

    # Concatenate t=0 with t=1:T
    filter_beliefs = jnp.concatenate([posterior_0[None], filt_rest], axis=0)
    predicted_beliefs = jnp.concatenate(
        [init_state_prob[None], pred_rest], axis=0
    )
    trans_matrices = jnp.concatenate(
        [dummy_trans[None], trans_rest], axis=0
    )

    # --- Backward pass ---
    def _backward_step(beta_next, step_data):
        filter_t, predicted_tp1, trans_tp1 = step_data
        # beta_next = P(s_{t+1} | data) / P(s_{t+1} | data_{1:t})
        # smoothed_t = filter_t * sum_j T(i→j) * beta_next[j] / predicted[j]
        ratio = beta_next / jnp.maximum(predicted_tp1, 1e-30)
        beta_t = filter_t * (trans_tp1 @ ratio)
        # Normalize for stability
        beta_t = beta_t / jnp.maximum(beta_t.sum(), 1e-30)
        return beta_t, beta_t

    # Backward scan: from T-1 down to 0
    # step_data for backward step t uses filter[t], predicted[t+1], trans[t+1]
    backward_inputs = (
        filter_beliefs[:-1],      # filter[0:T-1]
        predicted_beliefs[1:],    # predicted[1:T]
        trans_matrices[1:],       # trans[1:T]
    )
    _, smoothed_interior = jax.lax.scan(
        _backward_step,
        filter_beliefs[-1],  # init: last filter belief = smoothed belief
        backward_inputs,
        reverse=True,
    )
    # Concatenate: smoothed[0:T-1] from backward + filter[-1] as last
    smoothed = jnp.concatenate(
        [smoothed_interior, filter_beliefs[-1:]], axis=0
    )

    # --- Pairwise state probabilities P(s_{t-1}=i, s_t=j | data) ---
    def _pairwise(step_data):
        smooth_t, predicted_tp1, trans_tp1, smooth_tp1 = step_data
        # P(s_t=i, s_{t+1}=j | data) = smooth[t,i] * T[i,j] * smooth[t+1,j] / predicted[t+1,j]
        ratio = smooth_tp1 / jnp.maximum(predicted_tp1, 1e-30)
        joint = smooth_t[:, None] * trans_tp1 * ratio[None, :]
        # Normalize
        joint = joint / jnp.maximum(joint.sum(), 1e-30)
        return joint

    pairwise_inputs = (
        smoothed[:-1],
        predicted_beliefs[1:],
        trans_matrices[1:],
        smoothed[1:],
    )
    pairwise = jax.vmap(_pairwise)(pairwise_inputs)

    return SmootherResult(
        smoothed_state_prob=smoothed,
        pairwise_state_prob=pairwise,
        log_likelihood=total_ll,
    )


class ContingencyBeliefModel(SGDFittableMixin):
    """Contingency-belief model for multi-armed bandit behavior.

    Infers hidden task-state/contingency from choices and rewards using
    an input-output HMM with EM or SGD fitting.

    Transition model uses a design-matrix approach: each row of the
    transition matrix is parameterized as
    ``softmax(design_matrix @ coefficients[:, from_state, :])``,
    where the design matrix can include an intercept, covariates, and
    spline bases. For stationary models, the design matrix is just
    an intercept column.

    EM M-step for transitions uses per-row optimization with
    Dirichlet-Multinomial loss (Newton-CG via scipy). SGD optimizes
    all parameters jointly including transition coefficients.

    Parameters
    ----------
    n_states : int
        Number of latent contingency states.
    n_options : int
        Number of choice options.
    init_inverse_temperature : float
        Starting inverse temperature.
    init_diagonal : float
        Initial self-transition probability for diagonal initialization.
    concentration : float
        Dirichlet prior concentration (uniform part).
    stickiness : float
        Extra Dirichlet concentration on diagonal (self-transitions).
    transition_regularization : float
        L2 penalty on non-intercept transition coefficients.
    """

    def __init__(
        self,
        n_states: int,
        n_options: int,
        init_inverse_temperature: float = 1.0,
        init_diagonal: float = 0.9,
        concentration: float = 1.0,
        stickiness: float = 0.0,
        transition_regularization: float = 1e-5,
    ):
        self.n_states = n_states
        self.n_options = n_options
        self.inverse_temperature_ = init_inverse_temperature
        self.init_diagonal = init_diagonal
        self.concentration = concentration
        self.stickiness = stickiness
        self.transition_regularization = transition_regularization

        # Initialize parameters
        self.reward_probs_ = jnp.ones((n_states, n_options)) / 2
        self.state_values_ = jax.random.normal(
            jax.random.PRNGKey(0), (n_states, n_options)
        ) * 0.1

        # Transition coefficients: (n_coefficients, n_states, n_states - 1)
        # Initialized from diagonal transition matrix
        diag = np.full(n_states, init_diagonal)
        init_trans = np.diag(diag)
        off_diag = (1.0 - diag) / max(n_states - 1, 1)
        init_trans = init_trans + off_diag[:, None] * (1 - np.eye(n_states))
        init_logits = np.asarray(centered_softmax_inverse(jnp.array(init_trans)))
        # Start with intercept-only (1 coefficient)
        self.transition_coefficients_ = jnp.array(
            init_logits[None, :, :]  # (1, n_states, n_states - 1)
        )
        self._transition_design_matrix: Optional[Array] = None

        # Fitted state (public attributes per plan)
        self.state_posterior_: Optional[Array] = None
        self.smoothed_state_posterior_: Optional[Array] = None
        self._smoother_result: Optional[SmootherResult] = None
        self.log_likelihood_: Optional[float] = None
        self.log_likelihood_history_: Optional[list[float]] = None
        self._n_trials: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self._smoother_result is not None

    def _get_transition_logits(self) -> Array:
        """Get current transition logits from coefficients (intercept row)."""
        # For stationary model: logits = coefficients[0]
        # For non-stationary: this returns the intercept-only logits
        return self.transition_coefficients_[0]  # (n_states, n_states - 1)

    def _build_design_matrix(
        self, n_trials: int, transition_covariates: Optional[Array] = None
    ) -> Array:
        """Build the transition design matrix.

        For stationary (no covariates): intercept only, shape (n_trials, 1).
        For non-stationary: intercept + covariates, shape (n_trials, 1 + d_h).
        """
        intercept = jnp.ones((n_trials, 1))
        if transition_covariates is not None:
            return jnp.concatenate([intercept, transition_covariates], axis=1)
        return intercept

    def _get_transition_matrix_at(self, design_row: Array) -> Array:
        """Compute transition matrix for one time step."""
        return compute_transition_matrix_from_design(
            design_row, self.transition_coefficients_
        )

    def _smoother_kwargs(self, choices, rewards):
        """Build kwargs for filter/smoother calls, including covariates."""
        coefs = self.transition_coefficients_
        kwargs = dict(
            choices=choices,
            rewards=rewards,
            n_states=self.n_states,
            n_options=self.n_options,
            reward_probs=self.reward_probs_,
            state_values=self.state_values_,
            inverse_temperature=self.inverse_temperature_,
            transition_logits=coefs[0],  # intercept: (S, S-1)
        )
        # Pass covariate-driven transitions if non-stationary
        if coefs.shape[0] > 1 and self._transition_design_matrix is not None:
            kwargs["transition_weights"] = jnp.moveaxis(coefs[1:], 0, -1)
            kwargs["transition_covariates"] = self._transition_design_matrix[:, 1:]
        return kwargs

    def fit(
        self,
        choices: ArrayLike,
        rewards: ArrayLike,
        transition_covariates: Optional[ArrayLike] = None,
        max_iter: int = 50,
        tolerance: float = 1e-4,
    ) -> list[float]:
        """Fit via EM algorithm.

        Parameters
        ----------
        choices : ArrayLike, shape (n_trials,)
        rewards : ArrayLike, shape (n_trials,)
        transition_covariates : ArrayLike or None, shape (n_trials, d_h)
        max_iter : int
        tolerance : float

        Returns
        -------
        log_likelihoods : list of float
        """
        choices = jnp.asarray(choices, dtype=jnp.int32)
        rewards = jnp.asarray(rewards, dtype=jnp.int32)
        self._n_trials = int(choices.shape[0])

        if transition_covariates is not None:
            cov = jnp.asarray(transition_covariates)
        else:
            cov = None
        self._transition_design_matrix = self._build_design_matrix(
            self._n_trials, cov
        )

        # Expand coefficients if covariates added
        n_coefficients = self._transition_design_matrix.shape[1]
        if self.transition_coefficients_.shape[0] < n_coefficients:
            old = self.transition_coefficients_
            new = jnp.zeros((
                n_coefficients, self.n_states, self.n_states - 1
            ))
            new = new.at[:old.shape[0]].set(old)
            self.transition_coefficients_ = new

        log_likelihoods: list[float] = []
        prev_ll = float("-inf")

        for iteration in range(max_iter):
            # E-step
            kwargs = self._smoother_kwargs(choices, rewards)
            result = contingency_belief_smoother(**kwargs)
            self._smoother_result = result
            self.smoothed_state_posterior_ = result.smoothed_state_prob
            ll = float(result.log_likelihood)
            log_likelihoods.append(ll)

            if abs(ll - prev_ll) < tolerance and iteration > 0:
                logger.info(f"Converged at iteration {iteration + 1}")
                break
            prev_ll = ll

            # M-step
            self._m_step(choices, rewards, result)

        self.log_likelihood_ = log_likelihoods[-1]
        self.log_likelihood_history_ = log_likelihoods
        # Populate causal posterior from final parameters
        filter_kwargs = self._smoother_kwargs(choices, rewards)
        filter_result = contingency_belief_filter(**filter_kwargs)
        self.state_posterior_ = filter_result.state_posterior
        return log_likelihoods

    def _m_step(self, choices, rewards, result):
        """M-step: update reward_probs and transition coefficients."""
        gamma = result.smoothed_state_prob  # (T, S)
        xi = result.pairwise_state_prob  # (T-1, S, S)

        # Update reward_probs: weighted counts
        eps = 1e-10
        choice_onehot = (
            choices[:, None] == jnp.arange(self.n_options)[None, :]
        )  # (T, K)
        reward_counts = jnp.einsum(
            "ts,tk,t->sk", gamma, choice_onehot, rewards.astype(float)
        )
        total_counts = jnp.einsum("ts,tk->sk", gamma, choice_onehot)
        self.reward_probs_ = jnp.clip(
            reward_counts / jnp.maximum(total_counts, eps), eps, 1 - eps
        )

        # Update transition coefficients per row via Newton-CG
        alpha = get_transition_prior(
            self.concentration, self.stickiness, self.n_states
        )
        design = np.asarray(self._transition_design_matrix[:-1])  # (T-1, d)

        for from_state in range(self.n_states):
            x0 = np.asarray(
                self.transition_coefficients_[:, from_state, :].ravel()
            )
            response = np.asarray(xi[:, from_state, :])
            row_alpha = np.asarray(alpha[from_state])

            result_opt = scipy_minimize(
                fun=lambda c: float(dirichlet_neg_log_likelihood(
                    jnp.array(c), jnp.array(design), jnp.array(response),
                    jnp.array(row_alpha), self.transition_regularization,
                )),
                x0=x0,
                method="L-BFGS-B",
                jac=lambda c: np.asarray(_dirichlet_gradient(
                    jnp.array(c), jnp.array(design), jnp.array(response),
                    jnp.array(row_alpha), self.transition_regularization,
                )),
                options={"maxiter": 50},
            )
            n_coef = design.shape[1]
            self.transition_coefficients_ = self.transition_coefficients_.at[
                :, from_state, :
            ].set(jnp.array(result_opt.x.reshape((n_coef, -1))))

    def predict_state_posterior(
        self,
        choices: ArrayLike,
        rewards: ArrayLike,
        transition_covariates: Optional[ArrayLike] = None,
    ) -> Array:
        """Predict smoothed state posterior for given data."""
        choices = jnp.asarray(choices, dtype=jnp.int32)
        rewards = jnp.asarray(rewards, dtype=jnp.int32)
        n_trials = int(choices.shape[0])
        # Temporarily set design matrix for this prediction
        old_dm = self._transition_design_matrix
        if transition_covariates is not None:
            self._transition_design_matrix = self._build_design_matrix(
                n_trials, jnp.asarray(transition_covariates)
            )
        elif old_dm is None:
            self._transition_design_matrix = self._build_design_matrix(n_trials)
        kwargs = self._smoother_kwargs(choices, rewards)
        result = contingency_belief_smoother(**kwargs)
        self._transition_design_matrix = old_dm
        return result.smoothed_state_prob

    # --- SGDFittableMixin protocol ---

    def fit_sgd(
        self,
        choices: ArrayLike,
        rewards: ArrayLike,
        transition_covariates: Optional[ArrayLike] = None,
        optimizer=None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol=None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        SGD learns all parameters: reward_probs, state_values,
        inverse_temperature, and transition_coefficients.
        """
        choices = jnp.asarray(choices, dtype=jnp.int32)
        rewards = jnp.asarray(rewards, dtype=jnp.int32)
        self._n_trials = int(choices.shape[0])

        if transition_covariates is not None:
            cov = jnp.asarray(transition_covariates)
        else:
            cov = None
        self._transition_design_matrix = self._build_design_matrix(
            self._n_trials, cov
        )

        # Expand coefficients if covariates added
        n_coefficients = self._transition_design_matrix.shape[1]
        if self.transition_coefficients_.shape[0] < n_coefficients:
            old = self.transition_coefficients_
            new = jnp.zeros((
                n_coefficients, self.n_states, self.n_states - 1
            ))
            new = new.at[:old.shape[0]].set(old)
            self.transition_coefficients_ = new

        return super().fit_sgd(
            choices, rewards,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    @property
    def _n_timesteps(self) -> int:
        return self._n_trials

    def _check_sgd_initialized(self) -> None:
        pass

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            POSITIVE,
            UNCONSTRAINED,
            UNIT_INTERVAL,
        )

        params = {
            "reward_probs": self.reward_probs_,
            "state_values": self.state_values_,
            "inverse_temperature": jnp.array(self.inverse_temperature_),
            "transition_coefficients": self.transition_coefficients_,
        }
        spec = {
            "reward_probs": UNIT_INTERVAL,
            "state_values": UNCONSTRAINED,
            "inverse_temperature": POSITIVE,
            "transition_coefficients": UNCONSTRAINED,
        }
        return params, spec

    def _sgd_loss_fn(self, params: dict, choices: Array, rewards: Array) -> Array:
        coefs = params["transition_coefficients"]
        transition_logits = coefs[0]  # intercept: (n_states, n_states-1)

        kwargs = dict(
            choices=choices,
            rewards=rewards,
            n_states=self.n_states,
            n_options=self.n_options,
            reward_probs=params["reward_probs"],
            state_values=params["state_values"],
            inverse_temperature=params["inverse_temperature"],
            transition_logits=transition_logits,
        )

        # If non-stationary (>1 coefficient), extract covariate weights
        if coefs.shape[0] > 1 and self._transition_design_matrix is not None:
            # weights[i, j, k] = coefficients[k+1, i, j] for covariates
            transition_weights = jnp.moveaxis(coefs[1:], 0, -1)
            # Covariates from design matrix (exclude intercept column)
            kwargs["transition_weights"] = transition_weights
            kwargs["transition_covariates"] = self._transition_design_matrix[:, 1:]

        result = contingency_belief_filter(**kwargs)
        return -result.log_likelihood

    def _store_sgd_params(self, params: dict) -> None:
        self.reward_probs_ = params["reward_probs"]
        self.state_values_ = params["state_values"]
        self.inverse_temperature_ = float(params["inverse_temperature"])
        self.transition_coefficients_ = params["transition_coefficients"]

    def _finalize_sgd(self, choices: Array, rewards: Array) -> None:
        kwargs = self._smoother_kwargs(choices, rewards)
        result = contingency_belief_smoother(**kwargs)
        self._smoother_result = result
        self.smoothed_state_posterior_ = result.smoothed_state_prob
        self.log_likelihood_ = float(result.log_likelihood)
        # Also populate causal posterior
        filter_result = contingency_belief_filter(**kwargs)
        self.state_posterior_ = filter_result.state_posterior

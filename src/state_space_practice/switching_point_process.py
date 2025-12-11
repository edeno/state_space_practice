"""Switching point-process Kalman filter for spike-based observations.

This module implements a Switching Linear Dynamical System (SLDS) with
point-process (spike) observations using the Laplace-EKF approach from
Eden & Brown (2004).

The model combines:
- Oscillator network dynamics with switching discrete states (DIM/CNM style)
- Point-process observations (spikes) using Laplace approximation

EM Algorithm Components
-----------------------
For EM learning with point-process observations:

**E-step:**
1. Filter: Use `switching_point_process_filter()` from this module
2. Smoother: Use `switching_kalman_smoother()` from `switching_kalman.py`
   (smoother is observation-model agnostic once filter provides Gaussian posteriors)

**M-step for dynamics parameters:**
Use `switching_kalman_maximization_step()` from `switching_kalman.py` directly.
This function is observation-model agnostic - it operates on smoother outputs
(means, covariances, discrete state probabilities) which are Gaussian regardless
of observation model.

Example::

    from state_space_practice.switching_kalman import (
        switching_kalman_maximization_step,
        switching_kalman_smoother,
    )
    from state_space_practice.switching_point_process import (
        switching_point_process_filter,
        update_spike_glm_params,
    )

    # E-step
    filter_outputs = switching_point_process_filter(...)
    smoother_outputs = switching_kalman_smoother(*filter_outputs[:4], ...)

    # M-step for dynamics (A, Q, B, initial state, discrete transitions)
    (
        new_A, _, new_Q, _, new_init_mean, new_init_cov,
        new_discrete_trans, new_init_discrete_prob
    ) = switching_kalman_maximization_step(
        obs=spikes,  # Required arg but not used for dynamics
        state_cond_smoother_means=smoother_outputs[5],
        state_cond_smoother_covs=smoother_outputs[6],
        smoother_discrete_state_prob=smoother_outputs[2],
        smoother_joint_discrete_state_prob=smoother_outputs[3],
        pair_cond_smoother_cross_cov=smoother_outputs[7],
        pair_cond_smoother_means=smoother_outputs[8],
    )
    # NOTE: measurement_matrix and measurement_cov returns (indices 1, 3)
    # should be IGNORED for point-process models - they assume Gaussian obs.

    # M-step for spike GLM params (baseline, weights)
    new_spike_params = update_spike_glm_params(
        spikes, smoother_mean, current_spike_params, dt
    )

**Important notes on M-step:**

1. The raw M-step does NOT guarantee positive semi-definite covariances.
   When a discrete state has low probability or insufficient data, the process
   covariance can have negative eigenvalues. PSD enforcement (e.g., adding
   regularization ``Q = Q + eps*I``) should be handled at the model class level.

2. The ``measurement_matrix`` and ``measurement_cov`` returns from
   ``switching_kalman_maximization_step`` should be ignored for point-process
   models since they assume Gaussian observations.

References
----------
1. Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
   Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
   Neural Computation 16, 971-998.
2. Shumway, R.H., and Stoffer, D.S. (1991). Dynamic Linear Models With Switching.
3. Murphy, K.P. (1998). Switching Kalman Filters.
"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import symmetrize
from state_space_practice.utils import check_converged
from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
)
from state_space_practice.point_process_kalman import (
    _point_process_laplace_update,
)
from state_space_practice.switching_kalman import (
    _scale_likelihood,
    _update_discrete_state_probabilities,
    collapse_gaussian_mixture_per_discrete_state,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
)


@dataclass
class SpikeObsParams:
    """Spike observation model parameters.

    This dataclass holds the parameters for a linear log-intensity observation
    model for point-process (spike) data:

        log(lambda_n(t)) = baseline_n + weights_n @ x_t

    where lambda_n(t) is the firing rate of neuron n at time t, and x_t is the
    latent state.

    Attributes
    ----------
    baseline : Array, shape (n_neurons,)
        Baseline log-rate b_n for each neuron. When the latent state is zero,
        the firing rate is exp(baseline).
    weights : Array, shape (n_neurons, n_latent)
        Linear weights C mapping the oscillator state to log-rates.
        weights[n, :] gives the coupling of neuron n to each latent dimension.
    """

    baseline: Array
    weights: Array


def point_process_kalman_update(
    one_step_mean: Array,
    one_step_cov: Array,
    y_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
    diagonal_boost: float = 1e-9,
    grad_log_intensity_func: Callable[[Array], Array] | None = None,
    hess_log_intensity_func: Callable[[Array], Array] | None = None,
) -> tuple[Array, Array, float]:
    """Single point-process Laplace-EKF update for multiple neurons.

    Performs a Bayesian update of the latent state posterior given observed
    spike counts, using a Laplace (Gaussian) approximation to the posterior.

    The observation model is:
        y_n ~ Poisson(exp(log_intensity_func(x)[n]) * dt)

    The update uses a single Newton-Raphson step from the prior mean to
    approximate the posterior.

    Parameters
    ----------
    one_step_mean : Array, shape (n_latent,)
        Predicted mean from dynamics: A @ m_{t-1}
    one_step_cov : Array, shape (n_latent, n_latent)
        Predicted covariance: A @ P_{t-1} @ A.T + Q
    y_t : Array, shape (n_neurons,)
        Spike counts at time t for all neurons
    dt : float
        Time bin width in seconds
    log_intensity_func : Callable[[Array], Array]
        Function mapping state (n_latent,) to log-intensities (n_neurons,).
        Should return log(lambda) where lambda is firing rate in Hz.
    diagonal_boost : float, default=1e-9
        Small value added to precision matrix diagonal for numerical stability
        when solving linear systems.
    grad_log_intensity_func : Callable[[Array], Array] | None, optional
        Pre-computed gradient function (Jacobian) of log_intensity_func.
        If None, computed via jax.jacfwd(log_intensity_func).
        Passing pre-computed functions can improve compilation speed when
        this function is called repeatedly inside a JIT-compiled context.
    hess_log_intensity_func : Callable[[Array], Array] | None, optional
        Pre-computed Hessian function of log_intensity_func.
        If None, computed via jax.jacfwd of the gradient function.

    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
        Updated state mean after incorporating spike observations
    posterior_cov : Array, shape (n_latent, n_latent)
        Updated state covariance after incorporating spike observations
    log_likelihood : float
        Log p(y_t | y_{1:t-1}) approximated at posterior mode

    Notes
    -----
    The Laplace approximation uses the predicted mean as the expansion point
    for a single Newton step update. For multiple neurons, the gradients and
    Hessians are summed across neurons.

    The log-likelihood is computed as the sum of Poisson log-pmfs evaluated
    at the expected intensity (using the predicted mean).

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """
    # Delegate to the shared helper in point_process_kalman.py
    return _point_process_laplace_update(
        one_step_mean=one_step_mean,
        one_step_cov=one_step_cov,
        spike_indicator_t=y_t,
        dt=dt,
        log_intensity_func=log_intensity_func,
        diagonal_boost=diagonal_boost,
        grad_log_intensity_func=grad_log_intensity_func,
        hess_log_intensity_func=hess_log_intensity_func,
    )


def _point_process_predict_and_update(
    prev_state_cond_mean: Array,
    prev_state_cond_cov: Array,
    y_t: Array,
    continuous_transition_matrix: Array,
    process_cov: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
) -> tuple[Array, Array, float]:
    """Predict with dynamics, then update with spike observations.

    This function combines the one-step prediction (using dynamics parameters)
    with the point-process Laplace update. It is designed to be vmapped over
    different state pairs (i, j) in the switching filter.

    Returns the pair-conditional posterior:
        p(x_t | y_{1:t}, S_{t-1}=i, S_t=j)

    Parameters
    ----------
    prev_state_cond_mean : Array, shape (n_latent,)
        Previous state-conditional mean: m_{t-1|t-1}^i
    prev_state_cond_cov : Array, shape (n_latent, n_latent)
        Previous state-conditional covariance: P_{t-1|t-1}^i
    y_t : Array, shape (n_neurons,)
        Spike counts at time t for all neurons
    continuous_transition_matrix : Array, shape (n_latent, n_latent)
        State transition matrix for discrete state j: A_j
    process_cov : Array, shape (n_latent, n_latent)
        Process noise covariance for discrete state j: Q_j
    dt : float
        Time bin width in seconds
    log_intensity_func : Callable[[Array], Array]
        Function mapping state (n_latent,) to log-intensities (n_neurons,).

    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
        Pair-conditional posterior mean
    posterior_cov : Array, shape (n_latent, n_latent)
        Pair-conditional posterior covariance
    log_likelihood : float
        Log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)
    """
    # One-step prediction using dynamics for state j
    one_step_mean = continuous_transition_matrix @ prev_state_cond_mean
    one_step_cov = (
        continuous_transition_matrix
        @ prev_state_cond_cov
        @ continuous_transition_matrix.T
        + process_cov
    )
    one_step_cov = symmetrize(one_step_cov)

    # Point-process Laplace update
    return point_process_kalman_update(
        one_step_mean, one_step_cov, y_t, dt, log_intensity_func
    )


def _point_process_update_per_discrete_state_pair(
    prev_state_cond_mean: Array,
    prev_state_cond_cov: Array,
    y_t: Array,
    continuous_transition_matrix: Array,
    process_cov: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
) -> tuple[Array, Array, Array]:
    """Compute pair-conditional posteriors for all (i, j) state pairs.

    This function vmaps over both:
    - Previous discrete state i (varying prev_state_cond_mean/cov)
    - Next discrete state j (varying continuous_transition_matrix/process_cov)

    The result contains posteriors p(x_t | y_{1:t}, S_{t-1}=i, S_t=j) for all
    combinations of (i, j).

    Parameters
    ----------
    prev_state_cond_mean : Array, shape (n_latent, n_discrete_states)
        Previous state-conditional means, one per discrete state i
    prev_state_cond_cov : Array, shape (n_latent, n_latent, n_discrete_states)
        Previous state-conditional covariances, one per discrete state i
    y_t : Array, shape (n_neurons,)
        Spike counts at time t for all neurons
    continuous_transition_matrix : Array, shape (n_latent, n_latent, n_discrete_states)
        State transition matrices, one per discrete state j
    process_cov : Array, shape (n_latent, n_latent, n_discrete_states)
        Process noise covariances, one per discrete state j
    dt : float
        Time bin width in seconds
    log_intensity_func : Callable[[Array], Array]
        Function mapping state (n_latent,) to log-intensities (n_neurons,).

    Returns
    -------
    pair_cond_mean : Array, shape (n_latent, n_discrete_states, n_discrete_states)
        Pair-conditional posterior means. pair_cond_mean[:, i, j] is the
        posterior mean for p(x_t | y_{1:t}, S_{t-1}=i, S_t=j).
    pair_cond_cov : Array, shape (n_latent, n_latent, n_discrete_states, n_discrete_states)
        Pair-conditional posterior covariances.
    pair_cond_log_likelihood : Array, shape (n_discrete_states, n_discrete_states)
        Pair-conditional log-likelihoods. pair_cond_log_likelihood[i, j] is
        log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j).
    """

    # Create a version of predict_and_update that only takes array arguments
    # (log_intensity_func and dt are captured in the closure)
    def _update(prev_mean, prev_cov, A, Q):
        return _point_process_predict_and_update(
            prev_mean, prev_cov, y_t, A, Q, dt, log_intensity_func
        )

    # Double vmap:
    # Inner vmap: over previous state i (axis -1 of prev_mean/prev_cov)
    # Outer vmap: over next state j (axis -1 of A/Q)
    vmapped_update = jax.vmap(
        jax.vmap(
            _update,
            in_axes=(-1, -1, None, None),  # vmap over prev state i
            out_axes=-1,
        ),
        in_axes=(None, None, -1, -1),  # vmap over next state j
        out_axes=-1,
    )

    return vmapped_update(
        prev_state_cond_mean,
        prev_state_cond_cov,
        continuous_transition_matrix,
        process_cov,
    )


def _armijo_line_search(
    params: Array,
    delta: Array,
    gradient: Array,
    loss_fn: Callable[[Array], float],
    current_loss: float,
    beta: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 20,
) -> Array:
    """Backtracking line search with Armijo condition.

    Finds a step size alpha such that the sufficient decrease (Armijo)
    condition is satisfied:
        loss(params - alpha * delta) <= loss(params) - c * alpha * grad^T @ delta

    Parameters
    ----------
    params : Array, shape (n_params,)
        Current parameter values.
    delta : Array, shape (n_params,)
        Search direction (typically Newton direction H^{-1} @ g).
    gradient : Array, shape (n_params,)
        Gradient at current params.
    loss_fn : Callable[[Array], float]
        Loss function to minimize. Takes params and returns scalar loss.
    current_loss : float
        Loss at current params (avoids recomputation).
    beta : float, default=0.5
        Step reduction factor for backtracking.
    c : float, default=1e-4
        Armijo constant for sufficient decrease condition.
    max_iter : int, default=20
        Maximum number of backtracking iterations.

    Returns
    -------
    alpha : Array
        Step size satisfying Armijo condition.

    Notes
    -----
    The search starts with alpha=1 and repeatedly multiplies by beta
    until the Armijo condition is satisfied or max_iter is reached.
    """
    directional_derivative = jnp.dot(gradient, delta)

    def line_search_step(carry, _):
        alpha, _ = carry
        new_params_trial = params - alpha * delta
        new_loss = loss_fn(new_params_trial)

        # Armijo condition: new_loss <= current_loss - c * alpha * directional_derivative
        sufficient_decrease = new_loss <= current_loss - c * alpha * directional_derivative

        # Reduce alpha if not sufficient decrease
        new_alpha = jnp.where(sufficient_decrease, alpha, alpha * beta)
        return (new_alpha, sufficient_decrease), None

    (final_alpha, _), _ = jax.lax.scan(
        line_search_step, (jnp.array(1.0), jnp.array(False)), None, length=max_iter
    )

    return final_alpha


def _single_neuron_glm_loss(
    baseline: Array,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    dt: float,
) -> Array:
    """Poisson negative log-likelihood for a single neuron GLM.

    Computes the negative log-likelihood for a Poisson GLM observation model:
        y_{n,t} ~ Poisson(exp(baseline + weights @ x_t) * dt)

    This is the plug-in method that uses smoother_mean as a point estimate
    for the latent state x_t.

    Parameters
    ----------
    baseline : Array, shape ()
        Baseline log-rate for the neuron (0-D array).
    weights : Array, shape (n_latent,)
        Linear weights mapping latent state to log-rate.
    y_n : Array, shape (n_time,)
        Spike counts for the neuron at each timestep.
    smoother_mean : Array, shape (n_time, n_latent)
        Smoothed latent state estimates (used as design matrix).
    dt : float
        Time bin width in seconds.

    Returns
    -------
    loss : Array, shape ()
        Negative log-likelihood: -sum_t [y_t * eta_t - exp(eta_t) * dt]
        where eta_t = baseline + weights @ smoother_mean[t].

    Notes
    -----
    The Poisson log-likelihood is:
        log p(y | eta) = sum_t [y_t * log(lambda_t * dt) - lambda_t * dt - log(y_t!)]

    where lambda_t = exp(eta_t) is the firing rate. Since log(y_t!) doesn't
    depend on parameters, we omit it and return the negative of:
        sum_t [y_t * eta_t - exp(eta_t) * dt]

    This function is designed to be differentiable via JAX autodiff for
    gradient-based optimization.
    """
    # Linear predictor: eta_t = baseline + weights @ x_t
    # smoother_mean: (n_time, n_latent), weights: (n_latent,)
    eta = baseline + smoother_mean @ weights  # (n_time,)

    # Expected spike count: lambda_t * dt = exp(eta_t) * dt
    expected_counts = jnp.exp(eta) * dt  # (n_time,)

    # Poisson log-likelihood (without log(y!) term):
    # sum_t [y_t * eta_t - exp(eta_t) * dt]
    # Note: We include the dt scaling in the expected counts
    log_likelihood = jnp.sum(y_n * eta - expected_counts)

    # Return negative log-likelihood for minimization
    return -log_likelihood


def _single_neuron_glm_step(
    baseline: Array,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Single Newton-Raphson step for Poisson GLM parameter optimization.

    Takes one Newton step to minimize the negative log-likelihood
    of a Poisson GLM observation model. Uses backtracking line search
    to ensure descent. The update is:
        params_new = params - alpha * H^{-1} @ g

    where g is the gradient, H is the Hessian, and alpha is determined
    by backtracking line search.

    Parameters
    ----------
    baseline : Array, shape ()
        Current baseline log-rate for the neuron (0-D array).
    weights : Array, shape (n_latent,)
        Current linear weights mapping latent state to log-rate.
    y_n : Array, shape (n_time,)
        Spike counts for the neuron at each timestep.
    smoother_mean : Array, shape (n_time, n_latent)
        Smoothed latent state estimates (used as design matrix).
    dt : float
        Time bin width in seconds.

    Returns
    -------
    new_baseline : Array, shape ()
        Updated baseline after Newton step (0-D array).
    new_weights : Array, shape (n_latent,)
        Updated weights after Newton step.

    Notes
    -----
    For Poisson GLM with log-link, the negative log-likelihood is convex,
    so Newton's method is guaranteed to converge. The gradient and Hessian are:

        gradient = -X.T @ (y - mu)
        Hessian = X.T @ diag(mu) @ X

    where X is the design matrix [1, smoother_mean], y is spike counts,
    and mu = exp(X @ params) * dt is the expected count.

    We add a small regularization to the Hessian diagonal for numerical stability.
    Backtracking line search ensures the step decreases the objective.
    """
    n_latent = weights.shape[0]

    # Build design matrix: [1, smoother_mean] of shape (n_time, 1 + n_latent)
    n_time = smoother_mean.shape[0]
    ones = jnp.ones((n_time, 1))
    design_matrix = jnp.concatenate([ones, smoother_mean], axis=1)  # (n_time, 1+n_latent)

    # Concatenate parameters into a single vector: [baseline, weights]
    params = jnp.concatenate([jnp.atleast_1d(baseline), weights])  # (1 + n_latent,)

    # Compute linear predictor and expected counts
    eta = design_matrix @ params  # (n_time,)
    mu = jnp.exp(eta) * dt  # Expected spike counts (n_time,)

    # Compute gradient: -X.T @ (y - mu)
    residual = y_n - mu  # (n_time,)
    gradient = -design_matrix.T @ residual  # (1 + n_latent,)

    # Compute Hessian: X.T @ diag(mu) @ X
    weighted_design = design_matrix * mu[:, None]  # (n_time, 1+n_latent)
    hessian = design_matrix.T @ weighted_design  # (1+n_latent, 1+n_latent)

    # Add small regularization for numerical stability
    reg = 1e-6 * jnp.eye(1 + n_latent)
    hessian = hessian + reg

    # Newton direction: delta = H^{-1} @ g
    delta = jnp.linalg.solve(hessian, gradient)

    # Current loss for backtracking line search
    current_loss = jnp.sum(mu - y_n * eta)  # NLL without constant term

    # Define loss function for line search
    def loss_fn(p):
        eta_trial = design_matrix @ p
        mu_trial = jnp.exp(eta_trial) * dt
        return jnp.sum(mu_trial - y_n * eta_trial)

    # Backtracking line search with Armijo condition
    final_alpha = _armijo_line_search(params, delta, gradient, loss_fn, current_loss)

    # Apply final step
    new_params = params - final_alpha * delta

    # Extract updated baseline and weights
    new_baseline = new_params[0]
    new_weights = new_params[1:]

    return new_baseline, new_weights


def _single_neuron_glm_step_second_order(
    baseline: Array,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    smoother_cov: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Single Newton step for Poisson GLM with second-order expectation.

    Uses the second-order approximation that accounts for state uncertainty:
        E[exp(c @ x)] = exp(c @ m + 0.5 * c @ P @ c)

    This gives more accurate parameter estimates when the smoother
    covariance is significant.

    Parameters
    ----------
    baseline : Array, shape ()
        Current baseline log-rate for the neuron (0-D array).
    weights : Array, shape (n_latent,)
        Current linear weights mapping latent state to log-rate.
    y_n : Array, shape (n_time,)
        Spike counts for the neuron at each timestep.
    smoother_mean : Array, shape (n_time, n_latent)
        Smoothed latent state estimates.
    smoother_cov : Array, shape (n_time, n_latent, n_latent)
        Smoothed latent state covariances.
    dt : float
        Time bin width in seconds.

    Returns
    -------
    new_baseline : Array, shape ()
        Updated baseline after Newton step (0-D array).
    new_weights : Array, shape (n_latent,)
        Updated weights after Newton step.
    """
    n_latent = weights.shape[0]
    n_time = smoother_mean.shape[0]

    # Build design matrix with intercept: [1, smoother_mean]
    ones = jnp.ones((n_time, 1))
    design_matrix = jnp.concatenate([ones, smoother_mean], axis=1)

    # Concatenate parameters
    params = jnp.concatenate([jnp.atleast_1d(baseline), weights])

    # Compute variance correction: 0.5 * c @ P_t @ c for each timestep
    # This is a quadratic form: 0.5 * weights @ P_t @ weights
    def compute_variance_correction(P_t):
        return 0.5 * weights @ P_t @ weights

    variance_corrections = jax.vmap(compute_variance_correction)(smoother_cov)

    # Compute linear predictor with variance correction
    eta = design_matrix @ params + variance_corrections  # (n_time,)
    mu = jnp.exp(eta) * dt  # Expected spike counts with second-order correction

    # For the gradient w.r.t. baseline, the variance correction doesn't depend on it
    # For the gradient w.r.t. weights, we need d/dc [c @ m + 0.5 * c @ P @ c]
    #   = m + P @ c

    # The gradient of the expected log-likelihood term exp(eta) w.r.t params is:
    # d/d_params [sum_t mu_t] = sum_t mu_t * d_eta/d_params
    #
    # For the weights component, d_eta/dc = m_t + P_t @ c
    # For the baseline, d_eta/db = 1
    #
    # So we need an "effective design matrix" that accounts for the variance term

    # Effective design matrix for second-order method
    # Column 0: ones (for baseline)
    # Columns 1+: m_t + P_t @ weights (for weights)
    def compute_effective_design_row(m_t, P_t):
        effective_latent = m_t + P_t @ weights
        return jnp.concatenate([jnp.ones(1), effective_latent])

    effective_design = jax.vmap(compute_effective_design_row)(smoother_mean, smoother_cov)

    # Gradient: -X_eff.T @ (y - mu)
    residual = y_n - mu
    gradient = -effective_design.T @ residual

    # Hessian: For Poisson GLM, the Hessian also involves second derivatives
    # of the variance correction term. For simplicity, we use:
    # H ≈ X_eff.T @ diag(mu) @ X_eff
    # This is a good approximation when the variance correction is small
    weighted_design = effective_design * mu[:, None]
    hessian = effective_design.T @ weighted_design

    # Regularization
    reg = 1e-6 * jnp.eye(1 + n_latent)
    hessian = hessian + reg

    # Newton direction
    delta = jnp.linalg.solve(hessian, gradient)

    # Current loss for backtracking
    current_loss = jnp.sum(mu - y_n * (design_matrix @ params + variance_corrections))

    # Define loss function for line search (includes variance correction recomputation)
    def loss_fn(p):
        new_weights_trial = p[1:]

        # Recompute variance corrections with new weights
        def compute_var_corr_trial(P_t):
            return 0.5 * new_weights_trial @ P_t @ new_weights_trial

        var_corr_trial = jax.vmap(compute_var_corr_trial)(smoother_cov)

        eta_trial = design_matrix @ p + var_corr_trial
        mu_trial = jnp.exp(eta_trial) * dt
        return jnp.sum(mu_trial - y_n * (design_matrix @ p + var_corr_trial))

    # Backtracking line search with Armijo condition
    final_alpha = _armijo_line_search(params, delta, gradient, loss_fn, current_loss)

    new_params = params - final_alpha * delta
    new_baseline = new_params[0]
    new_weights = new_params[1:]

    return new_baseline, new_weights


def update_spike_glm_params(
    spikes: Array,
    smoother_mean: Array,
    current_params: SpikeObsParams,
    dt: float,
    max_iter: int = 10,
    smoother_cov: Array | None = None,
    use_second_order: bool = False,
) -> SpikeObsParams:
    """M-step for spike observation parameters.

    Updates the spike GLM parameters (baseline, weights) by maximizing
    E_q[log p(y | x; C, b)] with respect to C and b.

    Two methods are available:
    - Plug-in method (default): Uses smoother_mean directly, ignoring uncertainty.
    - Second-order method: Accounts for state uncertainty using the correction
      E[exp(c @ x)] = exp(c @ m + 0.5 * c @ P @ c).

    This runs Newton-Raphson optimization for each neuron independently
    for max_iter iterations.

    Parameters
    ----------
    spikes : Array, shape (n_time, n_neurons)
        Observed spike counts for all neurons at each timestep.
    smoother_mean : Array, shape (n_time, n_latent)
        Smoothed latent state estimates from the E-step.
    current_params : SpikeObsParams
        Current parameter estimates (for warm-starting).
    dt : float
        Time bin width in seconds.
    max_iter : int, default=10
        Maximum Newton iterations per neuron.
    smoother_cov : Array | None, shape (n_time, n_latent, n_latent), optional
        Smoothed latent state covariances. Required if use_second_order=True.
    use_second_order : bool, default=False
        If True, use second-order expectation method that accounts for
        state uncertainty. Requires smoother_cov to be provided.

    Returns
    -------
    SpikeObsParams
        Updated baseline and weights for all neurons.

    Notes
    -----
    The plug-in method treats the smoothed mean as a fixed design matrix,
    ignoring uncertainty in the latent state estimates. This is a common
    approximation that works well when the smoother covariance is small.

    The second-order method provides more accurate estimates when state
    uncertainty is significant, at the cost of additional computation.
    """
    n_neurons = spikes.shape[1]
    n_time = smoother_mean.shape[0]
    n_latent = smoother_mean.shape[1]

    # Validate inputs for second-order method
    if use_second_order:
        if smoother_cov is None:
            raise ValueError("smoother_cov required when use_second_order=True")
        expected_shape = (n_time, n_latent, n_latent)
        if smoother_cov.shape != expected_shape:
            raise ValueError(
                f"smoother_cov shape {smoother_cov.shape} incompatible with "
                f"expected shape {expected_shape}"
            )

    # Initialize with current parameters
    baselines = current_params.baseline
    weights = current_params.weights

    if use_second_order:

        # Run Newton iterations for all neurons (second-order)
        def iterate_all_neurons_second_order(carry, _):
            baselines, weights = carry

            def update_neuron(neuron_idx):
                b = baselines[neuron_idx]
                w = weights[neuron_idx]
                y_n = spikes[:, neuron_idx]
                new_b, new_w = _single_neuron_glm_step_second_order(
                    b, w, y_n, smoother_mean, smoother_cov, dt
                )
                return new_b, new_w

            neuron_indices = jnp.arange(n_neurons)
            new_baselines, new_weights = jax.vmap(update_neuron)(neuron_indices)

            return (new_baselines, new_weights), None

        (final_baselines, final_weights), _ = jax.lax.scan(
            iterate_all_neurons_second_order, (baselines, weights), None, length=max_iter
        )
    else:
        # Run Newton iterations for all neurons (plug-in)
        def iterate_all_neurons(carry, _):
            baselines, weights = carry

            def update_neuron(neuron_idx):
                b = baselines[neuron_idx]
                w = weights[neuron_idx]
                y_n = spikes[:, neuron_idx]
                new_b, new_w = _single_neuron_glm_step(b, w, y_n, smoother_mean, dt)
                return new_b, new_w

            neuron_indices = jnp.arange(n_neurons)
            new_baselines, new_weights = jax.vmap(update_neuron)(neuron_indices)

            return (new_baselines, new_weights), None

        (final_baselines, final_weights), _ = jax.lax.scan(
            iterate_all_neurons, (baselines, weights), None, length=max_iter
        )

    return SpikeObsParams(baseline=final_baselines, weights=final_weights)


def switching_point_process_filter(
    init_state_cond_mean: Array,
    init_state_cond_cov: Array,
    init_discrete_state_prob: Array,
    spikes: Array,
    discrete_transition_matrix: Array,
    continuous_transition_matrix: Array,
    process_cov: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
) -> tuple[Array, Array, Array, Array, float]:
    """Switching point-process Kalman filter for spike observations.

    This filter implements a Switching Linear Dynamical System (SLDS) with
    point-process (spike) observations using the Laplace-EKF approach. It
    maintains exact per-state-pair structure: p(x_t | y_{1:t}, S_{t-1}=i, S_t=j)
    for all state pairs (i, j).

    The model is:
        x_t = A_{s_t} @ x_{t-1} + w_t,  w_t ~ N(0, Q_{s_t})
        y_{n,t} ~ Poisson(exp(log_intensity_func(x_t)[n]) * dt)

    Parameters
    ----------
    init_state_cond_mean : Array, shape (n_latent, n_discrete_states)
        Initial state-conditional means p(x_0 | S_0 = j) for each discrete state.
    init_state_cond_cov : Array, shape (n_latent, n_latent, n_discrete_states)
        Initial state-conditional covariances for each discrete state.
    init_discrete_state_prob : Array, shape (n_discrete_states,)
        Initial discrete state probabilities p(S_0 = j).
    spikes : Array, shape (n_time, n_neurons)
        Observed spike counts for all neurons at each timestep.
    discrete_transition_matrix : Array, shape (n_discrete_states, n_discrete_states)
        Transition probabilities P(S_t = j | S_{t-1} = i). Entry [i, j] gives
        probability of transitioning from state i to state j.
    continuous_transition_matrix : Array, shape (n_latent, n_latent, n_discrete_states)
        State transition matrices A_j for each discrete state j.
    process_cov : Array, shape (n_latent, n_latent, n_discrete_states)
        Process noise covariances Q_j for each discrete state j.
    dt : float
        Time bin width in seconds.
    log_intensity_func : Callable[[Array], Array]
        Function mapping latent state (n_latent,) to log-intensities (n_neurons,).
        Should return log(lambda) where lambda is firing rate in Hz.

    Returns
    -------
    state_cond_filter_mean : Array, shape (n_time, n_latent, n_discrete_states)
        Filtered state-conditional means p(x_t | y_{1:t}, S_t = j).
    state_cond_filter_cov : Array, shape (n_time, n_latent, n_latent, n_discrete_states)
        Filtered state-conditional covariances.
    filter_discrete_state_prob : Array, shape (n_time, n_discrete_states)
        Filtered discrete state probabilities p(S_t = j | y_{1:t}).
    last_pair_cond_filter_mean : Array, shape (n_latent, n_discrete_states, n_discrete_states)
        Pair-conditional filter means at the last timestep. Entry [:, i, j] is
        the mean for p(x_T | y_{1:T}, S_{T-1}=i, S_T=j). Needed by smoother.
    marginal_log_likelihood : float
        Marginal log-likelihood log p(y_{1:T}).

    Notes
    -----
    The filter mirrors the structure of `switching_kalman_filter` but replaces
    the Gaussian observation update with a point-process Laplace-EKF update.

    At each timestep, the filter:
    1. Computes pair-conditional posteriors for all (i, j) state pairs
    2. Updates discrete state probabilities using the HMM forward algorithm
    3. Collapses pair-conditional to state-conditional via Gaussian mixture

    **Performance**: For production use, wrap this function with ``jax.jit``
    for significant speedups. The function uses ``jax.lax.scan`` and ``vmap``
    internally, which benefit greatly from JIT compilation::

        jitted_filter = jax.jit(switching_point_process_filter, static_argnums=(8,))
        results = jitted_filter(init_mean, init_cov, init_prob, spikes,
                                trans_mat, cont_trans, proc_cov, dt, log_intensity)

    Note that ``log_intensity_func`` (argument 8) must be marked as static
    since it's a Python callable.

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    [2] Murphy, K.P. (1998). Switching Kalman Filters.
    [3] Shumway, R.H., and Stoffer, D.S. (1991). Dynamic Linear Models With Switching.
    """

    def _step(
        carry: tuple[Array, Array, Array, float],
        y_t: Array,
    ) -> tuple[tuple[Array, Array, Array, float], tuple[Array, Array, Array, Array]]:
        """One step of the switching point-process filter.

        Parameters
        ----------
        carry : tuple
            prev_state_cond_filter_mean : Array, shape (n_latent, n_discrete_states)
                Previous state-conditional means.
            prev_state_cond_filter_cov : Array, shape (n_latent, n_latent, n_discrete_states)
                Previous state-conditional covariances.
            prev_filter_discrete_prob : Array, shape (n_discrete_states,)
                Previous discrete state probabilities.
            marginal_log_likelihood : float
                Accumulated marginal log-likelihood.
        y_t : Array, shape (n_neurons,)
            Spike counts at current timestep.

        Returns
        -------
        carry : tuple
            state_cond_filter_mean : Array, shape (n_latent, n_discrete_states)
                Posterior state-conditional means.
            state_cond_filter_cov : Array, shape (n_latent, n_latent, n_discrete_states)
                Posterior state-conditional covariances.
            filter_discrete_prob : Array, shape (n_discrete_states,)
                Posterior discrete state probabilities.
            marginal_log_likelihood : float
                Updated accumulated marginal log-likelihood.
        stack : tuple
            state_cond_filter_mean : Array, shape (n_latent, n_discrete_states)
                Posterior state-conditional means.
            state_cond_filter_cov : Array, shape (n_latent, n_latent, n_discrete_states)
                Posterior state-conditional covariances.
            filter_discrete_prob : Array, shape (n_discrete_states,)
                Posterior discrete state probabilities.
            pair_cond_filter_mean : Array, shape (n_latent, n_discrete_states, n_discrete_states)
                Pair-conditional filter means for smoother.
        """
        (
            prev_state_cond_filter_mean,
            prev_state_cond_filter_cov,
            prev_filter_discrete_prob,
            marginal_log_likelihood,
        ) = carry

        # 1. Compute pair-conditional posteriors p(x_t | y_{1:t}, S_{t-1}=i, S_t=j)
        #    for ALL (i, j) pairs using point-process Laplace update
        (
            pair_cond_filter_mean,
            pair_cond_filter_cov,
            pair_cond_log_likelihood,
        ) = _point_process_update_per_discrete_state_pair(
            prev_state_cond_filter_mean,
            prev_state_cond_filter_cov,
            y_t,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # 2. Scale likelihood for numerical stability
        pair_cond_likelihood_scaled, ll_max = _scale_likelihood(
            pair_cond_log_likelihood
        )

        # 3. Update discrete state probabilities (HMM forward step)
        (
            filter_discrete_prob,
            filter_backward_cond_prob,
            predictive_likelihood_term_sum,
        ) = _update_discrete_state_probabilities(
            pair_cond_likelihood_scaled,
            discrete_transition_matrix,
            prev_filter_discrete_prob,
        )

        # 4. Accumulate marginal log-likelihood
        marginal_log_likelihood += ll_max + jnp.log(predictive_likelihood_term_sum)

        # 5. Collapse pair-conditional Gaussians to state-conditional
        #    p(x_t | y_{1:t}, S_t=j) by marginalizing over S_{t-1}=i
        state_cond_filter_mean, state_cond_filter_cov = (
            collapse_gaussian_mixture_per_discrete_state(
                pair_cond_filter_mean,
                pair_cond_filter_cov,
                filter_backward_cond_prob,
            )
        )

        return (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_prob,
            marginal_log_likelihood,
        ), (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_prob,
            pair_cond_filter_mean,
        )

    # Run the filter using jax.lax.scan
    marginal_log_likelihood = jnp.array(0.0)
    (_, _, _, marginal_log_likelihood), (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_state_prob,
        pair_cond_filter_mean,
    ) = jax.lax.scan(
        _step,
        (
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            marginal_log_likelihood,
        ),
        spikes,
    )

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_state_prob,
        pair_cond_filter_mean[-1],  # Last timestep pair-conditional for smoother
        marginal_log_likelihood,
    )


class SwitchingSpikeOscillatorModel:
    """Switching oscillator network with spike-based observations.

    This model combines:
    - Oscillator network dynamics (DIM/CNM style transition matrices)
    - Switching discrete states for different coupling patterns
    - Point-process (spike) observations via Laplace-EKF

    The latent state x_t has dimension 2 * n_oscillators (amplitude and phase
    for each oscillator). The dynamics switch between n_discrete_states
    different configurations, each with potentially different coupling patterns.

    The observation model is a Poisson point-process with log-linear intensity:
        log(lambda_n(t)) = baseline_n + weights_n @ x_t

    Parameters
    ----------
    n_oscillators : int
        Number of latent oscillators. Must be positive. State dimension will
        be 2 * n_oscillators (amplitude and phase for each).
    n_neurons : int
        Number of observed neurons (spike trains). Must be positive.
    n_discrete_states : int
        Number of discrete network states (coupling configurations). Must be
        positive.
    sampling_freq : float
        Sampling frequency in Hz. Must be positive.
    dt : float
        Time bin width in seconds. Must be positive. Typically dt <= 1/sampling_freq
        for numerical stability.
    discrete_transition_diag : Array | None, optional
        Diagonal elements of the discrete transition matrix (self-transition
        probabilities). Values should be in [0, 1]. Shape must be (n_discrete_states,)
        if provided. If None, defaults to 0.95 for all states.
    update_continuous_transition_matrix : bool, default=True
        Whether to update A (dynamics) during M-step.
    update_process_cov : bool, default=True
        Whether to update Q (process noise) during M-step.
    update_discrete_transition_matrix : bool, default=True
        Whether to update Z (discrete transitions) during M-step.
    update_spike_params : bool, default=True
        Whether to update spike GLM parameters (baseline, weights) during M-step.
    update_init_mean : bool, default=True
        Whether to update initial mean during M-step.
    update_init_cov : bool, default=True
        Whether to update initial covariance during M-step.

    Attributes
    ----------
    n_latent : int
        Dimension of the continuous latent state (2 * n_oscillators).
    discrete_transition_diag : Array
        Diagonal of discrete transition matrix.

    Notes
    -----
    This model uses the exact per-state-pair structure for the switching filter,
    which maintains EM monotonicity (up to the Laplace approximation).

    The smoother is observation-model agnostic and reuses
    ``switching_kalman_smoother`` from the switching_kalman module.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> model = SwitchingSpikeOscillatorModel(
    ...     n_oscillators=2,
    ...     n_neurons=10,
    ...     n_discrete_states=3,
    ...     sampling_freq=100.0,
    ...     dt=0.01,
    ... )
    >>> model.n_latent
    4
    >>> model.n_discrete_states
    3
    """

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        discrete_transition_diag: Array | None = None,
        update_continuous_transition_matrix: bool = True,
        update_process_cov: bool = True,
        update_discrete_transition_matrix: bool = True,
        update_spike_params: bool = True,
        update_init_mean: bool = True,
        update_init_cov: bool = True,
    ) -> None:
        """Initialize the switching spike oscillator model.

        Parameters
        ----------
        n_oscillators : int
            Number of latent oscillators.
        n_neurons : int
            Number of observed neurons.
        n_discrete_states : int
            Number of discrete network states.
        sampling_freq : float
            Sampling frequency in Hz.
        dt : float
            Time bin width in seconds.
        discrete_transition_diag : Array | None, optional
            Diagonal of discrete transition matrix. Defaults to 0.95.
        update_continuous_transition_matrix : bool, default=True
            Update A during M-step.
        update_process_cov : bool, default=True
            Update Q during M-step.
        update_discrete_transition_matrix : bool, default=True
            Update Z during M-step.
        update_spike_params : bool, default=True
            Update spike GLM params during M-step.
        update_init_mean : bool, default=True
            Update initial mean during M-step.
        update_init_cov : bool, default=True
            Update initial covariance during M-step.

        Raises
        ------
        ValueError
            If any of n_oscillators, n_neurons, n_discrete_states, sampling_freq,
            or dt is not positive. Also if discrete_transition_diag has wrong shape.
        """
        # Validate input parameters
        if n_oscillators <= 0:
            raise ValueError(
                f"n_oscillators must be positive. Got {n_oscillators}."
            )
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive. Got {n_neurons}.")
        if n_discrete_states <= 0:
            raise ValueError(
                f"n_discrete_states must be positive. Got {n_discrete_states}."
            )
        if sampling_freq <= 0:
            raise ValueError(
                f"sampling_freq must be positive. Got {sampling_freq}."
            )
        if dt <= 0:
            raise ValueError(f"dt must be positive. Got {dt}.")
        if discrete_transition_diag is not None:
            if discrete_transition_diag.shape != (n_discrete_states,):
                raise ValueError(
                    f"discrete_transition_diag shape mismatch: expected "
                    f"({n_discrete_states},), got {discrete_transition_diag.shape}."
                )

        self.n_oscillators = n_oscillators
        self.n_neurons = n_neurons
        self.n_discrete_states = n_discrete_states
        self.sampling_freq = sampling_freq
        self.dt = dt
        self.n_latent = 2 * n_oscillators

        # Set default discrete transition diagonal if not provided
        if discrete_transition_diag is None:
            self.discrete_transition_diag = jnp.full(
                (n_discrete_states,), 0.95, dtype=jnp.float32
            )
        else:
            self.discrete_transition_diag = discrete_transition_diag

        # Store update flags for M-step
        self.update_continuous_transition_matrix = update_continuous_transition_matrix
        self.update_process_cov = update_process_cov
        self.update_discrete_transition_matrix = update_discrete_transition_matrix
        self.update_spike_params = update_spike_params
        self.update_init_mean = update_init_mean
        self.update_init_cov = update_init_cov

        # Placeholders for model parameters (initialized by _initialize_parameters)
        self.init_mean: Array
        self.init_cov: Array
        self.init_discrete_state_prob: Array
        self.discrete_transition_matrix: Array
        self.continuous_transition_matrix: Array
        self.process_cov: Array
        self.spike_params: SpikeObsParams

        # Placeholders for smoother results (computed in E-step)
        # Only state-conditional statistics are stored (needed for M-step)
        self.smoother_state_cond_mean: Array
        self.smoother_state_cond_cov: Array
        self.smoother_discrete_state_prob: Array
        self.smoother_joint_discrete_state_prob: Array
        self.smoother_pair_cond_cross_cov: Array
        self.smoother_pair_cond_means: Array

    def __repr__(self) -> str:
        """Return string representation of the model."""
        params = [
            f"n_oscillators={self.n_oscillators}",
            f"n_neurons={self.n_neurons}",
            f"n_discrete_states={self.n_discrete_states}",
            f"sampling_freq={self.sampling_freq}",
            f"dt={self.dt}",
        ]

        update_flags = {
            "A": self.update_continuous_transition_matrix,
            "Q": self.update_process_cov,
            "Z": self.update_discrete_transition_matrix,
            "spike": self.update_spike_params,
            "m0": self.update_init_mean,
            "P0": self.update_init_cov,
        }

        flags_str = ", ".join(f"Update({k})={v}" for k, v in update_flags.items())

        return f"<{self.__class__.__name__}: {', '.join(params)}, [{flags_str}]>"

    def _initialize_parameters(self, key: jax.random.PRNGKey) -> None:
        """Initialize all model parameters.

        Sets up initial values for all model parameters including:
        - Initial state mean and covariance
        - Discrete state transition probabilities
        - Continuous state transition matrices
        - Process noise covariances
        - Spike observation parameters (baseline and weights)

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random number generator key for reproducible initialization.

        Notes
        -----
        The initialization follows these conventions:
        - Initial state mean: drawn from standard normal distribution
        - Initial state covariance: identity matrix for each discrete state
        - Initial discrete state probabilities: uniform across states
        - Discrete transition matrix: based on `discrete_transition_diag`
        - Continuous transition matrix: uncoupled oscillators with default
          frequencies and damping coefficients
        - Process covariance: identity scaled by 0.01
        - Spike baseline: zero (corresponding to 1 Hz baseline rate)
        - Spike weights: small random values (scaled by 0.1)
        """
        k1, k2 = jax.random.split(key)

        # Initialize discrete state probabilities (uniform)
        self._initialize_discrete_state_prob()

        # Initialize discrete transition matrix
        self._initialize_discrete_transition_matrix()

        # Initialize continuous state (mean and covariance)
        self._initialize_continuous_state(k1)

        # Initialize continuous transition matrix
        self._initialize_continuous_transition_matrix()

        # Initialize process covariance
        self._initialize_process_covariance()

        # Initialize spike observation parameters
        self._initialize_spike_params(k2)

        # Validate all shapes
        self._validate_parameter_shapes()

    def _initialize_discrete_state_prob(self) -> None:
        """Initialize uniform discrete state probabilities."""
        self.init_discrete_state_prob = (
            jnp.ones(self.n_discrete_states) / self.n_discrete_states
        )

    def _initialize_discrete_transition_matrix(self) -> None:
        """Initialize discrete state transition matrix.

        Constructs the transition matrix based on `discrete_transition_diag`.
        Off-diagonal elements are set to distribute the remaining probability
        mass equally among other states.
        """
        diag = self.discrete_transition_diag
        transition_matrix = jnp.diag(diag)

        if self.n_discrete_states == 1:
            # Single state: transition matrix is just [[1.0]]
            self.discrete_transition_matrix = jnp.array([[1.0]])
            return

        # Compute off-diagonal values for each row
        off_diag = (1.0 - diag) / (self.n_discrete_states - 1.0)

        # Add off-diagonal elements, avoiding the diagonal
        transition_matrix = (
            transition_matrix
            + jnp.ones((self.n_discrete_states, self.n_discrete_states))
            * off_diag[:, None]
            - jnp.diag(off_diag)
        )

        # Ensure rows sum to 1 (handle potential floating point issues)
        self.discrete_transition_matrix = transition_matrix / jnp.sum(
            transition_matrix, axis=1, keepdims=True
        )

    def _initialize_continuous_state(self, key: jax.random.PRNGKey) -> None:
        """Initialize continuous state mean and covariance.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for sampling initial mean.
        """
        # Initial mean: sample from standard normal, same for all discrete states
        mean = jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros(self.n_latent),
            cov=jnp.eye(self.n_latent),
        )
        self.init_mean = jnp.stack([mean] * self.n_discrete_states, axis=1)

        # Initial covariance: identity for each discrete state
        self.init_cov = jnp.stack(
            [jnp.eye(self.n_latent)] * self.n_discrete_states, axis=2
        )

    def _initialize_continuous_transition_matrix(self) -> None:
        """Initialize continuous state transition matrices.

        Uses uncoupled oscillator dynamics with default frequencies
        (uniform in [5, 15] Hz) and damping coefficients (0.95).
        """
        # Default frequencies: spread across a reasonable range
        default_freqs = jnp.linspace(5.0, 15.0, self.n_oscillators)

        # Default damping: stable but not overly damped
        default_damping = jnp.full(self.n_oscillators, 0.95)

        # Construct uncoupled transition matrix
        A = construct_common_oscillator_transition_matrix(
            freqs=default_freqs,
            auto_regressive_coef=default_damping,
            sampling_freq=self.sampling_freq,
        )

        # Stack for each discrete state (same initial A for all states)
        self.continuous_transition_matrix = jnp.stack(
            [A] * self.n_discrete_states, axis=2
        )

    def _initialize_process_covariance(self) -> None:
        """Initialize process noise covariances.

        Uses block-diagonal structure with small variance (0.01) per oscillator.
        """
        # Default variance: small process noise
        default_variance = jnp.full(self.n_oscillators, 0.01)

        Q = construct_common_oscillator_process_covariance(variance=default_variance)

        # Stack for each discrete state
        self.process_cov = jnp.stack([Q] * self.n_discrete_states, axis=2)

    def _initialize_spike_params(self, key: jax.random.PRNGKey) -> None:
        """Initialize spike observation parameters.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for sampling initial weights.
        """
        # Baseline: zero (exp(0) = 1 Hz baseline firing rate)
        baseline = jnp.zeros(self.n_neurons)

        # Weights: small random values
        weights = (
            jax.random.normal(key, (self.n_neurons, self.n_latent)) * 0.1
        )

        self.spike_params = SpikeObsParams(baseline=baseline, weights=weights)

    def _validate_parameter_shapes(self) -> None:
        """Validate that all initialized parameters have correct shapes.

        Raises
        ------
        ValueError
            If any parameter has an incorrect shape.
        """
        if self.init_mean.shape != (self.n_latent, self.n_discrete_states):
            raise ValueError(
                f"init_mean shape mismatch: expected "
                f"({self.n_latent}, {self.n_discrete_states}), "
                f"got {self.init_mean.shape}."
            )
        if self.init_cov.shape != (
            self.n_latent,
            self.n_latent,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"init_cov shape mismatch: expected "
                f"({self.n_latent}, {self.n_latent}, {self.n_discrete_states}), "
                f"got {self.init_cov.shape}."
            )
        if self.init_discrete_state_prob.shape != (self.n_discrete_states,):
            raise ValueError(
                f"init_discrete_state_prob shape mismatch: expected "
                f"({self.n_discrete_states},), "
                f"got {self.init_discrete_state_prob.shape}."
            )
        if self.discrete_transition_matrix.shape != (
            self.n_discrete_states,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"discrete_transition_matrix shape mismatch: expected "
                f"({self.n_discrete_states}, {self.n_discrete_states}), "
                f"got {self.discrete_transition_matrix.shape}."
            )
        if self.continuous_transition_matrix.shape != (
            self.n_latent,
            self.n_latent,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"continuous_transition_matrix shape mismatch: expected "
                f"({self.n_latent}, {self.n_latent}, {self.n_discrete_states}), "
                f"got {self.continuous_transition_matrix.shape}."
            )
        if self.process_cov.shape != (
            self.n_latent,
            self.n_latent,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"process_cov shape mismatch: expected "
                f"({self.n_latent}, {self.n_latent}, {self.n_discrete_states}), "
                f"got {self.process_cov.shape}."
            )
        if self.spike_params.baseline.shape != (self.n_neurons,):
            raise ValueError(
                f"spike_params.baseline shape mismatch: expected "
                f"({self.n_neurons},), "
                f"got {self.spike_params.baseline.shape}."
            )
        if self.spike_params.weights.shape != (self.n_neurons, self.n_latent):
            raise ValueError(
                f"spike_params.weights shape mismatch: expected "
                f"({self.n_neurons}, {self.n_latent}), "
                f"got {self.spike_params.weights.shape}."
            )

    def _e_step(self, spikes: Array) -> Array:
        """E-step: Run filter and smoother, store posterior statistics.

        Performs the expectation step of the EM algorithm:
        1. Runs the switching point-process filter to compute filtered posteriors
        2. Runs the switching Kalman smoother (observation-model agnostic)
        3. Stores smoother outputs as model attributes for the M-step

        Parameters
        ----------
        spikes : Array, shape (n_time, n_neurons)
            Observed spike counts for all neurons at each timestep.

        Returns
        -------
        marginal_log_likelihood : Array, shape ()
            Marginal log-likelihood log p(y_{1:T}) computed during filtering.

        Notes
        -----
        After calling this method, the following attributes are set:
        - smoother_state_cond_mean: E[x_t | y_{1:T}, S_t=j]
        - smoother_state_cond_cov: Cov[x_t | y_{1:T}, S_t=j]
        - smoother_discrete_state_prob: P(S_t=j | y_{1:T})
        - smoother_joint_discrete_state_prob: P(S_t=j, S_{t+1}=k | y_{1:T})
        - smoother_pair_cond_cross_cov: Cov[x_{t+1}, x_t | y_{1:T}, S_t=j, S_{t+1}=k]
        - smoother_pair_cond_means: E[x_t | y_{1:T}, S_t=j, S_{t+1}=k]

        These are the sufficient statistics needed by the M-step for dynamics
        parameter updates via `switching_kalman_maximization_step`.
        """
        # Build log-intensity function from current spike parameters
        def log_intensity_func(state: Array) -> Array:
            """Compute log-intensity for all neurons given latent state."""
            return self.spike_params.baseline + self.spike_params.weights @ state

        # Run the switching point-process filter
        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            marginal_log_likelihood,
        ) = switching_point_process_filter(
            init_state_cond_mean=self.init_mean,
            init_state_cond_cov=self.init_cov,
            init_discrete_state_prob=self.init_discrete_state_prob,
            spikes=spikes,
            discrete_transition_matrix=self.discrete_transition_matrix,
            continuous_transition_matrix=self.continuous_transition_matrix,
            process_cov=self.process_cov,
            dt=self.dt,
            log_intensity_func=log_intensity_func,
        )

        # Run the switching Kalman smoother (observation-model agnostic)
        # The smoother operates on Gaussian posteriors regardless of observation model
        (
            _,  # overall_smoother_mean - marginalized over discrete states
            _,  # overall_smoother_cov - marginalized over discrete states
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            _,  # overall_smoother_cross_cov - marginalized over discrete states
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_covs,
            pair_cond_smoother_means,
        ) = switching_kalman_smoother(
            filter_mean=state_cond_filter_mean,
            filter_cov=state_cond_filter_cov,
            filter_discrete_state_prob=filter_discrete_state_prob,
            last_filter_conditional_cont_mean=last_pair_cond_filter_mean,
            process_cov=self.process_cov,
            continuous_transition_matrix=self.continuous_transition_matrix,
            discrete_state_transition_matrix=self.discrete_transition_matrix,
        )

        # Store smoother outputs as model attributes for M-step
        self.smoother_state_cond_mean = state_cond_smoother_means
        self.smoother_state_cond_cov = state_cond_smoother_covs
        self.smoother_discrete_state_prob = smoother_discrete_state_prob
        self.smoother_joint_discrete_state_prob = smoother_joint_discrete_state_prob
        self.smoother_pair_cond_cross_cov = pair_cond_smoother_cross_covs
        self.smoother_pair_cond_means = pair_cond_smoother_means

        return marginal_log_likelihood

    def _m_step_dynamics(self) -> None:
        """M-step for dynamics parameters: A, Q, discrete transitions, initial state.

        Updates the dynamics-related model parameters by calling
        `switching_kalman_maximization_step` with the stored smoother outputs
        from the E-step. Parameters are only updated if their corresponding
        update flag is True.

        This method updates:
        - continuous_transition_matrix (A): if update_continuous_transition_matrix=True
        - process_cov (Q): if update_process_cov=True
        - discrete_transition_matrix (Z): if update_discrete_transition_matrix=True
        - init_mean (m0): if update_init_mean=True
        - init_cov (P0): if update_init_cov=True
        - init_discrete_state_prob: always updated (from smoother posterior)

        Notes
        -----
        This method must be called after `_e_step()` which populates the smoother
        output attributes used here.

        The process covariance Q is regularized to ensure positive semi-definiteness
        by adding a small value (1e-8) times the identity matrix. This handles
        cases where low discrete state probability leads to numerically
        non-PSD estimates from the raw M-step.

        The measurement_matrix and measurement_cov returns from
        `switching_kalman_maximization_step` are ignored since they assume
        Gaussian observations, which don't apply to point-process models.
        """
        # Call the switching Kalman M-step for dynamics parameters
        # obs argument is required but not used for dynamics updates
        # We pass a dummy array since we don't use measurement_matrix/cov outputs
        n_time = self.smoother_state_cond_mean.shape[0]
        dummy_obs = jnp.zeros((n_time, 1))

        (
            new_A,
            _,  # measurement_matrix - ignored for point-process
            new_Q,
            _,  # measurement_cov - ignored for point-process
            new_init_mean,
            new_init_cov,
            new_discrete_transition,
            new_init_discrete_prob,
        ) = switching_kalman_maximization_step(
            obs=dummy_obs,
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_discrete_state_prob=self.smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
            pair_cond_smoother_means=self.smoother_pair_cond_means,
        )

        # Update continuous transition matrix if flag is True
        if self.update_continuous_transition_matrix:
            self.continuous_transition_matrix = new_A

        # Update process covariance if flag is True
        if self.update_process_cov:
            # Ensure PSD by adding small regularization
            # The raw M-step can produce non-PSD Q when a discrete state
            # has low probability or insufficient data
            eps = 1e-8
            Q_reg = new_Q + eps * jnp.eye(self.n_latent)[:, :, None]
            self.process_cov = Q_reg

        # Update discrete transition matrix if flag is True
        if self.update_discrete_transition_matrix:
            self.discrete_transition_matrix = new_discrete_transition

        # Update initial mean if flag is True
        if self.update_init_mean:
            self.init_mean = new_init_mean

        # Update initial covariance if flag is True
        if self.update_init_cov:
            self.init_cov = new_init_cov

        # Always update initial discrete state probabilities
        # (from smoother posterior at t=0)
        self.init_discrete_state_prob = new_init_discrete_prob

    def _m_step_spikes(self, spikes: Array) -> None:
        """M-step for spike observation parameters: baseline and weights.

        Updates the spike GLM parameters by calling `update_spike_glm_params`
        with the marginalized smoother mean (averaged over discrete states).
        Parameters are only updated if `update_spike_params` is True.

        Parameters
        ----------
        spikes : Array, shape (n_time, n_neurons)
            Observed spike counts for all neurons at each timestep.

        Notes
        -----
        This method must be called after `_e_step()` which populates the smoother
        output attributes used here.

        The spike GLM update uses the marginal smoother mean, which is computed
        by marginalizing the state-conditional smoother means over discrete states:

            E[x_t | y_{1:T}] = sum_j P(S_t=j | y_{1:T}) * E[x_t | y_{1:T}, S_t=j]

        This is the standard "plug-in" M-step for the spike parameters.
        """
        if not self.update_spike_params:
            return

        # Compute marginalized smoother mean by weighting state-conditional means
        # by discrete state probabilities
        # smoother_state_cond_mean: (n_time, n_latent, n_discrete_states)
        # smoother_discrete_state_prob: (n_time, n_discrete_states)
        # Result: (n_time, n_latent)
        smoother_mean = jnp.einsum(
            "tls,ts->tl",
            self.smoother_state_cond_mean,
            self.smoother_discrete_state_prob,
        )

        # Update spike GLM parameters
        self.spike_params = update_spike_glm_params(
            spikes=spikes,
            smoother_mean=smoother_mean,
            current_params=self.spike_params,
            dt=self.dt,
        )

    def fit(
        self,
        spikes: ArrayLike,
        max_iter: int = 50,
        tol: float = 1e-4,
        key: Array | None = None,
    ) -> list[float]:
        """Fit the model to spike data using the EM algorithm.

        Performs Expectation-Maximization (EM) to learn model parameters from
        observed spike data. The algorithm alternates between:
        - E-step: Compute posterior distributions over latent states
        - M-step: Update model parameters to maximize expected log-likelihood

        Parameters
        ----------
        spikes : ArrayLike, shape (n_time, n_neurons)
            Observed spike counts for all neurons at each timestep.
        max_iter : int, default=50
            Maximum number of EM iterations.
        tol : float, default=1e-4
            Convergence tolerance. The algorithm stops early if the relative
            change in log-likelihood is less than `tol`:
                |LL_new - LL_old| / avg < tol
            where avg = (|LL_new| + |LL_old|) / 2.
        key : Array | None, optional
            JAX random key for parameter initialization. If None, uses PRNGKey(0).

        Returns
        -------
        log_likelihoods : list[float]
            Marginal log-likelihood at each EM iteration. Length is the number
            of iterations performed (may be less than max_iter if converged).

        Raises
        ------
        ValueError
            If spikes has wrong shape (must be 2D with n_neurons columns).
            If log-likelihood becomes non-finite during iteration.

        Notes
        -----
        The EM algorithm for this model:

        **Initialization:**
        - Parameters are initialized via `_initialize_parameters()`
        - Spike weights are small random values
        - Transition matrices start as uncoupled oscillators

        **E-step:**
        - Run switching point-process filter (Laplace-EKF)
        - Run switching Kalman smoother (observation-model agnostic)
        - Store sufficient statistics for M-step

        **M-step:**
        - Update dynamics parameters (A, Q, Z, initial state) via
          `switching_kalman_maximization_step`
        - Update spike GLM parameters (baseline, weights) via
          `update_spike_glm_params`

        **Convergence:**
        - Monitor marginal log-likelihood from the E-step
        - EM guarantees monotonic increase (up to Laplace approximation)
        - Stop when relative change < tol or max_iter reached

        **Numerical Stability:**
        - The Laplace approximation can cause small violations of strict
          monotonicity in individual iterations
        - Large decreases or NaN/Inf indicate numerical issues

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax
        >>> model = SwitchingSpikeOscillatorModel(
        ...     n_oscillators=2,
        ...     n_neurons=10,
        ...     n_discrete_states=2,
        ...     sampling_freq=100.0,
        ...     dt=0.01,
        ... )
        >>> spikes = jax.random.poisson(jax.random.PRNGKey(0), 0.5, (100, 10))
        >>> log_likelihoods = model.fit(spikes, max_iter=20, key=jax.random.PRNGKey(42))
        >>> len(log_likelihoods)  # Number of iterations
        20
        """
        # Convert to JAX array
        spikes = jnp.asarray(spikes)

        # Validate input shape
        if spikes.ndim != 2:
            raise ValueError(
                f"spikes must be 2D array with shape (n_time, n_neurons), "
                f"got {spikes.ndim}D array with shape {spikes.shape}"
            )
        if spikes.shape[1] != self.n_neurons:
            raise ValueError(
                f"spikes shape[1] must match n_neurons={self.n_neurons}, "
                f"got shape {spikes.shape}"
            )

        # Set default random key if not provided
        if key is None:
            key = jax.random.PRNGKey(0)

        # Initialize parameters
        self._initialize_parameters(key)

        # Track log-likelihoods across iterations
        log_likelihoods: list[float] = []

        for iteration in range(max_iter):
            # E-step: compute posteriors
            marginal_ll = self._e_step(spikes)
            log_likelihoods.append(float(marginal_ll))

            # Check for numerical issues
            if not jnp.isfinite(marginal_ll):
                raise ValueError(
                    f"Non-finite log-likelihood at iteration {iteration}: "
                    f"{marginal_ll}. This may indicate numerical instability."
                )

            # Check convergence (after at least 2 iterations)
            if iteration > 0:
                is_converged, _ = check_converged(
                    log_likelihood=log_likelihoods[-1],
                    previous_log_likelihood=log_likelihoods[-2],
                    tolerance=tol,
                )

                if is_converged:
                    break

            # M-step: update parameters
            self._m_step_dynamics()
            self._m_step_spikes(spikes)

        return log_likelihoods

"""Switching point-process Kalman filter for spike-based observations.

This module implements a Switching Linear Dynamical System (SLDS) with
point-process (spike) observations using the Laplace-EKF approach from
Eden & Brown (2004).

The model combines:
- Oscillator network dynamics with switching discrete states (DIM/CNM style)
- Point-process observations (spikes) using Laplace approximation

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

from state_space_practice.kalman import symmetrize
from state_space_practice.switching_kalman import (
    _scale_likelihood,
    _update_discrete_state_probabilities,
    collapse_gaussian_mixture_per_discrete_state,
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
    # Compute gradients and Hessians of log-intensity function
    # grad: d(log_lambda)/d(state) -> (n_neurons, n_latent) Jacobian
    # hess: d^2(log_lambda)/d(state)^2 -> (n_neurons, n_latent, n_latent) Hessian
    grad_log_intensity = jax.jacfwd(log_intensity_func)
    hess_log_intensity = jax.jacfwd(grad_log_intensity)

    # Evaluate at the one-step predicted mean
    log_lambda = log_intensity_func(one_step_mean)  # (n_neurons,)
    conditional_intensity = jnp.exp(log_lambda) * dt  # Expected spike count

    # Innovation: observed - expected
    innovation = y_t - conditional_intensity  # (n_neurons,)

    # Jacobian: (n_neurons, n_latent)
    jacobian = grad_log_intensity(one_step_mean)  # (n_neurons, n_latent)

    # Hessian: (n_neurons, n_latent, n_latent)
    hessian = hess_log_intensity(one_step_mean)  # (n_neurons, n_latent, n_latent)

    # For Poisson likelihood with log-link:
    # log p(y | x) = sum_n [y_n * log(lambda_n * dt) - lambda_n * dt - log(y_n!)]
    # d/dx log p = sum_n [(y_n - lambda_n * dt) * d(log_lambda_n)/dx]
    # d^2/dx^2 log p = sum_n [(y_n - lambda_n * dt) * d^2(log_lambda_n)/dx^2
    #                         - lambda_n * dt * (d(log_lambda_n)/dx)^T @ (d(log_lambda_n)/dx)]

    # Information matrix contribution from each neuron:
    # Fisher information: lambda_n * dt * grad^T @ grad
    # (using expected Fisher information, not observed)
    # Sum over neurons: sum_n lambda_n * dt * jacobian[n]^T @ jacobian[n]

    # Weighted gradient for mean update
    # sum_n innovation_n * jacobian[n] = jacobian.T @ innovation
    gradient = jacobian.T @ innovation  # (n_latent,)

    # Information from likelihood (Fisher information approximation)
    # For each neuron: lambda_n * dt * outer(grad_n, grad_n)
    # Sum: jacobian.T @ diag(conditional_intensity) @ jacobian
    fisher_info = jacobian.T @ (
        conditional_intensity[:, None] * jacobian
    )  # (n_latent, n_latent)

    # Hessian correction from innovation (second-order term)
    # sum_n innovation_n * hessian[n]
    hessian_correction = jnp.einsum(
        "n,nij->ij", innovation, hessian
    )  # (n_latent, n_latent)

    # Inverse posterior covariance (precision)
    # P^{-1} = P_prior^{-1} + Fisher - innovation * Hessian
    prior_precision = jnp.linalg.pinv(one_step_cov)
    posterior_precision = prior_precision + fisher_info - hessian_correction

    # Posterior covariance
    posterior_cov = jnp.linalg.pinv(posterior_precision)
    posterior_cov = symmetrize(posterior_cov)

    # Posterior mean: Newton step from prior
    posterior_mean = one_step_mean + posterior_cov @ gradient

    # Log-likelihood: sum of Poisson log-pmfs at predicted intensity
    log_likelihood = jnp.sum(jax.scipy.stats.poisson.logpmf(y_t, conditional_intensity))

    return posterior_mean, posterior_cov, log_likelihood


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


def _single_neuron_glm_loss(
    baseline: float,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    dt: float,
) -> float:
    """Poisson negative log-likelihood for a single neuron GLM.

    Computes the negative log-likelihood for a Poisson GLM observation model:
        y_{n,t} ~ Poisson(exp(baseline + weights @ x_t) * dt)

    This is the plug-in method that uses smoother_mean as a point estimate
    for the latent state x_t.

    Parameters
    ----------
    baseline : float
        Baseline log-rate for the neuron.
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
    loss : float
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
    baseline: float,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    dt: float,
) -> tuple[float, Array]:
    """Single Newton-Raphson step for Poisson GLM parameter optimization.

    Takes one Newton step to minimize the negative log-likelihood
    of a Poisson GLM observation model. Uses backtracking line search
    to ensure descent. The update is:
        params_new = params - alpha * H^{-1} @ g

    where g is the gradient, H is the Hessian, and alpha is determined
    by backtracking line search.

    Parameters
    ----------
    baseline : float
        Current baseline log-rate for the neuron.
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
    new_baseline : float
        Updated baseline after Newton step.
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

    # Backtracking line search with Armijo condition
    # Parameters: alpha starts at 1, beta for reduction, c for sufficient decrease
    beta = 0.5  # Step reduction factor
    c = 1e-4  # Armijo constant

    # Expected decrease from gradient (for Armijo condition)
    # directional_derivative = gradient.T @ delta (should be positive for descent)
    directional_derivative = jnp.dot(gradient, delta)

    def line_search_step(carry, _):
        alpha, _ = carry
        new_params_trial = params - alpha * delta
        eta_trial = design_matrix @ new_params_trial
        mu_trial = jnp.exp(eta_trial) * dt
        new_loss = jnp.sum(mu_trial - y_n * eta_trial)

        # Armijo condition: new_loss <= current_loss - c * alpha * directional_derivative
        sufficient_decrease = new_loss <= current_loss - c * alpha * directional_derivative

        # Reduce alpha if not sufficient decrease
        new_alpha = jnp.where(sufficient_decrease, alpha, alpha * beta)
        return (new_alpha, sufficient_decrease), None

    # Run up to 20 backtracking iterations
    (final_alpha, _), _ = jax.lax.scan(
        line_search_step, (jnp.array(1.0), jnp.array(False)), None, length=20
    )

    # Apply final step
    new_params = params - final_alpha * delta

    # Extract updated baseline and weights
    new_baseline = new_params[0]
    new_weights = new_params[1:]

    return new_baseline, new_weights


def _single_neuron_glm_step_second_order(
    baseline: float,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    smoother_cov: Array,
    dt: float,
) -> tuple[float, Array]:
    """Single Newton step for Poisson GLM with second-order expectation.

    Uses the second-order approximation that accounts for state uncertainty:
        E[exp(c @ x)] = exp(c @ m + 0.5 * c @ P @ c)

    This gives more accurate parameter estimates when the smoother
    covariance is significant.

    Parameters
    ----------
    baseline : float
        Current baseline log-rate for the neuron.
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
    new_baseline : float
        Updated baseline after Newton step.
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

    # Backtracking line search
    beta = 0.5
    c_armijo = 1e-4
    directional_derivative = jnp.dot(gradient, delta)

    def line_search_step(carry, _):
        alpha, _ = carry
        new_params_trial = params - alpha * delta
        new_weights_trial = new_params_trial[1:]

        # Recompute variance corrections with new weights
        def compute_var_corr_trial(P_t):
            return 0.5 * new_weights_trial @ P_t @ new_weights_trial

        var_corr_trial = jax.vmap(compute_var_corr_trial)(smoother_cov)

        eta_trial = design_matrix @ new_params_trial + var_corr_trial
        mu_trial = jnp.exp(eta_trial) * dt
        new_loss = jnp.sum(mu_trial - y_n * (design_matrix @ new_params_trial + var_corr_trial))

        sufficient_decrease = new_loss <= current_loss - c_armijo * alpha * directional_derivative
        new_alpha = jnp.where(sufficient_decrease, alpha, alpha * beta)
        return (new_alpha, sufficient_decrease), None

    (final_alpha, _), _ = jax.lax.scan(
        line_search_step, (jnp.array(1.0), jnp.array(False)), None, length=20
    )

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

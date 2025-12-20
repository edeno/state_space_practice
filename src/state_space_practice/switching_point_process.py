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
from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
    project_coupled_transition_matrix,
)
from state_space_practice.point_process_kalman import _point_process_laplace_update
from state_space_practice.switching_kalman import (
    _divide_safe,
    _scale_likelihood,
    _update_discrete_state_probabilities,
    collapse_gaussian_mixture_per_discrete_state,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
)
from state_space_practice.utils import check_converged


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
    baseline : Array, shape (n_neurons,) or (n_neurons, n_discrete_states)
        Baseline log-rate b_n for each neuron. When the latent state is zero,
        the firing rate is exp(baseline). If provided per discrete state, the
        discrete state axis must be last.
    weights : Array, shape (n_neurons, n_latent) or (n_neurons, n_latent, n_discrete_states)
        Linear weights C mapping the oscillator state to log-rates.
        weights[n, :] gives the coupling of neuron n to each latent dimension.
        If provided per discrete state, the discrete state axis must be last.
    """

    baseline: Array
    weights: Array

    def tree_flatten(self):
        """Flatten for JAX pytree registration."""
        return (self.baseline, self.weights), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten for JAX pytree registration."""
        baseline, weights = children
        return cls(baseline=baseline, weights=weights)


# Register SpikeObsParams as a JAX pytree so it can be traced through jit
jax.tree_util.register_pytree_node(
    SpikeObsParams,
    SpikeObsParams.tree_flatten,
    SpikeObsParams.tree_unflatten,
)


def _is_per_state_spike_params(spike_params: SpikeObsParams) -> bool:
    """Check if spike params have per-state parameters.

    Parameters
    ----------
    spike_params : SpikeObsParams
        The spike observation parameters to check.

    Returns
    -------
    bool
        True if parameters are per-state (baseline 2D, weights 3D),
        False if parameters are shared (baseline 1D, weights 2D).

    Raises
    ------
    ValueError
        If parameter dimensions are inconsistent.
    """
    if spike_params.baseline.ndim == 1:
        if spike_params.weights.ndim != 2:
            raise ValueError(
                "spike_params.weights must have shape (n_neurons, n_latent) when "
                "spike_params.baseline is 1D (shared parameters)."
            )
        return False
    if spike_params.baseline.ndim == 2:
        if spike_params.weights.ndim != 3:
            raise ValueError(
                "spike_params.weights must have shape (n_neurons, n_latent, n_states) "
                "when spike_params.baseline is 2D (per-state parameters)."
            )
        return True
    raise ValueError(
        f"spike_params.baseline must be 1D (shared) or 2D (per-state), "
        f"got ndim={spike_params.baseline.ndim}"
    )


def _select_spike_params(spike_params: SpikeObsParams, state_index: Array) -> SpikeObsParams:
    """Select per-state spike params if a discrete-state axis is present.

    For shared parameters (baseline 1D, weights 2D), returns the params unchanged.
    For per-state parameters (baseline 2D, weights 3D), selects the slice for
    the given discrete state index.

    Parameters
    ----------
    spike_params : SpikeObsParams
        Spike observation parameters. Either shared across states:
        - baseline: (n_neurons,)
        - weights: (n_neurons, n_latent)
        Or per-state:
        - baseline: (n_neurons, n_discrete_states)
        - weights: (n_neurons, n_latent, n_discrete_states)
    state_index : Array
        Index of the discrete state to select (scalar). Only used when
        spike_params are per-state.

    Returns
    -------
    SpikeObsParams
        Parameters for the selected state, with shapes:
        - baseline: (n_neurons,)
        - weights: (n_neurons, n_latent)
    """
    if not _is_per_state_spike_params(spike_params):
        # Shared parameters - return unchanged (state_index is ignored)
        return spike_params

    # Per-state parameters - select the slice for this discrete state
    baseline = jnp.take(spike_params.baseline, state_index, axis=-1)
    weights = jnp.take(spike_params.weights, state_index, axis=-1)
    return SpikeObsParams(baseline=baseline, weights=weights)


@dataclass
class QRegularizationConfig:
    """Configuration for trust-region regularization of process covariances.

    Parameters
    ----------
    trust_region_weight : float
        Blend factor for new Q estimate: Q = w * new_Q + (1 - w) * old_Q.
    min_eigenvalue : float | None
        Lower bound for eigenvalues. None disables lower clipping.
    max_eigenvalue : float | None
        Upper bound for eigenvalues. None disables upper clipping.
    enabled : bool
        Whether to apply trust-region regularization and eigenvalue clipping.
    """

    trust_region_weight: float = 0.3  # More conservative blending
    min_eigenvalue: float | None = 0.01  # Process noise floor to prevent collapse
    max_eigenvalue: float | None = 1.0  # Prevent explosion
    enabled: bool = True


def point_process_kalman_update(
    one_step_mean: Array,
    one_step_cov: Array,
    y_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array, SpikeObsParams], Array],
    spike_params: SpikeObsParams,
    diagonal_boost: float = 1e-9,
    grad_log_intensity_func: Callable[[Array, SpikeObsParams], Array] | None = None,
    hess_log_intensity_func: Callable[[Array, SpikeObsParams], Array] | None = None,
    include_laplace_normalization: bool = True,
    max_newton_iter: int = 1,
    line_search_beta: float = 0.5,
) -> tuple[Array, Array, Array]:
    """Single point-process Laplace-EKF update for multiple neurons.

    Performs a Bayesian update of the latent state posterior given observed
    spike counts, using a Laplace (Gaussian) approximation to the posterior.

    The observation model is:
        y_n ~ Poisson(exp(log_intensity_func(x, spike_params)[n]) * dt)

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
    log_intensity_func : Callable[[Array, SpikeObsParams], Array]
        Function mapping (state, params) to log-intensities (n_neurons,).
        Should return log(lambda) where lambda is firing rate in Hz.
        The function takes state (n_latent,) and spike_params as arguments.
    spike_params : SpikeObsParams
        Spike observation parameters (baseline, weights). Passed as data to
        log_intensity_func, allowing JIT compilation without closure issues.
    diagonal_boost : float, default=1e-9
        Small value added to precision matrix diagonal for numerical stability
        when solving linear systems.
    grad_log_intensity_func : Callable[[Array, SpikeObsParams], Array] | None, optional
        Pre-computed gradient function (Jacobian) of log_intensity_func w.r.t. state.
        If None, computed via jax.jacfwd(log_intensity_func, argnums=0).
        Passing pre-computed functions can improve compilation speed when
        this function is called repeatedly inside a JIT-compiled context.
    hess_log_intensity_func : Callable[[Array, SpikeObsParams], Array] | None, optional
        Pre-computed Hessian function of log_intensity_func w.r.t. state.
        If None, computed via jax.jacfwd of the gradient function.
    include_laplace_normalization : bool, default=True
        If True, include the Laplace normalization and prior terms to approximate
        log p(y_t | y_{1:t-1}). If False, return the plug-in log-likelihood
        at the posterior mode without normalization.

    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
        Updated state mean after incorporating spike observations
    posterior_cov : Array, shape (n_latent, n_latent)
        Updated state covariance after incorporating spike observations
    log_likelihood : Array
        Log p(y_t | y_{1:t-1}) approximated at posterior mode (scalar array)

    Notes
    -----
    The Laplace approximation uses the predicted mean as the expansion point
    for a single Newton step update. For multiple neurons, the gradients and
    Hessians are summed across neurons.

    The log-likelihood is evaluated at the approximate posterior mode. When
    ``include_laplace_normalization`` is True, prior and normalization terms
    are added to approximate the marginal likelihood.

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """
    if spike_params.baseline.ndim != 1 or spike_params.weights.ndim != 2:
        raise ValueError(
            "point_process_kalman_update expects single-state spike_params with "
            "baseline shape (n_neurons,) and weights shape (n_neurons, n_latent)."
        )
    # Create a closure-free wrapper for the shared helper
    # This binds spike_params so _point_process_laplace_update sees Callable[[Array], Array]
    def _log_intensity_with_params(state: Array) -> Array:
        return log_intensity_func(state, spike_params)

    # Wrap grad/hess functions if provided
    _grad_func = None
    if grad_log_intensity_func is not None:
        def _grad_func(state: Array) -> Array:
            return grad_log_intensity_func(state, spike_params)

    _hess_func = None
    if hess_log_intensity_func is not None:
        def _hess_func(state: Array) -> Array:
            return hess_log_intensity_func(state, spike_params)

    # Delegate to the shared helper in point_process_kalman.py
    return _point_process_laplace_update(
        one_step_mean=one_step_mean,
        one_step_cov=one_step_cov,
        spike_indicator_t=y_t,
        dt=dt,
        log_intensity_func=_log_intensity_with_params,
        diagonal_boost=diagonal_boost,
        grad_log_intensity_func=_grad_func,
        hess_log_intensity_func=_hess_func,
        include_laplace_normalization=include_laplace_normalization,
        max_newton_iter=max_newton_iter,
        line_search_beta=line_search_beta,
    )


def _point_process_predict_and_update(
    prev_state_cond_mean: Array,
    prev_state_cond_cov: Array,
    y_t: Array,
    continuous_transition_matrix: Array,
    process_cov: Array,
    dt: float,
    log_intensity_func: Callable[[Array, SpikeObsParams], Array],
    spike_params: SpikeObsParams,
    include_laplace_normalization: bool = True,
    max_newton_iter: int = 1,
    line_search_beta: float = 0.5,
) -> tuple[Array, Array, Array]:
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
    log_intensity_func : Callable[[Array, SpikeObsParams], Array]
        Function mapping (state, params) to log-intensities (n_neurons,).
    spike_params : SpikeObsParams
        Spike observation parameters (baseline, weights).

    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
        Pair-conditional posterior mean
    posterior_cov : Array, shape (n_latent, n_latent)
        Pair-conditional posterior covariance
    log_likelihood : Array
        Log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j) (scalar array)
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
        one_step_mean,
        one_step_cov,
        y_t,
        dt,
        log_intensity_func,
        spike_params,
        include_laplace_normalization=include_laplace_normalization,
        max_newton_iter=max_newton_iter,
        line_search_beta=line_search_beta,
    )


def _point_process_update_per_discrete_state_pair(
    prev_state_cond_mean: Array,
    prev_state_cond_cov: Array,
    y_t: Array,
    continuous_transition_matrix: Array,
    process_cov: Array,
    dt: float,
    log_intensity_func: Callable[[Array, SpikeObsParams], Array],
    spike_params: SpikeObsParams,
    include_laplace_normalization: bool = True,
    max_newton_iter: int = 1,
    line_search_beta: float = 0.5,
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
    log_intensity_func : Callable[[Array, SpikeObsParams], Array]
        Function mapping (state, params) to log-intensities (n_neurons,).
    spike_params : SpikeObsParams
        Spike observation parameters (baseline, weights).

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

    n_discrete_states = continuous_transition_matrix.shape[-1]
    state_indices = jnp.arange(n_discrete_states)

    # Create a version of predict_and_update that only takes array arguments
    # (log_intensity_func, spike_params, dt are captured in the closure for vmap)
    def _update(prev_mean, prev_cov, A, Q, params_j):
        return _point_process_predict_and_update(
            prev_mean,
            prev_cov,
            y_t,
            A,
            Q,
            dt,
            log_intensity_func,
            params_j,
            include_laplace_normalization,
            max_newton_iter,
            line_search_beta,
        )

    def _update_for_state_j(
        state_index: Array, A: Array, Q: Array
    ) -> tuple[Array, Array, Array]:
        """Update for a single next-state j, vmapped over previous states i."""
        params_j = _select_spike_params(spike_params, state_index)
        return jax.vmap(
            _update,
            in_axes=(-1, -1, None, None, None),  # vmap over prev state i
            out_axes=-1,
        )(prev_state_cond_mean, prev_state_cond_cov, A, Q, params_j)

    # Outer vmap: over next state j (axis -1 of A/Q)
    vmapped_update = jax.vmap(
        _update_for_state_j,
        in_axes=(0, -1, -1),
        out_axes=-1,
    )

    result: tuple[Array, Array, Array] = vmapped_update(
        state_indices, continuous_transition_matrix, process_cov
    )
    return result


def _first_timestep_point_process_update(
    init_state_cond_mean: Array,
    init_state_cond_cov: Array,
    init_discrete_state_prob: Array,
    y_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array, SpikeObsParams], Array],
    spike_params: SpikeObsParams,
    include_laplace_normalization: bool = True,
    max_newton_iter: int = 1,
    line_search_beta: float = 0.5,
) -> tuple[Array, Array, Array, Array, Array]:
    """Handle first timestep with x₁ convention (update only, no prediction).

    For the first observation y₁, we treat init_state_cond_mean/cov as p(x₁ | S₁)
    and apply only the observation update (no dynamics prediction). This aligns
    with the EM M-step which sets init_state from smoother_means[0] = x₁|T.

    Parameters
    ----------
    init_state_cond_mean : Array, shape (n_latent, n_discrete_states)
        Prior belief about x₁ given S₁, p(x₁ | S₁)
    init_state_cond_cov : Array, shape (n_latent, n_latent, n_discrete_states)
        Prior covariance of x₁ given S₁
    init_discrete_state_prob : Array, shape (n_discrete_states,)
        Prior probability p(S₁)
    y_t : Array, shape (n_neurons,)
        Spike counts at first timestep
    dt : float
        Time bin width in seconds
    log_intensity_func : Callable[[Array, SpikeObsParams], Array]
        Function mapping (state, params) to log-intensities (n_neurons,).
    spike_params : SpikeObsParams
        Spike observation parameters (baseline, weights).
    include_laplace_normalization : bool, default=True
        If True, include Laplace normalization terms in log-likelihood

    Returns
    -------
    state_cond_filter_mean : Array, shape (n_latent, n_discrete_states)
        Posterior state-conditional means p(x₁ | y₁, S₁=j)
    state_cond_filter_cov : Array, shape (n_latent, n_latent, n_discrete_states)
        Posterior state-conditional covariances
    filter_discrete_prob : Array, shape (n_discrete_states,)
        Posterior discrete state probabilities p(S₁ | y₁)
    pair_cond_filter_mean : Array, shape (n_latent, n_discrete_states, n_discrete_states)
        For compatibility with smoother, diagonal in first two dims
        (no pair-conditioning at t=1 since there's no S₀)
    marginal_log_likelihood : Array
        Log p(y₁) contribution (scalar array)
    """
    n_discrete_states = init_state_cond_mean.shape[-1]

    # Apply point-process update directly to the prior (no dynamics prediction)
    # vmap over discrete states j with per-state spike params if provided
    state_indices = jnp.arange(n_discrete_states)

    def _update_for_state_j(
        prior_mean: Array, prior_cov: Array, state_index: Array
    ) -> tuple[Array, Array, Array]:
        """Apply observation update for a single discrete state j."""
        params_j = _select_spike_params(spike_params, state_index)
        return point_process_kalman_update(
            one_step_mean=prior_mean,  # Use prior directly, no A @ x prediction
            one_step_cov=prior_cov,
            y_t=y_t,
            dt=dt,
            log_intensity_func=log_intensity_func,
            spike_params=params_j,
            include_laplace_normalization=include_laplace_normalization,
            max_newton_iter=max_newton_iter,
            line_search_beta=line_search_beta,
        )

    # vmap over discrete states
    vmapped_update = jax.vmap(
        _update_for_state_j, in_axes=(-1, -1, 0), out_axes=(-1, -1, -1)
    )
    state_cond_filter_mean, state_cond_filter_cov, state_cond_log_lik = vmapped_update(
        init_state_cond_mean, init_state_cond_cov, state_indices
    )

    # Update discrete state probabilities using observation likelihood
    # At t=1, there's no transition from S₀, so we just use:
    # p(S₁=j | y₁) ∝ p(y₁ | S₁=j) * p(S₁=j)
    scaled_lik, ll_max = _scale_likelihood(state_cond_log_lik)
    unnorm_prob = scaled_lik * init_discrete_state_prob
    norm_const = jnp.sum(unnorm_prob)
    filter_discrete_prob = _divide_safe(unnorm_prob, norm_const)

    # Marginal log-likelihood contribution
    marginal_log_likelihood = ll_max + jnp.log(norm_const)

    # For smoother compatibility: create pair_cond_filter_mean
    # At t=1, there's no S₀, so we create a diagonal structure where
    # pair_cond_mean[:, i, j] = state_cond_mean[:, j] for all i
    # This ensures the smoother sees the right structure
    pair_cond_filter_mean = jnp.broadcast_to(
        state_cond_filter_mean[:, None, :],
        (state_cond_filter_mean.shape[0], n_discrete_states, n_discrete_states),
    )

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_prob,
        pair_cond_filter_mean,
        marginal_log_likelihood,
    )


def _armijo_line_search(
    params: Array,
    delta: Array,
    gradient: Array,
    loss_fn: Callable[[Array], Array],
    current_loss: Array,
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
    loss_fn : Callable[[Array], Array]
        Loss function to minimize. Takes params and returns scalar array loss.
    current_loss : Array
        Loss at current params (avoids recomputation, scalar array).
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
        sufficient_decrease = (
            new_loss <= current_loss - c * alpha * directional_derivative
        )

        # Reduce alpha if not sufficient decrease
        new_alpha = jnp.where(sufficient_decrease, alpha, alpha * beta)
        return (new_alpha, sufficient_decrease), None

    (final_alpha, _), _ = jax.lax.scan(
        line_search_step, (jnp.array(1.0), jnp.array(False)), None, length=max_iter
    )

    return final_alpha


def _single_neuron_glm_loss(
    baseline: ArrayLike,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    dt: float,
    time_weights: Array | None = None,
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
    time_weights : Array | None, shape (n_time,), optional
        Per-timestep weights (e.g., state responsibilities). If None, uses ones.

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

    if time_weights is None:
        time_weights = jnp.ones_like(expected_counts)

    # Poisson log-likelihood (without log(y!) term):
    # sum_t [y_t * eta_t - exp(eta_t) * dt]
    # Note: We include the dt scaling in the expected counts
    log_likelihood = jnp.sum(time_weights * (y_n * eta - expected_counts))

    # Return negative log-likelihood for minimization
    return -log_likelihood


def _single_neuron_glm_step(
    baseline: ArrayLike,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    dt: float,
    time_weights: Array | None = None,
    weight_l2: float = 0.0,
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
    time_weights : Array | None, shape (n_time,), optional
        Per-timestep weights (e.g., state responsibilities). If None, uses ones.
    weight_l2 : float, default=0.0
        L2 regularization strength on the weights (not baseline).
        Adds 0.5 * weight_l2 * ||weights||^2 to the objective.

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

        gradient = -X.T @ (y - mu) + l2_reg * [0, weights]
        Hessian = X.T @ diag(mu) @ X + l2_reg * diag([0, 1, 1, ...])

    where X is the design matrix [1, smoother_mean], y is spike counts,
    and mu = exp(X @ params) * dt is the expected count.

    We add a small regularization to the Hessian diagonal for numerical stability.
    Backtracking line search ensures the step decreases the objective.
    """
    n_latent = weights.shape[0]

    # Build design matrix: [1, smoother_mean] of shape (n_time, 1 + n_latent)
    n_time = smoother_mean.shape[0]
    ones = jnp.ones((n_time, 1))
    design_matrix = jnp.concatenate(
        [ones, smoother_mean], axis=1
    )  # (n_time, 1+n_latent)

    # Concatenate parameters into a single vector: [baseline, weights]
    params = jnp.concatenate([jnp.atleast_1d(baseline), weights])  # (1 + n_latent,)

    # Compute linear predictor and expected counts
    eta = design_matrix @ params  # (n_time,)
    mu = jnp.exp(eta) * dt  # Expected spike counts (n_time,)

    if time_weights is None:
        time_weights = jnp.ones_like(mu)

    # Compute gradient: -X.T @ (w * (y - mu)) + L2 penalty gradient
    residual = time_weights * (y_n - mu)  # (n_time,)
    gradient = -design_matrix.T @ residual  # (1 + n_latent,)

    # Add L2 penalty gradient: weight_l2 * [0, weights]
    # (no penalty on baseline, only on weights)
    l2_grad = jnp.concatenate([jnp.zeros(1), weight_l2 * weights])
    gradient = gradient + l2_grad

    # Compute Hessian: X.T @ diag(w * mu) @ X
    weighted_mu = time_weights * mu
    weighted_design = design_matrix * weighted_mu[:, None]  # (n_time, 1+n_latent)
    hessian = design_matrix.T @ weighted_design  # (1+n_latent, 1+n_latent)

    # Add L2 penalty to Hessian: weight_l2 * diag([0, 1, 1, ...])
    l2_hess_diag = jnp.concatenate([jnp.zeros(1), jnp.ones(n_latent) * weight_l2])
    hessian = hessian + jnp.diag(l2_hess_diag)

    # Add small regularization for numerical stability
    reg = 1e-6 * jnp.eye(1 + n_latent)
    hessian = hessian + reg

    # Newton direction: delta = H^{-1} @ g
    delta = jnp.linalg.solve(hessian, gradient)

    # Current loss for backtracking line search (NLL + L2 penalty)
    current_loss = jnp.sum(time_weights * (mu - y_n * eta)) + 0.5 * weight_l2 * jnp.sum(weights**2)

    # Define loss function for line search
    def loss_fn(p):
        eta_trial = design_matrix @ p
        mu_trial = jnp.exp(eta_trial) * dt
        weights_trial = p[1:]
        nll = jnp.sum(time_weights * (mu_trial - y_n * eta_trial))
        l2_penalty = 0.5 * weight_l2 * jnp.sum(weights_trial**2)
        return nll + l2_penalty

    # Backtracking line search with Armijo condition
    final_alpha = _armijo_line_search(params, delta, gradient, loss_fn, current_loss)

    # Apply final step
    new_params = params - final_alpha * delta

    # Extract updated baseline and weights
    new_baseline = new_params[0]
    new_weights = new_params[1:]

    return new_baseline, new_weights


def _neg_Q_single_neuron(
    params: Array,
    y_n: Array,
    smoother_mean: Array,
    smoother_cov: Array | None,
    dt: float,
    weight_l2: float,
    time_weights: Array | None = None,
) -> Array:
    """Negative expected log-likelihood for one neuron's Poisson GLM.

    This implements the exact latent-marginalized Q-function for the EM M-step:

        Q(b, w) = sum_t [y_t * (b + w^T m_t) - dt * E[exp(b + w^T x_t)]]
                  - 0.5 * lambda * ||w||^2

    Using the Gaussian moment generating function:
        E[exp(w^T x_t)] = exp(w^T m_t + 0.5 * w^T P_t w)

    Parameters
    ----------
    params : Array, shape (1 + n_latent,)
        Concatenated parameters [baseline, weights...].
    y_n : Array, shape (n_time,)
        Spike counts for one neuron.
    smoother_mean : Array, shape (n_time, n_latent)
        Smoothed latent means m_t.
    smoother_cov : Array or None, shape (n_time, n_latent, n_latent)
        Smoothed latent covariances P_t. If None, uses first-order
        approximation (ignores state uncertainty).
    dt : float
        Bin width in seconds.
    weight_l2 : float
        L2 penalty coefficient for the weights.
    time_weights : Array | None, shape (n_time,), optional
        Per-timestep weights (e.g., state responsibilities). If None, uses ones.

    Returns
    -------
    loss : Array, shape ()
        Negative expected log-likelihood plus L2 penalty.
    """
    b = params[0]
    w = params[1:]

    # Linear term: w^T m_t for each t, shape (n_time,)
    eta_lin = smoother_mean @ w

    if smoother_cov is None:
        # First-order approximation: ignore latent uncertainty
        quad = jnp.zeros(smoother_mean.shape[0])
    else:
        # Second-order: 0.5 * w^T P_t w for each t
        quad = 0.5 * jnp.einsum("tij,i,j->t", smoother_cov, w, w)

    # Full log-intensity expectation: b + w^T m_t + 0.5 * w^T P_t w
    eta = b + eta_lin + quad  # (n_time,)
    # Clip eta to prevent overflow: exp(20) ≈ 5e8 (reasonable max count per bin)
    eta_safe = jnp.clip(eta, -20.0, 20.0)
    mu = jnp.exp(eta_safe) * dt  # expected counts, (n_time,)

    if time_weights is None:
        time_weights = jnp.ones_like(mu)

    # Negative Q: sum(mu - y * (b + w^T m_t)) + 0.5 * lambda * ||w||^2
    # Note: the y_t term only uses (b + w^T m_t), not the variance correction
    nll = jnp.sum(time_weights * (mu - y_n * (b + eta_lin)))
    l2 = 0.5 * weight_l2 * jnp.sum(w**2)

    return nll + l2


def _single_neuron_glm_step_second_order(
    baseline: Array,
    weights: Array,
    y_n: Array,
    smoother_mean: Array,
    smoother_cov: Array,
    dt: float,
    time_weights: Array | None = None,
    weight_l2: float = 0.0,
) -> tuple[Array, Array]:
    """Single Newton step for Poisson GLM with exact second-order expectation.

    Uses JAX autodiff to compute exact gradient and Hessian of the
    latent-marginalized Q-function. This accounts for state uncertainty via:
        E[exp(w^T x)] = exp(w^T m + 0.5 * w^T P w)

    The Hessian includes all terms, specifically:
        H_ww = -sum_t mu_t * [(m_t + P_t w)(m_t + P_t w)^T + P_t] - lambda * I

    which was previously approximated by dropping the sum_t mu_t P_t term.

    Parameters
    ----------
    baseline : Array, shape ()
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
    time_weights : Array | None, shape (n_time,), optional
        Per-timestep weights (e.g., state responsibilities). If None, uses ones.
    weight_l2 : float, default=0.0
        L2 regularization strength on the weights (not baseline).

    Returns
    -------
    new_baseline : Array, shape ()
        Updated baseline after Newton step.
    new_weights : Array, shape (n_latent,)
        Updated weights after Newton step.
    """
    n_latent = weights.shape[0]
    params = jnp.concatenate([jnp.atleast_1d(baseline), weights])

    def loss_fn(p: Array) -> Array:
        return _neg_Q_single_neuron(
            p, y_n, smoother_mean, smoother_cov, dt, weight_l2, time_weights
        )

    # Compute exact gradient and Hessian via JAX autodiff
    # We minimize the negative Q-function, so gradient points uphill on -Q (downhill on Q)
    grad = jax.grad(loss_fn)(params)
    hess = jax.hessian(loss_fn)(params)

    # Regularize Hessian for numerical stability
    ridge = 1e-6
    hess_reg = hess + ridge * jnp.eye(1 + n_latent)

    # Newton direction for minimization: x_new = x - H^{-1} @ grad(neg_Q)
    delta = jnp.linalg.solve(hess_reg, grad)

    # Backtracking line search with Armijo condition
    current_loss = loss_fn(params)
    final_alpha = _armijo_line_search(
        params, delta, grad, loss_fn, current_loss
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
    weight_l2: float = 0.0,
    time_weights: Array | None = None,
) -> SpikeObsParams:
    """M-step for spike observation parameters.

    Updates the spike GLM parameters (baseline, weights) by maximizing
    E_q[log p(y | x; C, b)] with respect to C and b, with optional L2
    regularization on the weights.

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
    weight_l2 : float, default=0.0
        L2 regularization strength on the weights (not baseline).
        Adds 0.5 * weight_l2 * ||weights||^2 to the objective for each neuron.
    time_weights : Array | None, shape (n_time,), optional
        Per-timestep weights (e.g., state responsibilities). If None, uses ones.

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

    The L2 regularization is applied only to the weights, not the baseline,
    which helps prevent overfitting when the number of neurons is small
    relative to the latent dimensionality.
    """
    n_neurons = spikes.shape[1]
    n_time = smoother_mean.shape[0]
    n_latent = smoother_mean.shape[1]

    if time_weights is None:
        time_weights = jnp.ones(n_time)
    else:
        time_weights = jnp.asarray(time_weights)
        if time_weights.shape != (n_time,):
            raise ValueError(
                f"time_weights shape {time_weights.shape} incompatible with "
                f"expected shape ({n_time},)"
            )
        if not jnp.all(jnp.isfinite(time_weights)):
            raise ValueError("time_weights contains NaN or Inf values")
        if jnp.any(time_weights < 0):
            raise ValueError("time_weights must be non-negative")

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
                    b, w, y_n, smoother_mean, smoother_cov, dt, time_weights, weight_l2
                )
                return new_b, new_w

            neuron_indices = jnp.arange(n_neurons)
            new_baselines, new_weights = jax.vmap(update_neuron)(neuron_indices)

            return (new_baselines, new_weights), None

        (final_baselines, final_weights), _ = jax.lax.scan(
            iterate_all_neurons_second_order,
            (baselines, weights),
            None,
            length=max_iter,
        )
    else:
        # Run Newton iterations for all neurons (plug-in)
        def iterate_all_neurons(carry, _):
            baselines, weights = carry

            def update_neuron(neuron_idx):
                b = baselines[neuron_idx]
                w = weights[neuron_idx]
                y_n = spikes[:, neuron_idx]
                new_b, new_w = _single_neuron_glm_step(
                    b, w, y_n, smoother_mean, dt, time_weights, weight_l2
                )
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
    log_intensity_func: Callable[[Array, SpikeObsParams], Array],
    spike_params: SpikeObsParams,
    include_laplace_normalization: bool = True,
    max_newton_iter: int = 1,
    line_search_beta: float = 0.5,
) -> tuple[Array, Array, Array, Array, Array]:
    """Switching point-process Kalman filter for spike observations.

    This filter implements a Switching Linear Dynamical System (SLDS) with
    point-process (spike) observations using the Laplace-EKF approach. It
    maintains exact per-state-pair structure: p(x_t | y_{1:t}, S_{t-1}=i, S_t=j)
    for all state pairs (i, j).

    The model is:
        x_t = A_{s_t} @ x_{t-1} + w_t,  w_t ~ N(0, Q_{s_t})
        y_{n,t} ~ Poisson(exp(log_intensity_func(x_t, spike_params)[n]) * dt)

    Parameters
    ----------
    init_state_cond_mean : Array, shape (n_latent, n_discrete_states)
        Prior belief about x₁ given S₁, p(x₁ | S₁ = j) for each discrete state.
        This is the prior on the latent state *at* the first observation, before
        incorporating y₁. See "Time Indexing Convention" in Notes for details.
    init_state_cond_cov : Array, shape (n_latent, n_latent, n_discrete_states)
        Prior covariances of x₁ given S₁ for each discrete state.
    init_discrete_state_prob : Array, shape (n_discrete_states,)
        Prior discrete state probabilities p(S₁ = j).
    spikes : Array, shape (n_time, n_neurons)
        Observed spike counts for all neurons at each timestep. Observations
        are indexed as y_1, y_2, ..., y_T (1-indexed in math notation).
    discrete_transition_matrix : Array, shape (n_discrete_states, n_discrete_states)
        Transition probabilities P(S_t = j | S_{t-1} = i). Entry [i, j] gives
        probability of transitioning from state i to state j.
    continuous_transition_matrix : Array, shape (n_latent, n_latent, n_discrete_states)
        State transition matrices A_j for each discrete state j.
    process_cov : Array, shape (n_latent, n_latent, n_discrete_states)
        Process noise covariances Q_j for each discrete state j.
    dt : float
        Time bin width in seconds.
    log_intensity_func : Callable[[Array, SpikeObsParams], Array]
        Function mapping (latent state, params) to log-intensities (n_neurons,).
        Should return log(lambda) where lambda is firing rate in Hz. Takes the
        state (n_latent,) and spike_params as arguments.
    spike_params : SpikeObsParams
        Spike observation parameters (baseline, weights). For per-state spike
        parameters, provide baseline with shape (n_neurons, n_discrete_states)
        and weights with shape (n_neurons, n_latent, n_discrete_states); the
        discrete state axis must be last. Passed to log_intensity_func as data,
        not captured in closures. This ensures EM parameter updates take effect.
    include_laplace_normalization : bool, default=True
        If True, include Laplace normalization and prior terms in the
        observation log-likelihood used for discrete-state updates.

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
    marginal_log_likelihood : Array
        Marginal log-likelihood log p(y_{1:T}) (scalar array).

    Notes
    -----
    The filter mirrors the structure of `switching_kalman_filter` but replaces
    the Gaussian observation update with a point-process Laplace-EKF update.

    At each timestep, the filter:
    1. Computes pair-conditional posteriors for all (i, j) state pairs
    2. Updates discrete state probabilities using the HMM forward algorithm
    3. Collapses pair-conditional to state-conditional via Gaussian mixture

    **Time Indexing Convention (x₁ convention)**:
    This filter uses the x₁ convention where init parameters represent the state
    at the first observation time, aligning with the EM M-step:

    - ``init_state_cond_mean`` represents p(x₁ | S₁), the prior for x₁
    - ``init_discrete_state_prob`` represents p(S₁), the prior for the first
      discrete state
    - The first observation ``spikes[0]`` corresponds to y₁ in math notation
    - For y₁: apply observation update only (no dynamics prediction)
    - For y_t (t > 1): predict x_t = A @ x_{t-1} + w_t, then update with y_t

    This convention ensures consistency with the EM M-step, which sets
    init_state_cond_mean = smoother_means[0] = x₁|T. The returned filter
    outputs at index t represent p(x_{t+1} | y_{1:t+1}) (0-indexed in Python,
    corresponding to math time index t+1).

    **Performance**: For production use, wrap this function with ``jax.jit``
    for significant speedups. The function uses ``jax.lax.scan`` and ``vmap``
    internally, which benefit greatly from JIT compilation::

        jitted_filter = jax.jit(switching_point_process_filter, static_argnums=(8,))
        results = jitted_filter(init_mean, init_cov, init_prob, spikes,
                                trans_mat, cont_trans, proc_cov, dt,
                                log_intensity, spike_params)

    Note that ``log_intensity_func`` (argument 8) must be marked as static
    since it's a Python callable. The ``spike_params`` argument (a registered
    JAX pytree) is passed as data, so parameter updates during EM iterations
    take effect without recompilation.

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    [2] Murphy, K.P. (1998). Switching Kalman Filters.
    [3] Shumway, R.H., and Stoffer, D.S. (1991). Dynamic Linear Models With Switching.
    """
    # Input validation: ensure spikes is 2D (n_time, n_neurons)
    spikes = jnp.asarray(spikes)
    if spikes.ndim == 1:
        # Promote 1D single-neuron input to 2D
        spikes = spikes[:, None]
    elif spikes.ndim != 2:
        raise ValueError(
            f"spikes must be 1D or 2D array, got shape {spikes.shape}"
        )

    def _step(
        carry: tuple[Array, Array, Array, Array],
        y_t: Array,
    ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]]:
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
            marginal_log_likelihood : Array
                Accumulated marginal log-likelihood (scalar array).
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
            marginal_log_likelihood : Array
                Updated accumulated marginal log-likelihood (scalar array).
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
            spike_params,
            include_laplace_normalization,
            max_newton_iter,
            line_search_beta,
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

    # Handle first timestep with x₁ convention: update-only (no dynamics prediction)
    # init_state_cond_mean represents p(x₁ | S₁), the prior for the first observation
    (
        first_state_cond_mean,
        first_state_cond_cov,
        first_discrete_prob,
        first_pair_cond_mean,
        first_log_lik,
    ) = _first_timestep_point_process_update(
        init_state_cond_mean,
        init_state_cond_cov,
        init_discrete_state_prob,
        spikes[0],
        dt,
        log_intensity_func,
        spike_params,
        include_laplace_normalization,
        max_newton_iter,
        line_search_beta,
    )

    # Run predict-then-update for t=2,...,T
    # jax.lax.scan handles empty inputs (spikes[1:] when n_time=1) gracefully
    (_, _, _, marginal_log_likelihood), (
        rest_state_cond_filter_mean,
        rest_state_cond_filter_cov,
        rest_filter_discrete_state_prob,
        rest_pair_cond_filter_mean,
    ) = jax.lax.scan(
        _step,
        (
            first_state_cond_mean,
            first_state_cond_cov,
            first_discrete_prob,
            first_log_lik,
        ),
        spikes[1:],
    )

    # Prepend first timestep results
    state_cond_filter_mean = jnp.concatenate(
        [first_state_cond_mean[None, ...], rest_state_cond_filter_mean], axis=0
    )
    state_cond_filter_cov = jnp.concatenate(
        [first_state_cond_cov[None, ...], rest_state_cond_filter_cov], axis=0
    )
    filter_discrete_state_prob = jnp.concatenate(
        [first_discrete_prob[None, ...], rest_filter_discrete_state_prob], axis=0
    )
    pair_cond_filter_mean = jnp.concatenate(
        [first_pair_cond_mean[None, ...], rest_pair_cond_filter_mean], axis=0
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
    separate_spike_params : bool, default=False
        Whether to fit separate spike GLM parameters per discrete state.
    update_init_mean : bool, default=True
        Whether to update initial mean during M-step.
    update_init_cov : bool, default=True
        Whether to update initial covariance during M-step.
    q_regularization : QRegularizationConfig | None, optional
        Trust-region and eigenvalue clipping configuration for Q updates.
    spike_weight_l2 : float, default=0.01
        L2 regularization strength on the spike GLM weights (not baseline).
        Adds 0.5 * spike_weight_l2 * ||weights||^2 to the M-step objective
        for each neuron. The default (0.01) corresponds to a prior std of ~10
        on weight magnitude, which allows any biologically plausible firing
        rate modulation (up to ~1000x) while preventing pathological runaway
        values. This provides numerical stability without constraining
        realistic neurons.

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
        separate_spike_params: bool = False,
        update_init_mean: bool = True,
        update_init_cov: bool = True,
        q_regularization: QRegularizationConfig | None = None,
        spike_weight_l2: float = 0.01,
        max_newton_iter: int = 1,
        line_search_beta: float = 0.5,
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
        separate_spike_params : bool, default=False
            If True, fit a separate spike GLM (baseline, weights) per discrete state.
        update_init_mean : bool, default=True
            Update initial mean during M-step.
        update_init_cov : bool, default=True
            Update initial covariance during M-step.
        q_regularization : QRegularizationConfig | None, optional
            Trust-region and eigenvalue clipping configuration for Q updates.
        spike_weight_l2 : float, default=0.01
            L2 regularization strength on the spike GLM weights.

        Raises
        ------
        ValueError
            If any of n_oscillators, n_neurons, n_discrete_states, sampling_freq,
            or dt is not positive. Also if discrete_transition_diag has wrong shape
            or spike_weight_l2 is negative.
        """
        # Validate input parameters
        if n_oscillators <= 0:
            raise ValueError(f"n_oscillators must be positive. Got {n_oscillators}.")
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive. Got {n_neurons}.")
        if n_discrete_states <= 0:
            raise ValueError(
                f"n_discrete_states must be positive. Got {n_discrete_states}."
            )
        if sampling_freq <= 0:
            raise ValueError(f"sampling_freq must be positive. Got {sampling_freq}.")
        if dt <= 0:
            raise ValueError(f"dt must be positive. Got {dt}.")
        if discrete_transition_diag is not None:
            # Convert to array first to enable shape checking for lists/tuples
            discrete_transition_diag = jnp.asarray(discrete_transition_diag)
            if discrete_transition_diag.shape != (n_discrete_states,):
                raise ValueError(
                    f"discrete_transition_diag shape mismatch: expected "
                    f"({n_discrete_states},), got {discrete_transition_diag.shape}."
                )
            if not jnp.all(
                (discrete_transition_diag >= 0) & (discrete_transition_diag <= 1)
            ):
                raise ValueError(
                    "discrete_transition_diag values must be probabilities in [0, 1]."
                )
        if spike_weight_l2 < 0:
            raise ValueError(
                f"spike_weight_l2 must be non-negative. Got {spike_weight_l2}."
            )
        if q_regularization is not None:
            if not (0.0 <= q_regularization.trust_region_weight <= 1.0):
                raise ValueError(
                    "q_regularization.trust_region_weight must be in [0, 1]."
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
        self.separate_spike_params = separate_spike_params
        self.update_init_mean = update_init_mean
        self.update_init_cov = update_init_cov

        # Regularization parameters
        self.spike_weight_l2 = spike_weight_l2
        self.q_regularization = q_regularization or QRegularizationConfig()

        # Newton iteration parameters for point-process update
        self.max_newton_iter = max_newton_iter
        self.line_search_beta = line_search_beta

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

    def _initialize_parameters(self, key: Array) -> None:
        """Initialize all model parameters.

        Sets up initial values for all model parameters including:
        - Initial state mean and covariance
        - Discrete state transition probabilities
        - Continuous state transition matrices
        - Process noise covariances
        - Spike observation parameters (baseline and weights)

        Parameters
        ----------
        key : Array
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

    def _initialize_continuous_state(self, key: Array) -> None:
        """Initialize continuous state mean and covariance.

        Parameters
        ----------
        key : Array
            Random key for sampling initial mean.

        Notes
        -----
        Each discrete state gets a different random initial mean to break
        symmetry and allow EM to differentiate between states.
        """
        # Initial mean: sample independently for each discrete state to break symmetry
        keys = jax.random.split(key, self.n_discrete_states)
        means = jax.vmap(
            lambda k: jax.random.multivariate_normal(
                key=k, mean=jnp.zeros(self.n_latent), cov=jnp.eye(self.n_latent)
            )
        )(keys)
        self.init_mean = means.T  # Shape: (n_latent, n_discrete_states)

        # Initial covariance: identity for each discrete state
        self.init_cov = jnp.stack(
            [jnp.eye(self.n_latent)] * self.n_discrete_states, axis=2
        )

    def _initialize_continuous_transition_matrix(self) -> None:
        """Initialize continuous state transition matrices.

        Uses uncoupled oscillator dynamics with default frequencies
        (uniform in [5, 15] Hz) and damping coefficients that vary slightly
        across discrete states to break symmetry.

        Notes
        -----
        Each discrete state gets slightly different damping coefficients
        (ranging from 0.90 to 0.98) to break symmetry and allow EM to
        differentiate between states based on dynamics stability.
        """
        # Default frequencies: spread across a reasonable range
        default_freqs = jnp.linspace(5.0, 15.0, self.n_oscillators)

        # Damping varies per state to break symmetry: from 0.90 to 0.98
        damping_values = jnp.linspace(0.90, 0.98, self.n_discrete_states)

        # Construct transition matrix for each discrete state
        transition_matrices = []
        for i in range(self.n_discrete_states):
            damping = jnp.full(self.n_oscillators, damping_values[i])
            A = construct_common_oscillator_transition_matrix(
                freqs=default_freqs,
                auto_regressive_coef=damping,
                sampling_freq=self.sampling_freq,
            )
            transition_matrices.append(A)

        self.continuous_transition_matrix = jnp.stack(transition_matrices, axis=2)

    def _initialize_process_covariance(self) -> None:
        """Initialize process noise covariances.

        Uses block-diagonal structure with variance that varies across
        discrete states to break symmetry.

        Notes
        -----
        Each discrete state gets different process noise variance
        (ranging from 0.005 to 0.02) to break symmetry and allow EM to
        differentiate between states based on noise level.
        """
        # Variance varies per state to break symmetry: from 0.005 to 0.02
        variance_values = jnp.linspace(0.005, 0.02, self.n_discrete_states)

        # Construct process covariance for each discrete state
        process_covs = []
        for i in range(self.n_discrete_states):
            variance = jnp.full(self.n_oscillators, variance_values[i])
            Q = construct_common_oscillator_process_covariance(variance=variance)
            process_covs.append(Q)

        self.process_cov = jnp.stack(process_covs, axis=2)

    def _initialize_spike_params(self, key: Array) -> None:
        """Initialize spike observation parameters.

        Parameters
        ----------
        key : Array
            Random key for sampling initial weights.
        """
        if self.separate_spike_params:
            # Baseline: zero (exp(0) = 1 Hz baseline firing rate)
            baseline = jnp.zeros((self.n_neurons, self.n_discrete_states))

            # Weights: small random values, per discrete state (axis last)
            weights = jax.random.normal(
                key, (self.n_neurons, self.n_latent, self.n_discrete_states)
            ) * 0.1
        else:
            # Baseline: zero (exp(0) = 1 Hz baseline firing rate)
            baseline = jnp.zeros(self.n_neurons)

            # Weights: small random values
            weights = jax.random.normal(key, (self.n_neurons, self.n_latent)) * 0.1

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
        if self.separate_spike_params:
            expected_baseline = (self.n_neurons, self.n_discrete_states)
            expected_weights = (
                self.n_neurons,
                self.n_latent,
                self.n_discrete_states,
            )
        else:
            expected_baseline = (self.n_neurons,)
            expected_weights = (self.n_neurons, self.n_latent)

        if self.spike_params.baseline.shape != expected_baseline:
            raise ValueError(
                f"spike_params.baseline shape mismatch: expected "
                f"{expected_baseline}, got {self.spike_params.baseline.shape}."
            )
        if self.spike_params.weights.shape != expected_weights:
            raise ValueError(
                f"spike_params.weights shape mismatch: expected "
                f"{expected_weights}, got {self.spike_params.weights.shape}."
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

        # Log-intensity function takes (state, params) - params passed as data, not closure
        def log_intensity_func(state: Array, params: SpikeObsParams) -> Array:
            """Compute log-intensity for all neurons given latent state and params."""
            return params.baseline + params.weights @ state

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
            spike_params=self.spike_params,
            max_newton_iter=self.max_newton_iter,
            line_search_beta=self.line_search_beta,
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

        The process covariance Q can be regularized via a trust-region blend and
        eigenvalue clipping (see ``QRegularizationConfig``) to stabilize updates
        when the Laplace-EKF approximation produces unreliable estimates.

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
            cfg = self.q_regularization
            if cfg.enabled:
                # Blend new Q with previous Q (trust region)
                Q_blended = (
                    cfg.trust_region_weight * new_Q
                    + (1 - cfg.trust_region_weight) * self.process_cov
                )

                def clip_eigenvalues(Q: Array) -> Array:
                    Q = symmetrize(Q)
                    eigvals, eigvecs = jnp.linalg.eigh(Q)
                    if cfg.min_eigenvalue is not None:
                        eigvals = jnp.maximum(eigvals, cfg.min_eigenvalue)
                    if cfg.max_eigenvalue is not None:
                        eigvals = jnp.minimum(eigvals, cfg.max_eigenvalue)
                    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T

                # Clip eigenvalues per state
                Q_clipped = jax.vmap(clip_eigenvalues, in_axes=-1, out_axes=-1)(
                    Q_blended
                )
                self.process_cov = Q_clipped
            else:
                self.process_cov = new_Q

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

        If ``separate_spike_params`` is True, this method instead fits one
        spike GLM per discrete state using state-conditional means/covariances
        and state responsibilities as time weights.

        **Limitation (shared-parameter mode)**: This approach uses a single
        marginalized mean and ignores both the per-state posterior means and
        the posterior covariance. For a switching model with nonlinear log-link
        observation model, the correct EM objective would require:

        1. State-conditional expectations: E_q[log p(y|x) | S_t=j] for each state
        2. Integration over posterior covariance (second-order correction)

        The current implementation can introduce bias when state-conditional
        latents differ significantly. For improved accuracy, consider:

        - Using ``update_spike_glm_params_second_order`` with aggregated
          state-conditional covariances
        - Fitting separate observation parameters per discrete state

        """
        if not self.update_spike_params:
            return

        if self.separate_spike_params:
            # Minimum total weight required to update parameters for a state.
            # States with less weight keep their current parameters unchanged.
            MIN_STATE_WEIGHT = 1e-8

            def _update_single_state(
                state_index: int,
                current_baseline: Array,
                current_weights: Array,
                state_weights: Array,
                state_cond_mean: Array,
                state_cond_cov: Array,
            ) -> tuple[Array, Array]:
                """Update spike GLM params for a single discrete state."""
                current_params = SpikeObsParams(
                    baseline=current_baseline,
                    weights=current_weights,
                )

                def _keep_current(_: None) -> tuple[Array, Array]:
                    return current_baseline, current_weights

                def _do_update(_: None) -> tuple[Array, Array]:
                    updated = update_spike_glm_params(
                        spikes=spikes,
                        smoother_mean=state_cond_mean,
                        current_params=current_params,
                        dt=self.dt,
                        time_weights=state_weights,
                        weight_l2=self.spike_weight_l2,
                        smoother_cov=state_cond_cov,
                        use_second_order=True,
                    )
                    return updated.baseline, updated.weights

                # Use lax.cond for JIT-compatible conditional execution
                return jax.lax.cond(
                    jnp.sum(state_weights) < MIN_STATE_WEIGHT,
                    _keep_current,
                    _do_update,
                    None,
                )

            # Process all states using vmap for JIT compatibility
            state_indices = jnp.arange(self.n_discrete_states)

            # Transpose arrays to have state axis first for vmap
            # spike_params.baseline: (n_neurons, n_discrete_states)
            # spike_params.weights: (n_neurons, n_latent, n_discrete_states)
            # smoother_discrete_state_prob: (n_time, n_discrete_states)
            # smoother_state_cond_mean: (n_time, n_latent, n_discrete_states)
            # smoother_state_cond_cov: (n_time, n_latent, n_latent, n_discrete_states)

            updated_baselines, updated_weights = jax.vmap(
                _update_single_state,
                in_axes=(0, -1, -1, -1, -1, -1),
                out_axes=(-1, -1),
            )(
                state_indices,
                self.spike_params.baseline,
                self.spike_params.weights,
                self.smoother_discrete_state_prob,
                self.smoother_state_cond_mean,
                self.smoother_state_cond_cov,
            )

            self.spike_params = SpikeObsParams(
                baseline=updated_baselines,
                weights=updated_weights,
            )
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

        # Compute marginalized smoother covariance using law of total variance:
        # Var[x] = E[Var[x|S]] + Var[E[x|S]]
        #
        # E[Var[x|S]] = sum_j P(S=j) * Cov[x|S=j]
        # smoother_state_cond_cov: (n_time, n_latent, n_latent, n_discrete_states)
        expected_cov = jnp.einsum(
            "tlks,ts->tlk",
            self.smoother_state_cond_cov,
            self.smoother_discrete_state_prob,
        )

        # Var[E[x|S]] = sum_j P(S=j) * (m_j - m)(m_j - m)^T
        # where m = E[x] is the marginal mean computed above
        # smoother_state_cond_mean: (n_time, n_latent, n_discrete_states)
        # smoother_mean: (n_time, n_latent)
        mean_deviation = (
            self.smoother_state_cond_mean - smoother_mean[:, :, None]
        )  # (n_time, n_latent, n_discrete_states)
        # Outer product weighted by state probabilities
        var_of_mean = jnp.einsum(
            "tls,tks,ts->tlk",
            mean_deviation,
            mean_deviation,
            self.smoother_discrete_state_prob,
        )

        # Total marginal covariance
        smoother_cov = expected_cov + var_of_mean

        # Clip covariance eigenvalues to prevent numerical instability in
        # the second-order M-step. Large covariances (from early EM iterations
        # with poor initialization) can cause the Newton optimization to diverge.
        # Max eigenvalue of 1.0 corresponds to std dev ~1 in latent space.
        max_cov_eigenvalue = 1.0

        def clip_cov_eigenvalues(cov: Array) -> Array:
            cov = symmetrize(cov)
            eigvals, eigvecs = jnp.linalg.eigh(cov)
            eigvals_clipped = jnp.clip(eigvals, 0.0, max_cov_eigenvalue)
            return eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T

        smoother_cov_clipped = jax.vmap(clip_cov_eigenvalues)(smoother_cov)

        # Update spike GLM parameters with second-order correction
        self.spike_params = update_spike_glm_params(
            spikes=spikes,
            smoother_mean=smoother_mean,
            current_params=self.spike_params,
            dt=self.dt,
            weight_l2=self.spike_weight_l2,
            smoother_cov=smoother_cov_clipped,
            use_second_order=True,
        )

    def _project_parameters(self) -> None:
        """Project estimated parameters onto valid parameter spaces.

        This method ensures that model parameters satisfy their structural
        constraints after the M-step:

        1. **Transition matrix projection**: Projects each A_j to preserve
           oscillatory block structure using `project_coupled_transition_matrix`.
           Each 2x2 diagonal block is projected to the closest scaled rotation
           matrix [[a, -b], [b, a]], which preserves oscillatory dynamics.

        2. **Process covariance projection**: Ensures each Q_j is:
           - Symmetric: Q_j = (Q_j + Q_j.T) / 2
           - Positive semi-definite: Clips negative eigenvalues to a small
             positive value (1e-8)

        Notes
        -----
        This method should be called after `_m_step_dynamics()` in the EM loop.
        The projections are necessary because:

        - The unconstrained M-step for A can break the oscillatory block
          structure that is expected for coupled oscillator dynamics
        - The M-step for Q can produce non-symmetric or non-PSD matrices
          due to numerical issues or insufficient data for some discrete states

        The projections are idempotent: calling them multiple times has the
        same effect as calling once.

        The projections respect update flags: if `update_continuous_transition_matrix`
        or `update_process_cov` is False, the corresponding projection is skipped.
        """
        # Project each transition matrix to preserve oscillatory block structure
        if self.update_continuous_transition_matrix:
            self.continuous_transition_matrix = jnp.stack(
                [
                    project_coupled_transition_matrix(
                        self.continuous_transition_matrix[:, :, j]
                    )
                    for j in range(self.n_discrete_states)
                ],
                axis=-1,
            )

        # Project each process covariance to ensure PSD
        if self.update_process_cov:
            projected_Q_list = []
            for j in range(self.n_discrete_states):
                Q_j = self.process_cov[:, :, j]

                # Ensure symmetry
                Q_j = (Q_j + Q_j.T) / 2

                # Ensure PSD by eigenvalue clipping
                eigenvalues, eigenvectors = jnp.linalg.eigh(Q_j)
                # Clip negative eigenvalues to small positive value
                eigenvalues_clipped = jnp.maximum(eigenvalues, 1e-8)
                # Reconstruct PSD matrix
                Q_j_psd = eigenvectors @ jnp.diag(eigenvalues_clipped) @ eigenvectors.T
                # Ensure symmetry again (numerical precision)
                Q_j_psd = (Q_j_psd + Q_j_psd.T) / 2

                projected_Q_list.append(Q_j_psd)

            self.process_cov = jnp.stack(projected_Q_list, axis=-1)

    def fit(
        self,
        spikes: ArrayLike,
        max_iter: int = 50,
        tol: float = 1e-4,
        key: Array | None = None,
        skip_init: bool = False,
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
        skip_init : bool, default=False
            If True, skip parameter initialization and use existing parameters.
            This allows custom initialization before calling fit().

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

        # Initialize parameters (unless skip_init is True for custom initialization)
        if not skip_init:
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

            # Project parameters to valid spaces (oscillatory structure, PSD)
            self._project_parameters()

        return log_likelihoods

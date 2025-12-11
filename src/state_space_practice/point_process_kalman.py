import logging
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import (
    _kalman_smoother_update,
    psd_solve,
    sum_of_outer_products,
    symmetrize,
)
from state_space_practice.utils import check_converged

logger = logging.getLogger(__name__)


def log_conditional_intensity(design_matrix: ArrayLike, params: ArrayLike) -> Array:
    """Computes the log conditional intensity for a point process.

    Parameters
    ----------
    design_matrix : ArrayLike, shape (n_time, n_params)
        Design matrix (Z_k) used in the intensity function.
    params : ArrayLike, shape (n_params,)
        Parameters for the intensity function.

    Returns
    -------
    Array, shape (n_time,)
        Log conditional intensity (log(λ_k)).
    """
    return jnp.asarray(design_matrix) @ jnp.asarray(params)


def stochastic_point_process_filter(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
) -> tuple[Array, Array, float]:
    """Applies a Stochastic State Point Process Filter (SSPPF).

    This filter estimates a time-varying latent state ($x_k$) based on
    point process observations ($y_k$). It assumes a linear Gaussian state
    transition and a point process observation model where the conditional
    intensity $\lambda_k$ depends on the state.

    $$ x_k = A x_{k-1} + w_k, \quad w_k \sim N(0, Q) $$
    $$ \lambda_k = f(x_k, Z_k) $$
    $$ y_k \sim \text{Poisson}(\lambda_k \Delta t) \quad (\text{or Bernoulli}) $$

    The filter uses a local Gaussian approximation (Laplace-EKF approach)
    at each update step, utilizing the gradient and Hessian of the
    log-likelihood. It implements a single Newton-Raphson like step
    per time bin.

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
        Initial mean of the latent state ($x_0$).
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance of the latent state ($P_0$).
    design_matrix : ArrayLike, shape (n_time, n_params)
        Design matrix ($Z_k$) used in the intensity function.
    spike_indicator : ArrayLike, shape (n_time,)
        Observed spike counts or indicators ($y_k$).
    dt : float
        Time step size ($\Delta t$).
    transition_matrix : ArrayLike, shape (n_params, n_params)
        State transition matrix ($A$).
    process_cov : ArrayLike, shape (n_params, n_params)
        Process noise covariance ($Q$).
    log_conditional_intensity : callable
        Function `log_lambda(Z_k, x_k)` returning the log conditional
        intensity.

    Returns
    -------
    posterior_mean : Array, shape (n_time, n_params)
        Filtered posterior means ($x_{k|k}$).
    posterior_variance : Array, shape (n_time, n_params, n_params)
        Filtered posterior covariances ($P_{k|k}$).
    marginal_log_likelihood : float
        Total log-likelihood of the observations given the model.

    References
    ----------
    [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).


    """
    grad_log_conditional_intensity = jax.jacfwd(log_conditional_intensity, argnums=1)
    hess_log_conditional_intensity = jax.hessian(log_conditional_intensity, argnums=1)

    def _step(
        params_prev: tuple[Array, Array, float],
        args: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        """Point Process Adaptive Filter update step

        F : transition matrix
        Q : covariance matrix
        \theta_{k | k-1} :
        W_{k | k-1}: one_step_variance_params
        \theta_{k | k} : posterior_mean
        W_{k | k} : posterior_variance
        """

        # Unpack previous parameters
        mean_prev, variance_prev, marginal_log_likelihood = params_prev
        design_matrix_t, spike_indicator_t = args

        # One-step prediction
        one_step_mean = transition_matrix @ mean_prev
        one_step_covariance = (
            transition_matrix @ variance_prev @ transition_matrix.T + process_cov
        )
        one_step_covariance = symmetrize(one_step_covariance)

        # Compute the conditional intensity and innovation
        conditional_intensity = (
            jnp.exp(log_conditional_intensity(design_matrix_t, one_step_mean)) * dt
        )
        innovation = spike_indicator_t - conditional_intensity

        # Compute the posterior mean and variance
        one_step_grad = grad_log_conditional_intensity(design_matrix_t, one_step_mean)[
            None
        ]
        one_step_hess = hess_log_conditional_intensity(design_matrix_t, one_step_mean)

        inverse_posterior_covariance = (
            jnp.linalg.pinv(one_step_covariance)
            + ((one_step_grad.T * conditional_intensity) @ one_step_grad)
            - innovation * one_step_hess
        )
        posterior_covariance = jnp.linalg.pinv(inverse_posterior_covariance)
        posterior_mean = one_step_mean + posterior_covariance @ (
            one_step_grad.squeeze() * innovation
        )
        marginal_log_likelihood += jax.scipy.stats.poisson.logpmf(
            k=spike_indicator_t, mu=conditional_intensity
        )

        return (posterior_mean, posterior_covariance, marginal_log_likelihood), (
            posterior_mean,
            posterior_covariance,
        )

    marginal_log_likelihood = jnp.array(0.0)
    (_, _, marginal_log_likelihood), (
        filtered_mean,
        filtered_cov,
    ) = jax.lax.scan(
        _step,
        (init_mean_params, init_covariance_params, marginal_log_likelihood),
        (design_matrix, spike_indicator),
    )

    return filtered_mean, filtered_cov, marginal_log_likelihood


def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
) -> tuple[Array, Array, Array, float]:
    """
    Applies a Stochastic State Point Process Smoother (SSPPS).

    This smoother estimates a time-varying latent state ($x_k$) based on
    point process observations ($y_k$) using a Kalman smoother approach.
    It first applies a stochastic point process filter to obtain the filtered
    means and covariances, and then applies a Kalman smoother to refine these estimates.
    The smoother uses the filtered means and covariances to compute the smoothed
    means, covariances, and cross-covariances.
    $$ x_k = A x_{k-1} + w_k, \quad w_k \sim N(0, Q) $$
    $$ \lambda_k = f(x_k, Z_k) $$
    $$ y_k \sim \text{Poisson}(\lambda_k \Delta t) \quad (\text{or Bernoulli}) $$
    The smoother uses a Kalman smoother update step to refine the filtered estimates.
    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
        Initial mean of the latent state ($x_0$).
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance of the latent state ($P_0$).
    design_matrix : ArrayLike, shape (n_time, n_params)
        Design matrix ($Z_k$) used in the intensity function.
    spike_indicator : ArrayLike, shape (n_time,)
        Observed spike counts or indicators ($y_k$).
    dt : float
        Time step size ($\Delta t$).
    transition_matrix : ArrayLike, shape (n_params, n_params)
        State transition matrix ($A$).
    process_cov : ArrayLike, shape (n_params, n_params)
        Process noise covariance ($Q$).
    log_conditional_intensity : callable
        Function `log_lambda(Z_k, x_k)` returning the log conditional
        intensity.

    Returns
    -------
    smoother_mean : Array, shape (n_time, n_params)
        Smoothed posterior means ($x_{k|T}$).
    smoother_cov : Array, shape (n_time, n_params, n_params)
        Smoothed posterior covariances ($P_{k|T}$).
    smoother_cross_cov : Array, shape (n_time - 1, n_params, n_params)
        Smoothed cross-covariances ($P_{k|T, k-1}$).
    marginal_log_likelihood : float
        Total log-likelihood of the observations given the model.

    References
    ----------
    [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998 (2004).
    """
    filtered_mean, filtered_cov, marginal_log_likelihood = (
        stochastic_point_process_filter(
            init_mean_params,
            init_covariance_params,
            design_matrix,
            spike_indicator,
            dt,
            transition_matrix,
            process_cov,
            log_conditional_intensity,
        )
    )

    def _step(carry, args):
        (
            next_smoother_mean,
            next_smoother_cov,
        ) = carry

        filter_mean, filter_cov = args

        smoother_mean, smoother_cov, smoother_cross_cov = _kalman_smoother_update(
            next_smoother_mean,
            next_smoother_cov,
            filter_mean,
            filter_cov,
            process_cov,
            transition_matrix,
        )
        return (
            smoother_mean,
            smoother_cov,
        ), (
            smoother_mean,
            smoother_cov,
            smoother_cross_cov,
        )

    (_, _), (smoother_mean, smoother_cov, smoother_cross_cov) = jax.lax.scan(
        _step,
        (filtered_mean[-1], filtered_cov[-1]),
        (filtered_mean[:-1], filtered_cov[:-1]),
        reverse=True,
    )

    smoother_mean = jnp.concatenate((smoother_mean, filtered_mean[-1][None]))
    smoother_cov = jnp.concatenate((smoother_cov, filtered_cov[-1][None]))

    return smoother_mean, smoother_cov, smoother_cross_cov, marginal_log_likelihood


def kalman_maximization_step(
    smoother_mean: ArrayLike,
    smoother_cov: ArrayLike,
    smoother_cross_cov: ArrayLike,
) -> tuple[Array, Array, Array, Array]:
    """Maximization step for the Kalman filter.

    Parameters
    ----------
    smoother_mean : ArrayLike, shape (n_time, n_cont_states)
        smoother mean.
    smoother_cov : ArrayLike, shape (n_time, n_cont_states, n_cont_states)
        smoother covariance.
    smoother_cross_cov : ArrayLike, shape (n_time - 1, n_cont_states, n_cont_states)
        smoother cross-covariance.

    Returns
    -------
    transition_matrix : Array, shape (n_cont_states, n_cont_states)
        Transition matrix.
    process_cov : Array, shape (n_cont_states, n_cont_states)
        Process covariance.
    mean_init : Array, shape (n_cont_states,)
        Initial mean.
    cov_init : Array, shape (n_cont_states, n_cont_states)
        Initial covariance.

    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """
    smoother_mean = jnp.asarray(smoother_mean)
    smoother_cov = jnp.asarray(smoother_cov)
    smoother_cross_cov = jnp.asarray(smoother_cross_cov)

    n_time = smoother_mean.shape[0]

    # Compute intermediate expectation terms
    gamma = jnp.sum(smoother_cov, axis=0) + sum_of_outer_products(
        smoother_mean, smoother_mean
    )
    gamma1 = gamma - jnp.outer(smoother_mean[-1], smoother_mean[-1]) - smoother_cov[-1]
    gamma2 = gamma - jnp.outer(smoother_mean[0], smoother_mean[0]) - smoother_cov[0]
    beta = (
        smoother_cross_cov.sum(axis=0)
        + sum_of_outer_products(smoother_mean[:-1], smoother_mean[1:])
    ).T

    # Transition matrix
    transition_matrix = psd_solve(gamma1, beta.T).T

    # Process covariance
    process_cov = (gamma2 - transition_matrix @ beta.T) / (n_time - 1)
    process_cov = symmetrize(process_cov)

    # Initial mean and covariance
    init_mean = smoother_mean[0]
    init_cov = smoother_cov[0]

    return (
        transition_matrix,
        process_cov,
        init_mean,
        init_cov,
    )


def get_confidence_interval(
    posterior_mean: ArrayLike, posterior_covariance: ArrayLike, alpha: float = 0.01
) -> Array:
    """Get the confidence interval from the posterior covariance

    Parameters
    ----------
    posterior_mean : ArrayLike, shape (n_time, n_params)
    posterior_covariance : ArrayLike, shape (n_time, n_params, n_params)
    alpha : float, optional
        Confidence level, by default 0.01
    """
    posterior_mean = jnp.asarray(posterior_mean)
    posterior_covariance = jnp.asarray(posterior_covariance)
    z = jax.scipy.stats.norm.ppf(1 - alpha / 2)
    ci = z * jnp.sqrt(
        jnp.diagonal(posterior_covariance, axis1=-2, axis2=-1)
    )  # shape (n_time, n_params)

    return jnp.stack((posterior_mean - ci, posterior_mean + ci), axis=-1)


def steepest_descent_point_process_filter(
    init_mean_params: ArrayLike,
    x: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    epsilon: ArrayLike,
    log_receptive_field_model: Callable[[ArrayLike, ArrayLike], Array],
) -> Array:
    """Steepest Descent Point Process Filter (SDPPF)

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
    x : ArrayLike, shape (n_time,)
        Continuous-valued input signal
    spike_indicator : ArrayLike, shape (n_time,)
        Spike count
    dt : float
        Time step
    epsilon : ArrayLike, shape (n_params, n_params)
        Learning rate
    log_receptive_field_model : callable
        Function that takes in `x` and parameters and returns the log spike rate

    Returns
    -------
    posterior_mean : Array, shape (n_time, n_params)

    References
    ----------
    .. [1] Brown, E.N., Nguyen, D.P., Frank, L.M., Wilson, M.A., and Solo, V. (2001).
    An analysis of neural receptive field plasticity by point process adaptive filtering.
    Proceedings of the National Academy of Sciences 98, 12261–12266.
    https://doi.org/10.1073/pnas.201409398.

    .. [2] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).

    Notes
    -----
    Equation in [1] is for the likelihood while in [2] it is for the log likelihood.
    This implementation follows the formulation in [2].

    """
    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)

    def _update(
        mean_prev: Array, args: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        """Steepest Descent Point Process Filter update step"""
        x_t, spike_indicator_t = args
        conditional_intensity = jnp.exp(log_receptive_field_model(x_t, mean_prev)) * dt
        innovation = spike_indicator_t - conditional_intensity
        one_step_grad = grad_log_receptive_field_model(x_t, mean_prev)
        posterior_mean = mean_prev + epsilon @ one_step_grad * innovation

        return posterior_mean, posterior_mean

    return jax.lax.scan(_update, init_mean_params, (x, spike_indicator))[1]


class PointProcessModel:
    """Point Process State-Space Model with EM fitting.

    Implements the Eden & Brown (2004) adaptive point process filter/smoother
    with EM algorithm for parameter estimation.

    Model:
        x_k = A @ x_{k-1} + w_k,  w_k ~ N(0, Q)
        n_k ~ Poisson(exp(Z_k @ x_k) * dt)

    The EM algorithm estimates the state dynamics parameters (A, Q) while
    the latent states x_k are estimated via the E-step (filter/smoother).

    Parameters
    ----------
    n_state_dims : int
        Dimension of the latent state.
    dt : float
        Time step size.
    transition_matrix : ArrayLike, optional
        Initial state transition matrix A. Default is identity (random walk).
    process_cov : ArrayLike, optional
        Initial process noise covariance Q.
    init_mean : ArrayLike, optional
        Initial state mean.
    init_cov : ArrayLike, optional
        Initial state covariance.
    log_intensity_func : callable, optional
        Function log_lambda(Z_k, x_k) returning log conditional intensity.
        Default is linear: Z_k @ x_k.
    update_transition_matrix : bool
        Whether to update A in M-step. Default True.
    update_process_cov : bool
        Whether to update Q in M-step. Default True.
    update_init_state : bool
        Whether to update initial state in M-step. Default True.

    Attributes
    ----------
    smoother_mean : Array
        Smoothed state estimates after fitting.
    smoother_cov : Array
        Smoothed state covariances after fitting.
    smoother_cross_cov : Array
        Smoothed cross-covariances after fitting.

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """

    def __init__(
        self,
        n_state_dims: int,
        dt: float,
        transition_matrix: Optional[ArrayLike] = None,
        process_cov: Optional[ArrayLike] = None,
        init_mean: Optional[ArrayLike] = None,
        init_cov: Optional[ArrayLike] = None,
        log_intensity_func: Optional[Callable] = None,
        update_transition_matrix: bool = True,
        update_process_cov: bool = True,
        update_init_state: bool = True,
    ):
        self.n_state_dims = n_state_dims
        self.dt = dt

        # Initialize parameters
        if transition_matrix is None:
            self.transition_matrix = jnp.eye(n_state_dims)
        else:
            self.transition_matrix = jnp.asarray(transition_matrix)

        if process_cov is None:
            self.process_cov = jnp.eye(n_state_dims) * 1e-4
        else:
            self.process_cov = jnp.asarray(process_cov)

        if init_mean is None:
            self.init_mean = jnp.zeros(n_state_dims)
        else:
            self.init_mean = jnp.asarray(init_mean)

        if init_cov is None:
            self.init_cov = jnp.eye(n_state_dims)
        else:
            self.init_cov = jnp.asarray(init_cov)

        if log_intensity_func is None:
            self.log_intensity_func = log_conditional_intensity
        else:
            self.log_intensity_func = log_intensity_func

        # Update flags
        self.update_transition_matrix = update_transition_matrix
        self.update_process_cov = update_process_cov
        self.update_init_state = update_init_state

        # Results (populated after fit)
        self.smoother_mean: Optional[Array] = None
        self.smoother_cov: Optional[Array] = None
        self.smoother_cross_cov: Optional[Array] = None
        self.filtered_mean: Optional[Array] = None
        self.filtered_cov: Optional[Array] = None

    def _e_step(
        self, design_matrix: ArrayLike, spike_indicator: ArrayLike
    ) -> float:
        """E-step: Run filter and smoother to estimate latent states.

        Parameters
        ----------
        design_matrix : ArrayLike, shape (n_time, n_state_dims)
        spike_indicator : ArrayLike, shape (n_time,)

        Returns
        -------
        marginal_log_likelihood : float
        """
        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_log_likelihood,
        ) = stochastic_point_process_smoother(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
        )

        # Also store filtered results
        self.filtered_mean, self.filtered_cov, _ = stochastic_point_process_filter(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
        )

        return float(marginal_log_likelihood)

    def _m_step(self) -> None:
        """M-step: Update model parameters based on smoothed estimates."""
        if self.smoother_mean is None or self.smoother_cov is None:
            raise RuntimeError("Must run E-step before M-step")

        transition_matrix, process_cov, init_mean, init_cov = kalman_maximization_step(
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
        )

        if self.update_transition_matrix:
            self.transition_matrix = transition_matrix

        if self.update_process_cov:
            # Ensure positive definiteness
            process_cov = symmetrize(process_cov)
            # Ensure eigenvalues are positive (numerical stability)
            eigvals, eigvecs = jnp.linalg.eigh(process_cov)
            eigvals = jnp.maximum(eigvals, 1e-10)
            process_cov = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
            self.process_cov = process_cov

        if self.update_init_state:
            self.init_mean = init_mean
            self.init_cov = symmetrize(init_cov)

    def fit(
        self,
        design_matrix: ArrayLike,
        spike_indicator: ArrayLike,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[float]:
        """Fit the model using the EM algorithm.

        Parameters
        ----------
        design_matrix : ArrayLike, shape (n_time, n_state_dims)
            Design matrix for the intensity function.
        spike_indicator : ArrayLike, shape (n_time,)
            Observed spike counts or indicators.
        max_iter : int
            Maximum number of EM iterations.
        tolerance : float
            Convergence tolerance for relative change in log-likelihood.

        Returns
        -------
        log_likelihoods : list[float]
            Log-likelihood at each iteration.
        """
        design_matrix = jnp.asarray(design_matrix)
        spike_indicator = jnp.asarray(spike_indicator)

        log_likelihoods: list[float] = []
        previous_log_likelihood = -jnp.inf

        for iteration in range(max_iter):
            # E-step
            current_log_likelihood = self._e_step(design_matrix, spike_indicator)
            log_likelihoods.append(current_log_likelihood)

            # Check convergence
            is_converged, is_increasing = check_converged(
                current_log_likelihood, previous_log_likelihood, tolerance
            )

            if not is_increasing:
                logger.warning(
                    f"Log-likelihood decreased at iteration {iteration + 1}!"
                )

            if is_converged:
                logger.info(f"Converged after {iteration + 1} iterations.")
                break

            # M-step
            self._m_step()

            logger.info(
                f"Iteration {iteration + 1}/{max_iter}\t"
                f"Log-Likelihood: {current_log_likelihood:.4f}\t"
                f"Change: {(current_log_likelihood - previous_log_likelihood):.6f}"
            )
            previous_log_likelihood = current_log_likelihood

        if len(log_likelihoods) == max_iter:
            logger.warning("Reached maximum iterations without converging.")

        return log_likelihoods

    def get_rate_estimate(
        self,
        design_matrix: ArrayLike,
        use_smoothed: bool = True,
    ) -> Array:
        """Get the estimated firing rate.

        Parameters
        ----------
        design_matrix : ArrayLike, shape (n_time, n_state_dims) or (n_pos, n_state_dims)
            Design matrix to evaluate rate at.
        use_smoothed : bool
            If True, use smoothed estimates; otherwise use filtered.

        Returns
        -------
        rate : Array, shape (n_time,) or (n_time, n_pos)
            Estimated firing rate in Hz.
        """
        if use_smoothed:
            if self.smoother_mean is None:
                raise RuntimeError("Model has not been fitted yet.")
            state_estimate = self.smoother_mean
        else:
            if self.filtered_mean is None:
                raise RuntimeError("Model has not been fitted yet.")
            state_estimate = self.filtered_mean

        # state_estimate: (n_time, n_state_dims)
        # design_matrix: (n_time, n_state_dims) or (n_pos, n_state_dims)
        log_rate = state_estimate @ design_matrix.T
        rate = jnp.exp(log_rate) / self.dt

        return rate

    def get_confidence_interval(
        self, alpha: float = 0.05, use_smoothed: bool = True
    ) -> Array:
        """Get confidence intervals for the state estimates.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI).
        use_smoothed : bool
            If True, use smoothed estimates; otherwise use filtered.

        Returns
        -------
        ci : Array, shape (n_time, n_state_dims, 2)
            Lower and upper bounds of the confidence interval.
        """
        if use_smoothed:
            if self.smoother_mean is None or self.smoother_cov is None:
                raise RuntimeError("Model has not been fitted yet.")
            mean = self.smoother_mean
            cov = self.smoother_cov
        else:
            if self.filtered_mean is None or self.filtered_cov is None:
                raise RuntimeError("Model has not been fitted yet.")
            mean = self.filtered_mean
            cov = self.filtered_cov

        return get_confidence_interval(mean, cov, alpha=alpha)

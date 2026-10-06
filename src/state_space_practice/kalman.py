"""Kalman filter and smoother for linear Gaussian state-space models.

Implements the Kalman filter, Rauch-Tung-Striebel (RTS) smoother, and
the Expectation-Maximization (EM) algorithm's M-step for parameter estimation.

The assumed state-space model is:
$$ x_t = A x_{t-1} + w_t, \\quad w_t \\sim N(0, \\Sigma) $$
$$ y_t = H x_t + v_t, \\quad v_t \\sim N(0, R) $$

References
----------
1. Sarkka, S. (2013). Bayesian Filtering and Smoothing
  (Cambridge University Press) https://doi.org/10.1017/CBO9781139344203.
2. Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
   linear Gaussian models. Neural computation, 11(2), 305-345.

"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.stats.multivariate_normal
from jax.typing import ArrayLike

from state_space_practice.utils import (  # noqa: F401 — re-exported for backward compat
    _stabilizing_shift,
    _validate_filter_numerics,
    contains_tracer,
    project_psd_relative,
    psd_solve,
    stabilize_covariance,
    symmetrize,
    typed_jit,
    validate_finite_array,
    warn_if_not_positive_definite_in_graph,
)

# Gain solves (Kalman gain ``S^{-1} H P``, RTS gain ``P_pred^{-1} A P``) and the
# M-step regression solves use a purely scale-relative Cholesky shift. The
# linear-Gaussian recursions are equivariant under a change of units
# (x -> c x, covariances -> c^2 cov); an absolute shift breaks that, perturbing
# the posterior of any model whose covariances are comparable to or smaller
# than the shift. The matrices solved here are positive definite whenever the
# inputs are valid (R > 0 makes S > 0), so the shift only guards against
# round-off; the tiny absolute part only keeps an all-zero matrix
# factorizable. The relative shift is never below the dtype's machine
# epsilon (a 1e-14 shift rounds away in float32), and when even that is below
# the round-off of a singular float32 covariance whose diagonal entries differ
# in scale, the solve retries with a ``sqrt(eps)`` shift (see _gain_solve).
_GAIN_SOLVE_RELATIVE_BOOST = 1e-14
_GAIN_SOLVE_ABSOLUTE_BOOST = 1e-300


def _gain_solve(cov: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve ``cov @ x = rhs`` for a PD ``cov`` with a scale-relative shift.

    The relative shift is ``max(1e-14, eps)`` times each diagonal entry of
    ``cov`` (see :func:`state_space_practice.utils.psd_solve`), or
    ``sqrt(eps)`` when the smaller shift gives a non-finite Cholesky factor;
    ``eps`` is that of ``cov``'s own dtype, whose rounding it carries even
    when a float32 / float64 mix is solved in the promoted dtype. The shift
    is chosen and the matrix factored on a gradient-free copy; the factor is
    reused by :func:`jax.lax.custom_linear_solve`, which differentiates the
    stabilized system (including the shift) implicitly, so a failed
    factorization never reaches autodiff.
    """
    # custom_linear_solve needs the matrix, right-hand side and solution in
    # one floating dtype (integer inputs are promoted).
    dtype = jnp.result_type(cov, rhs, 1.0)
    # A float cov keeps its own precision's shift; an integer cov is exact, so
    # its shift follows the dtype the solve runs in.
    cov_dtype = jnp.result_type(cov)
    eps_dtype = cov_dtype if jnp.issubdtype(cov_dtype, jnp.inexact) else dtype
    eps = float(jnp.finfo(eps_dtype).eps)
    cov = symmetrize(jnp.asarray(cov, dtype=dtype))
    rhs = jnp.asarray(rhs, dtype=dtype)
    idx = jnp.arange(cov.shape[-1])

    def stabilized(cov: jax.Array, relative_boost: float | jax.Array) -> jax.Array:
        shift = _stabilizing_shift(cov, _GAIN_SOLVE_ABSOLUTE_BOOST, relative_boost)
        return cov.at[..., idx, idx].add(shift)

    def factor_with(relative_boost: float) -> tuple[jax.Array, jax.Array]:
        factor = jnp.linalg.cholesky(stabilized(cov_const, relative_boost))
        return jnp.asarray(relative_boost, dtype=dtype), factor

    cov_const = jax.lax.stop_gradient(cov)
    relative_boost, factor = factor_with(max(_GAIN_SOLVE_RELATIVE_BOOST, eps))
    relative_boost, factor = jax.lax.cond(
        jnp.all(jnp.isfinite(factor)),
        lambda: (relative_boost, factor),
        lambda: factor_with(eps**0.5),
    )
    matrix = stabilized(cov, relative_boost)
    return cast(
        jax.Array,
        jax.lax.custom_linear_solve(
            lambda x: matrix @ x,
            rhs,
            solve=lambda _, b: jax.scipy.linalg.cho_solve((factor, True), b),
            symmetric=True,
        ),
    )


def woodbury_kalman_gain(
    prior_cov: ArrayLike,
    emission_matrix: ArrayLike,
    emission_cov_diag: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Kalman gain using the Woodbury identity for diagonal R.

    When the observation noise covariance R is diagonal, the standard dense
    Kalman gain solve is O(D_obs^3). The Woodbury identity replaces that solve
    with O(D_obs * D_state^2 + D_state^3) work, which is much faster when
    D_obs >> D_state (e.g. many neurons, low-dimensional latent state).

    This compatibility helper still returns dense ``S`` and ``S_inv``, so its
    output memory is O(D_obs^2). Large-observation call sites that only need a
    gain-times-innovation action should use a factor/action API instead of
    materializing those dense matrices.

    Parameters
    ----------
    prior_cov : ArrayLike, shape (D_state, D_state)
        Prior (predicted) state covariance P.
    emission_matrix : ArrayLike, shape (D_obs, D_state)
        Observation matrix H.
    emission_cov_diag : ArrayLike, shape (D_obs,)
        Diagonal of observation noise covariance R.

    Returns
    -------
    K : jax.Array, shape (D_state, D_obs)
        Kalman gain.
    S : jax.Array, shape (D_obs, D_obs)
        Innovation covariance H P H' + R (for log-likelihood).
    S_inv : jax.Array, shape (D_obs, D_obs)
        Inverse of innovation covariance (via Woodbury).
    """
    prior_cov = jnp.asarray(prior_cov)
    emission_matrix = jnp.asarray(emission_matrix)
    emission_cov_diag = jnp.asarray(emission_cov_diag)

    if prior_cov.ndim != 2 or prior_cov.shape[0] != prior_cov.shape[1]:
        raise ValueError(
            "prior_cov must be a square 2D covariance matrix, "
            f"got shape {prior_cov.shape}."
        )
    if emission_matrix.ndim != 2:
        raise ValueError(
            "emission_matrix must have shape (D_obs, D_state), "
            f"got shape {emission_matrix.shape}."
        )
    D = prior_cov.shape[0]
    D_obs = emission_matrix.shape[0]
    if emission_matrix.shape[1] != D:
        raise ValueError(
            "emission_matrix state dimension must match prior_cov; "
            f"got {emission_matrix.shape[1]} and {D}."
        )
    if emission_cov_diag.ndim != 1:
        raise ValueError(
            "emission_cov_diag must be a 1D vector of positive variances, "
            f"got shape {emission_cov_diag.shape}."
        )
    if emission_cov_diag.shape != (D_obs,):
        raise ValueError(
            "emission_cov_diag length must match emission_matrix rows; "
            f"got {emission_cov_diag.shape} for D_obs={D_obs}."
        )
    if not contains_tracer(prior_cov, emission_matrix, emission_cov_diag):
        for name, arr in (
            ("prior_cov", prior_cov),
            ("emission_matrix", emission_matrix),
            ("emission_cov_diag", emission_cov_diag),
        ):
            validate_finite_array(name, arr)
        if not bool(jnp.all(emission_cov_diag > 0.0)):
            raise ValueError("emission_cov_diag entries must be positive.")

    dtype = jnp.result_type(prior_cov, emission_matrix, emission_cov_diag)
    prior_cov = prior_cov.astype(dtype)
    emission_matrix = emission_matrix.astype(dtype)
    emission_cov_diag = emission_cov_diag.astype(dtype)

    r_inv = 1.0 / emission_cov_diag
    R_inv_H = emission_matrix * r_inv[:, None]
    Ht_R_inv_H = emission_matrix.T @ R_inv_H
    I_D = jnp.eye(D, dtype=dtype)

    # Matrix inversion lemma without factorizing P:
    # S^{-1} = R^{-1} - R^{-1} H P (I + H' R^{-1} H P)^{-1} H' R^{-1}.
    # This remains finite for positive-semidefinite/rank-deficient P.
    small_inv_term = jnp.linalg.solve(I_D + Ht_R_inv_H @ prior_cov, R_inv_H.T)
    S_inv = symmetrize(jnp.diag(r_inv) - R_inv_H @ prior_cov @ small_inv_term)

    # Equivalent small-system expression for K = P H' S^{-1}; avoids a dense
    # (D_obs, D_obs) multiply when callers only consume K.
    K_rhs = jnp.linalg.solve(I_D + prior_cov @ Ht_R_inv_H, prior_cov)
    K = (R_inv_H @ K_rhs).T
    S = symmetrize(
        jnp.diag(emission_cov_diag) + emission_matrix @ prior_cov @ emission_matrix.T
    )
    return K, S, S_inv


def standard_kalman_gain(
    prior_cov: ArrayLike,
    emission_matrix: ArrayLike,
    emission_cov: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Compute Kalman gain using the standard formula.

    Parameters
    ----------
    prior_cov : ArrayLike, shape (D_state, D_state)
        Prior (predicted) state covariance P.
    emission_matrix : ArrayLike, shape (D_obs, D_state)
        Observation matrix H.
    emission_cov : ArrayLike, shape (D_obs, D_obs)
        Observation noise covariance R.

    Returns
    -------
    K : jax.Array, shape (D_state, D_obs)
        Kalman gain.
    S : jax.Array, shape (D_obs, D_obs)
        Innovation covariance H P H' + R.
    """
    emission_matrix = jnp.asarray(emission_matrix)
    S = symmetrize(emission_matrix @ prior_cov @ emission_matrix.T + emission_cov)
    K = psd_solve(S, emission_matrix @ prior_cov).T
    return K, S


def joseph_form_update(
    prior_cov: ArrayLike,
    kalman_gain: ArrayLike,
    emission_matrix: ArrayLike,
    emission_cov: ArrayLike,
) -> jax.Array:
    """Joseph form covariance update: always PSD by construction.

    Computes ``P_post = (I - K H) P (I - K H)' + K R K'``, which is
    a sum of PSD terms and therefore guaranteed PSD regardless of
    floating-point rounding.

    Parameters
    ----------
    prior_cov : ArrayLike, shape (D, D)
        Prior (predicted) state covariance.
    kalman_gain : ArrayLike, shape (D, D_obs)
        Kalman gain K.
    emission_matrix : ArrayLike, shape (D_obs, D)
        Observation matrix H.
    emission_cov : ArrayLike, shape (D_obs, D_obs)
        Observation noise covariance R.

    Returns
    -------
    jax.Array, shape (D, D)
        Posterior covariance, guaranteed PSD.
    """
    prior_cov = jnp.asarray(prior_cov)
    kalman_gain = jnp.asarray(kalman_gain)
    D = prior_cov.shape[0]
    I_KH = jnp.eye(D, dtype=prior_cov.dtype) - kalman_gain @ emission_matrix
    return symmetrize(
        I_KH @ prior_cov @ I_KH.T + kalman_gain @ emission_cov @ kalman_gain.T
    )


def _prepare_kalman_inputs(
    init_mean: ArrayLike,
    init_cov: ArrayLike,
    obs: ArrayLike,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    measurement_matrix: ArrayLike,
    measurement_cov: ArrayLike,
    *,
    validate: bool,
    filter_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Convert public linear-Gaussian inputs to arrays, optionally validating.

    With ``validate=False`` the inputs are only converted with ``jnp.asarray``.
    Otherwise shape checks always run. Value checks (finiteness, symmetry,
    positive definiteness, the float32 risk warning) run host-side on concrete
    inputs, including concrete constants closed over by a jitted caller. When
    any input is a JAX tracer they are skipped, and a non-positive-definite
    ``init_cov`` is reported at run time as a ``StateSpaceWarning`` (see
    :func:`~state_space_practice.utils.warn_if_not_positive_definite_in_graph`).

    Must be called directly from the public entry point: the float32 risk
    warning uses a fixed ``stacklevel`` that points at that function's caller.
    """
    if not validate:
        return (
            jnp.asarray(init_mean),
            jnp.asarray(init_cov),
            jnp.asarray(obs),
            jnp.asarray(transition_matrix),
            jnp.asarray(process_cov),
            jnp.asarray(measurement_matrix),
            jnp.asarray(measurement_cov),
        )

    # Under an active trace, jnp.asarray on a concrete (numpy) constant would
    # stage it; converting under ensure_compile_time_eval keeps it concrete.
    with jax.ensure_compile_time_eval():
        init_mean = jnp.asarray(init_mean)
        init_cov = jnp.asarray(init_cov)
        obs = jnp.asarray(obs)
        transition_matrix = jnp.asarray(transition_matrix)
        process_cov = jnp.asarray(process_cov)
        measurement_matrix = jnp.asarray(measurement_matrix)
        measurement_cov = jnp.asarray(measurement_cov)

    if init_mean.ndim != 1:
        raise ValueError(f"init_mean must be 1D, got shape {init_mean.shape}.")
    if obs.ndim != 2:
        raise ValueError(f"obs must have shape (n_time, n_obs_dim), got {obs.shape}.")
    if obs.shape[0] == 0:
        raise ValueError("obs must contain at least one time step.")

    n_state = init_mean.shape[0]
    n_obs = obs.shape[1]
    n_time = obs.shape[0]
    expected_shapes = {
        "init_cov": (n_state, n_state),
        "transition_matrix": (n_state, n_state),
        "process_cov": (n_state, n_state),
        "measurement_matrix": (n_obs, n_state),
    }
    actual_arrays = {
        "init_cov": init_cov,
        "transition_matrix": transition_matrix,
        "process_cov": process_cov,
        "measurement_matrix": measurement_matrix,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_arrays[name].shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {actual_arrays[name].shape}."
            )

    # measurement_cov may be a single constant matrix (n_obs, n_obs) or a
    # per-time-step stack with a leading time axis (n_time, n_obs, n_obs).
    measurement_cov_is_time_varying = measurement_cov.ndim == 3
    if measurement_cov_is_time_varying:
        if measurement_cov.shape != (n_time, n_obs, n_obs):
            raise ValueError(
                "measurement_cov with a leading time axis must have shape "
                f"(n_time={n_time}, {n_obs}, {n_obs}); "
                f"got {measurement_cov.shape}."
            )
    elif measurement_cov.shape != (n_obs, n_obs):
        raise ValueError(
            f"measurement_cov must have shape ({n_obs}, {n_obs}) or, for "
            f"time-varying observation noise, (n_time={n_time}, {n_obs}, "
            f"{n_obs}); got {measurement_cov.shape}."
        )

    arrays = (
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )

    # Value checks need concrete arrays: skip them when any input is traced
    # (bool()/float() on a tracer raises), but still report a
    # non-positive-definite init_cov from inside the computation. The shape
    # checks above are static and always run.
    if contains_tracer(*arrays):
        warn_if_not_positive_definite_in_graph(
            init_cov, name="init_cov", filter_name=filter_name
        )
        return arrays

    # Inside an active trace (concrete constants closed over by a jitted
    # caller) evaluate eagerly so the results stay concrete.
    with jax.ensure_compile_time_eval():
        for name, arr in (
            ("init_mean", init_mean),
            ("obs", obs),
            ("transition_matrix", transition_matrix),
            ("measurement_matrix", measurement_matrix),
            ("measurement_cov", measurement_cov),
        ):
            validate_finite_array(name, arr)

        if measurement_cov_is_time_varying:
            if not bool(
                jnp.allclose(
                    measurement_cov,
                    jnp.swapaxes(measurement_cov, -1, -2),
                    rtol=1e-6,
                    atol=1e-8,
                )
            ):
                asym = float(
                    jnp.max(
                        jnp.abs(measurement_cov - jnp.swapaxes(measurement_cov, -1, -2))
                    )
                )
                raise ValueError(
                    "measurement_cov must be symmetric at every time step "
                    f"(max|R - R.T| = {asym:g})."
                )
            # Per-bin R: require every slice positive definite. eigvalsh is
            # batched over the leading time axis (O(n_time * n_obs^3)) but runs
            # once per public call, since inner loops pass validate_inputs=False.
            per_slice_min_eig = jnp.linalg.eigvalsh(symmetrize(measurement_cov)).min(
                axis=-1
            )
            worst_time = int(jnp.argmin(per_slice_min_eig))
            min_slice_eig = float(per_slice_min_eig[worst_time])
            if not min_slice_eig > 0.0:
                raise ValueError(
                    "measurement_cov must be positive definite at every time "
                    f"step; the minimum eigenvalue is {min_slice_eig} at "
                    f"time step {worst_time}."
                )
        _validate_filter_numerics(
            init_cov,
            n_time=int(n_time),
            stacklevel=4,
            filter_name=filter_name,
            # A time-varying R was checked slice by slice above.
            measurement_cov=(
                None if measurement_cov_is_time_varying else measurement_cov
            ),
            process_cov=process_cov,
        )
    return arrays


@typed_jit
def kalman_measurement_update(
    prior_mean: ArrayLike,
    prior_cov: ArrayLike,
    obs: ArrayLike,
    measurement_matrix: ArrayLike,
    measurement_cov: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Kalman measurement update (no prediction step).

    Parameters
    ----------
    prior_mean : ArrayLike, shape (n_cont_states,)
        Prior state mean.
    prior_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        Prior state covariance.
    obs : ArrayLike, shape (n_obs_dim,)
        Observation.
    measurement_matrix : ArrayLike, shape (n_obs_dim, n_cont_states)
        Observation matrix H.
    measurement_cov : ArrayLike, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance R.

    Returns
    -------
    posterior_mean : jax.Array, shape (n_cont_states,)
    posterior_cov : jax.Array, shape (n_cont_states, n_cont_states)
    marginal_log_likelihood : jax.Array (scalar)
    """
    prior_mean = jnp.asarray(prior_mean)
    measurement_matrix = jnp.asarray(measurement_matrix)
    obs_mean = measurement_matrix @ prior_mean
    obs_cov = symmetrize(
        measurement_matrix @ prior_cov @ measurement_matrix.T + measurement_cov
    )

    residual_error = obs - obs_mean
    kalman_gain = _gain_solve(obs_cov, measurement_matrix @ prior_cov).T

    posterior_mean = prior_mean + kalman_gain @ residual_error
    posterior_cov = joseph_form_update(
        prior_cov, kalman_gain, measurement_matrix, measurement_cov
    )

    # Evaluated on the unboosted innovation covariance (the stabilised solve
    # above adds a tiny diagonal shift); keeping the two separate matches the
    # established likelihood values that the EM convergence checks depend on.
    # An innovation covariance that does not factor (numerically singular, e.g.
    # a rank-deficient H P H^T with negligible R) gives a non-finite likelihood
    # on purpose, although the stabilised gain keeps the posterior finite: EM's
    # non-finite check then rolls back or raises instead of accepting the
    # likelihood of a silently regularised model.
    marginal_log_likelihood = jnp.asarray(
        jax.scipy.stats.multivariate_normal.logpdf(x=obs, mean=obs_mean, cov=obs_cov)
    )

    return posterior_mean, posterior_cov, marginal_log_likelihood


@typed_jit
def _kalman_filter_update(
    mean_prev: jax.Array,
    cov_prev: jax.Array,
    obs: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Performs a single update step of the Kalman filter.

    Parameters
    ----------
    mean_prev : jax.Array, shape (n_cont_states,)
        Previous state mean, $$ m_{t-1} $$.
    cov_prev : jax.Array, shape (n_cont_states, n_cont_states)
        Previous state covariance, $$ P_{t-1} $$.
    obs : jax.Array, shape (n_obs_dim,)
        Data observation, $$ y_t $$.
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \\Sigma $$.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$.

    Returns
    -------
    posterior_mean : jax.Array, shape (n_cont_states,)
        Posterior state mean, $$ m_t $$.
    posterior_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Posterior state covariance, $$ P_t $$.
    marginal_log_likelihood : jax.Array
        Log-likelihood of the observation, $$ \\log p(y_t | y_{1:t-1}) $$ (scalar array).

    """
    # One step prediction
    one_step_mean = transition_matrix @ mean_prev
    one_step_cov = symmetrize(
        transition_matrix @ cov_prev @ transition_matrix.T + process_cov
    )

    # Measurement update
    return kalman_measurement_update(
        one_step_mean, one_step_cov, obs, measurement_matrix, measurement_cov
    )


@typed_jit
def _kalman_filter_impl(
    init_mean: jax.Array,
    init_cov: jax.Array,
    obs: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Jitted Kalman-filter scan. See :func:`kalman_filter` for the public API.

    ``measurement_cov`` may be a single ``(n_obs, n_obs)`` matrix (constant
    observation noise) or a per-time-step stack ``(n_time, n_obs, n_obs)``. The
    time-varying branch is selected at trace time from the static ndim, so each
    form compiles to its own specialization.

    All inputs are promoted to their common floating dtype before the scan,
    so e.g. a float32 ``init_mean`` / ``init_cov`` with float64 parameters
    runs in float64 instead of failing on a carry whose dtype changes after
    the first step.
    """
    dtype = jnp.result_type(
        float,
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )
    init_mean = jnp.asarray(init_mean, dtype=dtype)
    init_cov = jnp.asarray(init_cov, dtype=dtype)
    obs = jnp.asarray(obs, dtype=dtype)
    transition_matrix = jnp.asarray(transition_matrix, dtype=dtype)
    process_cov = jnp.asarray(process_cov, dtype=dtype)
    measurement_matrix = jnp.asarray(measurement_matrix, dtype=dtype)
    measurement_cov = jnp.asarray(measurement_cov, dtype=dtype)
    time_varying_measurement_cov = measurement_cov.ndim == 3

    def _step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        # ``obs_t`` or ``(obs_t, measurement_cov_t)``; ``Any`` because mypy
        # joins the union ``scan`` receives as ``xs`` to ``object``.
        step_inputs: Any,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        mean_prev, cov_prev, marginal_log_likelihood = carry
        if time_varying_measurement_cov:
            obs_t, measurement_cov_t = step_inputs
        else:
            obs_t = step_inputs
            measurement_cov_t = measurement_cov
        posterior_mean, posterior_cov, marginal_log_likelihood_t = (
            _kalman_filter_update(
                mean_prev,
                cov_prev,
                obs_t,
                transition_matrix,
                process_cov,
                measurement_matrix,
                measurement_cov_t,
            )
        )

        marginal_log_likelihood += marginal_log_likelihood_t

        return (posterior_mean, posterior_cov, marginal_log_likelihood), (
            posterior_mean,
            posterior_cov,
        )

    scan_inputs = (obs, measurement_cov) if time_varying_measurement_cov else obs
    marginal_log_likelihood = jnp.zeros((), dtype=dtype)
    (
        (_, _, marginal_log_likelihood),
        (
            filtered_mean,
            filtered_cov,
        ),
    ) = jax.lax.scan(
        _step,
        (init_mean, init_cov, marginal_log_likelihood),
        scan_inputs,
    )

    return filtered_mean, filtered_cov, marginal_log_likelihood


def kalman_filter(
    init_mean: ArrayLike,
    init_cov: ArrayLike,
    obs: ArrayLike,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    measurement_matrix: ArrayLike,
    measurement_cov: ArrayLike,
    validate_inputs: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Applies the Kalman filter to a sequence of observations.

    Parameters
    ----------
    init_mean : ArrayLike, shape (n_cont_states,)
        Initial state mean, $$ m_0 $$.
    init_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        Initial state covariance, $$ P_0 $$. Must be strictly positive
        definite.
    obs : ArrayLike, shape (n_time, n_obs_dim)
        Sequence of observations, $$ y_{1:T} $$.
    transition_matrix : ArrayLike, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \\Sigma $$.
    measurement_matrix : ArrayLike, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : ArrayLike, shape (n_obs_dim, n_obs_dim) or (n_time, n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$. A 2-D array is constant over
        time; a 3-D array with a leading time axis supplies a separate
        $$ R_t $$ per time step (used by the iterated-Laplace / IRLS smoother
        in :mod:`state_space_practice.temporal_rate_gp`). Every slice must be
        strictly positive definite.
    validate_inputs : bool, default=True
        If True, validate array shapes, finite observations, non-empty
        ``obs``, positive-definite ``init_cov`` / ``measurement_cov``, and
        positive-semidefinite ``process_cov``. Also warn when f32 + long T
        is at risk of losing PSD during the scan. Inner EM/SGD call sites
        that have already validated should pass ``False`` to skip the
        O(d^3) eigenvalue recomputation.

        The value checks need concrete inputs. Concrete arrays are checked
        even inside ``jax.jit`` (e.g. constants closed over by a jitted
        function). When any input is a JAX tracer (an argument of ``jax.jit``
        / ``jax.grad`` / ``jax.vmap``) only the shape checks run, and a
        non-positive-definite ``init_cov`` is reported at run time as a
        ``StateSpaceWarning`` naming its minimum eigenvalue instead of
        raising.

    Returns
    -------
    filtered_mean : jax.Array, shape (n_time, n_cont_states)
        Filtered state means, $$ m_{1:T} $$.
    filtered_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
        Filtered state covariances, $$ P_{1:T} $$.
    marginal_log_likelihood : jax.Array
        Total log likelihood of the observations, $$ \\sum_{t=1}^T \\log p(y_t | y_{1:t-1}) $$ (scalar array).

    Raises
    ------
    ValueError
        If ``validate_inputs=True`` and any public input is malformed, non-finite,
        or has an invalid covariance. Under tracing only malformed shapes raise.

    Warns
    -----
    StateSpaceWarning
        If ``validate_inputs=True``, ``init_cov`` is traced, and it is not
        positive definite (emitted when the computation runs).
    """
    (
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    ) = _prepare_kalman_inputs(
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
        validate=validate_inputs,
        filter_name="kalman_filter",
    )
    return _kalman_filter_impl(
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )


@typed_jit
def _kalman_smoother_update(
    next_smoother_mean: jax.Array,
    next_smoother_cov: jax.Array,
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    process_cov: jax.Array,
    transition_matrix: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Performs a single backward update step of the RTS smoother.

    Parameters
    ----------
    next_smoother_mean : jax.Array, shape (n_cont_states,)
        Smoothed mean from the next time step, $$ m_{t+1|T} $$.
    next_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Smoothed covariance from the next time step, $$ P_{t+1|T} $$.
    filter_mean : jax.Array, shape (n_cont_states,)
        Filtered mean from the current time step, $$ m_{t|t} $$.
    filter_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Filtered covariance from the current time step, $$ P_{t|t} $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \\Sigma $$.
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.

    Returns
    -------
    smoother_mean : jax.Array, shape (n_cont_states,)
        Smoothed state mean, $$ m_{t|T} $$.
    smoother_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Smoothed state covariance, $$ P_{t|T} $$.
    smoother_cross_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Smoothed cross-covariance, $$ P_{t, t+1|T} $$.

    """
    # Predicted mean m_{t+1|t}
    one_step_mean = transition_matrix @ filter_mean
    # Predicted covariance P_{t+1|t}
    one_step_cov = symmetrize(
        transition_matrix @ filter_cov @ transition_matrix.T + process_cov
    )

    # Smoother gain J_t
    smoother_kalman_gain = _gain_solve(one_step_cov, transition_matrix @ filter_cov).T

    # Smoothed mean m_{t|T}
    smoother_mean = filter_mean + smoother_kalman_gain @ (
        next_smoother_mean - one_step_mean
    )

    # Smoothed covariance P_{t|T}
    smoother_cov = symmetrize(
        filter_cov
        + smoother_kalman_gain
        @ (next_smoother_cov - one_step_cov)
        @ smoother_kalman_gain.T
    )
    # Lag-one cross covariance P_{t, t+1|T}
    smoother_cross_cov = smoother_kalman_gain @ next_smoother_cov

    return smoother_mean, smoother_cov, smoother_cross_cov


def _scan_with_boundary(
    step: Callable[[Any, Any], tuple[Any, tuple[Any, Any]]],
    init_carry: Any,
    sequences: Any,
    starts: Any,
    length: int,
    boundary: Any,
    *,
    reverse: bool = False,
) -> tuple[Any, Any, Any]:
    """``lax.scan`` whose stacked outputs gain one boundary entry without copies.

    Filters and smoothers handle one time step outside their scan (the first
    observation, or the terminal smoother step). Prepending/appending it with
    ``jnp.concatenate`` copies every stacked output, and scanning over sliced
    inputs (``x[1:]``, ``x[:-1]``) copies those inputs: each is a second
    full-size buffer at peak memory. Here ``step(carry, x) -> (carry, (y, z))``
    runs for ``i = 0, ..., length - 1`` (in reverse when ``reverse``) with
    ``x`` read in the loop as ``sequence[i + start]`` for each leaf of
    ``sequences`` (so no sliced copy is made), and each ``y`` is written into
    a ``(length + 1, ...)`` buffer carried through the loop (which XLA updates
    in place) whose remaining slot -- index 0, or index ``length`` when
    ``reverse`` -- holds ``boundary``. ``z`` is stacked as usual.

    Parameters
    ----------
    step : callable
        Scan body returning ``(carry, (y, z))``; ``y`` and ``z`` are pytrees.
    init_carry : pytree
    sequences : pytree of arrays
        Full-length per-time inputs.
    starts : pytree of int with the structure of ``sequences``
        Step ``i`` reads entry ``i + start`` of the matching sequence, i.e.
        the scan runs over ``sequence[start:start + length]``.
    length : int
        Number of scan steps.
    boundary : pytree matching ``y``
        The entry placed at index 0 (forward) or ``length`` (reverse).
    reverse : bool, default=False

    Returns
    -------
    carry : pytree
        Final carry.
    ys : pytree of arrays, each of shape ``(length + 1, ...)``
        Equal to ``concatenate([boundary[None], y_stack])`` (forward) or
        ``concatenate([y_stack, boundary[None]])`` (reverse), with the same
        dtype promotion.
    zs : pytree of arrays, each of shape ``(length, ...)``
    """
    x_spec = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape[1:], a.dtype), sequences
    )
    _, (y_spec, _) = jax.eval_shape(step, init_carry, x_spec)

    boundary_index = length if reverse else 0
    offset = 0 if reverse else 1

    def make_buffer(value: jax.Array, spec: jax.ShapeDtypeStruct) -> jax.Array:
        value = jnp.asarray(value)
        dtype = jnp.promote_types(value.dtype, spec.dtype)
        buffer = jnp.zeros((length + 1, *spec.shape), dtype)
        return buffer.at[boundary_index].set(value.astype(dtype))

    buffers = jax.tree.map(make_buffer, boundary, y_spec)

    # The step index is carried (not scanned over an arange) so that no
    # length-T index array is materialized either.
    def body(carry: tuple[Any, Any, jax.Array], _: None) -> tuple[Any, Any]:
        inner_carry, bufs, index = carry
        x = jax.tree.map(
            lambda start, a: jax.lax.dynamic_index_in_dim(
                a, index + start, keepdims=False
            ),
            starts,
            sequences,
        )
        inner_carry, (y, z) = step(inner_carry, x)
        bufs = jax.tree.map(
            lambda buf, value: jax.lax.dynamic_update_index_in_dim(
                buf, value.astype(buf.dtype), index + offset, axis=0
            ),
            bufs,
            y,
        )
        next_index = index - 1 if reverse else index + 1
        return (inner_carry, bufs, next_index), z

    first_index = jnp.asarray(length - 1 if reverse else 0)
    (carry, ys, _), zs = jax.lax.scan(
        body,
        (init_carry, buffers, first_index),
        None,
        length=length,
        reverse=reverse,
    )
    return carry, ys, zs


@typed_jit
def rts_backward_scan(
    filtered_mean: ArrayLike,
    filtered_cov: ArrayLike,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sequential RTS backward smoother given filtered means and covariances.

    This is the observable-agnostic backward pass: it consumes filtered
    state estimates from any forward filter (linear-Gaussian, point-
    process / Laplace-EKF, oscillator) and produces smoothed estimates.
    :func:`_kalman_smoother_impl` and
    :func:`~state_space_practice.multinomial_choice.multinomial_choice_smoother`
    call this helper so the RTS recurrence is implemented once. Filters that
    modify their one-step prediction (e.g. the position decoder) use
    :func:`rts_backward_scan_with_predictions` instead.

    Parameters
    ----------
    filtered_mean : ArrayLike, shape (n_time, n_cont_states)
    filtered_cov : ArrayLike, shape (n_time, n_cont_states, n_cont_states)
    transition_matrix : ArrayLike, shape (n_cont_states, n_cont_states)
    process_cov : ArrayLike, shape (n_cont_states, n_cont_states)

    Returns
    -------
    smoother_mean : jax.Array, shape (n_time, n_cont_states)
    smoother_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
    smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states)
        Lag-one cross-covariances P_{t, t+1|T}.
    """
    filtered_mean = jnp.asarray(filtered_mean)
    filtered_cov = jnp.asarray(filtered_cov)
    transition_matrix = jnp.asarray(transition_matrix)
    process_cov = jnp.asarray(process_cov)

    def _step(
        carry: tuple[jax.Array, jax.Array],
        args: tuple[jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array],
        tuple[tuple[jax.Array, jax.Array], jax.Array],
    ]:
        next_smoother_mean, next_smoother_cov = carry
        filter_mean, filter_cov = args
        smoother_mean, smoother_cov, smoother_cross_cov = _kalman_smoother_update(
            next_smoother_mean,
            next_smoother_cov,
            filter_mean,
            filter_cov,
            process_cov,
            transition_matrix,
        )
        return (smoother_mean, smoother_cov), (
            (smoother_mean, smoother_cov),
            smoother_cross_cov,
        )

    # The terminal smoothed moments equal the filtered ones.
    terminal = (filtered_mean[-1], filtered_cov[-1])
    _, (smoother_mean, smoother_cov), smoother_cross_cov = _scan_with_boundary(
        _step,
        terminal,
        (filtered_mean, filtered_cov),
        (0, 0),
        filtered_mean.shape[0] - 1,
        terminal,
        reverse=True,
    )
    return smoother_mean, smoother_cov, smoother_cross_cov


@typed_jit
def rts_backward_scan_with_predictions(
    filtered_mean: ArrayLike,
    filtered_cov: ArrayLike,
    predicted_mean: ArrayLike,
    predicted_cov: ArrayLike,
    transition_matrix: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """RTS backward smoother that uses the filter's stored one-step predictions.

    :func:`rts_backward_scan` recomputes ``m_{t+1|t} = A m_{t|t}`` and
    ``P_{t+1|t} = A P_{t|t} A^T + Q``, which is only right when the forward
    pass used exactly that prediction. Filters that modify the prediction --
    a control input, a pseudo-observation / penalty downdate, or covariance
    inflation (e.g. the position decoder) -- must hand the modified
    ``(m_{t+1|t}, P_{t+1|t})`` to the backward pass, which then uses::

        J_t = P_{t|t} A^T P_{t+1|t}^{-1}
        m_{t|T} = m_{t|t} + J_t (m_{t+1|T} - m_{t+1|t})
        P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t^T

    With unmodified predictions this reduces exactly to
    :func:`rts_backward_scan`.

    Parameters
    ----------
    filtered_mean : ArrayLike, shape (n_time, n_cont_states)
    filtered_cov : ArrayLike, shape (n_time, n_cont_states, n_cont_states)
    predicted_mean : ArrayLike, shape (n_time, n_cont_states)
        ``predicted_mean[t]`` is the prediction of ``x_t`` the filter used
        at step ``t`` (entry 0 is not used).
    predicted_cov : ArrayLike, shape (n_time, n_cont_states, n_cont_states)
        Matching predicted covariances (entry 0 is not used).
    transition_matrix : ArrayLike, shape (n_cont_states, n_cont_states)

    Returns
    -------
    smoother_mean : jax.Array, shape (n_time, n_cont_states)
    smoother_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
    smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states)
        Lag-one cross-covariances ``J_t P_{t+1|T}``.
    """
    filtered_mean = jnp.asarray(filtered_mean)
    filtered_cov = jnp.asarray(filtered_cov)
    predicted_mean = jnp.asarray(predicted_mean)
    predicted_cov = jnp.asarray(predicted_cov)
    A = jnp.asarray(transition_matrix)

    def _step(
        carry: tuple[jax.Array, jax.Array],
        args: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array],
        tuple[tuple[jax.Array, jax.Array], jax.Array],
    ]:
        next_mean, next_cov = carry
        f_mean, f_cov, p_mean_next, p_cov_next = args
        gain = _gain_solve(p_cov_next, A @ f_cov).T
        mean = f_mean + gain @ (next_mean - p_mean_next)
        cov = symmetrize(f_cov + gain @ (next_cov - p_cov_next) @ gain.T)
        return (mean, cov), ((mean, cov), gain @ next_cov)

    terminal = (filtered_mean[-1], filtered_cov[-1])
    _, (smoother_mean, smoother_cov), smoother_cross_cov = _scan_with_boundary(
        _step,
        terminal,
        (filtered_mean, filtered_cov, predicted_mean, predicted_cov),
        (0, 0, 1, 1),
        filtered_mean.shape[0] - 1,
        terminal,
        reverse=True,
    )
    return smoother_mean, smoother_cov, smoother_cross_cov


@typed_jit
def _kalman_smoother_impl(
    init_mean: jax.Array,
    init_cov: jax.Array,
    obs: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Jitted RTS smoother. See :func:`kalman_smoother` for the public API."""
    filtered_mean, filtered_cov, marginal_log_likelihood = _kalman_filter_impl(
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )
    smoother_mean, smoother_cov, smoother_cross_cov = rts_backward_scan(
        filtered_mean,
        filtered_cov,
        transition_matrix,
        process_cov,
    )
    return smoother_mean, smoother_cov, smoother_cross_cov, marginal_log_likelihood


def kalman_smoother(
    init_mean: ArrayLike,
    init_cov: ArrayLike,
    obs: ArrayLike,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    measurement_matrix: ArrayLike,
    measurement_cov: ArrayLike,
    validate_inputs: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Applies the Rauch-Tung-Striebel (RTS) smoother.

    Parameters
    ----------
    init_mean : ArrayLike, shape (n_cont_states,)
        Initial state mean, $$ m_0 $$.
    init_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        Initial state covariance, $$ P_0 $$. Must be strictly positive
        definite.
    obs : ArrayLike, shape (n_time, n_obs_dim)
        Sequence of observations, $$ y_{1:T} $$.
    transition_matrix : ArrayLike, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \\Sigma $$.
    measurement_matrix : ArrayLike, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : ArrayLike, shape (n_obs_dim, n_obs_dim) or (n_time, n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$. A 2-D array is constant over
        time; a 3-D array with a leading time axis supplies a separate
        $$ R_t $$ per time step (used by the iterated-Laplace / IRLS smoother
        in :mod:`state_space_practice.temporal_rate_gp`). Every slice must be
        strictly positive definite.
    validate_inputs : bool, default=True
        If True, validate array shapes, finite observations, non-empty
        ``obs``, positive-definite ``init_cov`` / ``measurement_cov``, and
        positive-semidefinite ``process_cov``. Also warn when f32 + long T
        is at risk of losing PSD during the scan. Inner EM call sites that
        have already validated should pass ``False`` to skip the O(d^3)
        eigenvalue recomputation.

        The value checks need concrete inputs. Concrete arrays are checked
        even inside ``jax.jit`` (e.g. constants closed over by a jitted
        function). When any input is a JAX tracer (an argument of ``jax.jit``
        / ``jax.grad`` / ``jax.vmap``) only the shape checks run, and a
        non-positive-definite ``init_cov`` is reported at run time as a
        ``StateSpaceWarning`` naming its minimum eigenvalue instead of
        raising.

    Returns
    -------
    smoother_mean : jax.Array, shape (n_time, n_cont_states)
        Smoothed state means, $$ m_{1:T|T} $$.
    smoother_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
        Smoothed state covariances, $$ P_{1:T|T} $$.
    smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states)
        Smoothed cross-covariances, $$ P_{t, t+1|T} $$.
    marginal_log_likelihood : jax.Array
        Total log likelihood of the observations (scalar array).

    Raises
    ------
    ValueError
        If ``validate_inputs=True`` and any public input is malformed, non-finite,
        or has an invalid covariance. Under tracing only malformed shapes raise.

    Warns
    -----
    StateSpaceWarning
        If ``validate_inputs=True``, ``init_cov`` is traced, and it is not
        positive definite (emitted when the computation runs).
    """
    (
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    ) = _prepare_kalman_inputs(
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
        validate=validate_inputs,
        filter_name="kalman_smoother",
    )
    return _kalman_smoother_impl(
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )


class _SmootherElement(NamedTuple):
    """Associative scan element for parallel RTS smoother.

    Each element (E, g, L) encodes one backward smoother step so that
    the composition of elements via the associative operator yields the
    full smoothed posterior.

    Attributes
    ----------
    E : jax.Array, shape (..., D, D)
        Smoother gain (analogous to J_t in the sequential formulation).
    g : jax.Array, shape (..., D)
        Bias term: m_{t|t} - J_t @ m_{t+1|t}.
    L : jax.Array, shape (..., D, D)
        Residual covariance: P_{t|t} - J_t @ P_{t+1|t} @ J_t.T.

    Note: shape prefix ``...`` indicates these may be batched by
    ``jax.lax.associative_scan``.
    """

    E: jax.Array
    g: jax.Array
    L: jax.Array


def parallel_kalman_smoother(
    filtered_means: ArrayLike,
    filtered_covariances: ArrayLike,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """RTS smoother via parallel associative scan.

    Algebraically equivalent to the sequential ``kalman_smoother`` backward
    pass but runs in O(log T) parallel depth on GPU/TPU via
    ``jax.lax.associative_scan``.

    Parameters
    ----------
    filtered_means : ArrayLike, shape (T, D)
        Filtered state means from the forward Kalman filter.
    filtered_covariances : ArrayLike, shape (T, D, D)
        Filtered state covariances from the forward Kalman filter.
    transition_matrix : ArrayLike, shape (D, D) or (T-1, D, D)
        State transition matrix. If 2-D, the same matrix is used at every
        time step. If 3-D, ``transition_matrix[t]`` is used for the
        transition from time ``t`` to ``t+1``.
    process_cov : ArrayLike, shape (D, D) or (T-1, D, D)
        Process noise covariance. Broadcasting rules follow
        ``transition_matrix``.

    Returns
    -------
    smoothed_means : jax.Array, shape (T, D)
        Smoothed state means.
    smoothed_covariances : jax.Array, shape (T, D, D)
        Smoothed state covariances.
    cross_covariances : jax.Array, shape (T-1, D, D)
        Lag-one cross-covariances P_{t, t+1|T}.

    References
    ----------
    Särkkä, S. & García-Fernández, Á.F. (2021). Temporal parallelization
    of Bayesian smoothers. IEEE Trans. Automatic Control 66(1), 299-306.
    """
    filtered_means = jnp.asarray(filtered_means)
    filtered_covariances = jnp.asarray(filtered_covariances)
    transition_matrix = jnp.asarray(transition_matrix)
    process_cov = jnp.asarray(process_cov)

    if filtered_means.ndim != 2:
        raise ValueError(
            f"filtered_means must have shape (T, D), got {filtered_means.shape}."
        )
    T, D = filtered_means.shape
    if T == 0:
        raise ValueError("filtered_means must contain at least one time step.")
    if filtered_covariances.shape != (T, D, D):
        raise ValueError(
            "filtered_covariances must have shape "
            f"(T={T}, D={D}, D={D}), got {filtered_covariances.shape}."
        )

    for name, value in (
        ("transition_matrix", transition_matrix),
        ("process_cov", process_cov),
    ):
        if value.ndim == 2:
            if value.shape != (D, D):
                raise ValueError(
                    f"{name} must have shape ({D}, {D}) or "
                    f"({T - 1}, {D}, {D}), got {value.shape}."
                )
        elif value.ndim == 3:
            if value.shape != (T - 1, D, D):
                raise ValueError(
                    f"{name} with a leading time axis must have shape "
                    f"({T - 1}, {D}, {D}) because entry t maps time t to "
                    f"t+1; got {value.shape}."
                )
        else:
            raise ValueError(
                f"{name} must have shape ({D}, {D}) or ({T - 1}, {D}, {D}), "
                f"got {value.shape}."
            )

    if not contains_tracer(
        filtered_means, filtered_covariances, transition_matrix, process_cov
    ):
        for name, arr in (
            ("filtered_means", filtered_means),
            ("filtered_covariances", filtered_covariances),
            ("transition_matrix", transition_matrix),
            ("process_cov", process_cov),
        ):
            validate_finite_array(name, arr)

    return _parallel_kalman_smoother_impl(
        filtered_means, filtered_covariances, transition_matrix, process_cov
    )


@typed_jit
def _parallel_kalman_smoother_impl(
    filtered_means: jax.Array,
    filtered_covariances: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Jitted parallel RTS smoother on validated inputs.

    See :func:`parallel_kalman_smoother`. ``transition_matrix`` and
    ``process_cov`` are ``(D, D)`` or ``(T-1, D, D)``.
    """
    T, D = filtered_means.shape

    # Broadcast time-invariant parameters to (T-1, D, D)
    A = jnp.broadcast_to(transition_matrix, (T - 1, D, D))
    Q = jnp.broadcast_to(process_cov, (T - 1, D, D))

    # Build per-timestep smoother elements for t = 0, ..., T-2
    def _build_element(
        filt_mean: jax.Array, filt_cov: jax.Array, A_t: jax.Array, Q_t: jax.Array
    ) -> _SmootherElement:
        pred_cov = symmetrize(A_t @ filt_cov @ A_t.T + Q_t)
        pred_mean = A_t @ filt_mean
        J = _gain_solve(pred_cov, A_t @ filt_cov).T  # smoother gain
        g = filt_mean - J @ pred_mean
        L = symmetrize(filt_cov - J @ pred_cov @ J.T)
        return _SmootherElement(E=J, g=g, L=L)

    elements = jax.vmap(_build_element)(
        filtered_means[:-1], filtered_covariances[:-1], A, Q
    )

    # Terminal element for t = T-1: no dependence on future state. It takes
    # the element dtypes so the scan's concatenations agree (a bare
    # jnp.zeros would be float64 under x64 even for float32 inputs).
    terminal = _SmootherElement(
        E=jnp.zeros((D, D), dtype=elements.E.dtype),
        g=filtered_means[-1].astype(elements.g.dtype),
        L=filtered_covariances[-1].astype(elements.L.dtype),
    )

    # Concatenate elements with terminal at end
    all_elements = _SmootherElement(
        E=jnp.concatenate([elements.E, terminal.E[None]], axis=0),
        g=jnp.concatenate([elements.g, terminal.g[None]], axis=0),
        L=jnp.concatenate([elements.L, terminal.L[None]], axis=0),
    )

    # Associative operator vmapped over the batch dimension that
    # associative_scan introduces when combining sub-sequences.
    @jax.vmap
    def _operator(elem1: _SmootherElement, elem2: _SmootherElement) -> _SmootherElement:
        E1, g1, L1 = elem1
        E2, g2, L2 = elem2
        E = E2 @ E1
        g = E2 @ g1 + g2
        L = symmetrize(E2 @ L1 @ E2.T + L2)
        return _SmootherElement(E=E, g=g, L=L)

    scanned = jax.lax.associative_scan(_operator, all_elements, reverse=True)

    smoothed_means = scanned.g
    smoothed_covariances = scanned.L

    # Cross-covariances: P_{t,t+1|T} = J_t @ P_{t+1|T}
    smoother_gains = elements.E  # (T-1, D, D)
    cross_covariances = jnp.einsum(
        "tij,tjk->tik", smoother_gains, smoothed_covariances[1:]
    )

    return smoothed_means, smoothed_covariances, cross_covariances


def sum_of_outer_products(x: ArrayLike, y: ArrayLike) -> jax.Array:
    """Compute the sum of outer products between corresponding vectors.

    Computes $$ S = \\sum_{t=1}^T x_t y_t^T $$.

    Parameters
    ----------
    x : ArrayLike, shape (T, N)
        First sequence of vectors.
    y : ArrayLike, shape (T, M)
        Second sequence of vectors.

    Returns
    -------
    jax.Array, shape (N, M)
        The sum of outer products.

    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x.ndim != 2:
        raise ValueError(f"x must have shape (T, N), got {x.shape}.")
    if y.ndim != 2:
        raise ValueError(f"y must have shape (T, M), got {y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y must have the same leading time dimension; "
            f"got {x.shape[0]} and {y.shape[0]}."
        )
    return x.T @ y


class InitialStatePrior(NamedTuple):
    """Prior on the initial state and the dynamics the E-step ran with.

    The filters in this package start from ``x_0 ~ N(init_mean, init_cov)``
    and *predict* before the first update, so the first observation is of
    ``x_1 = A x_0 + w_1``. The exact EM update of the initial-state prior
    therefore needs the smoothed ``x_0`` -- one RTS step behind the smoother
    output, which starts at ``x_1`` -- and that step needs the parameters
    the E-step used. Pass this to the M-step functions
    (:func:`kalman_maximization_step`,
    :func:`~state_space_practice.point_process_kalman.dynamics_only_m_step`).

    Attributes
    ----------
    init_mean : jax.Array, shape (n_cont_states,)
        Prior mean of ``x_0`` used by the E-step, $$ m_0 $$.
    init_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Prior covariance of ``x_0`` used by the E-step, $$ P_0 $$.
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        Transition matrix used by the E-step, $$ A $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Process covariance used by the E-step, $$ \\Sigma $$.
    """

    init_mean: jax.Array
    init_cov: jax.Array
    transition_matrix: jax.Array
    process_cov: jax.Array


def smooth_initial_state(
    prior: InitialStatePrior,
    first_smoother_mean: ArrayLike,
    first_smoother_cov: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Smoothed moments of ``x_0`` from those of ``x_1``: one RTS step back.

    With ``P_{1|0} = A P_0 A^T + Q`` (the filter's first prediction) and gain
    ``J_0 = P_0 A^T P_{1|0}^{-1}``::

        m_{0|T} = m_0 + J_0 (m_{1|T} - A m_0)
        P_{0|T} = P_0 + J_0 (P_{1|T} - P_{1|0}) J_0^T

    This is the RTS backward step with the prior ``(m_0, P_0)`` in place of
    the (absent) filtered moments of ``x_0``, so it reuses
    :func:`_kalman_smoother_update`.

    Parameters
    ----------
    prior : InitialStatePrior
        Initial-state prior and dynamics the E-step ran with.
    first_smoother_mean : ArrayLike, shape (n_cont_states,)
        Smoothed mean of the first time step, $$ m_{1|T} $$.
    first_smoother_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        Smoothed covariance of the first time step, $$ P_{1|T} $$.

    Returns
    -------
    init_smoother_mean : jax.Array, shape (n_cont_states,)
        $$ m_{0|T} $$, the exact EM update of the initial mean.
    init_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states)
        $$ P_{0|T} $$, the exact EM update of the initial covariance.
    """
    mean, cov, _ = smooth_initial_state_with_cross_cov(
        prior, first_smoother_mean, first_smoother_cov
    )
    return mean, cov


def smooth_initial_state_with_cross_cov(
    prior: InitialStatePrior,
    first_smoother_mean: ArrayLike,
    first_smoother_cov: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Smoothed ``x_0`` moments and the lag-one cross-covariance ``Cov(x_0, x_1 | y)``.

    As :func:`smooth_initial_state`, plus::

        C_{0,1} = Cov(x_0, x_1 | y_{1:T}) = J_0 P_{1|T},
        J_0 = P_0 A^T P_{1|0}^{-1},   P_{1|0} = A P_0 A^T + Q,

    the same convention (earlier state first) as the smoother's
    ``smoother_cross_cov``. These are the statistics the ``x_0 -> x_1``
    transition contributes to the EM update of ``A`` and ``Q``.

    Parameters
    ----------
    prior : InitialStatePrior
        Initial-state prior and dynamics the E-step ran with.
    first_smoother_mean : ArrayLike, shape (n_cont_states,)
        $$ m_{1|T} $$.
    first_smoother_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        $$ P_{1|T} $$.

    Returns
    -------
    init_smoother_mean : jax.Array, shape (n_cont_states,)
        $$ m_{0|T} $$.
    init_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states)
        $$ P_{0|T} $$.
    init_cross_cov : jax.Array, shape (n_cont_states, n_cont_states)
        $$ C_{0,1} = Cov(x_0, x_1 | y_{1:T}) $$.
    """
    return _kalman_smoother_update(
        jnp.asarray(first_smoother_mean),
        jnp.asarray(first_smoother_cov),
        jnp.asarray(prior.init_mean),
        jnp.asarray(prior.init_cov),
        jnp.asarray(prior.process_cov),
        jnp.asarray(prior.transition_matrix),
    )


def measurement_cov_residual_form(
    obs: ArrayLike,
    smoother_mean: ArrayLike,
    sum_smoother_cov: ArrayLike,
    measurement_matrix: ArrayLike,
) -> jax.Array:
    """Centred (residual) M-step estimate of the measurement covariance.

    ``R = (1/T) sum_t [(y_t - H m_t)(y_t - H m_t)^T + H P_t H^T]``, a sum of
    PSD terms, so it is PSD by construction and free of the cancellation in
    the equivalent ``(sum y y^T - H sum m y^T) / T`` when ``|y| >> sqrt(R)``.

    Parameters
    ----------
    obs : ArrayLike, shape (n_time, n_obs_dim)
    smoother_mean : ArrayLike, shape (n_time, n_cont_states)
    sum_smoother_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        $$ \\sum_t P_{t|T} $$.
    measurement_matrix : ArrayLike, shape (n_obs_dim, n_cont_states)

    Returns
    -------
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
    """
    obs = jnp.asarray(obs)
    smoother_mean = jnp.asarray(smoother_mean)
    measurement_matrix = jnp.asarray(measurement_matrix)
    n_time = obs.shape[0]
    residual = obs - smoother_mean @ measurement_matrix.T
    return (
        symmetrize(
            residual.T @ residual
            + measurement_matrix @ sum_smoother_cov @ measurement_matrix.T
        )
        / n_time
    )


def process_cov_residual_form(
    smoother_mean: ArrayLike,
    sum_next_cov: ArrayLike,
    sum_prev_cov: ArrayLike,
    sum_cross_cov: ArrayLike,
    transition_matrix: ArrayLike,
) -> jax.Array:
    """Centred (residual) M-step estimate of the process covariance.

    ``Q = sum_t E[(x_{t+1} - A x_t)(x_{t+1} - A x_t)^T]`` divided by the number
    of transitions (``n_time - 1``, ``n_time = smoother_mean.shape[0]``), where
    the sum is::

        sum_t [(m_{t+1} - A m_t)(.)^T + P_{t+1} - A C_t - C_t^T A^T + A P_t A^T]

    with ``C_t = Cov(x_t, x_{t+1} | y)`` (the smoother's cross-covariance
    convention). The mean part is a Gram matrix and the covariance part is
    the posterior covariance of ``x_{t+1} - A x_t``, so the estimate is PSD
    by construction and avoids the ``gamma2 - A beta^T`` cancellation. It is
    the M-step optimum for *any* given ``A`` (solved or held fixed).

    Parameters
    ----------
    smoother_mean : ArrayLike, shape (n_time, n_cont_states)
    sum_next_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        $$ \\sum_{t=2}^T P_{t|T} $$.
    sum_prev_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        $$ \\sum_{t=1}^{T-1} P_{t|T} $$.
    sum_cross_cov : ArrayLike, shape (n_cont_states, n_cont_states)
        $$ \\sum_{t=1}^{T-1} C_t $$.
    transition_matrix : ArrayLike, shape (n_cont_states, n_cont_states)

    Returns
    -------
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
    """
    smoother_mean = jnp.asarray(smoother_mean)
    n_time = smoother_mean.shape[0]
    A = jnp.asarray(transition_matrix)
    residual = smoother_mean[1:] - smoother_mean[:-1] @ A.T
    A_cross = A @ sum_cross_cov
    return symmetrize(
        residual.T @ residual
        + sum_next_cov
        - A_cross
        - A_cross.T
        + A @ sum_prev_cov @ A.T
    ) / (n_time - 1)


def kalman_maximization_step(
    obs: ArrayLike,
    smoother_mean: ArrayLike,
    smoother_cov: ArrayLike,
    smoother_cross_cov: ArrayLike,
    initial_state_prior: InitialStatePrior | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Performs the Maximization (M) step of the EM algorithm for Kalman filters.

    Updates the model parameters based on the expected sufficient statistics
    derived from the E-step (Kalman smoother).

    Parameters
    ----------
    obs : ArrayLike, shape (n_time, n_obs_dim)
        Observations, $$ y_{1:T} $$.
    smoother_mean : ArrayLike, shape (n_time, n_cont_states)
        Smoothed means, $$ m_{1:T|T} $$.
    smoother_cov : ArrayLike, shape (n_time, n_cont_states, n_cont_states)
        Smoothed covariances, $$ P_{1:T|T} $$.
    smoother_cross_cov : ArrayLike, shape (n_time - 1, n_cont_states, n_cont_states)
        Smoothed cross-covariances, $$ P_{t, t+1|T} $$.
    initial_state_prior : InitialStatePrior or None, optional
        The initial-state prior and dynamics the E-step ran with. The filter
        predicts before its first update, so ``init_mean`` / ``init_cov``
        are the prior of ``x_0``; with these parameters the M-step recovers
        the smoothed ``x_0`` (:func:`smooth_initial_state_with_cross_cov`)
        and is the exact maximiser of the expected complete-data
        log-likelihood of the filter's model (see Notes). Pass them for a
        monotone EM.

        .. deprecated:: 0.1.0
            Passing None is deprecated and will be removed in version 0.2.0
            (``initial_state_prior`` will become required). With None the
            returned initial moments are the smoothed moments of ``x_1`` and
            ``A`` / ``Sigma`` use only the ``T - 1`` transitions within
            ``x_{1:T}``: the exact M-step of a model whose prior sits on
            ``x_1``, which is not the filter's model and can decrease the
            log-likelihood (e.g. with a contractive ``A``).

    Returns
    -------
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        Updated transition matrix, $$ A $$.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Updated measurement matrix, $$ H $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Updated process covariance, $$ \\Sigma $$.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Updated measurement covariance, $$ R $$.
    init_mean : jax.Array, shape (n_cont_states,)
        Updated initial mean: $$ m_{0|T} $$ with ``initial_state_prior``,
        else $$ m_{1|T} $$.
    init_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Updated initial covariance: $$ P_{0|T} $$ with
        ``initial_state_prior``, else $$ P_{1|T} $$.

    Notes
    -----
    With ``initial_state_prior`` the transition statistics include the
    ``x_0 -> x_1`` transition, so ``A`` and ``Sigma`` are estimated from all
    ``T`` transitions of the filter's model. With ``m_{0|T}``, ``P_{0|T}``
    from the RTS step back to ``x_0`` and ``C_{t,t+1} = Cov(x_t, x_{t+1} | y)``
    (``C_{0,1} = J_0 P_{1|T}``, ``J_0 = P_0 A^T (A P_0 A^T + Sigma)^{-1}``)::

        S_t    = P_{t|T} + m_{t|T} m_{t|T}^T                   t = 0..T
        gamma1 = sum_{t=0}^{T-1} S_t
        beta   = sum_{t=0}^{T-1} (C_{t,t+1} + m_{t|T} m_{t+1|T}^T)^T
        A      = beta gamma1^{-1}
        Sigma  = (1/T) sum_{t=1}^{T} E[(x_t - A x_{t-1})(x_t - A x_{t-1})^T]

    (the last in the centred residual form of :func:`process_cov_residual_form`
    over ``x_{0:T}``). ``H`` and ``R`` use the ``T`` observations as before.
    Without ``initial_state_prior`` the sums run over ``x_{1:T}`` only
    (``T - 1`` transitions, ``Sigma`` divided by ``T - 1``).

    ``R`` and ``Sigma`` use the centred residual forms
    (:func:`measurement_cov_residual_form`, :func:`process_cov_residual_form`),
    which are PSD by construction; they equal the classical
    ``(alpha - H delta^T) / T`` and ``(gamma2 - A beta^T) / n_transitions``
    at the solved ``H`` / ``A`` up to roundoff. Eigenvalues are then floored
    at a scale-relative level
    (:func:`~state_space_practice.utils.project_psd_relative`), which logs a
    warning when it changes the estimate.

    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """
    if initial_state_prior is None:
        warnings.warn(
            "kalman_maximization_step(initial_state_prior=None) uses the x_1 "
            "prior, which is not the filter's model and can decrease the "
            "log-likelihood. Pass initial_state_prior=InitialStatePrior("
            "init_mean, init_cov, transition_matrix, process_cov) with the "
            "E-step's parameters. initial_state_prior=None will be removed in "
            "version 0.2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _kalman_maximization_step(
        jnp.asarray(obs),
        jnp.asarray(smoother_mean),
        jnp.asarray(smoother_cov),
        jnp.asarray(smoother_cross_cov),
        initial_state_prior,
    )


@typed_jit
def _kalman_maximization_step(
    obs: jax.Array,
    smoother_mean: jax.Array,
    smoother_cov: jax.Array,
    smoother_cross_cov: jax.Array,
    initial_state_prior: InitialStatePrior | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Jitted implementation of :func:`kalman_maximization_step`."""

    n_time: int = obs.shape[0]
    if n_time < 2:
        raise ValueError(
            "kalman_maximization_step requires at least 2 time steps to "
            "estimate transition dynamics."
        )

    # Observation statistics over x_{1:T}.
    sum_cov = jnp.sum(smoother_cov, axis=0)
    gamma = sum_cov + sum_of_outer_products(smoother_mean, smoother_mean)
    delta = sum_of_outer_products(obs, smoother_mean)

    # Measurement matrix and covariance
    measurement_matrix = _gain_solve(gamma, delta.T).T
    measurement_cov = project_psd_relative(
        measurement_cov_residual_form(obs, smoother_mean, sum_cov, measurement_matrix),
        name="kalman_maximization_step measurement_cov",
    )

    # Transition statistics: over x_{0:T} (all T transitions of the filter's
    # model) when the E-step's initial-state prior is known, else over x_{1:T}.
    if initial_state_prior is None:
        init_mean = smoother_mean[0]
        init_cov = smoother_cov[0]
        trans_mean = smoother_mean
        trans_cov = smoother_cov
        trans_cross_cov = smoother_cross_cov
    else:
        init_mean, init_cov, init_cross_cov = smooth_initial_state_with_cross_cov(
            initial_state_prior, smoother_mean[0], smoother_cov[0]
        )
        trans_mean = jnp.concatenate((init_mean[None], smoother_mean))
        trans_cov = jnp.concatenate((init_cov[None], smoother_cov))
        trans_cross_cov = jnp.concatenate((init_cross_cov[None], smoother_cross_cov))

    sum_trans_cov = jnp.sum(trans_cov, axis=0)
    sum_cross_cov = jnp.sum(trans_cross_cov, axis=0)
    gamma1 = (sum_trans_cov - trans_cov[-1]) + sum_of_outer_products(
        trans_mean[:-1], trans_mean[:-1]
    )
    beta = (sum_cross_cov + sum_of_outer_products(trans_mean[:-1], trans_mean[1:])).T

    # Transition matrix
    transition_matrix = _gain_solve(gamma1, beta.T).T

    # Process covariance
    process_cov = project_psd_relative(
        process_cov_residual_form(
            trans_mean,
            sum_next_cov=sum_trans_cov - trans_cov[0],
            sum_prev_cov=sum_trans_cov - trans_cov[-1],
            sum_cross_cov=sum_cross_cov,
            transition_matrix=transition_matrix,
        ),
        name="kalman_maximization_step process_cov",
    )

    return (
        transition_matrix,
        measurement_matrix,
        process_cov,
        measurement_cov,
        init_mean,
        init_cov,
    )


# Public names for the single-step updates (used by the notebooks); the
# underscore names remain the implementation.
kalman_filter_update = _kalman_filter_update
kalman_smoother_update = _kalman_smoother_update

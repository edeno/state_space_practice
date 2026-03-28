"""Shared fixtures and Hypothesis strategies for state space model tests."""

from typing import Tuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jax import Array, random

# Configure Hypothesis for JAX compatibility
# CI profile: More thorough testing for numerical algorithms
settings.register_profile("ci", max_examples=100, deadline=None, derandomize=True)
# Dev profile: Faster iteration during development
settings.register_profile("dev", max_examples=10, deadline=None)
settings.load_profile("dev")


# --- Hypothesis Strategies for State Space Models ---


@st.composite
def positive_definite_matrices(
    draw: st.DrawFn,
    n: int,
    min_eigenvalue: float = 0.01,
    max_eigenvalue: float = 10.0,
) -> np.ndarray:
    """Generate a random positive definite matrix.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    n : int
        Matrix dimension.
    min_eigenvalue : float
        Minimum eigenvalue for numerical stability.
    max_eigenvalue : float
        Maximum eigenvalue.

    Returns
    -------
    np.ndarray
        A positive definite matrix of shape (n, n).
    """
    # Generate eigenvalues with limited condition number
    # Limit max/min ratio to avoid ill-conditioned matrices
    eigenvalues = draw(
        arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(
                min_value=min_eigenvalue,
                max_value=max_eigenvalue,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
        )
    )
    # Ensure condition number is reasonable (max/min < 1000)
    eigenvalues = np.clip(eigenvalues, a_min=max(eigenvalues) / 1000, a_max=None)

    # Generate random orthogonal matrix via QR decomposition
    random_matrix = draw(
        arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    q, _ = np.linalg.qr(random_matrix)

    # Construct PD matrix: Q @ diag(eigenvalues) @ Q.T
    result: np.ndarray = q @ np.diag(eigenvalues) @ q.T
    return result


@st.composite
def stable_transition_matrices(
    draw: st.DrawFn,
    n: int,
    max_spectral_radius: float = 0.99,
) -> np.ndarray:
    """Generate a stable transition matrix (spectral radius < 1).

    Uses eigenvalue decomposition to guarantee stability.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    n : int
        Matrix dimension.
    max_spectral_radius : float
        Maximum spectral radius for stability.

    Returns
    -------
    np.ndarray
        A stable transition matrix of shape (n, n).
    """
    # Generate eigenvalues with magnitude < max_spectral_radius
    eigenvalue_magnitudes = draw(
        arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(
                min_value=0.1,
                max_value=max_spectral_radius,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
        )
    )

    # Generate random orthogonal matrix via QR
    random_matrix = draw(
        arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
        )
    )

    A: np.ndarray
    try:
        q, _ = np.linalg.qr(random_matrix)
        # Construct A = Q @ diag(eigenvalues) @ Q.T
        # This ensures eigenvalues are exactly what we specify
        A = q @ np.diag(eigenvalue_magnitudes) @ q.T
    except np.linalg.LinAlgError:
        # Fallback: diagonal matrix
        A = np.diag(eigenvalue_magnitudes)

    return A


@st.composite
def stochastic_matrices(
    draw: st.DrawFn,
    n: int,
    min_prob: float = 0.01,
) -> np.ndarray:
    """Generate a row-stochastic matrix (rows sum to 1).

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    n : int
        Matrix dimension.
    min_prob : float
        Minimum probability for each entry.

    Returns
    -------
    np.ndarray
        A stochastic matrix of shape (n, n).
    """
    # Generate positive entries
    matrix = draw(
        arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(
                min_value=min_prob,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    # Normalize rows to sum to 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    result: np.ndarray = matrix / row_sums
    return result


@st.composite
def probability_vectors(
    draw: st.DrawFn,
    n: int,
    min_prob: float = 0.01,
) -> np.ndarray:
    """Generate a probability vector (sums to 1).

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    n : int
        Vector dimension.
    min_prob : float
        Minimum probability for each entry.

    Returns
    -------
    np.ndarray
        A probability vector of shape (n,).
    """
    # Generate positive entries
    vector = draw(
        arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(
                min_value=min_prob,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    # Normalize to sum to 1
    result: np.ndarray = vector / vector.sum()
    return result


@st.composite
def kalman_model_params(
    draw: st.DrawFn,
    n_cont_states: int | None = None,
    n_obs_dim: int | None = None,
    max_cont_states: int = 4,
    max_obs_dim: int = 4,
) -> dict:
    """Generate valid Kalman filter model parameters.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    n_cont_states : int | None
        Number of continuous states. If None, randomly chosen.
    n_obs_dim : int | None
        Observation dimension. If None, randomly chosen.
    max_cont_states : int
        Maximum number of continuous states.
    max_obs_dim : int
        Maximum observation dimension.

    Returns
    -------
    dict
        Dictionary with keys: init_mean, init_cov, A, Q, H, R, n_cont_states, n_obs_dim
    """
    if n_cont_states is None:
        n_cont_states = draw(st.integers(min_value=1, max_value=max_cont_states))
    if n_obs_dim is None:
        n_obs_dim = draw(st.integers(min_value=1, max_value=max_obs_dim))

    init_mean = draw(
        arrays(
            dtype=np.float64,
            shape=(n_cont_states,),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    init_cov = draw(positive_definite_matrices(n_cont_states))
    A = draw(stable_transition_matrices(n_cont_states))
    Q = draw(positive_definite_matrices(n_cont_states, min_eigenvalue=0.001))

    # Observation matrix
    H = draw(
        arrays(
            dtype=np.float64,
            shape=(n_obs_dim, n_cont_states),
            elements=st.floats(
                min_value=-2.0,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    R = draw(positive_definite_matrices(n_obs_dim))

    return {
        "init_mean": init_mean,
        "init_cov": init_cov,
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
        "n_cont_states": n_cont_states,
        "n_obs_dim": n_obs_dim,
    }


@st.composite
def switching_kalman_model_params(
    draw: st.DrawFn,
    n_cont_states: int | None = None,
    n_obs_dim: int | None = None,
    n_discrete_states: int | None = None,
    max_cont_states: int = 3,
    max_obs_dim: int = 3,
    max_discrete_states: int = 3,
) -> dict:
    """Generate valid switching Kalman filter model parameters.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    n_cont_states : int | None
        Number of continuous states. If None, randomly chosen.
    n_obs_dim : int | None
        Observation dimension. If None, randomly chosen.
    n_discrete_states : int | None
        Number of discrete states. If None, randomly chosen.
    max_cont_states : int
        Maximum number of continuous states.
    max_obs_dim : int
        Maximum observation dimension.
    max_discrete_states : int
        Maximum number of discrete states.

    Returns
    -------
    dict
        Dictionary with all SKF parameters.
    """
    if n_cont_states is None:
        n_cont_states = draw(st.integers(min_value=1, max_value=max_cont_states))
    if n_obs_dim is None:
        n_obs_dim = draw(st.integers(min_value=1, max_value=max_obs_dim))
    if n_discrete_states is None:
        n_discrete_states = draw(st.integers(min_value=1, max_value=max_discrete_states))

    # Initial continuous state parameters per discrete state
    init_means = []
    init_covs = []
    for _ in range(n_discrete_states):
        init_means.append(
            draw(
                arrays(
                    dtype=np.float64,
                    shape=(n_cont_states,),
                    elements=st.floats(
                        min_value=-10.0,
                        max_value=10.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                )
            )
        )
        init_covs.append(draw(positive_definite_matrices(n_cont_states)))

    init_mean = np.stack(init_means, axis=-1)  # (n_cont_states, n_discrete_states)
    init_cov = np.stack(init_covs, axis=-1)  # (n_cont_states, n_cont_states, n_discrete_states)

    # Initial discrete state probability
    init_prob = draw(probability_vectors(n_discrete_states))

    # Transition matrices per discrete state
    As = []
    Qs = []
    Hs = []
    Rs = []
    for _ in range(n_discrete_states):
        As.append(draw(stable_transition_matrices(n_cont_states)))
        Qs.append(draw(positive_definite_matrices(n_cont_states, min_eigenvalue=0.001)))
        Hs.append(
            draw(
                arrays(
                    dtype=np.float64,
                    shape=(n_obs_dim, n_cont_states),
                    elements=st.floats(
                        min_value=-2.0,
                        max_value=2.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                )
            )
        )
        Rs.append(draw(positive_definite_matrices(n_obs_dim)))

    A = np.stack(As, axis=-1)  # (n_cont_states, n_cont_states, n_discrete_states)
    Q = np.stack(Qs, axis=-1)
    H = np.stack(Hs, axis=-1)  # (n_obs_dim, n_cont_states, n_discrete_states)
    R = np.stack(Rs, axis=-1)  # (n_obs_dim, n_obs_dim, n_discrete_states)

    # Discrete state transition matrix
    Z = draw(stochastic_matrices(n_discrete_states))

    return {
        "init_mean": init_mean,
        "init_cov": init_cov,
        "init_prob": init_prob,
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
        "Z": Z,
        "n_cont_states": n_cont_states,
        "n_obs_dim": n_obs_dim,
        "n_discrete_states": n_discrete_states,
    }


@st.composite
def gaussian_mixture_params(
    draw: st.DrawFn,
    n_dims: int | None = None,
    n_components: int | None = None,
    max_dims: int = 4,
    max_components: int = 5,
) -> dict:
    """Generate valid Gaussian mixture parameters.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    n_dims : int | None
        Dimension of each Gaussian.
    n_components : int | None
        Number of mixture components.
    max_dims : int
        Maximum dimension.
    max_components : int
        Maximum number of components.

    Returns
    -------
    dict
        Dictionary with keys: means, covs, weights, n_dims, n_components
    """
    if n_dims is None:
        n_dims = draw(st.integers(min_value=1, max_value=max_dims))
    if n_components is None:
        n_components = draw(st.integers(min_value=2, max_value=max_components))

    means = []
    covs = []
    for _ in range(n_components):
        means.append(
            draw(
                arrays(
                    dtype=np.float64,
                    shape=(n_dims,),
                    elements=st.floats(
                        min_value=-10.0,
                        max_value=10.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                )
            )
        )
        covs.append(draw(positive_definite_matrices(n_dims)))

    means_arr = np.stack(means, axis=-1)  # (n_dims, n_components)
    covs_arr = np.stack(covs, axis=-1)  # (n_dims, n_dims, n_components)
    weights = draw(probability_vectors(n_components))

    return {
        "means": means_arr,
        "covs": covs_arr,
        "weights": weights,
        "n_dims": n_dims,
        "n_components": n_components,
    }


def to_jax(*arrays: np.ndarray) -> tuple[jax.Array, ...]:
    """Convert numpy arrays to JAX arrays."""
    return tuple(jnp.array(arr) for arr in arrays)


# --- Shared Fixtures ---


@pytest.fixture(scope="session")
def simple_1d_model() -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Provides parameters and data for a simple 1D random walk model.

    Used by both test_kalman.py and test_switching_kalman.py.
    """
    key = random.PRNGKey(0)
    n_time = 50
    n_cont_states = 1
    n_obs_dim = 1

    init_mean = jnp.array([0.0])
    init_cov = jnp.eye(n_cont_states) * 1.0
    transition_matrix = jnp.eye(n_cont_states) * 1.0
    process_cov = jnp.eye(n_cont_states) * 0.1
    measurement_matrix = jnp.eye(n_obs_dim, n_cont_states)
    measurement_cov = jnp.eye(n_obs_dim) * 1.0

    true_states = [init_mean]
    obs = []
    k1, k2 = random.split(key)

    for t in range(1, n_time):
        w = random.multivariate_normal(k1, jnp.zeros(n_cont_states), process_cov)
        true_states.append(transition_matrix @ true_states[-1] + w)
        k1, _ = random.split(k1)

    for t in range(n_time):
        v = random.multivariate_normal(k2, jnp.zeros(n_obs_dim), measurement_cov)
        obs.append(measurement_matrix @ true_states[t] + v)
        k2, _ = random.split(k2)

    return (
        init_mean,
        init_cov,
        jnp.array(obs),
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )

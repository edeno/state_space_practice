# ruff: noqa: E402
"""Shared fixtures and Hypothesis strategies for state space model tests."""

import ast
import functools
import inspect
import os
import textwrap

import jax

jax.config.update("jax_enable_x64", True)

# Opt-in persistent XLA compilation cache. Much of the suite's wall time is
# compiling the same jitted filters; setting SSP_JAX_CACHE_DIR to a directory
# stores compiled executables there so that a *later, separate* pytest run
# reuses them instead of recompiling. It is off by default because the cache
# is not safe for concurrent writers: two pytest runs sharing one directory
# can leave truncated entries, which JAX reports as "Error reading persistent
# compilation cache entry" (a warning that ``filterwarnings = error`` turns
# into a test failure). Give concurrent runs separate directories, and clear
# the directory if it grows too large -- every compiled entry is kept.
_cache_dir = os.environ.get("SSP_JAX_CACHE_DIR", "")
if _cache_dir:
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
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
# Select with HYPOTHESIS_PROFILE=ci (CI sets this); defaults to "dev".
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# --- Automatic ``slow`` marking ---------------------------------------------
#
# Any test that runs EM, SGD or a full fit must be marked slow so the fast
# suite (``-m "not slow"``) stays quick. Rather than rely on every author
# remembering the decorator, a test is marked slow at collection time when
#
# 1. its own source, or the source of any fixture it (transitively) requests,
#    calls ``.fit(``, ``.fit_sgd(`` or ``run_em(`` outside a
#    ``with pytest.raises(...)`` block (input-validation tests that expect
#    ``fit`` to raise before doing any work stay in the fast suite), or
# 2. its node id contains an entry of ``_SLOW_TEST_REGISTRY`` (for tests whose
#    fitting happens in helper functions, which the source scan can't see).
#
# Explicit ``@pytest.mark.slow`` keeps working as before.

_FIT_METHODS = frozenset({"fit", "fit_sgd"})
_FIT_FUNCTIONS = frozenset({"run_em"})

# Node-id substrings (``file.py::Class`` or ``file.py::Class::test``) for tests
# that run EM / optimizers through helpers the source scan can't see. Chosen
# from ``--durations`` of the fast suite.
_SLOW_TEST_REGISTRY: tuple[str, ...] = (
    # EM driven through conftest.assert_em_rolls_back_on_ll_decrease
    "test_smith_learning_algorithm.py::TestSmithEMRollback::",
    "test_oscillator_models.py::TestOscillatorEMRollback::",
    # hand-rolled EM loops / optimizers
    "test_oscillator_models.py::TestOscillatorPaperStructure::"
    "test_em_pools_observation_covariance_across_states",
    "test_switching_kalman.py::TestSwitchingEMMonotonicity::",
    "test_switching_kalman.py::test_joint_dim_optimizer_improves_total_q_and_",
    "test_switching_point_process.py::TestMilestone8EndToEnd::",
    "test_switching_point_process.py::TestSecondOrderClippedWarmStart::"
    "test_single_neuron_converges_from_clipped_warm_start",
    "test_switching_point_process.py::TestSecondOrderClippedWarmStart::"
    "test_mixture_converges_from_clipped_warm_start",
)


def _is_fit_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in _FIT_METHODS or func.attr in _FIT_FUNCTIONS
    return isinstance(func, ast.Name) and func.id in _FIT_FUNCTIONS


def _is_pytest_raises_block(node: ast.AST) -> bool:
    if not isinstance(node, ast.With):
        return False
    for item in node.items:
        expr = item.context_expr
        if isinstance(expr, ast.Call):
            expr = expr.func
        if isinstance(expr, ast.Attribute) and expr.attr == "raises":
            return True
    return False


def _calls_fit_outside_raises(node: ast.AST) -> bool:
    if _is_pytest_raises_block(node):
        return False
    if _is_fit_call(node):
        return True
    return any(_calls_fit_outside_raises(child) for child in ast.iter_child_nodes(node))


@functools.cache
def _function_runs_fit(function) -> bool:
    try:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return False
    return _calls_fit_outside_raises(tree)


def _fixture_functions(item):
    """Functions of the fixtures ``item`` requests (closure), best effort."""
    fixtureinfo = getattr(item, "_fixtureinfo", None)
    if fixtureinfo is None:
        return []
    functions = []
    for name in fixtureinfo.names_closure:
        fixturedefs = fixtureinfo.name2fixturedefs.get(name) or ()
        functions.extend(getattr(fd, "func", None) for fd in fixturedefs)
    return [f for f in functions if f is not None]


def _item_runs_fit(item) -> bool:
    function = getattr(item, "function", None)
    if function is not None and _function_runs_fit(function):
        return True
    return any(_function_runs_fit(f) for f in _fixture_functions(item))


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("slow") is not None:
            continue
        in_registry = any(entry in item.nodeid for entry in _SLOW_TEST_REGISTRY)
        if in_registry or _item_runs_fit(item):
            item.add_marker(pytest.mark.slow)


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
        n_discrete_states = draw(
            st.integers(min_value=1, max_value=max_discrete_states)
        )

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
    init_cov = np.stack(
        init_covs, axis=-1
    )  # (n_cont_states, n_cont_states, n_discrete_states)

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
def simple_1d_model() -> tuple[Array, Array, Array, Array, Array, Array, Array]:
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

    for _t in range(1, n_time):
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


def assert_em_rolls_back_on_ll_decrease(
    model,
    fit_args,
    caplog,
    *,
    fit_kwargs=None,
    ll_sequence=(0.0, -1e6, -1e6),
):
    """Inject a synthetic LL drop into ``model._e_step`` and assert rollback.

    Used by the per-EM-loop rollback regression tests
    (TestEMRollbackOnDecrease, TestOscillatorEMRollback,
    TestSmithEMRollback). Replaces ``model._e_step`` with a wrapper
    that runs the real E-step (so smoother/filter state is still
    populated) and then returns successive values from
    ``ll_sequence`` instead of the real LL.

    Asserts a warning containing "rolling back" fires and that the
    bad LL was popped from the returned ``log_likelihoods``.
    """
    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.setdefault("max_iter", 5)
    original_e_step = model._e_step
    seq = iter(ll_sequence)

    def fake_e_step(*args, **kwargs):
        real = original_e_step(*args, **kwargs)
        try:
            val = next(seq)
        except StopIteration:
            return real
        # Match the real return's runtime type so downstream
        # ``float(...)`` / arithmetic in the EM loop sees the same
        # interface. JAX scalar arrays don't have a usable
        # ``type(real)(value)`` constructor; route via jnp.asarray.
        if hasattr(real, "dtype"):
            return jnp.asarray(val, dtype=real.dtype)
        return float(val)

    model._e_step = fake_e_step

    with caplog.at_level("WARNING"):
        lls = model.fit(*fit_args, **fit_kwargs)

    assert any("rolling back" in r.message.lower() for r in caplog.records), (
        f"Expected a rollback warning, got: {[r.message for r in caplog.records]}"
    )
    assert lls[0] == ll_sequence[0], (
        f"First LL should be the good one ({ll_sequence[0]}), got {lls}"
    )
    return lls


# --- Spike-field coupling validation fixtures ---


@pytest.fixture
def make_coupling_posterior():
    """Factory building a ``CouplingPosterior`` from explicit mean/var arrays.

    Imported lazily so collection does not depend on the coupling module existing.
    """

    def _make(
        beta_real_mean: npt.ArrayLike,
        beta_imag_mean: npt.ArrayLike,
        beta_real_var: npt.ArrayLike,
        beta_imag_var: npt.ArrayLike,
        samples: npt.ArrayLike | None = None,
        beta_real_imag_cov: npt.ArrayLike | None = None,
    ):
        from state_space_practice.coupling_validation import CouplingPosterior

        return CouplingPosterior(
            beta_real_mean=np.asarray(beta_real_mean, dtype=float),
            beta_imag_mean=np.asarray(beta_imag_mean, dtype=float),
            beta_real_var=np.asarray(beta_real_var, dtype=float),
            beta_imag_var=np.asarray(beta_imag_var, dtype=float),
            samples=None if samples is None else np.asarray(samples),
            beta_real_imag_cov=(
                None
                if beta_real_imag_cov is None
                else np.asarray(beta_real_imag_cov, dtype=float)
            ),
        )

    return _make


@pytest.fixture
def make_ground_truth():
    """Factory returning ``(beta_real_true, beta_imag_true, coupling_mask)`` arrays."""

    def _make(
        beta_real_true: npt.ArrayLike,
        beta_imag_true: npt.ArrayLike,
        coupling_mask: npt.ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(beta_real_true, dtype=float),
            np.asarray(beta_imag_true, dtype=float),
            np.asarray(coupling_mask, dtype=bool),
        )

    return _make


@pytest.fixture(scope="module")
def coupling_params_small():
    """A small Bernoulli-logistic coupling model: 3 neurons, 2 bands.

    Band 0 is couplable (neurons get known magnitude/phase), band 1 is a
    false-positive control (all-zero coupling). Stationary latent variance is 1
    per component (``process_noise_var / (1 - decay**2) = 1``), so coupling of
    magnitude 2 modulates the logit strongly. Base rate is ``sigmoid(baseline)``
    = 0.05 per 1 ms bin.
    """
    from state_space_practice.coupling_model import CouplingModelParams

    baseline_logit = float(np.log(0.05 / 0.95))
    return CouplingModelParams(
        osc_frequencies=jnp.array([6.0, 10.0]),
        osc_decay=jnp.array([0.99, 0.99]),
        process_noise_var=jnp.array([1.0 - 0.99**2, 1.0 - 0.99**2]),
        # neuron 0 -> band0 phase 0; neuron 1 -> band0 phase pi/2; neuron 2 -> band0 phase pi
        beta_real=jnp.array([[2.0, 0.0], [0.0, 0.0], [-2.0, 0.0]]),
        beta_imag=jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
        baseline=jnp.full((3,), baseline_logit),
        dt=1e-3,
    )


@pytest.fixture(scope="module")
def simulated_coupling_small(coupling_params_small):
    """A single simulation of ``coupling_params_small`` (5000 bins, seed 0)."""
    from state_space_practice.simulate_coupling import simulate_coupling

    return simulate_coupling(coupling_params_small, n_time=5000, seed=0)

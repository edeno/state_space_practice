"""Point-process Kalman filter and smoother for neural spike data.

This module implements state-space models with point-process (spike) observations
using the Laplace-EKF approach from Eden & Brown (2004).

The model is:
    x_k = A @ x_{k-1} + w_k,  w_k ~ N(0, Q)
    y_{n,k} ~ Poisson(exp(log_intensity_func(Z_k, x_k)[n]) * dt)

where x_k is the latent state, y_{n,k} is the spike count for neuron n at time k,
and log_intensity_func returns log firing rates for all neurons.

Multi-Neuron Support
--------------------
The filter supports multiple neurons sharing a common latent state:

- spike_indicator: (n_time, n_neurons) - spike counts for each neuron
- log_conditional_intensity(Z_k, x_k) returns (n_neurons,) log-intensities

For backwards compatibility, single-neuron inputs are automatically promoted:
- spike_indicator: (n_time,) is treated as (n_time, 1)
- scalar log-intensity output is wrapped to (1,)

References
----------
[1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
    Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
    Neural Computation 16, 971-998.
"""

from __future__ import annotations

import functools
import logging
import operator
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import DTypeLike

from state_space_practice.em_driver import (
    clear_attributes,
    restore_attributes,
    run_em,
    snapshot_attributes,
)
from state_space_practice.exceptions import NotFittedError
from state_space_practice.fitted_state import FittedAttribute, is_set
from state_space_practice.kalman import (
    InitialStatePrior,
    _kalman_smoother_update,
    process_cov_residual_form,
    smooth_initial_state_with_cross_cov,
    sum_of_outer_products,
)
from state_space_practice.parameter_transforms import (
    PSD_MATRIX,
    UNCONSTRAINED,
)
from state_space_practice.sgd_fitting import SGDFittableMixin, SGDParams, SGDParamSpec
from state_space_practice.utils import (
    _validate_filter_numerics as _validate_filter_numerics_impl,
)
from state_space_practice.utils import (
    contains_tracer,
    project_psd_relative,
    psd_cholesky,
    psd_logdet,
    psd_solve,
    symmetrize,
    typed_jit,
    validate_count_array,
    validate_scalar,
    warn_if_not_positive_definite_in_graph,
)

if TYPE_CHECKING:
    import optax

logger = logging.getLogger(__name__)


def _validate_filter_numerics(
    init_covariance: Array,
    n_time: int,
    stacklevel: int = 3,
) -> None:
    """Laplace-EKF wrapper around the shared validator in ``utils``.

    See :func:`state_space_practice.utils._validate_filter_numerics` for the
    full contract. Kept as a module-level thin wrapper so existing
    ``from state_space_practice.point_process_kalman import
    _validate_filter_numerics`` imports continue to work, and so the f32
    warning names this filter specifically.
    """
    _validate_filter_numerics_impl(
        init_covariance,
        n_time,
        stacklevel=stacklevel + 1,
        filter_name="stochastic_point_process_filter",
    )


def _validate_public_inputs(
    dt: float,
    init_mean_params: Array,
    init_covariance_params: Array,
    design_matrix: Array,
    spike_indicator: Array,
    transition_matrix: Array,
    process_cov: Array,
    *,
    filter_name: str,
    stacklevel: int,
) -> None:
    """Value checks of the public point-process filter and smoother.

    Host-side value checks need concrete inputs. Concrete arrays (including
    constants closed over by a jitted caller) are checked under
    ``jax.ensure_compile_time_eval`` so jnp ops on them are not staged. When
    any input is a tracer (jax.jit / jax.grad / jax.vmap argument) the checks
    are skipped, keeping the entry point traceable with its default
    arguments, and a non-positive-definite ``init_covariance_params`` is
    reported at run time as a ``StateSpaceWarning``.

    Raises on non-PSD ``init_covariance_params`` and warns on f32 + long T +
    ill-conditioned configs (see :func:`_validate_filter_numerics`).
    ``stacklevel`` is interpreted as if the caller had called
    :func:`_validate_filter_numerics` directly.
    """
    if contains_tracer(
        dt,
        init_mean_params,
        init_covariance_params,
        design_matrix,
        spike_indicator,
        transition_matrix,
        process_cov,
    ):
        warn_if_not_positive_definite_in_graph(
            init_covariance_params,
            name="init_covariance_params",
            filter_name=filter_name,
        )
        return
    with jax.ensure_compile_time_eval():
        validate_scalar(dt, "dt", positive=True)
        validate_count_array(spike_indicator, "spike_indicator")
        _validate_filter_numerics(
            init_covariance_params,
            n_time=spike_indicator.shape[0],
            stacklevel=stacklevel + 1,
        )


class BlockDiagonalStructure(NamedTuple):
    """Factored block-diagonal filter problem.

    Holds the per-neuron components of a point-process filter problem
    where ``A``, ``Q``, ``init_cov``, and the design matrix ``Z`` are all
    block-diagonal across neurons with the same per-neuron block size.
    Each neuron keeps its own diagonal blocks of ``A``, ``Q`` and
    ``init_cov``; only the design basis is shared.

    The block-diagonal contract, for ``n_neurons`` blocks of ``block_size``:

    1. Design: every neuron row ``Z[t, j]`` is zero outside its own state
       slice ``[j*block_size : (j+1)*block_size]`` and the in-slice part is
       the same shared basis ``Z_base[t]`` for all neurons, with the default
       linear log-intensity ``Z_base[t] @ x_j``. Not checked by the library:
       it holds by construction for ``PlaceFieldModel``, which builds every
       neuron's row from one spline basis.
    2. Parameters: ``init_cov``, ``A`` and ``Q`` are block-diagonal (zero
       off-block entries). Their diagonal blocks may differ per neuron.
       Checked by :func:`_block_diagonal_parameters_ok`.

    The dense-form filter can be replaced by a vmapped per-neuron scan
    using these components — see ``_stochastic_point_process_filter_
    block_diagonal`` for the implementation. The vmap runs a d=block_size
    filter per neuron rather than a d=n_neurons*block_size dense filter,
    which is ~n_neurons^2 cheaper on the hot Cholesky path.

    Attributes
    ----------
    A_blocks : Array, shape (n_neurons, block_size, block_size)
        Per-neuron transition matrices, the diagonal blocks of ``A``.
    Q_blocks : Array, shape (n_neurons, block_size, block_size)
        Per-neuron process noise covariances, the diagonal blocks of ``Q``.
    init_means_per_neuron : Array, shape (n_neurons, block_size)
        Per-neuron initial mean, sliced out of the concatenated state vector.
    init_covs_per_neuron : Array, shape (n_neurons, block_size, block_size)
        Per-neuron initial covariance, sliced out of the block-diagonal
        init_cov.
    Z_base : Array, shape (n_time, block_size)
        Shared spline basis, identical across neurons (contract item 1).
    n_neurons : int
        Number of neurons.
    block_size : int
        Per-neuron state dimension.
    """

    A_blocks: Array
    Q_blocks: Array
    init_means_per_neuron: Array
    init_covs_per_neuron: Array
    Z_base: Array
    n_neurons: int
    block_size: int


def _assemble_block_diagonal_matrix(blocks: Array) -> Array:
    """Scatter ``(n_blocks, nb, nb)`` diagonal blocks into a dense matrix.

    Returns the ``(n_blocks * nb, n_blocks * nb)`` block-diagonal matrix
    whose off-block entries are exact zeros.
    """
    n_blocks, nb = blocks.shape[0], blocks.shape[-1]
    n_state = n_blocks * nb
    idx = (jnp.arange(n_blocks) * nb)[:, None] + jnp.arange(nb)[None, :]
    dense = jnp.zeros((n_state, n_state), dtype=blocks.dtype)
    return dense.at[idx[:, :, None], idx[:, None, :]].set(blocks)


class BlockDiagonalCovariance:
    """Per-neuron diagonal blocks of a block-diagonal covariance sequence.

    On a block-diagonal problem the filter / smoother covariance at every
    time bin is block-diagonal across neurons::

        dense[t] = blockdiag(blocks[0, t], ..., blocks[n_neurons - 1, t])

    with ``dense`` of shape ``(n_time, n_state, n_state)`` and
    ``n_state = n_neurons * block_size``. This container keeps only the
    ``(n_neurons, n_time, block_size, block_size)`` blocks -- ``n_neurons``
    times less memory than the zero-padded dense array -- and exposes the
    block-local reductions its consumers need without materialising it:
    :meth:`sum` over time and integer time indexing (the EM M-step
    sufficient statistics), :meth:`neuron_blocks` (per-neuron rate maps) and
    :meth:`diagonal` (marginal variances / confidence intervals).

    ``shape``, ``ndim`` and ``len`` report the dense geometry so shape checks
    written against the dense array keep working. Integer time indexing
    follows ndarray rules (negative indices allowed, out-of-range raises
    ``IndexError``, a bool is rejected with ``TypeError``); any other index
    expression (slices, tuples, masks) is applied to the dense array.
    Operations whose ndarray meaning the container does not implement raise
    rather than return something different (e.g. ``sum`` requires
    ``axis=0``).

    This is not a ``jax.Array``: ``jax.numpy`` functions (``jnp.sum(cov)``,
    ``jnp.diagonal(cov)``, arithmetic) and ``jax.jit`` raise when handed the
    container. ``jnp.asarray(cov)``, ``np.asarray(cov)`` and
    :meth:`to_dense` are the supported ways to materialise the dense array;
    the hot paths use the block-local methods instead. ``blocks`` is
    read-only.
    """

    __slots__ = ("_blocks",)

    def __init__(self, blocks: ArrayLike):
        blocks = jnp.asarray(blocks)
        if blocks.ndim != 4 or blocks.shape[-1] != blocks.shape[-2]:
            raise ValueError(
                "blocks must have shape (n_neurons, n_time, block_size, "
                f"block_size), got {blocks.shape}."
            )
        self._blocks = blocks

    @property
    def blocks(self) -> Array:
        """The ``(n_neurons, n_time, block_size, block_size)`` diagonal blocks."""
        return self._blocks

    @property
    def n_neurons(self) -> int:
        return self._blocks.shape[0]

    @property
    def n_time(self) -> int:
        return self._blocks.shape[1]

    @property
    def block_size(self) -> int:
        return self._blocks.shape[-1]

    @property
    def n_state(self) -> int:
        return self.n_neurons * self.block_size

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the equivalent dense array, ``(n_time, n_state, n_state)``."""
        return (self.n_time, self.n_state, self.n_state)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> jnp.dtype:
        return self._blocks.dtype

    def __len__(self) -> int:
        return self.n_time

    def __repr__(self) -> str:
        return (
            f"BlockDiagonalCovariance(n_neurons={self.n_neurons}, "
            f"n_time={self.n_time}, block_size={self.block_size})"
        )

    def neuron_blocks(self, neuron_idx: int, time_slice: slice = slice(None)) -> Array:
        """One neuron's ``(n_t, block_size, block_size)`` diagonal blocks.

        Equals ``dense[time_slice, s, s]`` for that neuron's state slice
        ``s``, read with a single gather from the block storage.
        """
        return self._blocks[neuron_idx, time_slice]

    def diagonal(self) -> Array:
        """Marginal variances, shape ``(n_time, n_state)``.

        Takes no arguments; equals ``dense.diagonal(axis1=1, axis2=2)``.
        """
        var = jnp.diagonal(self._blocks, axis1=-2, axis2=-1)  # (n_neurons, T, nb)
        return jnp.transpose(var, (1, 0, 2)).reshape(self.n_time, self.n_state)

    def sum(self, axis: int) -> Array:
        """Sum over time, a dense ``(n_state, n_state)`` matrix (``dense.sum(0)``).

        Only the time axis is supported (``axis=0``, or ``-3``), and it must
        be passed explicitly: it is the reduction the EM M-step needs, and it
        stays block-local (one small dense matrix is built, never the
        ``(n_time, n_state, n_state)`` sequence).
        """
        if axis not in (0, -3):
            raise ValueError("BlockDiagonalCovariance.sum only supports axis=0 (time).")
        return _assemble_block_diagonal_matrix(self._blocks.sum(axis=1))

    def _time_index(self, t: Any) -> int:
        """Normalise an integer time index; ndarray bounds, bool rejected."""
        if isinstance(t, (bool, np.bool_)):
            raise TypeError(
                "BlockDiagonalCovariance time index must be an integer, not a bool."
            )
        idx = operator.index(t)
        if not -self.n_time <= idx < self.n_time:
            raise IndexError(
                f"time index {idx} is out of bounds for n_time={self.n_time}."
            )
        return idx

    def at_time(self, t: int) -> Array:
        """Dense ``(n_state, n_state)`` covariance of one time bin (``dense[t]``)."""
        return _assemble_block_diagonal_matrix(self._blocks[:, self._time_index(t)])

    def __getitem__(self, index: Any) -> Array:
        if not isinstance(index, (bool, np.bool_)):
            try:
                operator.index(index)
            except TypeError:
                # Any non-integer indexing (slices, tuples, masks) goes through
                # the dense array -- correct for every index expression, at
                # the dense cost.
                return self.to_dense()[index]
        return self.at_time(index)

    def to_dense(self) -> Array:
        """Materialise the dense ``(n_time, n_state, n_state)`` array."""
        return jax.vmap(_assemble_block_diagonal_matrix, in_axes=1)(self._blocks)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        return np.asarray(self.to_dense(), dtype=dtype)

    def __jax_array__(self) -> Array:
        return self.to_dense()


def _matrix_blocks(mat: Array, n_blocks: int, block_size: int) -> Array:
    """View an ``(n, n)`` matrix as ``(n_blocks, n_blocks, nb, nb)`` blocks."""
    return mat.reshape(n_blocks, block_size, n_blocks, block_size).transpose(0, 2, 1, 3)


def _diagonal_blocks(mat: Array, n_blocks: int, block_size: int) -> Array:
    """The ``(n_blocks, nb, nb)`` diagonal blocks of a square matrix."""
    idx = jnp.arange(n_blocks)
    return _matrix_blocks(mat, n_blocks, block_size)[idx, idx]


def _scaled_atol(mat: Array, atol: float) -> Array:
    """Relative-absolute tolerance ``atol * max(1, max|mat|)``."""
    return atol * jnp.maximum(1.0, jnp.max(jnp.abs(mat)))


@functools.partial(typed_jit, static_argnames=("n_neurons", "block_size", "atol"))
def _block_diagonal_parameters_ok(
    init_cov: Array,
    transition_matrix: Array,
    process_cov: Array,
    *,
    n_neurons: int,
    block_size: int,
    atol: float = 1e-10,
) -> Array:
    """Vectorised check of the parameter half of the block-diagonal contract.

    True iff ``init_cov``, ``transition_matrix`` and ``process_cov`` are
    block-diagonal with ``n_neurons`` blocks of ``block_size`` (all
    off-block entries zero) -- item 2 of the :class:`BlockDiagonalStructure`
    contract. The diagonal blocks may differ per neuron.
    One fused device computation returning a boolean scalar, so a caller
    pays a single host sync (``bool(...)``) instead of a Python loop of
    ``float()`` syncs; ``PlaceFieldModel`` runs it at fit entry and after
    every EM M-step.

    Tolerance model: the off-block-zero check for each matrix uses
    ``atol * max(1, max|mat|)``, so matrices with O(1) entries
    use the absolute floor while large-magnitude covariances get a
    proportionally larger tolerance. For f32 inputs pass a larger ``atol``
    (typically ``1e-6`` to ``1e-5``).
    """
    off_block = ~jnp.eye(n_neurons, dtype=bool)

    def off_block_max(mat: Array) -> Array:
        blocks = _matrix_blocks(mat, n_neurons, block_size)
        return jnp.max(jnp.where(off_block[:, :, None, None], jnp.abs(blocks), 0.0))

    ok = off_block_max(init_cov) <= _scaled_atol(init_cov, atol)
    ok &= off_block_max(transition_matrix) <= _scaled_atol(transition_matrix, atol)
    ok &= off_block_max(process_cov) <= _scaled_atol(process_cov, atol)
    return ok


def _build_block_structure_from_traced(
    init_mean: Array,
    init_cov: Array,
    transition_matrix: Array,
    process_cov: Array,
    design_matrix: Array,
    n_neurons: int,
    block_size: int,
) -> BlockDiagonalStructure:
    """Build a BlockDiagonalStructure from traced arrays using pure slicing.

    Unlike :func:`_block_diagonal_parameters_ok` -- whose verdict must be
    read back on the host (``bool(...)``) and which therefore cannot gate a
    branch inside ``jax.jit`` / ``jax.grad`` -- this helper assumes the
    caller has ALREADY verified block-diagonal structure (at fit entry time,
    with concrete arrays) and simply extracts the per-neuron factors -- the
    diagonal blocks of ``A``, ``Q`` and ``init_cov`` -- via static slicing.
    Off-block entries are ignored. It runs
    safely inside the jit boundary because ``n_neurons`` and ``block_size``
    are Python integers (not traced values) and all slicing indices are
    compile-time constants.

    Used by ``stochastic_point_process_filter``'s auto-dispatch path:
    the caller verifies the structure once at fit entry
    (``PlaceFieldModel._detect_block_structure``, which runs
    ``_block_diagonal_parameters_ok``) to get
    ``n_neurons`` / ``block_size``, then passes those integers into the
    jit-compiled loss function. Inside the loss function, the traced
    (init_mean, init_cov, A, Q, Z) arrays may differ from the
    detection-time values (e.g., the SGD optimizer has updated them). The
    result equals the dense filter only while they keep zero off-block
    entries, so the caller must only take this path for parameterizations
    that cannot leave the block-diagonal set: a diagonal or isotropic ``Q``
    (per-neuron values are fine -- each neuron gets its own block) and a
    ``transition_matrix`` that is not trained by gradient (a learned dense
    ``A`` has off-block gradients the block filter never sees, so
    ``PlaceFieldModel.fit_sgd`` dispatches dense when
    ``update_transition_matrix=True``).

    The EM M-step can write back a dense ``A``, so the model layer re-checks
    the structure after each M-step (``PlaceFieldModel._detect_block_structure``)
    and falls back to dense when it breaks.

    Parameters
    ----------
    init_mean : Array, shape (n_neurons * block_size,)
    init_cov : Array, shape (n_state, n_state)
    transition_matrix : Array, shape (n_state, n_state)
    process_cov : Array, shape (n_state, n_state)
    design_matrix : Array, shape (n_time, n_neurons, n_state) or (n_time, block_size)
        Either the block-expanded design matrix or, directly, the shared
        per-neuron basis ``Z_base``. ``PlaceFieldModel`` passes ``Z_base``
        so the expansion (>99% structural zeros) is never built.
    n_neurons : int
    block_size : int

    Returns
    -------
    BlockDiagonalStructure
    """
    nb = block_size
    # Per-neuron A, Q and init_cov: the diagonal blocks, gathered with
    # compile-time constant indices (``n_neurons`` and ``nb`` are Python
    # ints), so this is static slicing of the traced matrices.
    A_blocks = _diagonal_blocks(transition_matrix, n_neurons, nb)
    Q_blocks = _diagonal_blocks(process_cov, n_neurons, nb)
    init_covs_per_neuron = _diagonal_blocks(init_cov, n_neurons, nb)

    # Per-neuron init_mean: reshape the concatenated state vector.
    init_means_per_neuron = init_mean.reshape(n_neurons, nb)

    # Z_base: the shared basis itself when passed directly, else neuron 0's
    # own slice of the expanded design (all neurons share it by contract).
    Z_base = design_matrix if design_matrix.ndim == 2 else design_matrix[:, 0, 0:nb]

    return BlockDiagonalStructure(
        A_blocks=A_blocks,
        Q_blocks=Q_blocks,
        init_means_per_neuron=init_means_per_neuron,
        init_covs_per_neuron=init_covs_per_neuron,
        Z_base=Z_base,
        n_neurons=n_neurons,
        block_size=nb,
    )


def _validated_dispatch_block_structure(
    init_mean: Array,
    init_cov: Array,
    transition_matrix: Array,
    process_cov: Array,
    design_matrix: Array,
    spike_indicator: Array,
    block_n_neurons: int | None,
    block_size: int | None,
) -> BlockDiagonalStructure:
    """Check the block-dispatch request and extract its per-neuron factors.

    Shared entry of the block-diagonal dispatch in
    :func:`stochastic_point_process_filter` and
    :func:`stochastic_point_process_smoother`, called only when both
    ``block_n_neurons`` and ``block_size`` are set.

    Raises
    ------
    ValueError
        If ``spike_indicator`` is 1-D (single neuron), or if
        ``block_n_neurons * block_size`` does not match the state dimension
        of ``init_cov``.
    """
    # The callers only dispatch when both are set; assert so mypy narrows
    # block_n_neurons / block_size from int | None to int.
    assert block_n_neurons is not None and block_size is not None
    if spike_indicator.ndim == 1:
        raise ValueError(
            "block_n_neurons / block_size can only be passed for "
            "multi-neuron (2D spike_indicator) inputs."
        )
    # Shape guard: block_n_neurons * block_size must match the total state
    # dimension. A mismatch means the caller passed wrong detection integers
    # or the model state shape changed between detection and dispatch.
    expected_state_dim = block_n_neurons * block_size
    if init_cov.shape[-1] != expected_state_dim:
        raise ValueError(
            f"block dispatch shape mismatch: block_n_neurons="
            f"{block_n_neurons} * block_size={block_size} = "
            f"{expected_state_dim}, but init_cov has shape "
            f"{init_cov.shape}. Pass the block "
            f"structure of the current problem (PlaceFieldModel "
            f"re-derives it with _detect_block_structure)."
        )
    return _build_block_structure_from_traced(
        init_mean,
        init_cov,
        transition_matrix,
        process_cov,
        design_matrix,
        block_n_neurons,
        block_size,
    )


def _common_float_dtype(*arrays: Array) -> jnp.dtype:
    """Common floating dtype of the floating-point ``arrays``.

    Integer / boolean arrays (e.g. spike counts) do not take part, so they
    cannot promote a float32 problem; with no floating array at all the
    default float dtype is returned.
    """
    floating = [a for a in arrays if jnp.issubdtype(a.dtype, jnp.inexact)]
    return jnp.result_type(float, *floating)


def log_conditional_intensity(design_matrix: ArrayLike, params: ArrayLike) -> Array:
    """Computes the log conditional intensity for a point process.

    This is the default linear log-intensity function: log(λ) = Z @ x.

    For single-neuron models, this returns a scalar.
    For multi-neuron models, design_matrix should be (n_neurons, n_params)
    so this returns (n_neurons,) log-intensities.

    Parameters
    ----------
    design_matrix : ArrayLike, shape (n_params,) or (n_neurons, n_params)
        Design matrix (Z_k) used in the intensity function.
        For single neuron: (n_params,) row vector.
        For multi-neuron: (n_neurons, n_params) matrix.
    params : ArrayLike, shape (n_params,)
        Parameters (latent state) for the intensity function.

    Returns
    -------
    Array, shape () or (n_neurons,)
        Log conditional intensity (log(λ_k)).
        Scalar for single-neuron, (n_neurons,) for multi-neuron.
    """
    return jnp.asarray(design_matrix) @ jnp.asarray(params)


def _uses_default_linear_log_intensity(
    func: Callable[[ArrayLike, ArrayLike], Array],
) -> bool:
    """Return True when ``func`` is the module's default linear log-rate."""
    return func is log_conditional_intensity


def _logdet_psd(mat: Array, diagonal_boost: float = 1e-9) -> Array:
    """Log-determinant of a PSD matrix via Cholesky (stabilized).

    Uses ``logdet(A) = 2 * sum(log(diag(chol(A))))`` rather than
    ``sum(log(eigvalsh(A)))``. Cholesky is ~3-5x faster than eigvalsh
    for small-to-moderate PSD matrices and dominates the runtime of
    the Laplace-EKF filter's inner scan body — benchmarks show ~1.5-1.7x
    overall filter speedup on T=10k, d=36.

    Stability is ensured by adding ``diagonal_boost * I`` before
    factoring (a uniform eigenvalue shift) rather than clipping
    individual eigenvalues — the two strategies are equivalent for
    well-conditioned matrices and the shift is slightly more conservative
    for pathological ones. The caller is expected to hand in a PSD
    matrix; Kalman-filter covariances and posterior-precision matrices
    are PSD by construction.

    Parameters
    ----------
    mat : Array, shape (n, n)
        Symmetric positive semi-definite matrix.
    diagonal_boost : float
        Uniform eigenvalue shift for numerical stability before Cholesky.

    Returns
    -------
    Array
        Scalar log-determinant.
    """
    return psd_logdet(psd_cholesky(mat, diagonal_boost, relative_boost=0.0))


def _safe_expected_count(
    log_rate: Array,
    dt: float,
    min_log_count: float = -20.0,
    max_log_count: float = 20.0,
) -> Array:
    """Convert log-rate in Hz to expected count per bin with overflow protection.

    ``log(rate * dt)`` is clipped to ``[min_log_count, max_log_count]`` before
    exponentiation. ``max_log_count`` guards against overflow on pathological
    spike counts (see the calling filters' docstrings). ``min_log_count``
    floors the expected count at ``exp(min_log_count)`` so ``log(mu)`` stays
    finite; note this floor is not free of consequences. For any bin whose true
    ``rate * dt`` underflows the floor, ``y * log(mu)`` biases toward
    ``min_log_count * y`` and, because ``jnp.clip`` has zero gradient outside
    its range, that bin contributes no gradient in ``fit_sgd`` (dead learning
    signal). The default of ``-20`` corresponds to ~2e-9 counts/bin, so it only
    bites when the model drives a rate implausibly low.
    """
    dt_array = jnp.asarray(dt, dtype=log_rate.dtype)
    log_count = log_rate + jnp.log(dt_array)
    return jnp.exp(jnp.clip(log_count, min_log_count, max_log_count))


def _soft_expected_count(
    log_rate: Array,
    dt: float,
    max_log_count: float = 20.0,
) -> Array:
    """Expected count per bin with a *gradient-preserving* overflow cap.

    Equals :func:`_safe_expected_count` (``exp(log_rate*dt)``) only in the band
    ``-20 <= log_count <= max_log_count`` -- where neither function's clip is
    active. Above ``max_log_count`` it continues ``exp`` **logarithmically** --
    ``exp(cap) * (1 + log1p(overshoot))`` -- instead of hard-clipping. Below
    ``-20`` the two differ: ``_safe_expected_count`` floors ``log_count`` at its
    ``min_log_count`` while this function does not (see the no-lower-clip note).

    A hard ``jnp.clip`` has exactly zero gradient above the cap, so an SGD
    surrogate loss that overshoots it freezes with no signal to return: a
    dead-gradient stall, no better than the NaN a raw ``exp`` overflow would
    cause (see ``parameter_transforms.positive_capped`` for the same
    anti-pattern).

    A *linear* continuation ``exp(cap)*(1 + overshoot)`` fixes the gradient but
    reintroduces overflow: the value overflows for large finite inputs (e.g.
    float32 ``log_rate=1e30``, and even float64 well before ``+inf``), and its
    VJP sends a cotangent of order ``exp(cap)*overshoot`` back through ``exp``,
    which itself overflows to inf and then ``inf*0 = NaN`` at the clamped
    ``minimum``. The *logarithmic* continuation keeps the value finite for every
    finite ``log_rate`` (``log1p`` grows so slowly the product stays well under
    the dtype max) and the VJP cotangent bounded (order ``exp(cap)*log1p``), so
    both the value and the gradient are finite across the dtype range. The
    gradient ``exp(cap)/(1 + overshoot)`` stays strictly positive -- a restoring
    push toward the cap that decays with the overshoot but never vanishes or
    NaNs. Value and first derivative match ``exp`` at the cap.

    Unlike :func:`_safe_expected_count` there is no lower clip: the only consumer
    that needs ``log(mu)``, :func:`_soft_expected_count_and_log`, returns the
    analytic log directly, so the underflow direction needs no guard here. A
    value-only caller that needs ``log(mu)`` should read it from that companion
    rather than flooring ``log(count + eps)``.

    Parameters
    ----------
    log_rate : Array
        Log firing rate in Hz.
    dt : float
        Bin width in seconds.
    max_log_count : float, default 20.0
        Log expected-count above which ``exp`` is continued logarithmically.

    Returns
    -------
    Array
        Expected count per bin, finite (value and gradient) for every finite
        ``log_rate``.
    """
    count, _ = _soft_expected_count_and_log(log_rate, dt, max_log_count)
    return count


def _soft_expected_count_and_log(
    log_rate: Array,
    dt: float,
    max_log_count: float = 20.0,
) -> tuple[Array, Array]:
    """Soft-capped expected count and its exact log, from one decomposition.

    Returns ``(count, log_count)`` where ``count`` is :func:`_soft_expected_count`
    and ``log_count`` equals ``log(count)`` **analytically** -- derived from the
    same ``capped``/``overshoot`` terms, so the two share one cap and cannot
    drift out of the ``log_count == log(count)`` identity.

    A Poisson surrogate that needs ``log(mu)`` must read it from here rather than
    forming ``log(count + eps)``: once ``count`` underflows to ``0`` the ``+eps``
    floor pins the log at ``log(eps)``, zeroing the restoring gradient for
    positive spike counts. ``log_count`` stays finite and exact for every finite
    ``log_rate`` because it never exponentiates the (possibly very negative)
    capped log-count. See :func:`_soft_expected_count` for the overflow-side
    rationale behind the logarithmic continuation.

    Parameters
    ----------
    log_rate : Array
        Log firing rate in Hz.
    dt : float
        Bin width in seconds.
    max_log_count : float, default 20.0
        Log expected-count above which ``exp`` is continued logarithmically.

    Returns
    -------
    count : Array
        Expected count per bin, finite (value and gradient) for every finite
        ``log_rate``.
    log_count : Array
        The analytic ``log(rate * dt)`` under the same cap: equals ``log(count)``
        wherever ``count`` is representable and positive, and stays finite where
        ``count`` underflows to 0 (where ``log(count)`` itself would be ``-inf``).
    """
    dt_array = jnp.asarray(dt, dtype=log_rate.dtype)
    log_count = log_rate + jnp.log(dt_array)
    # exp is evaluated ONLY on the capped argument, so it never overflows. Above
    # the cap, ``log1p(overshoot)`` carries a bounded, positive gradient; because
    # log1p grows slowly, neither the value nor the VJP cotangent overflows, so
    # the gradient is finite (a linear ``overshoot`` here would NaN it).
    capped = jnp.minimum(log_count, max_log_count)
    overshoot = jnp.maximum(log_count - max_log_count, 0.0)
    count = jnp.exp(capped) * (1.0 + jnp.log1p(overshoot))
    # log(count) = capped + log(1 + log1p(overshoot)); log1p(log1p(...)) keeps it
    # exact without forming ``count`` first, so an underflowed count still yields
    # a finite, correctly-signed Poisson gradient.
    log_count_out = capped + jnp.log1p(jnp.log1p(overshoot))
    return count, log_count_out


#: Armijo sufficient-decrease constant of the Fisher-scoring line search.
_ARMIJO_C = 1e-4
#: Number of trial step sizes ``1, beta, ..., beta**(N - 1)`` (``beta`` is
#: ``line_search_beta``) evaluated per Fisher iteration.
_LINE_SEARCH_MAX_BACKTRACKS = 10
#: Fraction of time bins whose backtracking may be exhausted before the public
#: filters log a warning.
_LINE_SEARCH_FAIL_WARN_FRAC = 0.1


def _fisher_scoring_line_search(
    x0: Array,
    prior_precision: Array,
    fisher_step_at: Callable[[Array], tuple[Array, Array, Array]],
    neg_log_posterior: Callable[[Array], Array],
    max_newton_iter: int,
    line_search_beta: float,
) -> tuple[Array, Array, Array]:
    """Iterated Fisher scoring with an Armijo backtracking line search.

    Shared by :func:`_point_process_laplace_update` and
    :func:`glm_laplace_update`. ``fisher_step_at(x)`` returns
    ``(delta, post_prec, gradient)`` -- the Fisher direction
    ``delta = post_prec^{-1} gradient`` and posterior precision at ``x``, with
    ``gradient`` the gradient of the log-posterior -- and
    ``neg_log_posterior(x)`` the objective ``f`` the line search decreases.

    The trial step sizes are ``alpha = 1, beta, ..., beta^(N-1)`` with
    ``N = _LINE_SEARCH_MAX_BACKTRACKS``; the largest is accepted that
    satisfies the Armijo condition ``f(x + alpha delta) <= f(x) - c alpha gradient'
    delta`` with ``c = _ARMIJO_C``, up to a round-off slack of
    ``1e-12 (1 + |f(x)|)`` so a converged point still takes its (negligible)
    full step instead of freezing. Because ``post_prec`` is positive definite,
    ``gradient' delta >= 0`` and ``delta`` is a descent direction. When the
    full step is accepted the result is the same as without the Armijo
    condition. If no step size qualifies, ``x`` is kept (an uphill or
    insufficient step is never taken).

    With ``max_newton_iter == 0`` no measurement update is made: the prior
    ``(x0, prior_precision)`` is returned unchanged.

    The scan carries ``(x, delta, post_prec, loss, slope, n_failed)``, i.e.
    the Fisher step, the objective and the directional derivative *at the
    current point*. They are computed exactly once, when a point is accepted,
    and reused as the next iteration's search direction and references; the
    precision at the last accepted point is the returned posterior precision.

    Returns
    -------
    x : Array
        The accepted point after ``max_newton_iter`` iterations.
    post_prec : Array
        The Fisher posterior precision evaluated at ``x``.
    n_failed : Array
        Number of iterations (int32 scalar) whose backtracking was exhausted
        although the Fisher step predicted a decrease above roundoff
        (``0.5 gradient' delta > sqrt(eps) (1 + |f|)``). Exhaustion at an
        already-converged point is not counted.
    """
    eps_sqrt = float(jnp.finfo(x0.dtype).eps) ** 0.5

    def _line_search_step(
        carry: tuple[Array, Array, Array, Array, Array, Array], _: Array | None
    ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], None]:
        x, delta, _, current_loss, slope, n_failed = carry

        # Backtracking line search. The loss at the evaluated step size rides
        # along in the carry, so the winning loss is known without a second
        # evaluation at the accepted point.
        def _backtrack(
            alpha_carry: tuple[Array, Array, Array], _: Array | None
        ) -> tuple[tuple[Array, Array, Array], None]:
            alpha, _, _ = alpha_carry
            new_x = x + alpha * delta
            new_loss = neg_log_posterior(new_x)
            # The slack admits steps that change f only at round-off level,
            # so at a converged mode the (tiny) full step is still taken:
            # rejecting it would freeze x, and reverse-mode gradients
            # through the scan would then miss the Fisher map's contraction
            # and carry the error of an earlier, unconverged iterate.
            slack = 1e-12 * (1.0 + jnp.abs(current_loss))
            improved = new_loss <= current_loss - _ARMIJO_C * alpha * slope + slack
            new_alpha = jnp.where(improved, alpha, alpha * line_search_beta)
            return (new_alpha, improved, new_loss), None

        (final_alpha, line_search_improved, final_loss), _ = jax.lax.scan(
            _backtrack,
            (jnp.ones((), dtype=x.dtype), jnp.array(False), current_loss),
            None,
            length=_LINE_SEARCH_MAX_BACKTRACKS,
        )
        candidate_x = x + final_alpha * delta

        # Reject uphill / insufficient steps: reuse the improved flag from
        # the backtracking scan rather than re-evaluating neg_log_posterior.
        # If improved is False, no step size qualified -- keep current x and
        # count the failure unless the step was already negligible (a
        # converged point legitimately exhausts the backtracking).
        new_x = jnp.where(line_search_improved, candidate_x, x)
        new_loss = jnp.where(line_search_improved, final_loss, current_loss)
        exhausted = ~line_search_improved & (
            0.5 * slope > eps_sqrt * (1.0 + jnp.abs(current_loss))
        )

        # Fisher step at the accepted point: its precision is the posterior
        # precision if this was the last iteration, and its direction is the
        # next iteration's step.
        new_delta, new_post_prec, new_gradient = fisher_step_at(new_x)
        new_slope = new_gradient @ new_delta
        return (
            new_x,
            new_delta,
            new_post_prec,
            new_loss,
            new_slope,
            n_failed + exhausted.astype(jnp.int32),
        ), None

    if max_newton_iter == 0:
        return x0, prior_precision, jnp.zeros((), dtype=jnp.int32)
    delta0, post_prec0, gradient0 = fisher_step_at(x0)
    (x, _, post_prec, _, _, n_failed), _ = jax.lax.scan(
        _line_search_step,
        (
            x0,
            delta0,
            post_prec0,
            neg_log_posterior(x0),
            gradient0 @ delta0,
            jnp.zeros((), dtype=jnp.int32),
        ),
        None,
        length=max_newton_iter,
    )
    return x, post_prec, n_failed


def _log_line_search_failures(
    n_failed: Array, *, n_units: int, max_newton_iter: int, name: str, unit: str
) -> None:
    """Host-side logger for :func:`_warn_line_search_failures`."""
    n = int(np.sum(np.asarray(n_failed)))
    frac = n / n_units
    if frac > _LINE_SEARCH_FAIL_WARN_FRAC:
        logger.warning(
            "%s: the Fisher-scoring line search rejected all %d trial step "
            "sizes in %d/%d (%.1f%%) %s with max_newton_iter=%d; those "
            "updates kept a non-converged posterior mode. This usually means a "
            "poorly scaled or misspecified intensity (check the design matrix "
            "and max_log_count).",
            name,
            _LINE_SEARCH_MAX_BACKTRACKS,
            n,
            n_units,
            100.0 * frac,
            unit,
            max_newton_iter,
        )


def _warn_line_search_failures(
    n_failed_bins: Array,
    n_bins: int,
    max_newton_iter: int,
    filter_name: str,
    unit: str = "time bins",
) -> None:
    """Log (host side, jit-safe) when many updates exhausted the line search.

    ``n_failed_bins`` counts the updates (time bins, summed over neurons on
    the block path; or independent regressions) in which at least one Fisher
    iteration exhausted its backtracking with a non-negligible predicted
    decrease. A warning is logged when ``n_failed_bins / n_bins`` exceeds
    ``_LINE_SEARCH_FAIL_WARN_FRAC``. The check runs through
    :func:`jax.debug.callback`, so it reports eagerly and from inside
    ``jax.jit`` / ``jax.grad`` (e.g. an SGD loss) alike. Silent for
    ``max_newton_iter <= 1`` (no line search runs).

    Parameters
    ----------
    n_failed_bins : Array
        Integer scalar count of failed updates.
    n_bins : int
        Number of updates the count is out of (static).
    max_newton_iter : int
        Fisher-scoring iterations per update (static).
    filter_name : str
        Name of the calling routine, used in the log message.
    unit : str, default="time bins"
        What one update is, used in the log message.
    """
    if max_newton_iter <= 1 or n_bins == 0:
        return
    jax.debug.callback(
        functools.partial(
            _log_line_search_failures,
            n_units=n_bins,
            max_newton_iter=max_newton_iter,
            name=filter_name,
            unit=unit,
        ),
        n_failed_bins,
    )


@overload
def _point_process_laplace_update(
    one_step_mean: Array,
    one_step_cov: Array,
    spike_indicator_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
    diagonal_boost: float = ...,
    grad_log_intensity_func: Callable[[Array], Array] | None = ...,
    include_laplace_normalization: bool = ...,
    max_newton_iter: int = ...,
    line_search_beta: float = ...,
    max_log_count: float = ...,
    return_line_search_failures: Literal[False] = ...,
) -> tuple[Array, Array, Array]: ...


@overload
def _point_process_laplace_update(
    one_step_mean: Array,
    one_step_cov: Array,
    spike_indicator_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
    diagonal_boost: float = ...,
    grad_log_intensity_func: Callable[[Array], Array] | None = ...,
    include_laplace_normalization: bool = ...,
    max_newton_iter: int = ...,
    line_search_beta: float = ...,
    max_log_count: float = ...,
    return_line_search_failures: Literal[True] = ...,
) -> tuple[Array, Array, Array, Array]: ...


@overload
def _point_process_laplace_update(
    one_step_mean: Array,
    one_step_cov: Array,
    spike_indicator_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
    diagonal_boost: float = ...,
    grad_log_intensity_func: Callable[[Array], Array] | None = ...,
    include_laplace_normalization: bool = ...,
    max_newton_iter: int = ...,
    line_search_beta: float = ...,
    max_log_count: float = ...,
    return_line_search_failures: bool = ...,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]: ...


def _point_process_laplace_update(
    one_step_mean: Array,
    one_step_cov: Array,
    spike_indicator_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
    diagonal_boost: float = 0.0,
    grad_log_intensity_func: Callable[[Array], Array] | None = None,
    include_laplace_normalization: bool = True,
    max_newton_iter: int = 3,
    line_search_beta: float = 0.5,
    max_log_count: float = 20.0,
    return_line_search_failures: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
    """Single point-process Laplace-EKF update for multiple neurons.

    Performs a Bayesian update of the latent state posterior given observed
    spike counts, using a Gaussian (Laplace) approximation to the posterior.
    The approximation is built via **Fisher scoring** (expected Hessian /
    statistical linearization) rather than full Newton-Raphson with the
    observed Hessian.

    This is the core math for point-process observation updates, factored
    out to be reusable by both the non-switching and switching filters.

    The observation model is:
        y_n ~ Poisson(exp(log_intensity_func(x)[n]) * dt)

    Fisher scoring vs full Newton
    -----------------------------
    The posterior precision is built as

        Lambda_post = P_prior^{-1} + J' diag(lambda * dt) J

    where ``J`` is the Jacobian of ``log_intensity_func`` w.r.t. the state
    and ``lambda`` is the conditional intensity. The posterior covariance is
    ``P_post = Lambda_post^{-1}``. This is a sum of PSD matrices, so
    ``Lambda_post`` is PSD by construction and requires no
    eigenvalue stabilization.

    Full Newton would additionally subtract the observed Hessian correction
    ``sum_n (y_n - lambda_n * dt) * d^2(log lambda_n)/dx^2``, which is
    indefinite in general and can produce wildly large steps at non-MAP
    points. For **linear** log-intensities (``log lambda = Z @ x``, the
    default via :func:`log_conditional_intensity`) the second derivative is
    zero and Fisher scoring is mathematically identical to full Newton.
    For **nonlinear** intensities (e.g. KDE rate maps in
    :class:`PositionDecoder`), Fisher scoring produces better-conditioned,
    more stable updates. This matches the approach used in dynamax and
    generalized linear model IRLS.

    Parameters
    ----------
    one_step_mean : Array, shape (n_latent,)
        Predicted mean from dynamics: A @ m_{t-1}
    one_step_cov : Array, shape (n_latent, n_latent)
        Predicted covariance: A @ P_{t-1} @ A.T + Q
    spike_indicator_t : Array, shape (n_neurons,)
        Spike counts at time t for all neurons
    dt : float
        Time bin width in seconds
    log_intensity_func : Callable[[Array], Array]
        Function mapping state (n_latent,) to log-intensities (n_neurons,).
        Should return log(lambda) where lambda is firing rate in Hz.
    diagonal_boost : float, default=0.0
        Absolute floor of the diagonal shift added before each Cholesky
        factorization (prior covariance, posterior precision). The default
        leaves only the scale-relative shift of
        :func:`~state_space_practice.utils.psd_cholesky` (``1e-12 * |A_ii|``
        in float64), which keeps the update scale-equivariant: rescaling the
        latent state by ``c`` (all covariances by ``c**2``, the design by
        ``1/c``) rescales the output exactly. A positive value breaks that
        for covariances below ``~diagonal_boost / 1e-12`` in scale.
    grad_log_intensity_func : Callable[[Array], Array] | None, optional
        Pre-computed gradient function (Jacobian) of log_intensity_func.
        If None, computed via jax.jacfwd(log_intensity_func).
        Passing pre-computed functions can improve compilation speed when
        this function is called repeatedly inside a JIT-compiled context.
    include_laplace_normalization : bool, default=True
        If True, include the Laplace normalization and prior terms to approximate
        log p(y_t | y_{1:t-1}). If False, return the plug-in log-likelihood
        at the posterior mode without normalization.
    max_newton_iter : int, default=3
        Maximum number of Fisher scoring iterations. Use > 1 with line search
        for numerical stability with large spike counts (e.g., many neurons).
        (Named ``max_newton_iter`` for backwards compatibility; the inner
        iterations are Fisher steps, not full Newton.)
    line_search_beta : float, default=0.5
        Step size reduction factor for the backtracking line search. Only
        used when max_newton_iter > 1. Each iteration tries the step sizes
        ``1, beta, ..., beta**(N - 1)`` (``N = _LINE_SEARCH_MAX_BACKTRACKS``)
        and takes the largest that satisfies the Armijo sufficient-decrease
        condition on the negative log-posterior, up to a relative round-off
        slack; if none does, the iterate is kept (see
        :func:`_fisher_scoring_line_search`).
    max_log_count : float, default=20.0
        Ceiling on ``log(rate * dt)`` applied inside ``_safe_expected_count``
        to prevent overflow when the Fisher step produces implausibly large
        intensities. Callers should set this to a physiologically motivated
        value, e.g. ``log(max_firing_rate_hz * dt)``. The default of 20.0
        corresponds to ~2.4e9 Hz at dt=0.2s, which is high enough to mask
        pathological outlier bins; tighter ceilings (e.g. ``log(500 * dt)``)
        catch them.
    return_line_search_failures : bool, default=False
        If True, also return the number of Fisher iterations whose Armijo
        backtracking was exhausted (see :func:`_fisher_scoring_line_search`;
        always 0 for ``max_newton_iter <= 1``). The filters aggregate it to
        warn once per call.

    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
        Updated state mean after incorporating spike observations
    posterior_cov : Array, shape (n_latent, n_latent)
        Updated state covariance after incorporating spike observations
    log_likelihood : Array
        Approximate log p(y_t | y_{1:t-1}) using a Laplace expansion (scalar array).
    n_line_search_failures : Array, optional
        Only with ``return_line_search_failures=True`` (int32 scalar).

    Notes
    -----
    The Fisher-scoring iteration starts at the predicted mean; with
    ``max_newton_iter == 1`` it is a single step from there, otherwise up to
    ``max_newton_iter`` line-searched steps toward the posterior mode. For
    multiple neurons, the gradients and Jacobians are summed across neurons.

    For Poisson likelihood with log-link:
        log p(y | x) = sum_n [y_n * log(lambda_n * dt) - lambda_n * dt - log(y_n!)]
        gradient = sum_n [(y_n - lambda_n * dt) * d(log_lambda_n)/dx]
        Hessian = sum_n [(y_n - lambda_n * dt) * d^2(log_lambda_n)/dx^2
                        - lambda_n * dt * (d(log_lambda_n)/dx)^T @ (d(log_lambda_n)/dx)]

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """
    # Compute gradient of log-intensity function. The Hessian is NOT used
    # because we use Fisher scoring (expected Hessian) rather than the full
    # observed Hessian. For Poisson with log link, the Fisher information is
    # J' diag(lambda * dt) J, which is PSD by construction — no stabilization
    # of the posterior precision is required. The observed Hessian adds a
    # (y - lambda*dt) * d^2(log lambda)/dx^2 correction that can be indefinite
    # at non-MAP points; dropping it is the standard approach used in
    # dynamax, glmnet-style IRLS, and textbook Fisher scoring.
    #
    # For linear log-intensity (log lambda = Z @ x), the second derivative is
    # zero so Fisher scoring is mathematically identical to full Newton.
    # For nonlinear intensities (e.g. KDE rate maps in PositionDecoder),
    # Fisher scoring produces better-conditioned updates because it
    # avoids inverting a precision whose PSD-ness relies on jitter.
    if grad_log_intensity_func is None:
        grad_log_intensity_func = jax.jacfwd(log_intensity_func)
    grad_log_intensity = grad_log_intensity_func

    n_latent = one_step_mean.shape[0]
    identity = jnp.eye(n_latent, dtype=one_step_cov.dtype)
    # Factor the prior covariance once: the precision (used by the Fisher
    # steps and the quadratic form) and the log-determinant of the Laplace
    # normaliser both come from this factor, so they see the same jittered
    # matrix (psd_cholesky's scale-relative diagonal shift).
    prior_cho = psd_cholesky(one_step_cov, diagonal_boost=diagonal_boost)
    prior_precision = jax.scipy.linalg.cho_solve(prior_cho, identity)

    def _neg_log_posterior(x: Array) -> Array:
        """Negative log-posterior for line search."""
        log_lambda = log_intensity_func(x)
        cond_int = _safe_expected_count(log_lambda, dt, max_log_count=max_log_count)
        # Poisson log-likelihood (ignoring constant log(y!) term)
        # No floor needed: _safe_expected_count guarantees cond_int >= exp(-20) > 0
        log_lik = jnp.sum(spike_indicator_t * jnp.log(cond_int) - cond_int)
        # Gaussian prior log-probability (ignoring constant)
        delta = x - one_step_mean
        log_prior = -0.5 * delta @ (prior_precision @ delta)
        return -(log_lik + log_prior)

    def _fisher_step_at(x: Array) -> tuple[Array, Array, Array]:
        """Compute Fisher-scoring step and posterior precision at point x.

        Uses the expected Hessian (Fisher information) rather than the
        observed Hessian:

            -E[H_log_likelihood] = J' diag(lambda * dt) J    [PSD]

        Combined with the prior precision this gives a posterior precision
        that is PSD by construction:

            post_prec = prior_precision + J' diag(lambda * dt) J
        """
        log_lambda = log_intensity_func(x)
        conditional_intensity = _safe_expected_count(
            log_lambda, dt, max_log_count=max_log_count
        )
        innovation = spike_indicator_t - conditional_intensity
        jacobian = grad_log_intensity(x)

        # Likelihood gradient (same as full Newton)
        likelihood_gradient = jacobian.T @ innovation

        # Prior gradient: -prior_precision @ (x - one_step_mean)
        prior_gradient = -prior_precision @ (x - one_step_mean)

        # Full posterior gradient
        gradient = likelihood_gradient + prior_gradient

        # Fisher information (expected negative Hessian of log-likelihood).
        # J' diag(cond_int) J is a sum of rank-1 PSD terms. Adding the PSD
        # prior precision gives a PSD posterior precision — no stabilization
        # of indefiniteness is required.
        fisher_info = jacobian.T @ (conditional_intensity[:, None] * jacobian)
        post_prec = symmetrize(prior_precision + fisher_info)

        # Fisher-scoring direction
        delta = psd_solve(post_prec, gradient, diagonal_boost=diagonal_boost)
        return delta, post_prec, gradient

    if max_newton_iter == 1:
        # Single-step Fisher scoring (no line search overhead).
        # Evaluate at prior mean (one_step_mean), so prior gradient is zero.
        log_lambda = log_intensity_func(one_step_mean)
        conditional_intensity = _safe_expected_count(
            log_lambda, dt, max_log_count=max_log_count
        )
        innovation = spike_indicator_t - conditional_intensity
        jacobian = grad_log_intensity(one_step_mean)
        # Likelihood gradient only; prior gradient = -P^{-1}(x - m) = 0 at x = m
        likelihood_gradient = jacobian.T @ innovation
        prior_gradient = jnp.zeros_like(likelihood_gradient)
        gradient = likelihood_gradient + prior_gradient
        fisher_info = jacobian.T @ (conditional_intensity[:, None] * jacobian)
        posterior_precision = symmetrize(prior_precision + fisher_info)
        post_cho = psd_cholesky(posterior_precision, diagonal_boost=diagonal_boost)
        posterior_mean = one_step_mean + jax.scipy.linalg.cho_solve(post_cho, gradient)
        n_line_search_failures = jnp.zeros((), dtype=jnp.int32)
    else:
        # Iterative Fisher scoring with line search, started at the prior mean
        # (zero iterations return the prior unchanged).
        posterior_mean, posterior_precision, n_line_search_failures = (
            _fisher_scoring_line_search(
                one_step_mean,
                prior_precision,
                _fisher_step_at,
                _neg_log_posterior,
                max_newton_iter,
                line_search_beta,
            )
        )
        post_cho = psd_cholesky(posterior_precision, diagonal_boost=diagonal_boost)

    # Posterior covariance from the same factor. No post-hoc stabilization
    # needed because posterior_precision is PSD by construction (sum of two
    # PSD matrices); psd_cholesky adds a small diagonal boost for conditioning.
    posterior_cov = symmetrize(jax.scipy.linalg.cho_solve(post_cho, identity))

    # Log-likelihood at posterior mode (approximate)
    log_lambda_mode = log_intensity_func(posterior_mean)
    conditional_intensity_mode = _safe_expected_count(
        log_lambda_mode, dt, max_log_count=max_log_count
    )
    log_likelihood = jnp.sum(
        jax.scipy.stats.poisson.logpmf(spike_indicator_t, conditional_intensity_mode)
    )

    if include_laplace_normalization:
        # Laplace correction:
        #   log p(y) approx log p(y|x*) - 0.5 delta' P_prior^{-1} delta
        #                   - 0.5 log|P_prior| + 0.5 log|P_post|
        # The Gaussian +/- d/2 log(2*pi) terms cancel between the normalized
        # prior density and the Laplace integral.
        #
        # Both log-determinants come from factors already computed: the prior
        # factor that gave prior_precision, and the posterior-precision factor
        # that gave posterior_cov (log|P_post| = -log|Lambda_post|), so the
        # normaliser is exact for the matrices the update actually used.
        delta = posterior_mean - one_step_mean
        quad = delta @ (prior_precision @ delta)
        logdet_prior = psd_logdet(prior_cho)
        logdet_post = -psd_logdet(post_cho)
        log_prior = -0.5 * quad - 0.5 * logdet_prior
        log_likelihood = log_likelihood + log_prior + 0.5 * logdet_post

    if return_line_search_failures:
        return posterior_mean, posterior_cov, log_likelihood, n_line_search_failures
    return posterior_mean, posterior_cov, log_likelihood


class GLMFamily(NamedTuple):
    """Observation family for the GLM Laplace update.

    A family encapsulates everything that differs between exponential-family
    observation models with a canonical link, so the Fisher-scoring skeleton in
    :func:`glm_laplace_update` is shared. ``eta`` is the linear predictor (the
    natural parameter): for Poisson it is the log-rate, for Bernoulli the logit.

    Attributes
    ----------
    mean : callable ``eta -> mu``
        Conditional mean (expected count for Poisson, probability for Bernoulli).
    fisher_weight : callable ``(eta, mu) -> w``
        Diagonal Fisher-information weight; the Fisher info is ``J' diag(w) J``,
        PSD by construction. (Poisson ``w = mu``; Bernoulli ``w = mu (1 - mu)``.)
    loglik_plugin : callable ``(y, eta, mu) -> scalar``
        Log-likelihood up to an additive constant, used for the line search.
    loglik_normalized : callable ``(y, eta, mu) -> scalar``
        Fully normalized log-likelihood, used for the returned marginal LL.
    """

    mean: Callable[[Array], Array]
    fisher_weight: Callable[[Array, Array], Array]
    loglik_plugin: Callable[[Array, Array, Array], Array]
    loglik_normalized: Callable[[Array, Array, Array], Array]


def poisson_family(dt: float, max_log_count: float = 20.0) -> GLMFamily:
    """Poisson family with log link, matching :func:`_point_process_laplace_update`.

    The mean is ``exp(eta) * dt`` with the same overflow clipping as the legacy
    update, so ``glm_laplace_update(..., poisson_family(dt))`` reproduces it.
    Returns a fresh ``GLMFamily`` (with fresh closures) on each call, so hoist it
    out of any ``jit``/``scan`` loop to keep the cache warm.
    """
    validate_scalar(dt, "dt", positive=True)

    def mean(eta: Array) -> Array:
        return _safe_expected_count(eta, dt, max_log_count=max_log_count)

    def fisher_weight(_eta: Array, mu: Array) -> Array:
        return mu

    def loglik_plugin(y: Array, _eta: Array, mu: Array) -> Array:
        # Poisson log-likelihood without the constant log(y!) term.
        return jnp.sum(y * jnp.log(mu) - mu)

    def loglik_normalized(y: Array, _eta: Array, mu: Array) -> Array:
        return jnp.sum(jax.scipy.stats.poisson.logpmf(y, mu))

    return GLMFamily(mean, fisher_weight, loglik_plugin, loglik_normalized)


# Saturation guard: clip the logit so sigmoid/softplus cannot overflow. This is a
# numerical guard on the *mean/log-likelihood only* — the Jacobian used by the
# update differentiates the unclipped eta_func, so the clip is not part of the
# model gradient (it is inert at realistic |eta|, where sigmoid is already
# saturated). Do not "fix" the asymmetry by clipping inside the Jacobian.
_BERNOULLI_ETA_CLIP = 30.0


def _bernoulli_mean(eta: Array) -> Array:
    return jax.nn.sigmoid(jnp.clip(eta, -_BERNOULLI_ETA_CLIP, _BERNOULLI_ETA_CLIP))


def _bernoulli_fisher_weight(_eta: Array, mu: Array) -> Array:
    return mu * (1.0 - mu)


def _bernoulli_loglik(y: Array, eta: Array, _mu: Array) -> Array:
    # Bernoulli-logit log-pmf: y * eta - softplus(eta). Already normalized, so the
    # plug-in and normalized log-likelihoods coincide.
    eta_clipped = jnp.clip(eta, -_BERNOULLI_ETA_CLIP, _BERNOULLI_ETA_CLIP)
    return jnp.sum(y * eta_clipped - jax.nn.softplus(eta_clipped))


#: Bernoulli observation family with logit link (spikes are 0/1 per bin).
BERNOULLI_LOGIT_FAMILY = GLMFamily(
    mean=_bernoulli_mean,
    fisher_weight=_bernoulli_fisher_weight,
    loglik_plugin=_bernoulli_loglik,
    loglik_normalized=_bernoulli_loglik,
)


@overload
def glm_laplace_update(
    one_step_mean: ArrayLike,
    one_step_cov: ArrayLike,
    observations: ArrayLike,
    eta_func: Callable[[Array], Array],
    family: GLMFamily,
    diagonal_boost: float = ...,
    grad_eta_func: Callable[[Array], Array] | None = ...,
    include_laplace_normalization: bool = ...,
    max_newton_iter: int = ...,
    line_search_beta: float = ...,
    return_line_search_failures: Literal[False] = ...,
) -> tuple[Array, Array, Array]: ...


@overload
def glm_laplace_update(
    one_step_mean: ArrayLike,
    one_step_cov: ArrayLike,
    observations: ArrayLike,
    eta_func: Callable[[Array], Array],
    family: GLMFamily,
    diagonal_boost: float = ...,
    grad_eta_func: Callable[[Array], Array] | None = ...,
    include_laplace_normalization: bool = ...,
    max_newton_iter: int = ...,
    line_search_beta: float = ...,
    *,
    return_line_search_failures: Literal[True],
) -> tuple[Array, Array, Array, Array]: ...


def glm_laplace_update(
    one_step_mean: ArrayLike,
    one_step_cov: ArrayLike,
    observations: ArrayLike,
    eta_func: Callable[[Array], Array],
    family: GLMFamily,
    diagonal_boost: float = 0.0,
    grad_eta_func: Callable[[Array], Array] | None = None,
    include_laplace_normalization: bool = True,
    max_newton_iter: int = 3,
    line_search_beta: float = 0.5,
    return_line_search_failures: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
    """Family-generic Laplace measurement update via Fisher scoring.

    Generalizes :func:`_point_process_laplace_update` to any :class:`GLMFamily`
    with a canonical link; with ``poisson_family(dt)`` it reproduces the legacy
    Poisson update bit-for-bit (see ``test_glm_laplace``). The posterior precision
    ``prior_precision + J' diag(w) J`` is PSD by construction, so no eigenvalue
    stabilization is needed.

    Parameters
    ----------
    one_step_mean : ArrayLike, shape (n_latent,)
        Predicted mean ``A @ m_{t-1}``.
    one_step_cov : ArrayLike, shape (n_latent, n_latent)
        Predicted covariance ``A @ P_{t-1} @ A.T + Q``.
    observations : ArrayLike, shape (n_obs,)
        Observed counts (Poisson) or 0/1 indicators (Bernoulli) at this bin.
    eta_func : callable ``x -> eta``
        Maps state (n_latent,) to the linear predictor (n_obs,). May be nonlinear;
        the Jacobian is taken with ``jax.jacfwd`` if ``grad_eta_func`` is None (pass
        a constant ``grad_eta_func`` when ``eta_func`` is linear, to skip jacfwd).
    family : GLMFamily
        Observation family (e.g. :data:`BERNOULLI_LOGIT_FAMILY`).
    diagonal_boost, grad_eta_func, include_laplace_normalization, max_newton_iter,
    line_search_beta, return_line_search_failures
        As in :func:`_point_process_laplace_update`.

    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
    posterior_cov : Array, shape (n_latent, n_latent)
    log_likelihood : Array
        Approximate ``log p(y_t | y_{1:t-1})`` (Laplace expansion) if normalized.
    n_line_search_failures : Array, optional
        Only with ``return_line_search_failures=True`` (int32 scalar): the
        Fisher iterations whose backtracking was exhausted (see
        :func:`_fisher_scoring_line_search`).
    """
    one_step_mean = jnp.asarray(one_step_mean)
    one_step_cov = jnp.asarray(one_step_cov)
    observations = jnp.asarray(observations)
    if grad_eta_func is None:
        grad_eta_func = jax.jacfwd(eta_func)
    grad_eta = grad_eta_func

    n_latent = one_step_mean.shape[0]
    identity = jnp.eye(n_latent, dtype=one_step_cov.dtype)
    # Factor the prior covariance once: the precision (used by the Fisher
    # steps and the quadratic form) and the log-determinant of the Laplace
    # normaliser both come from this factor, so they see the same jittered
    # matrix (psd_cholesky's scale-relative diagonal shift).
    prior_cho = psd_cholesky(one_step_cov, diagonal_boost=diagonal_boost)
    prior_precision = jax.scipy.linalg.cho_solve(prior_cho, identity)

    def _neg_log_posterior(x: Array) -> Array:
        eta = eta_func(x)
        mu = family.mean(eta)
        log_lik = family.loglik_plugin(observations, eta, mu)
        delta = x - one_step_mean
        log_prior = -0.5 * delta @ (prior_precision @ delta)
        return -(log_lik + log_prior)

    def _fisher_step_at(x: Array) -> tuple[Array, Array, Array]:
        eta = eta_func(x)
        mu = family.mean(eta)
        innovation = observations - mu
        jacobian = grad_eta(x)
        likelihood_gradient = jacobian.T @ innovation
        prior_gradient = -prior_precision @ (x - one_step_mean)
        gradient = likelihood_gradient + prior_gradient
        weight = family.fisher_weight(eta, mu)
        fisher_info = jacobian.T @ (weight[:, None] * jacobian)
        post_prec = symmetrize(prior_precision + fisher_info)
        delta = psd_solve(post_prec, gradient, diagonal_boost=diagonal_boost)
        return delta, post_prec, gradient

    if max_newton_iter == 1:
        # Single Fisher step from the prior mean (prior gradient is zero there).
        eta = eta_func(one_step_mean)
        mu = family.mean(eta)
        innovation = observations - mu
        jacobian = grad_eta(one_step_mean)
        likelihood_gradient = jacobian.T @ innovation
        prior_gradient = jnp.zeros_like(likelihood_gradient)
        gradient = likelihood_gradient + prior_gradient
        weight = family.fisher_weight(eta, mu)
        fisher_info = jacobian.T @ (weight[:, None] * jacobian)
        posterior_precision = symmetrize(prior_precision + fisher_info)
        post_cho = psd_cholesky(posterior_precision, diagonal_boost=diagonal_boost)
        posterior_mean = one_step_mean + jax.scipy.linalg.cho_solve(post_cho, gradient)
        n_line_search_failures = jnp.zeros((), dtype=jnp.int32)
    else:
        posterior_mean, posterior_precision, n_line_search_failures = (
            _fisher_scoring_line_search(
                one_step_mean,
                prior_precision,
                _fisher_step_at,
                _neg_log_posterior,
                max_newton_iter,
                line_search_beta,
            )
        )
        post_cho = psd_cholesky(posterior_precision, diagonal_boost=diagonal_boost)

    posterior_cov = symmetrize(jax.scipy.linalg.cho_solve(post_cho, identity))

    eta_mode = eta_func(posterior_mean)
    mu_mode = family.mean(eta_mode)
    log_likelihood = family.loglik_normalized(observations, eta_mode, mu_mode)

    if include_laplace_normalization:
        # Log-determinants from the prior / posterior-precision factors, as
        # in _point_process_laplace_update.
        delta = posterior_mean - one_step_mean
        quad = delta @ (prior_precision @ delta)
        logdet_prior = psd_logdet(prior_cho)
        logdet_post = -psd_logdet(post_cho)
        log_prior = -0.5 * quad - 0.5 * logdet_prior
        log_likelihood = log_likelihood + log_prior + 0.5 * logdet_post

    if return_line_search_failures:
        return posterior_mean, posterior_cov, log_likelihood, n_line_search_failures
    return posterior_mean, posterior_cov, log_likelihood


@overload
def stochastic_point_process_filter(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = ...,
    max_log_count: float = ...,
    validate_inputs: bool = ...,
    block_n_neurons: int | None = ...,
    block_size: int | None = ...,
    force_dense: bool = ...,
    max_newton_iter: int = ...,
    return_block_covariances: Literal[False] = ...,
) -> tuple[Array, Array, Array]: ...


@overload
def stochastic_point_process_filter(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = ...,
    max_log_count: float = ...,
    validate_inputs: bool = ...,
    block_n_neurons: int | None = ...,
    block_size: int | None = ...,
    force_dense: bool = ...,
    max_newton_iter: int = ...,
    *,
    return_block_covariances: Literal[True],
) -> tuple[Array, Array | BlockDiagonalCovariance, Array]: ...


@overload
def stochastic_point_process_filter(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = ...,
    max_log_count: float = ...,
    validate_inputs: bool = ...,
    block_n_neurons: int | None = ...,
    block_size: int | None = ...,
    force_dense: bool = ...,
    max_newton_iter: int = ...,
    return_block_covariances: bool = ...,
) -> tuple[Array, Array | BlockDiagonalCovariance, Array]: ...


def stochastic_point_process_filter(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
    validate_inputs: bool = True,
    block_n_neurons: int | None = None,
    block_size: int | None = None,
    force_dense: bool = False,
    max_newton_iter: int = 3,
    return_block_covariances: bool = False,
) -> tuple[Array, Array | BlockDiagonalCovariance, Array]:
    """Applies a Stochastic State Point Process Filter (SSPPF).

    This filter estimates a time-varying latent state ($x_k$) based on
    point process observations ($y_k$). It assumes a linear Gaussian state
    transition and a point process observation model where the conditional
    intensity $\\lambda_k$ depends on the state.

    $$ x_k = A x_{k-1} + w_k, \\quad w_k \\sim N(0, Q) $$
    $$ \\lambda_{n,k} = f(x_k, Z_k)_n $$
    $$ y_{n,k} \\sim \\text{Poisson}(\\lambda_{n,k} \\Delta t) $$

    The filter uses a local Gaussian approximation (Laplace-EKF approach)
    at each update step, built by ``max_newton_iter`` Fisher-scoring
    iterations per time bin (default 3): the update uses the *expected* Hessian
    $E[\\nabla^2 \\log p]$ of the Poisson log-likelihood (equivalently, the
    negative of the inverse-intensity-weighted outer product) rather than
    the observed Hessian, which is what makes this Fisher scoring rather
    than Newton-Raphson. The practical effect is that the Hessian is always
    negative semidefinite, so no damping or trust-region safeguarding is
    needed for PSD of the covariance update.

    With ``max_newton_iter > 1`` each iteration is gated by a fixed-length
    Armijo backtracking scan (:func:`_fisher_scoring_line_search`); when more
    than 10% of the bins exhaust it, one warning is logged per call.
    ``max_newton_iter == 1`` is a single Fisher step without line search. For a true Newton
    step with observed Hessian + Armijo line search, see
    :func:`switching_point_process._single_neuron_glm_step_second_order`.

    Multi-Neuron Support
    --------------------
    The filter supports multiple neurons sharing a common latent state:

    - spike_indicator: (n_time, n_neurons) - spike counts for each neuron
    - log_conditional_intensity(Z_k, x_k) should return (n_neurons,)

    For backwards compatibility, single-neuron inputs work as before:
    - spike_indicator: (n_time,) is internally promoted to (n_time, 1)
    - scalar log-intensity output is wrapped to (1,)

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
        Initial mean of the latent state ($x_0$).
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance of the latent state ($P_0$).
    design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_params)
        Design matrix ($Z_k$) used in the intensity function.
        Shape depends on the log_conditional_intensity function.
        For multi-neuron with default linear intensity, use (n_time, n_neurons, n_params).
        On the block-diagonal path -- ``block_n_neurons`` / ``block_size``
        given, ``force_dense=False`` and the default linear
        :func:`log_conditional_intensity` -- the shared per-neuron basis
        ``Z_base`` of shape ``(n_time, block_size)`` may be passed instead of
        the block-expanded matrix; the block filter works from ``Z_base``
        directly. Any other configuration takes the dense path, which needs
        the design matrix in the form ``log_conditional_intensity`` expects.
    spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
        Observed spike counts or indicators ($y_k$).
        For single neuron: (n_time,)
        For multiple neurons: (n_time, n_neurons)
    dt : float
        Time step size ($\\Delta t$).
    transition_matrix : ArrayLike, shape (n_params, n_params)
        State transition matrix ($A$).
    process_cov : ArrayLike, shape (n_params, n_params)
        Process noise covariance ($Q$).
    log_conditional_intensity : callable
        Function `log_lambda(Z_k, x_k)` returning the log conditional
        intensity. Should return (n_neurons,) array for multi-neuron case,
        or scalar for single-neuron.
    include_laplace_normalization : bool, default=True
        If True, include Laplace normalization and prior terms in the
        marginal log-likelihood. If False, return the plug-in log-likelihood
        at the posterior mode without normalization.
    max_log_count : float, default=20.0
        Ceiling on ``log(rate * dt)`` used by the Laplace update's safe
        exponentiation to prevent overflow on pathological spike counts.
        Pass a physiologically motivated value (e.g.
        ``log(max_firing_rate_hz * dt)``) to catch outlier bins that would
        otherwise drive the Fisher step into a catastrophic region. The
        default of 20.0 corresponds to ~2.4e9 Hz at dt=0.2s and is kept
        for backwards compatibility with existing callers.
    return_block_covariances : bool, default=False
        Block-diagonal path only. If True, return the filtered covariance as
        a :class:`BlockDiagonalCovariance` holding the per-neuron
        ``(n_neurons, n_time, block_size, block_size)`` blocks instead of
        materialising the dense ``(n_time, n_params, n_params)`` array
        (``n_neurons`` times larger, >99% structural zeros). Ignored on the
        dense path, which always returns dense arrays.

    Returns
    -------
    posterior_mean : Array, shape (n_time, n_params)
        Filtered posterior means ($x_{k|k}$).
    posterior_variance : Array, shape (n_time, n_params, n_params)
        Filtered posterior covariances ($P_{k|k}$); a
        :class:`BlockDiagonalCovariance` of the same dense shape when
        ``return_block_covariances`` applies.
    marginal_log_likelihood : Array
        Total log-likelihood of the observations given the model (scalar array).

    Notes
    -----
    For multiple neurons, the observation log-likelihood term is the sum
    of independent Poisson log-pmfs:
        log p(y_t | x_t) = sum_n log Poisson(y_{n,t} | lambda_{n,t} * dt)
    When ``include_laplace_normalization`` is True, prior and normalization
    terms are added to approximate the marginal log-likelihood.

    The filter aggregates information from all neurons to update the shared
    latent state. More neurons provide more information, reducing posterior
    uncertainty.

    Numerical precision
    -------------------
    This filter requires **float64** for reliable long-sequence fits.
    In float32, accumulated roundoff in the covariance propagation can
    drive the posterior covariance below PSD after ~250-5000 time bins
    (depending on ``init_covariance_params`` conditioning), producing
    silent NaN output. Enable float64 BEFORE importing this module::

        import jax
        jax.config.update("jax_enable_x64", True)

    The entry-point validation (``validate_inputs=True``, default) warns
    when the dtype-and-conditioning combination is at risk and raises
    when ``init_covariance_params`` is not positive definite.

    References
    ----------
    [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).
    """
    # Convert to arrays
    init_mean_params = jnp.asarray(init_mean_params)
    init_covariance_params = jnp.asarray(init_covariance_params)
    design_matrix = jnp.asarray(design_matrix)
    spike_indicator = jnp.asarray(spike_indicator)
    transition_matrix = jnp.asarray(transition_matrix)
    process_cov = jnp.asarray(process_cov)

    # Numerical sanity check BEFORE the block dispatch below. The block path
    # (large-state / long-T) is exactly where these checks matter, so they
    # must run before dispatch, not only on the dense path. Gated behind
    # ``validate_inputs`` so tight inner loops (e.g. SGD) can pass False after
    # a single validation at the top of fit_sgd. stacklevel=4 so the warning
    # points at the user's call site: user -> fit_sgd -> _sgd_loss_fn ->
    # stochastic_point_process_filter -> _validate. Shape handling below is
    # static and always runs.
    if validate_inputs:
        _validate_public_inputs(
            dt,
            init_mean_params,
            init_covariance_params,
            design_matrix,
            spike_indicator,
            transition_matrix,
            process_cov,
            filter_name="stochastic_point_process_filter",
            stacklevel=4,
        )

    # Block-diagonal dispatch (opt-in via block_n_neurons / block_size).
    # The caller is responsible for verifying the block structure ONCE at
    # fit entry time (outside jax.jit / jax.grad; see the
    # BlockDiagonalStructure contract and _block_diagonal_parameters_ok) to
    # determine these integers. Inside
    # the jit boundary they are Python constants, not traced, so we can use
    # them for static slicing of the traced (init_mean, init_cov, A, Q,
    # design_matrix-or-Z_base) arrays into per-neuron factors. This is safe
    # because block extraction is pure array slicing — no host syncs.
    use_block_dispatch = (
        block_n_neurons is not None
        and block_size is not None
        and not force_dense
        and _uses_default_linear_log_intensity(log_conditional_intensity)
    )
    if use_block_dispatch:
        structure = _validated_dispatch_block_structure(
            init_mean_params,
            init_covariance_params,
            transition_matrix,
            process_cov,
            design_matrix,
            spike_indicator,
            block_n_neurons,
            block_size,
        )
        return _stochastic_point_process_filter_block_diagonal(
            structure,
            spike_indicator,
            dt,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
            max_newton_iter=max_newton_iter,
            return_block_covariances=return_block_covariances,
        )

    # (Numerical sanity check already ran above, before block dispatch.)

    # Promote single-neuron spike_indicator to (n_time, 1) for consistent handling
    single_neuron = spike_indicator.ndim == 1
    if single_neuron:
        spike_indicator = spike_indicator[:, None]

    result = _stochastic_point_process_filter_impl(
        init_mean_params,
        init_covariance_params,
        design_matrix,
        spike_indicator,
        transition_matrix,
        process_cov,
        dt=dt,
        log_conditional_intensity=log_conditional_intensity,
        include_laplace_normalization=include_laplace_normalization,
        max_log_count=max_log_count,
        max_newton_iter=max_newton_iter,
    )
    filtered_mean, filtered_cov, marginal_log_likelihood, n_failed_bins = result
    _warn_line_search_failures(
        n_failed_bins,
        spike_indicator.shape[0],
        max_newton_iter,
        "stochastic_point_process_filter",
    )
    return filtered_mean, filtered_cov, marginal_log_likelihood


@functools.partial(
    typed_jit,
    static_argnames=[
        "log_conditional_intensity",
        "include_laplace_normalization",
        "max_newton_iter",
    ],
)
def _stochastic_point_process_filter_impl(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    *,
    dt: float,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
    max_newton_iter: int = 3,
) -> tuple[Array, Array, Array, Array]:
    """JIT-compiled inner implementation of the point process filter.

    Returns the filtered means, covariances and marginal log-likelihood, plus
    the number of time bins whose Fisher-scoring line search was exhausted
    (always 0 for ``max_newton_iter <= 1``).
    """
    # Coerce the ArrayLike inputs to Array (identity on the arrays the public
    # entry point already passes) so the array algebra and the scan carry below
    # type as Array rather than the ArrayLike scalar union.
    init_mean_params = jnp.asarray(init_mean_params)
    init_covariance_params = jnp.asarray(init_covariance_params)
    design_matrix = jnp.asarray(design_matrix)
    spike_indicator = jnp.asarray(spike_indicator)
    transition_matrix = jnp.asarray(transition_matrix)
    process_cov = jnp.asarray(process_cov)

    # The scan carry must keep one dtype, so run in the common floating dtype
    # of everything that enters the state update (e.g. float32 init moments
    # with float64 dynamics run in float64). The design matrix and spikes are
    # left as given -- a custom log-intensity may index with them -- but
    # their floating dtypes take part in the promotion.
    dtype = _common_float_dtype(
        init_mean_params,
        init_covariance_params,
        transition_matrix,
        process_cov,
        design_matrix,
        spike_indicator,
    )
    init_mean_params = init_mean_params.astype(dtype)
    init_covariance_params = init_covariance_params.astype(dtype)
    transition_matrix = transition_matrix.astype(dtype)
    process_cov = process_cov.astype(dtype)

    # Pre-compute gradient function outside the scan.
    def _log_intensity_with_design(design_matrix_t: Array, x: Array) -> Array:
        log_lambda = log_conditional_intensity(design_matrix_t, x)
        return jnp.atleast_1d(log_lambda)

    _grad_log_intensity: Callable[[Array, Array], Array] = jax.jacfwd(
        _log_intensity_with_design, argnums=1
    )

    def _step(
        params_prev: tuple[Array, Array, Array, Array],
        args: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array]]:
        """Point Process Adaptive Filter update step."""
        mean_prev, variance_prev, marginal_log_likelihood, n_failed_bins = params_prev
        design_matrix_t, spike_indicator_t = args

        one_step_mean = transition_matrix @ mean_prev
        variance_prev_sym = symmetrize(variance_prev)
        one_step_covariance = symmetrize(
            transition_matrix @ variance_prev_sym @ transition_matrix.T + process_cov
        )

        def log_intensity_func(x: Array) -> Array:
            return _log_intensity_with_design(design_matrix_t, x)

        def grad_log_intensity_func(x: Array) -> Array:
            return _grad_log_intensity(design_matrix_t, x)

        posterior_mean, posterior_covariance, log_lik, n_failed = (
            _point_process_laplace_update(
                one_step_mean,
                one_step_covariance,
                spike_indicator_t,
                dt,
                log_intensity_func,
                grad_log_intensity_func=grad_log_intensity_func,
                include_laplace_normalization=include_laplace_normalization,
                max_log_count=max_log_count,
                max_newton_iter=max_newton_iter,
                return_line_search_failures=True,
            )
        )

        marginal_log_likelihood += log_lik
        n_failed_bins += (n_failed > 0).astype(jnp.int32)

        return (
            posterior_mean,
            posterior_covariance,
            marginal_log_likelihood,
            n_failed_bins,
        ), (
            posterior_mean,
            posterior_covariance,
        )

    marginal_log_likelihood = jnp.zeros((), dtype=dtype)
    (
        (_, _, marginal_log_likelihood, n_failed_bins),
        (
            filtered_mean,
            filtered_cov,
        ),
    ) = jax.lax.scan(
        _step,
        (
            init_mean_params,
            init_covariance_params,
            marginal_log_likelihood,
            jnp.zeros((), dtype=jnp.int32),
        ),
        (design_matrix, spike_indicator),
    )

    return filtered_mean, filtered_cov, marginal_log_likelihood, n_failed_bins


@functools.partial(
    typed_jit,
    static_argnames=("include_laplace_normalization", "max_newton_iter"),
)
def _block_diagonal_forward_core(
    A_blocks: Array,
    Q_blocks: Array,
    init_means_per_neuron: Array,
    init_covs_per_neuron: Array,
    Z_base: Array,
    spike_indicator: Array,
    dt: float,
    *,
    include_laplace_normalization: bool,
    max_log_count: float,
    max_newton_iter: int,
) -> tuple[Array, Array, Array, Array]:
    """JIT-compiled per-neuron forward Laplace-EKF scan.

    Takes the array fields of a :class:`BlockDiagonalStructure` explicitly,
    so its Python-int fields never become traced, and the option flags as
    static arguments. Each neuron's scan uses its own ``A_blocks[j]`` and
    ``Q_blocks[j]`` (vmapped alongside its initial state and spikes). The vmapped ``lax.scan`` is therefore traced and
    compiled once per (shapes, dtypes, options) and reused across calls --
    every EM iteration of ``PlaceFieldModel.fit`` -- instead of being
    re-traced on each call. ``dt`` and ``max_log_count`` are ordinary
    (traced) scalars, so changing their values does not recompile.

    See :func:`_run_forward_block_diagonal` for the returned arrays.
    """
    # One carry dtype for every neuron's scan (see
    # _stochastic_point_process_filter_impl).
    dtype = _common_float_dtype(
        A_blocks,
        Q_blocks,
        init_means_per_neuron,
        init_covs_per_neuron,
        Z_base,
        spike_indicator,
    )
    A_blocks = A_blocks.astype(dtype)
    Q_blocks = Q_blocks.astype(dtype)
    init_means_per_neuron = init_means_per_neuron.astype(dtype)
    init_covs_per_neuron = init_covs_per_neuron.astype(dtype)
    Z_base = Z_base.astype(dtype)

    def _step_one_neuron(
        A_j: Array,
        Q_j: Array,
        carry: tuple[Array, Array, Array, Array],
        args: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array]]:
        mean_prev, cov_prev, ll_acc, n_failed_bins = carry
        z_row_t, y_t = args
        one_step_mean = A_j @ mean_prev
        cov_prev_sym = symmetrize(cov_prev)
        one_step_cov = symmetrize(A_j @ cov_prev_sym @ A_j.T + Q_j)
        spike_as_vec = jnp.atleast_1d(y_t)

        def _lin(x_block: Array) -> Array:
            return jnp.atleast_1d(z_row_t @ x_block)

        def _grad(_x_block: Array) -> Array:
            # Analytical gradient: d(log_intensity)/dx = z_row_t.
            # Safe because the public filter / smoother only dispatch to
            # the block path when the log-intensity is the default linear
            # one (the ``_uses_default_linear_log_intensity`` term of
            # ``use_block_dispatch``); a nonlinear intensity (e.g.
            # PositionDecoder's KDE rate map) always takes the dense path.
            return z_row_t[None, :]

        post_mean, post_cov, log_lik_step, n_failed = _point_process_laplace_update(
            one_step_mean,
            one_step_cov,
            spike_as_vec,
            dt,
            _lin,
            grad_log_intensity_func=_grad,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
            max_newton_iter=max_newton_iter,
            return_line_search_failures=True,
        )
        return (
            (
                post_mean,
                post_cov,
                ll_acc + log_lik_step,
                n_failed_bins + (n_failed > 0).astype(jnp.int32),
            ),
            (post_mean, post_cov),
        )

    def _run_one_neuron(
        A_j: Array,
        Q_j: Array,
        init_mean_j: Array,
        init_cov_j: Array,
        spikes_j: Array,
    ) -> tuple[Array, Array, Array, Array]:
        init_carry = (
            init_mean_j,
            init_cov_j,
            jnp.array(0.0, dtype=init_mean_j.dtype),
            jnp.zeros((), dtype=jnp.int32),
        )
        (_, _, ll_j, n_failed_j), (means_j, covs_j) = jax.lax.scan(
            functools.partial(_step_one_neuron, A_j, Q_j),
            init_carry,
            (Z_base, spikes_j),
        )
        return means_j, covs_j, ll_j, n_failed_j

    return jax.vmap(_run_one_neuron, in_axes=(0, 0, 0, 0, 1))(
        A_blocks, Q_blocks, init_means_per_neuron, init_covs_per_neuron, spike_indicator
    )


def _run_forward_block_diagonal(
    structure: BlockDiagonalStructure,
    spike_indicator: Array,
    dt: float,
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
    max_newton_iter: int = 3,
) -> tuple[Array, Array, Array, Array]:
    """Run the per-neuron forward Laplace-EKF filter in block form.

    Thin wrapper around the compiled :func:`_block_diagonal_forward_core`
    that unpacks a ``BlockDiagonalStructure``. Shared by the block filter
    and (through :func:`_block_diagonal_smoother_core`) the block smoother.

    Parameters
    ----------
    structure : BlockDiagonalStructure
    spike_indicator : Array, shape (n_time, n_neurons)
    dt : float
    include_laplace_normalization : bool, default=True
    max_log_count : float, default=20.0
    max_newton_iter : int, default=3

    Returns
    -------
    fwd_means : Array, shape (n_neurons, n_time, block_size)
        Per-neuron forward posterior means at each time step.
    fwd_covs : Array, shape (n_neurons, n_time, block_size, block_size)
        Per-neuron forward posterior covariances.
    lls_per_neuron : Array, shape (n_neurons,)
        Per-neuron marginal log-likelihood (sum over time steps).
    n_failed_per_neuron : Array, shape (n_neurons,)
        Per-neuron number of time bins whose Fisher-scoring line search was
        exhausted (0 for ``max_newton_iter <= 1``).
    """
    fwd = _block_diagonal_forward_core(
        structure.A_blocks,
        structure.Q_blocks,
        structure.init_means_per_neuron,
        structure.init_covs_per_neuron,
        structure.Z_base,
        spike_indicator,
        dt,
        include_laplace_normalization=include_laplace_normalization,
        max_log_count=max_log_count,
        max_newton_iter=max_newton_iter,
    )
    return fwd


@functools.partial(
    typed_jit,
    static_argnames=("include_laplace_normalization", "max_newton_iter"),
)
def _block_diagonal_smoother_core(
    A_blocks: Array,
    Q_blocks: Array,
    init_means_per_neuron: Array,
    init_covs_per_neuron: Array,
    Z_base: Array,
    spike_indicator: Array,
    dt: float,
    *,
    include_laplace_normalization: bool,
    max_log_count: float,
    max_newton_iter: int,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """JIT-compiled per-neuron forward filter plus backward RTS pass.

    Same compile-once contract as :func:`_block_diagonal_forward_core`:
    the forward core is inlined, so the forward and backward passes run as
    one executable, compiled once per (shapes, dtypes, options) and reused
    on every call.

    Returns ``(fwd_means, fwd_covs, lls_per_neuron, n_failed_per_neuron,
    smoother_means, smoother_covs, smoother_cross_covs)``, all per neuron:
    means are ``(n_neurons, n_time, nb)``, covariances
    ``(n_neurons, n_time, nb, nb)`` and cross-covariances
    ``(n_neurons, n_time - 1, nb, nb)``; ``n_failed_per_neuron`` counts the
    bins whose line search was exhausted.
    """
    fwd_means, fwd_covs, lls_per_neuron, n_failed_per_neuron = (
        _block_diagonal_forward_core(
            A_blocks,
            Q_blocks,
            init_means_per_neuron,
            init_covs_per_neuron,
            Z_base,
            spike_indicator,
            dt,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
            max_newton_iter=max_newton_iter,
        )
    )

    # Backward RTS smoother pass per neuron. _kalman_smoother_update is
    # observation-model-agnostic — it uses only A, Q, and the filtered
    # Gaussian moments. For block-diagonal A and Q, the backward pass
    # decomposes into independent per-neuron smoothers (neuron j with its
    # own A_j, Q_j), exactly matching the dense filter's backward pass on
    # the block-diagonal problem.
    def _backward_step(
        A_j: Array,
        Q_j: Array,
        carry: tuple[Array, Array],
        args: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        next_smoother_mean, next_smoother_cov = carry
        filter_mean, filter_cov = args
        sm, sc, scc = _kalman_smoother_update(
            next_smoother_mean,
            next_smoother_cov,
            filter_mean,
            filter_cov,
            Q_j,
            A_j,
        )
        return (sm, sc), (sm, sc, scc)

    def _run_backward_one_neuron(
        A_j: Array, Q_j: Array, means_j: Array, covs_j: Array
    ) -> tuple[Array, Array, Array]:
        # Initial carry: the last-time-step filtered posterior.
        (_, _), (sm_rev, sc_rev, scc_rev) = jax.lax.scan(
            functools.partial(_backward_step, A_j, Q_j),
            (means_j[-1], covs_j[-1]),
            (means_j[:-1], covs_j[:-1]),
            reverse=True,
        )
        # Append the last time step's filter posterior (no backward update)
        sm_full = jnp.concatenate((sm_rev, means_j[-1][None]))
        sc_full = jnp.concatenate((sc_rev, covs_j[-1][None]))
        return sm_full, sc_full, scc_rev

    smoother_means, smoother_covs, smoother_cross_covs = jax.vmap(
        _run_backward_one_neuron
    )(A_blocks, Q_blocks, fwd_means, fwd_covs)
    return (
        fwd_means,
        fwd_covs,
        lls_per_neuron,
        n_failed_per_neuron,
        smoother_means,
        smoother_covs,
        smoother_cross_covs,
    )


def _assemble_block_diagonal_covs(covs_per_neuron: Array) -> Array:
    """Reassemble ``(n_neurons, n_time, nb, nb)`` blocks into the dense
    ``(n_time, n_state, n_state)`` block-diagonal array.

    One scatter per time step (vmapped over the time axis of the block
    storage directly, without an intermediate transposed copy). Used only
    when a caller asks for dense covariances; the block path otherwise hands
    out a :class:`BlockDiagonalCovariance`.
    """
    return jax.vmap(_assemble_block_diagonal_matrix, in_axes=1)(covs_per_neuron)


def _concatenate_neuron_means(means_per_neuron: Array) -> Array:
    """``(n_neurons, n_time, nb)`` per-neuron means -> ``(n_time, n_neurons * nb)``."""
    n_neurons, n_time, nb = means_per_neuron.shape
    return jnp.transpose(means_per_neuron, (1, 0, 2)).reshape(n_time, n_neurons * nb)


def _package_block_covs(
    covs_per_neuron: Array, return_block_covariances: bool
) -> Array | BlockDiagonalCovariance:
    """Per-neuron covariance blocks as a container or as the dense array."""
    if return_block_covariances:
        return BlockDiagonalCovariance(covs_per_neuron)
    return _assemble_block_diagonal_covs(covs_per_neuron)


def _stochastic_point_process_filter_block_diagonal(
    structure: BlockDiagonalStructure,
    spike_indicator: Array,
    dt: float,
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
    max_newton_iter: int = 3,
    return_block_covariances: bool = False,
) -> tuple[Array, Array | BlockDiagonalCovariance, Array]:
    """Block-diagonal Laplace-EKF filter via vmapped per-neuron scans.

    Consumes a ``BlockDiagonalStructure`` (from
    ``_build_block_structure_from_traced``) and runs ``n_neurons`` independent
    filters in parallel via ``jax.vmap``. Each per-neuron filter operates
    at ``d=block_size`` state dimension rather than ``d=n_neurons*block_size``,
    giving ~``n_neurons^2`` speedup on the per-step Cholesky.

    Mathematical equivalence
    ------------------------
    For a block-diagonal problem, the dense filter's posterior decomposes
    into ``n_neurons`` independent per-neuron posteriors. Specifically:

    - ``A @ P @ A^T + Q`` is block-diagonal → each block is
      ``A_j @ P_j @ A_j^T + Q_j`` computed independently, with neuron
      ``j``'s own diagonal blocks ``A_j``, ``Q_j``.
    - ``J^T diag(rate*dt) J`` (Fisher info) for neuron j depends only
      on that neuron's basis slice and its own weights, so it contributes
      only to block j of the posterior precision.
    - The Laplace normalization's ``logdet(P_prior)`` for a block-
      diagonal matrix equals ``sum_j logdet(P_j_prior)``, so per-neuron
      logdet contributions sum correctly to the full-problem logdet.
    - Same for ``logdet(P_post)``.
    - The quadratic form ``(x* - m)^T P_prior^{-1} (x* - m)`` also
      decomposes into a sum over neurons because ``P_prior`` is
      block-diagonal.

    Therefore: per-neuron ``log p(y_t^j | y_{1:t-1})`` contributions
    sum to the full-problem marginal log-likelihood.

    This equivalence is exact for a single Fisher step
    (``max_newton_iter == 1``). For ``max_newton_iter > 1`` (the default is
    3) the two paths differ only when a line search backtracks: the dense update runs one *global* backtracking line search
    (a single step size gating all neurons on the summed neg-log-posterior),
    whereas this block path backtracks *per neuron* independently. The
    per-neuron search is arguably better conditioned, but the resulting
    means and log-likelihoods no longer agree with the dense path to
    roundoff.

    Output shape compatibility
    --------------------------
    ``filtered_mean`` is always the concatenated ``(n_time, n_state)``
    array. ``filtered_cov`` is, by default, the dense
    ``(n_time, n_state, n_state)`` block-diagonal array reassembled from the
    per-neuron blocks, so this is a drop-in replacement for the dense
    filter; with ``return_block_covariances=True`` it is instead a
    :class:`BlockDiagonalCovariance` holding the per-neuron blocks -- the
    same dense ``shape``, but without the ``n_neurons``-fold zero padding.
    ``PlaceFieldModel`` uses the container; its consumers (EM M-step,
    rate maps, confidence intervals) only need block-local quantities.

    Parameters
    ----------
    structure : BlockDiagonalStructure
        Per-neuron factored filter problem.
    spike_indicator : Array, shape (n_time, n_neurons)
        Spike counts per neuron per time bin. Must be 2D (not 1D);
        callers with single-neuron problems should dispatch to the
        dense filter, not this function.
    dt : float
        Time bin width.
    include_laplace_normalization : bool, default=True
    max_log_count : float, default=20.0
    max_newton_iter : int, default=3
    return_block_covariances : bool, default=False
        Return ``filtered_cov`` as a :class:`BlockDiagonalCovariance`
        instead of the dense array.

    Returns
    -------
    filtered_mean : Array, shape (n_time, n_neurons * block_size)
        Concatenated per-neuron posterior means.
    filtered_cov : Array or BlockDiagonalCovariance
        Block-diagonal posterior covariance of dense shape
        ``(n_time, n_neurons * block_size, n_neurons * block_size)``.
    marginal_log_likelihood : Array, scalar
        Total log-likelihood summed across neurons.
    """
    means_per_neuron, covs_per_neuron, lls_per_neuron, n_failed_per_neuron = (
        _run_forward_block_diagonal(
            structure,
            spike_indicator,
            dt,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
            max_newton_iter=max_newton_iter,
        )
    )
    _warn_line_search_failures(
        jnp.sum(n_failed_per_neuron),
        int(spike_indicator.size),
        max_newton_iter,
        "stochastic_point_process_filter (block-diagonal path)",
    )
    # means_per_neuron: (n_neurons, n_time, nb)
    # covs_per_neuron: (n_neurons, n_time, nb, nb)
    # lls_per_neuron: (n_neurons,)
    filtered_mean = _concatenate_neuron_means(means_per_neuron)
    filtered_cov = _package_block_covs(covs_per_neuron, return_block_covariances)

    # Marginal log-likelihood: sum across neurons. For block-diagonal
    # problems, log p(y_t | y_{1:t-1}) decomposes as a sum of per-neuron
    # contributions — the Laplace normalization's logdet of a block-
    # diagonal matrix equals the sum of its blocks' logdets, and the
    # quadratic form similarly decomposes.
    return filtered_mean, filtered_cov, jnp.sum(lls_per_neuron)


def _stochastic_point_process_smoother_block_diagonal(
    structure: BlockDiagonalStructure,
    spike_indicator: Array,
    dt: float,
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
    return_filtered: bool = False,
    max_newton_iter: int = 3,
    return_block_covariances: bool = False,
) -> tuple[Array | BlockDiagonalCovariance, ...]:
    """Block-diagonal RTS smoother via vmapped per-neuron backward pass.

    Runs the block-diagonal forward filter and the standard RTS backward
    pass INDEPENDENTLY per neuron via jax.vmap, in one compiled call
    (:func:`_block_diagonal_smoother_core`). The smoother is
    observation-model-agnostic — it operates only on Gaussian moments —
    so the block-diagonal decomposition propagates through the backward
    pass without any additional algebra.

    Mathematical equivalence
    ------------------------
    The RTS backward step ``_kalman_smoother_update`` uses only:

    - Per-time-step filter mean and covariance (per-neuron in the block form)
    - ``transition_matrix`` (per-neuron ``A_blocks[j]``)
    - ``process_cov`` (per-neuron ``Q_blocks[j]``)

    All of these are block-diagonal by construction. The smoother gain
    ``J_t = P_{t|t} A^T (A P_{t|t} A^T + Q)^{-1}`` has block-diagonal
    structure when ``P_{t|t}``, ``A``, and ``Q`` are all block-diagonal,
    so the backward update decomposes neuron-by-neuron. Running the
    backward pass per-neuron is mathematically identical to running it
    on the full dense state *given the same filtered inputs*. The forward
    filter it consumes carries the same caveat as
    ``_stochastic_point_process_filter_block_diagonal``: the two paths
    agree to roundoff only for ``max_newton_iter == 1``; for
    ``max_newton_iter > 1`` the per-neuron vs global line search makes the
    filtered moments (and hence the smoothed output) differ from the dense
    path.

    Output shape compatibility
    --------------------------
    By default returns dense ``(n_time, n_state, n_state)`` covariances and
    ``(n_time - 1, n_state, n_state)`` cross-covariances reassembled from
    the per-neuron blocks, matching the dense smoother API. With
    ``return_block_covariances=True`` every covariance output is a
    :class:`BlockDiagonalCovariance` instead (same dense ``shape``, per-
    neuron block storage), which is what ``PlaceFieldModel`` consumes.

    Cross-cov index convention: ``_kalman_smoother_update`` returns
    ``J_t @ P_{t+1|T}``, the smoothed lag-one cross-cov ``P_{t, t+1|T}``
    (current-next). EM M-step consumers expect this convention.

    Parameters
    ----------
    structure : BlockDiagonalStructure
    spike_indicator : Array, shape (n_time, n_neurons)
    dt : float
    include_laplace_normalization : bool, default=True
    max_log_count : float, default=20.0
    return_filtered : bool, default=False
        If True, also return the filtered mean and covariance.
    max_newton_iter : int, default=3
    return_block_covariances : bool, default=False

    Returns
    -------
    smoother_mean : Array, shape (n_time, n_state)
    smoother_cov : Array or BlockDiagonalCovariance
        Dense shape ``(n_time, n_state, n_state)``.
    smoother_cross_cov : Array or BlockDiagonalCovariance
        Dense shape ``(n_time - 1, n_state, n_state)``.
    marginal_log_likelihood : Array, scalar
    filtered_mean, filtered_cov : optional, if return_filtered=True
    """
    (
        fwd_means,
        fwd_covs,
        lls_per_neuron,
        n_failed_per_neuron,
        smoother_means_per_neuron,
        smoother_covs_per_neuron,
        smoother_cross_covs_per_neuron,
    ) = _block_diagonal_smoother_core(
        structure.A_blocks,
        structure.Q_blocks,
        structure.init_means_per_neuron,
        structure.init_covs_per_neuron,
        structure.Z_base,
        spike_indicator,
        dt,
        include_laplace_normalization=include_laplace_normalization,
        max_log_count=max_log_count,
        max_newton_iter=max_newton_iter,
    )

    _warn_line_search_failures(
        jnp.sum(n_failed_per_neuron),
        int(spike_indicator.size),
        max_newton_iter,
        "stochastic_point_process_smoother (block-diagonal path)",
    )
    smoother_mean = _concatenate_neuron_means(smoother_means_per_neuron)
    smoother_cov = _package_block_covs(
        smoother_covs_per_neuron, return_block_covariances
    )
    smoother_cross_cov = _package_block_covs(
        smoother_cross_covs_per_neuron, return_block_covariances
    )
    marginal_ll = jnp.sum(lls_per_neuron)

    if return_filtered:
        return (
            smoother_mean,
            smoother_cov,
            smoother_cross_cov,
            marginal_ll,
            _concatenate_neuron_means(fwd_means),
            _package_block_covs(fwd_covs, return_block_covariances),
        )
    return smoother_mean, smoother_cov, smoother_cross_cov, marginal_ll


@overload
def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = ...,
    return_filtered: Literal[False] = ...,
    max_log_count: float = ...,
    validate_inputs: bool = ...,
    block_n_neurons: int | None = ...,
    block_size: int | None = ...,
    force_dense: bool = ...,
    max_newton_iter: int = ...,
    return_block_covariances: Literal[False] = ...,
) -> tuple[Array, Array, Array, Array]: ...


@overload
def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = ...,
    return_filtered: Literal[True] = ...,
    max_log_count: float = ...,
    validate_inputs: bool = ...,
    block_n_neurons: int | None = ...,
    block_size: int | None = ...,
    force_dense: bool = ...,
    max_newton_iter: int = ...,
    return_block_covariances: Literal[False] = ...,
) -> tuple[Array, Array, Array, Array, Array, Array]: ...


@overload
def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = ...,
    return_filtered: Literal[False] = ...,
    max_log_count: float = ...,
    validate_inputs: bool = ...,
    block_n_neurons: int | None = ...,
    block_size: int | None = ...,
    force_dense: bool = ...,
    max_newton_iter: int = ...,
    return_block_covariances: Literal[True] = ...,
) -> tuple[
    Array,
    Array | BlockDiagonalCovariance,
    Array | BlockDiagonalCovariance,
    Array,
]: ...


@overload
def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = ...,
    return_filtered: Literal[True] = ...,
    max_log_count: float = ...,
    validate_inputs: bool = ...,
    block_n_neurons: int | None = ...,
    block_size: int | None = ...,
    force_dense: bool = ...,
    max_newton_iter: int = ...,
    return_block_covariances: Literal[True] = ...,
) -> tuple[
    Array,
    Array | BlockDiagonalCovariance,
    Array | BlockDiagonalCovariance,
    Array,
    Array,
    Array | BlockDiagonalCovariance,
]: ...


@overload
def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = ...,
    return_filtered: bool = ...,
    max_log_count: float = ...,
    validate_inputs: bool = ...,
    block_n_neurons: int | None = ...,
    block_size: int | None = ...,
    force_dense: bool = ...,
    max_newton_iter: int = ...,
    return_block_covariances: bool = ...,
) -> tuple[Array | BlockDiagonalCovariance, ...]: ...


def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = True,
    return_filtered: bool = False,
    max_log_count: float = 20.0,
    validate_inputs: bool = True,
    block_n_neurons: int | None = None,
    block_size: int | None = None,
    force_dense: bool = False,
    max_newton_iter: int = 3,
    return_block_covariances: bool = False,
) -> tuple[Array | BlockDiagonalCovariance, ...]:
    """Applies a Stochastic State Point Process Smoother (SSPPS).

    This smoother estimates a time-varying latent state ($x_k$) based on
    point process observations ($y_k$) using a Kalman smoother approach.
    It first applies a stochastic point process filter to obtain the filtered
    means and covariances, and then applies a Kalman smoother to refine these estimates.

    $$ x_k = A x_{k-1} + w_k, \\quad w_k \\sim N(0, Q) $$
    $$ \\lambda_{n,k} = f(x_k, Z_k)_n $$
    $$ y_{n,k} \\sim \\text{Poisson}(\\lambda_{n,k} \\Delta t) $$

    Multi-Neuron Support
    --------------------
    The smoother supports multiple neurons sharing a common latent state.
    See `stochastic_point_process_filter` for details on multi-neuron inputs.

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
        Initial mean of the latent state ($x_0$).
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance of the latent state ($P_0$).
    design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_params)
        Design matrix ($Z_k$) used in the intensity function.
        Shape depends on the log_conditional_intensity function. On the
        block-diagonal path the shared per-neuron basis ``Z_base`` of shape
        ``(n_time, block_size)`` may be passed instead (see
        :func:`stochastic_point_process_filter`).
    spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
        Observed spike counts or indicators ($y_k$).
        For single neuron: (n_time,)
        For multiple neurons: (n_time, n_neurons)
    dt : float
        Time step size ($\\Delta t$).
    transition_matrix : ArrayLike, shape (n_params, n_params)
        State transition matrix ($A$).
    process_cov : ArrayLike, shape (n_params, n_params)
        Process noise covariance ($Q$).
    log_conditional_intensity : callable
        Function `log_lambda(Z_k, x_k)` returning the log conditional
        intensity. Should return (n_neurons,) for multi-neuron case.
    include_laplace_normalization : bool, default=True
        If True, include Laplace normalization and prior terms in the
        marginal log-likelihood. If False, return the plug-in log-likelihood
        at the posterior mode without normalization.
    return_block_covariances : bool, default=False
        Block-diagonal path only: return every covariance output (smoothed,
        cross and, with ``return_filtered``, filtered) as a
        :class:`BlockDiagonalCovariance` of per-neuron blocks instead of a
        dense array. See :func:`stochastic_point_process_filter`.

    Returns
    -------
    smoother_mean : Array, shape (n_time, n_params)
        Smoothed posterior means ($x_{k|T}$).
    smoother_cov : Array or BlockDiagonalCovariance, shape (n_time, n_params, n_params)
        Smoothed posterior covariances ($P_{k|T}$); a
        :class:`BlockDiagonalCovariance` of the same dense shape when
        ``return_block_covariances`` applies.
    smoother_cross_cov : Array or BlockDiagonalCovariance, shape (n_time - 1, n_params, n_params)
        Smoothed lag-one cross-covariances, indexed as
        ``Cov(x_t, x_{t+1} | y_{1:T})`` for ``t = 0, ..., T - 2``; a
        :class:`BlockDiagonalCovariance` when ``return_block_covariances``
        applies.
    marginal_log_likelihood : Array
        Total log-likelihood of the observations given the model (scalar array).
    filtered_mean, filtered_cov : Array, optional
        Only with ``return_filtered=True``: the filter output, as returned by
        :func:`stochastic_point_process_filter` (``filtered_cov`` is a
        :class:`BlockDiagonalCovariance` when ``return_block_covariances``
        applies).

    Notes
    -----
    The smoother is observation-model agnostic - it operates only on the
    Gaussian posteriors from the filter. The multi-neuron handling is done
    entirely in the filter step.

    References
    ----------
    [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998 (2004).
    """
    # Convert to arrays up-front so the dispatch branch can slice.
    init_mean_params = jnp.asarray(init_mean_params)
    init_covariance_params = jnp.asarray(init_covariance_params)
    design_matrix = jnp.asarray(design_matrix)
    spike_indicator = jnp.asarray(spike_indicator)
    transition_matrix = jnp.asarray(transition_matrix)
    process_cov = jnp.asarray(process_cov)

    # Validate BEFORE the block dispatch below (the dense path would otherwise
    # inherit this check from the inner filter call, but the block path
    # returns early and would skip it). stacklevel=3: user -> smoother ->
    # _validate.
    if validate_inputs:
        _validate_public_inputs(
            dt,
            init_mean_params,
            init_covariance_params,
            design_matrix,
            spike_indicator,
            transition_matrix,
            process_cov,
            filter_name="stochastic_point_process_smoother",
            stacklevel=3,
        )

    # Block-diagonal dispatch: same opt-in contract as the filter.
    # See stochastic_point_process_filter's block-dispatch comment for
    # the full rationale. If both block_n_neurons and block_size are
    # provided (as Python ints) and force_dense is False, we short-
    # circuit to _stochastic_point_process_smoother_block_diagonal
    # which vmaps the forward and backward passes per-neuron.
    use_block_dispatch = (
        block_n_neurons is not None
        and block_size is not None
        and not force_dense
        and _uses_default_linear_log_intensity(log_conditional_intensity)
    )
    if use_block_dispatch:
        structure = _validated_dispatch_block_structure(
            init_mean_params,
            init_covariance_params,
            transition_matrix,
            process_cov,
            design_matrix,
            spike_indicator,
            block_n_neurons,
            block_size,
        )
        return _stochastic_point_process_smoother_block_diagonal(
            structure,
            spike_indicator,
            dt,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
            return_filtered=return_filtered,
            max_newton_iter=max_newton_iter,
            return_block_covariances=return_block_covariances,
        )

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
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
            # Already validated at this smoother's entry point above; skip
            # the inner filter's redundant re-validation.
            validate_inputs=False,
            max_newton_iter=max_newton_iter,
        )
    )

    smoother_mean, smoother_cov, smoother_cross_cov = (
        _stochastic_point_process_smoother_backward(
            filtered_mean,
            filtered_cov,
            process_cov,
            transition_matrix,
        )
    )

    result = (smoother_mean, smoother_cov, smoother_cross_cov, marginal_log_likelihood)
    if return_filtered:
        return result + (filtered_mean, filtered_cov)
    return result


@typed_jit
def _stochastic_point_process_smoother_backward(
    filtered_mean: Array,
    filtered_cov: Array,
    process_cov: Array,
    transition_matrix: Array,
) -> tuple[Array, Array, Array]:
    """JIT-compiled backward pass of the point process smoother."""

    def _step(
        carry: tuple[Array, Array], args: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
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

    return smoother_mean, smoother_cov, smoother_cross_cov


def dynamics_only_m_step(
    smoother_mean: ArrayLike,
    smoother_cov: ArrayLike,
    smoother_cross_cov: ArrayLike,
    fixed_transition_matrix: ArrayLike | None = None,
    initial_state_prior: InitialStatePrior | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Dynamics-only M-step: update (A, Q, init_mean, init_cov) from smoother outputs.

    Unlike :func:`state_space_practice.kalman.kalman_maximization_step`, this
    variant does not update measurement parameters (H, R) — only the latent
    dynamics (transition matrix, process covariance) and the initial state.
    Used by point-process / spike models whose observation model is a GLM
    fit separately, not a linear-Gaussian emission.

    Exact EM (``initial_state_prior`` given)
    ----------------------------------------
    The filters start from ``x_0 ~ N(m_0, P_0)`` and observe ``x_1 .. x_T``,
    so the complete-data log-likelihood has ``T`` transitions
    ``x_{t-1} -> x_t``, ``t = 1..T``. With the prior, the smoothed moments of
    ``x_0`` and ``Cov(x_0, x_1 | y)`` are recovered one RTS step behind the
    smoother (:func:`~state_space_practice.kalman.smooth_initial_state_with_cross_cov`)
    and the ``x_0 ->
    x_1`` transition enters the ``A`` and ``Q`` sufficient statistics, so
    ``A``, ``Q``, ``m_0`` and ``P_0`` jointly maximise the expected
    complete-data log-likelihood ``Q(theta)`` given the E-step moments (the
    M-step is exact; ``tests/test_oracle_point_process.py`` checks that the
    gradient of ``Q(theta)`` vanishes at the returned values). ``Q`` is then
    divided by ``T``.

    Without the prior (deprecated), only the ``T - 1`` observed-to-observed
    transitions are used (``Q`` divided by ``T - 1``) and the initial state is
    set to the smoothed ``x_1``; that is not an exact EM step.

    Parameters
    ----------
    smoother_mean : ArrayLike, shape (n_time, n_cont_states)
        smoother mean.
    smoother_cov : ArrayLike, shape (n_time, n_cont_states, n_cont_states)
        smoother covariance.
    smoother_cross_cov : ArrayLike, shape (n_time - 1, n_cont_states, n_cont_states)
        Smoothed lag-one cross-covariances, indexed as
        ``Cov(x_t, x_{t+1} | y_{1:T})`` for ``t = 0, ..., T - 2``.
    fixed_transition_matrix : ArrayLike or None, optional
        If None (default), solve for the unconstrained ML transition matrix
        ``A = beta gamma1^{-1}``. If an array is supplied, ``A`` is held fixed
        at it and returned unchanged. Either way ``Q`` is the centred residual
        form :func:`~state_space_practice.kalman.process_cov_residual_form`
        at that ``A`` -- the M-step optimum for the given dynamics, PSD by
        construction (at the solved ``A`` it equals the Roweis-Ghahramani
        ``(gamma2 - A beta^T) / n_transitions`` up to roundoff).
    initial_state_prior : InitialStatePrior or None, optional
        The initial-state prior and dynamics the E-step ran with. If given,
        the M-step is exact EM (see above): the ``x_0 -> x_1`` transition is
        included in the ``A`` / ``Q`` statistics and the initial state is the
        smoothed ``x_0``. If None, the smoothed moments of ``x_1`` are
        returned as the initial state and only ``T - 1`` transitions are
        used, which is not an EM step and can decrease the log-likelihood.

        .. deprecated::
            Passing ``None`` emits a ``DeprecationWarning`` and will be
            removed in version 0.2.0; pass the prior the E-step ran with.

    Returns
    -------
    transition_matrix : Array, shape (n_cont_states, n_cont_states)
        Transition matrix (the unconstrained solve, or ``fixed_transition_matrix``).
    process_cov : Array, shape (n_cont_states, n_cont_states)
        Process covariance, eigenvalues floored at a scale-relative level
        (:func:`~state_space_practice.utils.project_psd_relative`, which logs
        a warning when the floor changes the estimate).
    mean_init : Array, shape (n_cont_states,)
        Initial mean.
    cov_init : Array, shape (n_cont_states, n_cont_states)
        Initial covariance.

    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """
    if initial_state_prior is None:
        warnings.warn(
            "dynamics_only_m_step with initial_state_prior=None uses only the "
            "T - 1 observed transitions, which is not an exact EM step. Pass the "
            "InitialStatePrior the E-step ran with instead. "
            "It will be removed in version 0.2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _dynamics_only_m_step(
        smoother_mean,
        smoother_cov,
        smoother_cross_cov,
        fixed_transition_matrix,
        initial_state_prior,
    )


@typed_jit
def _dynamics_only_m_step(
    smoother_mean: ArrayLike,
    smoother_cov: ArrayLike,
    smoother_cross_cov: ArrayLike,
    fixed_transition_matrix: ArrayLike | None,
    initial_state_prior: InitialStatePrior | None,
) -> tuple[Array, Array, Array, Array]:
    """Jitted core of :func:`dynamics_only_m_step` (no deprecation check)."""
    smoother_mean = jnp.asarray(smoother_mean)
    smoother_cov = jnp.asarray(smoother_cov)
    smoother_cross_cov = jnp.asarray(smoother_cross_cov)

    n_time = smoother_mean.shape[0]
    if initial_state_prior is None and n_time < 2:
        raise ValueError(
            "dynamics_only_m_step requires at least 2 time steps to "
            "estimate transition dynamics (or an initial_state_prior, which "
            "adds the x_0 -> x_1 transition)."
        )
    if n_time < 1:
        raise ValueError("dynamics_only_m_step requires at least 1 time step.")

    sum_cov = jnp.sum(smoother_cov, axis=0)
    sum_cross_cov = smoother_cross_cov.sum(axis=0)

    if initial_state_prior is None:
        # Deprecated: the T - 1 transitions x_t -> x_{t+1}, t = 1..T-1.
        init_mean = smoother_mean[0]
        init_cov = smoother_cov[0]
        means = smoother_mean
        sum_state_cov = sum_cov
        first_cov = smoother_cov[0]
    else:
        # Exact EM: prepend the smoothed x_0 so the sums run over all T
        # transitions x_{t-1} -> x_t, t = 1..T.
        init_mean, init_cov, init_cross_cov = smooth_initial_state_with_cross_cov(
            initial_state_prior, smoother_mean[0], smoother_cov[0]
        )
        means = jnp.concatenate((init_mean[None], smoother_mean), axis=0)
        sum_state_cov = sum_cov + init_cov
        sum_cross_cov = sum_cross_cov + init_cross_cov
        first_cov = init_cov

    # sum over the "previous" (x_{t-1}) and "next" (x_t) ends of the transitions.
    sum_prev_cov = sum_state_cov - smoother_cov[-1]
    sum_next_cov = sum_state_cov - first_cov

    if fixed_transition_matrix is None:
        # Unconstrained ML transition matrix A = beta gamma1^{-1}.
        gamma1 = sum_prev_cov + sum_of_outer_products(means[:-1], means[:-1])
        beta = (sum_cross_cov + sum_of_outer_products(means[:-1], means[1:])).T
        transition_matrix = psd_solve(gamma1, beta.T).T
    else:
        transition_matrix = jnp.asarray(fixed_transition_matrix)

    # Process covariance: centred residual form at the chosen A (divides by
    # the number of transitions, len(means) - 1).
    process_cov = project_psd_relative(
        process_cov_residual_form(
            means,
            sum_next_cov=sum_next_cov,
            sum_prev_cov=sum_prev_cov,
            sum_cross_cov=sum_cross_cov,
            transition_matrix=transition_matrix,
        ),
        name="dynamics_only_m_step process_cov",
    )

    return (
        transition_matrix,
        process_cov,
        init_mean,
        init_cov,
    )


def get_confidence_interval(
    posterior_mean: ArrayLike,
    posterior_covariance: ArrayLike | BlockDiagonalCovariance,
    alpha: float = 0.05,
) -> Array:
    """Get the confidence interval from the posterior covariance

    Parameters
    ----------
    posterior_mean : ArrayLike, shape (n_time, n_params)
    posterior_covariance : ArrayLike or BlockDiagonalCovariance, shape (n_time, n_params, n_params)
        Dense covariances, or a :class:`BlockDiagonalCovariance` (whose
        marginal variances are read from the per-neuron blocks without
        materialising the dense array).
    alpha : float, optional
        Significance level in ``(0, 1)``, by default ``0.05``. Returns a
        ``1 - alpha`` confidence interval (i.e. the default 0.05 gives a
        95% CI, not a 5% CI). This matches the default of
        :meth:`PointProcessModel.get_confidence_interval`.
    """
    posterior_mean = jnp.asarray(posterior_mean)
    if isinstance(posterior_covariance, BlockDiagonalCovariance):
        # Block-diagonal path: the marginal variances come straight from the
        # per-neuron blocks; no dense (n_time, n_params, n_params) array.
        variances = posterior_covariance.diagonal()
    else:
        variances = jnp.diagonal(jnp.asarray(posterior_covariance), axis1=-2, axis2=-1)
    z = jax.scipy.stats.norm.ppf(1 - alpha / 2)
    ci = z * jnp.sqrt(variances)  # shape (n_time, n_params)

    return jnp.stack((posterior_mean - ci, posterior_mean + ci), axis=-1)


def steepest_descent_point_process_filter(
    init_mean_params: ArrayLike,
    x: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    epsilon: ArrayLike,
    log_receptive_field_model: Callable[[ArrayLike, ArrayLike], Array],
    max_log_count: float = 20.0,
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
    max_log_count : float, default=20.0
        Ceiling on ``log(rate * dt)`` applied inside ``_safe_expected_count``
        to prevent overflow on pathological spike counts. Pass a physiologically
        motivated value (e.g. ``log(max_firing_rate_hz * dt)``) if you expect
        outlier bins. The default of 20.0 matches the Laplace-EKF filter's
        default for backwards compatibility.

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
    # Convert ArrayLike inputs to Array
    init_mean_params_arr: Array = jnp.asarray(init_mean_params)
    x_arr: Array = jnp.asarray(x)
    spike_indicator_arr: Array = jnp.asarray(spike_indicator)
    epsilon_arr: Array = jnp.asarray(epsilon)

    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)

    def _update(mean_prev: Array, args: tuple[Array, Array]) -> tuple[Array, Array]:
        """Steepest Descent Point Process Filter update step"""
        x_t, spike_indicator_t = args
        conditional_intensity = _safe_expected_count(
            log_receptive_field_model(x_t, mean_prev),
            dt,
            max_log_count=max_log_count,
        )
        innovation = spike_indicator_t - conditional_intensity
        one_step_grad = grad_log_receptive_field_model(x_t, mean_prev)
        posterior_mean = mean_prev + epsilon_arr @ one_step_grad * innovation

        return posterior_mean, posterior_mean

    return jax.lax.scan(_update, init_mean_params_arr, (x_arr, spike_indicator_arr))[1]


class PointProcessModel(SGDFittableMixin):
    """Point Process State-Space Model with EM fitting.

    Implements the Eden & Brown (2004) adaptive point process filter/smoother
    with EM algorithm for parameter estimation.

    Model:
        x_k = A @ x_{k-1} + w_k,  w_k ~ N(0, Q)
        n_{j,k} ~ Poisson(exp(log_intensity_func(Z_k, x_k)[j]) * dt)

    The EM algorithm estimates the state dynamics parameters (A, Q) while
    the latent states x_k are estimated via the E-step (filter/smoother).

    Multi-Neuron Support
    --------------------
    The model supports multiple neurons sharing a common latent state:

    - spike_indicator: (n_time, n_neurons) - spike counts for each neuron
    - log_intensity_func(Z_k, x_k) should return (n_neurons,) log-intensities

    For backwards compatibility, single-neuron inputs work as before:
    - spike_indicator: (n_time,) is internally promoted to (n_time, 1)
    - scalar log-intensity output is wrapped to (1,)

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
        For multi-neuron, should return (n_neurons,) array.
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
    filtered_mean, filtered_cov : Array
        Filtered state estimates and covariances after fitting.
    log_likelihood_ : float
        Marginal log-likelihood at the fitted parameters (from the last
        :meth:`fit` or :meth:`fit_sgd`).
    log_likelihood_history_ : list[float]
        Per-iteration EM log-likelihoods, or per-step SGD training objective,
        of the last fit.
    converged_ : bool
        Whether the last fit met its convergence criterion.
    n_iter_ : int or None
        EM iterations of the last fit; ``None`` after :meth:`fit_sgd`.

    Reading a fitted attribute before fitting raises ``NotFittedError``.

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """

    # Results, set by fit / fit_sgd.
    smoother_mean: FittedAttribute[Array] = FittedAttribute()
    smoother_cov: FittedAttribute[Array] = FittedAttribute()
    smoother_cross_cov: FittedAttribute[Array] = FittedAttribute()
    filtered_mean: FittedAttribute[Array] = FittedAttribute()
    filtered_cov: FittedAttribute[Array] = FittedAttribute()
    _sgd_n_time: FittedAttribute[int] = FittedAttribute()

    # The posteriors above, cleared when a fit fails.
    _fit_output_attrs = (
        "smoother_mean",
        "smoother_cov",
        "smoother_cross_cov",
        "filtered_mean",
        "filtered_cov",
    )

    def __init__(
        self,
        n_state_dims: int,
        dt: float,
        transition_matrix: ArrayLike | None = None,
        process_cov: ArrayLike | None = None,
        init_mean: ArrayLike | None = None,
        init_cov: ArrayLike | None = None,
        log_intensity_func: Callable[[ArrayLike, ArrayLike], Array] | None = None,
        update_transition_matrix: bool = True,
        update_process_cov: bool = True,
        update_init_state: bool = True,
        max_newton_iter: int = 3,
    ):
        validate_scalar(dt, "dt", positive=True)
        self.n_state_dims = n_state_dims
        self.dt = dt
        self.max_newton_iter = max_newton_iter

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

        # Validate supplied parameter shapes at the boundary rather than
        # letting a mismatch surface as an opaque error deep in the first
        # filter scan.
        for name, expected_shape, arr in (
            ("transition_matrix", (n_state_dims, n_state_dims), self.transition_matrix),
            ("process_cov", (n_state_dims, n_state_dims), self.process_cov),
            ("init_mean", (n_state_dims,), self.init_mean),
            ("init_cov", (n_state_dims, n_state_dims), self.init_cov),
        ):
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape} for "
                    f"n_state_dims={n_state_dims}, got {arr.shape}."
                )

        if log_intensity_func is None:
            self.log_intensity_func: Callable[[ArrayLike, ArrayLike], Array] = (
                log_conditional_intensity
            )
        else:
            self.log_intensity_func = log_intensity_func

        # Update flags
        self.update_transition_matrix = update_transition_matrix
        self.update_process_cov = update_process_cov
        self.update_init_state = update_init_state

    def _e_step(self, design_matrix: ArrayLike, spike_indicator: ArrayLike) -> float:
        """E-step: Run filter and smoother to estimate latent states.

        Parameters
        ----------
        design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_state_dims)
            Design matrix for the intensity function.
        spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Observed spike counts. Single neuron: (n_time,), multi-neuron: (n_time, n_neurons).

        Returns
        -------
        marginal_log_likelihood : float
        """
        # The smoother returns the filtered moments from its single forward
        # pass (return_filtered=True), so we don't run a second, redundant
        # forward filter — the forward pass is the dominant per-iteration cost.
        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_log_likelihood,
            self.filtered_mean,
            self.filtered_cov,
        ) = stochastic_point_process_smoother(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
            # fit() / fit_sgd() validate once at entry; skip re-validation on
            # each EM iteration's E-step and the post-SGD smoothing pass.
            validate_inputs=False,
            return_filtered=True,
            max_newton_iter=self.max_newton_iter,
        )

        return float(marginal_log_likelihood)

    def _m_step(self) -> None:
        """M-step: Update model parameters based on smoothed estimates."""
        if not (
            is_set(self, "smoother_mean")
            and is_set(self, "smoother_cov")
            and is_set(self, "smoother_cross_cov")
        ):
            raise RuntimeError("Must run E-step before M-step")

        # When A is frozen, the process-cov update must be taken against that
        # frozen A via the full quadratic form; the Roweis-Ghahramani shortcut
        # is the M-step optimum only at the freshly solved (unconstrained) A.
        fixed_transition_matrix = (
            None if self.update_transition_matrix else self.transition_matrix
        )
        # The parameters the E-step ran with: the initial-state update is the
        # smoothed x_0, one RTS step behind the smoother's x_1.
        initial_state_prior = InitialStatePrior(
            init_mean=self.init_mean,
            init_cov=self.init_cov,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
        )
        transition_matrix, process_cov, init_mean, init_cov = dynamics_only_m_step(
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            fixed_transition_matrix=fixed_transition_matrix,
            initial_state_prior=initial_state_prior,
        )

        if self.update_transition_matrix:
            self.transition_matrix = transition_matrix

        if self.update_process_cov:
            # dynamics_only_m_step already floors the eigenvalues at a
            # scale-relative level (project_psd_relative).
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
        design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_state_dims)
            Design matrix for the intensity function.
            Shape depends on the log_intensity_func.
        spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Observed spike counts or indicators.
            For single neuron: (n_time,)
            For multiple neurons: (n_time, n_neurons)
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
        validate_count_array(spike_indicator, "spike_indicator")

        # Numerical sanity check once at the top: validate PSD and warn
        # on f32 risk. Skips per-iteration re-validation in the E-step.
        _validate_filter_numerics(
            jnp.asarray(self.init_cov), n_time=spike_indicator.shape[0]
        )

        snapshot_keys = self._fit_output_attrs + (
            "transition_matrix",
            "process_cov",
            "init_mean",
            "init_cov",
        )

        # A rejected E-step (non-finite or decreasing LL) restores the last
        # accepted (parameters, smoother) pair so get_rate_estimate /
        # get_confidence_interval never serve a diverged posterior; a
        # non-finite *first* E-step clears the posteriors it installed.
        result = run_em(
            lambda: float(self._e_step(design_matrix, spike_indicator)),
            self._m_step,
            lambda: snapshot_attributes(self, snapshot_keys),
            lambda state: restore_attributes(self, state),
            max_iter=max_iter,
            tol=tolerance,
            logger=logger,
            on_first_nonfinite="clear",
            clear_state=lambda: clear_attributes(self, self._fit_output_attrs),
        )
        self._record_fit_result(
            result.log_likelihoods,
            result.converged,
            n_iter=len(result.log_likelihoods),
        )
        return result.log_likelihoods

    # --- SGDFittableMixin protocol ---

    def fit_sgd(
        self,
        design_matrix: ArrayLike,
        spike_indicator: ArrayLike,
        optimizer: optax.GradientTransformation | None = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: float | None = None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        design_matrix : ArrayLike
            Design matrix for the intensity function.
        spike_indicator : ArrayLike
            Observed spike counts.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.
        convergence_tol : float or None
            If set, stop early when loss change < tol for 5 consecutive steps.

        Returns
        -------
        log_likelihoods : list of float
        """
        return super().fit_sgd(
            design_matrix,
            spike_indicator,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    def _prepare_sgd_data(
        self, design_matrix: ArrayLike, spike_indicator: ArrayLike
    ) -> tuple[tuple[Array, Array], dict[str, Any]]:
        """Validate the ``fit_sgd`` data and record its length.

        Runs after ``fit_sgd`` has validated its settings, so a rejected call
        leaves the model untouched.
        """
        design_matrix = jnp.asarray(design_matrix)
        spike_indicator = jnp.asarray(spike_indicator)
        validate_count_array(spike_indicator, "spike_indicator")
        if spike_indicator.ndim == 1:
            spike_indicator = spike_indicator[:, None]

        # Numerical sanity check once at the top: validate PSD and warn
        # on f32 risk. The SGD loss_fn runs inside jax.jit, which cannot
        # call eigvalsh's host-side float() conversion, so per-step
        # re-validation is both wasteful and forbidden. stacklevel=5: user
        # -> fit_sgd -> SGDFittableMixin.fit_sgd -> this hook -> wrapper.
        _validate_filter_numerics(
            jnp.asarray(self.init_cov), n_time=spike_indicator.shape[0], stacklevel=5
        )

        self._sgd_n_time = spike_indicator.shape[0]
        return (design_matrix, spike_indicator), {}

    @property
    def _n_timesteps(self) -> int:
        return self._sgd_n_time

    def _build_param_spec(self) -> tuple[SGDParams, SGDParamSpec]:
        params: SGDParams = {}
        spec: SGDParamSpec = {}

        if self.update_transition_matrix:
            params["transition_matrix"] = self.transition_matrix
            spec["transition_matrix"] = UNCONSTRAINED

        if self.update_process_cov:
            params["process_cov"] = self.process_cov
            spec["process_cov"] = PSD_MATRIX

        if self.update_init_state:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED
            params["init_cov"] = self.init_cov
            spec["init_cov"] = PSD_MATRIX

        return params, spec

    def _sgd_loss_fn(
        self, params: SGDParams, design_matrix: Array, spike_indicator: Array
    ) -> Array:
        A = self._sgd_param(params, "transition_matrix")
        Q = self._sgd_param(params, "process_cov")
        m0 = self._sgd_param(params, "init_mean")
        P0 = self._sgd_param(params, "init_cov")

        _, _, marginal_ll = stochastic_point_process_filter(
            init_mean_params=m0,
            init_covariance_params=P0,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=A,
            process_cov=Q,
            log_conditional_intensity=self.log_intensity_func,
            # fit_sgd validated once at the top; skip per-step
            # re-validation inside the jit'd loss fn.
            validate_inputs=False,
            max_newton_iter=self.max_newton_iter,
        )
        return -marginal_ll

    def _store_sgd_params(self, params: SGDParams) -> None:
        if "transition_matrix" in params:
            self.transition_matrix = params["transition_matrix"]
        if "process_cov" in params:
            self.process_cov = params["process_cov"]
        if "init_mean" in params:
            self.init_mean = params["init_mean"]
        if "init_cov" in params:
            self.init_cov = params["init_cov"]

    def _finalize_sgd(self, design_matrix: Array, spike_indicator: Array) -> float:
        return self._e_step(design_matrix, spike_indicator)

    def get_rate_estimate(
        self,
        design_matrix: ArrayLike,
        use_smoothed: bool = True,
        evaluate_at_all_positions: bool = True,
    ) -> Array:
        """Get the estimated firing rate using the model's log-intensity function.

        This method computes firing rates by evaluating the stored log_intensity_func,
        supporting both single-neuron and multi-neuron models with arbitrary
        (possibly nonlinear) intensity functions.

        Parameters
        ----------
        design_matrix : ArrayLike
            Design matrix to evaluate rate at. Shape depends on usage:
            - If evaluate_at_all_positions=True (default): (n_pos, n_state_dims)
              where n_pos can be n_time or any number of positions/conditions
            - If evaluate_at_all_positions=False: same shape as used during fit,
              e.g., (n_time, n_state_dims) for single-neuron or
              (n_time, n_neurons, n_state_dims) for multi-neuron

        use_smoothed : bool
            If True, use smoothed estimates; otherwise use filtered.

        evaluate_at_all_positions : bool, default=True
            If True, evaluate the rate at all positions in design_matrix for each
            time point, returning shape (n_time, n_pos) for single-neuron or
            (n_time, n_pos, n_neurons) for multi-neuron.
            If False, evaluate time-aligned: design_matrix[t] with state[t],
            returning shape (n_time,) for single-neuron or (n_time, n_neurons)
            for multi-neuron.

        Returns
        -------
        rate : Array
            Estimated firing rate in Hz. Shape depends on evaluate_at_all_positions
            and whether the model is single/multi-neuron (see above).

        Notes
        -----
        The rate is computed as exp(log_intensity_func(design_matrix, x)).
        This generalizes to arbitrary intensity functions, not just the default
        linear Z @ x.
        """
        if use_smoothed:
            if not is_set(self, "smoother_mean"):
                raise NotFittedError("Model has not been fitted yet.")
            state_estimate = self.smoother_mean
        else:
            if not is_set(self, "filtered_mean"):
                raise NotFittedError("Model has not been fitted yet.")
            state_estimate = self.filtered_mean

        design_matrix = jnp.asarray(design_matrix)

        if evaluate_at_all_positions:
            # Evaluate rate at all positions for each time point
            # For each (time, position) pair, compute log_intensity_func(design[pos], state[time])
            # vmap over positions (inner), then over times (outer)
            def rate_at_time(state_t: Array) -> Array:
                # For this time's state, evaluate at all positions
                return jax.vmap(lambda dm: self.log_intensity_func(dm, state_t))(
                    design_matrix
                )

            log_rate = jax.vmap(rate_at_time)(state_estimate)  # (n_time, n_pos, ...)
        else:
            # Time-aligned evaluation: design_matrix[t] with state_estimate[t]
            log_rate = jax.vmap(self.log_intensity_func)(design_matrix, state_estimate)

        return jnp.exp(log_rate)

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
            if not (is_set(self, "smoother_mean") and is_set(self, "smoother_cov")):
                raise NotFittedError("Model has not been fitted yet.")
            mean = self.smoother_mean
            cov = self.smoother_cov
        else:
            if not (is_set(self, "filtered_mean") and is_set(self, "filtered_cov")):
                raise NotFittedError("Model has not been fitted yet.")
            mean = self.filtered_mean
            cov = self.filtered_cov

        return get_confidence_interval(mean, cov, alpha=alpha)

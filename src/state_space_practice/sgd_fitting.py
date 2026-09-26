"""Shared SGDFittableMixin for gradient-based model fitting.

Provides a generic fit_sgd() method that delegates to model-specific
hooks for parameter specification, loss computation, and post-optimization
finalization.

Models must implement:
- _build_param_spec() -> tuple[dict, dict]
- _sgd_loss_fn(params, *args, **kwargs) -> Array
- _finalize_sgd(*args, **kwargs) -> None
- _n_timesteps: int (property or attribute)

and either declare ``_sgd_param_attrs`` (param key -> attribute name, used by
the default ``_store_sgd_params``) or override ``_store_sgd_params``.
Optional hooks: ``_check_sgd_initialized`` (default no-op) and
``_prepare_sgd_data`` (default: pass the data through unchanged).
"""

import contextlib
import hashlib
import logging
import math
import weakref
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any, ClassVar, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from state_space_practice.parameter_transforms import (
    transform_to_constrained,
    transform_to_unconstrained,
)
from state_space_practice.utils import validate_int

logger = logging.getLogger(__name__)


#: Default optimizer for ``fit_sgd``. A single module-level instance, so the
#: per-model compiled-step cache (keyed on the optimizer's identity) is reused
#: by every ``fit_sgd`` call that does not pass its own optimizer.
_DEFAULT_OPTIMIZER = optax.chain(
    optax.clip_by_global_norm(10.0),
    optax.adam(1e-2),
)

#: Compiled SGD steps kept per model instance (least recently used evicted).
_MAX_CACHED_STEPS_PER_MODEL = 4

#: ``id(model) -> {cache key -> _CompiledSGDStep}``. Keyed by id (models need
#: not be hashable) and emptied by a ``weakref.finalize`` when the model is
#: collected; the compiled steps only hold a weak reference to their model.
#: Kept outside the instance so models stay deep-copyable and picklable.
_SGD_STEP_CACHE: dict[int, dict[Hashable, "_CompiledSGDStep"]] = {}

_MISSING = object()


class _Uncacheable(Exception):
    """A value whose identity cannot be captured by a fingerprint."""


def _fingerprint(value: object, keepalive: list[object]) -> Hashable:
    """Hashable summary of ``value`` that changes whenever its content does.

    Arrays are summarized by shape, dtype and a digest of their bytes; plain
    scalars and strings by value (floats by ``repr``, so ``-0.0`` and ``nan``
    compare as themselves); containers recursively. Any other object is
    summarized by identity and appended to ``keepalive`` so its id cannot be
    reused while the fingerprint is alive.

    Raises
    ------
    _Uncacheable
        If ``value`` contains a JAX tracer.
    """
    if value is _MISSING or value is None:
        return ("const", value is None)
    if isinstance(value, jax.core.Tracer):
        raise _Uncacheable
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):  # e.g. typed PRNG keys
            keepalive.append(value)
            return ("object", id(value))
        digest = hashlib.blake2b(
            np.ascontiguousarray(arr).tobytes(), digest_size=16
        ).digest()
        return ("array", arr.shape, arr.dtype.str, digest)
    if isinstance(value, (bool, int, str, bytes)):
        return ("value", type(value), value)
    if isinstance(value, (float, complex)):
        return ("value", type(value), repr(value))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_fingerprint(v, keepalive) for v in value))
    if isinstance(value, dict):
        return (
            dict,
            tuple((k, _fingerprint(v, keepalive)) for k, v in value.items()),
        )
    keepalive.append(value)
    return ("object", id(value))


@contextlib.contextmanager
def _recording_attribute_reads(obj: object, reads: set[str]) -> Iterator[bool]:
    """Record the attribute names read on ``obj`` inside the block.

    Temporarily swaps ``obj.__class__`` for a subclass whose
    ``__getattribute__`` logs each name, so reads made by any method or
    property of the model are captured. Yields ``False`` (and records
    nothing) when the class cannot be swapped, e.g. for ``__slots__`` layouts.
    """
    cls = type(obj)

    def __getattribute__(self: object, name: str) -> Any:
        reads.add(name)
        return cls.__getattribute__(self, name)

    swapped = False
    try:
        obj.__class__ = cast(Any, type(cls))(
            cls.__name__,
            (cls,),
            {
                "__getattribute__": __getattribute__,
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
            },
        )
        swapped = True
    except TypeError:
        pass
    try:
        yield swapped
    finally:
        if swapped:
            obj.__class__ = cls


#: ``(treedef, is_array_mask, static_leaves)`` from :func:`_split_leaves`.
_LeafStructure = tuple[Any, tuple[bool, ...], tuple[Any, ...]]


def _split_leaves(tree: object) -> tuple[list[Any], _LeafStructure]:
    """Split a pytree into its array leaves and a hashable static remainder.

    Array leaves become jit arguments; every other leaf (Python scalars,
    strings, ...) stays a compile-time constant and is part of the returned
    structure, which keys the compiled-step cache.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    is_array = tuple(isinstance(leaf, (jax.Array, np.ndarray)) for leaf in leaves)
    dynamic = [leaf for leaf, dyn in zip(leaves, is_array) if dyn]
    static = tuple(leaf for leaf, dyn in zip(leaves, is_array) if not dyn)
    return dynamic, (treedef, is_array, static)


def _merge_leaves(dynamic: list[Any], structure: _LeafStructure) -> Any:
    """Inverse of :func:`_split_leaves`."""
    treedef, is_array, static = structure
    dynamic_iter, static_iter = iter(dynamic), iter(static)
    leaves = [next(dynamic_iter) if dyn else next(static_iter) for dyn in is_array]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _unbuilt_step(*args: Any) -> Any:
    raise RuntimeError("SGD step used before it was built.")


class _CompiledSGDStep:
    """A jitted SGD step plus the model state its traces baked in.

    ``_sgd_loss_fn`` may read model attributes (design matrices, fixed
    hyperparameters, ...); jit freezes whatever values those reads returned
    while tracing. Each trace therefore records the attributes it read and a
    fingerprint of their values, and the step is reused only while every
    recorded attribute still has the same fingerprint.
    """

    def __init__(self, keepalive: list[object]) -> None:
        self.train_step: Callable[..., Any] = _unbuilt_step
        self.state_fingerprints: dict[str, Hashable] = {}
        self.cacheable = True
        self.keepalive = keepalive

    def record_trace(self, model: object, reads: set[str], recorded: bool) -> None:
        """Fingerprint the attributes one trace read (called while tracing)."""
        if not recorded:
            self.cacheable = False
            return
        instance_dict = vars(model)
        try:
            for name in reads:
                if name.startswith("__"):
                    continue
                self.state_fingerprints[name] = _fingerprint(
                    instance_dict.get(name, _MISSING), self.keepalive
                )
        except _Uncacheable:
            self.cacheable = False

    def matches(self, model: object) -> bool:
        """Whether ``model``'s recorded attributes still hold the traced values."""
        instance_dict = vars(model)
        scratch: list[object] = []
        try:
            return all(
                _fingerprint(instance_dict.get(name, _MISSING), scratch) == fp
                for name, fp in self.state_fingerprints.items()
            )
        except _Uncacheable:
            return False


def _model_step_cache(model: object) -> dict[Hashable, _CompiledSGDStep] | None:
    """Per-model compiled-step cache, or None if the model is not weakref-able."""
    model_id = id(model)
    cache = _SGD_STEP_CACHE.get(model_id)
    if cache is None:
        try:
            weakref.finalize(model, _SGD_STEP_CACHE.pop, model_id, None)
        except TypeError:
            return None
        cache = _SGD_STEP_CACHE[model_id] = {}
    return cache


def _tree_all_finite(tree: object) -> bool:
    """Return True when every numeric leaf in a pytree is finite."""
    return all(
        bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_all_finite_array(tree: object) -> jax.Array:
    """JIT-compatible finite check for numeric pytree leaves."""
    checks = [
        jnp.all(jnp.isfinite(jnp.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    if not checks:
        return jnp.array(True)
    return jnp.all(jnp.stack(checks))


def reconstruct_per_state_array(
    params: dict, prefix: str, fallback: Array, n_discrete_states: int
) -> Array:
    """Reassemble a ``(..., n_discrete_states)`` array from per-state SGD params.

    Per-state PSD parameters are exposed to the optimizer as separate keys
    (``"init_cov_0"``, ``"init_cov_1"``, ...). If ``params`` holds none of
    them the ``fallback`` array is returned unchanged; otherwise each state's
    slice comes from ``params`` when present and from ``fallback`` otherwise.

    Parameters
    ----------
    params : dict
        Optimized parameters; the per-state entries are keyed
        ``f"{prefix}_{j}"`` and each has shape ``fallback.shape[:-1]``, e.g.
        ``(n_latent, n_latent)`` for ``init_cov``.
    prefix : str
        Key prefix of the per-state entries, e.g. ``"init_cov"``.
    fallback : Array, shape (..., n_discrete_states)
        Current stacked array, discrete-state axis last, e.g.
        ``(n_latent, n_latent, n_discrete_states)``. Supplies every state
        slice absent from ``params``.
    n_discrete_states : int
        Number of discrete states (length of the trailing axis).

    Returns
    -------
    Array, shape (..., n_discrete_states)
        ``fallback`` itself (same object) when ``params`` has no
        ``f"{prefix}_*"`` key, else the restacked array.
    """
    if not any(k.startswith(f"{prefix}_") for k in params):
        return fallback
    return jnp.stack(
        [
            params.get(f"{prefix}_{j}", fallback[..., j])
            for j in range(n_discrete_states)
        ],
        axis=-1,
    )


#: Errors raised when a loss needs concrete values of its data (e.g. host-side
#: validation of the data inside ``_sgd_loss_fn``).
_DATA_TRACING_ERRORS = (
    jax.errors.ConcretizationTypeError,
    jax.errors.TracerArrayConversionError,
    jax.errors.TracerBoolConversionError,
    jax.errors.TracerIntegerConversionError,
)


def _build_sgd_step(
    model_ref: Callable[[], Any],
    optimizer: Any,
    param_spec: dict,
    n_timesteps: float,
    data_structure: _LeafStructure,
    frozen_structure: _LeafStructure,
    baked_leaves: tuple[list[Any], list[Any]] | None = None,
) -> _CompiledSGDStep:
    """Build the jitted SGD step ``(unc_p, opt_st, frozen_leaves, data_leaves)``.

    With ``baked_leaves=(frozen_leaves, data_leaves)`` those values are closed
    over as compile-time constants and the corresponding arguments are
    ignored -- the fallback for losses that need concrete data while tracing.
    """
    # Keep the optimizer (and hence its id) alive as long as the entry.
    entry = _CompiledSGDStep(keepalive=[optimizer])

    def _loss_inner(unc_p, frozen_leaves, data_leaves):
        model = model_ref()
        frozen = _merge_leaves(frozen_leaves, frozen_structure)
        args, kwargs = _merge_leaves(data_leaves, data_structure)
        p = transform_to_constrained(unc_p, param_spec, static_params=frozen)
        return model._sgd_loss_fn(p, *args, **kwargs) / n_timesteps

    def _sgd_train_step(unc_p, opt_st, frozen_leaves, data_leaves):
        if baked_leaves is not None:
            frozen_leaves, data_leaves = baked_leaves
        # Runs only while tracing: record which model attributes the loss
        # reads, so a cached step is never reused with stale values.
        model = model_ref()
        reads: set[str] = set()
        with _recording_attribute_reads(model, reads) as recorded:
            loss, grads = jax.value_and_grad(_loss_inner)(
                unc_p, frozen_leaves, data_leaves
            )
        entry.record_trace(model, reads, recorded)
        updates, new_opt_st = optimizer.update(grads, opt_st, unc_p)
        new_unc_p = optax.apply_updates(unc_p, updates)
        step_finite = (
            _tree_all_finite_array(grads)
            & _tree_all_finite_array(updates)
            & _tree_all_finite_array(new_unc_p)
        )
        return loss, new_unc_p, new_opt_st, step_finite

    entry.train_step = jax.jit(_sgd_train_step)
    return entry


class SGDFittableMixin:
    """Mixin providing fit_sgd() for state-space models.

    Models must implement:
    - _build_param_spec() -> tuple[dict, dict]
    - _sgd_loss_fn(params, *args, **kwargs) -> Array
    - _finalize_sgd(*args, **kwargs) -> None
    - _n_timesteps: int (property or attribute)

    and store the optimized parameters either by declaring
    ``_sgd_param_attrs`` (consumed by the default ``_store_sgd_params``) or
    by overriding ``_store_sgd_params``.

    Models whose parameters are allocated lazily (e.g. on the first ``fit``)
    may override ``_check_sgd_initialized`` to raise before optimization
    starts; the default is a no-op for models that allocate in ``__init__``.
    Models that need to validate or canonicalize the data ``fit_sgd``
    receives (or record its length for ``_n_timesteps``) override
    ``_prepare_sgd_data`` instead of re-declaring ``fit_sgd``.
    """

    #: Optimized-parameter key -> model attribute name, consumed by the
    #: default ``_store_sgd_params``. A subclass attribute *replaces* (does
    #: not merge with) the parent's mapping; extend it explicitly with
    #: ``{**Parent._sgd_param_attrs, "key": "attr"}``.
    _sgd_param_attrs: ClassVar[Mapping[str, str]] = {}

    def _check_sgd_initialized(self) -> None:
        return

    def _prepare_sgd_data(
        self, *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Validate / canonicalize the data passed to ``fit_sgd``.

        Called first thing in ``fit_sgd`` with the positional data and any
        model-specific keyword arguments (everything except the optimizer
        settings). The returned ``(args, kwargs)`` replace the originals and
        are forwarded to ``_sgd_loss_fn`` and ``_finalize_sgd``, so a model
        can coerce inputs to arrays, reject malformed data before any JAX
        work, or record the sequence length that ``_n_timesteps`` reports,
        without overriding ``fit_sgd`` itself. The default is the identity.
        """
        return args, kwargs

    def _store_sgd_params(self, params: dict) -> None:
        """Copy the optimized parameters onto the model.

        Default implementation driven by ``_sgd_param_attrs``: every mapped
        key present in ``params`` is assigned to its attribute; absent keys
        are skipped, so a param spec may drop entries. Models with derived
        storage (a per-state slot, a re-stabilized covariance, a matrix
        rebuilt from scientific parameters) override this method and call
        ``super()._store_sgd_params(params)`` for the plain keys.
        """
        attrs = self._sgd_param_attrs
        if not attrs:
            raise NotImplementedError(
                f"{type(self).__name__} must declare `_sgd_param_attrs` or "
                "override `_store_sgd_params` to store optimized parameters."
            )
        for key, attr in attrs.items():
            if key in params:
                setattr(self, attr, params[key])

    def _finalize_convergence(self, converged: bool, max_iter: int) -> None:
        """Record the EM convergence flag and warn if the fit did not converge.

        Shared terminal-convergence policy for the EM fitters that subclass this
        mixin: sets ``self.converged_`` and, on max-iter exhaustion, logs a
        warning under the model's own module logger. The per-loop convergence
        criterion (which differs across fitters) stays in each ``fit`` method.
        """
        self.converged_ = converged
        if not converged and max_iter > 1:
            logging.getLogger(type(self).__module__).warning(
                "%s.fit did not converge in %d EM iterations; the returned "
                "parameters are the last iterate, not a converged fit "
                "(increase max_iter or relax tolerance).",
                type(self).__name__,
                max_iter,
            )

    def _compiled_sgd_step(
        self,
        optimizer: Any,
        param_spec: dict,
        n_timesteps: float,
        data_structure: _LeafStructure,
        frozen_structure: _LeafStructure,
    ) -> tuple[
        Callable[..., Any],
        _CompiledSGDStep,
        dict[Hashable, _CompiledSGDStep] | None,
        Hashable,
    ]:
        """Return a jitted SGD step, reusing this model's cached one if valid.

        The step fuses loss + grad + ``optimizer.update`` + ``apply_updates``
        into one compiled graph; without jit the optimizer update and the
        softplus/adam primitives dispatch one at a time through Python (~65x
        slower on CPU; worse on GPU due to per-primitive sync). It takes
        ``(unc_params, opt_state, frozen_leaves, data_leaves)``.

        Steps are cached per model instance under ``(id(optimizer),
        param_spec, n_timesteps, static data / frozen structure)``; jit's own
        cache handles new array shapes. Because ``_sgd_loss_fn`` may read
        model attributes, which jit freezes at trace time, a cached step is
        reused only while every attribute its traces read still holds the
        same value (see :class:`_CompiledSGDStep`). Nothing on the model may
        change *during* one ``fit_sgd`` loop: ``_store_sgd_params`` only runs
        after it.

        Returns
        -------
        train_step : callable
        entry : _CompiledSGDStep
        cache : dict or None
            The model's step cache (None when the model cannot be cached).
        key : hashable or None
            The entry's key in ``cache``.
        """
        cache = _model_step_cache(self)
        key: Hashable = None
        if cache is not None:
            key = (
                id(optimizer),
                tuple(sorted(param_spec.items())),
                n_timesteps,
                data_structure,
                frozen_structure,
            )
            try:
                cached = cache.pop(key, None)
            except TypeError:  # an unhashable static leaf or transform
                cache, key, cached = None, None, None
            if cache is not None and cached is not None and cached.matches(self):
                cache[key] = cached  # re-insert as most recently used
                return cached.train_step, cached, cache, key

        entry = _build_sgd_step(
            weakref.ref(self) if cache is not None else (lambda: self),
            optimizer,
            param_spec,
            n_timesteps,
            data_structure,
            frozen_structure,
        )
        if cache is not None:
            cache[key] = entry
            while len(cache) > _MAX_CACHED_STEPS_PER_MODEL:
                cache.pop(next(iter(cache)))
        return entry.train_step, entry, cache, key

    def fit_sgd(
        self,
        *args,
        optimizer: object | None = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: float | None = None,
        **kwargs,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        *args, **kwargs
            Passed through ``_prepare_sgd_data`` and then to _sgd_loss_fn
            and _finalize_sgd.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.
        convergence_tol : float or None
            If set, stop early when the direction-agnostic relative change
            ``|ΔLL| / avg(|LL|) < tol`` for 5 consecutive steps.
            This is a dimensionless fraction (e.g., ``1e-4`` means 0.01%
            relative change), consistent with the EM convergence check.

        Returns
        -------
        log_likelihoods : list of float
            One entry per evaluated optimization step that produced a finite
            loss. When the final candidate is finite, the final entry is
            rewritten to the log likelihood of the stored final parameters.
        """
        # Validate the plain settings before the hook: ``_prepare_sgd_data`` may
        # mutate the model (e.g. record the sequence length).
        num_steps = validate_int(num_steps, "num_steps", nonnegative=True)
        args, kwargs = self._prepare_sgd_data(*args, **kwargs)

        self._check_sgd_initialized()
        params, param_spec = self._build_param_spec()

        if not param_spec or not any(spec.trainable for spec in param_spec.values()):
            raise ValueError("No learnable parameters — nothing to optimize.")

        frozen_params = {
            k: params[k] for k, spec in param_spec.items() if not spec.trainable
        }
        unc_params = transform_to_unconstrained(
            params,
            param_spec,
            include_non_trainable=False,
        )
        n_timesteps = float(self._n_timesteps)
        if not math.isfinite(n_timesteps) or n_timesteps <= 0.0:
            raise ValueError(
                "_n_timesteps must be positive and finite for SGD fitting."
            )
        if not _tree_all_finite(unc_params):
            raise ValueError(
                "Initial unconstrained SGD parameters contain NaN or inf. "
                "Check the model's initial parameter values."
            )

        if optimizer is None:
            optimizer = _DEFAULT_OPTIMIZER
        if not hasattr(optimizer, "init") or not hasattr(optimizer, "update"):
            raise ValueError(
                "optimizer must be an optax GradientTransformation with "
                "init and update methods."
            )
        opt_state = optimizer.init(unc_params)

        def _loss_inner(unc_p):
            p = transform_to_constrained(
                unc_p,
                param_spec,
                static_params=frozen_params,
            )
            return self._sgd_loss_fn(p, *args, **kwargs) / n_timesteps

        # The data and the frozen parameters are jit *arguments* (array leaves
        # only; other leaves stay compile-time constants and key the cache),
        # so a later fit_sgd call on this model with same-shaped data reuses
        # the compiled step instead of re-tracing and re-compiling it.
        data_dynamic, data_structure = _split_leaves((args, kwargs))
        frozen_dynamic, frozen_structure = _split_leaves(frozen_params)
        train_step, step_entry, step_cache, cache_key = self._compiled_sgd_step(
            optimizer, param_spec, n_timesteps, data_structure, frozen_structure
        )

        log_likelihoods: list[float] = []
        converged = False
        # last_valid_unc_params tracks the most recent params that
        # produced finite loss. It is updated ONLY after the finite check
        # and BEFORE swapping in the candidate, so on NaN recovery we
        # roll back to a set of params we actually confirmed as good —
        # not the post-update params that then produced NaN.
        last_valid_unc_params = unc_params
        stall_count = 0
        data_baked = False

        # Python loop (not lax.scan) to support NaN checks and verbose
        # logging without JIT closure issues with self.
        for step in range(num_steps):
            try:
                loss, new_unc_params, new_opt_state, step_finite = train_step(
                    unc_params, opt_state, frozen_dynamic, data_dynamic
                )
            except _DATA_TRACING_ERRORS:
                if step > 0 or data_baked:
                    raise
                data_baked = True
                # The loss needs concrete data while tracing (e.g. it validates
                # its inputs on the host): bake the data in as constants for
                # this call instead, uncached (the pre-cache behaviour).
                logger.debug(
                    "%s._sgd_loss_fn needs concrete data; compiling an uncached "
                    "SGD step with the data baked in.",
                    type(self).__name__,
                )
                if step_cache is not None:
                    step_cache.pop(cache_key, None)
                step_cache = None
                step_entry = _build_sgd_step(
                    lambda: self,
                    optimizer,
                    param_spec,
                    n_timesteps,
                    data_structure,
                    frozen_structure,
                    baked_leaves=(frozen_dynamic, data_dynamic),
                )
                step_entry.cacheable = False
                train_step = step_entry.train_step
                loss, new_unc_params, new_opt_state, step_finite = train_step(
                    unc_params, opt_state, frozen_dynamic, data_dynamic
                )

            # One device->host round trip per step; ``float`` / ``bool`` on
            # each value separately would block three times.
            loss_host, step_finite_host = jax.device_get((loss, step_finite))
            if not math.isfinite(loss_host):
                logger.warning(
                    "SGD step %d: NaN/inf loss — restoring last valid params "
                    "and stopping.",
                    step,
                )
                unc_params = last_valid_unc_params
                break

            ll = -float(loss_host) * n_timesteps
            log_likelihoods.append(ll)

            if not bool(step_finite_host):
                last_valid_unc_params = unc_params
                logger.warning(
                    "SGD step %d: NaN/inf gradient or parameter update -- keeping "
                    "the current finite-loss params and stopping.",
                    step,
                )
                break

            # train_step just confirmed these input params are finite.
            # Snapshot NOW, before swapping in the candidate — so that
            # on the next iteration's potential NaN, we can restore to
            # this known-good state rather than to the failing
            # post-update state.
            last_valid_unc_params = unc_params
            unc_params = new_unc_params
            opt_state = new_opt_state

            if verbose and (step % 10 == 0 or step == num_steps - 1):
                print(f"SGD step {step}: LL={ll:.2f}")

            if convergence_tol is not None and len(log_likelihoods) >= 2:
                # Use relative change for convergence, consistent with EM's
                # check_converged in utils.py. This avoids stalling too early
                # for problems with large total LL.
                avg = (abs(log_likelihoods[-1]) + abs(log_likelihoods[-2])) / 2
                rel_change = abs(log_likelihoods[-1] - log_likelihoods[-2]) / max(
                    avg, 1e-10
                )
                if rel_change < convergence_tol:
                    stall_count += 1
                else:
                    stall_count = 0
                if stall_count >= 5:
                    if verbose:
                        print(f"SGD converged at step {step}.")
                    logger.info("SGD converged at step %d.", step)
                    converged = True
                    break

        if step_cache is not None and not step_entry.cacheable:
            # A trace read state that cannot be fingerprinted: never reuse it.
            step_cache.pop(cache_key, None)

        final_loss = _loss_inner(unc_params)
        if not bool(jnp.isfinite(final_loss)):
            logger.warning(
                "Final SGD parameters produced NaN/inf loss -- restoring "
                "last valid params."
            )
            unc_params = last_valid_unc_params
        elif log_likelihoods:
            log_likelihoods[-1] = -float(final_loss) * n_timesteps

        final_params = transform_to_constrained(
            unc_params,
            param_spec,
            static_params=frozen_params,
        )
        self._store_sgd_params(final_params)
        self.log_likelihood_history_ = log_likelihoods
        self.converged_ = converged
        self._finalize_sgd(*args, **kwargs)

        return log_likelihoods

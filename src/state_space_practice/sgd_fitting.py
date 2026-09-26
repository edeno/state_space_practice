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

import logging
import math
from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.parameter_transforms import (
    transform_to_constrained,
    transform_to_unconstrained,
)
from state_space_practice.utils import validate_int

logger = logging.getLogger(__name__)


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
        import optax

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
            optimizer = optax.chain(
                optax.clip_by_global_norm(10.0),
                optax.adam(1e-2),
            )
        if not hasattr(optimizer, "init") or not hasattr(optimizer, "update"):
            raise ValueError(
                "optimizer must be an optax GradientTransformation with "
                "init and update methods."
            )
        opt_state = optimizer.init(unc_params)

        # jit fuses loss + grad + optimizer.update + apply_updates into a
        # single compiled graph. Without this the optimizer update and the
        # softplus/adam primitives dispatch one at a time through Python
        # (~65x slower on CPU; worse on GPU due to per-primitive sync).
        # Safe because nothing inside self mutates during the SGD loop —
        # _store_sgd_params only runs after the loop. If a subclass ever
        # mutates self attributes inside _sgd_loss_fn, jit will silently
        # freeze stale values; keep that invariant.
        def _loss_inner(unc_p):
            p = transform_to_constrained(
                unc_p,
                param_spec,
                static_params=frozen_params,
            )
            return self._sgd_loss_fn(p, *args, **kwargs) / n_timesteps

        @jax.jit
        def train_step(unc_p, opt_st):
            loss, grads = jax.value_and_grad(_loss_inner)(unc_p)
            updates, new_opt_st = optimizer.update(grads, opt_st, unc_p)
            new_unc_p = optax.apply_updates(unc_p, updates)
            step_finite = (
                _tree_all_finite_array(grads)
                & _tree_all_finite_array(updates)
                & _tree_all_finite_array(new_unc_p)
            )
            return loss, new_unc_p, new_opt_st, step_finite

        log_likelihoods: list[float] = []
        converged = False
        # last_valid_unc_params tracks the most recent params that
        # produced finite loss. It is updated ONLY after the finite check
        # and BEFORE swapping in the candidate, so on NaN recovery we
        # roll back to a set of params we actually confirmed as good —
        # not the post-update params that then produced NaN.
        last_valid_unc_params = unc_params
        stall_count = 0

        # Python loop (not lax.scan) to support NaN checks and verbose
        # logging without JIT closure issues with self.
        for step in range(num_steps):
            loss, new_unc_params, new_opt_state, step_finite = train_step(
                unc_params, opt_state
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

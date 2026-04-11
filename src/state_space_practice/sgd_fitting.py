"""Shared SGDFittableMixin for gradient-based model fitting.

Provides a generic fit_sgd() method that delegates to model-specific
hooks for parameter specification, loss computation, and post-optimization
finalization.

Models must implement:
- _build_param_spec() -> tuple[dict, dict]
- _sgd_loss_fn(params, *args, **kwargs) -> Array
- _store_sgd_params(params: dict) -> None
- _finalize_sgd(*args, **kwargs) -> None
- _check_sgd_initialized() -> None
- _n_timesteps: int (property or attribute)
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class SGDFittableMixin:
    """Mixin providing fit_sgd() for state-space models.

    Models must implement:
    - _build_param_spec() -> tuple[dict, dict]
    - _sgd_loss_fn(params, *args, **kwargs) -> Array
    - _store_sgd_params(params: dict) -> None
    - _finalize_sgd(*args, **kwargs) -> None
    - _check_sgd_initialized() -> None
    - _n_timesteps: int (property or attribute)
    """

    def fit_sgd(
        self,
        *args,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        **kwargs,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        *args, **kwargs
            Passed to _sgd_loss_fn and _finalize_sgd.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.
        convergence_tol : float or None
            If set, stop early when |ΔLL| < tol for 5 consecutive steps,
            where LL is the total (unnormalized) log-likelihood.

        Returns
        -------
        log_likelihoods : list of float
        """
        import optax

        from state_space_practice.parameter_transforms import (
            transform_to_constrained,
            transform_to_unconstrained,
        )

        self._check_sgd_initialized()
        params, param_spec = self._build_param_spec()

        if not param_spec:
            raise ValueError("No learnable parameters — nothing to optimize.")

        unc_params = transform_to_unconstrained(params, param_spec)
        n_timesteps = float(self._n_timesteps)

        # Build trainable mask and wrap optimizer
        # Frozen parameter handling uses two complementary mechanisms:
        # 1. stop_gradient in transform_to_constrained() zeroes gradients
        #    for non-trainable params — this is the correctness mechanism.
        # 2. optax.masked prevents the optimizer from accumulating momentum/
        #    second-moment state for frozen params — this is a resource
        #    optimization that also prevents stale optimizer state if params
        #    are later thawed.
        trainable_mask = {k: spec.trainable for k, spec in param_spec.items()}

        if optimizer is None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(10.0),
                optax.adam(1e-2),
            )
        optimizer = optax.masked(optimizer, trainable_mask)
        opt_state = optimizer.init(unc_params)

        # jit is essential for performance: without it, every step retraces
        # the full filter graph through Python dispatch (~65x slower on CPU,
        # more on GPU due to per-primitive host/device sync). Safe because
        # nothing inside self mutates during the SGD loop below —
        # _store_sgd_params only runs after the loop. If a subclass ever
        # mutates self attributes inside _sgd_loss_fn, jit will silently
        # freeze stale values; keep that invariant.
        @jax.jit
        @jax.value_and_grad
        def loss_fn(unc_p):
            p = transform_to_constrained(unc_p, param_spec)
            return self._sgd_loss_fn(p, *args, **kwargs) / n_timesteps

        log_likelihoods: list[float] = []
        # last_valid_unc_params tracks the most recent params that
        # produced finite loss. It is updated ONLY after the finite check
        # and BEFORE apply_updates, so on NaN recovery we roll back to a
        # set of params we actually confirmed as good — not the post-
        # update params that then produced NaN on the next iteration.
        last_valid_unc_params = unc_params
        stall_count = 0

        # Python loop (not lax.scan) to support NaN checks and verbose
        # logging without JIT closure issues with self.
        for step in range(num_steps):
            loss, grads = loss_fn(unc_params)

            if not jnp.isfinite(loss):
                logger.warning(
                    "SGD step %d: NaN/inf loss — restoring last valid params "
                    "and stopping.",
                    step,
                )
                unc_params = last_valid_unc_params
                break

            # loss_fn just confirmed these params are finite. Snapshot
            # NOW, before the update mutates them — so that on the next
            # iteration's potential NaN, we can restore to this known-
            # good state rather than to the failing post-update state.
            last_valid_unc_params = unc_params

            updates, opt_state = optimizer.update(grads, opt_state, unc_params)
            unc_params = optax.apply_updates(unc_params, updates)

            ll = -float(loss) * n_timesteps
            log_likelihoods.append(ll)

            if verbose and (step % 10 == 0 or step == num_steps - 1):
                logger.info("SGD step %d: LL=%.2f", step, ll)

            if convergence_tol is not None and len(log_likelihoods) >= 2:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < convergence_tol:
                    stall_count += 1
                else:
                    stall_count = 0
                if stall_count >= 5:
                    logger.info("SGD converged at step %d.", step)
                    break

        final_params = transform_to_constrained(unc_params, param_spec)
        self._store_sgd_params(final_params)
        self.log_likelihood_history_ = log_likelihoods
        self._finalize_sgd(*args, **kwargs)

        return log_likelihoods

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
            If set, stop early when the relative change
            ``|ΔLL| / avg(|LL|) < tol`` for 5 consecutive steps.
            This is a dimensionless fraction (e.g., ``1e-4`` means 0.01%
            relative change), consistent with the EM convergence check.

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

        # jit fuses loss + grad + optimizer.update + apply_updates into a
        # single compiled graph. Without this the optimizer update and the
        # softplus/adam primitives dispatch one at a time through Python
        # (~65x slower on CPU; worse on GPU due to per-primitive sync).
        # Safe because nothing inside self mutates during the SGD loop —
        # _store_sgd_params only runs after the loop. If a subclass ever
        # mutates self attributes inside _sgd_loss_fn, jit will silently
        # freeze stale values; keep that invariant.
        def _loss_inner(unc_p):
            p = transform_to_constrained(unc_p, param_spec)
            return self._sgd_loss_fn(p, *args, **kwargs) / n_timesteps

        @jax.jit
        def train_step(unc_p, opt_st):
            loss, grads = jax.value_and_grad(_loss_inner)(unc_p)
            updates, new_opt_st = optimizer.update(grads, opt_st, unc_p)
            new_unc_p = optax.apply_updates(unc_p, updates)
            return loss, new_unc_p, new_opt_st

        log_likelihoods: list[float] = []
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
            loss, new_unc_params, new_opt_state = train_step(
                unc_params, opt_state
            )

            if not jnp.isfinite(loss):
                logger.warning(
                    "SGD step %d: NaN/inf loss — restoring last valid params "
                    "and stopping.",
                    step,
                )
                unc_params = last_valid_unc_params
                break

            # train_step just confirmed these input params are finite.
            # Snapshot NOW, before swapping in the candidate — so that
            # on the next iteration's potential NaN, we can restore to
            # this known-good state rather than to the failing
            # post-update state.
            last_valid_unc_params = unc_params
            unc_params = new_unc_params
            opt_state = new_opt_state

            ll = -float(loss) * n_timesteps
            log_likelihoods.append(ll)

            if verbose and (step % 10 == 0 or step == num_steps - 1):
                print(f"SGD step {step}: LL={ll:.2f}")

            if convergence_tol is not None and len(log_likelihoods) >= 2:
                # Use relative change for convergence, consistent with EM's
                # check_converged in utils.py. This avoids stalling too early
                # for problems with large total LL.
                avg = (abs(log_likelihoods[-1]) + abs(log_likelihoods[-2])) / 2
                rel_change = abs(log_likelihoods[-1] - log_likelihoods[-2]) / max(avg, 1e-10)
                if rel_change < convergence_tol:
                    stall_count += 1
                else:
                    stall_count = 0
                if stall_count >= 5:
                    if verbose:
                        print(f"SGD converged at step {step}.")
                    logger.info("SGD converged at step %d.", step)
                    break

        final_params = transform_to_constrained(unc_params, param_spec)
        self._store_sgd_params(final_params)
        self.log_likelihood_history_ = log_likelihoods
        self._finalize_sgd(*args, **kwargs)

        return log_likelihoods

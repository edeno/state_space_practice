# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.parameter_transforms import (
    POSITIVE,
    STOCHASTIC_ROW,
    UNCONSTRAINED,
    frozen,
    transform_to_constrained,
    transform_to_unconstrained,
)
from state_space_practice.sgd_fitting import SGDFittableMixin


class TestStochasticRowTransform:
    def test_roundtrip_2x2(self) -> None:
        Z = jnp.array([[0.7, 0.3], [0.2, 0.8]])
        unc = STOCHASTIC_ROW.to_unconstrained(Z)
        recovered = STOCHASTIC_ROW.to_constrained(unc)
        np.testing.assert_allclose(recovered, Z, atol=1e-6)

    def test_roundtrip_3x3(self) -> None:
        Z = jnp.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        unc = STOCHASTIC_ROW.to_unconstrained(Z)
        recovered = STOCHASTIC_ROW.to_constrained(unc)
        np.testing.assert_allclose(recovered, Z, atol=1e-6)

    def test_rows_sum_to_one(self) -> None:
        logits = jnp.array([[1.0, -1.0], [-0.5, 0.5]])
        Z = STOCHASTIC_ROW.to_constrained(logits)
        np.testing.assert_allclose(Z.sum(axis=1), jnp.ones(2), atol=1e-7)

    def test_all_positive(self) -> None:
        logits = jnp.array([[10.0, -10.0], [-5.0, 5.0]])
        Z = STOCHASTIC_ROW.to_constrained(logits)
        assert jnp.all(Z > 0)

    def test_gradient_finite(self) -> None:
        def loss(logits):
            Z = STOCHASTIC_ROW.to_constrained(logits)
            return Z.sum()

        logits = jnp.array([[1.0, -1.0], [-0.5, 0.5]])
        g = jax.grad(loss)(logits)
        assert jnp.all(jnp.isfinite(g))

    def test_1d_vector(self) -> None:
        """Single row (e.g., initial state probabilities)."""
        pi = jnp.array([0.6, 0.3, 0.1])
        unc = STOCHASTIC_ROW.to_unconstrained(pi)
        recovered = STOCHASTIC_ROW.to_constrained(unc)
        np.testing.assert_allclose(recovered, pi, atol=1e-6)


class TestTrainableFlag:
    def test_default_trainable(self) -> None:
        assert POSITIVE.trainable is True
        assert UNCONSTRAINED.trainable is True
        assert STOCHASTIC_ROW.trainable is True

    def test_frozen_creates_non_trainable(self) -> None:
        f = frozen(POSITIVE)
        assert f.trainable is False
        # Transform functions should still work
        x = jnp.array(1.0)
        unc = f.to_unconstrained(x)
        recovered = f.to_constrained(unc)
        np.testing.assert_allclose(recovered, x, rtol=1e-5)

    def test_frozen_does_not_mutate_original(self) -> None:
        f = frozen(POSITIVE)
        assert POSITIVE.trainable is True
        assert f.trainable is False

    def test_stop_gradient_on_frozen(self) -> None:
        spec = {
            "learnable": POSITIVE,
            "frozen_param": frozen(POSITIVE),
        }
        params = {
            "learnable": jnp.array(1.0),
            "frozen_param": jnp.array(2.0),
        }
        unc = transform_to_unconstrained(params, spec)

        def loss(unc_p):
            p = transform_to_constrained(unc_p, spec)
            return p["learnable"] + p["frozen_param"]

        g = jax.grad(loss)(unc)
        # Gradient should flow through learnable param
        assert jnp.isfinite(g["learnable"])
        assert float(g["learnable"]) != 0.0
        # Gradient should be zero for frozen param (stop_gradient)
        assert float(g["frozen_param"]) == 0.0


class _ToyModel(SGDFittableMixin):
    """Minimal model for testing the mixin."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._initialized = True

    @property
    def _n_timesteps(self):
        return 100

    def _check_sgd_initialized(self):
        if not self._initialized:
            raise RuntimeError("Not initialized")

    def _build_param_spec(self):
        params = {"scale": jnp.array(self.scale)}
        spec = {"scale": POSITIVE}
        return params, spec

    def _sgd_loss_fn(self, params, target):
        # Simple quadratic loss: minimize (scale - target)^2 * n_timesteps
        return ((params["scale"] - target) ** 2) * self._n_timesteps

    def _store_sgd_params(self, params):
        self.scale = float(params["scale"])

    def _finalize_sgd(self, target):
        self.is_fitted = True


class TestSGDFittableMixin:
    def test_basic_optimization(self) -> None:
        import optax

        model = _ToyModel(scale=3.0)
        target = jnp.array(5.0)
        optimizer = optax.adam(1e-1)
        lls = model.fit_sgd(target, optimizer=optimizer, num_steps=200)
        # Scale should move toward target
        assert abs(model.scale - 5.0) < 0.5
        assert model.is_fitted

    def test_returns_log_likelihoods(self) -> None:
        model = _ToyModel(scale=0.1)
        lls = model.fit_sgd(jnp.array(5.0), num_steps=50)
        assert isinstance(lls, list)
        assert len(lls) == 50

    def test_convergence_tol(self) -> None:
        model = _ToyModel(scale=4.9)
        lls = model.fit_sgd(
            jnp.array(5.0), num_steps=500, convergence_tol=1e-8
        )
        # Should converge early
        assert len(lls) < 500

    def test_stores_log_likelihood_history(self) -> None:
        model = _ToyModel(scale=0.1)
        lls = model.fit_sgd(jnp.array(5.0), num_steps=20)
        assert hasattr(model, "log_likelihood_history_")
        assert model.log_likelihood_history_ == lls

    def test_no_learnable_params_raises(self) -> None:
        class _EmptyModel(SGDFittableMixin):
            _n_timesteps = 10

            def _check_sgd_initialized(self):
                pass

            def _build_param_spec(self):
                return {}, {}

            def _sgd_loss_fn(self, params):
                return jnp.array(0.0)

            def _store_sgd_params(self, params):
                pass

            def _finalize_sgd(self):
                pass

        with pytest.raises(ValueError, match="No learnable parameters"):
            _EmptyModel().fit_sgd(num_steps=10)

    def test_not_initialized_raises(self) -> None:
        model = _ToyModel(scale=1.0)
        model._initialized = False
        with pytest.raises(RuntimeError, match="Not initialized"):
            model.fit_sgd(jnp.array(5.0))

    def test_custom_optimizer(self) -> None:
        import optax

        model = _ToyModel(scale=0.1)
        optimizer = optax.adam(1e-1)
        lls = model.fit_sgd(jnp.array(5.0), optimizer=optimizer, num_steps=50)
        assert len(lls) == 50

    def test_nan_recovery_restores_last_finite_params(self) -> None:
        """Regression test: NaN recovery must roll back to params that
        were confirmed finite by a previous loss_fn call, NOT to the
        post-update params that just produced NaN.

        Earlier versions of the mixin assigned ``last_valid_unc_params``
        at the END of each loop iteration — AFTER ``optax.apply_updates``.
        Consequence: at step N+1 NaN, ``last_valid_unc_params`` was the
        post-step-N-update params — exactly the params the just-failed
        ``loss_fn`` was evaluated on. Restoration was a no-op, and
        downstream ``_finalize_sgd`` ran on the NaN-producing params.

        The fix captures the snapshot AFTER confirming finite loss,
        BEFORE ``apply_updates``. That guarantees the snapshot is a
        set of params that ``loss_fn`` has actually evaluated and
        approved.

        Test mechanic: use a loss that NaNs when ``scale`` crosses a
        threshold. The trajectory is linear (constant gradient), so
        each SGD step moves ``scale`` by a fixed amount. With the
        correct fix, after NaN recovery ``model.scale`` must equal the
        LAST finite-loss value, not the post-update value that
        triggered NaN.
        """

        class _NanAboveThreshold(SGDFittableMixin):
            """Loss = -scale, NaN once scale > 1.4."""

            def __init__(self):
                self.scale = 0.0
                self._initialized = True

            @property
            def _n_timesteps(self):
                return 100

            def _check_sgd_initialized(self):
                pass

            def _build_param_spec(self):
                return {"scale": jnp.array(self.scale)}, {
                    "scale": UNCONSTRAINED
                }

            def _sgd_loss_fn(self, params, _target):
                # Loss = -scale (maximize scale). NaN once scale > 1.4.
                return jnp.where(
                    params["scale"] > 1.4,
                    jnp.array(jnp.nan),
                    -params["scale"] * self._n_timesteps,
                )

            def _store_sgd_params(self, params):
                self.scale = float(params["scale"])

            def _finalize_sgd(self, _target):
                pass

        import optax

        # Gradient of -scale wrt scale is -1, so with SGD lr=0.5 each
        # step increments scale by 0.5. Trajectory:
        #   step 0: scale=0.0 → loss=0,    post=0.5 → last_valid=0.0
        #   step 1: scale=0.5 → loss=-0.5, post=1.0 → last_valid=0.5
        #   step 2: scale=1.0 → loss=-1.0, post=1.5 → last_valid=1.0
        #   step 3: scale=1.5 → loss=NaN  → restore to last_valid=1.0
        #
        # CORRECT fix (snapshot after finite confirmation, before update):
        #   model.scale after recovery == 1.0 (step 2's evaluated params)
        #
        # BUGGY version (snapshot at end of loop, after update):
        #   last_valid at NaN time would be 1.5 → restored model.scale == 1.5
        #
        # BROKEN alternative (snapshot at top of loop, before loss_fn):
        #   last_valid at top of step 3 is 1.5 → restored model.scale == 1.5
        #   — same as buggy, because unc_params doesn't change between
        #   end of prev iter and top of next iter.
        #
        # Only the correct fix restores to 1.0.
        model = _NanAboveThreshold()
        optimizer = optax.sgd(learning_rate=0.5)

        lls = model.fit_sgd(
            jnp.array(0.0), optimizer=optimizer, num_steps=10
        )

        # Finite LL for steps 0, 1, 2 (loss = 0, -0.5, -1.0 → LL = 0, 50, 100)
        assert len(lls) == 3, f"expected 3 finite steps, got {len(lls)}"
        assert all(np.isfinite(ll) for ll in lls)
        np.testing.assert_allclose(lls, [0.0, 50.0, 100.0], atol=1e-6)

        # The critical assertion: model.scale must equal the LAST
        # finite-loss scale value (1.0), NOT the post-update value
        # from step 2 (1.5) that triggered NaN at step 3.
        assert np.isfinite(model.scale)
        np.testing.assert_allclose(model.scale, 1.0, atol=1e-6)

    def test_nan_at_step_zero_is_noop_recovery(self) -> None:
        """If the very first loss evaluation returns NaN, recovery has
        nothing valid to restore — the user's init was already bad.
        The contract is: ``model.scale`` is left at its init value,
        ``lls`` is empty, and ``fit_sgd`` does not crash.
        """

        class _AlwaysNan(SGDFittableMixin):
            def __init__(self):
                self.scale = 2.0  # already above the NaN threshold
                self._initialized = True

            @property
            def _n_timesteps(self):
                return 100

            def _check_sgd_initialized(self):
                pass

            def _build_param_spec(self):
                return {"scale": jnp.array(self.scale)}, {
                    "scale": UNCONSTRAINED
                }

            def _sgd_loss_fn(self, params, _target):
                return jnp.where(
                    params["scale"] > 1.4,
                    jnp.array(jnp.nan),
                    -params["scale"] * self._n_timesteps,
                )

            def _store_sgd_params(self, params):
                self.scale = float(params["scale"])

            def _finalize_sgd(self, _target):
                pass

        model = _AlwaysNan()
        lls = model.fit_sgd(jnp.array(0.0), num_steps=10)
        assert lls == []
        np.testing.assert_allclose(model.scale, 2.0, atol=1e-6)

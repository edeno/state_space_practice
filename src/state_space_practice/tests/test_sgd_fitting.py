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

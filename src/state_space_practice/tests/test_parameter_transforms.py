# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import random

from state_space_practice.parameter_transforms import (
    POSITIVE,
    PSD_MATRIX,
    UNCONSTRAINED,
    UNIT_INTERVAL,
    transform_to_constrained,
    transform_to_unconstrained,
)


class TestPositiveTransform:
    def test_roundtrip(self) -> None:
        x = jnp.array([0.01, 1.0, 100.0])
        unc = POSITIVE.to_unconstrained(x)
        recovered = POSITIVE.to_constrained(unc)
        np.testing.assert_allclose(recovered, x, rtol=1e-5)

    def test_always_positive(self) -> None:
        unc = jnp.array([-10.0, 0.0, 10.0])
        x = POSITIVE.to_constrained(unc)
        assert jnp.all(x > 0)

    def test_gradient_finite(self) -> None:
        grad_fn = jax.grad(lambda unc: POSITIVE.to_constrained(unc).sum())
        g = grad_fn(jnp.array([0.0, 1.0, -1.0]))
        assert jnp.all(jnp.isfinite(g))
        assert jnp.all(g > 0)  # exp is always positive


class TestUnitIntervalTransform:
    def test_roundtrip(self) -> None:
        x = jnp.array([0.01, 0.5, 0.99])
        recovered = UNIT_INTERVAL.to_constrained(UNIT_INTERVAL.to_unconstrained(x))
        np.testing.assert_allclose(recovered, x, rtol=1e-5)

    def test_bounds(self) -> None:
        unc = jnp.linspace(-10, 10, 100)
        x = UNIT_INTERVAL.to_constrained(unc)
        assert jnp.all(x > 0) and jnp.all(x < 1)

    def test_gradient_finite(self) -> None:
        grad_fn = jax.grad(lambda unc: UNIT_INTERVAL.to_constrained(unc).sum())
        g = grad_fn(jnp.array([0.0, 2.0, -2.0]))
        assert jnp.all(jnp.isfinite(g))


class TestUnconstrainedTransform:
    def test_identity(self) -> None:
        x = jnp.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(UNCONSTRAINED.to_unconstrained(x), x)
        np.testing.assert_allclose(UNCONSTRAINED.to_constrained(x), x)


class TestPSDMatrixTransform:
    def test_roundtrip(self) -> None:
        P = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        unc = PSD_MATRIX.to_unconstrained(P)
        recovered = PSD_MATRIX.to_constrained(unc)
        np.testing.assert_allclose(recovered, P, atol=1e-5)

    def test_always_psd(self) -> None:
        unc = jnp.array([0.5, -0.3, 0.1])  # arbitrary reals for 2x2
        P = PSD_MATRIX.to_constrained(unc)
        eigvals = jnp.linalg.eigvalsh(P)
        assert jnp.all(eigvals > 0)

    def test_roundtrip_3x3(self) -> None:
        key = random.PRNGKey(42)
        L = jnp.tril(random.normal(key, (3, 3)))
        L = L.at[jnp.diag_indices(3)].set(jnp.abs(jnp.diag(L)) + 0.1)
        P = L @ L.T
        unc = PSD_MATRIX.to_unconstrained(P)
        recovered = PSD_MATRIX.to_constrained(unc)
        np.testing.assert_allclose(recovered, P, atol=1e-5)

    def test_gradient_flows(self) -> None:
        def loss(unc):
            P = PSD_MATRIX.to_constrained(unc)
            return jnp.trace(P)

        unc = jnp.array([0.5, -0.3, 0.1])
        g = jax.grad(loss)(unc)
        assert jnp.all(jnp.isfinite(g))


class TestDictTransforms:
    def test_roundtrip(self) -> None:
        spec = {"q": POSITIVE, "beta": POSITIVE, "decay": UNIT_INTERVAL}
        params = {
            "q": jnp.array(0.01),
            "beta": jnp.array(2.0),
            "decay": jnp.array(0.95),
        }
        unc = transform_to_unconstrained(params, spec)
        recovered = transform_to_constrained(unc, spec)
        for k in params:
            np.testing.assert_allclose(recovered[k], params[k], rtol=1e-5)

    def test_gradient_through_dict(self) -> None:
        spec = {"q": POSITIVE, "decay": UNIT_INTERVAL}
        params = {"q": jnp.array(0.1), "decay": jnp.array(0.9)}
        unc = transform_to_unconstrained(params, spec)

        def loss(unc_params):
            p = transform_to_constrained(unc_params, spec)
            return p["q"] + p["decay"]

        g = jax.grad(loss)(unc)
        assert all(jnp.isfinite(g[k]) for k in g)

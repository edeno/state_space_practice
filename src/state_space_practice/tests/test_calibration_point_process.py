"""Simulation-based calibration of the Laplace-EKF point-process smoothers.

If parameters and data are drawn from the model and the smoother is run at
the true parameters, the standardised errors ``z = (x_true - m_{t|T}) /
sd_{t|T}`` of an *exact* posterior satisfy ``E[z] = 0`` and ``E[z^2] = 1``
exactly (whatever the shape of the posterior), and for a near-Gaussian
posterior the ``|z| < 1.645`` coverage is ~0.90. Pooling ``z`` over many
small replicates therefore tests the Laplace smoother's means *and*
variances, which a single-seed RMSE threshold cannot.

Tolerances. The ``z`` of one replicate are correlated (over time and
latent dimensions), so standard errors are computed from the per-replicate
averages of ``z``, ``z^2`` and the coverage indicator (a cluster-robust SE
with the replicate as the cluster): ``SE = std(replicate averages) /
sqrt(n_replicates)``. Calibration assertions use +-4 SE (a false-alarm
probability ~1e-4 per statistic), plus the pinned-value assertions where
the approximation is measurably miscalibrated.

All tests run full filter/smoother pipelines and are marked slow.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.place_field_model import PlaceFieldModel
from state_space_practice.point_process_kalman import stochastic_point_process_smoother
from state_space_practice.position_decoder import (
    PlaceFieldRateMaps,
    position_decoder_smoother,
)

_Z90 = 1.6448536269514722  # two-sided 90% normal quantile


def _affine_log_rate(design_t, x):
    """log lambda_n = b_n + w_n . x with ``design_t[n] = [b_n, w_n]``."""
    return design_t[:, 0] + design_t[:, 1:] @ x


def _calibration_stats(z: np.ndarray) -> dict:
    """Pooled mean / variance / 90% coverage of ``z`` and cluster SEs.

    ``z`` has shape ``(n_replicates, ...)``; each replicate is a cluster.
    """
    n_rep = z.shape[0]
    flat = z.reshape(n_rep, -1)
    stats = {}
    for name, values in (
        ("mean", flat),
        ("second_moment", flat**2),
        ("coverage", (np.abs(flat) < _Z90).astype(float)),
    ):
        per_rep = values.mean(axis=1)
        stats[name] = float(per_rep.mean())
        stats[f"{name}_se"] = float(per_rep.std(ddof=1) / np.sqrt(n_rep))
    stats["n"] = int(flat.size)
    return stats


def _describe(stats: dict) -> str:
    return (
        f"mean={stats['mean']:.4f}+-{stats['mean_se']:.4f}, "
        f"E[z^2]={stats['second_moment']:.4f}+-{stats['second_moment_se']:.4f}, "
        f"coverage90={stats['coverage']:.4f}+-{stats['coverage_se']:.4f} "
        f"(n={stats['n']})"
    )


def _assert_calibrated(stats: dict, n_se: float = 4.0) -> None:
    msg = _describe(stats)
    assert abs(stats["mean"]) < n_se * stats["mean_se"], msg
    assert abs(stats["second_moment"] - 1.0) < n_se * stats["second_moment_se"], msg
    assert abs(stats["coverage"] - 0.9) < n_se * stats["coverage_se"], msg
    # Guard: the SEs are small enough for the test to have power against a
    # 10% variance miscalibration.
    assert stats["second_moment_se"] < 0.025, msg


# ---------------------------------------------------------------------------
# Point-process smoother (random AR(1) models, Poisson GLM observations)
# ---------------------------------------------------------------------------


def _simulate_replicates(rng, n_rep, n_time, n_latent, n_neurons, rate_range, dt):
    """Draw (params, x_{1:T}, spikes) for ``n_rep`` random models."""
    a = rng.uniform(0.85, 0.98, (n_rep, n_latent))
    q = rng.uniform(0.01, 0.05, (n_rep, n_latent))
    init_mean = rng.normal(0.0, 0.3, (n_rep, n_latent))
    init_var = rng.uniform(0.1, 0.3, (n_rep, n_latent))
    weights = rng.normal(0.0, 0.7, (n_rep, n_neurons, n_latent))
    baseline = np.log(rng.uniform(*rate_range, (n_rep, n_neurons)))
    x = init_mean + np.sqrt(init_var) * rng.normal(size=(n_rep, n_latent))
    xs, spikes = [], []
    for _ in range(n_time):
        x = a * x + np.sqrt(q) * rng.normal(size=(n_rep, n_latent))
        rate = np.exp(baseline + np.einsum("rnd,rd->rn", weights, x))
        xs.append(x)
        spikes.append(rng.poisson(rate * dt))
    design = np.concatenate([baseline[..., None], weights], axis=-1)
    design = np.broadcast_to(design[:, None], (n_rep, n_time) + design.shape[1:])
    params = {
        "init_mean": init_mean,
        "init_cov": np.einsum("rd,de->rde", init_var, np.eye(n_latent)),
        "A": np.einsum("rd,de->rde", a, np.eye(n_latent)),
        "Q": np.einsum("rd,de->rde", q, np.eye(n_latent)),
        "design": np.ascontiguousarray(design),
    }
    return params, np.stack(xs, 1), np.stack(spikes, 1).astype(float)


def _pp_smoother_z(seed, rate_range, n_rep=2000, n_time=20, n_latent=2, n_neurons=3):
    rng = np.random.default_rng(seed)
    dt = 0.02
    params, x_true, spikes = _simulate_replicates(
        rng, n_rep, n_time, n_latent, n_neurons, rate_range, dt
    )

    def smooth_one(m0, P0, A, Q, design, y):
        sm, sc, _, _ = stochastic_point_process_smoother(
            m0, P0, design, y, dt, A, Q, _affine_log_rate, max_newton_iter=3
        )
        return sm, jnp.diagonal(sc, axis1=-2, axis2=-1)

    sm, sv = jax.jit(jax.vmap(smooth_one))(
        jnp.asarray(params["init_mean"]),
        jnp.asarray(params["init_cov"]),
        jnp.asarray(params["A"]),
        jnp.asarray(params["Q"]),
        jnp.asarray(params["design"]),
        jnp.asarray(spikes),
    )
    z = (x_true - np.asarray(sm)) / np.sqrt(np.asarray(sv))
    return z, spikes


@pytest.mark.slow
class TestPointProcessSmootherCalibration:
    """2000 random 2-latent / 3-neuron models, T=20 (80000 pooled z)."""

    def test_moderate_rate_smoother_is_calibrated(self) -> None:
        """Rates 20-60 Hz at dt=20 ms (0.4-1.2 expected spikes / bin / neuron).

        Observed (seed 0): mean 0.007+-0.011, E[z^2] 1.009+-0.013,
        coverage 0.899+-0.002.
        """
        z, spikes = _pp_smoother_z(seed=0, rate_range=(20.0, 60.0))
        # Guard: the regime is informative (spikes in most bins).
        assert spikes.mean() > 0.4
        _assert_calibrated(_calibration_stats(z))

    def test_low_rate_smoother_is_calibrated(self) -> None:
        """Rates 2-6 Hz at dt=20 ms (0.04-0.12 expected spikes / bin / neuron).

        Most bins have no spike, so each update is small and the posterior
        stays close to the (Gaussian) prior; the skew of the zero-count
        likelihood (the ~0.13 sd mode-vs-mean offsets seen against the exact
        posterior in ``test_oracle_point_process``) averages out over
        replicates, and the pooled calibration is indistinguishable from
        exact. Observed (seed 1): mean -0.017+-0.012, E[z^2] 1.011+-0.014,
        coverage 0.898+-0.003 -- so this asserts calibration rather than
        pinning a miscalibrated coverage.
        """
        z, spikes = _pp_smoother_z(seed=1, rate_range=(2.0, 6.0))
        assert spikes.mean() < 0.15  # guard: really low-rate
        stats = _calibration_stats(z)
        _assert_calibrated(stats)


# ---------------------------------------------------------------------------
# Position decoder (synthetic arena, known rate maps)
# ---------------------------------------------------------------------------


def _synthetic_rate_maps(rng, n_side=5, extent=60.0, peak_hz=25.0, width=10.0):
    """Gaussian place fields on a jittered lattice, 1 cm grid."""
    edges = np.linspace(0.0, extent, int(extent) + 1)
    gx, gy = np.meshgrid(edges, edges)  # (n_grid_y, n_grid_x)
    lattice = np.linspace(0.1 * extent, 0.9 * extent, n_side)
    centers = np.stack(np.meshgrid(lattice, lattice), -1).reshape(-1, 2)
    centers = centers + rng.normal(0.0, 2.0, centers.shape)
    d2 = (gx[None] - centers[:, 0, None, None]) ** 2 + (
        gy[None] - centers[:, 1, None, None]
    ) ** 2
    rates = 1.0 + peak_hz * np.exp(-0.5 * d2 / width**2)
    return PlaceFieldRateMaps(rates, edges, edges)


def _decoder_z(seed, n_rep=600, n_time=40, dt=0.02, q_pos=50.0):
    """Standardised smoothed position errors pooled over replicates.

    Trajectories are drawn from the decoder's own random-walk prior
    (``include_velocity=False``: per-step covariance ``q_pos * dt * I``)
    started from ``N(init_position, init_cov)``, and spikes from the same
    bilinear log-rate the decoder evaluates, so the model is exactly
    specified.
    """
    rng = np.random.default_rng(seed)
    rate_maps = _synthetic_rate_maps(rng)
    init_cov = 4.0 * np.eye(2)
    log_rate = jax.jit(jax.vmap(rate_maps.log_rate))
    zs, n_spikes = [], []
    for _ in range(n_rep):
        init_position = rng.uniform(20.0, 40.0, 2)
        x0 = init_position + 2.0 * rng.normal(size=2)
        steps = np.sqrt(q_pos * dt) * rng.normal(size=(n_time, 2))
        path = x0 + np.cumsum(steps, axis=0)
        rates = np.exp(np.asarray(log_rate(jnp.asarray(path))))
        spikes = rng.poisson(rates * dt).astype(float)
        result = position_decoder_smoother(
            spikes,
            rate_maps,
            dt,
            q_pos=q_pos,
            include_velocity=False,
            init_position=init_position,
            init_cov=init_cov,
        )
        mean = np.asarray(result.position_mean)
        sd = np.sqrt(np.diagonal(np.asarray(result.position_cov), axis1=1, axis2=2))
        zs.append((path - mean) / sd)
        n_spikes.append(spikes.sum(axis=1).mean())
    return np.stack(zs), float(np.mean(n_spikes))


# ---------------------------------------------------------------------------
# PlaceFieldModel smoother at the true parameters
# ---------------------------------------------------------------------------


def _place_field_z(seed, n_rep=600, n_time=300, dt=0.02, q=1e-4, init_var=0.05):
    """Standardised smoothed log-rate errors at the animal's position.

    One fixed trajectory (and spline design); per replicate the weights are
    drawn from the model's random-walk prior around a place-field-shaped
    ``init_mean`` and spikes from ``Poisson(exp(Z_t x_t) dt)``. The E-step
    runs at the true ``(A = I, Q = q I, init_mean, init_cov = init_var I)``.
    The error is taken on ``eta_t = Z_t x_t`` (the log-rate), whose
    posterior variance is ``Z_t P_{t|T} Z_t'``: weights of basis functions
    the animal never visits are prior-dominated and uninformative.
    """
    rng = np.random.default_rng(seed)
    # Two laps of a noisy loop through a 100 x 100 cm arena.
    phase = 4.0 * np.pi * np.arange(n_time) / n_time
    position = 50.0 + 35.0 * np.stack([np.cos(phase), np.sin(phase)], axis=1)
    position += rng.normal(0.0, 2.0, position.shape)
    model = PlaceFieldModel(dt=dt, n_interior_knots=1)
    design = model._build_design_matrix(position)
    Z = np.asarray(design)
    n_basis = Z.shape[1]
    bump = np.exp(-0.5 * np.sum((position - [55.0, 45.0]) ** 2, axis=1) / 20.0**2)
    target = np.log(10.0) + 1.5 * bump
    init_mean = np.linalg.lstsq(Z, target, rcond=None)[0]
    model.transition_matrix = jnp.eye(n_basis)
    model.process_cov = q * jnp.eye(n_basis)
    model.init_mean = jnp.asarray(init_mean)
    model.init_cov = init_var * jnp.eye(n_basis)
    # patsy's bs() drops the first B-spline, so the basis row is exactly 0
    # at the lower data bound in x or y: eta is deterministically 0 there
    # and carries no calibration information.
    informative = np.linalg.norm(Z, axis=1) > 1e-8
    assert informative.mean() > 0.98
    zs, n_spikes = [], []
    for _ in range(n_rep):
        x = init_mean + np.sqrt(init_var) * rng.normal(size=n_basis)
        weights = x + np.cumsum(np.sqrt(q) * rng.normal(size=(n_time, n_basis)), 0)
        eta = np.sum(Z * weights, axis=1)
        spikes = rng.poisson(np.exp(eta) * dt).astype(float)
        model._e_step(design, jnp.asarray(spikes))
        sm = np.asarray(model.smoother_mean)
        sc = np.asarray(model.smoother_cov)
        eta_mean = np.sum(Z * sm, axis=1)
        eta_var = np.einsum("tb,tbc,tc->t", Z, sc, Z)
        zs.append((eta - eta_mean)[informative] / np.sqrt(eta_var[informative]))
        n_spikes.append(spikes.mean())
    return np.stack(zs), float(np.mean(n_spikes))


@pytest.mark.slow
class TestPositionDecoderCalibration:
    def test_smoothed_position_calibration_is_pinned(self) -> None:
        """600 trajectories x 40 bins x 2 coordinates on a 60 x 60 cm arena
        with 25 Gaussian place fields (~2.5 spikes / bin in total).

        Observed (seed 0): mean 0.054+-0.018, E[z^2] 1.094+-0.021, coverage
        0.888+-0.004 (seed 1: -0.010, 1.092, 0.886). The smoothed position
        covariance is ~9% too small: the decoder evaluates the rate maps by
        bilinear interpolation of the log-rate, which is piecewise linear in
        position, so the Fisher precision ``J' diag(lambda dt) J`` uses a
        Jacobian that is constant within a grid cell and jumps across cell
        edges, and the Laplace curvature at the mode misses the kinks'
        contribution to the posterior spread. This is an approximation
        property of the decoder's likelihood surrogate, not of the Laplace
        filter (whose own calibration is tested above), so the observed
        values are pinned (+-4 SE) and the overconfidence is asserted to
        stay mild.
        """
        z, spikes_per_bin = _decoder_z(seed=0)
        assert spikes_per_bin > 1.0  # guard: informative spiking
        stats = _calibration_stats(z)
        msg = _describe(stats)
        assert abs(stats["mean"]) < 4 * stats["mean_se"], msg
        assert abs(stats["second_moment"] - 1.094) < 4 * stats["second_moment_se"], msg
        assert abs(stats["coverage"] - 0.888) < 4 * stats["coverage_se"], msg
        # Mild: variance at most ~20% too small, coverage within 3 points.
        assert 0.95 < stats["second_moment"] < 1.2, msg
        assert stats["coverage"] > 0.87, msg


@pytest.mark.slow
class TestPlaceFieldSmootherCalibration:
    def test_smoother_at_true_parameters_is_calibrated(self) -> None:
        """600 replicates x ~300 bins, 16-weight spline model, ~0.28 spikes/bin.

        Observed (seed 0): mean -0.001+-0.020, E[z^2] 1.000+-0.021,
        coverage 0.901+-0.004. Over seeds 0-2, E[z^2] is 0.96-1.00 and
        coverage 0.901-0.908: at most slightly *under*-confident (~2 SE; the
        Laplace variance at the mode of a low-count Poisson likelihood can
        exceed the posterior variance a little), within the +-4 SE
        tolerance.
        """
        z, spikes_per_bin = _place_field_z(seed=0)
        assert spikes_per_bin > 0.1  # guard
        _assert_calibrated(_calibration_stats(z))

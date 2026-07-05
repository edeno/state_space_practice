# ruff: noqa: E402
"""Tests for the temporal log-Gaussian Cox process (Matern-3/2 rate GP).

The state-space iterated-Laplace inference is checked against an independent,
dense Gaussian-process Laplace implementation (a plain NumPy Newton solve on the
full Matern-3/2 Gram matrix). Because the Matern-3/2 SDE reproduces the Matern
kernel exactly, the two share the *same* prior, so the posterior mode and the
Laplace log-evidence must agree to numerical precision -- a strong oracle that a
subtly wrong evidence formula (or an unconverged mode) cannot pass.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy.special import gammaln

from state_space_practice.temporal_rate_gp import (
    TemporalRateGP,
    infer_log_rate,
    infer_log_rate_batch,
    poisson_log_rate_site,
)


def _sample_matern32_latent(
    n_time: int, dt: float, variance: float, lengthscale: float, seed: int
) -> np.ndarray:
    """Draw one zero-mean Matern-3/2 GP path via a dense Cholesky factor."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_time) * dt
    tau = np.abs(times[:, None] - times[None, :])
    scaled = np.sqrt(3.0) * tau / lengthscale
    gram = variance * (1.0 + scaled) * np.exp(-scaled) + 1e-8 * np.eye(n_time)
    return np.linalg.cholesky(gram) @ rng.standard_normal(n_time)


# --- Dense GP-Laplace oracle (independent NumPy reference) ---------------------


def _matern32_gram(
    times: np.ndarray, variance: float, lengthscale: float
) -> np.ndarray:
    tau = np.abs(times[:, None] - times[None, :])
    scaled = np.sqrt(3.0) * tau / lengthscale
    return variance * (1.0 + scaled) * np.exp(-scaled)


def _dense_laplace_lgcp(
    counts: np.ndarray,
    dt: float,
    variance: float,
    lengthscale: float,
    mean: float = 0.0,
    n_newton: int = 300,
    jitter: float = 1e-9,
    min_weight: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Return (posterior mode of the zero-mean latent g, Laplace log-evidence).

    Model: ``y_t ~ Poisson(exp(mean + g_t + log dt))``, ``g ~ N(0, K)`` with ``K``
    the Matern-3/2 Gram matrix. Evidence is Rasmussen & Williams eq. 3.32.
    """
    counts = np.asarray(counts, dtype=float)
    n_time = counts.size
    times = np.arange(n_time) * dt
    offset = mean + np.log(dt)
    gram = _matern32_gram(times, variance, lengthscale) + jitter * np.eye(n_time)
    gram_inv = np.linalg.inv(gram)

    g = np.zeros(n_time)
    for _ in range(n_newton):
        rate = np.exp(g + offset)
        weight = np.maximum(rate, min_weight)
        gradient = (counts - rate) - gram_inv @ g
        step = np.linalg.solve(np.diag(weight) + gram_inv, gradient)
        g = g + step
        if np.max(np.abs(step)) < 1e-13:
            break

    rate = np.exp(g + offset)
    weight = np.maximum(rate, min_weight)
    log_lik = np.sum(counts * (g + offset) - rate - gammaln(counts + 1.0))
    weight_sqrt = np.sqrt(weight)
    b_matrix = np.eye(n_time) + weight_sqrt[:, None] * gram * weight_sqrt[None, :]
    log_evidence = (
        -0.5 * g @ (gram_inv @ g) + log_lik - 0.5 * np.linalg.slogdet(b_matrix)[1]
    )
    return g, float(log_evidence)


# --- Fixtures -----------------------------------------------------------------


@pytest.fixture
def small_counts() -> jnp.ndarray:
    """A short, reproducible count sequence for the dense oracle comparisons."""
    key = random.PRNGKey(0)
    n_time, dt = 24, 0.1
    times = jnp.arange(n_time) * dt
    true_rate = jnp.exp(0.7 * jnp.sin(2.0 * jnp.pi * times / 1.5))  # around 1 Hz
    return random.poisson(key, true_rate * dt).astype(float)


# --- Poisson IRLS site --------------------------------------------------------


def test_poisson_site_matches_irls_working_response():
    """The site is the Poisson IRLS linearization: R = 1/rate, y~ = g + (y-rate)/rate."""
    g = jnp.array([0.0, 0.5, -0.3, 1.2])
    counts = jnp.array([2.0, 0.0, 5.0, 3.0])
    offset = jnp.log(0.1)

    working_response, site_variance, expected_count = poisson_log_rate_site(
        g, counts, offset
    )

    rate = jnp.exp(g + offset)
    np.testing.assert_allclose(expected_count, rate, rtol=1e-10)
    np.testing.assert_allclose(site_variance, 1.0 / rate, rtol=1e-10)
    np.testing.assert_allclose(working_response, g + (counts - rate) / rate, rtol=1e-8)


@pytest.mark.parametrize("bad_min_weight", [0.0, -1.0, np.nan, np.inf])
def test_poisson_site_rejects_invalid_min_weight(bad_min_weight):
    with pytest.raises(ValueError, match="min_weight"):
        poisson_log_rate_site(
            jnp.array([0.0]),
            jnp.array([0.0]),
            jnp.array(0.0),
            min_weight=bad_min_weight,
        )


# --- Dense-oracle correctness -------------------------------------------------


def test_posterior_mode_matches_dense_gp_laplace(small_counts):
    """State-space Laplace mode equals the dense GP-Laplace mode."""
    variance, lengthscale, dt = 1.5, 0.4, 0.1
    dense_g, _ = _dense_laplace_lgcp(
        np.asarray(small_counts), dt, variance, lengthscale
    )
    result = infer_log_rate(small_counts, dt, variance, lengthscale, n_iter=40)
    # mean=0, so the returned log-rate mean IS the zero-mean latent g.
    np.testing.assert_allclose(
        np.asarray(result.log_rate_mean), dense_g, atol=1e-4, rtol=1e-4
    )


def test_log_evidence_matches_dense_gp_laplace(small_counts):
    """State-space Laplace log-evidence equals R&W eq. 3.32 on the dense GP."""
    variance, lengthscale, dt = 1.5, 0.4, 0.1
    _, dense_evidence = _dense_laplace_lgcp(
        np.asarray(small_counts), dt, variance, lengthscale
    )
    result = infer_log_rate(small_counts, dt, variance, lengthscale, n_iter=40)
    # An O(1)-or-worse error (the failure mode of an approximate energy) would
    # blow past this; require agreement to a few 1e-2 on an O(10) quantity.
    assert abs(float(result.log_marginal_likelihood) - dense_evidence) < 2e-2


def test_evidence_prefers_true_lengthscale_over_dense(small_counts):
    """Evidence ranks candidate lengthscales the same way the dense oracle does."""
    variance, dt = 1.5, 0.1
    lengthscales = [0.15, 0.4, 1.2]
    ssm = [
        float(
            infer_log_rate(
                small_counts, dt, variance, ell, n_iter=40
            ).log_marginal_likelihood
        )
        for ell in lengthscales
    ]
    dense = [
        _dense_laplace_lgcp(np.asarray(small_counts), dt, variance, ell)[1]
        for ell in lengthscales
    ]
    assert int(np.argmax(ssm)) == int(np.argmax(dense))


def test_laplace_iteration_converges(small_counts):
    """The reported final Newton update is at the mode (near machine zero)."""
    result = infer_log_rate(small_counts, 0.1, 1.5, 0.4, n_iter=40)
    assert float(result.max_abs_update) < 1e-8


@pytest.mark.parametrize("bad_n_iter", [0, -1, 1.5])
def test_infer_log_rate_rejects_invalid_n_iter(small_counts, bad_n_iter):
    with pytest.raises(ValueError, match="n_iter"):
        infer_log_rate(small_counts, 0.1, 1.5, 0.4, n_iter=bad_n_iter)


def test_infer_log_rate_rejects_empty_counts():
    with pytest.raises(ValueError, match="at least one"):
        infer_log_rate(jnp.array([]), 0.1, 1.5, 0.4)


@pytest.mark.parametrize("bad_min_weight", [0.0, -1.0, np.nan, np.inf])
def test_infer_log_rate_rejects_invalid_min_weight(small_counts, bad_min_weight):
    with pytest.raises(ValueError, match="min_weight"):
        infer_log_rate(small_counts, 0.1, 1.5, 0.4, min_weight=bad_min_weight)


def test_nonzero_mean_offsets_log_rate(small_counts):
    """A baseline log-rate `mean` shifts the posterior mode by that offset in the

    dense oracle too (the latent g is defined zero-mean, f = mean + g)."""
    variance, lengthscale, dt, mean = 1.5, 0.4, 0.1, 1.3
    dense_g, _ = _dense_laplace_lgcp(
        np.asarray(small_counts), dt, variance, lengthscale, mean=mean
    )
    result = infer_log_rate(
        small_counts, dt, variance, lengthscale, mean=mean, n_iter=40
    )
    # Returned log-rate mean is f = mean + g.
    np.testing.assert_allclose(
        np.asarray(result.log_rate_mean), mean + dense_g, atol=1e-4, rtol=1e-4
    )


# --- Behavioral: recovery, calibration, differentiability ---------------------


@pytest.mark.slow
def test_recovers_known_smooth_rate_better_than_empirical():
    """Posterior rate is closer to the true smooth rate than the raw empirical rate."""
    key = random.PRNGKey(1)
    n_time, dt = 500, 0.02
    times = jnp.arange(n_time) * dt
    true_log_rate = 1.0 + 0.9 * jnp.sin(2.0 * jnp.pi * times / 2.0)
    true_rate = jnp.exp(true_log_rate)
    counts = random.poisson(key, true_rate * dt).astype(float)

    result = infer_log_rate(
        counts, dt, variance=1.0, lengthscale=0.3, mean=1.0, n_iter=30
    )
    # Log-normal posterior mean of the rate.
    rate_est = jnp.exp(result.log_rate_mean + 0.5 * result.log_rate_var)
    empirical_rate = counts / dt

    rmse_gp = float(jnp.sqrt(jnp.mean((rate_est - true_rate) ** 2)))
    rmse_empirical = float(jnp.sqrt(jnp.mean((empirical_rate - true_rate) ** 2)))
    assert rmse_gp < 0.5 * rmse_empirical


@pytest.mark.slow
def test_credible_band_covers_truth():
    """A 95% posterior band covers most of the true log-rate."""
    key = random.PRNGKey(2)
    n_time, dt = 500, 0.02
    times = jnp.arange(n_time) * dt
    true_log_rate = 1.0 + 0.9 * jnp.sin(2.0 * jnp.pi * times / 2.0)
    counts = random.poisson(key, jnp.exp(true_log_rate) * dt).astype(float)

    result = infer_log_rate(
        counts, dt, variance=1.0, lengthscale=0.3, mean=1.0, n_iter=30
    )
    sd = jnp.sqrt(result.log_rate_var)
    lo = result.log_rate_mean - 1.96 * sd
    hi = result.log_rate_mean + 1.96 * sd
    coverage = float(jnp.mean((true_log_rate >= lo) & (true_log_rate <= hi)))
    assert coverage > 0.85


@pytest.mark.slow
def test_evidence_gradient_matches_finite_difference(small_counts):
    """d(log evidence)/d(log variance, log lengthscale) matches finite differences."""
    dt = 0.1

    def log_evidence(log_theta):
        variance = jnp.exp(log_theta[0])
        lengthscale = jnp.exp(log_theta[1])
        return infer_log_rate(
            small_counts, dt, variance, lengthscale, n_iter=40
        ).log_marginal_likelihood

    log_theta = jnp.log(jnp.array([1.5, 0.4]))
    grad = jax.grad(log_evidence)(log_theta)

    eps = 1e-5
    fd = np.zeros(2)
    for i in range(2):
        bump = log_theta.at[i].add(eps)
        drop = log_theta.at[i].add(-eps)
        fd[i] = float((log_evidence(bump) - log_evidence(drop)) / (2 * eps))
    np.testing.assert_allclose(np.asarray(grad), fd, rtol=1e-3, atol=1e-3)


# --- TemporalRateGP model class -----------------------------------------------


def test_predict_before_fit_raises():
    model = TemporalRateGP(dt=0.1)
    with pytest.raises(RuntimeError):
        model.predict_rate()


@pytest.mark.parametrize("bad_min_weight", [0.0, -1.0, np.nan, np.inf])
def test_temporal_rate_gp_rejects_invalid_min_weight(bad_min_weight):
    with pytest.raises(ValueError, match="min_weight"):
        TemporalRateGP(dt=0.1, min_weight=bad_min_weight)


def test_predict_rate_positive_and_correct_shape(small_counts):
    model = TemporalRateGP(dt=0.1, variance=1.0, lengthscale=0.4, mean=0.0, n_iter=15)
    model.fit_sgd(small_counts, num_steps=3)
    rate = model.predict_rate()
    assert rate.shape == small_counts.shape
    assert bool(jnp.all(rate > 0.0))


def test_credible_interval_brackets_median(small_counts):
    model = TemporalRateGP(dt=0.1, variance=1.0, lengthscale=0.4, n_iter=15)
    model.fit_sgd(small_counts, num_steps=3)
    lo, hi = model.credible_interval(level=0.95)
    log_rate_mean, _ = model.predict_log_rate()
    median_rate = jnp.exp(log_rate_mean)
    assert bool(jnp.all(lo > 0.0))
    assert bool(jnp.all(hi > lo))
    assert bool(jnp.all((median_rate >= lo) & (median_rate <= hi)))


def test_frozen_mean_is_not_updated(small_counts):
    model = TemporalRateGP(
        dt=0.1, variance=1.0, lengthscale=0.4, mean=0.7, update_mean=False, n_iter=12
    )
    model.fit_sgd(small_counts, num_steps=6)
    assert float(model.mean_) == pytest.approx(0.7, abs=1e-12)


def test_learned_variance_and_lengthscale_stay_positive(small_counts):
    model = TemporalRateGP(dt=0.1, variance=1.0, lengthscale=0.4, n_iter=12)
    model.fit_sgd(small_counts, num_steps=8)
    assert float(model.variance_) > 0.0
    assert float(model.lengthscale_) > 0.0


@pytest.mark.slow
def test_fit_sgd_improves_evidence_and_moves_lengthscale_toward_truth():
    """On GP-simulated data, fitting increases the evidence and pulls the

    lengthscale from a deliberately-too-smooth start toward the true value."""
    lengthscale_true, variance_true = 0.3, 1.0
    n_time, dt, baseline = 500, 0.02, 2.0
    latent = _sample_matern32_latent(
        n_time, dt, variance_true, lengthscale_true, seed=0
    )
    rate = np.exp(baseline + latent)  # Hz
    rng = np.random.default_rng(1)
    counts = jnp.asarray(rng.poisson(rate * dt).astype(float))

    lengthscale_init = lengthscale_true * 4.0
    model = TemporalRateGP(
        dt=dt,
        variance=1.0,
        lengthscale=lengthscale_init,
        mean=baseline,
        n_iter=15,
    )
    log_likelihoods = model.fit_sgd(counts, num_steps=120)

    assert log_likelihoods[-1] > log_likelihoods[0]  # evidence improved
    # Started too smooth; learned lengthscale should shrink toward the truth.
    assert float(model.lengthscale_) < lengthscale_init
    assert 0.4 * lengthscale_true < float(model.lengthscale_) < 3.0 * lengthscale_true


# --- Multi-neuron batching ----------------------------------------------------


@pytest.fixture
def multineuron_counts() -> jnp.ndarray:
    """Reproducible (n_neurons, n_time) counts with per-neuron baselines/phases."""
    key = random.PRNGKey(3)
    n_neurons, n_time, dt = 4, 24, 0.1
    times = jnp.arange(n_time) * dt
    keys = random.split(key, n_neurons)
    baselines = jnp.array([0.5, 0.0, -0.3, 0.8])
    rows = []
    for j in range(n_neurons):
        rate = jnp.exp(baselines[j] + 0.6 * jnp.sin(2.0 * jnp.pi * times / 1.5 + j))
        rows.append(random.poisson(keys[j], rate * dt).astype(float))
    return jnp.stack(rows)  # (n_neurons, n_time)


def test_batch_inference_matches_per_neuron_loop(multineuron_counts):
    """Batched inference with shared hyperparameters equals a per-neuron loop."""
    dt, variance, lengthscale, mean = 0.1, 1.2, 0.4, 0.0
    batch = infer_log_rate_batch(
        multineuron_counts, dt, variance, lengthscale, mean, n_iter=40
    )
    assert batch.log_rate_mean.shape == multineuron_counts.shape
    assert batch.log_marginal_likelihood.shape == (multineuron_counts.shape[0],)
    for j in range(multineuron_counts.shape[0]):
        single = infer_log_rate(
            multineuron_counts[j], dt, variance, lengthscale, mean, n_iter=40
        )
        np.testing.assert_allclose(
            np.asarray(batch.log_rate_mean[j]),
            np.asarray(single.log_rate_mean),
            rtol=1e-7,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            float(batch.log_marginal_likelihood[j]),
            float(single.log_marginal_likelihood),
            rtol=1e-7,
        )


def test_batch_inference_per_neuron_hyperparameters(multineuron_counts):
    """Per-neuron (n_neurons,) hyperparameters route to the matching single fit."""
    n_neurons = multineuron_counts.shape[0]
    dt = 0.1
    variance = jnp.linspace(0.8, 1.6, n_neurons)
    lengthscale = jnp.linspace(0.2, 0.6, n_neurons)
    mean = jnp.linspace(-0.2, 0.4, n_neurons)
    batch = infer_log_rate_batch(
        multineuron_counts, dt, variance, lengthscale, mean, n_iter=40
    )
    for j in range(n_neurons):
        single = infer_log_rate(
            multineuron_counts[j],
            dt,
            float(variance[j]),
            float(lengthscale[j]),
            float(mean[j]),
            n_iter=40,
        )
        np.testing.assert_allclose(
            np.asarray(batch.log_rate_mean[j]),
            np.asarray(single.log_rate_mean),
            rtol=1e-7,
            atol=1e-7,
        )


def test_batch_inference_rejects_wrong_hyperparameter_length(multineuron_counts):
    with pytest.raises(ValueError, match="n_neurons"):
        infer_log_rate_batch(
            multineuron_counts, 0.1, jnp.array([1.0, 1.0]), 0.4, 0.0, n_iter=10
        )


def test_batch_inference_rejects_empty_counts():
    with pytest.raises(ValueError, match="at least one"):
        infer_log_rate_batch(jnp.empty((2, 0)), 0.1, 1.0, 0.4)
    with pytest.raises(ValueError, match="at least one"):
        infer_log_rate_batch(jnp.empty((0, 3)), 0.1, 1.0, 0.4)


@pytest.mark.parametrize("bad_min_weight", [0.0, -1.0, np.nan, np.inf])
def test_batch_inference_rejects_invalid_min_weight(
    multineuron_counts, bad_min_weight
):
    with pytest.raises(ValueError, match="min_weight"):
        infer_log_rate_batch(
            multineuron_counts,
            0.1,
            1.0,
            0.4,
            min_weight=bad_min_weight,
        )


def test_fit_sgd_rejects_empty_counts():
    model = TemporalRateGP(dt=0.1)
    with pytest.raises(ValueError, match="at least one"):
        model.fit_sgd(jnp.array([]), num_steps=0)


def test_fit_sgd_multineuron_shared_shapes(multineuron_counts):
    n_neurons = multineuron_counts.shape[0]
    model = TemporalRateGP(dt=0.1, variance=1.0, lengthscale=0.4, mean=0.0, n_iter=12)
    model.fit_sgd(multineuron_counts, num_steps=4)
    # Shared variance/lengthscale are scalars; baseline mean is per-neuron.
    assert np.ndim(model.variance_) == 0
    assert np.ndim(model.lengthscale_) == 0
    assert np.asarray(model.mean_).shape == (n_neurons,)
    rate = model.predict_rate()
    assert rate.shape == multineuron_counts.shape
    assert bool(jnp.all(rate > 0.0))


def test_fit_sgd_multineuron_per_neuron_hyperparameters(multineuron_counts):
    n_neurons = multineuron_counts.shape[0]
    model = TemporalRateGP(
        dt=0.1,
        variance=1.0,
        lengthscale=0.4,
        n_iter=12,
        share_hyperparameters=False,
    )
    model.fit_sgd(multineuron_counts, num_steps=4)
    assert np.asarray(model.variance_).shape == (n_neurons,)
    assert np.asarray(model.lengthscale_).shape == (n_neurons,)


def test_single_neuron_path_unchanged_by_batch_support(small_counts):
    """A 1D fit still yields scalar hyperparameters and a 1D rate."""
    model = TemporalRateGP(dt=0.1, variance=1.0, lengthscale=0.4, n_iter=12)
    model.fit_sgd(small_counts, num_steps=4)
    assert np.ndim(model.variance_) == 0
    assert model.predict_rate().shape == small_counts.shape


@pytest.mark.slow
def test_fit_sgd_multineuron_shared_recovers_lengthscale():
    """Pooling neurons with a shared lengthscale recovers the true value."""
    lengthscale_true, variance_true = 0.3, 1.0
    n_neurons, n_time, dt, baseline = 6, 400, 0.02, 2.0
    rows = []
    for j in range(n_neurons):
        latent = _sample_matern32_latent(
            n_time, dt, variance_true, lengthscale_true, seed=j
        )
        rng = np.random.default_rng(100 + j)
        rows.append(rng.poisson(np.exp(baseline + latent) * dt).astype(float))
    counts = jnp.asarray(np.stack(rows))

    model = TemporalRateGP(
        dt=dt,
        variance=1.0,
        lengthscale=lengthscale_true * 4.0,
        mean=baseline,
        n_iter=15,
    )
    log_likelihoods = model.fit_sgd(counts, num_steps=100)
    assert log_likelihoods[-1] > log_likelihoods[0]
    assert 0.4 * lengthscale_true < float(model.lengthscale_) < 3.0 * lengthscale_true

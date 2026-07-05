r"""Temporal firing-rate estimation as a log-Gaussian Cox process.

Estimates a smooth, time-varying firing rate from binned spike counts:

.. math::

    y_k \sim \mathrm{Poisson}(\exp(\mu + g_k)\,\Delta t),
    \qquad g \sim \mathrm{GP}(0, k_{\mathrm{Matern\text{-}3/2}}),

where ``mu`` is a baseline log-rate and ``g`` is a zero-mean Gaussian process
whose smoothness (``variance``, ``lengthscale``) is a learnable hyperparameter.
This is a log-Gaussian Cox process (Moller et al., 1998); the GP intensity view
for neural data is Cunningham et al. (2008) and Adams et al. (2009).

The GP prior is represented as the Matern-3/2 linear-Gaussian SDE in
:mod:`state_space_practice.gp_ssm`, so inference reuses the Kalman
filter/smoother in :mod:`state_space_practice.kalman`. The only non-Gaussian
piece is the per-bin Poisson likelihood, which is handled by **iterated Laplace /
Gauss-Newton** smoothing: at each iteration the Poisson likelihood is replaced by
its local Gaussian (IRLS) site and a linear-Gaussian RTS smoother is run; the
iteration is repeated to the posterior mode. For a canonical (log) link the
Poisson observed and expected Hessians coincide, so this is exact Newton on the
concave log-posterior and converges to the true mode (Rasmussen & Williams,
2006, ch. 3; Nickisch et al., 2018).

Numerical precision
-------------------
Like the rest of the Laplace-EKF stack, long sequences require float64. Enable it
before importing::

    import jax
    jax.config.update("jax_enable_x64", True)

References
----------
Moller, J., Syversveen, A. & Waagepetersen, R. (1998). Log Gaussian Cox
    Processes. Scand. J. Statist. 25(3), 451-482.
Cunningham, J., Shenoy, K. & Sahani, M. (2008). Fast Gaussian Process Methods for
    Point Process Intensity Estimation. ICML.
Rasmussen, C. E. & Williams, C. K. I. (2006). Gaussian Processes for Machine
    Learning. MIT Press. (Ch. 3: Laplace approximation; eq. 3.32.)
Nickisch, H., Solin, A. & Grigorevskiy, A. (2018). State Space Gaussian Processes
    with Non-Gaussian Likelihood. ICML, PMLR 80:3789-3798.
"""

import operator
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln, ndtri
from jax.typing import ArrayLike

from state_space_practice.gp_ssm import matern32_continuous, matern32_discretize
from state_space_practice.kalman import kalman_smoother
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import (
    _validate_filter_numerics,
    validate_count_array,
    validate_scalar,
)

#: Default floor on the Fisher weight ``rate`` to keep the site variance finite
#: at (near-)zero-rate bins. exp(-20)/dt Hz is far below any real firing rate.
_DEFAULT_MIN_WEIGHT = 1e-9


class LaplaceRateResult(NamedTuple):
    """Posterior summary from :func:`infer_log_rate`.

    Attributes
    ----------
    log_rate_mean : Array, shape (n_time,)
        Posterior mode of the log-rate ``f = mean + g``.
    log_rate_var : Array, shape (n_time,)
        Posterior (Laplace) variance of the log-rate at the mode.
    log_marginal_likelihood : Array, shape ()
        Laplace approximation to ``log p(y | hyperparameters)`` (scalar array).
    max_abs_update : Array, shape ()
        Infinity-norm of the final Newton update to ``g`` (a convergence
        diagnostic; small means the iteration reached the mode).
    """

    log_rate_mean: Array
    log_rate_var: Array
    log_marginal_likelihood: Array
    max_abs_update: Array


def poisson_log_rate_site(
    latent: ArrayLike,
    counts: ArrayLike,
    offset: ArrayLike,
    min_weight: float = _DEFAULT_MIN_WEIGHT,
) -> tuple[Array, Array, Array]:
    r"""Local Gaussian (IRLS) site for the Poisson log-rate likelihood.

    Linearizing ``log p(y | g) = y (g + offset) - exp(g + offset)`` at the
    current ``g`` gives a Gaussian pseudo-observation with Fisher weight
    ``W = exp(g + offset)`` (the expected count), working response
    ``y~ = g + (y - W) / W``, and site variance ``R = 1 / W``. A linear-Gaussian
    update against ``(y~, R)`` is one Newton step on the Poisson log-posterior.

    Parameters
    ----------
    latent : ArrayLike, shape (n_time,)
        Current zero-mean latent ``g`` (log-rate minus baseline).
    counts : ArrayLike, shape (n_time,)
        Observed spike counts per bin.
    offset : ArrayLike
        Constant added inside the exponential, ``mean + log(dt)`` (scalar or
        broadcastable). ``exp(g + offset)`` is the expected count per bin.
    min_weight : float
        Floor on the Fisher weight to keep ``R`` finite at near-zero rate.

    Returns
    -------
    working_response : Array, shape (n_time,)
        IRLS working response ``y~``.
    site_variance : Array, shape (n_time,)
        Site variance ``R = 1 / W``.
    expected_count : Array, shape (n_time,)
        ``W = exp(g + offset)``, the expected count per bin.
    """
    latent = jnp.asarray(latent)
    counts = jnp.asarray(counts)
    expected_count = jnp.exp(latent + offset)
    weight = jnp.maximum(expected_count, min_weight)
    working_response = latent + (counts - expected_count) / weight
    site_variance = 1.0 / weight
    return working_response, site_variance, expected_count


def _infer_log_rate_traced(
    counts: Array,
    dt: ArrayLike,
    variance: ArrayLike,
    lengthscale: ArrayLike,
    mean: ArrayLike,
    n_iter: int,
    min_weight: float,
) -> LaplaceRateResult:
    """Traced core of :func:`infer_log_rate` (safe under ``jit`` / ``grad``).

    Does no host-side validation, so hyperparameters may be tracers. The public
    :func:`infer_log_rate` validates concrete inputs before delegating here.
    """
    offset = mean + jnp.log(dt)
    _F, _L, _Qc, measurement_vector, stationary_cov = matern32_continuous(
        variance, lengthscale, validate=False
    )
    transition, process_cov = matern32_discretize(
        variance, lengthscale, dt, validate=False
    )
    measurement_matrix = measurement_vector[None, :]  # (1, 2)
    init_mean = jnp.zeros(2)
    n_time = counts.shape[0]

    def _smooth_at(g: Array) -> tuple[Array, Array, Array]:
        working_response, site_variance, _ = poisson_log_rate_site(
            g, counts, offset, min_weight
        )
        smoother_mean, smoother_cov, _cross, marginal_ll = kalman_smoother(
            init_mean,
            stationary_cov,
            working_response[:, None],
            transition,
            process_cov,
            measurement_matrix,
            site_variance[:, None, None],
            validate_inputs=False,
        )
        return smoother_mean[:, 0], smoother_cov[:, 0, 0], marginal_ll

    def _newton_step(g: Array, _: None) -> tuple[Array, Array]:
        g_new, _var, _ll = _smooth_at(g)
        return g_new, jnp.max(jnp.abs(g_new - g))

    g_mode, updates = jax.lax.scan(_newton_step, jnp.zeros(n_time), None, length=n_iter)
    max_abs_update = updates[-1]

    # Evaluate the Laplace evidence at the converged mode. The Kalman filter's
    # marginal likelihood of the mode's Gaussian sites, corrected by (true
    # Poisson - Gaussian site) log-likelihoods, is the state-space form of the
    # GP-Laplace evidence (Rasmussen & Williams eq. 3.32; Nickisch et al. 2018).
    working_response, site_variance, expected_count = poisson_log_rate_site(
        g_mode, counts, offset, min_weight
    )
    _mode_again, smoother_cov, _cross, kf_marginal_ll = kalman_smoother(
        init_mean,
        stationary_cov,
        working_response[:, None],
        transition,
        process_cov,
        measurement_matrix,
        site_variance[:, None, None],
        validate_inputs=False,
    )
    log_rate_var = smoother_cov[:, 0, 0]

    true_log_lik = jnp.sum(
        counts * (g_mode + offset) - expected_count - gammaln(counts + 1.0)
    )
    site_log_lik = jnp.sum(
        -0.5
        * (
            jnp.log(2.0 * jnp.pi * site_variance)
            + (working_response - g_mode) ** 2 / site_variance
        )
    )
    log_marginal_likelihood = kf_marginal_ll + true_log_lik - site_log_lik

    return LaplaceRateResult(
        log_rate_mean=jnp.asarray(mean + g_mode),
        log_rate_var=log_rate_var,
        log_marginal_likelihood=log_marginal_likelihood,
        max_abs_update=max_abs_update,
    )


def infer_log_rate(
    counts: ArrayLike,
    dt: ArrayLike,
    variance: ArrayLike,
    lengthscale: ArrayLike,
    mean: ArrayLike = 0.0,
    n_iter: int = 25,
    min_weight: float = _DEFAULT_MIN_WEIGHT,
) -> LaplaceRateResult:
    r"""Infer the posterior log-rate for fixed Matern-3/2 hyperparameters.

    Runs iterated Laplace / Gauss-Newton smoothing to the posterior mode and
    returns the mode, its Laplace variance, and the Laplace log-evidence (used
    to learn hyperparameters). Inference for fixed hyperparameters is exact up to
    the Laplace approximation of the Poisson likelihood.

    Parameters
    ----------
    counts : ArrayLike, shape (n_time,)
        Non-negative integer spike counts per bin.
    dt : ArrayLike
        Bin width in seconds (strictly positive).
    variance : ArrayLike
        Marginal prior variance of the log-rate fluctuations (strictly positive).
    lengthscale : ArrayLike
        Correlation time of the log-rate in seconds (strictly positive).
    mean : ArrayLike, default 0.0
        Baseline log-rate ``mu``; the prior mean of ``f = mu + g``.
    n_iter : int, default 25
        Number of Newton iterations. The iteration is a fixed-length scan so the
        evidence stays differentiable for hyperparameter learning; the concave
        Poisson-GP posterior converges quadratically, so the default is ample and
        extra steps are stable no-ops. ``max_abs_update`` reports convergence.
    min_weight : float
        Floor on the Fisher weight; see :func:`poisson_log_rate_site`.

    Returns
    -------
    LaplaceRateResult
        Posterior mode, variance, log-evidence, and convergence diagnostic.

    Notes
    -----
    Differentiable in ``variance``, ``lengthscale``, and ``mean`` (host-side
    validation of those is skipped when they are tracers), so
    ``jax.grad(lambda th: infer_log_rate(..., *th).log_marginal_likelihood)``
    gives the marginal-likelihood gradient used by
    :class:`TemporalRateGP.fit_sgd`.
    """
    counts = jnp.asarray(counts)
    if counts.ndim != 1:
        raise ValueError(f"counts must be 1D (n_time,), got shape {counts.shape}.")
    validate_count_array(counts, "counts")
    try:
        n_iter = operator.index(n_iter)
    except TypeError as exc:
        raise ValueError("n_iter must be a positive integer.") from exc
    if n_iter <= 0:
        raise ValueError("n_iter must be a positive integer.")
    # Validate scalar hyperparameters only when concrete: under jax.grad they
    # arrive as tracers, and host-side float() checks would break tracing. The
    # model layer constrains them positive via parameter_transforms.
    for value, name in (
        (dt, "dt"),
        (variance, "variance"),
        (lengthscale, "lengthscale"),
    ):
        if not isinstance(value, jax.core.Tracer):
            validate_scalar(value, name, positive=True)
    if not isinstance(mean, jax.core.Tracer):
        validate_scalar(mean, "mean")

    return _infer_log_rate_traced(
        counts, dt, variance, lengthscale, mean, n_iter, min_weight
    )


class TemporalRateGP(SGDFittableMixin):
    r"""Temporal firing-rate GP with marginal-likelihood hyperparameter learning.

    Wraps :func:`infer_log_rate` in the package's SGD-fitting convention: the
    Matern-3/2 ``variance`` and ``lengthscale`` (and, optionally, the baseline
    log-rate ``mean``) are learned by maximizing the Laplace log-evidence via
    :meth:`fit_sgd` (optax Adam on the unconstrained hyperparameters, with
    positivity enforced through :mod:`state_space_practice.parameter_transforms`).

    After fitting, :meth:`predict_rate` returns the posterior mean firing rate and
    :meth:`credible_interval` a posterior band.

    Parameters
    ----------
    dt : float
        Bin width in seconds (strictly positive).
    variance : float, default 1.0
        Initial marginal prior variance of the log-rate fluctuations.
    lengthscale : float, default 1.0
        Initial correlation time of the log-rate in seconds.
    mean : float, default 0.0
        Initial baseline log-rate ``mu``.
    n_iter : int, default 25
        Newton iterations per inference call (see :func:`infer_log_rate`).
    update_variance, update_lengthscale, update_mean : bool, default True
        Whether each hyperparameter is learned. A frozen hyperparameter is held
        at its initial value.
    min_weight : float
        Fisher-weight floor; see :func:`poisson_log_rate_site`.

    Attributes
    ----------
    log_rate_mean_, log_rate_var_ : Array, shape (n_time,)
        Posterior mode and variance of the log-rate, set by :meth:`fit_sgd`.
    log_marginal_likelihood_ : float
        Laplace log-evidence at the fitted hyperparameters.
    log_likelihood_history_ : list of float
        Log-evidence at each SGD step (from the mixin).
    """

    def __init__(
        self,
        dt: float,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        mean: float = 0.0,
        n_iter: int = 25,
        update_variance: bool = True,
        update_lengthscale: bool = True,
        update_mean: bool = True,
        min_weight: float = _DEFAULT_MIN_WEIGHT,
    ) -> None:
        self.dt = validate_scalar(dt, "dt", positive=True)
        self.variance = validate_scalar(variance, "variance", positive=True)
        self.lengthscale = validate_scalar(lengthscale, "lengthscale", positive=True)
        self.mean = validate_scalar(mean, "mean")
        try:
            self.n_iter = operator.index(n_iter)
        except TypeError as exc:
            raise ValueError("n_iter must be a positive integer.") from exc
        if self.n_iter <= 0:
            raise ValueError("n_iter must be a positive integer.")
        self.update_variance = bool(update_variance)
        self.update_lengthscale = bool(update_lengthscale)
        self.update_mean = bool(update_mean)
        self.min_weight = float(min_weight)

        # Data and posterior, populated by fit_sgd.
        self._counts: Optional[Array] = None
        self._sgd_n_time: int = 0
        self.log_rate_mean_: Optional[Array] = None
        self.log_rate_var_: Optional[Array] = None
        self.log_marginal_likelihood_: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"TemporalRateGP(dt={self.dt}, variance={self.variance:.4g}, "
            f"lengthscale={self.lengthscale:.4g}, mean={self.mean:.4g})"
        )

    # -- fitted-hyperparameter accessors --------------------------------------

    @property
    def variance_(self) -> float:
        return self.variance

    @property
    def lengthscale_(self) -> float:
        return self.lengthscale

    @property
    def mean_(self) -> float:
        return self.mean

    # -- public fitting / prediction ------------------------------------------

    def fit_sgd(  # type: ignore[override]
        self,
        counts: ArrayLike,
        *,
        num_steps: int = 200,
        optimizer: Optional[object] = None,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
    ) -> list[float]:
        """Learn hyperparameters by maximizing the Laplace log-evidence.

        Parameters
        ----------
        counts : ArrayLike, shape (n_time,)
            Non-negative integer spike counts per bin.
        num_steps, optimizer, verbose, convergence_tol
            Forwarded to :meth:`SGDFittableMixin.fit_sgd`.

        Returns
        -------
        list of float
            Log-evidence at each optimization step.
        """
        counts = jnp.asarray(counts)
        if counts.ndim != 1:
            raise ValueError(f"counts must be 1D (n_time,), got shape {counts.shape}.")
        validate_count_array(counts, "counts")
        self._counts = counts
        self._sgd_n_time = int(counts.shape[0])

        # Numerical-precision guard: the Laplace-EKF smoother needs float64 for
        # long sequences. Validate the stationary prior once (also warns on
        # f32 + long T), matching the other filter entry points.
        _, _, _, _, stationary_cov = matern32_continuous(
            self.variance, self.lengthscale
        )
        _validate_filter_numerics(
            stationary_cov,
            n_time=self._sgd_n_time,
            stacklevel=3,
            filter_name="TemporalRateGP.fit_sgd",
        )

        return super().fit_sgd(
            counts,
            num_steps=num_steps,
            optimizer=optimizer,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    def _check_fitted(self) -> None:
        if self.log_rate_mean_ is None:
            raise RuntimeError("Model is not fitted. Call fit_sgd(counts) first.")

    def predict_log_rate(self) -> tuple[Array, Array]:
        """Posterior mode and variance of the log-rate ``f`` (after fitting)."""
        self._check_fitted()
        assert self.log_rate_mean_ is not None and self.log_rate_var_ is not None
        return self.log_rate_mean_, self.log_rate_var_

    def predict_rate(self) -> Array:
        r"""Posterior mean firing rate ``E[exp(f)] = exp(f_mean + 0.5 f_var)`` in Hz.

        The rate is log-normal under the Gaussian posterior on ``f``, so its mean
        includes the ``+0.5 * variance`` Jensen correction over the plug-in
        ``exp(f_mean)``.
        """
        self._check_fitted()
        assert self.log_rate_mean_ is not None and self.log_rate_var_ is not None
        return jnp.exp(self.log_rate_mean_ + 0.5 * self.log_rate_var_)

    def credible_interval(self, level: float = 0.95) -> tuple[Array, Array]:
        """Posterior credible band on the firing rate.

        Parameters
        ----------
        level : float, default 0.95
            Central probability mass (0 < level < 1).

        Returns
        -------
        lower, upper : Array, shape (n_time,)
            Band on ``exp(f)`` from the Gaussian posterior on ``f``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}.")
        self._check_fitted()
        assert self.log_rate_mean_ is not None and self.log_rate_var_ is not None
        z = ndtri(0.5 * (1.0 + level))
        sd = jnp.sqrt(self.log_rate_var_)
        lower = jnp.exp(self.log_rate_mean_ - z * sd)
        upper = jnp.exp(self.log_rate_mean_ + z * sd)
        return lower, upper

    # -- SGDFittableMixin hooks ------------------------------------------------

    @property
    def _n_timesteps(self) -> int:
        return self._sgd_n_time

    def _check_sgd_initialized(self) -> None:
        if self._counts is None:
            raise RuntimeError(
                "No data. Call fit_sgd(counts), not SGDFittableMixin.fit_sgd() "
                "directly."
            )

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            POSITIVE,
            UNCONSTRAINED,
            frozen,
        )

        params = {
            "variance": jnp.asarray(self.variance),
            "lengthscale": jnp.asarray(self.lengthscale),
            "mean": jnp.asarray(self.mean),
        }
        spec = {
            "variance": POSITIVE if self.update_variance else frozen(POSITIVE),
            "lengthscale": POSITIVE if self.update_lengthscale else frozen(POSITIVE),
            "mean": UNCONSTRAINED if self.update_mean else frozen(UNCONSTRAINED),
        }
        return params, spec

    def _sgd_loss_fn(self, params: dict, counts: Array) -> Array:
        result = _infer_log_rate_traced(
            counts,
            self.dt,
            params["variance"],
            params["lengthscale"],
            params["mean"],
            self.n_iter,
            self.min_weight,
        )
        return -result.log_marginal_likelihood

    def _store_sgd_params(self, params: dict) -> None:
        self.variance = float(params["variance"])
        self.lengthscale = float(params["lengthscale"])
        self.mean = float(params["mean"])

    def _finalize_sgd(self, counts: Array) -> None:
        result = infer_log_rate(
            counts,
            self.dt,
            self.variance,
            self.lengthscale,
            self.mean,
            n_iter=self.n_iter,
            min_weight=self.min_weight,
        )
        self.log_rate_mean_ = result.log_rate_mean
        self.log_rate_var_ = result.log_rate_var
        self.log_marginal_likelihood_ = float(result.log_marginal_likelihood)

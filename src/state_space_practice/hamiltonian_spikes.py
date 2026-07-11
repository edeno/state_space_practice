"""Hamiltonian Spike Model (Point-Process Observation).

Uses Hamiltonian dynamics with a Poisson/Point-Process readout for spike data.

See docs/hamiltonian_architecture.md for why this family is standalone
(no linear-Gaussian EM integration, SGD-only fitting).
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.hamiltonian_core import (
    _BaseModelStubs,
    default_init_mean,
    ekf_rts_backward_pass,
    mlp_l2_penalty,
    point_process_laplace_update,
)
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_predict_step_with_jacobian,
    init_mlp_params,
    leapfrog_step,
)
from state_space_practice.oscillator_models import BaseModel
from state_space_practice.parameter_transforms import (
    PSD_MATRIX,
    POSITIVE,
    UNCONSTRAINED,
    ParameterTransform,
    frozen,
)
from state_space_practice.point_process_kalman import _soft_expected_count_and_log
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import stabilize_covariance, validate_count_array


class HamiltonianSpikeModel(_BaseModelStubs, BaseModel, SGDFittableMixin):
    """Spike Model with Hamiltonian dynamics and Point-Process observations."""

    def __init__(
        self,
        n_oscillators: int,
        n_sources: int,  # n_neurons
        sampling_freq: float,
        hidden_dims: Optional[List[int]] = None,
        seed: int = 42,
    ):
        super().__init__(
            n_oscillators=n_oscillators,
            n_discrete_states=1,
            n_sources=n_sources,
            sampling_freq=sampling_freq,
        )
        self.dt = 1.0 / sampling_freq
        self.hidden_dims = hidden_dims or [32, 32]
        self.key = jax.random.PRNGKey(seed)
        k_mlp, k_obs, k_init = jax.random.split(self.key, 3)

        # Latent dynamics (the Hamiltonian)
        self.mlp_params = init_mlp_params(self.n_oscillators, self.hidden_dims, k_mlp)
        self.omega = 1.0

        # Observation: log-linear Poisson intensity
        self.C = jax.random.normal(k_obs, (n_sources, self.n_cont_states)) * 0.1
        self.d = jnp.zeros((n_sources,))

        self._initialize_parameters(k_init)
        self._sgd_n_time = 0

    def _initialize_parameters(self, key: Array) -> None:
        self.init_discrete_state_prob = jnp.ones((1,))
        self.discrete_transition_matrix = jnp.eye(1)
        m0 = default_init_mean(self.n_oscillators)
        self.init_mean = jnp.stack([m0], axis=1)
        self.init_cov = jnp.stack([jnp.eye(self.n_cont_states) * 0.1], axis=2)

        self.measurement_matrix = (
            jnp.zeros((self.n_sources, self.n_cont_states, 1)).at[:, :, 0].set(self.C)
        )
        self.process_cov = jnp.stack([jnp.eye(self.n_cont_states) * 1e-4], axis=2)
        self.continuous_transition_matrix = jnp.stack(
            [jnp.eye(self.n_cont_states)], axis=2
        )

    def transition_func(self, x: Array, params: Dict[str, Array]) -> Array:
        """Deterministic Hamiltonian transition."""
        return leapfrog_step(x, params, apply_mlp, self.dt)

    def filter(
        self, spikes: Array, params: Dict[str, Any]
    ) -> Tuple[Array, Array, Array]:
        """Apply Point-Process EKF (Laplace-EKF) to spikes."""
        spikes = self._validate_spikes(spikes)
        return cast(
            Tuple[Array, Array, Array],
            self._filter_jit(spikes, self._complete_filter_params(params)),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _filter_jit(
        self, spikes: Array, params: Dict[str, Any]
    ) -> Tuple[Array, Array, Array]:
        """JIT-compiled filter core with all mutable inputs passed explicitly."""
        trans_params = {**params["mlp"], "omega": params["omega"]}
        C, d = params["C"], params["d"]
        Q = params["Q"]

        def step(carry, y_t):
            m_prev, P_prev = carry
            m_pred, P_pred = ekf_predict_step(
                m_prev, P_prev, trans_params, apply_mlp, Q, self.dt
            )
            m_post, P_post, ll = point_process_laplace_update(
                m_pred,
                P_pred,
                y_t,
                C,
                d,
                self.dt,
            )
            return (m_post, P_post), (m_post, P_post, ll)

        m0 = params["init_mean"]
        P0 = params["init_cov"]
        _, (means, covs, lls) = jax.lax.scan(step, (m0, P0), spikes)
        return means, covs, lls

    def smooth(self, spikes: Array, params: Dict[str, Any]) -> Tuple[Array, Array]:
        """Apply Point-Process RTS Smoother to spikes."""
        spikes = self._validate_spikes(spikes)
        return cast(
            Tuple[Array, Array],
            self._smooth_jit(spikes, self._complete_filter_params(params)),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _smooth_jit(self, spikes: Array, params: Dict[str, Any]) -> Tuple[Array, Array]:
        """JIT-compiled smoother core with mutable inputs passed explicitly."""
        trans_params = {**params["mlp"], "omega": params["omega"]}
        C, d = params["C"], params["d"]
        Q = params["Q"]

        def forward_step(carry, y_t):
            m_prev, P_prev = carry
            m_pred, P_pred, F_t = ekf_predict_step_with_jacobian(
                m_prev, P_prev, trans_params, apply_mlp, Q, self.dt
            )
            m_post, P_post, _ = point_process_laplace_update(
                m_pred,
                P_pred,
                y_t,
                C,
                d,
                self.dt,
                compute_log_likelihood=False,
            )
            return (m_post, P_post), (m_post, P_post, m_pred, P_pred, F_t)

        m0 = params["init_mean"]
        P0 = params["init_cov"]
        _, (m_f, P_f, m_p, P_p, F) = jax.lax.scan(forward_step, (m0, P0), spikes)
        return ekf_rts_backward_pass(m_f, P_f, m_p, P_p, F)

    def _validate_spikes(self, spikes: Array, *, allow_empty: bool = True) -> Array:
        """Validate public spike input and return it as a JAX array."""
        spikes = jnp.asarray(spikes)
        if spikes.ndim != 2:
            raise ValueError(
                f"spikes must have shape (n_time, n_sources); got {spikes.shape}."
            )
        if spikes.shape[1] != self.n_sources:
            raise ValueError(
                f"spikes must have {self.n_sources} sources, got {spikes.shape[1]}."
            )
        validate_count_array(spikes, "spikes", allow_empty=allow_empty)
        return spikes

    def _complete_filter_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fill covariance defaults before entering a JIT-compiled method.

        ``self`` is static in the compiled cores, so reading mutable covariance
        attributes there would capture stale values in JAX's compilation cache.
        """
        complete = dict(params)
        complete.setdefault("Q", self.process_cov[:, :, 0])
        complete.setdefault("init_cov", self.init_cov[:, :, 0])
        return complete

    def fit_sgd(  # type: ignore[override]
        self,
        observations: Array,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
    ) -> list[float]:
        """Fit the model by gradient descent.

        ``use_filter=True`` optimizes the marginal Laplace-EKF likelihood and
        learns the process covariance. ``use_filter=False`` is a deterministic
        rollout surrogate intended for warm starts; it does not depend on or
        update the process covariance.
        """
        observations = self._validate_spikes(observations, allow_empty=False)
        self._sgd_n_time = observations.shape[0]
        return SGDFittableMixin.fit_sgd(
            self,
            observations,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
            use_filter=use_filter,
            l2_reg=l2_reg,
        )

    def _build_param_spec(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, ParameterTransform]]:
        params = {
            "mlp": self.mlp_params,
            "omega": self.omega,
            "C": self.C,
            "d": self.d,
            "init_mean": self.init_mean[:, 0],
            "init_cov": self.init_cov[:, :, 0],
            "Q": self.process_cov[:, :, 0],
        }
        spec = {
            "mlp": UNCONSTRAINED,
            "omega": POSITIVE,
            "C": UNCONSTRAINED,
            "d": UNCONSTRAINED,
            "init_mean": UNCONSTRAINED,
            # Initial covariance is part of the dynamic filter inputs so JIT
            # cannot cache a stale self.init_cov, but it is not learned here.
            "init_cov": frozen(PSD_MATRIX),
            "Q": PSD_MATRIX,
        }
        return params, spec

    def _sgd_loss_fn(
        self,
        params: Dict[str, Any],
        spikes: Array,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
        **kwargs,
    ) -> Array:
        if use_filter:
            _, _, lls = self._filter_jit(spikes, params)
            lik_loss = -jnp.sum(lls)
        else:
            trans_params = {**params["mlp"], "omega": params["omega"]}
            C, d = params["C"], params["d"]
            x0 = params["init_mean"]

            def scan_fn(x_prev, _):
                x_next = self.transition_func(x_prev, trans_params)
                return x_next, x_next

            _, x_traj = jax.lax.scan(scan_fn, x0, jnp.arange(spikes.shape[0]))
            log_lambda = x_traj @ C.T + d
            # Overflow-safe, gradient-preserving Poisson NLL. An unclipped exp
            # overflows to +inf on a divergent rollout (0*log(inf) / (inf-inf) =
            # NaN); a hard clip would zero the gradient above the cap and freeze
            # SGD. _soft_expected_count_and_log continues exp logarithmically past
            # the cap and returns log(mu) analytically under that same cap.
            # log_rates equals log(rates) wherever rates is representable and
            # positive, and stays finite (not -inf) where rates underflows to 0 --
            # so a positive spike count keeps a finite restoring gradient there
            # that log(rates + eps) would kill.
            rates, log_rates = _soft_expected_count_and_log(log_lambda, self.dt)
            lik_loss = jnp.sum(
                rates - spikes * log_rates + jax.scipy.special.gammaln(spikes + 1.0)
            )

        return lik_loss + l2_reg * mlp_l2_penalty(params["mlp"])

    def fit(self, *args, **kwargs):
        """Hamiltonian models do not support linear EM."""
        raise NotImplementedError(
            "HamiltonianSpikeModel does not support the linear EM path "
            "(fit()). Please use fit_sgd() for non-linear optimization."
        )

    def _store_sgd_params(self, params: Dict[str, Any]) -> None:
        self.mlp_params = params["mlp"]
        self.omega = params["omega"]
        self.C = params["C"]
        self.d = params["d"]
        self.measurement_matrix = (
            jnp.zeros((self.n_sources, self.n_cont_states, 1)).at[:, :, 0].set(self.C)
        )
        self.init_mean = self.init_mean.at[:, 0].set(params["init_mean"])
        if "Q" in params:
            self.process_cov = jnp.stack([stabilize_covariance(params["Q"])], axis=2)

    def _finalize_sgd(self, spikes, **kwargs):
        """Run filter + smoother to populate fitted states after SGD."""
        params = self._build_param_spec()[0]
        means, covs, lls = self.filter(spikes, params)
        self.filtered_means_ = means
        self.filtered_covs_ = covs
        self.log_likelihood_ = float(jnp.sum(lls))
        sm_means, sm_covs = self.smooth(spikes, params)
        self.smoothed_means_ = sm_means
        self.smoothed_covs_ = sm_covs

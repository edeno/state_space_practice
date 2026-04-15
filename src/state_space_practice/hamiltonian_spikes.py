"""Hamiltonian Spike Model (Point-Process Observation).

Uses Hamiltonian dynamics with a Poisson/Point-Process readout for spike data.
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.kalman import psd_solve, symmetrize
from state_space_practice.point_process_kalman import _logdet_psd
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_predict_step_with_jacobian,
    ekf_smooth_step,
    get_transition_jacobian,
    init_mlp_params,
    leapfrog_step,
)
from state_space_practice.oscillator_models import BaseModel
from state_space_practice.parameter_transforms import PSD_MATRIX, POSITIVE, UNCONSTRAINED, ParameterTransform
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import stabilize_covariance

class HamiltonianSpikeModel(BaseModel, SGDFittableMixin):
    """Spike Model with Hamiltonian dynamics and Point-Process observations."""
    
    def __init__(
        self,
        n_oscillators: int,
        n_sources: int, # n_neurons
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

        # 1. Latent Parameters (The Hamiltonian)
        self.mlp_params = init_mlp_params(self.n_cont_states, self.hidden_dims, k_mlp)
        self.omega = 1.0 # Initial frequency guess (1 rad/s)

        # 2. Observation Parameters (Log-linear Intensity)
        self.C = jax.random.normal(k_obs, (n_sources, self.n_cont_states)) * 0.1
        self.d = jnp.zeros((n_sources,))

        # BaseModel compliance
        self._initialize_parameters(k_init)
        self._sgd_n_time = 0

    def _initialize_parameters(self, key: Array) -> None:
        self.init_discrete_state_prob = jnp.ones((1,))
        self.discrete_transition_matrix = jnp.eye(1)
        m0 = jnp.concatenate([jnp.full((self.n_oscillators,), 0.1), jnp.zeros((self.n_oscillators,))])
        self.init_mean = jnp.stack([m0], axis=1)
        self.init_cov = jnp.stack([jnp.eye(self.n_cont_states) * 0.1], axis=2)
        
        self.measurement_matrix = jnp.zeros((self.n_sources, self.n_cont_states, 1)).at[:, :, 0].set(self.C)
        self.process_cov = jnp.stack([jnp.eye(self.n_cont_states) * 1e-4], axis=2)
        self.continuous_transition_matrix = jnp.stack([jnp.eye(self.n_cont_states)], axis=2)

    def transition_func(self, x: Array, params: Dict[str, Array]) -> Array:
        """Deterministic Hamiltonian transition."""
        return leapfrog_step(x, params, apply_mlp, self.dt)

    @partial(jax.jit, static_argnums=(0,))
    def filter(self, spikes: Array, params: Dict[str, Any]) -> Tuple[Array, Array, Array]:
        """Apply Point-Process EKF (Laplace-EKF) to spikes.

        Returns
        -------
        means : (T, n_cont)
        covs : (T, n_cont, n_cont)
        log_likelihoods : (T,)
        """
        mlp_params = params["mlp"]
        omega = params["omega"]
        current_trans_params = {**mlp_params, "omega": omega}
        
        C = params["C"]
        d = params["d"]
        Q = params.get("Q", self.process_cov[:, :, 0])

        def step(carry, y_t):
            m_prev, P_prev = carry

            # 1. Predict (Nonlinear)
            m_pred, P_pred = ekf_predict_step(
                m_prev, P_prev, current_trans_params, apply_mlp, Q, self.dt
            )

            # 2. Update (Laplace Approximation for Point-Process)
            def log_lik_fn(x):
                rate = jnp.exp(jnp.dot(C, x) + d) * self.dt
                return jnp.sum(y_t * jnp.log(rate + 1e-10) - rate)

            # Single Newton step (Laplace approximation)
            rate_pred = jnp.exp(jnp.dot(C, m_pred) + d) * self.dt
            grad = jnp.dot(C.T, y_t - rate_pred)
            H_lik = jnp.dot(C.T, (rate_pred[:, None] * C))

            n = P_pred.shape[0]
            I_n = jnp.eye(n)
            P_post = symmetrize(P_pred - P_pred @ psd_solve(I_n + H_lik @ P_pred, H_lik @ P_pred))

            m_post = m_pred + P_post @ grad

            # Laplace-approximated marginal: log p(y_t | y_{1:t-1})
            ll_at_mode = log_lik_fn(m_post)
            delta = m_post - m_pred
            quad = delta @ psd_solve(P_pred, delta)
            ll = (ll_at_mode
                  - 0.5 * quad
                  - 0.5 * _logdet_psd(P_pred)
                  + 0.5 * _logdet_psd(P_post))

            return (m_post, P_post), (m_post, P_post, ll)

        m0 = params["init_mean"]
        P0 = self.init_cov[:, :, 0]

        _, (means, covs, lls) = jax.lax.scan(step, (m0, P0), spikes)
        return means, covs, lls

    @partial(jax.jit, static_argnums=(0,))
    def smooth(self, spikes: Array, params: Dict[str, Any]) -> Tuple[Array, Array]:
        """Apply Point-Process RTS Smoother to spikes."""
        mlp_params = params["mlp"]
        omega = params["omega"]
        current_trans_params = {**mlp_params, "omega": omega}
        C, d = params["C"], params["d"]
        Q = params.get("Q", self.process_cov[:, :, 0])

        def forward_step(carry, y_t):
            m_prev, P_prev = carry
            m_pred, P_pred, F_t = ekf_predict_step_with_jacobian(
                m_prev, P_prev, current_trans_params, apply_mlp, Q, self.dt
            )
            
            rate_pred = jnp.exp(jnp.dot(C, m_pred) + d) * self.dt
            grad = jnp.dot(C.T, y_t - rate_pred)
            H_lik = jnp.dot(C.T, (rate_pred[:, None] * C))
            n = P_pred.shape[0]
            I_n = jnp.eye(n)
            P_post = symmetrize(P_pred - P_pred @ psd_solve(I_n + H_lik @ P_pred, H_lik @ P_pred))
            m_post = m_pred + P_post @ grad
            
            return (m_post, P_post), (m_post, P_post, m_pred, P_pred, F_t)

        m0 = params["init_mean"]
        P0 = self.init_cov[:, :, 0]
        _, (m_f, P_f, m_p, P_p, F) = jax.lax.scan(forward_step, (m0, P0), spikes)

        def backward_step(carry, inputs):
            m_s_next, P_s_next = carry
            m_f_t, P_f_t, m_p_next, P_p_next, F_next = inputs
            m_s, P_s = ekf_smooth_step(m_f_t, P_f_t, m_p_next, P_p_next, m_s_next, P_s_next, F_next)
            return (m_s, P_s), (m_s, P_s)

        init_smooth = (m_f[-1], P_f[-1])
        bw_inputs = (m_f[:-1], P_f[:-1], m_p[1:], P_p[1:], F[1:])
        _, (m_s_rev, P_s_rev) = jax.lax.scan(backward_step, init_smooth, bw_inputs, reverse=True)
        m_s = jnp.concatenate([m_s_rev, m_f[-1:]], axis=0)
        P_s = jnp.concatenate([P_s_rev, P_f[-1:]], axis=0)
        return m_s, P_s

    def fit_sgd(
        self,
        observations: Array,
        key: Array | None = None,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        use_filter: bool = False,
        **kwargs,
    ) -> list[float]:
        """Override fit_sgd to handle the use_filter flag."""
        self._sgd_n_time = observations.shape[0]
        return SGDFittableMixin.fit_sgd(
            self,
            observations,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
            use_filter=use_filter,
            **kwargs,
        )

    def _build_param_spec(self) -> Tuple[Dict[str, Any], Dict[str, ParameterTransform]]:
        params = {
            "mlp": self.mlp_params,
            "omega": self.omega,
            "C": self.C,
            "d": self.d,
            "init_mean": self.init_mean[:, 0],
            "Q": self.process_cov[:, :, 0],
        }
        spec = {
            "mlp": UNCONSTRAINED,
            "omega": POSITIVE,
            "C": UNCONSTRAINED,
            "d": UNCONSTRAINED,
            "init_mean": UNCONSTRAINED,
            "Q": PSD_MATRIX,
        }
        return params, spec

    def _sgd_loss_fn(
        self, params: Dict[str, Any], spikes: Array, use_filter: bool = False, l2_reg: float = 1e-4, **kwargs
    ) -> Array:
        """Negative log-likelihood loss with L2 regularization."""
        mlp_params = params["mlp"]
        omega = params["omega"]
        current_trans_params = {**mlp_params, "omega": omega}
        
        # 1. Prediction Loss
        if use_filter:
            _, _, lls = self.filter(spikes, params)
            lik_loss = -jnp.sum(lls)
        else:
            C, d = params["C"], params["d"]
            x0 = params["init_mean"]

            def scan_fn(x_prev, _):
                x_next = self.transition_func(x_prev, current_trans_params)
                return x_next, x_next

            _, x_traj = jax.lax.scan(scan_fn, x0, jnp.arange(spikes.shape[0]))
            log_lambda = jnp.dot(x_traj, C.T) + d
            rates = jnp.exp(log_lambda) * self.dt
            lik_loss = -jnp.sum(spikes * jnp.log(rates + 1e-10) - rates)
            
        # 2. L2 Regularization
        l2_penalty = 0.0
        for k, v in mlp_params.items():
            if k.startswith("w"):
                l2_penalty += jnp.sum(v**2)
                
        return lik_loss + l2_reg * l2_penalty

    def _store_sgd_params(self, params: Dict[str, Any]) -> None:
        self.mlp_params = params["mlp"]
        self.omega = params["omega"]
        self.C = params["C"]
        self.d = params["d"]
        self.init_mean = self.init_mean.at[:, 0].set(params["init_mean"])
        if "Q" in params:
            self.process_cov = jnp.stack(
                [stabilize_covariance(params["Q"])], axis=2
            )

    # Required Abstract Stubs
    def _initialize_measurement_matrix(self, key=None): pass
    def _initialize_measurement_covariance(self): pass
    def _initialize_continuous_transition_matrix(self): pass
    def _initialize_process_covariance(self): pass
    def _project_parameters(self): pass
    def _check_sgd_initialized(self): pass
    def _finalize_sgd(self, *args, **kwargs): pass

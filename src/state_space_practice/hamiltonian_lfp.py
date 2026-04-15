"""Hamiltonian LFP Model (Gaussian Observation).

Uses Hamiltonian dynamics with a linear-Gaussian readout for voltage data.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional, List
from functools import partial
from jax import Array

from state_space_practice.kalman import psd_solve, joseph_form_update
from state_space_practice.utils import stabilize_covariance
from state_space_practice.oscillator_models import BaseModel
from state_space_practice.parameter_transforms import PSD_MATRIX, POSITIVE, UNCONSTRAINED, ParameterTransform
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.nonlinear_dynamics import (
    leapfrog_step, apply_mlp, init_mlp_params, ekf_predict_step,
    ekf_predict_step_with_jacobian,
    get_transition_jacobian, ekf_smooth_step
)

class HamiltonianLFPModel(BaseModel, SGDFittableMixin):
    """LFP Model with Hamiltonian dynamics and Gaussian noise."""
    
    def __init__(
        self,
        n_oscillators: int,
        n_sources: int,
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
        self.omega = 1.0

        # 2. Observation Parameters (Linear Projection)
        self.C = jax.random.normal(k_obs, (n_sources, self.n_cont_states)) * 0.1
        self.d = jnp.zeros((n_sources,))
        self.obs_noise_std = 0.1

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
        self.measurement_cov = jnp.zeros((self.n_sources, self.n_sources, 1)).at[:, :, 0].set(jnp.eye(self.n_sources) * self.obs_noise_std**2)
        self.process_cov = jnp.stack([jnp.eye(self.n_cont_states) * 1e-4], axis=2)
        self.continuous_transition_matrix = jnp.stack([jnp.eye(self.n_cont_states)], axis=2)

    def transition_func(self, x: Array, params: Dict[str, Array]) -> Array:
        """Deterministic Hamiltonian transition."""
        return leapfrog_step(x, params, apply_mlp, self.dt)

    @partial(jax.jit, static_argnums=(0,))
    def filter(self, lfp_data: Array, params: Dict[str, Any]) -> Tuple[Array, Array, Array]:
        """Apply EKF filter to LFP data."""
        mlp_params = params["mlp"]
        omega = params["omega"]
        current_trans_params = {**mlp_params, "omega": omega}
        C, d = params["C"], params["d"]
        R = jnp.eye(self.n_sources) * self.obs_noise_std**2
        Q = params.get("Q", self.process_cov[:, :, 0])

        def step(carry, y_t):
            m_prev, P_prev = carry
            m_pred, P_pred = ekf_predict_step(m_prev, P_prev, current_trans_params, apply_mlp, Q, self.dt)
            S = C @ P_pred @ C.T + R
            K = psd_solve(S, C @ P_pred).T
            err = y_t - (C @ m_pred + d)
            m_post = m_pred + K @ err
            P_post = joseph_form_update(P_pred, K, C, R)
            sign, logdet = jnp.linalg.slogdet(S)
            ll = -0.5 * (err @ psd_solve(S, err) + logdet + self.n_sources * jnp.log(2 * jnp.pi))
            return (m_post, P_post), (m_post, P_post, ll)

        m0 = params["init_mean"]
        P0 = self.init_cov[:, :, 0]
        _, (means, covs, lls) = jax.lax.scan(step, (m0, P0), lfp_data)
        return means, covs, lls

    @partial(jax.jit, static_argnums=(0,))
    def smooth(self, lfp_data: Array, params: Dict[str, Any]) -> Tuple[Array, Array]:
        """Apply EKF-RTS Smoother to LFP data."""
        mlp_params = params["mlp"]
        omega = params["omega"]
        current_trans_params = {**mlp_params, "omega": omega}
        C, d = params["C"], params["d"]
        R = jnp.eye(self.n_sources) * self.obs_noise_std**2
        Q = params.get("Q", self.process_cov[:, :, 0])

        def forward_step(carry, y_t):
            m_prev, P_prev = carry
            m_pred, P_pred, F_t = ekf_predict_step_with_jacobian(
                m_prev, P_prev, current_trans_params, apply_mlp, Q, self.dt
            )
            S = C @ P_pred @ C.T + R
            K = psd_solve(S, C @ P_pred).T
            m_post = m_pred + K @ (y_t - (C @ m_pred + d))
            P_post = joseph_form_update(P_pred, K, C, R)
            return (m_post, P_post), (m_post, P_post, m_pred, P_pred, F_t)

        m0 = params["init_mean"]
        P0 = self.init_cov[:, :, 0]
        _, (m_f, P_f, m_p, P_p, F) = jax.lax.scan(forward_step, (m0, P0), lfp_data)

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

    def _build_param_spec(self) -> Tuple[Dict[str, Any], Dict[str, ParameterTransform]]:
        params = {
            "mlp": self.mlp_params, "omega": self.omega,
            "C": self.C, "d": self.d,
            "init_mean": self.init_mean[:, 0],
            "Q": self.process_cov[:, :, 0],
        }
        spec = {
            "mlp": UNCONSTRAINED, "omega": POSITIVE,
            "C": UNCONSTRAINED, "d": UNCONSTRAINED,
            "init_mean": UNCONSTRAINED,
            "Q": PSD_MATRIX,
        }
        return params, spec

    def fit_sgd(
        self,
        observations: Array,
        key: Array | None = None,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        use_filter: bool = True,
        **kwargs,
    ) -> list[float]:
        self._sgd_n_time = observations.shape[0]
        return SGDFittableMixin.fit_sgd(self, observations, optimizer=optimizer, num_steps=num_steps, verbose=verbose, convergence_tol=convergence_tol, use_filter=use_filter, **kwargs)

    def _sgd_loss_fn(
        self, params: Dict[str, Any], lfp_data: Array, use_filter: bool = True, l2_reg: float = 1e-4, **kwargs
    ) -> Array:
        mlp_params = params["mlp"]
        omega = params["omega"]
        current_trans_params = {**mlp_params, "omega": omega}
        
        if use_filter:
            _, _, lls = self.filter(lfp_data, params)
            lik_loss = -jnp.sum(lls)
        else:
            C, d = params["C"], params["d"]
            x0 = params["init_mean"]
            def scan_fn(x_prev, _):
                x_next = self.transition_func(x_prev, current_trans_params)
                return x_next, x_next
            _, x_traj = jax.lax.scan(scan_fn, x0, jnp.arange(lfp_data.shape[0]))
            lik_loss = jnp.sum((lfp_data - (jnp.dot(x_traj, C.T) + d)) ** 2)
            
        l2_penalty = jnp.sum(jnp.array([jnp.sum(v**2) for k, v in mlp_params.items() if k.startswith("w")]))
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

    def _initialize_measurement_matrix(self, key=None): pass
    def _initialize_measurement_covariance(self): pass
    def _initialize_continuous_transition_matrix(self): pass
    def _initialize_process_covariance(self): pass
    def _project_parameters(self): pass
    def _check_sgd_initialized(self): pass
    def _finalize_sgd(self, *args, **kwargs): pass

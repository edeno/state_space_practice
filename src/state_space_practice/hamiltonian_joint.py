"""Joint Hamiltonian Model for LFP and Spikes.

Unifies continuous voltage (LFP) and sparse point-processes (Spikes)
under a single shared Hamiltonian latent trajectory.

See docs/hamiltonian_architecture.md for why this family is standalone
(no linear-Gaussian EM integration, SGD-only fitting).
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.kalman import joseph_form_update, psd_solve, symmetrize
from state_space_practice.point_process_kalman import _logdet_psd
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_predict_step_with_jacobian,
    ekf_smooth_step,
    init_mlp_params,
    leapfrog_step,
)
from state_space_practice.oscillator_models import BaseModel
from state_space_practice.parameter_transforms import PSD_MATRIX, POSITIVE, UNCONSTRAINED, ParameterTransform
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import stabilize_covariance

class JointHamiltonianModel(BaseModel, SGDFittableMixin):
    """Joint Model combining Gaussian LFP and Poisson Spikes."""
    
    def __init__(
        self,
        n_oscillators: int,
        n_lfp_sources: int,
        n_spike_sources: int,
        sampling_freq: float,
        hidden_dims: Optional[List[int]] = None,
        seed: int = 42,
    ):
        super().__init__(
            n_oscillators=n_oscillators,
            n_discrete_states=1,
            n_sources=n_lfp_sources + n_spike_sources, # Total sources
            sampling_freq=sampling_freq,
        )
        self.dt = 1.0 / sampling_freq
        self.n_lfp = n_lfp_sources
        self.n_spikes = n_spike_sources
        self.hidden_dims = hidden_dims or [32, 32]
        self.key = jax.random.PRNGKey(seed)
        k_mlp, k_lfp, k_spk, k_init = jax.random.split(self.key, 4)

        # 1. Shared Latent Dynamics
        self.mlp_params = init_mlp_params(self.n_cont_states, self.hidden_dims, k_mlp)
        self.omega = 1.0

        # 2. LFP Head (Gaussian)
        self.C_lfp = jax.random.normal(k_lfp, (n_lfp_sources, self.n_cont_states)) * 0.1
        self.d_lfp = jnp.zeros((n_lfp_sources,))
        self.obs_noise_std = 0.1

        # 3. Spike Head (Poisson)
        self.C_spikes = jax.random.normal(k_spk, (n_spike_sources, self.n_cont_states)) * 0.1
        self.d_spikes = jnp.zeros((n_spike_sources,))

        # BaseModel compliance
        self._initialize_parameters(k_init)
        self._sgd_n_time = 0

    def _initialize_parameters(self, key: Array) -> None:
        self.init_discrete_state_prob = jnp.ones((1,))
        self.discrete_transition_matrix = jnp.eye(1)
        m0 = jnp.concatenate([jnp.full((self.n_oscillators,), 0.1), jnp.zeros((self.n_oscillators,))])
        self.init_mean = jnp.stack([m0], axis=1)
        self.init_cov = jnp.stack([jnp.eye(self.n_cont_states) * 0.1], axis=2)
        
        # Observation noise covariance for LFP part
        self.measurement_cov = jnp.zeros((self.n_sources, self.n_sources, 1))
        # Initial continuous transition (dummy)
        self.continuous_transition_matrix = jnp.stack([jnp.eye(self.n_cont_states)], axis=2)
        self.process_cov = jnp.stack([jnp.eye(self.n_cont_states) * 1e-4], axis=2)

    def transition_func(self, x: Array, params: Dict[str, Array]) -> Array:
        return leapfrog_step(x, params, apply_mlp, self.dt)

    @partial(jax.jit, static_argnums=(0,))
    def filter(self, lfp_data: Array, spike_data: Array, params: Dict[str, Any]) -> Tuple[Array, Array, Array]:
        """Hybrid EKF: Sequentially update from LFP then Spikes."""
        mlp_params = params["mlp"]
        omega = params["omega"]
        current_trans_params = {**mlp_params, "omega": omega}
        
        C_l = params["C_lfp"]
        d_l = params["d_lfp"]
        C_s = params["C_spikes"]
        d_s = params["d_spikes"]
        
        R_l = jnp.eye(self.n_lfp) * self.obs_noise_std**2
        Q = params.get("Q", self.process_cov[:, :, 0])

        def step(carry, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_prev, P_prev = carry

            # 1. Predict (Nonlinear Hamiltonian)
            m_pred, P_pred = ekf_predict_step(
                m_prev, P_prev, current_trans_params, apply_mlp, Q, self.dt
            )

            # 2. LFP Update (Standard Kalman)
            S_l = C_l @ P_pred @ C_l.T + R_l
            K_l = psd_solve(S_l, C_l @ P_pred).T

            err_l = y_lfp_t - (C_l @ m_pred + d_l)
            m_mid = m_pred + K_l @ err_l
            P_mid = joseph_form_update(P_pred, K_l, C_l, R_l)

            sign, logdet = jnp.linalg.slogdet(S_l)
            n_lfp = err_l.shape[0]
            ll_lfp = -0.5 * (err_l @ psd_solve(S_l, err_l) + logdet
                             + n_lfp * jnp.log(2 * jnp.pi))

            # 3. Spike Update (Laplace Approximation)
            rate_mid = jnp.exp(jnp.dot(C_s, m_mid) + d_s) * self.dt
            grad_s = jnp.dot(C_s.T, y_spike_t - rate_mid)
            H_s = jnp.dot(C_s.T, (rate_mid[:, None] * C_s))

            n = P_mid.shape[0]
            I_n = jnp.eye(n)
            P_post = symmetrize(P_mid - P_mid @ psd_solve(I_n + H_s @ P_mid, H_s @ P_mid))
            m_post = m_mid + P_post @ grad_s

            # Laplace-approximated marginal for spike observation
            rate_post = jnp.exp(jnp.dot(C_s, m_post) + d_s) * self.dt
            ll_spike_at_mode = jnp.sum(y_spike_t * jnp.log(rate_post + 1e-10) - rate_post)
            delta_s = m_post - m_mid
            quad_s = delta_s @ psd_solve(P_mid, delta_s)
            ll_spike = (ll_spike_at_mode
                        - 0.5 * quad_s
                        - 0.5 * _logdet_psd(P_mid)
                        + 0.5 * _logdet_psd(P_post))

            return (m_post, P_post), (m_post, P_post, ll_lfp + ll_spike)

        m0 = params["init_mean"]
        P0 = self.init_cov[:, :, 0]
        
        _, (means, covs, lls) = jax.lax.scan(step, (m0, P0), (lfp_data, spike_data))
        return means, covs, lls

    @partial(jax.jit, static_argnums=(0,))
    def smooth(self, lfp_data: Array, spike_data: Array, params: Dict[str, Any]) -> Tuple[Array, Array]:
        """Apply EKF-RTS Smoother to joint data."""
        mlp_params = params["mlp"]
        omega = params["omega"]
        trans_params = {**mlp_params, "omega": omega}
        
        C_l, d_l = params["C_lfp"], params["d_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]
        R_l = jnp.eye(self.n_lfp) * self.obs_noise_std**2
        Q = params.get("Q", self.process_cov[:, :, 0])

        # 1. Forward Pass (Store predicted and filtered states)
        def forward_step(carry, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_prev, P_prev = carry
            
            # Predict (returns Jacobian to avoid recomputation)
            m_pred, P_pred, F_t = ekf_predict_step_with_jacobian(
                m_prev, P_prev, trans_params, apply_mlp, Q, self.dt
            )
            
            # LFP Update
            S_l = C_l @ P_pred @ C_l.T + R_l
            K_l = psd_solve(S_l, C_l @ P_pred).T
            m_mid = m_pred + K_l @ (y_lfp_t - (C_l @ m_pred + d_l))
            P_mid = joseph_form_update(P_pred, K_l, C_l, R_l)

            # Spike Update
            rate_mid = jnp.exp(jnp.dot(C_s, m_mid) + d_s) * self.dt
            grad_s = jnp.dot(C_s.T, y_spike_t - rate_mid)
            H_s = jnp.dot(C_s.T, (rate_mid[:, None] * C_s))
            n = P_mid.shape[0]
            I_n = jnp.eye(n)
            P_post = symmetrize(P_mid - P_mid @ psd_solve(I_n + H_s @ P_mid, H_s @ P_mid))
            m_post = m_mid + P_post @ grad_s
            
            return (m_post, P_post), (m_post, P_post, m_pred, P_pred, F_t)

        m0 = params["init_mean"]
        P0 = self.init_cov[:, :, 0]
        _, (m_f, P_f, m_p, P_p, F) = jax.lax.scan(forward_step, (m0, P0), (lfp_data, spike_data))

        # 2. Backward Pass (Smoothing)
        def backward_step(carry, inputs):
            m_s_next, P_s_next = carry
            m_f_t, P_f_t, m_p_next, P_p_next, F_next = inputs
            
            m_s, P_s = ekf_smooth_step(m_f_t, P_f_t, m_p_next, P_p_next, m_s_next, P_s_next, F_next)
            return (m_s, P_s), (m_s, P_s)

        # Start from the last filtered state
        init_smooth = (m_f[-1], P_f[-1])
        
        # We need to shift the predicted states and Jacobians for the backward pass
        bw_inputs = (m_f[:-1], P_f[:-1], m_p[1:], P_p[1:], F[1:])
        _, (m_s_rev, P_s_rev) = jax.lax.scan(backward_step, init_smooth, bw_inputs, reverse=True)
        
        # Concatenate the last filtered state (which is already smooth)
        m_s = jnp.concatenate([m_s_rev, m_f[-1:]], axis=0)
        P_s = jnp.concatenate([P_s_rev, P_f[-1:]], axis=0)
        
        return m_s, P_s

    def fit_sgd(
        self,
        lfp_obs: Array,
        spike_obs: Array,
        key: Array | None = None,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        use_filter: bool = True,
        **kwargs,
    ) -> list[float]:
        self._sgd_n_time = lfp_obs.shape[0]
        return SGDFittableMixin.fit_sgd(
            self,
            lfp_obs,
            spike_obs,
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
            "C_lfp": self.C_lfp,
            "d_lfp": self.d_lfp,
            "C_spikes": self.C_spikes,
            "d_spikes": self.d_spikes,
            "init_mean": self.init_mean[:, 0],
            "Q": self.process_cov[:, :, 0],
        }
        spec = {
            "mlp": UNCONSTRAINED, "omega": POSITIVE,
            "C_lfp": UNCONSTRAINED, "d_lfp": UNCONSTRAINED,
            "C_spikes": UNCONSTRAINED, "d_spikes": UNCONSTRAINED,
            "init_mean": UNCONSTRAINED,
            "Q": PSD_MATRIX,
        }
        return params, spec

    def _sgd_loss_fn(
        self, params: Dict[str, Any], lfp_data: Array, spike_data: Array, 
        use_filter: bool = True, l2_reg: float = 1e-4, **kwargs
    ) -> Array:
        # 1. Prediction Loss
        if use_filter:
            _, _, lls = self.filter(lfp_data, spike_data, params)
            lik_loss = -jnp.sum(lls)
        else:
            # Surrogate loss: deterministic rollout (LFP SSE + plug-in
            # Poisson NLL).  This is NOT the joint state-space marginal
            # likelihood — it drops observation-noise scale, process
            # prior, and latent uncertainty.  Useful for warm-starting.
            mlp_params, omega = params["mlp"], params["omega"]
            trans_params = {**mlp_params, "omega": omega}
            C_l, d_l = params["C_lfp"], params["d_lfp"]
            C_s, d_s = params["C_spikes"], params["d_spikes"]
            x0 = params["init_mean"]

            def scan_fn(x_prev, _):
                x_next = self.transition_func(x_prev, trans_params)
                return x_next, x_next

            _, x_traj = jax.lax.scan(scan_fn, x0, jnp.arange(lfp_data.shape[0]))
            
            # LFP MSE
            y_l_pred = jnp.dot(x_traj, C_l.T) + d_l
            mse_l = jnp.sum((lfp_data - y_l_pred)**2)
            
            # Spike NLL
            log_lambda = jnp.dot(x_traj, C_s.T) + d_s
            rates = jnp.exp(log_lambda) * self.dt
            nll_s = -jnp.sum(spike_data * jnp.log(rates + 1e-10) - rates)
            
            lik_loss = mse_l + nll_s
            
        # 2. L2 Regularization
        l2_penalty = 0.0
        for k, v in params["mlp"].items():
            if k.startswith("w"):
                l2_penalty += jnp.sum(v**2)
                
        return lik_loss + l2_reg * l2_penalty

    def fit(self, *args, **kwargs):
        """Hamiltonian models do not support linear EM."""
        raise NotImplementedError(
            "JointHamiltonianModel does not support the linear EM path (fit()). "
            "Please use fit_sgd() for non-linear optimization."
        )

    def _store_sgd_params(self, params: Dict[str, Any]) -> None:
        self.mlp_params = params["mlp"]
        self.omega = params["omega"]
        self.C_lfp = params["C_lfp"]
        self.d_lfp = params["d_lfp"]
        self.C_spikes = params["C_spikes"]
        self.d_spikes = params["d_spikes"]
        self.init_mean = self.init_mean.at[:, 0].set(params["init_mean"])
        if "Q" in params:
            self.process_cov = jnp.stack(
                [stabilize_covariance(params["Q"])], axis=2
            )

        # Resynchronize BaseModel fields
        self.measurement_matrix = jnp.zeros((self.n_sources, self.n_cont_states, 1)).at[:self.n_lfp, :, 0].set(self.C_lfp)
        self.measurement_matrix = self.measurement_matrix.at[self.n_lfp:, :, 0].set(self.C_spikes)

    # BaseModel stubs
    def _initialize_measurement_matrix(self, key=None): pass
    def _initialize_measurement_covariance(self): pass
    def _initialize_continuous_transition_matrix(self): pass
    def _initialize_process_covariance(self): pass
    def _project_parameters(self): pass
    def _check_sgd_initialized(self): pass
    def _finalize_sgd(self, lfp_data, spike_data=None, **kwargs):
        """Run filter + smoother to populate fitted states after SGD."""
        if spike_data is None:
            raise ValueError("spike_data required for _finalize_sgd")
        params = self._build_param_spec()[0]
        means, covs, lls = self.filter(lfp_data, spike_data, params)
        self.filtered_means_ = means
        self.filtered_covs_ = covs
        self.log_likelihood_ = float(jnp.sum(lls))
        sm_means, sm_covs = self.smooth(lfp_data, spike_data, params)
        self.smoothed_means_ = sm_means
        self.smoothed_covs_ = sm_covs

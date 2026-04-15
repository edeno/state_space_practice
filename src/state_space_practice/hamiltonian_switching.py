"""Switching Joint Hamiltonian Model.

Combines multiple Hamiltonian energy landscapes with switching discrete states,
sharing a single multimodal (LFP + Spikes) observation head.
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.hamiltonian_joint import JointHamiltonianModel
from state_space_practice.kalman import joseph_form_update, psd_solve, symmetrize
from state_space_practice.point_process_kalman import _logdet_psd
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_predict_step_with_jacobian,
    ekf_smooth_step,
    get_transition_jacobian,
    init_mlp_params,
)
from state_space_practice.parameter_transforms import (
    STOCHASTIC_ROW,
    UNCONSTRAINED,
    ParameterTransform,
)
from state_space_practice.switching_kalman import collapse_gaussian_mixture
from state_space_practice.utils import divide_safe as _divide_safe
from state_space_practice.utils import scale_likelihood as _scale_likelihood


class SwitchingHamiltonianJointModel(JointHamiltonianModel):
    """Switching Model with multiple Hamiltonian energy landscapes."""
    
    def __init__(
        self,
        n_oscillators: int,
        n_discrete_states: int,
        n_lfp_sources: int,
        n_spike_sources: int,
        sampling_freq: float,
        hidden_dims: Optional[List[int]] = None,
        seed: int = 42,
    ):
        super().__init__(
            n_oscillators=n_oscillators,
            n_lfp_sources=n_lfp_sources,
            n_spike_sources=n_spike_sources,
            sampling_freq=sampling_freq,
            hidden_dims=hidden_dims,
            seed=seed
        )
        self.n_discrete_states = n_discrete_states
        
        # Multiple Hamiltonians
        keys = jax.random.split(self.key, n_discrete_states)
        self.mlp_params = jax.vmap(partial(init_mlp_params, self.n_cont_states, self.hidden_dims))(keys)
        self.omega = jnp.ones((n_discrete_states,))
        
        self.discrete_transition_matrix = jnp.eye(n_discrete_states) * 0.95 + \
                                         jnp.ones((n_discrete_states, n_discrete_states)) * 0.05 / n_discrete_states
        
        self._initialize_parameters(self.key)

    def _initialize_parameters(self, key: Array) -> None:
        self.init_discrete_state_prob = jnp.ones((self.n_discrete_states,)) / self.n_discrete_states
        m0 = jnp.concatenate([jnp.full((self.n_oscillators,), 0.1), jnp.zeros((self.n_oscillators,))])
        self.init_mean = jnp.stack([m0] * self.n_discrete_states, axis=1)
        self.init_cov = jnp.stack([jnp.eye(self.n_cont_states) * 0.1] * self.n_discrete_states, axis=2)
        self.process_cov = jnp.stack([jnp.eye(self.n_cont_states) * 1e-4] * self.n_discrete_states, axis=2)

    @partial(jax.jit, static_argnums=(0,))
    def filter(self, lfp_data: Array, spike_data: Array, params: Dict[str, Any]) -> Tuple[Array, Array, Array, Array]:
        """Switching EKF with Gaussian Collapse (Kim Filter)."""
        mlp_params, omega = params["mlp"], params["omega"]
        Z = params["Z"]
        C_l, d_l = params["C_lfp"], params["d_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]
        R_l = jnp.eye(self.n_lfp) * self.obs_noise_std**2
        Q = self.process_cov
        K_states = self.n_discrete_states

        def step(carry, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_prev, P_prev, pi_prev = carry

            def predict_j_k(mj, Pj, k):
                trans_k = {**jax.tree_util.tree_map(lambda x: x[k], mlp_params), "omega": omega[k]}
                return ekf_predict_step(mj, Pj, trans_k, apply_mlp, Q[:, :, k], self.dt)

            v_predict = jax.vmap(jax.vmap(predict_j_k, in_axes=(None, None, 0)), in_axes=(1, 2, None))
            m_p_jk, P_p_jk = v_predict(m_prev, P_prev, jnp.arange(K_states))

            joint_pi_pred = pi_prev[:, None] * Z
            pi_pred_k = jnp.sum(joint_pi_pred, axis=0)

            def collapse_k(k):
                w_jk = _divide_safe(joint_pi_pred[:, k], pi_pred_k[k])
                return collapse_gaussian_mixture(m_p_jk[:, k, :].T, P_p_jk[:, k, :, :].transpose(1, 2, 0), w_jk)

            m_p_k, P_p_k = jax.vmap(collapse_k)(jnp.arange(K_states))
            m_p_k, P_p_k = m_p_k.T, P_p_k.transpose(1, 2, 0)

            def update_k(k):
                mp, Pp = m_p_k[:, k], P_p_k[:, :, k]
                S_l = C_l @ Pp @ C_l.T + R_l
                K_l = psd_solve(S_l, C_l @ Pp).T
                m_mid = mp + K_l @ (y_lfp_t - (C_l @ mp + d_l))
                P_mid = joseph_form_update(Pp, K_l, C_l, R_l)
                err_l = y_lfp_t - (C_l @ mp + d_l)
                sign, logdet = jnp.linalg.slogdet(S_l)
                n_lfp = err_l.shape[0]
                ll_l = -0.5 * (err_l @ psd_solve(S_l, err_l) + logdet
                               + n_lfp * jnp.log(2 * jnp.pi))

                rate_mid = jnp.exp(jnp.dot(C_s, m_mid) + d_s) * self.dt
                grad_s = jnp.dot(C_s.T, y_spike_t - rate_mid)
                H_s = jnp.dot(C_s.T, (rate_mid[:, None] * C_s))
                n = P_mid.shape[0]
                I_n = jnp.eye(n)
                P_post = symmetrize(P_mid - P_mid @ psd_solve(I_n + H_s @ P_mid, H_s @ P_mid))
                m_post = m_mid + P_post @ grad_s

                # Laplace-approximated marginal for spike observation
                rate_post = jnp.exp(jnp.dot(C_s, m_post) + d_s) * self.dt
                ll_s_mode = jnp.sum(y_spike_t * jnp.log(rate_post + 1e-10) - rate_post)
                delta_s = m_post - m_mid
                quad_s = delta_s @ psd_solve(P_mid, delta_s)
                ll_s = (ll_s_mode
                        - 0.5 * quad_s
                        - 0.5 * _logdet_psd(P_mid)
                        + 0.5 * _logdet_psd(P_post))
                return m_post, P_post, ll_l + ll_s

            m_filt, P_filt, lls_k = jax.vmap(update_k)(jnp.arange(K_states))
            scaled_lik, ll_max = _scale_likelihood(lls_k)
            pi_filt = _divide_safe(scaled_lik * pi_pred_k, jnp.sum(scaled_lik * pi_pred_k))
            
            return (m_filt.T, P_filt.transpose(1, 2, 0), pi_filt), (m_filt.T, P_filt.transpose(1, 2, 0), pi_filt, ll_max + jnp.log(jnp.sum(scaled_lik * pi_pred_k)))

        m0 = params["init_mean"]
        P0 = self.init_cov
        pi0 = params["init_pi"]
        _, (means, covs, probs, marginal_lls) = jax.lax.scan(step, (m0, P0, pi0), (lfp_data, spike_data))
        return means, covs, probs, marginal_lls

    @partial(jax.jit, static_argnums=(0,))
    def smooth(self, lfp_data: Array, spike_data: Array, params: Dict[str, Any]) -> Tuple[Array, Array, Array]:
        """Switching EKF-RTS smoother with Kim-style discrete-state smoothing.

        Returns smoothed continuous means and covariances per discrete state,
        plus smoothed discrete-state probabilities.

        Returns
        -------
        smoothed_means : Array, shape (n_time, n_latent, n_discrete_states)
        smoothed_covs : Array, shape (n_time, n_latent, n_latent, n_discrete_states)
        smoothed_probs : Array, shape (n_time, n_discrete_states)
        """
        mlp_params, omega = params["mlp"], params["omega"]
        Z = params["Z"]
        C_l, d_l = params["C_lfp"], params["d_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]
        R_l = jnp.eye(self.n_lfp) * self.obs_noise_std**2
        Q = self.process_cov
        K_states = self.n_discrete_states

        def forward_step(carry, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_prev, P_prev, pi_prev = carry
            
            def predict_j_k(mj, Pj, k):
                trans_k = {**jax.tree_util.tree_map(lambda x: x[k], mlp_params), "omega": omega[k]}
                m_pred, P_pred, F_jk = ekf_predict_step_with_jacobian(
                    mj, Pj, trans_k, apply_mlp, Q[:, :, k], self.dt
                )
                return m_pred, P_pred, F_jk

            v_predict = jax.vmap(jax.vmap(predict_j_k, in_axes=(None, None, 0)), in_axes=(1, 2, None))
            m_p_jk, P_p_jk, F_jk = v_predict(m_prev, P_prev, jnp.arange(K_states))
            
            joint_pi_pred = pi_prev[:, None] * Z
            pi_pred_k = jnp.sum(joint_pi_pred, axis=0)
            
            def collapse_k(k):
                w_jk = _divide_safe(joint_pi_pred[:, k], pi_pred_k[k])
                return collapse_gaussian_mixture(m_p_jk[:, k, :].T, P_p_jk[:, k, :, :].transpose(1, 2, 0), w_jk)

            m_p_k, P_p_k = jax.vmap(collapse_k)(jnp.arange(K_states))
            m_p_k, P_p_k = m_p_k.T, P_p_k.transpose(1, 2, 0)

            def update_k(k):
                mp, Pp = m_p_k[:, k], P_p_k[:, :, k]
                S_l = C_l @ Pp @ C_l.T + R_l
                K_l = psd_solve(S_l, C_l @ Pp).T
                m_mid = mp + K_l @ (y_lfp_t - (C_l @ mp + d_l))
                P_mid = joseph_form_update(Pp, K_l, C_l, R_l)
                err_l = y_lfp_t - (C_l @ mp + d_l)
                sign, logdet = jnp.linalg.slogdet(S_l)
                ll_l = -0.5 * (err_l @ psd_solve(S_l, err_l) + logdet)
                rate_mid = jnp.exp(jnp.dot(C_s, m_mid) + d_s) * self.dt
                grad_s = jnp.dot(C_s.T, y_spike_t - rate_mid)
                H_s = jnp.dot(C_s.T, (rate_mid[:, None] * C_s))
                n = P_mid.shape[0]
                I_n = jnp.eye(n)
                P_post = symmetrize(P_mid - P_mid @ psd_solve(I_n + H_s @ P_mid, H_s @ P_mid))
                m_post = m_mid + P_post @ grad_s
                rate_post = jnp.exp(jnp.dot(C_s, m_post) + d_s) * self.dt
                ll_s = jnp.sum(y_spike_t * jnp.log(rate_post + 1e-10) - rate_post)
                return m_post, P_post, ll_l + ll_s

            m_f, P_f, lls_k = jax.vmap(update_k)(jnp.arange(K_states))
            m_f, P_f = m_f.T, P_f.transpose(1, 2, 0)
            scaled_lik, _ = _scale_likelihood(lls_k)
            pi_filt = _divide_safe(scaled_lik * pi_pred_k, jnp.sum(scaled_lik * pi_pred_k))
            return (m_f, P_f, pi_filt), (m_f, P_f, m_p_k, P_p_k, F_jk, joint_pi_pred, pi_pred_k, pi_filt)

        m0 = params["init_mean"]
        P0 = self.init_cov
        pi0 = params["init_pi"]
        _, (m_filt, P_filt, m_pred, P_pred, F_all, joint_pi_all, pi_pred_all, pi_filt_all) = jax.lax.scan(forward_step, (m0, P0, pi0), (lfp_data, spike_data))

        def backward_step(carry, inputs):
            m_s_next, P_s_next, pi_s_next = carry
            m_f_t, P_f_t, m_p_next, P_p_next, F_next_jk, joint_pi_next, pi_pred_next_k, pi_filt_t = inputs

            # Smooth discrete states
            # pi_smooth[t,k] = pi_filt[t,k] * sum_j(Z[k,j] * pi_smooth[t+1,j] / pi_pred[t+1,j])
            ratio = _divide_safe(pi_s_next, pi_pred_next_k)
            pi_s_t = pi_filt_t * (Z @ ratio)
            pi_s_t = _divide_safe(pi_s_t, jnp.sum(pi_s_t))  # normalize

            def smooth_k(k):
                w_jk = _divide_safe(joint_pi_next[:, k], pi_pred_next_k[k])
                F_avg = jnp.sum(w_jk[:, None, None] * F_next_jk[:, k, :, :], axis=0)
                return ekf_smooth_step(m_f_t[:, k], P_f_t[:, :, k], m_p_next[:, k], P_p_next[:, :, k], m_s_next[:, k], P_s_next[:, :, k], F_avg)
            m_s, P_s = jax.vmap(smooth_k)(jnp.arange(K_states))
            return (m_s.T, P_s.transpose(1, 2, 0), pi_s_t), (m_s.T, P_s.transpose(1, 2, 0), pi_s_t)

        # Use last filtered probs as initial smoothed probs
        init_s = (m_filt[-1], P_filt[-1], pi_filt_all[-1])
        bw_in = (m_filt[:-1], P_filt[:-1], m_pred[1:], P_pred[1:], F_all[1:], joint_pi_all[1:], pi_pred_all[1:], pi_filt_all[:-1])
        _, (m_smooth_rev, P_smooth_rev, pi_smooth_rev) = jax.lax.scan(backward_step, init_s, bw_in, reverse=True)
        m_s = jnp.concatenate([m_smooth_rev, m_filt[-1:]], axis=0)
        P_s = jnp.concatenate([P_smooth_rev, P_filt[-1:]], axis=0)
        pi_s = jnp.concatenate([pi_smooth_rev, pi_filt_all[-1:]], axis=0)
        return m_s, P_s, pi_s

    def _build_param_spec(self) -> Tuple[Dict[str, Any], Dict[str, ParameterTransform]]:
        params, spec = super()._build_param_spec()
        # Remove inherited single-matrix Q: the switching model uses per-state
        # process_cov (shape n x n x K) read directly from self.process_cov
        # in filter/smoother.  Exposing a single Q for optimization would be
        # structurally inconsistent with the per-state dynamics.
        params.pop("Q", None)
        spec.pop("Q", None)
        params["init_mean"] = self.init_mean
        params["Z"] = self.discrete_transition_matrix
        params["init_pi"] = self.init_discrete_state_prob
        spec["init_mean"] = UNCONSTRAINED
        spec["Z"] = STOCHASTIC_ROW
        spec["init_pi"] = STOCHASTIC_ROW
        return params, spec

    def _sgd_loss_fn(self, params: Dict[str, Any], lfp_data: Array, spike_data: Array, **kwargs) -> Array:
        _, _, _, marginal_lls = self.filter(lfp_data, spike_data, params)
        lik_loss = -jnp.sum(marginal_lls)
        l2_penalty = jnp.sum(jnp.array([jnp.sum(v**2) for k, v in params["mlp"].items() if k.startswith("w")]))
        return lik_loss + kwargs.get("l2_reg", 1e-4) * l2_penalty

    def fit(self, *args, **kwargs):
        """Hamiltonian models do not support linear EM."""
        raise NotImplementedError(
            "SwitchingHamiltonianJointModel does not support the linear EM path (fit()). "
            "Please use fit_sgd() for non-linear optimization."
        )

    def _store_sgd_params(self, params: Dict[str, Any]) -> None:
        self.mlp_params = params["mlp"]
        self.omega = params["omega"]
        self.C_lfp = params["C_lfp"]
        self.d_lfp = params["d_lfp"]
        self.C_spikes = params["C_spikes"]
        self.d_spikes = params["d_spikes"]
        self.init_mean = params["init_mean"]
        self.discrete_transition_matrix = params["Z"]
        self.init_discrete_state_prob = params["init_pi"]

        # Resynchronize BaseModel fields
        # Shared measurement matrix across states for now
        self.measurement_matrix = jnp.zeros((self.n_sources, self.n_cont_states, self.n_discrete_states))
        # Set for all discrete states
        for k in range(self.n_discrete_states):
            self.measurement_matrix = self.measurement_matrix.at[:self.n_lfp, :, k].set(self.C_lfp)
            self.measurement_matrix = self.measurement_matrix.at[self.n_lfp:, :, k].set(self.C_spikes)

    def _finalize_sgd(self, lfp_data, spike_data=None, **kwargs):
        """Run filter + smoother to populate fitted states after SGD.

        Overrides the single-regime parent: the switching filter returns
        four arrays (means, covs, discrete_probs, marginal_lls) and the
        smoother returns three (means, covs, discrete_probs).
        """
        if spike_data is None:
            raise ValueError("spike_data required for _finalize_sgd")
        params = self._build_param_spec()[0]
        means, covs, probs, lls = self.filter(lfp_data, spike_data, params)
        self.filtered_means_ = means
        self.filtered_covs_ = covs
        self.filtered_discrete_probs_ = probs
        self.log_likelihood_ = float(jnp.sum(lls))
        sm_means, sm_covs, sm_probs = self.smooth(lfp_data, spike_data, params)
        self.smoothed_means_ = sm_means
        self.smoothed_covs_ = sm_covs
        self.smoothed_discrete_probs_ = sm_probs

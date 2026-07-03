"""Switching Joint Hamiltonian Model.

Combines multiple Hamiltonian energy landscapes with switching discrete states,
sharing a single multimodal (LFP + Spikes) observation head.

See docs/hamiltonian_architecture.md for why this family is standalone
(no linear-Gaussian EM integration, SGD-only fitting).
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.hamiltonian_core import (
    gaussian_measurement_update,
    mlp_l2_penalty,
    point_process_laplace_update,
)
from state_space_practice.hamiltonian_joint import JointHamiltonianModel
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_predict_step_with_jacobian,
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
            seed=seed,
        )
        self.n_discrete_states = n_discrete_states

        keys = jax.random.split(self.key, n_discrete_states)
        self.mlp_params = jax.vmap(
            partial(init_mlp_params, self.n_cont_states, self.hidden_dims)
        )(keys)
        self.omega = jnp.ones((n_discrete_states,))

        self.discrete_transition_matrix = (
            jnp.eye(n_discrete_states) * 0.95
            + jnp.ones((n_discrete_states, n_discrete_states))
            * 0.05
            / n_discrete_states
        )

        self._initialize_parameters(self.key)

    def _initialize_parameters(self, key: Array) -> None:
        from state_space_practice.hamiltonian_core import default_init_mean

        self.init_discrete_state_prob = (
            jnp.ones((self.n_discrete_states,)) / self.n_discrete_states
        )
        m0 = default_init_mean(self.n_oscillators)
        self.init_mean = jnp.stack([m0] * self.n_discrete_states, axis=1)
        self.init_cov = jnp.stack(
            [jnp.eye(self.n_cont_states) * 0.1] * self.n_discrete_states, axis=2
        )
        self.process_cov = jnp.stack(
            [jnp.eye(self.n_cont_states) * 1e-4] * self.n_discrete_states, axis=2
        )

    def _per_state_pred_collapse(
        self,
        m_prev: Array,
        P_prev: Array,
        pi_prev: Array,
        Z: Array,
        mlp_params: Dict[str, Any],
        omega: Array,
        Q_all: Array,
        K_states: int,
        with_jacobian: bool,
    ):
        """Predict step + Gaussian-mixture collapse for the switching filter.

        Returns the per-state collapsed predicted mean/cov (and Jacobian
        if requested), the joint discrete prior P(s_{t-1}, s_t), and the
        marginal predicted prior P(s_t).
        """

        def predict_j_k(mj, Pj, k):
            trans_k = {
                **jax.tree_util.tree_map(lambda x: x[k], mlp_params),
                "omega": omega[k],
            }
            if with_jacobian:
                return ekf_predict_step_with_jacobian(
                    mj, Pj, trans_k, apply_mlp, Q_all[:, :, k], self.dt
                )
            return ekf_predict_step(
                mj, Pj, trans_k, apply_mlp, Q_all[:, :, k], self.dt
            )

        v_predict = jax.vmap(
            jax.vmap(predict_j_k, in_axes=(None, None, 0)),
            in_axes=(1, 2, None),
        )
        if with_jacobian:
            m_p_jk, P_p_jk, F_jk = v_predict(m_prev, P_prev, jnp.arange(K_states))
        else:
            m_p_jk, P_p_jk = v_predict(m_prev, P_prev, jnp.arange(K_states))
            F_jk = None

        joint_pi_pred = pi_prev[:, None] * Z
        pi_pred_k = jnp.sum(joint_pi_pred, axis=0)

        def collapse_k(k):
            w_jk = _divide_safe(joint_pi_pred[:, k], pi_pred_k[k])
            return collapse_gaussian_mixture(
                m_p_jk[:, k, :].T,
                P_p_jk[:, k, :, :].transpose(1, 2, 0),
                w_jk,
            )

        m_p_k, P_p_k = jax.vmap(collapse_k)(jnp.arange(K_states))
        m_p_k = m_p_k.T
        P_p_k = P_p_k.transpose(1, 2, 0)
        return m_p_k, P_p_k, F_jk, joint_pi_pred, pi_pred_k

    @partial(jax.jit, static_argnums=(0,))
    def filter(
        self,
        lfp_data: Array,
        spike_data: Array,
        params: Dict[str, Any],
    ) -> Tuple[Array, Array, Array, Array]:
        """Switching EKF with Gaussian Collapse (Kim Filter)."""
        mlp_params, omega = params["mlp"], params["omega"]
        Z = params["Z"]
        C_l, d_l = params["C_lfp"], params["d_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]
        R_l = self._r_lfp()
        Q = self.process_cov
        K_states = self.n_discrete_states

        def step(carry, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_prev, P_prev, pi_prev = carry

            m_p_k, P_p_k, _, joint_pi_pred, pi_pred_k = self._per_state_pred_collapse(
                m_prev, P_prev, pi_prev,
                Z, mlp_params, omega, Q, K_states,
                with_jacobian=False,
            )
            del joint_pi_pred  # not used in filter (only smoother needs it)

            def update_k(k):
                m_mid, P_mid, ll_l = gaussian_measurement_update(
                    m_p_k[:, k], P_p_k[:, :, k], y_lfp_t, C_l, d_l, R_l,
                )
                m_post, P_post, ll_s = point_process_laplace_update(
                    m_mid, P_mid, y_spike_t, C_s, d_s, self.dt,
                )
                return m_post, P_post, ll_l + ll_s

            m_filt, P_filt, lls_k = jax.vmap(update_k)(jnp.arange(K_states))
            scaled_lik, ll_max = _scale_likelihood(lls_k)
            pi_filt = _divide_safe(
                scaled_lik * pi_pred_k, jnp.sum(scaled_lik * pi_pred_k)
            )
            m_filt_T = m_filt.T
            P_filt_T = P_filt.transpose(1, 2, 0)
            marginal_ll = ll_max + jnp.log(jnp.sum(scaled_lik * pi_pred_k))

            return (
                (m_filt_T, P_filt_T, pi_filt),
                (m_filt_T, P_filt_T, pi_filt, marginal_ll),
            )

        m0 = params["init_mean"]
        P0 = self.init_cov
        pi0 = params["init_pi"]
        _, (means, covs, probs, marginal_lls) = jax.lax.scan(
            step, (m0, P0, pi0), (lfp_data, spike_data),
        )
        return means, covs, probs, marginal_lls

    @partial(jax.jit, static_argnums=(0,))
    def smooth(
        self,
        lfp_data: Array,
        spike_data: Array,
        params: Dict[str, Any],
    ) -> Tuple[Array, Array, Array]:
        """Switching EKF-RTS smoother with Kim-style discrete-state smoothing.

        Returns
        -------
        smoothed_means : (n_time, n_latent, n_discrete_states)
        smoothed_covs : (n_time, n_latent, n_latent, n_discrete_states)
        smoothed_probs : (n_time, n_discrete_states)
        """
        mlp_params, omega = params["mlp"], params["omega"]
        Z = params["Z"]
        C_l, d_l = params["C_lfp"], params["d_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]
        R_l = self._r_lfp()
        Q = self.process_cov
        K_states = self.n_discrete_states

        def forward_step(carry, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_prev, P_prev, pi_prev = carry

            (
                m_p_k,
                P_p_k,
                F_jk,
                joint_pi_pred,
                pi_pred_k,
            ) = self._per_state_pred_collapse(
                m_prev, P_prev, pi_prev,
                Z, mlp_params, omega, Q, K_states,
                with_jacobian=True,
            )

            def update_k(k):
                # Per-state gaussian + Laplace update. Log-likelihoods
                # here go into _scale_likelihood for relative discrete-
                # state weighting only, so the Gaussian normalization
                # constant is dropped. The Laplace correction terms remain
                # state-dependent and must be included.
                m_mid, P_mid, ll_l = gaussian_measurement_update(
                    m_p_k[:, k], P_p_k[:, :, k], y_lfp_t, C_l, d_l, R_l,
                    include_normalization_const=False,
                )
                m_post, P_post, ll_s = point_process_laplace_update(
                    m_mid, P_mid, y_spike_t, C_s, d_s, self.dt,
                )
                return m_post, P_post, ll_l + ll_s

            m_f, P_f, lls_k = jax.vmap(update_k)(jnp.arange(K_states))
            m_f = m_f.T
            P_f = P_f.transpose(1, 2, 0)
            scaled_lik, _ = _scale_likelihood(lls_k)
            pi_filt = _divide_safe(
                scaled_lik * pi_pred_k, jnp.sum(scaled_lik * pi_pred_k)
            )
            return (
                (m_f, P_f, pi_filt),
                (m_f, P_f, m_p_k, P_p_k, F_jk, joint_pi_pred, pi_pred_k, pi_filt),
            )

        m0 = params["init_mean"]
        P0 = self.init_cov
        pi0 = params["init_pi"]
        _, (
            m_filt, P_filt, m_pred, P_pred, F_all, joint_pi_all, pi_pred_all, pi_filt_all,
        ) = jax.lax.scan(forward_step, (m0, P0, pi0), (lfp_data, spike_data))

        # Backward pass: per-state EKF smoother + Kim discrete-state smoother.
        # Cannot use ekf_rts_backward_pass directly because each discrete
        # state has its own filtered/predicted trajectory and the
        # transition Jacobian is averaged across (j, k) source states.
        def backward_step(carry, inputs):
            m_s_next, P_s_next, pi_s_next = carry
            (
                m_f_t, P_f_t, m_p_next, P_p_next, F_next_jk,
                joint_pi_next, pi_pred_next_k, pi_filt_t,
            ) = inputs

            ratio = _divide_safe(pi_s_next, pi_pred_next_k)
            pi_s_t = pi_filt_t * (Z @ ratio)
            pi_s_t = _divide_safe(pi_s_t, jnp.sum(pi_s_t))

            def smooth_k(k):
                from state_space_practice.nonlinear_dynamics import ekf_smooth_step

                w_jk = _divide_safe(joint_pi_next[:, k], pi_pred_next_k[k])
                F_avg = jnp.sum(w_jk[:, None, None] * F_next_jk[:, k, :, :], axis=0)
                return ekf_smooth_step(
                    m_f_t[:, k], P_f_t[:, :, k],
                    m_p_next[:, k], P_p_next[:, :, k],
                    m_s_next[:, k], P_s_next[:, :, k],
                    F_avg,
                )

            m_s, P_s = jax.vmap(smooth_k)(jnp.arange(K_states))
            return (
                (m_s.T, P_s.transpose(1, 2, 0), pi_s_t),
                (m_s.T, P_s.transpose(1, 2, 0), pi_s_t),
            )

        init_s = (m_filt[-1], P_filt[-1], pi_filt_all[-1])
        bw_in = (
            m_filt[:-1], P_filt[:-1],
            m_pred[1:], P_pred[1:], F_all[1:],
            joint_pi_all[1:], pi_pred_all[1:],
            pi_filt_all[:-1],
        )
        _, (m_smooth_rev, P_smooth_rev, pi_smooth_rev) = jax.lax.scan(
            backward_step, init_s, bw_in, reverse=True,
        )
        m_s = jnp.concatenate([m_smooth_rev, m_filt[-1:]], axis=0)
        P_s = jnp.concatenate([P_smooth_rev, P_filt[-1:]], axis=0)
        pi_s = jnp.concatenate([pi_smooth_rev, pi_filt_all[-1:]], axis=0)
        return m_s, P_s, pi_s

    def _build_param_spec(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, ParameterTransform]]:
        params, spec = super()._build_param_spec()
        # Drop the inherited single-matrix Q: this model uses a per-state
        # process_cov (n x n x K) read directly from self.process_cov in
        # filter/smoother. Exposing a single Q for optimization would be
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

    def _sgd_loss_fn(
        self,
        params: Dict[str, Any],
        lfp_data: Array,
        spike_data: Array,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
        **kwargs,
    ) -> Array:
        # The parent's deterministic-rollout surrogate (use_filter=False) is a
        # single-trajectory warm-start with no analogue under discrete-state
        # switching, so this model supports only the filter-based marginal loss.
        # Reject use_filter=False loudly rather than silently ignoring it.
        if not use_filter:
            raise NotImplementedError(
                "SwitchingHamiltonianJointModel supports only the filter-based "
                "SGD loss (use_filter=True); the deterministic-rollout surrogate "
                "used by the non-switching Hamiltonian models is not defined for "
                "the switching model."
            )
        _, _, _, marginal_lls = self.filter(lfp_data, spike_data, params)
        lik_loss = -jnp.sum(marginal_lls)
        return lik_loss + l2_reg * mlp_l2_penalty(params["mlp"])

    def fit(self, *args, **kwargs):
        """Hamiltonian models do not support linear EM."""
        raise NotImplementedError(
            "SwitchingHamiltonianJointModel does not support the linear EM path "
            "(fit()). Please use fit_sgd() for non-linear optimization."
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

        # Resync BaseModel measurement_matrix across all discrete states.
        mm = jnp.zeros((self.n_sources, self.n_cont_states, self.n_discrete_states))
        for k in range(self.n_discrete_states):
            mm = mm.at[: self.n_lfp, :, k].set(self.C_lfp)
            mm = mm.at[self.n_lfp :, :, k].set(self.C_spikes)
        self.measurement_matrix = mm

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
        self.smoother_state_cond_mean = sm_means
        self.smoother_state_cond_cov = sm_covs
        self.smoother_discrete_state_prob = sm_probs

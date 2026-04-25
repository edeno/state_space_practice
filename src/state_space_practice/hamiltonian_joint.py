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

from state_space_practice.hamiltonian_core import (
    _BaseModelStubs,
    default_init_mean,
    ekf_rts_backward_pass,
    gaussian_measurement_update,
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
)
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import stabilize_covariance


class JointHamiltonianModel(_BaseModelStubs, BaseModel, SGDFittableMixin):
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
            n_sources=n_lfp_sources + n_spike_sources,
            sampling_freq=sampling_freq,
        )
        self.dt = 1.0 / sampling_freq
        self.n_lfp = n_lfp_sources
        self.n_spikes = n_spike_sources
        self.hidden_dims = hidden_dims or [32, 32]
        self.key = jax.random.PRNGKey(seed)
        k_mlp, k_lfp, k_spk, k_init = jax.random.split(self.key, 4)

        # Shared latent dynamics (the Hamiltonian)
        self.mlp_params = init_mlp_params(self.n_cont_states, self.hidden_dims, k_mlp)
        self.omega = 1.0

        # LFP head (Gaussian)
        self.C_lfp = jax.random.normal(k_lfp, (n_lfp_sources, self.n_cont_states)) * 0.1
        self.d_lfp = jnp.zeros((n_lfp_sources,))
        self.obs_noise_std = 0.1

        # Spike head (Poisson)
        self.C_spikes = jax.random.normal(k_spk, (n_spike_sources, self.n_cont_states)) * 0.1
        self.d_spikes = jnp.zeros((n_spike_sources,))

        self._initialize_parameters(k_init)
        self._sgd_n_time = 0

    def _initialize_parameters(self, key: Array) -> None:
        self.init_discrete_state_prob = jnp.ones((1,))
        self.discrete_transition_matrix = jnp.eye(1)
        m0 = default_init_mean(self.n_oscillators)
        self.init_mean = jnp.stack([m0], axis=1)
        self.init_cov = jnp.stack([jnp.eye(self.n_cont_states) * 0.1], axis=2)

        self.measurement_cov = jnp.zeros((self.n_sources, self.n_sources, 1))
        self.continuous_transition_matrix = jnp.stack(
            [jnp.eye(self.n_cont_states)], axis=2
        )
        self.process_cov = jnp.stack(
            [jnp.eye(self.n_cont_states) * 1e-4], axis=2
        )

    def transition_func(self, x: Array, params: Dict[str, Array]) -> Array:
        return leapfrog_step(x, params, apply_mlp, self.dt)

    def _r_lfp(self) -> Array:
        return jnp.eye(self.n_lfp) * self.obs_noise_std ** 2

    @partial(jax.jit, static_argnums=(0,))
    def filter(
        self,
        lfp_data: Array,
        spike_data: Array,
        params: Dict[str, Any],
    ) -> Tuple[Array, Array, Array]:
        """Hybrid EKF: sequentially update from LFP then Spikes."""
        trans_params = {**params["mlp"], "omega": params["omega"]}
        C_l, d_l = params["C_lfp"], params["d_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]
        R_l = self._r_lfp()
        Q = params.get("Q", self.process_cov[:, :, 0])

        def step(carry, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_prev, P_prev = carry

            m_pred, P_pred = ekf_predict_step(
                m_prev, P_prev, trans_params, apply_mlp, Q, self.dt
            )

            # Sequential: LFP update first, then point-process update on
            # the LFP posterior. The two log-likelihoods sum to the joint
            # marginal because the observations are conditionally
            # independent given x_t.
            m_mid, P_mid, ll_lfp = gaussian_measurement_update(
                m_pred, P_pred, y_lfp_t, C_l, d_l, R_l,
            )
            m_post, P_post, ll_spike = point_process_laplace_update(
                m_mid, P_mid, y_spike_t, C_s, d_s, self.dt,
            )
            return (m_post, P_post), (m_post, P_post, ll_lfp + ll_spike)

        m0 = params["init_mean"]
        P0 = self.init_cov[:, :, 0]
        _, (means, covs, lls) = jax.lax.scan(
            step, (m0, P0), (lfp_data, spike_data),
        )
        return means, covs, lls

    @partial(jax.jit, static_argnums=(0,))
    def smooth(
        self,
        lfp_data: Array,
        spike_data: Array,
        params: Dict[str, Any],
    ) -> Tuple[Array, Array]:
        """Apply EKF-RTS Smoother to joint data."""
        trans_params = {**params["mlp"], "omega": params["omega"]}
        C_l, d_l = params["C_lfp"], params["d_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]
        R_l = self._r_lfp()
        Q = params.get("Q", self.process_cov[:, :, 0])

        def forward_step(carry, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_prev, P_prev = carry
            m_pred, P_pred, F_t = ekf_predict_step_with_jacobian(
                m_prev, P_prev, trans_params, apply_mlp, Q, self.dt
            )
            m_mid, P_mid, _ = gaussian_measurement_update(
                m_pred, P_pred, y_lfp_t, C_l, d_l, R_l,
                include_normalization_const=False,
            )
            m_post, P_post, _ = point_process_laplace_update(
                m_mid, P_mid, y_spike_t, C_s, d_s, self.dt,
                compute_log_likelihood=False,
            )
            return (m_post, P_post), (m_post, P_post, m_pred, P_pred, F_t)

        m0 = params["init_mean"]
        P0 = self.init_cov[:, :, 0]
        _, (m_f, P_f, m_p, P_p, F) = jax.lax.scan(
            forward_step, (m0, P0), (lfp_data, spike_data),
        )
        return ekf_rts_backward_pass(m_f, P_f, m_p, P_p, F)

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
            self, lfp_obs, spike_obs,
            optimizer=optimizer, num_steps=num_steps, verbose=verbose,
            convergence_tol=convergence_tol, use_filter=use_filter, **kwargs,
        )

    def _build_param_spec(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, ParameterTransform]]:
        params = {
            "mlp": self.mlp_params, "omega": self.omega,
            "C_lfp": self.C_lfp, "d_lfp": self.d_lfp,
            "C_spikes": self.C_spikes, "d_spikes": self.d_spikes,
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
        self,
        params: Dict[str, Any],
        lfp_data: Array,
        spike_data: Array,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
        **kwargs,
    ) -> Array:
        if use_filter:
            _, _, lls = self.filter(lfp_data, spike_data, params)
            lik_loss = -jnp.sum(lls)
        else:
            # Surrogate loss: deterministic rollout. NOT the joint
            # state-space marginal — drops observation-noise scale,
            # process prior, and latent uncertainty. Useful for
            # warm-starting.
            trans_params = {**params["mlp"], "omega": params["omega"]}
            C_l, d_l = params["C_lfp"], params["d_lfp"]
            C_s, d_s = params["C_spikes"], params["d_spikes"]
            x0 = params["init_mean"]

            def scan_fn(x_prev, _):
                x_next = self.transition_func(x_prev, trans_params)
                return x_next, x_next

            _, x_traj = jax.lax.scan(scan_fn, x0, jnp.arange(lfp_data.shape[0]))

            mse_l = jnp.sum((lfp_data - (x_traj @ C_l.T + d_l)) ** 2)
            log_lambda = x_traj @ C_s.T + d_s
            rates = jnp.exp(log_lambda) * self.dt
            nll_s = -jnp.sum(spike_data * jnp.log(rates + 1e-10) - rates)
            lik_loss = mse_l + nll_s

        return lik_loss + l2_reg * mlp_l2_penalty(params["mlp"])

    def fit(self, *args, **kwargs):
        """Hamiltonian models do not support linear EM."""
        raise NotImplementedError(
            "JointHamiltonianModel does not support the linear EM path "
            "(fit()). Please use fit_sgd() for non-linear optimization."
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

        # Resync BaseModel measurement_matrix with the two heads.
        self.measurement_matrix = (
            jnp.zeros((self.n_sources, self.n_cont_states, 1))
            .at[: self.n_lfp, :, 0].set(self.C_lfp)
            .at[self.n_lfp :, :, 0].set(self.C_spikes)
        )

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

"""Hamiltonian LFP Model (Gaussian Observation).

Uses Hamiltonian dynamics with a linear-Gaussian readout for voltage data.

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
    gaussian_measurement_update,
    mlp_l2_penalty,
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
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import stabilize_covariance, validate_scalar


class HamiltonianLFPModel(_BaseModelStubs, BaseModel, SGDFittableMixin):
    """LFP Model with Hamiltonian dynamics and Gaussian noise."""

    def __init__(
        self,
        n_oscillators: int,
        n_sources: int,
        sampling_freq: float,
        hidden_dims: Optional[List[int]] = None,
        seed: int = 42,
        obs_noise_std: float = 0.1,
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

        # Observation: linear projection
        self.C = jax.random.normal(k_obs, (n_sources, self.n_cont_states)) * 0.1
        self.d = jnp.zeros((n_sources,))
        obs_noise_std = validate_scalar(obs_noise_std, "obs_noise_std", positive=True)
        self._initial_measurement_cov = jnp.eye(n_sources) * obs_noise_std**2

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
        self.measurement_cov = (
            jnp.zeros((self.n_sources, self.n_sources, 1))
            .at[:, :, 0]
            .set(self._initial_measurement_cov)
        )
        self.process_cov = jnp.stack([jnp.eye(self.n_cont_states) * 1e-4], axis=2)
        self.continuous_transition_matrix = jnp.stack(
            [jnp.eye(self.n_cont_states)], axis=2
        )

    @property
    def obs_noise_std(self) -> float:
        """Read-only scalar summary of the LFP measurement covariance.

        The LFP observation noise is a full learnable covariance stored in
        ``measurement_cov``; this returns ``sqrt(mean(diag(R)))``. It is read-only
        (unlike the joint model's settable property): the LFP model has no scalar
        noise parameter to reset ``R`` to, so mutate ``self.measurement_cov`` or
        refit to change the noise. Provided for API parity with
        ``JointHamiltonianModel.obs_noise_std``.
        """
        R = self.measurement_cov[:, :, 0]
        return float(jnp.sqrt(jnp.mean(jnp.diag(R))))

    def transition_func(self, x: Array, params: Dict[str, Array]) -> Array:
        """Deterministic Hamiltonian transition."""
        return leapfrog_step(x, params, apply_mlp, self.dt)

    def filter(
        self, lfp_data: Array, params: Dict[str, Any]
    ) -> Tuple[Array, Array, Array]:
        """Apply EKF filter to LFP data."""
        lfp_data = self._validate_lfp_data(lfp_data)
        return cast(
            Tuple[Array, Array, Array],
            self._filter_jit(lfp_data, self._complete_filter_params(params)),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _filter_jit(
        self, lfp_data: Array, params: Dict[str, Any]
    ) -> Tuple[Array, Array, Array]:
        """JIT-compiled filter core with all mutable inputs passed explicitly."""
        trans_params = {**params["mlp"], "omega": params["omega"]}
        C, d = params["C"], params["d"]
        R, Q = params["R"], params["Q"]

        def step(carry, y_t):
            m_prev, P_prev = carry
            m_pred, P_pred = ekf_predict_step(
                m_prev, P_prev, trans_params, apply_mlp, Q, self.dt
            )
            m_post, P_post, ll = gaussian_measurement_update(
                m_pred, P_pred, y_t, C, d, R
            )
            return (m_post, P_post), (m_post, P_post, ll)

        m0 = params["init_mean"]
        P0 = params["init_cov"]
        _, (means, covs, lls) = jax.lax.scan(step, (m0, P0), lfp_data)
        return means, covs, lls

    def smooth(self, lfp_data: Array, params: Dict[str, Any]) -> Tuple[Array, Array]:
        """Apply EKF-RTS Smoother to LFP data."""
        lfp_data = self._validate_lfp_data(lfp_data)
        return cast(
            Tuple[Array, Array],
            self._smooth_jit(lfp_data, self._complete_filter_params(params)),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _smooth_jit(
        self, lfp_data: Array, params: Dict[str, Any]
    ) -> Tuple[Array, Array]:
        """JIT-compiled smoother core with all mutable inputs passed explicitly."""
        trans_params = {**params["mlp"], "omega": params["omega"]}
        C, d = params["C"], params["d"]
        R, Q = params["R"], params["Q"]

        def forward_step(carry, y_t):
            m_prev, P_prev = carry
            m_pred, P_pred, F_t = ekf_predict_step_with_jacobian(
                m_prev, P_prev, trans_params, apply_mlp, Q, self.dt
            )
            m_post, P_post, _ = gaussian_measurement_update(
                m_pred,
                P_pred,
                y_t,
                C,
                d,
                R,
                include_normalization_const=False,
            )
            return (m_post, P_post), (m_post, P_post, m_pred, P_pred, F_t)

        m0 = params["init_mean"]
        P0 = params["init_cov"]
        _, (m_f, P_f, m_p, P_p, F) = jax.lax.scan(forward_step, (m0, P0), lfp_data)
        return ekf_rts_backward_pass(m_f, P_f, m_p, P_p, F)

    def _validate_lfp_data(self, lfp_data: Array, *, allow_empty: bool = True) -> Array:
        """Validate public LFP input and return it as a JAX array."""
        lfp_data = jnp.asarray(lfp_data)
        if lfp_data.ndim != 2:
            raise ValueError(
                f"lfp_data must have shape (n_time, n_sources); got {lfp_data.shape}."
            )
        if lfp_data.shape[1] != self.n_sources:
            raise ValueError(
                f"lfp_data must have {self.n_sources} sources, got {lfp_data.shape[1]}."
            )
        if not allow_empty and lfp_data.shape[0] == 0:
            raise ValueError("lfp_data must contain at least one observation.")
        if not bool(jnp.all(jnp.isfinite(lfp_data))):
            raise ValueError("lfp_data must contain only finite values.")
        return lfp_data

    def _complete_filter_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fill covariance defaults before entering a JIT-compiled method."""
        complete = dict(params)
        complete.setdefault("R", self.measurement_cov[:, :, 0])
        complete.setdefault("Q", self.process_cov[:, :, 0])
        complete.setdefault("init_cov", self.init_cov[:, :, 0])
        return complete

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
            "R": self.measurement_cov[:, :, 0],
            "Q": self.process_cov[:, :, 0],
        }
        spec = {
            "mlp": UNCONSTRAINED,
            "omega": POSITIVE,
            "C": UNCONSTRAINED,
            "d": UNCONSTRAINED,
            "init_mean": UNCONSTRAINED,
            "init_cov": frozen(PSD_MATRIX),
            "R": PSD_MATRIX,
            "Q": PSD_MATRIX,
        }
        return params, spec

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
        observations = self._validate_lfp_data(observations, allow_empty=False)
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

    def _sgd_loss_fn(
        self,
        params: Dict[str, Any],
        lfp_data: Array,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
        **kwargs,
    ) -> Array:
        if use_filter:
            _, _, lls = self._filter_jit(lfp_data, params)
            lik_loss = -jnp.sum(lls)
        else:
            # Surrogate loss: deterministic rollout SSE. NOT the
            # Gaussian state-space marginal — drops observation noise,
            # process prior, and latent uncertainty. Useful for
            # warm-starting dynamics before switching to use_filter=True.
            trans_params = {**params["mlp"], "omega": params["omega"]}
            C, d = params["C"], params["d"]
            x0 = params["init_mean"]

            def scan_fn(x_prev, _):
                x_next = self.transition_func(x_prev, trans_params)
                return x_next, x_next

            _, x_traj = jax.lax.scan(scan_fn, x0, jnp.arange(lfp_data.shape[0]))
            lik_loss = jnp.sum((lfp_data - (x_traj @ C.T + d)) ** 2)

        return lik_loss + l2_reg * mlp_l2_penalty(params["mlp"])

    def fit(self, *args, **kwargs):
        """Hamiltonian models do not support linear EM."""
        raise NotImplementedError(
            "HamiltonianLFPModel does not support the linear EM path "
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
        if "R" in params:
            self.measurement_cov = jnp.stack(
                [stabilize_covariance(params["R"])], axis=2
            )
        if "Q" in params:
            self.process_cov = jnp.stack([stabilize_covariance(params["Q"])], axis=2)

    def _finalize_sgd(self, lfp_data, **kwargs):
        """Run filter + smoother to populate fitted states after SGD."""
        params = self._build_param_spec()[0]
        means, covs, lls = self.filter(lfp_data, params)
        self.filtered_means_ = means
        self.filtered_covs_ = covs
        self.log_likelihood_ = float(jnp.sum(lls))
        sm_means, sm_covs = self.smooth(lfp_data, params)
        self.smoothed_means_ = sm_means
        self.smoothed_covs_ = sm_covs

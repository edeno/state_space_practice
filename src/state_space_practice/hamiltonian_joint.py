"""Joint Hamiltonian Model for LFP and Spikes.

Unifies continuous voltage (LFP) and sparse point-processes (Spikes)
under a single shared Hamiltonian latent trajectory.

See docs/hamiltonian_architecture.md for why this family is standalone
(no linear-Gaussian EM integration, SGD-only fitting).
"""

import warnings
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.hamiltonian_core import (
    _BaseModelStubs,
    default_init_mean,
    gaussian_measurement_update,
    mlp_l2_penalty,
    point_process_laplace_update,
    poisson_rollout_nll,
    run_ekf_filter,
    run_ekf_smoother,
)
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    init_mlp_params,
    leapfrog_step,
)
from state_space_practice.oscillator_models import BaseModel
from state_space_practice.parameter_transforms import (
    POSITIVE,
    PSD_MATRIX,
    UNCONSTRAINED,
    ParameterTransform,
    frozen,
)
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import (
    psd_cholesky,
    psd_logdet,
    stabilize_covariance,
    validate_count_array,
    validate_scalar,
)


class JointHamiltonianModel(_BaseModelStubs, BaseModel, SGDFittableMixin):
    """Joint Model combining Gaussian LFP and Poisson Spikes."""

    def __init__(
        self,
        n_oscillators: int,
        n_lfp_sources: int,
        n_spike_sources: int,
        sampling_freq: float,
        hidden_dims: list[int] | None = None,
        seed: int = 42,
        obs_noise_std: float = 0.1,
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
        self.mlp_params = init_mlp_params(self.n_oscillators, self.hidden_dims, k_mlp)
        self.omega: float | Array = 1.0

        # LFP head (Gaussian)
        self.C_lfp = jax.random.normal(k_lfp, (n_lfp_sources, self.n_cont_states)) * 0.1
        self.d_lfp = jnp.zeros((n_lfp_sources,))
        self.obs_noise_std = obs_noise_std

        # Spike head (Poisson)
        self.C_spikes = (
            jax.random.normal(k_spk, (n_spike_sources, self.n_cont_states)) * 0.1
        )
        self.d_spikes = jnp.zeros((n_spike_sources,))

        self._initialize_parameters(k_init)
        self._sgd_n_time = 0

    def _initialize_parameters(self, key: Array) -> None:
        self.init_discrete_state_prob = jnp.ones((1,))
        self.discrete_transition_matrix = jnp.eye(1)
        m0 = default_init_mean(self.n_oscillators)
        self.init_mean = jnp.stack([m0], axis=1)
        self.init_cov = jnp.stack([jnp.eye(self.n_cont_states) * 0.1], axis=2)

        self.measurement_matrix = (
            jnp.zeros((self.n_sources, self.n_cont_states, 1))
            .at[: self.n_lfp, :, 0]
            .set(self.C_lfp)
            .at[self.n_lfp :, :, 0]
            .set(self.C_spikes)
        )
        self.measurement_cov = (
            jnp.zeros((self.n_sources, self.n_sources, 1))
            .at[: self.n_lfp, : self.n_lfp, 0]
            .set(self.R_lfp)
        )
        self.continuous_transition_matrix = jnp.stack(
            [jnp.eye(self.n_cont_states)], axis=2
        )
        self.process_cov = jnp.stack([jnp.eye(self.n_cont_states) * 1e-4], axis=2)

    def transition_func(self, x: Array, params: dict[str, Array]) -> Array:
        return leapfrog_step(x, params, apply_mlp, self.dt)

    @property
    def obs_noise_std(self) -> float:
        """Scalar summary/configuration for the LFP measurement covariance."""
        return self._obs_noise_std

    @obs_noise_std.setter
    def obs_noise_std(self, value: float) -> None:
        value = validate_scalar(value, "obs_noise_std", positive=True)
        # Setting the scalar resets R_lfp to isotropic sigma**2 * I. If R_lfp is
        # currently a full (non-isotropic) covariance -- e.g. one learned by
        # fit_sgd -- that structure would be silently discarded, so warn. The
        # getter only exposes a scalar summary, so a caller cannot otherwise see
        # what the assignment destroys.
        current_R = getattr(self, "R_lfp", None)
        if current_R is not None:
            isotropic = jnp.allclose(
                current_R, jnp.eye(self.n_lfp) * jnp.mean(jnp.diag(current_R))
            )
            if not bool(isotropic):
                warnings.warn(
                    "Setting obs_noise_std resets R_lfp to an isotropic "
                    "sigma**2 * I and discards the current non-isotropic LFP "
                    "measurement covariance (e.g. one learned by fit_sgd). Set "
                    "obs_noise_std before fitting, or assign self.R_lfp directly "
                    "to keep a full covariance.",
                    UserWarning,
                    stacklevel=2,
                )
        self._obs_noise_std = value
        self.R_lfp = jnp.eye(self.n_lfp) * value**2
        if hasattr(self, "measurement_cov"):
            R_all_states = jnp.broadcast_to(
                self.R_lfp[:, :, None],
                (self.n_lfp, self.n_lfp, self.n_discrete_states),
            )
            self.measurement_cov = self.measurement_cov.at[
                : self.n_lfp, : self.n_lfp, :
            ].set(R_all_states)

    def _r_lfp(self) -> Array:
        return self.R_lfp

    def filter(
        self,
        lfp_data: Array,
        spike_data: Array,
        params: dict[str, Any],
    ) -> tuple[Array, Array, Array]:
        """Hybrid EKF: sequentially update from LFP then Spikes."""
        lfp_data, spike_data = self._validate_joint_data(lfp_data, spike_data)
        return cast(
            tuple[Array, Array, Array],
            self._filter_jit(
                lfp_data,
                spike_data,
                self._complete_filter_params(params),
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _filter_jit(
        self,
        lfp_data: Array,
        spike_data: Array,
        params: dict[str, Any],
    ) -> tuple[Array, Array, Array]:
        """JIT-compiled filter core with mutable inputs passed explicitly."""
        trans_params = {**params["mlp"], "omega": params["omega"]}
        C_l, d_l, R_l = params["C_lfp"], params["d_lfp"], params["R_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]

        def update(m_pred, P_pred, obs_t):
            # Sequential: LFP update first, then point-process update on
            # the LFP posterior. The two log-likelihoods sum to the joint
            # marginal because the observations are conditionally
            # independent given x_t.
            y_lfp_t, y_spike_t = obs_t
            m_mid, P_mid, ll_lfp = gaussian_measurement_update(
                m_pred, P_pred, y_lfp_t, C_l, d_l, R_l
            )
            m_post, P_post, ll_spike = point_process_laplace_update(
                m_mid, P_mid, y_spike_t, C_s, d_s, self.dt
            )
            return m_post, P_post, ll_lfp + ll_spike

        return run_ekf_filter(
            (lfp_data, spike_data),
            params["init_mean"],
            params["init_cov"],
            trans_params,
            params["Q"],
            self.dt,
            update,
        )

    def smooth(
        self,
        lfp_data: Array,
        spike_data: Array,
        params: dict[str, Any],
    ) -> tuple[Array, Array]:
        """Apply EKF-RTS Smoother to joint data."""
        lfp_data, spike_data = self._validate_joint_data(lfp_data, spike_data)
        return cast(
            tuple[Array, Array],
            self._smooth_jit(
                lfp_data,
                spike_data,
                self._complete_filter_params(params),
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _smooth_jit(
        self,
        lfp_data: Array,
        spike_data: Array,
        params: dict[str, Any],
    ) -> tuple[Array, Array]:
        """JIT-compiled smoother core with mutable inputs passed explicitly."""
        trans_params = {**params["mlp"], "omega": params["omega"]}
        C_l, d_l, R_l = params["C_lfp"], params["d_lfp"], params["R_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]

        def update(m_pred, P_pred, obs_t):
            y_lfp_t, y_spike_t = obs_t
            m_mid, P_mid, _ = gaussian_measurement_update(
                m_pred,
                P_pred,
                y_lfp_t,
                C_l,
                d_l,
                R_l,
                include_normalization_const=False,
            )
            m_post, P_post, _ = point_process_laplace_update(
                m_mid, P_mid, y_spike_t, C_s, d_s, self.dt, compute_log_likelihood=False
            )
            return m_post, P_post

        return run_ekf_smoother(
            (lfp_data, spike_data),
            params["init_mean"],
            params["init_cov"],
            trans_params,
            params["Q"],
            self.dt,
            update,
        )

    def _validate_joint_data(
        self,
        lfp_data: Array,
        spike_data: Array,
        *,
        allow_empty: bool = True,
    ) -> tuple[Array, Array]:
        """Validate aligned public LFP and spike observations."""
        lfp_data = jnp.asarray(lfp_data)
        spike_data = jnp.asarray(spike_data)
        if lfp_data.ndim != 2 or lfp_data.shape[1] != self.n_lfp:
            raise ValueError(
                "lfp_data must have shape (n_time, n_lfp_sources); "
                f"expected second dimension {self.n_lfp}, got {lfp_data.shape}."
            )
        if spike_data.ndim != 2 or spike_data.shape[1] != self.n_spikes:
            raise ValueError(
                "spike_data must have shape (n_time, n_spike_sources); "
                f"expected second dimension {self.n_spikes}, got {spike_data.shape}."
            )
        if lfp_data.shape[0] != spike_data.shape[0]:
            raise ValueError(
                "lfp_data and spike_data must have the same number of time rows; "
                f"got {lfp_data.shape[0]} and {spike_data.shape[0]}."
            )
        if not allow_empty and lfp_data.shape[0] == 0:
            raise ValueError("joint observations must contain at least one time row.")
        if not bool(jnp.all(jnp.isfinite(lfp_data))):
            raise ValueError("lfp_data must contain only finite values.")
        validate_count_array(spike_data, "spike_data", allow_empty=allow_empty)
        return lfp_data, spike_data

    def _complete_filter_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fill covariance defaults before entering a JIT-compiled method."""
        complete = dict(params)
        complete.setdefault("R_lfp", self.R_lfp)
        complete.setdefault("Q", self.process_cov[:, :, 0])
        complete.setdefault("init_cov", self.init_cov[:, :, 0])
        return complete

    def fit_sgd(  # type: ignore[override]
        self,
        lfp_obs: Array,
        spike_obs: Array,
        optimizer: object | None = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: float | None = None,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
    ) -> list[float]:
        lfp_obs, spike_obs = self._validate_joint_data(
            lfp_obs, spike_obs, allow_empty=False
        )
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
            l2_reg=l2_reg,
        )

    def _build_param_spec(
        self,
    ) -> tuple[dict[str, Any], dict[str, ParameterTransform]]:
        params = {
            "mlp": self.mlp_params,
            "omega": self.omega,
            "C_lfp": self.C_lfp,
            "d_lfp": self.d_lfp,
            "C_spikes": self.C_spikes,
            "d_spikes": self.d_spikes,
            "init_mean": self.init_mean[:, 0],
            "init_cov": self.init_cov[:, :, 0],
            "R_lfp": self.R_lfp,
            "Q": self.process_cov[:, :, 0],
        }
        spec = {
            "mlp": UNCONSTRAINED,
            "omega": POSITIVE,
            "C_lfp": UNCONSTRAINED,
            "d_lfp": UNCONSTRAINED,
            "C_spikes": UNCONSTRAINED,
            "d_spikes": UNCONSTRAINED,
            "init_mean": UNCONSTRAINED,
            "init_cov": frozen(PSD_MATRIX),
            "R_lfp": PSD_MATRIX,
            "Q": PSD_MATRIX,
        }
        return params, spec

    def _sgd_loss_fn(
        self,
        params: dict[str, Any],
        lfp_data: Array,
        spike_data: Array,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
        **kwargs,
    ) -> Array:
        if use_filter:
            _, _, lls = self._filter_jit(lfp_data, spike_data, params)
            lik_loss = -jnp.sum(lls)
        else:
            # Surrogate loss: deterministic rollout. NOT the joint
            # state-space marginal — it uses normalized observation
            # likelihoods but drops the process prior and latent uncertainty.
            # Useful for warm-starting.
            C_l, d_l = params["C_lfp"], params["d_lfp"]
            C_s, d_s = params["C_spikes"], params["d_spikes"]
            x_traj = self._rollout_trajectory(params, lfp_data.shape[0])

            residual_l = lfp_data - (x_traj @ C_l.T + d_l)
            R_l_cho = psd_cholesky(params["R_lfp"])
            solved_l = jax.scipy.linalg.cho_solve(R_l_cho, residual_l.T).T
            logdet_R_l = psd_logdet(R_l_cho)
            nll_l = 0.5 * (
                jnp.sum(residual_l * solved_l)
                + lfp_data.shape[0] * (logdet_R_l + self.n_lfp * jnp.log(2.0 * jnp.pi))
            )
            nll_s = poisson_rollout_nll(x_traj @ C_s.T + d_s, spike_data, self.dt)
            lik_loss = nll_l + nll_s

        return lik_loss + l2_reg * mlp_l2_penalty(params["mlp"])

    def _store_sgd_params(self, params: dict[str, Any]) -> None:
        self.mlp_params = params["mlp"]
        self.omega = params["omega"]
        self.C_lfp = params["C_lfp"]
        self.d_lfp = params["d_lfp"]
        self.C_spikes = params["C_spikes"]
        self.d_spikes = params["d_spikes"]
        self.init_mean = self.init_mean.at[:, 0].set(params["init_mean"])
        if "R_lfp" in params:
            self.R_lfp = stabilize_covariance(params["R_lfp"])
            self._obs_noise_std = float(jnp.sqrt(jnp.mean(jnp.diag(self.R_lfp))))
        if "Q" in params:
            self.process_cov = jnp.stack([stabilize_covariance(params["Q"])], axis=2)

        # Resync BaseModel measurement_matrix with the two heads.
        self.measurement_matrix = (
            jnp.zeros((self.n_sources, self.n_cont_states, 1))
            .at[: self.n_lfp, :, 0]
            .set(self.C_lfp)
            .at[self.n_lfp :, :, 0]
            .set(self.C_spikes)
        )
        self.measurement_cov = (
            jnp.zeros((self.n_sources, self.n_sources, 1))
            .at[: self.n_lfp, : self.n_lfp, 0]
            .set(self.R_lfp)
        )


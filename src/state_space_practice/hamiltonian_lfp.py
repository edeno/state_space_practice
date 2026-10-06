"""Hamiltonian LFP Model (Gaussian Observation).

Uses Hamiltonian dynamics with a linear-Gaussian readout for voltage data.

See docs/hamiltonian_architecture.md for why this family has no
linear-Gaussian EM integration and is fit by SGD only.
"""

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.hamiltonian_core import (
    HamiltonianModelBase,
    _SingleRegimeHamiltonianModel,
    default_init_mean,
    mlp_l2_penalty,
)
from state_space_practice.nonlinear_dynamics import init_mlp_params
from state_space_practice.parameter_transforms import (
    POSITIVE,
    PSD_MATRIX,
    UNCONSTRAINED,
    ParameterTransform,
    frozen,
)
from state_space_practice.utils import (
    stabilize_covariance,
    validate_finite_array,
    validate_scalar,
)

if TYPE_CHECKING:
    import optax


class HamiltonianLFPModel(_SingleRegimeHamiltonianModel):
    """LFP Model with Hamiltonian dynamics and Gaussian noise."""

    _observation_model = "gaussian"
    _sgd_param_attrs = {**HamiltonianModelBase._sgd_param_attrs, "C": "C", "d": "d"}

    def __init__(
        self,
        n_oscillators: int,
        n_sources: int,
        sampling_freq: float,
        hidden_dims: list[int] | None = None,
        seed: int = 42,
        obs_noise_std: float = 0.1,
    ):
        """Build the model with random initial weights.

        Parameters
        ----------
        n_oscillators : int
            Number of latent oscillators; the state holds one position and
            one momentum per oscillator (``n_cont_states = 2 * n_oscillators``).
        n_sources : int
            Number of LFP channels.
        sampling_freq : float
            Sampling rate in Hz; the leapfrog step is ``1 / sampling_freq``.
        hidden_dims : list of int or None
            Hidden-layer widths of the MLP that parameterizes the Hamiltonian
            (default ``[32, 32]``).
        seed : int
            Seed for the initial MLP and observation weights.
        obs_noise_std : float
            Initial LFP noise standard deviation; the measurement covariance
            starts at ``obs_noise_std**2 * I`` and is learned by ``fit_sgd``.

        Raises
        ------
        ValueError
            If ``obs_noise_std`` is not a positive finite scalar.
        """
        super().__init__(
            n_oscillators=n_oscillators,
            n_discrete_states=1,
            n_sources=n_sources,
            sampling_freq=sampling_freq,
            hidden_dims=hidden_dims,
            seed=seed,
        )
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

    def _initialize_parameters(self, key: Array) -> None:
        self.init_discrete_state_prob = jnp.ones((1,))
        self.discrete_transition_matrix = jnp.eye(1)
        m0 = default_init_mean(self.n_oscillators)
        self.init_mean = jnp.stack([m0], axis=1)
        self.init_cov = jnp.stack([jnp.eye(self.n_cont_states) * 0.1], axis=2)

        self.measurement_matrix = self.C[:, :, None]
        self.measurement_cov = self._initial_measurement_cov[:, :, None]
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

    def filter(
        self, lfp_data: ArrayLike, params: dict[str, Any]
    ) -> tuple[Array, Array, Array]:
        """Apply EKF filter to LFP data."""
        lfp_data = self._validate_lfp_data(lfp_data)
        return self._filter_jit(lfp_data, self._complete_filter_params(params))

    def smooth(
        self, lfp_data: ArrayLike, params: dict[str, Any]
    ) -> tuple[Array, Array]:
        """Apply EKF-RTS Smoother to LFP data."""
        lfp_data = self._validate_lfp_data(lfp_data)
        return self._smooth_jit(lfp_data, self._complete_filter_params(params))

    def _validate_lfp_data(
        self, lfp_data: ArrayLike, *, allow_empty: bool = True
    ) -> Array:
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
        validate_finite_array("lfp_data", lfp_data)
        return lfp_data

    def fit_sgd(
        self,
        observations: ArrayLike,
        optimizer: "optax.GradientTransformation | None" = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: float | None = None,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
    ) -> list[float]:
        """Fit by gradient descent on the negative log-likelihood + L2 penalty.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            LFP observations.
        optimizer : optax.GradientTransformation or None, optional
            Default: ``adam(1e-2)`` with global-norm gradient clipping.
        num_steps : int, optional
            Number of optimization steps.
        verbose : bool, optional
            Print progress every 10 steps.
        convergence_tol : float or None, optional
            Stop early once the relative LL change stays below this for 5
            consecutive steps.
        use_filter : bool, optional
            ``True`` optimizes the marginal EKF log-likelihood and learns the
            process covariance; ``False`` optimizes a deterministic-rollout
            surrogate (no process prior, no latent uncertainty) for warm starts.
        l2_reg : float, optional
            Weight of the L2 penalty on the MLP weights.

        Returns
        -------
        log_likelihoods : list of float
            Training objective per accepted optimization step: the
            log-likelihood (or the rollout surrogate) minus the L2 penalty.
            ``log_likelihood_`` is instead the filter's marginal
            log-likelihood at the fitted parameters.
        """
        return super().fit_sgd(
            observations,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
            use_filter=use_filter,
            l2_reg=l2_reg,
        )

    def _validate_fit_data(self, observations: Array) -> tuple[Array, ...]:
        return (self._validate_lfp_data(observations, allow_empty=False),)

    def _complete_filter_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fill covariance defaults before entering the jitted cores.

        The compiled cores read only ``params``, so the current model
        covariances must travel with it rather than be captured from ``self``
        (which would freeze stale values into the compile cache).
        """
        complete = dict(params)
        complete.setdefault("R", self.measurement_cov[:, :, 0])
        complete.setdefault("Q", self.process_cov[:, :, 0])
        complete.setdefault("init_cov", self.init_cov[:, :, 0])
        return complete

    def _build_param_spec(
        self,
    ) -> tuple[dict[str, Any], dict[str, ParameterTransform]]:
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

    def _sgd_loss_fn(
        self,
        params: dict[str, Any],
        lfp_data: Array,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
        **kwargs: Any,
    ) -> Array:
        if use_filter:
            _, _, lls = self._filter_jit(lfp_data, params)
            lik_loss = -jnp.sum(lls)
        else:
            # Surrogate loss: deterministic rollout SSE. NOT the
            # Gaussian state-space marginal — drops observation noise,
            # process prior, and latent uncertainty. Useful for
            # warm-starting dynamics before switching to use_filter=True.
            C, d = params["C"], params["d"]
            x_traj = self._rollout_trajectory(params, lfp_data.shape[0])
            lik_loss = jnp.sum((lfp_data - (x_traj @ C.T + d)) ** 2)

        return lik_loss + l2_reg * mlp_l2_penalty(params["mlp"])

    def _store_sgd_params(self, params: dict[str, Any]) -> None:
        super()._store_sgd_params(params)
        self.measurement_matrix = self.C[:, :, None]
        if "R" in params:
            self.measurement_cov = jnp.stack(
                [stabilize_covariance(params["R"])], axis=2
            )

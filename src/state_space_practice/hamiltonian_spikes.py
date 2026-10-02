"""Hamiltonian Spike Model (Point-Process Observation).

Uses Hamiltonian dynamics with a Poisson/Point-Process readout for spike data.

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
    default_init_mean,
    mlp_l2_penalty,
    poisson_rollout_nll,
)
from state_space_practice.nonlinear_dynamics import init_mlp_params
from state_space_practice.parameter_transforms import (
    POSITIVE,
    PSD_MATRIX,
    UNCONSTRAINED,
    ParameterTransform,
    frozen,
)
from state_space_practice.utils import validate_count_array

if TYPE_CHECKING:
    import optax


class HamiltonianSpikeModel(HamiltonianModelBase):
    """Spike Model with Hamiltonian dynamics and Point-Process observations.

    ``fit_sgd(spikes, use_filter=True)`` optimizes the marginal Laplace-EKF
    likelihood and learns the process covariance; ``use_filter=False`` is a
    deterministic-rollout surrogate intended for warm starts that neither
    depends on nor updates the process covariance.
    """

    _observation_model = "poisson"
    _sgd_param_attrs = {**HamiltonianModelBase._sgd_param_attrs, "C": "C", "d": "d"}

    def __init__(
        self,
        n_oscillators: int,
        n_sources: int,  # n_neurons
        sampling_freq: float,
        hidden_dims: list[int] | None = None,
        seed: int = 42,
    ):
        """Build the model with random initial weights.

        Parameters
        ----------
        n_oscillators : int
            Number of latent oscillators; the state holds one position and
            one momentum per oscillator (``n_cont_states = 2 * n_oscillators``).
        n_sources : int
            Number of neurons (spike-count channels).
        sampling_freq : float
            Sampling rate in Hz; also the bin rate of the spike counts.
        hidden_dims : list of int or None
            Hidden-layer widths of the MLP that parameterizes the Hamiltonian
            (default ``[32, 32]``).
        seed : int
            Seed for the initial MLP and observation weights.
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

        # Observation: log-linear Poisson intensity
        self.C = jax.random.normal(k_obs, (n_sources, self.n_cont_states)) * 0.1
        self.d = jnp.zeros((n_sources,))

        self._initialize_parameters(k_init)

    def _initialize_parameters(self, key: Array) -> None:
        self.init_discrete_state_prob = jnp.ones((1,))
        self.discrete_transition_matrix = jnp.eye(1)
        m0 = default_init_mean(self.n_oscillators)
        self.init_mean = jnp.stack([m0], axis=1)
        self.init_cov = jnp.stack([jnp.eye(self.n_cont_states) * 0.1], axis=2)

        self.measurement_matrix = self.C[:, :, None]
        self.process_cov = jnp.stack([jnp.eye(self.n_cont_states) * 1e-4], axis=2)
        self.continuous_transition_matrix = jnp.stack(
            [jnp.eye(self.n_cont_states)], axis=2
        )

    def filter(
        self, spikes: ArrayLike, params: dict[str, Any]
    ) -> tuple[Array, Array, Array]:
        """Apply Point-Process EKF (Laplace-EKF) to spikes."""
        spikes = self._validate_spikes(spikes)
        return self._filter_jit(spikes, self._complete_filter_params(params))

    def smooth(self, spikes: ArrayLike, params: dict[str, Any]) -> tuple[Array, Array]:
        """Apply Point-Process RTS Smoother to spikes."""
        spikes = self._validate_spikes(spikes)
        return self._smooth_jit(spikes, self._complete_filter_params(params))

    def _validate_spikes(self, spikes: ArrayLike, *, allow_empty: bool = True) -> Array:
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
        """Fit by gradient descent on the negative log-likelihood.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            Non-negative integer spike counts.
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
            ``True`` optimizes the marginal Laplace-EKF log-likelihood and
            learns the process covariance; ``False`` optimizes a
            deterministic-rollout surrogate (no process prior, no latent
            uncertainty) for warm starts.
        l2_reg : float, optional
            Weight of the L2 penalty on the MLP weights.

        Returns
        -------
        log_likelihoods : list of float
            Log-likelihood (or surrogate) per accepted optimization step.
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
        return (self._validate_spikes(observations, allow_empty=False),)

    def _complete_filter_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fill covariance defaults before entering the jitted cores.

        The compiled cores read only ``params``, so the current model
        covariances must travel with it rather than be captured from ``self``
        (which would freeze stale values into the compile cache).
        """
        complete = dict(params)
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
        params: dict[str, Any],
        spikes: Array,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
        **kwargs: Any,
    ) -> Array:
        if use_filter:
            _, _, lls = self._filter_jit(spikes, params)
            lik_loss = -jnp.sum(lls)
        else:
            # Surrogate loss: deterministic rollout scored under the Poisson
            # observation model (no process prior, no latent uncertainty).
            C, d = params["C"], params["d"]
            x_traj = self._rollout_trajectory(params, spikes.shape[0])
            lik_loss = poisson_rollout_nll(x_traj @ C.T + d, spikes, self.dt)

        return lik_loss + l2_reg * mlp_l2_penalty(params["mlp"])

    def _store_sgd_params(self, params: dict[str, Any]) -> None:
        super()._store_sgd_params(params)
        self.measurement_matrix = self.C[:, :, None]

"""Hamiltonian Oscillator State-Space Models.

This module implements a standalone Hamiltonian Point-Process model
that performs latent-state inference using a symplectic nonlinear transition
and a Point-Process Laplace update.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.parameter_transforms import UNCONSTRAINED, ParameterTransform
from state_space_practice.sgd_fitting import SGDFittableMixin

if TYPE_CHECKING:
    from state_space_practice.hamiltonian_point_process import HamiltonianFilterResult
else:
    HamiltonianFilterResult = Any

logger = logging.getLogger(__name__)

# --- Task 2: Structured Hamiltonian Dynamics ---

def init_structured_hamiltonian_params(
    n_oscillators: int, 
    key: Array
) -> Dict[str, Array]:
    """Initialize interpretable Hamiltonian parameters."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    params = {
        "log_mass": jnp.zeros((n_oscillators,)), # mass = 1
        "log_stiffness": jnp.zeros((n_oscillators,)), # k = 1
        "coupling_q": jax.random.normal(k1, (n_oscillators, n_oscillators)) * 0.01,
    }
    # Zero diagonal for coupling
    params["coupling_q"] = params["coupling_q"] * (1.0 - jnp.eye(n_oscillators))
    return params

def kinetic_energy(p: Array, params: Dict[str, Array]) -> Array:
    """T(p) = 0.5 * p^T * M^-1 * p"""
    mass = jnp.exp(params["log_mass"])
    return 0.5 * jnp.sum((p**2) / mass)

def potential_energy(q: Array, params: Dict[str, Array]) -> Array:
    """V(q) = 0.5 * q^T * K * q"""
    stiffness = jnp.exp(params["log_stiffness"])
    return 0.5 * jnp.sum(stiffness * (q**2))

def coupling_energy(q: Array, params: Dict[str, Array]) -> Array:
    """C(q) = 0.5 * q^T * W * q"""
    K_coupled = params["coupling_q"]
    W = 0.5 * (K_coupled + K_coupled.T)
    return 0.5 * q @ W @ q

def hamiltonian(state: Array, params: Dict[str, Array]) -> Array:
    """H(q, p) = T(p) + V(q) + C(q)"""
    q = state[0::2]
    p = state[1::2]
    return kinetic_energy(p, params) + potential_energy(q, params) + coupling_energy(q, params)

def symplectic_transition(state: Array, params: Dict[str, Array], dt: float) -> Array:
    """Single-step Leapfrog integrator for Hamiltonian dynamics."""
    q, p = state[0::2], state[1::2]
    
    def get_dH_dq(q_val, p_val):
        st = jnp.zeros_like(state)
        st = st.at[0::2].set(q_val)
        st = st.at[1::2].set(p_val)
        return jax.grad(hamiltonian)(st, params)[0::2]

    def get_dH_dp(q_val, p_val):
        st = jnp.zeros_like(state)
        st = st.at[0::2].set(q_val)
        st = st.at[1::2].set(p_val)
        return jax.grad(hamiltonian)(st, params)[1::2]

    # 1. p(t + dt/2) = p(t) - (dt/2) * dH/dq
    p_mid = p - (dt / 2.0) * get_dH_dq(q, p)
    # 2. q(t + dt) = q(t) + dt * dH/dp(p_mid)
    q_next = q + dt * get_dH_dp(q, p_mid)
    # 3. p(t + dt) = p_mid - (dt/2) * dH/dq(q_next)
    p_next = p_mid - (dt / 2.0) * get_dH_dq(q_next, p_mid)
    
    return jnp.stack([q_next, p_next], axis=1).flatten()

def transition_jacobian(state: Array, params: Dict[str, Array], dt: float) -> Array:
    """Local Jacobian of the symplectic transition."""
    return jax.jacfwd(lambda s: symplectic_transition(s, params, dt))(state)

# --- Task 1: Standalone SGD Model ---

class HamiltonianPointProcessModel(SGDFittableMixin):
    """Standalone Hamiltonian model for point-process observations.
    
    V1: No BaseModel inheritance. Structured Hamiltonian.
    """
    
    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        sampling_freq: float,
        dt: float,
        seed: int = 42,
    ):
        self.n_oscillators = n_oscillators
        self.n_neurons = n_neurons
        self.n_latent = 2 * n_oscillators
        self.sampling_freq = sampling_freq
        self.dt = dt
        self.key = jax.random.PRNGKey(seed)
        k_hamiltonian, k_C = jax.random.split(self.key)

        # Dynamics
        self.hamiltonian_params = init_structured_hamiltonian_params(n_oscillators, k_hamiltonian)
        self.process_cov = 1e-4 * jnp.eye(self.n_latent)

        # Initialization
        self.init_mean = jnp.zeros((self.n_latent,))
        self.init_cov = 1e-1 * jnp.eye(self.n_latent)

        # Observation (Log-linear)
        self.C = jax.random.normal(k_C, (n_neurons, self.n_latent)) * 0.1
        self.d = jnp.zeros((n_neurons,))
        
        self._sgd_n_time = 0
        self.log_likelihood_history_ = []
        
        # Inference results
        self.filtered_mean = None
        self.filtered_cov = None
        self.smoother_mean = None
        self.smoother_cov = None

    def fit_sgd(
        self,
        design_matrix: Array, # API consistency
        spike_indicator: Array,
        key: Optional[Array] = None,
        num_steps: int = 100,
        **kwargs
    ) -> List[float]:
        self._sgd_n_time = spike_indicator.shape[0]
        # Important: pass design_matrix as first positional argument to super().fit_sgd
        # so it reaches _sgd_loss_fn(params, design_matrix, spikes)
        history = SGDFittableMixin.fit_sgd(
            self,
            design_matrix,
            spike_indicator, 
            key=self.key if key is None else key, 
            num_steps=num_steps, 
            **kwargs
        )
        self.log_likelihood_history_ = history
        return history

    @property
    def _n_timesteps(self) -> int:
        return self._sgd_n_time

    # --- SGD Contract Implementation ---

    def _build_param_spec(self) -> Tuple[Dict[str, Any], Dict[str, ParameterTransform]]:
        params = {
            "hamiltonian": self.hamiltonian_params,
            "C": self.C,
            "d": self.d,
            "init_mean": self.init_mean,
        }
        spec = {
            "hamiltonian": UNCONSTRAINED,
            "C": UNCONSTRAINED,
            "d": UNCONSTRAINED,
            "init_mean": UNCONSTRAINED,
        }
        return params, spec

    def _sgd_loss_fn(self, params: Dict[str, Any], design_matrix: Array, spikes: Array, **kwargs) -> Array:
        """Negative marginal log-likelihood via Laplace-EKF."""
        from state_space_practice.hamiltonian_point_process import hamiltonian_point_process_filter
        
        res = hamiltonian_point_process_filter(
            init_mean=params["init_mean"],
            init_cov=self.init_cov,
            process_cov=self.process_cov,
            hamiltonian_params=params["hamiltonian"],
            C=params["C"],
            d=params["d"],
            spike_indicator=spikes,
            dt=self.dt
        )
        return -res.marginal_log_likelihood

    def _store_sgd_params(self, params: Dict[str, Any]) -> None:
        self.hamiltonian_params = params["hamiltonian"]
        self.C = params["C"]
        self.d = params["d"]
        self.init_mean = params["init_mean"]

    def _finalize_sgd(self, design_matrix: Array, spikes: Array, **kwargs) -> None:
        """Run full filter/smoother after SGD."""
        res = self.filter(spikes)
        self.filtered_mean, self.filtered_cov = res.filtered_mean, res.filtered_cov
        self.smoother_mean, self.smoother_cov = self.smooth(res)

    def _check_sgd_initialized(self) -> None:
        if self._sgd_n_time == 0:
            raise RuntimeError("Model must be fitted with fit_sgd before use.")

    def filter(self, spike_indicator: Array) -> HamiltonianFilterResult:
        """Apply nonlinear point-process filter."""
        from state_space_practice.hamiltonian_point_process import hamiltonian_point_process_filter
        return hamiltonian_point_process_filter(
            init_mean=self.init_mean,
            init_cov=self.init_cov,
            process_cov=self.process_cov,
            hamiltonian_params=self.hamiltonian_params,
            C=self.C,
            d=self.d,
            spike_indicator=spike_indicator,
            dt=self.dt
        )

    def smooth(self, filter_result: HamiltonianFilterResult) -> Tuple[Array, Array]:
        """Apply nonlinear RTS smoother."""
        from state_space_practice.hamiltonian_point_process import hamiltonian_point_process_smoother
        return hamiltonian_point_process_smoother(
            filtered_mean=filter_result.filtered_mean,
            filtered_cov=filter_result.filtered_cov,
            linearization_matrices=filter_result.linearization_matrices,
            process_cov=self.process_cov
        )

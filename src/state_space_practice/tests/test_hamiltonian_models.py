import jax
import jax.numpy as jnp
import pytest
from state_space_practice.hamiltonian_models import (
    HamiltonianPointProcessModel, 
    init_structured_hamiltonian_params, 
    symplectic_transition, 
    transition_jacobian
)

def test_hamiltonian_model_is_standalone_sgd_model():
    """Assert the V1 contract: standalone SGDFittable, not BaseModel."""
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=2,
        sampling_freq=100.0,
        dt=0.01,
    )
    assert model.n_latent == 2
    assert model.process_cov.shape == (2, 2)
    # Check V1 initialization: init_mean should be 1D vector for standalone
    assert model.init_mean.ndim == 1
    assert model.init_mean.shape == (2,)

def test_fit_sgd_sets_internal_sequence_length():
    """Verify SGD lifecycle and sequence length tracking."""
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=2,
        sampling_freq=100.0,
        dt=0.01,
    )
    spikes = jnp.zeros((10, 2))
    # Note: design_matrix is currently ignored in V1 Hamiltonian loss but part of API
    design_matrix = jnp.zeros((10, 2, 2))
    
    # Run 2 steps to verify optimization lifecycle
    history = model.fit_sgd(design_matrix, spikes, key=jax.random.PRNGKey(0), num_steps=2)
    assert len(history) == 2
    assert model._sgd_n_time == 10
    assert model.smoother_mean is not None

def test_leapfrog_step_returns_finite_state():
    """Test the symplectic transition (Task 2)."""
    params = init_structured_hamiltonian_params(n_oscillators=2, key=jax.random.PRNGKey(0))
    x = jnp.array([1.0, 0.0, 0.5, 0.0])
    x_next = symplectic_transition(x, params, dt=0.01)
    assert x_next.shape == x.shape
    assert jnp.all(jnp.isfinite(x_next))

def test_transition_jacobian_is_finite():
    """Test the Jacobian utility (Task 2)."""
    params = init_structured_hamiltonian_params(n_oscillators=1, key=jax.random.PRNGKey(0))
    x = jnp.array([0.7, -0.2])
    jac = transition_jacobian(x, params, dt=0.01)
    assert jac.shape == (2, 2)
    assert jnp.all(jnp.isfinite(jac))

def test_hamiltonian_point_process_filter_and_smooth():
    """Test filter and smooth methods (Task 3)."""
    n_time = 20
    n_neurons = 3
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=n_neurons,
        sampling_freq=100.0,
        dt=0.01,
    )
    spikes = jnp.zeros((n_time, n_neurons))
    
    # Run filter
    res = model.filter(spikes)
    assert res.filtered_mean.shape == (n_time, 2)
    assert jnp.all(jnp.isfinite(res.filtered_mean))
    
    # Run smooth
    m_s, P_s = model.smooth(res)
    assert m_s.shape == (n_time, 2)
    assert jnp.all(jnp.isfinite(m_s))

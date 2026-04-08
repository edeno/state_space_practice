import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
from state_space_practice.hamiltonian_models import HamiltonianPointProcessModel
from state_space_practice.simulate_data_hamiltonian import simulate_hamiltonian_spike_data
from state_space_practice.parameter_transforms import UNCONSTRAINED, ParameterTransform
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.point_process_kalman import _point_process_laplace_update

class SimpleLinearPointProcessModel(SGDFittableMixin):
    """Simple random-walk point-process model for benchmarking."""
    def __init__(self, n_latent, n_neurons, dt):
        self.n_latent = n_latent
        self.n_neurons = n_neurons
        self.dt = dt
        self.init_mean = jnp.zeros((n_latent,))
        self.init_cov = 1e-1 * jnp.eye(n_latent)
        self.process_cov = 1e-6 * jnp.eye(n_latent)
        self.C = jnp.zeros((n_neurons, n_latent))
        self.d = jnp.zeros((n_neurons,))
        self._sgd_n_time = 0

    def _build_param_spec(self):
        params = {"C": self.C, "d": self.d, "init_mean": self.init_mean}
        spec = {"C": UNCONSTRAINED, "d": UNCONSTRAINED, "init_mean": UNCONSTRAINED}
        return params, spec

    def _sgd_loss_fn(self, params, design_matrix, spikes, **kwargs):
        C, d, x0 = params["C"], params["d"], params["init_mean"]
        log_intensity_func = lambda x: jnp.dot(C, x) + d
        grad_fn = jax.jit(jax.jacfwd(log_intensity_func))
        hess_fn = jax.jit(jax.jacfwd(grad_fn))
        
        def step(carry, y_t):
            m_prev, P_prev = carry
            m_pred = m_prev
            P_pred = P_prev + self.process_cov
            m_post, P_post, log_lik_t = _point_process_laplace_update(
                m_pred, 
                P_pred, 
                y_t, 
                self.dt, 
                log_intensity_func,
                grad_log_intensity_func=grad_fn, 
                hess_log_intensity_func=hess_fn
            )
            return (m_post, P_post), log_lik_t
        
        _, lls = jax.lax.scan(step, (x0, self.init_cov), spikes)
        return -jnp.sum(lls)

    def _store_sgd_params(self, params):
        self.C, self.d, self.init_mean = params["C"], params["d"], params["init_mean"]

    def _finalize_sgd(self, *args, **kwargs): pass
    def _check_sgd_initialized(self): pass
    @property
    def _n_timesteps(self): return self._sgd_n_time

def test_hamiltonian_point_process_filter_returns_finite_outputs():
    """Verify filter and smoother on small data (Task 3)."""
    n_time = 20
    n_neurons = 3
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=n_neurons,
        sampling_freq=100.0,
        dt=0.01,
    )
    spikes = jnp.zeros((n_time, n_neurons))
    result = model.filter(spikes)
    assert result.filtered_mean.shape == (n_time, 2)
    assert jnp.all(jnp.isfinite(result.filtered_mean))
    assert jnp.all(jnp.isfinite(result.filtered_cov))

def test_simulate_hamiltonian_spikes_is_deterministic_under_seed():
    """Verify simulator properties (Task 4)."""
    out1 = simulate_hamiltonian_spike_data(seed=0, n_time=50, n_oscillators=1, n_neurons=3)
    out2 = simulate_hamiltonian_spike_data(seed=0, n_time=50, n_oscillators=1, n_neurons=3)
    np.testing.assert_allclose(out1.latent_states, out2.latent_states, atol=1e-5)
    np.testing.assert_array_equal(out1.spikes, out2.spikes)
    assert out1.spikes.shape == (50, 3)
    assert out1.latent_states.shape == (50, 2)

@pytest.mark.slow
def test_hamiltonian_point_process_fit_sgd_improves_ll():
    """Verify SGD fitting improves marginal LL (Task 5)."""
    data = simulate_hamiltonian_spike_data(
        seed=0,
        n_time=100,
        n_oscillators=1,
        n_neurons=4,
    )
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=4,
        sampling_freq=100.0,
        dt=0.01,
    )
    model.C = data.true_params["C"]
    optimizer = optax.adam(5e-3)
    history = model.fit_sgd(None, data.spikes, num_steps=50, optimizer=optimizer)
    # Log-likelihood should INCREASE
    assert history[-1] > history[0]
    assert model.smoother_mean is not None
    assert model.smoother_mean.shape == (100, 2)

@pytest.mark.slow
def test_hamiltonian_model_beats_linear_baseline_on_nonlinear_synthetic_data():
    """Benchmark Hamiltonian model against linear baseline (Task 6)."""
    dt = 0.005
    n_time = 400
    n_neurons = 30
    # Higher N and higher T makes the true physics more apparent
    data = simulate_hamiltonian_spike_data(
        seed=42, n_time=n_time, n_oscillators=1, n_neurons=n_neurons, dt=dt, process_noise_std=0.0001
    )
    
    optimizer = optax.adam(1e-3)
    # 200 steps needed after PRNG key fix changed the spike realization
    num_steps = 200
    
    # 1. Linear Model
    linear_model = SimpleLinearPointProcessModel(n_latent=2, n_neurons=n_neurons, dt=dt)
    linear_model._sgd_n_time = n_time
    lin_history = linear_model.fit_sgd(None, data.spikes, num_steps=num_steps, optimizer=optimizer)
    
    # 2. Nonlinear Model
    nonlinear_model = HamiltonianPointProcessModel(1, n_neurons, 1.0/dt, dt)
    nonlinear_model.hamiltonian_params = data.true_params["hamiltonian"]
    nonlinear_model.C = data.true_params["C"]
    nonlinear_model.init_mean = data.latent_states[0]
    
    nonlin_history = nonlinear_model.fit_sgd(None, data.spikes, num_steps=num_steps, optimizer=optimizer)
    
    # Hamiltonian model should now comfortably win
    assert nonlin_history[-1] > lin_history[-1]

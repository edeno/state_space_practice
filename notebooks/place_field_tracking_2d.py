# %% [markdown]
# # 2D Place Field Tracking with Point-Process State-Space Model
#
# Track a drifting 2D place field using the Laplace-EKF point-process
# filter/smoother with EM for parameter estimation.
#
# Model:
# - Latent state x_t ∈ R^{n_basis}: time-varying GLM weights
# - Dynamics: x_t = A x_{t-1} + w_t,  w_t ~ N(0, Q)
# - Observation: y_t ~ Poisson(exp(Z_t @ x_t) * dt)
# - Z_t: 2D tensor-product B-spline basis evaluated at animal position

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from state_space_practice.point_process_kalman import (
    kalman_maximization_step,
    log_conditional_intensity,
    stochastic_point_process_smoother,
)
from state_space_practice.simulate_data import simulate_2d_moving_place_field

# %%
# --- Simulate data ---
data = simulate_2d_moving_place_field(
    total_time=600.0, dt=0.004, drift_speed=0.05,
    peak_rate=80.0, background_rate=2.0, n_basis_per_dim=5,
)

print(f"n_time: {len(data['time'])}")
print(f"n_basis: {data['design_matrix'].shape[1]}")
print(f"total spikes: {data['spikes'].sum()}")

# %%
# --- Set up model ---
n_time = len(data["time"])
n_basis = data["design_matrix"].shape[1]
dt = data["dt"]

# Design matrix: (n_time, 1, n_basis) for single neuron
# The filter expects (n_time, n_neurons, n_params) for multi-neuron,
# but for single neuron with default log_conditional_intensity(Z, x) = Z @ x,
# we need Z to be (n_basis,) per time step.
design_matrix = jnp.asarray(data["design_matrix"])  # (n_time, n_basis)

# Spikes: (n_time,)
spikes = jnp.asarray(data["spikes"])

# Initial state: start with small weights (near-uniform low rate)
init_mean = jnp.zeros(n_basis)
init_cov = jnp.eye(n_basis) * 1.0

# Dynamics: random walk with small process noise
A = jnp.eye(n_basis)
Q = jnp.eye(n_basis) * 1e-5

# %%
# --- Run EM ---
# For the random-walk place field model, we fix A=I and learn only Q (isotropic)
# and initial conditions.  This avoids instability from unconstrained A.
n_em_iter = 5
ll_history = []

for em_iter in range(n_em_iter):
    # E-step: smoother
    smoother_mean, smoother_cov, smoother_cross_cov, marginal_ll = (
        stochastic_point_process_smoother(
            init_mean,
            init_cov,
            design_matrix,
            spikes,
            dt,
            A,
            Q,
            log_conditional_intensity,
        )
    )

    ll_val = float(marginal_ll)
    ll_history.append(ll_val)
    print(f"EM iter {em_iter}: marginal LL = {ll_val:.1f}")

    # M-step: update Q (isotropic) and initial conditions, keeping A = I
    # Q = (1/(T-1)) * sum_t [E[x_t x_t'] - E[x_t x_{t-1}'] - E[x_{t-1} x_t'] + E[x_{t-1} x_{t-1}'] ]
    # With A=I this simplifies to the variance of the state increments.
    _, Q_new, init_mean_new, init_cov_new = kalman_maximization_step(
        smoother_mean, smoother_cov, smoother_cross_cov
    )
    # Diagonal Q, clamped to positive
    q_diag = jnp.maximum(jnp.diag(Q_new), 1e-10)
    Q = jnp.diag(q_diag)
    init_mean = init_mean_new
    init_cov = init_cov_new

# %%
# --- Reconstruct estimated firing rate ---
# log(lambda_t) = Z_t @ x_t  (smoothed weights)
log_rate_estimated = jnp.sum(design_matrix * smoother_mean, axis=1)
rate_estimated = jnp.exp(log_rate_estimated)

# %%
# --- Plot results ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. True vs estimated firing rate (time series, first 10s)
t_plot = 10.0  # seconds to plot
n_plot = int(t_plot / dt)
ax = axes[0, 0]
ax.plot(data["time"][:n_plot], data["true_rate"][:n_plot], "k-", alpha=0.5, label="True rate")
ax.plot(data["time"][:n_plot], np.array(rate_estimated[:n_plot]), "r-", alpha=0.7, label="Estimated rate")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("True vs Estimated Rate (first 10s)")
ax.legend()

# 2. True place field at session start
ax = axes[0, 1]
arena = data["position"]
n_grid = 50
x_grid = np.linspace(0, 100, n_grid)
y_grid = np.linspace(0, 100, n_grid)
xx, yy = np.meshgrid(x_grid, y_grid)

# True rate at start: background + (peak - background) * exp(-d^2 / 2sigma^2)
peak_rate = 80.0
background_rate = 2.0
pf_sigma = 12.0
center_start = data["true_center"][0]
dist_sq = (xx - center_start[0]) ** 2 + (yy - center_start[1]) ** 2
true_field_start = background_rate + (peak_rate - background_rate) * np.exp(-dist_sq / (2 * pf_sigma**2))
im = ax.pcolormesh(x_grid, y_grid, true_field_start, cmap="hot")
ax.set_title("True Place Field (t=0)")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
plt.colorbar(im, ax=ax, label="Rate (Hz)")

# 3. True place field at session end
ax = axes[0, 2]
center_end = data["true_center"][-1]
dist_sq = (xx - center_end[0]) ** 2 + (yy - center_end[1]) ** 2
true_field_end = background_rate + (peak_rate - background_rate) * np.exp(-dist_sq / (2 * pf_sigma**2))
im = ax.pcolormesh(x_grid, y_grid, true_field_end, cmap="hot")
ax.set_title("True Place Field (t=end)")
ax.set_xlabel("x (cm)")
plt.colorbar(im, ax=ax, label="Rate (Hz)")

# 4. Estimated place field at session start (from smoothed weights)
ax = axes[1, 0]
from patsy import dmatrix

grid_positions = np.column_stack([xx.ravel(), yy.ravel()])
Z_grid = np.asarray(
    dmatrix(
        "te(bs(x, df=5), bs(y, df=5)) - 1",
        {"x": grid_positions[:, 0], "y": grid_positions[:, 1]},
    )
)
# Use weights from early in session (average first 1000 steps)
weights_start = np.array(smoother_mean[:1000].mean(axis=0))
log_rate_grid = Z_grid @ weights_start
rate_grid = np.exp(log_rate_grid).reshape(n_grid, n_grid)
im = ax.pcolormesh(x_grid, y_grid, rate_grid, cmap="hot")
ax.set_title("Estimated Place Field (t=0)")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
plt.colorbar(im, ax=ax, label="Rate (Hz)")

# 5. Estimated place field at session end
ax = axes[1, 1]
weights_end = np.array(smoother_mean[-1000:].mean(axis=0))
log_rate_grid = Z_grid @ weights_end
rate_grid = np.exp(log_rate_grid).reshape(n_grid, n_grid)
im = ax.pcolormesh(x_grid, y_grid, rate_grid, cmap="hot")
ax.set_title("Estimated Place Field (t=end)")
ax.set_xlabel("x (cm)")
plt.colorbar(im, ax=ax, label="Rate (Hz)")

# 6. Place field center trajectory
ax = axes[1, 2]
# Estimate center over time by finding peak of estimated field in blocks
# Use a finer grid for center estimation
n_grid_fine = 100
x_fine = np.linspace(0, 100, n_grid_fine)
y_fine = np.linspace(0, 100, n_grid_fine)
xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
grid_fine = np.column_stack([xx_fine.ravel(), yy_fine.ravel()])
Z_grid_fine = np.asarray(
    dmatrix(
        "te(bs(x, df=5), bs(y, df=5)) - 1",
        {"x": grid_fine[:, 0], "y": grid_fine[:, 1]},
    )
)

n_blocks = 20
block_size = n_time // n_blocks
estimated_centers = []
for i in range(n_blocks):
    w_block = np.array(smoother_mean[i * block_size : (i + 1) * block_size].mean(axis=0))
    log_r = Z_grid_fine @ w_block
    peak_idx = np.argmax(log_r)
    estimated_centers.append(grid_fine[peak_idx])
estimated_centers = np.array(estimated_centers)

true_center_blocks = np.array(
    [data["true_center"][i * block_size : (i + 1) * block_size].mean(axis=0) for i in range(n_blocks)]
)

ax.plot(true_center_blocks[:, 0], true_center_blocks[:, 1], "k-o", label="True center", markersize=4)
ax.plot(estimated_centers[:, 0], estimated_centers[:, 1], "r-s", label="Estimated center", markersize=4)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title("Place Field Center Drift")
ax.legend()
ax.set_aspect("equal")
# Set limits around the data
all_centers = np.vstack([true_center_blocks, estimated_centers])
margin = 10
ax.set_xlim(all_centers[:, 0].min() - margin, all_centers[:, 0].max() + margin)
ax.set_ylim(all_centers[:, 1].min() - margin, all_centers[:, 1].max() + margin)

plt.tight_layout()
plt.savefig("output/place_field_tracking_2d.png", dpi=150)
plt.show()
print("Done!")

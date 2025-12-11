"""Eden & Brown 2004 Point Process Filter with Spline Basis Functions.

This script demonstrates tracking a place field that moves over time using:
- B-spline basis functions for position (instead of parametric Gaussian)
- Point process filter/smoother from the library

The state θ_t represents the spline coefficients at each time, which evolve
via a random walk. This allows the firing rate as a function of position
to change smoothly over time.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix

from state_space_practice.point_process_kalman import PointProcessModel

# --- Simulation ---


def simulate_moving_place_field(
    dt: float = 0.020,
    total_time: float = 2000.0,
    track_length: float = 300.0,
    speed: float = 125.0,
    max_rate: float = 20.0,
    place_field_width: float = 20.0,
    center_start: float = 100.0,
    center_end: float = 200.0,
    seed: int = 42,
):
    """Simulate spikes from a place field that drifts over time.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    total_time : float
        Total simulation time in seconds.
    track_length : float
        Length of the linear track in cm.
    speed : float
        Running speed in cm/s.
    max_rate : float
        Peak firing rate in Hz.
    place_field_width : float
        Standard deviation of the Gaussian place field in cm.
    center_start : float
        Initial place field center in cm.
    center_end : float
        Final place field center in cm.
    seed : int
        Random seed.

    Returns
    -------
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time,)
    spike_indicator : ndarray, shape (n_time,)
    true_centers : ndarray, shape (n_time,)
        The true place field center at each time.
    true_rate : ndarray, shape (n_time,)
        The true firing rate at each time.
    """
    np.random.seed(seed)

    n_time = int(total_time / dt)
    time = np.arange(n_time) * dt

    # Generate position: animal runs back and forth
    run_out = np.arange(0, track_length, speed * dt)
    run_back = np.arange(track_length, 0, -speed * dt)
    run = np.concatenate([run_out, run_back])
    n_repeats = int(np.ceil(n_time / len(run)))
    position = np.tile(run, n_repeats)[:n_time]

    # Place field center drifts linearly over time
    true_centers = np.linspace(center_start, center_end, n_time)

    # Compute true firing rate
    true_rate = max_rate * np.exp(
        -((position - true_centers) ** 2) / (2 * place_field_width**2)
    )

    # Generate spikes
    spike_indicator = np.random.poisson(true_rate * dt)

    return time, position, spike_indicator, true_centers, true_rate, dt


# --- Design Matrix ---


def create_spline_design_matrix(
    position: np.ndarray, n_basis: int = 10, degree: int = 3
):
    """Create B-spline design matrix for position.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
        Position values.
    n_basis : int
        Number of basis functions.
    degree : int
        Degree of the spline (3 = cubic).

    Returns
    -------
    design_matrix : ndarray, shape (n_time, n_basis)
    """
    formula = f"bs(x, df={n_basis}, degree={degree}, include_intercept=True) - 1"
    design_matrix = dmatrix(formula, {"x": position})
    return np.asarray(design_matrix), formula


# --- Visualization ---


def plot_raster(time, position, spike_indicator, ax=None):
    """Plot spike raster colored by position."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    spike_times = time[spike_indicator > 0]
    spike_positions = position[spike_indicator > 0]
    ax.scatter(spike_times, spike_positions, s=1, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (cm)")
    ax.set_title("Spike Raster")
    return ax


def plot_rate_heatmap(
    time,
    position,
    rate_estimate,
    design_matrix,
    dt,
    ax=None,
    title="Estimated Firing Rate",
    vmax=None,
):
    """Plot firing rate as function of position over time.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time,)
    rate_estimate : ndarray, shape (n_time, n_basis)
        Spline coefficients over time.
    design_matrix : ndarray, shape (n_time, n_basis)
    dt : float
    ax : matplotlib axis
    title : str
    vmax : float
        Maximum value for colorbar.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    # Create grid for visualization
    pos_grid = np.linspace(position.min(), position.max(), 100)
    formula = (
        f"bs(x, df={design_matrix.shape[1]}, degree=3, include_intercept=True) - 1"
    )
    design_grid = np.asarray(dmatrix(formula, {"x": pos_grid}))

    # Compute rate at each position for each time
    # rate_estimate: (n_time, n_basis), design_grid: (n_pos, n_basis)
    log_rate = rate_estimate @ design_grid.T  # (n_time, n_pos)
    rate = np.exp(log_rate) / dt  # Convert to Hz

    # Subsample time for visualization
    time_subsample = 100
    t_idx = np.arange(0, len(time), time_subsample)

    t_grid, p_grid = np.meshgrid(time[t_idx], pos_grid)
    im = ax.pcolormesh(
        t_grid, p_grid, rate[t_idx].T, shading="auto", vmin=0, vmax=vmax, cmap="viridis"
    )
    plt.colorbar(im, ax=ax, label="Firing Rate (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (cm)")
    ax.set_title(title)
    return ax


def plot_place_field_snapshots(
    time,
    position,
    rate_estimate,
    design_matrix,
    true_centers,
    dt,
    n_snapshots: int = 5,
):
    """Plot place field at different time points."""
    fig, axes = plt.subplots(1, n_snapshots, figsize=(15, 3), sharey=True)

    pos_grid = np.linspace(position.min(), position.max(), 100)
    formula = (
        f"bs(x, df={design_matrix.shape[1]}, degree=3, include_intercept=True) - 1"
    )
    design_grid = np.asarray(dmatrix(formula, {"x": pos_grid}))

    snapshot_times = np.linspace(0, len(time) - 1, n_snapshots).astype(int)

    for ax, t_idx in zip(axes, snapshot_times):
        # Estimated rate
        log_rate = rate_estimate[t_idx] @ design_grid.T
        rate = np.exp(log_rate) / dt

        ax.plot(pos_grid, rate, "b-", label="Estimated")
        ax.axvline(true_centers[t_idx], color="r", linestyle="--", label="True center")
        ax.set_xlabel("Position (cm)")
        ax.set_title(f"t = {time[t_idx]:.0f}s")
        ax.set_ylim(0, None)

    axes[0].set_ylabel("Firing Rate (Hz)")
    axes[0].legend()
    fig.suptitle("Place Field Evolution")
    plt.tight_layout()
    return fig, axes


def plot_center_tracking(time, true_centers, estimated_centers, ci=None, ax=None):
    """Plot true vs estimated place field center over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(time, true_centers, "r--", label="True center", linewidth=2)
    ax.plot(time, estimated_centers, "b-", label="Estimated center", alpha=0.8)

    if ci is not None:
        ax.fill_between(time, ci[:, 0], ci[:, 1], alpha=0.3, color="b")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (cm)")
    ax.set_title("Place Field Center Tracking")
    ax.legend()
    return ax


def estimate_center_from_splines(rate_estimate, design_matrix, position, dt):
    """Estimate place field center as the position of maximum rate.

    Parameters
    ----------
    rate_estimate : ndarray, shape (n_time, n_basis)
    design_matrix : ndarray, shape (n_time, n_basis)
    position : ndarray, shape (n_time,)
    dt : float

    Returns
    -------
    centers : ndarray, shape (n_time,)
    """
    pos_grid = np.linspace(position.min(), position.max(), 200)
    formula = (
        f"bs(x, df={design_matrix.shape[1]}, degree=3, include_intercept=True) - 1"
    )
    design_grid = np.asarray(dmatrix(formula, {"x": pos_grid}))

    log_rate = rate_estimate @ design_grid.T  # (n_time, n_pos)
    centers = pos_grid[np.argmax(log_rate, axis=1)]
    return centers


def compute_rate_uncertainty(
    mean_estimate, cov_estimate, design_matrix, position, dt, n_std: float = 2.0
):
    """Compute confidence intervals for firing rate using the delta method.

    For log_rate = Z @ θ, the variance of log_rate is Z @ Σ @ Z.T
    Then rate = exp(log_rate) / dt, so by delta method:
        Var(rate) ≈ (d rate / d log_rate)^2 * Var(log_rate)
                  = rate^2 * Var(log_rate)

    Parameters
    ----------
    mean_estimate : ndarray, shape (n_time, n_basis)
        Posterior mean of spline coefficients.
    cov_estimate : ndarray, shape (n_time, n_basis, n_basis)
        Posterior covariance of spline coefficients.
    design_matrix : ndarray, shape (n_time, n_basis)
    position : ndarray, shape (n_time,)
    dt : float
    n_std : float
        Number of standard deviations for CI.

    Returns
    -------
    rate_mean : ndarray, shape (n_time, n_pos)
        Mean firing rate at each position.
    rate_lower : ndarray, shape (n_time, n_pos)
        Lower CI bound.
    rate_upper : ndarray, shape (n_time, n_pos)
        Upper CI bound.
    pos_grid : ndarray, shape (n_pos,)
        Position grid.
    """
    pos_grid = np.linspace(position.min(), position.max(), 100)
    formula = (
        f"bs(x, df={design_matrix.shape[1]}, degree=3, include_intercept=True) - 1"
    )
    design_grid = np.asarray(dmatrix(formula, {"x": pos_grid}))  # (n_pos, n_basis)

    # Mean log rate: (n_time, n_pos)
    log_rate_mean = mean_estimate @ design_grid.T

    # Variance of log rate at each position: Z @ Σ @ Z.T
    # For each time t: var_log_rate[t, p] = design_grid[p] @ cov[t] @ design_grid[p].T
    # Vectorized: (n_time, n_pos)
    log_rate_var = np.einsum("pb,tbc,pc->tp", design_grid, cov_estimate, design_grid)

    log_rate_std = np.sqrt(log_rate_var)

    # CI on log scale
    log_rate_lower = log_rate_mean - n_std * log_rate_std
    log_rate_upper = log_rate_mean + n_std * log_rate_std

    # Transform to rate scale
    rate_mean = np.exp(log_rate_mean) / dt
    rate_lower = np.exp(log_rate_lower) / dt
    rate_upper = np.exp(log_rate_upper) / dt

    return rate_mean, rate_lower, rate_upper, pos_grid


def estimate_center_with_uncertainty(
    mean_estimate, cov_estimate, design_matrix, position, n_samples: int = 100
):
    """Estimate place field center and uncertainty via sampling.

    Since center = argmax(rate) is not differentiable, we sample from the
    posterior and compute the center for each sample.

    Parameters
    ----------
    mean_estimate : ndarray, shape (n_time, n_basis)
    cov_estimate : ndarray, shape (n_time, n_basis, n_basis)
    design_matrix : ndarray, shape (n_time, n_basis)
    position : ndarray, shape (n_time,)
    n_samples : int
        Number of posterior samples.

    Returns
    -------
    center_mean : ndarray, shape (n_time,)
    center_std : ndarray, shape (n_time,)
    center_lower : ndarray, shape (n_time,)
        2.5th percentile.
    center_upper : ndarray, shape (n_time,)
        97.5th percentile.
    """
    pos_grid = np.linspace(position.min(), position.max(), 200)
    formula = (
        f"bs(x, df={design_matrix.shape[1]}, degree=3, include_intercept=True) - 1"
    )
    design_grid = np.asarray(dmatrix(formula, {"x": pos_grid}))

    n_time = mean_estimate.shape[0]
    centers_samples = np.zeros((n_time, n_samples))

    for i in range(n_samples):
        # Sample θ from posterior at each time
        # θ_sample[t] ~ N(mean[t], cov[t])
        theta_samples = np.array(
            [
                np.random.multivariate_normal(mean_estimate[t], cov_estimate[t])
                for t in range(n_time)
            ]
        )
        # Compute log rate and find argmax
        log_rate = theta_samples @ design_grid.T
        centers_samples[:, i] = pos_grid[np.argmax(log_rate, axis=1)]

    center_mean = np.mean(centers_samples, axis=1)
    center_std = np.std(centers_samples, axis=1)
    center_lower = np.percentile(centers_samples, 2.5, axis=1)
    center_upper = np.percentile(centers_samples, 97.5, axis=1)

    return center_mean, center_std, center_lower, center_upper


def plot_place_field_with_uncertainty(
    time,
    rate_mean,
    rate_lower,
    rate_upper,
    pos_grid,
    true_centers,
    n_snapshots: int = 5,
):
    """Plot place field snapshots with uncertainty bands."""
    fig, axes = plt.subplots(1, n_snapshots, figsize=(15, 3), sharey=True)

    snapshot_times = np.linspace(0, len(time) - 1, n_snapshots).astype(int)

    for ax, t_idx in zip(axes, snapshot_times):
        ax.fill_between(
            pos_grid,
            rate_lower[t_idx],
            rate_upper[t_idx],
            alpha=0.3,
            color="b",
            label="95% CI",
        )
        ax.plot(pos_grid, rate_mean[t_idx], "b-", label="Estimated")
        ax.axvline(true_centers[t_idx], color="r", linestyle="--", label="True center")
        ax.set_xlabel("Position (cm)")
        ax.set_title(f"t = {time[t_idx]:.0f}s")
        ax.set_ylim(0, None)

    axes[0].set_ylabel("Firing Rate (Hz)")
    axes[0].legend()
    fig.suptitle("Place Field Evolution with Uncertainty")
    plt.tight_layout()
    return fig, axes


# --- Main ---


def main():
    # Simulate data
    print("Simulating moving place field...")
    time, position, spike_indicator, true_centers, true_rate, dt = (
        simulate_moving_place_field(
            total_time=2000.0,
            center_start=100.0,
            center_end=200.0,
            max_rate=30.0,
            seed=42,
        )
    )
    print(f"  Total spikes: {spike_indicator.sum()}")
    print(f"  Mean rate: {spike_indicator.sum() / (len(time) * dt):.2f} Hz")

    # Create design matrix
    n_basis = 10
    design_matrix, formula = create_spline_design_matrix(position, n_basis=n_basis)
    design_matrix_jax = jnp.asarray(design_matrix)
    print(f"  Design matrix shape: {design_matrix.shape}")

    # Set up model
    n_params = design_matrix.shape[1]
    model = PointProcessModel(
        n_state_dims=n_params,
        dt=dt,
        transition_matrix=jnp.eye(n_params),  # Random walk
        process_cov=jnp.eye(n_params) * 1e-4,  # Initial process covariance
        init_mean=jnp.zeros(n_params),
        init_cov=jnp.eye(n_params) * 1.0,
        # Learn process covariance via EM
        update_transition_matrix=False,  # Keep random walk structure
        update_process_cov=True,  # Learn optimal smoothness
        update_init_state=True,  # Learn initial state
    )

    # Run EM to learn parameters
    print("Running EM algorithm...")
    log_likelihoods = model.fit(
        design_matrix=design_matrix_jax,
        spike_indicator=jnp.asarray(spike_indicator),
        max_iter=20,
        tolerance=1e-4,
    )
    print(f"  Initial log-likelihood: {log_likelihoods[0]:.2f}")
    print(f"  Final log-likelihood: {log_likelihoods[-1]:.2f}")
    print(f"  Iterations: {len(log_likelihoods)}")

    # Extract results from model
    filtered_mean = np.asarray(model.filtered_mean)
    filtered_cov = np.asarray(model.filtered_cov)
    smoothed_mean = np.asarray(model.smoother_mean)
    smoothed_cov = np.asarray(model.smoother_cov)

    # Estimate centers (point estimates)
    filtered_centers = estimate_center_from_splines(
        filtered_mean, design_matrix, position, dt
    )
    smoothed_centers = estimate_center_from_splines(
        smoothed_mean, design_matrix, position, dt
    )

    # Compute rate uncertainty for smoothed estimates
    print("Computing rate uncertainty...")
    rate_mean, rate_lower, rate_upper, pos_grid = compute_rate_uncertainty(
        smoothed_mean, smoothed_cov, design_matrix, position, dt, n_std=2.0
    )

    # Compute center uncertainty via sampling (subsampled for speed)
    print("Computing center uncertainty (sampling)...")
    subsample = 100  # Subsample time for faster computation
    center_mean, center_std, center_lower, center_upper = estimate_center_with_uncertainty(
        smoothed_mean[::subsample],
        smoothed_cov[::subsample],
        design_matrix,
        position,
        n_samples=200,
    )
    time_subsampled = time[::subsample]

    # --- Plotting ---
    print("Generating plots...")

    # Figure 1: Raster
    fig1, ax1 = plt.subplots(figsize=(12, 3))
    plot_raster(time, position, spike_indicator, ax=ax1)

    # Figure 2: Rate heatmaps
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    vmax = 40  # Common scale
    plot_rate_heatmap(
        time,
        position,
        filtered_mean,
        design_matrix,
        dt,
        ax=axes2[0],
        title="Filtered Rate Estimate",
        vmax=vmax,
    )
    plot_rate_heatmap(
        time,
        position,
        smoothed_mean,
        design_matrix,
        dt,
        ax=axes2[1],
        title="Smoothed Rate Estimate",
        vmax=vmax,
    )
    plt.tight_layout()

    # Figure 3: Place field snapshots with uncertainty
    fig3, _ = plot_place_field_with_uncertainty(
        time,
        rate_mean,
        rate_lower,
        rate_upper,
        pos_grid,
        true_centers,
        n_snapshots=5,
    )

    # Figure 4: Center tracking with uncertainty
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(time, true_centers, "r--", label="True center", linewidth=2)
    ax4.plot(time, smoothed_centers, "b-", label="Smoothed estimate", alpha=0.8)
    ax4.fill_between(
        time_subsampled,
        center_lower,
        center_upper,
        alpha=0.3,
        color="b",
        label="95% CI (sampled)",
    )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Position (cm)")
    ax4.set_title("Place Field Center Tracking with Uncertainty")
    ax4.legend()
    plt.tight_layout()

    # Print tracking error
    filter_rmse = np.sqrt(np.mean((filtered_centers - true_centers) ** 2))
    smoother_rmse = np.sqrt(np.mean((smoothed_centers - true_centers) ** 2))
    print(f"\nCenter tracking RMSE:")
    print(f"  Filter:   {filter_rmse:.2f} cm")
    print(f"  Smoother: {smoother_rmse:.2f} cm")

    # Save figures
    fig1.savefig("notebooks/spline_raster.png", dpi=150, bbox_inches="tight")
    fig2.savefig("notebooks/spline_rate_heatmap.png", dpi=150, bbox_inches="tight")
    fig3.savefig("notebooks/spline_place_field_uncertainty.png", dpi=150, bbox_inches="tight")
    fig4.savefig("notebooks/spline_center_tracking.png", dpi=150, bbox_inches="tight")
    print("\nFigures saved to notebooks/")

    plt.show()


if __name__ == "__main__":
    main()

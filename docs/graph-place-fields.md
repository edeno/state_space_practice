# Drifting graph place fields

`GraphPlaceFieldModel.fit()` estimates each neuron's drift scale from training
observations using joint-trajectory Poisson Laplace evidence, an explicit static
candidate, positive-scale profiles, and bounded L-BFGS with full-sequence gradients.
Install the pinned `spatial` extra and enable x64 before importing the package.

This runnable example uses constructor and fitting defaults, including the full
basis of a small environment:

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from neurospatial import Environment
from state_space_practice.graph_place_field import GraphPlaceFieldModel

rng = np.random.default_rng(52)
env = Environment.from_samples(np.linspace(0, 10, 201)[:, None], bin_size=2.0)
times = np.arange(300) * 0.1
positions = env.bin_centers[rng.integers(env.n_bins, size=len(times))]
spikes = rng.poisson(np.exp(np.linspace(-1, 2, len(times))))
model = GraphPlaceFieldModel(env, dt=0.1)
model.fit(times, positions, spikes)
log_rate = model.predict_log_rate_trajectory()  # neuron x time x active bin
print(model.drift_scale, model.field_drift_scale_, model.converged_)
```

For a larger environment, specify a manageable rank, such as `rank=8`; rank is a
spatial resolution choice. The validation benchmark uses rank 4 on small linear
tracks and rank 8 on branching and analytic-field tracks. It does not establish
adequate performance for every rank or geometry.

For a supplied scale, use
`GraphPlaceFieldModel(env, dt, update_drift_scale=False, init_drift_scale=q)`.
`q=0` gives a reduced static-state solve with compatible evidence constants.
Other enabled parameters still learn. Also set `update_kappa2=False` to hold
the entire positive Q covariance fixed. Setting every `update_*` flag to False
performs inference at the supplied parameters. `fit_mle()` is the explicit alias
for the normal estimator. `fit_em()` (or `fit(method="em")`) and `fit_sgd()`
retain experimental alternatives; `inference_method="sequential"` retains the
older local Laplace filter/RTS approximation. These alternatives do not carry
the automatic estimator's validation guarantees.

The coefficient model remains `P0=tau2*diag(S)`, `Q_c=q_c*diag(S)`, with
`S=(kappa2+lambda)**(-alpha)`. Stored q and tau2 retain coefficient-variance units.
`field_drift_scale_ = q * sum(S) / n_active_bins` is the mean active-bin log-rate
increment variance **per transition**; divide by dt for field diffusion per
second. When kappa2 is fitted, raw q and shape can trade off. The field variance
is easier to interpret across shapes; it does not identify a biological motion
speed. Changing dt does not automatically rescale a supplied q.

The normal estimator fits component baseline prior means and keeps all spatial
contrast prior means zero. Fitting every prior mean together with its initial
variance can collapse that variance; `initial_mean_mode="full"` is experimental.
`warm_start=True` initializes from an aggregated static map and resets q before
profiling. `warm_start=False` resumes compatible stored parameters and still
profiles drift. Failed refits clear fitted outputs and diagnostics.

For optimized fits, check `converged_`, `optimizer_result_.gradient_norm`, `parameter_bound_hits_`,
and `smoother_diagnostics_`. Inference with every parameter fixed has no optimizer
result. The optimizer threshold is 1e-7 per time row in
transformed field-variance coordinates, with a budget of 500 per nuisance solve;
a small relative evidence change alone does not establish convergence. Joint
Newton inference has a 50-step cap and 1e-8 remaining-update tolerance;
unconverged evidence is rejected. Profiles use 49 log-grid points, refine sampled
basins, and alternate three times with nuisance fitting. An all-static candidate
also refits its nuisance parameters. This is a finite, deterministic search,
not a guarantee of a global optimum for arbitrary datasets.

`drift_profiles_` holds **conditional** profiles at final nuisance parameters,
in field-variance units. Compare the chosen scale with these profiles, inspect
endpoints and near-equivalent solutions, and expand bounds when appropriate.
A flat profile or small fitted scale can reflect insufficient information. These
profiles are not joint confidence intervals. Positive q bounds map coefficient
bounds [1e-8, 10] using the constructor's spectral shape. Numerical bounds on
initial field variance [1e-8, 1e4], kappa2 [1e-6, 1e4*(1+max retained eigenvalue)]
and mean coefficients [-200, 200] keep searches finite; bound hits are reported.

Histories contain nuisance rounds, competing static fits and line-search trials,
so they can decrease. The last entry and `log_likelihood_` describe the selected
fit; `n_iter_` is None for gradient fits. `score()` recomputes the selected
approximate sequence evidence from the fitted initial prior. It does not continue
from the training posterior or score a causal forecast. Posterior trajectories
and the missing-block benchmark are retrospective reconstructions.

Out-of-bounds positions skip observations while retaining every transition on the
full time grid. Mask held-out counts too, and use training rows only in fitting.
The first row conditions directly on P0. Neurons have independent latent paths,
with shared fitted spectral shape and initial amplitude. Joint inference uses
ordinary Poisson observations; `max_firing_rate_hz` clamps only the causal
sequential filter outputs, which are retained separately as `filtered_mean` and
`filtered_cov`. Those causal outputs differ from joint smoothed trajectories.

See the [execution and acceptance plan](plans/2026-07-16-drifting-graph-gp-place-field.md#remaining-work-before-merge-automatic-drift-scale-estimation-2026-10-06)
and [reproducible validation](validation/graph-place-field-learning/README.md).

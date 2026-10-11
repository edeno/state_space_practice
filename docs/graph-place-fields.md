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
log_rate = model.predict_log_rate_trajectory()  # neuron 0: (n_time, n_bins)
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
performs inference at fixed parameters, but with the default `warm_start=True`
the prior mean `init_mean` is first set to the static GLM MAP of the training
data, so the fixed prior mean is data-derived. To infer at a fully supplied
prior, set `model.init_mean` (shape `(n_neurons, rank)`) and fit with
`warm_start=False`. `fit_mle()` is the explicit alias for the normal estimator.
`fit_em()` (or `fit(method="em")`) and `fit_sgd()` retain experimental
alternatives; `inference_method="sequential"` retains the older local Laplace
filter/RTS approximation. These alternatives do not carry
the automatic estimator's validation guarantees. `fit(verbose=...)` affects only
`method="em"` (per-iteration log messages); the normal estimator ignores it.

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
ordinary Poisson observations and does not run the causal filter, which no part
of joint fitting or prediction uses. `inference_method="sequential"` runs it and
keeps its past-only estimates as `filtered_mean` and `filtered_cov`;
`max_firing_rate_hz` clamps only that filter.

## GPU

The joint smoother is a sequential scan over time by default. On a GPU each of
its tiny per-step factorizations is a separate kernel launch, so a single neuron
runs several times slower than on a CPU. `parallel=True` (the default on a
GPU/TPU, `parallel=None`) runs each linear pass by associative scans instead:
on an A100 this was 49-129x faster for one neuron (3,000-70,930 time bins,
rank 20) and 5-20x for 8-32 neurons at once, with the same fit; a full `fit()`
of 8 neurons over 3,000 bins took 165 s instead of 2,065 s. On a CPU the
parallel pass is 1.7-4x slower, so `None` keeps the sequential pass there.

The parallel pass needs three to four times the device memory with gradients:
about 12-17 GiB per neuron at 70,930 bins and rank 20 (sequential: about 4
GiB), scaling with `n_time * rank**2`. Set `neuron_batch_size` to infer at most
that many neurons at once when all neurons do not fit. Gradients then recompute
each chunk rather than storing it, which cost 17-34 % more per gradient on an
A100 during `fit()` (and about 2x under `fit_sgd` with learnable drift):

```python
model = GraphPlaceFieldModel(env, dt=0.02, rank=20, neuron_batch_size=4)
```

For large recordings, also set `store_covariances=False`. By default the fit
keeps the per-time posterior covariances (`smoother_cov`, `smoother_cross_cov`,
and under sequential inference `filtered_cov`), each
`(n_neurons, n_time, rank, rank)`: about 0.6 GB per
neuron and array for an hour of 20 ms bins at rank 20. Without them the model
keeps only the means and small per-neuron summaries, fitting and EM give the
same results, and `predict_rate_map` recomputes the requested neuron's
covariances. With `neuron_batch_size` set, the final inference also runs in
blocks of that many neurons and keeps finished blocks in host memory.

See the [reproducible validation](validation/graph-place-field-learning/README.md).

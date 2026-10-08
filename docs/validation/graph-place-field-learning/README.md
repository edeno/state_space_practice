# Automatic graph drift-scale validation

The protocol was frozen after pilot seeds 0/1 and before independent final seeds
100--119. The [plan](../../plans/2026-07-16-drifting-graph-gp-place-field.md#execution-decisions-and-frozen-protocol-2026-10-08)
records settings, thresholds, and the pilot refinement from coefficient q to mean
field-increment variance when spatial shape is also fitted.

Run from the repository root, with the pinned spatial/test extras, x64 enabled
by each script, and without substituting an editable neurospatial checkout:

```bash
uv run --no-sync python notebooks/graph_place_field_joint_reference.py
uv run --no-sync python notebooks/graph_place_field_estimation_validation.py \
    --section optimizer --inference joint \
    --output-dir docs/validation/graph-place-field-estimation
uv run --no-sync python notebooks/graph_place_field_learning_validation.py \
    --section matched --output-dir docs/validation/graph-place-field-learning
uv run --no-sync python notebooks/graph_place_field_learning_validation.py \
    --section application --output-dir docs/validation/graph-place-field-learning
```

`joint_reference.json` compares fifteen small spatial q-profile points against
an independently constructed dense whitened Gaussian prior, Poisson posterior,
Hessian, covariance and normalized Laplace evidence. Joint evidence agrees to
6.2e-9 nats and modes to 1.5e-8 posterior SD. Converged sequential local updates
have errors up to 0.925 nats. A static rare-count counterexample changes sequential
evidence by 2.36 nats when count order reverses; joint evidence is invariant.
These results verify the approximation implemented, not exact posterior accuracy.
Existing scalar Bayes-integral artifacts retain the independent exact-posterior
comparison and its approximation limits.

`../graph-place-field-estimation/optimizer_joint.json` contains the twenty original
scalar datasets, three starts, and independent zero/positive reference searches
with expanded ranges and meshes. Profile-initialized L-BFGS meets the 0.001-nat
gate in **60/60 fits**, with maximum gap 5.7e-14 and projected-gradient convergence
in every fit. Plain L-BFGS also passes here (maximum gap 2.3e-6). The joint-objective Adam comparison (`../graph-place-field-estimation/sgd_joint.json`)
uses the same starts and expanded zero/positive profile. Adam reaches the 0.001-nat
gate in 10/60 fits at 200 steps and 36/60 at 1000 steps (worst gaps 5.57 and
0.148 nats). It remains an experimental alternative. The sequential optimizer
report and historical Adam report remain separate, with their inference settings
recorded. This scalar result does not prove joint spatial identification.

`matched.json` reports known-nuisance and jointly learned fits on three informative
positive conditions, static controls, and sparse/short controls; each has twenty
independent sessions and three neurons. Every retained mode, including baselines,
follows the stated prior/random walk. The recovery gate applies to the median
session mean ratio of estimated/generating field-increment variance (0.5--2),
with raw coefficient ratios reported too. Zero fractions, profile support,
parameter-bound hits, gradient convergence, field RMSE and coverage are retained.
Coverage concerns pointwise 90% Gaussian intervals conditional on fitted
hyperparameters; it excludes hyperparameter uncertainty. Bootstrap intervals
resample sessions, not time rows or neurons.

`application.json` reports independent analytic static/smooth/remapping/sparse
sessions. Training/validation/test use one-second blocks on the full grid. Automatic
fits use training counts only; fixed q and occupancy-map widths use validation
selection only. The task is retrospective interpolation across missing blocks,
not future forecasting. Test gains use Poisson-lognormal marginal predictions,
with session-paired 95% bootstrap intervals in bits per test spike. Fixed and
automatic graph fits have the same rank and nuisance-parameter policy. The tuned
fixed-q margin is -0.02 bits/spike in every condition; static controls use that
margin versus static, and the three moving conditions require a positive lower
bound versus static. Occupancy-map superiority is not a gate.

Runtime includes compilation and descriptive fitting wall time. Concurrent
validation processes share a machine, so their times do not establish a hardware
speedup or an isolated fit-only cost comparison. Biological recordings, arbitrary
geometries/ranks, exact Bayesian calibration, and causal prediction remain open.

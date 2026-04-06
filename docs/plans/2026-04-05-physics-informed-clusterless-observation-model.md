# Physics-Informed Clusterless Marked Point-Process Observation Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build an offline clusterless marked point-process decoder whose observation model uses known extracellular-space constraints to explain waveform drift, overlap, and geometry-dependent mark variation without requiring spike sorting.

**Architecture:** Keep the latent decoder simple and move the modeling complexity into the observation model. Represent each event through probe-aware marks, then predict those marks with a physics-informed forward model composed of: a constrained extracellular scaffold, a low-dimensional drift transformation, and a small neural residual for mismatch that the scaffold cannot capture. Use the resulting marked point-process likelihood inside a dedicated observation update rather than forcing it through the existing spike-count-only filter interface.

**Tech Stack:** JAX, Optax, NumPy, existing modules in `src/state_space_practice/position_decoder.py` and `src/state_space_practice/simulate_data.py`, pytest. Use pure JAX neural components in V1 rather than adding a new neural-network framework.

---

## Why This Revision

The scientific motivation here is not just “decode without sorting.” It is to use known extracellular constraints to explain precisely the phenomena that make sorting fragile: drift, overlapping spikes, geometry-dependent amplitudes, and slow waveform deformation.

That means the model should be more PINN-like in the observation model, not primarily in the latent dynamics. This revision therefore makes four changes:

1. the latent motion model is deliberately simple
2. the observation model becomes the primary modeling target
3. physical structure is encoded explicitly before adding any neural residual flexibility
4. overlap and drift are treated as observation-model phenomena rather than as arbitrary latent-state noise

## Non-Goals for V1

- No raw waveform-to-mark extraction pipeline
- No streaming or real-time guarantees
- No adaptive GMM birth/death process
- No multi-region hierarchy
- No optogenetic masking logic
- No claim that spike sorting is obsolete
- No full PDE or finite-element tissue model
- No unconstrained neural observation model

## State and Observation Design

Use a latent state of either:

$$
z_t = [x_t, y_t, v_{x,t}, v_{y,t}]
$$

or, if drift is enabled,

$$
z_t = [x_t, y_t, v_{x,t}, v_{y,t}, d_{x,t}, d_{y,t}].
$$

The decoder state remains intentionally lightweight. The PINN-like structure enters through the event observation model.

For each detected event, assume a geometry-aware mark vector such as:

$$
m_i = [a_{1,i}, \ldots, a_{K,i}, w_i, c_i]
$$

where $a_{k,i}$ are amplitudes or summary statistics tied to electrode geometry, $w_i$ is a width-like feature, and $c_i$ is an optional center-of-mass or spatial summary. The first implementation should use precomputed mark features rather than raw waveforms.

The observation model for an event should be decomposed as

$$
\mu(m_i \mid s_i, z_t) = f_{\text{physics}}(s_i, z_t, g) + f_{\text{drift}}(d_t, g) + f_{\text{residual},\theta}(s_i, z_t, g),
$$

where:

- $s_i$ is a latent event source or component descriptor
- $g$ denotes probe geometry and electrode coordinates
- $f_{\text{physics}}$ is a constrained forward model based on extracellular attenuation and smooth spatial coupling
- $f_{\text{drift}}$ is a low-dimensional drift transformation
- $f_{\text{residual},\theta}$ is a small neural residual term

The neural residual should be explicitly regularized so that the physical scaffold explains as much of the mark structure as possible.

Within each time bin, define a proper marked-observation log-likelihood of the form

$$
\log p(\mathcal{E}_t \mid z_t) = -\Delta t \sum_c \lambda_c(z_t) + \sum_{i \in \mathcal{E}_t} \log \left(\sum_c \lambda_c(z_t) p_\theta(m_i \mid c, z_t, g)\right),
$$

where $\mathcal{E}_t$ is the set of marks in bin $t$. This gives the decoder a concrete objective with gradients and Hessians.

## Physics Constraints for the Observation Model

The observation model should encode at least the following constraints:

1. **Distance attenuation:** expected amplitude decreases with source-electrode distance.
2. **Spatial smoothness across channels:** neighboring electrodes should have correlated responses for nearby sources.
3. **Low-dimensional drift:** changes in mark geometry across time should be explainable by a small drift state before invoking arbitrary component changes.
4. **Overlap consistency:** ambiguous events should be explainable by mixtures or superpositions of plausible sources, not by impossible single-source marks.
5. **Residual economy:** the neural residual should be penalized so it does not absorb structure already captured by the physical scaffold.

## V1 Physical Scaffold Assumptions

To keep the first implementation identifiable and testable, the physical scaffold should make a small number of explicit assumptions rather than vaguely appealing to extracellular physics.

1. **Quasi-static observation model:** ignore full time-dependent field propagation and model marks as instantaneous summaries of a local extracellular event.
2. **Known probe geometry:** electrode coordinates are treated as known inputs and remain fixed.
3. **Local isotropy near the probe:** use a simple distance-based attenuation law or closely related smooth decay model rather than a full anisotropic tissue model.
4. **Low-dimensional drift:** represent drift as a translation or small affine warp of the source-to-probe mapping, not arbitrary per-channel waveform drift.
5. **Simple overlap rule:** start with additive or mixture-style superposition in mark space for overlapping events.
6. **Fixed noise scale in V1:** keep observation noise parameters fixed or tightly regularized so the model cannot explain everything by inflating variance.

These assumptions are deliberately limited. The goal of V1 is not to be biophysically complete; it is to encode enough correct structure that the model fails for the right reasons and improves on geometry-agnostic baselines.

## Success Criteria

1. A synthetic clusterless dataset can be generated with position, marks, and optional drift.
2. The physics-informed observation model returns finite likelihoods, gradients, and stable parameter updates.
3. On synthetic data with drift or overlap, the physics-informed model outperforms a geometry-agnostic mark likelihood.
4. The neural residual remains small when the physical scaffold is sufficient.
5. The code path is compatible with later extensions for richer source models or waveform features.

### Task 1: Define the Geometry-Aware Mark Data Interface

**Files:**
- Create: `src/state_space_practice/clusterless_decoder.py`
- Create: `src/state_space_practice/extracellular_observation.py`
- Test: `src/state_space_practice/tests/test_clusterless_decoder.py`

**Step 1: Add typed containers for marks and binned observations**

Define structures for:

- event times or bin indices
- mark arrays of shape `(n_events, n_mark_dims)`
- electrode geometry arrays
- optional per-event channel neighborhoods

For example:

```python
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class MarkedSpikeObservations:
	bin_index: jnp.ndarray
	marks: jnp.ndarray
	electrode_positions: jnp.ndarray
	n_time: int
```

**Step 2: Add validation helpers**

Write input checks for shape consistency, sorted bin indices, valid mark dimensionality, and consistency between mark channels and electrode geometry.

**Step 3: Write failing tests first**

Add tests for valid construction and informative failure on malformed inputs.

**Step 4: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_clusterless_decoder.py -v`

Expected: PASS for input validation tests.

### Task 2: Implement the Physics-Informed Forward Observation Model

**Files:**
- Modify: `src/state_space_practice/extracellular_observation.py`
- Test: `src/state_space_practice/tests/test_clusterless_decoder.py`

**Step 1: Add the physical scaffold**

Implement helpers that map latent event source descriptors and electrode coordinates to expected marks under simple extracellular assumptions. Start with a constrained forward model such as distance-based attenuation plus smooth channel coupling.

For V1, make the scaffold explicit and narrow:

- use known electrode coordinates
- represent each event by a low-dimensional source descriptor such as source location plus amplitude
- predict channel amplitudes with a monotone distance-decay law
- derive width or center-of-mass style marks from the predicted channel pattern rather than learning them freely

**Step 2: Add an explicit drift transformation**

Implement a low-dimensional drift mapping that deforms expected marks through a small translation or affine correction tied to the drift state.

**Step 3: Add a small neural residual**

Add a pure-JAX residual network that predicts a bounded correction to the physical scaffold. Keep it intentionally small.

**Step 4: Add PINN-style penalties**

Implement regularizers such as:

- monotone distance attenuation penalty
- neighboring-channel smoothness penalty
- drift smoothness penalty
- residual magnitude penalty

These penalties should be part of training, not ad hoc post-processing.

**Step 5: Write failing tests first**

Add tests verifying that:

- predicted amplitudes decrease with distance in a simple geometry setup
- nearby channels receive more similar predictions than distant channels
- zero drift leaves marks unchanged
- the residual network output is bounded and finite

**Step 6: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_clusterless_decoder.py -v`

Expected: PASS for physical-scaffold tests.

### Task 3: Implement the Physics-Informed Marked Likelihood

**Files:**
- Modify: `src/state_space_practice/extracellular_observation.py`
- Modify: `src/state_space_practice/clusterless_decoder.py`
- Test: `src/state_space_practice/tests/test_clusterless_decoder.py`

**Step 1: Build the event-level marked likelihood**

Implement `log_mark_likelihood(...)` using the observation model from Task 2.

**Step 2: Aggregate to the bin level**

Implement a function that returns

- the scalar marked point-process log-likelihood for all events in a bin
- its gradient with respect to latent state
- and, if tractable, its Hessian or a numerically stable approximation

**Step 3: Add tests**

Verify that:

- likelihoods are finite
- under the same geometry and latent state, a physically compatible mark yields higher log-likelihood than an implausible one
- empty bins are handled correctly
- overlap-like marks are assigned higher probability under a mixture-compatible model than under a single-source-only special case

**Step 4: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_clusterless_decoder.py -v`

Expected: PASS for mark-likelihood tests.

### Task 4: Add a Simple Clusterless Decoder Core

**Files:**
- Modify: `src/state_space_practice/clusterless_decoder.py`
- Reference: `src/state_space_practice/position_decoder.py`
- Reference: `src/state_space_practice/extracellular_observation.py`
- Test: `src/state_space_practice/tests/test_clusterless_decoder.py`

**Step 1: Reuse existing position dynamics**

Use `build_position_dynamics(...)` from `position_decoder.py` for the base kinematics and extend it with a small random-walk drift state only if needed.

**Step 2: Implement a dedicated marked-observation update**

Do not try to force the first implementation through `point_process_kalman.py`, which currently assumes binned spike counts and a log-intensity function over neurons. Instead, add a local Laplace-style or Newton-style update in `clusterless_decoder.py` that consumes the physics-informed bin-level marked log-likelihood from Task 3.

**Step 3: Implement the decoding loop**

Implement a narrow offline decoder API such as:

```python
def decode_clusterless_position(
	observations: MarkedSpikeObservations,
	init_mean,
	init_cov,
	transition_matrix,
	process_cov,
	mark_model,
):
	...
```

The first version can use autodiff or numerical derivatives for the marked-observation objective if that keeps the code simple, but the update should still optimize a true marked point-process log-likelihood.

**Step 4: Write end-to-end tests**

Add tests that decode a tiny synthetic dataset and check that outputs are finite with the expected shapes.

**Step 5: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_clusterless_decoder.py -v`

Expected: PASS for decoder smoke tests.

### Task 5: Add Synthetic Data Generation with Extracellular Constraints

**Files:**
- Modify: `src/state_space_practice/simulate_data.py`
- Test: `src/state_space_practice/tests/test_clusterless_decoder.py`

**Step 1: Add a synthetic generator**

Create a helper that simulates:

- latent position trajectory
- latent event sources tied to position or place-field occupancy
- event times or bin counts
- event marks generated from the constrained extracellular forward model
- optional slow probe drift applied through the drift transform
- optional overlap events formed from mixed or superposed sources

**Step 2: Keep the first generator minimal**

Use a fixed number of source templates and simple geometry. Do not add bursting, artifacts, or component birth/death yet.

**Step 3: Add tests**

Verify that generated datasets are reproducible under a fixed seed and satisfy expected shapes and finite-value checks.

**Step 4: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_clusterless_decoder.py -v`

Expected: PASS for synthetic data tests.

### Task 6: Validate Against Non-Physical Baselines

**Files:**
- Modify: `src/state_space_practice/tests/test_clusterless_decoder.py`
- Optional analysis script: `notebooks/place_field_tracking_2d.py`

**Step 1: Define a naive baseline**

Use one of:

- a random-walk prior without mark information
- a geometry-agnostic Gaussian mark model
- nearest-component mark assignment followed by simple position averaging

**Step 2: Compare on synthetic data**

Measure:

- position RMSE
- negative log-likelihood
- robustness to a modest amount of mark drift
- robustness to overlap-like events

**Step 3: Add a PINN-specific ablation**

Compare:

- physical scaffold only
- physical scaffold plus neural residual
- unconstrained neural residual without physics penalties

The desired outcome is that the scaffold-plus-residual model performs best or is most stable, while the unconstrained residual is less interpretable or less data-efficient.

**Step 4: Use realistic success criteria**

The first milestone succeeds if the decoder beats geometry-agnostic baselines offline on controlled synthetic datasets with drift or overlap, and if the neural residual does not simply overwhelm the physical scaffold.

## Verification Checklist

- [ ] Input containers validate marks, bins, and electrode geometry correctly
- [ ] Physical-scaffold predictions satisfy basic attenuation and spatial smoothness tests
- [ ] Bin-level marked point-process log-likelihood is finite and differentiable on small synthetic examples
- [ ] Decoder returns finite trajectories and covariances on synthetic data
- [ ] Synthetic generator supports drift and overlap without shape or stability failures
- [ ] Physics-informed observation model outperforms a geometry-agnostic baseline on at least one controlled benchmark
- [ ] Neural residual remains controlled rather than replacing the physical scaffold entirely

## Known Implementation Risks

- The current point-process filter expects spike counts and a log-intensity function, so clusterless marked observations need a separate measurement update path.
- A physical scaffold that is too weak will push all explanatory burden into the neural residual, defeating the purpose of the PINN design.
- Mixture-style marked likelihoods can become numerically unstable if component variances collapse; the first implementation should keep noise scales fixed and well-conditioned.
- A realistic clusterless decoder is much easier to stage if probe geometry and mark-generation parameters are treated as known in the initial synthetic benchmark.

## Deferred Research Extensions

- Learning richer source-template families from data
- More expressive drift fields than low-dimensional translations or affine warps
- Artifact components and history-dependent marks
- Streaming inference and GPU real-time optimization
- Full extracellular field models or PDE-constrained observation networks
- Multi-region and stimulation-aware models

## Expected Outcome

This plan produces a genuinely more PINN-like clusterless decoder in the current repository: the observation model is anchored to known extracellular constraints, the neural component is used as a residual rather than a replacement for physics, and the resulting decoder directly targets the failure modes that make spike sorting brittle.

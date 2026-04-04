# Multinomial Choice Learning Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Build a state-space model that tracks an animal's evolving valuation of multiple choice options in a multi-armed bandit task, where the latent state is a vector of option values that drift over time and choices are multinomial (softmax) observations.

**Architecture:** The latent state is `x_t ∈ R^{K-1}` representing the animal's relative value for each of K-1 choice options (option 0 is the reference, fixed at 0 for identifiability). Values evolve via random walk. The observation is the animal's choice, modeled as `Categorical(softmax([0, x_t]))`. Inference uses the Laplace-EKF. EM learns the drift rate (process noise) and inverse temperature.

**Tech Stack:** JAX, existing Laplace-EKF pattern from `smith_learning_algorithm.py`, `psd_solve`/`symmetrize`/`_kalman_smoother_update` from `kalman.py`.

**Prerequisite Gates:**

- Use this document as the design contract and the companion task breakdown in `docs/plans/2026-04-04-multinomial-choice-tasks.md` as the execution checklist.
- Verify that `smith_learning_algorithm.py` and `kalman.py` contain the APIs referenced here before implementation.
- If the companion task document and this design document disagree, resolve the discrepancy in the plan before writing code.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_smith_learning_algorithm.py src/state_space_practice/tests/test_kalman.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- Before declaring the plan complete, run the targeted tests plus the neighbor regression tests in the same environment and confirm the expected pass/fail transitions for each task.

**Feasibility Status:** READY

**Codebase Reality Check:**

- Reusable primitives exist: Laplace-EKF patterns in `src/state_space_practice/smith_learning_algorithm.py` and linear-algebra/smoother utilities in `src/state_space_practice/kalman.py`.
- Planned new module is required: `src/state_space_practice/multinomial_choice.py`.

**Claude Code Execution Notes:**

- Use this file as the design contract and the companion task-breakdown plan as the execution sequence.
- Add early equivalence gates: K=2 multinomial should agree with the existing Smith-style binary setting before broadening to K>2.
- Keep the softmax update implementation analytically grounded (explicit gradient/Hessian) and add finite-check assertions around Newton updates.

**MVP Scope Lock (implement now):**

- Implement K=3 choices as the primary supported case first (general code can still accept K>2).
- Use scalar process-noise parameterization (`Q = q * I`) and a single inverse-temperature parameter.
- Require K=2 consistency to Smith-style behavior as the core acceptance gate.

**Defer Until Post-MVP:**

- Rich process-noise structures and hierarchical priors.
- Multi-condition temperature schedules and context-dependent choice policies.

**References:**

- Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006). Cortical substrates for exploratory decisions in humans. Nature 441, 876-879.
- Piray, P. & Daw, N.D. (2021). A simple model for learning in volatile environments. PLoS Computational Biology 17(4), e1007963.
- Smith, A.C., Frank, L.M., Wirth, S. et al. (2004). Dynamic analysis of learning in behavioral experiments. J Neuroscience 24(2), 447-461.
- Wilson, R.C. & Collins, A.G.E. (2019). Ten simple rules for the computational modeling of behavioral data. eLife 8, e49547.

---

## Mathematical Model

### Generative model

```
Latent value state (K-1 free parameters, option 0 is reference):
    x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
    x_t ∈ R^{K-1}

Full value vector (for softmax):
    v_t = [0, x_t]  ∈ R^K   (prepend 0 for reference option)

Choice observation:
    c_t ~ Categorical(softmax(β * v_t))
    P(c_t = k) = exp(β * v_{t,k}) / Σ_j exp(β * v_{t,j})

    β: inverse temperature (exploration vs exploitation)
```

### Identifiability

The softmax is invariant to adding a constant to all values: `softmax(x + c) = softmax(x)`. Without a constraint, the values drift together and the model is non-identifiable. We fix option 0 as the reference (value = 0) and estimate K-1 relative values. This is the standard approach in multinomial logistic regression.

### Laplace approximation for softmax

The softmax observation model is nonlinear but log-concave. The Laplace-EKF linearizes around the current state estimate:

```
Step 1: Construct full K-dim one-hot for the chosen option:
    e_k = [0, ..., 1, ..., 0] ∈ R^K   (1 at index k)

Step 2: Compute softmax probabilities from full value vector:
    p = softmax(β * [0, x]) ∈ R^K

Step 3: Gradient of log P(c_t=k | x) w.r.t. x (K-1 free params):
    ∇_x = β * (e_k[1:] - p[1:])
    Note: when choice=0 (reference), e_k[1:] = 0, so ∇_x = -β * p[1:]
    (all free values decrease, which is correct)

Step 4: Negative Hessian w.r.t. x (K-1 × K-1 submatrix):
    H = β² * (diag(p[1:]) - outer(p[1:], p[1:]))
    This is the Fisher information restricted to free parameters.
    Always PSD (log-concavity), so the Laplace approximation is
    well-behaved — no BFGS failure modes.
```

Because the Hessian is analytic and the posterior is log-concave, we use
iterative Newton (2-3 steps with convergence check) rather than BFGS. For
moderate β and well-conditioned priors, a single Newton step is often
sufficient, but with large β or a prior mean far from the mode, Newton can
overshoot. Iterating to convergence ensures the Laplace mode is accurate,
which is critical for the K=2 consistency test with SmithLearningModel.

**Note on approximation:** Even with iterative Newton, the Laplace-EKF is
an approximation (Gaussian posterior at each step). The Smith model uses
BFGS to find the exact Laplace mode for each scalar update; the
multinomial model uses Newton iteration on the K-1 dimensional problem.

### EM M-step

The M-step updates:

1. **Process noise Q** (scalar × I for simplicity):

   The multivariate M-step formula (using smoother sufficient statistics):
   ```
   Q_hat = (1/(T-1)) * Σ_{t=1}^{T-1} [
       (m_{t|T} - m_{t-1|T})(m_{t|T} - m_{t-1|T})'
       + P_{t|T} + P_{t-1|T}
       - 2 * P_{t-1,t|T}
   ]
   ```
   where `P_{t-1,t|T}` is the smoother cross-covariance from
   `_kalman_smoother_update`. Since Q is scalar × I, take the mean
   of the diagonal: `q = mean(diag(Q_hat))`, clamped to `>= 1e-8`.

   **Important:** This uses `smoother_cross_cov[t]` which represents
   `Cov(x_t, x_{t+1} | y_{1:T})` — the cross-covariance between
   consecutive trials. Convention: index `t` refers to the earlier trial.

2. **Inverse temperature β**: Two-phase optimization:
   (a) Coarse grid search over candidates (default: `[0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 12]`).
   (b) Golden-section refinement around the best grid point (bracket = neighboring
   grid values, 10 refinement steps). This handles β >> 12 without needing a
   huge grid. Use `jax.vmap` over the grid to vectorize the filter passes.
   The `init_cov` is fixed (not learned via EM), so BIC parameter count
   is K+1: Q (1 scalar) + β (1 scalar) + init_mean (K-1 values).

### Log-likelihood approximation

The marginal log-likelihood stored by the filter is the observation
log-likelihood evaluated at the prior mean (`log p(c_t | x_{t|t-1})`),
not the true Laplace-approximated marginal. This is the same convention
as `SmithLearningModel` and is standard for Laplace-EKF EM monitoring.

### Relationship to existing models

- **SmithLearningModel:** K=2 case with reference-option parameterization is equivalent to Smith's binomial model (value of option 1 relative to option 0 ≈ logit-space learning state).
- **K=2 consistency test:** With 2 options, the multinomial model should produce results equivalent to SmithLearningModel. This is a key validation test.

---

## Implementation Scope

### Phase 1 (this plan): Core model
- Task 1: Softmax observation update (Laplace-EKF step)
- Task 2: Filter and smoother
- Task 3: `MultinomialChoiceModel` class with EM
- Task 4: Simulation, summary, plotting, and null model comparison

### Phase 2 (future): Extensions
- Covariate-driven dynamics (reward history as input)
- Joint choice + neural observations (shared latent value state)
- Comparison with Q-learning / RL models

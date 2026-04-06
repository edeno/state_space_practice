# RL in State-Space Form: Covariate-Driven Choice Models

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Extend the multinomial choice model and Smith learning model with covariate-driven dynamics so that option values update in response to observable trial events (rewards, cues, reward history). This transforms the random-walk learning model into a reinforcement learning model in state-space form, where value updates are explained by covariates rather than pure drift, and estimation uses Kalman filtering/smoothing rather than point estimates.

**Scientific motivation:** The current `MultinomialChoiceModel` tracks evolving option values but treats all changes as unexplained drift (`x_t = x_{t-1} + noise`). In reality, values change because the animal receives rewards, encounters cues, or experiences prediction errors. Modeling these drivers explicitly lets us:

1. Ask what drives learning (reward vs. cue vs. time)
2. Separate signal (covariate-driven value updates) from noise (unexplained drift)
3. Build toward joint neural-behavioral models where neural data constrains the value state alongside covariates and choices
4. Compare against standard RL models (Q-learning, Rescorla-Wagner) using the same data and model selection tools (BIC, cross-validation)

**Architecture:** Two extensions to the existing state-space choice model:

1. **Dynamics covariates (input-driven value updates):** `x_t = x_{t-1} + B @ u_t + w_t` where `u_t` is a vector of trial-level covariates (reward received, reward prediction error, cue identity, etc.) and `B` is a learned input-gain matrix. This is the Kalman filter with control inputs.

2. **Observation covariates (context-dependent choice policy):** `p(c_t = k) = softmax(β * [0, x_t] + Θ' z_t)` where `z_t` is a vector of trial-level choice-context features (e.g., position at choice point, time since last reward, trial number). `Θ` modifies choice probabilities without changing the latent value state.

**Tech Stack:** JAX, existing `MultinomialChoiceModel` and `SmithLearningModel`, `psd_solve`/`symmetrize`/`_kalman_smoother_update` from `kalman.py`.

**Prerequisite Gates:**

- Multinomial choice model tests must pass before starting.
- Verify that `kalman.py` M-step supports input matrices (it currently does not — the M-step learns A but assumes no control input B).
- If SmithLearningModel needs parallel changes for K=2 consistency, implement those as part of this plan.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_smith_learning_algorithm.py src/state_space_practice/tests/test_kalman.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

**Feasibility Status:** READY (builds directly on completed multinomial choice model)

**Codebase Reality Check:**

- Reusable: `MultinomialChoiceModel` filter/smoother with softmax Laplace-EKF, `_kalman_smoother_update` for RTS smoothing, `psd_solve`/`symmetrize` for linear algebra.
- New module required: `src/state_space_practice/covariate_choice.py`
- New tests required: `src/state_space_practice/tests/test_covariate_choice.py`
- Existing `MultinomialChoiceModel` stays unchanged — new module extends, does not modify.

**Claude Code Execution Notes:**

- Start with dynamics covariates only (the RL-relevant case). Observation covariates are Phase 2.
- The M-step for B has a closed-form solution using smoother sufficient statistics — do not use gradient-based optimization.
- Add a Rescorla-Wagner equivalence test: with a single reward covariate and no unexplained drift (Q → 0), the model should recover standard RL update behavior.
- Keep K=2 parity with a covariate-extended SmithLearningModel as an acceptance gate.

**MVP Scope Lock (implement now):**

- Dynamics covariates with learned B matrix and scalar residual process noise Q.
- Support common RL covariates: reward received (per option), reward prediction error, trial number.
- EM learns B, Q, and β jointly. B M-step is closed-form from smoother statistics.
- Require Rescorla-Wagner equivalence test and K=2 Smith parity test.
- Provide `simulate_rl_choice_data` with known B for validation.

**Defer Until Post-MVP:**

- Observation covariates (context-dependent choice policy via Θ).
- Hierarchical priors over B across sessions or subjects.
- Joint neural-behavioral model with covariates (combine with position decoder).
- Non-stationary B (time-varying learning rates).
- Full RL model comparison toolkit (Q-learning, actor-critic baselines).

**References:**

- Rescorla, R.A. & Wagner, A.R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A.H. Black & W.F. Prokasy (Eds.), Classical conditioning II, 64-99.
- Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006). Cortical substrates for exploratory decisions in humans. Nature 441, 876-879.
- Piray, P. & Daw, N.D. (2021). A simple model for learning in volatile environments. PLoS Computational Biology 17(4), e1007963.
- Smith, A.C., Frank, L.M., Wirth, S. et al. (2004). Dynamic analysis of learning in behavioral experiments. J Neuroscience 24(2), 447-461.
- Gershman, S.J. (2015). A Unifying Probabilistic View of Associative Learning. PLoS Computational Biology 11(11), e1004567.

---

## Mathematical Model

### Current model (random walk)

```text
x_t = x_{t-1} + w_t,   w_t ~ N(0, q I)
c_t ~ Categorical(softmax(β * [0, x_t]))
```

All value changes are unexplained drift. The model tracks *that* values change but not *why*.

### Extended model (covariate-driven dynamics)

```text
x_t = x_{t-1} + B @ u_t + w_t,   w_t ~ N(0, q I)
c_t ~ Categorical(softmax(β * [0, x_t]))

x_t ∈ R^(K-1)     latent relative values
u_t ∈ R^d          observed covariates at trial t
B ∈ R^(K-1 × d)   input-gain matrix (learned)
q                  residual process noise (learned)
β                  inverse temperature (learned)
```

### What B captures

Each column of B maps one covariate to value updates for all K-1 options:

- **Reward covariate** (d=K-1 columns, one per non-reference option):
  `u_t = [r_{1,t}, r_{2,t}, ..., r_{K-1,t}]` where `r_{k,t} = 1` if option k was rewarded.
  `B_reward` learns the per-option reward sensitivity (learning rate in RL terms).

- **Global reward signal** (d=1):
  `u_t = [reward_received_t]` regardless of which option.
  `B_global` learns how reward affects all option values.

- **Reward prediction error** (d=1 per option):
  `u_t = [r_{k,t} - p_{k,t-1}]` where `p_{k,t-1}` is the model's predicted choice probability.
  This makes the model equivalent to Rescorla-Wagner when `q → 0`.

### Inference

The filter prediction step changes from:
```text
pred_mean = filt_mean                    (random walk)
```
to:
```text
pred_mean = filt_mean + B @ u_t          (covariate-driven)
pred_cov  = filt_cov + Q                 (unchanged)
```

The softmax observation update is **unchanged** — it still applies the Laplace-EKF to the predicted state. The smoother is also unchanged (the transition matrix is still identity; the control input B only shifts the prediction mean).

### EM M-step for B

Given smoother statistics, the M-step for B is a multivariate regression:

```text
B_hat = [ Σ_t (m_{t|T} - m_{t-1|T}) u_t' ] @ [ Σ_t u_t u_t' ]^{-1}
```

This is the standard least-squares solution: regress the smoothed value increments `Δm_t = m_{t|T} - m_{t-1|T}` onto the covariates `u_t`. The residual variance gives the updated Q:

```text
residual_t = Δm_t - B_hat @ u_t
Q_hat = (1/(T-1)) * Σ_t [ residual_t residual_t' + P_t + P_{t-1} - 2 C_{t-1,t} ]
q = mean(diag(Q_hat)), clamped >= 1e-8
```

### Relationship to RL

| RL concept | State-space equivalent |
|---|---|
| Q-values | Latent state `x_t` (relative to reference) |
| Learning rate α | Columns of B (per-covariate, per-option) |
| Exploration rate ε | Inverse temperature β (softmax) |
| Value update rule | Prediction step `x_t = x_{t-1} + B @ u_t + noise` |
| Belief uncertainty | Posterior covariance `P_t` (not available in standard RL) |

Key advantages over standard RL:
1. **Uncertainty quantification:** posterior covariance at every trial
2. **Smoothed estimates:** RTS smoother uses future data for offline analysis
3. **Principled model comparison:** BIC/marginal likelihood vs. RL with ad-hoc fitting
4. **Multiple learning signals:** B matrix handles arbitrary covariates, not just reward
5. **Path to neural integration:** same latent state can drive neural observation model

### Relationship to Rescorla-Wagner

With K=2, a single binary reward covariate `u_t = r_t ∈ {0, 1}`, and `q → 0` (no unexplained drift), the dynamics become:

```text
x_t = x_{t-1} + b * r_t
```

This is exactly the Rescorla-Wagner update `V_t = V_{t-1} + α * δ_t` when `b = α` and `δ_t = r_t` (the prediction error simplifies to reward when the expected value is absorbed into the state). The state-space version adds posterior uncertainty and the ability to smooth.

---

## Implementation Scope

### Phase 1 (this plan): Dynamics covariates

- Task 1: Covariate-driven prediction step + tests
- Task 2: Filter and smoother with covariates + tests
- Task 3: `CovariateChoiceModel` class with EM for B, Q, β + tests
- Task 4: Simulation (`simulate_rl_choice_data`), Rescorla-Wagner equivalence test, diagnostics

### Phase 2 (future): Observation covariates

- Add `Θ' z_t` to the softmax linear predictor
- M-step for Θ via Newton (concave in Θ)
- Enables context-dependent choice policy (e.g., spatial position at choice point)

### Phase 3 (future): Joint neural-behavioral model

- Combine covariate-driven value dynamics with spike observation model
- Neural firing rate: `log λ_n = b_n + w_n' Z(position) + α_n' x_t`
- Position can be either observed (tracked) or latent (inferred from spikes)
- Connects to Plan 4 (joint belief-state decoder) with covariates

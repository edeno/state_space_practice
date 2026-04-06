# Switching Choice Model: Strategy-Dependent Multi-Armed Bandit

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Build a switching state-space choice model where discrete latent states represent behavioral strategies (e.g., exploit, explore, reset) that control how continuous option values evolve and drive choices. This bridges two existing approaches: the Julia lab's HMM (discrete contingency states with fixed reward tables) and our `CovariateChoiceModel` (continuous values with a single dynamics regime).

**Scientific motivation:** In the Frank Lab's 6-arm spatial bandit task, animals don't use one fixed strategy. They alternate between exploiting known good arms, exploring alternatives, and resetting after contingency changes. The Julia HMM captures contingency switches by enumerating all possible reward schedules, but this requires knowing the task structure a priori. The switching choice model instead infers strategy switches directly from behavior — what changes is not "which contingency is active" but "how the animal is learning."

This distinction matters because:
1. Strategy switches are a property of the animal, not the task. They can be correlated with neural states.
2. The model doesn't need the experimenter's contingency table. It works on any multi-armed bandit task.
3. Per-state parameters are directly interpretable: a state with high β and low Q is exploitation; a state with low β and high Q is exploration.
4. The same framework extends to joint neural-behavioral inference, where hippocampal/mPFC neural modes may align with behavioral strategy switches.

**Architecture:** A switching linear dynamical system with softmax observations:

```
Discrete state (behavioral strategy):
    s_t ~ Categorical(T[s_{t-1}, :])

Continuous values (strategy-dependent dynamics):
    x_t = A_{s_t} @ x_{t-1} + B_{s_t} @ u_t + w_t,   w_t ~ N(0, Q_{s_t})

Choice (strategy-dependent policy):
    c_t ~ softmax(β_{s_t} * [0, x_t] + Θ_{s_t} @ z_t)
```

The key insight: **every parameter from `CovariateChoiceModel` becomes per-state.** The discrete state selects which learning rate, decay, exploration level, and choice biases are active on each trial.

**Tech Stack:** JAX, `switching_kalman.py` (GPB2 filter, mixture collapse, discrete state updates, RTS smoother), `multinomial_choice.py` (`_softmax_update_core`), `covariate_choice.py` (covariate prediction, M-steps for B and Q).

**Prerequisite Gates:**

- `CovariateChoiceModel` tests must pass (DONE — 55 tests).
- `switching_kalman.py` tests must pass.
- Verify `_softmax_update_core` accepts `obs_offset` parameter (DONE).
- Verify `switching_point_process.py` pattern for per-state-pair updates is stable.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_choice.py -v`
- Neighbor regression: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_switching_kalman.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

**Feasibility Status:** READY (all components exist; this is integration, not invention)

**Codebase Reality Check:**

- Reusable components:
  - `_softmax_update_core` from `multinomial_choice.py` — Laplace-EKF update for softmax choice
  - `_scale_likelihood`, `_update_discrete_state_probabilities`, `collapse_gaussian_mixture_per_discrete_state` from `switching_kalman.py` — GPB2 machinery
  - `_rts_smoother_pass`, `_kalman_smoother_update` from `multinomial_choice.py` / `kalman.py` — backward smoother
  - `covariate_predict`, `m_step_input_gain`, `m_step_obs_weights` from `covariate_choice.py` — M-steps
  - `switching_point_process.py` — architectural pattern for per-state-pair Laplace updates inside a switching filter
- New module required: `src/state_space_practice/switching_choice.py`
- New tests required: `src/state_space_practice/tests/test_switching_choice.py`

**Claude Code Execution Notes:**

- Follow the `switching_point_process.py` implementation pattern exactly. The per-state-pair update, mixture collapse, and scan structure are proven.
- The softmax update replaces the point-process update; everything else (discrete state inference, smoother) is reused.
- JIT compilation time will be the main concern. Start with S=2 states to validate, then test S=3.
- The single-state equivalence test (S=1 → `CovariateChoiceModel`) is the critical acceptance gate.

**MVP Scope Lock (implement now):**

- S=2 discrete states (exploit vs. explore) as the primary case.
- Per-state scalar parameters: β_s, Q_s, decay_s.
- Shared B matrix and Θ matrix across states (per-state B and Θ are deferred — they multiply the parameter count by S and complicate the M-step).
- Shared init_mean across states.
- EM learns: per-state Q, per-state β (grid search), per-state decay, transition matrix T, shared B and Θ.
- Require three acceptance tests: S=1 parity, synthetic two-state recovery, and improved LL over non-switching model on switching data.

**Defer Until Post-MVP:**

- Per-state B and Θ matrices.
- More than 3 discrete states.
- Joint neural-behavioral switching (adding spike observations per state).
- Cross-session state persistence.
- Group-level EM across subjects.

**References:**

- Linderman, S.W., Johnson, M.J., Miller, A.C. et al. (2017). Bayesian learning and inference in recurrent switching linear dynamical systems. AISTATS.
- Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006). Cortical substrates for exploratory decisions in humans. Nature 441, 876-879.
- Wilson, R.C. & Collins, A.G.E. (2019). Ten simple rules for the computational modeling of behavioral data. eLife 8, e49547.
- Smith, A.C. et al. (2004). Dynamic analysis of learning in behavioral experiments. J Neuroscience 24(2), 447-461.

---

## Mathematical Model

### Generative model

```text
Discrete state (S states, Markov chain):
    s_t ~ Categorical(T[s_{t-1}, :])
    T is S × S transition matrix (learned)

Continuous option values (K-1 free parameters, per-state dynamics):
    x_t = decay_{s_t} * x_{t-1} + B @ u_t + w_t
    w_t ~ N(0, q_{s_t} * I)

    decay_{s_t}: per-state scalar decay (mean reversion)
    q_{s_t}: per-state scalar process noise
    B: shared input-gain matrix (reward → value updates)
    u_t: observed dynamics covariates (reward history, etc.)

Full value vector (for softmax):
    v_t = [0, x_t] ∈ R^K

Choice:
    c_t ~ Categorical(softmax(β_{s_t} * v_t + Θ @ z_t))

    β_{s_t}: per-state inverse temperature
    Θ: shared observation weights (stay bias, spatial bias)
    z_t: observed choice-context covariates
```

### Scientific interpretation of states

| Parameter | Exploit state | Explore state |
|---|---|---|
| β (inverse temperature) | High (~5-10) — choose best option reliably | Low (~0.5-2) — choose more randomly |
| Q (process noise) | Low (~0.001) — values are stable | High (~0.05) — values are volatile |
| decay | High (~0.99) — remember past rewards | Low (~0.8) — forget quickly, reset values |
| B (learning rate) | Shared — same reward sensitivity | Shared — same reward sensitivity |

The shared B means the animal learns from reward at the same rate in both states — what changes is how much the values drift (Q) and how deterministically the animal follows them (β). This is the key behavioral signature of explore vs. exploit.

### Inference

The switching Kalman filter maintains S Gaussians, one per discrete state:

```text
For each trial t:
    For each state pair (s_{t-1}=i, s_t=j):
        1. Predict: x_t^{ij} = decay_j * x_{t-1}^i + B @ u_t,
                    P_t^{ij} = decay_j^2 * P_{t-1}^i + Q_j
        2. Update: softmax Laplace update with β_j and choice c_t
        3. Log-likelihood: log p(c_t | x_t^{ij}, β_j)

    Scale likelihoods, update discrete state probabilities:
        p(s_t=j | data) ∝ Σ_i T[i,j] * p(s_{t-1}=i | data) * p(c_t | s_t=j)

    Collapse mixture: GPB2 reduces S² Gaussians back to S
        (one per current discrete state)
```

This is O(S² × K²) per trial. With S=2 and K=6, that's 4 × 25 = 100 operations per trial — very fast.

### EM M-steps

| Parameter | M-step | Notes |
|---|---|---|
| Q_s (per-state process noise) | Closed-form weighted regression | Weight each trial by p(s_t = j), same formula as `_m_step_process_noise` but weighted |
| decay_s (per-state decay) | Closed-form weighted regression | Weight each trial by p(s_t = j), same formula as `_m_step_decay` but weighted |
| β_s (per-state inverse temperature) | Grid search + golden section, per state | Run S separate grid searches, each using state-weighted filter LL |
| T (transition matrix) | Closed-form from joint smoothed probabilities | Standard: T[i,j] ∝ Σ_t p(s_{t-1}=i, s_t=j) — already in `switching_kalman_maximization_step` |
| B (shared input-gain) | Closed-form regression | Weighted by sum of all state probabilities (= 1 per trial), so same as non-switching M-step |
| Θ (shared obs weights) | Newton M-step | Same as non-switching, since Θ doesn't depend on state |

### Relationship to existing models

**Our CovariateChoiceModel:** The S=1 special case. With one discrete state, the switching model reduces exactly to `CovariateChoiceModel` with the same parameters.

**Julia Q-learner:** The S=1 case with decay and covariates is already comparable (previous analysis). The switching model adds strategy states on top.

**Julia HMM:** Both have discrete latent states, but:
- HMM states = which contingency is active (needs contingency table)
- Our states = which behavioral strategy is active (learned from behavior)
- HMM produces Q-values as expected reward under state belief
- Our model produces Q-values as smoothed continuous latent values
- HMM can be compared against our model via BIC on the same data

**Plan 8 (Joint Learning + Drift):** Plan 8 has a factored state (scalar θ + per-neuron weights) with Smith binomial observations. This plan has a single continuous state with multinomial softmax observations. Plan 8 is the neural + behavioral integration; this plan is behavioral only. They could eventually be combined, but this plan is self-contained and much simpler.

### Path to neural integration

Once the behavioral switching model works, adding neural observations per state is straightforward:

```text
Spike observation (per neuron n):
    log λ_{n,t} = b_n^{s_t} + w_n^{s_t}' Z(position_t) + α_n^{s_t}' x_t
    y_{n,t} ~ Poisson(λ_{n,t} * dt)
```

This would replace `_softmax_update_core` with a joint update that applies both the softmax and the point-process Laplace steps sequentially on the same latent state. The discrete state then captures both behavioral strategy switches AND neural coding mode switches. This connects to:
- **S1 (CA1 Represented-State Switching):** the discrete states may align with local vs. nonlocal content
- **S2 (Value-Gated Sequence Expression):** the discrete states may predict when replay occurs

This neural extension is NOT in the current plan scope. It should be a separate follow-up plan after the behavioral switching model is validated.

### Gap: group-level EM

The Julia SpatialBanditTask repo fits all models with a two-level EM that pools across subjects. This plan fits each session independently. Comparing switching model results across animals requires a separate group-level EM framework, which is not in any current plan. For now, per-animal fitting followed by second-level statistics is the practical approach.

---

## Implementation Scope

### Phase 1 (this plan): Behavioral switching choice model

- Task 1: Per-state-pair softmax update (following `switching_point_process.py` pattern)
- Task 2: Switching choice filter with GPB2 + scan
- Task 3: `SwitchingChoiceModel` class with EM
- Task 4: Simulation, S=1 parity, two-state recovery, comparison to non-switching model

### Phase 2 (future): Per-state covariates

- Per-state B_s and Θ_s (strategy-dependent learning rates and biases)
- More than 3 discrete states

### Phase 3 (future): Joint neural-behavioral switching

- Add spike observations per state
- Connects to S1/S2 in the scientific track
- Discrete states constrained by both behavior and neural coding

### Phase 4 (future): Group-level inference

- Two-level EM across subjects
- Population-level priors over per-subject parameters
- Direct comparison to Julia EM framework

# Shared contracts — the transition-logit contract across the three switching families

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Index

1. [Parametrisation and layout contract](#parametrisation-and-layout-contract)
2. [Time-alignment contract](#time-alignment-contract)
3. [Keyword contract](#keyword-contract)
4. [None-path invariant](#none-path-invariant)
5. [`discrete_transitions` module API](#discrete_transitions-module-api)
6. [M-step contract](#m-step-contract)
7. [Validation contract](#validation-contract)
8. [Model attribute contract](#model-attribute-contract)
9. [Sibling-mirroring checklist](#sibling-mirroring-checklist)

Every phase links here. Do not weaken a contract in one family without changing
it in all three.

## Parametrisation and layout contract

Per row `i` (current state), the transition logits are

```text
logit_t[i, j] = eta[i, j] + Gamma[:, i, j] . u_t + W[:, i, j] . x_{t-1}      j = 0..S-2
logit_t[i, S-1] = 0                                                        (reference column)
T_t[i, :] = softmax(logit_t[i, :])
```

- `S = n_discrete_states`. The **last** state is the reference column of every
  row, exactly the convention of `centered_softmax` / `centered_softmax_inverse`
  (moved from `contingency_belief.py:84-130`) and of `STOCHASTIC_ROW`
  (`parameter_transforms.py:225-252`, which drops the last column). Checked:
  `STOCHASTIC_ROW.to_unconstrained(Z) == centered_softmax_inverse(Z)`.
- `eta` is **not** a new parameter. It is `centered_softmax_inverse(discrete_transition_matrix)`,
  so `discrete_transition_matrix` (shape `(S, S)`, row-stochastic) remains the
  baseline matrix every caller already passes, snapshots and learns. With all
  weights zero, `T_t == discrete_transition_matrix` up to the log/exp round trip
  (~1e-16).
- `transition_covariates` (`u_t`): `(n_time, n_features)`, float, finite. A design
  matrix the caller builds; no implicit intercept column (the intercept is `eta`).
- `transition_weights` (`Gamma`): `(n_features, S, S - 1)`.
- `transition_state_weights` (`W`, phase 2): `(n_cont_states, S, S - 1)`, where
  `n_cont_states` is the continuous latent dimension of the family
  (`n_cont_states` / `n_latent` / `n_options - 1`).
- Discrete-state axes trail (CLAUDE.md), so the contractions are
  `jnp.einsum("tk,kij->tij", u, Gamma)` and `jnp.einsum("di,dij->ij", m_prev, W)`
  with `m_prev` of shape `(n_cont_states, S)` (column `i` = the collapsed mean
  given `S_{t-1} = i`).
- The realised **transition stack** is `(n_time, S, S)`; entry `t` is the
  row-stochastic matrix for the step *into* time `t` (`P(S_t = j | S_{t-1} = i)`).
  Entry `0` is the baseline matrix and is never read.

Invariant (do not weaken): a relabelling of the discrete states
`perm` maps `(Z, Gamma, W)` to the parameters given in
[designs.md H](designs.md#h-relabelling-transform) and produces
`T'_t = T_t[perm][:, perm]` exactly; the existing relabelling invariance
(`tests/test_invariances.py:332-345`) must hold with the new options.

## Time-alignment contract

- `transition_covariates[t]` drives `S_{t-1} -> S_t`, i.e. it is paired with
  observation `t`. Row `0` is never read. This is the `contingency_belief`
  convention (`contingency_belief.py:577-583` scans `transition_covariates[1:]`;
  the M-step pairs `xi[t]` with design row `t + 1`, `:1298-1302`).
- Filters scan over `stack[1:]` alongside `obs[1:]`.
- The GPB1 smoother's backward step at time `t` (combining filter `t` with
  smoothed `t + 1`) uses `stack[t + 1]`; it scans over `stack[1:]` aligned with
  `filter_mean[:-1]`.
- Viterbi backward scores for `pair_log_liks[t]` (observation `t + 1`) use
  `log(stack[t + 1])`.
- The M-step response `xi[t] = P(S_t = i, S_{t+1} = j | y)` (shape
  `(n_time - 1, S, S)`, index `t` = pair `(t, t + 1)`) pairs with design row
  `[1, u_{t+1} (, m_{t|T}^{i})]` — covariates shifted by one, phase-2 state means
  not shifted (they are the means at the *source* time).
- Simulators draw `s_t` from `T_t` built from `u_t` and `x_{t-1}`.

## Keyword contract

Names are fixed (other plans use them):

| Where | Phase 1 (keyword-only) | Phase 2 (keyword-only) |
| --- | --- | --- |
| `switching_kalman_filter`, `switching_kalman_viterbi`, `switching_kalman_smoother` | `transition_covariates=None, transition_weights=None` | `transition_state_weights=None` |
| `switching_point_process_filter` and `_switching_point_process_filter_jit` | same | same |
| `switching_choice_filter`, `_switching_choice_filter_jit`, `switching_choice_smoother` | same | same |
| `utils.hmm_viterbi` | `transition_matrix` may be `(n_time, K, K)` (no new keyword) | — |
| `simulate_switching_kalman.simulate`, `simulate_switching_spike_oscillator`, `simulate_switching_choice_data` | `transition_covariates=None, transition_weights=None` | `transition_state_weights=None` |
| `BaseModel.fit` / `.fit_sgd`, `SwitchingSpikeOscillatorModel.fit` / `.fit_sgd`, `BaseSwitchingPointProcessModel.fit`, `SwitchingChoiceModel.fit` / `.fit_sgd` | `transition_covariates=None` (weights are model attributes) | — (weights are model attributes) |
| the three base-class constructors | `transition_regularization: float = 1e-5` | — |

Rules:

- At the **function** level `transition_covariates` and `transition_weights` are
  both-or-neither (`ValueError` otherwise), mirroring the choice filter's
  `covariates` / `input_gain` rule (`switching_choice.py:331-343`).
  `transition_state_weights` may be given alone.
- At the **model** level `fit(..., transition_covariates=u)` with
  `transition_weights is None` zero-initialises the weights
  `(n_features, S, S - 1)`; a non-`None` weights attribute whose feature axis
  does not match `u.shape[1]` raises `ValueError` (fail loud, no silent reset).
  `transition_state_weights` is enabled by setting the attribute to an array
  before `fit` / `fit_sgd` (there is no covariate to bind); phase 2 adds a
  constructor option `recurrent_transitions: bool = False` that zero-initialises it.
- `predict_proba` / `decode` are unchanged (they read the smoothed posterior).

## None-path invariant

With every new keyword at `None`, the jitted functions trace exactly as today:
the scan receives a `None` leaf in `xs` and the fixed `discrete_transition_matrix`
stays a closed-over constant (verified: `lax.scan(f, c, (obs, None))` traces with
`num_consts=1` and no per-step transition input). Outputs are `array_equal`
with and without the keywords passed explicitly as `None`. The model classes
install the count-based `discrete_transition_matrix` from
`switching_kalman_maximization_step` exactly as before, and the SGD parameter
specs are unchanged. Tests in every phase assert this (fast).

## `discrete_transitions` module API

New module `src/state_space_practice/discrete_transitions.py`; added to
`[tool.mypy] files` in `pyproject.toml`; imports only `jax`, `jax.numpy`,
`numpy` (validation) and `state_space_practice.utils` (never `switching_kalman`,
which imports it). Full code in [designs.md A/E](designs.md).

```python
def centered_log_softmax(logits: Array) -> Array          # (..., S-1) -> (..., S)   moved verbatim
def centered_softmax(logits: Array) -> Array              # (..., S-1) -> (..., S)   moved verbatim
def centered_softmax_inverse(probs: Array) -> Array       # (..., S) -> (..., S-1)   moved verbatim

def transition_logits_stack(
    baseline_logits: Array,                # (S, S-1)
    transition_covariates: Array | None,   # (n_time, n_features)
    transition_weights: Array | None,      # (n_features, S, S-1)
    n_time: int,
) -> Array                                 # (n_time, S, S-1); entry 0 = baseline

def state_dependent_logits(                # phase 2
    base_logits_t: Array,                  # (S, S-1)
    transition_state_weights: Array,       # (n_cont_states, S, S-1)
    prev_state_cond_mean: Array,           # (n_cont_states, S)
) -> Array                                 # (S, S-1)

def transition_matrix_stack(
    discrete_transition_matrix: Array,     # (S, S)
    transition_covariates: Array | None = None,
    transition_weights: Array | None = None,
    transition_state_weights: Array | None = None,   # phase 2
    state_cond_means: Array | None = None,           # phase 2: (n_time, n_cont_states, S), filter means
    *, n_time: int | None = None,
) -> Array | None                          # None when no option is given; else (n_time, S, S)

def resolve_scan_transitions(
    discrete_transition_matrix: Array,
    transition_covariates: Array | None,
    transition_weights: Array | None,
    transition_state_weights: Array | None,          # phase 2 (phase 1 omits this argument)
    n_time: int,
) -> tuple[Array | None, Callable[[Array | None, Array], Array]]
    # (per_step_xs, transition_at_step). per_step_xs is None on the fixed path,
    # stack[1:] (probabilities) on the covariate path, logits[1:] on the state path.
    # transition_at_step(xs_t, prev_state_cond_mean) -> (S, S) is called inside the scan body.

def validate_transition_inputs(
    discrete_transition_matrix, transition_covariates, transition_weights,
    transition_state_weights, n_time: int, n_cont_states: int,
) -> None                                  # host-side; no-op on tracers

def dirichlet_neg_log_likelihood(coefficients_flat, design_matrix, response, alpha, l2_penalty=1e-5) -> Array
                                           # moved verbatim from contingency_belief.py:225-278
def optimize_transition_rows(
    x0_all: Array,          # (S, n_coef * (S-1))
    response_all: Array,    # (S, n_samples, S)
    alpha_all: Array,       # (S, S)
    design_all: Array,      # (S, n_samples, n_coef)   <- per-row designs (generalises contingency_belief.py:803-846)
    l2_penalty: Array,
) -> Array                  # (S, n_coef * (S-1)); jitted at module level

def maximize_transition_coefficients(
    smoother_joint_discrete_state_prob: Array,      # (n_time-1, S, S)
    discrete_transition_matrix: Array,              # (S, S) current baseline
    transition_covariates: Array | None = None,
    transition_weights: Array | None = None,
    transition_state_weights: Array | None = None,  # phase 2
    state_cond_smoother_means: Array | None = None, # phase 2: (n_time, n_cont_states, S)
    *, transition_prior: Array | None = None, l2_penalty: float = 1e-5,
) -> tuple[Array, Array | None, Array | None]       # (new Z, new Gamma or None, new W or None)
```

`contingency_belief.py` re-exports the three softmax helpers,
`dirichlet_neg_log_likelihood` and keeps a thin `_optimize_transition_rows`
wrapper that broadcasts its shared design to per-row designs (so its jitted
compile-once test only needs its monkeypatch target moved).

## M-step contract

- Objective per source row `i` (maximised; written as the minimised negative in
  `dirichlet_neg_log_likelihood`):
  `sum_t xi[t, i, :] . log softmax(design_i[t] @ coef_i) / N + sum_j (alpha_eff[i, j] - 1) mean_t log softmax(...)[j] - l2 * ||coef_i[1:]||^2`,
  with `N = n_time - 1` and `l2 = transition_regularization` on the
  non-intercept rows of `coef_i` (i.e. on `Gamma` and `W`, never on `eta`).
  `alpha_eff = 1 + (alpha - 1) / N` where `alpha` is the existing Dirichlet
  `transition_prior` (`get_transition_prior`, `>= 1`; all-ones when `None`).
  The rescaling is required for consistency with the fixed path:
  `dirichlet_neg_log_likelihood` averages its data term over `N` and weights
  the pseudo-counts per sample, so an unscaled `alpha - 1` would act as
  `N (alpha - 1)` counts, whereas `_switching_kalman_m_step_inner`
  (`switching_kalman.py:2275-2281`) adds `alpha - 1` counts. With
  `alpha_eff` and an intercept-only design the logit M-step equals the
  count-based MAP `(counts + alpha - 1)` row-normalised (a required test). The
  pseudo-count term remains the documented approximation of
  `contingency_belief.py:239-244` once covariates enter (not a conjugate prior).
- Solver: per-row BFGS via `jax.scipy.optimize.minimize` (`maxiter=50`), warm-started
  from the current coefficients — the request's "gradient ascent" default,
  refined to the quasi-Newton solver the repo already uses for this objective.
- Identifiability gate: `row_occupancy[i] = sum_t sum_j xi[t, i, j]` (expected
  transitions out of `i`). Rows with `row_occupancy < minimum_state_occupancy(n_coef)`
  (`n_coef = 1 + n_features (+ n_cont_states)`, i.e. regressors + 1,
  `switching_kalman.py:2063-2071`) keep their previous coefficients, and
  `warn_low_occupancy_states(row_occupancy, min_occ, "maximize_transition_coefficients",
  "their transition logits kept their previous values")`
  (`switching_kalman.py:2074-2110`) is called when the inputs are concrete.
- Outputs: `new_Z = centered_softmax(new_eta)` (row-stochastic by construction),
  `new_Gamma`, `new_W`. The count-based `discrete_state_transition` from
  `switching_kalman_maximization_step` is **not** installed on this path.
- SGD equivalence: the SGD losses add `(n_time - 1) * l2 * (sum(Gamma**2) + sum(W**2))`
  so EM and SGD optimise the same penalised objective (the EM data term is
  averaged over `N`; the SGD loss is the unaveraged `-log p(y)`). Precedent:
  the spike-weight penalty at `switching_point_process.py:3455-3468`.

## Validation contract

`validate_transition_inputs` runs on the host at every public entry point
(filter wrappers, smoothers, Viterbi, model `fit` / `fit_sgd`) and is a no-op
when any input is a tracer (`utils.contains_tracer`, precedent
`switching_point_process.py:396-397`). It raises `ValueError` when:

- exactly one of `transition_covariates` / `transition_weights` is given;
- `transition_covariates.shape != (n_time, n_features)` or it is non-finite;
- `transition_weights.shape != (n_features, S, S - 1)` or
  `transition_state_weights.shape != (n_cont_states, S, S - 1)`, or either is non-finite;
- any new option is given and `discrete_transition_matrix` has an exact zero
  (structural zeros are not representable by a softmax row; message names
  `centered_softmax_inverse`).

Static shape checks additionally run at trace time inside the jitted point-process
core (`_validate_switching_point_process_filter_shapes`,
`switching_point_process.py:286-370`), matching how that core already checks
its other arguments. Model-level `_validate_parameter_shapes` methods check the
weights' shapes when the attributes are not `None`.

## Model attribute contract

| Attribute | Gaussian `BaseModel` | Point-process bases | `SwitchingChoiceModel` |
| --- | --- | --- | --- |
| baseline matrix (existing) | `discrete_transition_matrix` | `discrete_transition_matrix` | `discrete_transition_matrix_` |
| covariate weights | `transition_weights` (`None` default) | `transition_weights` | `transition_weights_` |
| state weights (phase 2) | `transition_state_weights` | `transition_state_weights` | `transition_state_weights_` |
| bound covariates (private) | `_transition_covariates` | `_transition_covariates` | `_transition_covariates` |
| L2 penalty | `transition_regularization` | `transition_regularization` | `transition_regularization` |
| forwarding helper | `_transition_kwargs()` -> dict of the filter/smoother keywords | same | same |
| install helper | `_install_discrete_transition(count_based_Z)` | same | same |
| EM snapshot | `_EM_SNAPSHOT_KEYS` + `transition_weights`, `transition_state_weights` | `_snapshot_em_state` attrs / SSOM `_snapshot_params` + both | (no snapshot today; unchanged) |
| SGD keys | `"transition_weights"`, `"transition_state_weights"` (`UNCONSTRAINED`), only when `update_discrete_transition_matrix` and the option is active | same | same |

Each family follows its own naming style (trailing underscore only in the choice
model, as its existing attributes do).

## Sibling-mirroring checklist

Every item below exists in each family; a phase that touches an item in one
family lists the others' rows as tasks. (Memory rules
`fix-sibling-implementations`, `dim-pp-mirrors-gaussian-dim`.)

| Item | Gaussian (`switching_kalman.py`, `oscillator_models.py`) | Point process (`switching_point_process.py`, `point_process_models.py`) | Choice (`switching_choice.py`) |
| --- | --- | --- | --- |
| filter scan body takes the per-step matrix | `_step` `:871-988`, call `:947-957`, scan `:1032-1042` | `_step` `:2194-2321`, call `:2277-2292`, scan `:2355-2376` | `_step` `:432-490`, call `:464-469`, scan inputs `:499` |
| host-side validation at the public entry | `switching_kalman_filter` is `@jax.jit`: validate in a thin wrapper section before tracing (see [designs.md B](designs.md#b-gaussian-filter-integration)) | wrapper `:2453-2455` | wrapper `:233-249` |
| GPB1 smoother per-step matrix | `_step` `:1436-1618`, call `:1538-1542`, scan `:1639-1648` | reuses the Gaussian smoother (`_e_step` `:2989-2992`) | `switching_choice_smoother` `:595-633` (per-step call `:601-608`) |
| Viterbi | `switching_kalman_viterbi` `:1140-1215`; `utils.hmm_viterbi` `:1797-1805` | none (no Viterbi entry point) | none |
| E-step forwards keywords | `_e_step` `:913-923`, `:963-966` | `_e_step` `:2934-2947`, `:2989-2992` | `_run_filter` `:912-931`, `_run_smoother` `:1026-1034` |
| install site(s) | `_m_step` `:1027-1028`; DIM `:2082-2083` | `_m_step_dynamics` `:3124-3125`; DIM-PP `:1368-1369` | `_m_step` `:1115-1118` |
| fit binds covariates | `fit` `:1082-1142`, `fit_sgd` `:1146-1194` | SSOM `fit` `:3841-`, base `fit_sgd` `:3313-3359`; `BaseSwitchingPointProcessModel.fit` `:516-568` | `fit` `:933-1011` (`:965-971`), `fit_sgd` `:1122-1173` (`:1159-1165`) |
| SGD spec / loss / store | COM `:1420-1462`; CNM `:1745-1811`; DIM `:2169-2313`; base store `:1205-1217` + `_sgd_param_attrs` `:404-413` | `_shared_sgd_param_spec` `:3379-3409`; `_sgd_loss_fn` `:3411-3470`; `_store_sgd_params` `:3472-3485` | `_build_param_spec` `:1179-1203`; `_sgd_loss_fn` `:1205-1227`; `_store_sgd_params` `:1229-1238` |
| snapshot / restore | `_EM_SNAPSHOT_KEYS` `:415-438`; warm-init restore `:794-800` | `_snapshot_em_state` `:474-497`; SSOM `_snapshot_params` `:3966-3984` | n/a |
| parameter-shape validation | `OscillatorParameterBase._validate_parameter_shapes` `:288-299` | `_validate_parameter_shapes` `:2786-2836` | constructor checks `:742-761` (add the weights) |
| predicted discrete prior for diagnostics | n/a | n/a | `_populate_uncertainty` `:813-822` |
| simulator | `simulate_switching_kalman.simulate` `:98-130` | `simulate_switching_spike_oscillator` `:15-`, `_step` `:142-166` | `simulate_switching_choice_data` `:1260-1359` |
| identity tests (None bit-for-bit; zero weights `1e-12`; one-hot covariate switches at the right step) | phase 1a | phase 1b | phase 1c |

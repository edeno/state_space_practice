# Shared contracts — multi-map place fields

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Contracts referenced by more than one phase. Each appears once; phases link by anchor and
must not weaken them.

- [HMM forward–backward](#hmm-forward-backward) — used by phase 1 (constant transitions) and phase 1b (time-varying stack)
- [Model attributes](#model-attributes) — the fitted-state names and shapes phase 1 defines and phase 1b extends

## HMM forward-backward

`state_space_practice.utils.hmm_filter(initial_probs, transition_matrix, log_likelihoods)`
and `state_space_practice.utils.hmm_forward_backward(...)` (full code in
[designs A](designs.md#a-exact-hmm-forwardbackward-in-log-space)).

| Argument / field | Shape | Semantics |
| --- | --- | --- |
| `initial_probs` | `(n_states,)` | `P(s_1 = k)`; exact zeros are structural and stay zero |
| `transition_matrix` | `(n_states, n_states)` **or** `(n_time - 1, n_states, n_states)` | row-stochastic `Z[i, j] = P(s_{t+1} = j \| s_t = i)`; in a stack, index `t` moves the chain from bin `t` to bin `t + 1` |
| `log_likelihoods` | `(n_time, n_states)` | `log p(y_t \| s_t = k)`, fully normalised |
| `HMMPosterior.smoothed_prob` | `(n_time, n_states)` | `P(s_t \| y_{1:T})` |
| `HMMPosterior.pairwise_prob` | `(n_time - 1, n_states, n_states)` | `P(s_t = i, s_{t+1} = j \| y_{1:T})`; slice `t` sums over `j` to `smoothed_prob[t]` and over `i` to `smoothed_prob[t + 1]` |
| `HMMPosterior.filtered_prob` | `(n_time, n_states)` | `P(s_t \| y_{1:t})` |
| `HMMPosterior.log_likelihood` / `hmm_filter(...)[1]` | `()` | `log p(y_{1:T})`, differentiable in all arguments |

Invariants (do not weaken):

- Exact: matches path enumeration to round-off (`rtol 1e-10`) for both the constant and the
  stacked transition input. Phase 1 pins this; phase 1b relies on the stacked branch being
  already exact and adds no new approximation.
- Discrete-state axis trails, time axis leads (project convention).
- Both functions are `jax.jit`-compiled and accept a 3-D stack without any change of
  signature — this is what makes the recurrent-transitions extension additive.
- Structural zeros in `transition_matrix` or `initial_probs` never receive posterior mass;
  an impossible observation (every reachable state at `-inf`) yields NaN (fail loud, no
  silent dynamics fallback).

`hmm_viterbi` (`src/state_space_practice/utils.py:1769-1824`) takes a `(K, K)` matrix in
phase 1. Phase 1b consumes the recurrent-transitions plan's `(n_time, K, K)`
branch: entry t is the transition into bin t and entry 0 is ignored. Its
backward step uses `[t + 1]`. The model pads its `(n_time - 1, K, K)` edge
stack with an ignored identity row before this call; the 2-D path stays
bit-identical. Reject a stack with the wrong leading length.

## Model attributes

`state_space_practice.multi_map_place_field.MultiMapPlaceFieldModel`, after `fit` /
`fit_sgd` (map axis trails everywhere). Accessing any of these before a fit raises
`NotFittedError` through `_check_fitted` for the methods; the attributes themselves are
`None`. `from_parameters(...)` sets the parameter block and leaves the posterior block
`None`.

| Attribute | Shape / type | Meaning |
| --- | --- | --- |
| `weights_` | `(n_neurons, n_basis, n_maps)` | per-neuron, per-map GLM weights; `exp(Z @ weights_[n, :, k])` is neuron `n`'s rate (Hz) under map `k` |
| `transition_matrix_` | `(n_maps, n_maps)` | row-stochastic map transitions (phase 1b: the *baseline* matrix at zero covariates) |
| `init_map_prob_` | `(n_maps,)` | `P(s_1)` |
| `smoothed_map_prob_` | `(n_time, n_maps)` | `P(s_t \| y)` |
| `filtered_map_prob_` | `(n_time, n_maps)` | `P(s_t \| y_{1:t})` |
| `pairwise_map_prob_` | `(n_time - 1, n_maps, n_maps)` | `P(s_t, s_{t+1} \| y)` |
| `switch_probability_` | `(n_time - 1,)` | `1 - trace(pairwise_map_prob_[t])`: probability that the map changes between bins `t` and `t + 1` |
| `log_likelihood_` | `float` | unpenalised marginal log-likelihood of the final parameters on the training data |
| `log_likelihood_history_` | `list[float]` | the objective the fitter maximised per accepted iteration/step: marginal log-likelihood plus the log prior (weights penalty, Dirichlet pseudo-counts); equals the marginal log-likelihood when `penalty=0` and the transition prior is flat |
| `converged_` | `bool` | EM relative-change criterion met / SGD stall criterion met |
| `n_neurons_`, `n_basis_` | `int` | recorded at fit; a refit with different shapes raises `ValueError` |
| `viterbi_path()` | `(n_time,)` int | most likely map sequence on the training data |

Label convention: map 0 is the most-occupied map (`smoothed_map_prob_.sum(0)` descending,
ties by original index). Phase 1b adds `transition_logits_` `(n_maps, n_maps - 1)` and
`transition_weights_` `(n_maps, n_maps - 1, d_h)` and keeps every row above unchanged.

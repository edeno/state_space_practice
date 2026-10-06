# Designs — multi-map place fields

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Per-component designs with complete code. Phase files reference these sections by
anchor; this file does not repeat the task lists. All code assumes float64
(`jax.config.update("jax_enable_x64", True)` before import, as the whole library does).

Sections:

- [A. Exact HMM forward–backward in log space](#a-exact-hmm-forwardbackward-in-log-space) (`utils.hmm_filter`, `utils.hmm_forward_backward`)
- [B. Per-map Poisson log-likelihoods](#b-per-map-poisson-log-likelihoods)
- [C. Weighted, penalised Poisson-GLM M-step](#c-weighted-penalised-poisson-glm-m-step)
- [D. k-means initialisation on windowed population rate maps](#d-k-means-initialisation-on-windowed-population-rate-maps)
- [E. Transition and initial-state M-step; occupancy ordering](#e-transition-and-initial-state-m-step-occupancy-ordering)
- [F. Model class wiring (EM via `run_em`, SGD via `SGDFittableMixin`)](#f-model-class-wiring)
- [G. Two-map session simulator](#g-two-map-session-simulator)
- [H. Exact path-enumeration oracle for tests](#h-exact-path-enumeration-oracle-for-tests)
- [I. Behavioural alignment helpers](#i-behavioural-alignment-helpers)
- [J. Tensor-product spline difference penalty](#j-tensor-product-spline-difference-penalty)
- [K. Slow per-map drift — sketch, not scheduled](#k-slow-per-map-drift--sketch-not-scheduled)

---

## A. Exact HMM forward–backward in log space

**Where:** `src/state_space_practice/utils.py`, inserted directly after `hmm_viterbi`
(ends at `utils.py:1824`) and before `zero_preserving_log` (`utils.py:1827`). New names:
`HMMPosterior`, `hmm_filter`, `hmm_forward_backward`. `hmm_viterbi` is untouched in
phase 1 (phase 1b adapts edge stacks to the recurrent plan's existing 3-D Viterbi branch).

**Why in `utils` and not the model module:** the library's other discrete-state helpers
(`hmm_viterbi` `utils.py:1769-1824`, `make_discrete_transition_matrix` `:1722-1759`,
`find_permutation` `:1895-1921`) live there, and the recurrent-transitions plan needs the
same recursion with a time-varying transition stack. The interface below is the
[forward–backward contract](shared-contracts.md#hmm-forward-backward).

**Why not reuse `contingency_belief._contingency_belief_smoother_jit`
(`contingency_belief.py:652-800`):** that smoother is in linear probability space with
`jnp.maximum(…, 1e-30)` floors (`:751-754`, `:783-786`) and is fused with the bandit
observation model. The switching Kalman filter's discrete update
(`switching_kalman.py:454-570`) is in log space but is one step of a GPB filter with
support masks, not a standalone HMM pass. A ~60-line standalone log-space recursion is
the simplest correct thing and is exact-oracle-testable ([H](#h-exact-path-enumeration-oracle-for-tests)).

```python
class HMMPosterior(NamedTuple):
    """Exact discrete-state posteriors of a hidden Markov model.

    Attributes
    ----------
    smoothed_prob : Array, shape (n_time, n_states)
        ``P(s_t = k | y_{1:T})``.
    pairwise_prob : Array, shape (n_time - 1, n_states, n_states)
        ``P(s_t = i, s_{t+1} = j | y_{1:T})``; ``pairwise_prob[t].sum(1) == smoothed_prob[t]``
        and ``pairwise_prob[t].sum(0) == smoothed_prob[t + 1]``.
    filtered_prob : Array, shape (n_time, n_states)
        ``P(s_t = k | y_{1:t})``.
    log_likelihood : Array, shape ()
        ``log p(y_{1:T})``.
    """

    smoothed_prob: Array
    pairwise_prob: Array
    filtered_prob: Array
    log_likelihood: Array


def _log_transition_stack(transition_matrix: Array, n_time: int) -> Array:
    """``(n_time - 1, K, K)`` log-transition stack from a ``(K, K)`` matrix or a stack.

    Exact zeros stay ``-inf`` (:func:`zero_preserving_log`), so a structurally
    forbidden transition never receives posterior mass.
    """
    log_trans = zero_preserving_log(jnp.asarray(transition_matrix))
    if log_trans.ndim == 2:
        return jnp.broadcast_to(log_trans, (n_time - 1, *log_trans.shape))
    if log_trans.ndim != 3 or log_trans.shape[0] != n_time - 1:
        raise ValueError(
            "transition_matrix must be (n_states, n_states) or a stack "
            f"(n_time - 1, n_states, n_states); got {log_trans.shape} for n_time={n_time}."
        )
    return log_trans


def _hmm_forward(
    initial_probs: Array, transition_matrix: Array, log_likelihoods: Array
) -> tuple[Array, Array]:
    """Normalised log filtered probabilities ``(n_time, K)`` and ``log p(y_{1:T})``."""
    log_likelihoods = jnp.asarray(log_likelihoods)
    n_time = log_likelihoods.shape[0]
    log_trans = _log_transition_stack(transition_matrix, n_time)

    log_joint_0 = zero_preserving_log(jnp.asarray(initial_probs)) + log_likelihoods[0]
    log_norm_0 = logsumexp(log_joint_0)
    log_alpha_0 = log_joint_0 - log_norm_0

    def _step(log_alpha_prev, inputs):
        log_trans_t, ll_t = inputs
        # predict: log sum_i alpha_{t-1}(i) Z_t[i, j]; update: + log p(y_t | s_t = j)
        log_joint = logsumexp(log_alpha_prev[:, None] + log_trans_t, axis=0) + ll_t
        log_norm = logsumexp(log_joint)  # log p(y_t | y_{1:t-1})
        log_alpha = log_joint - log_norm
        return log_alpha, (log_alpha, log_norm)

    _, (log_alpha_rest, log_norms) = jax.lax.scan(
        _step, log_alpha_0, (log_trans, log_likelihoods[1:])
    )
    log_alpha = jnp.concatenate([log_alpha_0[None], log_alpha_rest], axis=0)
    return log_alpha, log_norm_0 + jnp.sum(log_norms)


@jax.jit
def hmm_filter(
    initial_probs: Array, transition_matrix: Array, log_likelihoods: Array
) -> tuple[Array, Array]:
    """Forward pass of a hidden Markov model.

    Parameters
    ----------
    initial_probs : Array, shape (n_states,)
        ``P(s_1 = k)``. Exact zeros are structural (kept at probability 0).
    transition_matrix : Array, shape (n_states, n_states) or (n_time - 1, n_states, n_states)
        Row-stochastic, ``Z[i, j] = P(s_{t+1} = j | s_t = i)``. A stack gives the
        transition into bin ``t + 1`` at index ``t`` (time-varying transitions).
    log_likelihoods : Array, shape (n_time, n_states)
        ``log p(y_t | s_t = k)``.

    Returns
    -------
    filtered_prob : Array, shape (n_time, n_states)
    log_likelihood : Array, shape ()
        ``log p(y_{1:T})``. Differentiable in every argument (the SGD objective).

    Notes
    -----
    Standard scaled forward recursion in log space (Rabiner 1989). If every
    reachable state has ``-inf`` likelihood at some bin (an impossible
    observation) the result is NaN: the library fails loud rather than
    substituting a dynamics prediction here.
    """
    log_alpha, log_likelihood = _hmm_forward(initial_probs, transition_matrix, log_likelihoods)
    return jnp.exp(log_alpha), log_likelihood


@jax.jit
def hmm_forward_backward(
    initial_probs: Array, transition_matrix: Array, log_likelihoods: Array
) -> HMMPosterior:
    """Exact smoothed, pairwise and filtered posteriors of a hidden Markov model.

    Arguments as in :func:`hmm_filter`. The backward messages are renormalised
    at every step; posteriors are normalised per bin (per pair slice), so the
    normalisation constants cancel exactly.
    """
    log_likelihoods = jnp.asarray(log_likelihoods)
    n_time, n_states = log_likelihoods.shape
    log_trans = _log_transition_stack(transition_matrix, n_time)
    log_alpha, log_likelihood = _hmm_forward(initial_probs, transition_matrix, log_likelihoods)

    def _backward(log_beta_next, inputs):
        log_trans_t, ll_next = inputs
        log_beta = logsumexp(log_trans_t + (ll_next + log_beta_next)[None, :], axis=1)
        log_beta = log_beta - logsumexp(log_beta)
        return log_beta, log_beta

    _, log_beta_rest = jax.lax.scan(
        _backward,
        jnp.zeros(n_states, dtype=log_likelihoods.dtype),
        (log_trans, log_likelihoods[1:]),
        reverse=True,
    )
    log_beta = jnp.concatenate(
        [log_beta_rest, jnp.zeros((1, n_states), dtype=log_likelihoods.dtype)], axis=0
    )

    log_post = log_alpha + log_beta
    smoothed = jnp.exp(log_post - logsumexp(log_post, axis=1, keepdims=True))
    log_pair = (
        log_alpha[:-1, :, None]
        + log_trans
        + (log_likelihoods[1:] + log_beta[1:])[:, None, :]
    )
    pairwise = jnp.exp(log_pair - logsumexp(log_pair, axis=(1, 2), keepdims=True))
    return HMMPosterior(smoothed, pairwise, jnp.exp(log_alpha), log_likelihood)
```

Imports needed at the top of `utils.py`: `from typing import NamedTuple` and
`from jax.scipy.special import logsumexp` (check what is already imported before adding).
`utils.py` is not in the mypy `files` list, but write it typed anyway.

Index conventions (verify in the oracle test): `pairwise[t]` is the joint of `(s_t, s_{t+1})`
and uses `log_trans[t]`; with a stack, `transition_matrix[t]` is the matrix that moves the
chain from bin `t` to bin `t + 1`. This is the pairing `contingency_belief._m_step` documents
at `contingency_belief.py:1298-1302` (response `xi[t]` ↔ design row `t + 1`).

---

## B. Per-map Poisson log-likelihoods

**Where:** `src/state_space_practice/multi_map_place_field.py`, public function.

```python
@functools.partial(jax.jit, static_argnames=("max_log_count", "batch_size"))
def per_map_log_likelihoods(
    design_matrix: Array,
    spikes: Array,
    weights: Array,
    dt: float,
    max_log_count: float,
    batch_size: int = 1024,
) -> Array:
    """``log p(y_t | s_t = k)`` for every time bin and map.

    Parameters
    ----------
    design_matrix : Array, shape (n_time, n_basis)
        Spatial basis evaluated at the animal's position at each bin.
    spikes : Array, shape (n_time, n_neurons)
        Spike counts.
    weights : Array, shape (n_neurons, n_basis, n_maps)
        Per-neuron, per-map GLM weights (map axis last).
    dt : float
        Bin width in seconds.
    max_log_count : float
        Ceiling on ``log(rate * dt)`` (see ``_safe_expected_count``); the same
        value the M-step uses, so E- and M-step evaluate one objective.
    batch_size : int
        Bins evaluated per ``lax.map`` batch. The full ``(n_time, n_neurons,
        n_maps)`` expected-count tensor is never materialised.

    Returns
    -------
    Array, shape (n_time, n_maps)
        Fully normalised Poisson log-likelihood summed over neurons (includes
        ``-lgamma(y + 1)``), so ``hmm_filter`` returns the true marginal
        log-likelihood, comparable across ``n_maps`` and with ``PlaceFieldModel.score``.
    """
    design_matrix = jnp.asarray(design_matrix)
    spikes = jnp.asarray(spikes, dtype=design_matrix.dtype)
    log_factorial = jnp.sum(gammaln(spikes + 1.0), axis=1)  # (n_time,)

    def _one_bin(inputs):
        z_t, y_t = inputs
        log_rate = jnp.einsum("b,nbk->nk", z_t, weights)  # (n_neurons, n_maps)
        mu = _safe_expected_count(log_rate, dt, max_log_count=max_log_count)
        return jnp.sum(y_t[:, None] * jnp.log(mu) - mu, axis=0)  # (n_maps,)

    log_lik = jax.lax.map(_one_bin, (design_matrix, spikes), batch_size=batch_size)
    return log_lik - log_factorial[:, None]
```

`_safe_expected_count` is `point_process_kalman.py:589-610` (already imported into
`place_field_model.py` at `:61`). `gammaln` from `jax.scipy.special`. Memory: the result is
`n_time × n_maps` (18 MB for 150 k bins, K = 3, with the smoothed/pairwise posteriors).

---

## C. Weighted, penalised Poisson-GLM M-step

**Where:** `multi_map_place_field.py`.

**Alternatives considered (and why not):**

- `switching_point_process.update_spike_glm_params` (`switching_point_process.py:1813-1968`,
  plug-in branch `:1946-1965`) fits `baseline + weights @ x` with a *scalar* L2 on the
  weights only. The place-field parametrisation has no separate baseline (the spline basis
  spans constants; a graph basis carries the constant null mode), and we need a general PSD
  penalty matrix (ridge or smoothness). Adding a baseline would also break the K = 1 parity
  with `PlaceFieldModel._fit_stationary_glm` (`place_field_model.py:652-821`), which has
  none.
- `point_process_kalman.glm_laplace_update` (`:1333-1464`) is a single-bin MAP update with a
  Gaussian prior; per-bin responsibilities would have to be smuggled in as offsets
  (`log gamma_t`, `-inf` at zero) and the count vector rescaled. Fragile.

So the solver is a small dedicated Newton loop that **reuses the shared Newton primitives**
`_ridged_newton_direction` (`switching_point_process.py:1035-1045`), `_descent_step`
(Armijo + gradient fallback, `:1048-1090`) and `_warn_newton_fallbacks` (`:1093-1105`),
and the same `_safe_expected_count` as the E-step. Same structure as `_fit_stationary_glm`'s
`_fit_one` (`place_field_model.py:769-800`) plus per-bin weights, a penalty matrix and a
line search.

```python
def _weighted_poisson_glm_newton(
    design_matrix: Array,
    counts: Array,
    time_weights: Array,
    penalty: Array,
    initial_weights: Array,
    dt: float,
    max_log_count: float,
    max_iter: int,
) -> tuple[Array, Array]:
    """``max_iter`` line-searched Newton steps on one neuron's weighted penalised GLM.

    Objective (convex): ``-sum_t gamma_t [y_t log mu_t - mu_t] + 0.5 w' P w`` with
    ``mu_t = exp(Z_t w) dt`` (clipped at ``max_log_count``).

    Parameters
    ----------
    design_matrix : (n_time, n_basis); counts : (n_time,); time_weights : (n_time,)
        Responsibilities ``gamma_{t,k}`` of the map being fitted.
    penalty : (n_basis, n_basis)  PSD penalty ``P``.
    initial_weights : (n_basis,)

    Returns
    -------
    weights : (n_basis,); n_fallbacks : () int
        Number of Newton steps that fell back to the gradient direction.
    """

    def _objective(w: Array) -> Array:
        mu = _safe_expected_count(design_matrix @ w, dt, max_log_count=max_log_count)
        nll = -jnp.sum(time_weights * (counts * jnp.log(mu) - mu))
        return nll + 0.5 * w @ (penalty @ w)

    def _step(w: Array, _: None) -> tuple[Array, Array]:
        mu = _safe_expected_count(design_matrix @ w, dt, max_log_count=max_log_count)
        gradient = design_matrix.T @ (time_weights * (mu - counts)) + penalty @ w
        hessian = design_matrix.T @ ((time_weights * mu)[:, None] * design_matrix) + penalty
        delta = _ridged_newton_direction(hessian, gradient)
        w_new, fell_back = _descent_step(
            w, delta, gradient, _objective, _objective(w), beta=0.5, max_iter=10
        )
        return w_new, fell_back

    weights, fell_back = jax.lax.scan(_step, initial_weights, None, length=max_iter)
    return weights, jnp.sum(fell_back)


@functools.partial(jax.jit, static_argnames=("max_log_count", "max_iter"))
def update_map_weights(
    design_matrix: Array,
    spikes: Array,
    responsibilities: Array,
    penalty: Array,
    weights: Array,
    dt: float,
    max_log_count: float,
    max_iter: int,
) -> tuple[Array, Array]:
    """M-step for every (neuron, map) pair, vmapped over both axes.

    Parameters
    ----------
    spikes : (n_time, n_neurons); responsibilities : (n_time, n_maps)
    weights : (n_neurons, n_basis, n_maps)  warm start.

    Returns
    -------
    new_weights : (n_neurons, n_basis, n_maps); n_fallbacks : () int
    """
    per_neuron = jax.vmap(
        _weighted_poisson_glm_newton,
        in_axes=(None, 1, None, None, 0, None, None, None),
        out_axes=(0, 0),
    )
    per_map = jax.vmap(
        per_neuron, in_axes=(None, None, 1, None, 2, None, None, None), out_axes=(2, 0)
    )
    new_weights, n_fallbacks = per_map(
        design_matrix, spikes, responsibilities, penalty, weights, dt, max_log_count, max_iter
    )
    return new_weights, jnp.sum(n_fallbacks)
```

**Warm start for the very first M-step** (from the initial responsibilities): the
intercept-matching rule of `place_field_model.py:742-790`, with the per-map weighted mean
count. Shared `intercept_direction`; one ``(n_neurons, n_maps)`` table of targets.

```python
def _intercept_matching_weights(
    design_matrix: Array, spikes: Array, responsibilities: Array, penalty: Array, dt: float
) -> Array:
    """Constant-log-rate start ``w = target * (Z'Z + P)^{-1} Z' 1`` per (neuron, map)."""
    n_time = design_matrix.shape[0]
    gram = design_matrix.T @ design_matrix + penalty
    direction = psd_solve(gram, design_matrix.T @ jnp.ones(n_time, design_matrix.dtype))
    weighted_counts = responsibilities.T @ spikes  # (n_maps, n_neurons)
    exposure = responsibilities.sum(axis=0)  # (n_maps,)
    mean_count = (weighted_counts + 1.0) / (exposure[:, None] + 1.0)  # phantom spike
    target_log_rate = jnp.log(mean_count / dt)  # (n_maps, n_neurons)
    return jnp.einsum("kn,b->nbk", target_log_rate, direction)
```

**Occupancy gate.** Maps whose expected occupancy `gamma.sum(0)[k]` is below
`minimum_state_occupancy(n_basis)` (`switching_kalman.py:2063-2071`, = `n_basis + 1`) keep
their previous weights; `warn_low_occupancy_states(occupancy, min_occ,
"MultiMapPlaceFieldModel M-step", "their weights kept their previous values")`
(`switching_kalman.py:2074-2110`) logs the warning and returns the gated indices. Run the
vmapped update for all maps and mask: `jnp.where(keep[None, None, :], new, old)`. This
mirrors `_m_step_spikes` (`switching_point_process.py:3245-3270`).

**Penalty resolution.** `penalty` (constructor) is a float `lam` → `lam * I(n_basis)`, or an
`(n_basis, n_basis)` array validated with
`validate_covariance(P, "penalty", require_positive_definite=False)` (`utils.py:1308-1405`,
symmetric PSD). Resolved once at fit time when `n_basis` is known. The default `1.0`
matches `_fit_stationary_glm(prior_precision=1.0)` so K = 1 reproduces `PlaceFieldModel`'s
warm-start GLM exactly ([phase 1 validation](phase-1-static-maps-hmm.md#validation-slice)).
For a graph basis, `np.diag(basis.eigvals) * lam` is the Laplacian (spectral) smoothness
penalty (`graph_place_field.py:58-92` `GraphBasis.eigvals`); for the tensor spline basis use
[J](#j-tensor-product-spline-difference-penalty).

**Log prior** (shared by EM's reported objective and the SGD loss so both fitters optimise
one function, as `switching_point_process._sgd_loss_fn` does at `:3455-3468`):

```python
def _log_prior(self, weights: Array, transition_matrix: Array) -> Array:
    quad = jnp.einsum("nbk,bc,nck->", weights, self._penalty, weights)
    log_prior = -0.5 * quad
    if self.n_maps > 1 and self.update_transition_matrix:
        pseudo = self._transition_prior - 1.0  # (K, K), >= 0
        log_z = zero_preserving_log(transition_matrix)
        # 0 * (-inf) would be NaN: a flat prior contributes exactly zero.
        log_prior = log_prior + jnp.sum(jnp.where(pseudo > 0, pseudo * log_z, 0.0))
    return log_prior
```

---

## D. k-means initialisation on windowed population rate maps

**Where:** `multi_map_place_field.py`, public function `kmeans_map_initialization`.

Low et al. (2021) identified map states by k-means clustering of trial-wise population
activity vectors. Our basis-agnostic analogue: for each window of `window_bins` bins, the
least-squares projection of every neuron's counts onto the spatial basis (a linear
"rate map in basis coordinates"); one feature vector per window; k-means with a fixed seed.

```python
def kmeans_map_initialization(
    design_matrix: ArrayLike,
    spikes: ArrayLike,
    n_maps: int,
    window_bins: int,
    penalty: ArrayLike,
    seed: int,
    *,
    soft_floor: float = 0.05,
) -> tuple[Array, np.ndarray]:
    """Initial map responsibilities from k-means on windowed population rate maps.

    Parameters
    ----------
    design_matrix : (n_time, n_basis); spikes : (n_time, n_neurons)
    n_maps : int; window_bins : int
        Window length in bins (e.g. 2 s). ``ceil(n_time / window_bins)`` must be
        at least ``n_maps``.
    penalty : (n_basis, n_basis)
        Ridge added to each window's Gram matrix (the model's resolved penalty).
    seed : int
        ``random_state`` of ``sklearn.cluster.KMeans`` (``n_init=10``).
    soft_floor : float
        Mass spread over the other maps so the first M-step sees every map.

    Returns
    -------
    responsibilities : Array, shape (n_time, n_maps)
        ``(1 - soft_floor)`` on the window's cluster, ``soft_floor / n_maps`` elsewhere.
    labels : np.ndarray, shape (n_windows,)
        Cluster label per window (for diagnostics).
    """
    Z = np.asarray(design_matrix, dtype=float)
    y = np.asarray(spikes, dtype=float)
    n_time, n_basis = Z.shape
    n_windows = -(-n_time // window_bins)
    if n_windows < n_maps:
        raise ValueError(
            f"{n_windows} windows of {window_bins} bins cannot seed {n_maps} maps; "
            "shorten init_window_bins or lower n_maps."
        )
    pad = n_windows * window_bins - n_time
    Zw = np.pad(Z, ((0, pad), (0, 0))).reshape(n_windows, window_bins, n_basis)
    yw = np.pad(y, ((0, pad), (0, 0))).reshape(n_windows, window_bins, -1)
    gram = np.einsum("wtb,wtc->wbc", Zw, Zw) + np.asarray(penalty)[None]
    cross = np.einsum("wtb,wtn->wbn", Zw, yw)
    coef = np.linalg.solve(gram, cross)  # (n_windows, n_basis, n_neurons)
    features = coef.reshape(n_windows, -1)
    features = (features - features.mean(0)) / (features.std(0) + 1e-12)

    from sklearn.cluster import KMeans  # deferred: slow import

    labels = KMeans(n_clusters=n_maps, n_init=10, random_state=seed).fit(features).labels_
    one_hot = np.eye(n_maps)[labels]
    resp = (1.0 - soft_floor) * one_hot + soft_floor / n_maps
    resp = np.repeat(resp, window_bins, axis=0)[:n_time]
    return jnp.asarray(resp), labels
```

For `n_maps == 1` the model skips this and uses all-ones responsibilities. `scikit-learn` is
already a runtime dependency (`pyproject.toml` `dependencies`, used by
`oscillator_models.py:687`), imported lazily like there.

---

## E. Transition and initial-state M-step; occupancy ordering

```python
def _transition_matrix_map_estimate(
    pairwise_prob: Array, transition_prior: Array, previous: Array
) -> Array:
    """Row-normalised expected transition counts plus Dirichlet pseudo-counts.

    Mirrors ``switching_kalman.py:2275-2281``. A row with zero expected mass and a
    flat prior (a map never visited) keeps its previous row instead of 0/0.
    """
    expected = pairwise_prob.sum(axis=0) + (transition_prior - 1.0)
    row_sum = expected.sum(axis=1, keepdims=True)
    return jnp.where(row_sum > 0, expected / jnp.where(row_sum > 0, row_sum, 1.0), previous)
```

`transition_prior = get_transition_prior(concentration, stickiness, n_maps)`
(`contingency_belief.py:200-222`, entries `>= 1`). Initial map probability:
`gamma[0] / gamma[0].sum()` (`switching_kalman.py:2284-2287`).

**Occupancy ordering** (after EM/SGD, `n_maps > 1`): `perm = np.argsort(-occupancy, kind="stable")`
with `occupancy = smoothed_map_prob_.sum(0)`; apply `weights_[..., perm]`,
`transition_matrix_[perm][:, perm]`, `init_map_prob_[perm]`, `smoothed_map_prob_[:, perm]`,
`filtered_map_prob_[:, perm]`, `pairwise_map_prob_[:, perm][:, :, perm]`,
`_per_map_log_likelihoods[:, perm]`. Label 0 is always the most-occupied map; ties resolve by
original index. Every fitted quantity is permutation-equivariant, so two fits from
column-permuted initial responsibilities end identical to round-off
([phase 1 test](phase-1-static-maps-hmm.md#validation-slice)).

---

## F. Model class wiring

Class `MultiMapPlaceFieldModel(SGDFittableMixin)` in `multi_map_place_field.py`. Public
attribute names and shapes are the [model contract](shared-contracts.md#model-attributes).

```python
class MultiMapPlaceFieldModel(SGDFittableMixin):
    def __init__(
        self,
        n_maps: int,
        dt: float,
        *,
        penalty: float | ArrayLike = 1.0,
        transition_concentration: float = 1.0,
        transition_stickiness: float = 0.0,
        init_transition_diag: float = 0.99,
        init_window_bins: int = 250,
        max_newton_iter: int = 5,
        max_firing_rate_hz: float = 500.0,
        update_transition_matrix: bool = True,
        update_init_map_prob: bool = True,
        seed: int = 0,
    ) -> None:
        self.n_maps = validate_int(n_maps, "n_maps", positive=True)
        self.dt = validate_scalar(dt, "dt", positive=True)
        ...  # validate the rest with validate_scalar / validate_int; store
        # Fitted state (None before fit / from_parameters)
        self.weights_: Array | None = None
        self.transition_matrix_: Array | None = None
        self.init_map_prob_: Array | None = None
        self.smoothed_map_prob_: Array | None = None
        self.filtered_map_prob_: Array | None = None
        self.pairwise_map_prob_: Array | None = None
        self.switch_probability_: Array | None = None
        self.log_likelihood_: float | None = None
        self.log_likelihood_history_: list[float] = []
        self.converged_: bool | None = None
        self.n_neurons_: int | None = None
        self.n_basis_: int | None = None
        self._per_map_log_likelihoods: Array | None = None
        self._penalty: Array | None = None
        self._n_time: int = 0

    @property
    def _max_log_count(self) -> float:  # place_field_model.py:852-862
        return float(np.log(self.max_firing_rate_hz * self.dt))

    def _check_fitted(self, method_name: str) -> None:  # place_field_model.py:550-565
        if self.weights_ is None:
            raise NotFittedError(
                "Model has not been fitted. Call model.fit(design_matrix, spikes) "
                f"before {method_name}()."
            )
```

**Data validation** (`_validate_data(design_matrix, spikes) -> tuple[Array, Array]`):
`validate_finite_array("design_matrix", Z)` (`utils.py:1044`), 2-D; `validate_count_array(
spikes, "spikes", allow_empty=False)` (`utils.py:1016-1041`), 2-D `(n_time, n_neurons)`
(a 1-D vector is reshaped to one neuron, as `PlaceFieldModel.fit` does at `:1266-1267`);
equal `n_time`; on refit, `n_basis`/`n_neurons` must match the fitted shapes. Sets
`n_neurons_`, `n_basis_`, `_n_time`, `_penalty` (section C), `_transition_prior`.

**E-step / M-step / snapshot** (the `run_em` hooks; `em_driver.py:58-77`):

```python
    def _e_step(self, Z: Array, y: Array) -> float:
        ll_maps = per_map_log_likelihoods(Z, y, self.weights_, self.dt, self._max_log_count)
        post = hmm_forward_backward(self.init_map_prob_, self.transition_matrix_, ll_maps)
        self._per_map_log_likelihoods = ll_maps
        self.smoothed_map_prob_ = post.smoothed_prob
        self.filtered_map_prob_ = post.filtered_prob
        self.pairwise_map_prob_ = post.pairwise_prob
        self.log_likelihood_ = float(post.log_likelihood)
        # EM maximises the penalised objective; it is exact (HMM E-step, MAP M-steps)
        # and hence non-decreasing, which is what run_em's rollback checks assume.
        return self.log_likelihood_ + float(
            self._log_prior(self.weights_, self.transition_matrix_)
        )

    def _m_step(self, Z: Array, y: Array) -> None:
        gamma, xi = self.smoothed_map_prob_, self.pairwise_map_prob_
        if self.n_maps > 1:
            if self.update_transition_matrix:
                self.transition_matrix_ = _transition_matrix_map_estimate(
                    xi, self._transition_prior, self.transition_matrix_
                )
            if self.update_init_map_prob:
                self.init_map_prob_ = gamma[0] / gamma[0].sum()
        self.weights_ = self._update_weights(Z, y, gamma, self.weights_, self.max_newton_iter)

    def _update_weights(self, Z, y, gamma, old, max_iter) -> Array:
        occupancy = gamma.sum(axis=0)
        min_occ = minimum_state_occupancy(self.n_basis_)
        low = warn_low_occupancy_states(
            occupancy, min_occ, "MultiMapPlaceFieldModel M-step",
            "their weights kept their previous values",
        )
        new, n_fallbacks = update_map_weights(
            Z, y, gamma, self._penalty, old, self.dt, self._max_log_count, max_iter
        )
        _warn_newton_fallbacks(n_fallbacks, "MultiMapPlaceFieldModel M-step")
        if low:
            keep = jnp.asarray([k not in low for k in range(self.n_maps)])
            new = jnp.where(keep[None, None, :], new, old)
        return new

    def _snapshot(self) -> dict[str, object]:
        # Immutable JAX arrays and a float: referencing them is safe (em_driver.py:105-109).
        return {name: getattr(self, name) for name in _EM_STATE_ATTRS}

    def _restore(self, state: dict[str, object]) -> None:
        for name, value in state.items():
            setattr(self, name, value)


# Module-level, defined before the class: everything the EM snapshot must hold so a
# rollback restores parameters and posteriors together.
_EM_STATE_ATTRS = (
    "weights_", "transition_matrix_", "init_map_prob_", "smoothed_map_prob_",
    "filtered_map_prob_", "pairwise_map_prob_", "_per_map_log_likelihoods",
    "log_likelihood_",
)
```

**Initialisation** (`_initialize_parameters(Z, y, gamma0)`): weights from
`_intercept_matching_weights` then one `update_map_weights(..., max_iter=15)` pass;
`transition_matrix_ = make_discrete_transition_matrix(jnp.full(K, init_transition_diag), K)`
(`utils.py:1722-1759`); `init_map_prob_ = ones(K) / K`. `gamma0` is the user's
`initial_responsibilities` (validated: shape `(n_time, K)`, rows sum to 1 — reuse
`validate_probability_vector` per row is O(T) Python; instead check
`jnp.allclose(gamma0.sum(1), 1)` and non-negativity vectorised) or
[k-means](#d-k-means-initialisation-on-windowed-population-rate-maps); all-ones for K = 1.

**`fit`:**

```python
    def fit(
        self,
        design_matrix: ArrayLike,
        spikes: ArrayLike,
        *,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        initial_responsibilities: ArrayLike | None = None,
        verbose: bool = False,
    ) -> list[float]:
        Z, y = self._validate_data(design_matrix, spikes)
        gamma0 = self._initial_responsibilities(Z, y, initial_responsibilities)
        self._initialize_parameters(Z, y, gamma0)
        result = run_em(
            lambda: self._e_step(Z, y),
            lambda: self._m_step(Z, y),
            self._snapshot,
            self._restore,
            max_iter=max_iter,
            tol=tolerance,
            on_first_nonfinite="raise",
            logger=logger,
            on_iteration=(lambda i, ll, ch: logger.info(...)) if verbose else None,
        )
        self.log_likelihood_history_ = result.log_likelihoods
        self.converged_ = result.converged
        self._finalize_fit()
        return self.log_likelihood_history_
```

`run_em` defaults (`stop_on_decrease=True`, `decrease_tol=tol`) are right for an exact EM:
a decrease is a bug, not approximation noise. The snapshot holds parameters *and*
posteriors (like `PlaceFieldModel.fit`, `place_field_model.py:1335-1352`), so no
`refresh_after_restore`. `_finalize_fit()` = occupancy ordering (section E) +
`switch_probability_ = switch_probability(self.pairwise_map_prob_)` (section I).

`log_likelihood_history_` holds what the fitter maximised — the marginal log-likelihood
plus the log prior (weights penalty, Dirichlet pseudo-counts). `log_likelihood_` is the
unpenalised marginal log-likelihood of the final parameters. Docstrings must say so.

**SGD hooks** (`sgd_fitting.py:354-373` protocol; `fit_sgd` at `:506-535` pops the optimizer
settings and forwards the remaining kwargs to `_prepare_sgd_data`, `:384-397`):

```python
    _sgd_param_attrs = {
        "weights": "weights_",
        "transition_matrix": "transition_matrix_",
        "init_map_prob": "init_map_prob_",
    }

    def fit_sgd(self, design_matrix, spikes, *, initial_responsibilities=None,
                optimizer=None, num_steps=200, verbose=False, convergence_tol=None):
        return super().fit_sgd(
            design_matrix, spikes, initial_responsibilities=initial_responsibilities,
            optimizer=optimizer, num_steps=num_steps, verbose=verbose,
            convergence_tol=convergence_tol,
        )

    def _prepare_sgd_data(self, design_matrix, spikes, *, initial_responsibilities=None):
        Z, y = self._validate_data(design_matrix, spikes)
        if self.weights_ is None:  # cold start: same starting point as fit()
            gamma0 = self._initial_responsibilities(Z, y, initial_responsibilities)
            self._initialize_parameters(Z, y, gamma0)
        return (Z, y), {}

    @property
    def _n_timesteps(self) -> int:
        return self._n_time

    def _build_param_spec(self) -> tuple[dict, dict]:
        params = {"weights": self.weights_}
        spec = {"weights": UNCONSTRAINED}
        if self.n_maps > 1:  # a (1, 1) STOCHASTIC_ROW has no free parameter
            if self.update_transition_matrix:
                params["transition_matrix"] = self.transition_matrix_
                spec["transition_matrix"] = STOCHASTIC_ROW
            if self.update_init_map_prob:
                params["init_map_prob"] = self.init_map_prob_
                spec["init_map_prob"] = STOCHASTIC_ROW  # (K,) works: softmax over the last axis
        return params, spec

    def _sgd_loss_fn(self, params: dict, Z: Array, y: Array) -> Array:
        W = params["weights"]
        T = params.get("transition_matrix", self.transition_matrix_)
        pi = params.get("init_map_prob", self.init_map_prob_)
        ll_maps = per_map_log_likelihoods(Z, y, W, self.dt, self._max_log_count)
        _, log_lik = hmm_filter(pi, T, ll_maps)
        return -(log_lik + self._log_prior(W, T))

    def _finalize_sgd(self, Z: Array, y: Array) -> None:
        self._e_step(Z, y)  # installs posteriors and log_likelihood_
        self._finalize_fit()
```

The mixin stores `log_likelihood_history_` (per-step objective, `sgd_fitting.py:719`) and
`converged_` before calling `_finalize_sgd` (`:720-721`), consistent with the EM meaning
above. `STOCHASTIC_ROW` (`parameter_transforms.py:225-252`) requires strictly positive
entries (`:232`): the initial matrix from `init_transition_diag < 1` satisfies it.

**Prediction / scoring API:**

```python
    @classmethod
    def from_parameters(cls, weights, transition_matrix, init_map_prob, dt, **kwargs):
        """A model with known parameters (no data fit); posteriors are None."""
        # validate_transition_matrix (utils.py:1408-1457), validate_probability_vector
        # (utils.py:1460-1499), validate_finite_array on weights; set n_neurons_, n_basis_,
        # _penalty (from kwargs' penalty), _transition_prior.

    def predict_map_posterior(self, design_matrix, spikes) -> HMMPosterior:
        """E-step on new data with the fitted parameters (nothing stored)."""

    def score(self, design_matrix, spikes) -> float:
        """Marginal log-likelihood of held-out data (forward pass only), starting from
        ``init_map_prob_`` -- the same convention as PlaceFieldModel.score
        (place_field_model.py:1886-1979)."""

    def predict_rate_map(self, design_matrix, map_index=None, neuron_index=None) -> np.ndarray:
        """Rates in Hz, ``exp(Z @ weights)``: (n_points, n_neurons, n_maps) or the
        requested slice. Basis-agnostic: pass ``evaluate_basis(grid, basis_info)``
        (place_field_model.py:160-190) or ``basis.eigvecs`` (graph_place_field.py:58-92)."""

    def viterbi_path(self) -> Array:
        """hmm_viterbi(init_map_prob_, transition_matrix_, _per_map_log_likelihoods)
        (utils.py:1769-1824); NotFittedError if no data has been fitted."""

    @property
    def n_free_params(self) -> int:
        """K * n_neurons * n_basis + K (K - 1) + (K - 1) (only the updated blocks)."""

    def bic(self) -> float:  # -2 log_likelihood_ + n_free_params * log(n_time)
    def aic(self) -> float:
```

`n_free_params` counts nominal parameters; the penalty makes the effective number smaller,
so BIC is conservative (favours smaller K). Documented in the docstring; see
[overview open question 2](overview.md#open-questions).

**Model selection helper** (module-level):

```python
def fit_over_n_maps(
    design_matrix, spikes, dt, n_maps_grid, *, held_out=None, **model_kwargs
) -> list[dict]:
    """Fit one model per K; rows {"n_maps", "log_likelihood", "bic", "held_out_log_likelihood", "model"}."""

def select_n_maps(rows, criterion="bic", tolerance_nats=1.0) -> int:
    """Smallest K whose criterion is within ``tolerance_nats`` of the best
    (min BIC / max held-out LL): a parsimony rule so near-ties pick the simpler model."""
```

---

## G. Two-map session simulator

**Where:** `src/state_space_practice/simulate_data.py` (in the mypy `files` list —
annotate fully).

1. Lift the lawnmower trajectory of `simulate_2d_moving_place_field`
   (`simulate_data.py:138-166`) into `_lawnmower_trajectory(n_time, dt, arena_size,
   speed, rng) -> np.ndarray`, called from `simulate_2d_moving_place_field` in place of
   those lines. The only RNG draw in that block is `rng.normal(0, step_size * 0.5,
   (n_time, 2))` at `:164`; keeping the draw inside the helper in the same order leaves
   `simulate_2d_moving_place_field` **bit-identical** (verified by the baseline-capture task
   in phase 1).
2. Add:

```python
def gaussian_place_field_rate(
    position: np.ndarray,
    centers: np.ndarray,
    peak_rate: float,
    background_rate: float,
    place_field_sigma: float,
) -> np.ndarray:
    """Rates (Hz) of Gaussian place fields, shape (n_points, n_neurons).

    position : (n_points, 2); centers : (n_neurons, 2).
    """
    dist_sq = np.sum((position[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return background_rate + (peak_rate - background_rate) * np.exp(
        -dist_sq / (2.0 * place_field_sigma**2)
    )


def simulate_multi_map_session(
    n_neurons: int = 8,
    n_maps: int = 2,
    total_time: float = 300.0,
    dt: float = 0.02,
    arena_size: float = 80.0,
    speed: float = 20.0,
    peak_rate: float = 30.0,
    background_rate: float = 1.0,
    place_field_sigma: float = 10.0,
    dwell_time: float = 20.0,
    map_separation: float = 1.0,
    n_interior_knots: int = 4,
    transition_covariate: np.ndarray | None = None,
    transition_gain: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Population spikes from ``n_maps`` place-field maps selected by a hidden Markov state.

    At each bin the shared discrete state ``s_t`` picks the map; neuron ``n`` fires
    as Poisson(rate_{n, s_t}(position_t) * dt). Map 0 centres are uniform in the
    arena; map k centres are ``c_0 + map_separation * (c'_k - c_0)`` with
    independent ``c'_k`` -- ``map_separation=1`` is a full remap, ``0`` makes every
    map identical (spikes carry no information about ``s_t``). The chain switches
    with hazard ``h_t = min(dt / dwell_time * exp(transition_gain * covariate_t), 0.5)``
    to a uniformly chosen other map (``transition_covariate`` is ``(n_time,)``,
    standardised by the caller; ``None`` gives a constant hazard).

    Returns
    -------
    dict with ``time``, ``position`` (n_time, 2), ``spikes`` (n_time, n_neurons) int,
    ``states`` (n_time,) int, ``true_rate`` (n_time, n_neurons) Hz, ``centers``
    (n_neurons, 2, n_maps), ``transition_matrix`` (n_maps, n_maps) [constant-hazard
    case], ``design_matrix`` (n_time, n_basis) from ``build_2d_spline_basis``,
    ``basis_info``, ``dt``.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_time = int(total_time / dt)
    position = _lawnmower_trajectory(n_time, dt, arena_size, speed, rng)

    margin = place_field_sigma
    base = rng.uniform(margin, arena_size - margin, (n_neurons, 2))
    alt = rng.uniform(margin, arena_size - margin, (n_neurons, 2, n_maps))
    centers = base[:, :, None] + map_separation * (alt - base[:, :, None])
    centers[:, :, 0] = base

    hazard = np.full(n_time, dt / dwell_time)
    if transition_covariate is not None:
        hazard = hazard * np.exp(transition_gain * np.asarray(transition_covariate))
    hazard = np.minimum(hazard, 0.5)
    states = np.empty(n_time, dtype=int)
    states[0] = rng.integers(n_maps)
    switch = rng.random(n_time) < hazard
    other = rng.integers(1, max(n_maps, 2), n_time)  # offset to a different map
    for t in range(1, n_time):  # sequential: the destination depends on the current map
        states[t] = (states[t - 1] + other[t]) % n_maps if switch[t] else states[t - 1]

    rate_per_map = np.stack(
        [
            gaussian_place_field_rate(
                position, centers[:, :, k], peak_rate, background_rate, place_field_sigma
            )
            for k in range(n_maps)
        ],
        axis=-1,
    )  # (n_time, n_neurons, n_maps)
    true_rate = np.take_along_axis(rate_per_map, states[:, None, None], axis=-1)[..., 0]
    spikes = rng.poisson(true_rate * dt)
    design_matrix, basis_info = build_2d_spline_basis(position, n_interior_knots)
    h = dt / dwell_time
    transition_matrix = np.full((n_maps, n_maps), h / max(n_maps - 1, 1))
    np.fill_diagonal(transition_matrix, 1.0 - h)
    return {
        "time": np.arange(n_time) * dt, "position": position, "spikes": spikes,
        "states": states, "true_rate": true_rate, "centers": centers,
        "transition_matrix": transition_matrix, "design_matrix": design_matrix,
        "basis_info": basis_info, "dt": dt,
    }
```

The Python loop over `n_time` is a simulator's sequential Markov sampling (15 k iterations,
milliseconds), not a library array op. For `n_maps == 1` the chain never switches.

---

## H. Exact path-enumeration oracle for tests

**Where:** test-side only (`tests/test_multi_map_place_field.py` and the
`hmm_forward_backward` tests in `tests/test_utils.py`; a shared helper can live in
`tests/oracles.py` next to `_logsumexp` at `oracles.py:437-442`). Same pattern as
`_all_paths` / `path_posterior` in `tests/test_oracle_switching_point_process.py:125-126,
238-261`, but with no continuous state the path weight is closed form, so this is an
**exact** reference for the HMM E-step.

```python
def enumerate_hmm_posterior(initial_probs, transition_matrix, log_likelihoods) -> dict:
    """Exact HMM posteriors by summing over every one of the K**T discrete paths."""
    ll = np.asarray(log_likelihoods)
    n_time, n_states = ll.shape
    trans = np.asarray(transition_matrix)
    if trans.ndim == 2:
        trans = np.broadcast_to(trans, (n_time - 1, n_states, n_states))
    paths = np.array(list(itertools.product(range(n_states), repeat=n_time)))  # (P, T)
    t_idx = np.arange(n_time)
    log_init = np.log(np.asarray(initial_probs)[paths[:, 0]])  # (P,)
    # log Z[s_t, s_{t+1}] for every step, (P, T - 1); cumulative prefix priors below
    log_step = np.log(trans[np.arange(n_time - 1)[None, :], paths[:, :-1], paths[:, 1:]])
    log_lik_prefix = np.cumsum(ll[t_idx[None, :], paths], axis=1)  # (P, T)
    log_w = log_init + log_step.sum(1) + log_lik_prefix[:, -1]
    log_z = _logsumexp(log_w)
    w = np.exp(log_w - log_z)
    smoothed = np.zeros((n_time, n_states))
    pairwise = np.zeros((n_time - 1, n_states, n_states))
    filtered = np.zeros((n_time, n_states))
    for t in range(n_time):
        np.add.at(smoothed[t], paths[:, t], w)
        # Filtered marginal: weight of the prefix s_{1:t+1} with y_{1:t+1}. Every
        # prefix appears once per continuation with the same weight, so
        # normalising over all P rows already normalises over prefixes.
        log_w_t = log_init + log_step[:, :t].sum(1) + log_lik_prefix[:, t]
        w_t = np.exp(log_w_t - _logsumexp(log_w_t))
        np.add.at(filtered[t], paths[:, t], w_t)
    for t in range(n_time - 1):
        np.add.at(pairwise[t], (paths[:, t], paths[:, t + 1]), w)
    return {"log_lik": float(log_z), "smoothed": smoothed, "pairwise": pairwise,
            "filtered": filtered, "viterbi": paths[np.argmax(log_w)]}
```

`_logsumexp` is the NumPy helper at `tests/oracles.py:437-442`; `itertools` is already
imported there (`:37`). Use `T <= 6`, `K <= 3` (729 paths). Compare at `rtol=1e-10` with
`_close` (`tests/test_invariances.py:82-88`) — these are identities.

---

## I. Behavioural alignment helpers

**Where:** `multi_map_place_field.py`, module-level, NumPy (post-hoc analysis, not jitted).

```python
def switch_probability(pairwise_prob: ArrayLike) -> Array:
    """``P(s_{t+1} != s_t | y)`` per transition, shape (n_time - 1,): one minus the
    trace of each pairwise slice."""
    pairwise_prob = jnp.asarray(pairwise_prob)
    return 1.0 - jnp.trace(pairwise_prob, axis1=1, axis2=2)


def lagged_correlation(x: ArrayLike, covariate: ArrayLike, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation of ``x[t]`` with ``covariate[t + lag]`` for ``lag`` in
    ``[-max_lag, max_lag]``. Returns ``(lags, correlations)``. ``x`` and ``covariate``
    are truncated to a common length; ``switch_probability_`` has ``n_time - 1``
    entries indexed by the *source* bin ``t``, so pass ``covariate[:-1]`` to align
    on the source bin or ``covariate[1:]`` on the destination bin."""


def circular_shift_null(x, covariate, n_shuffles: int, rng: np.random.Generator, min_shift: int) -> np.ndarray:
    """Zero-lag correlations after circularly shifting ``x`` by random offsets in
    ``[min_shift, n - min_shift)``: a null that preserves both autocorrelations."""


def covariate_by_map(smoothed_prob: ArrayLike, covariate: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Posterior-weighted mean and standard deviation of ``covariate`` in each map,
    ``mean_k = sum_t gamma_tk c_t / sum_t gamma_tk``; both shape (n_maps,)."""
```

These operationalise Low et al.'s post-hoc speed correlation. Phase 1b tests the same
relationship as a transition *parameter*.

---

## J. Tensor-product spline difference penalty

```python
def tensor_spline_difference_penalty(n_per_dim: int, order: int = 2) -> np.ndarray:
    """P-spline difference penalty (Eilers & Marx 1996) on an ``n x n`` tensor-product
    coefficient grid, shape (n_per_dim**2, n_per_dim**2).

    ``kron(D'D, I) + kron(I, D'D)`` with ``D`` the ``order``-th difference matrix. The
    two terms make the matrix invariant to whether the grid is flattened x-major or
    y-major, so it matches ``build_2d_spline_basis`` (place_field_model.py:79-157)
    without depending on patsy's ``te()`` column order. ``n_per_dim`` is
    ``n_interior_knots + 3`` for its cubic B-splines (``basis_info["n_basis"]`` is
    ``n_per_dim**2``). Scale it (``lam * P``) before passing it as ``penalty``.
    """
    D = np.diff(np.eye(n_per_dim), n=order, axis=0)
    P1 = D.T @ D
    eye = np.eye(n_per_dim)
    return np.kron(P1, eye) + np.kron(eye, P1)
```

Check: a coefficient grid linear in both indices has zero second differences → quadratic
form exactly 0; a bump has a positive form. A pure ridge is `lam * np.eye(n_basis)`.

---

## K. Slow per-map drift — sketch, not scheduled

Model: `w_{n,k,t} = w_{n,k,t-1} + eps`, `eps ~ N(0, Q_k)` per map (map-specific process
noise), shared discrete `s_t`; spikes from block `s_t` only. Per neuron the continuous state
is the stacked `n_maps * n_basis` vector; neurons are independent given `s_t`
(shared design, per-neuron weights), so the problem is block-diagonal across neurons exactly
as `BlockDiagonalStructure` describes (`point_process_kalman.py:148-200`), and the
per-neuron block itself is block-diagonal across maps *conditional on the discrete path*
(the observation at `t` touches only block `s_t`; predictions are independent per map).

**How the GPB collapse composes with the block structure.** The switching filter's
collapse per destination state `j` (`collapse_gaussian_mixture_per_discrete_state`,
imported at `switching_point_process.py:123`) mixes Gaussians with different means:
`Cov = sum_i w_i P_i + sum_i w_i (m_i - m)(m_i - m)'`. The spread term is **not**
block-diagonal across maps — the collapse correlates a neuron's map-0 and map-1 weights even
though the generative model never does. Three options, none evidenced yet:

1. keep the full `(K n_basis)^2` per-neuron covariance (exact GPB1);
2. drop the cross-map spread terms after every collapse ("block-GPB" approximation; the
   marginal per-map covariances stay exact, cross-map correlations are discarded);
3. treat the inactive maps' weights as frozen within a dwell (no diffusion while
   unobserved) — changes the model, not just the inference.

**Memory** (float64, per-neuron blocks, storing state-conditional smoother output as the
switching models do, `switching_point_process.py:2996-3004`), for `T = 150 000` bins
(10 min at 4 ms), `N = 100`, `B = 36`, `K = 3`:

| Quantity | Floats | Bytes |
| --- | --- | --- |
| static-map model (phase 1): `(T, K)` likelihoods + posteriors + `(T-1, K, K)` pairwise | 2.3e6 | 18 MB |
| `PlaceFieldModel` today, one map: `(N, T, B, B)` smoother covariance | 1.9e10 | 155 GB |
| option 1, state-conditional covariances `(N, T, (KB)^2, K)` | 5.2e11 | 4.2 TB |
| option 2, per-map blocks `(N, T, K, B, B, K)` | 1.7e11 | 1.4 TB |
| option 2 with collapsed (marginal) posterior only `(N, T, K, B, B)` | 5.8e10 | 470 GB |

Even the single-map `PlaceFieldModel` is only practical at `N ~ 10-30`, `dt = 20 ms`,
`T ~ 3e4` (then `(N, T, B, B)` is 6-19 GB); the drift model multiplies that by `K` to `K^2`.
It therefore needs (a) segment-wise processing — sessions or chunks with boundary
conditions, which is what `docs/plans/masks-and-multi-sequence/` provides — and (b) a
decision between options 1-3 backed by a small exact-oracle study (the `T <= 5`
path-enumeration harness in `tests/test_oracle_switching_point_process.py` extends
directly). Neither exists yet, so drift is **not planned here**; see
[overview → Deliberately not in this plan](overview.md#scope-and-dependency-policy) for the
trigger.

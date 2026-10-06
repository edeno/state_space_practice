# Designs — per-component code and rationale

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Sections

- [A. Transition-stack resolution and in-scan step](#a-transition-stack-resolution-and-in-scan-step)
- [B. Gaussian filter integration](#b-gaussian-filter-integration)
- [C. GPB1 smoother integration](#c-gpb1-smoother-integration)
- [D. Viterbi integration](#d-viterbi-integration)
- [E. Logit M-step](#e-logit-m-step)
- [F. Oracle extension](#f-oracle-extension)
- [G. Phase-2 collapse approximation and exact grid oracle](#g-phase-2-collapse-approximation-and-exact-grid-oracle)
- [H. Relabelling transform](#h-relabelling-transform)
- [I. Simulators](#i-simulators)
- [J. Sibling integration notes](#j-sibling-integration-notes)

Shapes follow [shared-contracts](shared-contracts.md#parametrisation-and-layout-contract).
Code below is the intended implementation, not pseudocode; adapt names to the
surrounding file only where noted.

## A. Transition-stack resolution and in-scan step

`src/state_space_practice/discrete_transitions.py` (new). The three softmax
helpers are moved verbatim from `contingency_belief.py:84-130` and are not
repeated here.

```python
"""Covariate- and state-dependent transition matrices for switching models.

The discrete transition probabilities of a switching state-space model are a
row-wise softmax of logits that are linear in observed covariates ``u_t``
(input-output HMM, Bengio & Frasconi 1995) and, optionally, in the continuous
latent state ``x_{t-1}`` (recurrent SLDS, Linderman et al. 2017)::

    logit_t[i, j] = eta[i, j] + Gamma[:, i, j] . u_t + W[:, i, j] . x_{t-1}

with the last state as the reference column (logit 0) of every row. ``eta`` is
the centered-softmax inverse of a row-stochastic baseline matrix, so the
baseline matrix every switching filter already takes stays the parameter the
caller passes and learns.

Shapes: ``eta (S, S-1)``, ``Gamma (n_features, S, S-1)``,
``W (n_cont_states, S, S-1)``, ``u (n_time, n_features)``; the realised stack is
``(n_time, S, S)`` with entry ``t`` the matrix of the step into time ``t``
(entry 0 is the baseline and is never read).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from state_space_practice.utils import contains_tracer


def transition_logits_stack(
    baseline_logits: Array,
    transition_covariates: Array | None,
    transition_weights: Array | None,
    n_time: int,
) -> Array:
    """Baseline plus covariate logits for every time step.

    Parameters
    ----------
    baseline_logits : Array, shape (n_discrete_states, n_discrete_states - 1)
    transition_covariates : Array or None, shape (n_time, n_features)
        Row ``t`` drives the transition into time ``t``; row 0 is never read.
    transition_weights : Array or None, shape (n_features, n_discrete_states, n_discrete_states - 1)
    n_time : int

    Returns
    -------
    Array, shape (n_time, n_discrete_states, n_discrete_states - 1)
    """
    logits = jnp.broadcast_to(baseline_logits, (n_time, *baseline_logits.shape))
    if transition_covariates is None:
        return logits
    return logits + jnp.einsum("tk,kij->tij", transition_covariates, transition_weights)


def state_dependent_logits(
    base_logits_t: Array,
    transition_state_weights: Array,
    prev_state_cond_mean: Array,
) -> Array:
    """Add the continuous-state term for one step (plug-in at the collapsed mean).

    Parameters
    ----------
    base_logits_t : Array, shape (n_discrete_states, n_discrete_states - 1)
    transition_state_weights : Array, shape (n_cont_states, n_discrete_states, n_discrete_states - 1)
    prev_state_cond_mean : Array, shape (n_cont_states, n_discrete_states)
        ``E[x_{t-1} | S_{t-1} = i, y_{1:t-1}]`` in column ``i``.

    Returns
    -------
    Array, shape (n_discrete_states, n_discrete_states - 1)
    """
    return base_logits_t + jnp.einsum(
        "di,dij->ij", prev_state_cond_mean, transition_state_weights
    )


def transition_matrix_stack(
    discrete_transition_matrix: Array,
    transition_covariates: Array | None = None,
    transition_weights: Array | None = None,
    transition_state_weights: Array | None = None,
    state_cond_means: Array | None = None,
    *,
    n_time: int | None = None,
) -> Array | None:
    """Realised row-stochastic transition matrices, or None on the fixed path.

    Returns None when neither covariates nor state weights are given, so a
    caller can keep its fixed-matrix code path untouched. Otherwise returns the
    ``(n_time, S, S)`` stack; entry ``t`` uses ``transition_covariates[t]`` and
    ``state_cond_means[t - 1]`` (entry 0 is the baseline matrix).
    """
    if transition_covariates is None and transition_state_weights is None:
        return None
    if n_time is None:
        n_time = (
            state_cond_means.shape[0]
            if state_cond_means is not None
            else transition_covariates.shape[0]
        )
    eta = centered_softmax_inverse(discrete_transition_matrix)
    logits = transition_logits_stack(
        eta, transition_covariates, transition_weights, n_time
    )
    if transition_state_weights is not None:
        state_part = jnp.einsum(
            "tdi,dij->tij", state_cond_means[:-1], transition_state_weights
        )
        logits = logits.at[1:].add(state_part)
    return centered_softmax(logits)


def resolve_scan_transitions(
    discrete_transition_matrix: Array,
    transition_covariates: Array | None,
    transition_weights: Array | None,
    transition_state_weights: Array | None,
    n_time: int,
) -> tuple[Array | None, Callable[[Array | None, Array], Array]]:
    """Per-step scan input and the in-scan rule that turns it into ``T_t``.

    Returns ``(per_step, transition_at_step)``. ``per_step`` is scanned over
    alongside ``obs[1:]``: None on the fixed path (so ``lax.scan`` sees a None
    leaf and the fixed matrix stays a closed-over constant -- the trace is
    identical to the one without this machinery), ``stack[1:]`` of
    probabilities on the covariate-only path, ``logits[1:]`` on the
    state-dependent path (the softmax must wait for the previous collapsed
    mean, which lives in the scan carry).
    ``transition_at_step(per_step_t, prev_state_cond_mean) -> (S, S)``.
    """
    if transition_covariates is None and transition_state_weights is None:
        return None, lambda _unused, _prev_mean: discrete_transition_matrix
    eta = centered_softmax_inverse(discrete_transition_matrix)
    logits = transition_logits_stack(
        eta, transition_covariates, transition_weights, n_time
    )
    if transition_state_weights is None:
        return centered_softmax(logits)[1:], lambda z_t, _prev_mean: z_t
    weights = transition_state_weights
    return logits[1:], lambda base_t, prev_mean: centered_softmax(
        state_dependent_logits(base_t, weights, prev_mean)
    )


def validate_transition_inputs(
    discrete_transition_matrix: Array,
    transition_covariates: Array | None,
    transition_weights: Array | None,
    transition_state_weights: Array | None,
    n_time: int,
    n_cont_states: int,
) -> None:
    """Host-side checks of the transition options; a no-op on tracers."""
    if (transition_covariates is None) != (transition_weights is None):
        raise ValueError(
            "transition_covariates and transition_weights must both be given or "
            "both be None."
        )
    if transition_covariates is None and transition_state_weights is None:
        return
    if contains_tracer(
        discrete_transition_matrix,
        transition_covariates,
        transition_weights,
        transition_state_weights,
    ):
        return
    z = np.asarray(discrete_transition_matrix)
    n_states = z.shape[0]
    if np.any(z == 0.0):
        raise ValueError(
            "discrete_transition_matrix has an exact zero: a softmax row is "
            "strictly positive, so covariate- or state-dependent transitions "
            "cannot keep a structural zero (centered_softmax_inverse would lift "
            "it to 1e-10). Use the fixed-matrix path for forbidden transitions."
        )
    if transition_covariates is not None:
        u = np.asarray(transition_covariates)
        if u.ndim != 2 or u.shape[0] != n_time:
            raise ValueError(
                f"transition_covariates must have shape (n_time={n_time}, "
                f"n_features), got {u.shape}."
            )
        if not np.all(np.isfinite(u)):
            raise ValueError("transition_covariates must be finite.")
        g = np.asarray(transition_weights)
        expected = (u.shape[1], n_states, n_states - 1)
        if g.shape != expected:
            raise ValueError(
                f"transition_weights must have shape {expected} "
                f"(n_features, n_discrete_states, n_discrete_states - 1), got {g.shape}."
            )
        if not np.all(np.isfinite(g)):
            raise ValueError("transition_weights must be finite.")
    if transition_state_weights is not None:
        w = np.asarray(transition_state_weights)
        expected = (n_cont_states, n_states, n_states - 1)
        if w.shape != expected:
            raise ValueError(
                f"transition_state_weights must have shape {expected} "
                f"(n_cont_states, n_discrete_states, n_discrete_states - 1), got {w.shape}."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError("transition_state_weights must be finite.")
```

Phase 1a ships everything above except the `transition_state_weights` /
`state_cond_means` arguments and `state_dependent_logits` (phase 2a adds them;
in phase 1a `resolve_scan_transitions` has no `transition_state_weights`
parameter and its first branch returns after the covariate case).

Rationale: the `None` leaf in `xs` is what keeps the fixed path a single,
unchanged trace (checked: `jax.make_jaxpr` of a scan over `(obs, None)` has
`num_consts=1` for the closed-over matrix and no per-step transition input).
Probabilities, not logits, are precomputed on the covariate-only path so the
in-scan body does one array read; the state-dependent path must defer the
softmax because the previous collapsed mean is in the carry.

## B. Gaussian filter integration

`switching_kalman.py`. The jitted body becomes `_switching_kalman_filter_jit`;
the public `switching_kalman_filter` becomes a thin validating wrapper with the
same signature plus the new keywords (the structure `switching_point_process_filter`
/ `_switching_point_process_filter_jit` already has, `switching_point_process.py:2415-2470`).
Callers that traced through `switching_kalman_filter` (the SGD losses) still
trace through the jitted core; a caller may still wrap it in `jax.jit`.

```python
def switching_kalman_filter(
    init_state_cond_mean, init_state_cond_cov, init_discrete_state_prob, obs,
    discrete_transition_matrix, continuous_transition_matrix, process_cov,
    measurement_matrix, measurement_cov,
    *,
    transition_covariates: jax.Array | None = None,
    transition_weights: jax.Array | None = None,
):
    """...existing docstring...

    transition_covariates : jax.Array or None, shape (n_time, n_features), keyword-only
        Design matrix of the transition covariates; row ``t`` drives the step
        into time ``t`` (row 0 is never read). With ``transition_weights`` the
        discrete transition matrix becomes time-varying,
        ``T_t = softmax_rows(centered_softmax_inverse(Z) + u_t @ Gamma)``
        (see :mod:`state_space_practice.discrete_transitions`). ``None`` keeps
        the fixed matrix ``Z`` and the code path unchanged.
    transition_weights : jax.Array or None, shape (n_features, n_discrete_states, n_discrete_states - 1), keyword-only
        Covariate logit weights ``Gamma``; the last discrete state is the
        reference column. Both-or-neither with ``transition_covariates``.
    """
    validate_transition_inputs(
        discrete_transition_matrix, transition_covariates, transition_weights,
        None, n_time=jnp.shape(obs)[0], n_cont_states=jnp.shape(init_state_cond_mean)[0],
    )
    return _switching_kalman_filter_jit(
        init_state_cond_mean, ..., measurement_cov,
        transition_covariates=transition_covariates,
        transition_weights=transition_weights,
    )


@jax.jit
def _switching_kalman_filter_jit(..., *, transition_covariates=None, transition_weights=None):
    ...
    def _step(carry, xs):
        obs_t, transition_input = xs                       # was: obs_t
        (prev_state_cond_filter_mean, ..., prev_support) = carry
        ...  # pair update unchanged
        transition_t = transition_at_step(transition_input, prev_state_cond_filter_mean)
        (...) = _update_discrete_state_probabilities(
            pair_cond_marginal_log_likelihood,
            transition_t,                                  # was: discrete_transition_matrix
            prev_filter_discrete_prob,
            prev_support,
        )
        ...  # unchanged

    per_step, transition_at_step = resolve_scan_transitions(
        discrete_transition_matrix, transition_covariates, transition_weights,
        obs.shape[0],
    )
    ... = jax.lax.scan(_step, (...), (obs[1:], per_step))   # was: obs[1:]
```

Everything else in the body (`switching_kalman.py:990-1076`) is unchanged.
Phase 2a adds `transition_state_weights` to the wrapper, the core and the
`resolve_scan_transitions` call; nothing else in the body changes because the
state-dependent rule already receives `prev_state_cond_filter_mean`.

## C. GPB1 smoother integration

`switching_kalman_smoother` (`switching_kalman.py:1386-1683`) stays `@jax.jit`
(its shape checks are static; the values it consumes were validated by the
filter that produced them). The stack is resolved once, outside the scan, from
`filter_mean` (which is exactly the per-step collapsed mean the filter used,
so the phase-2 reconstruction is bit-identical to the filter's in-scan value):

```python
@jax.jit
def switching_kalman_smoother(
    filter_mean, filter_cov, filter_discrete_state_prob, process_cov,
    continuous_transition_matrix, discrete_state_transition_matrix,
    *,
    transition_covariates=None, transition_weights=None,   # phase 2a adds transition_state_weights=None
):
    stack = transition_matrix_stack(
        discrete_state_transition_matrix, transition_covariates, transition_weights,
        # phase 2a: transition_state_weights, state_cond_means=filter_mean,
        n_time=filter_mean.shape[0],
    )

    def _step(carry, args):
        (next_state_cond_smoother_mean, next_state_cond_smoother_cov,
         next_smoother_discrete_prob) = carry
        state_cond_filter_mean, state_cond_filter_cov, filter_discrete_prob, transition_next = args
        transition_next = (
            discrete_state_transition_matrix if transition_next is None else transition_next
        )
        ...
        (...) = _update_smoother_discrete_probabilities(
            filter_discrete_prob, transition_next, next_smoother_discrete_prob,
        )
        ...  # unchanged

    xs = (
        filter_mean[:-1], filter_cov[:-1], filter_discrete_state_prob[:-1],
        None if stack is None else stack[1:],       # T_{t+1}: the step S_t -> S_{t+1}
    )
    ... = jax.lax.scan(_step, init_carry, xs, reverse=True)
```

`transition_next is None` is a Python-level check on the scan-body argument,
resolved at trace time (the leaf is `None` only when `stack is None`).

`switching_choice_smoother` (`switching_choice.py:544-659`) resolves its stack
the same way and adds `None if stack is None else stack[1:]` to its `xs`
(`:626-631`), passing `discrete_state_transition_matrix=transition_next` in its
per-step call (`:601-608`). Because that inner call is a two-step smoother whose
single backward step is the step at `t`, a fixed `(S, S)` matrix per call is
exactly right.

## D. Viterbi integration

`switching_kalman_viterbi` (`switching_kalman.py:1079-1230`; not jitted, so
`validate_transition_inputs` runs directly next to the existing prior check at
`:1104-1117`):

```python
per_step, transition_at_step = resolve_scan_transitions(
    discrete_transition_matrix, transition_covariates, transition_weights, obs.shape[0]
)
stack = transition_matrix_stack(discrete_transition_matrix, transition_covariates,
                                transition_weights, n_time=obs.shape[0])

def _step(carry, xs):
    obs_t, transition_input = xs
    prev_mean, prev_cov, prev_prob, prev_support = carry
    ...
    transition_t = transition_at_step(transition_input, prev_mean)
    (...) = _update_discrete_state_probabilities(pair_cond_log_lik, transition_t, prev_prob, prev_support)
    ...

_, pair_log_liks = jax.lax.scan(_step, (...), (obs[1:], per_step))

# backward: pair_log_liks[t] scores observation t+1, i.e. the step S_t -> S_{t+1} = stack[t+1]
log_trans = zero_preserving_log(
    discrete_transition_matrix if stack is None else stack[1:]
)  # (K, K) or (T-1, K, K)

def _viterbi_backward(best_next_score, t):
    log_trans_t = log_trans if stack is None else log_trans[t]
    scores = log_trans_t + pair_log_liks[t] + best_next_score[None, :]
    ...
```

Phase 2a: the forward pass needs no change beyond the extra keyword; the
backward pass reconstructs the stack from the forward pass's collapsed means, so
the forward scan also stacks `state_cond_mean` as an output and the stack is
built after the scan with `state_cond_means=jnp.concatenate([first_state_cond_mean[None], means_rest])`.

`utils.hmm_viterbi` (`utils.py:1769-1824`): accept `transition_matrix` of
ndim 2 or 3 (entry `t` = the step into time `t`). A 3-D stack must have
shape `(T, K, K)` where `T = log_likelihoods.shape[0]`; validate this and
ignore row 0. Edge-stack callers such as multi-map pad their `(T-1, K, K)`
stack at the call boundary; do not infer two indexing conventions from shape:

```python
log_transition_matrix = zero_preserving_log(transition_matrix)
time_varying = log_transition_matrix.ndim == 3

def _backward_step(best_next_score, t):
    log_trans_t = log_transition_matrix[t + 1] if time_varying else log_transition_matrix
    scores = log_trans_t + best_next_score + log_likelihoods[t + 1]
    ...
```

## E. Logit M-step

`discrete_transitions.py`, continued. `dirichlet_neg_log_likelihood` moves
verbatim from `contingency_belief.py:225-278` (its `l2_penalty` applies to
`coefficients[1:]`, i.e. every non-intercept coefficient row — that is exactly
`Gamma` and `W`).

```python
@jax.jit
def optimize_transition_rows(
    x0_all: Array,
    response_all: Array,
    alpha_all: Array,
    design_all: Array,
    l2_penalty: Array,
) -> Array:
    """BFGS-optimise the transition coefficients of every source state.

    Generalises the contingency-belief optimizer to a design matrix per source
    state, which the recurrent model needs (row ``i`` regresses on
    ``E[x | S = i]``).

    Parameters
    ----------
    x0_all : Array, shape (n_states, n_coef * (n_states - 1))
    response_all : Array, shape (n_states, n_samples, n_states)
        ``xi[t, i, :]`` for source state ``i``.
    alpha_all : Array, shape (n_states, n_states)
    design_all : Array, shape (n_states, n_samples, n_coef)
    l2_penalty : Array, shape ()

    Returns
    -------
    Array, shape (n_states, n_coef * (n_states - 1))
    """

    def _optimize_one_row(x0_flat, response_row, alpha_row, design_row):
        def loss(c):
            return dirichlet_neg_log_likelihood(
                c, design_row, response_row, alpha_row, l2_penalty
            )

        return jax.scipy.optimize.minimize(
            loss, x0_flat, method="BFGS", options={"maxiter": 50}
        ).x

    return jax.vmap(_optimize_one_row)(x0_all, response_all, alpha_all, design_all)


def maximize_transition_coefficients(
    smoother_joint_discrete_state_prob: Array,
    discrete_transition_matrix: Array,
    transition_covariates: Array | None = None,
    transition_weights: Array | None = None,
    transition_state_weights: Array | None = None,
    state_cond_smoother_means: Array | None = None,
    *,
    transition_prior: Array | None = None,
    l2_penalty: float = 1e-5,
) -> tuple[Array, Array | None, Array | None]:
    """M-step of the transition logits from the smoothed pair probabilities.

    Maximises ``sum_t sum_ij xi[t, i, j] log T_t[i, j]`` (with the Dirichlet-style
    pseudo-counts ``transition_prior - 1`` and an L2 penalty on the non-intercept
    coefficients) over ``eta`` (returned as a row-stochastic matrix), ``Gamma``
    and ``W`` by per-row BFGS. Under the recurrent model the design uses the
    smoothed state-conditional mean ``E[x_t | S_t = i, y_{1:T}]`` in place of
    ``x_t`` (the same collapsed-mean plug-in the filter uses). Source states
    with fewer expected transitions than ``minimum_state_occupancy(n_coef)``
    keep their previous coefficients and are reported by
    ``warn_low_occupancy_states``.

    Parameters
    ----------
    smoother_joint_discrete_state_prob : Array, shape (n_time - 1, S, S)
        ``xi[t, i, j] = P(S_t = i, S_{t+1} = j | y_{1:T})``.
    discrete_transition_matrix : Array, shape (S, S)
        Current baseline; its centered-softmax inverse is the intercept row.
    transition_covariates : Array or None, shape (n_time, n_features)
        Row ``t + 1`` pairs with ``xi[t]``.
    transition_weights : Array or None, shape (n_features, S, S - 1)
    transition_state_weights : Array or None, shape (n_cont_states, S, S - 1)
    state_cond_smoother_means : Array or None, shape (n_time, n_cont_states, S)
        Row ``t`` pairs with ``xi[t]`` (the source time).
    transition_prior : Array or None, shape (S, S)
        Dirichlet alphas (``>= 1``); None means all ones (no pseudo-counts).
        Worth ``alpha - 1`` expected transitions, exactly as in
        :func:`switching_kalman.switching_kalman_maximization_step`.
    l2_penalty : float

    Returns
    -------
    discrete_transition_matrix : Array, shape (S, S)
    transition_weights : Array or None, shape (n_features, S, S - 1)
    transition_state_weights : Array or None, shape (n_cont_states, S, S - 1)
    """
    # Imported here: switching_kalman imports this module.
    from state_space_practice.switching_kalman import (
        minimum_state_occupancy,
        warn_low_occupancy_states,
    )

    xi = jnp.asarray(smoother_joint_discrete_state_prob)
    n_pairs, n_states, _ = xi.shape
    eta = centered_softmax_inverse(jnp.asarray(discrete_transition_matrix))
    blocks = [jnp.ones((n_pairs, 1), dtype=xi.dtype)]
    coefs = [eta[None]]
    n_features = 0
    if transition_covariates is not None:
        u = jnp.asarray(transition_covariates)
        n_features = u.shape[1]
        blocks.append(u[1:])                       # row t + 1 drives S_t -> S_{t+1}
        coefs.append(jnp.asarray(transition_weights))
    design_shared = jnp.concatenate(blocks, axis=1)                 # (n_pairs, 1 + F)
    design_all = jnp.broadcast_to(design_shared, (n_states, *design_shared.shape))
    if transition_state_weights is not None:
        means_prev = jnp.moveaxis(jnp.asarray(state_cond_smoother_means)[:-1], -1, 0)
        design_all = jnp.concatenate([design_all, means_prev], axis=-1)  # (S, n_pairs, 1+F+n_cont)
        coefs.append(jnp.asarray(transition_state_weights))
    coef0 = jnp.concatenate(coefs, axis=0)                          # (n_coef, S, S-1)
    n_coef = coef0.shape[0]
    x0_all = coef0.transpose(1, 0, 2).reshape(n_states, -1)
    response_all = xi.transpose(1, 0, 2)                             # (S, n_pairs, S)
    # dirichlet_neg_log_likelihood averages the data term over the N pairs and
    # weights the pseudo-counts per sample, so pass 1 + (alpha - 1) / N to make
    # the prior worth (alpha - 1) *counts*, as in _switching_kalman_m_step_inner.
    alpha = (
        jnp.ones((n_states, n_states), dtype=xi.dtype)
        if transition_prior is None
        else 1.0 + (jnp.asarray(transition_prior, dtype=xi.dtype) - 1.0) / n_pairs
    )
    optimized = optimize_transition_rows(
        x0_all, response_all, alpha, design_all, jnp.asarray(l2_penalty, dtype=xi.dtype)
    )
    new_coef = optimized.reshape(n_states, n_coef, n_states - 1).transpose(1, 0, 2)

    # Identifiability gate: a source state with too few expected transitions
    # cannot support a regression of n_coef coefficients per column.
    row_occupancy = jnp.sum(xi, axis=(0, 2))
    min_occupancy = minimum_state_occupancy(n_coef)
    ok = row_occupancy >= min_occupancy
    new_coef = jnp.where(ok[None, :, None], new_coef, coef0)
    if not contains_tracer(row_occupancy):
        warn_low_occupancy_states(
            row_occupancy,
            min_occupancy,
            "maximize_transition_coefficients",
            "their transition logits kept their previous values",
        )

    new_matrix = centered_softmax(new_coef[0])
    new_gamma = new_coef[1 : 1 + n_features] if transition_covariates is not None else None
    new_state_weights = (
        new_coef[1 + n_features :] if transition_state_weights is not None else None
    )
    return new_matrix, new_gamma, new_state_weights
```

The `contingency_belief._optimize_transition_rows` name is kept as a two-line
wrapper (`return optimize_transition_rows(x0_all, response_all, alpha_all,
jnp.broadcast_to(design, (x0_all.shape[0], *design.shape)), l2_penalty)`) so
`ContingencyBeliefModel._m_step` (`contingency_belief.py:1309-1315`) is
unchanged apart from the import.

Install helper (one per base class; the point-process and choice versions are
identical apart from attribute names, see the
[model attribute contract](shared-contracts.md#model-attribute-contract)):

```python
def _transition_kwargs(self) -> dict:
    """Keywords forwarded to the switching filter / GPB1 smoother."""
    kwargs = {}
    if self._transition_covariates is not None:
        kwargs["transition_covariates"] = self._transition_covariates
        kwargs["transition_weights"] = self.transition_weights
    # phase 2: if self.transition_state_weights is not None: kwargs["transition_state_weights"] = ...
    return kwargs

def _install_discrete_transition(self, count_based_matrix: Array) -> None:
    """Install the M-step's discrete transition estimate (fixed or logit path)."""
    if not self.update_discrete_transition_matrix:
        return
    if not self._transition_kwargs():
        self.discrete_transition_matrix = count_based_matrix
        return
    new_matrix, new_weights, new_state_weights = maximize_transition_coefficients(
        self.smoother_joint_discrete_state_prob,
        self.discrete_transition_matrix,
        transition_covariates=self._transition_covariates,
        transition_weights=self.transition_weights,
        # phase 2: transition_state_weights=..., state_cond_smoother_means=self.smoother_state_cond_mean,
        transition_prior=self.transition_prior,
        l2_penalty=self.transition_regularization,
    )
    self.discrete_transition_matrix = new_matrix
    if new_weights is not None:
        self.transition_weights = new_weights
    # phase 2: if new_state_weights is not None: self.transition_state_weights = new_state_weights

def _bind_transition_covariates(self, transition_covariates, n_time: int) -> None:
    if transition_covariates is None:
        self._transition_covariates = None
        return
    u = jnp.asarray(transition_covariates, dtype=float)
    n_states = self.n_discrete_states
    if self.transition_weights is None:
        self.transition_weights = jnp.zeros((u.shape[1], n_states, n_states - 1))
    validate_transition_inputs(
        self.discrete_transition_matrix, u, self.transition_weights, None,
        n_time=n_time, n_cont_states=<family's latent dim>,
    )
    self._transition_covariates = u
```

Rationale for BFGS over plain gradient ascent: the objective is a smooth
multinomial-logit log-likelihood per row, the repo already solves exactly this
objective with jitted per-row BFGS for the contingency model, and warm-starting
from the current coefficients makes each EM iteration a few quasi-Newton steps.

## F. Oracle extension

`tests/oracles.py:473-651` (`switching_lgssm_exact_posterior`). Accept a
`(n_time, K, K)` stack; the constant case is broadcast (entry 0 unused), the
path prior uses the per-step matrix:

```python
Z = np.asarray(discrete_transition_matrix, dtype=np.float64)
if Z.ndim == 2:
    Z = np.broadcast_to(Z, (n_time, *Z.shape))          # entry 0 is never read
...
with np.errstate(divide="ignore"):
    log_pi = np.log(pi)
    log_Z = np.log(Z)                                    # (T, K, K)
...
for t in range(1, n_time):
    lp = lp + log_Z[t, s[t - 1], s[t]]                   # was: log_Z[s[t - 1], s[t]]
```

Everything downstream is already per time step. Update the module docstring
(`tests/oracles.py:29`) to `S_t | S_{t-1} = i ~ Z_t[i, :]`.
`switching_q_from_statistics` keeps its summed `counts * log(Z)` term
(`tests/oracles.py:774`) for the fixed case; the time-varying expected
transition term used by the M-step tests is computed directly in the test as
`np.sum(oracle.smoothed_joint_discrete_prob * np.log(stack[1:]))`.

The library side is driven from the same stack: build `stack` with
`transition_matrix_stack` from the covariates, feed the oracle the stack and the
library the covariates + weights; both then use identical `T_t` (up to the
1e-16 round trip).

## G. Phase-2 collapse approximation and exact grid oracle

**The approximation.** Under GPB2 the discrete update at time `t` needs
`P(S_t = j | S_{t-1} = i, y_{1:t-1}) = E[softmax_j(eta_i + Gamma_i u_t + W_i x_{t-1}) | S_{t-1} = i, y_{1:t-1}]`,
an expectation over the Gaussian `x_{t-1} | S_{t-1} = i, y_{1:t-1} ~ N(m^i_{t-1|t-1}, P^i_{t-1|t-1})`
(the collapsed, state-conditional filter moment the scan carries). V1 uses the
plug-in `softmax(eta_i + Gamma_i u_t + W_i m^i_{t-1|t-1})`. The gap is
`O(||W_i||^2 tr P^i)` (second-order Taylor term of the softmax), so it vanishes
as the filter covariance scale goes to zero — the quantity the trend tests
drive. The smoother keeps the forward pass's `T_t` fixed (reconstructed from
`filter_mean`), which is the same structural choice as holding the discrete
variational factor's transition potentials fixed from the forward pass in
Laplace-EM for rSLDS (Zoltowski, Pillow & Linderman 2020).

**IO-HMM equivalence (fixed-point test).** The plug-in model is exactly a
phase-1 covariate model whose covariates are the filter's own collapsed means:
with `u_t = vec(m_{t-1|t-1})` of length `n_cont_states * S` (state-major:
`u[d * S + i'] = m[d, i']`) and
`Gamma[d * S + i', i, j] = delta(i, i') * W[d, i, j]`, `eta + u_t @ Gamma`
reproduces `eta + W . m^i` row by row. So

```python
out_a = switching_kalman_filter(..., transition_state_weights=W)            # phase 2 path
m = out_a[0]                                                                # (T, n_cont, S) state-conditional means
u = jnp.concatenate([jnp.zeros((1, n_cont * S)), m[:-1].reshape(T - 1, -1)])  # row t = vec(m_{t-1})
gamma = jnp.einsum("dij,ik->dkij", W, jnp.eye(S)).reshape(n_cont * S, S, S - 1)
out_b = switching_kalman_filter(..., transition_covariates=u, transition_weights=gamma)  # phase 1 path
# every output of out_a equals out_b to 1e-12
```

This checks the in-scan rule against the precomputed-stack rule and, since the
smoother uses `transition_matrix_stack(..., state_cond_means=filter_mean)`,
also that the smoother sees exactly the matrices the filter used.

**Primitive-level trend test (deterministic quadrature).** For a 1-D latent the
exact row expectation is a one-dimensional Gaussian integral; Gauss-Hermite
quadrature makes it exact to ~1e-12, so the trend assertion is not confounded
by Monte-Carlo noise (which would floor at ~1e-3 for 1e6 samples):

```python
nodes, weights = np.polynomial.hermite_e.hermegauss(200)     # weight function exp(-x^2/2)
weights = weights / np.sqrt(2.0 * np.pi)

def plugin_gap(row_logits, row_state_weights, mean, var):
    x = mean + np.sqrt(var) * nodes                          # (200,)
    exact = (weights[:, None] * np.asarray(centered_softmax(
        jnp.asarray(row_logits)[None, :] + x[:, None] * row_state_weights[None, :]
    ))).sum(axis=0)
    plug = np.asarray(centered_softmax(jnp.asarray(row_logits + mean * row_state_weights)))
    return np.max(np.abs(exact - plug))

gaps = [plugin_gap(np.array([0.5, -0.3]), np.array([2.0, -1.5]), 0.2, c) for c in (1.0, 0.3, 0.1, 0.03, 0.01)]
# assert strictly decreasing and gaps[0] > 1e-2
```

**Filter-level exact reference: grid forward-backward for a 1-D recurrent SLDS.**
`n_cont_states = 1`, `K` states, `T <= 8`, grid `x` of `G` points with spacing
`dx`. Unlike the path-enumeration oracle, the transition depends on `x_{t-1}`,
so each path's continuous posterior is no longer Gaussian; the grid recursion
integrates it directly. Cost `O(T K^2 G^2)`.

```python
from scipy.stats import norm

def grid_recurrent_posterior(model, grid, transition_fn):
    """Exact (to quadrature) filter/smoother of a 1-D recurrent switching LGSSM.

    model: dict with m0 (1,K), P0 (1,1,K), pi (K,), y (T,1), A (1,1,K), Q (1,1,K),
    H (1,1,K), R (1,1,K).  transition_fn(x_prev) -> (K, K) row-stochastic.
    Returns filtered_prob (T,K), smoothed_prob (T,K), joint (T-1,K,K), log_lik.
    """
    y, K, T = model["y"][:, 0], model["pi"].size, model["y"].shape[0]
    G, dx = grid.size, grid[1] - grid[0]
    Tm = np.stack([transition_fn(x) for x in grid])                       # (G, K, K)
    obs = lambda t: np.stack([norm.pdf(y[t], model["H"][0, 0, j] * grid,
                                       np.sqrt(model["R"][0, 0, j])) for j in range(K)], -1)   # (G, K)
    dyn = [norm.pdf(grid[None, :], model["A"][0, 0, j] * grid[:, None],
                    np.sqrt(model["Q"][0, 0, j])) for j in range(K)]      # K x (G_prev, G_next)
    alpha = np.stack([model["pi"][j] * norm.pdf(grid, model["m0"][0, j],
                      np.sqrt(model["P0"][0, 0, j])) for j in range(K)], -1) * obs(0)
    log_c, alphas = [np.log(alpha.sum() * dx)], []
    alpha /= alpha.sum() * dx
    alphas.append(alpha)
    for t in range(1, T):
        pred = np.zeros((G, K))                                            # p(x_t, S_t=j | y_{1:t-1})
        for j in range(K):
            for i in range(K):
                pred[:, j] += (alpha[:, i] * Tm[:, i, j]) @ dyn[j] * dx
        new = pred * obs(t)
        log_c.append(np.log(new.sum() * dx))
        alpha = new / (new.sum() * dx)
        alphas.append(alpha)
    filtered_prob = np.stack([a.sum(0) * dx for a in alphas])
    # backward: beta_t(x, i) = sum_j T(x)[i,j] * ∫ N(x'; A_j x, Q_j) p(y_{t+1}|x',j) beta_{t+1}(x',j) dx'
    beta = np.ones((G, K))
    smoothed, joint = [alphas[-1]], []
    for t in range(T - 2, -1, -1):
        lik_next = obs(t + 1) * beta                                       # (G_next, K)
        back = np.zeros((G, K))
        pair = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                inner = dyn[j] @ lik_next[:, j] * dx                        # (G_prev,)
                back[:, i] += Tm[:, i, j] * inner
                pair[i, j] = (alphas[t][:, i] * Tm[:, i, j] * inner).sum() * dx
        gamma = alphas[t] * back
        smoothed.append(gamma / (gamma.sum() * dx))
        joint.append(pair / pair.sum())
        beta = back / back.max()                                           # rescale only
    return dict(filtered_prob=filtered_prob,
                smoothed_prob=np.stack([s.sum(0) * dx for s in smoothed[::-1]]),
                joint=np.stack(joint[::-1]), log_lik=float(np.sum(log_c)))
```

Self-check (required test): with `transition_fn = lambda x: Z` (no state
dependence) the grid oracle must match `switching_lgssm_exact_posterior` on
`filtered_discrete_prob`, `smoothed_discrete_prob`, `smoothed_joint_discrete_prob`
and `log_likelihood` to `1e-6` with `G = 1201` over `[-8, 8]` (choose the
grid from the union of `±8 sd` of the path posteriors, as
`tests/test_approximation_trends.py:88-91` does).

Trend test: scale `P0`, `Q` by `c in (1, 0.3, 0.1, 0.03, 0.01)` (keep `R`), fix
`W` and the data; RMS gap of the library filter's discrete probabilities and
the `|log_lik|` gap against the grid oracle decrease strictly and the loosest
gap exceeds `1e-2`.

**Recorded follow-up (exact expectation).** See overview Open Question 2. The
single swap point is the `transition_at_step` rule returned by
`resolve_scan_transitions` on the state path: it receives the carry's collapsed
mean today and would receive the collapsed covariance as well.

## H. Relabelling transform

Permuting the discrete states changes the reference column. The invariance test
builds the permuted parameters by appending the reference column, permuting
both discrete axes, and re-centering on the new last column (softmax rows are
shift-invariant, so `T'_t = T_t[perm][:, perm]` exactly):

```python
def permute_transition_params(Z, Gamma, W, perm):
    S = Z.shape[0]
    def recenter(full):                     # full (..., S, S) logits with the reference column present
        p = full[..., perm, :][..., :, perm]
        return p[..., :-1] - p[..., -1:]
    eta_full = np.concatenate([np.asarray(centered_softmax_inverse(jnp.asarray(Z))), np.zeros((S, 1))], -1)
    Z_new = np.asarray(centered_softmax(jnp.asarray(recenter(eta_full))))
    Gamma_new = None if Gamma is None else recenter(np.concatenate([Gamma, np.zeros((*Gamma.shape[:-1], 1))], -1))
    W_new = None if W is None else recenter(np.concatenate([W, np.zeros((*W.shape[:-1], 1))], -1))
    return Z_new, Gamma_new, W_new
```

Extend `tests/test_invariances.py:332-345` (`test_discrete_state_relabelling`)
with covariates (phase 1a) and state weights (phase 2a) using this helper; the
assertions (`prob[:, perm]`, `joint[:, perm][:, :, perm]`, means unchanged) are
the existing ones.

## I. Simulators

`simulate/simulate_switching_kalman.py:98-130` (`simulate`, NumPy):

```python
def simulate(A, B0, Q, R, Z, X_0, S_0, T, s=None, seed: int = 14,
             *, transition_covariates=None, transition_weights=None,
             transition_state_weights=None):                   # phase 2b
    ...
    eta = np.asarray(centered_softmax_inverse(jnp.asarray(Z)))   # (M, M-1)
    def _row_probs(s_prev, t, x_prev):
        logits = eta[s_prev].copy()
        if transition_covariates is not None:
            logits += transition_covariates[t] @ transition_weights[:, s_prev, :]
        if transition_state_weights is not None:
            logits += x_prev @ transition_state_weights[:, s_prev, :]
        full = np.concatenate([logits, [0.0]])
        full -= full.max()
        p = np.exp(full)
        return p / p.sum()
    for t in range(1, T):
        if blnSimS:
            probs = Z[s[t - 1], :] if (transition_covariates is None and transition_state_weights is None) else _row_probs(s[t - 1], t, x[t - 1])
            s[t] = np.nonzero(rng.multinomial(1, probs))[0][0]
        ...
```

The fixed path keeps `Z[s[t - 1], :]` so existing seeds reproduce existing data.

`simulate/simulate_switching_spikes.py:142-166` (`_step`, JAX): scan over
`covariates[1:]` (zeros `(n_time, 1)` with zero weights when `None`, so the scan
shape is static and the fixed path's random stream is unchanged — the draw is
`categorical(key, log_probs)` with `log_probs = log Z[s_prev]` exactly as today
when no option is given; use a Python branch, not a zero-weight addition, to
keep the fixed path bit-identical):

```python
def _step(carry, u_t):
    x_prev, s_prev, key = carry
    key, key_discrete, key_continuous, key_spikes = jax.random.split(key, 4)
    if transition_covariates is None and transition_state_weights is None:
        log_probs = jnp.log(discrete_transition_matrix[s_prev])
    else:
        logits = eta[s_prev]
        if transition_covariates is not None:
            logits = logits + u_t @ transition_weights[:, s_prev, :]
        if transition_state_weights is not None:
            logits = logits + x_prev @ transition_state_weights[:, s_prev, :]
        log_probs = centered_log_softmax(logits)
    s_t = jax.random.categorical(key_discrete, log_probs)
    ...
xs = None if transition_covariates is None else transition_covariates[1:]
jax.lax.scan(_step, (x_0, s_0, key), xs, length=n_time - 1)
```

`switching_choice.simulate_switching_choice_data` (`switching_choice.py:1260-1359`):
the state and value scans are separate today (`:1325-1341`). Phase 1c adds
`u_t` to `_state_step` (states still drawn first); phase 2b merges the two scans
into one because `s_t` depends on `x_{t-1}`, consuming the *pre-split*
`state_keys[t]` / `value_keys[t]` per step so the fixed path reproduces the
current draws (a baseline capture of `simulate_switching_choice_data(n_trials=20, seed=42)`
before the change is checked in as a test constant, see phase 2b fixtures).

## J. Sibling integration notes

Point-process filter (`switching_point_process.py`): the jitted core takes the
keywords, calls `resolve_scan_transitions` once and scans over
`(spikes[1:], per_step)`; the body change is the two-line `transition_t` swap
at `:2284-2289`. Static shape checks go into
`_validate_switching_point_process_filter_shapes` (`:286-370`) next to the
existing `discrete_transition_matrix` check (`:319-324`); value checks into the
wrapper via `validate_transition_inputs` next to `_validate_discrete_state_transitions`
(`:2453-2455`). The `_validate_discrete_state_transitions` row-sum check
(`:410-415`) still applies to the baseline matrix.

Choice filter (`switching_choice.py`): the jitted core is
`functools.partial(jax.jit, static_argnames=[...])` with `None` defaults; the
new keywords follow the same pattern as `covariates` / `input_gain`
(`:331-343`). Scan inputs become
`(choices[1:], cov_arr[1:], obs_cov_arr[1:], per_step)` (`:499`). The public
wrapper (`:233-249`) passes positionally today; pass the new ones by keyword.
`_populate_uncertainty` (`:813-822`): with a stack, `predicted_disc[t] = disc_probs[t-1] @ stack[t]`
(`jnp.einsum("ti,tij->tj", disc_probs[:-1], stack[1:])`); the model recomputes
the stack with `transition_matrix_stack(..., state_cond_means=result.filtered_values)`.

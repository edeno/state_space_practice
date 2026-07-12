"""Switching Kalman filter and smoother and EM algorithm.

References
----------
1. Shumway, R.H., and Stoffer, D.S. (1991). Dynamic Linear Models With Switching. 8.
2. Murphy, K.P. (1998). Switching kalman filters.
3. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2022). Switching Functional Network Models of Oscillatory Brain Dynamics. In 2022 56th Asilomar Conference on Signals, Systems, and Computers (IEEE), pp. 607–612. https://doi.org/10.1109/IEEECONF56349.2022.10052077.
4. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2024). Switching Models of Oscillatory Networks Greatly Improve Inference of Dynamic Functional Connectivity. Preprint at arXiv.
5. https://github.com/Stephen-Lab-BU/Switching_Oscillator_Networks
"""

import math
import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from state_space_practice.kalman import (
    _kalman_filter_update,
    _kalman_smoother_update,
    kalman_measurement_update,
    psd_solve,
    stabilize_covariance,
)
from state_space_practice.utils import debug_print_if
from state_space_practice.utils import divide_safe as _divide_safe
from state_space_practice.utils import safe_log as _safe_log
from state_space_practice.utils import spectral_radius as _spectral_radius
from state_space_practice.utils import (
    stabilize_probability_vector as _stabilize_probability_vector,
)

_kalman_filter_update_per_discrete_state_pair = jax.vmap(
    jax.vmap(
        _kalman_filter_update,
        in_axes=(-1, -1, None, None, None, None, None),
        out_axes=-1,
    ),
    in_axes=(None, None, None, -1, -1, -1, -1),
    out_axes=-1,
)  # shape (n_discrete_states, n_discrete_states, n_obs_dim)


@jax.jit
def collapse_gaussian_mixture(
    conditional_means_x: jax.Array,
    conditional_cov: jax.Array,
    mixing_weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Collapse a mixture of Gaussians.

    Parameters
    ----------
    conditional_means_x : jax.Array, shape (n_dims, n_discrete_states)
        E[X | S = j]
    conditional_cov : jax.Array, shape (n_dims, n_dims, n_discrete_states)
        Cov[X | S = j]
    mixing_weights : jax.Array, shape (n_discrete_states,)
        P[S = j]

    Returns
    -------
    unconditional_mean_x : jax.Array, shape (n_dims,)
        E[X]
    unconditional_cov_x : jax.Array, shape (n_dims, n_dims)
        Cov[X]
    """
    unconditional_mean_x = conditional_means_x @ mixing_weights  # E[X]
    diff_x = conditional_means_x - unconditional_mean_x[:, None]

    # Cov[X] via the law of total covariance:
    #   Cov[X] = E[Cov[X | S]] + Cov[E[X | S]]
    unconditional_cov_x = (
        conditional_cov @ mixing_weights + (diff_x * mixing_weights) @ diff_x.T
    )

    return unconditional_mean_x, unconditional_cov_x


def collapse_gaussian_mixture_cross_covariance(
    conditional_means_x: jax.Array,
    conditional_means_y: jax.Array,
    conditional_cross_cov: jax.Array,
    mixing_weights: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute cross-covariance when collapsing a Gaussian mixture.

    Parameters
    ----------
    conditional_means_x : jax.Array, shape (n_dims, n_discrete_states)
        E[X | S = j]
    conditional_means_y : jax.Array, shape (n_dims, n_discrete_states)
        E[Y | S = j]
    conditional_cross_cov : jax.Array, shape (n_dims, n_dims, n_discrete_states)
        Cov[X, Y | S = j], conditional cross-covariance per discrete state.
    mixing_weights : jax.Array, shape (n_discrete_states,)
        P[S = j]

    Returns
    -------
    unconditional_mean_x : jax.Array, shape (n_dims,)
        E[X]
    unconditional_mean_y : jax.Array, shape (n_dims,)
        E[Y]
    unconditional_cov_xy : jax.Array, shape (n_dims, n_dims)
        Cov[X, Y]
    """

    unconditional_mean_x = conditional_means_x @ mixing_weights  # E[X]
    unconditional_mean_y = conditional_means_y @ mixing_weights  # E[Y]

    diff_x = conditional_means_x - unconditional_mean_x[:, None]
    diff_y = conditional_means_y - unconditional_mean_y[:, None]

    # Cov[X, Y] via the law of total covariance:
    #   Cov[X, Y] = E[Cov[X, Y | S]] + Cov[E[X | S], E[Y | S]]
    unconditional_cov_xy = (
        conditional_cross_cov @ mixing_weights + (diff_x * mixing_weights) @ diff_y.T
    )

    return unconditional_mean_x, unconditional_mean_y, unconditional_cov_xy


collapse_gaussian_mixture_per_discrete_state = jax.vmap(
    collapse_gaussian_mixture, in_axes=(-1, -1, -1), out_axes=(-1, -1)
)
collapse_gaussian_mixture_over_next_discrete_state = jax.vmap(
    collapse_gaussian_mixture, in_axes=(1, 2, 0), out_axes=(-1, -1)
)
collapse_cross_gaussian_mixture_across_states = jax.vmap(
    collapse_gaussian_mixture_cross_covariance, in_axes=(2, 2, 3, 1), out_axes=(1, 1, 2)
)


def _cap_covariance_trace(
    cov: jax.Array,
    max_allowed_trace: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Cap a covariance so its trace does not exceed ``max_allowed_trace``.

    Returns the capped covariance *and* the multiplicative scale ``alpha`` in
    (0, 1] that was applied (``cov * alpha``). The caller needs ``alpha`` to
    rescale the paired lag-one cross-covariance by ``sqrt(alpha)`` -- a
    congruence transform of the joint block ``[[P_t, C], [C^T, P_{t+1}]]`` --
    so capping ``P_t`` cannot leave the joint indefinite.

    Pure (no telemetry): this runs under ``jax.vmap``, where ``lax.cond`` lowers
    to ``select`` and both branches execute, so a per-element ``debug_print_if``
    here would fire on every lane regardless of the predicate. The divergence
    signal is emitted once at the call site via ``_warn_if_cov_cap_engaged``.

    Parameters
    ----------
    cov : jax.Array, shape (n_cont_states, n_cont_states)
        A single (per-lane) covariance matrix.
    max_allowed_trace : jax.Array
        Scalar trace cap.

    Returns
    -------
    capped_cov : jax.Array, shape (n_cont_states, n_cont_states)
        ``cov`` scaled by ``alpha`` so its trace is at most ``max_allowed_trace``.
    alpha : jax.Array
        The applied scale in (0, 1] (``sqrt(alpha)`` rescales the cross-cov).
    """
    trace = jnp.trace(cov)
    # scale = min(1, max/trace); guard trace==0 (then no scaling is needed).
    scale = jnp.minimum(1.0, max_allowed_trace / jnp.where(trace > 0.0, trace, 1.0))
    return scale * cov, scale


def _log_prob_preserve_zeros(prob: jax.Array) -> jax.Array:
    """Log-probabilities mapping exact zeros to ``-inf`` (impossible stays impossible)."""
    prob = jnp.asarray(prob)
    if not jnp.issubdtype(prob.dtype, jnp.inexact):
        prob = prob.astype(jnp.float32)
    tiny = jnp.finfo(prob.dtype).tiny
    return jnp.where(prob > 0.0, jnp.log(jnp.maximum(prob, tiny)), -jnp.inf)


def _normalize_initial_discrete_prob(prob: jax.Array) -> jax.Array:
    """Normalize a caller-supplied initial discrete-state prior, preserving zeros.

    The initial distribution is a user *input*, not a computed posterior. An
    exact zero declares a state impossible at t=1 and must survive into the
    posterior. Unlike :func:`stabilize_probability_vector` -- which floors
    underflowed *computed* posteriors back above zero to prevent permanent
    state lockout -- this must NOT floor zeros, otherwise the first-observation
    likelihood can resurrect a state the caller declared impossible (and, via
    ``_log_prob_preserve_zeros``, the Viterbi first state too).

    Malformed input is handled by severity, with a divergence signal
    (``debug_print_if``) in every case: a vector that does not sum to a positive
    value (all-zero / all-non-positive) is *rejected* as NaN (fail loud); a
    non-finite or negative *entry* alongside an otherwise-positive sum is
    *clamped to 0 and renormalized* (a sign error must not pull mass onto a
    state) rather than rejected -- so this is defense-in-depth telemetry, not a
    hard rejection of every stray negative.

    Parameters
    ----------
    prob : jax.Array, shape (n_discrete_states,)
        Caller-supplied prior ``p(S_1)``; exact zeros are structural.

    Returns
    -------
    jax.Array, shape (n_discrete_states,)
        ``prob`` normalized to sum 1 with structural zeros preserved; all-NaN
        when the (sanitized) vector does not sum to a positive value.
    """
    prob = jnp.asarray(prob)
    prob = prob.astype(jnp.result_type(prob, 1.0))
    # Sanitize non-finite entries and clamp sign errors to zero (an invalid
    # negative probability must not pull mass onto a state) without flooring
    # legitimate structural zeros.
    cleaned = jnp.maximum(jnp.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = jnp.sum(cleaned)
    debug_print_if(
        jnp.any(~jnp.isfinite(prob)) | jnp.any(prob < 0.0) | (total <= 0.0),
        "switching_kalman: initial_discrete_state_prob was non-finite, "
        "negative, or summed to <= 0 (sum={s:.2e}); structural zeros in an "
        "otherwise valid prior are preserved, but an all-zero / non-positive "
        "prior is rejected as NaN.",
        s=total,
    )
    # A prior that does not sum to a positive value is not a distribution.
    # Fail loud with NaN so the caller's non-finite check rejects it, rather
    # than returning an all-zero "posterior" that looks finite downstream.
    return jnp.where(total > 0.0, _divide_safe(cleaned, total), jnp.nan)


def _warn_if_cov_cap_engaged(
    covs: jax.Array,
    max_allowed_trace: jax.Array,
    label: str,
    cap_multiplier: float | None = None,
    *,
    upper_allowed_trace: jax.Array | None = None,
    moderate: bool = False,
) -> None:
    """Emit ONE divergence signal if any covariance-lane trace exceeds the cap.

    ``covs`` has its two leading axes as the (n_cont, n_cont) covariance and any
    number of trailing discrete-state axes. Reducing to a single scalar
    predicate here -- rather than inside the vmapped ``_cap_covariance_trace`` --
    is what makes the signal fire only on a genuine blow-up (the cap engaging on
    some lane), not on every fit. Called from the (non-vmapped) smoother scan
    body, so ``debug_print_if``'s ``lax.cond`` behaves conditionally.

    Two severity tiers are supported so a caller that applies a *moderate*
    trust-region cap (below the catastrophic threshold) can still surface that
    the cap engaged instead of clipping EM statistics silently:

    - ``moderate=False`` (default): catastrophic-divergence wording, fired when a
      lane trace exceeds ``max_allowed_trace``.
    - ``moderate=True``: lower-severity "trust region engaged" wording. Pass
      ``upper_allowed_trace`` (the catastrophic threshold) to fire only in the
      band ``(max_allowed_trace, upper_allowed_trace]`` so the moderate note and
      a separate catastrophic warning remain mutually exclusive.
    """
    if cap_multiplier is None:
        cap_multiplier = _COV_CAP_MULTIPLIER

    traces = jnp.trace(covs, axis1=0, axis2=1)
    engaged = traces > max_allowed_trace
    if upper_allowed_trace is not None:
        engaged = engaged & (traces <= upper_allowed_trace)
    if moderate:
        debug_print_if(
            jnp.any(engaged),
            f"switching_kalman: {label} smoother covariance engaged the "
            f"{cap_multiplier:.0e}x-filter trust region (max trace {{m:.2e}}); "
            f"EM statistics at those steps were clipped. This is a "
            f"moment-matching safety net, not necessarily divergence, but "
            f"persistent firing means the GPB2 smoother covariances are growing "
            f"abnormally.",
            m=jnp.max(jnp.where(engaged, traces, -jnp.inf)),
        )
    else:
        debug_print_if(
            jnp.any(engaged),
            f"switching_kalman: {label} smoother covariance exceeded the "
            f"{cap_multiplier:.0e}x-filter cap (max trace {{m:.2e}}); the GPB "
            f"smoother has diverged and the returned posterior + EM statistics "
            f"are unreliable.",
            m=jnp.max(traces),
        )


_COV_CAP_MULTIPLIER = 1e8
_GPB2_COV_CAP_MULTIPLIER = 1e2


def _compute_max_allowed_trace(
    state_cond_filter_cov: jax.Array,
) -> jax.Array:
    """Compute the maximum allowed covariance trace from filter covariances.

    Parameters
    ----------
    state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        State-conditional filter covariances. The last axis is the discrete
        state dimension, which is what the inner ``vmap`` broadcasts over.

    Returns
    -------
    jax.Array
        Scalar maximum trace, equal to ``max(trace(cov[:, :, k]) for k in states) * _COV_CAP_MULTIPLIER + 1.0``.
        The additive 1.0 prevents the cap from collapsing to zero when filter
        covariances are identically zero.
    """
    max_filter_trace = jnp.max(jax.vmap(jnp.trace, in_axes=-1)(state_cond_filter_cov))
    return max_filter_trace * _COV_CAP_MULTIPLIER + 1.0


def _guard_smoother_mean(mean: jax.Array, label: str) -> jax.Array:
    """Preserve finite, representable smoother means; fail loud otherwise.

    A smoother mean has no valid magnitude threshold: future observations can
    legitimately move it arbitrarily far from the current filtered mean, so any
    local cap (absolute or scale-relative) can corrupt a correct RTS posterior.
    Finite, representable means are therefore returned unchanged.

    The only genuine failure is a mean that is non-finite, or so large that
    squaring it (as the M-step second-moment statistics do) would overflow the
    dtype. Those are replaced with NaN rather than consumed as a fabricated
    posterior. The value is never saturated at a threshold -- a clipped mean is
    still a fabricated posterior.

    Propagating NaN is a fail-loud signal, not a rollback: what the caller does
    with it is caller-dependent. ``PlaceFieldModel``-style point-process ``fit``
    loops guard on a non-finite log-likelihood and roll the EM step back;
    callers without that guard (e.g. ``SwitchingChoiceModel.fit``) instead
    surface the NaN in their reported objective. Either way the fabricated
    posterior never silently overwrites the parameters.

    Parameters
    ----------
    mean : jax.Array
        Smoother mean array (any shape).
    label : str
        Smoother label (``"GPB1"`` / ``"GPB2"``) for the telemetry message.

    Returns
    -------
    jax.Array
        ``mean`` unchanged when finite and representable; otherwise all-NaN.
    """
    # sqrt(max) is the largest magnitude whose square is still representable.
    safe_second_moment_abs = jnp.sqrt(jnp.finfo(mean.dtype).max)
    unrepresentable = jnp.any(~jnp.isfinite(mean)) | jnp.any(
        jnp.abs(mean) > safe_second_moment_abs
    )
    debug_print_if(
        unrepresentable,
        f"switching_kalman: {label} smoother mean is non-finite or too large to "
        "square without overflow (max |mean| {m:.2e}); propagating NaN as a "
        "fail-loud signal (a non-finite-guarded EM loop rolls the step back).",
        m=jnp.max(jnp.abs(mean)),
    )
    return jnp.where(unrepresentable, jnp.nan, mean)


def _stabilize_probability_vector_preserving_zeros(
    probabilities: jax.Array,
) -> jax.Array:
    """Floor underflow only on currently-nonzero entries; hold exact zeros at 0.

    An exact zero is *structural*: a state the model cannot occupy at this
    timestep, because either the prior or every transition into it is zero.
    Flooring it -- as :func:`stabilize_probability_vector` does unconditionally
    -- resurrects an impossible state, and (across a deterministic transition)
    leaks that mass onto the wrong state at the next step.

    Crucially, a probability vector's exact zeros are *time-indexed*: the
    posterior/marginal at time ``t`` is exactly 0 precisely at the states not in
    the structural support ``S_t``, so masking on ``value > 0`` gives the
    per-timestep support without any separate reachability computation. Nonzero
    entries are still floored (then renormalized) to recover from numerical
    underflow; in the healthy all-positive regime this equals
    :func:`stabilize_probability_vector` exactly.

    Parameters
    ----------
    probabilities : jax.Array, shape (n_discrete_states,)
        A discrete probability vector whose exact zeros are structural.

    Returns
    -------
    jax.Array, shape (n_discrete_states,)
        Renormalized vector with exact zeros preserved and nonzero entries
        floored; uniform recovery if the whole vector underflowed to zero.
    """
    probabilities = jnp.asarray(probabilities)
    support = probabilities > 0.0
    stabilized = _stabilize_probability_vector(probabilities)
    # Preserve structural zeros when some state has positive mass; if the whole
    # vector underflowed to zero there is no support left to distinguish, so
    # fall back to stabilize_probability_vector's uniform recovery.
    masked = jnp.where(
        jnp.any(support), jnp.where(support, stabilized, 0.0), stabilized
    )
    return _divide_safe(masked, jnp.sum(masked))


# Log-space floor for a structurally-supported discrete state (matches the
# linear _DISCRETE_PROB_STABILITY_FLOOR = 1e-10 in utils). A supported state
# whose posterior underflows to 0 is floored here so it is recovered rather than
# locked out; a structurally forbidden state is masked to -inf instead.
_LOG_DISCRETE_STABILITY_FLOOR = math.log(1e-10)
_DISCRETE_STABILITY_FLOOR = 1e-10


def _floor_supported_marginal(prob: jax.Array, support: jax.Array) -> jax.Array:
    """Lift a supported state that underflowed to *exactly 0* to the stability floor.

    The GPB smoothers read the *filter marginal* and infer the structural
    support from its zeros. A supported state whose forward posterior underflowed
    to exactly 0 would then be lost (mistaken for impossible) even though future
    data supports it, so it is lifted to ``_DISCRETE_STABILITY_FLOOR`` to stay
    strictly positive and recoverable. Only *exact* zeros are lifted, so a
    legitimately tiny *positive* marginal (e.g. from a small caller-supplied prior
    like 1e-12) is not inflated -- it is preserved up to the final
    renormalization (dividing by the new sum). Structurally forbidden states
    (``support`` False) stay exactly 0, and a NaN marginal (malformed/diverged)
    propagates rather than being masked to 0.

    Parameters
    ----------
    prob : jax.Array, shape (n_discrete_states,)
        The (linear-space) filter marginal ``M_{t|t}(j)``.
    support : jax.Array, shape (n_discrete_states,)
        Boolean structural support ``S_t``.

    Returns
    -------
    jax.Array, shape (n_discrete_states,)
        Renormalized marginal with supported exact-zeros lifted to the floor and
        forbidden states at exactly 0; all-NaN if ``prob`` contains any NaN.
    """
    lifted = jnp.where(prob > 0.0, prob, _DISCRETE_STABILITY_FLOOR)
    floored = jnp.where(support, lifted, 0.0)
    result = _divide_safe(floored, jnp.sum(floored))
    return jnp.where(jnp.any(jnp.isnan(prob)), jnp.nan, result)


def _update_discrete_state_probabilities(
    pair_cond_marginal_log_likelihood: jax.Array,
    discrete_transition_matrix: jax.Array,
    prev_filter_discrete_prob: jax.Array,
    prev_support: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the discrete state probabilities in log space with a support mask.

    Works entirely with log-likelihoods and normalizes with ``logsumexp`` over
    the structurally-supported ``(S_{t-1}=i, S_t=j)`` pairs. This is what lets a
    state whose posterior underflowed to exactly 0 be recovered (its Boolean
    support, not its value, keeps it alive) and prevents a structurally
    impossible state -- prior/transition zero -- from setting the scaling
    reference or pulling any mass.

    Parameters
    ----------
    pair_cond_marginal_log_likelihood : jax.Array, shape (n_discrete_states, n_discrete_states)
        ``log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)`` (NOT pre-scaled).
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Z(i, j) = P(S_t=j | S_{t-1}=i)
    prev_filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        M_{t-1|t-1}(i) = Pr(S_{t-1}=i | y_{1:t-1})
    prev_support : jax.Array, shape (n_discrete_states,)
        Boolean structural support ``S_{t-1}`` (states reachable from the prior
        through the transition structure by t-1). Threaded through the scan
        rather than inferred from ``prev_filter_discrete_prob`` values, which
        cannot distinguish an underflowed reachable state from an impossible one.

    Returns
    -------
    filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        Updated discrete state probabilities, M_{t|t}(j) = Pr(S_t=j | y_{1:t})
    filter_backward_cond_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
        Mixing weights for the discrete states, W^{i|j} = Pr(S_{t-1}=i | S_t=j, y_{1:t})
    log_predictive : jax.Array
        ``log p(y_t | y_{1:t-1})`` contribution to the marginal log-likelihood.
    next_support : jax.Array, shape (n_discrete_states,)
        Boolean structural support ``S_t`` for the next step.
    """
    # Log-prior over the previous state. The Boolean support decides possibility,
    # not the value: a structurally forbidden state is -inf, while a supported
    # state whose posterior underflowed to exactly 0 is floored (recovered).
    log_prev = jnp.where(
        prev_support,
        jnp.maximum(
            _log_prob_preserve_zeros(prev_filter_discrete_prob),
            _LOG_DISCRETE_STABILITY_FLOOR,
        ),
        -jnp.inf,
    )
    log_transition = _log_prob_preserve_zeros(discrete_transition_matrix)
    # Structural log-prior over pairs: -inf on any structurally impossible pair
    # (forbidden source, or zero transition). Once a pair is impossible its
    # likelihood is irrelevant -- but ``NaN + (-inf) = NaN`` and
    # ``+inf + (-inf) = NaN``, so a garbage likelihood in an impossible lane
    # would poison the logsumexp and NaN the whole posterior despite valid
    # supported lanes. Substitute a finite placeholder into impossible lanes
    # BEFORE adding, so the -inf structural prior cleanly zeros them out.
    dynamics_log_joint = log_transition + log_prev[:, None]
    impossible_pair = jnp.isneginf(dynamics_log_joint)
    safe_pair_log_lik = jnp.where(
        impossible_pair, 0.0, pair_cond_marginal_log_likelihood
    )
    log_joint = safe_pair_log_lik + dynamics_log_joint

    # log p(y_t | y_{1:t-1}): the reference is a logsumexp over supported pairs
    # only (impossible pairs are -inf), so no impossible state can win it.
    log_predictive = logsumexp(log_joint)

    # Degenerate recovery: if every supported pair has -inf likelihood (an
    # impossible observation), fall back to the dynamics prediction (drop the
    # likelihood) for the posterior so the discrete state follows the transition
    # structure instead of zero-locking, without leaking onto impossible states.
    # Gate on -inf specifically, NOT on ~isfinite: a +inf pair likelihood (a
    # pathological zero-variance-like observation) is not an *impossible*
    # observation, so it must not be silently discarded for the dynamics
    # prediction; it flows through and surfaces as a fail-loud NaN below. A NaN
    # predictive likewise propagates rather than being masked by the dynamics.
    log_joint = jnp.where(jnp.isneginf(log_predictive), dynamics_log_joint, log_joint)

    next_support = jnp.any(
        prev_support[:, None] & (discrete_transition_matrix > 0.0), axis=0
    )
    supported_col = next_support[None, :]  # (1, n_states)

    # Per-destination-state (S_t=j) marginal and backward conditional, BOTH from
    # the log joint. Computing W^{i|j} as a per-column softmax keeps the mixing
    # weights well-defined even when a destination column underflows in linear
    # space (its relative weights are still finite), so W^{i|j} columns sum to 1
    # for every supported state -- and the caller's pair = W * marginal then stays
    # column-consistent with the (possibly floored) marginal.
    #
    # A structurally-forbidden destination column (j not in next_support) is all
    # -inf. Substitute a finite placeholder into those columns BEFORE the column
    # reductions, then mask the results back: otherwise JAX differentiates through
    # ``-inf - -inf`` and returns NaN gradients even though the forward value is
    # masked (this breaks differentiable callers like switching-choice SGD).
    safe_log_joint = jnp.where(supported_col, log_joint, 0.0)
    log_marginal = jnp.where(
        next_support, logsumexp(safe_log_joint, axis=0), -jnp.inf
    )  # (n_states,), over S_{t-1}=i
    log_norm = logsumexp(log_marginal)  # scalar (over supported columns)
    # M_{t|t}(j) = Pr(S_t=j | y_{1:t})
    filter_discrete_prob = jnp.where(
        next_support, jnp.exp(log_marginal - log_norm), 0.0
    )
    # W^{i|j} = Pr(S_{t-1}=i | S_t=j, y_{1:t}); forbidden columns -> 0. Substitute
    # a finite placeholder into ANY all -inf column -- structurally forbidden OR a
    # supported column whose likelihood underflowed to -inf (reachable in the
    # point-process filter at zero intensity). This keeps exp(...) NaN-free (both
    # the forward value and the gradient); such a column's weights are then 0.
    safe_log_marginal = jnp.where(jnp.isfinite(log_marginal), log_marginal, 0.0)
    filter_backward_cond_prob = jnp.where(
        supported_col,
        jnp.exp(log_joint - safe_log_marginal[None, :]),
        0.0,
    )
    # Floor supported-but-underflowed marginals in the RETURNED value so the GPB
    # smoothers (which read the support from the marginal's zeros) do not mistake
    # a reachable *underflowed* state (finite but tiny log_marginal) for an
    # impossible one; those columns have a per-column-normalized backward that
    # sums to 1, so pair = W * floored_marginal stays consistent.
    #
    # A reachable destination the data *rules out* this step (its likelihood
    # column is all -inf, so log_marginal[j] = -inf) is NOT floored: its backward
    # column is all zero (the observation is impossible for j), so its Bayes
    # marginal is exactly 0. Leaving it at 0 keeps pair.sum(axis=0) == marginal
    # (0 == 0) -- flooring it to 1e-10 against a zero backward column would make
    # the GPB collapse fabricate a zero-weight posterior for a state with mass.
    # (It re-enters the support next step via the log-prior floor if reachable.)
    floorable = next_support & jnp.isfinite(log_marginal)
    filter_discrete_prob = _floor_supported_marginal(filter_discrete_prob, floorable)
    # Floor a degenerate -inf contribution so the marginal LL stays finite for
    # EM; keep honest finite (even very negative) values and propagate NaN.
    log_predictive = jnp.where(
        jnp.isneginf(log_predictive), _LOG_DISCRETE_STABILITY_FLOOR, log_predictive
    )
    return (
        filter_discrete_prob,
        filter_backward_cond_prob,
        log_predictive,
        next_support,
    )


def _first_timestep_discrete_update(
    state_cond_log_lik: jax.Array,
    init_discrete_state_prob: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """First-timestep discrete posterior in log space, preserving structural zeros.

    Shared by the Gaussian (:func:`_first_timestep_kalman_update`) and
    point-process first-timestep updates so both honor a caller's structural
    zero (an impossible state stays exactly 0), represent a legitimately tiny
    prior faithfully, and fail loud (NaN) on a malformed all-zero prior. At t=1
    there is no transition from S₀, so masking structurally impossible states
    (prior 0 -> -inf) BEFORE normalizing keeps an impossible-but-well-fitting
    state from setting the ``logsumexp`` reference and underflowing every
    supported state:

    ``p(S_1=j | y_1) = softmax_j( log p(y_1 | S_1=j) + log p(S_1=j) )``.

    Parameters
    ----------
    state_cond_log_lik : jax.Array, shape (n_discrete_states,)
        ``log p(y_1 | S_1=j)`` for each discrete state.
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Caller-supplied prior ``p(S_1)``; exact zeros are structural.

    Returns
    -------
    filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        Posterior ``p(S_1=j | y_1)``; a structural zero stays exactly 0.
    marginal_log_likelihood : jax.Array
        ``log p(y_1)`` contribution (scalar array); NaN on a malformed prior.
    """
    init_discrete_state_prob = _normalize_initial_discrete_prob(
        init_discrete_state_prob
    )
    # ``_log_prob_preserve_zeros``: exact zero -> -inf; positive -> its true log,
    # floored only at the dtype's tiny (NOT 1e-10), so a tiny prior like 1e-12 is
    # faithful. Malformed (NaN) prior -> NaN, so the posterior/LL fail loud.
    log_prior = jnp.where(
        jnp.isnan(init_discrete_state_prob),
        jnp.nan,
        _log_prob_preserve_zeros(init_discrete_state_prob),
    )
    # A structural-zero-prior state (log_prior = -inf) is impossible at t=1; its
    # likelihood is irrelevant, but ``NaN/+inf lik + (-inf) = NaN`` would poison
    # the logsumexp and NaN the whole posterior despite valid supported states.
    # Substitute a finite placeholder into those impossible lanes before adding.
    # (A NaN log_prior is the malformed-prior fail-loud signal -- isneginf is
    # False there -- so it is preserved, not masked.)
    impossible = jnp.isneginf(log_prior)
    safe_state_cond_log_lik = jnp.where(impossible, 0.0, state_cond_log_lik)
    log_unnorm = safe_state_cond_log_lik + log_prior
    marginal_log_likelihood = logsumexp(log_unnorm)
    # Degenerate recovery (mirrors the later-timestep dynamics fallback in
    # _update_discrete_state_probabilities): an impossible first observation
    # (every state's likelihood -inf) makes exp(log_unnorm - marginal) =
    # exp(-inf - -inf) = NaN. With no dynamics at t=1, the "prediction" is the
    # prior itself, so drop the likelihood and fall back to the normalized prior.
    # ``logsumexp(log_prior) == log(sum prior) == 0`` (the prior is normalized),
    # so ``exp(log_prior - 0)`` returns the prior (structural zeros preserved).
    # Gate on -inf specifically: a +inf likelihood (pathological, not impossible)
    # or a NaN marginal (malformed prior) must NOT be recovered to the prior --
    # it flows through and surfaces as a fail-loud NaN posterior below.
    obs_impossible = jnp.isneginf(marginal_log_likelihood)
    posterior_log_unnorm = jnp.where(obs_impossible, log_prior, log_unnorm)
    posterior_log_norm = jnp.where(
        obs_impossible, logsumexp(log_prior), marginal_log_likelihood
    )
    filter_discrete_prob = jnp.exp(posterior_log_unnorm - posterior_log_norm)
    # Floor only reachable states with a *finite* posterior log-unnorm (genuine
    # underflow, recoverable). A state whose likelihood is -inf (the data rules
    # it out at t=1) or whose prior is a structural zero has posterior_log_unnorm
    # = -inf and stays exactly 0 -- matching the scan body's
    # ``floorable = support & isfinite(log_marginal)``, so a data-ruled-out
    # first state is not fabricated to 1e-10. (NaN -> not floorable, propagates.)
    floorable = jnp.isfinite(posterior_log_unnorm)
    filter_discrete_prob = _floor_supported_marginal(filter_discrete_prob, floorable)
    # Floor a degenerate -inf contribution so the marginal LL stays finite for
    # EM accumulation (matching the scan body); NaN (malformed prior) propagates.
    marginal_log_likelihood = jnp.where(
        jnp.isneginf(marginal_log_likelihood),
        _LOG_DISCRETE_STABILITY_FLOOR,
        marginal_log_likelihood,
    )
    return filter_discrete_prob, marginal_log_likelihood


def _first_timestep_kalman_update(
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    obs_t: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Handle first timestep with x₁ convention (measurement update only).

    For the first observation y₁, we treat init_state_cond_mean/cov as p(x₁ | S₁)
    and apply only the measurement update (no dynamics prediction). This aligns
    with the EM M-step which sets init_state from smoother_means[0] = x₁|T.

    Parameters
    ----------
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Prior belief about x₁ given S₁, p(x₁ | S₁)
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Prior covariance of x₁ given S₁
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Prior probability p(S₁)
    obs_t : jax.Array, shape (n_obs_dim,)
        Observation at first timestep y₁
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Observation matrices H_j for each discrete state
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Observation noise covariances R_j for each discrete state

    Returns
    -------
    state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Posterior state-conditional means p(x₁ | y₁, S₁=j)
    state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Posterior state-conditional covariances
    filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        Posterior discrete state probabilities p(S₁ | y₁)
    pair_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
        For smoother compatibility. At t=1 there is no S₀, so this is broadcast
        to be constant across the S₀ axis (not diagonal); the GPB2 smoother
        relies on that constant-across-i property at the first timestep.
    pair_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional covariance, broadcast over the nonexistent S₀ axis.
    pair_cond_filter_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
        Pair-filter probabilities, with P(S₀=i | S₁=j, y₁) represented as
        uniform because S₀ does not exist under the x₁ convention.
    marginal_log_likelihood : jax.Array
        Log p(y₁) contribution (scalar array)
    """
    n_discrete_states = init_state_cond_mean.shape[-1]

    # Apply measurement update for each discrete state (no dynamics prediction)
    # vmap over discrete states j
    vmapped_update = jax.vmap(
        kalman_measurement_update,
        in_axes=(-1, -1, None, -1, -1),
        out_axes=(-1, -1, -1),
    )
    state_cond_filter_mean, state_cond_filter_cov, state_cond_log_lik = vmapped_update(
        init_state_cond_mean,
        init_state_cond_cov,
        obs_t,
        measurement_matrix,
        measurement_cov,
    )

    # Zero-preserving log-space discrete posterior (shared with the
    # point-process first-step): honors a structural-zero prior, keeps a tiny
    # prior faithful, and fails loud (NaN) on a malformed prior.
    filter_discrete_prob, marginal_log_likelihood = _first_timestep_discrete_update(
        state_cond_log_lik, init_discrete_state_prob
    )

    # For smoother compatibility: create pair_cond_filter_mean and cov
    # At t=1, there's no S₀, so broadcast to be constant across the S₀ axis
    pair_cond_filter_mean = jnp.broadcast_to(
        state_cond_filter_mean[:, None, :],
        (state_cond_filter_mean.shape[0], n_discrete_states, n_discrete_states),
    )
    pair_cond_filter_cov = jnp.broadcast_to(
        state_cond_filter_cov[:, :, None, :],
        (*state_cond_filter_cov.shape[:2], n_discrete_states, n_discrete_states),
    )
    pair_cond_filter_prob = jnp.broadcast_to(
        filter_discrete_prob[None, :] / n_discrete_states,
        (n_discrete_states, n_discrete_states),
    )

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_prob,
        pair_cond_filter_mean,
        pair_cond_filter_cov,
        pair_cond_filter_prob,
        marginal_log_likelihood,
    )


@jax.jit
def switching_kalman_filter(
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    obs: jax.Array,
    discrete_transition_matrix: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[
    jax.Array,  # Filtered mean of the continuous latent state
    jax.Array,  # Filtered covariance of the continuous latent state
    jax.Array,  # Filtered probability of the discrete states
    jax.Array,  # Pair-conditional filter mean trajectory
    jax.Array,  # Pair-conditional filter covariance trajectory
    jax.Array,  # Pair-conditional discrete probability trajectory
    jax.Array,  # Marginal log likelihood of the observations (scalar array)
]:
    """Switching Kalman filter for a linear Gaussian state space model with discrete states.

    This filter uses the x₁ convention where init parameters represent the state
    at the first observation time (not before it). For the first observation y₁,
    only a measurement update is applied (no dynamics prediction). For subsequent
    observations y_t (t > 1), the standard predict-then-update cycle is used.

    This convention ensures consistency with the EM M-step, which sets
    init_state_cond_mean = smoother_means[0] = x₁|T.

    Parameters
    ----------
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Prior belief about x₁ given S₁, p(x₁ | S₁ = j) for each discrete state.
        This is the prior on the latent state *at* the first observation, before
        incorporating y₁.
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Prior covariance of x₁ given S₁ for each discrete state.
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Prior discrete state probabilities p(S₁ = j).
    obs : jax.Array, shape (n_time, n_obs_dim)
        Observations $y_{1:T}$
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Transition matrix for the discrete states $B$
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Transition matrix for the continuous states $A$
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Process noise covariance matrix. $\\Sigma$
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Map observations to the continuous states $H$
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Measurement variance. $R$

    Returns
    -------
    state_cond_filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        Filtered mean of the continuous latent state
    state_cond_filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        Filtered covariance of the continuous latent state
    filter_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        Filtered probability of the discrete states
    pair_cond_filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional filter mean trajectory E[x_t | S_{t-1}=i, S_t=j, y_{1:t}].
        The first timestep uses the x_1 convention (broadcast over the
        nonexistent S_0). GPB1 callers use the last timestep ``[-1]``; the GPB2
        smoother consumes the full trajectory.
    pair_cond_filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional filter covariance trajectory Cov[x_t | S_{t-1}=i, S_t=j, y_{1:t}].
    pair_cond_filter_prob : jax.Array, shape (n_time, n_discrete_states, n_discrete_states)
        Pair-filter discrete probability trajectory
        ``P(S_{t-1}=i, S_t=j | y_{1:t})``. The first timestep broadcasts over
        the nonexistent previous state and sums to the filtered probability for
        ``S_1``.
    marginal_log_likelihood : jax.Array
        Marginal log likelihood of the observations (scalar array)

    """

    def _step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        obs_t: jax.Array,
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],  # Next carry
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],  # Stacked output
    ]:
        """One step of the switching Kalman filter.

        Parameters
        ----------
        carry : tuple
            prev_state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
                Previous state mean.
            prev_state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
                Previous state covariance.
            prev_filter_discrete_prob : jax.Array, shape (n_discrete_states,)
                Previous discrete state probabilities
            pair_cond_marginal_log_likelihood : jax.Array
                Previous marginal log likelihood (scalar array)
        obs_t : jax.Array, shape (n_obs_dim,)
            Observation at time t

        Returns
        -------
        carry : tuple
            prev_state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
                Posterior state mean.
            prev_state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
                Posterior state covariance.
            prev_filter_discrete_prob : jax.Array, shape (n_discrete_states,)
                Posterior discrete state probabilities
            marginal_log_likelihood : jax.Array
                Posterior marginal log likelihood (scalar array)
        stack : tuple
            state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
                Posterior state mean.
            state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
                Posterior state covariance.
            filter_discrete_prob : jax.Array, shape (n_discrete_states,)
                Posterior discrete state probabilities
            pair_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
                Conditional means of the continuous latent state
        """
        (
            prev_state_cond_filter_mean,
            prev_state_cond_filter_cov,
            prev_filter_discrete_prob,
            marginal_log_likelihood,
            prev_support,
        ) = carry

        # Kalman update for each pair of discrete states
        # P(x_t | y_{1:t}, S_t = j, S_{t-1} = i)
        # vmap twice over the discrete states
        (
            pair_cond_filter_mean,  # x^{ij}_{t|t}
            pair_cond_filter_cov,  # V^{ij}_{t|t}
            pair_cond_marginal_log_likelihood,  # log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)
        ) = _kalman_filter_update_per_discrete_state_pair(
            prev_state_cond_filter_mean,  # x^i_{t-1|t-1}
            prev_state_cond_filter_cov,  # V^i_{t-1|t-1}
            obs_t,  # y_t
            continuous_transition_matrix,  # A
            process_cov,  # Sigma
            measurement_matrix,  # H
            measurement_cov,  # R
        )

        (
            filter_discrete_prob,  # M_{t|t}(j) = P(S_t=j | y_{1:t})
            filter_backward_cond_prob,  # P(S_{t-1}=i | S_t=j, y_{1:t})
            log_predictive,  # log p(y_t | y_{1:t-1})
            next_support,  # S_t
        ) = _update_discrete_state_probabilities(
            pair_cond_marginal_log_likelihood,  # shape (n_discrete_states, n_discrete_states)
            discrete_transition_matrix,  # P(S_t=j | S_{t-1}=i)
            prev_filter_discrete_prob,  # M_{t-1|t-1}(i)
            prev_support,  # S_{t-1}
        )
        pair_cond_filter_prob = (
            filter_backward_cond_prob * filter_discrete_prob[None, :]
        )

        marginal_log_likelihood += log_predictive

        # Collapse pair-conditional Gaussians P(x_t | ..., S_t=j, S_{t-1}=i)
        # over S_{t-1}=i using weights P(S_{t-1}=i | S_t=j, y_{1:t})
        # to get state-conditional Gaussians P(x_t | ..., S_t=j)
        state_cond_filter_mean, state_cond_filter_cov = (
            collapse_gaussian_mixture_per_discrete_state(
                pair_cond_filter_mean,  # x^{ij}_{t|t}
                pair_cond_filter_cov,  # V^{ij}_{t|t}
                filter_backward_cond_prob,  # P(S_{t-1}=i | S_t=j, y_{1:t})
            )
        )

        return (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_prob,
            marginal_log_likelihood,
            next_support,
        ), (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_prob,
            pair_cond_filter_mean,
            pair_cond_filter_cov,
            pair_cond_filter_prob,
        )

    # Handle first timestep with x₁ convention: measurement update only (no dynamics)
    # init_state_cond_mean represents p(x₁ | S₁), the prior for the first observation
    (
        first_state_cond_mean,
        first_state_cond_cov,
        first_discrete_prob,
        first_pair_cond_mean,
        first_pair_cond_cov,
        first_pair_cond_prob,
        first_log_lik,
    ) = _first_timestep_kalman_update(
        init_state_cond_mean,
        init_state_cond_cov,
        init_discrete_state_prob,
        obs[0],
        measurement_matrix,
        measurement_cov,
    )

    # Run predict-then-update for t=2,...,T
    # jax.lax.scan handles empty inputs (obs[1:] when n_time=1) gracefully.
    # The scan now emits the full pair-conditional filter trajectory as stacked
    # outputs (the GPB2 smoother needs every timestep, not just the last).
    # Structural support S_1 from the *sanitized* prior (the same normalization
    # _first_timestep_discrete_update applies before forming the posterior), not
    # the raw prior. Otherwise a non-finite prior entry (e.g. +inf) that the
    # sanitizer clamps to 0 would be marked reachable here and resurrected to
    # ~1e-10 at the next step via the log-prior floor, disagreeing with its
    # exactly-0 first-step posterior. Threaded through the scan and advanced one
    # transition per step, so an underflowed reachable state is not mistaken for
    # a structurally impossible one.
    first_support = _normalize_initial_discrete_prob(init_discrete_state_prob) > 0.0
    (
        (_, _, _, marginal_log_likelihood, _),
        (
            rest_state_cond_filter_mean,
            rest_state_cond_filter_cov,
            rest_filter_discrete_state_prob,
            rest_pair_cond_filter_mean,
            rest_pair_cond_filter_cov,
            rest_pair_cond_filter_prob,
        ),
    ) = jax.lax.scan(
        _step,
        (
            first_state_cond_mean,
            first_state_cond_cov,
            first_discrete_prob,
            first_log_lik,
            first_support,
        ),
        obs[1:],
    )

    # Prepend first timestep results
    state_cond_filter_mean = jnp.concatenate(
        [first_state_cond_mean[None, ...], rest_state_cond_filter_mean], axis=0
    )
    state_cond_filter_cov = jnp.concatenate(
        [first_state_cond_cov[None, ...], rest_state_cond_filter_cov], axis=0
    )
    filter_discrete_state_prob = jnp.concatenate(
        [first_discrete_prob[None, ...], rest_filter_discrete_state_prob], axis=0
    )
    # Full pair-conditional filter trajectories, E[x_t | S_{t-1}=i, S_t=j, y_{1:t}]
    # and its covariance. The first timestep uses the x_1 convention (broadcast
    # over the nonexistent S_0). The GPB2 smoother needs the whole trajectory;
    # GPB1 callers take the last timestep, ``pair_cond_filter_mean[-1]``.
    pair_cond_filter_mean = jnp.concatenate(
        [first_pair_cond_mean[None, ...], rest_pair_cond_filter_mean], axis=0
    )
    pair_cond_filter_cov = jnp.concatenate(
        [first_pair_cond_cov[None, ...], rest_pair_cond_filter_cov], axis=0
    )
    pair_cond_filter_prob = jnp.concatenate(
        [first_pair_cond_prob[None, ...], rest_pair_cond_filter_prob], axis=0
    )

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_state_prob,
        pair_cond_filter_mean,
        pair_cond_filter_cov,
        pair_cond_filter_prob,
        marginal_log_likelihood,
    )


def switching_kalman_viterbi(
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    obs: jax.Array,
    discrete_transition_matrix: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> jax.Array:
    """Find the most likely discrete state sequence for a switching Kalman model.

    Runs the GPB2 filter forward pass to collect pair-conditional
    log-likelihoods ``log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)``, then
    applies a pairwise Viterbi algorithm that accounts for the dependence
    of the emission on both the current and previous discrete state.

    Parameters are identical to :func:`switching_kalman_filter`.

    Returns
    -------
    states : jax.Array, shape (n_time,)
        Most likely discrete state sequence (integer-valued).
    """
    n_discrete_states = init_state_cond_mean.shape[-1]

    # Viterbi is not JIT-decorated, so reject a malformed prior loudly here
    # rather than returning an arbitrary path from a NaN/degenerate posterior.
    _init_prob = jnp.asarray(init_discrete_state_prob)
    if not (
        bool(jnp.all(jnp.isfinite(_init_prob)))
        and bool(jnp.sum(jnp.where(_init_prob > 0.0, _init_prob, 0.0)) > 0.0)
    ):
        raise ValueError(
            "switching_kalman_viterbi: init_discrete_state_prob must be finite "
            "and sum to a positive value (it defines the initial support); got "
            f"{jnp.asarray(_init_prob)}."
        )

    # --- First timestep: measurement update only (x₁ convention) -----------
    (
        first_state_cond_mean,
        first_state_cond_cov,
        first_discrete_prob,
        _,
        _,
        _,
        _,
    ) = _first_timestep_kalman_update(
        init_state_cond_mean,
        init_state_cond_cov,
        init_discrete_state_prob,
        obs[0],
        measurement_matrix,
        measurement_cov,
    )
    if obs.shape[0] == 1:
        return jnp.array([jnp.argmax(first_discrete_prob)], dtype=jnp.int32)

    # --- Forward pass: collect pair log-likelihoods -----------------------
    def _step(carry, obs_t):
        prev_mean, prev_cov, prev_prob, prev_support = carry

        # Pair-conditional Kalman update (same as in the filter)
        (
            pair_cond_filter_mean,
            pair_cond_filter_cov,
            pair_cond_log_lik,  # (K, K): log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)
        ) = _kalman_filter_update_per_discrete_state_pair(
            prev_mean,
            prev_cov,
            obs_t,
            continuous_transition_matrix,
            process_cov,
            measurement_matrix,
            measurement_cov,
        )

        # Collapse mixture (same as filter) to get state-conditional
        # means/covs for the next step (log-space, support-masked).
        (
            filter_prob,
            backward_cond_prob,
            _,
            next_support,
        ) = _update_discrete_state_probabilities(
            pair_cond_log_lik,
            discrete_transition_matrix,
            prev_prob,
            prev_support,
        )

        state_cond_mean, state_cond_cov = collapse_gaussian_mixture_per_discrete_state(
            pair_cond_filter_mean,
            pair_cond_filter_cov,
            backward_cond_prob,
        )

        return (
            state_cond_mean,
            state_cond_cov,
            filter_prob,
            next_support,
        ), pair_cond_log_lik

    _, pair_log_liks = jax.lax.scan(
        _step,
        (
            first_state_cond_mean,
            first_state_cond_cov,
            first_discrete_prob,
            jnp.asarray(init_discrete_state_prob) > 0.0,  # S_1
        ),
        obs[1:],
    )
    # pair_log_liks: (T-1, K, K) where entry (t, i, j) is
    # log p(y_{t+1} | y_{1:t}, S_t=i, S_{t+1}=j)

    # --- Pairwise Viterbi -------------------------------------------------
    # Backward pass: accumulate best future scores with pair log-likelihoods
    log_trans = _log_prob_preserve_zeros(discrete_transition_matrix)

    def _viterbi_backward(best_next_score, t):
        # scores[i, j] = log A(i,j) + pair_log_lik(t, i, j) + best_future(j)
        scores = log_trans + pair_log_liks[t] + best_next_score[None, :]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    n_rest = pair_log_liks.shape[0]
    best_second_score, best_next_states = jax.lax.scan(
        _viterbi_backward,
        jnp.zeros(n_discrete_states),
        jnp.arange(n_rest),
        reverse=True,
    )

    # Best first state: first_discrete_prob is already p(S_1 | y_1),
    # so the first-obs likelihood is already encoded — do not add it again.
    first_state = jnp.argmax(
        _log_prob_preserve_zeros(first_discrete_prob) + best_second_score
    )

    # Forward trace
    def _viterbi_forward(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    _, states = jax.lax.scan(_viterbi_forward, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


_kalman_smoother_update_per_discrete_state_pair = jax.vmap(
    jax.vmap(
        _kalman_smoother_update, in_axes=(None, None, -1, -1, None, None), out_axes=-1
    ),
    in_axes=(-1, -1, None, None, -1, -1),
    out_axes=-1,
)

# GPB2 triple vmap: produces S³ triple-conditional smoother posteriors of x_t
# indexed by (i, j, k) = (S_{t-1}, S_t, S_{t+1}).
#
# Each RTS update for the triple (i, j, k) combines
#   next smoother : E[x_{t+1} | S_t=j, S_{t+1}=k, y_{1:T}]   (carry, pair-cond)
#   filter        : E[x_t     | S_{t-1}=i, S_t=j, y_{1:t}]   (pair-cond filter)
#   dynamics      : A_k, Q_k  (the x_t -> x_{t+1} transition is governed by
#                              S_{t+1}=k, matching the filter/GPB1/M-step)
# The middle state S_t=j is SHARED between the carry (its first axis) and the
# pair-conditional filter (its second axis); the two are matched on j rather
# than treated as independent. Argument order of ``_kalman_smoother_update`` is
# (next_mean, next_cov, filter_mean, filter_cov, Q, A) with shapes
#   carry_mean (L, S_j, S_k), carry_cov (L, L, S_j, S_k),
#   filter_mean (L, S_i, S_j), filter_cov (L, L, S_i, S_j),
#   Q (L, L, S_k), A (L, L, S_k).
#
# Outer vmap: k = S_{t+1} (carry 2nd axis + dynamics)
# Middle vmap: j = S_t (carry 1st axis + filter 2nd axis, shared)
# Inner vmap: i = S_{t-1} (filter 1st axis)
#
# Output shapes: mean (L, S_i, S_j, S_k), cov/cross (L, L, S_i, S_j, S_k)
_gpb2_kalman_smoother_update_triple = jax.vmap(
    jax.vmap(
        jax.vmap(
            _kalman_smoother_update,
            in_axes=(None, None, 1, 2, None, None),
            out_axes=-1,
        ),
        in_axes=(1, 2, 2, 3, None, None),
        out_axes=-1,
    ),
    in_axes=(2, 3, None, None, 2, 2),
    out_axes=-1,
)


def _collapse_triple_to_pair(
    triple_mean: jax.Array,
    triple_cov: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Collapse S³ triple-conditional Gaussians to S² pair-conditional.

    Marginalizes over the last conditioning variable (S_{t+1}=k) for each
    pair (S_{t-1}=i, S_t=j).

    Parameters
    ----------
    triple_mean : jax.Array, shape (n_latent, S_i, S_j, S_k)
    triple_cov : jax.Array, shape (n_latent, n_latent, S_i, S_j, S_k)
    weights : jax.Array, shape (S_j, S_k)
        P(S_{t+1}=k | S_t=j, y_{1:T}) — forward conditional probability.

    Returns
    -------
    pair_mean : jax.Array, shape (n_latent, S_i, S_j)
    pair_cov : jax.Array, shape (n_latent, n_latent, S_i, S_j)
    """
    # For each (i, j): collapse over k using weights[j, :]
    # collapse_gaussian_mixture expects: means (L, S_k), cov (L, L, S_k), weights (S_k,)
    #
    # After outer vmap slices j: triple_mean becomes (L, S_i, S_k)
    # Inner vmap iterates over S_i (axis -2), leaving (L, S_k) per slice — correct.
    _collapse_over_k_for_fixed_j = jax.vmap(
        collapse_gaussian_mixture,
        in_axes=(-2, -2, None),  # vmap over i (second-to-last), keep k as mixture
        out_axes=(-1, -1),
    )
    _collapse_over_k = jax.vmap(
        _collapse_over_k_for_fixed_j,
        in_axes=(-2, -2, 0),  # vmap over j (second-to-last of 4D), weights axis 0
        out_axes=(-1, -1),
    )
    return _collapse_over_k(triple_mean, triple_cov, weights)


def _update_smoother_discrete_probabilities(
    filter_discrete_prob: jax.Array,
    discrete_state_transition_matrix: jax.Array,
    next_smoother_discrete_prob: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """

    Parameters
    ----------
    filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        Pr(S_t=j | y_{1:t}), shape (n_discrete_states,), M_{t | t}(j)
    discrete_state_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Z[j, k] = Pr(S_{t+1}=k | S_t=j) -- row-stochastic (row = current state,
        column = next state), matching the usage below.
    next_smoother_discrete_prob : jax.Array, shape (n_discrete_states,)
        Pr(S_{t+1}=k | y_{1:T}), shape (n_discrete_states,) M_{t+1 | T}(k)

    Returns
    -------
    smoother_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Pr(S_t=j | y_{1:T}), shape (n_discrete_states,),  M_{t | T}(j)
    smoother_backward_cond_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
        Pr(S_t=j | S_{t+1}=k, y_{1:T}), shape (n_discrete_states, n_discrete_states), U^{j | k}_t
    joint_smoother_discrete_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
        Pr(S_t=j, S_{t+1}=k | y_{1:T}), shape (n_discrete_states, n_discrete_states)
    smoother_forward_cond_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
        Pr(S_{t+1}=k | S{t}=j, y_{1:T}), shape (n_discrete_states, n_discrete_states), W^{k | j}_t

    """
    # Stabilize filter input to prevent underflow propagation.
    # Note: next_smoother_discrete_prob is NOT stabilized here — it is
    # stabilized in the carry output so that the stored smoother_prob[t+1]
    # and the value used to compute joint_prob[t] are always identical.
    filter_discrete_prob = _stabilize_probability_vector_preserving_zeros(
        filter_discrete_prob
    )

    # Discrete smoother prob
    # P(S_t = j, S_{t+1} = k | y_{1:T})
    # Unnormalized joint P(S_t=j) * P(S_{t+1}=k | S_t=j)
    unnormalized = filter_discrete_prob[:, None] * discrete_state_transition_matrix
    # Normalize columns to get P(S_t=j | S_{t+1}=k, y_{1:t})
    smoother_backward_cond_prob = _divide_safe(
        unnormalized, jnp.sum(unnormalized, axis=0)
    )

    joint_smoother_discrete_prob = (
        smoother_backward_cond_prob * next_smoother_discrete_prob
    )
    # P(S_t = j | y_{1:T})
    smoother_discrete_state_prob = jnp.sum(joint_smoother_discrete_prob, axis=1)
    # P(S_{t+1} = k | S_t = j, y_{1:T})
    smoother_forward_cond_prob = _divide_safe(
        joint_smoother_discrete_prob, smoother_discrete_state_prob[:, None]
    )

    return (
        smoother_discrete_state_prob,
        smoother_backward_cond_prob,
        joint_smoother_discrete_prob,
        smoother_forward_cond_prob,
    )


@jax.jit
def switching_kalman_smoother(
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    filter_discrete_state_prob: jax.Array,
    last_filter_conditional_cont_mean: jax.Array,
    process_cov: jax.Array,
    continuous_transition_matrix: jax.Array,
    discrete_state_transition_matrix: jax.Array,
) -> tuple[
    jax.Array,  # Overall smoother mean
    jax.Array,  # Overall smoother covariance
    jax.Array,  # Smoother discrete state probabilities
    jax.Array,  # Smoother joint discrete state probabilities
    jax.Array,  # Overall smoother cross covariance
    jax.Array,  # State conditional smoother means
    jax.Array,  # State conditional smoother covariances
    jax.Array,  # Pair conditional smoother cross covariances
    jax.Array,  # Pair conditional smoother means
]:
    """GPB1/IMM approximate switching Kalman smoother.

    This is an approximate smoother: the forward pass collapses K^2 mixture
    components to K at each step, so the state-conditional quantities (means
    and covariances) are approximate. The pair-conditional cross-covariances
    and means are computed from these collapsed quantities. For exact
    pair-conditional structure, use ``switching_kalman_smoother_gpb2``.

    Parameters
    ----------
    filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    filter_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    last_filter_conditional_cont_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    discrete_state_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)

    Returns
    -------
    overall_smoother_mean : jax.Array, shape (n_time, n_cont_states)
    overall_smoother_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    overall_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states)
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    pair_cond_smoother_cross_covs : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    pair_cond_smoother_means : jax.Array, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=j, S_{t+1}=k] - needed for correct M-step beta computation.
    """

    def _step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        args: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ]:
        """

        Parameters
        ----------
        carry : tuple
            next_smoother_mean : jax.Array, shape (n_cont_states, n_discrete_states)
            next_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
            next_discrete_state_prob : jax.Array, shape (n_discrete_states,)
            next_conditional_cont_means : jax.Array, unused compatibility slot
        args : tuple
            state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
            state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
            filter_discrete_prob : jax.Array, shape (n_discrete_states,)

        Returns
        -------
        carry : tuple
            next_state_cond_smoother_mean : jax.Array, shape (n_cont_states, n_discrete_states)
            next_state_cond_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
            next_smoother_discrete_prob : jax.Array, shape (n_discrete_states,)
            next_pair_cond_smoother_mean : jax.Array, unused compatibility slot
        args : tuple
            state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
            state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
            filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        """
        (
            next_state_cond_smoother_mean,
            next_state_cond_smoother_cov,
            next_smoother_discrete_prob,
            _unused_next_pair_cond_smoother_mean,
        ) = carry

        state_cond_filter_mean, state_cond_filter_cov, filter_discrete_prob = args

        # 1. Smooth for each discrete state pair
        (
            pair_cond_smoother_mean,  # E[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_discrete_states, n_discrete_states)
            pair_cond_smoother_covs,  # Cov[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
            pair_cond_smoother_cross_covs,  # Cov[X_t, X_{t+1} | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        ) = _kalman_smoother_update_per_discrete_state_pair(
            next_state_cond_smoother_mean,  # E[X_{t+1} | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states)
            next_state_cond_smoother_cov,  # Cov[X_{t+1} | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_cont_states, n_discrete_states)
            state_cond_filter_mean,  # E[X_t | y_{1:t}, S_t=j], shape (n_cont_states, n_discrete_states)
            state_cond_filter_cov,  # Cov[X_t | y_{1:t}, S_t=j], shape (n_cont_states, n_cont_states, n_discrete_states)
            process_cov,  # Cov[X_{t+1} | X_t], shape (n_cont_states, n_cont_states, n_discrete_states)
            continuous_transition_matrix,  # E[X_{t+1} | X_t], shape (n_cont_states, n_cont_states, n_discrete_states)
        )

        # 1b. Stabilize pair-conditional smoother outputs (GPB1 safety net).
        # Cap at 1e8x the filter covariance trace. This is large enough to
        # not interfere with normal EM dynamics (where smoother cov can
        # legitimately exceed filter cov by 10-100x from GPB1 collapse) but
        # catches the exponential blowup that leads to overflow on long
        # sequences (where growth reaches 10^100+).
        max_allowed_trace = _compute_max_allowed_trace(state_cond_filter_cov)

        _warn_if_cov_cap_engaged(pair_cond_smoother_covs, max_allowed_trace, "GPB1")
        _cap = partial(_cap_covariance_trace, max_allowed_trace=max_allowed_trace)
        pair_cond_smoother_covs, cov_scales = jax.vmap(
            jax.vmap(_cap, in_axes=-1, out_axes=(-1, -1)),
            in_axes=-1,
            out_axes=(-1, -1),
        )(pair_cond_smoother_covs)  # cov_scales: (n_discrete_states, n_discrete_states)

        # Rescale the lag-one cross-covariance coherently with the covariance
        # cap: if ``P_t`` was scaled by ``alpha`` per (S_t, S_{t+1}) lane, scale
        # ``C`` by ``sqrt(alpha)``. This is a congruence transform of the joint
        # block ``[[P_t, C], [C^T, P_{t+1}]]``, so a PSD joint (guaranteed by the
        # RTS smoother pre-cap) stays PSD -- unlike an independent Frobenius cap,
        # which can leave the joint indefinite and corrupt the M-step ``beta``.
        pair_cond_smoother_cross_covs = (
            jnp.sqrt(cov_scales) * pair_cond_smoother_cross_covs
        )

        # Never clip a finite smoother mean: future data can legitimately move
        # it far from the current filtered mean, so there is no valid magnitude
        # threshold. Only fail loud (propagate NaN) on a non-finite or
        # overflow-prone mean, letting the EM loop roll the step back.
        pair_cond_smoother_mean = _guard_smoother_mean(pair_cond_smoother_mean, "GPB1")

        # 2. Compute discrete state intermediates
        (
            smoother_discrete_state_prob,  # Pr(S_t=j | y_{1:T}), shape (n_discrete_states,),  M_{t | T}(j)
            smoother_backward_cond_prob,  # Pr(S_t=j | S_{t+1}=k, y_{1:T}), shape (n_discrete_states, n_discrete_states), U^{j | k}_t
            joint_smoother_discrete_prob,  # Pr(S_t=j, S_{t+1}=k | y_{1:T}), shape (n_discrete_states, n_discrete_states)
            smoother_forward_cond_prob,  # Pr(S_{t+1}=k | S{t}=j, y_{1:T}), shape (n_discrete_states, n_discrete_states), W^{k | j}_t
        ) = _update_smoother_discrete_probabilities(
            filter_discrete_prob,  # Pr(S_t=j | y_{1:t}), shape (n_discrete_states,), M_{t | t}(j)
            discrete_state_transition_matrix,  # Z[j,k] = Pr(S_{t+1}=k | S_t=j), row-stochastic, shape (n_discrete_states, n_discrete_states)
            next_smoother_discrete_prob,  # Pr(S_{t+1}=k | y_{1:T}), shape (n_discrete_states,) M_{t+1 | T}(k)
        )

        # 3. Collapse conditional mean and covariance (n_states x n_states -> n_states)
        (
            state_cond_smoother_means,  # E[X_t | y_{1:T}, S_{t}=j], shape (n_cont_states, n_discrete_states)
            state_cond_smoother_covs,  # Cov[X_t | y_{1:T}, S_{t}=j], shape (n_cont_states, n_cont_states, n_discrete_states)
        ) = collapse_gaussian_mixture_over_next_discrete_state(
            pair_cond_smoother_mean,  # E[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_discrete_states, n_discrete_states)
            pair_cond_smoother_covs,  # Cov[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
            smoother_forward_cond_prob,  # Pr(S_{t+1} = k | S{t} = j, y_{1:T}), shape (n_discrete_states, n_discrete_states), W^{k | j}_t
        )

        # 4. Collapse to single mean and covariance (n_states -> 1)
        (
            overall_smoother_mean,  # E[X_t | y_{1:T}], shape (n_cont_states,), x_{t|T}
            overall_smoother_covs,  # Cov[X_t | y_{1:T}], shape (n_cont_states, n_cont_states)
        ) = collapse_gaussian_mixture(
            state_cond_smoother_means,  # E[X_t | y_{1:T}, S_{t}=j], shape (n_cont_states, n_discrete_states)
            state_cond_smoother_covs,  # Cov[X_t | y_{1:T}, S_{t}=j], shape (n_cont_states, n_cont_states, n_discrete_states)
            smoother_discrete_state_prob,  # Pr(S_t = j | y_{1:T}), shape (n_discrete_states,),  M_{t | T}(j)
        )

        # 5. Collapse lag-one cross covariance. _kalman_smoother_update returns
        # Cov(x_t, x_{t+1}), so keep x_t as the first argument throughout the
        # mixture collapses.
        next_pair_cond_smoother_mean = jnp.broadcast_to(
            next_state_cond_smoother_mean[:, None, :],
            pair_cond_smoother_mean.shape,
        )

        (
            smoother_mean_t_cond_Stplus1,  # E[X_t | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states), x^{()k}_{t | T}
            state_cond_smoother_mean_tplus1,  # E[X_{t+1} | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states), x^{()k}_{t+1 | T}
            state_cond_smoother_cross_cov,  # Cov(X_t, X_{t+1} | y_{1:T}, S_{t+1}=k), shape (n_cont_states, n_cont_states, n_discrete_states), V^k_{t, t+1 | T}:
        ) = collapse_cross_gaussian_mixture_across_states(
            pair_cond_smoother_mean,  # E[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_discrete_states, n_discrete_states), x^{(j)k}_{t | T}
            # GPB1 approximation: E[X_{t+1} | y, S_{t+1}=k],
            # broadcast over S_t=j.
            next_pair_cond_smoother_mean,
            pair_cond_smoother_cross_covs,  # Cov(X_t, X_{t+1} | y_{1:T}, S_t=j, S_{t+1}=k), shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states), V^{j(k)}_{t,t+1 | T}
            smoother_backward_cond_prob,  # Pr(S_t=j | S_{t+1}=k, y_{1:T}), shape (n_discrete_states, n_discrete_states), U^{j | k}_t
        )

        # Cross collapse to a single Gaussian
        # overall_smoother_cross_cov, shape (n_cont_states, n_cont_states)
        _, _, overall_smoother_cross_cov = collapse_gaussian_mixture_cross_covariance(
            smoother_mean_t_cond_Stplus1,  # E[X_t | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states), x^{()k}_{t | T}
            state_cond_smoother_mean_tplus1,  # E[X_{t+1} | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states), x^{()k}_{t+1 | T}
            state_cond_smoother_cross_cov,  # V^k_{t, t+1 | T}: state-conditional smoother cross covariance
            next_smoother_discrete_prob,  # Pr(S_{t+1} = k | y_{1:T}), M_{t+1 | T}(k)
        )

        # Stabilize smoother_discrete_state_prob in the carry only,
        # so it is consistent when used as next_smoother_discrete_prob
        # in the next backward step. The output arrays store the
        # un-stabilized version for exact joint/marginal consistency.
        # Preserve structural zeros (a state impossible at this timestep must
        # not be resurrected to 1e-10).
        stabilized_smoother_prob = _stabilize_probability_vector_preserving_zeros(
            smoother_discrete_state_prob
        )

        return (
            state_cond_smoother_means,
            state_cond_smoother_covs,
            stabilized_smoother_prob,
            next_pair_cond_smoother_mean,
        ), (
            overall_smoother_mean,
            overall_smoother_covs,
            smoother_discrete_state_prob,
            joint_smoother_discrete_prob,
            overall_smoother_cross_cov,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_covs,
            pair_cond_smoother_mean,  # E[X_t | y_{1:T}, S_t=j, S_{t+1}=k]
        )

    init_carry = (
        filter_mean[-1],  # shape (n_cont_states, n_discrete_states)
        filter_cov[-1],  # shape (n_cont_states, n_cont_states, n_discrete_states)
        filter_discrete_state_prob[-1],  # shape (n_discrete_states,)
        last_filter_conditional_cont_mean,  # shape (n_cont_states, n_discrete_states, n_discrete_states)
    )

    (
        _,
        (
            overall_smoother_mean,
            overall_smoother_covs,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            overall_smoother_cross_cov,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_covs,
            pair_cond_smoother_means,
        ),
    ) = jax.lax.scan(
        _step,
        init_carry,
        (
            filter_mean[:-1],
            filter_cov[:-1],
            filter_discrete_state_prob[:-1],
        ),
        reverse=True,
    )

    # Guard the terminal mean (appended outside the scan, so the per-step
    # _guard_smoother_mean does not cover it): a finite-but-unrepresentable
    # terminal mean must fail loud like the interior, not pass straight through.
    last_filter_mean = _guard_smoother_mean(filter_mean[-1], "GPB1")
    last_smoother_mean, last_smoother_cov = collapse_gaussian_mixture(
        last_filter_mean, filter_cov[-1], filter_discrete_state_prob[-1]
    )
    overall_smoother_mean = jnp.concatenate(
        [overall_smoother_mean, last_smoother_mean[None]], axis=0
    )
    overall_smoother_covs = jnp.concatenate(
        [overall_smoother_covs, last_smoother_cov[None]], axis=0
    )
    smoother_discrete_state_prob = jnp.concatenate(
        [smoother_discrete_state_prob, filter_discrete_state_prob[-1][None]], axis=0
    )
    state_cond_smoother_means = jnp.concatenate(
        [state_cond_smoother_means, last_filter_mean[None]], axis=0
    )
    state_cond_smoother_covs = jnp.concatenate(
        [state_cond_smoother_covs, filter_cov[-1][None]], axis=0
    )

    return (
        overall_smoother_mean,
        overall_smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        overall_smoother_cross_cov,
        state_cond_smoother_means,
        state_cond_smoother_covs,
        pair_cond_smoother_cross_covs,
        pair_cond_smoother_means,
    )


@jax.jit
def switching_kalman_smoother_gpb2(
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    filter_discrete_state_prob: jax.Array,
    pair_cond_filter_mean: jax.Array,
    pair_cond_filter_cov: jax.Array,
    pair_cond_filter_prob: jax.Array,
    process_cov: jax.Array,
    continuous_transition_matrix: jax.Array,
) -> tuple[
    jax.Array,  # overall_smoother_mean
    jax.Array,  # overall_smoother_covs
    jax.Array,  # smoother_discrete_state_prob
    jax.Array,  # smoother_joint_discrete_state_prob
    jax.Array,  # overall_smoother_cross_cov
    jax.Array,  # state_cond_smoother_means
    jax.Array,  # state_cond_smoother_covs
    jax.Array,  # pair_cond_smoother_cross_covs
    jax.Array,  # pair_cond_smoother_means
    jax.Array,  # pair_cond_smoother_covs_mstep
    jax.Array,  # next_pair_cond_smoother_means
]:
    """GPB2 (Kim second-order) switching Kalman smoother.

    Carries pair-conditional (S_t, S_{t+1}) Gaussians and pair discrete
    probabilities through the backward pass instead of collapsing to
    state-conditional at each step (GPB1). Each
    backward step combines the pair-conditional smoother of ``x_{t+1}`` (the
    carry) with the pair-conditional *filter* of ``x_t`` to form the
    triple-conditional posterior ``E[x_t | S_{t-1}=i, S_t=j, S_{t+1}=k,
    y_{1:T}]``, which is then marginalized two ways:

    * over ``S_{t+1}=k`` with the carried pair smoother probabilities
      ``P(S_t=j, S_{t+1}=k | y_{1:T})`` to give the pair-conditional smoother
      ``E[x_t | S_{t-1}=i, S_t=j]`` carried to the next backward step;
    * over ``S_{t-1}=i`` with pair-filter probabilities
      ``P(S_{t-1}=i | S_t=j, y_{1:t})`` to give ``E[x_t | S_t=j, S_{t+1}=k]``
      and its covariance / cross-covariance, the GPB2 pair-conditional
      sufficient statistics the M-step consumes.

    The ``x_t -> x_{t+1}`` transition is governed by ``A[..., S_{t+1}]``,
    matching :func:`switching_kalman_filter`, :func:`switching_kalman_smoother`
    (GPB1), and the M-step. Unlike GPB1 this consumes the full pair-conditional
    filter trajectory, which is what makes the S² backward recursion
    well-defined. This remains the GPB/Kim moment-matching approximation, not
    exact path enumeration.

    Parameters
    ----------
    filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        State-conditional filter mean E[x_t | S_t=j, y_{1:t}].
    filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        State-conditional filter covariance.
    filter_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        M_{t|t}(j) = P(S_t=j | y_{1:t}).
    pair_cond_filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional filter mean E[x_t | S_{t-1}=i, S_t=j, y_{1:t}], as
        returned (whole trajectory) by :func:`switching_kalman_filter`.
    pair_cond_filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional filter covariance Cov[x_t | S_{t-1}=i, S_t=j, y_{1:t}].
    pair_cond_filter_prob : jax.Array, shape (n_time, n_discrete_states, n_discrete_states)
        Pair-filter discrete probabilities
        ``P(S_{t-1}=i, S_t=j | y_{1:t})``. These provide the past-state
        conditioning weights used when marginalizing the GPB2 triple smoother.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)

    Returns
    -------
    Eleven arrays, in the same layout as before: overall smoother mean/cov,
    smoother discrete marginal/joint probabilities, overall cross-covariance,
    state-conditional smoother means/covs, and the pair-conditional
    cross-covariances, means, covariances, and next-step means used by the
    M-step. See the type annotation above for the order.

    Computational cost is ~2x GPB1 for S=2 (8 vs 4 RTS updates per step).
    """

    def _step(carry, args):
        (
            next_pair_cond_smoother_mean,  # E[x_{t+1} | S_t=j, S_{t+1}=k], (L, Sj, Sk)
            next_pair_cond_smoother_cov,  # (L, L, Sj, Sk)
            next_pair_smoother_prob,  # P(S_t=j, S_{t+1}=k | y), (Sj, Sk)
        ) = carry

        (
            pair_filter_mean,  # E[x_t | S_{t-1}=i, S_t=j, y_{1:t}], (L, Si, Sj)
            pair_filter_cov,  # (L, L, Si, Sj)
            filter_discrete_prob,  # M_{t|t}(j), (S,)
            pair_filter_prob,  # P(S_{t-1}=i, S_t=j | y_{1:t}), (Si, Sj)
        ) = args

        # 1. Triple-conditional RTS update: E[x_t | S_{t-1}=i, S_t=j, S_{t+1}=k].
        # The middle S_t=j is shared between the carry (pair smoother of x_{t+1})
        # and the pair-conditional filter of x_t; dynamics use A_k, Q_k.
        (
            triple_mean,  # (L, Si, Sj, Sk)
            triple_cov,  # (L, L, Si, Sj, Sk)
            triple_cross,  # (L, L, Si, Sj, Sk) = Cov[x_t, x_{t+1} | i, j, k]
        ) = _gpb2_kalman_smoother_update_triple(
            next_pair_cond_smoother_mean,  # (L, Sj, Sk)
            next_pair_cond_smoother_cov,  # (L, L, Sj, Sk)
            pair_filter_mean,  # (L, Si, Sj)
            pair_filter_cov,  # (L, L, Si, Sj)
            process_cov,  # Q_k
            continuous_transition_matrix,  # A_k
        )

        # 1b. Stabilize triple-conditional outputs. GPB2 can feed moment-matched
        # pair covariances back through the RTS update, so apply a moderate
        # covariance trust region relative to the pair-filter covariance. The
        # moderate cap is what is actually applied to triple_cov, so surface a
        # lower-severity note when it engages instead of clipping silently; the
        # huge-cap warning stays as a separate catastrophic-divergence signal.
        pair_filter_traces = jax.vmap(
            jax.vmap(jnp.trace, in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1
        )(pair_filter_cov)  # (Si, Sj)
        max_filter_trace = jnp.max(pair_filter_traces)
        max_allowed = max_filter_trace * _GPB2_COV_CAP_MULTIPLIER + 1.0
        catastrophic_allowed = max_filter_trace * _COV_CAP_MULTIPLIER + 1.0

        _warn_if_cov_cap_engaged(
            triple_cov,
            max_allowed,
            "GPB2",
            cap_multiplier=_GPB2_COV_CAP_MULTIPLIER,
            upper_allowed_trace=catastrophic_allowed,
            moderate=True,
        )
        _warn_if_cov_cap_engaged(
            triple_cov,
            catastrophic_allowed,
            "GPB2",
        )
        _cap = partial(_cap_covariance_trace, max_allowed_trace=max_allowed)
        triple_cov, triple_cov_scales = jax.vmap(
            jax.vmap(
                jax.vmap(_cap, in_axes=-1, out_axes=(-1, -1)),
                in_axes=-1,
                out_axes=(-1, -1),
            ),
            in_axes=-1,
            out_axes=(-1, -1),
        )(triple_cov)  # triple_cov_scales: (Si, Sj, Sk)

        # Never clip a finite smoother mean (see GPB1 above / _guard_smoother_mean):
        # only fail loud on a non-finite or overflow-prone mean.
        triple_mean = _guard_smoother_mean(triple_mean, "GPB2")

        # Rescale the cross-covariance coherently with the covariance cap (see
        # GPB1): C scaled by sqrt(alpha) where alpha scaled the triple
        # covariance. This congruence transform keeps the joint block PSD,
        # unlike an independent Frobenius cross-cap.
        triple_cross = jnp.sqrt(triple_cov_scales) * triple_cross

        # 2. Discrete pair smoother probabilities for (S_t=j, S_{t+1}=k).
        joint_smoother_discrete_prob = next_pair_smoother_prob
        smoother_discrete_state_prob = jnp.sum(joint_smoother_discrete_prob, axis=1)
        smoother_forward_cond_prob = _divide_safe(
            joint_smoother_discrete_prob,
            smoother_discrete_state_prob[:, None],
        )

        # 3. Pair-filter backward probability for (S_{t-1}=i, S_t=j). This uses
        # the actual pair-filter posterior, including the observation likelihood
        # at time t, instead of reconstructing P(S_{t-1}|S_t) from one-step
        # marginals alone.
        smoother_backward_cond_prob_prev = _divide_safe(
            pair_filter_prob,
            filter_discrete_prob[None, :],
        )

        # 4. Carry: pair-conditional smoother E[x_t | S_{t-1}=i, S_t=j] obtained
        # by marginalizing the triple over the future S_{t+1}=k with W^{k|j}.
        (
            carry_pair_cond_mean,  # (L, Si, Sj)
            carry_pair_cond_cov,  # (L, L, Si, Sj)
        ) = _collapse_triple_to_pair(
            triple_mean,
            triple_cov,
            smoother_forward_cond_prob,
        )

        # 5. M-step statistics: pair-conditional E[x_t | S_t=j, S_{t+1}=k] and
        # its covariance / cross-covariance, from marginalizing the triple over
        # the past S_{t-1}=i with V^{i|j}. Because E[x_{t+1} | i,j,k] does not
        # depend on i (the carry conditions only on j,k), the cross-covariance
        # has no i-spread term.
        mstep_pair_cond_means = jnp.einsum(
            "lijk,ij->ljk", triple_mean, smoother_backward_cond_prob_prev
        )
        # Cov[x_t | S_t=j, S_{t+1}=k] = E_i[Cov] + Var_i[E]
        mstep_pair_cond_covs = jnp.einsum(
            "abijk,ij->abjk", triple_cov, smoother_backward_cond_prob_prev
        )
        mean_diff = triple_mean - mstep_pair_cond_means[:, None, :, :]
        mstep_pair_cond_covs += jnp.einsum(
            "aijk,bijk,ij->abjk",
            mean_diff,
            mean_diff,
            smoother_backward_cond_prob_prev,
        )
        pair_cond_smoother_cross_covs = jnp.einsum(
            "abijk,ij->abjk", triple_cross, smoother_backward_cond_prob_prev
        )
        mstep_next_pair_cond_means = next_pair_cond_smoother_mean  # (L, Sj, Sk)

        pair_smoother_prob_prev = (
            smoother_backward_cond_prob_prev * smoother_discrete_state_prob[None, :]
        )

        # 6. State-conditional smoother E[x_t | S_t=j] by marginalizing the
        # (S_t=j, S_{t+1}=k) pair over the future with W^{k|j}.
        (
            state_cond_smoother_means,  # (L, Sj)
            state_cond_smoother_covs,  # (L, L, Sj)
        ) = collapse_gaussian_mixture_over_next_discrete_state(
            mstep_pair_cond_means,
            mstep_pair_cond_covs,
            smoother_forward_cond_prob,
        )

        # 7. Overall smoother (collapse S_j -> 1).
        (
            overall_smoother_mean,
            overall_smoother_covs,
        ) = collapse_gaussian_mixture(
            state_cond_smoother_means,
            state_cond_smoother_covs,
            smoother_discrete_state_prob,
        )

        pair_means_flat = mstep_pair_cond_means.reshape(
            mstep_pair_cond_means.shape[0], -1
        )
        next_pair_means_flat = mstep_next_pair_cond_means.reshape(
            mstep_next_pair_cond_means.shape[0], -1
        )
        pair_cross_flat = pair_cond_smoother_cross_covs.reshape(
            pair_cond_smoother_cross_covs.shape[0],
            pair_cond_smoother_cross_covs.shape[1],
            -1,
        )
        joint_smoother_prob_flat = joint_smoother_discrete_prob.reshape(-1)

        # Overall lag-one cross covariance Cov(x_t, x_{t+1}) via the law of
        # total covariance over pair states (S_t, S_{t+1}).
        _, _, overall_smoother_cross_cov = collapse_gaussian_mixture_cross_covariance(
            pair_means_flat,
            next_pair_means_flat,
            pair_cross_flat,
            joint_smoother_prob_flat,
        )

        # Stabilize the pair probability matrix in the carry only; the output
        # arrays store the un-stabilized value for joint/marginal consistency.
        # Preserve structural zeros so an impossible (S_t, S_{t+1}) pair is not
        # resurrected.
        stabilized_pair_smoother_prob = _stabilize_probability_vector_preserving_zeros(
            pair_smoother_prob_prev.reshape(-1)
        ).reshape(pair_smoother_prob_prev.shape)

        return (
            carry_pair_cond_mean,  # E[x_t | S_{t-1}=i, S_t=j] — next carry
            carry_pair_cond_cov,
            stabilized_pair_smoother_prob,
        ), (
            overall_smoother_mean,
            overall_smoother_covs,
            smoother_discrete_state_prob,
            joint_smoother_discrete_prob,
            overall_smoother_cross_cov,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_covs,
            mstep_pair_cond_means,  # E[x_t | S_t=j, S_{t+1}=k]
            mstep_pair_cond_covs,  # Cov[x_t | S_t=j, S_{t+1}=k]
            mstep_next_pair_cond_means,  # E[x_{t+1} | S_t=j, S_{t+1}=k]
        )

    # Initialize carry from the last timestep, where smoother == filter (no
    # future data). The pair-conditional filter at T-1 is exactly
    # E[x_{T-1} | S_{T-2}=j, S_{T-1}=k, y_{1:T-1}].
    init_carry = (
        pair_cond_filter_mean[-1],
        pair_cond_filter_cov[-1],
        pair_cond_filter_prob[-1],
    )

    (
        _,
        (
            overall_smoother_mean,
            overall_smoother_covs,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            overall_smoother_cross_cov,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_covs,
            pair_cond_smoother_means,
            pair_cond_smoother_covs_mstep,
            next_pair_cond_smoother_means,
        ),
    ) = jax.lax.scan(
        _step,
        init_carry,
        (
            pair_cond_filter_mean[:-1],
            pair_cond_filter_cov[:-1],
            filter_discrete_state_prob[:-1],
            pair_cond_filter_prob[:-1],
        ),
        reverse=True,
    )

    # Append last timestep (same as GPB1). Guard the terminal mean too, since
    # the per-step _guard_smoother_mean does not cover this appended value.
    last_filter_mean = _guard_smoother_mean(filter_mean[-1], "GPB2")
    last_overall_mean, last_overall_cov = collapse_gaussian_mixture(
        last_filter_mean,
        filter_cov[-1],
        filter_discrete_state_prob[-1],
    )
    overall_smoother_mean = jnp.concatenate(
        [overall_smoother_mean, last_overall_mean[None]], axis=0
    )
    overall_smoother_covs = jnp.concatenate(
        [overall_smoother_covs, last_overall_cov[None]], axis=0
    )
    smoother_discrete_state_prob = jnp.concatenate(
        [smoother_discrete_state_prob, filter_discrete_state_prob[-1][None]], axis=0
    )
    state_cond_smoother_means = jnp.concatenate(
        [state_cond_smoother_means, last_filter_mean[None]], axis=0
    )
    state_cond_smoother_covs = jnp.concatenate(
        [state_cond_smoother_covs, filter_cov[-1][None]], axis=0
    )

    return (
        overall_smoother_mean,
        overall_smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        overall_smoother_cross_cov,
        state_cond_smoother_means,
        state_cond_smoother_covs,
        pair_cond_smoother_cross_covs,
        pair_cond_smoother_means,
        pair_cond_smoother_covs_mstep,
        next_pair_cond_smoother_means,
    )


def weighted_sum_of_outer_products(
    x: jax.Array, y: jax.Array, weights: jax.Array
) -> jax.Array:
    """Compute the weighted outer sum of two arrays.
    Parameters
    ----------
    x : jax.Array, shape (n_time, x_dims, n_discrete_states)
        First array.
    y : jax.Array, shape (n_time, y_dims, n_discrete_states)
        Second array.
    weights : jax.Array, shape (n_time, n_discrete_states)
        Weights for the outer sum.

    Returns
    -------
    weighted_sum_of_outer_products: jax.Array, shape (x_dims, y_dims, n_discrete_states)
        Weighted outer sum of x and y.
    """
    return jnp.einsum("tcm, tdm, tm -> cdm", x, y, weights)


psd_solve_per_discrete_state = jax.vmap(
    lambda x, y: psd_solve(x, y.T).T, in_axes=(-1, -1), out_axes=-1
)

cov_solve_per_discrete_state = jax.vmap(
    lambda x, y, z, n: stabilize_covariance(_divide_safe(x - y @ z.T, n)),
    in_axes=(-1, -1, -1, -1),
    out_axes=-1,
)


@jax.jit
def _switching_kalman_m_step_inner(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    gamma1: jax.Array,
    beta: jax.Array,
    transition_pseudo_counts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """JIT-compiled inner M-step. All arrays are concrete (no None branching).

    ``gamma1`` and ``beta`` are the transition sufficient statistics,
    pre-computed by the outer ``switching_kalman_maximization_step`` which
    resolves the Optional pair-conditional paths before calling this function.
    ``gamma2`` is reconstructed from the state-conditional moments at t >= 1.
    With exact E-step moments this equals the corresponding pair-conditional
    second moment by total expectation. With GPB/IMM moment matching it remains
    an approximation because the smoother does not expose next-time pair
    covariances for a fully pair-consistent Q update.
    ``transition_pseudo_counts`` is zeros for ML or ``(alpha - 1)`` for MAP.
    """
    n_time = smoother_discrete_state_prob.sum(axis=0)
    n_time_1 = smoother_discrete_state_prob[1:].sum(axis=0)

    # Compute intermediate expectation terms
    gamma = jnp.sum(
        state_cond_smoother_covs * smoother_discrete_state_prob[:, None, None], axis=0
    ) + weighted_sum_of_outer_products(
        state_cond_smoother_means,
        state_cond_smoother_means,
        smoother_discrete_state_prob,
    )

    delta = weighted_sum_of_outer_products(
        obs[..., None], state_cond_smoother_means, smoother_discrete_state_prob
    )
    alpha = weighted_sum_of_outer_products(
        obs[..., None], obs[..., None], smoother_discrete_state_prob
    )

    first_gamma = (
        state_cond_smoother_covs[0] * smoother_discrete_state_prob[0, None, None]
    ) + weighted_sum_of_outer_products(
        state_cond_smoother_means[:1],
        state_cond_smoother_means[:1],
        smoother_discrete_state_prob[:1],
    )
    gamma2 = gamma - first_gamma

    # Measurement matrix and covariance
    measurement_matrix = psd_solve_per_discrete_state(gamma, delta)
    measurement_cov = cov_solve_per_discrete_state(
        alpha, measurement_matrix, delta, n_time
    )

    # Transition matrix
    continuous_transition_matrix = psd_solve_per_discrete_state(gamma1, beta)

    # Process covariance
    process_cov = cov_solve_per_discrete_state(
        gamma2, continuous_transition_matrix, beta, n_time_1
    )

    # Initial mean and covariance
    init_state_cond_mean = state_cond_smoother_means[0]
    init_state_cond_cov = state_cond_smoother_covs[0]

    # Discrete transition matrix (MAP with optional Dirichlet prior)
    expected_counts = smoother_joint_discrete_state_prob.sum(axis=0)
    expected_counts = expected_counts + transition_pseudo_counts
    discrete_state_transition = _divide_safe(
        expected_counts,
        jnp.sum(expected_counts, axis=1, keepdims=True),
    )

    # Initial discrete state probabilities
    init_discrete_state_prob = smoother_discrete_state_prob[0]
    init_discrete_state_prob = _divide_safe(
        init_discrete_state_prob, jnp.sum(init_discrete_state_prob)
    )

    return (
        continuous_transition_matrix,
        measurement_matrix,
        process_cov,
        measurement_cov,
        init_state_cond_mean,
        init_state_cond_cov,
        discrete_state_transition,
        init_discrete_state_prob,
    )


def switching_kalman_maximization_step(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
    transition_prior: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Maximization step for the switching Kalman filter.

    Parameters
    ----------
    obs : jax.Array, shape (n_time, n_obs_dim)
        Observations.
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        smoother mean.
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        smoother covariance.
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        smoother discrete state probabilities.
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
        smoother joint discrete state probabilities.
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        smoother cross-covariance.
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional
        means for transition sufficient statistics. If None, uses the approximate factored form.
    pair_cond_smoother_covs : jax.Array | None, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Cov[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional
        covariances for gamma1. If None, falls back to state-conditional covariances.
    next_pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_{t+1} | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional
        next-step means for beta. If None, falls back to state-conditional means.
    transition_prior : jax.Array | None, shape (n_discrete_states, n_discrete_states)
        Dirichlet prior alpha parameters for the discrete transition matrix.
        If provided, adds (alpha - 1) pseudo-counts to the expected transition
        counts (MAP estimate). Use ``get_transition_prior(concentration, stickiness,
        n_states)`` from ``contingency_belief`` to construct. If None, uses the
        standard ML estimate.

    Returns
    -------
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Transition matrix.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Measurement matrix.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Process covariance.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Measurement covariance.
    init_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Initial mean of the continuous latent state.
    init_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Initial covariance of the continuous latent state.
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Transition matrix for the discrete states.
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Initial discrete state probabilities.


    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """

    if obs.shape[0] < 2:
        raise ValueError(
            "switching_kalman_maximization_step requires at least two time "
            "steps to estimate transition and process-noise parameters."
        )

    # Resolve Optional args into concrete arrays before calling JIT inner.
    # Compute gamma1 and beta (transition sufficient statistics) here
    # because the code path depends on which Optional args are provided.

    # Transition sufficient statistics (gamma1, beta), weighted by the joint
    # probability P(S_t=i, S_{t+1}=j). Delegated to the shared
    # compute_transition_sufficient_stats so the two identical implementations
    # cannot drift apart.
    gamma1, beta = compute_transition_sufficient_stats(
        state_cond_smoother_means=state_cond_smoother_means,
        state_cond_smoother_covs=state_cond_smoother_covs,
        smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
        pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
        pair_cond_smoother_means=pair_cond_smoother_means,
        pair_cond_smoother_covs=pair_cond_smoother_covs,
        next_pair_cond_smoother_means=next_pair_cond_smoother_means,
    )

    # Transition prior pseudo-counts (zeros = no prior = ML estimate)
    n_discrete_states = smoother_discrete_state_prob.shape[1]
    if transition_prior is not None:
        transition_prior = jnp.asarray(transition_prior)
        expected_shape = (n_discrete_states, n_discrete_states)
        if transition_prior.shape != expected_shape:
            raise ValueError(
                f"transition_prior must have shape {expected_shape}, "
                f"got {transition_prior.shape}."
            )
        if not isinstance(transition_prior, jax.core.Tracer):
            if not bool(jnp.all(jnp.isfinite(transition_prior))):
                raise ValueError("transition_prior must contain only finite values.")
            if not bool(jnp.all(transition_prior >= 1.0)):
                raise ValueError(
                    "transition_prior entries must be >= 1.0 for this MAP "
                    "update, because alpha - 1 is added as non-negative "
                    "pseudo-counts."
                )
        transition_pseudo_counts = transition_prior - 1.0
    else:
        transition_pseudo_counts = jnp.zeros((n_discrete_states, n_discrete_states))

    return _switching_kalman_m_step_inner(
        obs,
        state_cond_smoother_means,
        state_cond_smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        gamma1,
        beta,
        transition_pseudo_counts,
    )


def _compute_expected_complete_log_likelihood_reference(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
    discrete_transition_matrix: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
) -> jax.Array:
    """Compute the expected complete-data log-likelihood E_q[log p(y, x, s | θ)].

    This is the Q-function that the EM algorithm maximizes for a fixed
    approximate posterior. The optional pair-conditional inputs make the
    transition term closer to the GPB2 approximation used by the M-step.

    Parameters
    ----------
    obs : jax.Array, shape (n_time, n_obs_dim)
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional
        quantities for the transition Q-function term.
    pair_cond_smoother_covs : jax.Array | None, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Cov[X_t | y_{1:T}, S_t=i, S_{t+1}=j].
    next_pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_{t+1} | y_{1:T}, S_t=i, S_{t+1}=j].

    Returns
    -------
    expected_complete_ll : jax.Array
        E_q[log p(y, x, s | θ)] (scalar array)

    Notes
    -----
    When GPB2 pair-conditional quantities are provided, the transition
    Q-function term uses pair-conditional means and covariances for x_t.
    Cov[x_{t+1} | S_t, S_{t+1}] is still approximated with the
    state-conditional Cov[x_{t+1} | S_{t+1}] since the GPB2 smoother does
    not produce that quantity directly. This affects only diagnostics, not
    the closed-form parameter updates.
    """
    n_time = obs.shape[0]
    n_discrete_states = smoother_discrete_state_prob.shape[1]
    n_cont_states = state_cond_smoother_means.shape[1]

    # 1. E_q[log p(s_1)] - initial discrete state
    log_init_discrete = jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(init_discrete_state_prob)
    )

    # 2. E_q[log p(x_1 | s_1)] - initial continuous state
    log_init_cont = jnp.zeros(())
    for j in range(n_discrete_states):
        # E_q[log N(x_1; μ_0^j, Σ_0^j) | s_1=j]
        # = -0.5 * (log|Σ_0^j| + tr(Σ_0^j^{-1} E_q[(x_1 - μ_0^j)(x_1 - μ_0^j)^T | s_1=j]))
        mean_j = init_state_cond_mean[:, j]
        cov_j = init_state_cond_cov[:, :, j]

        # E_q[(x_1 - μ_0^j)(x_1 - μ_0^j)^T | s_1=j]
        smoother_mean_j = state_cond_smoother_means[0, :, j]
        smoother_cov_j = state_cond_smoother_covs[0, :, :, j]
        diff = smoother_mean_j - mean_j
        expected_outer = smoother_cov_j + jnp.outer(diff, diff)

        log_det = jnp.linalg.slogdet(cov_j)[1]
        trace_term = jnp.trace(psd_solve(cov_j, expected_outer))
        log_prob_j = -0.5 * (n_cont_states * jnp.log(2 * jnp.pi) + log_det + trace_term)
        log_init_cont += smoother_discrete_state_prob[0, j] * log_prob_j

    # 3. E_q[sum_t log p(s_t | s_{t-1})] - discrete state transitions
    log_discrete_trans = jnp.sum(
        smoother_joint_discrete_state_prob * _safe_log(discrete_transition_matrix)
    )

    # 4. E_q[sum_t log p(x_t | x_{t-1}, s_t)] - continuous state transitions
    log_cont_trans = jnp.zeros(())
    for j in range(n_discrete_states):
        A_j = continuous_transition_matrix[:, :, j]
        Q_j = process_cov[:, :, j]
        log_det_Q = jnp.linalg.slogdet(Q_j)[1]

        for t in range(n_time - 1):
            # Sum over source states i weighted by P(s_t=i, s_{t+1}=j | y_{1:T})
            for i in range(n_discrete_states):
                weight = smoother_joint_discrete_state_prob[t, i, j]

                # E_q[(x_{t+1} - A_j x_t)(x_{t+1} - A_j x_t)^T | s_t=i, s_{t+1}=j]
                # Use pair-conditional quantities when available (GPB2),
                # otherwise fall back to state-conditional (GPB1 approximate).
                if pair_cond_smoother_means is not None:
                    m_t_ij = pair_cond_smoother_means[t, :, i, j]
                else:
                    m_t_ij = state_cond_smoother_means[t, :, i]

                if next_pair_cond_smoother_means is not None:
                    m_t1_ij = next_pair_cond_smoother_means[t, :, i, j]
                else:
                    m_t1_ij = state_cond_smoother_means[t + 1, :, j]

                if pair_cond_smoother_covs is not None:
                    V_t_ij = pair_cond_smoother_covs[t, :, :, i, j]
                else:
                    V_t_ij = state_cond_smoother_covs[t, :, :, i]

                # For V_{t+1}, we don't have pair-conditional Cov[x_{t+1} | S_t, S_{t+1}]
                # separately (only Cov[x_t | S_t, S_{t+1}]). Use state-conditional.
                V_t1_j = state_cond_smoother_covs[t + 1, :, :, j]

                # Stored as Cov[x_t, x_{t+1} | ...] by the RTS helper.
                cross_cov_t_t1_ij = pair_cond_smoother_cross_cov[t, :, :, i, j]

                # E[x_{t+1} x_{t+1}^T | ...]
                E_xt1_xt1 = V_t1_j + jnp.outer(m_t1_ij, m_t1_ij)
                # E[x_t x_t^T | ...]
                E_xt_xt = V_t_ij + jnp.outer(m_t_ij, m_t_ij)
                # E[x_{t+1} x_t^T | ...]
                E_xt1_xt = cross_cov_t_t1_ij.T + jnp.outer(m_t1_ij, m_t_ij)

                # E[(x_{t+1} - A x_t)(x_{t+1} - A x_t)^T]
                # = E[x_{t+1} x_{t+1}^T] - A E[x_t x_{t+1}^T] - E[x_{t+1} x_t^T] A^T + A E[x_t x_t^T] A^T
                expected_residual = (
                    E_xt1_xt1
                    - A_j @ E_xt1_xt.T
                    - E_xt1_xt @ A_j.T
                    + A_j @ E_xt_xt @ A_j.T
                )

                trace_term = jnp.trace(psd_solve(Q_j, expected_residual))
                log_prob = -0.5 * (
                    n_cont_states * jnp.log(2 * jnp.pi) + log_det_Q + trace_term
                )
                log_cont_trans += jnp.where(weight > 0, weight * log_prob, 0.0)

    # 5. E_q[sum_t log p(y_t | x_t, s_t)] - observations
    log_obs = jnp.zeros(())
    for j in range(n_discrete_states):
        H_j = measurement_matrix[:, :, j]
        R_j = measurement_cov[:, :, j]
        log_det_R = jnp.linalg.slogdet(R_j)[1]
        n_obs = obs.shape[1]

        for t in range(n_time):
            weight = smoother_discrete_state_prob[t, j]

            m_t_j = state_cond_smoother_means[t, :, j]
            V_t_j = state_cond_smoother_covs[t, :, :, j]

            # E[(y_t - H x_t)(y_t - H x_t)^T | s_t=j]
            pred_mean = H_j @ m_t_j
            diff = obs[t] - pred_mean
            # E[x_t x_t^T | s_t=j]
            E_xt_xt = V_t_j + jnp.outer(m_t_j, m_t_j)
            # E[(y - Hx)(y - Hx)^T] = (y - H m)(y - H m)^T + H V H^T
            expected_residual = jnp.outer(diff, diff) + H_j @ V_t_j @ H_j.T

            trace_term = jnp.trace(psd_solve(R_j, expected_residual))
            log_prob = -0.5 * (n_obs * jnp.log(2 * jnp.pi) + log_det_R + trace_term)
            log_obs += jnp.where(weight > 0, weight * log_prob, 0.0)

    return (
        log_init_discrete
        + log_init_cont
        + log_discrete_trans
        + log_cont_trans
        + log_obs
    )


def _compute_posterior_entropy_reference(
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    state_cond_smoother_covs: jax.Array,
) -> jax.Array:
    """Compute the entropy of the approximate posterior H(q).

    For the switching Kalman filter with mixture collapse approximation:
    H(q) = H(q(s)) + E_q(s)[H(q(x|s))]

    Parameters
    ----------
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)

    Returns
    -------
    entropy : jax.Array
        H(q(x, s)) (scalar array)
    """
    n_time = smoother_discrete_state_prob.shape[0]
    n_discrete_states = smoother_discrete_state_prob.shape[1]
    n_cont_states = state_cond_smoother_covs.shape[1]

    # 1. Entropy of discrete state sequence
    # H(q(s)) = -sum_t E_q[log q(s_t | s_{t-1})]
    # For t=1: -sum_j q(s_1=j) log q(s_1=j)
    discrete_entropy = -jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(smoother_discrete_state_prob[0])
    )

    # For t>1: -sum_{t,i,j} q(s_{t-1}=i, s_t=j) log q(s_t=j | s_{t-1}=i)
    # q(s_t=j | s_{t-1}=i) = q(s_{t-1}=i, s_t=j) / q(s_{t-1}=i)
    for t in range(n_time - 1):
        marginal_prev = smoother_discrete_state_prob[t]
        joint = smoother_joint_discrete_state_prob[t]
        cond = _divide_safe(joint, marginal_prev[:, None])
        discrete_entropy -= jnp.sum(joint * _safe_log(cond))

    # 2. Entropy of continuous states given discrete states
    # H(q(x|s)) = sum_t sum_j q(s_t=j) * H(q(x_t | s_t=j))
    # For Gaussian: H(N(μ, Σ)) = 0.5 * (k + k*log(2π) + log|Σ|)
    cont_entropy = jnp.zeros(())
    for j in range(n_discrete_states):
        for t in range(n_time):
            weight = smoother_discrete_state_prob[t, j]
            cov_j = state_cond_smoother_covs[t, :, :, j]
            log_det = jnp.linalg.slogdet(cov_j)[1]
            gaussian_entropy = 0.5 * (
                n_cont_states * (1 + jnp.log(2 * jnp.pi)) + log_det
            )
            cont_entropy += jnp.where(weight > 0, weight * gaussian_entropy, 0.0)

    return discrete_entropy + cont_entropy


# ---------------------------------------------------------------------------
# Vectorized ELBO functions (JIT-compatible, no Python loops)
# ---------------------------------------------------------------------------


def _weighted_gaussian_log_prob(
    mean: jax.Array,
    cov: jax.Array,
    smoother_mean: jax.Array,
    smoother_cov: jax.Array,
    n_cont_states: int,
) -> jax.Array:
    """Log N(smoother_mean; mean, cov) including the expected covariance term."""
    diff = smoother_mean - mean
    expected_outer = smoother_cov + jnp.outer(diff, diff)
    log_det = jnp.linalg.slogdet(cov)[1]
    trace_term = jnp.trace(psd_solve(cov, expected_outer))
    return -0.5 * (n_cont_states * jnp.log(2 * jnp.pi) + log_det + trace_term)


@jax.jit
def compute_expected_complete_log_likelihood(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
    discrete_transition_matrix: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
) -> jax.Array:
    """Vectorized expected complete-data log-likelihood E_q[log p(y, x, s | θ)].

    Equivalent to ``_compute_expected_complete_log_likelihood_reference`` but
    uses vectorized JAX operations instead of Python loops, making it
    JIT-compilable and significantly faster for long sequences.

    See ``_compute_expected_complete_log_likelihood_reference`` for full
    parameter documentation.
    """
    n_cont_states = state_cond_smoother_means.shape[1]
    n_obs = obs.shape[1]

    # 1. Initial discrete state
    log_init_discrete = jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(init_discrete_state_prob)
    )

    # 2. Initial continuous state — vmap over j
    log_probs_init = jax.vmap(
        lambda mean_j, cov_j, sm_j, sc_j: _weighted_gaussian_log_prob(
            mean_j, cov_j, sm_j, sc_j, n_cont_states
        )
    )(
        init_state_cond_mean.T,  # (K, n)
        jnp.moveaxis(init_state_cond_cov, -1, 0),  # (K, n, n)
        state_cond_smoother_means[0].T,  # (K, n)
        jnp.moveaxis(state_cond_smoother_covs[0], -1, 0),  # (K, n, n)
    )  # (K,)
    log_init_cont = jnp.sum(smoother_discrete_state_prob[0] * log_probs_init)

    # 3. Discrete state transitions
    log_discrete_trans = jnp.sum(
        smoother_joint_discrete_state_prob * _safe_log(discrete_transition_matrix)
    )

    # 4. Continuous state transitions — resolve Optional branching, then vectorize
    # Resolve pair-conditional means/covs to concrete arrays
    if pair_cond_smoother_means is not None:
        m_t = pair_cond_smoother_means  # (T-1, n, K_i, K_j)
    else:
        # Broadcast state-cond means: E[x_t | S_t=i] for all j
        m_t = jnp.broadcast_to(
            state_cond_smoother_means[:-1, :, :, None],
            smoother_joint_discrete_state_prob.shape[:1]
            + state_cond_smoother_means.shape[1:2]
            + smoother_joint_discrete_state_prob.shape[1:],
        )

    if next_pair_cond_smoother_means is not None:
        m_t1 = next_pair_cond_smoother_means  # (T-1, n, K_i, K_j)
    else:
        # Broadcast state-cond means: E[x_{t+1} | S_{t+1}=j] for all i
        m_t1 = jnp.broadcast_to(
            state_cond_smoother_means[1:, :, None, :],
            smoother_joint_discrete_state_prob.shape[:1]
            + state_cond_smoother_means.shape[1:2]
            + smoother_joint_discrete_state_prob.shape[1:],
        )

    if pair_cond_smoother_covs is not None:
        V_t = pair_cond_smoother_covs  # (T-1, n, n, K_i, K_j)
    else:
        # Broadcast state-cond covs: Cov[x_t | S_t=i] for all j
        V_t = jnp.broadcast_to(
            state_cond_smoother_covs[:-1, :, :, :, None],
            smoother_joint_discrete_state_prob.shape[:1]
            + state_cond_smoother_covs.shape[1:3]
            + smoother_joint_discrete_state_prob.shape[1:],
        )

    # V_{t+1} is always state-conditional
    V_t1 = jnp.broadcast_to(
        state_cond_smoother_covs[1:, :, :, None, :],
        smoother_joint_discrete_state_prob.shape[:1]
        + state_cond_smoother_covs.shape[1:3]
        + smoother_joint_discrete_state_prob.shape[1:],
    )

    def _cont_trans_log_prob_single(
        weight, m_t_ij, m_t1_ij, V_t_ij, V_t1_ij, cross_cov_ij, A_j, Q_j, log_det_Q
    ):
        """Log-prob for a single (t, i, j) triple."""
        E_xt1_xt1 = V_t1_ij + jnp.outer(m_t1_ij, m_t1_ij)
        E_xt_xt = V_t_ij + jnp.outer(m_t_ij, m_t_ij)
        # Stored lag covariance is Cov[x_t, x_{t+1}], so transpose it for
        # E[x_{t+1} x_t^T].
        E_xt1_xt = cross_cov_ij.T + jnp.outer(m_t1_ij, m_t_ij)
        expected_residual = (
            E_xt1_xt1 - A_j @ E_xt1_xt.T - E_xt1_xt @ A_j.T + A_j @ E_xt_xt @ A_j.T
        )
        trace_term = jnp.trace(psd_solve(Q_j, expected_residual))
        log_prob = -0.5 * (n_cont_states * jnp.log(2 * jnp.pi) + log_det_Q + trace_term)
        return jnp.where(weight > 0, weight * log_prob, 0.0)

    # vmap over i (source state): weight(i,), mean(n,i)->axis -1, cov(n,n,i)->axis -1
    _over_i = jax.vmap(
        _cont_trans_log_prob_single,
        in_axes=(0, -1, -1, -1, -1, -1, None, None, None),
    )
    # vmap over t (time): everything has t as axis 0
    _over_ti = jax.vmap(
        _over_i,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )

    # For each j: compute over all (t, i) and sum
    def _sum_for_j(A_j, Q_j, weights_j, m_t_j, m_t1_j, V_t_j, V_t1_j, cross_cov_j):
        """Sum log-probs over (t, i) for a single destination state j.

        weights_j: (T-1, K_i)
        m_t_j:     (T-1, n, K_i)
        V_t_j:     (T-1, n, n, K_i)
        cross_cov_j: (T-1, n, n, K_i)
        """
        log_det_Q = jnp.linalg.slogdet(Q_j)[1]
        return jnp.sum(
            _over_ti(
                weights_j,
                m_t_j,
                m_t1_j,
                V_t_j,
                V_t1_j,
                cross_cov_j,
                A_j,
                Q_j,
                log_det_Q,
            )
        )

    # vmap over j (destination state)
    log_cont_trans = jnp.sum(
        jax.vmap(
            _sum_for_j,
            in_axes=(2, 2, 2, 3, 3, 4, 4, 4),
        )(
            continuous_transition_matrix,  # (:, :, j)
            process_cov,  # (:, :, j)
            smoother_joint_discrete_state_prob,  # (:, i, j) -> axis 1 for i-within-j
            m_t,  # (:, :, i, j) -> axis 3
            m_t1,  # (:, :, i, j) -> axis 3
            V_t,  # (:, :, :, i, j) -> axis 4
            V_t1,  # (:, :, :, i, j) -> axis 4
            pair_cond_smoother_cross_cov,  # (:, :, :, i, j) -> axis 4
        )
    )

    # 5. Observations — vmap over j, vectorize over t
    def _obs_log_prob_for_j(H_j, R_j, weights_j, means_j, covs_j):
        """Sum observation log-probs over t for a single state j."""
        log_det_R = jnp.linalg.slogdet(R_j)[1]

        def _single_t(weight, m_t_j, V_t_j, y_t):
            pred_mean = H_j @ m_t_j
            diff = y_t - pred_mean
            expected_residual = jnp.outer(diff, diff) + H_j @ V_t_j @ H_j.T
            trace_term = jnp.trace(psd_solve(R_j, expected_residual))
            log_prob = -0.5 * (n_obs * jnp.log(2 * jnp.pi) + log_det_R + trace_term)
            return jnp.where(weight > 0, weight * log_prob, 0.0)

        return jnp.sum(jax.vmap(_single_t)(weights_j, means_j, covs_j, obs))

    log_obs = jnp.sum(
        jax.vmap(
            _obs_log_prob_for_j,
            in_axes=(2, 2, 1, 2, 3),
        )(
            measurement_matrix,  # (n_obs, n_cont, K) -> axis 2
            measurement_cov,  # (n_obs, n_obs, K) -> axis 2
            smoother_discrete_state_prob,  # (T, K) -> axis 1
            state_cond_smoother_means,  # (T, n_cont, K) -> axis 2
            state_cond_smoother_covs,  # (T, n_cont, n_cont, K) -> axis 3
        )
    )

    return (
        log_init_discrete
        + log_init_cont
        + log_discrete_trans
        + log_cont_trans
        + log_obs
    )


@jax.jit
def compute_posterior_entropy(
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    state_cond_smoother_covs: jax.Array,
) -> jax.Array:
    """Vectorized posterior entropy H(q).

    Equivalent to ``_compute_posterior_entropy_reference`` but uses vectorized
    JAX operations instead of Python loops.

    See ``_compute_posterior_entropy_reference`` for full parameter documentation.
    """
    n_cont_states = state_cond_smoother_covs.shape[1]

    # 1. Discrete entropy: t=0
    discrete_entropy = -jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(smoother_discrete_state_prob[0])
    )

    # t>0: vectorize over all time steps at once
    marginal_prev = smoother_discrete_state_prob[:-1]  # (T-1, K)
    joint = smoother_joint_discrete_state_prob  # (T-1, K, K)
    cond = _divide_safe(joint, marginal_prev[:, :, None])  # (T-1, K, K)
    discrete_entropy -= jnp.sum(joint * _safe_log(cond))

    # 2. Continuous entropy: vmap slogdet over (T, K)
    # state_cond_smoother_covs: (T, n, n, K) -> need (T, K, n, n) for vmap
    covs_tk = jnp.moveaxis(state_cond_smoother_covs, -1, 1)  # (T, K, n, n)
    T, K = covs_tk.shape[:2]
    covs_flat = covs_tk.reshape(T * K, n_cont_states, n_cont_states)
    log_dets = jax.vmap(lambda c: jnp.linalg.slogdet(c)[1])(covs_flat)
    log_dets = log_dets.reshape(T, K)  # (T, K)

    gaussian_entropies = 0.5 * (
        n_cont_states * (1 + jnp.log(2 * jnp.pi)) + log_dets
    )  # (T, K)
    weights = smoother_discrete_state_prob  # (T, K)
    cont_entropy = jnp.sum(jnp.where(weights > 0, weights * gaussian_entropies, 0.0))

    return discrete_entropy + cont_entropy


@jax.jit
def compute_markov_posterior_entropy(
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    state_cond_smoother_covs: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    pair_cond_smoother_covs: jax.Array | None = None,
) -> jax.Array:
    """Approximate trajectory entropy for the switching smoother posterior.

    The legacy ``compute_posterior_entropy`` sums marginal entropies
    ``H(x_t | S_t)`` and therefore over-counts entropy for a correlated Kalman
    trajectory. This function uses the Markov factorization

        H(x_{1:T}, s_{1:T}) = H(s) + E[H(x_T | s_T)]
            + sum_t E[H(x_t | x_{t+1}, s_t, s_{t+1})]

    with the available GPB pair lag covariances. For a single-state linear
    Gaussian model this reduces to the exact RTS trajectory entropy.
    """
    n_cont_states = state_cond_smoother_covs.shape[1]

    # Discrete Markov-chain entropy H(s_1:T).
    discrete_entropy = -jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(smoother_discrete_state_prob[0])
    )
    marginal_prev = smoother_discrete_state_prob[:-1]
    cond = _divide_safe(smoother_joint_discrete_state_prob, marginal_prev[:, :, None])
    discrete_entropy -= jnp.sum(smoother_joint_discrete_state_prob * _safe_log(cond))

    def _gaussian_entropy_from_cov(cov: jax.Array) -> jax.Array:
        cov = stabilize_covariance(cov, min_eigenvalue=1e-12)
        log_det = jnp.linalg.slogdet(cov)[1]
        return 0.5 * (n_cont_states * (1.0 + jnp.log(2.0 * jnp.pi)) + log_det)

    # Terminal entropy E[H(x_T | S_T)].
    terminal_covs = jnp.moveaxis(state_cond_smoother_covs[-1], -1, 0)
    terminal_entropies = jax.vmap(_gaussian_entropy_from_cov)(terminal_covs)
    terminal_entropy = jnp.sum(smoother_discrete_state_prob[-1] * terminal_entropies)

    if pair_cond_smoother_covs is not None:
        cov_t = pair_cond_smoother_covs
    else:
        cov_t = jnp.broadcast_to(
            state_cond_smoother_covs[:-1, :, :, :, None],
            pair_cond_smoother_cross_cov.shape,
        )

    cov_t1 = jnp.broadcast_to(
        state_cond_smoother_covs[1:, :, :, None, :],
        pair_cond_smoother_cross_cov.shape,
    )

    def _conditional_entropy(
        weight: jax.Array,
        cov_prev: jax.Array,
        cov_next: jax.Array,
        cross_prev_next: jax.Array,
    ) -> jax.Array:
        # cross_prev_next is Cov[x_t, x_{t+1}]. The Gaussian conditional
        # covariance is P_t - P_{t,t+1} P_{t+1}^{-1} P_{t+1,t}.
        gain = psd_solve(cov_next, cross_prev_next.T).T
        cond_cov = cov_prev - gain @ cross_prev_next.T
        entropy = _gaussian_entropy_from_cov(cond_cov)
        return jnp.where(weight > 0.0, weight * entropy, 0.0)

    weights = smoother_joint_discrete_state_prob.reshape(-1)
    flat_cov_t = jnp.transpose(cov_t, (0, 3, 4, 1, 2)).reshape(
        -1, n_cont_states, n_cont_states
    )
    flat_cov_t1 = jnp.transpose(cov_t1, (0, 3, 4, 1, 2)).reshape(
        -1, n_cont_states, n_cont_states
    )
    flat_cross = jnp.transpose(pair_cond_smoother_cross_cov, (0, 3, 4, 1, 2)).reshape(
        -1, n_cont_states, n_cont_states
    )
    conditional_entropy = jnp.sum(
        jax.vmap(_conditional_entropy)(weights, flat_cov_t, flat_cov_t1, flat_cross)
    )

    return discrete_entropy + terminal_entropy + conditional_entropy


@jax.jit
def compute_elbo(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
    discrete_transition_matrix: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
) -> jax.Array:
    """Compute the GPB approximate lower-bound diagnostic.

    ELBO = E_q[log p(y, x, s | θ)] - E_q[log q(x, s)]
         = E_q[log p(y, x, s | θ)] + H(q)

    The continuous entropy uses a trajectory-aware Markov Gaussian entropy
    rather than a sum of marginal entropies. For a single-state linear Gaussian
    model this equals the exact Kalman evidence. With GPB switching
    approximations it remains a diagnostic, not a strict monotonicity
    certificate.

    Parameters
    ----------
    obs : jax.Array, shape (n_time, n_obs_dim)
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. Optional GPB2 Q-function input.
    pair_cond_smoother_covs : jax.Array | None, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Cov[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. Optional GPB2 Q-function input.
    next_pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_{t+1} | y_{1:T}, S_t=i, S_{t+1}=j]. Optional GPB2 Q-function input.

    Returns
    -------
    elbo : jax.Array
        The evidence lower bound (scalar array)
    """
    expected_ll = compute_expected_complete_log_likelihood(
        obs=obs,
        state_cond_smoother_means=state_cond_smoother_means,
        state_cond_smoother_covs=state_cond_smoother_covs,
        smoother_discrete_state_prob=smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
        pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
        init_state_cond_mean=init_state_cond_mean,
        init_state_cond_cov=init_state_cond_cov,
        init_discrete_state_prob=init_discrete_state_prob,
        continuous_transition_matrix=continuous_transition_matrix,
        process_cov=process_cov,
        measurement_matrix=measurement_matrix,
        measurement_cov=measurement_cov,
        discrete_transition_matrix=discrete_transition_matrix,
        pair_cond_smoother_means=pair_cond_smoother_means,
        pair_cond_smoother_covs=pair_cond_smoother_covs,
        next_pair_cond_smoother_means=next_pair_cond_smoother_means,
    )

    entropy = compute_markov_posterior_entropy(
        smoother_discrete_state_prob=smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
        state_cond_smoother_covs=state_cond_smoother_covs,
        pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
        pair_cond_smoother_covs=pair_cond_smoother_covs,
    )

    return expected_ll + entropy


def compute_transition_sufficient_stats(
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute sufficient statistics for transition matrix estimation.

    Parameters
    ----------
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional means.
    pair_cond_smoother_covs : jax.Array | None, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Cov[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional covariances.
    next_pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_{t+1} | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional next means.

    Returns
    -------
    gamma1 : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        E[x_t x_t^T] weighted by joint probability P(S_t=i, S_{t+1}=j), summed over i.
    beta : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        E[x_{t+1} x_t^T] weighted by joint probability.
    """
    if pair_cond_smoother_means is not None:
        # gamma1[a,b,j] = sum_{t,i} w_t^{ij} * (Cov[x_t | i,j] + m_t^{ij} (m_t^{ij})^T)
        if pair_cond_smoother_covs is not None:
            gamma1 = jnp.einsum(
                "tij, tabij -> abj",
                smoother_joint_discrete_state_prob,
                pair_cond_smoother_covs,
            )
        else:
            gamma1 = jnp.einsum(
                "tij, tabi -> abj",
                smoother_joint_discrete_state_prob,
                state_cond_smoother_covs[:-1],
            )
        gamma1 += jnp.einsum(
            "tij, taij, tbij -> abj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_means,
            pair_cond_smoother_means,
        )

        # beta[c,d,j] = sum_{t,i} w_t^{ij} * (Cov[x_{t+1}, x_t | i,j] + m_{t+1}^{ij} (m_t^{ij})^T)
        beta = jnp.einsum(
            "tij,tdcij->cdj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov,
        )
        if next_pair_cond_smoother_means is not None:
            beta += jnp.einsum(
                "tdij,tcij,tij->cdj",
                pair_cond_smoother_means,
                next_pair_cond_smoother_means,
                smoother_joint_discrete_state_prob,
            )
        else:
            beta += jnp.einsum(
                "tdij,tcj,tij->cdj",
                pair_cond_smoother_means,
                state_cond_smoother_means[1:],
                smoother_joint_discrete_state_prob,
            )
    else:
        # Approximate factored form (original implementation)
        gamma1 = jnp.einsum(
            "tij, tabi -> abj",
            smoother_joint_discrete_state_prob,
            state_cond_smoother_covs[:-1],
        ) + jnp.einsum(
            "tij, tai, tbi -> abj",
            smoother_joint_discrete_state_prob,
            state_cond_smoother_means[:-1],
            state_cond_smoother_means[:-1],
        )

        beta = jnp.einsum(
            "tij,tdcij->cdj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov,
        )
        beta += jnp.einsum(
            "tdi,tcj,tij->cdj",
            state_cond_smoother_means[:-1],
            state_cond_smoother_means[1:],
            smoother_joint_discrete_state_prob,
        )

    return gamma1, beta


def compute_process_covariance_sufficient_stats(
    continuous_transition_matrix: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute fixed-transition residual scatter for a constrained Q M-step.

    The generic switching M-step estimates ``A`` and then uses the simplifying
    identity valid at that unconstrained optimum.  Models such as CNM keep ``A``
    fixed, so their process-noise update instead needs the complete residual
    second moment

    ``S = Gamma2 - A Beta.T - Beta A.T + A Gamma1 A.T``.

    Returns the unnormalized per-state residual scatter and the corresponding
    destination-state counts.  Optional GPB2 pair-conditioned statistics are
    forwarded to the shared transition-statistics implementation.
    """
    gamma1, beta = compute_transition_sufficient_stats(
        state_cond_smoother_means=state_cond_smoother_means,
        state_cond_smoother_covs=state_cond_smoother_covs,
        smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
        pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
        pair_cond_smoother_means=pair_cond_smoother_means,
        pair_cond_smoother_covs=pair_cond_smoother_covs,
        next_pair_cond_smoother_means=next_pair_cond_smoother_means,
    )

    weights = smoother_discrete_state_prob[1:]
    gamma2 = jnp.sum(
        state_cond_smoother_covs[1:] * weights[:, None, None, :], axis=0
    ) + jnp.einsum(
        "tai,tbi,ti->abi",
        state_cond_smoother_means[1:],
        state_cond_smoother_means[1:],
        weights,
    )
    state_counts = jnp.sum(weights, axis=0)

    def residual_scatter(
        A: jax.Array, gamma1_j: jax.Array, beta_j: jax.Array, gamma2_j: jax.Array
    ) -> jax.Array:
        residual = gamma2_j - A @ beta_j.T - beta_j @ A.T + A @ gamma1_j @ A.T
        return 0.5 * (residual + residual.T)

    scatter = jax.vmap(residual_scatter, in_axes=(-1, -1, -1, -1), out_axes=-1)(
        continuous_transition_matrix, gamma1, beta, gamma2
    )
    return scatter, state_counts


def compute_transition_q_function(
    A: jax.Array,
    gamma1: jax.Array,
    beta: jax.Array,
    process_cov: jax.Array | None = None,
) -> jax.Array:
    """Compute the Q-function contribution from transition matrix.

    The Q-function for A (ignoring terms independent of A) is:
        Q(A) ∝ -0.5 * tr(A^T A gamma1 - 2 A^T beta)
    when the process covariance is the identity. For non-identity process
    covariance this function uses the corresponding Mahalanobis weighting.

    We return the negative Q-function since we want to minimize.

    Parameters
    ----------
    A : jax.Array, shape (n_cont, n_cont)
        Transition matrix.
    gamma1 : jax.Array, shape (n_cont, n_cont)
        E[x_t x_t^T] summed over time, weighted by joint discrete state probs.
    beta : jax.Array, shape (n_cont, n_cont)
        E[x_{t+1} x_t^T] summed over time, weighted by joint discrete state probs.
    process_cov : jax.Array | None, shape (n_cont, n_cont), optional
        Process covariance Q for Mahalanobis weighting. If None, uses identity
        weighting for backwards compatibility.

    Returns
    -------
    jax.Array
        Negative Q-function value (to be minimized, scalar array).
    """
    if process_cov is None:
        return 0.5 * jnp.trace(A.T @ A @ gamma1) - jnp.trace(A.T @ beta)

    q_inv_A = psd_solve(process_cov, A)
    q_inv_beta = psd_solve(process_cov, beta)
    return 0.5 * jnp.trace(q_inv_A @ gamma1 @ A.T) - jnp.trace(A.T @ q_inv_beta)


def compute_transition_q_from_params(
    damping: jax.Array,
    freq: jax.Array,
    coupling_strength: jax.Array,
    phase_diff: jax.Array,
    sampling_freq: float,
    gamma1: jax.Array,
    beta: jax.Array,
    process_cov: jax.Array | None = None,
) -> jax.Array:
    """Compute Q-function from oscillator parameters.

    This is the objective to minimize in the reparameterized M-step.

    Parameters
    ----------
    damping : jax.Array, shape (n_oscillators,)
        Damping coefficients.
    freq : jax.Array, shape (n_oscillators,)
        Frequencies in Hz.
    coupling_strength : jax.Array, shape (n_oscillators, n_oscillators)
        Coupling strengths (0 on diagonal).
    phase_diff : jax.Array, shape (n_oscillators, n_oscillators)
        Phase differences (0 on diagonal).
    sampling_freq : float
        Sampling frequency.
    gamma1 : jax.Array, shape (n_cont, n_cont)
        Sufficient statistic.
    beta : jax.Array, shape (n_cont, n_cont)
        Sufficient statistic.
    process_cov : jax.Array | None, shape (n_cont, n_cont), optional
        Process covariance for Mahalanobis weighting.

    Returns
    -------
    jax.Array
        Negative Q-function value (to be minimized, scalar array).
    """
    from state_space_practice.oscillator_utils import (
        construct_directed_influence_transition_matrix,
    )

    A = construct_directed_influence_transition_matrix(
        freqs=freq,
        damping_coeffs=damping,
        coupling_strengths=coupling_strength,
        phase_diffs=phase_diff,
        sampling_freq=sampling_freq,
    )
    return compute_transition_q_function(A, gamma1, beta, process_cov=process_cov)


def optimize_dim_transition_params(
    gamma1: jax.Array,
    beta: jax.Array,
    init_params: dict,
    sampling_freq: float,
    process_cov: jax.Array | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    raise_on_failure: bool = False,
) -> dict:
    """Optimize oscillator parameters to maximize Q-function.

    Uses JAX autodiff + BFGS optimizer.

    Parameters
    ----------
    gamma1 : jax.Array, shape (n_cont, n_cont)
        Sufficient statistic E[x_t x_t^T].
    beta : jax.Array, shape (n_cont, n_cont)
        Sufficient statistic E[x_{t+1} x_t^T].
    init_params : dict
        Initial parameter values (damping, freq, coupling_strength, phase_diff).
    sampling_freq : float
        Sampling frequency.
    process_cov : jax.Array | None, optional
        Process covariance for Mahalanobis weighting of the transition
        residual. If None, uses identity weighting.
    max_iter : int
        Maximum optimization iterations.
    tol : float
        Convergence tolerance.
    raise_on_failure : bool
        Controls handling of an unusable optimizer result (non-finite solution
        or objective). If True, raise RuntimeError; if False, emit a
        RuntimeWarning and return the parameters as reported. Note that JAX
        BFGS's ``success=False`` flag alone is not treated as a failure, since
        it fires on benign line-search terminations that still yield a good
        solution.

    Returns
    -------
    dict
        Optimized parameters with keys: damping, freq, coupling_strength, phase_diff.
    """
    from jax.scipy.optimize import minimize

    from state_space_practice.oscillator_utils import (
        construct_directed_influence_transition_matrix,
    )

    n_osc = len(init_params["damping"])

    if sampling_freq <= 0.0:
        raise ValueError("sampling_freq must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    # Optimize in transformed coordinates for stability:
    # - damping: sigmoid maps (-inf, inf) -> (0, max_damping)
    # - frequency: tanh maps (-inf, inf) -> (-Nyquist, Nyquist)
    # - coupling_strength: sigmoid maps (-inf, inf) -> (0, max_coupling)
    # - phase_diff: unconstrained, with diagonal entries excluded
    max_damping = 0.995
    max_freq = 0.5 * float(sampling_freq)
    max_coupling = 0.5
    offdiag_i, offdiag_j = jnp.where(~jnp.eye(n_osc, dtype=bool))

    def _sigmoid(x: jax.Array) -> jax.Array:
        return max_damping * jax.nn.sigmoid(x)

    def _inv_sigmoid(y: jax.Array) -> jax.Array:
        y_clipped = jnp.clip(y / max_damping, 1e-6, 1.0 - 1e-6)
        return jnp.log(y_clipped / (1.0 - y_clipped))

    def _bounded_freq(x: jax.Array) -> jax.Array:
        return max_freq * jnp.tanh(x)

    def _inv_bounded_freq(y: jax.Array) -> jax.Array:
        y_clipped = jnp.clip(y / max_freq, -1.0 + 1e-6, 1.0 - 1e-6)
        return jnp.arctanh(y_clipped)

    def _bounded_coupling(x: jax.Array) -> jax.Array:
        return max_coupling * jax.nn.sigmoid(x)

    def _inv_bounded_coupling(y: jax.Array) -> jax.Array:
        y_clipped = jnp.clip(y / max_coupling, 1e-6, 1.0 - 1e-6)
        return jnp.log(y_clipped / (1.0 - y_clipped))

    def pack_unconstrained(params: dict) -> jax.Array:
        """Map physical params to unconstrained coordinates."""
        damping = jnp.asarray(params["damping"])
        freq = jnp.asarray(params["freq"])
        coupling = jnp.asarray(params["coupling_strength"])
        phase = jnp.asarray(params["phase_diff"])
        offdiag_coupling = coupling[offdiag_i, offdiag_j]
        offdiag_phase = phase[offdiag_i, offdiag_j]
        offdiag_phase = jnp.where(
            offdiag_coupling < 0.0,
            offdiag_phase + jnp.pi,
            offdiag_phase,
        )
        offdiag_coupling = jnp.abs(offdiag_coupling)
        return jnp.concatenate(
            [
                _inv_sigmoid(damping),
                _inv_bounded_freq(freq),
                _inv_bounded_coupling(offdiag_coupling),
                offdiag_phase,
            ]
        )

    def unpack_constrained(flat: jax.Array) -> dict:
        """Map unconstrained coordinates to physical params."""
        idx = 0
        damping = _sigmoid(flat[idx : idx + n_osc])
        idx += n_osc
        freq = _bounded_freq(flat[idx : idx + n_osc])
        idx += n_osc
        n_offdiag = n_osc * (n_osc - 1)
        coupling = jnp.zeros((n_osc, n_osc), dtype=flat.dtype)
        coupling = coupling.at[offdiag_i, offdiag_j].set(
            _bounded_coupling(flat[idx : idx + n_offdiag])
        )
        idx += n_offdiag
        phase = jnp.zeros((n_osc, n_osc), dtype=flat.dtype)
        phase = phase.at[offdiag_i, offdiag_j].set(flat[idx : idx + n_offdiag])
        return {
            "damping": damping,
            "freq": freq,
            "coupling_strength": coupling,
            "phase_diff": phase,
        }

    def loss(flat_params: jax.Array) -> jax.Array:
        params = unpack_constrained(flat_params)
        return compute_transition_q_from_params(
            damping=params["damping"],
            freq=params["freq"],
            coupling_strength=params["coupling_strength"],
            phase_diff=params["phase_diff"],
            sampling_freq=sampling_freq,
            gamma1=gamma1,
            beta=beta,
            process_cov=process_cov,
        )

    # Run optimizer in unconstrained space
    init_flat = pack_unconstrained(init_params)
    result = minimize(
        loss,
        init_flat,
        method="BFGS",
        tol=tol,
        options={"maxiter": max_iter},
    )
    # JAX's BFGS sets ``success=False`` for benign line-search terminations even
    # when the returned iterate is a good solution (a very common outcome on this
    # reparameterized objective), so that flag alone is not an actionable failure
    # signal. Treat the optimization as failed only when it returns a non-finite
    # solution or objective, which is unambiguously unusable downstream.
    solution_finite = bool(jax.device_get(jnp.all(jnp.isfinite(result.x))))
    objective_finite = bool(jax.device_get(jnp.isfinite(result.fun)))
    if not (solution_finite and objective_finite):
        status = int(jax.device_get(result.status))
        nit = int(jax.device_get(result.nit))
        fun = float(jax.device_get(result.fun))
        message = (
            "DIM transition parameter optimization produced a non-finite "
            f"solution (status={status}, nit={nit}, objective={fun:.6g})."
        )
        if raise_on_failure:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    opt_params = unpack_constrained(result.x)

    # Post-check: verify spectral radius of resulting A matrix.
    # If unstable, uniformly scale damping AND coupling so the
    # spectral radius of the reconstructed A is <= 0.99.
    A_opt = construct_directed_influence_transition_matrix(
        freqs=opt_params["freq"],
        damping_coeffs=opt_params["damping"],
        coupling_strengths=opt_params["coupling_strength"],
        phase_diffs=opt_params["phase_diff"],
        sampling_freq=sampling_freq,
    )
    # Spectral radius is computed on host (eigvals has no GPU/TPU lowering);
    # the optimizer has already returned, so this runs eagerly and stays
    # backend-portable. Here we scale the params (not A directly), so we need
    # the scalar radius rather than utils.stabilize_transition_matrix.
    radius = _spectral_radius(A_opt)
    safe_scale = 0.99 / radius if radius > 0.99 else 1.0
    opt_params["damping"] = opt_params["damping"] * safe_scale
    opt_params["coupling_strength"] = opt_params["coupling_strength"] * safe_scale

    return opt_params


def optimize_dim_transition_params_joint(
    gamma1: jax.Array,
    beta: jax.Array,
    init_params: dict,
    sampling_freq: float,
    process_cov: jax.Array | None = None,
    max_spectral_radius: float = 0.99,
    max_damping: float = 0.995,
    max_iter: int = 100,
    tol: float = 1e-6,
    max_backtracking_steps: int = 20,
    raise_on_failure: bool = False,
) -> dict:
    """Jointly optimize shared and state-specific DIM transition parameters.

    Unlike :func:`optimize_dim_transition_params`, this solves one objective
    across all discrete states. Frequency and damping are shared variables;
    coupling and phase remain state-specific. Every objective evaluation uses
    the same differentiable global stability scale as ``DirectedInfluenceModel``
    so the optimized objective matches the transition matrices installed by the
    model.

    Parameters
    ----------
    gamma1, beta : jax.Array, shape (n_cont, n_cont, n_discrete_states)
        GPB transition sufficient statistics.
    init_params : dict
        ``damping`` and ``freq`` have shape ``(n_oscillators,)``;
        ``coupling_strength`` and ``phase_diff`` have shape
        ``(n_oscillators, n_oscillators, n_discrete_states)``.
    sampling_freq : float
        Sampling frequency in Hz.
    process_cov : jax.Array | None
        Per-state process covariance stack. Identity weighting is used when
        omitted.
    max_spectral_radius : float
        Target upper bound on the spectral radius of each state's transition
        matrix; passed to the differentiable stability scale. Must lie in
        ``(0, 1)``.
    max_damping : float
        Upper bound on the intrinsic per-oscillator damping in the bounded
        reparameterization. Must lie in ``(0, 1)``.
    max_iter, tol : int, float
        BFGS controls.
    max_backtracking_steps : int
        Maximum safeguard steps between the initial and proposed unconstrained
        parameters if BFGS returns a worse objective.
    raise_on_failure : bool
        Raise instead of warning and returning the initial constrained point
        when BFGS produces a non-finite result.

    Returns
    -------
    dict
        Shared ``damping``/``freq`` and state-specific
        ``coupling_strength``/``phase_diff``.
    """
    from jax.scipy.optimize import minimize

    from state_space_practice.oscillator_utils import (
        compute_directed_influence_stability_scale,
    )

    gamma1 = jnp.asarray(gamma1)
    beta = jnp.asarray(beta)
    damping0 = jnp.asarray(init_params["damping"])
    freq0 = jnp.asarray(init_params["freq"])
    coupling0 = jnp.asarray(init_params["coupling_strength"])
    phase0 = jnp.asarray(init_params["phase_diff"])

    if sampling_freq <= 0.0:
        raise ValueError("sampling_freq must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if max_backtracking_steps < 0:
        raise ValueError("max_backtracking_steps must be non-negative.")
    if not 0.0 < max_spectral_radius < 1.0:
        raise ValueError("max_spectral_radius must lie in (0, 1).")
    if not 0.0 < max_damping < 1.0:
        raise ValueError("max_damping must lie in (0, 1).")
    if gamma1.ndim != 3 or beta.shape != gamma1.shape:
        raise ValueError(
            "gamma1 and beta must have matching shape "
            "(n_cont, n_cont, n_discrete_states)."
        )

    n_osc = damping0.shape[0]
    n_states = gamma1.shape[-1]
    n_cont = 2 * n_osc
    expected_stats_shape = (n_cont, n_cont, n_states)
    if gamma1.shape != expected_stats_shape:
        raise ValueError(
            f"gamma1 shape must be {expected_stats_shape}, got {gamma1.shape}."
        )
    if freq0.shape != (n_osc,):
        raise ValueError(f"freq shape must be ({n_osc},), got {freq0.shape}.")
    expected_network_shape = (n_osc, n_osc, n_states)
    if coupling0.shape != expected_network_shape:
        raise ValueError(
            "coupling_strength shape must be "
            f"{expected_network_shape}, got {coupling0.shape}."
        )
    if phase0.shape != expected_network_shape:
        raise ValueError(
            f"phase_diff shape must be {expected_network_shape}, got {phase0.shape}."
        )

    process_cov_arr = None if process_cov is None else jnp.asarray(process_cov)
    if process_cov_arr is not None and process_cov_arr.shape != expected_stats_shape:
        raise ValueError(
            f"process_cov shape must be {expected_stats_shape}, "
            f"got {process_cov_arr.shape}."
        )

    arrays_to_check = [gamma1, beta, damping0, freq0, coupling0, phase0]
    if process_cov_arr is not None:
        arrays_to_check.append(process_cov_arr)
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in arrays_to_check):
        raise ValueError("Joint DIM optimizer inputs must contain only finite values.")

    max_freq = 0.5 * float(sampling_freq)
    max_coupling = 0.5
    offdiag_i, offdiag_j = jnp.where(~jnp.eye(n_osc, dtype=bool))
    n_offdiag = n_osc * (n_osc - 1)

    def _bounded_damping(x: jax.Array) -> jax.Array:
        return max_damping * jax.nn.sigmoid(x)

    def _inv_bounded_damping(y: jax.Array) -> jax.Array:
        ratio = jnp.clip(y / max_damping, 1e-6, 1.0 - 1e-6)
        return jnp.log(ratio) - jnp.log1p(-ratio)

    def _bounded_freq(x: jax.Array) -> jax.Array:
        return max_freq * jnp.tanh(x)

    def _inv_bounded_freq(y: jax.Array) -> jax.Array:
        ratio = jnp.clip(y / max_freq, -1.0 + 1e-6, 1.0 - 1e-6)
        return jnp.arctanh(ratio)

    def _bounded_coupling(x: jax.Array) -> jax.Array:
        return max_coupling * jax.nn.sigmoid(x)

    def _inv_bounded_coupling(y: jax.Array) -> jax.Array:
        ratio = jnp.clip(y / max_coupling, 1e-6, 1.0 - 1e-6)
        return jnp.log(ratio) - jnp.log1p(-ratio)

    def pack_unconstrained(params: dict) -> jax.Array:
        damping = jnp.asarray(params["damping"])
        freq = jnp.asarray(params["freq"])
        coupling = jnp.asarray(params["coupling_strength"])
        phase = jnp.asarray(params["phase_diff"])
        offdiag_coupling = coupling[offdiag_i, offdiag_j, :]
        offdiag_phase = phase[offdiag_i, offdiag_j, :]
        offdiag_phase = jnp.where(
            offdiag_coupling < 0.0,
            offdiag_phase + jnp.pi,
            offdiag_phase,
        )
        offdiag_coupling = jnp.abs(offdiag_coupling)
        return jnp.concatenate(
            [
                _inv_bounded_damping(damping),
                _inv_bounded_freq(freq),
                _inv_bounded_coupling(offdiag_coupling).reshape(-1),
                offdiag_phase.reshape(-1),
            ]
        )

    def unpack_constrained(flat: jax.Array) -> dict:
        idx = 0
        damping = _bounded_damping(flat[idx : idx + n_osc])
        idx += n_osc
        freq = _bounded_freq(flat[idx : idx + n_osc])
        idx += n_osc
        network_size = n_offdiag * n_states
        coupling_values = _bounded_coupling(flat[idx : idx + network_size]).reshape(
            n_offdiag, n_states
        )
        idx += network_size
        phase_values = flat[idx : idx + network_size].reshape(n_offdiag, n_states)

        coupling = jnp.zeros(expected_network_shape, dtype=flat.dtype)
        coupling = coupling.at[offdiag_i, offdiag_j, :].set(coupling_values)
        phase = jnp.zeros(expected_network_shape, dtype=flat.dtype)
        phase = phase.at[offdiag_i, offdiag_j, :].set(phase_values)
        return {
            "damping": damping,
            "freq": freq,
            "coupling_strength": coupling,
            "phase_diff": phase,
        }

    def loss(flat_params: jax.Array) -> jax.Array:
        params = unpack_constrained(flat_params)
        scale = compute_directed_influence_stability_scale(
            params["freq"],
            params["damping"],
            params["coupling_strength"],
            sampling_freq,
            max_spectral_radius=max_spectral_radius,
        )
        effective_damping = params["damping"] * scale
        effective_coupling = params["coupling_strength"] * scale

        if process_cov_arr is None:
            per_state = jax.vmap(
                lambda coupling, phase, gamma, cross: compute_transition_q_from_params(
                    damping=effective_damping,
                    freq=params["freq"],
                    coupling_strength=coupling,
                    phase_diff=phase,
                    sampling_freq=sampling_freq,
                    gamma1=gamma,
                    beta=cross,
                ),
                in_axes=(-1, -1, -1, -1),
            )(
                effective_coupling,
                params["phase_diff"],
                gamma1,
                beta,
            )
        else:
            per_state = jax.vmap(
                lambda coupling, phase, gamma, cross, cov: (
                    compute_transition_q_from_params(
                        damping=effective_damping,
                        freq=params["freq"],
                        coupling_strength=coupling,
                        phase_diff=phase,
                        sampling_freq=sampling_freq,
                        gamma1=gamma,
                        beta=cross,
                        process_cov=cov,
                    )
                ),
                in_axes=(-1, -1, -1, -1, -1),
            )(
                effective_coupling,
                params["phase_diff"],
                gamma1,
                beta,
                process_cov_arr,
            )
        return jnp.sum(per_state)

    init_params_stacked = {
        "damping": damping0,
        "freq": freq0,
        "coupling_strength": coupling0,
        "phase_diff": phase0,
    }
    init_flat = pack_unconstrained(init_params_stacked)
    init_loss = float(jax.device_get(loss(init_flat)))
    if not math.isfinite(init_loss):
        raise ValueError("Initial joint DIM parameters produce a non-finite objective.")
    result = minimize(
        loss,
        init_flat,
        method="BFGS",
        tol=tol,
        options={"maxiter": max_iter},
    )

    solution_finite = bool(jax.device_get(jnp.all(jnp.isfinite(result.x))))
    candidate_loss = (
        float(jax.device_get(loss(result.x))) if solution_finite else float("nan")
    )
    objective_finite = bool(jax.device_get(jnp.isfinite(result.fun))) and math.isfinite(
        candidate_loss
    )
    if not (solution_finite and objective_finite):
        status = int(jax.device_get(result.status))
        nit = int(jax.device_get(result.nit))
        fun = float(jax.device_get(result.fun))
        message = (
            "Joint DIM transition optimization produced a non-finite solution "
            f"(status={status}, nit={nit}, objective={fun:.6g})."
        )
        if raise_on_failure:
            raise RuntimeError(message)
        warnings.warn(
            message + " Keeping the initial parameters.",
            RuntimeWarning,
            stacklevel=2,
        )
        return unpack_constrained(init_flat)

    accepted_flat = result.x
    accepted_loss = candidate_loss
    objective_slack = tol * max(1.0, abs(init_loss))
    if accepted_loss > init_loss + objective_slack:
        accepted_flat = init_flat
        direction = result.x - init_flat
        for step in range(1, max_backtracking_steps + 1):
            trial_flat = init_flat + (0.5**step) * direction
            trial_loss = float(jax.device_get(loss(trial_flat)))
            if math.isfinite(trial_loss) and trial_loss <= init_loss + objective_slack:
                accepted_flat = trial_flat
                break

    return unpack_constrained(accepted_flat)

# Designs — derivations and code

[← back to PLAN.md](PLAN.md) · [contracts](shared-contracts.md) · [overview](overview.md)

Sections

1. [Problem and notation](#problem-and-notation)
2. [Iteration semantics](#iteration-semantics)
3. [Pseudo-observation sites in information form](#pseudo-observation-sites)
4. [The Gauss–Newton step is one linear smoother pass](#gauss-newton-step-as-a-linear-smoother-pass)
5. [Line search on the joint log posterior](#line-search)
6. [Laplace evidence at the mode](#laplace-evidence)
7. [Convergence and diagnostics](#convergence-and-diagnostics)
8. [Block-diagonal path](#block-diagonal-path)
9. [Parallel-in-time pass](#parallel-in-time-pass)
10. [Position-decoder sites](#position-decoder-sites)
11. [Gradients](#gradients)
12. [Numerical notes and memory](#numerical-notes)
13. [Alternatives considered](#alternatives-considered)
14. [References](#references)

All code below is JAX (`jnp`, `lax.scan`, `vmap`), float64, time axis leading. Helper names
match [shared-contracts.md](shared-contracts.md).

## Problem and notation

Model (as the filters in `point_process_kalman.py` implement it, see the scan at
`point_process_kalman.py:1855-1864` and `kalman.InitialStatePrior`,
`kalman.py:1251-1278`):

```
x_0 ~ N(m_0, P_0),   x_t = A x_{t-1} + w_t,  w_t ~ N(0, Q),      t = 1..T
y_{t,n} ~ Poisson(μ_{t,n}),  μ_t = family.mean(η_t),  η_t = h(Z_t, x_t)   (n = 1..N)
```

`h` is `log_conditional_intensity`; by default `h(Z_t, x) = Z_t x` (`:523-547`). The one-pass
filter observes `x_1 = A x_0 + w_1` first, so the smoother output starts at `x_1` and `x_0`
is marginalised: `x_1 ~ N(A m_0, P_{1|0})`, `P_{1|0} = A P_0 Aᵀ + Q`. The joint log
posterior over `X = (x_1, …, x_T)` (dropping constants) is

```
Ψ(X) = Σ_t ℓ_t(x_t) − ½ (x_1 − A m_0)ᵀ P_{1|0}⁻¹ (x_1 − A m_0) − ½ Σ_{t≥2} (x_t − A x_{t−1})ᵀ Q⁻¹ (x_t − A x_{t−1})
ℓ_t(x) = Σ_n log p(y_{t,n} | η_{t,n}(x))          # family.loglik_plugin
```

For the canonical Poisson log link, `∂ℓ_t/∂η = y_t − μ_t` and `−∂²ℓ_t/∂η² = diag(μ_t)`
(observed = expected Hessian). With linear `η`, `Ψ` is strictly concave in `X` (a sum of a
concave likelihood and a strictly concave Gaussian prior), so the MAP path is unique and
Newton = Gauss–Newton (Bell 1994; Fahrmeir 1992). With a nonlinear `η` (decoder), the
Gauss–Newton Hessian drops `Σ_n (y − μ)_n ∂²η_n/∂x²` and the iteration is the iterated
extended Kalman smoother / posterior-linearisation smoother with analytical linearisation
(García-Fernández et al. 2017); it remains an ascent method under the line search because
`G_t` is PSD.

## Iteration semantics

Fixed in [shared-contracts.md#public-keywords](shared-contracts.md#public-keywords). Passes:

| pass | `parallel=False` | `parallel=True` |
| --- | --- | --- |
| 1 | today's Laplace-EKF filter (`max_newton_iter` Fisher steps per bin) + RTS: `X^{(1)}` | damped Gauss–Newton pass from the nominal path `x̂_t = init_mean` through associative scans: `X^{(1)}` |
| 2..k | damped Gauss–Newton pass linearised at `X^{(j−1)}` | same, parallel |
| final | linear pass at `X^{(k)}`: covariances, cross-covariances, evidence, un-taken step | same, parallel |

Why the parallel variant does not keep the Laplace-EKF initialisation: that pass is a
nonlinear sequential recursion with O(T) span and would dominate the wall clock at
`T ~ 1e6` (measured one-pass cost on this CPU: 0.165 s at `T = 1e4, d = 2, N = 50`, i.e.
~17 s at `T = 1e6`), defeating the purpose; Yaghoobi et al. (2021) likewise start the
parallel IEKS from a nominal trajectory. Both variants converge to the same unique MAP path
for the Poisson–linear model, which is what the phase-2 comparison test asserts.

## Pseudo-observation sites

Given the linearisation point `x̂_t` (contract:
[shared-contracts.md#pseudo-observation-site-contract](shared-contracts.md#pseudo-observation-site-contract)):

```python
def _pseudo_observation_sites(
    eta_hat: Array, counts_t: Array, family: GLMFamily
) -> tuple[Array, Array, Array]:
    """Fisher weight ``w``, likelihood score ``r`` and mean ``mu`` at ``eta_hat``.

    Parameters
    ----------
    eta_hat : Array, shape (n_eta,)
    counts_t : Array, shape (n_obs,)
    family : GLMFamily

    Returns
    -------
    weight, residual, mean : Array, shape (n_eta,)
    """
    mu = family.mean(eta_hat)
    weight = family.fisher_weight(eta_hat, mu)
    score = counts_t - mu if family.score is None else family.score(counts_t, eta_hat, mu)
    return weight, score, mu


def _information_site(jacobian: Array, weight: Array, residual: Array, x_hat: Array):
    """Information matrix ``G = J' diag(w) J`` and vector ``g = G x_hat + J' r``."""
    info_matrix = jacobian.T @ (weight[:, None] * jacobian)
    info_vector = info_matrix @ x_hat + jacobian.T @ residual
    return symmetrize(info_matrix), info_vector


def _site_log_value(state, x_hat, info_matrix, info_vector, log_value_at_hat):
    """Evaluate a complete quadratic site, including its Taylor constant."""
    delta = state - x_hat
    score = info_vector - info_matrix @ x_hat
    return log_value_at_hat + score @ delta - 0.5 * delta @ info_matrix @ delta
```

Derivation of `g_t`: the IRLS working response is `ỹ_t = η̂_t + W_t⁻¹ r_t` observed through
`ỹ_t = c_t + J_t x_t + ε_t`, `ε_t ~ N(0, W_t⁻¹)`, `c_t = η̂_t − J_t x̂_t`. Its Gaussian
information about `x_t` is `G_t = J_tᵀ W_t J_t` and
`g_t = J_tᵀ W_t (ỹ_t − c_t) = J_tᵀ (W_t J_t x̂_t + r_t) = G_t x̂_t + J_tᵀ r_t`. Both stay
finite as `w → 0` (a zero-count bin with vanishing rate) where `W_t⁻¹` does not — this is
why the information form is used everywhere rather than `kalman_filter` with
`R_t = W_t⁻¹` (compare `temporal_rate_gp.poisson_log_rate_site`,
`temporal_rate_gp.py:190-233`, which needs the `min_weight` floor for that reason).

`family.mean` for the Poisson family is `_safe_expected_count` (`point_process_kalman.py:589-610`,
log-count clipped to `[−20, max_log_count]`), so the sites see exactly the clipping the
one-pass filter uses.

## Gauss–Newton step as a linear smoother pass

**Claim.** With sites `(G_t, g_t)` formed at the path `X̂`, the RTS-smoothed mean of the
linear-Gaussian model `x_t = A x_{t−1} + w_t`, pseudo-observation `(G_t, g_t)` equals the
Gauss–Newton target `X̂ + H⁻¹ ∇Ψ(X̂)`, where `H = Σ⁻¹ + blockdiag(G_t)` is the (negative)
Gauss–Newton Hessian and `Σ⁻¹` the block-tridiagonal prior precision.

*Proof.* The pseudo-model posterior over `X` is Gaussian with precision `Σ⁻¹ + G` and mean
`X*` solving `(Σ⁻¹ + G) X* = Σ⁻¹ μ + g`, where `μ` is the prior mean path
(`μ_1 = A m_0`, `μ_t = A μ_{t−1}`) and `g = (g_t)_t`. Subtract `(Σ⁻¹ + G) X̂`:
`(Σ⁻¹ + G)(X* − X̂) = Σ⁻¹ (μ − X̂) + g − G X̂ = Σ⁻¹ (μ − X̂) + Jᵀ r = ∇Ψ(X̂)`, since
`∇_X [−½ (X − μ)ᵀ Σ⁻¹ (X − μ)] = Σ⁻¹ (μ − X)` and `∇_X Σ_t ℓ_t = (J_tᵀ r_t)_t`. The RTS
smoother computes exactly this posterior mean (Paninski et al. 2010 make the same
observation: the Kalman smoother is one Newton step on the block-tridiagonal objective in
O(T)). ∎

**Per-bin measurement update, information form** (mirrors
`_point_process_laplace_update`'s single-step branch, `point_process_kalman.py:1145-1162`,
with the linearisation point decoupled from the predicted mean):

```python
def _linearised_measurement_update(
    one_step_mean: Array,   # (d,)   A m_{t-1|t-1}
    one_step_cov: Array,    # (d, d) A P_{t-1|t-1} A' + Q (symmetrised)
    info_matrix: Array,     # (d, d) G_t
    info_vector: Array,     # (d,)   g_t
    diagonal_boost: float = 0.0,
) -> tuple[Array, Array, Array]:
    """Kalman measurement update of one information-form pseudo-observation.

    Returns ``(post_mean, post_cov, gaussian_terms)`` where ``gaussian_terms``
    is ``-0.5 d' P_pred^{-1} d - 0.5 log|P_pred| + 0.5 log|P_post|`` with
    ``d = post_mean - one_step_mean`` -- the prior and normaliser part of the
    per-bin Laplace evidence, from the same two Cholesky factors the update
    used (one jitter policy, as in ``_point_process_laplace_update``).
    """
    n_latent = one_step_mean.shape[0]
    identity = jnp.eye(n_latent, dtype=one_step_cov.dtype)
    prior_cho = psd_cholesky(one_step_cov, diagonal_boost=diagonal_boost)
    prior_precision = jax.scipy.linalg.cho_solve(prior_cho, identity)
    post_precision = symmetrize(prior_precision + info_matrix)
    post_cho = psd_cholesky(post_precision, diagonal_boost=diagonal_boost)
    # post_mean = P_post (P_pred^{-1} m_pred + g) written in delta form:
    delta = jax.scipy.linalg.cho_solve(post_cho, info_vector - info_matrix @ one_step_mean)
    post_mean = one_step_mean + delta
    post_cov = symmetrize(jax.scipy.linalg.cho_solve(post_cho, identity))
    gaussian_terms = (
        -0.5 * delta @ (prior_precision @ delta)
        - 0.5 * psd_logdet(prior_cho)
        - 0.5 * psd_logdet(post_cho)  # log|P_post| = -log|post_precision|
    )
    return post_mean, post_cov, gaussian_terms
```

Check against the existing Fisher step: with `x̂ = m_pred` the delta is
`P_post Jᵀ r`, the `max_newton_iter=1` update at `:1161`. With `x̂ ≠ m_pred`,
`g − G m_pred = Jᵀ r + G (x̂ − m_pred)`: the pseudo-observation update equals one Fisher
step *from `x̂`* under the prior `N(m_pred, P_pred)`.

**Sequential forward pass** (dense; the block path is the same body per neuron):

```python
def _linearised_forward_scan(problem: _LinearisationProblem, path: Array):
    A, Q = problem.transition_matrix, problem.process_cov
    family = problem.family

    def _step(carry, args):
        mean_prev, cov_prev, evidence, plugin = carry
        design_t, counts_t, x_hat = args
        one_step_mean = A @ mean_prev
        one_step_cov = symmetrize(A @ symmetrize(cov_prev) @ A.T + Q)

        eta_hat = jnp.atleast_1d(problem.log_intensity(design_t, x_hat))     # (n_eta,)
        jacobian = problem.grad_log_intensity(design_t, x_hat)                # (n_eta, d)
        weight, residual, mu = _pseudo_observation_sites(eta_hat, counts_t, family)
        info_matrix, info_vector = _information_site(jacobian, weight, residual, x_hat)
        log_prior_extra = 0.0
        if problem.extra_sites is not None:                                   # phase 3
            g_extra, v_extra, log_prior_extra = problem.extra_sites(x_hat)
            info_matrix, info_vector = info_matrix + g_extra, info_vector + v_extra

        post_mean, post_cov, gaussian_terms = _linearised_measurement_update(
            one_step_mean, one_step_cov, info_matrix, info_vector
        )
        # Per-bin Laplace evidence in the combined form (#laplace-evidence):
        loglik_t = family.loglik_normalized(counts_t, eta_hat, mu)
        evidence_t = _site_log_value(
            post_mean, x_hat, info_matrix, info_vector, loglik_t + log_prior_extra
        ) + gaussian_terms
        return (post_mean, post_cov, evidence + evidence_t, plugin + loglik_t), (
            post_mean,
            post_cov,
        )

    init = (problem.init_mean, problem.init_cov, jnp.zeros(()), jnp.zeros(()))
    (_, _, evidence, plugin), (filtered_mean, filtered_cov) = jax.lax.scan(
        _step, init, (problem.design_matrix, problem.counts, path)
    )
    return filtered_mean, filtered_cov, evidence, plugin
```

The sequential pass is then `rts_backward_scan(filtered_mean, filtered_cov, A, Q)`
(`kalman.py:778-842`), which returns `(smoother_mean, smoother_cov, smoother_cross_cov)`
in the `Cov(x_t, x_{t+1} | ·)` convention the M-steps expect.

For the default linear intensity `grad_log_intensity(Z_t, x) = Z_t` (constant); pass an
analytical gradient as the block path does (`:1926-1933`) rather than `jax.jacfwd`.

## Line search

Each Gauss–Newton pass is damped: with the current path `X`, target `X*` from the linear
pass, direction `D = X* − X`, choose the largest
`α ∈ {1, ½, …, 2⁻¹¹}` with

```
Ψ(X + α D) ≥ Ψ(X) + c α ∇Ψ(X)·D − slack,   c = _ARMIJO_C = 1e-4,  slack = _MERIT_RTOL (1 + |Ψ(X)|),  _MERIT_RTOL = 1e-10
```

(the constants of `temporal_rate_gp.py:82-86` and `point_process_kalman.py:727`; the
slack lets a converged iterate take its round-off-sized full step so the unrolled gradient
sees the contraction, as explained at `multinomial_choice.py:226-233`). If no step passes,
keep `X` (`α = 0`) and count it in `n_unaccepted_steps` (the policy of
`_fisher_scoring_line_search`, `point_process_kalman.py:816-824`, and of
`multinomial_choice`; *not* the smallest-step fallback of `temporal_rate_gp`, which can
decrease `Ψ` — phase 4 records this change). If `Ψ(X)` is not finite, take the full step
and count it in `n_nonfinite_merit` (as `temporal_rate_gp.py:330-339`).

`∇Ψ(X)·D ≥ 0` because `D = H⁻¹ ∇Ψ` with `H` positive definite.

**Prior quadratic form along the segment.** `Ψ` restricted to the segment is
`Σ_t ℓ_t(x_t + α d_t) − ½ (q_xx + 2 α q_xd + α² q_dd)`, where the three scalars come from
the Markov prior residuals (new helpers in `kalman.py`, generalising
`temporal_rate_gp._prior_whitened_residuals`, `temporal_rate_gp.py:89-115`, to a non-zero
first mean and to the `cho_factor` convention of `psd_cholesky`):

```python
def markov_prior_residuals(
    states: Array, transition_matrix: Array, first_mean: Array | None = None
) -> Array:
    """Residuals ``[x_1 - first_mean, x_2 - A x_1, ..., x_T - A x_{T-1}]``.

    Parameters
    ----------
    states : Array, shape (n_time, n_state)
    transition_matrix : Array, shape (n_state, n_state)
    first_mean : Array, shape (n_state,) or None
        Prior mean of the first state (``A m_0``); ``None`` means zero, which is
        the homogeneous form used for a search direction.

    Returns
    -------
    Array, shape (n_time, n_state)
    """
    first = states[0] if first_mean is None else states[0] - first_mean
    innovations = states[1:] - states[:-1] @ transition_matrix.T
    return jnp.concatenate([first[None], innovations], axis=0)


def markov_prior_quadratic_form(
    residuals_a: Array,
    residuals_b: Array,
    first_cho: tuple[Array, bool],
    process_cho: tuple[Array, bool],
) -> Array:
    """``Σ_t a_t' Σ_t^{-1} b_t`` with ``Σ_1 = P_{1|0}`` and ``Σ_t = Q`` for t >= 2.

    ``first_cho`` / ``process_cho`` are :func:`psd_cholesky` factors, so the
    same stabilised matrices are used for every quadratic form.
    """
    first = residuals_a[0] @ jax.scipy.linalg.cho_solve(first_cho, residuals_b[0])
    rest = jnp.sum(
        residuals_a[1:] * jax.scipy.linalg.cho_solve(process_cho, residuals_b[1:].T).T
    )
    return first + rest
```

**Line-search step** (one Gauss–Newton pass; `psi_log_lik(path)` is
`Σ_t family.loglik_plugin` at `η(path)` vectorised over time; `_N_BACKTRACK = 12`):

```python
def _gauss_newton_line_search(
    problem, prior, path, target, psi_log_lik
) -> tuple[Array, Array, Array, Array, Array]:
    """Armijo-damped step from ``path`` toward ``target``.

    Returns ``(new_path, step, psi_new, nonfinite_merit, unaccepted)``.
    """
    direction = target - path
    A = problem.transition_matrix
    res_x = markov_prior_residuals(path, A, first_mean=prior.first_mean)
    res_d = markov_prior_residuals(direction, A)                       # homogeneous
    q_xx = markov_prior_quadratic_form(res_x, res_x, prior.first_cho, prior.process_cho)
    q_xd = markov_prior_quadratic_form(res_x, res_d, prior.first_cho, prior.process_cho)
    q_dd = markov_prior_quadratic_form(res_d, res_d, prior.first_cho, prior.process_cho)

    steps = 0.5 ** jnp.arange(_N_BACKTRACK, dtype=path.dtype)          # largest first
    loglik_current, dloglik = jax.jvp(psi_log_lik, (path,), (direction,))  # value and Σ_t r_t·(J_t d_t)
    psi_current = loglik_current - 0.5 * q_xx
    slope = dloglik - q_xd                                              # ∇Ψ(X)·D
    # One (n_time, n_obs) evaluation per trial step; lax.map keeps peak memory at
    # one path's worth instead of 12 (parallel=True at T=1e6 would otherwise
    # materialise 12 * T * n_obs values).
    trial_loglik = jax.lax.map(lambda a: psi_log_lik(path + a * direction), steps)
    psi_trial = trial_loglik - 0.5 * (q_xx + 2.0 * steps * q_xd + steps**2 * q_dd)
    slack = _MERIT_RTOL * (1.0 + jnp.abs(psi_current))
    accept = psi_trial >= psi_current + _ARMIJO_C * steps * slope - slack

    merit_finite = jnp.isfinite(psi_current)
    any_accepted = jnp.any(accept)
    index = jnp.argmax(accept)                                           # largest accepted
    step = jnp.where(merit_finite, jnp.where(any_accepted, steps[index], 0.0), 1.0)
    step = jax.lax.stop_gradient(step)                                   # discrete choice
    psi_new = jnp.where(
        merit_finite, jnp.where(any_accepted, psi_trial[index], psi_current), psi_trial[0]
    )
    new_path = path + step * direction
    return new_path, step, psi_new, ~merit_finite, merit_finite & ~any_accepted
```

`prior` is a small NamedTuple built once per core call:
`first_mean = A m_0`, `first_cho = psd_cholesky(A P_0 Aᵀ + Q)`, `process_cho = psd_cholesky(Q)`.
For `extra_sites` (decoder), `psi_log_lik` also adds `Σ_t log_prior_extra(x_t)` (the exact
penalty), so the merit is the true joint objective, not its Gauss–Newton model.

## Laplace evidence

At the accepted path `X̂` (final pass), the Laplace approximation to `log p(y)` is the
marginal likelihood of the linear-Gaussian pseudo-model corrected by the difference between
the true and the Gaussian-site log-likelihoods at the mode (Rasmussen & Williams 2006
eq. 3.32 in state-space form; Nickisch et al. 2018; exactly the expression at
`temporal_rate_gp.py:357-386`):

```
log Z_Laplace = LL_KF(ỹ_{1:T}) + Σ_t [ log p(y_t | η̂_t) − log N(ỹ_t; η̂_t, W_t⁻¹) ]
```

**Per-bin decomposition without `W⁻¹`.** For a Gaussian likelihood the per-step marginal is
exact at any point, in particular at the filtered mean `m_{t|t}` (Bayes' rule
`p(ỹ) = p(ỹ|x) p(x) / p(x|ỹ)`):

```
log p(ỹ_t | ỹ_{1:t−1}) = log N(ỹ_t; c_t + J_t m_{t|t}, W_t⁻¹) − ½ δ_tᵀ P_{t|t−1}⁻¹ δ_t − ½ log|P_{t|t−1}| + ½ log|P_{t|t}|,   δ_t = m_{t|t} − m_{t|t−1}
```

(the `½ d log 2π` terms cancel between the two Gaussians over `x`). Subtracting the site
term, with `Δ_t = J_t (m_{t|t} − x̂_t)` so that `c_t + J_t m_{t|t} = η̂_t + Δ_t` and
`ỹ_t − η̂_t = r_t / w_t` componentwise:

```
log N(ỹ; η̂ + Δ, W⁻¹) − log N(ỹ; η̂, W⁻¹)
  = −½ Σ_n [ w_n (r_n/w_n − Δ_n)² − w_n (r_n/w_n)² ]         (the ½ log(2π/w_n) terms cancel)
  = Σ_n [ r_n Δ_n − ½ w_n Δ_n² ]
```

Hence

```
log Z_Laplace = Σ_t { log p(y_t | η̂_t) + Σ_n [ r_{t,n} Δ_{t,n} − ½ w_{t,n} Δ_{t,n}² ] − ½ δ_tᵀ P_{t|t−1}⁻¹ δ_t − ½ log|P_{t|t−1}| + ½ log|P_{t|t}| }
```

which is what `_linearised_forward_scan` accumulates (`evidence_t`). No `log w`, no `r²/w`;
at a masked or zero-weight bin the bracket is `0`. It is algebraically identical to
`kf_marginal_ll + true_log_lik − site_log_lik` of `temporal_rate_gp.py:364-386`, whose
`log(2π · site_variance)` and `(ỹ − g)²/site_variance` terms are the ones that cancel
above; the phase-1 cross-check test pins the two to rtol 1e-8.

The Laplace covariances are the final pass's `smoother_cov` (marginal blocks of
`H⁻¹`) and `smoother_cross_cov` (its lag-one off-diagonal blocks); the test reconstructs
the full `H⁻¹` from them with `_full_covariance_from_smoother`
(`tests/test_oracle_gp.py:294-311`) and compares with the dense inverse.

With `include_laplace_normalization=False` the returned `marginal_log_likelihood` is
`plugin_log_likelihood = Σ_t log p(y_t | η̂_t)`.

## Convergence and diagnostics

After the last Gauss–Newton pass the final linear pass yields `X*_final`;
`max_abs_update = max|X*_final − X̂|` is the step the next pass *would* take. Near the mode
Newton converges quadratically, so a converged iteration has `max_abs_update` at round-off;
the test `converged := max_abs_update ≤ convergence_tol (1 + max|X̂|)` with
`convergence_tol = 1e-6` is the relative-change criterion applied to the fixed-point
residual rather than to the last accepted step (which is what `temporal_rate_gp` reports as
`max_abs_update`; phase 4 maps its result onto this stricter quantity). The per-pass
record is in [shared-contracts.md#iterated-smoother-diagnostics](shared-contracts.md#iterated-smoother-diagnostics).

Core skeleton:

```python
def _iterated_laplace_smoother_core(problem, initial_path, *, n_gauss_newton_passes,
                                    convergence_tol, parallel):
    prior = _markov_prior(problem)
    psi_log_lik = _make_path_log_likelihood(problem)   # Σ_t loglik_plugin (+ extra log prior)
    psi_0 = psi_log_lik(initial_path) - 0.5 * _prior_quadratic(prior, problem, initial_path)

    def _pass(carry, _):
        path, n_nonfinite, n_unaccepted = carry
        out = _linearised_smoother_pass(problem, path, parallel=parallel)
        new_path, step, psi_new, nonfinite, unaccepted = _gauss_newton_line_search(
            problem, prior, path, out.smoother_mean, psi_log_lik
        )
        max_abs_step = jnp.max(jnp.abs(new_path - path))
        carry = (new_path, n_nonfinite + nonfinite.astype(jnp.int32),
                 n_unaccepted + unaccepted.astype(jnp.int32))
        return carry, (psi_new, step, max_abs_step)

    (path, n_nonfinite, n_unaccepted), (psis, steps, max_steps) = jax.lax.scan(
        _pass, (initial_path, jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)),
        None, length=n_gauss_newton_passes,
    )
    final = _linearised_smoother_pass(problem, path, parallel=parallel)
    max_abs_update = jnp.max(jnp.abs(final.smoother_mean - path))
    converged = max_abs_update <= convergence_tol * (1.0 + jnp.max(jnp.abs(path)))
    diagnostics = IteratedSmootherDiagnostics(
        n_gauss_newton_passes=n_gauss_newton_passes,
        log_posterior=jnp.concatenate([psi_0[None], psis]),
        step_sizes=steps, max_abs_step=max_steps, max_abs_update=max_abs_update,
        converged=converged, n_nonfinite_merit=n_nonfinite, n_unaccepted_steps=n_unaccepted,
    )
    # The returned moments are those of the linear pass at the *accepted* path
    # (Laplace covariances at the mode); its smoother_mean is only the diagnostic.
    return final._replace(smoother_mean=path), diagnostics
```

Note the last line: the returned smoothed mean is the accepted path `X̂`, not the un-taken
target `X*_final` — the covariances belong to the linearisation at `X̂`. At convergence the
two agree to `max_abs_update`.

## Block-diagonal path

On the block path (`BlockDiagonalStructure`, `point_process_kalman.py:148-200`) neuron `j`
is an independent problem of dimension `block_size` with design `Z_base[:, None, :]`
(`(n_time, 1, nb)`), counts `spike_indicator[:, j][:, None]`, `A_j`, `Q_j`, `m_{0,j}`,
`P_{0,j}` and the linear intensity. `Ψ` separates over neurons, so the block path is
`jax.vmap(_iterated_laplace_smoother_core over neurons)` on per-neuron `_LinearisationProblem`s
with per-neuron line searches (independent `α_j`), started from the per-neuron
`(n_neurons, n_time, nb)` output of `_block_diagonal_smoother_core` (`:2037-2122`). Output
packaging reuses `_concatenate_neuron_means` and `_package_block_covs` (`:2137-2149`).
Diagnostics aggregate as stated in the contract.

Dense vs block agreement: identical to round-off when every neuron's line search accepts
the same step (in practice: all `α = 1`). When damping differs, the paths differ until
convergence (both then reach the same MAP) — the same caveat the block filter documents at
`:2191-2198`.

## Parallel-in-time pass

`_linearised_smoother_pass(..., parallel=True)`:

1. sites `(G_t, g_t)` for all `t` by `jax.vmap` over time (the `(T, N, d)` Jacobian is a
   transient of the vmap body; `G`, `g` are `(T, d, d)`, `(T, d)`);
2. **parallel information filter** (new `kalman._parallel_information_filter`) →
   `(filtered_mean, filtered_cov)`;
3. `kalman.parallel_kalman_smoother(filtered_mean, filtered_cov, A, Q)` (`kalman.py:1072-1216`)
   → smoothed means, covariances, cross-covariances;
4. evidence terms vectorised over `t` (below).

**Filter elements** (Särkkä & García-Fernández 2021, filtering elements
`(A_k, b_k, C_k, η_k, J_k)`; rewritten with `(G_k, g_k)` through the push-through identity
`Q Hᵀ (H Q Hᵀ + W⁻¹)⁻¹ = (I + Q G)⁻¹ Q Hᵀ W`, so nothing is divided by `w`):

```
k = 1:  m⁻ = A m_0,  P⁻ = A P_0 Aᵀ + Q,  P_1 = (I + P⁻ G_1)⁻¹ P⁻,  m_1 = m⁻ + P_1 (g_1 − G_1 m⁻)
        A_1 = 0,  b_1 = m_1,  C_1 = P_1,  η_1 = 0,  J_1 = 0
k ≥ 2:  M_k = (I + Q G_k)⁻¹
        A_k = M_k A,   b_k = M_k Q g_k,   C_k = M_k Q,   η_k = Aᵀ M_kᵀ g_k,   J_k = Aᵀ M_kᵀ G_k A
```

(`(I + G Q)⁻¹ = M_kᵀ` since `G`, `Q` are symmetric; `M_k Q = (Q⁻¹ + G_k)⁻¹` is symmetric
when `Q` is invertible and is symmetrised numerically otherwise.) `I + Q G` is nonsingular
for any PSD `Q`, `G` (its eigenvalues are `1 + eig(QG) ≥ 1`), so a singular `Q` is allowed.

**Associative operator** (`e_i` earlier, `e_j` later; one LU factor of `I + C_i J_j` serves
both the plain and the transposed system):

```python
class _FilterElement(NamedTuple):
    A: Array   # (..., d, d)
    b: Array   # (..., d)
    C: Array   # (..., d, d)
    eta: Array # (..., d)
    J: Array   # (..., d, d)


@jax.vmap
def _combine_filter_elements(e1: _FilterElement, e2: _FilterElement) -> _FilterElement:
    d = e1.b.shape[-1]
    identity = jnp.eye(d, dtype=e1.b.dtype)
    lu = jax.scipy.linalg.lu_factor(identity + e1.C @ e2.J)        # (I + C_i J_j)
    solve = lambda rhs: jax.scipy.linalg.lu_solve(lu, rhs)
    solve_t = lambda rhs: jax.scipy.linalg.lu_solve(lu, rhs, trans=1)  # (I + J_j C_i)^{-1} = ((I + C_i J_j)^T)^{-1}
    A = e2.A @ solve(e1.A)
    b = e2.A @ solve(e1.b + e1.C @ e2.eta) + e2.b
    C = symmetrize(e2.A @ solve(e1.C) @ e2.A.T + e2.C)
    eta = e1.A.T @ solve_t(e2.eta - e2.J @ e1.b) + e1.eta
    J = symmetrize(e1.A.T @ solve_t(e2.J) @ e1.A + e1.J)
    return _FilterElement(A, b, C, eta, J)


def _parallel_information_filter(init_mean, init_cov, transition_matrix, process_cov,
                                 info_matrices, info_vectors):
    """Filtered means/covariances of a linear-Gaussian model with information-form
    observations ``(G_t, g_t)`` by associative scan (O(log T) span)."""
    A, Q = transition_matrix, process_cov
    d = A.shape[0]
    identity = jnp.eye(d, dtype=A.dtype)

    m_pred = A @ init_mean
    P_pred = symmetrize(A @ init_cov @ A.T + Q)
    P_1 = symmetrize(jnp.linalg.solve(identity + P_pred @ info_matrices[0], P_pred))
    m_1 = m_pred + P_1 @ (info_vectors[0] - info_matrices[0] @ m_pred)
    first = _FilterElement(jnp.zeros((d, d)), m_1, P_1, jnp.zeros(d), jnp.zeros((d, d)))

    def _generic(G_k, g_k):
        M = jnp.linalg.solve(identity + Q @ G_k, identity)          # (I + Q G_k)^{-1}
        MQ = symmetrize(M @ Q)
        return _FilterElement(
            A=M @ A, b=MQ @ g_k, C=MQ, eta=A.T @ (M.T @ g_k),
            J=symmetrize(A.T @ (M.T @ G_k) @ A),
        )

    rest = jax.vmap(_generic)(info_matrices[1:], info_vectors[1:])
    elements = jax.tree_util.tree_map(
        lambda f, r: jnp.concatenate([f[None], r], axis=0), first, rest
    )
    scanned = jax.lax.associative_scan(_combine_filter_elements, elements)
    return scanned.b, scanned.C          # m_{t|t}, P_{t|t} for t = 1..T
```

The public covariance-form wrapper `parallel_kalman_filter(init_mean, init_cov, obs,
transition_matrix, process_cov, measurement_matrix, measurement_cov, validate_inputs=True)`
accepts `measurement_matrix` of shape `(n_obs, d)` or `(n_time, n_obs, d)` and
`measurement_cov` `(n_obs, n_obs)` or `(n_time, n_obs, n_obs)`, forms
`G_t = H_tᵀ R_t⁻¹ H_t`, `g_t = H_tᵀ R_t⁻¹ y_t` with `psd_solve`, runs the filter and returns
`(filtered_mean, filtered_cov, marginal_log_likelihood)` with the LL evaluated vectorised
as `Σ_t log N(y_t; H_t m_{t|t−1}, H_t P_{t|t−1} H_tᵀ + R_t)` from
`m_{t|t−1} = A m_{t−1|t−1}`, `P_{t|t−1} = A P_{t−1|t−1} Aᵀ + Q` (prepend the prior). It is
the time-varying-`H_t` capability `kalman_filter` lacks (`kalman.py:315`), in parallel form.

**Evidence in parallel.** With `prev = concat(init, filtered[:-1])`:

```python
m_pred = prev_mean @ A.T
P_pred = jax.vmap(lambda P: symmetrize(A @ P @ A.T + Q))(prev_cov)
delta = filtered_mean - m_pred
prior_cho = psd_cholesky(P_pred)                                    # batched
quad = jnp.einsum("ti,ti->t", delta, jax.vmap(lambda c, v: cho_solve(c, v))(prior_cho, delta))
gaussian_terms = -0.5 * quad - 0.5 * psd_logdet(prior_cho) + 0.5 * psd_logdet(psd_cholesky(filtered_cov))
# info_matrices/vectors already include all extra sites; log_prior_extra is
# evaluated at path (a zero vector when extra_sites is None).
site_values = jax.vmap(_site_log_value)(
    filtered_mean, path, info_matrices, info_vectors, loglik_t + log_prior_extra
)
evidence = jnp.sum(site_values + gaussian_terms)
```

`psd_cholesky` / `psd_logdet` accept batches (`utils.py:114-183`). The only numerical
difference from the sequential pass is that `log|P_{t|t}|` is taken from a factor of the
covariance rather than of the precision (O(1e-12) relative).

Both passes use `_site_log_value` on the combined information site. Adding only
`log_prior_extra(path[t])` would omit the extra site's linear and quadratic
terms at `filtered_mean[t]`. Those terms generally remain nonzero at convergence:
the filtered mean of the pseudo-model is not its smoothed linearisation point.

**Block path.** `jax.lax.map` over neurons of the single-neuron parallel core (not `vmap`):
the elements are `(n_time, nb, nb)` per neuron and `vmap` would hold all neurons' elements
at once (see [#numerical-notes](#numerical-notes)).

## Position-decoder sites

`_run_filter_scan` (`position_decoder.py:958-1178`) applies, before each Laplace update, a
rank-1 "track penalty" update at the *predicted* mean (`:1096-1107`): with `pen = pen(m⁻[:2])`,
`g = ∇pen`, it is the Kalman update of the scalar pseudo-observation
`z = gᵀ x + v`, `v ~ N(0, 2 pen + 1e-12)`, observed value `gᵀ m⁻ − 2 pen`. In Gauss–Newton
terms this is the GN model of the log-prior term `−pen(x)` written as `−½ ρ(x)²` with
`ρ = sqrt(2 pen)`, `∇ρ = g / ρ`: `G^{pen} = ∇ρ ∇ρᵀ = g gᵀ / (2 pen)` and the information
vector at the linearisation point is `−ρ ∇ρ = −g`. Moving the linearisation point to the
smoothed path gives the decoder's `extra_sites`:

```python
def _track_penalty_sites(x_hat: Array, penalty_value_fn, penalty_grad_fn, n_state: int):
    """Gauss-Newton pseudo-observation of the distance-to-track penalty at ``x_hat``.

    Returns ``(G_pen, g_pen, log_prior)`` with ``G_pen = g g' / (2 pen + 1e-12)``,
    ``g_pen = G_pen x_hat - g * 2 pen / (2 pen + 1e-12)`` and ``log_prior = -pen``;
    the ``1e-12`` is the same guard as the filter's Woodbury denominator.
    """
    pen = penalty_value_fn(x_hat[:2])
    grad_xy = penalty_grad_fn(x_hat[:2])
    grad = jnp.concatenate([grad_xy, jnp.zeros(n_state - 2, dtype=x_hat.dtype)])
    denom = 2.0 * pen + 1e-12
    info_matrix = jnp.outer(grad, grad) / denom
    info_vector = info_matrix @ x_hat - grad * (2.0 * pen / denom)
    return info_matrix, info_vector, -pen
```

(`g_pen` follows the `g = G x̂ + Jᵀ W (z − H x̂)` convention with `H = gᵀ`,
`W = 1/denom`, `z − H x̂ = −2 pen`.) The merit adds the *exact* `−Σ_t pen(x_t[:2])`. The
exterior part of the penalty (`½ dist²/σ²`, `:1072-1078`) is exactly of squared-residual
form, so GN is Newton there; the bilinear interior map is not, so GN is a PSD surrogate
Hessian — an ascent direction, made monotone by the line search.

Adaptive inflation (`:1109-1131`) is a filter-side robustness heuristic with no joint
objective; with `n_iterations > 1` it affects only the initialisation path
(`parallel=False`). The Gauss–Newton passes, covariances and evidence are those of the
un-inflated dynamics `(A, Q)` from `build_position_dynamics` (`:136-197`). A test asserts
that inflated and un-inflated runs converge to the same MAP path.

The nonlinear intensity uses `log_intensity_func` / `grad_log_intensity_func` from
`_run_filter_scan` (`:1028-1060`), hoisted so both the iteration-0 scan and the linearised
passes share them; with `parallel=True` the nominal path is `init_position` broadcast.

## Gradients

`fit_sgd` differentiates the evidence with respect to `(A, Q, m_0, P_0)` (models) or the
Matérn hyperparameters (phase 4). Decision for this plan: **unrolled** reverse-mode through
the fixed-length `lax.scan` of Gauss–Newton passes and the final pass, with the step-size
choice under `stop_gradient` (as `temporal_rate_gp.py:343`). At a converged fixed point the
full step is always accepted and the unrolled derivative equals the fixed-point
(implicit-function-theorem) derivative up to the contraction factor; away from convergence
it is the derivative of what was actually computed. Cost: reverse-mode memory grows as
`n_iterations × T × d²` (each pass stores its filtered/smoothed covariances). Implicit
differentiation at the fixed point (`jax.lax.custom_root` or a hand-written VJP solving one
linear pass with the transposed system) halves memory and removes the dependence on
`n_iterations`; it is a listed follow-up with a trigger in phase 1.

## Numerical notes

- **float64 required** as for the whole Laplace-EKF stack (`CLAUDE.md`); the public entry
  validation (`_validate_public_inputs`, `point_process_kalman.py:96-146`) runs once as
  today.
- **Cholesky shifts**: every factorisation is `psd_cholesky` (scale-relative `1e-12`
  shift, `utils.py:74-160`); the evidence's two log-determinants come from the factors the
  update used (same policy as `:1192-1208`). Scale equivariance of the iterated smoother
  under `x → c x` therefore holds like the filter's (`tests/test_point_process_kalman.py::TestScaleEquivariance`).
- **Singular `Q`**: `psd_cholesky(Q)` gives a zero-variance direction the shift
  `1e-12 · 1e-8 · max_j Q_jj` (`_stabilizing_shift`, `utils.py:74-112`), i.e. a finite
  precision `~1e20 / max Q`; the Gauss–Newton direction satisfies the constraint to
  round-off, so the whitened innovation contributes `~(1e-16 · scale)² · 1e20 / max Q`,
  negligible. Documented, not special-cased.
- **Zero-weight bins**: handled by the information form; no `min_weight` floor anywhere
  in this plan (phase 4 reproduces `temporal_rate_gp`'s floor through its family's
  `fisher_weight`).
- **Memory of a parallel pass** (elements `A, C, J`: `3 T d²`; `b, η`: `2 T d`; scan
  intermediates ≈ 2×; float64):
  `bytes ≈ 8 · T · (3 d² + 2 d) · 3`. Examples: `d = 2, T = 1e6`: 0.4 GB; `d = 8, T = 1e6`:
  5 GB; `d = 36` (one place-field neuron), `T = 1e6`: 95 GB (infeasible → `lax.map` over
  neurons and `T ≲ 2e5` at `d = 36` on a 24 GB device). The sequential pass is
  `O(T d²)` for outputs only (`filtered_cov`, `smoother_cov`, `cross_cov`: `3 T d²`).
- **Measured one-pass sequential cost** (this machine, CPU, arm64, jax 0.10.2, second
  call): `T = 1e4, d = 2, N = 50`: 0.165 s; `d = 8`: 0.213 s. Linear in `T`.

## Alternatives considered

- **Extend `kalman_filter` with time-varying `H_t` and call `kalman_smoother` with
  `R_t = W_t⁻¹`** (the `temporal_rate_gp` route). Rejected for the general case: `R_t`
  is infinite at zero-weight bins (needs a floor that perturbs the answer), the covariance
  form solves an `N × N` innovation system per bin (`N = 50` neurons vs `d = 2..8`), and
  the information-form update already exists as the template. The parallel wrapper
  `parallel_kalman_filter` does provide time-varying `H_t` for linear-Gaussian users.
- **Keep the Laplace-EKF initialisation under `parallel=True`.** Rejected: O(T) span
  dominates (see [#iteration-semantics](#iteration-semantics)).
- **`lax.while_loop` early exit.** Rejected for V1: not reverse-differentiable; fixed
  length matches `temporal_rate_gp` and compiles once. Open Question 1 in overview.md.
- **Smallest-step fallback when no Armijo step passes** (`temporal_rate_gp`). Rejected:
  can decrease `Ψ`; keeping the iterate preserves monotonicity and matches the other two
  line searches in the library.
- **Convergence on the last accepted step** instead of the un-taken next step. Rejected:
  the fixed-point residual is the quantity the Laplace covariances depend on; the final
  pass computes it for free.
- **A new module** for the core. Rejected: it would import `GLMFamily`,
  `BlockDiagonalStructure` and the block cores from `point_process_kalman.py`, which must
  import the core back (cycle); the core lives in `point_process_kalman.py` as a new
  section, the parallel filter and prior helpers in `kalman.py`.

## References

- Bell, B. M. (1994). The iterated Kalman smoother as a Gauss–Newton method. *SIAM J. Optim.* 4(3), 626–636. — The IKS iteration is Gauss–Newton on the joint MAP objective; one linear smoother pass computes the step.
- Fahrmeir, L. (1992). Posterior mode estimation by extended Kalman filtering for multivariate dynamic generalized linear models. *JASA* 87(418), 501–509. — Working-response (IRLS) pseudo-observations for dynamic GLMs; Fisher scoring = Newton for canonical links.
- García-Fernández, Á. F., Svensson, L. & Särkkä, S. (2017). Iterated posterior linearization smoother. *IEEE TAC* 62(4), 2056–2063. — With analytical linearisation the IPLS is the IEKS; convergence requires damping in general.
- Yaghoobi, F., Corenflos, A., Hassan, S. & Särkkä, S. (2021). Parallel iterated extended and sigma-point Kalman smoothers. *ICASSP*. (JAX code: github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers.) — Each IEKS iteration as associative scans over the linearised model, started from a nominal trajectory; O(log T) span per iteration.
- Särkkä, S. & García-Fernández, Á. F. (2021). Temporal parallelization of Bayesian smoothers. *IEEE TAC* 66(1), 299–306. — Filtering elements `(A, b, C, η, J)`, smoothing elements `(E, g, L)` and their associative operators.
- Paninski, L., Ahmadian, Y., Ferreira, D. G., Koyama, S., Rahnama Rad, K., Vidne, M., Vogelstein, J. & Wu, W. (2010). A new look at state-space models for neural data. *J. Comput. Neurosci.* 29, 107–126. — Block-tridiagonal Newton MAP path estimation in O(T); the Laplace covariance is the inverse block-tridiagonal Hessian.
- Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N. (2004). Dynamic analysis of neural encoding by point process adaptive filtering. *Neural Computation* 16, 971–998. — The one-pass Laplace-EKF filter (iteration 0).
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*, ch. 3 (eq. 3.32). — Laplace evidence at the mode.
- Nickisch, H., Solin, A. & Grigorevskiy, A. (2018). State space Gaussian processes with non-Gaussian likelihood. *ICML*. — Site-based (IRLS) Laplace inference and evidence in Kalman form.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed., Part II. — Mode estimation for non-Gaussian state-space models by iterating the linear-Gaussian smoother on artificial observations.

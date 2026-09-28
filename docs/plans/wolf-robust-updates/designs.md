# Designs

[← back to PLAN.md](PLAN.md) · [contracts](shared-contracts.md)

- [G1 — Weighted Gaussian measurement update](#g1-update)
- [G2 — Generalised evidence (the objective)](#g2-evidence)
- [G3 — Weighted observation M-step and its bias](#g3-bias)
- [G4 — EM through `run_em` and SGD](#g4-em-sgd)
- [S1 — Shared weight in the switching filter](#s1-shared-weight)
- [S2 — Weighted switching M-step](#s2-switching-mstep)
- [P1 — Deviance-weighted GLM Laplace update](#p1-deviance)
- [P2 — Point-process filters, block path, position decoder](#p2-filters)

Notation: `m, P` prior (predicted) mean and covariance; `y` observation (`d = n_obs`); `H`, `R` measurement matrix and covariance; `e = y − H m`; `w ∈ [0, 1]` the weight; `logpdf(x; μ, Σ)` is `jax.scipy.stats.multivariate_normal.logpdf`.

<a id="g1-update"></a>

## G1 — Weighted Gaussian measurement update

**Paper.** Prop. 3.1 / Algorithm 1: the WoLF update is the Kalman update with `R⁻¹` replaced by `w² R⁻¹`, i.e. `R → R / w²`, where `w = W(y, ŷ)` and `ŷ = H m` is the prior predictive mean. That form is infinite at `w = 0`; rewrite it with `S̃ = w² H P Hᵀ + R`:

```text
K  = P Hᵀ (H P Hᵀ + R / w²)⁻¹ = w² P Hᵀ S̃⁻¹ =: w² K̃
P⁺ = (I − K H) P (I − K H)ᵀ + K (R / w²) Kᵀ = (I − K H) P (I − K H)ᵀ + w² K̃ R K̃ᵀ
```

Both are finite for every `w ∈ [0, 1]`; at `w = 0`, `K = 0` and `P⁺ = P` (the observation is dropped, the paper's TMD case). For `w > 0` this is algebraically the standard Joseph-form update with `R / w²`, which the phase 1 equivalence test checks at `rtol = 1e-10`.

**Code** (`kalman.py`, next to `kalman_measurement_update`):

```python
_LOG_2PI = math.log(2.0 * math.pi)  # kalman.py does not import numpy; add ``import math``


def standardized_residual(residual: jax.Array, measurement_cov: jax.Array) -> jax.Array:
    """``L^{-1} residual`` with ``L L^T = measurement_cov`` (shape (n_obs,))."""
    chol = jnp.linalg.cholesky(symmetrize(measurement_cov))
    return jax.scipy.linalg.solve_triangular(chol, residual, lower=True)


def weighted_kalman_measurement_update(
    prior_mean, prior_cov, obs, measurement_matrix, measurement_cov, weight
):
    """WoLF measurement update for a given scalar ``weight`` (Duran-Martin et
    al. 2024, Prop. 3.1), written with ``S~ = w^2 H P H^T + R`` so it is
    finite at ``weight == 0``. Returns ``(posterior_mean, posterior_cov,
    log_likelihood, objective)`` where ``log_likelihood`` is the unweighted
    predictive log density and ``objective`` the per-step generalised
    evidence (see G2)."""
    n_obs = obs.shape[0]
    w2 = weight * weight
    obs_mean = measurement_matrix @ prior_mean
    residual = obs - obs_mean
    HP = measurement_matrix @ prior_cov
    HPHt = symmetrize(HP @ measurement_matrix.T)
    obs_cov = symmetrize(HPHt + measurement_cov)          # S, unweighted
    scaled_obs_cov = symmetrize(w2 * HPHt + measurement_cov)  # S~
    unit_gain = _gain_solve(scaled_obs_cov, HP).T          # K~ = P H^T S~^{-1}
    kalman_gain = w2 * unit_gain
    posterior_mean = prior_mean + kalman_gain @ residual
    identity = jnp.eye(prior_mean.shape[0], dtype=prior_cov.dtype)
    I_KH = identity - kalman_gain @ measurement_matrix
    posterior_cov = symmetrize(
        I_KH @ prior_cov @ I_KH.T + w2 * unit_gain @ measurement_cov @ unit_gain.T
    )
    log_likelihood = jnp.asarray(
        jax.scipy.stats.multivariate_normal.logpdf(x=obs, mean=obs_mean, cov=obs_cov)
    )
    _, logdet_R = jnp.linalg.slogdet(measurement_cov)
    objective = jax.scipy.stats.multivariate_normal.logpdf(
        x=weight * residual, mean=jnp.zeros(n_obs, obs.dtype), cov=scaled_obs_cov
    ) + 0.5 * (1.0 - w2) * (n_obs * _LOG_2PI + logdet_R)
    return posterior_mean, posterior_cov, log_likelihood, objective
```

`kalman_measurement_update(..., *, robust_weight=None)` then reads:

```python
@functools.partial(jax.jit, static_argnames=("robust_weight",))
def kalman_measurement_update(prior_mean, prior_cov, obs, measurement_matrix,
                              measurement_cov, *, robust_weight=None):
    if robust_weight is None:
        ...  # existing body, byte for byte (lines 455-475 today)
        return posterior_mean, posterior_cov, marginal_log_likelihood
    residual = obs - measurement_matrix @ prior_mean
    weight = jax.lax.stop_gradient(
        robust_weight(standardized_residual(residual, measurement_cov))
    )
    mean, cov, ll, objective = weighted_kalman_measurement_update(
        prior_mean, prior_cov, obs, measurement_matrix, measurement_cov, weight
    )
    return mean, cov, ll, RobustOutput(objective=objective, weights=weight)
```

Notes: the standardisation uses `R` (paper Eq. 18), not `S` — with `S` the weight would depend on the prior covariance and under-detect outliers early in a run when `P` is large. `_gain_solve` keeps its scale-relative shift, so the robust update is unit-equivariant like the standard one (extend `test_invariances.TestKalmanInvariances.test_observation_units` with a robust case). `jnp.linalg.slogdet` on `R` costs `O(d³)` once per step, the same order as the solve. A time-varying `R_t` (3-D `measurement_cov`) is used per step unchanged.

<a id="g2-evidence"></a>

## G2 — Generalised evidence (the objective)

The generalised posterior at step `t` is `q_t(x) ∝ q_{t|t−1}(x) N(y; Hx, R)^{w²}` (paper Eq. 14–15). Its normaliser is the natural per-step objective:

```text
Z_t(w) = ∫ N(x; m, P) N(y; H x, R)^{w²} dx
```

Using `N(y; μ, R)^{w²} = N(y; μ, R / w²) · exp(c(w))` with `c(w) = ½(1 − w²)(d log 2π + log|R|) − d log w`, and `∫ N(x; m, P) N(y; Hx, R / w²) dx = N(y; Hm, H P Hᵀ + R / w²)`:

```text
log Z_t(w) = log N(y; Hm, HPHᵀ + R/w²) + c(w)
           = logpdf(w e; 0, S̃) + ½ (1 − w²) (d log 2π + log|R|)        (S̃ = w² HPHᵀ + R)
```

The second line (the `d log w` terms cancel) is what G1 computes. Checks: `w = 1` gives the standard predictive log density; `w → 0` gives `−½ log|R| + ½ log|R| = 0` (a dropped observation contributes nothing); it is finite for all `w ∈ [0, 1]`. Phase 1 verifies the constant against 1-D numerical quadrature.

Why this and not `Σ_t w_t² log p(y_t | y_{1:t−1})`: the weighted M-step (G3) maximises `E_q[log p(x_{0:T}) + Σ_t w_t² log N(y_t; H x_t, R)]`, whose EM lower bound is `log Z_w = Σ_t log Z_t` (with `q` the generalised posterior computed by the robust filter + RTS — exact, because a tempered Gaussian likelihood is Gaussian). The tempered sum of predictive densities is not the quantity that bound targets, and it is also unbounded below at `w = 0`.

<a id="g3-bias"></a>

## G3 — Weighted observation M-step and its bias

With weights `w_t` fixed at the E-step's values and smoothed moments `m_t, P_t`, maximising the expected tempered complete-data objective over the observation block gives

```text
H = (Σ_t w_t² y_t m_tᵀ) (Σ_t w_t² (P_t + m_t m_tᵀ))⁻¹
R = Σ_t w_t² [ (y_t − H m_t)(y_t − H m_t)ᵀ + H P_t Hᵀ ] / Σ_t w_t²
```

A, Q and the initial-state update are unchanged (the state prior is not tempered). `measurement_cov_residual_form` gains `weights=None`; with weights, `sum_smoother_cov` must be the weighted sum `Σ w_t² P_t` and the divisor is `Σ w_t²`:

```python
def measurement_cov_residual_form(obs, smoother_mean, sum_smoother_cov,
                                  measurement_matrix, *, weights=None,
                                  previous_measurement_cov=None):
    residual = obs - smoother_mean @ measurement_matrix.T
    if weights is None:
        scatter = residual.T @ residual
        denominator = obs.shape[0]
    else:
        if previous_measurement_cov is None:
            raise ValueError("weights require previous_measurement_cov")
        w2 = jnp.asarray(weights) ** 2
        scatter = (residual * w2[:, None]).T @ residual
        denominator = jnp.sum(w2)
    numerator = symmetrize(
        scatter + measurement_matrix @ sum_smoother_cov @ measurement_matrix.T
    )
    if weights is None:
        return numerator / denominator
    # Safe even when transformed by vmap, which may evaluate both branches.
    estimate = numerator / jnp.where(denominator > 0, denominator, 1.0)
    return jnp.where(denominator > 0, estimate, previous_measurement_cov)
```

and in `_kalman_maximization_step` (jitted; `robust_weights=None` is a valid pytree argument, exactly like `initial_state_prior=None` at `kalman.py:1585`):

```python
if robust_weights is None:
    sum_cov_obs, gamma_obs, delta = sum_cov, gamma, delta   # today's code
    measurement_matrix = _gain_solve(gamma_obs, delta.T).T
    measurement_cov = project_psd_relative(
        measurement_cov_residual_form(obs, smoother_mean, sum_cov_obs,
                                      measurement_matrix), ...)
else:
    w2 = robust_weights ** 2
    sum_cov_obs = jnp.sum(smoother_cov * w2[:, None, None], axis=0)
    gamma_obs = sum_cov_obs + (smoother_mean * w2[:, None]).T @ smoother_mean
    delta = (obs * w2[:, None]).T @ smoother_mean
    H_prev = previous_params["measurement_matrix"]
    R_prev = previous_params["measurement_cov"]
    has_observations = jnp.sum(w2) > 0.0
    safe_gamma = jnp.where(has_observations, gamma_obs, jnp.eye(gamma_obs.shape[0]))
    H_candidate = _gain_solve(safe_gamma, delta.T).T
    R_candidate = measurement_cov_residual_form(
        obs, smoother_mean, sum_cov_obs, H_candidate, weights=robust_weights,
        previous_measurement_cov=R_prev,
    )
    measurement_matrix = jnp.where(has_observations, H_candidate, H_prev)
    measurement_cov = jax.lax.cond(
        has_observations,
        lambda R: project_psd_relative(R, name="robust measurement_cov"),
        lambda _R: R_prev,
        R_candidate,
    )
```

`sum_cov` and `gamma` (unweighted) keep feeding the transition block.

The public Gaussian M-step gains keyword-only `previous_params=None`, following
the switching M-step's existing dictionary convention. Whenever `robust_weights`
is supplied, require previous `measurement_matrix` and `measurement_cov` entries
with compatible shapes; this static validation also works under `jit`. Model
callers and the EM test harness pass the E-step's parameter snapshot. Exactly
zero total squared weight leaves H and R unchanged, including no PSD projection
of the retained R; dynamics and initial-state updates still run. Positive totals
use the weighted maximizer. Do not insert an arbitrary positive floor into the
divisor, which would change that maximizer.

**Monotonicity.** At fixed weights this is exact EM for `log Z_w`, so the objective cannot decrease across one M-step. The next E-step recomputes weights from the new parameters, so across iterations the scheme is *generalised* EM; `run_em`'s decrease check remains the safety net (overview, open question 1).

**Bias on clean data.** Because the IMQ down-weights large legitimate residuals, the weighted scatter under-estimates `R` even without outliers, and the bias compounds through the weights until a fixed point. With `z ~ N(0, I_d)`, `κ(c, d, s) = E[w² ‖z‖²] / (d E[w²])` for a filter running at `R̂ = s R`; iterating `s ← s κ(c, d, s)` (computed by quadrature over the χ²_d radial density):

| `d` | `c` | one-step `R̂/R` at truth | EM fixed point `s*` (plain IMQ) | `s*` with `core = √χ²_d(0.99)` |
| --- | --- | --- | --- | --- |
| 1 | 2 | 0.746 | 0.691 | 0.976 |
| 1 | 3 | 0.849 | 0.828 | 0.986 |
| 1 | 4 | 0.902 | 0.893 | — |
| 3 | 2 | 0.786 | 0.756 | 0.988 |
| 3 | 3 | 0.866 | 0.851 | 0.993 |
| 3 | 4 | 0.910 | 0.903 | — |
| 8 | 3 | 0.896 | 0.889 | 0.996 |
| 8 | 5 | 0.944 | 0.942 | — |

The fixed point is benign (no collapse), but a 10–17 % downward bias in `R` is a real effect. Decision: no consistency-correction factor (it would break the exact fixed-weight M-step), document the bias in `kalman_maximization_step` / `switching_kalman_maximization_step` docstrings with this table's gist, and recommend `imq_weight(c, core=sqrt(chi2.ppf(0.99, n_obs)))` for EM fits. The phase 1 EM recovery test uses that setting and a 15 % tolerance.

<a id="g4-em-sgd"></a>

## G4 — EM through `run_em`, and SGD

`run_em` (`em_driver.py:58-77`) receives `e_step: Callable[[], float]` and compares successive values (`em_driver.py:223, 255-275`). Nothing in the driver knows about likelihoods. Therefore:

- **Minimal change:** each model's `_e_step` returns `RobustOutput.objective` when `self.robust_weight is not None` and the marginal LL otherwise; the `lambda: float(self._e_step(...))` closures (`oscillator_models.py:1131`, `point_process_kalman.py:3220`, `place_field_model.py:1364-1365`) need no edit. `EMResult.log_likelihoods` then holds the objective; the `fit` docstrings say so.
- The E-step also stores the weights on the model (`filter_robust_weights`) for the M-step and as a diagnostic; the attribute joins the rollback snapshot so a restored state carries matching weights.
- `decrease_tol` is left as-is (open question 1).

**SGD.** `_sgd_loss_fn` returns `−objective` when robust. Weights are computed under `jax.lax.stop_gradient` inside the filters ([contracts, invariant 1](shared-contracts.md#invariants)). Reason: `log Z_w → 0` when every `w_t → 0` (G2), and `0` typically exceeds the value at the true parameters (each Gaussian term is `≈ −½ d log(2πe)` on standardised data), so an optimiser allowed to differentiate through `w_t(R)` has a spurious "shrink R until everything is an outlier" direction. With the weights frozen per evaluation the loss is the fixed-weight objective, and each SGD step re-evaluates weights at the new parameters — the same majorise-then-optimise structure as the EM path. The phase 1 gradient test asserts finite non-zero gradients w.r.t. `R`, `A`, `init_cov` with the weights frozen.

<a id="s1-shared-weight"></a>

## S1 — Shared weight in the switching filter

`switching_kalman_filter` updates every `(S_{t−1}=i, S_t=j)` pair with its own prior predictive (`switching_kalman.py:933-945`). Two options were considered:

- *Per-pair weights* `w_{ij}`: a poorly fitting state has a larger innovation, hence a smaller weight, hence a per-step evidence closer to `0` — i.e. **larger** than the well-fitting state's. Tempering per pair systematically favours the wrong discrete state. Rejected.
- *One shared weight per time step* from the mixture prior predictive: all pairs are tempered equally, so their relative evidences keep their ordering; as `w → 0` every pair's term → 0 and the discrete posterior follows the transition prior (the outlier is ignored for both continuous and discrete states). Chosen.

Shared weight at step `t > 1` (inside `_step`, before the per-pair update):

```python
# prev_filter_discrete_prob: (K,), Z = discrete_transition_matrix: (K, K)
pair_prior = prev_filter_discrete_prob[:, None] * discrete_transition_matrix   # (K, K)
pair_prior = pair_prior / jnp.sum(pair_prior)


# Per-pair prior predictive means  yhat_ij = H_j A_j m^i  -> (n_obs, K_i, K_j):
# vmap over the destination state j (trailing axis of A and H), and inside it
# over the source state i (trailing axis of the previous state-conditional means).
def _predicted_obs_for_state(A_j, H_j):
    return jax.vmap(lambda m_i: H_j @ (A_j @ m_i), in_axes=-1, out_axes=-1)(
        prev_state_cond_filter_mean
    )  # (n_obs, K_i)


pred_mean = jax.vmap(_predicted_obs_for_state, in_axes=(-1, -1), out_axes=-1)(
    continuous_transition_matrix, measurement_matrix
)  # (n_obs, K_i, K_j)
predicted_obs = jnp.einsum("nij,ij->n", pred_mean, pair_prior)
state_prior = jnp.sum(pair_prior, axis=0)                                       # (K,)
mean_measurement_cov = jnp.einsum("abj,j->ab", measurement_cov, state_prior)    # (n_obs, n_obs)
weight = jax.lax.stop_gradient(
    robust_weight(standardized_residual(obs_t - predicted_obs, mean_measurement_cov))
)
```

For the oscillator models `measurement_cov` is shared across states (`_stack_shared_measurement_covariance`, `oscillator_models.py:322-337`), so `mean_measurement_cov == R` exactly. At `t = 1` (`_first_timestep_kalman_update`) use `state_prior = _normalize_initial_discrete_prob(init_discrete_state_prob)` and `pred_mean[:, j] = H_j m_j` (no dynamics).

Per-pair update with the shared weight: a new `_weighted_kalman_filter_update_per_discrete_state_pair` built like `switching_kalman.py:49-57` but over

```python
def _weighted_kalman_filter_update(mean_prev, cov_prev, obs, A, Q, H, R, weight):
    one_step_mean = A @ mean_prev
    one_step_cov = symmetrize(A @ cov_prev @ A.T + Q)
    return weighted_kalman_measurement_update(one_step_mean, one_step_cov, obs, H, R, weight)
```

with `in_axes` `(-1, -1, None, None, None, None, None, None)` inside and `(None, None, None, -1, -1, -1, -1, None)` outside, `out_axes=-1` (four outputs: means, covs, unweighted LLs `(K, K)`, objectives `(K, K)`). The first step vmaps `weighted_kalman_measurement_update` over `j` with `in_axes=(-1, -1, None, -1, -1, None)`.

Discrete update and log-likelihoods:

```python
filter_discrete_prob, backward, log_predictive_objective, next_support = (
    _update_discrete_state_probabilities(pair_objective, Z, prev_prob, prev_support)
)
# Unweighted predictive LL for reporting: same prior, unweighted per-pair LLs.
_, _, log_predictive_ll, _ = _update_discrete_state_probabilities(
    pair_cond_marginal_log_likelihood, Z, prev_prob, prev_support
)
marginal_log_likelihood += log_predictive_ll
objective += log_predictive_objective
```

The first step uses `_first_timestep_discrete_update` twice in the same way. The carry gains `objective`; the scan emits `weight` per step; the return appends `RobustOutput(objective, weights)` (`(n_time,)`, first-step weight prepended). The `None` branch is the existing scan body untouched.

<a id="s2-switching-mstep"></a>

## S2 — Weighted switching M-step

`switching_kalman_maximization_step(..., robust_weights=None)` forms

```python
obs_weights = (smoother_discrete_state_prob
               if robust_weights is None
               else smoother_discrete_state_prob * (robust_weights ** 2)[:, None])
```

and passes it to `_switching_kalman_m_step_inner` together with a static flag `use_robust_weights`. Inside, when the flag is set, the observation block uses `obs_weights` in place of `smoother_discrete_state_prob` for `weighted_cov_sum`, `gamma`, `delta` and the R scatter / divisor (`switching_kalman.py:2206-2251`), while `gamma2` (2222-2229) and everything from 2253 on keep using the unweighted probabilities (compute a separate unweighted `gamma` for `gamma2`). When the flag is not set the function body is unchanged. The transition occupancy gate (`2514-2539`) stays on unweighted occupancy. Add a separate observation-block gate on `obs_weights.sum(0) > 0`: a state with zero effective observation weight retains its previous H and R while its dynamics can still update. Require `previous_params` whenever robust weights are given and measurement parameters are estimated; reuse the safe solve/division and exact-zero fallback of G3 per state. Shared observation parameters use the effective weight pooled over the states that share them, with the same zero-total fallback.

<a id="p1-deviance"></a>

## P1 — Deviance-weighted GLM Laplace update

**What the paper gives.** App. D.1 extends WoLF to exponential families by tempering the moment-matched Gaussian log-likelihood of the exponential-family EKF (Ollivier 2018) by `W²` (Eq. 60) and explicitly leaves the choice of `W` for non-Gaussian families to future work. Our Fisher-scoring update *is* that Gaussian approximation (precision `P⁻¹ + Jᵀ diag(v) J` with `v` the Fisher weight), so tempering it by `w²` per observation is the D.1 update; the standardisation, the per-neuron granularity and the validation are ours.

**Why not the Pearson residual.** Per bin, a single spike at a low expected count is a large Pearson residual even though it is ordinary data (`P(Y ≥ 1) = 4.9 %` at `μ = 0.05`), so an IMQ on Pearson residuals down-weights spikes relative to silence and biases the rate:

| `μ` | `y` | Pearson `z` | `w²` (c = 4) | deviance `z` | `w²` (c = 4) | `w²` (c = 4, core = 3) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 1 | 4.25 | 0.47 | 2.02 | 0.80 | 1.00 |
| 0.05 | 2 | 8.72 | 0.17 | 3.29 | 0.60 | 0.90 |
| 0.05 | 8 | 35.6 | 0.012 | 8.08 | 0.20 | 0.22 |
| 0.5 | 8 | 10.6 | 0.13 | 5.42 | 0.35 | 0.44 |
| 2.0 | 8 | 4.24 | 0.47 | 3.19 | 0.61 | 0.93 |

Implied rate bias `μ̂/μ` on clean Poisson data at `c = 4`: Pearson-IMQ 0.47 (`μ = 0.05`), 0.81 (`μ = 0.2`); deviance-IMQ 0.80, 0.91; deviance-IMQ with `core = 3`: 0.995, 0.998. The signed deviance residual `z = sign(y − μ) √(2[y log(y/μ) − (y − μ)])` is the signed square root of a likelihood-ratio statistic, so it is calibrated in tail-probability terms even at tiny `μ` (Wilks), and with `core = 3` ordinary spikes are untouched while an 8-spike burst is down-weighted to `w² ≈ 0.2`. Decision: standardise with the unit deviance; recommend `core ≈ 3` for spikes.

**Family additions** (`GLMFamily`, trailing optional fields so four-field positional construction keeps working). Reuse `loglik_per_obs` if masks or the GLM plan already added it; preserve existing `score` and `validate_observations` fields and their order. Optional additions are passed by keyword:

```python
class GLMFamily(NamedTuple):
    mean: Callable[[Array], Array]
    fisher_weight: Callable[[Array, Array], Array]
    loglik_plugin: Callable[[Array, Array, Array], Array]
    loglik_normalized: Callable[[Array, Array, Array], Array]
    unit_deviance: Callable[[Array, Array], Array] | None = None   # (y, mu) -> (n_obs,)
    loglik_per_obs: Callable[[Array, Array, Array], Array] | None = None  # (y, eta, mu) -> (n_obs,)
```

Poisson: `unit_deviance = 2 * (xlogy(y, y) - y * log(mu) - (y - mu))` (`jax.scipy.special.xlogy`, so `y = 0` is exact and differentiable), `loglik_per_obs = jax.scipy.stats.poisson.logpmf(y, mu)`. Bernoulli: `unit_deviance = -2 * (y * log(mu) + (1 - y) * log1p(-mu))` and `loglik_per_obs = y * eta_c - softplus(eta_c)` with the same clipping as `_bernoulli_loglik`. `glm_laplace_update` raises `ValueError` when `robust_weight` is given and either field is `None`.

**Weights and the tempered update** (both siblings; shown for the family-generic one):

```python
def _deviance_weights(y, mu, unit_deviance, robust_weight):
    z = jnp.sign(y - mu) * jnp.sqrt(jnp.maximum(unit_deviance(y, mu), 0.0))
    return jax.lax.stop_gradient(jax.vmap(lambda zi: robust_weight(zi[None]))(z))

# at the top of the update, before any Fisher iteration:
eta0 = eta_func(one_step_mean); mu0 = family.mean(eta0)
w = _deviance_weights(observations, mu0, family.unit_deviance, robust_weight)   # (n_obs,)
w2 = w * w

# _neg_log_posterior: log_lik = jnp.sum(w2 * family.loglik_per_obs(observations, eta, mu))
# _fisher_step_at and the single-step branch:
likelihood_gradient = jacobian.T @ (w2 * innovation)
fisher_info = jacobian.T @ ((w2 * weight)[:, None] * jacobian)
```

(The legacy `_point_process_laplace_update` uses the inline Poisson deviance and `poisson.logpmf` per observation; its line search and Fisher step change identically.) With `robust_weight=None` the existing code runs unchanged.

**Returned quantities** (with `x*`, `η*`, `μ*` at the robust mode, `Λ_w = P⁻¹ + Jᵀ diag(w² v*) J` the robust posterior precision whose Cholesky the update already has):

```text
per_obs   = loglik_per_obs(y, η*, μ*)
objective = Σ w² per_obs  [+ −½ δᵀP⁻¹δ − ½ log|P| − ½ log|Λ_w|      if normalised]
log_likelihood (unweighted) = Σ per_obs
            [+ −½ δᵀP⁻¹δ − ½ log|P| − ½ log|Λ_1|                   if normalised]
            with Λ_1 = P⁻¹ + Jᵀ diag(v*) J  (one extra psd_cholesky + psd_logdet)
```

`objective` is the Laplace approximation of `log ∫ N(x; m, P) Π_n p(y_n | x)^{w_n²} dx`; `log_likelihood` is the Laplace approximation of the unweighted evidence expanded at the robust mode (an approximation whose error grows as weights depart from 1; its purpose is comparison across `c`, not inference). The `RobustOutput.weights` are `w` (`(n_obs,)`).

<a id="p2-filters"></a>

## P2 — Point-process filters, block path, position decoder

- **Dense filter** (`_stochastic_point_process_filter_impl`): `robust_weight` joins `static_argnames`; the scan step calls the Laplace update with it, accumulates `objective` in the carry and emits `weights_t` per step; the public filter returns `RobustOutput(objective, weights (n_time, n_neurons))` appended.
- **Block-diagonal core** (`_block_diagonal_forward_core`): the same static argument; each per-neuron scan calls the update on its single-neuron problem, so it produces that neuron's own weight — identical to the dense per-neuron weight because neuron `n`'s prior predictive depends only on block `n` (block-diagonal `A`, `Q`, `P`, design). `vmap` output `(n_neurons, n_time)` is transposed to `(n_time, n_neurons)`; per-neuron objectives are summed. Dense/block equivalence therefore holds under robust weights up to the pre-existing per-neuron-vs-global line-search caveat (`point_process_kalman.py:2191-2198`), and the equivalence test is extended.
- **Smoother** (`stochastic_point_process_smoother` and the block smoother): pass-through; `RobustOutput` appended after the optional filtered pair.
- **Position decoder** (`_run_filter_scan`): `robust_weight` joins `static_argnames`. In `_step`, after the track-penalty downdate (which moves the predicted mean) and before the inflation block, compute `w = _deviance_weights(spike_t, cond_int0, poisson_unit_deviance, robust_weight)` from the prior-predictive rate; when `inflate`, the surprise statistic uses `w2 * innovation` and `cond_int * w2` so an ignored spike does not inflate the prior; the Laplace update receives `robust_weight` and its weights (recomputed from the same inputs, hence identical) are emitted per step; the scan returns `objective` and `weights` in addition, and `DecoderResult(..., robust=RobustOutput(...))`. `marginal_log_likelihood` stays unweighted.
- **Models.** `PointProcessModel`, `PlaceFieldModel`, `PositionDecoder` take `robust_weight=None` in their constructors and forward it to every filter/smoother call (`_e_step`, `_sgd_loss_fn`, `_finalize_sgd`, `decode`); `_e_step` returns the objective and stores `filter_robust_weights` when robust (`PlaceFieldModel.score` and its filter call stay unweighted). Their M-steps are dynamics-only (`dynamics_only_m_step`) and need no weighting.

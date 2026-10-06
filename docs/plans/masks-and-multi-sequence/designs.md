# Designs

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Per-component algorithms with complete code and rationale. Phases link in by
anchor; this file does not repeat their task lists. Code samples are written
against the conventions of the modules they land in (`jnp`, `lax.scan`, time
axis leading, discrete-state axis trailing, float64).

- [D1 — Masked Gaussian measurement update (static-shape trick)](#d1--masked-gaussian-measurement-update-static-shape-trick)
- [D2 — Masked Gaussian M-step (imputation form)](#d2--masked-gaussian-m-step-imputation-form)
- [D3 — Masked Laplace-EKF update](#d3--masked-laplace-ekf-update)
- [D4 — Multi-sequence sufficient statistics (two-pass)](#d4--multi-sequence-sufficient-statistics-two-pass)
- [D5 — Batching strategy and measured memory](#d5--batching-strategy-and-measured-memory)
- [D6 — Oracle extensions for masks and sequences](#d6--oracle-extensions-for-masks-and-sequences)
- [D7 — Block-diagonal covariances with a sequence axis](#d7--block-diagonal-covariances-with-a-sequence-axis)

---

## D1 — Masked Gaussian measurement update (static-shape trick)

**Problem.** `kalman_measurement_update` (`kalman.py:426-475`) forms
`S = H P H^T + R`, solves `K^T = S^{-1} H P` with `_gain_solve` (`:461`), applies
the Joseph form (`:464-466`) and evaluates `logpdf(y; H m, S)` (`:471-473`).
Conditioning on the observed sub-vector would change array shapes per step,
which `lax.scan` and `jit` cannot do. Keep the shapes and make the masked
channels inert instead.

**Transform.** With `m = mask_t.astype(dtype)` (`(n_obs,)`), `D = diag(m)`:

```
H_m = D H                    zero the masked rows of H
R_m = D R D + (I - D)        masked rows/cols of R replaced by an identity block
y_m = where(mask, y, 0)      masked entries zeroed (NaN-safe)
```

Then `S_m = H_m P H_m^T + R_m` is block-diagonal after a permutation:
`S_oo` on the observed block, `I` on the masked block, zero cross-blocks. The
masked rows of `H_m P` are exactly zero, so the masked rows of `K^T = S_m^{-1} H_m P`
are exactly zero (the Cholesky factor has no coupling between the blocks; the
triangular solves propagate exact zeros). Hence the masked columns of `K` are
zero, `K y_m` and `K R_m K^T` see only observed channels, and the Joseph form
equals the observed-only update. For the log-likelihood,

```
logpdf(y_m; H_m m, S_m) = -1/2 [ n_obs log 2π + log|S_oo| + r_o^T S_oo^{-1} r_o ]
                        = log N(y_o; H_o m, S_oo)  -  1/2 n_masked log 2π ,
```

so adding `1/2 n_masked log 2π` gives the exact marginal over observed channels.
A fully masked bin gives `K = 0`, posterior = prediction (Joseph form
`(I)P(I)^T + 0`, symmetrised, equal to the already-symmetric prediction
bit-for-bit) and LL `-1/2 n_obs log 2π + 1/2 n_obs log 2π = 0`.

The `_gain_solve` stabilising shift (`kalman.py:55-101`) adds `1e-14 x` the
diagonal; on the identity block that is `1 + 1e-14`, harmless. The LL uses the
unshifted `S_m` as today.

**Code** (`kalman.py`, next to `kalman_measurement_update`):

```python
_LOG_2PI = math.log(2.0 * math.pi)


def _mask_gaussian_measurement(
    obs: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
    obs_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Static-shape masking of one Gaussian observation.

    Masked rows of ``H`` are zeroed, the masked rows/columns of ``R`` are
    replaced by an identity block and the masked entries of ``obs`` by zero,
    so the update with the returned arrays equals the update on the observed
    sub-vector while every array keeps its shape.

    Parameters
    ----------
    obs : jax.Array, shape (n_obs,)
    measurement_matrix : jax.Array, shape (n_obs, n_state)
    measurement_cov : jax.Array, shape (n_obs, n_obs)
    obs_mask : jax.Array, bool, shape (n_obs,)
        True where the entry was observed.

    Returns
    -------
    obs_masked : jax.Array, shape (n_obs,)
    measurement_matrix_masked : jax.Array, shape (n_obs, n_state)
    measurement_cov_masked : jax.Array, shape (n_obs, n_obs)
    n_masked : jax.Array, scalar
        Number of masked entries, for the ``+ n_masked/2 log 2π``
        log-likelihood correction.
    """
    m = obs_mask.astype(measurement_matrix.dtype)
    obs_masked = jnp.where(obs_mask, obs, 0.0)
    measurement_matrix_masked = measurement_matrix * m[:, None]
    measurement_cov_masked = measurement_cov * (m[:, None] * m[None, :]) + jnp.diag(
        1.0 - m
    )
    return obs_masked, measurement_matrix_masked, measurement_cov_masked, jnp.sum(1.0 - m)
```

`kalman_measurement_update` becomes:

```python
@jax.jit
def kalman_measurement_update(
    prior_mean, prior_cov, obs, measurement_matrix, measurement_cov,
    obs_mask: jax.Array | None = None,
):
    if obs_mask is not None:
        obs, measurement_matrix, measurement_cov, n_masked = _mask_gaussian_measurement(
            obs, measurement_matrix, measurement_cov, obs_mask
        )
    obs_mean = measurement_matrix @ prior_mean
    ...  # unchanged lines 456-473
    marginal_log_likelihood = jnp.asarray(
        jax.scipy.stats.multivariate_normal.logpdf(x=obs, mean=obs_mean, cov=obs_cov)
    )
    if obs_mask is not None:
        marginal_log_likelihood = marginal_log_likelihood + 0.5 * n_masked * _LOG_2PI
    return posterior_mean, posterior_cov, marginal_log_likelihood
```

`obs_mask` is a *trailing positional-or-keyword* parameter here and on
`_kalman_filter_update` (`kalman.py:478-526`) — not keyword-only — because
`jax.vmap` maps keyword arguments over axis 0 unconditionally; the switching
filter's `in_axes` (`switching_kalman.py:49-57`, `:747-751`) must be able to
name the mask's axis (`None`). Public entry points (`kalman_filter`,
`kalman_smoother`, ...) take it keyword-only.

In `_kalman_filter_impl` (`kalman.py:529-610`) the mask joins the scan inputs
under a static branch:

```python
def _kalman_filter_impl(..., measurement_cov, obs_mask=None):
    ...
    if obs_mask is not None:
        obs = jnp.where(obs_mask, obs, 0.0)   # NaN-safe before anything else

    def _step(carry, step_inputs):
        mean_prev, cov_prev, marginal_log_likelihood = carry
        if obs_mask is not None:
            step_inputs, obs_mask_t = step_inputs
        else:
            obs_mask_t = None
        if time_varying_measurement_cov:
            obs_t, measurement_cov_t = step_inputs
        else:
            obs_t, measurement_cov_t = step_inputs, measurement_cov
        posterior_mean, posterior_cov, ll_t = _kalman_filter_update(
            mean_prev, cov_prev, obs_t, transition_matrix, process_cov,
            measurement_matrix, measurement_cov_t, obs_mask_t,
        )
        ...

    scan_inputs = (obs, measurement_cov) if time_varying_measurement_cov else obs
    if obs_mask is not None:
        scan_inputs = (scan_inputs, obs_mask)
```

Four `jit` specialisations (mask × time-varying R) — acceptable; the `None`
branches are the pre-existing code ([C5](shared-contracts.md#c5--backwards-compatibility-off-means-the-old-code-runs)).

**Why not `woodbury_kalman_gain` / `standard_kalman_gain`?** Neither is used by
the filters (`kalman.py:104-232` have no callers outside tests). They consume
`(P, H, R)` and compose with the same transform (`woodbury` needs the diagonal
of `R_m`, which is `diag(R) * m + (1 - m)`), but they are out of the change set.

## D2 — Masked Gaussian M-step (imputation form)

**Complete data and expectations.** For bin `t` with mask `m_t`, observed block
`o`, missing block `m`, state posterior `N(m_t, P_t)` and the E-step's `(H, R)`:

```
y^m | x, y^o  ~  N( H_m x + R_mo R_oo^{-1} (y_o - H_o x),  R_mm - R_mo R_oo^{-1} R_om )
```

Embedding everything in full `n_obs` shape with `D = diag(m)`, `R_m = D R D + (I - D)`
(so `D R_m^{-1} D` is `R_oo^{-1}` embedded) and `W = (I - D) R (D R_m^{-1} D)`
(`R_mo R_oo^{-1}` embedded):

```
ỹ_t = E[y_t | data]          = D y + (I - D) H m_t + W D (y - H m_t)
B_t = d E[y_t | x_t] / d x_t = (I - D) H - W H          (rows: missing only)
C_t = Cov[y_t | x_t, y^o]    = (I - D) R (I - D) - W R (I - D)
E[y y^T] = ỹ ỹ^T + C_t + B_t P_t B_t^T
E[y x^T] = ỹ m_t^T + B_t P_t
E[x x^T] = m_t m_t^T + P_t
```

Checks: no mask → `W = 0`, `ỹ = y`, `B = 0`, `C = 0` (today's statistics).
Fully masked → `ỹ = H m_t`, `B = H`, `C = R`; the bin is then excluded by its
weight `e_t = 0` ([C3](shared-contracts.md#c3--masked-bin-semantics-m-step-statistics)).
Diagonal `R` → `W = 0`, and the `R` update reduces to
`(1/N) sum_t e_t [ D_t (r r^T + H P H^T) D_t + (I - D_t) R_old (I - D_t) ]`, the
classical missing-data form (masked channels keep `R_old` for that bin) —
this is the closed form phase 2's diagonal-`R` test checks.

**M-step with weights** `e_t = any(m_t)`, `N = sum_t e_t`:

```
H_new = ( sum_t e_t E[y x^T] ) ( sum_t e_t E[x x^T] )^{-1}
R_new = (1/N) sum_t e_t [ (ỹ_t - H_new m_t)(ỹ_t - H_new m_t)^T
                          + (B_t - H_new) P_t (B_t - H_new)^T + C_t ]
```

The residual form is a sum of PSD terms (PSD by construction, as today's
`measurement_cov_residual_form`, `kalman.py:1364-1396`); it equals
`E[(y - H x)(y - H x)^T]` expanded — verify: `ỹỹ^T + C + B P B^T - H(m ỹ^T + P B^T) - (ỹ m^T + B P)H^T + H(mm^T + P)H^T`
`= (ỹ - Hm)(ỹ - Hm)^T + C + (B - H) P (B - H)^T`.

**Per-bin code** (`kalman.py`, public — the switching M-step reuses it):

```python
def masked_observation_moments(
    obs_t: jax.Array,
    obs_mask_t: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
    smoother_mean_t: jax.Array,
    smoother_cov_t: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Expected observation, state coefficient and conditional covariance of one
    partially observed Gaussian observation given the smoothed state.

    Missing entries are imputed by their conditional mean given the observed
    entries and the state posterior under the current ``(H, R)``:
    ``E[y] = D y + (I - D) H m + W D (y - H m)`` with ``W = R_mo R_oo^{-1}``
    (embedded in full shape), ``B = (I - D) H - W H`` the coefficient of the
    state in the missing entries' conditional mean, and
    ``C = R_mm - R_mo R_oo^{-1} R_om`` their conditional covariance. With no
    missing entries ``E[y] = y``, ``B = 0``, ``C = 0``.

    Parameters
    ----------
    obs_t : jax.Array, shape (n_obs,)
        Observation; masked entries may hold any value.
    obs_mask_t : jax.Array, bool, shape (n_obs,)
    measurement_matrix : jax.Array, shape (n_obs, n_state)
    measurement_cov : jax.Array, shape (n_obs, n_obs)
    smoother_mean_t : jax.Array, shape (n_state,)
    smoother_cov_t : jax.Array, shape (n_state, n_state)

    Returns
    -------
    expected_obs : jax.Array, shape (n_obs,)
    state_coefficient : jax.Array, shape (n_obs, n_state)
    conditional_cov : jax.Array, shape (n_obs, n_obs)
    """
    dtype = measurement_matrix.dtype
    m = obs_mask_t.astype(dtype)
    not_m = 1.0 - m
    y0 = jnp.where(obs_mask_t, obs_t, 0.0)
    R = measurement_cov
    R_masked = R * (m[:, None] * m[None, :]) + jnp.diag(not_m)
    R_oo_inv = m[:, None] * psd_solve(R_masked, jnp.diag(m))      # R_oo^{-1}, embedded
    W = (not_m[:, None] * R) @ R_oo_inv                            # R_mo R_oo^{-1}
    predicted = measurement_matrix @ smoother_mean_t
    expected_obs = y0 + not_m * predicted + W @ (y0 - m * predicted)
    state_coefficient = not_m[:, None] * measurement_matrix - W @ measurement_matrix
    conditional_cov = (not_m[:, None] * R) * not_m[None, :] - W @ (R * not_m[None, :])
    return expected_obs, state_coefficient, symmetrize(conditional_cov)
```

(`smoother_cov_t` is unused inside — it enters the statistics below — but keeping
it in the signature lets the same `vmap` axes serve the statistics function.)

**Statistics over time** (single sequence; phase 4 adds the sequence axis in
[D4](#d4--multi-sequence-sufficient-statistics-two-pass)):

```python
def _masked_observation_statistics(obs, obs_mask, H, R, s_mean, s_cov):
    """(gamma, delta, n_bins, per-bin pieces) for the masked H / R update."""
    y_tilde, B, C = jax.vmap(
        masked_observation_moments, in_axes=(0, 0, None, None, 0, 0)
    )(obs, obs_mask, H, R, s_mean, s_cov)
    e = jnp.any(obs_mask, axis=1).astype(s_mean.dtype)              # (n_time,)
    gamma = jnp.einsum("t,tab->ab", e, s_cov) + jnp.einsum("t,ta,tb->ab", e, s_mean, s_mean)
    delta = jnp.einsum("t,ta,tb->ab", e, y_tilde, s_mean) + jnp.einsum(
        "t,tab,tbc->ac", e, B, s_cov
    )
    return gamma, delta, jnp.sum(e), (y_tilde, B, C, e)


def _masked_measurement_scatter(H, s_mean, s_cov, pieces):
    y_tilde, B, C, e = pieces
    resid = y_tilde - s_mean @ H.T
    b_minus_h = B - H[None]
    return symmetrize(
        jnp.einsum("t,ta,tb->ab", e, resid, resid)
        + jnp.einsum("t,tab,tbc,tdc->ad", e, b_minus_h, s_cov, b_minus_h)
        + jnp.einsum("t,tab->ab", e, C)
    )
```

In `_kalman_maximization_step` (`kalman.py:1554-1628`) the observation block
(`:1571-1581`) gets a static branch: `obs_mask is None` → the existing four
lines untouched; otherwise `gamma, delta, n_bins, pieces = _masked_observation_statistics(...)`,
`H = _gain_solve(gamma, delta.T).T`, `R = project_psd_relative(_masked_measurement_scatter(...) / n_bins, ...)`.
The transition block (`:1583-1619`) is unchanged. `kalman_maximization_step`
(`:1448-1551`) gains keyword-only `obs_mask=None` and passes it through (the
jitted inner function takes it as a pytree argument; `None` is static).

**Switching version** (`switching_kalman.py`). Vectorise the per-bin function
over the trailing state axis and over time:

```python
_masked_observation_moments_per_state = jax.vmap(
    masked_observation_moments,
    in_axes=(None, None, -1, -1, -1, -1),
    out_axes=(-1, -1, -1),
)  # H (m, n, K), R (m, m, K), mean (n, K), cov (n, n, K) -> (m, K), (m, n, K), (m, m, K)
_masked_observation_moments_over_time = jax.vmap(
    _masked_observation_moments_per_state, in_axes=(0, 0, None, None, 0, 0)
)  # -> (T, m, K), (T, m, n, K), (T, m, m, K)
```

Refactor `_switching_kalman_m_step_inner` (`switching_kalman.py:2150-2300`) so
its observation block is two helpers whose unmasked branches are **the existing
einsum expressions verbatim** (bit-identical):

```python
def _switching_observation_statistics(obs, means, covs, probs, obs_mask, H_prev, R_prev):
    """gamma_obs (n, n, K), delta (m, n, K), n_obs (K,), pieces for the scatter."""
    if obs_mask is None:
        weighted_cov_sum = jnp.sum(covs * probs[:, None, None], axis=0)          # :2210-2212
        gamma = weighted_cov_sum + weighted_sum_of_outer_products(means, means, probs)  # :2213-2217
        delta = weighted_sum_of_outer_products(obs[..., None], means, probs)     # :2218-2220
        return gamma, delta, probs.sum(axis=0), (obs, None, None, probs, weighted_cov_sum)
    y_tilde, B, C = _masked_observation_moments_over_time(obs, obs_mask, H_prev, R_prev, means, covs)
    w = probs * jnp.any(obs_mask, axis=1)[:, None]                                # e_t p_tj
    weighted_cov_sum = jnp.sum(covs * w[:, None, None], axis=0)
    gamma = weighted_cov_sum + weighted_sum_of_outer_products(means, means, w)
    delta = weighted_sum_of_outer_products(y_tilde, means, w) + jnp.einsum(
        "tmnj,tnpj,tj->mpj", B, covs, w
    )
    return gamma, delta, w.sum(axis=0), (y_tilde, B, C, w, weighted_cov_sum)


def _switching_measurement_scatter(H, means, covs, pieces):
    y_tilde, B, C, w, weighted_cov_sum = pieces
    predicted_obs = jnp.einsum("okj,tkj->toj", H, means)                        # :2238-2240
    if B is None:
        obs_residual = y_tilde[..., None] - predicted_obs                        # :2241
        return weighted_sum_of_outer_products(obs_residual, obs_residual, w) + jnp.einsum(
            "okj,klj,plj->opj", H, weighted_cov_sum, H                           # :2242-2246
        )
    obs_residual = y_tilde - predicted_obs
    b_minus_h = B - H[None]
    return (
        weighted_sum_of_outer_products(obs_residual, obs_residual, w)
        + jnp.einsum("tmnj,tnpj,tqpj,tj->mqj", b_minus_h, covs, b_minus_h, w)
        + jnp.einsum("tmpj,tj->mpj", C, w)
    )
```

`n_time` for the `R` normaliser (`:2206`, `:2249`) becomes the returned
`n_obs` (`probs.sum(0)` unmasked — identical). `gamma2` (`:2222-2229`) must keep
using the *unmasked* `probs` (dynamics are not masked): compute
`first_gamma` / `gamma2` from `probs`, not from the masked `gamma`. The jitted
inner function gets a static flag `use_obs_mask` and takes `H_prev`, `R_prev`
(the E-step's parameters, already available to the caller as
`previous_params`, `oscillator_models.py:1058-1063`). Callers with
`estimate_measurement_params=False` (point-process oscillator models,
`point_process_models.py:1351`, `switching_point_process.py:3073`) never pass a
mask and are unaffected.

## D3 — Masked Laplace-EKF update

`_point_process_laplace_update` (`point_process_kalman.py:927-1212`) and
`glm_laplace_update` (`:1333-1464`) reduce over neurons in exactly three places:
the score `J^T innovation` (`:1126`, `:1155`, `:1409`, `:1424`), the Fisher
information `J^T diag(w) J` (`:1138`, `:1158`, `:1413`, `:1428`) and the
log-likelihood sums (`:1099`, `:1188-1190`, `:1399`, `:1450`). A per-neuron
weight vector `mask_w = obs_mask_t.astype(dtype)` multiplies each:

```python
# in _point_process_laplace_update, new trailing keyword-only parameter
#     obs_mask_t: Array | None = None,
if obs_mask_t is not None:
    mask_w = obs_mask_t.astype(one_step_cov.dtype)
    spike_indicator_t = jnp.where(obs_mask_t, spike_indicator_t, 0.0)   # NaN-safe

def _weighted(v):            # identity when there is no mask -> old code path
    return v if obs_mask_t is None else mask_w * v

# _neg_log_posterior:   log_lik = jnp.sum(_weighted(spike_indicator_t * jnp.log(cond_int) - cond_int))
# _fisher_step_at:      innovation = _weighted(spike_indicator_t - conditional_intensity)
#                       fisher_info = jacobian.T @ (_weighted(conditional_intensity)[:, None] * jacobian)
# single-step branch:   same two substitutions
# log_likelihood:       jnp.sum(_weighted(jax.scipy.stats.poisson.logpmf(spike_indicator_t, conditional_intensity_mode)))
```

`glm_laplace_update` consumes the shared
[`GLMFamily.loglik_per_obs` contract](../glm-families-nb-zig/shared-contracts.md#c1-glmfamily-extended-contract).
If this plan lands first, append that optional field and supply it for Poisson
and Bernoulli; the GLM and WoLF plans reuse the same field. Do not add `weights=`
to the scalar callbacks. With no mask, retain their existing calls unchanged.
With a mask, sanitize `y` with `where` before evaluating any family function:

```python
def _masked_glm_terms(y, eta, family, obs_mask):
    """Score/information per predictor and normalized scalar masked likelihood."""
    if family.loglik_per_obs is None:
        raise ValueError("obs_mask requires family.loglik_per_obs")
    n_obs, n_eta = y.shape[0], eta.shape[0]
    if n_obs == 0 or n_eta % n_obs:
        raise ValueError("predictors must use the predictor-major observation layout")
    y_safe = jnp.where(obs_mask, y, 0.0)
    mu = family.mean(eta)
    score = y_safe - mu if family.score is None else family.score(y_safe, eta, mu)
    predictor_mask = jnp.tile(obs_mask, n_eta // n_obs)
    score = jnp.where(predictor_mask, score, 0.0)
    weight = jnp.where(predictor_mask, family.fisher_weight(eta, mu), 0.0)
    per_obs = family.loglik_per_obs(y_safe, eta, mu)  # (n_obs,), NOT (n_eta,)
    loglik = jnp.sum(jnp.where(obs_mask, per_obs, 0.0))
    return score, weight, loglik
```

The masked line search and evidence both use this normalized likelihood; its
latent gradient equals the masked score. The unmasked path keeps the old
plugin/normalized distinction. Validate the optional callback at the public
boundary and all unmasked observations through `family.validate_observations`
when supplied, otherwise the count validator. In particular, ZIG has N mask
entries, 2N score/information entries, and continuous observations; masking
must neither compare their lengths for equality nor apply count validation.
Add any absent optional `score`/`validate_observations` fields with the GLM
plan's defaults if implementing this integration before that plan.

Correctness of the fully masked bin: `innovation = 0` and `fisher_info = 0`, so
`post_prec = prior_precision`, `posterior_mean = one_step_mean` (the Fisher step
is `0`; with line search the gradient is zero so the iterate is kept),
`posterior_cov = one_step_cov` (up to the `psd_cholesky` shift round trip:
`cho_solve(chol(P^{-1}), I)` — assert with `rtol=1e-10`, not bitwise), and the
Laplace normaliser gives `-1/2 logdet_prior + 1/2 logdet_post = 0`, `quad = 0`,
`logpmf` sum `0` → LL contribution exactly 0 ([C2](shared-contracts.md#c2--masked-bin-semantics-e-step-and-log-likelihood)).

Block-diagonal path: `_block_diagonal_forward_core` (`:1869-1979`) vmaps
`_run_one_neuron` with `in_axes=(0, 0, 0, 0, 1)` (`:1977`); the mask joins as a
sixth argument with `in_axes` `1` (or `None` when absent), sliced per bin in the
scan alongside `y_t` (`:1917`) and passed as a 1-element `obs_mask_t`.
`_block_diagonal_smoother_core` (`:2033-2122`) forwards it. The per-neuron
independence means masking neuron `j` at bin `t` touches only neuron `j`'s
scan — no cross-neuron effect, which the block-vs-dense equivalence test checks.

## D4 — Multi-sequence sufficient statistics (two-pass)

**Why two passes.** The library's `R` and `Q` updates use centred residual forms
evaluated at the *installed* `H` / `A` (`kalman.py:1364-1445`,
`switching_kalman.py:2237-2269`), chosen for PSD-by-construction. With several
sequences, `H_new` / `A_new` depend on all sequences, so per-sequence residual
scatters can only be formed after the solve:

1. **Pass 1** — per sequence, the moment statistics (`gamma`, `delta`, `gamma1`,
   `beta`, `gamma2`, weights, initial-state moments); `vmap` over sequences,
   `jax.tree.map(sum over axis 0)`; solve `H_new`, `A_new` from the sums.
2. **Pass 2** — per sequence, the residual scatters at `H_new` / `A_new`;
   `vmap`, sum, divide by the summed weights; PSD floor as today.

Both passes are einsums over already-computed posteriors (no scans), so the cost
is negligible next to the E-step.

**Weights** come from the mask ([C6](shared-contracts.md#c6--multi-sequence-inputs-padding-and-sequence-lengths)):
`e[s, t]` for observation statistics, `tw[s, t]` (transition exists) for every
`t -> t + 1` statistic, `1` for the initial state.

**Gaussian `kalman_maximization_step` (prior on `x_0`).** Per sequence `s`,
with `m0_s, P0_s, C01_s = smooth_initial_state_with_cross_cov(prior, mean[s, 0], cov[s, 0])`
(`kalman.py:1320-1361`, vmapped over `s`):

```
trans_mean_s = [m0_s, mean_s]                        (T+1, n)
trans_cov_s  = [P0_s, cov_s]                         (T+1, n, n)
trans_cross_s= [C01_s, cross_s]                      (T, n, n)
w_s          = [1, tw_s]                             (T,)   weight of transition t-1 -> t, t = 1..T
gamma1 += sum_t w_s[t] (trans_cov_s[t-1] + m m^T)_{t-1}
beta   += sum_t w_s[t] (trans_cross_s[t-1] + m_{t-1} m_t^T)^T
n_trans += sum_t w_s[t]
Q scatter (pass 2) += sum_t w_s[t] [ (m_t - A m_{t-1})(.)^T + P_t - A C - C^T A^T + A P_{t-1} A^T ]
init_mean = mean_s m0_s ;  init_cov = mean_s [ P0_s + (m0_s - init_mean)(m0_s - init_mean)^T ]
```

The initial-state update is the exact maximiser for a shared `N(m_0, P_0)` prior
across sequences (the between-sequence spread term is what makes `P_0` the
maximiser rather than the mean covariance). `process_cov_residual_form`
(`kalman.py:1399-1445`) is refactored into `_process_scatter(...)` (unnormalised,
exactly the expression at `:1437-1444`) plus the division, so the single-sequence
path stays bit-identical and the batched path sums scatters.

**Switching `switching_kalman_maximization_step` (prior on `x_1`).** Per
sequence: `_switching_observation_statistics` ([D2](#d2--masked-gaussian-m-step-imputation-form))
with weights `e_s[t] p_s[t, j]`; `compute_transition_sufficient_stats`
(`switching_kalman.py:3087-3184`) with
`smoother_joint_discrete_state_prob[s] * tw_s[:, None, None]` (it is linear in the
joint probabilities, so pre-multiplying is exact); `gamma2` from
`probs_s[1:] * tw_s[:, None]`; expected transition counts likewise weighted;
initial state per discrete state `j`:

```
w_sj = probs_s[0, j]
init_mean_j = sum_s w_sj mean_s[0, :, j] / sum_s w_sj
init_cov_j  = sum_s w_sj [ cov_s[0, :, :, j] + (mean_s[0, :, j] - init_mean_j)(.)^T ] / sum_s w_sj
pi_0        = mean_s probs_s[0]
```

Implement as `_switching_kalman_m_step_batched(...)`, dispatching from
`switching_kalman_maximization_step` on `state_cond_smoother_means.ndim == 4`;
the 3-D path calls the existing `_switching_kalman_m_step_inner` unchanged.
`compute_transition_sufficient_stats` and
`compute_process_covariance_sufficient_stats` (`:3187-3241`) gain
`transition_weight: jax.Array | None = None` and accept a leading sequence axis
(detected from `ndim`; batched inputs are `vmap`ped then summed); their existing
unbatched, unweighted branch is untouched. This is what
`DirectedInfluenceModel._m_step_reparameterized`
(`oscillator_models.py:2089-2097`) and
`CorrelatedNoiseModel._m_step_constrained_process_covariance` (`:1648-1658`)
need.

**Dynamics-only `dynamics_only_m_step` (point process, prior on `x_0`).** Same
as the Gaussian transition block without `H` / `R`; `_dynamics_only_m_step`
(`point_process_kalman.py:2743-2817`) dispatches on `smoother_mean.ndim == 3`
to `_dynamics_only_m_step_batched(..., transition_weight)`. `PlaceFieldModel._m_step`
(`place_field_model.py:1026-1137`) computes the same sums inline from
`BlockDiagonalCovariance` containers; its batched version uses
[D7](#d7--block-diagonal-covariances-with-a-sequence-axis).

## D5 — Batching strategy and measured memory

Decision: batch with `jax.vmap` over the leading sequence axis inside each
model's E-step ([C9](shared-contracts.md#c9--where-batching-lives-not-in-run_em)),
through `sequences.map_over_sequences` so the strategy has one switch.

Measured in this session (jax 0.10.2, CPU, float64,
`jax.jit(...).lower(...).compile().memory_analysis()`; timings after warm-up):

| E-step | Size | vmap temp / output | lax.map temp / output | Runtime vmap / lax.map |
| --- | --- | --- | --- | --- |
| `_kalman_filter_impl` | S=20, T=2000, n=4, m=4 | 1.5 MB / 6.4 MB | 0.4 MB / 6.4 MB | 8 ms / 97 ms |
| `switching_kalman_filter` + GPB1 smoother | S=20, T=2000, n=4, m=4, K=2 | 54 MB / 52 MB | (not measured) | — |

Output size is set by the stored posteriors (`S T n²` floats per covariance
sequence; the switching filter also emits `pair_cond_filter_cov`, `S T n² K²`
floats = 20.5 MB of the 52 MB above) and is identical for both strategies;
`lax.map` only trims temporaries, at ~12x wall time on CPU. So `vmap` is the
default. The phase-4 and phase-5 smoke tests re-measure at the executor's
realistic sizes with the same `memory_analysis()` call and record the numbers;
the rule for flipping `_SEQUENCE_MAP` to `"map"` is temp > 4x output *and*
total > ~50% of device memory. For the block-diagonal point-process path the
stored blocks are `3 S n_neurons T nb²` floats (forward, smoothed, cross): at
S=10, T=2000, n_neurons=20, nb=16 that is 2.5 GB — the same as a single 20 000-bin
session today, i.e. multi-sequence changes nothing about total posterior
storage, only about how it is laid out.

## D6 — Oracle extensions for masks and sequences

All in `src/state_space_practice/tests/oracles.py`; plain NumPy, no library
code, as today.

**Dense posterior with masked entries.** `lgssm_dense_posterior`
(`oracles.py:198-272`) gains `obs_mask: np.ndarray | None = None`. Conditioning
on observed entries only is a row/column selection of the joint built by
`lgssm_joint_prior` (`:129-195`):

```python
def _condition_subset(mean_x, cov_x, mean_y, cov_y, cov_xy, y, keep):
    """Moments of x | y[keep]; the prior when nothing is kept."""
    if not keep.any():
        return mean_x.copy(), 0.5 * (cov_x + cov_x.T)
    return _condition(
        mean_x, cov_x, mean_y[keep], cov_y[np.ix_(keep, keep)], cov_xy[:, keep], y[keep]
    )

# in lgssm_dense_posterior, after y = obs.reshape(-1):
keep = np.ones(y.size, bool) if obs_mask is None else np.asarray(obs_mask, bool).reshape(-1)
y = np.where(keep, y, 0.0)                     # masked entries may be NaN
joint_mean, joint_cov = _condition_subset(mean_x, cov_x, mean_y, cov_y, cov_xy, y, keep)
# filtered loop (:244-257): kk = keep[:k]
fm, fc = _condition_subset(mean_x[sl], cov_x[sl, sl], mean_y[:k], cov_y[:k, :k], cov_xy[sl, :k], y[:k], kk)
filtered_ll[t - 1] = (
    gaussian_logpdf(y[:k][kk], mean_y[:k][kk], cov_y[:k, :k][np.ix_(kk, kk)]) if kk.any() else 0.0
)
```

`switching_lgssm_exact_posterior` (`:473-646`) passes `obs_mask` through to each
per-path `lgssm_dense_posterior` call (`:528-537`); nothing else changes, so the
exact switching filter with masks comes for free.

**Augmented-Q oracle for the masked M-step.** The masked M-step maximises
`Q_aug(θ) = E[log p(x_{0:T}, y_{1:T} | θ) | y^o, θ_old]` with the expectation
over `(x, y^m)` under `θ_old`. Build the joint posterior of `z = (x_{0:T}, y_{1:T})`
given `y^o`:

```python
def lgssm_dense_posterior_with_observations(init_mean, init_cov, obs, A, Q, H, R, obs_mask):
    """Posterior mean / cov of the stacked (x_{0:T}, y_{1:T}) given the observed entries."""
    n_time, m = obs.shape
    mean_x, cov_x, mean_y, cov_y, cov_xy = lgssm_joint_prior(init_mean, init_cov, A, Q, H, R, n_time)
    keep = np.asarray(obs_mask, bool).reshape(-1)
    y = np.where(keep, obs.reshape(-1), 0.0)
    mean_z = np.concatenate([mean_x, mean_y])
    cov_z = np.block([[cov_x, cov_xy], [cov_xy.T, cov_y]])
    n_x = mean_x.size
    cov_zy = cov_z[:, n_x:][:, keep]
    return _condition(mean_z, cov_z, mean_y[keep], cov_y[np.ix_(keep, keep)], cov_zy, y[keep])
```

Then `lgssm_expected_complete_log_likelihood` (`:294-353`) gets a sibling
`lgssm_expected_complete_log_likelihood_augmented(z_mean, z_cov, n_time, n, m, ...)`
whose observation term (`:349-352` today uses the fixed `obs`) becomes, with
`Szz` second moments read from the `(x, y)` blocks,
`E[(y_t - H x_t)(.)^T] = S_yy - H S_xy^T - S_xy H^T + H S_xx H^T` for the bins
with `e_t = 1`, and the dynamics terms unchanged. The phase-2 test asserts the
masked `kalman_maximization_step` zeroes the finite-difference gradient of this
`Q_aug` (`_fd_gradient`, `test_oracle_kalman.py:331-351`) and that the exact
observed-data LL from the masked `lgssm_dense_posterior` is non-decreasing over
EM iterations (pattern of `test_full_em_matches_dense_oracle_q_monotonically`,
`:454-473`).

**Sequences.** Exact statistics are additive: for the switching oracle, run
`switching_path_sufficient_statistics` (`:689-752`) per sequence and sum the
dicts; `switching_q_from_statistics` (`:755-818`) then evaluates the summed
`Q`, which is what the batched M-step must maximise (phase 4's exactness test).
For the Gaussian oracle, `lgssm_expected_complete_log_likelihood` per sequence,
summed. Padding is tested separately by comparing padded vs unpadded inputs.

## D7 — Block-diagonal covariances with a sequence axis

`BlockDiagonalCovariance` (`point_process_kalman.py:216-368`) stores
`(n_neurons, n_time, nb, nb)` blocks and requires `ndim == 4` (`:254-258`). A
`vmap`ped block E-step returns `(n_sequences, n_neurons, n_time, nb, nb)`.
Extend the container rather than materialise anything:

- accept `ndim in (4, 5)`; `n_sequences` property (`None` when 4-D);
- `shape` reports `(n_sequences, n_time, n_state, n_state)` when batched;
- `sum(axis=0)` on a batched container raises with a message pointing at
  `weighted_time_sum`; new
  `weighted_time_sum(weights)` — `weights` `(n_time,)` or `(n_sequences, n_time)`
  float — returns the dense `(n_state, n_state)` `sum_{s,t} w[s,t] blocks[s,:,t]`
  assembled block-diagonally (this is the reduction the batched M-step needs:
  transition-weighted sums); on a 4-D container with `weights=None` it equals
  `sum(axis=0)` (assert this in the container tests);
- `at_time(t, sequence=None)` / `__getitem__` with a `(sequence, t)` tuple;
- `sequence(s)` returns the 4-D container of one sequence (a view of `blocks[s]`);
- `neuron_blocks(j, time_slice, sequence=None)`, `diagonal(sequence=None)`.

`PlaceFieldModel._m_step` batched branch (`place_field_model.py:1026-1137`):

```python
tw = transition_weights(obs_mask)                    # (S, T-1)
m0, P0, C01 = jax.vmap(smooth_initial_state_with_cross_cov, in_axes=(None, 0, 0))(prior, sm[:, 0], sc.sequence_first())
# sc.sequence_first(): (S, n_state, n_state) dense first-bin covariances, one small matrix per sequence
means = jnp.concatenate([m0[:, None], sm], axis=1)   # (S, T+1, n)
w = jnp.concatenate([jnp.ones((S, 1)), tw], axis=1)  # (S, T) weight of transition t-1 -> t
prev_w = jnp.concatenate([tw, jnp.zeros((S, 1), dtype=tw.dtype)], axis=1)  # (S, T)
sum_prev_cov = sc.weighted_time_sum(prev_w) + P0.sum(0)   # x_0 .. x_{T-1}; x_T has weight zero
sum_next_cov = sc.weighted_time_sum(w)                                                    # x_1 .. x_T (weighted)
sum_cross_cov = scc.weighted_time_sum(tw) + C01.sum(0)
gamma1 = sum_prev_cov + jnp.einsum("st,sta,stb->ab", w, means[:, :-1], means[:, :-1])
beta = (sum_cross_cov + jnp.einsum("st,sta,stb->ab", w, means[:, :-1], means[:, 1:])).T
```

then `A_new`, the residual `Q` scatter with the same weights divided by
`w.sum()`, the diagonal / isotropic constraint and floors as today
(`:1120-1127`), and the initial state averaged over sequences as in
[D4](#d4--multi-sequence-sufficient-statistics-two-pass). `sc.weighted_time_sum(w)`
for the `x_1..x_T` sum must weight bin `t` by the existence of the transition
into it (`w[:, t]`), which is `bin_weight` — identical for `t < len_s`.

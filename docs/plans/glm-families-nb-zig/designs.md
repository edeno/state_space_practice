# Designs — family math, code and rationale

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Each section is one component. Phases link here by anchor; this file does not
repeat the phase task lists. Notation: `eta` is the linear predictor, `mu` the
inverse-link value the family stores in its `mean` slot, `y` the observation,
`J = d eta / d x` the `(n_eta, n_state)` Jacobian the update already computes.

Contents

1. [Why the update needs a `score` hook](#1-why-the-update-needs-a-score-hook)
2. [Negative-binomial family](#2-negative-binomial-family)
3. [Zero-inflated-gamma family](#3-zero-inflated-gamma-family)
4. [Threading `family` through the filter and smoother](#4-threading-family-through-the-filter-and-smoother)
5. [`PointProcessModel` family plumbing](#5-pointprocessmodel-family-plumbing)
6. [Simulators](#6-simulators)
7. [Extending the grid-quadrature oracle and the calibration harness](#7-extending-the-grid-quadrature-oracle-and-the-calibration-harness)

---

## 1. Why the update needs a `score` hook

`glm_laplace_update` (`src/state_space_practice/point_process_kalman.py:1333-1464`)
hard-codes the likelihood score as `jacobian.T @ (observations - mu)` at lines
1407-1409 and 1422-1424. That is the score of an exponential family **with its
canonical link** (`d loglik / d eta = y - mu`), which is true for Poisson-log and
Bernoulli-logit and is what the `GLMFamily` docstring (`:1215-1234`) assumes.

Neither new family is canonical in that sense:

* Negative binomial with a log link: `d loglik / d eta = (y - mu) / (1 + mu / r)`
  (derived in §2). `y - mu` is wrong by the shrinkage factor `r / (r + mu)`.
* Zero-inflated gamma: the score of the log-scale predictor is
  `1[y > 0] ((y - loc) / theta - k)`, which is not `y - mu` for any choice of
  `mu` (§3).

So the minimal generalisation that keeps the existing path byte-identical is an
**optional per-predictor score** on the family:

```python
class GLMFamily(NamedTuple):
    mean: Callable[[Array], Array]
    fisher_weight: Callable[[Array, Array], Array]
    loglik_plugin: Callable[[Array, Array, Array], Array]
    loglik_normalized: Callable[[Array, Array, Array], Array]
    #: Per-predictor score ``d loglik / d eta`` as ``(y, eta, mu) -> (n_eta,)``.
    #: ``None`` (the default, and the canonical-link case) means ``y - mu``.
    score: Callable[[Array, Array, Array], Array] | None = None
    #: Normalized log likelihood per observation (n_obs,), shared with masks/WoLF.
    loglik_per_obs: Callable[[Array, Array, Array], Array] | None = None
```

and, inside `glm_laplace_update`, one Python-level branch used at both call sites
(the two `innovation = observations - mu` lines):

```python
    def _score(eta: Array, mu: Array) -> Array:
        # Canonical link: score = y - mu. Non-canonical families supply it.
        if family.score is None:
            return observations - mu
        return family.score(observations, eta, mu)
```

Because `family` is a Python object (static under `jit`), `family.score is None`
is resolved at trace time and the traced program for `poisson_family` /
`BERNOULLI_LOGIT_FAMILY` is unchanged — the existing parity test
(`tests/test_glm_laplace.py:36-68`) and the `point_process_glm` golden case
(`tests/test_em_golden_regression.py:239-248`) pin that.

The Fisher information stays `J' diag(w) J` with `w = family.fisher_weight`;
for the new families `w` is the **expected** negative Hessian per predictor
(derived below), so the posterior precision remains PSD by construction and the
update remains Fisher scoring, not Newton. A general check that holds for every
family, canonical or not, is `w = E_y[score^2]` (the information identity); the
canonical-only identity `w = d mean / d eta` tested at
`tests/test_glm_laplace.py:196-214` must **not** be extended to the new
families.

`n_eta` need not equal `n_obs`: nothing in the update compares their lengths
except the (now family-owned) score, so a family may map `n_obs` observations to
`n_eta = k * n_obs` stacked predictors (§3).

**Masking interface shared with the masks and WoLF plans.** All built-in families
provide `loglik_per_obs(y, eta, mu)` of shape `(n_obs,)`; its sum equals
`loglik_normalized`. Keep the existing three-argument scalar likelihood callbacks
unchanged. With a mask, use the sum of the selected per-observation normalized
terms for both line search and evidence (the omitted constants are independent
of the latent state). With no mask, call the original scalar callbacks so their
traced expressions remain unchanged. A custom family without `loglik_per_obs`
continues to work unmasked, but a masked call raises `ValueError` at the wrapper.
Poisson supplies `poisson.logpmf(y, mu)`; Bernoulli supplies the existing clipped
logit expression without its reduction. Predictor-major layout means an
observation mask `(N,)` is tiled `k` times for the score and Fisher vector `(k*N,)`;
likelihood terms use the original `(N,)` mask. Validate zero-filled observations
with the selected family's validator, so masked ZIG data remain continuous.

---

## 2. Negative-binomial family

### 2.1 Model and derivation

For neuron `n` in a bin of width `dt`, `eta = log rate (Hz)`,
`mu = exp(eta) * dt` (with the same overflow clip as Poisson, via
`_safe_expected_count`, `point_process_kalman.py:589-610`) and dispersion
`r > 0` (scalar or per neuron):

```
y ~ NB(mean mu, dispersion r)          E[y] = mu,   Var[y] = mu + mu^2 / r
p(y) = Gamma(y + r) / (Gamma(r) y!) * (r / (r + mu))^r * (mu / (r + mu))^y
log p(y) = [lgamma(y + r) - lgamma(r)] - lgamma(y + 1)
           - r * log1p(mu / r) + y * (log mu - log(r + mu))
```

With `d mu / d eta = mu`:

```
d log p / d mu   = y / mu - (y + r) / (r + mu)
d log p / d eta  = y - mu (y + r) / (r + mu) = r (y - mu) / (r + mu) = (y - mu) / (1 + mu / r)
-d^2 log p / d eta^2 (observed) = (y + r) * mu r / (r + mu)^2
E_y[ -d^2 log p / d eta^2 ]     = (mu + r) * mu r / (r + mu)^2 = mu r / (r + mu) = mu / (1 + mu / r)
Var_y[score] = Var[y] r^2 / (r + mu)^2 = mu (1 + mu / r) r^2 / (r + mu)^2 = mu / (1 + mu / r)   (information identity holds)
```

So `score = (y - mu) / (1 + mu / r)` and `fisher_weight w = mu / (1 + mu / r)`.
As `r -> inf` both tend to the Poisson `y - mu` and `mu`, and `log p` tends to
the Poisson log-pmf. The observed information `(y + r) mu r / (r + mu)^2` is
also non-negative here (so Newton would be PSD too), but the repo's updates use
the expected information throughout; the Laplace covariance returned for NB is
therefore the **Fisher-scoring** covariance (inverse of prior precision plus
expected information at the mode), which coincides with Poisson's as
`r -> inf`. Say exactly this in the family docstring.

`loglik_plugin` (line search only, `x`-dependent part): drop `lgamma(y + r) -
lgamma(r) - lgamma(y + 1) + r log r`, keep

```
plugin = sum_n [ y log mu - (y + r) log(r + mu) ]        (d/d eta = score above)
```

### 2.2 Stable `lgamma(y + r) - lgamma(r)`

`jax.scipy.stats.nbinom.logpmf` and a naive `gammaln(y + r) - gammaln(r)` lose
`eps * |lgamma(r)|` absolute accuracy: `~2e-7` at `r = 1e8`, `~2e-5` at
`r = 1e10`. The `r -> inf` Poisson parity test (1e-8 on the marginal LL)
needs `r ~ 1e10` to make the *model* gap `O(1/r)` negligible, and the
digamma-difference gradient `psi(y + r) - psi(r)` suffers the same cancellation
when SGD drives `r` large on Poisson-like data. Use Stirling for large `r`:

```
lgamma(z) = (z - 1/2) log z - z + log(2 pi)/2 + 1/(12 z) - 1/(360 z^3) + ...
lgamma(y + r) - lgamma(r)
    = (r - 1/2) log1p(y / r) + y log(y + r) - y + 1/(12 (y + r)) - 1/(12 r) + O(y / r^4)
```

```python
_STIRLING_SWITCH = 1e3  # both branches are accurate to ~1e-12 at the switch


def _log_gamma_ratio(y: Array, r: Array) -> Array:
    """``lgamma(y + r) - lgamma(r)`` without the large-``r`` cancellation.

    For ``r <= 1e3`` the direct difference is accurate to ``~eps * lgamma(r)``
    (``< 2e-12``); above it the Stirling form is used, whose truncation error is
    below ``1e-12``. Both branches are finite everywhere on ``y >= 0, r > 0``,
    so ``jnp.where`` is safe under differentiation.
    """
    direct = jax.scipy.special.gammaln(y + r) - jax.scipy.special.gammaln(r)
    stirling = (
        (r - 0.5) * jnp.log1p(y / r)
        + y * jnp.log(y + r)
        - y
        + 1.0 / (12.0 * (y + r))
        - 1.0 / (12.0 * r)
    )
    return jnp.where(r > _STIRLING_SWITCH, stirling, direct)
```

Reference for tests: for integer `y`, `lgamma(y + r) - lgamma(r) =
sum_{i=0}^{y-1} log(r + i)` (exact, `math.fsum` in Python floats) and its
`r`-derivative is `sum_{i<y} 1/(r + i)`.

### 2.3 Code

```python
def negative_binomial_family(
    dt: float, dispersion: ArrayLike, max_log_count: float = 20.0
) -> GLMFamily:
    """Negative-binomial family with log link for over-dispersed counts.

    ``y ~ NB(mean = exp(eta) * dt, dispersion = r)`` with
    ``Var[y] = mu + mu**2 / r``; ``r -> inf`` recovers ``poisson_family``.
    The log link is not the NB's canonical link, so the update is Fisher
    scoring: ``score = (y - mu) / (1 + mu / r)`` and the weight is the expected
    information ``mu / (1 + mu / r)``; the Laplace covariance is the
    Fisher-scoring covariance, which coincides with Poisson's as ``r -> inf``.

    Parameters
    ----------
    dt : float
        Bin width in seconds (``eta`` is a log rate in Hz).
    dispersion : ArrayLike, shape () or (n_obs,)
        Dispersion ``r > 0`` (shared or per neuron). May be a tracer (SGD).
    max_log_count : float, default 20.0
        Overflow ceiling on ``log(rate * dt)``, as in :func:`poisson_family`.

    Returns a fresh ``GLMFamily`` each call; hoist it out of ``jit``/``scan``
    loops (see :func:`poisson_family`).
    """
    validate_scalar(dt, "dt", positive=True)
    r = jnp.asarray(dispersion)
    if not contains_tracer(r):
        if r.ndim > 1:
            raise ValueError(f"dispersion must be a scalar or 1-D, got shape {r.shape}.")
        if not bool(jnp.all(jnp.isfinite(r)) and jnp.all(r > 0)):
            raise ValueError("dispersion must be finite and strictly positive.")

    def mean(eta: Array) -> Array:
        return _safe_expected_count(eta, dt, max_log_count=max_log_count)

    def _shrink(mu: Array) -> Array:
        return r / (r + mu)  # == 1 / (1 + mu / r), no cancellation for large r

    def score(y: Array, _eta: Array, mu: Array) -> Array:
        return (y - mu) * _shrink(mu)

    def fisher_weight(_eta: Array, mu: Array) -> Array:
        return mu * _shrink(mu)

    def loglik_plugin(y: Array, _eta: Array, mu: Array) -> Array:
        # x-dependent part only; mu >= exp(-20) so log(mu) is finite.
        return jnp.sum(y * jnp.log(mu) - (y + r) * jnp.log(r + mu))

    def loglik_per_obs(y: Array, _eta: Array, mu: Array) -> Array:
        return (
            _log_gamma_ratio(y, r)
            - jax.scipy.special.gammaln(y + 1.0)
            - r * jnp.log1p(mu / r)
            + y * (jnp.log(mu) - jnp.log(r + mu))
        )

    def loglik_normalized(y: Array, eta: Array, mu: Array) -> Array:
        return jnp.sum(loglik_per_obs(y, eta, mu))

    return GLMFamily(mean, fisher_weight, loglik_plugin, loglik_normalized,
                     score=score, loglik_per_obs=loglik_per_obs)
```

`contains_tracer` is already imported in `point_process_kalman.py:61`;
`validate_scalar` at `:68`. Under SGD `dispersion` is a tracer and the
host-side checks are skipped (same policy as `_validate_public_inputs`,
`:96-145`).

### 2.4 Why dispersion is not updated by EM

The M-step objective `Q(r) = E_{x|y}[ sum_{t,n} log NB(y_tn | mu_tn(x_t), r) ]`
contains `E[log(r + mu(x))]` and `E[lgamma(y + r) - lgamma(r)]`-type terms under
the Gaussian smoother posterior; there is no closed form. A workable route is a
per-neuron 1-D Newton on `Q(r)` with the expectation over `eta_tn ~ N(z_tn' m_t,
z_tn' P_t z_tn)` taken by Gauss-Hermite quadrature (scalar Gaussian per bin, ~20
nodes). It is left as an Open Question ([overview](overview.md#open-questions));
the shipped answer is SGD on the marginal likelihood via a `POSITIVE`
transform (§5), which needs no M-step.

---

## 3. Zero-inflated-gamma family

### 3.1 The model (Wei et al. 2020)

Wei et al. (2020, *Neurons, Behavior, Data analysis, and Theory* 3(2);
bioRxiv 10.1101/637652; arXiv 2006.03737), eq. (2) and Sec. 4.3, model a
deconvolved trace `s` of neuron `i` at time `t` as

```
s_ti ~ (1 - q_ti) * delta(0) + q_ti * gamma(s_ti; k_ti, a_ti, loc_i)
```

with `q` the probability of a non-zero response, `a` the gamma **scale**, `k`
the gamma shape and `loc_i` the gamma location, "fixed as the minimum spike
size `s_min`". In their encoding fits `a(theta)` and `q(theta)` are functions
of the covariate while "we fix the shape parameter `k` to be a constant for
individual neurons (i.e., `k` is neuron-dependent but not `theta`-dependent)";
the mean is `q (k a + loc)` and the variance `q k a^2 + q (k a + loc)^2 (1 - q)`.
Their objective (eq. 3) is the sum of `log(1 - q) 1[s = 0]` and
`(log q + (k - 1) log(s - s_min) - (s - s_min)/a - k log a - log Gamma(k)) 1[s > s_min]`.

This plan follows that exactly: per neuron, **two linear predictors in the
latent** — `eta_q = logit q` and `eta_s = log theta` (scale) — plus fixed
`shape k_n > 0` and `loc_n >= 0`.

### 3.2 Design alternatives

| | Design | Verdict |
|---|---|---|
| (a) | Generalise `GLMFamily` to `k` predictors per observation: `mean` takes `(n_obs, k)`, `fisher_weight` returns `(n_obs, k, k)` blocks, Jacobian `(n_obs*k, n_state)`. | Correct but more than ZIG needs (§3.3 shows the expected information is **diagonal** across the two predictors), and the `(n_obs, k, k)` weight path would be a second code path next to `diag(w)`. |
| (b) | Separate `zig_laplace_update` sharing `_fisher_scoring_line_search` (`:736-856`). | Duplicates the prior factorisation, the single-step branch, the Laplace normaliser and the failure counting of `glm_laplace_update` (~80 lines) for one family. |
| **(a′) chosen** | Keep one flat predictor vector `eta` of length `n_eta = 2 n_obs` (layout in [shared-contracts C4](shared-contracts.md#c4-stacked-predictor-layout)); the family's `score` and `fisher_weight` return `(2 n_obs,)`; the Jacobian is `(2 n_obs, n_state)` automatically. Needs only the Phase-1 `score` hook plus an observation validator. | Zero change to the update; `k = 1` path untouched; exact because the expected cross-information is identically zero. If a future family has non-zero expected cross-information between its predictors (e.g. a latent-dependent gamma shape, §3.6), (a) becomes necessary — that is the revisit trigger. |

### 3.3 Derivation

Per neuron, `z = 1[y > 0]`, `q = sigmoid(eta_q)`, `theta = exp(eta_s)`,
`e = y - loc` (only used when `z = 1`):

```
loglik = (1 - z) log(1 - q) + z [ log q + (k - 1) log e - e / theta - k log theta - lgamma(k) ]

d/d eta_q  = (1 - z)(-q) + z (1 - q)      = z - q
d/d eta_s  = z ( e / theta - k )                       (d/d eta_s of -e exp(-eta_s) - k eta_s)

-d^2/d eta_q^2      = q (1 - q)
-d^2/d eta_s^2      = z e / theta
-d^2/d eta_q d eta_s = 0                                (the eta_q score does not involve eta_s)

E[z] = q ;   E[z e / theta] = q * E[e | z = 1] / theta = q * k theta / theta = q k
=> expected information per neuron = diag( q (1 - q),  q k ),  cross term exactly 0
```

Across neurons the observations are conditionally independent, so over the
stacked `2 n_obs` predictors the expected information is `diag(w)` with
`w = [q_n (1 - q_n)]_n ++ [q_n k_n]_n`, all entries `>= 0`: `J' diag(w) J` is PSD
by construction and analytic. The information identity also holds:
`E[(z - q)^2] = q(1 - q)` and `E[z^2 (e/theta - k)^2] = q * Var[e/theta | z=1] = q k`.

Plugin (x-dependent) log-likelihood, with `log q = eta_q - softplus(eta_q)`,
`log(1 - q) = -softplus(eta_q)`, `log theta = eta_s`:

```
plugin = sum_n [ (1 - z) (-softplus(eta_q)) + z ( eta_q - softplus(eta_q) - e exp(-eta_s) - k eta_s ) ]
normalized = plugin + sum_n z [ (k - 1) log e - lgamma(k) ]
```

`d plugin / d eta_q = z - q` and `d plugin / d eta_s = z (e/theta - k)`: the
line search objective is consistent with the score.

### 3.4 Code

```python
_ZIG_LOG_SCALE_CLIP = 20.0  # theta = exp(eta_s) overflow guard; same asymmetry note as _BERNOULLI_ETA_CLIP


def zero_inflated_gamma_family(shape: ArrayLike, loc: ArrayLike = 0.0) -> GLMFamily:
    """Zero-inflated gamma family for deconvolved calcium traces (Wei et al. 2020).

    ``y = 0`` with probability ``1 - q``; otherwise ``y - loc ~ Gamma(shape=k,
    scale=theta)``. Two linear predictors per observation are stacked in
    ``eta`` of length ``2 * n_obs``: ``eta[:n_obs] = logit q`` and
    ``eta[n_obs:] = log theta`` (scale). ``k`` and ``loc`` are fixed per
    observation (Wei et al. fix ``k`` per neuron and ``loc`` at the
    deconvolution's minimum spike size). ``mean(eta)`` returns
    ``concat([q, theta])`` in the same layout; the observation mean
    ``q (k theta + loc)`` is :func:`zero_inflated_gamma_mean`.

    The update is Fisher scoring with the expected information
    ``diag(q (1 - q)) ⊕ diag(q k)`` — analytic, PSD, and exactly block-diagonal
    across the two predictors (their expected cross-information is zero).
    ``dt`` plays no role: the trace is not a count.

    Parameters
    ----------
    shape : ArrayLike, shape () or (n_obs,)
        Gamma shape ``k > 0``.
    loc : ArrayLike, shape () or (n_obs,), default 0.0
        Gamma location ``>= 0``; positive observations must exceed it.
    """
    k = jnp.asarray(shape)
    loc_arr = jnp.asarray(loc)
    if not contains_tracer(k, loc_arr):
        for name, arr in (("shape", k), ("loc", loc_arr)):
            if arr.ndim > 1:
                raise ValueError(f"{name} must be a scalar or 1-D, got shape {arr.shape}.")
            if not bool(jnp.all(jnp.isfinite(arr))):
                raise ValueError(f"{name} must be finite.")
        if not bool(jnp.all(k > 0)):
            raise ValueError("shape must be strictly positive.")
        if not bool(jnp.all(loc_arr >= 0)):
            raise ValueError("loc must be non-negative.")

    def _split(v: Array) -> tuple[Array, Array]:
        n = v.shape[0] // 2
        return v[:n], v[n:]

    def _clipped(eta: Array) -> tuple[Array, Array]:
        eta_q, eta_s = _split(eta)
        return (
            jnp.clip(eta_q, -_BERNOULLI_ETA_CLIP, _BERNOULLI_ETA_CLIP),
            jnp.clip(eta_s, -_ZIG_LOG_SCALE_CLIP, _ZIG_LOG_SCALE_CLIP),
        )

    def mean(eta: Array) -> Array:
        eta_q, eta_s = _clipped(eta)
        return jnp.concatenate([jax.nn.sigmoid(eta_q), jnp.exp(eta_s)])

    def _parts(y: Array, mu: Array) -> tuple[Array, Array, Array, Array]:
        q, theta = _split(mu)
        positive = y > 0
        # Double-where: the masked-out branch must be finite so no NaN leaks
        # through the gradient of log / division.
        excess = jnp.where(positive, y - loc_arr, 1.0)
        return q, theta, positive.astype(mu.dtype), excess

    def score(y: Array, _eta: Array, mu: Array) -> Array:
        q, theta, z, excess = _parts(y, mu)
        return jnp.concatenate([z - q, z * (excess / theta - k)])

    def fisher_weight(_eta: Array, mu: Array) -> Array:
        q, _theta = _split(mu)
        return jnp.concatenate([q * (1.0 - q), q * k * jnp.ones_like(q)])

    def _plugin_per_obs(y: Array, eta: Array, mu: Array) -> Array:
        eta_q, eta_s = _clipped(eta)
        _q, theta, z, excess = _parts(y, mu)
        softplus_q = jax.nn.softplus(eta_q)
        return (
            (1.0 - z) * (-softplus_q)
            + z * (eta_q - softplus_q - excess / theta - k * eta_s)
        )

    def loglik_plugin(y: Array, eta: Array, mu: Array) -> Array:
        return jnp.sum(_plugin_per_obs(y, eta, mu))

    def loglik_per_obs(y: Array, eta: Array, mu: Array) -> Array:
        _q, _theta, z, excess = _parts(y, mu)
        return _plugin_per_obs(y, eta, mu) + (
            z * ((k - 1.0) * jnp.log(excess) - jax.scipy.special.gammaln(k))
        )

    def loglik_normalized(y: Array, eta: Array, mu: Array) -> Array:
        return jnp.sum(loglik_per_obs(y, eta, mu))

    return GLMFamily(
        mean,
        fisher_weight,
        loglik_plugin,
        loglik_normalized,
        score=score,
        loglik_per_obs=loglik_per_obs,
        validate_observations=functools.partial(_validate_zig_observations, loc=loc_arr),
    )


def zero_inflated_gamma_mean(eta: Array, shape: ArrayLike, loc: ArrayLike = 0.0) -> Array:
    """Observation mean ``q (k theta + loc)`` of the ZIG family, shape ``(n_obs,)``.

    ``eta`` uses the stacked layout of :func:`zero_inflated_gamma_family`
    (``[logit q, log theta]``), with the same clipping.
    """
    n = eta.shape[0] // 2
    q = jax.nn.sigmoid(jnp.clip(eta[:n], -_BERNOULLI_ETA_CLIP, _BERNOULLI_ETA_CLIP))
    theta = jnp.exp(jnp.clip(eta[n:], -_ZIG_LOG_SCALE_CLIP, _ZIG_LOG_SCALE_CLIP))
    return q * (jnp.asarray(shape) * theta + jnp.asarray(loc))
```

### 3.5 Observation validation

`_validate_public_inputs` (`point_process_kalman.py:96-145`) calls
`validate_count_array` at `:140`, as do `PointProcessModel.fit` (`:3180`) and
`fit_sgd` (`:3266`). ZIG observations are continuous, so validation becomes
family-owned through an additional optional `GLMFamily` field,
`validate_observations: Callable[[ArrayLike, str], None] | None = None`
(`None` ⇒ `validate_count_array`, so Poisson/Bernoulli/NB are unchanged):

```python
def _validate_zig_observations(obs: ArrayLike, name: str, *, loc: Array) -> None:
    """Finite, non-negative, and every positive value exceeds ``loc``."""
    arr = np.asarray(obs, dtype=float)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    loc_np = np.broadcast_to(np.asarray(loc, dtype=float), arr.shape[-1:])
    positive = arr > 0
    if np.any(arr[positive] <= np.broadcast_to(loc_np, arr.shape)[positive]):
        raise ValueError(
            f"{name}: positive values must exceed the gamma location ``loc``; "
            "set loc below the smallest positive observation."
        )


def _observation_validator(family: GLMFamily | None):
    if family is None or family.validate_observations is None:
        return validate_count_array
    return family.validate_observations
```

`_validate_public_inputs` gains a keyword-only `family: GLMFamily | None = None`
and calls `_observation_validator(family)(spike_indicator, "spike_indicator")`;
`PointProcessModel` calls the same helper on `self._family`.

### 3.6 Non-goals with a trigger

* **Latent-dependent shape `k(x)`**: `d loglik / d k = z (log e - log theta -
  psi(k))` and `-d^2 loglik / d eta_s d k = z`, so `E[...] = q != 0`: the
  expected information would couple `eta_s` and `k` and the flat design (a′)
  would no longer be exact. Revisit with design (a) only if a real dataset
  needs it (Wei et al. did not).
* **Ties at `loc`**: Wei et al.'s objective (eq. 3) uses `1[s > s_min]` and
  leaves values in `(0, s_min]` in neither component. This plan rejects them at
  validation (§3.5); users set `loc` just below their smallest positive value.

---

## 4. Threading `family` through the filter and smoother

`stochastic_point_process_filter` (`point_process_kalman.py:1467-1734`) and
`stochastic_point_process_smoother` (`:2392-2602`) never see a family today: the
dense scan body `_step` (`:1800-1846`) calls the Poisson-only
`_point_process_laplace_update` at `:1820-1833`, and the block path
(`_block_diagonal_forward_core`, `:1873-1979`) does the same at `:1935-1946`.

Additive change (every default reproduces the current program):

```python
# stochastic_point_process_filter(..., return_block_covariances=False, family: GLMFamily | None = None)
    if family is not None and max_log_count != 20.0:
        warnings.warn(
            "max_log_count is ignored when a family is given; configure the "
            "overflow ceiling on the family (e.g. negative_binomial_family("
            "dt, r, max_log_count=...)).",
            StateSpaceWarning,
            stacklevel=2,
        )
    use_block_dispatch = (
        block_n_neurons is not None
        and block_size is not None
        and not force_dense
        and family is None                      # block path is Poisson-only
        and _uses_default_linear_log_intensity(log_conditional_intensity)
    )
    ...
    result = _stochastic_point_process_filter_impl(..., max_newton_iter=max_newton_iter, family=family)
```

```python
@functools.partial(
    jax.jit,
    static_argnames=["log_conditional_intensity", "include_laplace_normalization", "max_newton_iter", "family"],
)
def _stochastic_point_process_filter_impl(..., max_newton_iter: int = 3, family: GLMFamily | None = None):
    ...
    def _step(params_prev, args):
        ...
        if family is None:
            posterior_mean, posterior_covariance, log_lik, n_failed = _point_process_laplace_update(
                ...  # unchanged
            )
        else:
            posterior_mean, posterior_covariance, log_lik, n_failed = glm_laplace_update(
                one_step_mean,
                one_step_covariance,
                spike_indicator_t,
                log_intensity_func,
                family,
                grad_eta_func=grad_log_intensity_func,
                include_laplace_normalization=include_laplace_normalization,
                max_newton_iter=max_newton_iter,
                return_line_search_failures=True,
            )
```

`family` is a static argument: a `NamedTuple` of functions is hashable (by
identity), and — verified in this planning session — a family whose closures
capture a **traced** dispersion works as a static argument under an outer
`jax.jit(jax.value_and_grad(...))` and eagerly under `jax.grad`, with the
gradient matching finite differences. Two consequences to document on the
filter: (1) each `negative_binomial_family(...)` call is a new cache key, so
callers hoist the family (the model does, §5); (2) under SGD the family is
rebuilt inside the loss from the traced `dispersion`, which is traced once per
compiled step (the outer step is cached by `SGDFittableMixin`).

The smoother mirrors this: `family` keyword, the same `and family is None` in
its `use_block_dispatch` (`:2526-2531`), and `family=family` forwarded to its
inner filter call (`:2571-2588`). `StateSpaceWarning` must be imported from
`state_space_practice.exceptions` (it is not among the imports at `:31-70`).

`glm_laplace_update` is defined at `:1333`, before the filter at `:1467`; no
reordering is needed.

---

## 5. `PointProcessModel` family plumbing

`PointProcessModel` (`point_process_kalman.py:2928-3474`). Additive keyword
arguments on `__init__` (`:2993-3006`) and one attribute set:

```python
    def __init__(self, n_state_dims, dt, ..., max_newton_iter: int = 3,
                 family: str | GLMFamily = "poisson",
                 dispersion: ArrayLike | None = None,
                 update_dispersion: bool = False):
        ...
        self.family_kind, self._custom_family = self._resolve_family(family)
        if self.family_kind != "negative_binomial":
            if dispersion is not None or update_dispersion:
                raise ValueError("dispersion / update_dispersion apply only to family='negative_binomial'.")
        elif dispersion is None:
            raise ValueError("family='negative_binomial' requires dispersion.")
        self.dispersion = None if dispersion is None else jnp.asarray(dispersion)
        self.update_dispersion = update_dispersion
        self._family = self._build_family()

    @staticmethod
    def _resolve_family(family) -> tuple[str, GLMFamily | None]:
        if isinstance(family, GLMFamily):
            return "custom", family
        if family in ("poisson", "negative_binomial"):
            return family, None
        raise ValueError(f"family must be 'poisson', 'negative_binomial' or a GLMFamily, got {family!r}.")

    def _build_family(self) -> GLMFamily | None:
        """The family object the E-step and SGD loss pass to the filter.

        ``None`` for Poisson selects the filter's original update. Built once
        (and again whenever ``dispersion`` changes) so the filter's compile
        cache sees one family object across EM iterations.
        """
        if self.family_kind == "poisson":
            return None
        if self.family_kind == "negative_binomial":
            return negative_binomial_family(self.dt, self.dispersion)
        return self._custom_family

    def _check_observation_shape(self, spike_indicator: Array) -> None:
        """Per-neuron dispersion must match the neuron axis (after 1-D promotion)."""
        if self.dispersion is not None and self.dispersion.ndim == 1:
            n_neurons = 1 if spike_indicator.ndim == 1 else spike_indicator.shape[1]
            if self.dispersion.shape[0] != n_neurons:
                raise ValueError(
                    f"dispersion has {self.dispersion.shape[0]} entries but "
                    f"spike_indicator has {n_neurons} neurons."
                )
```

Call sites:

* `_e_step` (`:3089-3103`) and `_finalize_sgd` (`:3358-3370`): add
  `family=self._family` to the smoother call.
* `fit` (`:3178-3186`) and `fit_sgd` (`:3264-3279`): call
  `self._check_observation_shape(spike_indicator)` after the array coercion.
* `_build_param_spec` (`:3294-3312`): append

  ```python
        if self.update_dispersion:
            params["dispersion"] = self.dispersion
            spec["dispersion"] = POSITIVE   # softplus; import from parameter_transforms (:96-99)
  ```

* `_sgd_loss_fn` (`:3314-3336`):

  ```python
        family = (
            negative_binomial_family(self.dt, params["dispersion"])
            if self.update_dispersion
            else self._family
        )
        _, _, marginal_ll = stochastic_point_process_filter(..., max_newton_iter=self.max_newton_iter, family=family)
  ```

* `_store_sgd_params` (`:3338-3346`): `if "dispersion" in params:
  self.dispersion = params["dispersion"]; self._family = self._build_family()`.

Phase 2 extends `_resolve_family` with `"zero_inflated_gamma"`, adds
`gamma_shape: ArrayLike | None = None, gamma_loc: ArrayLike = 0.0` (required /
optional for that kind, rejected for others), `_build_family` returns
`zero_inflated_gamma_family(self.gamma_shape, self.gamma_loc)`, and
`get_rate_estimate` (`:3373-3444`, which returns `exp(eta)` at `:3444`) raises
`ValueError("get_rate_estimate is defined for count families; for
'zero_inflated_gamma' evaluate zero_inflated_gamma_mean on the linear
predictor")` when `self.family_kind == "zero_inflated_gamma"`.

The `SGDFittableMixin` step cache (`sgd_fitting.py:437-501`) records every
model attribute the loss reads while tracing, so reading `self._family`,
`self.update_dispersion` and `self.dt` in `_sgd_loss_fn` keeps cached steps
from being reused with stale values.

---

## 6. Simulators

`simulate_data.py` is NumPy-only and mypy-gated (`pyproject.toml:157`), so the
new functions are typed NumPy code.

```python
def simulate_negative_binomial_counts(
    rate_hz: ArrayLike,
    dt: float,
    dispersion: ArrayLike,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Over-dispersed spike counts with mean ``rate_hz * dt`` and variance
    ``mean + mean**2 / dispersion`` (negative binomial; ``dispersion -> inf``
    is Poisson). ``rate_hz`` is any shape; ``dispersion`` broadcasts against it.
    """
    if rng is None:
        rng = np.random.default_rng()
    mean = np.asarray(rate_hz, dtype=float) * dt
    r = np.broadcast_to(np.asarray(dispersion, dtype=float), mean.shape)
    # numpy's NB(n, p) has mean n (1 - p) / p; p = r / (r + mean) gives mean ``mean``.
    counts: np.ndarray = rng.negative_binomial(r, r / (r + mean))
    return counts
```

(`Generator.negative_binomial` accepts real-valued `n` and array arguments;
the mean/variance identity was checked numerically during planning.)

```python
def simulate_zero_inflated_gamma_traces(
    n_time: int,
    n_neurons: int,
    n_latent: int = 2,
    transition: float = 0.95,
    process_var: float = 0.05,
    shape: float | np.ndarray = 2.0,
    loc: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Calcium-like traces from a shared AR(1) latent with per-neuron ZIG observations.

    ``x_t = transition * x_{t-1} + N(0, process_var I)``, ``x_0 ~ N(0, I)``;
    per neuron ``logit q = a_n + b_n . x_t`` and ``log theta = c_n + d_n . x_t``
    with ``a_n ~ N(-0.5, 0.5)``, ``b_n, d_n ~ N(0, 0.6)``, ``c_n ~ N(0, 0.3)``;
    ``y = Bernoulli(q) * (loc + Gamma(shape, theta))``.

    Returns dict with ``latent`` (n_time, n_latent), ``traces`` (n_time,
    n_neurons), ``design`` (n_time, 2 * n_neurons, 1 + n_latent) in the stacked
    affine layout (row ``n``: ``[a_n, b_n]``; row ``n_neurons + n``: ``[c_n,
    d_n]``; column 0 is the intercept), ``shape`` (n_neurons,), ``loc``,
    ``transition``, ``process_var``, and the per-neuron ``q`` / ``theta``.
    """
```

Body: draw the coefficient arrays, run the AR(1) with a Python loop over time
(NumPy simulators in this module already loop; `n_time <= 1e4`), compute
`eta_q, eta_s`, `q = 1 / (1 + exp(-eta_q))`, `theta = exp(eta_s)`,
`y = (rng.random(...) < q) * (loc + rng.gamma(shape, theta))`. Return the design
so `_affine_log_rate` (`tests/test_oracle_point_process.py:76-78`) yields the
stacked `eta` directly.

For JAX-side tests (`tests/recovery_helpers.py:204-228` pattern), an NB sampler
as a gamma-Poisson mixture (JAX has no `negative_binomial` sampler; checked):

```python
def simulate_negative_binomial_spikes(x_true, C, d, dt, dispersion, key=None) -> Array:
    """NB counts from a latent trajectory: Poisson(rate * dt * G), G ~ Gamma(r, 1/r)."""
    if key is None:
        key = jax.random.PRNGKey(78)
    key_gamma, key_poisson = jax.random.split(key)
    log_rates = x_true @ C.T + d
    means = jnp.exp(jnp.clip(log_rates, -5, 3)) * dt
    r = jnp.asarray(dispersion)
    gain = jax.random.gamma(key_gamma, r, shape=means.shape) / r
    return jax.random.poisson(key_poisson, means * gain)
```

---

## 7. Extending the grid-quadrature oracle and the calibration harness

`tests/test_oracle_point_process.py` hard-codes Poisson in three places: the
simulator (`:91-121`, `rng.poisson` at `:118`), the grid likelihood
(`:191-240`, `:210-212`), and the reference recursion's score/information
(`:261-267`). Introduce one observation-model record and thread it through
`_Problem`:

```python
class _Observation(NamedTuple):
    """Everything the oracle needs about an observation model, in NumPy."""
    name: str
    family: object | None            # GLMFamily for the library call; None = legacy Poisson path
    n_eta_per_neuron: int            # 1 (Poisson, NB), 2 (ZIG: rows [logit q ; log theta])
    sample: Callable                 # (rng, eta) -> y   for one time step, eta (n_eta,)
    loglik_grid: Callable            # (eta_grid (G, n_eta), y (n_neurons,)) -> (G,)  normalized log-lik
    score_info: Callable             # (eta (n_eta,), y) -> (score (n_eta,), weight (n_eta,))  expected info


def _poisson_observation(dt) -> _Observation:
    def sample(rng, eta):
        return rng.poisson(np.exp(eta) * dt)
    def loglik_grid(eta_grid, y):
        mu = np.exp(eta_grid) * dt
        return np.sum(y * np.log(mu) - mu - gammaln(y + 1.0), axis=1)
    def score_info(eta, y):
        mu = np.exp(eta) * dt
        return y - mu, mu
    return _Observation("poisson", None, 1, sample, loglik_grid, score_info)


def _negative_binomial_observation(dt, r) -> _Observation:
    def sample(rng, eta):
        mu = np.exp(eta) * dt
        return rng.negative_binomial(r, r / (r + mu))
    def loglik_grid(eta_grid, y):
        mu = np.exp(eta_grid) * dt
        return np.sum(scipy.stats.nbinom.logpmf(y, r, r / (r + mu)), axis=1)
    def score_info(eta, y):
        mu = np.exp(eta) * dt
        shrink = r / (r + mu)
        return (y - mu) * shrink, mu * shrink
    return _Observation("nb", negative_binomial_family(dt, r), 1, sample, loglik_grid, score_info)


def _zig_observation(k, loc) -> _Observation:
    def _split(eta):
        n = eta.shape[-1] // 2
        return eta[..., :n], eta[..., n:]
    def sample(rng, eta):
        eta_q, eta_s = _split(eta)
        q, theta = 1 / (1 + np.exp(-eta_q)), np.exp(eta_s)
        return (rng.random(q.shape) < q) * (loc + rng.gamma(k, theta))
    def loglik_grid(eta_grid, y):
        eta_q, eta_s = _split(eta_grid)
        z = (y > 0).astype(float)
        e = np.where(y > 0, y - loc, 1.0)
        log_q = eta_q - np.logaddexp(0.0, eta_q)
        log_1mq = -np.logaddexp(0.0, eta_q)
        gamma_term = (k - 1) * np.log(e) - e / np.exp(eta_s) - k * eta_s - gammaln(k)
        return np.sum((1 - z) * log_1mq + z * (log_q + gamma_term), axis=1)
    def score_info(eta, y):
        eta_q, eta_s = _split(eta)
        q, theta = 1 / (1 + np.exp(-eta_q)), np.exp(eta_s)
        z = (y > 0).astype(float)
        e = np.where(y > 0, y - loc, 1.0)
        return (np.concatenate([z - q, z * (e / theta - k)]),
                np.concatenate([q * (1 - q), q * k * np.ones_like(q)]))
    return _Observation("zig", zero_inflated_gamma_family(k, loc), 2, sample, loglik_grid, score_info)
```

Changes to the existing helpers (keep Poisson defaults so current tests are
unchanged):

* `_Problem` gains `obs: _Observation`. `_simulate_problem(..., obs=None)`
  builds `design` with `n_neurons * obs.n_eta_per_neuron` rows (ZIG: two
  intercept/weight rows per neuron) and samples with `obs.sample(rng, eta)`.
* `_grid_posterior`: replace `:210-212` by `eta = design_t[:, 0][None, :] +
  points @ design_t[:, 1:].T` and `loglik = obs.loglik_grid(eta, y_t)`.
* `_log_evidence_terms` → `obs.score_info(eta, y_t)` plus the normalized
  log-lik from `obs.loglik_grid(eta[None], y_t)[0]`; `_laplace_filter_reference`
  is then **Fisher scoring** (the docstring's "pure Newton" remark holds only
  for the canonical Poisson case — update it).
* `_run_laplace` passes `family=problem.obs.family` to the smoother; `dt` is
  irrelevant for ZIG but the smoother still requires a positive value (pass the
  problem's `dt`).

Calibration (`tests/test_calibration_point_process.py`): `_simulate_replicates`
(`:86-110`) takes a `sample(rng, rate_times_dt) -> counts` callable (default
`rng.poisson`) and `_pp_smoother_z` (`:113-135`) a `family=None` forwarded to the
smoother. The family object is built once outside the `vmap` (concrete
dispersion). The SBC rank test (`tests/test_sbc_ranks.py:190-228`) gets the same
two knobs via a small `_pp_rank_stat(seed, sample, family)` helper.

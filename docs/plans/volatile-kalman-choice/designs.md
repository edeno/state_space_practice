# Designs — Volatile Kalman filter for the choice models

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Each section is referenced from a phase file by anchor. Code here is the
intended implementation; phases list the tasks and tests around it.

Notation (Piray & Daw 2020): `m` posterior mean, `w` posterior variance, `v`
volatility (= the process variance used for the next prediction), `k` Kalman
gain, `λ` volatility learning rate, `v0` initial volatility, `σ²` outcome
noise (Gaussian variant), `ω` the binary variant's inference-only "noise
parameter". Trial `t` runs over `0..T-1` in code; the reference's `t = 1..T`.

Contents

1. [Reference recursions (transcribed)](#vkf-reference)
2. [JAX core: masked, optionally shared-volatility VKF](#vkf-jax-core)
3. [NumPy oracle for the tests](#vkf-oracle)
4. [`VolatileKalmanChoiceModel`](#vkf-choice-model)
5. [Block-change bandit simulator](#vkf-simulator)
6. [Hybrid: volatility as the process noise of the Laplace-EKF choice filter](#hybrid-filter)
7. [Hybrid model hooks on `CovariateChoiceModel`](#hybrid-model-hooks)
8. [Identifiability gate](#identifiability-gate)
9. [Model-comparison harness](#comparison-harness)
10. [Generative agents for the confusion matrix](#comparison-simulators)

---

## 1. Reference recursions (transcribed) <a id="vkf-reference"></a>

Source of truth for the recursions is the authors' reference code,
`github.com/payampiray/VKF` (branch `master`, files `vkf.m` and `vkf_bin.m`),
transcribed verbatim below, together with the equation numbers of the
published article (Piray & Daw 2020, *PLoS Comput Biol* 16(7): e1007963; the
numbers below are the PLoS numbering, read from the article's JATS full text;
the comments inside `vkf.m` cite the bioRxiv preprint's numbering, which
happens to coincide for Eqs 9-13).

### 1a. Gaussian-observation VKF (`vkf.m`)

```matlab
function [predictions, signals] = vkf(outcomes,lambda,v0,sigma2)
% Inputs:  outcomes (T x C), 0<lambda<1, v0>0, sigma2>0
% Note: outputs of VKF also depends on initial variance (w0), which is
% assumed here w0 = sigma2

w0      = sigma2;
[T,C] = size(outcomes);
m       = zeros(1,C);
w       = w0*ones(1,C);
v       = v0*ones(1,C);

for t  = 1:T
    o = outcomes(t,:);
    predictions(t,:) = m;          % prediction BEFORE seeing o_t
    volatility(t,:) = v;           % volatility BEFORE the update

    mpre        = m;
    wpre        = w;

    delta_m     = o - m;
    k           = (w+v)./(w+v + sigma2);                            % Eq 9
    m           = m + k.*delta_m;                                   % Eq 10
    w           = (1-k).*(w+v);                                     % Eq 11

    wcov        = (1-k).*wpre;                                      % Eq 12
    delta_v     = (m-mpre).^2 + w + wpre - 2*wcov - v;
    v           = v +lambda.*delta_v;                               % Eq 13

    learning_rate(t,:) = k;
    prediction_error(t,:) = delta_m;
    volatility_error(t,:) = delta_v;
end
```

Article equations (PLoS numbering), matching the code line for line:

```
Eq 9   k_t       = (w_{t-1} + v_{t-1}) / (w_{t-1} + v_{t-1} + σ²)
Eq 10  m_t       = m_{t-1} + k_t (o_t − m_{t-1})
Eq 11  w_t       = (1 − k_t)(w_{t-1} + v_{t-1})
Eq 12  w_{t-1,t} = (1 − k_t) w_{t-1}                       (autocovariance)
Eq 13  v_t       = v_{t-1} + λ [ (m_t − m_{t-1})² + w_{t-1} + w_t − 2 w_{t-1,t} − v_{t-1} ]
       where E[(x_t − x_{t-1})²] = (m_t − m_{t-1})² + w_{t-1} + w_t − 2 w_{t-1,t}
```

Article: "the Kalman filter algorithm is a special case of the VKF in which
λ = 0 and the process variance is equal to v_0 on all trials." (The
reference code rejects `lambda <= 0`; our implementation accepts `λ = 0`
because that special case is the exact fallback used by the hybrid model and
the Kalman-filter oracle test.)

### 1b. Binary-outcome VKF (`vkf_bin.m`)

```matlab
function [predictions, signals] = vkf_bin(outcomes,lambda,v0,omega)
% Inputs:  outcomes (T x C) binary, 0<lambda<1, v0>0, omega>0 (noise parameter)
% Note: outputs of VKF also depends on initial variance (w0), which is
% assumed here w0 = omega

w0 = omega;
m       = zeros(1,C);
w       = w0*ones(1,C);
v       = v0*ones(1,C);

sigmoid = @(x)1./(1+exp(-x));
for t  = 1:T
    o = outcomes(t,:);
    predictions(t,:) = m;
    volatility(t,:) = v;

    mpre        = m;
    wpre        = w;

    delta_m     = o - sigmoid(m);
    k           = (w+v)./(w+v+ omega);
    alpha       = sqrt(w+v);
    m           = m + alpha.*delta_m;
    w           = (1-k).*(w+v);

    wcov        = (1-k).*wpre;
    delta_v     = (m-mpre).^2 + w + wpre - 2*wcov - v;
    v           = v +lambda.*delta_v;

    learning_rate(t,:) = alpha;
    prediction_error(t,:) = delta_m;
    volatility_error(t,:) = delta_v;
end
```

Article equations (PLoS numbering):

```
Eq 14  k_t       = (w_{t-1} + v_{t-1}) / (w_{t-1} + v_{t-1} + ω)
Eq 15  α_t       = sqrt(w_{t-1} + v_{t-1})
Eq 16  m_t       = m_{t-1} + α_t (o_t − s(m_{t-1})),   s(x) = 1 / (1 + exp(−x))
Eq 17  w_t       = (1 − k_t)(w_{t-1} + v_{t-1})
Eq 18  w_{t-1,t} = (1 − k_t) w_{t-1}
Eq 19  v_t       = v_{t-1} + λ [ (m_t − m_{t-1})² + w_{t-1} + w_t − 2 w_{t-1,t} − v_{t-1} ]
```

Article prose on ω: "we have introduced a parameter, ω>0, specifically for
inference (i.e. does not exist in the generative model). We call this
parameter the noise parameter, because its effects on volatility are similar
to the noise parameter, σ² for linear observations (through k_t)."

Two facts to keep in mind when designing around these:

- **The mean update of the binary VKF uses `α_t = sqrt(w+v)`, not the gain
  `k_t`.** `k_t` only enters the variance recursion. `α_t` is not bounded by
  1 and `m` lives on the logit scale (`s(m)` is the predicted outcome
  probability).
- **Positivity of `v`.** Algebra on Eq 13: `w_t + w_{t-1} − 2 w_{t-1,t} =
  (1−k) v_{t-1} + k w_{t-1}`, so `v_t = (1 − λ k_t) v_{t-1} + λ [(m_t −
  m_{t-1})² + k_t w_{t-1}]`. Every term is ≥ 0 and `λ k_t < 1`, hence
  `v_t > 0` whenever `v_{t-1} > 0`. The same holds for the binary variant
  (same `w`, `k`, `wcov`). Tests assert this property.

### 1c. Generative model (for orientation only; not implemented)

Article Eq 1: `x_t = x_{t-1} + e_t`, `e_t ~ N(0, v)`; Eq 6: precision
`z_t = z_{t-1} ε_t` with `ε_t` a rescaled Beta variable of mean 1 whose spread
is controlled by `λ` (details in the article's S1 Appendix, which was not
read). A simulator of this multiplicative-diffusion process is deliberately
not part of this plan (see overview Open Questions); the block-change bandit
simulator (§5) is the motivating scenario instead.

---

## 2. JAX core: masked, optionally shared-volatility VKF <a id="vkf-jax-core"></a>

Module: `src/state_space_practice/volatile_kalman.py`.

Two extensions over the reference, both switched off by default so that the
default call reproduces `vkf.m` / `vkf_bin.m`:

1. **Observation mask** (`outcome_mask`, `(T, C)` in `{0, 1}`). A bandit
   reveals only the chosen option's reward. With `k` and `α` multiplied by
   the mask, an unobserved cue gets `m` unchanged, `w ← w + v` (the prior
   variance grows by the volatility, the standard Kalman treatment of a
   missing observation, as in Daw et al. 2006's Kalman bandit) and
   `δv = 0`, so its volatility is unchanged. Multiplying by `1.0` is exact
   in IEEE arithmetic, so a mask of ones is bit-identical to no mask.
   `δv` is also multiplied by the mask because the expression
   `(w+v) + w − 2w − v` is only zero up to round-off.
2. **Shared volatility** (`shared_volatility=True`): one `v` for all cues,
   updated with the mean of the observed cues' `δv`. With one cue this is
   the reference recursion exactly; with several fully observed cues it is a
   modelling choice (documented). In a bandit exactly one cue is observed
   per trial, so `v ← v + λ δv_chosen`.

```python
"""Volatile Kalman filter (VKF) and a reward-learning choice model.

The VKF (Piray & Daw 2020) is a Kalman filter whose process variance -- the
"volatility" -- is itself tracked online, so the learning rate rises after
changes in the outcome statistics. ``volatile_kalman_filter`` and
``binary_volatile_kalman_filter`` transcribe the reference recursions
(github.com/payampiray/VKF: ``vkf.m`` and ``vkf_bin.m``) as pure functions
under ``jax.lax.scan``, with two additive options: an observation mask for
partially observed cues (a bandit reveals only the chosen option's reward)
and a volatility shared across cues. ``VolatileKalmanChoiceModel`` learns
option values from rewards with the VKF and models choices as a softmax of
those values.

References
----------
[1] Piray, P. & Daw, N.D. (2020). A simple model for learning in volatile
    environments. PLoS Computational Biology 16(7), e1007963.
[2] Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006).
    Cortical substrates for exploratory decisions in humans. Nature 441,
    876-879.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.utils import contains_tracer

ObservationModel = Literal["gaussian", "binary"]


class VolatileKalmanState(NamedTuple):
    """Carry of the VKF scan.

    Attributes
    ----------
    mean : Array, shape (n_cues,)
        Posterior mean ``m`` after the last update.
    variance : Array, shape (n_cues,)
        Posterior variance ``w`` after the last update.
    volatility : Array, shape (n_cues,) or ()
        Volatility ``v``; a scalar when shared across cues.
    """

    mean: Array
    variance: Array
    volatility: Array


class VolatileKalmanResult(NamedTuple):
    """Per-trial signals of the VKF (the reference code's ``signals`` struct).

    Attributes
    ----------
    predictions : Array, shape (n_trials, n_cues)
        ``m`` before the update on trial t: the prediction of outcome t.
    volatility : Array, shape (n_trials, n_cues) or (n_trials,)
        ``v`` before the update on trial t (the process variance of the
        prediction at t); one column when shared across cues.
    learning_rate : Array, shape (n_trials, n_cues)
        Kalman gain ``k_t`` (Gaussian) or ``alpha_t = sqrt(w + v)`` (binary),
        zero for unobserved cues.
    prediction_error : Array, shape (n_trials, n_cues)
        ``o_t - m_{t-1}`` (Gaussian) or ``o_t - sigmoid(m_{t-1})`` (binary).
    volatility_prediction_error : Array, shape (n_trials, n_cues) or (n_trials,)
        ``E[(x_t - x_{t-1})^2] - v_{t-1}``; zero for unobserved cues.
    posterior_means : Array, shape (n_trials, n_cues)
        ``m`` after the update on trial t.
    posterior_variances : Array, shape (n_trials, n_cues)
        ``w`` after the update on trial t.
    predicted_variances : Array, shape (n_trials, n_cues)
        ``w_{t-1} + v_{t-1}``: the prior variance of the prediction at t.
    """

    predictions: Array
    volatility: Array
    learning_rate: Array
    prediction_error: Array
    volatility_prediction_error: Array
    posterior_means: Array
    posterior_variances: Array
    predicted_variances: Array


def _volatility_update(volatility: Array, learning_rate: Array, delta: Array) -> Array:
    """``v + lambda * delta`` (Eq 13 / Eq 19), shared with the hybrid filter."""
    return volatility + learning_rate * delta


def volatile_kalman_step(
    state: VolatileKalmanState,
    outcome: Array,
    outcome_mask: Array,
    volatility_learning_rate: Array,
    observation_noise: Array,
    observation: ObservationModel,
    shared_volatility: bool,
) -> tuple[VolatileKalmanState, tuple[Array, ...]]:
    """One VKF trial for all cues (Eqs 9-13, or 14-19 for binary outcomes).

    Parameters
    ----------
    state : VolatileKalmanState
        ``(m, w, v)`` after the previous trial.
    outcome : Array, shape (n_cues,)
    outcome_mask : Array, shape (n_cues,)
        1.0 where the cue's outcome was observed, 0.0 elsewhere.
    volatility_learning_rate : Array, shape ()
        ``lambda`` in [0, 1).
    observation_noise : Array, shape ()
        ``sigma^2`` (Gaussian) or ``omega`` (binary), > 0.
    observation : {"gaussian", "binary"}
    shared_volatility : bool
        Whether ``state.volatility`` is a scalar shared by all cues.

    Returns
    -------
    new_state : VolatileKalmanState
    outputs : tuple
        The eight per-trial arrays of :class:`VolatileKalmanResult`, in order.
    """
    m_pre, w_pre, v_pre = state
    v_cue = jnp.broadcast_to(v_pre, m_pre.shape)
    prior_var = w_pre + v_cue  # w_{t-1} + v_{t-1}
    k = outcome_mask * prior_var / (prior_var + observation_noise)  # Eq 9 / Eq 14
    if observation == "gaussian":
        delta_m = outcome - m_pre
        alpha = k
    else:
        delta_m = outcome - jax.nn.sigmoid(m_pre)
        alpha = outcome_mask * jnp.sqrt(prior_var)  # Eq 15
    m = m_pre + alpha * delta_m  # Eq 10 / Eq 16
    w = (1.0 - k) * prior_var  # Eq 11 / Eq 17
    w_cov = (1.0 - k) * w_pre  # Eq 12 / Eq 18
    # Eq 13 / Eq 19 bracket. The mask makes an unobserved cue's term exactly
    # zero (the expression is zero only up to round-off otherwise).
    delta_v = outcome_mask * ((m - m_pre) ** 2 + w + w_pre - 2.0 * w_cov - v_cue)
    if shared_volatility:
        n_observed = jnp.sum(outcome_mask)
        delta_v_used = jnp.sum(delta_v) / jnp.maximum(n_observed, 1.0)
    else:
        delta_v_used = delta_v
    v = _volatility_update(v_pre, volatility_learning_rate, delta_v_used)
    new_state = VolatileKalmanState(m, w, v)
    outputs = (m_pre, v_pre, alpha, delta_m, delta_v_used, m, w, prior_var)
    return new_state, outputs


@partial(jax.jit, static_argnames=("observation", "shared_volatility"))
def _volatile_kalman_scan(
    outcomes: Array,
    outcome_mask: Array,
    volatility_learning_rate: Array,
    initial_volatility: Array,
    observation_noise: Array,
    init_mean: Array,
    init_variance: Array,
    observation: ObservationModel,
    shared_volatility: bool,
) -> VolatileKalmanResult:
    """Jitted VKF over ``(n_trials, n_cues)`` outcomes (no validation)."""
    n_cues = outcomes.shape[1]
    v_init = (
        jnp.asarray(initial_volatility)
        if shared_volatility
        else jnp.broadcast_to(jnp.asarray(initial_volatility), (n_cues,))
    )
    init = VolatileKalmanState(init_mean, init_variance, v_init)

    def _step(state, inputs):
        outcome_t, mask_t = inputs
        return volatile_kalman_step(
            state,
            outcome_t,
            mask_t,
            volatility_learning_rate,
            observation_noise,
            observation,
            shared_volatility,
        )

    _, outputs = jax.lax.scan(_step, init, (outcomes, outcome_mask))
    return VolatileKalmanResult(*outputs)
```

Public wrappers (host-side validation only on concrete inputs, mirroring
`covariate_choice_filter`, `covariate_choice.py:210-320`):

```python
def volatile_kalman_filter(
    outcomes: ArrayLike,
    volatility_learning_rate: float,
    initial_volatility: float,
    observation_noise: float,
    *,
    outcome_mask: ArrayLike | None = None,
    init_mean: ArrayLike | None = None,
    init_variance: ArrayLike | None = None,
    observation: ObservationModel = "gaussian",
    shared_volatility: bool = False,
) -> VolatileKalmanResult:
    """Volatile Kalman filter for Gaussian outcomes (Piray & Daw 2020, Eqs 9-13).

    With the defaults (no mask, per-cue volatility, ``init_mean = 0``,
    ``init_variance = observation_noise``) this is the reference
    implementation ``vkf.m`` applied to each column of ``outcomes``.

    Parameters
    ----------
    outcomes : ArrayLike, shape (n_trials, n_cues) or (n_trials,)
    volatility_learning_rate : float
        ``lambda`` in ``[0, 1)``. ``0`` is the Kalman filter with process
        variance ``initial_volatility`` on every trial.
    initial_volatility : float
        ``v0 > 0``.
    observation_noise : float
        Outcome variance ``sigma^2 > 0``.
    outcome_mask : ArrayLike or None, shape (n_trials, n_cues)
        1 where the cue's outcome was observed. An unobserved cue keeps its
        mean, its variance grows by the volatility and its volatility is
        unchanged. Default: everything observed.
    init_mean : ArrayLike or None, shape (n_cues,)
        Default zeros.
    init_variance : ArrayLike or None, shape (n_cues,)
        Default ``observation_noise`` (the reference code's ``w0 = sigma2``).
    observation : {"gaussian", "binary"}
        ``"binary"`` selects Eqs 14-19 (:func:`binary_volatile_kalman_filter`).
    shared_volatility : bool
        One volatility for all cues, updated with the mean volatility
        prediction error of the observed cues.

    Returns
    -------
    VolatileKalmanResult

    Raises
    ------
    ValueError
        If ``volatility_learning_rate`` is outside ``[0, 1)``,
        ``initial_volatility`` or ``observation_noise`` is not positive,
        ``outcomes`` is not finite, or the mask shape does not match.
    """
    ...  # coerce outcomes to (T, C) float, mask to float (T, C) of ones by default,
    ...  # validate when not contains_tracer(...), then _volatile_kalman_scan(...)


def binary_volatile_kalman_filter(
    outcomes, volatility_learning_rate, initial_volatility, noise_parameter, *,
    outcome_mask=None, init_mean=None, init_variance=None, shared_volatility=False,
) -> VolatileKalmanResult:
    """Binary-outcome VKF (Eqs 14-19); ``noise_parameter`` is the article's omega.

    Outcomes must lie in [0, 1]. ``predictions`` are on the logit scale;
    ``jax.nn.sigmoid(predictions)`` is the predicted outcome probability.
    Default ``init_variance = noise_parameter`` (``w0 = omega`` in ``vkf_bin.m``).
    """
    return volatile_kalman_filter(..., observation="binary", ...)
```

Validation calls: `validate_finite_array("outcomes", ...)` (`utils.py:1044`),
`validate_unit_interval_array("outcomes", ...)` for binary (`utils.py:1090`),
scalar checks on `λ`, `v0`, noise guarded by `contains_tracer`
(`utils.py:910`). A 1-D `outcomes` is reshaped to `(T, 1)`.

---

## 3. NumPy oracle for the tests <a id="vkf-oracle"></a>

Home: `src/state_space_practice/tests/oracles.py` (the module already hosts
the exact LGSSM oracles). Direct port of `vkf.m` / `vkf_bin.m`, loop over
trials, no JAX, so the JAX scan is checked against an independent
transcription of the same equations:

```python
def reference_volatile_kalman(
    outcomes: np.ndarray, lam: float, v0: float, noise: float, binary: bool = False
) -> dict[str, np.ndarray]:
    """NumPy transcription of ``vkf.m`` (``binary=False``) / ``vkf_bin.m``.

    Returns the reference ``signals`` struct plus the post-update ``m``/``w``:
    keys ``predictions, volatility, learning_rate, prediction_error,
    volatility_prediction_error, posterior_means, posterior_variances``.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    n_trials, n_cues = outcomes.shape
    m = np.zeros(n_cues)
    w = np.full(n_cues, noise)  # w0 = sigma2 (or omega)
    v = np.full(n_cues, v0)
    out = {key: np.empty((n_trials, n_cues)) for key in (
        "predictions", "volatility", "learning_rate", "prediction_error",
        "volatility_prediction_error", "posterior_means", "posterior_variances",
    )}
    for t in range(n_trials):
        o = outcomes[t]
        out["predictions"][t] = m
        out["volatility"][t] = v
        mpre, wpre = m, w
        if binary:
            delta_m = o - 1.0 / (1.0 + np.exp(-m))
            k = (w + v) / (w + v + noise)
            alpha = np.sqrt(w + v)
        else:
            delta_m = o - m
            k = (w + v) / (w + v + noise)
            alpha = k
        m = m + alpha * delta_m
        w = (1 - k) * (wpre + v)
        wcov = (1 - k) * wpre
        delta_v = (m - mpre) ** 2 + w + wpre - 2 * wcov - v
        v = v + lam * delta_v
        out["learning_rate"][t] = alpha
        out["prediction_error"][t] = delta_m
        out["volatility_prediction_error"][t] = delta_v
        out["posterior_means"][t] = m
        out["posterior_variances"][t] = w
    return out
```

Hand-checkable trace (Gaussian, `λ = 0.5`, `v0 = 1`, `σ² = 1`, outcomes
`[1, 0, 2]`), exact fractions for the first two trials, used as a second,
transcription-independent check in `test_volatile_kalman.py`:

| t | prior var `w+v` | `k_t` | `m_t` (post) | `w_t` | `δv_t` | `v_t` (post) |
|---|---|---|---|---|---|---|
| 1 | 2 | 2/3 | 2/3 | 2/3 | 4/9 | 11/9 |
| 2 | 17/9 | 17/26 | 3/13 | 17/26 | −527/3042 | 2303/2028 |
| 3 | ≈ 1.78945 | ≈ 0.64151 | ≈ 1.36574 | ≈ 0.64151 | ≈ 0.97911 | ≈ 1.62516 |

(Row 3 decimals are rounded from the recursion; assert them at `atol=1e-4`,
rows 1-2 at `1e-12`.)

Independent Kalman oracle: with `λ = 0` the Gaussian VKF is the Kalman filter
with `A = 1`, `Q = v0`, `H = 1`, `R = σ²`, prior `N(0, σ²)` propagated once
before the first observation — exactly `kalman.kalman_filter(init_mean=[0.],
init_cov=[[σ²]], obs=outcomes[:, None], A=[[1.]], Q=[[v0]], H=[[1.]],
R=[[σ²]])` (`kalman.py:613-622`; its scan predicts before each update, see
`_kalman_filter_impl`'s `_step` at `kalman.py:570-592`). Its filtered means
and variances must equal `posterior_means` / `posterior_variances` to
`1e-10`, and `predictions[1:] == posterior_means[:-1]`.

---

## 4. `VolatileKalmanChoiceModel` <a id="vkf-choice-model"></a>

Decisions (recorded; see overview for what would break them):

- **Values are learned from rewards, not inferred from choices**, so all `K`
  option values are identified by the reward scale. The model therefore uses
  full-`K` value vectors (no reference option; the softmax is shift-invariant
  so this loses nothing). Trial-aligned attributes keep the `*_option_*_`
  names of `MultinomialChoiceModel` (`multinomial_choice.py:772-814`) with
  shape `(T, K)`.
- **Choice input is the VKF prediction `m`** (`softmax(β m_t)`), for both
  observation models. For the binary variant `m` is on the logit scale, so
  `β` is per logit unit.
- **`observation="binary"` default**: the target task's rewards are binary
  and the binary VKF is what Piray & Daw fit to binary-outcome tasks. The
  Gaussian variant is one keyword away (graded rewards).
- **`shared_volatility=True` default**: in the spatial bandit the reward
  contingencies of all patches change together at epoch boundaries, so
  volatility is an environment-level quantity; a shared `v` also lets a
  change detected on the chosen option raise the learning rate (and the
  prior variance) of the unchosen ones. Per-option volatility is available
  (`shared_volatility=False`) and is what the reference code computes when
  every cue is observed; in a bandit an unchosen option's per-option
  volatility can never move (`δv = 0` under the mask), which is the reason
  it is not the default.
- **`λ, v0, ω (or σ²), β` are all learned by default** (`SGDFittableMixin`);
  `init_variance` is fixed at the noise parameter as in the reference code.
  A `(v0, ω)` ridge is expected (both enter `k` through `w + v`); the
  recovery test reports it and the identifiability report (overview
  Dependency policy) is the tool to quantify it.

```python
class VolatileKalmanChoiceModel(SGDFittableMixin):
    """Multi-armed bandit whose option values are learned by a volatile Kalman filter.

    Each option's value is the VKF prediction of its reward; only the chosen
    option's reward is observed on a trial, so the other options keep their
    value while their uncertainty grows by the current volatility. The
    volatility is tracked online (Piray & Daw 2020), so the learning rate
    rises after reward-contingency changes. Choices are
    ``Categorical(softmax(beta * values_t))``.

    Parameters
    ----------
    n_options : int
    observation : {"binary", "gaussian"}
        Reward model. ``"binary"`` (Eqs 14-19 of the reference) expects
        rewards in [0, 1] and keeps values on the logit scale; ``"gaussian"``
        (Eqs 9-13) treats rewards as continuous outcomes.
    shared_volatility : bool
        One volatility for all options (default) or one per option.
    init_volatility_learning_rate : float
        Starting ``lambda`` in [0, 1).
    init_initial_volatility : float
        Starting ``v0 > 0``.
    init_observation_noise : float
        Starting ``omega`` (binary) or ``sigma^2`` (Gaussian), > 0.
    init_inverse_temperature : float
        Starting softmax ``beta > 0``.
    learn_volatility_learning_rate, learn_initial_volatility,
    learn_observation_noise, learn_inverse_temperature : bool
        Which parameters ``fit_sgd`` optimises (all True by default).
    """

    _sgd_param_attrs = {
        "volatility_learning_rate": "volatility_learning_rate",
        "initial_volatility": "initial_volatility",
        "observation_noise": "observation_noise",
        "inverse_temperature": "inverse_temperature",
    }

    def __init__(self, n_options, observation="binary", shared_volatility=True,
                 init_volatility_learning_rate=0.1, init_initial_volatility=0.1,
                 init_observation_noise=1.0, init_inverse_temperature=1.0,
                 learn_volatility_learning_rate=True, learn_initial_volatility=True,
                 learn_observation_noise=True, learn_inverse_temperature=True):
        # validate: n_options >= 2; 0 <= lambda < 1; v0 > 0; noise > 0; beta > 0;
        # observation in {"binary", "gaussian"} -> ValueError otherwise
        ...
        self._choices: Array | None = None
        self._rewards: Array | None = None
        self._result: VolatileKalmanResult | None = None
        self._n_trials: int | None = None
        self.log_likelihood_: float | None = None
        self.log_likelihood_history_: list[float] | None = None
        self.converged_: bool | None = None
        # trial-aligned outputs, populated by _finalize_sgd
        self.predicted_option_values_ = None      # (T, K)  m before reward t
        self.filtered_option_values_ = None       # (T, K)  m after reward t
        self.predicted_option_variances_ = None   # (T, K)  w + v before reward t
        self.filtered_option_variances_ = None    # (T, K)  w after reward t
        self.volatility_ = None                   # (T,) or (T, K)
        self.learning_rate_ = None                # (T, K)  k_t or alpha_t
        self.volatility_prediction_error_ = None  # (T,) or (T, K)
        self.predicted_choice_entropy_ = None     # (T,)
        self.surprise_ = None                     # (T,)

    # --- data binding -------------------------------------------------------
    def _learning_inputs(self, choices: Array, rewards: Array) -> tuple[Array, Array]:
        """Outcome matrix (T, K) and mask (T, K) = one_hot(choices)."""
        mask = jax.nn.one_hot(choices, self.n_options, dtype=rewards.dtype)
        outcomes = jnp.broadcast_to(rewards[:, None], mask.shape)
        return outcomes, mask

    def _prepare_sgd_data(self, choices, rewards):
        """Validate choices / rewards, record ``_n_trials`` (SGDFittableMixin hook)."""
        choices_np = np.asarray(choices)
        # 1-D, >= 2 trials, validate_choice_indices; rewards same length, finite,
        # in [0, 1] when observation == "binary"  -> ValueError otherwise
        choices_arr = jnp.asarray(choices_np, dtype=jnp.int32)
        rewards_arr = jnp.asarray(rewards, dtype=float)
        self._n_trials = int(choices_arr.shape[0])
        self._choices, self._rewards = choices_arr, rewards_arr
        return (choices_arr, rewards_arr), {}

    def fit_sgd(self, choices, rewards, optimizer=None, num_steps=200,
                verbose=False, convergence_tol=None) -> list[float]:
        """Fit lambda, v0, the noise parameter and beta by maximum likelihood.

        Parameters
        ----------
        choices : ArrayLike, shape (n_trials,)
            Chosen option per trial, integers in ``[0, n_options)``.
        rewards : ArrayLike, shape (n_trials,)
            Reward of the chosen option on each trial (in [0, 1] for the
            binary observation model).
        optimizer, num_steps, verbose, convergence_tol
            As in :meth:`SGDFittableMixin.fit_sgd`.
        """
        return super().fit_sgd(choices, rewards, optimizer=optimizer,
                               num_steps=num_steps, verbose=verbose,
                               convergence_tol=convergence_tol)

    # --- SGDFittableMixin protocol -----------------------------------------
    @property
    def _n_timesteps(self) -> int:
        return self._n_trials

    def _build_param_spec(self):
        params, spec = {}, {}
        if self.learn_volatility_learning_rate:
            params["volatility_learning_rate"] = jnp.array(self.volatility_learning_rate)
            spec["volatility_learning_rate"] = UNIT_INTERVAL
        if self.learn_initial_volatility:
            params["initial_volatility"] = jnp.array(self.initial_volatility)
            spec["initial_volatility"] = POSITIVE
        if self.learn_observation_noise:
            params["observation_noise"] = jnp.array(self.observation_noise)
            spec["observation_noise"] = POSITIVE
        if self.learn_inverse_temperature:
            params["inverse_temperature"] = jnp.array(self.inverse_temperature)
            spec["inverse_temperature"] = POSITIVE
        return params, spec

    def _run(self, params: dict, choices: Array, rewards: Array) -> VolatileKalmanResult:
        # Read a model attribute only for parameters not being optimised
        # (see MultinomialChoiceModel._sgd_loss_fn, multinomial_choice.py:1020-1027).
        def _param(key):
            return params[key] if key in params else jnp.asarray(getattr(self, key))
        outcomes, mask = self._learning_inputs(choices, rewards)
        noise = _param("observation_noise")
        return _volatile_kalman_scan(
            outcomes, mask, _param("volatility_learning_rate"),
            _param("initial_volatility"), noise,
            jnp.zeros(self.n_options), jnp.broadcast_to(noise, (self.n_options,)),
            self.observation, self.shared_volatility,
        )

    def _choice_log_probs(self, params, choices, rewards):
        result = self._run(params, choices, rewards)
        logits = params.get("inverse_temperature", jnp.asarray(self.inverse_temperature)) * result.predictions
        return jax.nn.log_softmax(logits, axis=1)[jnp.arange(choices.shape[0]), choices]

    def _sgd_loss_fn(self, params, choices, rewards):
        return -jnp.sum(self._choice_log_probs(params, choices, rewards))

    def _store_sgd_params(self, params):
        for key, attr in self._sgd_param_attrs.items():
            if key in params:
                setattr(self, attr, float(params[key]))

    def _finalize_sgd(self, choices, rewards):
        self._result = self._run({}, choices, rewards)
        self.log_likelihood_ = float(jnp.sum(self._choice_log_probs({}, choices, rewards)))
        self._populate_uncertainty(choices)

    def _populate_uncertainty(self, choices):
        from state_space_practice.behavioral_uncertainty import (
            categorical_entropy, compute_surprise,
        )
        r = self._result
        self.predicted_option_values_ = r.predictions
        self.filtered_option_values_ = r.posterior_means
        self.predicted_option_variances_ = r.predicted_variances
        self.filtered_option_variances_ = r.posterior_variances
        self.volatility_ = r.volatility
        self.learning_rate_ = r.learning_rate
        self.volatility_prediction_error_ = r.volatility_prediction_error
        probs = jax.nn.softmax(self.inverse_temperature * r.predictions, axis=1)
        self.predicted_choice_entropy_ = categorical_entropy(probs)
        self.surprise_ = compute_surprise(probs, choices)

    # --- public API mirroring MultinomialChoiceModel -------------------------
    def predictive_log_likelihood(self, choices, rewards) -> float:
        """Sum over trials of ``log p(c_t | c_{<t}, r_{<t})`` at the current parameters.

        Causal, so on a prefix it equals the corresponding partial sum of the
        full sequence; the model-comparison harness uses the difference of
        full-sequence and prefix values as a held-out log-likelihood.
        """
        (choices_arr, rewards_arr), _ = self._validate_inputs(choices, rewards)  # no state change
        return float(jnp.sum(self._choice_log_probs({}, choices_arr, rewards_arr)))

    def choice_probabilities(self) -> Array:
        """Predicted choice probabilities ``softmax(beta * values_t)``, shape (T, K).

        Unlike :meth:`MultinomialChoiceModel.choice_probabilities`, which uses
        smoothed latent values, these are the causal predictions the fitted
        agent made before each choice.
        """
        self._check_fitted("choice_probabilities")
        return jax.nn.softmax(self.inverse_temperature * self._result.predictions, axis=1)

    @property
    def n_free_params(self) -> int:  # sum of the four learn_* flags
    def bic(self) -> float:          # -2 LL + n_free_params * log(n_trials), as multinomial_choice.py:1182-1190
    def compare_to_null(self) -> dict:  # uniform 1/K null, keys as multinomial_choice.py:1192-1215
    def summary(self) -> str
    def __repr__(self) -> str
    @property
    def is_fitted(self) -> bool: return self._result is not None
    def _check_fitted(self, method): raise NotFittedError(...) if not fitted
```

`_prepare_sgd_data` must not mutate the model before validation passes
(see `_bind_covariates`, `covariate_choice.py:735-762`); `_validate_inputs`
is the shared pure validator it and `predictive_log_likelihood` call.

---

## 5. Block-change bandit simulator <a id="vkf-simulator"></a>

A VKF agent playing a bandit whose reward probabilities change every
`block_length` trials (the epoch structure of the Frank-lab spatial bandit,
Comrie et al. 2024). Ground truth is the agent's own VKF state. Implemented
as one `lax.scan` with `jax.random` (as `simulate_switching_choice_data`,
`switching_choice.py:1318-1352`), calling `volatile_kalman_step` so the
simulator and the filter share one recursion.

```python
class SimulatedVolatileChoiceData(NamedTuple):
    """Bandit data generated by a VKF agent under block-wise reward contingencies.

    Attributes
    ----------
    choices : Array, shape (n_trials,)
    rewards : Array, shape (n_trials,)
        Reward of the chosen option (0/1).
    reward_probs : Array, shape (n_trials, n_options)
        Task ground truth: reward probability of every option on every trial.
    change_points : Array, shape (n_blocks - 1,)
        Trial indices at which ``reward_probs`` changes.
    true_values : Array, shape (n_trials, n_options)
        Agent's VKF predictions used for the choice on each trial.
    true_volatility : Array, shape (n_trials,) or (n_trials, n_options)
    true_learning_rate : Array, shape (n_trials, n_options)
    true_probs : Array, shape (n_trials, n_options)
        Softmax choice probabilities of the agent.
    """


def simulate_volatile_bandit_data(
    n_trials: int = 400,
    n_options: int = 3,
    block_length: int = 100,
    block_reward_probs: ArrayLike | None = None,   # (n_blocks, K); default: 0.8 on option (b mod K), 0.2 elsewhere
    volatility_learning_rate: float = 0.2,
    initial_volatility: float = 0.1,
    observation_noise: float = 1.0,
    inverse_temperature: float = 2.0,
    observation: ObservationModel = "binary",
    shared_volatility: bool = True,
    seed: int = 42,
) -> SimulatedVolatileChoiceData:
    key = jax.random.PRNGKey(seed)
    n_blocks = -(-n_trials // block_length)
    if block_reward_probs is None:
        block_reward_probs = 0.2 + 0.6 * jax.nn.one_hot(jnp.arange(n_blocks) % n_options, n_options)
    reward_probs = jnp.repeat(jnp.asarray(block_reward_probs), block_length, axis=0)[:n_trials]
    change_points = jnp.arange(block_length, n_trials, block_length)
    v_init = initial_volatility if shared_volatility else jnp.full(n_options, initial_volatility)
    init = VolatileKalmanState(jnp.zeros(n_options), jnp.full(n_options, observation_noise), jnp.asarray(v_init))

    def _trial(state, inputs):
        key_t, p_t = inputs
        k_choice, k_reward = jax.random.split(key_t)
        probs = jax.nn.softmax(inverse_temperature * state.mean)
        choice = jax.random.categorical(k_choice, jnp.log(probs))
        reward = jax.random.bernoulli(k_reward, p_t[choice]).astype(float)
        mask = jax.nn.one_hot(choice, n_options)
        new_state, out = volatile_kalman_step(
            state, jnp.full(n_options, reward), mask,
            jnp.asarray(volatility_learning_rate), jnp.asarray(observation_noise),
            observation, shared_volatility,
        )
        return new_state, (choice, reward, probs, out[0], out[1], out[2])

    _, (choices, rewards, probs, values, volatility, lr) = jax.lax.scan(
        _trial, init, (jax.random.split(key, n_trials), reward_probs)
    )
    return SimulatedVolatileChoiceData(choices, rewards, reward_probs, change_points,
                                       values, volatility, lr, probs)
```

Consistency check used by the tests: `volatile_kalman_filter(rewards
broadcast, mask = one_hot(choices), same parameters)` reproduces
`true_values` / `true_volatility` to `1e-10`.

---

## 6. Hybrid: volatility as the process noise of the Laplace-EKF choice filter <a id="hybrid-filter"></a>

`CovariateChoiceModel`'s filter (`covariate_choice.py:323-384`) is

```
carry = (filt_mean, filt_cov, total_ll)                          # :343
pred_mean = A @ filt_mean + input_gain @ u_t                     # :347
pred_cov  = A @ filt_cov @ A.T + Q,   Q = eye(k_free) * process_noise   # :348, :339
post_mean, post_cov, ll, gap = _softmax_update_core(pred_mean, pred_cov, choice_t, K, beta, obs_offset)  # :354-361
```

The volatile variant keeps every line and substitutes:

| Reference VKF quantity | Hybrid quantity (latent dimension `d = K − 1`) |
|---|---|
| `v_{t-1}` (process variance) | `v`, carried scalar; `Q_t = eye(d) * v` replaces the constant `Q` |
| `w_{t-1} + v_{t-1}` | `pred_cov = A P_{t-1} A^T + Q_t` (already computed) |
| Gaussian update `m_t, w_t` | the Laplace-EKF softmax update's `post_mean, post_cov` (unchanged) |
| `m_t − m_{t-1}` | dynamics residual `r_t = post_mean − pred_mean = m_t − A m_{t-1} − B u_t` (the covariate-driven change is not volatility) |
| `w_{t-1,t} = (1 − k) w_{t-1}` | `C_t = J_t P_t`, `J_t = P_{t-1} A^T pred_cov^{-1}` — the RTS gain/cross-covariance already used at `covariate_choice.py:409-412`; with `A = 1`, `d = 1` this is `w_{t-1} (w+v)^{-1} w_t = (1 − k) w_{t-1}` |
| `E[(x_t − x_{t-1})²]` | `E‖x_t − A x_{t-1} − B u_t‖² / d = (‖r_t‖² + tr P_t + tr(A P_{t-1} A^T) − 2 tr(A C_t)) / d` — the per-trial summand of the process-noise M-step (`multinomial_choice.py:1059-1061`) evaluated with filtered instead of smoothed moments |
| Eq 13 | `v ← v + λ (E‖·‖²/d − v)` via `volatile_kalman._volatility_update` |

So `dynamics="volatile"` is an online, exponentially weighted (rate `λ`)
version of the EM process-noise update: with `λ = 0` it *is* the fixed-`q`
filter (`v ≡ v0`), and the smoother is unchanged because
`_rts_smoother_pass_with_predictions` (`covariate_choice.py:387-426`)
consumes the stored `pred_cov` (which already contains `Q_t`).

Positivity: `tr P_t + tr(A P_{t-1} A^T) − 2 tr(A C_t)` is the trace of
`Cov(x_t − A x_{t-1} | c_{1:t})` under the Gaussian approximation, hence ≥ 0
(scalar check: `(1−k) v + k w_{t-1}`), so `v_next = (1 − λ) v + λ · (≥ 0) > 0`.

```python
class VolatileChoiceFilterResult(NamedTuple):
    """`ChoiceFilterResult` fields plus the volatility trajectory.

    Attributes
    ----------
    filtered_values, filtered_covariances, predicted_values,
    predicted_covariances, marginal_log_likelihood
        As in :class:`multinomial_choice.ChoiceFilterResult`.
    volatility : Array, shape (n_trials,)
        Process variance ``v_{t-1}`` used for the prediction at trial t.
    volatility_prediction_error : Array, shape (n_trials,)
        ``E||x_t - A x_{t-1} - B u_t||^2 / (K-1) - v_{t-1}``.
    """
    filtered_values: Array
    filtered_covariances: Array
    predicted_values: Array
    predicted_covariances: Array
    marginal_log_likelihood: Array
    volatility: Array
    volatility_prediction_error: Array


def _volatility_prediction_error(
    filt_cov_prev: Array, pred_mean: Array, pred_cov: Array,
    post_mean: Array, post_cov: Array, transition_matrix: Array, volatility: Array,
) -> Array:
    """VKF Eq 13 bracket for a ``(K-1)``-dimensional latent with dynamics ``A``.

    ``E||x_t - A x_{t-1} - b_t||^2 / (K-1) - v_{t-1}`` under the filter's
    Gaussian approximation, with the lag-one cross-covariance
    ``C_t = P_{t-1} A^T P_pred^{-1} P_t``. For a scalar latent, ``A = 1`` and a
    linear-Gaussian update this equals ``(m_t - m_{t-1})^2 + w_t + w_{t-1}
    - 2 (1 - k_t) w_{t-1} - v_{t-1}`` of the reference code.
    """
    d = post_mean.shape[0]
    residual = post_mean - pred_mean
    gain = psd_solve(pred_cov, transition_matrix @ filt_cov_prev).T  # P_{t-1} A^T P_pred^{-1}
    cross_cov = gain @ post_cov                                       # Cov(x_{t-1}, x_t | c_{1:t})
    propagated = transition_matrix @ filt_cov_prev @ transition_matrix.T
    expected_sq_change = (
        residual @ residual + jnp.trace(post_cov) + jnp.trace(propagated)
        - 2.0 * jnp.trace(transition_matrix @ cross_cov)
    ) / d
    return expected_sq_change - volatility


@partial(jax.jit, static_argnames=("n_options",))
def _volatile_covariate_choice_filter_jit(
    choices, n_options, covariates, input_gain, obs_covariates, obs_weights,
    volatility_learning_rate, initial_volatility, inverse_temperature, decay,
    init_mean, init_cov,
) -> VolatileChoiceFilterResult:
    """Laplace-EKF choice filter whose process variance is a tracked volatility."""
    k_free = n_options - 1
    eye = jnp.eye(k_free)
    A = eye * decay

    def _step(carry, inputs):
        filt_mean, filt_cov, v, total_ll = carry
        choice_t, u_t, z_t = inputs
        pred_mean = A @ filt_mean + input_gain @ u_t
        pred_cov = A @ filt_cov @ A.T + eye * v
        obs_offset = obs_weights @ z_t
        post_mean, post_cov, ll, newton_gap = _softmax_update_core(
            pred_mean, pred_cov, choice_t, n_options, inverse_temperature,
            obs_offset=obs_offset,
        )
        delta_v = _volatility_prediction_error(
            filt_cov, pred_mean, pred_cov, post_mean, post_cov, A, v
        )
        v_next = _volatility_update(v, volatility_learning_rate, delta_v)
        total_ll = total_ll + ll
        return (post_mean, post_cov, v_next, total_ll), (
            post_mean, post_cov, pred_mean, pred_cov, v, delta_v, newton_gap
        )

    init_carry = (init_mean, init_cov, jnp.asarray(initial_volatility), jnp.array(0.0))
    (_, _, _, marginal_ll), (filt_vals, filt_covs, pred_vals, pred_covs, vols, dvs, gaps) = (
        jax.lax.scan(_step, init_carry, (choices, covariates, obs_covariates))
    )
    _warn_if_newton_unconverged(gaps, "volatile_covariate_choice_filter")
    return VolatileChoiceFilterResult(filt_vals, filt_covs, pred_vals, pred_covs,
                                      marginal_ll, vols, dvs)
```

Public wrappers `volatile_covariate_choice_filter(choices, n_options,
covariates=None, input_gain=None, obs_covariates=None, obs_weights=None,
volatility_learning_rate=0.1, initial_volatility=0.01,
inverse_temperature=1.0, decay=1.0, init_mean=None, init_cov=None)` (same
coercions as `covariate_choice_filter`, `covariate_choice.py:265-320`, plus
`0 <= λ < 1`, `v0 > 0`) and `volatile_covariate_choice_smoother(...)`, which
runs the volatile filter and then the existing backward pass. The tail of
`covariate_choice_smoother` (`covariate_choice.py:464-487`: build `A`, call
`_rts_smoother_pass_with_predictions`, concatenate the last filtered state,
build `ChoiceSmootherResult`) is factored into `_smooth_filter_result(filt, A)`
so both smoothers call it; the fixed path's outputs must be unchanged
bit-for-bit.

`_covariate_choice_filter_jit` itself is **not modified**: the fixed path
keeps its own compiled function, which is what guarantees bit-identical
default behaviour.

---

## 7. Hybrid model hooks on `CovariateChoiceModel` <a id="hybrid-model-hooks"></a>

Additive keywords on `CovariateChoiceModel.__init__` (`covariate_choice.py:566-616`):

```python
dynamics: Literal["fixed", "volatile"] = "fixed",
init_volatility_learning_rate: float = 0.1,
learn_volatility_learning_rate: bool = True,
```

Semantics in volatile mode: **`process_noise` is the initial volatility
`v0`** and `learn_process_noise` decides whether it is learned (the POSITIVE
transform already exists at `multinomial_choice.py:1012-1014`); the new
scalar `volatility_learning_rate` (UNIT_INTERVAL) is the only added
parameter. Hooks:

| Hook (existing location) | Change |
|---|---|
| `__repr__` (`:618-626`) | append `lambda=...` when volatile |
| `_filter_kwargs` (`:630-639`) | add `volatility_learning_rate` when volatile; the base `process_noise` entry is passed as `initial_volatility` |
| `_run_filter` / `_run_smoother` (`:641-648`) | dispatch to `volatile_covariate_choice_filter` / `_smoother` when volatile |
| `fit` (`:764-803`) | volatile: `raise ValueError("dynamics='volatile' has no closed-form M-step for the volatility learning rate; use fit_sgd")` before `_bind_covariates` |
| `_build_param_spec` (`:847-858`) | add `params["volatility_learning_rate"]`, `spec[...] = UNIT_INTERVAL` when volatile and learnable |
| `_sgd_loss_fn` (`:860-895`) | volatile: call `_volatile_covariate_choice_filter_jit(..., _param("volatility_learning_rate", ...), _param("process_noise", ...), beta, decay, zeros, eye)` |
| `_store_sgd_params` (`:897-904`) | store `volatility_learning_rate` as float |
| `n_free_params` (`:938-951`) | `+1` when volatile and `learn_volatility_learning_rate` |
| `_populate_uncertainty` (inherited, `multinomial_choice.py:772-814`) | override: `super()`, then when volatile run `_run_filter` once more and set `volatility_` `(T,)`, `volatility_prediction_error_` `(T,)`, and `learning_rate_` `(T, K-1) = 1 - diag(P_t) / diag(P_pred_t)` (fraction of prior variance removed by the choice; equals `k_t` for a scalar Gaussian update) |
| `summary` (`:957-968`) | add a `volatility_learning_rate` row when volatile |

Also added to `covariate_choice.py` (used by phase 2 tests and by the
comparison harness):

```python
def signed_reward_covariates(choices: ArrayLike, rewards: ArrayLike, n_options: int) -> Array:
    """Lagged signed-outcome covariates for a bandit, shape (n_trials, n_options).

    ``u[t, k] = (2 r_{t-1} - 1)`` if option ``k`` was chosen on trial ``t-1``
    and 0 otherwise; row 0 is zero. Follows the module convention that
    ``covariates[t]`` drives the transition ``x_{t-1} -> x_t``. With a free
    input gain ``B`` (shape ``(K-1, K)``) this expresses Rescorla-Wagner-like
    updates of the relative values, including the effect of rewarding the
    reference option (column 0).
    """
    chosen = jax.nn.one_hot(jnp.asarray(choices), n_options)
    signed = chosen * (2.0 * jnp.asarray(rewards, dtype=float) - 1.0)[:, None]
    return jnp.concatenate([jnp.zeros((1, n_options)), signed[:-1]], axis=0)
```

---

## 8. Identifiability gate <a id="identifiability-gate"></a>

The hybrid adds a second knob (`λ`) on how fast values move; the inverse
temperature `β` sets how sharply values map to choices. Both can trade off
(a faster-moving value with a smaller `β` produces similar choice
probabilities), so before the hybrid is used for scientific claims the
fitted model's parameter Hessian must show no near-null direction mixing
`{volatility_learning_rate, process_noise}` with `inverse_temperature`.

Primary tool: `.identifiability_report()` from
`docs/plans/identifiability-diagnostics/` (see overview Dependency policy);
the gate test asserts, using that report's own null-direction criterion,
that no flagged direction has loadings on both groups.

Interim (only while that plan has not landed; replaced by the report when it
does — do not keep both): finite-difference Hessian of the unconstrained SGD
loss at the fitted optimum, then an eigen-analysis:

```python
def _unconstrained_hessian(model, choices, covariates):
    params, spec = model._build_param_spec()
    unc = transform_to_unconstrained(params, spec, include_non_trainable=False)
    keys = sorted(unc)
    flat0 = np.array([float(unc[k]) for k in keys])

    def loss(flat):
        u = {k: jnp.asarray(x) for k, x in zip(keys, flat)}
        return float(model._sgd_loss_fn(transform_to_constrained(u, spec), choices))

    h = 1e-3
    H = np.zeros((len(keys), len(keys)))
    for i in range(len(keys)):
        for j in range(i, len(keys)):
            e_i, e_j = np.eye(len(keys))[i] * h, np.eye(len(keys))[j] * h
            H[i, j] = H[j, i] = (
                loss(flat0 + e_i + e_j) - loss(flat0 + e_i - e_j)
                - loss(flat0 - e_i + e_j) + loss(flat0 - e_i - e_j)
            ) / (4 * h * h)
    return keys, H
```

Gate criterion: with `evals, evecs = eigh(H)`, the smallest eigenvalue must
exceed `1e-3 * evals.max()`; additionally, for every eigenvector with
`eval < 1e-2 * evals.max()`, the squared loading on
`{volatility_learning_rate, process_noise}` and on `inverse_temperature`
must not both exceed 0.2 (a near-flat direction that is a joint
volatility/`β` trade-off). Record the observed spectrum in the test
docstring. The gate runs on the simulated block-change data at `T = 600`;
its failure is a **scientific** blocker (documented in the CHANGELOG entry
and the class docstring as "use for claims only after the identifiability
report passes"), not a merge blocker for the code.

---

## 9. Model-comparison harness <a id="comparison-harness"></a>

Module: `src/state_space_practice/choice_model_comparison.py`. Three
candidates, one data interface `(choices, rewards)`:

| Candidate | Model | Data preparation |
|---|---|---|
| `"volatile"` | `VolatileKalmanChoiceModel(n_options)` | `fit_sgd(choices, rewards)` |
| `"fixed_q"` | `CovariateChoiceModel(n_options, n_covariates=n_options, init_decay=0.95, learn_decay=True)` | `fit_sgd(choices, covariates=signed_reward_covariates(...))` |
| `"switching"` | `SwitchingChoiceModel(n_options, n_discrete_states=2, n_covariates=n_options, init_inverse_temperatures=[0.5, 3.0], init_process_noises=[0.05, 0.005])` | `fit_sgd(choices, covariates=signed_reward_covariates(...))` |

```python
CandidateName = Literal["volatile", "fixed_q", "switching"]
CANDIDATES: tuple[CandidateName, ...] = ("volatile", "fixed_q", "switching")


class FittedCandidate(NamedTuple):
    """One fitted candidate and its scores.

    Attributes
    ----------
    name : str
    model : object
    log_likelihood : float
        Training-sequence log-likelihood at the fitted parameters.
    n_free_params : int
    bic : float
        ``-2 log_likelihood + n_free_params * log(n_train)``.
    heldout_log_likelihood : float
        ``log p(c_{n_train:T} | c_{<n_train}, rewards)`` at the fitted
        parameters (NaN when nothing is held out).
    """


def count_free_parameters(model) -> int:
    """Free parameters as the optimiser sees them.

    ``n_free_params`` when the model defines it, else the number of
    unconstrained coordinates of ``_build_param_spec`` (a row-stochastic
    ``S x S`` matrix counts ``S (S-1)``, etc.).
    """
    if hasattr(model, "n_free_params"):
        return int(model.n_free_params)
    params, spec = model._build_param_spec()
    unc = transform_to_unconstrained(params, spec, include_non_trainable=False)
    return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(unc)))


def sequence_log_likelihood(name, model, choices, rewards, n_options) -> float:
    """``sum_t log p(c_t | c_{<t}, r_{<t})`` at the model's current parameters.

    Uses the public filter functions with the fitted attributes; causal, so a
    prefix's value is the corresponding partial sum of the full sequence.
    """
    if name == "volatile":
        return model.predictive_log_likelihood(choices, rewards)
    u = signed_reward_covariates(choices, rewards, n_options)
    if name == "fixed_q":
        return float(covariate_choice_filter(
            choices, n_options, covariates=u, input_gain=model.input_gain_,
            process_noise=model.process_noise, inverse_temperature=model.inverse_temperature,
            decay=model.decay,
        ).marginal_log_likelihood)
    return float(switching_choice_filter(
        choices, n_options, n_discrete_states=model.n_discrete_states, covariates=u,
        input_gain=model.input_gain_, process_noises=model.process_noises_,
        inverse_temperatures=model.inverse_temperatures_, decays=model.decays_,
        discrete_transition_matrix=model.discrete_transition_matrix_,
        init_mean=model.init_mean_, init_cov=model.init_cov_,
    ).marginal_log_likelihood)


def fit_candidate(name, choices, rewards, n_options, *, num_steps=150) -> object:
    """Construct the candidate with the defaults above and ``fit_sgd`` it."""


def compare_candidates(
    choices, rewards, n_options, *, train_fraction=0.75, num_steps=150,
    candidates=CANDIDATES,
) -> list[FittedCandidate]:
    """Fit every candidate on the first ``train_fraction`` of trials and score it.

    BIC uses the training log-likelihood and ``n_train``; the held-out
    log-likelihood is ``sequence_log_likelihood(full) -
    sequence_log_likelihood(prefix)``, i.e. the predictive log-probability
    of the held-out trials given the training history (the filters are run
    over the whole sequence so the held-out trials see their true history).
    """
```

Winner selection: `argmin` BIC and `argmax` held-out LL. A confusion matrix
over generators (rows) and winners (columns) is what the slow test and the
script report.

---

## 10. Generative agents for the confusion matrix <a id="comparison-simulators"></a>

One task, three agents, one interface. Task: block-wise reward
probabilities as in §5. Agents:

| Agent | Generative process | Parameters (defaults) |
|---|---|---|
| `"volatile"` | §5 simulator | `λ = 0.3, v0 = 0.1, ω = 1, β = 2` (binary, shared) |
| `"fixed_q"` | `x_t = a x_{t-1} + B u_t + N(0, q I)`, `u_t = signed_reward_covariates`, `c_t ~ softmax(β [0, x_t])` | `a = 0.95, q = 0.005, β = 3`, `B[k-1, k] = +0.6`, `B[:, 0] = -0.6` (rewarding the reference option lowers every relative value) |
| `"switching"` | same with `s_t ~ Markov(Z)`, per-state `β_s, q_s` | `β = [0.5, 4.0], q = [0.05, 0.002], Z = 0.97 I + 0.03/2` |

```python
class SimulatedBanditData(NamedTuple):
    choices: Array          # (T,)
    rewards: Array          # (T,)
    reward_probs: Array     # (T, K)
    change_points: Array    # (n_blocks - 1,)
    true_probs: Array       # (T, K) agent's choice probabilities
    generator: str


def simulate_bandit_agent(agent: CandidateName, n_trials=400, n_options=3,
                          block_length=100, seed=0, **agent_params) -> SimulatedBanditData:
```

Implementation: `lax.scan` over trials; the carry holds the agent state
(`VolatileKalmanState`, or `(x, s)` for the latent agents) and the last
trial's `(choice, reward)` from which the current `u_t` is formed; rewards
`~ Bernoulli(reward_probs[t, choice])`. The `"volatile"` branch delegates to
`simulate_volatile_bandit_data` and repacks. Sanity tests: empirical reward
rate of each option within a block matches `reward_probs` (blocks with ≥ 50
pulls), `true_probs` rows sum to one, change points at multiples of
`block_length`.

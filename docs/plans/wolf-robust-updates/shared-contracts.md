# Shared contracts

[← back to PLAN.md](PLAN.md)

Contracts used by two or more phases. Each appears once; phases link in by anchor and must not weaken them.

- [`RobustWeight` protocol](#robustweight)
- [`IMQWeight` / `imq_weight`](#imq-weight)
- [`RobustOutput`](#robustoutput)
- [Return-arity rule](#return-arity)
- [The two objectives](#objectives)
- [Weight granularity](#weight-granularity)
- [Invariants](#invariants)

All new symbols live in `src/state_space_practice/utils.py` (imported by `kalman.py`, `switching_kalman.py` and `point_process_kalman.py`; `utils` imports none of them, so no cycles). `imq_weight` is also re-exported lazily from the package root.

<a id="robustweight"></a>

## `RobustWeight` protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class RobustWeight(Protocol):
    """Callable mapping a standardised residual to a weight in ``[0, 1]``.

    Instances are passed to the filters as *static* ``jax.jit`` arguments,
    so they must be hashable and compare equal by value (a frozen dataclass,
    not a lambda: a lambda hashes by identity and forces a recompilation
    on every call). ``__call__`` runs inside ``jax.lax.scan`` / ``jax.vmap``
    and must be a pure ``jax.numpy`` function of its argument.
    """

    def __call__(self, standardized_residual: Array) -> Array:
        """Weight for one observation vector.

        Parameters
        ----------
        standardized_residual : Array, shape (n,)
            Residual of the observation from its prior predictive, in
            standard-deviation units (see the per-path definitions below).

        Returns
        -------
        Array, shape ()
            Weight ``w`` in ``[0, 1]``; the log-likelihood is tempered by
            ``w**2`` (Duran-Martin et al. 2024, Eq. 15).
        """
```

Semantics the filters rely on: `w = 1` is the standard update; `w = 0` drops the observation. Boundedness (paper Thm 3.2) needs `sup w < ∞` and `sup w² ‖y‖ < ∞`; the IMQ satisfies both. A weight that violates them still runs, but the bounded-influence test in phase 1 pins the IMQ only.

<a id="imq-weight"></a>

## `IMQWeight` / `imq_weight`

```python
@dataclasses.dataclass(frozen=True)
class IMQWeight:
    """Inverse multi-quadratic weight ``(1 + max(0, ‖z‖² − core²) / c²)^{-1/2}``.

    With ``core=0`` this is the Mahalanobis-standardised IMQ weight of
    Duran-Martin et al. (2024), Eq. (18) ("WoLF-MD"); ``c`` is the soft
    threshold in standard-deviation units. With ``core > 0`` the weight is
    exactly 1 while ``‖z‖ <= core`` and decays as the IMQ beyond it, which
    removes the down-weighting of ordinary data (see the M-step bias notes
    in the docstring of ``kalman_maximization_step``).
    """

    c: float
    core: float = 0.0

    def __call__(self, standardized_residual: Array) -> Array:
        z = jnp.asarray(standardized_residual)
        excess = jnp.maximum(jnp.sum(z * z) - self.core**2, 0.0)
        return (1.0 + excess / self.c**2) ** -0.5


def imq_weight(c: float, core: float = 0.0) -> IMQWeight:
    """Build an :class:`IMQWeight`; ``c > 0``, ``core >= 0`` (validated with
    :func:`validate_scalar`). ``c`` is required: no default is defensible
    across observation dimensions (``‖z‖² ~ χ²_n`` under the model, so a
    useful ``c`` grows like ``√n``; ``c² = χ²_n(0.99)`` puts ``w = 1/√2`` at
    the 99 % quantile). The paper tunes ``c`` by Bayesian optimisation."""
```

Invariants: `IMQWeight(2.0) == IMQWeight(2.0)` and equal hashes (jit cache hit); `imq_weight(c=1e150)(z) == 1.0` exactly in float64 for any realistic `z` (used by the `c → ∞` tests); output dtype follows the input.

<a id="robustoutput"></a>

## `RobustOutput`

```python
class RobustOutput(NamedTuple):
    """Extra outputs of a robust (``robust_weight`` given) filter call.

    Attributes
    ----------
    objective : Array, shape ()
        Generalised-Bayes log evidence of the observations (the sum over
        time of the per-step terms defined in the "objectives" contract).
        This is what EM monitors and what ``fit_sgd`` maximises.
    weights : Array
        The weights used. Shape ``()`` for a single Gaussian update,
        ``(n_time,)`` for a Gaussian / switching filter run,
        ``(n_obs,)`` for a single GLM Laplace update and
        ``(n_time, n_obs)`` for a point-process filter run.
    """

    objective: Array
    weights: Array
```

`robust_objective` in the plan brief is `RobustOutput.objective`. The weights are returned because the M-steps need them and because they are the natural artifact-detection diagnostic; models keep them as `filter_robust_weights`.

<a id="return-arity"></a>

## Return-arity rule

Every function that gains `robust_weight` keeps its current return value **unchanged and bit-identical** when `robust_weight is None`. When a weight is given, exactly one element, a `RobustOutput`, is appended **after all existing outputs, including any other optional trailing outputs** (`n_line_search_failures`, `filtered_mean, filtered_cov`). Concretely:

| Function | `robust_weight=None` | `robust_weight=IMQWeight(...)` |
| --- | --- | --- |
| `kalman_measurement_update` | `(mean, cov, ll)` | `(mean, cov, ll, RobustOutput)` |
| `kalman_filter` | `(means, covs, ll)` | `(means, covs, ll, RobustOutput)` |
| `kalman_smoother` | `(sm, sc, scc, ll)` | `(sm, sc, scc, ll, RobustOutput)` |
| `switching_kalman_filter` | 7-tuple | 7-tuple + `RobustOutput` |
| `_point_process_laplace_update` / `glm_laplace_update` | `(mean, cov, ll[, n_failed])` | `(mean, cov, ll[, n_failed], RobustOutput)` |
| `stochastic_point_process_filter` | `(means, covs, ll)` | `(means, covs, ll, RobustOutput)` |
| `stochastic_point_process_smoother` | `(sm, sc, scc, ll[, fm, fc])` | `(sm, sc, scc, ll[, fm, fc], RobustOutput)` |
| `position_decoder_filter` / `_smoother` | `DecoderResult` | `DecoderResult` with `.robust` set (attribute, default `None`) |

`kalman_maximization_step` and `switching_kalman_maximization_step` take `robust_weights=` (an array of weights, plural) and their return values never change. With robust weights, Gaussian `kalman_maximization_step` also requires `previous_params` containing the E-step's measurement matrix and covariance; the switching function reuses its existing `previous_params` argument when estimating measurement parameters. Zero effective observation weight preserves those parameters exactly, separately from the dynamics occupancy gate.

Plain-Python wrappers (`kalman_filter`, `kalman_smoother`, `glm_laplace_update`, the point-process filter/smoother) get `@overload` stubs distinguishing `robust_weight: None` from `robust_weight: RobustWeight`, following the existing pattern at `point_process_kalman.py:1300-1331`. Jitted functions document the rule in their docstrings instead (their static type is erased by `jax.jit`).

<a id="objectives"></a>

## The two objectives

For observation `y_t` with prior predictive `N(ŷ_t, S_t)` and weight `w_t`:

- **`log_likelihood`** (unchanged meaning): the *unweighted* one-step predictive log density evaluated on the robust filter's predictive, `log N(y_t; ŷ_t, S_t)`, summed over time. Comparable across values of `c` and with `robust_weight=None` fits of the same data; it is *not* the quantity EM monitors when robust.
- **`RobustOutput.objective`**: the generalised-Bayes log evidence `Σ_t log ∫ q(x_t | y_{1:t−1}) p(y_t | x_t)^{w_t²} dx_t`. For the Gaussian path this is closed form ([designs.md](designs.md#g2-evidence)); for the point-process path it is the Laplace approximation with the log-likelihood tempered by `w²` ([designs.md](designs.md#p1-deviance)). It equals `log_likelihood` when every `w_t = 1` and each per-step term is `0` when `w_t = 0`. The weighted M-step is the exact maximiser of its EM lower bound at fixed weights.

Models report the objective as their fit history (`fit` return value, `log_likelihoods`, `log_likelihood_`, `bic`/`aic`) when robust, and say so in their docstrings.

<a id="weight-granularity"></a>

## Weight granularity

- **Gaussian path (phases 1–2): one weight per time step.** `z_t = L_t⁻¹ (y_t − ŷ_t)` with `L_t L_tᵀ = R_t` (Cholesky of the *measurement* covariance, not the innovation covariance; paper Eq. 18) and `ŷ_t = H m_{t|t−1}`. For the switching filter `ŷ_t` and `R_t` are the mixture-averaged prior predictive ([designs.md](designs.md#s1-shared-weight)).
- **Point-process path (phase 3): one weight per (time step, observation dimension).** `z_{t,n} = sign(y_{t,n} − μ_{t,n}) √d(y_{t,n}, μ_{t,n})` with `d` the family's unit deviance and `μ_{t,n}` the mean at the prior predictive `η = eta_func(m_{t|t−1})`; each `z_{t,n}` is passed to the weight as a length-1 vector. Rationale (bursts and sorting errors are per-neuron; the block-diagonal fast path can only see one neuron at a time) and the Pearson-vs-deviance numbers are in [designs.md](designs.md#p1-deviance). Recommended setting for spikes: `imq_weight(c, core=3.0)`.

<a id="invariants"></a>

## Invariants

1. **Weights are constants for autodiff.** Every filter wraps the weight computation in `jax.lax.stop_gradient`. The gradient of `RobustOutput.objective` is therefore the gradient of the fixed-weight objective, matching the M-step's treatment. Do not remove this: without it gradient descent has a spurious direction (shrink R so every `w → 0` and the objective `→ 0`).
2. **Weights are evaluated at the prior predictive** (before any Fisher-scoring iteration, after any prediction-side modification such as the position decoder's track-penalty downdate), never at the posterior. This is what makes the influence bounded.
3. **The `None` branch is the pre-existing code**, not a `w = 1` special case of the new code, so bit-identity holds by construction; the new code is reached only through a static Python branch on `robust_weight is None`.
4. **Static, not traced.** `robust_weight` is in `static_argnames` of every jitted function it passes through, and is keyword-only everywhere.
5. **Finite at `w = 0`.** No formula divides by `w`; the Gaussian update uses `S̃ = w² H P Hᵀ + R`, and objectives use the forms in designs.md. In the M-step, zero total squared observation weight preserves the previous H/R; safe denominators and solve operands prevent invalid intermediate arithmetic, including under `jit`/`vmap`. Dynamics remain eligible for update.
6. **Siblings change together.** `_point_process_laplace_update` and `glm_laplace_update`; the dense and block-diagonal point-process cores; the three oscillator `_sgd_loss_fn`s.

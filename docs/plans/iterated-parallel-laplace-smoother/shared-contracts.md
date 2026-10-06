# Shared contracts

[← back to PLAN.md](PLAN.md)

Contracts used by two or more phases. Each appears once; phases link by anchor and must not
weaken them.

- [Public keywords and return layout](#public-keywords)
- [`IteratedSmootherDiagnostics`](#iterated-smoother-diagnostics)
- [Pseudo-observation site contract](#pseudo-observation-site-contract)
- [Gauss–Newton core interface](#gauss-newton-core-interface)
- [Linear-pass interface](#linear-pass-interface)
- [Warning contract](#warning-contract)

## Public keywords

Added, keyword-only, to `stochastic_point_process_smoother` (`point_process_kalman.py`)
and `position_decoder_smoother` (`position_decoder.py`); mirrored as constructor arguments
(stored as attributes) on `PointProcessModel`, `PlaceFieldModel` and `PositionDecoder`.

```python
*,
n_iterations: int = 1,          # phase 1 (phase 3 for the decoder)
parallel: bool = False,         # phase 2 (phase 3 for the decoder)
convergence_tol: float = 1e-6,  # phase 1
return_diagnostics: bool = False,  # phase 1
```

Semantics (fixed; do not weaken):

- `n_iterations` is the total number of smoother passes that produce a path. Validated with
  `utils.validate_int(..., positive=True)`.
  - `parallel=False`: pass 1 is **today's** one-pass Laplace-EKF filter + RTS smoother
    (with its own per-bin `max_newton_iter` Fisher iterations); passes 2..`n_iterations`
    are Armijo-damped Gauss–Newton passes linearised at the current smoothed path. So
    `n_gauss_newton_passes = n_iterations - 1`.
  - `parallel=True`: no Laplace-EKF pass. The iteration starts from the nominal trajectory
    `x̂_t = init_mean` (broadcast over time; `init_position` for the decoder) and all
    `n_iterations` passes are damped Gauss–Newton passes run through associative scans.
    `n_gauss_newton_passes = n_iterations`.
  - Whenever `n_gauss_newton_passes >= 1`, one extra *final* linear pass is run at the
    accepted path; it supplies the returned covariances, cross-covariances, evidence and
    the un-taken Gauss–Newton step used for the convergence test. It never moves the path.
- `n_iterations=1, parallel=False` executes exactly the code that exists today and returns
  arrays that are `assert_array_equal`-identical to today's. No final pass is run.
- `convergence_tol`: `converged := max_abs_update <= convergence_tol * (1 + max|x̂|)`
  where `max_abs_update` is the infinity norm of the un-taken final Gauss–Newton step.
  Validated with `validate_scalar(..., positive=True)`.
- `return_diagnostics=True` appends one `IteratedSmootherDiagnostics` as the **last**
  element of the returned tuple (after `filtered_mean, filtered_cov` when
  `return_filtered=True`). For `position_decoder_smoother` the return becomes
  `(DecoderResult, IteratedSmootherDiagnostics)`.
- `marginal_log_likelihood` semantics:
  - `n_iterations=1, parallel=False`: today's quantity (sum of per-bin sequential Laplace
    evidence terms).
  - otherwise: the Laplace evidence at the accepted path of the final pass
    (designs.md#laplace-evidence). With `include_laplace_normalization=False` it is the
    plug-in `Σ_t log p(y_t | η̂_t)` at that path.
- `filtered_mean, filtered_cov` (with `return_filtered=True`):
  - `parallel=False`: the iteration-0 Laplace-EKF filter output (a causal filter of the
    counts), identical to today's.
  - `parallel=True`: the filtered moments of the linear-Gaussian pseudo-model at the final
    linearisation. These are **not** a causal filter of the counts (the linearisation used
    all data); the docstring must say so.
- `return_block_covariances=True` keeps its meaning on the block path for every returned
  covariance (smoothed, cross, filtered).

Model attributes: `self.n_iterations`, `self.parallel` (validated in `__init__`), and after
a fit/decode `self.smoother_diagnostics_: IteratedSmootherDiagnostics | None` (included in
the EM snapshot/restore dictionaries).

## Iterated smoother diagnostics

Defined once in `point_process_kalman.py`; imported by `position_decoder.py` and (phase 4)
mapped onto `temporal_rate_gp.LaplaceRateResult`.

```python
class IteratedSmootherDiagnostics(NamedTuple):
    """Convergence record of the Gauss-Newton (iterated Laplace) smoother.

    Attributes
    ----------
    n_gauss_newton_passes : int
        Number of damped Gauss-Newton passes run (``n_iterations - 1`` after a
        Laplace-EKF initialisation, ``n_iterations`` from a nominal trajectory).
    log_posterior : Array, shape (n_gauss_newton_passes + 1,)
        Joint log posterior (up to constants) of the accepted path before the
        first Gauss-Newton pass and after each one. Non-decreasing by
        construction of the line search.
    step_sizes : Array, shape (n_gauss_newton_passes,)
        Accepted Armijo step length per pass (1 = full Gauss-Newton step,
        0 = rejected, path kept).
    max_abs_step : Array, shape (n_gauss_newton_passes,)
        Infinity norm of the taken update ``step * direction`` per pass.
    max_abs_update : Array, shape ()
        Infinity norm of the un-taken Gauss-Newton step at the final path
        (``nan`` when no final pass was run).
    converged : Array, shape ()
        ``max_abs_update <= convergence_tol * (1 + max|path|)``.
    n_nonfinite_merit : Array, shape ()
        Passes whose log posterior was not finite, so the full step was taken
        (int32).
    n_unaccepted_steps : Array, shape ()
        Passes in which no trial step passed the Armijo test, so the path was
        kept (int32).
    """
```

Block-diagonal path aggregation: `log_posterior` is summed over neurons; `step_sizes` is
the per-pass **minimum** over neurons; `max_abs_step` and `max_abs_update` are maxima over
neurons; `converged` is the logical AND; the counts are sums. All leaves are JAX arrays so
the tuple can be returned from jitted code; `n_gauss_newton_passes` is a Python `int`.

## Pseudo-observation site contract

For time bin `t`, linearisation point `x̂_t`, observation vector `y_t` (shape
`(n_obs,)`), and a `GLMFamily` (`point_process_kalman.py:1215-1239`):

```
η̂_t = log_conditional_intensity(Z_t, x̂_t)            # (n_eta,)   linear predictor
J_t  = ∂η/∂x |_{x̂_t}                                   # (n_eta, d) (= Z_t for the default linear intensity)
μ̂_t = family.mean(η̂_t)                                # (n_eta,)
w_t  = family.fisher_weight(η̂_t, μ̂_t)                 # (n_eta,)   Fisher weight (Poisson: μ̂_t)
r_t  = family.score(y_t, η̂_t, μ̂_t)                    # (n_eta,)   likelihood score; y_t − μ̂_t only if score is None
G_t  = J_tᵀ diag(w_t) J_t                              # (d, d)     information matrix (PSD)
g_t  = G_t x̂_t + J_tᵀ r_t                              # (d,)       information vector
```

`(G_t, g_t)` is the *only* form in which a pseudo-observation enters a linear pass
(sequential update, parallel filter elements, evidence). `W_t^{-1}` and the working
response `ỹ_t = η̂_t + r_t / w_t` are never formed. Equivalent statement in the usual
notation: `ỹ_t ~ N(c_t + J_t x_t, W_t^{-1})` with offset `c_t = η̂_t − J_t x̂_t`, since
`J_tᵀ W_t (ỹ_t − c_t) = g_t`.

Composition rules other plans rely on:

- **Masked bin** (masks plan): `G_t = 0`, `g_t = 0`, and the bin's log-likelihood term is
  `0`. Nothing else changes: the sequential update reduces to the prediction and the
  parallel element to the pure transition element.
- **Additional pseudo-observations** (decoder track penalty, phase 3): sites add:
  `G_t ← G_t + G_t^{pen}`, `g_t ← g_t + g_t^{pen}`, and the merit gains the corresponding
  exact log-prior term. Evidence evaluates the combined quadratic site at the
  filtered mean, including its Taylor constant and both correction terms;
  adding only the extra log-prior value at the smoothed path is insufficient.
- **Non-canonical family** (NB/ZIG plan): preserve the caller's `family` in every
  pass and in the merit/evidence calculations. `r_t` is the family score, with
  the canonical residual used only when `score is None`; `n_eta` may differ
  from `n_obs`. Fisher information is expected information. Combined with the
  correct score and prior gradient, the step is an ascent direction; the line
  search guarantees monotonicity. A supplied family always uses the dense path.

## Gauss–Newton core interface

Private, in `point_process_kalman.py`. Single-problem (one state vector, one design); the
block-diagonal path `vmap`s it over per-neuron problems, the parallel block path
`lax.map`s it. Not jitted itself (callers jit); the family and intensity callables may
close over tracers (phase 4 needs this for a traced baseline log-rate).

```python
class _LinearisationProblem(NamedTuple):
    init_mean: Array               # (d,)
    init_cov: Array                # (d, d)
    transition_matrix: Array       # (d, d)
    process_cov: Array             # (d, d)
    design_matrix: Array           # (n_time, ...) as log_conditional_intensity expects
    counts: Array                  # (n_time, n_obs) float
    log_intensity: Callable[[Array, Array], Array]        # (Z_t, x) -> (n_eta,)
    grad_log_intensity: Callable[[Array, Array], Array]   # (Z_t, x) -> (n_eta, d)
    family: GLMFamily
    include_laplace_normalization: bool
    extra_sites: Callable[[Array], tuple[Array, Array, Array]] | None
    # phase 3: x̂_t -> (G_t^extra, g_t^extra, log_prior_extra(x̂_t)); None otherwise


def _iterated_laplace_smoother_core(
    problem: _LinearisationProblem,
    initial_path: Array,                 # (n_time, d)
    *,
    n_gauss_newton_passes: int,          # static, >= 1
    convergence_tol: float,
    parallel: bool,                      # static; phase 1 supports False only
) -> tuple[_LinearPassOutput, IteratedSmootherDiagnostics]:
```

Invariants: `log_posterior` non-decreasing; the returned `_LinearPassOutput` is evaluated
at the accepted path (never at an un-taken target); `step` choices are under
`jax.lax.stop_gradient`; no host callbacks inside (warnings are raised by the public
wrappers).

## Linear-pass interface

```python
class _LinearPassOutput(NamedTuple):
    smoother_mean: Array        # (n_time, d)   Gauss-Newton target (posterior mean of the pseudo-model)
    smoother_cov: Array         # (n_time, d, d)
    smoother_cross_cov: Array   # (n_time - 1, d, d)  Cov(x_t, x_{t+1} | ỹ)
    filtered_mean: Array        # (n_time, d)
    filtered_cov: Array         # (n_time, d, d)
    evidence: Array             # ()  Laplace evidence at the linearisation path (designs.md#laplace-evidence)
    plugin_log_likelihood: Array  # ()  Σ_t log p(y_t | η̂_t) at the linearisation path


def _linearised_smoother_pass(
    problem: _LinearisationProblem, path: Array, *, parallel: bool
) -> _LinearPassOutput:
```

`parallel=False` (phase 1): information-form forward `lax.scan` +
`kalman.rts_backward_scan`. `parallel=True` (phase 2): `kalman._parallel_information_filter`
+ `kalman.parallel_kalman_smoother` + vectorised evidence terms. Both branches must agree to
rtol 1e-9 on the same `(problem, path)`; that agreement is a phase-2 test.

## Warning contract

Host-side, from the public wrappers only, skipped when the diagnostics contain tracers
(`utils.contains_tracer`), exactly like `temporal_rate_gp._warn_laplace_diagnostics`
(`temporal_rate_gp.py:149-187`):

- Not converged (`n_gauss_newton_passes >= 1` and `not converged`):
  `StateSpaceWarning`, message
  `"{name}: the Gauss-Newton iteration did not converge in n_iterations={k} pass(es) (final relative update {u:.3g} > convergence_tol={tol:g}); the smoothed path, covariances and evidence are unconverged. Increase n_iterations."`
- Fallbacks (`n_nonfinite_merit > 0 or n_unaccepted_steps > 0`): `StateSpaceWarning`,
  message
  `"{name}: the Gauss-Newton line search fell back in {a} pass(es) with a non-finite log posterior (full step taken) and kept the path in {b} pass(es) where no step length passed the Armijo test. Check the dynamics and prior scales and enable float64."`
- `stacklevel` points at the user's call of the public function.

One warning per condition per call; never inside `jax.jit` / `jax.grad` (the SGD loss);
the models' `_finalize_sgd` / `_e_step` run eagerly and therefore do warn.

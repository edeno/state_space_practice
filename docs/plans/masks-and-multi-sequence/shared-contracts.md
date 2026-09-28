# Shared contracts

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Contracts that more than one phase implements against. Each appears once; phases
link in by anchor. "Do not weaken" clauses are load-bearing for later phases.

- [C1 — `obs_mask` argument](#c1--obs_mask-argument) (phases 1-5)
- [C2 — Masked-bin semantics: E-step and log-likelihood](#c2--masked-bin-semantics-e-step-and-log-likelihood) (phases 1-5)
- [C3 — Masked-bin semantics: M-step statistics](#c3--masked-bin-semantics-m-step-statistics) (phases 2-5)
- [C4 — `validate_observation_mask`](#c4--validate_observation_mask) (phases 1-3)
- [C5 — Backwards compatibility: "off" means the old code runs](#c5--backwards-compatibility-off-means-the-old-code-runs) (phases 1-5)
- [C6 — Multi-sequence inputs, padding and sequence lengths](#c6--multi-sequence-inputs-padding-and-sequence-lengths) (phases 4-5)
- [C7 — `state_space_practice.sequences` helper API](#c7--state_space_practicesequences-helper-api) (phases 4-5)
- [C8 — `SGDFittableMixin._n_timesteps` under masks and sequences](#c8--sgdfittablemixin_n_timesteps-under-masks-and-sequences) (phases 2-5)
- [C9 — Where batching lives (not in `run_em`)](#c9--where-batching-lives-not-in-run_em) (phases 4-5)

---

## C1 — `obs_mask` argument

Every public filter / smoother / fit entry point that gains mask support takes the
same keyword-only argument:

```python
obs_mask: ArrayLike | None = None
```

- Boolean array. `True` = the entry **was observed**; `False` = missing.
- Accepted shapes for a single sequence: `(n_time,)` (whole-bin mask, broadcast
  across the observation axis) or `(n_time, n_obs)` (per-entry mask). `n_obs` is
  `n_obs_dim` for Gaussian observations and `n_neurons` for spike observations
  (a 1-D `spike_indicator` of shape `(n_time,)` pairs with a mask of shape
  `(n_time,)`).
- For multi-sequence inputs ([C6](#c6--multi-sequence-inputs-padding-and-sequence-lengths))
  the same shapes with a leading `n_sequences` axis.
- Non-boolean dtypes are rejected (`ValueError`), including 0/1 integers: an
  explicit `mask.astype(bool)` at the call site is cheap and removes a whole
  class of "I passed the spike counts as the mask" mistakes.
- **Masked entries of `obs` / `spike_indicator` may hold any value, including
  NaN.** Implementations must therefore zero-fill with `jnp.where(mask, obs, 0.0)`
  before any arithmetic — never `obs * mask`, which propagates NaN. Value
  validation (finiteness, non-negative integer counts) applies to unmasked
  entries only.
- Name and semantics are fixed by cross-plan agreement (other plans call
  `kalman_filter(..., obs_mask=...)` etc.). Do not rename, do not invert the
  polarity.

Functions that gain the argument: `kalman_measurement_update` (trailing
positional-or-keyword, see phase 1 for why), `kalman_filter`, `kalman_smoother`,
`switching_kalman_filter`, `switching_kalman_maximization_step`,
`kalman_maximization_step`, `stochastic_point_process_filter`,
`stochastic_point_process_smoother`, `glm_laplace_update`, and the `fit` /
`fit_sgd` / `score` methods of `BaseModel` (oscillator family),
`PointProcessModel` and `PlaceFieldModel`.

`parallel_kalman_smoother` does **not** gain the argument: it consumes filtered
moments only (`kalman.py:1072-1216`) and never touches `H`, `R` or `y`. A masked
run is `parallel_kalman_smoother(*kalman_filter(..., obs_mask=m)[:2], A, Q)`.
Phase 1 documents this in its docstring and tests it against the masked oracle.

## C2 — Masked-bin semantics: E-step and log-likelihood

For every filter in scope the following hold exactly (not approximately), and
phase tests assert them:

1. **Partially masked bin (Gaussian):** the update conditions on the observed
   sub-vector `y_o` only, with `H_o = H[o, :]` and `R_oo = R[o, o]`. The
   log-likelihood contribution is `log N(y_o; H_o m_pred, H_o P_pred H_o^T + R_oo)`.
   Posterior moments equal dense Gaussian conditioning on the observed entries
   (the `tests/oracles.py` reference with masked rows dropped).
2. **Partially masked bin (point process):** masked neurons contribute nothing
   to the Fisher score, the Fisher information or the log-likelihood; the
   Laplace normaliser is unchanged (it involves only the state prior and the
   posterior precision).
3. **Fully masked bin:** a predict-only step. Posterior mean and covariance equal
   the one-step prediction (`A m, A P A^T + Q`, symmetrised), and the bin
   contributes **exactly 0** to the log-likelihood.
4. **Smoothers** need no mask: the RTS backward pass (`rts_backward_scan`,
   `_stochastic_point_process_smoother_backward`, the parallel smoother, the GPB
   smoothers) is observation-agnostic and consumes the masked filter output.
5. **Switching filters:** the per-pair log-likelihoods `log p(y_t | ..., S_{t-1}=i, S_t=j)`
   are the masked ones, so a fully masked bin leaves the discrete posterior at
   its one-step prediction (all pairs contribute 0) and `log_predictive = 0`.

Do not weaken (3): later phases rely on "padding bins contribute 0 to the LL"
for the multi-sequence LL identity.

## C3 — Masked-bin semantics: M-step statistics

The M-steps in scope maximise an expected complete-data log-likelihood. Under
masks the complete data are `(x_{0:T}, y^o, y^m_partial)`: latent states, the
observed entries, and the missing entries **of bins that have at least one
observed entry**. Consequences (derivation and code in
[designs D2](designs.md#d2--masked-gaussian-m-step-imputation-form)):

1. **Observation statistics (H, R):** a bin with no observed entry has weight
   `e_t = 0` and is excluded. A bin with some observed entries enters with weight
   1, its missing entries imputed by their conditional expectation given the
   observed entries and the state posterior under the E-step's `(H, R)`
   (Shumway & Stoffer's missing-data EM). The normaliser for `R` is
   `sum_t e_t`, not `n_time`.
2. **Dynamics statistics (A, Q, initial state):** unchanged by masks. Every bin
   in a sequence is a latent state whether or not it was observed, so all `T`
   transitions (`T` with the `x_0` prior, `T - 1` in the prior-on-`x_1`
   convention) enter. Only *padding* bins
   ([C6](#c6--multi-sequence-inputs-padding-and-sequence-lengths)) are excluded.
3. **Point-process GLM statistics** (`PlaceFieldModel._fit_stationary_glm`
   warm start): a masked `(bin, neuron)` pair has weight 0 in that neuron's
   score and Hessian.
4. **Discrete-state statistics** (`Z`, `pi_0`): unchanged by masks; padding
   transitions excluded as in (2).

The EM under these statistics is a valid EM for the observed-data likelihood
(each bin's complete-data augmentation marginalises back to the observed
entries) and is monotone in the exact observed-data log-likelihood; phase 2 tests
this against the dense oracle.

## C4 — `validate_observation_mask`

One canonicaliser, in `src/state_space_practice/utils.py`, used by every entry
point that accepts [C1](#c1--obs_mask-argument):

```python
def validate_observation_mask(
    obs_mask: ArrayLike,
    *,
    n_time: int,
    n_obs: int,
    name: str = "obs_mask",
) -> jax.Array:
    """Canonicalise an observation mask to a boolean ``(n_time, n_obs)`` array.

    ``True`` marks an observed entry. A 1-D ``(n_time,)`` mask marks whole
    time bins and is broadcast across the observation axis. Only the shape and
    the dtype are checked, both static, so the function is safe to call on a
    traced mask inside ``jax.jit`` / ``jax.grad`` / ``jax.vmap``.

    Parameters
    ----------
    obs_mask : ArrayLike, shape (n_time,) or (n_time, n_obs)
        Boolean observation indicator.
    n_time, n_obs : int
        Expected leading (time) and trailing (observation) sizes.
    name : str
        Argument name used in error messages.

    Returns
    -------
    jax.Array, bool, shape (n_time, n_obs)

    Raises
    ------
    ValueError
        If the dtype is not boolean or the shape is neither accepted form.
    """
    with jax.ensure_compile_time_eval():
        mask = jnp.asarray(obs_mask)
    if mask.dtype != jnp.bool_:
        raise ValueError(f"{name} must be a boolean array, got dtype {mask.dtype}.")
    if mask.shape == (n_time,):
        return jnp.broadcast_to(mask[:, None], (n_time, n_obs))
    if mask.shape != (n_time, n_obs):
        raise ValueError(
            f"{name} must have shape ({n_time},) or ({n_time}, {n_obs}), "
            f"got {mask.shape}."
        )
    return mask
```

Multi-sequence callers (phases 4-5) apply it per sequence through `jax.vmap`, or
call it with the leading axis folded into `n_time` after checking `ndim`.

## C5 — Backwards compatibility: "off" means the old code runs

Decided policy: additive keyword-only options; no signature breaks; no
deprecation phases.

Stronger than "same numbers": when `obs_mask is None` **and** the observations
have no leading sequence axis, **the pre-existing code path executes with no
added operations**. Implement every mask / batch feature behind a static Python
branch (`if obs_mask is None:` / `if observations.ndim == 2:`) that leaves the
existing expression untouched, so the "off" result is bit-for-bit identical and
`test_em_golden_regression.py` cannot move. Tests assert this with
`np.testing.assert_array_equal`, not `assert_allclose`.

Two weaker equalities are also tested and documented, and are *not* promised
bit-for-bit:

- an all-`True` mask equals no mask (`assert_array_equal` in the E-step, where
  the masked arithmetic is `x * 1.0`, `x + 0.0`, `where(True, x, 0)` — exact
  up to the sign of zero; `rtol=1e-12` in M-steps, which use different
  reductions);
- an `n_sequences == 1` batched input equals the 2-D input (`rtol=1e-12`:
  batched reductions have a different XLA reduction order).

## C6 — Multi-sequence inputs, padding and sequence lengths

- Observations with a leading sequence axis: `(n_sequences, n_time_max, n_obs)`
  (Gaussian), `(n_sequences, n_time_max, n_neurons)` spikes with
  `(n_sequences, n_time_max, ...)` design matrices / `(n_sequences, n_time_max, 2)`
  positions. Shorter sequences are right-padded to `n_time_max`.
- `obs_mask` carries the padding: padded bins are all-`False`. It has shape
  `(n_sequences, n_time_max)` or `(n_sequences, n_time_max, n_obs)`.
- **Sequence length** `len_s = 1 + (index of the last bin of sequence s with any
  observed entry)`. Bins `t >= len_s` are *padding*: they do not exist in the
  model. Bins `t < len_s` with no observed entry are *missing-data bins*: latent
  states that exist and connect their neighbours ([C2](#c2--masked-bin-semantics-e-step-and-log-likelihood) item 3,
  [C3](#c3--masked-bin-semantics-m-step-statistics) item 2). A trailing stretch of
  fully masked real bins is therefore treated as padding; that changes nothing
  in the observed-data likelihood (those latents marginalise out) and only
  speeds EM up, which is why the mask alone suffices and no separate
  `sequence_lengths` argument exists.
- Every sequence must have `len_s >= 1` (at least one observed entry); `fit`
  raises `ValueError` otherwise, naming the sequence.
- E-steps run over the full `n_time_max` for static shapes. Padding bins are
  predict-only and contribute 0 to the LL; their smoothed moments equal their
  filtered (= predicted) moments and never influence real bins (the RTS
  correction through a predict-only step is exactly zero).
- M-step weights: `bin_weight[s, t] = (t < len_s)` and
  `transition_weight[s, t] = (t + 1 < len_s)` for the transition `t -> t + 1`.
  Observation statistics use `e[s, t] = any(mask[s, t, :])` (a subset of the bin
  weight). Initial-state statistics always use `t = 0`, which exists for every
  sequence.
- The total log-likelihood is the sum over sequences. Stored posteriors keep the
  leading sequence axis (`smoother_mean` of shape `(n_sequences, n_time_max, n)`,
  etc.); models expose `sequence_lengths_` (int array, `(n_sequences,)`) after a
  multi-sequence fit so callers can slice off the padding.

## C7 — `state_space_practice.sequences` helper API

New module (phase 4) shared by every multi-sequence adopter. Pure functions, no
model knowledge.

```python
def pad_sequences(
    sequences: Sequence[ArrayLike],
    obs_masks: Sequence[ArrayLike] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Right-pad ragged sequences into one array and the observation mask.

    Returns ``(padded, obs_mask)`` with ``padded`` of shape
    ``(n_sequences, n_time_max, *trailing)`` (padding entries are zero) and
    ``obs_mask`` boolean ``(n_sequences, n_time_max, n_obs)`` -- the given
    per-sequence masks (or all-True) right-padded with False.
    """

def sequence_lengths(obs_mask: ArrayLike) -> jax.Array:
    """``1 + index of the last observed bin`` per sequence, shape (n_sequences,) int32.

    ``obs_mask`` is ``(n_sequences, n_time_max)`` or ``(n_sequences, n_time_max, n_obs)``.
    A sequence with no observed entry has length 0.
    """

def bin_weights(obs_mask: ArrayLike) -> jax.Array:
    """Float ``(n_sequences, n_time_max)``: 1.0 where ``t < sequence_lengths``."""

def transition_weights(obs_mask: ArrayLike) -> jax.Array:
    """Float ``(n_sequences, n_time_max - 1)``: 1.0 where ``t + 1 < sequence_lengths``."""

def observed_bin_weights(obs_mask: ArrayLike) -> jax.Array:
    """Float ``(n_sequences, n_time_max)``: 1.0 where any entry of the bin is observed."""

def n_observed_bins(obs_mask: ArrayLike | None, n_time: int) -> int:
    """Number of bins with at least one observed entry (``n_time`` when ``obs_mask`` is None).

    Sums over a leading sequence axis when present. Host-side (returns ``int``).
    """

def canonicalize_sequences(
    observations: ArrayLike, obs_mask: ArrayLike | None, *, n_obs: int
) -> tuple[jax.Array, jax.Array | None, bool]:
    """Add a leading sequence axis to 2-D inputs; validate 3-D ones.

    Returns ``(observations, obs_mask, was_batched)``. ``obs_mask`` is passed
    through :func:`validate_observation_mask` per sequence (None stays None).
    Raises ``ValueError`` if a batched input has a sequence with no observed
    entry, or if the mask's leading axis does not match.
    """

def map_over_sequences(fn: Callable[..., T], *args: Any) -> T:
    """``jax.vmap(fn)(*args)`` over the leading axis (the batching primitive).

    One place to switch the strategy: ``_SEQUENCE_MAP = "vmap"`` (default) or
    ``"map"`` (``jax.lax.map``, sequential, lower peak temporaries, ~10x slower
    on CPU in the phase-4 measurement). Not a public knob; see designs D5.
    """
```

Add `src/state_space_practice/sequences.py` to `[tool.mypy] files`
(`pyproject.toml:134-159`) when it lands.

## C8 — `SGDFittableMixin._n_timesteps` under masks and sequences

`SGDFittableMixin` divides the loss by `float(self._n_timesteps)`
(`sgd_fitting.py:327`, `:560-564`, `:586`) and rescales the reported LL by it
(`:658`, `:711`). Contract: **`_n_timesteps` is the number of bins with at least
one observed entry, summed over sequences** (`sequences.n_observed_bins`). Each
adopter sets its `_sgd_n_time` from the canonicalised mask in its `fit_sgd`
(oscillator `oscillator_models.py:1186`, `PointProcessModel`
`point_process_kalman.py:3269`, `PlaceFieldModel` `place_field_model.py:1487`).
The mixin itself does not change. The `_n_timesteps` value must stay `> 0`
(guaranteed by [C6](#c6--multi-sequence-inputs-padding-and-sequence-lengths)'s
`len_s >= 1` rule).

## C9 — Where batching lives (not in `run_em`)

`run_em` (`em_driver.py:58-356`) talks to a model only through four callables
and sees a single float per E-step (`:223`). It stays untouched. Batching over
sequences is implemented **inside each model's `_e_step` / `_m_step`** (and
`_sgd_loss_fn`), using the [C7](#c7--state_space_practicesequences-helper-api)
helpers and the batched statistics functions of
[designs D4](designs.md#d4--multi-sequence-sufficient-statistics-two-pass): the
E-step returns `sum_s LL_s`, so `run_em`'s convergence, rollback and snapshot
logic is unchanged. Rationale: the driver has no notion of which arrays are
per-sequence or how a model's statistics combine; putting a `vmap` there would
require it to know model internals it was designed not to know
(`em_driver.py:12-14`). `SGDFittableMixin.fit_sgd` likewise stays unchanged:
`_sgd_loss_fn` receives the batched arrays (data args flow through
`_prepare_sgd_data`, `sgd_fitting.py:544`) and sums over sequences itself.

## GLM family masking integration

Use the [shared GLM family contract](../glm-families-nb-zig/shared-contracts.md#c1-glmfamily-extended-contract): likelihood terms and masks are observation-sized, while scores and information are predictor-sized. Tile the observation mask for predictor-major families; ZIG has two predictor groups. Sanitize masked observations before calling the selected family's validator or evaluating its likelihood. Masked likelihoods use the optional normalized `loglik_per_obs` callback; existing scalar callbacks keep their three-argument signatures. Poisson/Bernoulli/NB/ZIG supply the callback, custom families without it remain supported on unmasked calls.

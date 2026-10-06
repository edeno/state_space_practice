# ruff: noqa: E402
"""Gradient checks for every ``SGDFittableMixin`` loss.

For each model the harness captures exactly the problem ``fit_sgd`` would
optimise -- the data after ``_prepare_sgd_data`` / the model's own setup, the
parameter spec, and the loss ``_sgd_loss_fn(transform_to_constrained(u))`` --
without running the optimiser. At a random point in unconstrained space it
compares ``jax.grad`` with central finite differences, one random direction
per parameter group plus one joint direction, in float64. It also checks the
constrained <-> unconstrained round trips that fit_sgd relies on for warm
starts: ``to_unconstrained(to_constrained(u)) == u`` at the random point and
``to_constrained(to_unconstrained(p)) == p`` at the model's parameters, up to
the documented jitter of ``PSD_MATRIX`` (1e-9 on the diagonal) and the
open-interval clamp of ``UNIT_INTERVAL`` / ``positive_capped`` at exact
endpoints (1e-6).

The coupling EKF / Polya-Gamma estimators are not ``SGDFittableMixin``
models and expose no differentiable loss (they are Newton / Gibbs
regressions), so they have nothing to check here.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.parameter_transforms import (
    PSD_MATRIX,
    transform_to_constrained,
    transform_to_unconstrained,
)
from state_space_practice.sgd_fitting import SGDFittableMixin

# Every check compiles a loss and its gradient (5-30 s of XLA compilation).
pytestmark = pytest.mark.slow

_FIT_SETTINGS = ("optimizer", "num_steps", "verbose", "convergence_tol")


class _Captured(Exception):
    """Raised by the patched ``fit_sgd`` to abort right after data prep."""


def capture_sgd_problem(monkeypatch, model, *data, **kwargs):
    """Return ``(args, kwargs)`` that ``fit_sgd`` would pass to the loss.

    The model's own ``fit_sgd`` runs its setup (binding covariates, resolving
    shapes, initialising parameters) and then calls
    ``SGDFittableMixin.fit_sgd``, which is patched to run
    ``_prepare_sgd_data`` and stop.
    """
    captured = {}

    def fake_fit_sgd(self, *args, **kw):
        for key in _FIT_SETTINGS:
            kw.pop(key, None)
        args, kw = self._prepare_sgd_data(*args, **kw)
        captured["args"], captured["kwargs"] = args, kw
        raise _Captured

    monkeypatch.setattr(SGDFittableMixin, "fit_sgd", fake_fit_sgd)
    with pytest.raises(_Captured):
        # A patched no-op: nothing is optimised (see fake_fit_sgd).
        model.fit_sgd(*data, **kwargs)
    monkeypatch.undo()
    return captured["args"], captured["kwargs"]


def _assert_trees_close(actual, expected, tol, msg):
    a_leaves, a_def = jax.tree_util.tree_flatten(actual)
    e_leaves, e_def = jax.tree_util.tree_flatten(expected)
    assert a_def == e_def, msg
    for a, e in zip(a_leaves, e_leaves):
        np.testing.assert_allclose(a, e, rtol=tol, atol=tol, err_msg=msg)


def _tree_add(a, b, scale=1.0):
    return jax.tree_util.tree_map(lambda x, y: x + scale * y, a, b)


def _tree_dot(a, b):
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.vdot, a, b))
    return float(sum(leaves))


def _random_like(tree, rng, scale=1.0):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    new = [scale * rng.normal(size=np.shape(x)) for x in leaves]
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in new])


def _zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def check_sgd_loss(
    model,
    args,
    kwargs,
    seed=0,
    perturb=0.05,
    h=1e-5,
    rtol=2e-4,
    loss_kwargs=None,
):
    """Finite-difference check of the fit_sgd loss and transform round trips."""
    loss_kwargs = {**kwargs, **(loss_kwargs or {})}
    params, spec = model._build_param_spec()
    frozen_params = {k: params[k] for k, s in spec.items() if not s.trainable}
    unc0 = transform_to_unconstrained(params, spec, include_non_trainable=False)
    assert unc0, "no trainable parameters"

    # Round trip at the model's parameters (con -> unc -> con).
    back = transform_to_constrained(unc0, spec, static_params=frozen_params)
    for key, value in params.items():
        _assert_trees_close(
            back[key], value, 1e-6, f"round trip con->unc->con of {key!r}"
        )

    rng = np.random.default_rng(seed)
    unc = _tree_add(unc0, _random_like(unc0, rng), perturb)

    # Round trip at a random unconstrained point (unc -> con -> unc).
    con = transform_to_constrained(unc, spec, static_params=frozen_params)
    trainable_con = {k: v for k, v in con.items() if spec[k].trainable}
    trainable_spec = {k: s for k, s in spec.items() if s.trainable}
    unc_back = transform_to_unconstrained(trainable_con, trainable_spec)
    con_back = transform_to_constrained(unc_back, trainable_spec)
    for key in unc:
        if trainable_spec[key].to_constrained is PSD_MATRIX.to_constrained:
            # PSD_MATRIX adds a 1e-9 jitter before its Cholesky: exact to 1e-9
            # in constrained space, which in Cholesky-softplus coordinates is
            # ~1e-9 / lambda_min (1e-5 for the Hamiltonian Q = 1e-4 I).
            _assert_trees_close(
                con_back[key], trainable_con[key], 1e-8, f"PSD round trip of {key!r}"
            )
        else:
            _assert_trees_close(
                unc_back[key], unc[key], 1e-7, f"round trip unc->con->unc of {key!r}"
            )

    def loss_fn(u):
        p = transform_to_constrained(u, spec, static_params=frozen_params)
        return model._sgd_loss_fn(p, *args, **loss_kwargs)

    loss = jax.jit(loss_fn)
    value = float(loss(unc))
    assert np.isfinite(value), value
    grad = jax.jit(jax.grad(loss_fn))(unc)
    assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(grad))

    directions = {}
    for key in unc:
        d = _zeros_like(unc)
        d[key] = _random_like(unc[key], rng)
        directions[key] = d
    directions["<all>"] = _random_like(unc, rng)

    failures = []
    checked_nonzero = 0
    for name, d in directions.items():
        ad = _tree_dot(grad, d)
        fd = (
            float(loss(_tree_add(unc, d, h))) - float(loss(_tree_add(unc, d, -h)))
        ) / (2 * h)
        tol = rtol * max(abs(fd), abs(ad)) + 1e-7 * max(1.0, abs(value))
        if abs(ad - fd) > tol:
            failures.append(f"{name}: autodiff {ad:.8g} vs FD {fd:.8g}")
        if abs(fd) > 1e-6 * max(1.0, abs(value)):
            checked_nonzero += 1
    assert not failures, "; ".join(failures)
    # Guard: the check is informative (most directions have a real slope).
    assert checked_nonzero >= max(1, len(directions) // 2), checked_nonzero
    return value


# ---------------------------------------------------------------------------
# Behavioural models
# ---------------------------------------------------------------------------


def test_multinomial_choice_loss(monkeypatch):
    from state_space_practice.multinomial_choice import MultinomialChoiceModel

    choices = np.random.default_rng(0).integers(0, 3, 30)
    model = MultinomialChoiceModel(
        n_options=3, init_process_noise=0.05, init_inverse_temperature=1.5
    )
    args, kwargs = capture_sgd_problem(monkeypatch, model, choices)
    check_sgd_loss(model, args, kwargs)


def test_covariate_choice_loss(monkeypatch):
    from state_space_practice.covariate_choice import (
        CovariateChoiceModel,
        simulate_rl_choice_data,
    )

    data = simulate_rl_choice_data(n_trials=30, seed=1, inverse_temperature=1.0)
    obs = np.random.default_rng(1).normal(size=(30, 1))
    model = CovariateChoiceModel(
        n_options=3,
        n_covariates=2,
        n_obs_covariates=1,
        init_decay=0.9,
        learn_decay=True,
    )
    model.input_gain_ = 0.3 * jnp.eye(2)
    model.obs_weights_ = jnp.array([[0.0], [0.4], [-0.2]])
    args, kwargs = capture_sgd_problem(
        monkeypatch, model, data.choices, covariates=data.covariates, obs_covariates=obs
    )
    check_sgd_loss(model, args, kwargs)


def test_switching_choice_loss(monkeypatch):
    from state_space_practice.covariate_choice import simulate_rl_choice_data
    from state_space_practice.switching_choice import SwitchingChoiceModel

    data = simulate_rl_choice_data(n_trials=25, seed=2, inverse_temperature=1.0)
    obs = np.random.default_rng(2).normal(size=(25, 1))
    model = SwitchingChoiceModel(
        n_options=3,
        n_discrete_states=2,
        n_covariates=2,
        n_obs_covariates=1,
        init_inverse_temperatures=[0.8, 2.5],
        init_process_noises=[0.02, 0.1],
        init_decays=[0.95, 0.8],
    )
    model.input_gain_ = 0.3 * jnp.eye(2)
    model.obs_weights_ = jnp.array([[0.0], [0.4], [-0.2]])
    args, kwargs = capture_sgd_problem(
        monkeypatch, model, data.choices, covariates=data.covariates, obs_covariates=obs
    )
    check_sgd_loss(model, args, kwargs)


@pytest.mark.parametrize("per_state", [False, True])
def test_contingency_belief_loss(monkeypatch, per_state):
    from state_space_practice.contingency_belief import ContingencyBeliefModel

    rng = np.random.default_rng(3)
    T = 30
    choices = rng.integers(0, 3, T)
    rewards = rng.integers(0, 2, T)
    model = ContingencyBeliefModel(
        n_states=2,
        n_options=3,
        n_obs_covariates=1,
        per_state_obs_weights=per_state,
        seed=0,
    )
    args, kwargs = capture_sgd_problem(
        monkeypatch,
        model,
        choices,
        rewards,
        transition_covariates=rng.normal(size=(T, 1)),
        obs_design_matrix=rng.normal(size=(T, 1)),
    )
    check_sgd_loss(model, args, kwargs)


def test_smith_learning_loss(monkeypatch):
    from state_space_practice.smith_learning_algorithm import SmithLearningModel

    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(0, 0.3, 30))
    y = (rng.random(30) < 1 / (1 + np.exp(-x))).astype(int)
    model = SmithLearningModel(max_possible_correct=1, init_learning_variance=0.3)
    args, kwargs = capture_sgd_problem(monkeypatch, model, y)
    check_sgd_loss(model, args, kwargs)


@pytest.mark.parametrize("share", [True, False])
def test_temporal_rate_gp_loss(monkeypatch, share):
    from state_space_practice.temporal_rate_gp import TemporalRateGP

    counts = np.random.default_rng(5).poisson(0.3, size=(2, 40))
    model = TemporalRateGP(dt=0.05, share_hyperparameters=share, n_iter=10)
    args, kwargs = capture_sgd_problem(monkeypatch, model, counts)
    check_sgd_loss(model, args, kwargs)


# ---------------------------------------------------------------------------
# Hamiltonian models
# ---------------------------------------------------------------------------


def _ham_data(n_time=20, n_sources=2, n_neurons=3, seed=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n_time) * 0.01
    lfp = np.stack([np.sin(2 * np.pi * 8 * t + k) for k in range(n_sources)], 1)
    lfp = lfp * 0.1 + 0.02 * rng.normal(size=lfp.shape)
    spikes = rng.poisson(0.2, size=(n_time, n_neurons))
    return lfp, spikes


@pytest.mark.parametrize("use_filter", [True, False])
def test_hamiltonian_lfp_loss(monkeypatch, use_filter):
    from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel

    lfp, _ = _ham_data()
    model = HamiltonianLFPModel(
        n_oscillators=1, n_sources=2, sampling_freq=100.0, hidden_dims=[4]
    )
    args, kwargs = capture_sgd_problem(monkeypatch, model, lfp, use_filter=use_filter)
    check_sgd_loss(model, args, kwargs)


@pytest.mark.parametrize("use_filter", [True, False])
def test_hamiltonian_spike_loss(monkeypatch, use_filter):
    from state_space_practice.hamiltonian_spikes import HamiltonianSpikeModel

    _, spikes = _ham_data()
    model = HamiltonianSpikeModel(
        n_oscillators=1, n_sources=3, sampling_freq=100.0, hidden_dims=[4]
    )
    args, kwargs = capture_sgd_problem(
        monkeypatch, model, spikes, use_filter=use_filter
    )
    check_sgd_loss(model, args, kwargs)


@pytest.mark.parametrize("use_filter", [True, False])
def test_hamiltonian_joint_loss(monkeypatch, use_filter):
    from state_space_practice.hamiltonian_joint import JointHamiltonianModel

    lfp, spikes = _ham_data()
    model = JointHamiltonianModel(
        n_oscillators=1,
        n_lfp_sources=2,
        n_spike_sources=3,
        sampling_freq=100.0,
        hidden_dims=[4],
    )
    args, kwargs = capture_sgd_problem(
        monkeypatch, model, lfp, spikes, use_filter=use_filter
    )
    check_sgd_loss(model, args, kwargs)


def test_switching_hamiltonian_loss(monkeypatch):
    from state_space_practice.hamiltonian_switching import (
        SwitchingHamiltonianJointModel,
    )

    lfp, spikes = _ham_data()
    model = SwitchingHamiltonianJointModel(
        n_oscillators=1,
        n_lfp_sources=2,
        n_spike_sources=3,
        n_discrete_states=2,
        sampling_freq=100.0,
        hidden_dims=[4],
    )
    args, kwargs = capture_sgd_problem(monkeypatch, model, lfp, spikes)
    check_sgd_loss(model, args, kwargs)


# ---------------------------------------------------------------------------
# Point-process / place-field / oscillator models (read-only here)
# ---------------------------------------------------------------------------


def test_point_process_model_loss(monkeypatch):
    from state_space_practice.point_process_kalman import PointProcessModel

    rng = np.random.default_rng(7)
    T = 40
    design = rng.normal(size=(T, 2)) * 0.5
    spikes = rng.poisson(0.3, size=T)
    model = PointProcessModel(n_state_dims=2, dt=0.02)
    args, kwargs = capture_sgd_problem(monkeypatch, model, design, spikes)
    check_sgd_loss(model, args, kwargs)


def test_place_field_model_loss(monkeypatch):
    from state_space_practice.place_field_model import PlaceFieldModel
    from state_space_practice.simulate_data import simulate_2d_moving_place_field

    sim = simulate_2d_moving_place_field(
        total_time=2.0,
        dt=0.02,
        arena_size=80.0,
        peak_rate=25.0,
        background_rate=1.0,
        n_interior_knots=2,
        rng=np.random.default_rng(8),
    )
    model = PlaceFieldModel(dt=0.02, n_interior_knots=2, init_process_noise=1e-3)
    args, kwargs = capture_sgd_problem(
        monkeypatch, model, sim["position"], sim["spikes"]
    )
    check_sgd_loss(model, args, kwargs, perturb=0.02)


@pytest.mark.parametrize("name", ["com", "cnm", "dim"])
def test_oscillator_model_loss(monkeypatch, name):
    from state_space_practice import oscillator_models as om
    from state_space_practice.simulate import scenarios

    sim = getattr(scenarios, f"simulate_{name}_scenario")(n_time=40, seed=9)
    p = sim["params"]
    common = dict(
        n_oscillators=p["n_oscillators"],
        n_discrete_states=p["n_discrete_states"],
        sampling_freq=p["sampling_freq"],
        freqs=p["freqs"],
        damping_coef=p["damping"],
        process_variance=p["process_variance"],
        measurement_variance=p["measurement_variance"],
    )
    if name == "com":
        model = om.CommonOscillatorModel(n_sources=p["n_sources"], **common)
    else:
        cls = om.CorrelatedNoiseModel if name == "cnm" else om.DirectedInfluenceModel
        model = cls(
            phase_difference=p["phase_difference"],
            coupling_strength=p["coupling_strength"],
            **common,
        )
    args, kwargs = capture_sgd_problem(
        monkeypatch, model, sim["obs"], key=jax.random.PRNGKey(0)
    )
    check_sgd_loss(model, args, kwargs, perturb=0.02)


def test_switching_spike_oscillator_loss(monkeypatch):
    from state_space_practice.switching_point_process import (
        SwitchingSpikeOscillatorModel,
    )

    key = jax.random.PRNGKey(10)
    spikes = jax.random.poisson(key, jnp.ones((30, 3)) * 0.2)
    model = SwitchingSpikeOscillatorModel(
        n_oscillators=1,
        n_neurons=3,
        n_discrete_states=2,
        sampling_freq=100.0,
        dt=0.01,
    )
    args, kwargs = capture_sgd_problem(monkeypatch, model, spikes, key=key)
    check_sgd_loss(model, args, kwargs, perturb=0.02)

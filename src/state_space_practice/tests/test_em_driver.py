"""Tests for the shared EM driver (:mod:`state_space_practice.em_driver`)."""

import logging

import numpy as np
import pytest

from state_space_practice.em_driver import run_em


class ScriptedModel:
    """A stand-in model whose E-step returns a scripted log-likelihood sequence.

    ``params`` counts the M-steps that have been applied; ``posterior`` is set
    by the E-step from the parameter count, so a rollback is visible as a
    (params, posterior) pair that belongs to an earlier iteration.
    """

    def __init__(self, lls):
        self.lls = list(lls)
        self.params = 0
        self.posterior = None
        self.e_calls = 0
        self.cleared = False

    def e_step(self):
        ll = self.lls[min(self.e_calls, len(self.lls) - 1)]
        self.e_calls += 1
        self.posterior = ("posterior-for-params", self.params)
        return ll

    def m_step(self):
        self.params += 1

    def snapshot(self):
        return {"params": self.params, "posterior": self.posterior}

    def restore(self, state):
        self.params = state["params"]
        self.posterior = state["posterior"]

    def clear(self):
        self.cleared = True
        self.posterior = None

    def run(self, **kwargs):
        return run_em(self.e_step, self.m_step, self.snapshot, self.restore, **kwargs)


class ParamScriptedModel(ScriptedModel):
    """A ScriptedModel whose log-likelihood is a function of the parameters.

    ``lls[k]`` is the log-likelihood after ``k`` M-steps, so re-running the
    E-step under restored parameters reproduces that iterate's value, as a
    real model would.
    """

    def e_step(self):
        self.e_calls += 1
        self.posterior = ("posterior-for-params", self.params)
        return self.lls[self.params]


def test_converges_and_keeps_consistent_state():
    model = ScriptedModel([-100.0, -50.0, -50.0000001])
    result = model.run(max_iter=10, tol=1e-4)

    assert result.converged and not result.reached_max_iter
    assert result.log_likelihoods == [-100.0, -50.0, -50.0000001]
    # Two M-steps ran; the stored posterior belongs to the current parameters.
    assert model.params == 2
    assert model.posterior == ("posterior-for-params", 2)


def test_decrease_rolls_back_to_previous_accepted_state(caplog):
    model = ScriptedModel([-100.0, -50.0, -80.0])
    with caplog.at_level(logging.WARNING):
        result = model.run(max_iter=10, tol=1e-4)

    assert result.log_likelihoods == [-100.0, -50.0]
    assert not result.converged
    # Rolled back to the snapshot taken before the second M-step.
    assert model.params == 1
    assert model.posterior == ("posterior-for-params", 1)
    assert any("rolling back" in r.message.lower() for r in caplog.records)


def test_lenient_decrease_tol_continues_on_small_decrease():
    model = ScriptedModel([-100.0, -50.0, -50.1, -49.0, -49.0])
    result = model.run(max_iter=10, tol=1e-6, decrease_tol=1e-2)

    # The 0.2 % decrease is within decrease_tol, so EM continues and converges.
    assert result.converged
    assert result.log_likelihoods == [-100.0, -50.0, -50.1, -49.0, -49.0]


def test_continue_on_decrease_restores_best_state():
    # LL after 0..4 M-steps; the best iterate is params=1.
    model = ParamScriptedModel([-100.0, -40.0, -60.0, -70.0, -65.0])
    warnings = []
    result = model.run(
        max_iter=4,
        tol=1e-4,
        stop_on_decrease=False,
        require_increase_to_converge=True,
        track_best=True,
        warn=warnings.append,
    )

    # Decreases are warned about but kept; the final E-step (params=4, -65)
    # is appended; then the best state (params=1) is restored and the
    # E-step re-run under it appends its LL.
    assert result.log_likelihoods == [-100.0, -40.0, -60.0, -70.0, -65.0, -40.0]
    assert result.reached_max_iter and not result.converged
    assert sum("decreased" in w for w in warnings) == 2
    assert model.params == 1
    assert model.posterior == ("posterior-for-params", 1)


def test_track_best_warns_when_restored_e_step_is_nonfinite():
    model = ScriptedModel([-100.0, -40.0, -60.0, -70.0, -70.0, np.nan])
    warnings = []
    result = model.run(
        max_iter=4,
        tol=1e-4,
        stop_on_decrease=False,
        track_best=True,
        warn=warnings.append,
    )

    # The best state (params=1) was restored, but its re-run E-step is NaN:
    # nothing non-finite enters the history and the failure is reported.
    assert model.params == 1
    assert model.e_calls == 6
    assert result.log_likelihoods == [-100.0, -40.0, -60.0, -70.0, -70.0]
    assert any("best" in w and "non-finite" in w for w in warnings)


def test_require_increase_to_converge_rejects_converged_decrease():
    # -50 -> -50.005 is within tol (converged) but a decrease beyond
    # decrease_tol; it must not end EM.
    lls = [-100.0, -50.0, -50.005, -40.0, -40.0]
    kwargs = {
        "max_iter": 10,
        "tol": 1e-2,
        "decrease_tol": 1e-6,
        "stop_on_decrease": False,
    }
    warnings = []

    model = ScriptedModel(lls)
    result = model.run(
        require_increase_to_converge=True, warn=warnings.append, **kwargs
    )
    assert any("decreased at iteration 3" in w for w in warnings)
    assert result.converged
    assert result.log_likelihoods == lls
    assert model.params == 4

    # Without the flag the same decrease counts as convergence.
    baseline = ScriptedModel(lls)
    result = baseline.run(require_increase_to_converge=False, **kwargs)
    assert result.converged and result.log_likelihoods == lls[:3]
    assert baseline.params == 2


@pytest.mark.parametrize("policy", ["break", "clear"])
def test_first_nonfinite_policies(policy):
    model = ScriptedModel([np.nan])
    result = model.run(
        max_iter=3,
        tol=1e-4,
        on_first_nonfinite=policy,
        clear_state=model.clear if policy == "clear" else None,
    )

    assert result.log_likelihoods == []  # never a NaN in the history
    assert not result.converged and not result.reached_max_iter
    assert model.cleared is (policy == "clear")


def test_first_nonfinite_raise_policy():
    model = ScriptedModel([np.inf])
    with pytest.raises(ValueError, match="Non-finite"):
        model.run(max_iter=3, tol=1e-4, on_first_nonfinite="raise")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"on_first_nonfinite": "clear"}, "clear_state"),
        ({"on_first_nonfinite": "break", "clear_state": lambda: None}, "clear_state"),
        ({"on_first_nonfinite": "raise", "clear_state": lambda: None}, "clear_state"),
        ({"on_first_nonfinite": "ignore"}, "on_first_nonfinite"),
        ({"max_iter": 0}, "max_iter"),
    ],
)
def test_invalid_arguments_raise_before_any_e_step(kwargs, match):
    model = ScriptedModel([-100.0, -50.0])
    run_kwargs = {"max_iter": 3, "tol": 1e-4, **kwargs}
    with pytest.raises(ValueError, match=match):
        model.run(**run_kwargs)
    assert model.e_calls == 0


def test_later_nonfinite_rolls_back(caplog):
    model = ScriptedModel([-100.0, np.nan])
    with caplog.at_level(logging.WARNING):
        result = model.run(max_iter=5, tol=1e-4, on_first_nonfinite="raise")

    assert result.log_likelihoods == [-100.0]
    assert model.params == 0 and model.posterior == ("posterior-for-params", 0)
    assert any(
        "non-finite log-likelihood" in r.message.lower()
        and "rolling back" in r.message.lower()
        for r in caplog.records
    )


def test_refresh_after_restore_warns_on_nonfinite_refresh():
    model = ScriptedModel([-100.0, -50.0, -80.0, np.nan])
    warnings = []
    result = model.run(
        max_iter=10, tol=1e-4, refresh_after_restore=True, warn=warnings.append
    )

    assert result.log_likelihoods == [-100.0, -50.0]
    assert model.e_calls == 4  # the refresh E-step ran
    assert any("non-finite" in w and "restor" in w for w in warnings)


def test_refresh_after_restore_recomputes_posteriors():
    model = ScriptedModel([-100.0, -50.0, -80.0, -55.0])
    result = model.run(max_iter=10, tol=1e-4, refresh_after_restore=True)

    assert result.log_likelihoods == [-100.0, -50.0]
    # After restoring params=1, an extra E-step ran under those parameters.
    assert model.e_calls == 4
    assert model.posterior == ("posterior-for-params", 1)


def test_final_e_step_after_max_iter_appends_or_rolls_back():
    improving = ScriptedModel([-100.0, -90.0, -80.0])
    result = improving.run(max_iter=2, tol=1e-4)
    assert result.reached_max_iter and not result.converged
    assert result.log_likelihoods == [-100.0, -90.0, -80.0]
    assert improving.params == 2

    worsening = ScriptedModel([-100.0, -90.0, -95.0])
    result = worsening.run(max_iter=2, tol=1e-4)
    assert result.log_likelihoods == [-100.0, -90.0]
    assert worsening.params == 1  # last M-step rolled back
    assert worsening.posterior == ("posterior-for-params", 1)

    skipped = ScriptedModel([-100.0, -90.0, -80.0])
    result = skipped.run(max_iter=2, tol=1e-4, final_e_step=False)
    assert result.log_likelihoods == [-100.0, -90.0]
    assert skipped.e_calls == 2


def test_final_e_step_nonfinite_rolls_back(caplog):
    model = ScriptedModel([-100.0, -90.0, np.nan])
    with caplog.at_level(logging.WARNING):
        result = model.run(max_iter=2, tol=1e-4)

    assert model.e_calls == 3  # the final E-step ran
    assert result.log_likelihoods == [-100.0, -90.0]
    assert model.params == 1
    assert model.posterior == ("posterior-for-params", 1)
    assert any("final e-step" in r.message.lower() for r in caplog.records)


def test_m_step_on_convergence_runs_extra_m_step():
    model = ScriptedModel([-100.0, -50.0, -50.0, -45.0])
    result = model.run(max_iter=10, tol=1e-4, m_step_on_convergence=True)

    assert result.converged
    # Converged at the third E-step, then one more M-step + E-step (-45).
    assert result.log_likelihoods == [-100.0, -50.0, -50.0, -45.0]
    assert model.params == 3
    assert model.posterior == ("posterior-for-params", 3)


@pytest.mark.parametrize("bad_ll", [-80.0, np.nan])
def test_m_step_on_convergence_rolls_back_and_warns(bad_ll):
    model = ScriptedModel([-100.0, -50.0, -50.0, bad_ll])
    warnings = []
    result = model.run(
        max_iter=10, tol=1e-4, m_step_on_convergence=True, warn=warnings.append
    )

    assert model.e_calls == 4  # the post-convergence E-step ran
    assert result.converged
    assert result.log_likelihoods == [-100.0, -50.0, -50.0]
    # Back to the converged iterate, with the posterior of those parameters.
    assert model.params == 2
    assert model.posterior == ("posterior-for-params", 2)
    assert len(warnings) == 1 and "rolling back" in warnings[0]


def test_m_step_on_convergence_uses_decrease_tol():
    # A post-convergence decrease within decrease_tol is noise, not a rollback.
    model = ScriptedModel([-100.0, -50.0, -50.0, -50.1])
    result = model.run(
        max_iter=10, tol=1e-6, decrease_tol=1e-2, m_step_on_convergence=True
    )

    assert result.log_likelihoods == [-100.0, -50.0, -50.0, -50.1]
    assert model.params == 3


def test_iteration_and_warning_hooks_receive_messages():
    seen = []
    warnings = []
    model = ScriptedModel([-100.0, -50.0, -80.0])
    model.run(
        max_iter=10,
        tol=1e-4,
        on_iteration=lambda i, ll, change: seen.append((i, ll, change)),
        warn=warnings.append,
    )

    assert [s[:2] for s in seen] == [(0, -100.0), (1, -50.0), (2, -80.0)]
    assert np.isnan(seen[0][2]) and seen[1][2] == 50.0
    assert len(warnings) == 1 and "rolling back" in warnings[0]

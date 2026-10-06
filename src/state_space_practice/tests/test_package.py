"""Tests for the package surface: lazy public API, version, x64 import warning."""

import os
import subprocess
import sys
import textwrap

import pytest

import state_space_practice as ssp
from state_space_practice import (
    NonFiniteLikelihoodError,
    NotFittedError,
    StateSpaceWarning,
)


def _run_python(code: str, **env_overrides: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter that imports this checkout."""
    env = os.environ.copy()
    env.pop("JAX_ENABLE_X64", None)
    src_dir = os.path.dirname(os.path.dirname(ssp.__file__))
    env["PYTHONPATH"] = os.pathsep.join(
        [src_dir, *filter(None, [env.get("PYTHONPATH")])]
    )
    env["JAX_PLATFORMS"] = "cpu"
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


class TestPublicAPI:
    def test_all_matches_lazy_registry(self):
        lazy = set(ssp._LAZY_API)
        eager = {
            "__version__",
            "NonFiniteLikelihoodError",
            "NotFittedError",
            "StateSpaceWarning",
        }
        assert set(ssp.__all__) == lazy | eager
        assert len(ssp.__all__) == len(set(ssp.__all__))

    @pytest.mark.parametrize("name", sorted(ssp._LAZY_API))
    def test_lazy_name_resolves_to_defining_module(self, name):
        obj = getattr(ssp, name)
        assert obj.__module__ == f"state_space_practice.{ssp._LAZY_API[name]}"
        assert name in dir(ssp)

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="no_such_name"):
            ssp.no_such_name  # noqa: B018

    def test_top_level_place_field_model_is_the_defining_class(self):
        from state_space_practice import PlaceFieldModel
        from state_space_practice.place_field_model import (
            PlaceFieldModel as Direct,
        )

        assert PlaceFieldModel is Direct

    def test_version_is_a_nonempty_string(self):
        assert isinstance(ssp.__version__, str)
        assert ssp.__version__[0].isdigit()

    def test_exception_hierarchy(self):
        assert issubclass(StateSpaceWarning, UserWarning)
        assert issubclass(NotFittedError, RuntimeError)
        assert issubclass(NonFiniteLikelihoodError, ValueError)


class TestImportSideEffects:
    def test_top_level_import_is_lazy(self):
        result = _run_python(
            """
            import jax
            jax.config.update("jax_enable_x64", True)
            import sys
            import state_space_practice
            heavy = [m for m in ("state_space_practice.place_field_model",
                                 "state_space_practice.switching_point_process",
                                 "state_space_practice.kalman", "optax", "patsy")
                     if m in sys.modules]
            print("HEAVY", heavy)
            """
        )
        assert result.returncode == 0, result.stderr
        assert "HEAVY []" in result.stdout

    def test_warns_when_x64_disabled(self):
        result = _run_python(
            """
            import warnings
            warnings.simplefilter("always")
            with warnings.catch_warnings(record=True) as caught:
                import state_space_practice
            hits = [w for w in caught
                    if issubclass(w.category, state_space_practice.StateSpaceWarning)]
            print("N_WARN", len(hits))
            print(hits[0].message if hits else "")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "N_WARN 1" in result.stdout
        assert 'jax.config.update("jax_enable_x64", True)' in result.stdout

    @pytest.mark.parametrize("how", ["config", "env"])
    def test_silent_when_x64_enabled(self, how):
        prelude = (
            'import jax; jax.config.update("jax_enable_x64", True)'
            if how == "config"
            else ""
        )
        env = {"JAX_ENABLE_X64": "1"} if how == "env" else {}
        result = _run_python(
            f"""
            {prelude}
            import warnings
            warnings.simplefilter("error")
            import state_space_practice
            print("IMPORTED")
            """,
            **env,
        )
        assert result.returncode == 0, result.stderr
        assert "IMPORTED" in result.stdout

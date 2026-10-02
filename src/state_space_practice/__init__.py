"""State-space models in JAX: Kalman, point-process, and switching models.

The public API below is loaded lazily (PEP 562): ``import state_space_practice``
only imports JAX, and each model module is imported on first attribute access.

This library needs float64 for long point-process sequences. Enable it before
importing::

    import jax

    jax.config.update("jax_enable_x64", True)

    from state_space_practice import PlaceFieldModel
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

from state_space_practice.exceptions import (
    NonFiniteLikelihoodError,
    NotFittedError,
    StateSpaceWarning,
)

try:
    from state_space_practice._version import __version__
except ImportError:  # pragma: no cover - _version.py is written at build/install
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("state_space_practice")
    except PackageNotFoundError:
        __version__ = "0+unknown"

# Public name -> submodule that defines it.
_LAZY_API: dict[str, str] = {
    "kalman_filter": "kalman",
    "kalman_smoother": "kalman",
    "switching_kalman_filter": "switching_kalman",
    "switching_kalman_smoother": "switching_kalman",
    "run_em": "em_driver",
    "PointProcessModel": "point_process_kalman",
    "PlaceFieldModel": "place_field_model",
    "PositionDecoder": "position_decoder",
    "CommonOscillatorModel": "oscillator_models",
    "CorrelatedNoiseModel": "oscillator_models",
    "DirectedInfluenceModel": "oscillator_models",
    "SwitchingSpikeOscillatorModel": "switching_point_process",
    "MultinomialChoiceModel": "multinomial_choice",
    "CovariateChoiceModel": "covariate_choice",
    "SwitchingChoiceModel": "switching_choice",
    "ContingencyBeliefModel": "contingency_belief",
    "SmithLearningModel": "smith_learning_algorithm",
}

__all__ = [
    "__version__",
    "NonFiniteLikelihoodError",
    "NotFittedError",
    "StateSpaceWarning",
    "kalman_filter",
    "kalman_smoother",
    "switching_kalman_filter",
    "switching_kalman_smoother",
    "run_em",
    "PointProcessModel",
    "PlaceFieldModel",
    "PositionDecoder",
    "CommonOscillatorModel",
    "CorrelatedNoiseModel",
    "DirectedInfluenceModel",
    "SwitchingSpikeOscillatorModel",
    "MultinomialChoiceModel",
    "CovariateChoiceModel",
    "SwitchingChoiceModel",
    "ContingencyBeliefModel",
    "SmithLearningModel",
]

if TYPE_CHECKING:
    from state_space_practice.contingency_belief import ContingencyBeliefModel
    from state_space_practice.covariate_choice import CovariateChoiceModel
    from state_space_practice.em_driver import run_em
    from state_space_practice.kalman import kalman_filter, kalman_smoother
    from state_space_practice.multinomial_choice import MultinomialChoiceModel
    from state_space_practice.oscillator_models import (
        CommonOscillatorModel,
        CorrelatedNoiseModel,
        DirectedInfluenceModel,
    )
    from state_space_practice.place_field_model import PlaceFieldModel
    from state_space_practice.point_process_kalman import PointProcessModel
    from state_space_practice.position_decoder import PositionDecoder
    from state_space_practice.smith_learning_algorithm import SmithLearningModel
    from state_space_practice.switching_choice import SwitchingChoiceModel
    from state_space_practice.switching_kalman import (
        switching_kalman_filter,
        switching_kalman_smoother,
    )
    from state_space_practice.switching_point_process import (
        SwitchingSpikeOscillatorModel,
    )


def __getattr__(name: str) -> Any:
    module_name = _LAZY_API.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value  # cache: later lookups bypass __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


_X64_RECIPE = (
    "state_space_practice was imported with JAX float64 disabled "
    "(jax_enable_x64=False). The Laplace-EKF point-process filters can lose "
    "covariance PSD and silently produce NaN on long sequences in float32. "
    "Enable float64 before importing this library:\n\n"
    "    import jax\n"
    '    jax.config.update("jax_enable_x64", True)\n\n'
    "    from state_space_practice import PlaceFieldModel\n\n"
    "or set the environment variable JAX_ENABLE_X64=1. Silence this warning "
    'with warnings.filterwarnings("ignore", category=StateSpaceWarning).'
)


def _warn_if_x64_disabled() -> None:
    import jax

    if not jax.config.read("jax_enable_x64"):
        warnings.warn(_X64_RECIPE, StateSpaceWarning, stacklevel=3)


_warn_if_x64_disabled()

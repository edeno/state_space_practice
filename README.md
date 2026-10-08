# state_space_practice

State-space models in JAX for neural data: Kalman filtering and smoothing,
point-process (spike) observation models with a Laplace-EKF, switching linear
dynamical systems, coupled-oscillator networks, and behavioural latent-state
models (choice, learning).

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/); `uv.lock` is
committed, so a clone reproduces the tested environment (Python 3.10-3.12;
uv provisions an interpreter if needed).

```bash
git clone https://github.com/edeno/state_space_practice
cd state_space_practice
uv sync --extra test --extra coupling   # editable install + dev/test tools
uv run pytest -m "not slow"             # fast suite
```

Optional extras:

| Extra       | Adds                          | Used by                                        |
|-------------|-------------------------------|------------------------------------------------|
| `plot`      | matplotlib                    | the `plot_*` helpers (imported lazily)         |
| `coupling`  | polyagamma                    | the Polya-Gamma coupling estimator             |
| `spatial`   | neurospatial >= 0.8.0         | `graph_place_field`                            |
| `notebooks` | matplotlib, pandas            | the exploratory scripts in `notebooks/`        |
| `gpu`       | `jax[cuda]`                   | GPU runs                                       |
| `test`      | pytest, hypothesis, ruff, mypy, matplotlib | development                       |

neurospatial 0.8.0 is not on PyPI yet. Under uv the `spatial` extra resolves
from a git commit pinned in `[tool.uv.sources]`. With pip, install it first:
`pip install "neurospatial @ git+https://github.com/edeno/neurospatial@e81dec62dbada29c70cfb3640488f429cae20608"`,
then `pip install -e ".[spatial]"`.

The `notebooks/` and `scripts/` files import the package from the editable
install. Some also import local data loaders from a gitignored `data/`
directory; run those from the repository root with `PYTHONPATH=.`.

## Float64 is required

The Laplace-EKF point-process filters (and every model built on them, such as
`PlaceFieldModel`) need float64 for long sequences: in float32, roundoff in the
covariance recursion can break positive-definiteness after a few hundred to a
few thousand bins and the filter silently returns NaN. Enable x64 **before**
importing the library:

```python
import jax

jax.config.update("jax_enable_x64", True)  # or set JAX_ENABLE_X64=1

from state_space_practice import PlaceFieldModel
```

Importing `state_space_practice` with x64 disabled emits a
`StateSpaceWarning` with this recipe.

## Package layout

`import state_space_practice` is cheap: the main entry points
(`kalman_filter`, `kalman_smoother`, `switching_kalman_filter`,
`switching_kalman_smoother`, `run_em`, `PlaceFieldModel`, `PointProcessModel`,
`PositionDecoder`, the oscillator and choice models, `SmithLearningModel`) are
loaded on first access. Everything else lives in the submodules
(`state_space_practice.kalman`, `.point_process_kalman`, ...).

## Development

```bash
uv run ruff check src/ && uv run ruff format --check src/ notebooks/ scripts/
uv run mypy                       # all package modules, including untyped bodies
uv run pytest -m "not slow"       # fast suite; drop -m for the full suite
uvx pre-commit install            # ruff, ruff-format and nbstripout on commit
```

Tests that run EM, SGD or full fits are marked `slow`, either explicitly or
automatically by `tests/conftest.py` (any test whose body, or a fixture it
uses, calls a fitting method or `run_em(` outside `pytest.raises`). CI runs lint, mypy and the
fast suite on Python 3.10-3.12 for every push and pull request, and the full
suite nightly. Set `HYPOTHESIS_PROFILE=ci` for the thorough Hypothesis profile.

## Roadmap status

Roadmaps were reconciled on 2026-07-08 against the checked-in code. The old
P1/P2/P2.5/P3/P6 queue has shipped; active next work should start from the
missing modules in:

- [Execution roadmap](docs/plans/2026-04-04-execution-roadmap.md) — including the
  [2026-09-28 extension queue](docs/plans/2026-04-04-execution-roadmap.md#extension-queue-2026-09-28):
  eleven phased plans under `docs/plans/<slug>/` (identifiability diagnostics,
  robust updates, new GLM families, masks and multi-sequence fitting, the
  iterated/parallel Laplace smoother, recurrent switching transitions,
  multi-map place fields, a successor-representation basis, a volatile Kalman
  choice model, theta-sweep amplitude tracking, streaming filters)
- [Spatial bandit latent roadmap](docs/plans/2026-04-05-bandit-latent-roadmap.md)
- [Drifting graph-GP place fields](docs/plans/2026-07-16-drifting-graph-gp-place-field.md)
  — Stages 0–2 implemented: geometry-aware static and drifting place-field
  inference and posterior field trajectories. Automatic drift-scale fitting
  now uses joint Laplace evidence and profile-initialized L-BFGS; fixed scales
  and experimental EM/Adam remain available. See the [usage guide](docs/graph-place-fields.md)
  for units and diagnostics. Richer dynamics and external estimator comparisons
  remain deferred.

Spike-only latent oscillator coupling plans are treated as exploratory or
blocked as scientific estimators unless an observed field signal anchors the
latent. Earlier working notes (`SCRATCHPAD.md`, `TASKS.md`) are archived under
[docs/plans/archive/](docs/plans/archive/).

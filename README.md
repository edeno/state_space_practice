# state_space_practice

State-space modeling practice repo for neural decoding, behavioral latent-state
models, and oscillator/coupling experiments.

## Roadmap status

Roadmaps were reconciled on 2026-07-08 against the checked-in code. The old
P1/P2/P2.5/P3/P6 queue has shipped; active next work should start from the
missing modules in:

- [Execution roadmap](docs/plans/2026-04-04-execution-roadmap.md)
- [Spatial bandit latent roadmap](docs/plans/2026-04-05-bandit-latent-roadmap.md)

Spike-only latent oscillator coupling plans are treated as exploratory or
blocked as scientific estimators unless an observed field signal anchors the
latent.

## Verification

```bash
uv run pytest -m "not slow" --tb=short
```

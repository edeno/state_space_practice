"""Benchmark the joint Laplace smoother's sequential vs parallel-in-time linear pass.

One case per process, so ``peak_bytes_in_use`` on an accelerator belongs to that
case. Simulates drifting place fields on a 1-D track, builds a
``GraphPlaceFieldModel``, and times the batched joint-smoother evidence and its
gradient with respect to a shared drift scale.

Timing protocol: compile + first call measured separately; then steady-state
calls, each blocked with ``jax.block_until_ready`` on the whole output pytree;
min and median reported.

Example::

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        uv run python scripts/benchmarks/joint_laplace_parallel_benchmark.py \\
        --parallel false --n-time 3000 --n-neurons 8 \\
        --out baseline.json --save-outputs baseline.npz
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import statistics
import time
from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from neurospatial import Environment  # noqa: E402

from state_space_practice.graph_place_field import (  # noqa: E402
    GraphPlaceFieldModel,
    _vmap_by_noise_structure,
)
from state_space_practice.laplace_smoothing import poisson_laplace_smoother  # noqa: E402

DT = 0.05
DRIFT = 1e-3


def simulate(
    n_time: int, n_neurons: int, seed: int
) -> tuple[Environment, np.ndarray, np.ndarray, np.ndarray]:
    """Track environment, times, positions (n_time, 1) and counts (n_time, n_neurons)."""
    env = Environment.from_samples(np.linspace(0, 250, 1001)[:, None], bin_size=5.0)
    rng = np.random.default_rng(seed)
    times = np.arange(n_time) * DT
    positions = env.bin_centers[(np.arange(n_time) // 2) % env.n_bins]
    starts = rng.uniform(40, 210, n_neurons)
    ends = np.clip(starts + rng.normal(0, 40, n_neurons), 20, 230)
    centres = np.linspace(starts, ends, n_time)  # (n_time, n_neurons)
    rates = 0.5 + 15 * np.exp(-0.5 * ((positions[:, :1] - centres) / 15) ** 2)
    counts = rng.poisson(rates * DT).astype(float)
    return env, times, positions, counts


def parallel_kwargs(function: Callable[..., object], parallel: bool) -> dict[str, bool]:
    """Pass ``parallel`` only to code that has the keyword (baseline runs predate it)."""
    if "parallel" in inspect.signature(function).parameters:
        return {"parallel": parallel}
    if parallel:
        raise SystemExit("This source tree has no `parallel` keyword.")
    return {}


def time_calls(fn, n_runs: int) -> dict[str, float]:  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    jax.block_until_ready(fn())
    first = time.perf_counter() - start
    jax.block_until_ready(fn())  # warm-up
    runs = []
    for _ in range(n_runs):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        runs.append(time.perf_counter() - start)
    return dict(
        compile_and_first_s=first, min_s=min(runs), median_s=statistics.median(runs)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parallel", choices=("false", "true"), required=True)
    parser.add_argument("--n-time", type=int, required=True)
    parser.add_argument("--n-neurons", type=int, required=True)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-outputs")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument(
        "--neuron-batch-size",
        type=int,
        help="Infer at most this many neurons at once (lax.map over chunks).",
    )
    args = parser.parse_args()
    use_parallel = args.parallel == "true"
    smoother_kwargs = parallel_kwargs(poisson_laplace_smoother, use_parallel)

    env, times, positions, counts = simulate(args.n_time, args.n_neurons, args.seed)
    model = GraphPlaceFieldModel(env, DT, rank=args.rank)
    (Z, spikes, valid), _ = model._prepare_sgd_data(times, positions, counts)
    S = model._spectral_shape_current()
    P0 = jnp.diag(model.tau2 * S)
    A = model.transition_matrix
    m0 = model.init_mean

    def one(mean: jax.Array, y: jax.Array, q: jax.Array):  # type: ignore[no-untyped-def]
        return poisson_laplace_smoother(
            mean,
            P0,
            A,
            jnp.diag(q * S),
            Z,
            y,
            valid,
            dt=DT,
            static_noise=False,
            **smoother_kwargs,
        )

    def per_neuron(q: jax.Array):  # type: ignore[no-untyped-def]
        # The library's own neuron batching, including its checkpointed chunks.
        return _vmap_by_noise_structure(
            lambda _flag, mean, y: one(mean, y, q),
            (False,) * args.n_neurons,
            m0,
            spikes,
            batch_size=args.neuron_batch_size,
        )

    def evidence(q: jax.Array) -> jax.Array:
        return jnp.sum(per_neuron(q).log_evidence)

    q = jnp.asarray(DRIFT)
    value = jax.jit(evidence)
    value_and_grad = jax.jit(jax.value_and_grad(evidence))
    n_runs = 3 if args.n_time >= 12000 else 5
    record: dict[str, object] = dict(
        parallel=use_parallel,
        n_time=args.n_time,
        n_neurons=args.n_neurons,
        neuron_batch_size=args.neuron_batch_size,
        rank=args.rank,
        seed=args.seed,
        backend=jax.default_backend(),
        device_kind=jax.devices()[0].device_kind,
        jax_version=jax.__version__,
        host=platform.node(),
        value=time_calls(lambda: value(q), n_runs),
        value_and_grad=time_calls(lambda: value_and_grad(q), n_runs),
    )
    result = jax.jit(per_neuron)(q)
    outputs: dict[str, np.ndarray] = dict(
        log_evidence=np.asarray(result.log_evidence),
        mean_first_neuron=np.asarray(result.mean[0]),
        converged=np.asarray(result.converged),
        evidence_grad=np.asarray(value_and_grad(q)[1]),
    )
    if args.fit:
        fit_model = GraphPlaceFieldModel(
            env,
            DT,
            rank=args.rank,
            **parallel_kwargs(GraphPlaceFieldModel.__init__, use_parallel),
        )
        start = time.perf_counter()
        fit_model.fit(times, positions, counts, profile_points=9, profile_rounds=2)
        record["fit"] = dict(
            wall_s=time.perf_counter() - start,
            log_likelihood=fit_model.log_likelihood_,
            converged=bool(fit_model.converged_),
        )
        outputs["fit_drift_scale"] = np.asarray(fit_model.drift_scale)
        outputs["fit_log_likelihood"] = np.asarray(fit_model.log_likelihood_)
    stats = jax.devices()[0].memory_stats()
    record["peak_bytes_in_use"] = stats.get("peak_bytes_in_use") if stats else None
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    if args.save_outputs:
        np.savez(args.save_outputs, **outputs)
    print(json.dumps(record))


if __name__ == "__main__":
    main()

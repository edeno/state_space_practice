"""Small spatial profile audit against an independently assembled dense posterior.

Run with the spatial/test extras::
    uv run --no-sync python notebooks/graph_place_field_joint_reference.py

This measures implementation accuracy of Laplace inference, not exact Bayes error.
"""

from pathlib import Path
import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from graph_place_field_validation import Environment, software_versions  # noqa: E402
from state_space_practice.graph_place_field import (  # noqa: E402
    GraphPlaceFieldModel,
    _masked_graph_point_process_filter,
)
from state_space_practice.kalman import rts_backward_scan  # noqa: E402
from state_space_practice.laplace_smoothing import poisson_laplace_smoother  # noqa: E402
from state_space_practice.tests.graph_math_reference import joint_poisson_laplace  # noqa: E402


def experiment():
    env = Environment.from_samples(np.linspace(0, 10, 201)[:, None], bin_size=2.0)
    model = GraphPlaceFieldModel(env, 0.1, rank=3)
    phi = np.asarray(model.basis.eigvecs)
    shape = np.asarray(model._spectral_shape_current())
    mean = np.zeros(3)
    mean[0] = np.log(5) / phi[0, 0]
    p = np.diag(shape)
    a = np.eye(3)
    records = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        t = np.arange(60) * 0.1
        positions = (5 + 4.9 * np.sin(t))[:, None]
        z, valid = model._design_and_spikes(t, positions, np.zeros(60))[::2]
        states = np.empty((60, 3))
        states[0] = mean + rng.normal(size=3) * np.sqrt(shape)
        states[1:] = states[0] + np.cumsum(
            rng.normal(size=(59, 3)) * np.sqrt(0.03 * shape), axis=0
        )
        counts = rng.poisson(
            0.1 * np.exp(np.sum(np.asarray(z) * states, axis=1))
        ).astype(float)
        valid = np.asarray(valid).copy()
        valid[[3, 9, 10, 17]] = False
        for q in (0.0, 0.0003, 0.003, 0.03, 0.3):
            noise = np.diag(q * shape)
            dense = joint_poisson_laplace(
                mean, p, noise, np.asarray(z), counts, valid, 0.1
            )
            joint = poisson_laplace_smoother(
                jnp.asarray(mean),
                jnp.asarray(p),
                jnp.asarray(a),
                jnp.asarray(noise),
                z,
                jnp.asarray(counts),
                jnp.asarray(valid),
                dt=0.1,
            )
            fm, fc, ll = _masked_graph_point_process_filter(
                jnp.asarray(mean),
                jnp.asarray(p),
                z,
                jnp.asarray(counts),
                jnp.asarray(valid),
                jnp.asarray(a),
                jnp.asarray(noise),
                dt=0.1,
                max_log_count=600.0,
                max_newton_iter=25,
            )
            sm, sc, _ = rts_backward_scan(fm, fc, jnp.asarray(a), jnp.asarray(noise))
            scale = np.sqrt(np.diagonal(dense.covariance, axis1=1, axis2=2))
            records.append(
                dict(
                    seed=seed,
                    q=q,
                    dense_evidence=dense.log_evidence,
                    sequential_evidence=float(ll),
                    joint_evidence=float(joint.log_evidence),
                    sequential_mode_max_sd=float(
                        np.max(np.abs(np.asarray(sm) - dense.mean) / scale)
                    ),
                    joint_mode_max_sd=float(
                        np.max(np.abs(np.asarray(joint.mean) - dense.mean) / scale)
                    ),
                    joint_covariance_max_error=float(
                        np.max(np.abs(np.asarray(joint.covariance) - dense.covariance))
                    ),
                    joint_lag_max_error=float(
                        np.max(
                            np.abs(
                                np.asarray(joint.cross_covariance)
                                - dense.cross_covariance
                            )
                        )
                    ),
                    relative_newton_step=float(joint.relative_newton_step),
                )
            )
    orders = []
    for y in ([8.0, 0.0], [0.0, 8.0]):
        _, _, ll = _masked_graph_point_process_filter(
            jnp.zeros(1),
            jnp.eye(1),
            jnp.ones((2, 1)),
            jnp.array(y),
            jnp.ones(2, dtype=bool),
            jnp.eye(1),
            jnp.zeros((1, 1)),
            dt=0.1,
            max_log_count=600.0,
            max_newton_iter=25,
        )
        joint = poisson_laplace_smoother(
            jnp.zeros(1),
            jnp.eye(1),
            jnp.eye(1),
            jnp.zeros((1, 1)),
            jnp.ones((2, 1)),
            jnp.array(y),
            jnp.ones(2, dtype=bool),
            dt=0.1,
        )
        orders.append(
            dict(
                counts=y,
                sequential_evidence=float(ll),
                joint_evidence=float(joint.log_evidence),
            )
        )
    return dict(
        software=software_versions(),
        protocol=dict(
            seeds=[0, 1, 2],
            rank=3,
            n_time=60,
            true_q=0.03,
            known_nuisance=True,
            local_newton_iterations=25,
            oracle="Independent dense whitened innovation posterior/Hessian, SciPy trust-exact",
        ),
        records=records,
        static_order_counterexample=orders,
        summary=dict(
            max_joint_evidence_error=max(
                abs(r["joint_evidence"] - r["dense_evidence"]) for r in records
            ),
            max_joint_mode_error_sd=max(r["joint_mode_max_sd"] for r in records),
            max_sequential_evidence_error=max(
                abs(r["sequential_evidence"] - r["dense_evidence"]) for r in records
            ),
        ),
    )


if __name__ == "__main__":
    result = experiment()
    path = Path("docs/validation/graph-place-field-learning/joint_reference.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))

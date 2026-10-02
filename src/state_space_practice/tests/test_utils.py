"""Tests for the utils module."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.utils import (
    _root_figure,
    _validate_filter_numerics,
    check_converged,
    clip_eigenvalues_relative,
    compute_state_overlap,
    divide_safe,
    floor_variances_relative,
    hmm_viterbi,
    make_discrete_transition_matrix,
    project_psd,
    project_psd_relative,
    psd_cholesky,
    psd_logdet,
    psd_solve,
    relative_psd_floor,
    safe_log,
    scale_likelihood,
    shift_to_psd,
    spectral_radius,
    stabilize_covariance,
    stabilize_probability_vector,
    stabilize_transition_matrix,
    symmetrize,
    typed_jit,
    validate_covariance,
    validate_int,
    validate_probability_vector,
    validate_transition_matrix,
    zero_preserving_log,
)


class TestTypedJit:
    """typed_jit must behave exactly like jax.jit at runtime."""

    def test_compiles_once_and_matches_eager(self) -> None:
        n_traces = 0

        def f(x: jax.Array) -> jax.Array:
            nonlocal n_traces
            n_traces += 1
            return jnp.sin(x) * 2.0

        jitted = typed_jit(f)
        x = jnp.linspace(0.0, 1.0, 5)
        first = jitted(x)
        second = jitted(x + 1.0)

        assert n_traces == 1  # second call hit the compilation cache
        np.testing.assert_allclose(first, jnp.sin(x) * 2.0)
        np.testing.assert_allclose(second, jnp.sin(x + 1.0) * 2.0)

    def test_forwards_static_argnames(self) -> None:
        def f(x: jax.Array, power: int) -> jax.Array:
            # Python control flow on `power` needs it to be static.
            return x**power if power > 0 else jnp.ones_like(x)

        x = jnp.array([2.0, 3.0])
        static = typed_jit(f, static_argnames=("power",))
        np.testing.assert_allclose(static(x, power=2), [4.0, 9.0])
        np.testing.assert_allclose(static(x, power=0), [1.0, 1.0])

        # Guard: without static_argnames the same function cannot trace.
        with pytest.raises(jax.errors.TracerBoolConversionError):
            typed_jit(f)(x, 2)


class TestRootFigure:
    """Plot helpers call tight_layout on the root Figure, never a SubFigure."""

    def test_axes_in_subfigure_resolve_to_root_figure(self) -> None:
        plt = pytest.importorskip("matplotlib.pyplot")
        fig = plt.figure()
        try:
            sub = fig.subfigures(1, 2)[0]
            ax = sub.subplots()
            # Guard: ax.figure is the SubFigure, which has no tight_layout.
            assert ax.figure is sub
            assert not hasattr(ax.figure, "tight_layout")

            root = _root_figure(ax)
            assert root is fig
            root.tight_layout()
        finally:
            plt.close(fig)


class TestCheckConverged:
    """Tests for the check_converged function."""

    def test_identical_values_converged(self) -> None:
        """Identical log-likelihoods should indicate convergence."""
        ll = -100.0
        is_converged, is_increasing = check_converged(ll, ll, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True

    def test_small_relative_change_converged(self) -> None:
        """Small relative change below tolerance should converge."""
        ll_prev = -1000.0
        # Change of 0.01 on average of 1000 = 1e-5 relative change
        ll_curr = -999.99
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True

    def test_large_relative_change_not_converged(self) -> None:
        """Large relative change above tolerance should not converge."""
        ll_prev = -100.0
        ll_curr = -90.0  # 10% change
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_converged is False
        assert is_increasing is True

    def test_increasing_log_likelihood(self) -> None:
        """Increasing log-likelihood should be flagged as increasing."""
        ll_prev = -100.0
        ll_curr = -50.0  # LL increased (less negative)
        _, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_increasing is True

    def test_decreasing_log_likelihood_beyond_tolerance(self) -> None:
        """Decreasing log-likelihood beyond tolerance should flag as not increasing."""
        ll_prev = -50.0
        ll_curr = -100.0  # LL decreased significantly
        _, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_increasing is False

    def test_slight_decrease_within_tolerance_is_increasing(self) -> None:
        """Slight decrease within tolerance should still be considered increasing."""
        ll_prev = -100.0
        ll_curr = -100.00005  # Decrease by 5e-5, within 1e-4 tolerance
        _, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_increasing is True

    def test_zero_log_likelihoods(self) -> None:
        """Zero log-likelihoods should be handled without division errors."""
        is_converged, is_increasing = check_converged(0.0, 0.0, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True

    def test_negative_infinity_previous(self) -> None:
        """Negative infinity previous LL (initial state) should not converge."""
        ll_prev = -np.inf
        ll_curr = -100.0
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        # Any finite value is an improvement from -inf
        assert is_converged is False
        assert is_increasing is True

    def test_both_negative_infinity(self) -> None:
        """Both -inf should be considered converged (no change)."""
        ll_prev = -np.inf
        ll_curr = -np.inf
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        # inf - inf = nan, but the function should handle this
        # The actual behavior depends on implementation
        assert isinstance(is_converged, bool)
        assert isinstance(is_increasing, bool)

    def test_positive_log_likelihoods(self) -> None:
        """Positive log-likelihoods (valid in some contexts) should work."""
        ll_prev = 10.0
        ll_curr = 10.0001
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True

    def test_tolerance_boundary_exact(self) -> None:
        """Test behavior exactly at the tolerance boundary."""
        ll_prev = -100.0
        # Create a change exactly at tolerance
        avg = 100.0  # |ll_prev| + |ll_curr| / 2 ≈ 100
        tolerance = 1e-4
        delta = tolerance * avg  # 0.01
        ll_curr = ll_prev + delta

        is_converged, _ = check_converged(ll_curr, ll_prev, tolerance=tolerance)
        # At boundary, should not be converged (strictly less than)
        assert is_converged is False

    def test_custom_tolerance(self) -> None:
        """Custom tolerance values should be respected."""
        ll_prev = -100.0
        ll_curr = -99.0  # 1% change

        # With 1% tolerance, should converge
        is_converged_loose, _ = check_converged(ll_curr, ll_prev, tolerance=0.02)
        assert is_converged_loose is True

        # With 0.1% tolerance, should not converge
        is_converged_tight, _ = check_converged(ll_curr, ll_prev, tolerance=0.001)
        assert is_converged_tight is False

    def test_returns_python_bools(self) -> None:
        """Function should return Python bools, not numpy bools."""
        is_converged, is_increasing = check_converged(-100.0, -100.0)
        assert type(is_converged) is bool
        assert type(is_increasing) is bool

    def test_very_small_values(self) -> None:
        """Very small log-likelihoods should be handled correctly."""
        ll_prev = -1e-10
        ll_curr = -1e-10
        is_converged, is_increasing = check_converged(ll_curr, ll_prev)
        assert is_converged is True
        assert is_increasing is True

    def test_very_large_values(self) -> None:
        """Very large log-likelihoods should be handled correctly."""
        ll_prev = -1e10
        ll_curr = -1e10 + 1  # Small absolute change, tiny relative change
        is_converged, is_increasing = check_converged(ll_curr, ll_prev)
        assert is_converged is True
        assert is_increasing is True

    def test_near_zero_absolute_wobble_converges(self) -> None:
        """Near-zero likelihoods should not explode the relative-change check."""
        is_converged, is_increasing = check_converged(-1e-12, 1e-12, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True


class TestLinearAlgebraUtilities:
    """Tests for linear algebra utilities moved from kalman.py."""

    def test_symmetrize_makes_symmetric(self) -> None:
        A = jnp.array([[1.0, 2.0], [4.0, 5.0]])
        S = symmetrize(A)
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        np.testing.assert_allclose(S, jnp.array([[1.0, 3.0], [3.0, 5.0]]))

    def test_symmetrize_batch(self) -> None:
        """Symmetrize should handle batched matrices (last two dims)."""
        A = jnp.stack([jnp.array([[1.0, 2.0], [4.0, 5.0]]), jnp.eye(2) * 3.0])
        S = symmetrize(A)
        np.testing.assert_allclose(S, jnp.swapaxes(S, -1, -2), atol=1e-12)

    def test_psd_solve_matches_numpy(self) -> None:
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x = psd_solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, atol=1e-6)

    def test_psd_solve_relative_boost_scales_with_max_diag(self) -> None:
        """psd_solve's relative_boost must scale with the diagonal of A.

        For a large-scale matrix (max_diag ~ 1e6) the effective shift is
        ``relative_boost * max_diag`` (~1e-6 here), which must still
        preserve the relative signal.
        """
        # max_diag ~ 1e6, well-conditioned otherwise
        A = jnp.array([[1.0e6, 100.0], [100.0, 5.0e5]])
        b = jnp.array([1.0, 2.0])

        # Reference solution via direct inverse
        expected = np.linalg.solve(np.asarray(A), np.asarray(b))

        # psd_solve should still agree to reasonable precision despite
        # the larger effective shift — the relative shift preserves
        # relative signal.
        x = psd_solve(A, b)
        np.testing.assert_allclose(
            np.asarray(x),
            expected,
            rtol=1e-5,
            atol=0,
            err_msg="relative boost should not swamp the signal",
        )

    def test_psd_solve_small_scale_unchanged(self) -> None:
        """On an O(1) well-conditioned matrix the relative shift
        (``1e-12 * max_diag``) is negligible against the eigenvalues."""
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x = psd_solve(A, b)
        expected = np.linalg.solve(np.asarray(A), np.asarray(b))
        np.testing.assert_allclose(np.asarray(x), expected, atol=1e-6)

    def test_psd_solve_near_singular_stabilized(self) -> None:
        """A rank-deficient PSD matrix has infinitely many solutions;
        this test only asserts ``psd_solve`` returns a finite output
        (does not crash or NaN), not that the solution is unique or
        meaningful. The regression target is the f32 NaN / Cholesky-
        failure mode, not solution correctness.
        """
        # Construct a 4x4 rank-2 PSD matrix
        A_low_rank = jax.random.normal(jax.random.PRNGKey(0), (4, 2))
        A = A_low_rank @ A_low_rank.T  # rank 2, 2 zero eigenvalues
        b = jnp.array([1.0, 2.0, 3.0, 4.0])
        x = psd_solve(A, b)
        assert jnp.all(jnp.isfinite(x)), (
            "psd_solve should return finite output on rank-deficient input"
        )

    def test_psd_solve_relative_boost_zero_matches_old_behavior(self) -> None:
        """``relative_boost=0.0`` disables relative scaling entirely.

        With the default ``diagonal_boost=0.0`` only the ``sqrt(tiny)``
        floor remains, so the solve is the plain Cholesky solve.
        """
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x_default = psd_solve(A, b)
        x_no_relative = psd_solve(A, b, relative_boost=0.0)
        # The default relative shift (1e-12 * 4) is negligible here.
        np.testing.assert_allclose(
            np.asarray(x_default), np.asarray(x_no_relative), atol=1e-11
        )
        np.testing.assert_allclose(
            np.asarray(x_no_relative),
            np.linalg.solve(np.asarray(A), np.asarray(b)),
            rtol=1e-14,
        )

    def test_psd_solve_zero_matrix_falls_back_to_tiny_floor(self) -> None:
        """All-zero matrix: ``max_diag=0``, so the relative shift vanishes
        and only the ``sqrt(tiny)`` floor regularizes the otherwise-singular
        system. The solver must still produce a finite output."""
        A = jnp.zeros((3, 3))
        b = jnp.array([1.0, 2.0, 3.0])
        x = psd_solve(A, b)
        assert jnp.all(jnp.isfinite(x))

    @pytest.mark.parametrize("scale", [1e-10, 1e-6, 1e6])
    def test_default_shift_is_scale_equivariant(self, scale) -> None:
        """psd_solve(s A, b) = psd_solve(A, b) / s and logdet(s A) =
        logdet(A) + n log s: the default shift is relative to the scale."""
        rng = np.random.default_rng(0)
        m = rng.normal(size=(5, 3))
        A = jnp.asarray(m @ m.T + 1e-3 * np.eye(5))  # eigenvalues ~1e-3 .. 10
        b = jnp.asarray(rng.normal(size=5))
        x = psd_solve(A, b)
        np.testing.assert_allclose(psd_solve(scale * A, b), x / scale, rtol=1e-11)
        logdet = psd_logdet(psd_cholesky(A))
        np.testing.assert_allclose(
            psd_logdet(psd_cholesky(scale * A)),
            logdet + 5 * np.log(scale),
            rtol=1e-12,
        )
        if scale < 1.0:
            # Guard: an absolute shift (the former 1e-9 default) is not
            # equivariant once the matrix is small.
            x_abs = psd_solve(scale * A, b, diagonal_boost=1e-9)
            assert not np.allclose(np.asarray(x_abs) * scale, np.asarray(x), rtol=1e-3)

    def test_shift_is_per_coordinate_and_block_local(self) -> None:
        """The default shift is ``1e-12 * diag(A)``: equivariant under a
        per-coordinate rescaling ``D A D`` and identical whether a
        block-diagonal matrix is factored whole or block by block."""
        rng = np.random.default_rng(1)
        m = rng.normal(size=(4, 2))
        A = m @ m.T + 1e-4 * np.eye(4)  # near-singular: the shift matters
        D = np.diag([0.1, 1.0, 30.0, 0.03])
        b = rng.normal(size=4)
        x = np.asarray(psd_solve(jnp.asarray(A), jnp.asarray(b)))
        x_scaled = np.asarray(psd_solve(jnp.asarray(D @ A @ D), jnp.asarray(D @ b)))
        np.testing.assert_allclose(D @ x_scaled, x, rtol=1e-9)
        # Block-local: blockdiag(A, 1e-6 B) whole == per block.
        B = 1e-6 * (A[:2, :2] + np.eye(2))
        whole = np.zeros((6, 6))
        whole[:4, :4], whole[4:, 4:] = A, B
        logdet_whole = psd_logdet(psd_cholesky(jnp.asarray(whole)))
        logdet_blocks = psd_logdet(psd_cholesky(jnp.asarray(A))) + psd_logdet(
            psd_cholesky(jnp.asarray(B))
        )
        np.testing.assert_allclose(logdet_whole, logdet_blocks, rtol=1e-14)
        # Guard: a shift relative to max|diag| would not be block-local here.
        shift_max = 1e-12 * np.max(np.diag(whole))
        logdet_max_shift = np.linalg.slogdet(whole + shift_max * np.eye(6))[1]
        assert abs(logdet_max_shift - float(logdet_blocks)) > 1e-6

    def test_explicit_absolute_boost_is_still_honoured(self) -> None:
        """Callers that pass ``diagonal_boost`` keep an absolute floor."""
        A = jnp.eye(2) * 1e-6
        b = jnp.ones(2)
        np.testing.assert_allclose(psd_solve(A, b), 1e6 * np.ones(2), rtol=1e-10)
        np.testing.assert_allclose(
            psd_solve(A, b, diagonal_boost=1e-6), 0.5e6 * np.ones(2), rtol=1e-10
        )

    def test_float32_default_shift_is_machine_epsilon(self) -> None:
        """In float32 a 1e-12 relative shift rounds away; the default uses
        float32 epsilon, which keeps a singular PSD matrix factorable."""
        singular = jnp.ones((2, 2), dtype=jnp.float32)
        assert np.isfinite(float(psd_logdet(psd_cholesky(singular))))
        assert not np.isfinite(
            float(psd_logdet(psd_cholesky(singular, relative_boost=1e-12)))
        )

    def test_psd_solve_batched_boost_matches_vmap(self) -> None:
        """Direct batches should get the same per-matrix boost as vmapped calls."""
        A = jnp.stack([jnp.eye(2), 1.0e6 * jnp.eye(2)])
        b = jnp.stack([jnp.ones(2), jnp.ones(2)])

        batched = psd_solve(A, b, diagonal_boost=0.0, relative_boost=1e-6)
        vmapped = jax.vmap(
            lambda A_i, b_i: psd_solve(
                A_i, b_i, diagonal_boost=0.0, relative_boost=1e-6
            )
        )(A, b)

        np.testing.assert_allclose(batched, vmapped, rtol=1e-6, atol=1e-12)
        assert float(batched[0, 0]) > 0.99

    def test_project_psd_clips_negative_eigvals(self) -> None:
        Q = jnp.array([[1.0, 0.5], [0.5, -0.1]])  # indefinite
        Q_psd = project_psd(Q, min_eigenvalue=1e-4)
        eigvals = jnp.linalg.eigvalsh(Q_psd)
        assert jnp.all(eigvals >= 1e-4 - 1e-8)

    def test_stabilize_covariance_preserves_psd_input(self) -> None:
        C = jnp.array([[2.0, 0.5], [0.5, 1.0]])  # PSD already
        stabilized = stabilize_covariance(C)
        np.testing.assert_allclose(stabilized, C, atol=1e-6)


def _degenerate_block_cov(coupling: float) -> jax.Array:
    """A 4x4 symmetric matrix with paired (degenerate) eigenvalues 0.1 +/- c.

    This is the eigenvalue structure that makes eigenvector-reconstruction
    projections (project_psd / stabilize_covariance) produce NaN gradients:
    the ``1/(lambda_i - lambda_j)`` terms diverge on the degenerate pairs.
    """
    return jnp.array(
        [
            [0.1, 0.0, coupling, 0.0],
            [0.0, 0.1, 0.0, coupling],
            [coupling, 0.0, 0.1, 0.0],
            [0.0, coupling, 0.0, 0.1],
        ]
    )


class TestShiftToPsd:
    """Tests for the gradient-safe PSD shift used inside SGD losses."""

    def test_identity_on_psd_input(self) -> None:
        C = jnp.array([[2.0, 0.5], [0.5, 1.0]])  # PSD already
        np.testing.assert_allclose(shift_to_psd(C), C, atol=1e-12)

    def test_lifts_indefinite_to_psd(self) -> None:
        Q = _degenerate_block_cov(0.5)  # eigenvalues 0.1 +/- 0.5 -> min -0.4
        assert jnp.linalg.eigvalsh(Q).min() < 0.0  # guard: input is indefinite
        lifted = shift_to_psd(Q, min_eigenvalue=1e-6)
        assert jnp.linalg.eigvalsh(lifted).min() >= 1e-6 - 1e-9

    def test_shift_is_isotropic_and_symmetric(self) -> None:
        Q = _degenerate_block_cov(0.5)
        lifted = shift_to_psd(Q)
        # Only the diagonal is shifted (by a single scalar); off-diagonals unchanged.
        np.testing.assert_allclose(
            lifted - Q, jnp.diag(jnp.diag(lifted - Q)), atol=1e-12
        )
        diag_shift = jnp.diag(lifted - Q)
        np.testing.assert_allclose(diag_shift, diag_shift[0], atol=1e-12)
        np.testing.assert_allclose(lifted, lifted.T, atol=1e-12)

    def test_gradient_finite_through_degenerate_indefinite(self) -> None:
        # The whole point of shift_to_psd: a differentiated projection that stays
        # finite where eigenvector-reconstruction projections NaN. Guard that the
        # barrier is actually active (Q indefinite) at the evaluation point.
        def loss(c: float) -> jax.Array:
            return shift_to_psd(_degenerate_block_cov(c)).sum()

        c_eval = 0.5
        assert jnp.linalg.eigvalsh(_degenerate_block_cov(c_eval)).min() < 0.0
        grad = jax.grad(loss)(c_eval)
        assert bool(jnp.isfinite(grad))
        # The eigenvector-reconstruction projection NaNs here -- this is the bug
        # shift_to_psd exists to avoid.
        assert not bool(
            jnp.isfinite(
                jax.grad(
                    lambda c: stabilize_covariance(_degenerate_block_cov(c)).sum()
                )(c_eval)
            )
        )


class TestProbabilityUtilities:
    """Tests for probability utilities moved from switching_kalman.py."""

    def test_divide_safe_zero_denominator(self) -> None:
        """divide_safe should return 0 when denominator is 0."""
        result = divide_safe(jnp.array(5.0), jnp.array(0.0))
        assert float(result) == 0.0

    def test_divide_safe_normal(self) -> None:
        result = divide_safe(jnp.array(6.0), jnp.array(2.0))
        assert float(result) == 3.0

    def test_divide_safe_elementwise(self) -> None:
        num = jnp.array([2.0, 4.0, 6.0])
        den = jnp.array([1.0, 0.0, 3.0])
        result = divide_safe(num, den)
        np.testing.assert_allclose(result, jnp.array([2.0, 0.0, 2.0]))

    def test_divide_safe_zero_denominator_gradient_is_finite(self) -> None:
        grad = jax.grad(lambda d: divide_safe(jnp.array(1.0), d))(jnp.array(0.0))
        assert float(grad) == 0.0

    def test_safe_log_at_floor_boundary(self) -> None:
        """safe_log should floor small inputs to log(_LOG_PROB_FLOOR), not -inf."""
        result = safe_log(jnp.array(0.0))
        np.testing.assert_allclose(result, np.log(1e-10), rtol=1e-6)

    def test_safe_log_normal(self) -> None:
        """safe_log should match jnp.log for values above the floor."""
        x = jnp.array(0.5)
        np.testing.assert_allclose(safe_log(x), jnp.log(x), atol=1e-6)

    def test_safe_log_is_monotone_at_floor(self) -> None:
        below = safe_log(jnp.array(0.5e-10))
        above = safe_log(jnp.array(1.01e-10))
        assert float(below) < float(above)

    def test_safe_log_zero_gradient_is_finite(self) -> None:
        grad = jax.grad(lambda x: safe_log(x))(jnp.array(0.0))
        assert float(grad) == 0.0

    def test_stabilize_probability_vector_sums_to_one(self) -> None:
        p = jnp.array([0.1, 0.2, 0.7])
        stabilized = stabilize_probability_vector(p)
        np.testing.assert_allclose(float(jnp.sum(stabilized)), 1.0, atol=1e-12)

    def test_stabilize_probability_vector_all_zeros_gives_uniform(self) -> None:
        """All-zero input should be lifted to uniform after re-normalization."""
        p = jnp.zeros(4)
        stabilized = stabilize_probability_vector(p)
        np.testing.assert_allclose(stabilized, jnp.ones(4) / 4.0, atol=1e-6)
        np.testing.assert_allclose(float(jnp.sum(stabilized)), 1.0, atol=1e-12)

    def test_stabilize_probability_vector_preserves_dominant_state(self) -> None:
        """A peaked distribution should remain peaked after stabilization."""
        p = jnp.array([1.0, 0.0, 0.0])
        stabilized = stabilize_probability_vector(p)
        assert float(stabilized[0]) > 0.99

    def test_stabilize_probability_vector_sanitizes_nan(self) -> None:
        p = jnp.array([0.5, jnp.nan, 0.5])
        stabilized = stabilize_probability_vector(p)
        assert bool(jnp.all(jnp.isfinite(stabilized)))
        np.testing.assert_allclose(float(jnp.sum(stabilized)), 1.0, atol=1e-12)
        assert float(stabilized[1]) < 1e-6

    def test_stabilize_probability_vector_all_nan_gives_uniform(self) -> None:
        p = jnp.array([jnp.nan, jnp.nan, jnp.nan])
        stabilized = stabilize_probability_vector(p)
        np.testing.assert_allclose(stabilized, jnp.ones(3) / 3.0, atol=1e-6)

    def test_scale_likelihood_subtracts_max(self) -> None:
        ll = jnp.array([-10.0, -5.0, -20.0])
        scaled, ll_max = scale_likelihood(ll)
        assert float(ll_max) == -5.0
        # The max-scaled likelihood entry should be 1.0
        assert float(jnp.max(scaled)) == 1.0

    def test_scale_likelihood_all_neg_inf_uses_zero_max(self) -> None:
        """All -inf input: ll_max should fall back to 0.0, preventing NaN."""
        ll = jnp.full((3,), -jnp.inf)
        scaled, ll_max = scale_likelihood(ll)
        assert float(ll_max) == 0.0
        # exp(-inf - 0) = 0.0
        np.testing.assert_allclose(scaled, jnp.zeros(3))

    def test_scale_likelihood_positive_infinity_uses_indicator(self) -> None:
        ll = jnp.array([1.0, jnp.inf, jnp.inf])
        scaled, ll_max = scale_likelihood(ll)
        assert bool(jnp.isposinf(ll_max))
        np.testing.assert_allclose(scaled, jnp.array([0.0, 1.0, 1.0]))

    def test_zero_preserving_log_contract(self) -> None:
        """Exact zero -> -inf, positive -> exact log (no floor), NaN/neg -> NaN."""
        p = jnp.array([0.0, 0.25, 1e-300, jnp.nan, -0.1])
        result = np.asarray(zero_preserving_log(p))

        assert result[0] == -np.inf
        # Unfloored: 1e-300 keeps its true log (far below e.g. log(1e-10)).
        np.testing.assert_allclose(result[1:3], np.log([0.25, 1e-300]))
        assert np.isnan(result[3]) and np.isnan(result[4])


class TestValidateCovariance:
    """Tests for validate_covariance (symmetric-PSD guard)."""

    def test_symmetric_pd_passes(self) -> None:
        validate_covariance(jnp.eye(3))  # should not raise
        # Guard: the check is live for this input -- flipping one diagonal
        # entry's sign makes it indefinite and must be rejected.
        with pytest.raises(ValueError, match="not positive definite"):
            validate_covariance(jnp.eye(3).at[0, 0].set(-1.0))

    def test_per_state_stack_passes(self) -> None:
        cov = jnp.stack([jnp.eye(2), 2.0 * jnp.eye(2)], axis=-1)  # (2, 2, 2)
        validate_covariance(cov)
        # Guard: every slice is checked, not just the first.
        with pytest.raises(ValueError, match=r"\[\.\.\., 1\]"):
            validate_covariance(cov.at[0, 0, 1].set(-2.0))

    def test_asymmetric_raises(self) -> None:
        # Symmetric eigvalsh would NOT catch this; the raw-matrix check must.
        asym = jnp.array([[1.0, 0.5], [-0.5, 1.0]])
        with pytest.raises(ValueError, match="not symmetric"):
            validate_covariance(asym)

    def test_indefinite_raises(self) -> None:
        indef = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
        with pytest.raises(ValueError, match="not positive definite"):
            validate_covariance(indef)

    def test_per_state_names_offending_index(self) -> None:
        cov = jnp.stack([jnp.eye(2), jnp.array([[1.0, 2.0], [2.0, 1.0]])], axis=-1)
        with pytest.raises(ValueError, match=r"\[\.\.\., 1\]"):
            validate_covariance(cov)

    def test_semidefinite_allowed_when_not_requiring_pd(self) -> None:
        psd_singular = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        validate_covariance(psd_singular, require_positive_definite=False)
        with pytest.raises(ValueError, match="not positive definite"):
            validate_covariance(psd_singular, require_positive_definite=True)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            validate_covariance(jnp.ones((2, 3)))

    def test_non_finite_raises(self) -> None:
        # diag([inf, 1]) is "symmetric" and has all eigenvalues >= 0, so only an
        # explicit finiteness check catches it.
        with pytest.raises(ValueError, match="non-finite"):
            validate_covariance(jnp.diag(jnp.array([jnp.inf, 1.0])))

    def test_empty_matrix_raises(self) -> None:
        # A 0x0 matrix satisfies the square check and the symmetry / eigenvalue
        # checks vacuously; only an explicit empty guard rejects it.
        with pytest.raises(ValueError, match="empty"):
            validate_covariance(jnp.zeros((0, 0)))

    def test_empty_state_stack_raises(self) -> None:
        # A (d, d, 0) stack builds an empty per-slice loop, so every slice
        # check is skipped and the input would otherwise pass silently.
        with pytest.raises(ValueError, match="empty"):
            validate_covariance(jnp.zeros((2, 2, 0)))


class TestValidateTransitionMatrix:
    """Tests for validate_transition_matrix (row-stochastic guard)."""

    def test_row_stochastic_passes(self) -> None:
        valid = jnp.array([[0.9, 0.1], [0.2, 0.8]])
        validate_transition_matrix(valid)
        # Guard: perturbing one row off the simplex must be rejected.
        with pytest.raises(ValueError, match="sum to 1"):
            validate_transition_matrix(valid.at[1, 1].set(0.9))

    def test_rows_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            validate_transition_matrix(jnp.array([[0.9, 0.9], [0.1, 0.1]]))

    def test_negative_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            validate_transition_matrix(jnp.array([[1.2, -0.2], [0.3, 0.7]]))

    def test_non_finite_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            validate_transition_matrix(jnp.array([[jnp.nan, 0.0], [0.5, 0.5]]))

    def test_empty_matrix_raises(self) -> None:
        # A 0x0 matrix is square and makes the non-negativity / row-sum
        # reductions vacuously pass; an explicit empty guard is required.
        with pytest.raises(ValueError, match="empty"):
            validate_transition_matrix(jnp.zeros((0, 0)))


class TestValidateProbabilityVector:
    """Tests for validate_probability_vector (simplex guard)."""

    def test_valid_simplex_passes(self) -> None:
        valid = jnp.array([0.2, 0.3, 0.5])
        validate_probability_vector(valid)
        # Guard: a vector 0.1 off the simplex must be rejected.
        with pytest.raises(ValueError, match="sum to 1"):
            validate_probability_vector(valid.at[2].set(0.6))

    def test_unnormalized_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            validate_probability_vector(jnp.array([0.5, 0.6, 0.3]))

    def test_negative_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            validate_probability_vector(jnp.array([1.2, -0.2]))

    def test_non_finite_raises(self) -> None:
        # [0.5, nan] would slip past the sum check: abs(nan - 1) > atol is False.
        with pytest.raises(ValueError, match="non-finite"):
            validate_probability_vector(jnp.array([0.5, jnp.nan]))

    def test_matrix_raises(self) -> None:
        with pytest.raises(ValueError, match="1D probability vector"):
            validate_probability_vector(jnp.array([[0.25, 0.25], [0.25, 0.25]]))


class TestValidateInt:
    """validate_int accepts exactly what ``operator.index`` accepts."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (3, 3),
            (np.int32(3), 3),
            (np.array(3), 3),
            (jnp.array(3), 3),
            (True, 1),
        ],
        ids=["int", "np_int32", "np_0d", "jax_0d", "bool"],
    )
    def test_accepts_integer_like(self, value, expected: int) -> None:
        result = validate_int(value, "n")
        assert type(result) is int and result == expected

    @pytest.mark.parametrize(
        "value",
        [3.0, np.float64(3.0), np.array([3]), "3", None],
        ids=["float", "np_float", "1d_array", "str", "none"],
    )
    def test_rejects_non_integers(self, value) -> None:
        with pytest.raises(ValueError, match="n must be an integer"):
            validate_int(value, "n")

    def test_range_checks(self) -> None:
        assert validate_int(0, "n", nonnegative=True) == 0
        with pytest.raises(ValueError, match="n must be a positive integer"):
            validate_int(0, "n", positive=True)
        with pytest.raises(ValueError, match="n must be a non-negative integer"):
            validate_int(-1, "n", nonnegative=True)


class TestDiscreteStateUtilities:
    """Tests for discrete-state transition, Viterbi, and alignment helpers."""

    def test_make_discrete_transition_matrix_shape_guard(self) -> None:
        with pytest.raises(ValueError, match="diag must have shape"):
            make_discrete_transition_matrix(jnp.array([0.9, 0.8]), 3)

    def test_make_discrete_transition_matrix_requires_state(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            make_discrete_transition_matrix(jnp.array([]), 0)

    def test_hmm_viterbi_preserves_structural_zero_transitions(self) -> None:
        initial_probs = jnp.array([1.0, 0.0])
        transition_matrix = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        # Observations prefer state 0 at every time, but the transition matrix
        # forces alternation from the known initial state.
        log_likelihoods = jnp.array([[0.0, -10.0], [0.0, -10.0], [0.0, -10.0]])

        states = hmm_viterbi(initial_probs, transition_matrix, log_likelihoods)

        np.testing.assert_array_equal(np.asarray(states), np.array([0, 1, 0]))

    def test_compute_state_overlap_uses_linear_memory_scatter(self) -> None:
        z1 = jnp.array([0, 0, 1, 2, 2])
        z2 = jnp.array([1, 1, 1, 0, 2])
        overlap = compute_state_overlap(z1, z2)
        expected = jnp.array(
            [
                [0, 2, 0],
                [0, 1, 0],
                [1, 0, 1],
            ],
            dtype=jnp.int32,
        )
        np.testing.assert_array_equal(overlap, expected)

    def test_compute_state_overlap_shape_guard(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            compute_state_overlap(jnp.array([0, 1]), jnp.array([0]))


def _scaled_rotation(scale: float, theta: float) -> jnp.ndarray:
    """A non-symmetric 2x2 block ``scale * R(theta)`` with eigenvalue magnitude
    ``scale`` (complex conjugate pair) -- the case that motivates using
    ``eigvals`` over ``eigvalsh``."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return scale * jnp.array([[c, -s], [s, c]])


class TestSpectralRadius:
    def test_matches_eigenvalue_magnitude_on_nonsymmetric_matrix(self) -> None:
        A = _scaled_rotation(scale=1.7, theta=0.6)
        # guard: A is genuinely non-symmetric, so eigvalsh would be wrong here.
        assert not np.allclose(np.asarray(A), np.asarray(A).T)
        assert float(spectral_radius(A)) == pytest.approx(1.7, rel=1e-6)


class TestStabilizeTransitionMatrix:
    def test_scales_unstable_matrix_exactly_to_the_bound(self) -> None:
        A = _scaled_rotation(scale=2.0, theta=0.4)  # radius 2.0 > 0.99
        stabilized = stabilize_transition_matrix(A, max_spectral_radius=0.99)
        # radius pulled to exactly the bound...
        assert float(spectral_radius(stabilized)) == pytest.approx(0.99, rel=1e-6)
        # ...by a uniform scale (every entry * 0.99/2.0), preserving structure.
        np.testing.assert_allclose(
            np.asarray(stabilized), np.asarray(A) * (0.99 / 2.0), rtol=1e-6
        )

    def test_leaves_stable_matrix_unchanged(self) -> None:
        A = _scaled_rotation(scale=0.5, theta=0.4)  # radius 0.5 < 0.99
        # guard: input is genuinely within the bound (test isn't vacuous).
        assert float(spectral_radius(A)) < 0.99
        stabilized = stabilize_transition_matrix(A, max_spectral_radius=0.99)
        np.testing.assert_array_equal(np.asarray(stabilized), np.asarray(A))

    def test_integer_matrix_is_scaled_not_truncated(self) -> None:
        A = jnp.array([[2, 1], [0, 2]])  # integer dtype, radius 2
        assert jnp.issubdtype(A.dtype, jnp.integer)
        stabilized = stabilize_transition_matrix(A, max_spectral_radius=0.99)
        assert jnp.issubdtype(stabilized.dtype, jnp.floating)
        np.testing.assert_allclose(
            np.asarray(stabilized), [[0.99, 0.495], [0.0, 0.99]], rtol=1e-12
        )

    def test_block_size_not_dividing_n_warns_about_uniform_fallback(
        self, caplog
    ) -> None:
        A = jnp.diag(jnp.array([2.0, 0.5, 0.5]))
        with caplog.at_level(logging.WARNING, logger="state_space_practice.utils"):
            stabilized = stabilize_transition_matrix(
                A, max_spectral_radius=0.99, block_size=2
            )
        assert any("does not divide" in r.getMessage() for r in caplog.records)
        # the fallback is one uniform scale over the whole matrix
        np.testing.assert_allclose(np.asarray(stabilized), np.asarray(A) * 0.495)


class TestRelativePsdFloor:
    """Scale-relative eigenvalue floors for M-step covariance projections."""

    @pytest.mark.parametrize("scale", [1e-12, 1.0, 1e8])
    def test_floor_scales_with_matrix(self, scale) -> None:
        eigs = jnp.array([1.0, 1e-3, -1e-20]) * scale
        np.testing.assert_allclose(relative_psd_floor(eigs), 1e-8 * scale, rtol=1e-12)

    def test_zero_matrix_uses_absolute_floor(self) -> None:
        assert float(relative_psd_floor(jnp.zeros(3))) == pytest.approx(1e-30)

    @pytest.mark.parametrize("scale", [1e-10, 1.0, 1e6])
    def test_well_conditioned_matrix_passes_through(self, scale) -> None:
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]]) * scale
        projected, n_floored = clip_eigenvalues_relative(cov)
        np.testing.assert_allclose(projected, cov, rtol=1e-12)
        assert int(n_floored) == 0

    def test_small_scale_matrix_is_not_clamped_to_an_absolute_floor(self) -> None:
        """A covariance in volts^2 (~1e-10) keeps its value; the former
        ``stabilize_covariance(min_eigenvalue=1e-8)`` returned 1e-8 I."""
        cov = jnp.eye(2) * 1e-10
        np.testing.assert_allclose(project_psd_relative(cov), cov, rtol=1e-12)
        # guard: the absolute floor really clamps this matrix.
        np.testing.assert_allclose(stabilize_covariance(cov), jnp.eye(2) * 1e-8)

    def test_indefinite_matrix_is_floored_and_logged(self, caplog) -> None:
        cov = jnp.array([[1.0, 0.0], [0.0, -0.5]])
        with caplog.at_level("WARNING", logger="state_space_practice.utils"):
            projected = project_psd_relative(cov, name="test_cov")
            jax.effects_barrier()
        eigs = np.linalg.eigvalsh(np.asarray(projected))
        np.testing.assert_allclose(eigs, [1e-8, 1.0], rtol=1e-6)
        assert any("test_cov" in r.getMessage() for r in caplog.records)

    def test_materially_negative_eigenvalue_is_reported_as_indefinite(
        self, caplog
    ) -> None:
        """Flooring eigenvalue -5 is not rank deficiency: the log names the
        minimum eigenvalue and points at inconsistent sufficient statistics."""
        cov = jnp.array([[1.0, 0.0], [0.0, -5.0]])
        with caplog.at_level("WARNING", logger="state_space_practice.utils"):
            project_psd_relative(cov, name="test_cov")
            jax.effects_barrier()
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert "indefinite" in messages[0] and "-5" in messages[0]
        assert "inconsistent sufficient statistics" in messages[0]
        assert "rank deficient" not in messages[0]

    def test_roundoff_negative_eigenvalue_is_reported_as_rank_deficient(
        self, caplog
    ) -> None:
        cov = jnp.array([[1.0, 1.0], [1.0, 1.0 - 1e-15]])  # rank one + round-off
        with caplog.at_level("WARNING", logger="state_space_practice.utils"):
            project_psd_relative(cov, name="test_cov")
            jax.effects_barrier()
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert "rank deficient" in messages[0]
        assert "indefinite" not in messages[0]

    def test_no_log_when_nothing_changes(self, caplog) -> None:
        with caplog.at_level("WARNING", logger="state_space_practice.utils"):
            project_psd_relative(jnp.eye(2), name="test_cov")
            jax.effects_barrier()
        assert not caplog.records

    def test_logging_works_under_jit(self, caplog) -> None:
        with caplog.at_level("WARNING", logger="state_space_practice.utils"):
            out = jax.jit(lambda c: project_psd_relative(c, name="jit_cov"))(
                jnp.diag(jnp.array([1.0, 0.0]))
            )
            jax.block_until_ready(out)
            jax.effects_barrier()
        assert any("jit_cov" in r.getMessage() for r in caplog.records)

    def test_small_variance_next_to_large_one_is_unchanged(self) -> None:
        """The floor is per-coordinate: a PD matrix with variances 1e4 and
        1e-5 (ratio 1e-9, below 1e-8 * lambda_max) is left as it is."""
        cov = jnp.diag(jnp.array([1e4, 1e-5]))
        projected, n_floored = clip_eigenvalues_relative(cov)
        np.testing.assert_array_equal(projected, cov)
        assert int(n_floored) == 0

    @pytest.mark.parametrize(
        "cov",
        [
            # rank one (two perfectly correlated coordinates + an independent one)
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
            # indefinite
            [[1.0, 0.9, 0.3], [0.9, 0.5, 0.0], [0.3, 0.0, 1.0]],
        ],
    )
    def test_floor_is_equivariant_under_diagonal_rescaling(self, cov) -> None:
        cov = jnp.asarray(cov)
        s = jnp.array([1e-6, 1.0, 1e6])
        scaled_then_floored, n_scaled = clip_eigenvalues_relative(
            s[:, None] * cov * s[None, :]
        )
        floored, n = clip_eigenvalues_relative(cov)
        # guard: the floor actually binds for this matrix.
        assert int(n) > 0 and int(n_scaled) == int(n)
        np.testing.assert_allclose(
            scaled_then_floored / (s[:, None] * s[None, :]), floored, rtol=1e-9
        )
        assert np.linalg.eigvalsh(np.asarray(floored)).min() > 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_well_conditioned_psd_matrix_is_returned_unchanged(self, seed) -> None:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(4, 4))
        cov = jnp.asarray(X @ X.T + 0.5 * np.eye(4))
        cov = jnp.asarray(
            np.diag([1e-3, 1.0, 1e2, 1e5]) @ cov @ np.diag([1e-3, 1.0, 1e2, 1e5])
        )
        cov = 0.5 * (cov + cov.T)
        projected, n_floored = clip_eigenvalues_relative(cov)
        assert int(n_floored) == 0
        np.testing.assert_array_equal(projected, cov)

    def test_floor_variances_relative(self) -> None:
        v = jnp.array([1e-6, 1e-20, 2e-6])
        np.testing.assert_allclose(
            floor_variances_relative(v, warn=False), [1e-6, 2e-14, 2e-6]
        )


class TestValidateFilterNumericsTracing:
    def test_skipped_under_tracing(self) -> None:
        """Host-side checks cannot inspect a tracer; they are skipped so the
        public filters stay traceable (an eager call still validates)."""

        def f(cov):
            _validate_filter_numerics(cov, n_time=10)
            return jnp.sum(cov)

        assert float(jax.jit(f)(-jnp.eye(2))) == -2.0
        with pytest.raises(ValueError, match="not positive definite"):
            f(-jnp.eye(2))

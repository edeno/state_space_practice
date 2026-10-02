# ruff: noqa: E402
"""Tests for the switching_choice module."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.exceptions import NotFittedError
from state_space_practice.switching_choice import (
    SwitchingChoiceModel,
    _softmax_predict_and_update,
    _softmax_update_per_state_pair,
    simulate_switching_choice_data,
    switching_choice_filter,
    switching_choice_smoother,
)


class TestSoftmaxPredictAndUpdate:
    """Tests for the single-pair predict + softmax update."""

    def test_output_shapes(self):
        k_free = 2  # K=3 options
        mean = jnp.zeros(k_free)
        cov = jnp.eye(k_free)
        A = 0.95 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        post_mean, post_cov, ll, _ = _softmax_predict_and_update(
            mean,
            cov,
            jnp.int32(0),
            A,
            Q,
            3,
            1.0,
            B,
            u,
            obs_offset,
        )
        assert post_mean.shape == (k_free,)
        assert post_cov.shape == (k_free, k_free)
        assert ll.shape == ()

    def test_log_likelihood_finite(self):
        k_free = 2
        mean = jnp.array([0.5, -0.3])
        cov = jnp.eye(k_free) * 0.5
        A = 0.9 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, _, ll, _ = _softmax_predict_and_update(
            mean,
            cov,
            jnp.int32(1),
            A,
            Q,
            3,
            2.0,
            B,
            u,
            obs_offset,
        )
        assert jnp.isfinite(ll)

    def test_covariance_psd(self):
        k_free = 2
        mean = jnp.zeros(k_free)
        cov = jnp.eye(k_free)
        A = 0.95 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, post_cov, _, _ = _softmax_predict_and_update(
            mean,
            cov,
            jnp.int32(0),
            A,
            Q,
            3,
            1.0,
            B,
            u,
            obs_offset,
        )
        eigvals = jnp.linalg.eigvalsh(post_cov)
        assert jnp.all(eigvals > -1e-10)


class TestSoftmaxUpdatePerStatePair:
    """Tests for the double-vmapped per-state-pair update."""

    def test_output_shapes(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([1.0, 3.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, pair_cov, pair_ll, _ = _softmax_update_per_state_pair(
            mean,
            cov,
            jnp.int32(0),
            A,
            Q,
            3,
            betas,
            B,
            u,
            obs_offset,
        )
        assert pair_mean.shape == (k_free, S, S)
        assert pair_cov.shape == (k_free, k_free, S, S)
        assert pair_ll.shape == (S, S)

    def test_single_state_matches_softmax_update(self):
        """S=1 should match _softmax_predict_and_update directly."""
        k_free = 2
        mean = jnp.zeros((k_free, 1))
        cov = jnp.eye(k_free)[:, :, None]
        A = (0.95 * jnp.eye(k_free))[:, :, None]
        Q = (0.01 * jnp.eye(k_free))[:, :, None]
        betas = jnp.array([2.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, pair_cov, pair_ll, _ = _softmax_update_per_state_pair(
            mean,
            cov,
            jnp.int32(1),
            A,
            Q,
            3,
            betas,
            B,
            u,
            obs_offset,
        )

        ref_mean, ref_cov, ref_ll, _ = _softmax_predict_and_update(
            mean[:, 0],
            cov[:, :, 0],
            jnp.int32(1),
            A[:, :, 0],
            Q[:, :, 0],
            3,
            2.0,
            B,
            u,
            obs_offset,
        )

        np.testing.assert_allclose(pair_mean[:, 0, 0], ref_mean, atol=1e-10)
        np.testing.assert_allclose(pair_cov[:, :, 0, 0], ref_cov, atol=1e-10)
        np.testing.assert_allclose(float(pair_ll[0, 0]), float(ref_ll), atol=1e-10)

    def test_different_betas_give_different_posteriors(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([0.5, 5.0])  # very different
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, _, _, _ = _softmax_update_per_state_pair(
            mean,
            cov,
            jnp.int32(0),
            A,
            Q,
            3,
            betas,
            B,
            u,
            obs_offset,
        )
        # Posteriors should differ across next-state axis
        assert not jnp.allclose(pair_mean[:, 0, 0], pair_mean[:, 0, 1])

    def test_all_log_likelihoods_finite(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([1.0, 3.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, _, pair_ll, _ = _softmax_update_per_state_pair(
            mean,
            cov,
            jnp.int32(0),
            A,
            Q,
            3,
            betas,
            B,
            u,
            obs_offset,
        )
        assert jnp.all(jnp.isfinite(pair_ll))


class TestSwitchingChoiceFilter:
    """Tests for the switching choice filter."""

    def test_output_shapes(self):
        n_trials, K, S = 50, 3, 2
        choices = jax.random.randint(jax.random.PRNGKey(0), (n_trials,), 0, K)
        result = switching_choice_filter(
            choices,
            n_options=K,
            n_discrete_states=S,
        )
        assert result.filtered_values.shape == (n_trials, K - 1, S)
        assert result.filtered_covs.shape == (n_trials, K - 1, K - 1, S)
        assert result.discrete_state_probs.shape == (n_trials, S)
        assert result.marginal_log_likelihood.shape == ()

    def test_discrete_probs_sum_to_one(self):
        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        result = switching_choice_filter(choices, n_options=3, n_discrete_states=2)
        row_sums = result.discrete_state_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_discrete_probs_nonnegative(self):
        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        result = switching_choice_filter(choices, n_options=3, n_discrete_states=2)
        assert jnp.all(result.discrete_state_probs >= -1e-10)

    def test_marginal_ll_finite(self):
        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        result = switching_choice_filter(choices, n_options=3, n_discrete_states=2)
        assert jnp.isfinite(result.marginal_log_likelihood)

    def test_preserves_structural_zero_prior(self):
        """A structurally-impossible state (prior 0 + identity Z) stays exactly 0.

        The support mask must come from the sanitized prior's exact zeros
        (`_normalize_initial_discrete_prob` keeps them); flooring the prior to
        ~1e-10 first would resurrect state 1, which then persists under identity
        transitions.
        """
        choices = jnp.array([0, 1, 0, 1, 0])
        result = switching_choice_filter(
            choices,
            n_options=2,
            n_discrete_states=2,
            init_discrete_prob=jnp.array([1.0, 0.0]),
            discrete_transition_matrix=jnp.eye(2),
        )
        np.testing.assert_array_equal(
            np.asarray(result.discrete_state_probs[:, 1]), np.zeros(len(choices))
        )

    def test_marginal_ll_gradient_finite_with_structural_zeros(self):
        """jax.grad through the choice filter with a forbidden column is not NaN.

        Prior [1, 0] + identity transitions make state 1's destination column
        all -inf at t>=2; the gradient-safe -inf handling must keep the gradient
        finite for the SGD path.
        """
        choices = jnp.array([0, 0, 1, 0])

        def marginal_ll(beta):
            return switching_choice_filter(
                choices,
                n_options=2,
                n_discrete_states=2,
                inverse_temperatures=jnp.array([beta, beta]),
                init_discrete_prob=jnp.array([1.0, 0.0]),
                discrete_transition_matrix=jnp.eye(2),
            ).marginal_log_likelihood

        grad = jax.grad(marginal_ll)(1.0)
        assert bool(jnp.isfinite(marginal_ll(1.0)))
        assert bool(jnp.isfinite(grad))

    def test_rejects_invalid_choice_index(self):
        choices = jnp.array([0, 3, 1])

        with pytest.raises(ValueError, match="All choices"):
            switching_choice_filter(choices, n_options=3, n_discrete_states=2)

    def test_single_state_matches_covariate_filter(self):
        """S=1 must produce identical filtered values to CovariateChoiceModel.

        Uses nonzero init_mean, nonzero covariates, and decay != 1 to
        exercise the full prediction path including A @ x at trial 0.
        """
        from state_space_practice.covariate_choice import _covariate_choice_filter_jit

        choices = jax.random.randint(jax.random.PRNGKey(42), (100,), 0, 4)
        k_free = 3
        q = 0.01
        beta = 2.0
        decay = 0.6  # nontrivial decay

        init_mean = jnp.array([0.5, -0.3, 0.1])
        init_cov = jnp.eye(k_free) * 0.5
        covariates = jax.random.normal(jax.random.PRNGKey(1), (100, 2)) * 0.1
        input_gain = jax.random.normal(jax.random.PRNGKey(2), (k_free, 2)) * 0.1

        # Switching filter with S=1
        result_sw = switching_choice_filter(
            choices,
            n_options=4,
            n_discrete_states=1,
            process_noises=jnp.array([q]),
            inverse_temperatures=jnp.array([beta]),
            decays=jnp.array([decay]),
            init_mean=init_mean,
            init_cov=init_cov,
            covariates=covariates,
            input_gain=input_gain,
        )

        # Non-switching CovariateChoiceModel filter
        result_cov = _covariate_choice_filter_jit(
            choices,
            4,
            covariates,
            input_gain,
            jnp.zeros((100, 1)),  # obs_covariates
            jnp.zeros((4, 1)),  # obs_weights
            q,
            beta,
            decay,
            init_mean,
            init_cov,
        )

        # Filtered values must match exactly (same Newton steps, same prediction)
        sw_values = result_sw.filtered_values[:, :, 0]  # (T, K-1)
        cov_values = result_cov.filtered_values  # (T, K-1)
        np.testing.assert_allclose(sw_values, cov_values, atol=1e-6)

    def test_two_state_switching_detected(self):
        """First half exploit (deterministic), second half explore (random)."""
        key = jax.random.PRNGKey(0)
        # Exploit phase: always choose 0
        exploit = jnp.zeros(50, dtype=jnp.int32)
        # Explore phase: random
        explore = jax.random.randint(key, (50,), 0, 3)
        choices = jnp.concatenate([exploit, explore])

        result = switching_choice_filter(
            choices,
            n_options=3,
            n_discrete_states=2,
            inverse_temperatures=jnp.array([5.0, 0.5]),  # high vs low beta
            process_noises=jnp.array([0.001, 0.05]),
        )
        # The filter should detect some difference in state probs
        # between the two halves
        first_half = result.discrete_state_probs[:25].mean(axis=0)
        second_half = result.discrete_state_probs[75:].mean(axis=0)
        assert not jnp.allclose(first_half, second_half, atol=0.05)


class TestSwitchingChoiceSmootherControlInput:
    """The smoother must see the dynamics input ``B @ u_t`` the filter used.

    Regression: ``SwitchingChoiceModel._run_smoother`` called
    ``switching_kalman_smoother``, whose backward step predicts ``A_k m_t``
    without the control input, so with covariates every smoothed mean was
    compared against a prediction missing ``B u_{t+1}`` (a 3.2-unit error on
    the data below). With identical per-state parameters the switching
    smoother must reduce exactly to the control-aware covariate smoother.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        from state_space_practice.covariate_choice import simulate_rl_choice_data

        return simulate_rl_choice_data(
            n_trials=40, n_options=3, seed=3, inverse_temperature=1.0
        )

    def _model(self, n_covariates=2):
        return SwitchingChoiceModel(
            n_options=3,
            n_discrete_states=2,
            n_covariates=n_covariates,
            init_inverse_temperatures=[1.5, 1.5],
            init_process_noises=[0.05, 0.05],
            init_decays=[0.9, 0.9],
        )

    def test_identical_states_match_covariate_smoother(self, data):
        from state_space_practice.covariate_choice import covariate_choice_smoother

        B = 0.5 * jnp.eye(2)
        model = self._model()
        model.input_gain_ = B
        model._covariates = data.covariates
        smooth = model._run_smoother(model._run_filter(data.choices, data.covariates))
        ref = covariate_choice_smoother(
            data.choices,
            3,
            covariates=data.covariates,
            input_gain=B,
            process_noise=0.05,
            inverse_temperature=1.5,
            decay=0.9,
        )
        # 1e-8: the batched (vmapped) and unbatched Newton iterates differ at
        # round-off level.
        np.testing.assert_allclose(smooth[0], ref.smoothed_values, atol=1e-8)
        np.testing.assert_allclose(smooth[1], ref.smoothed_covariances, atol=1e-8)
        np.testing.assert_allclose(smooth[4], ref.smoother_cross_cov, atol=1e-8)
        for s in range(2):
            np.testing.assert_allclose(
                smooth[5][..., s], ref.smoothed_values, atol=1e-8
            )
        # Guard: the control input is large enough that ignoring it (the old
        # behaviour, reproduced by a zero control input) is far off.
        filt = model._run_filter(data.choices, data.covariates)
        k = 2
        stale = switching_choice_smoother(
            filt.filtered_values,
            filt.filtered_covs,
            filt.discrete_state_probs,
            0.05 * jnp.stack([jnp.eye(k)] * 2, axis=-1),
            0.9 * jnp.stack([jnp.eye(k)] * 2, axis=-1),
            model.discrete_transition_matrix_,
            jnp.zeros((40, k)),
        )
        assert float(jnp.max(jnp.abs(stale[0] - ref.smoothed_values))) > 1.0

    def test_zero_control_matches_library_smoother(self, data):
        from state_space_practice.switching_kalman import switching_kalman_smoother

        model = SwitchingChoiceModel(
            n_options=3,
            n_discrete_states=2,
            init_inverse_temperatures=[0.5, 3.0],
            init_process_noises=[0.01, 0.2],
            init_decays=[1.0, 0.8],
        )
        filt = model._run_filter(data.choices)
        ours = model._run_smoother(filt)
        eye = jnp.eye(2)
        lib = switching_kalman_smoother(
            filt.filtered_values,
            filt.filtered_covs,
            filt.discrete_state_probs,
            jnp.stack([q * eye for q in model.process_noises_], axis=-1),
            jnp.stack([a * eye for a in model.decays_], axis=-1),
            model.discrete_transition_matrix_,
        )
        assert len(ours) == len(lib)
        for a, b in zip(ours, lib):
            np.testing.assert_allclose(a, b, atol=1e-12)
        # Guard: the per-state parameters differ, so the discrete smoother
        # actually moved away from uniform.
        assert float(jnp.max(jnp.abs(ours[2] - 0.5))) > 0.05


class TestSwitchingChoiceInitialDiscretePrior:
    """First-trial handling of the caller-supplied discrete prior ``p(S_1)``.

    Contract (``_normalize_initial_discrete_prob`` +
    ``_first_timestep_discrete_update``): a prior that does not sum to a
    positive value fails loud with NaN; a tiny positive prior is used as is,
    not floored; a NaN, infinite or negative entry in an otherwise valid prior
    is clamped to a structural zero and the rest renormalized.
    """

    CHOICES = jnp.array([1, 0, 1, 1])

    def _filter(self, prior):
        """Filter with state-dependent first-trial likelihoods (distinct betas)."""
        n_states = len(prior)
        return switching_choice_filter(
            self.CHOICES,
            n_options=2,
            n_discrete_states=n_states,
            inverse_temperatures=jnp.linspace(0.5, 4.0, n_states),
            init_mean=jnp.array([2.0]),
            init_discrete_prob=jnp.asarray(prior),
            discrete_transition_matrix=jnp.eye(n_states),
        )

    @pytest.mark.parametrize(
        "prior",
        [[0.0, 0.0], [-0.5, 0.0], [np.nan, np.nan]],
        ids=["all_zero", "non_positive", "all_nan"],
    )
    def test_prior_without_positive_mass_fails_loud(self, prior):
        result = self._filter(prior)
        assert np.isnan(float(result.marginal_log_likelihood))
        assert np.all(np.isnan(np.asarray(result.discrete_state_probs[0])))

    def test_tiny_prior_is_not_floored(self):
        """Posterior odds = likelihood ratio x prior odds, with prior odds 1e-12.

        Flooring the prior to the 1e-10 stability floor would inflate the
        posterior odds of state 1 a hundredfold.
        """
        tiny = 1e-12
        posterior_uniform = np.asarray(self._filter([0.5, 0.5]).discrete_state_probs[0])
        posterior_tiny = np.asarray(
            self._filter([1.0 - tiny, tiny]).discrete_state_probs[0]
        )
        likelihood_ratio = posterior_uniform[1] / posterior_uniform[0]
        # Guard: the states' first-trial likelihoods differ, so the odds are
        # informative about how the prior entered.
        assert abs(np.log(likelihood_ratio)) > 0.1
        np.testing.assert_allclose(
            posterior_tiny[1] / posterior_tiny[0],
            likelihood_ratio * tiny / (1.0 - tiny),
            rtol=1e-6,
        )

    @pytest.mark.parametrize(
        "bad_entry", [np.nan, np.inf, -0.2], ids=["nan", "inf", "negative"]
    )
    def test_invalid_entry_becomes_structural_zero(self, bad_entry):
        """``[0.3, bad, 0.1]`` behaves exactly like ``[0.75, 0, 0.25]``."""
        result = self._filter([0.3, bad_entry, 0.1])
        reference = self._filter([0.75, 0.0, 0.25])
        assert np.isfinite(float(reference.marginal_log_likelihood))  # guard

        probs = np.asarray(result.discrete_state_probs)
        np.testing.assert_array_equal(probs[:, 1], np.zeros(len(self.CHOICES)))
        np.testing.assert_allclose(
            probs, np.asarray(reference.discrete_state_probs), rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            float(result.marginal_log_likelihood),
            float(reference.marginal_log_likelihood),
            rtol=1e-12,
        )


class TestSwitchingChoiceModel:
    """Tests for the SwitchingChoiceModel class."""

    def test_fit_returns_log_likelihoods(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        lls = model.fit(choices, max_iter=5)
        assert len(lls) > 0
        assert all(np.isfinite(ll) for ll in lls)

    def test_is_fitted(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (50,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        assert not model.is_fitted
        model.fit(choices, max_iter=3)
        assert model.is_fitted

    @pytest.mark.parametrize(
        "attr",
        [
            "log_likelihood_",
            "smoothed_discrete_probs_",
            "per_state_predicted_variances_",
        ],
    )
    def test_fitted_attribute_unavailable_before_fit(self, attr):
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        with pytest.raises(NotFittedError, match=attr):
            getattr(model, attr)
        assert not hasattr(model, attr)

    def test_discrete_state_posterior_shape(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (80,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit(choices, max_iter=3)
        assert model.smoothed_discrete_probs_.shape == (80, 2)
        np.testing.assert_allclose(
            model.smoothed_discrete_probs_.sum(axis=1), 1.0, atol=1e-5
        )

    def test_em_does_not_update_betas(self):
        """EM does not update inverse_temperatures (no closed-form M-step).

        Per-state betas are learned via SGD only. EM updates Q and Z.
        """
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        # Start with equal betas
        model = SwitchingChoiceModel(
            n_options=3,
            n_discrete_states=2,
            init_inverse_temperatures=jnp.array([2.0, 2.0]),
        )
        model.fit(choices, max_iter=5)
        # Betas should be unchanged (EM doesn't update them)
        np.testing.assert_allclose(model.inverse_temperatures_, [2.0, 2.0])

    def test_sgd_learns_different_betas(self):
        """SGD should learn different betas for exploit/explore data."""
        from state_space_practice.switching_choice import (
            SwitchingChoiceModel,
            simulate_switching_choice_data,
        )

        sim = simulate_switching_choice_data(
            n_trials=200,
            n_options=3,
            inverse_temperatures=jnp.array([5.0, 0.5]),
            process_noises=jnp.array([0.001, 0.05]),
            seed=42,
        )
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(sim.choices, num_steps=100)
        # Per-state betas should differ after SGD
        assert (
            abs(float(model.inverse_temperatures_[0] - model.inverse_temperatures_[1]))
            > 0.1
        )

    def test_sgd_improves_ll(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        lls = model.fit_sgd(choices, num_steps=30)
        assert lls[-1] > lls[0]

    def test_sgd_model_is_fitted(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (60,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(choices, num_steps=15)
        assert model.is_fitted

    def test_fit_rejects_invalid_choice_index(self):
        choices = jnp.array([0, 3, 1])
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)

        with pytest.raises(ValueError, match="All choices"):
            model.fit(choices, max_iter=1)

    def test_fit_sgd_rejects_invalid_choice_index(self):
        choices = jnp.array([0, 3, 1])
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)

        with pytest.raises(ValueError, match="All choices"):
            model.fit_sgd(choices, num_steps=1)

    def test_fit_log_likelihood_matches_final_parameters(self):
        choices = jnp.array(
            [0, 1, 1, 2, 2, 2, 0, 1, 2, 0, 2, 1],
            dtype=jnp.int32,
        )
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)

        model.fit(choices, max_iter=1)
        fresh_ll = float(model._run_filter(choices, None, None).marginal_log_likelihood)

        assert np.isclose(model.log_likelihood_, fresh_ll)


class TestSimulateSwitchingChoiceData:
    """Tests for the simulation helper."""

    def test_output_shapes(self):
        from state_space_practice.switching_choice import simulate_switching_choice_data

        sim = simulate_switching_choice_data(n_trials=100, n_options=3, seed=42)
        assert sim.choices.shape == (100,)
        assert sim.true_values.shape == (100, 2)  # K-1
        assert sim.true_states.shape == (100,)
        assert sim.true_probs.shape == (100, 3)

    def test_choices_valid(self):
        from state_space_practice.switching_choice import simulate_switching_choice_data

        sim = simulate_switching_choice_data(n_trials=200, n_options=4, seed=0)
        assert jnp.all(sim.choices >= 0)
        assert jnp.all(sim.choices < 4)

    def test_states_valid(self):
        from state_space_practice.switching_choice import simulate_switching_choice_data

        sim = simulate_switching_choice_data(
            n_trials=200,
            n_options=3,
            n_discrete_states=3,
            seed=0,
            process_noises=jnp.array([0.001, 0.01, 0.05]),
            inverse_temperatures=jnp.array([5.0, 2.0, 0.5]),
        )
        assert jnp.all(sim.true_states >= 0)
        assert jnp.all(sim.true_states < 3)

    def test_seed_reproducibility(self):
        from state_space_practice.switching_choice import simulate_switching_choice_data

        s1 = simulate_switching_choice_data(n_trials=50, n_options=3, seed=42)
        s2 = simulate_switching_choice_data(n_trials=50, n_options=3, seed=42)
        np.testing.assert_array_equal(s1.choices, s2.choices)


class TestModelComparison:
    """Tests for switching vs non-switching model comparison."""

    def test_switching_beats_nonswitching_on_switching_data(self):
        from state_space_practice.covariate_choice import CovariateChoiceModel
        from state_space_practice.switching_choice import (
            SwitchingChoiceModel,
            simulate_switching_choice_data,
        )

        sim = simulate_switching_choice_data(
            n_trials=200,
            n_options=3,
            n_discrete_states=2,
            inverse_temperatures=jnp.array([5.0, 0.5]),
            process_noises=jnp.array([0.001, 0.05]),
            seed=42,
        )

        # Switching model
        model_sw = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model_sw.fit_sgd(sim.choices, num_steps=100)

        # Non-switching model
        model_ns = CovariateChoiceModel(n_options=3)
        model_ns.fit_sgd(sim.choices, num_steps=100)

        # Switching model should have better LL on switching data
        assert model_sw.log_likelihood_ > model_ns.log_likelihood_


class TestSwitchingChoiceUncertainty:
    """Tests for switching choice model uncertainty summaries."""

    def test_uncertainty_populated_after_sgd(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (80,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(choices, num_steps=15)
        assert model.predicted_option_variances_ is not None
        assert model.predicted_option_variances_.shape == (80, 3)
        assert model.surprise_ is not None
        assert model.surprise_.shape == (80,)
        assert model.predicted_choice_entropy_ is not None
        assert model.predicted_choice_entropy_.shape == (80,)

    def test_uncertainty_populated_after_em(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (60,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit(choices, max_iter=3)
        assert model.predicted_option_variances_ is not None
        assert model.surprise_ is not None

    def test_surprise_is_positive(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (50,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(choices, num_steps=10)
        assert jnp.all(model.surprise_ >= 0)


class TestBetweenStateVariance:
    """Law-of-total-variance between-state term used by the uncertainty
    summaries (``predicted_option_variances_`` / ``smoothed_...``)."""

    def test_no_cancellation_for_large_nearby_means(self):
        """Means ~1e5 that differ by 1e-4: E[m^2] - E[m]^2 cancels to noise
        of order eps * 1e10 ~ 1e-6, while the true variance is 2.5e-9."""
        from state_space_practice.switching_choice import _between_state_variance

        means = jnp.array([[[1e5, 1e5 + 1e-4]]])  # (T=1, K=1, S=2)
        probs = jnp.array([[0.5, 0.5]])
        exact = 0.25 * (1e-4) ** 2  # p (1 - p) (m_1 - m_0)^2
        var = float(_between_state_variance(means, probs)[0, 0])
        np.testing.assert_allclose(var, exact, rtol=1e-3)
        # guard: the uncentred formula really is catastrophically wrong here
        e_mean = jnp.einsum("tks,ts->tk", means, probs)
        e_mean_sq = jnp.einsum("tks,ts->tk", means**2, probs)
        uncentred = float((e_mean_sq - e_mean**2)[0, 0])
        # (it returns 0 or noise of either sign: >= 50% relative error)
        assert abs(uncentred - exact) > 0.5 * exact

    def test_matches_direct_definition_and_is_nonnegative(self):
        from state_space_practice.switching_choice import _between_state_variance

        rng = np.random.default_rng(0)
        means = rng.standard_normal((6, 3, 4)) * 5 + 50
        probs = rng.dirichlet(np.ones(4), size=6)
        var = np.asarray(
            _between_state_variance(jnp.asarray(means), jnp.asarray(probs))
        )
        mbar = np.einsum("tks,ts->tk", means, probs)
        direct = np.einsum("tks,ts->tk", (means - mbar[..., None]) ** 2, probs)
        np.testing.assert_allclose(var, direct, rtol=1e-12)
        assert np.all(var >= 0)


# Shape combinations for parametrized uncertainty tests. Crucially includes
# at least one non-square (K-1 != S) config — square configs like (3, 2) can
# hide axis-ordering bugs because dimensions are numerically indistinguishable.
UNCERTAINTY_SHAPE_CASES = [
    pytest.param((3, 2), id="K3_S2_square"),
    pytest.param((4, 2), id="K4_S2_more_options"),
    pytest.param((3, 5), id="K3_S5_more_states"),
]


@pytest.fixture(
    scope="class",
    params=UNCERTAINTY_SHAPE_CASES,
)
def fitted_switching_choice_model(request):
    """Fit a SwitchingChoiceModel at the given (n_options, n_discrete_states).

    Class-scoped so each shape is only fit once across the tests that use it.
    """
    from state_space_practice.switching_choice import SwitchingChoiceModel

    n_options, n_discrete_states = request.param
    n_trials = 40
    choices = jax.random.randint(jax.random.PRNGKey(0), (n_trials,), 0, n_options)
    model = SwitchingChoiceModel(
        n_options=n_options,
        n_discrete_states=n_discrete_states,
    )
    model.fit_sgd(choices, num_steps=5)
    return model, choices, n_trials, n_options, n_discrete_states


class TestSwitchingChoiceUncertaintyShapes:
    """Shape-parametrized tests for the uncertainty summaries.

    These use a fixture that sweeps ``(n_options, n_discrete_states)`` including
    a non-square case (K-1 != S) so axis-ordering bugs cannot hide behind
    numerically indistinguishable dimensions.
    """

    def test_uncertainty_populated(self, fitted_switching_choice_model):
        model, _, T, K, S = fitted_switching_choice_model
        assert model.predicted_option_variances_ is not None
        assert model.predicted_option_variances_.shape == (T, K)
        assert model.smoothed_option_variances_ is not None
        assert model.smoothed_option_variances_.shape == (T, K)
        assert model.per_state_predicted_variances_ is not None
        assert model.per_state_predicted_variances_.shape == (T, K, S)
        assert model.surprise_ is not None
        assert model.surprise_.shape == (T,)
        assert model.predicted_choice_entropy_ is not None
        assert model.predicted_choice_entropy_.shape == (T,)

    def test_reference_option_variance_is_zero(
        self,
        fitted_switching_choice_model,
    ):
        model, *_ = fitted_switching_choice_model
        np.testing.assert_allclose(
            model.predicted_option_variances_[:, 0],
            0.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            model.smoothed_option_variances_[:, 0],
            0.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            model.per_state_predicted_variances_[:, 0, :],
            0.0,
            atol=1e-8,
        )

    def test_variances_finite_and_nonnegative(
        self,
        fitted_switching_choice_model,
    ):
        model, *_ = fitted_switching_choice_model
        assert jnp.all(jnp.isfinite(model.predicted_option_variances_))
        assert jnp.all(jnp.isfinite(model.smoothed_option_variances_))
        assert jnp.all(model.predicted_option_variances_ >= -1e-8)
        assert jnp.all(model.smoothed_option_variances_ >= -1e-8)

    def test_predicted_uses_prior_not_posterior(
        self,
        fitted_switching_choice_model,
    ):
        """Predicted (prior) variance should be >= filtered (posterior)."""
        model, *_ = fitted_switching_choice_model

        pred_var = model.predicted_option_variances_[:, 1:].mean()  # exclude ref

        # Extract filtered covariance diagonals independently: (T, K-1, S).
        # Use explicit einsum to avoid the jnp.diagonal axis-ordering pitfall.
        filt_diag = jnp.einsum("tiis->tis", model._filter_result.filtered_covs)
        # Reconstruct predicted (prior) state probs the same way the model does.
        disc_probs = model._filter_result.discrete_state_probs  # (T, S)
        init_prob = jnp.ones(model.n_discrete_states) / model.n_discrete_states
        predicted_disc = jnp.concatenate(
            [
                init_prob[None, :],
                disc_probs[:-1] @ model.discrete_transition_matrix_,
            ],
            axis=0,
        )
        filt_var = jnp.einsum("tks,ts->tk", filt_diag, predicted_disc).mean()
        assert float(pred_var) >= float(filt_var) - 1e-6

    def test_variance_law_of_total_variance(
        self,
        fitted_switching_choice_model,
    ):
        """``predicted_option_variances_`` must equal E[Var(x|s)] + Var(E[x|s]).

        The expected value is computed independently from ``result.predicted_covs``
        via a Python loop over (trial, option, state), so any shape/axis bug in
        how ``per_state_predicted_variances_`` is populated is detectable — this
        test does NOT re-use that attribute on both sides of the assertion.
        """
        model, *_ = fitted_switching_choice_model
        result = model._filter_result

        T = int(result.predicted_covs.shape[0])
        K = int(model.n_options)
        S = int(model.n_discrete_states)

        covs_np = np.asarray(result.predicted_covs)  # (T, K-1, K-1, S)
        vals_np = np.asarray(result.predicted_values)  # (T, K-1, S)

        # Reconstruct predicted (prior) state probs the same way the model does.
        disc_probs = np.asarray(result.discrete_state_probs)  # (T, S)
        trans = np.asarray(model.discrete_transition_matrix_)  # (S, S)
        init_prob = np.ones(S) / S
        predicted_disc = np.concatenate(
            [
                init_prob[None, :],
                disc_probs[:-1] @ trans,
            ],
            axis=0,
        )  # (T, S)

        expected_total = np.zeros((T, K))
        for t in range(T):
            for k in range(K):
                if k == 0:
                    # Reference option: per-state mean and variance are 0,
                    # so the total variance is identically 0.
                    continue
                k_idx = k - 1  # index into (K-1)-sized arrays
                e_var = 0.0
                e_mean = 0.0
                e_mean_sq = 0.0
                for s in range(S):
                    p = float(predicted_disc[t, s])
                    v = float(covs_np[t, k_idx, k_idx, s])
                    m = float(vals_np[t, k_idx, s])
                    e_var += p * v
                    e_mean += p * m
                    e_mean_sq += p * m * m
                var_mean = e_mean_sq - e_mean * e_mean
                expected_total[t, k] = e_var + var_mean

        np.testing.assert_allclose(
            np.asarray(model.predicted_option_variances_),
            expected_total,
            atol=1e-6,
        )


@pytest.mark.slow
class TestSwitchingChoiceRecovery:
    """Generative recovery: simulate → fit → check state sequence recovery."""

    def test_predicted_choice_probs_track_generative(self):
        """Fit on switching data and check that predicted choice
        probabilities correlate with the true generative probabilities.

        This is a predictive-distribution recovery test. We don't require
        recovery of exact state labels (which is under-determined for
        choice-only data with shared value dynamics); we require that
        the fitted predictive distribution matches the generative
        distribution in expectation.
        """
        from state_space_practice.switching_choice import (
            SwitchingChoiceModel,
            simulate_switching_choice_data,
        )

        sim = simulate_switching_choice_data(
            n_trials=400,
            n_options=3,
            n_discrete_states=2,
            seed=21,
        )
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(sim.choices, num_steps=100)

        # Reconstruct model predictive probabilities from filter output:
        # marginalize state-conditional predictive values over discrete probs
        filt_values = np.asarray(model._filter_result.predicted_values)  # (T, K-1, S)
        disc_probs = np.asarray(model._filter_result.discrete_state_probs)  # (T, S)
        betas = np.asarray(model.inverse_temperatures_)  # (S,)

        # Per-state choice probabilities
        zero_ref = np.zeros((filt_values.shape[0], 1, filt_values.shape[2]))
        full_vals = np.concatenate([zero_ref, filt_values], axis=1)  # (T, K, S)
        per_state_logits = betas[None, None, :] * full_vals  # (T, K, S)
        per_state_logits -= per_state_logits.max(axis=1, keepdims=True)
        per_state_probs = np.exp(per_state_logits)
        per_state_probs /= per_state_probs.sum(axis=1, keepdims=True)
        pred_probs = np.einsum("tks,ts->tk", per_state_probs, disc_probs)

        true_probs = np.asarray(sim.true_probs)

        # Mean absolute error between predicted and true choice probs
        mae = np.mean(np.abs(pred_probs - true_probs))
        assert mae < 0.15, f"predictive MAE {mae:.3f} >= 0.15"

    def test_fit_sgd_improves_log_likelihood_from_simulation(self):
        """SGD should improve LL on data drawn from the model's family."""
        from state_space_practice.switching_choice import (
            SwitchingChoiceModel,
            simulate_switching_choice_data,
        )

        sim = simulate_switching_choice_data(
            n_trials=300,
            n_options=3,
            n_discrete_states=2,
            seed=5,
        )
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(sim.choices, num_steps=50)
        history = model.log_likelihood_history_
        # LL should improve from start to end
        assert history[-1] > history[0] + 0.5, (
            f"LL did not improve: {history[0]:.3f} -> {history[-1]:.3f}"
        )
        assert np.isfinite(history[-1])

    def test_sgd_learns_distinct_per_state_params(self):
        """SGD should learn per-state betas that differ from each other.

        Note: exact state segmentation from choice data alone is
        under-determined (the generative model has shared value dynamics),
        so we test parameter distinguishability rather than state recovery.
        The predictive-distribution test above validates the model's
        overall recovery quality.
        """
        sim = simulate_switching_choice_data(
            n_trials=300,
            n_options=3,
            n_discrete_states=2,
            seed=21,
        )
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(sim.choices, num_steps=100)
        # Per-state betas should differ (true gap is 4.5)
        gap = float(
            abs(model.inverse_temperatures_[0] - model.inverse_temperatures_[1])
        )
        assert gap > 0.1, (
            f"Per-state beta gap {gap:.3f} < 0.1 after SGD "
            f"(learned: {model.inverse_temperatures_})"
        )


class TestSwitchingChoiceValidation:
    """Construction-time guards on per-state hyperparameters."""

    def test_rejects_negative_inverse_temperature(self):
        with pytest.raises(
            ValueError, match="inverse_temperatures must be strictly positive"
        ):
            SwitchingChoiceModel(
                n_options=3,
                n_discrete_states=2,
                init_inverse_temperatures=[-1.0, 1.0],
            )

    def test_rejects_decay_outside_unit_interval(self):
        with pytest.raises(ValueError, match="decays must lie in"):
            SwitchingChoiceModel(
                n_options=3,
                n_discrete_states=2,
                init_decays=[1.5, 0.9],
            )

    def test_converged_flag_false_when_not_converged(self):
        # Deterministic: one iteration cannot satisfy the convergence check, so
        # converged_ must be False. (The True direction for this same
        # `while abs(ll-prev_ll)<tol and iteration>0` loop is covered by the
        # contingency-belief test, whose data converges; switching_choice EM
        # does not reliably converge on random choices in a bounded budget.)
        rng = np.random.default_rng(0)
        choices = rng.integers(0, 3, size=200)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit(choices, max_iter=1)
        assert model.converged_ is False


class TestSwitchingChoiceMStepExactness:
    """The EM M-step maximises its (GPB1-approximate) expected objective.

    With ``joint[t, i, s] = P(s_t=i, s_{t+1}=s | y)`` and the smoother's
    state-conditional moments, the process-noise objective of state ``s`` is

        Q_s(q) = -sum_t w_t(s) [ k/2 log q + e_t(s) / (2 q) ],
        w_t(s) = sum_i joint[t, i, s],
        e_t(s) = ||m_{t+1,s} - a_s m_{t,s} - b_{t+1}||^2 + tr P_{t+1,s}
                 + a_s^2 tr P_{t,s} - 2 a_s tr C_t(s),
        C_t(s) = sum_i joint[t, i, s] C_t(i, s) / w_t(s),

    and the transition objective is ``sum_t sum_ij joint[t,i,j] log Z_ij``.
    Both are written out here with explicit loops; the returned parameters
    must be stationary (Lagrange condition for the simplex rows).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def em_inputs(cls):
        from state_space_practice.covariate_choice import simulate_rl_choice_data

        data = simulate_rl_choice_data(
            n_trials=60, n_options=3, seed=5, inverse_temperature=1.0
        )
        model = SwitchingChoiceModel(
            n_options=3,
            n_discrete_states=2,
            n_covariates=2,
            init_inverse_temperatures=[0.7, 2.5],
            init_process_noises=[0.02, 0.2],
            init_decays=[1.0, 0.85],
        )
        model.input_gain_ = 0.4 * jnp.eye(2)
        model._covariates = data.covariates
        filt = model._run_filter(data.choices, data.covariates)
        smooth = model._run_smoother(filt)
        return model, data, filt, smooth

    def _q_objective(self, model, data, smooth, s, q):
        joint = np.asarray(smooth[3])
        means = np.asarray(smooth[5])
        covs = np.asarray(smooth[6])
        cross = np.asarray(smooth[7])
        b = np.asarray(data.covariates) @ np.asarray(model.input_gain_).T
        a = float(model.decays_[s])
        k = means.shape[1]
        total = 0.0
        for t in range(joint.shape[0]):
            w = joint[t, :, s].sum()
            C = sum(joint[t, i, s] * cross[t, :, :, i, s] for i in range(2)) / w
            r = means[t + 1, :, s] - a * means[t, :, s] - b[t + 1]
            e = (
                r @ r
                + np.trace(covs[t + 1, :, :, s])
                + a**2 * np.trace(covs[t, :, :, s])
                - 2 * a * np.trace(C)
            )
            total -= w * (0.5 * k * np.log(q) + e / (2 * q))
        return total

    def test_process_noise_per_state_is_stationary(self, em_inputs):
        model, data, filt, smooth = em_inputs
        q_old = np.asarray(model.process_noises_).copy()
        fresh = SwitchingChoiceModel(
            n_options=3,
            n_discrete_states=2,
            n_covariates=2,
            init_inverse_temperatures=model.inverse_temperatures_,
            init_process_noises=q_old,
            init_decays=model.decays_,
        )
        fresh.input_gain_ = model.input_gain_
        fresh._covariates = data.covariates
        fresh._m_step(smooth)
        q_new = np.asarray(fresh.process_noises_)
        for s in range(2):
            eps = 1e-6 * q_new[s]
            g = (
                self._q_objective(model, data, smooth, s, q_new[s] + eps)
                - self._q_objective(model, data, smooth, s, q_new[s] - eps)
            ) / (2 * eps)
            f = self._q_objective(model, data, smooth, s, q_new[s])
            assert abs(g) < 1e-6 * abs(f) / q_new[s], (s, g)
            assert f >= self._q_objective(model, data, smooth, s, q_old[s])
        # Guard: the two states' estimates differ (the weights matter).
        assert abs(np.log(q_new[0] / q_new[1])) > 0.1, q_new

        # Transition rows: grad_ij of sum N_ij log Z_ij is N_ij / Z_ij, which
        # the Lagrange condition requires to equal the row's total count.
        joint = np.asarray(smooth[3])
        N = joint.sum(axis=0)
        Z = np.asarray(fresh.discrete_transition_matrix_)
        np.testing.assert_allclose(Z.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            N / Z, np.broadcast_to(N.sum(axis=1, keepdims=True), N.shape), rtol=1e-8
        )

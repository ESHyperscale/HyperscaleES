"""
Test: Fitness normalization for stable gradient estimation.

PAPER CLAIM: Raw fitness scores are converted to normalized scores with baseline
subtraction. This is standard in ES to reduce variance and ensure stable updates.

DESIGN DECISION: The convert_fitnesses function performs:
1. Mean subtraction (baseline): centers scores around zero
2. Variance normalization: scales to unit variance
3. Optional group-wise normalization for noise_reuse scenarios

This ensures that the gradient estimate is well-conditioned regardless of the
absolute scale of the fitness function.
"""
import pytest
import jax
import jax.numpy as jnp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.noiser.open_es import OpenES


class TestFitnessNormalization:
    """Verify fitness conversion produces properly normalized scores."""

    def test_normalized_scores_have_zero_mean(self, eggroll_noiser):
        """
        Normalized fitness scores should have approximately zero mean.
        
        This is baseline subtraction - critical for unbiased gradient estimation.
        """
        frozen_noiser_params, noiser_params = eggroll_noiser
        
        # Various raw score distributions
        test_cases = [
            jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),  # uniform
            jnp.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7]),  # high baseline
            jnp.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0]),  # mixed signs
            jnp.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]),  # large scale
        ]
        
        for raw_scores in test_cases:
            normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
            mean = jnp.mean(normalized)
            assert jnp.abs(mean) < 1e-5, f"Mean {mean} should be ~0 for scores {raw_scores}"

    def test_normalized_scores_have_unit_variance(self, eggroll_noiser):
        """
        Normalized fitness scores should have approximately unit variance.
        
        This ensures consistent gradient magnitudes regardless of fitness scale.
        """
        frozen_noiser_params, noiser_params = eggroll_noiser
        
        # Test with scores that have different variances
        test_cases = [
            jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),  # small variance
            jnp.array([10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0]),  # large variance
        ]
        
        for raw_scores in test_cases:
            normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
            var = jnp.var(normalized)
            # Note: normalization uses sqrt(var + 1e-5), so variance should be ~1
            assert jnp.abs(var - 1.0) < 0.1, f"Variance {var} should be ~1 for scores {raw_scores}"

    def test_normalization_preserves_ordering(self, eggroll_noiser):
        """
        Normalization should preserve the relative ordering of scores.
        
        Higher raw scores should still be higher after normalization.
        """
        frozen_noiser_params, noiser_params = eggroll_noiser
        
        raw_scores = jnp.array([1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0])
        normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        
        # Check that argmax and argmin are preserved
        assert jnp.argmax(raw_scores) == jnp.argmax(normalized), "Max position should be preserved"
        assert jnp.argmin(raw_scores) == jnp.argmin(normalized), "Min position should be preserved"
        
        # Check full ordering
        raw_order = jnp.argsort(raw_scores)
        normalized_order = jnp.argsort(normalized)
        assert jnp.allclose(raw_order, normalized_order), "Ordering should be preserved"

    def test_constant_scores_handled_gracefully(self, eggroll_noiser):
        """
        All-constant scores should not produce NaN or Inf.
        
        CODE: Division by sqrt(var + 1e-5) handles zero variance gracefully.
        """
        frozen_noiser_params, noiser_params = eggroll_noiser
        
        constant_scores = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, constant_scores)
        
        # Should not be NaN or Inf
        assert not jnp.any(jnp.isnan(normalized)), "Normalization should not produce NaN"
        assert not jnp.any(jnp.isinf(normalized)), "Normalization should not produce Inf"
        
        # All should be zero (constant input, mean subtraction = 0)
        assert jnp.allclose(normalized, 0.0), "Constant scores should normalize to zero"

    def test_group_normalization_when_enabled(self, small_param):
        """
        With group_size > 0, normalization is done within groups.
        
        This is useful for noise_reuse where multiple epochs share the same noise
        and should be compared within groups.
        """
        group_size = 4
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            sigma=0.1,
            lr=0.01,
            group_size=group_size,
        )
        
        # 8 scores, 2 groups of 4
        raw_scores = jnp.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        
        # Each group should have zero mean within the group
        group1 = normalized[:4]
        group2 = normalized[4:]
        
        # Group means should be approximately zero
        # Note: the implementation uses global variance, so this is approximate
        assert jnp.abs(jnp.mean(group1)) < 0.1, "Group 1 mean should be ~0"
        assert jnp.abs(jnp.mean(group2)) < 0.1, "Group 2 mean should be ~0"

    def test_open_es_uses_same_normalization(self, small_param, open_es_config):
        """
        OpenES should use the same normalization as EggRoll.
        
        This ensures consistent behavior across noiser implementations.
        """
        frozen_eggroll, noiser_eggroll = EggRoll.init_noiser(
            {"w": small_param}, sigma=0.1, lr=0.01
        )
        frozen_openes, noiser_openes = OpenES.init_noiser(
            {"w": small_param}, sigma=0.1, lr=0.01
        )
        
        raw_scores = jnp.array([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0])
        
        norm_eggroll = EggRoll.convert_fitnesses(frozen_eggroll, noiser_eggroll, raw_scores)
        norm_openes = OpenES.convert_fitnesses(frozen_openes, noiser_openes, raw_scores)
        
        assert jnp.allclose(norm_eggroll, norm_openes), \
            "EggRoll and OpenES should produce identical normalized scores"

    def test_normalization_is_differentiable(self, eggroll_noiser):
        """
        Normalization should be differentiable for potential future use cases.
        
        Even though ES doesn't backprop through fitness, this is good hygiene.
        """
        frozen_noiser_params, noiser_params = eggroll_noiser
        
        def normalize_fn(scores):
            return jnp.sum(EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, scores))
        
        raw_scores = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # This should not raise an error
        grad = jax.grad(normalize_fn)(raw_scores)
        
        assert grad.shape == raw_scores.shape, "Gradient should have same shape as input"
        assert not jnp.any(jnp.isnan(grad)), "Gradient should not contain NaN"

    def test_large_population_normalization(self, eggroll_noiser):
        """
        Normalization should work correctly for large populations.
        """
        frozen_noiser_params, noiser_params = eggroll_noiser
        
        key = jax.random.key(999)
        large_scores = jax.random.normal(key, (1024,)) * 10 + 50  # mean ~50, std ~10
        
        normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, large_scores)
        
        assert jnp.abs(jnp.mean(normalized)) < 0.01, "Large population should still have ~0 mean"
        assert jnp.abs(jnp.var(normalized) - 1.0) < 0.1, "Large population should have ~1 variance"

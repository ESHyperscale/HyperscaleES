"""
Test: High-rank accumulation from low-rank perturbations.

PAPER CLAIM: "Although individual perturbations are low-rank, the expression on the
right side is actually high-rank, due to the properties of sums of low-rank matrices."

The gradient estimate is: E[F(θ+σε)ε] where ε = AB^T is rank-r
But the sum Σ_i w_i A_i B_i^T can have rank up to min(N*r, min(m,n)).

DESIGN DECISION: This is the key insight that makes EGGROLL as expressive as
full-rank ES while being much more efficient. The final parameter update can
explore a high-dimensional subspace even though each individual perturbation
is constrained to a low-rank subspace.
"""
import pytest
import jax
import jax.numpy as jnp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperscalees.noiser.eggroll import get_lora_update_params

from conftest import compute_matrix_rank


class TestHighRankAccumulation:
    """Verify that summing low-rank perturbations produces high-rank updates."""

    def test_sum_of_rank1_exceeds_rank1(self, medium_param, es_key):
        """
        Sum of N independent rank-1 matrices should have rank > 1 (generically).
        
        This is the fundamental property: low-rank individuals → high-rank sum.
        """
        frozen_noiser_params = {"rank": 1, "noise_reuse": 0}
        
        # Sum perturbations from multiple independent threads
        total_perturbation = jnp.zeros_like(medium_param)
        num_perturbations = 16
        
        for thread_id in range(0, num_perturbations * 2, 2):  # Even threads only
            A, B = get_lora_update_params(
                frozen_noiser_params,
                0.1,  # base_sigma
                (0, thread_id),  # iterinfo
                medium_param,
                es_key
            )
            total_perturbation += A @ B.T
        
        # Individual perturbations are rank-1
        # But the sum should be higher rank
        accumulated_rank = compute_matrix_rank(total_perturbation)
        
        assert accumulated_rank > 1, \
            f"Sum of {num_perturbations} rank-1 matrices should have rank > 1, got {accumulated_rank}"
        
        print(f"\nSum of {num_perturbations} rank-1 perturbations has rank {accumulated_rank}")

    def test_accumulated_rank_grows_with_population(self, medium_param, es_key):
        """
        Accumulated rank should grow (roughly) with population size.
        
        With N independent rank-r perturbations, we expect rank up to min(N*r, min(m,n)).
        """
        frozen_noiser_params = {"rank": 1, "noise_reuse": 0}
        m, n = medium_param.shape  # 64, 32
        
        ranks_by_population = []
        
        for num_perturbations in [4, 8, 16, 32]:
            total_perturbation = jnp.zeros_like(medium_param)
            
            for thread_id in range(0, num_perturbations * 2, 2):
                A, B = get_lora_update_params(
                    frozen_noiser_params,
                    0.1,  # base_sigma
                    (0, thread_id),  # iterinfo
                    medium_param,
                    es_key
                )
                total_perturbation += A @ B.T
            
            accumulated_rank = compute_matrix_rank(total_perturbation)
            ranks_by_population.append((num_perturbations, int(accumulated_rank)))
        
        # Rank should generally increase with population
        # (not strictly monotonic due to random alignment, but trend should be clear)
        print(f"\nRank growth: {ranks_by_population}")
        
        # At minimum, larger populations should not have lower rank
        # And final rank should approach min(m,n) = 32
        final_rank = ranks_by_population[-1][1]
        assert final_rank >= 16, f"With 32 rank-1 perturbations, should have rank >= 16, got {final_rank}"

    def test_higher_base_rank_accumulates_faster(self, medium_param, es_key):
        """
        Higher rank perturbations should accumulate to full rank faster.
        """
        num_perturbations = 8
        m, n = medium_param.shape
        
        ranks_by_base_rank = []
        
        for base_rank in [1, 2, 4, 8]:
            frozen_noiser_params = {"rank": base_rank, "noise_reuse": 0}
            total_perturbation = jnp.zeros_like(medium_param)
            
            for thread_id in range(0, num_perturbations * 2, 2):
                A, B = get_lora_update_params(
                    frozen_noiser_params,
                    0.1,  # base_sigma
                    (0, thread_id),  # iterinfo
                    medium_param,
                    es_key
                )
                total_perturbation += A @ B.T
            
            accumulated_rank = compute_matrix_rank(total_perturbation)
            ranks_by_base_rank.append((base_rank, int(accumulated_rank)))
        
        print(f"\nRank accumulation with 8 perturbations: {ranks_by_base_rank}")
        
        # Higher base rank should lead to higher accumulated rank (or saturate at min(m,n))
        # r=1 → should be ≤8, r=8 → could reach min(64,32)=32
        assert ranks_by_base_rank[-1][1] >= ranks_by_base_rank[0][1], \
            "Higher base rank should not decrease accumulated rank"

    def test_full_rank_achievable_with_sufficient_population(self, small_param, es_key):
        """
        With enough population members, we should achieve full rank.
        
        For an 8x4 matrix, min(m,n)=4, so we need at least 4 rank-1 perturbations
        (generically) to span the full space.
        """
        frozen_noiser_params = {"rank": 1, "noise_reuse": 0}
        m, n = small_param.shape  # 8, 4
        max_rank = min(m, n)  # 4
        
        # Use more than enough perturbations to ensure full rank
        num_perturbations = max_rank * 4  # 16 rank-1 = plenty for rank 4
        
        total_perturbation = jnp.zeros_like(small_param)
        for thread_id in range(0, num_perturbations * 2, 2):
            A, B = get_lora_update_params(
                frozen_noiser_params,
                0.1,  # base_sigma
                (0, thread_id),  # iterinfo
                small_param,
                es_key
            )
            total_perturbation += A @ B.T
        
        accumulated_rank = compute_matrix_rank(total_perturbation)
        
        assert accumulated_rank == max_rank, \
            f"Should achieve full rank {max_rank} with {num_perturbations} perturbations, got {accumulated_rank}"

    def test_weighted_sum_respects_fitness_weights(self, medium_param, es_key):
        """
        The actual update is a weighted sum: Σ w_i A_i B_i^T.
        
        Higher-weighted perturbations should contribute more to the final update.
        """
        frozen_noiser_params = {"rank": 1, "noise_reuse": 0}
        
        # Get two perturbations
        A0, B0 = get_lora_update_params(
            frozen_noiser_params, 0.1, (0, 0), medium_param, es_key
        )
        A2, B2 = get_lora_update_params(
            frozen_noiser_params, 0.1, (0, 2), medium_param, es_key
        )
        
        pert0 = A0 @ B0.T
        pert2 = A2 @ B2.T
        
        # Weight perturbation 0 heavily
        weights = jnp.array([10.0, -10.0, 1.0, -1.0])  # Antithetic pairs
        weighted_sum = weights[0] * pert0 + weights[2] * pert2
        
        # The weighted sum should be more aligned with pert0
        alignment_0 = jnp.sum(weighted_sum * pert0) / (jnp.linalg.norm(weighted_sum) * jnp.linalg.norm(pert0) + 1e-8)
        alignment_2 = jnp.sum(weighted_sum * pert2) / (jnp.linalg.norm(weighted_sum) * jnp.linalg.norm(pert2) + 1e-8)
        
        assert jnp.abs(alignment_0) > jnp.abs(alignment_2), \
            f"Higher-weighted perturbation should dominate: {alignment_0:.3f} vs {alignment_2:.3f}"

    def test_antithetic_cancellation_preserves_rank(self, medium_param, es_key):
        """
        Antithetic pairs with equal weights cancel, but unequal weights contribute.
        
        If f(θ+ε) > f(θ-ε), the net contribution is proportional to the difference.
        """
        frozen_noiser_params = {"rank": 1, "noise_reuse": 0}
        
        # Get an antithetic pair
        A_plus, B_plus = get_lora_update_params(
            frozen_noiser_params, 0.1, (0, 0), medium_param, es_key
        )
        A_minus, B_minus = get_lora_update_params(
            frozen_noiser_params, 0.1, (0, 1), medium_param, es_key
        )
        
        pert_plus = A_plus @ B_plus.T
        pert_minus = A_minus @ B_minus.T
        
        # Equal weights → cancellation
        equal_sum = pert_plus + pert_minus
        assert jnp.allclose(equal_sum, 0.0, atol=1e-5), \
            "Equal weights on antithetic pair should cancel"
        
        # Unequal weights → net contribution
        unequal_sum = 2.0 * pert_plus + 1.0 * pert_minus  # Net positive for +ε direction
        assert not jnp.allclose(unequal_sum, 0.0), \
            "Unequal weights should not cancel"
        
        # The net should be aligned with the +ε direction
        net_rank = compute_matrix_rank(unequal_sum)
        assert net_rank == 1, f"Net contribution should be rank 1, got {net_rank}"

    def test_paper_claim_high_rank_update(self, large_param, es_key):
        """
        Integration test: verify the paper's claim that EGGROLL produces
        high-rank updates from low-rank perturbations.
        
        PAPER: "this still results in a high-rank update but with significant
        memory and computation savings"
        """
        m, n = large_param.shape  # 256, 128
        rank = 4
        population_size = 64  # N/2 = 32 unique perturbations (antithetic pairs)
        
        frozen_noiser_params = {"rank": rank, "noise_reuse": 0}
        
        # Simulate the weighted sum that would occur in an actual update
        # with random fitness-like weights
        key = jax.random.key(777)
        weights = jax.random.normal(key, (population_size,))
        
        total_perturbation = jnp.zeros_like(large_param)
        for i in range(population_size):
            thread_id = i
            A, B = get_lora_update_params(
                frozen_noiser_params,
                0.1,  # base_sigma
                (0, thread_id),  # iterinfo
                large_param,
                es_key
            )
            total_perturbation += weights[i] * (A @ B.T)
        
        accumulated_rank = compute_matrix_rank(total_perturbation)
        
        # With 32 unique rank-4 perturbations, theoretical max is min(32*4, 128) = 128
        # In practice, we should get close to full rank
        print(f"\nLarge-scale test: {population_size} members, rank-{rank} perturbations")
        print(f"Parameter shape: {m}x{n}, max possible rank: {min(m,n)}")
        print(f"Achieved rank: {accumulated_rank}")
        
        # Should achieve at least half of the theoretical max
        assert accumulated_rank >= min(m, n) // 2, \
            f"Should achieve substantial rank, got {accumulated_rank}"

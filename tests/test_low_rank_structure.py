"""
Test: Low-rank perturbation structure.

PAPER CLAIM: EGGROLL generates perturbations as AB^T where A ∈ R^{m×r}, B ∈ R^{n×r}
with r << min(m,n). This reduces auxiliary storage from mn to r(m+n).

DESIGN DECISION: Individual perturbations are strictly low-rank, but the sum
across a population is high-rank (tested separately in test_high_rank_accumulation.py).
"""
import pytest
import jax
import jax.numpy as jnp

from conftest import compute_matrix_rank

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperscalees.noiser.eggroll import get_lora_update_params


class TestLowRankPerturbationStructure:
    """Verify that EGGROLL perturbations have the claimed low-rank structure."""

    def test_perturbation_returns_two_factors(self, small_param, es_key, eggroll_config):
        """
        get_lora_update_params returns (A, B) factors, not a full matrix.
        
        This is the key memory optimization: we store r(m+n) instead of mn.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        iterinfo = (0, 0)  # epoch=0, thread_id=0
        
        A, B = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            small_param,
            es_key
        )
        
        m, n = small_param.shape  # 8, 4
        r = eggroll_config["rank"]  # 4
        
        # A should be m x r, B should be n x r
        assert A.shape == (m, r), f"A shape {A.shape} != expected ({m}, {r})"
        assert B.shape == (n, r), f"B shape {B.shape} != expected ({n}, {r})"

    def test_reconstructed_perturbation_has_low_rank(self, small_param, es_key, eggroll_config):
        """
        The reconstructed perturbation A @ B.T has rank at most r.
        
        PAPER: "low-rank matrix perturbation AB^T"
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        iterinfo = (0, 0)
        
        A, B = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            small_param,
            es_key
        )
        
        # Reconstruct the full perturbation
        perturbation = A @ B.T
        
        # Verify shape matches parameter
        assert perturbation.shape == small_param.shape
        
        # Verify rank is at most r
        rank = compute_matrix_rank(perturbation)
        assert rank <= eggroll_config["rank"], \
            f"Perturbation rank {rank} > specified rank {eggroll_config['rank']}"

    @pytest.mark.parametrize("rank", [1, 2, 4, 8])
    def test_rank_parameter_controls_perturbation_rank(self, medium_param, es_key, rank):
        """
        The rank parameter directly controls the maximum rank of perturbations.
        
        This is crucial for the memory/compute tradeoff discussed in the paper.
        """
        frozen_noiser_params = {
            "rank": rank,
            "noise_reuse": 0,
        }
        iterinfo = (0, 0)
        
        A, B = get_lora_update_params(
            frozen_noiser_params,
            0.1,  # sigma
            iterinfo,
            medium_param,
            es_key
        )
        
        perturbation = A @ B.T
        computed_rank = compute_matrix_rank(perturbation)
        
        # Rank should be exactly r (generically, random matrices have full rank)
        assert computed_rank == rank, \
            f"Expected rank {rank}, got {computed_rank}"

    def test_rank_one_is_outer_product(self, small_param, es_key):
        """
        At rank=1, the perturbation is a simple outer product a * b^T.
        
        This is the minimal low-rank case, highly efficient.
        """
        frozen_noiser_params = {
            "rank": 1,
            "noise_reuse": 0,
        }
        iterinfo = (0, 0)
        
        A, B = get_lora_update_params(
            frozen_noiser_params,
            0.1,
            iterinfo,
            small_param,
            es_key
        )
        
        # At rank 1, A and B should be column vectors
        assert A.shape[1] == 1
        assert B.shape[1] == 1
        
        # Perturbation should be exactly rank 1
        perturbation = A @ B.T
        rank = compute_matrix_rank(perturbation)
        assert rank == 1

    def test_storage_savings(self, large_param, eggroll_config):
        """
        Verify storage savings: r(m+n) << mn for reasonable r.
        
        PAPER: "reducing the auxiliary storage from mn to r(m+n) per layer"
        """
        m, n = large_param.shape  # 256, 128
        r = eggroll_config["rank"]  # 4
        
        full_rank_storage = m * n  # 32768
        low_rank_storage = r * (m + n)  # 4 * 384 = 1536
        
        savings_ratio = full_rank_storage / low_rank_storage
        
        # With r=4 on 256x128, we should save ~21x storage
        assert savings_ratio > 10, \
            f"Storage savings ratio {savings_ratio:.1f}x is less than expected"
        
        # Document the actual savings
        print(f"\nStorage: full-rank={full_rank_storage}, low-rank={low_rank_storage}, "
              f"savings={savings_ratio:.1f}x")

    def test_different_thread_ids_produce_different_perturbations(self, small_param, es_key, eggroll_config):
        """
        Different thread_ids should produce different (independent) perturbations.
        
        This is essential for population-based optimization.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        
        perturbations = []
        for thread_id in [0, 2, 4, 6]:  # Even thread_ids (positive sigma)
            iterinfo = (0, thread_id)
            A, B = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                iterinfo,
                small_param,
                es_key
            )
            perturbations.append(A @ B.T)
        
        # All perturbations should be different
        for i in range(len(perturbations)):
            for j in range(i + 1, len(perturbations)):
                assert not jnp.allclose(perturbations[i], perturbations[j]), \
                    f"Perturbations for thread {i*2} and {j*2} are identical"

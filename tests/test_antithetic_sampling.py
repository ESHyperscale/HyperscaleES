"""
Test: Antithetic (mirrored) sampling for variance reduction.

PAPER CLAIM: EGGROLL uses antithetic sampling where thread_id % 2 controls the sign
of the perturbation. Thread pairs (0,1), (2,3), etc. use the same base noise but
with opposite signs (±σ).

DESIGN DECISION: This variance reduction technique is standard in ES and is
critical for stable gradient estimation. The mirrored samples cancel out noise
in the gradient estimate, reducing variance without additional fitness evaluations.
"""
import pytest
import jax
import jax.numpy as jnp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperscalees.noiser.eggroll import get_lora_update_params, get_nonlora_update_params


class TestAntitheticSampling:
    """Verify that EGGROLL implements antithetic (mirrored) sampling correctly."""

    def test_even_odd_thread_pairs_have_opposite_sign(self, small_param, es_key, eggroll_config):
        """
        Thread pairs (2k, 2k+1) should produce perturbations with opposite signs.
        
        CODE: sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        
        # Test multiple pairs
        for base_thread in [0, 2, 4, 6]:
            even_iterinfo = (0, base_thread)
            odd_iterinfo = (0, base_thread + 1)
            
            A_even, B_even = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                even_iterinfo,
                small_param,
                es_key
            )
            
            A_odd, B_odd = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                odd_iterinfo,
                small_param,
                es_key
            )
            
            # A contains the sigma scaling, so A_even = -A_odd
            # B is the same for both (no sigma)
            assert jnp.allclose(A_even, -A_odd), \
                f"A vectors for threads {base_thread} and {base_thread+1} are not negatives"
            assert jnp.allclose(B_even, B_odd), \
                f"B vectors for threads {base_thread} and {base_thread+1} should be identical"

    def test_perturbation_matrices_are_negatives(self, small_param, es_key, eggroll_config):
        """
        The full perturbation matrices A @ B.T should be exact negatives for paired threads.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        
        for base_thread in [0, 2, 4]:
            even_iterinfo = (0, base_thread)
            odd_iterinfo = (0, base_thread + 1)
            
            A_even, B_even = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                even_iterinfo,
                small_param,
                es_key
            )
            
            A_odd, B_odd = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                odd_iterinfo,
                small_param,
                es_key
            )
            
            pert_even = A_even @ B_even.T
            pert_odd = A_odd @ B_odd.T
            
            assert jnp.allclose(pert_even, -pert_odd), \
                f"Perturbations for threads {base_thread} and {base_thread+1} are not negatives"

    def test_nonlora_params_also_use_antithetic_sampling(self, small_param, es_key, eggroll_config):
        """
        Non-LoRA parameters (biases, etc.) also use antithetic sampling.
        
        This ensures consistent variance reduction across all parameter types.
        """
        # Use a 1D param to simulate bias
        bias_param = jnp.ones((16,), dtype=jnp.float32)
        
        frozen_noiser_params = {
            "noise_reuse": 0,
        }
        
        for base_thread in [0, 2, 4]:
            even_iterinfo = (0, base_thread)
            odd_iterinfo = (0, base_thread + 1)
            
            noise_even = get_nonlora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                even_iterinfo,
                bias_param,
                es_key
            )
            
            noise_odd = get_nonlora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                odd_iterinfo,
                bias_param,
                es_key
            )
            
            assert jnp.allclose(noise_even, -noise_odd), \
                f"Non-LoRA noise for threads {base_thread} and {base_thread+1} are not negatives"

    def test_antithetic_pairs_share_base_noise(self, small_param, es_key, eggroll_config):
        """
        Antithetic pairs use the same random draw, just with opposite sign.
        
        CODE: true_thread_idx = thread_id // 2
        This means threads 0,1 share the same base key, threads 2,3 share another, etc.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        
        # Get perturbations for threads 0 and 1
        A_0, B_0 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            (0, 0),
            small_param,
            es_key
        )
        
        A_1, B_1 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            (0, 1),
            small_param,
            es_key
        )
        
        # Get perturbations for threads 2 and 3
        A_2, B_2 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            (0, 2),
            small_param,
            es_key
        )
        
        A_3, B_3 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            (0, 3),
            small_param,
            es_key
        )
        
        # Threads 0,1 should share base noise (B identical)
        assert jnp.allclose(B_0, B_1), "Threads 0 and 1 should share base B matrix"
        
        # Threads 2,3 should share base noise
        assert jnp.allclose(B_2, B_3), "Threads 2 and 3 should share base B matrix"
        
        # But threads 0,2 should have different base noise
        assert not jnp.allclose(B_0, B_2), "Threads 0 and 2 should have different B matrices"

    def test_antithetic_cancellation_in_mean(self, small_param, es_key, eggroll_config):
        """
        When fitnesses are equal, antithetic pairs should cancel in the mean update.
        
        This is the variance reduction property: if f(θ+ε) ≈ f(θ-ε), the noise
        contribution cancels and we get a cleaner gradient signal.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        
        num_pairs = 4
        perturbations = []
        
        for pair_idx in range(num_pairs):
            even_iterinfo = (0, pair_idx * 2)
            odd_iterinfo = (0, pair_idx * 2 + 1)
            
            A_even, B_even = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                even_iterinfo,
                small_param,
                es_key
            )
            
            A_odd, B_odd = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                odd_iterinfo,
                small_param,
                es_key
            )
            
            # Add both perturbations (simulating equal fitness weights)
            perturbations.append(A_even @ B_even.T)
            perturbations.append(A_odd @ B_odd.T)
        
        # With equal weights, the sum should be zero (perfect cancellation)
        total = sum(perturbations)
        assert jnp.allclose(total, 0.0, atol=1e-5), \
            "Antithetic pairs with equal weights should cancel to zero"

    def test_population_size_must_be_even(self):
        """
        Document that population size should be even for proper antithetic pairing.
        
        DESIGN: The codebase assumes pairs (2k, 2k+1), so odd population sizes
        would leave an unpaired member.
        """
        # This is more of a documentation test - the code doesn't enforce this
        # but users should be aware
        for pop_size in [8, 16, 32, 64, 128]:
            assert pop_size % 2 == 0, "Population size should be even for antithetic sampling"

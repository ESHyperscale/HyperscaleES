"""
Test: Deterministic noise generation via key folding.

PAPER CLAIM: EGGROLL uses deterministic noise generation via JAX random key folding.
This enables "noise reuse" - the ability to reconstruct the same perturbations
from just (epoch, thread_id) without storing them.

DESIGN DECISION: By folding keys with (epoch, thread_id), we get reproducible
perturbations that can be regenerated during the update phase without storing
them during the forward pass. This is critical for memory efficiency.
"""
import pytest
import jax
import jax.numpy as jnp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperscalees.noiser.eggroll import get_lora_update_params, get_nonlora_update_params


class TestNoiseDeterminism:
    """Verify that noise generation is deterministic and reproducible."""

    def test_same_inputs_produce_same_noise(self, small_param, es_key, eggroll_config):
        """
        Calling get_lora_update_params with identical inputs produces identical outputs.
        
        This is the foundation of noise reuse - we can reconstruct perturbations.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        iterinfo = (0, 4)  # epoch=0, thread_id=4
        
        A1, B1 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            small_param,
            es_key
        )
        
        A2, B2 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            small_param,
            es_key
        )
        
        assert jnp.allclose(A1, A2), "Same inputs should produce identical A"
        assert jnp.allclose(B1, B2), "Same inputs should produce identical B"

    def test_different_epochs_produce_different_noise(self, small_param, es_key, eggroll_config):
        """
        Different epochs produce different perturbations (when noise_reuse=0).
        
        This allows the algorithm to explore different directions over time.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        thread_id = 4
        
        A_epoch0, B_epoch0 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            (0, thread_id),
            small_param,
            es_key
        )
        
        A_epoch1, B_epoch1 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            (1, thread_id),
            small_param,
            es_key
        )
        
        A_epoch2, B_epoch2 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            (2, thread_id),
            small_param,
            es_key
        )
        
        # All should be different
        assert not jnp.allclose(B_epoch0, B_epoch1), "Different epochs should have different noise"
        assert not jnp.allclose(B_epoch1, B_epoch2), "Different epochs should have different noise"
        assert not jnp.allclose(B_epoch0, B_epoch2), "Different epochs should have different noise"

    def test_noise_reuse_repeats_noise_across_epochs(self, small_param, es_key, eggroll_config):
        """
        With noise_reuse > 0, the same noise is used for multiple consecutive epochs.
        
        PAPER: "take multiple updates within a single sequence using Noise-Reuse ES"
        
        CODE: true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse
        """
        noise_reuse = 3
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": noise_reuse,
        }
        thread_id = 4
        
        # Epochs 0, 1, 2 should all use the same noise (true_epoch = 0)
        perturbations_group1 = []
        for epoch in [0, 1, 2]:
            A, B = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                (epoch, thread_id),
                small_param,
                es_key
            )
            perturbations_group1.append((A, B))
        
        # Epochs 3, 4, 5 should all use different noise (true_epoch = 1)
        perturbations_group2 = []
        for epoch in [3, 4, 5]:
            A, B = get_lora_update_params(
                frozen_noiser_params,
                eggroll_config["sigma"],
                (epoch, thread_id),
                small_param,
                es_key
            )
            perturbations_group2.append((A, B))
        
        # Within group 1, all should be identical
        for i in range(len(perturbations_group1) - 1):
            A_i, B_i = perturbations_group1[i]
            A_j, B_j = perturbations_group1[i + 1]
            assert jnp.allclose(B_i, B_j), \
                f"Epochs {i} and {i+1} should have same noise with noise_reuse={noise_reuse}"
        
        # Within group 2, all should be identical
        for i in range(len(perturbations_group2) - 1):
            A_i, B_i = perturbations_group2[i]
            A_j, B_j = perturbations_group2[i + 1]
            assert jnp.allclose(B_i, B_j), \
                f"Epochs {3+i} and {3+i+1} should have same noise with noise_reuse={noise_reuse}"
        
        # But group 1 and group 2 should be different
        assert not jnp.allclose(perturbations_group1[0][1], perturbations_group2[0][1]), \
            "Different noise_reuse groups should have different noise"

    def test_different_keys_produce_different_noise(self, small_param, eggroll_config):
        """
        Different base keys produce different perturbations.
        
        This is how different parameters get independent noise.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        iterinfo = (0, 0)
        
        key1 = jax.random.key(100)
        key2 = jax.random.key(200)
        key3 = jax.random.key(300)
        
        _, B1 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            small_param,
            key1
        )
        
        _, B2 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            small_param,
            key2
        )
        
        _, B3 = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            small_param,
            key3
        )
        
        assert not jnp.allclose(B1, B2), "Different keys should produce different noise"
        assert not jnp.allclose(B2, B3), "Different keys should produce different noise"
        assert not jnp.allclose(B1, B3), "Different keys should produce different noise"

    def test_key_folding_is_commutative_in_reproduction(self, small_param, es_key, eggroll_config):
        """
        The key folding produces consistent results regardless of call order.
        
        This ensures we can reconstruct perturbations in any order during updates.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        
        # Generate noise for several (epoch, thread_id) combinations in order
        results_ordered = {}
        for epoch in [0, 1, 2]:
            for thread_id in [0, 2, 4]:
                A, B = get_lora_update_params(
                    frozen_noiser_params,
                    eggroll_config["sigma"],
                    (epoch, thread_id),
                    small_param,
                    es_key
                )
                results_ordered[(epoch, thread_id)] = (A, B)
        
        # Generate same combinations in different order
        results_shuffled = {}
        for thread_id in [4, 0, 2]:
            for epoch in [2, 0, 1]:
                A, B = get_lora_update_params(
                    frozen_noiser_params,
                    eggroll_config["sigma"],
                    (epoch, thread_id),
                    small_param,
                    es_key
                )
                results_shuffled[(epoch, thread_id)] = (A, B)
        
        # All results should match
        for key in results_ordered:
            A_ord, B_ord = results_ordered[key]
            A_shuf, B_shuf = results_shuffled[key]
            assert jnp.allclose(A_ord, A_shuf), f"Results for {key} should match regardless of call order"
            assert jnp.allclose(B_ord, B_shuf), f"Results for {key} should match regardless of call order"

    def test_nonlora_noise_is_also_deterministic(self, es_key, eggroll_config):
        """
        Non-LoRA parameter noise is also deterministically reproducible.
        """
        bias_param = jnp.ones((32,), dtype=jnp.float32)
        frozen_noiser_params = {"noise_reuse": 0}
        iterinfo = (5, 10)
        
        noise1 = get_nonlora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            bias_param,
            es_key
        )
        
        noise2 = get_nonlora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            bias_param,
            es_key
        )
        
        assert jnp.allclose(noise1, noise2), "Non-LoRA noise should be deterministic"

    def test_noise_depends_on_param_shape(self, es_key, eggroll_config):
        """
        Noise shape matches parameter shape, even with same key and iterinfo.
        
        This is important for supporting different layer sizes.
        """
        frozen_noiser_params = {
            "rank": eggroll_config["rank"],
            "noise_reuse": 0,
        }
        iterinfo = (0, 0)
        
        small = jnp.ones((8, 4), dtype=jnp.float32)
        medium = jnp.ones((16, 8), dtype=jnp.float32)
        large = jnp.ones((32, 16), dtype=jnp.float32)
        
        A_small, B_small = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            small,
            es_key
        )
        
        A_medium, B_medium = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            medium,
            es_key
        )
        
        A_large, B_large = get_lora_update_params(
            frozen_noiser_params,
            eggroll_config["sigma"],
            iterinfo,
            large,
            es_key
        )
        
        # Shapes should match params
        assert A_small.shape == (8, eggroll_config["rank"])
        assert B_small.shape == (4, eggroll_config["rank"])
        
        assert A_medium.shape == (16, eggroll_config["rank"])
        assert B_medium.shape == (8, eggroll_config["rank"])
        
        assert A_large.shape == (32, eggroll_config["rank"])
        assert B_large.shape == (16, eggroll_config["rank"])

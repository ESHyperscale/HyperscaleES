"""
Test: Forward pass equivalence - do_mm implements efficient low-rank computation.

PAPER CLAIM: EGGROLL computes x @ (W + AB^T) as x @ W + x @ B @ A^T,
avoiding explicit formation of the perturbed matrix. This reduces the cost
of a forward pass from O(mn) to O(r(m+n)).

DESIGN DECISION: The do_mm method is the core optimization. Instead of:
    perturbed_W = W + sigma * A @ B.T
    output = x @ perturbed_W.T
We compute:
    output = x @ W.T + x @ B @ A.T * sigma

This is mathematically equivalent but computationally cheaper when r << min(m,n).
"""
import pytest
import jax
import jax.numpy as jnp
import optax

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperscalees.noiser.eggroll import EggRoll, get_lora_update_params
from hyperscalees.noiser.open_es import OpenES


class TestForwardEquivalence:
    """Verify that do_mm computes the correct perturbed forward pass."""

    def test_do_mm_matches_explicit_perturbation(self, small_param, es_key, eggroll_config):
        """
        do_mm(x) should equal x @ (W + perturbation).T computed explicitly.
        
        This is the core correctness check for the EGGROLL optimization.
        """
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            eggroll_config["lr"],
            rank=eggroll_config["rank"],
        )
        
        iterinfo = (0, 4)  # epoch=0, thread_id=4
        batch_size = 3
        input_dim = small_param.shape[1]  # 4
        
        # Random input
        x = jax.random.normal(jax.random.key(123), (batch_size, input_dim))
        
        # Method 1: Use do_mm (the efficient implementation)
        output_efficient = EggRoll.do_mm(
            frozen_noiser_params, noiser_params, small_param, es_key, iterinfo, x
        )
        
        # Method 2: Compute explicitly by forming the perturbed matrix
        A, B = get_lora_update_params(
            frozen_noiser_params,
            noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]),
            iterinfo,
            small_param,
            es_key
        )
        perturbation = A @ B.T  # Full perturbation matrix
        perturbed_W = small_param + perturbation
        output_explicit = x @ perturbed_W.T
        
        assert jnp.allclose(output_efficient, output_explicit, atol=1e-5), \
            "do_mm should match explicit perturbation computation"

    def test_do_mm_without_iterinfo_is_standard_matmul(self, small_param, es_key, eggroll_config):
        """
        When iterinfo is None, do_mm should just compute x @ W.T (no perturbation).
        
        This is used for evaluation without noise.
        """
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            eggroll_config["lr"],
            rank=eggroll_config["rank"],
        )
        
        batch_size = 5
        input_dim = small_param.shape[1]
        x = jax.random.normal(jax.random.key(456), (batch_size, input_dim))
        
        # do_mm with iterinfo=None
        output = EggRoll.do_mm(
            frozen_noiser_params, noiser_params, small_param, es_key, None, x
        )
        
        # Should be standard matmul
        expected = x @ small_param.T
        
        assert jnp.allclose(output, expected), \
            "do_mm with iterinfo=None should be standard matmul"

    def test_do_Tmm_matches_transposed_computation(self, small_param, es_key, eggroll_config):
        """
        do_Tmm computes x @ (W + perturbation) (transposed version).
        
        This is used for gradients or different layer configurations.
        """
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            eggroll_config["lr"],
            rank=eggroll_config["rank"],
        )
        
        iterinfo = (0, 2)
        batch_size = 3
        output_dim = small_param.shape[0]  # 8
        
        # Input for transposed multiply
        x = jax.random.normal(jax.random.key(789), (batch_size, output_dim))
        
        # Method 1: Use do_Tmm
        output_efficient = EggRoll.do_Tmm(
            frozen_noiser_params, noiser_params, small_param, es_key, iterinfo, x
        )
        
        # Method 2: Explicit computation
        A, B = get_lora_update_params(
            frozen_noiser_params,
            noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]),
            iterinfo,
            small_param,
            es_key
        )
        perturbation = A @ B.T
        perturbed_W = small_param + perturbation
        output_explicit = x @ perturbed_W
        
        assert jnp.allclose(output_efficient, output_explicit, atol=1e-5), \
            "do_Tmm should match explicit transposed computation"

    def test_different_thread_ids_produce_different_outputs(self, small_param, es_key, eggroll_config):
        """
        Different thread_ids should produce different forward pass outputs.
        
        This is essential for population diversity.
        """
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            eggroll_config["lr"],
            rank=eggroll_config["rank"],
        )
        
        x = jax.random.normal(jax.random.key(111), (2, small_param.shape[1]))
        
        outputs = []
        for thread_id in [0, 2, 4, 6]:
            iterinfo = (0, thread_id)
            output = EggRoll.do_mm(
                frozen_noiser_params, noiser_params, small_param, es_key, iterinfo, x
            )
            outputs.append(output)
        
        # All outputs should be different
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not jnp.allclose(outputs[i], outputs[j]), \
                    f"Outputs for threads {i*2} and {j*2} should be different"

    def test_antithetic_pairs_bracket_base_output(self, small_param, es_key, eggroll_config):
        """
        Antithetic pairs (thread 2k, 2k+1) should produce outputs that average
        to approximately the unperturbed output.
        
        This is a consequence of the +σ/-σ perturbations.
        """
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            eggroll_config["lr"],
            rank=eggroll_config["rank"],
        )
        
        x = jax.random.normal(jax.random.key(222), (2, small_param.shape[1]))
        
        # Unperturbed output
        base_output = EggRoll.do_mm(
            frozen_noiser_params, noiser_params, small_param, es_key, None, x
        )
        
        # Antithetic pair outputs
        output_plus = EggRoll.do_mm(
            frozen_noiser_params, noiser_params, small_param, es_key, (0, 0), x
        )
        output_minus = EggRoll.do_mm(
            frozen_noiser_params, noiser_params, small_param, es_key, (0, 1), x
        )
        
        # Average should equal base (exactly, due to linearity)
        avg_output = (output_plus + output_minus) / 2
        
        assert jnp.allclose(avg_output, base_output), \
            "Average of antithetic pair should equal unperturbed output"

    def test_sigma_scales_perturbation_magnitude(self, small_param, es_key):
        """
        Larger sigma should produce larger deviations from base output.
        """
        x = jax.random.normal(jax.random.key(333), (2, small_param.shape[1]))
        iterinfo = (0, 0)
        
        deviations = []
        for sigma in [0.01, 0.1, 1.0]:
            frozen_noiser_params, noiser_params = EggRoll.init_noiser(
                {"w": small_param}, sigma=sigma, lr=0.01, rank=4
            )
            
            base = EggRoll.do_mm(
                frozen_noiser_params, noiser_params, small_param, es_key, None, x
            )
            perturbed = EggRoll.do_mm(
                frozen_noiser_params, noiser_params, small_param, es_key, iterinfo, x
            )
            
            deviation = jnp.sqrt(jnp.mean((perturbed - base) ** 2))
            deviations.append(float(deviation))
        
        # Deviations should increase with sigma
        assert deviations[0] < deviations[1] < deviations[2], \
            f"Deviations {deviations} should increase with sigma"

    def test_rank_affects_perturbation_structure(self, medium_param, es_key):
        """
        Higher rank should allow for more diverse perturbations.
        """
        x = jax.random.normal(jax.random.key(444), (4, medium_param.shape[1]))
        
        outputs_rank1 = []
        outputs_rank8 = []
        
        for rank, output_list in [(1, outputs_rank1), (8, outputs_rank8)]:
            frozen_noiser_params, noiser_params = EggRoll.init_noiser(
                {"w": medium_param}, sigma=0.1, lr=0.01, rank=rank
            )
            
            for thread_id in [0, 2, 4, 6]:
                output = EggRoll.do_mm(
                    frozen_noiser_params, noiser_params, medium_param, es_key,
                    (0, thread_id), x
                )
                output_list.append(output)
        
        # Compute variance across different thread outputs
        var_rank1 = jnp.var(jnp.stack(outputs_rank1))
        var_rank8 = jnp.var(jnp.stack(outputs_rank8))
        
        # Higher rank should generally allow more variation
        # (though this is a weak test - just checking they're both non-zero)
        assert var_rank1 > 0 and var_rank8 > 0, "Both ranks should produce variation"

    def test_vmapped_forward_over_population(self, small_param, es_key, eggroll_config):
        """
        do_mm should work correctly when vmapped over a population.
        
        This is how it's actually used in practice.
        """
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            eggroll_config["lr"],
            rank=eggroll_config["rank"],
        )
        
        num_envs = 8
        batch_size = 2
        input_dim = small_param.shape[1]
        
        # Batched inputs and iterinfos
        x_batch = jax.random.normal(jax.random.key(555), (num_envs, batch_size, input_dim))
        iterinfos = (jnp.zeros(num_envs, dtype=jnp.int32), jnp.arange(num_envs))
        
        # Vmap over population
        def single_forward(x, iterinfo):
            return EggRoll.do_mm(
                frozen_noiser_params, noiser_params, small_param, es_key, iterinfo, x
            )
        
        outputs = jax.vmap(single_forward)(
            x_batch,
            (iterinfos[0], iterinfos[1])
        )
        
        assert outputs.shape == (num_envs, batch_size, small_param.shape[0])
        
        # Verify against manual loop
        for i in range(num_envs):
            expected = single_forward(x_batch[i], (iterinfos[0][i], iterinfos[1][i]))
            assert jnp.allclose(outputs[i], expected), f"Vmapped output {i} should match manual"

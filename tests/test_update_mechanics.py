"""
Test: Update mechanics - gradient estimation and parameter updates.

PAPER CLAIM: EGGROLL estimates gradients via the ES formula:
    ∇_θ E[F(θ+σε)] ≈ (1/σ) E[F(θ+σε)ε]
    
With low-rank ε = AB^T, the update aggregates weighted low-rank perturbations
across the population.

DESIGN DECISION: The _do_update and do_updates methods compute gradient estimates
by weighting perturbations by their normalized fitnesses. Higher fitness perturbations
contribute more to the final update direction.
"""
import pytest
import jax
import jax.numpy as jnp
import optax

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.noiser.open_es import OpenES
from hyperscalees.models.common import MM_PARAM, PARAM

from conftest import make_iterinfo


class TestUpdateMechanics:
    """Verify that ES gradient estimation and updates work correctly."""

    def test_higher_fitness_perturbation_dominates_update(self, small_param, es_key, eggroll_config):
        """
        When one perturbation has much higher fitness, the update should
        move parameters toward that perturbation direction.
        
        We verify this by:
        1. Creating a population where one thread has much higher fitness
        2. Computing the perturbation for that high-fitness thread
        3. Checking that the parameter update is positively correlated with that perturbation
        """
        from hyperscalees.noiser.eggroll import get_lora_update_params
        
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            eggroll_config["lr"],
            solver=optax.sgd,
            rank=eggroll_config["rank"],
        )
        
        num_envs = 8
        iterinfos = make_iterinfo(num_envs)
        
        # Give thread 6 (even = +sigma direction) very high fitness, all others low
        # Thread 7 is its antithetic pair with -sigma
        fitnesses = jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 10.0, -1.0])
        normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, fitnesses)
        
        # Get the perturbation for thread 6 (the high-fitness one)
        A_high, B_high = get_lora_update_params(
            frozen_noiser_params,
            noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]),
            (0, 6),  # epoch=0, thread_id=6
            small_param,
            es_key
        )
        high_fitness_perturbation = A_high @ B_high.T
        
        es_tree_key = {"w": es_key}
        es_map = {"w": MM_PARAM}
        
        _, new_params = EggRoll.do_updates(
            frozen_noiser_params, noiser_params,
            {"w": small_param}, es_tree_key,
            normalized, iterinfos, es_map
        )
        
        # The update direction
        param_delta = new_params["w"] - small_param
        
        # Compute correlation: update should be aligned with high-fitness perturbation
        # (positive dot product means moving toward that direction)
        correlation = jnp.sum(param_delta * high_fitness_perturbation)
        
        # The update should be positively correlated with the high-fitness perturbation
        assert correlation > 0, \
            f"Update should move toward high-fitness perturbation, got correlation={float(correlation):.4f}"

    def test_equal_antithetic_fitnesses_cancel_to_no_update(self, small_param, es_key, eggroll_config):
        """
        When antithetic pairs have equal fitness, their contributions should cancel.
        
        This is a key property: f(θ+ε) = f(θ-ε) implies ε contributes nothing to gradient.
        """
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            lr=1.0,  # Large LR to make any non-zero update visible
            solver=optax.sgd,
            rank=eggroll_config["rank"],
        )
        
        num_envs = 8
        iterinfos = make_iterinfo(num_envs)
        
        # Equal fitness for all - after normalization, all are 0
        fitnesses = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, fitnesses)
        
        es_tree_key = {"w": es_key}
        es_map = {"w": MM_PARAM}
        
        _, new_params = EggRoll.do_updates(
            frozen_noiser_params, noiser_params,
            {"w": small_param}, es_tree_key,
            normalized, iterinfos, es_map
        )
        
        # With all equal (zero normalized) fitness, update should be zero
        assert jnp.allclose(new_params["w"], small_param, atol=1e-6), \
            "Equal fitnesses should produce no update"

    def test_update_magnitude_scales_with_lr(self, small_param, es_key, eggroll_config):
        """
        Larger learning rate should produce larger parameter changes.
        """
        num_envs = 8
        iterinfos = make_iterinfo(num_envs)
        fitnesses_raw = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        changes = []
        for lr in [0.001, 0.01, 0.1]:
            frozen_noiser_params, noiser_params = EggRoll.init_noiser(
                {"w": small_param},
                eggroll_config["sigma"],
                lr=lr,
                solver=optax.sgd,
                rank=eggroll_config["rank"],
            )
            
            normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, fitnesses_raw)
            es_tree_key = {"w": es_key}
            es_map = {"w": MM_PARAM}
            
            _, new_params = EggRoll.do_updates(
                frozen_noiser_params, noiser_params,
                {"w": small_param}, es_tree_key,
                normalized, iterinfos, es_map
            )
            
            change = jnp.sqrt(jnp.mean((new_params["w"] - small_param) ** 2))
            changes.append(float(change))
        
        # Changes should increase with LR
        assert changes[0] < changes[1] < changes[2], \
            f"Update magnitudes {changes} should increase with LR"

    def test_update_improves_simple_fitness(self, es_key):
        """
        On a simple optimization problem, updates should improve fitness.
        
        We use a direct test: optimize a single parameter to equal a target value.
        This is the simplest possible ES task - no neural network, just one scalar.
        """
        # Simple task: optimize a scalar to equal 2.0
        target = 2.0
        param = jnp.array([[0.5]])  # Start at 0.5, want to reach 2.0
        
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": param},
            sigma=0.1,
            lr=0.5,
            solver=optax.sgd,
            rank=1,
        )
        
        es_tree_key = {"w": es_key}
        es_map = {"w": MM_PARAM}
        
        def fitness(p):
            """Negative squared distance from target."""
            return -((p[0, 0] - target) ** 2)
        
        initial_fitness = float(fitness(param))
        num_envs = 32
        num_epochs = 20
        
        for epoch in range(num_epochs):
            iterinfos = make_iterinfo(num_envs, epoch)
            
            # Evaluate fitness for each perturbed parameter
            fitnesses = []
            for i in range(num_envs):
                # Get the perturbed parameter
                perturbed = EggRoll.get_noisy_standard(
                    frozen_noiser_params, noiser_params, param,
                    es_key, (iterinfos[0][i], iterinfos[1][i])
                )
                fitnesses.append(fitness(perturbed))
            
            raw_fitnesses = jnp.array(fitnesses)
            normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_fitnesses)
            
            noiser_params, new_params = EggRoll.do_updates(
                frozen_noiser_params, noiser_params,
                {"w": param}, es_tree_key,
                normalized, iterinfos, es_map
            )
            param = new_params["w"]
        
        final_fitness = float(fitness(param))
        
        # Fitness should improve (get closer to 0, which is the max)
        assert final_fitness > initial_fitness, \
            f"Fitness should improve: {initial_fitness:.4f} -> {final_fitness:.4f}"
        
        # And parameter should be closer to target
        initial_distance = abs(0.5 - target)
        final_distance = abs(float(param[0, 0]) - target)
        assert final_distance < initial_distance, \
            f"Should get closer to target: {initial_distance:.4f} -> {final_distance:.4f}"

    def test_optimizer_state_updates(self, small_param, es_key, eggroll_config):
        """
        The optimizer state should be updated after do_updates.
        
        This is important for optimizers with momentum (Adam, etc.).
        """
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            eggroll_config["sigma"],
            eggroll_config["lr"],
            solver=optax.adam,
            rank=eggroll_config["rank"],
        )
        
        # Store initial optimizer state for comparison
        # For Adam, the state contains (count, mu, nu) - we'll check that count increments
        initial_opt_state = noiser_params["opt_state"]
        
        num_envs = 8
        iterinfos = make_iterinfo(num_envs)
        fitnesses = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, fitnesses)
        
        es_tree_key = {"w": es_key}
        es_map = {"w": MM_PARAM}
        
        noiser_params_after, new_params = EggRoll.do_updates(
            frozen_noiser_params, noiser_params,
            {"w": small_param}, es_tree_key,
            normalized, iterinfos, es_map
        )
        
        after_opt_state = noiser_params_after["opt_state"]
        
        # The optimizer state should have been updated
        # For most optax optimizers, there's a step count or moment estimates that change
        # We can flatten and compare the pytrees
        initial_flat = jax.tree_util.tree_leaves(initial_opt_state)
        after_flat = jax.tree_util.tree_leaves(after_opt_state)
        
        # At least one leaf should be different after an update
        any_changed = any(
            not jnp.allclose(i, a) 
            for i, a in zip(initial_flat, after_flat)
            if hasattr(i, 'shape')  # Only compare arrays
        )
        
        assert any_changed, "Optimizer state should change after update"

    def test_frozen_nonlora_skips_bias_updates(self, small_param, es_key):
        """
        With freeze_nonlora=True, non-MM parameters (biases) should not update.
        """
        bias = jnp.ones((8,), dtype=jnp.float32) * 0.5
        
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param, "b": bias},
            sigma=0.1,
            lr=0.1,
            solver=optax.sgd,
            freeze_nonlora=True,
            rank=4,
        )
        
        num_envs = 8
        iterinfos = make_iterinfo(num_envs)
        fitnesses = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, fitnesses)
        
        es_tree_key = {"w": es_key, "b": jax.random.fold_in(es_key, 100)}
        es_map = {"w": MM_PARAM, "b": PARAM}  # 0 = PARAM for full-rank treatment
        
        _, new_params = EggRoll.do_updates(
            frozen_noiser_params, noiser_params,
            {"w": small_param, "b": bias}, es_tree_key,
            normalized, iterinfos, es_map
        )
        
        # Bias should not change with freeze_nonlora=True
        assert jnp.allclose(new_params["b"], bias), \
            "Bias should not change when freeze_nonlora=True"
        
        # But weights should still update
        assert not jnp.allclose(new_params["w"], small_param), \
            "Weights should still update"

    def test_population_scaling_in_update(self, small_param, es_key, eggroll_config):
        """
        The update formula includes sqrt(N) scaling for population size.
        
        CODE: return -(new_grad * jnp.sqrt(fitnesses.size)).astype(param.dtype)
        """
        # Test with different population sizes
        for num_envs in [8, 32, 128]:
            frozen_noiser_params, noiser_params = EggRoll.init_noiser(
                {"w": small_param},
                eggroll_config["sigma"],
                eggroll_config["lr"],
                solver=optax.sgd,
                rank=eggroll_config["rank"],
            )
            
            iterinfos = make_iterinfo(num_envs)
            # Create fitnesses that would produce similar normalized values
            fitnesses = jnp.linspace(0, 1, num_envs)
            normalized = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, fitnesses)
            
            es_tree_key = {"w": es_key}
            es_map = {"w": MM_PARAM}
            
            _, new_params = EggRoll.do_updates(
                frozen_noiser_params, noiser_params,
                {"w": small_param}, es_tree_key,
                normalized, iterinfos, es_map
            )
            
            # Just verify it runs without error for various population sizes
            assert new_params["w"].shape == small_param.shape

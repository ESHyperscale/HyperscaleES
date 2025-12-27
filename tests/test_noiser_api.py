"""
Test: Noiser API contract across implementations.

DESIGN DECISION: All noiser implementations (EggRoll, OpenES, BaseNoiser) should
follow the same interface contract. This ensures they are interchangeable and
that the Model class can work with any noiser.

The Noiser interface consists of:
- init_noiser: Initialize noiser state
- do_mm: Noised matrix multiplication (x @ W.T)
- do_Tmm: Noised transposed matrix multiplication (x @ W)
- do_emb: Noised embedding lookup
- get_noisy_standard: Get noised parameter directly
- convert_fitnesses: Normalize raw fitness scores
- do_updates: Apply ES update to parameters
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
from hyperscalees.noiser.base_noiser import Noiser as BaseNoiser

from conftest import make_iterinfo


# List of noiser classes to test
NOISER_CLASSES = [
    pytest.param(EggRoll, id="EggRoll"),
    pytest.param(OpenES, id="OpenES"),
]


class TestNoiserAPIContract:
    """Verify all noiser implementations follow the same interface."""

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_init_noiser_returns_correct_structure(self, noiser_cls, small_param):
        """
        init_noiser should return (frozen_noiser_params, noiser_params) tuple.
        """
        frozen_params, noiser_params = noiser_cls.init_noiser(
            {"w": small_param},
            sigma=0.1,
            lr=0.01,
            solver=optax.sgd,
        )
        
        assert isinstance(frozen_params, dict), "frozen_noiser_params should be a dict"
        assert isinstance(noiser_params, dict), "noiser_params should be a dict"
        
        # noiser_params should contain sigma and opt_state
        assert "sigma" in noiser_params, "noiser_params should contain 'sigma'"
        assert "opt_state" in noiser_params, "noiser_params should contain 'opt_state'"

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_do_mm_signature(self, noiser_cls, small_param, es_key):
        """
        do_mm should accept (frozen_noiser_params, noiser_params, param, key, iterinfo, x).
        """
        frozen_params, noiser_params = noiser_cls.init_noiser(
            {"w": small_param}, sigma=0.1, lr=0.01
        )
        
        batch_size = 2
        input_dim = small_param.shape[1]
        x = jax.random.normal(jax.random.key(123), (batch_size, input_dim))
        
        # With iterinfo
        output = noiser_cls.do_mm(
            frozen_params, noiser_params, small_param, es_key, (0, 0), x
        )
        assert output.shape == (batch_size, small_param.shape[0])
        
        # Without iterinfo (None)
        output_eval = noiser_cls.do_mm(
            frozen_params, noiser_params, small_param, es_key, None, x
        )
        assert output_eval.shape == (batch_size, small_param.shape[0])

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_do_Tmm_signature(self, noiser_cls, small_param, es_key):
        """
        do_Tmm should accept same args and compute transposed multiplication.
        """
        frozen_params, noiser_params = noiser_cls.init_noiser(
            {"w": small_param}, sigma=0.1, lr=0.01
        )
        
        batch_size = 2
        output_dim = small_param.shape[0]  # Transpose: input is output_dim
        x = jax.random.normal(jax.random.key(456), (batch_size, output_dim))
        
        output = noiser_cls.do_Tmm(
            frozen_params, noiser_params, small_param, es_key, (0, 0), x
        )
        assert output.shape == (batch_size, small_param.shape[1])

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_get_noisy_standard_signature(self, noiser_cls, small_param, es_key):
        """
        get_noisy_standard should return a noised version of the parameter.
        """
        frozen_params, noiser_params = noiser_cls.init_noiser(
            {"w": small_param}, sigma=0.1, lr=0.01
        )
        
        # With iterinfo
        noisy_param = noiser_cls.get_noisy_standard(
            frozen_params, noiser_params, small_param, es_key, (0, 0)
        )
        assert noisy_param.shape == small_param.shape
        
        # Without iterinfo - should return original
        original_param = noiser_cls.get_noisy_standard(
            frozen_params, noiser_params, small_param, es_key, None
        )
        assert jnp.allclose(original_param, small_param)

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_convert_fitnesses_signature(self, noiser_cls, small_param):
        """
        convert_fitnesses should normalize raw scores.
        """
        frozen_params, noiser_params = noiser_cls.init_noiser(
            {"w": small_param}, sigma=0.1, lr=0.01
        )
        
        raw_scores = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        normalized = noiser_cls.convert_fitnesses(
            frozen_params, noiser_params, raw_scores
        )
        
        assert normalized.shape == raw_scores.shape
        assert jnp.abs(jnp.mean(normalized)) < 0.01  # Should be ~zero mean

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_do_updates_signature(self, noiser_cls, small_param, es_key):
        """
        do_updates should return (new_noiser_params, new_params).
        """
        from hyperscalees.models.common import MM_PARAM
        
        frozen_params, noiser_params = noiser_cls.init_noiser(
            {"w": small_param}, sigma=0.1, lr=0.01, solver=optax.sgd
        )
        
        num_envs = 8
        iterinfos = make_iterinfo(num_envs)
        fitnesses = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        normalized = noiser_cls.convert_fitnesses(frozen_params, noiser_params, fitnesses)
        
        es_tree_key = {"w": es_key}
        es_map = {"w": MM_PARAM}
        
        new_noiser_params, new_params = noiser_cls.do_updates(
            frozen_params, noiser_params,
            {"w": small_param}, es_tree_key,
            normalized, iterinfos, es_map
        )
        
        assert isinstance(new_noiser_params, dict)
        assert isinstance(new_params, dict)
        assert "w" in new_params
        assert new_params["w"].shape == small_param.shape


class TestNoiserBehaviorEquivalence:
    """Test that different noisers have consistent behavior where expected."""

    def test_eval_mode_matches_across_noisers(self, small_param, es_key):
        """
        In evaluation mode (iterinfo=None), all noisers should return identical results.
        """
        x = jax.random.normal(jax.random.key(111), (3, small_param.shape[1]))
        
        outputs = []
        for noiser_cls in [EggRoll, OpenES]:
            frozen_params, noiser_params = noiser_cls.init_noiser(
                {"w": small_param}, sigma=0.1, lr=0.01
            )
            
            output = noiser_cls.do_mm(
                frozen_params, noiser_params, small_param, es_key, None, x
            )
            outputs.append(output)
        
        # All should be identical (just x @ W.T)
        for i in range(len(outputs) - 1):
            assert jnp.allclose(outputs[i], outputs[i+1]), \
                "Eval mode should be identical across noisers"

    def test_fitness_normalization_matches_across_noisers(self, small_param):
        """
        Fitness normalization should produce identical results across noisers.
        
        Both EggRoll and OpenES should use the same baseline subtraction and
        variance normalization. This test verifies they agree AND that the
        result has the expected statistical properties.
        """
        raw_scores = jnp.array([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0])
        
        normalized_results = []
        for noiser_cls in [EggRoll, OpenES]:
            frozen_params, noiser_params = noiser_cls.init_noiser(
                {"w": small_param}, sigma=0.1, lr=0.01
            )
            
            normalized = noiser_cls.convert_fitnesses(
                frozen_params, noiser_params, raw_scores
            )
            normalized_results.append(normalized)
        
        # Both should agree
        assert jnp.allclose(normalized_results[0], normalized_results[1]), \
            "Fitness normalization should be identical"
        
        # AND they should have the correct statistical properties
        for i, result in enumerate(normalized_results):
            noiser_name = ["EggRoll", "OpenES"][i]
            assert jnp.abs(jnp.mean(result)) < 1e-5, \
                f"{noiser_name} normalization should have zero mean"
            assert jnp.abs(jnp.var(result) - 1.0) < 0.1, \
                f"{noiser_name} normalization should have unit variance"


class TestBaseNoiserContract:
    """Test that BaseNoiser provides sensible defaults."""

    def test_base_noiser_init_returns_empty_dicts(self, small_param):
        """
        BaseNoiser.init_noiser should return empty dicts (no-op default).
        """
        frozen_params, noiser_params = BaseNoiser.init_noiser(
            {"w": small_param}, sigma=0.1, lr=0.01
        )
        
        assert frozen_params == {}
        assert noiser_params == {}

    def test_base_noiser_do_mm_is_standard_matmul(self, small_param, es_key):
        """
        BaseNoiser.do_mm should just be x @ W.T (no noise).
        """
        x = jax.random.normal(jax.random.key(222), (3, small_param.shape[1]))
        
        output = BaseNoiser.do_mm({}, {}, small_param, es_key, (0, 0), x)
        expected = x @ small_param.T
        
        assert jnp.allclose(output, expected), \
            "BaseNoiser.do_mm should be standard matmul"

    def test_base_noiser_do_updates_is_identity(self, small_param, es_key):
        """
        BaseNoiser.do_updates should not change parameters.
        """
        num_envs = 8
        iterinfos = make_iterinfo(num_envs)
        fitnesses = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        es_tree_key = {"w": es_key}
        es_map = {"w": 1}
        
        new_noiser_params, new_params = BaseNoiser.do_updates(
            {}, {},
            {"w": small_param}, es_tree_key,
            fitnesses, iterinfos, es_map
        )
        
        assert jnp.allclose(new_params["w"], small_param), \
            "BaseNoiser.do_updates should not change params"


class TestNoiserConfigOptions:
    """Test that configuration options work correctly."""

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_different_solvers_work(self, noiser_cls, small_param):
        """
        Different optax solvers should work with the noiser.
        """
        solvers = [optax.sgd, optax.adam, optax.adamw]
        
        for solver in solvers:
            frozen_params, noiser_params = noiser_cls.init_noiser(
                {"w": small_param},
                sigma=0.1,
                lr=0.01,
                solver=solver,
            )
            
            assert "opt_state" in noiser_params
            # Just verify it initialized without error

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_solver_kwargs_passed_through(self, noiser_cls, small_param):
        """
        solver_kwargs should be passed to the optax solver.
        """
        frozen_params, noiser_params = noiser_cls.init_noiser(
            {"w": small_param},
            sigma=0.1,
            lr=0.01,
            solver=optax.adam,
            solver_kwargs={"b1": 0.8, "b2": 0.99},
        )
        
        # If kwargs were wrong, this would have raised an error
        assert "opt_state" in noiser_params

    @pytest.mark.parametrize("noiser_cls", NOISER_CLASSES)
    def test_group_size_option(self, noiser_cls, small_param):
        """
        group_size option should be stored in frozen_noiser_params.
        """
        frozen_params, noiser_params = noiser_cls.init_noiser(
            {"w": small_param},
            sigma=0.1,
            lr=0.01,
            group_size=4,
        )
        
        assert "group_size" in frozen_params
        assert frozen_params["group_size"] == 4

    def test_eggroll_rank_option(self, small_param):
        """
        EggRoll should accept and store the rank option.
        """
        frozen_params, noiser_params = EggRoll.init_noiser(
            {"w": small_param},
            sigma=0.1,
            lr=0.01,
            rank=8,
        )
        
        assert "rank" in frozen_params
        assert frozen_params["rank"] == 8

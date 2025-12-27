"""
Shared fixtures for EGGROLL test suite.

These fixtures provide common test infrastructure for verifying the fundamental
claims of the EGGROLL paper: low-rank perturbations, antithetic sampling,
deterministic noise generation, and efficient gradient estimation.
"""
import pytest
import jax
import jax.numpy as jnp
import optax

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import hyperscalees as hs
from hyperscalees.noiser.eggroll import EggRoll, get_lora_update_params, get_nonlora_update_params
from hyperscalees.noiser.open_es import OpenES
from hyperscalees.noiser.base_noiser import Noiser as BaseNoiser
from hyperscalees.models.common import MLP, MM, Parameter, Linear, simple_es_tree_key, merge_inits


# ============================================================================
# Random key fixtures
# ============================================================================

@pytest.fixture
def base_key():
    """Base PRNG key for reproducible tests."""
    return jax.random.key(42)


@pytest.fixture
def model_key(base_key):
    """Key for model initialization."""
    return jax.random.fold_in(base_key, 0)


@pytest.fixture
def es_key(base_key):
    """Key for ES perturbation generation."""
    return jax.random.fold_in(base_key, 1)


# ============================================================================
# Parameter fixtures
# ============================================================================

@pytest.fixture
def small_param():
    """Small 2D parameter for detailed inspection (8x4 matrix)."""
    return jnp.ones((8, 4), dtype=jnp.float32) * 0.1


@pytest.fixture
def medium_param():
    """Medium 2D parameter for rank tests (64x32 matrix)."""
    key = jax.random.key(123)
    return jax.random.normal(key, (64, 32), dtype=jnp.float32) * 0.1


@pytest.fixture
def large_param():
    """Larger 2D parameter for scalability tests (256x128 matrix)."""
    key = jax.random.key(456)
    return jax.random.normal(key, (256, 128), dtype=jnp.float32) * 0.1


# ============================================================================
# Noiser configuration fixtures
# ============================================================================

@pytest.fixture
def eggroll_config():
    """Standard EggRoll configuration."""
    return {
        "sigma": 0.1,
        "lr": 0.01,
        "rank": 4,
        "group_size": 0,
        "freeze_nonlora": False,
        "noise_reuse": 0,
    }


@pytest.fixture
def eggroll_rank1_config():
    """EggRoll with rank=1 (minimal low-rank)."""
    return {
        "sigma": 0.1,
        "lr": 0.01,
        "rank": 1,
        "group_size": 0,
        "freeze_nonlora": False,
        "noise_reuse": 0,
    }


@pytest.fixture
def open_es_config():
    """Standard OpenES configuration for comparison."""
    return {
        "sigma": 0.1,
        "lr": 0.01,
        "group_size": 0,
        "freeze_nonlora": False,
        "noise_reuse": 0,
    }


# ============================================================================
# Initialized noiser fixtures
# ============================================================================

@pytest.fixture
def eggroll_noiser(small_param, eggroll_config):
    """Initialized EggRoll noiser with small param."""
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        {"w": small_param},
        eggroll_config["sigma"],
        eggroll_config["lr"],
        solver=optax.sgd,
        rank=eggroll_config["rank"],
        group_size=eggroll_config["group_size"],
        freeze_nonlora=eggroll_config["freeze_nonlora"],
        noise_reuse=eggroll_config["noise_reuse"],
    )
    return frozen_noiser_params, noiser_params


@pytest.fixture
def open_es_noiser(small_param, open_es_config):
    """Initialized OpenES noiser with small param."""
    frozen_noiser_params, noiser_params = OpenES.init_noiser(
        {"w": small_param},
        open_es_config["sigma"],
        open_es_config["lr"],
        solver=optax.sgd,
        group_size=open_es_config["group_size"],
        freeze_nonlora=open_es_config["freeze_nonlora"],
        noise_reuse=open_es_config["noise_reuse"],
    )
    return frozen_noiser_params, noiser_params


# ============================================================================
# MLP model fixtures
# ============================================================================

@pytest.fixture
def mlp_setup(model_key, es_key):
    """
    Complete MLP setup for end-to-end testing.
    
    Returns a dict with all components needed to run forward/update:
    - frozen_params, params, scan_map, es_map: model initialization
    - es_tree_key: perturbation keys
    - frozen_noiser_params, noiser_params: EggRoll noiser state
    """
    in_dim, out_dim = 8, 2
    hidden_dims = [16, 16]
    
    frozen_params, params, scan_map, es_map = MLP.rand_init(
        model_key,
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=hidden_dims,
        use_bias=True,
        activation="relu",
        dtype="float32"
    )
    
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        sigma=0.1,
        lr=0.01,
        solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999},
        rank=4,
    )
    
    return {
        "frozen_params": frozen_params,
        "params": params,
        "scan_map": scan_map,
        "es_map": es_map,
        "es_tree_key": es_tree_key,
        "frozen_noiser_params": frozen_noiser_params,
        "noiser_params": noiser_params,
        "in_dim": in_dim,
        "out_dim": out_dim,
    }


# ============================================================================
# Iteration info fixtures (for population-based testing)
# ============================================================================

@pytest.fixture
def small_population_iterinfo():
    """Iteration info for small population (8 members)."""
    num_envs = 8
    epoch = 0
    return (jnp.full(num_envs, epoch, dtype=jnp.int32), jnp.arange(num_envs))


@pytest.fixture
def medium_population_iterinfo():
    """Iteration info for medium population (64 members)."""
    num_envs = 64
    epoch = 0
    return (jnp.full(num_envs, epoch, dtype=jnp.int32), jnp.arange(num_envs))


@pytest.fixture
def large_population_iterinfo():
    """Iteration info for larger population (256 members)."""
    num_envs = 256
    epoch = 0
    return (jnp.full(num_envs, epoch, dtype=jnp.int32), jnp.arange(num_envs))


# ============================================================================
# Helper functions (not fixtures, but commonly used)
# ============================================================================

def compute_matrix_rank(matrix, tol=1e-5):
    """
    Compute numerical rank of a matrix using SVD.
    
    A singular value is considered zero if it's less than tol * max(singular_values).
    """
    s = jnp.linalg.svd(matrix, compute_uv=False)
    threshold = tol * jnp.max(s)
    return jnp.sum(s > threshold)


def make_iterinfo(num_envs, epoch=0):
    """Create iterinfo tuple for a given population size and epoch."""
    return (jnp.full(num_envs, epoch, dtype=jnp.int32), jnp.arange(num_envs))

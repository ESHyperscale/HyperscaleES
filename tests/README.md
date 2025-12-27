# EGGROLL Test Suite

This test suite serves as **living documentation** for the EGGROLL codebase, verifying the fundamental claims from the [EGGROLL paper](https://www.alphaxiv.org/abs/2511.16652).

## Running Tests

```bash
# From the HyperscaleES directory
pip install pytest
pytest

# Run specific test file
pytest tests/test_low_rank_structure.py -v

# Run with output
pytest -v -s
```

## Test Structure

| File | Paper Claim | What It Verifies |
|------|------------|------------------|
| `test_low_rank_structure.py` | Low-rank perturbations | AB^T has rank ≤ r, storage savings |
| `test_antithetic_sampling.py` | Variance reduction | ±σ mirrored sampling, pair cancellation |
| `test_noise_determinism.py` | Reproducible noise | Key folding, noise reuse |
| `test_fitness_normalization.py` | Stable updates | Zero-mean, unit-variance normalization |
| `test_forward_equivalence.py` | Efficient computation | do_mm matches explicit perturbation |
| `test_update_mechanics.py` | ES gradient estimation | Updates improve fitness |
| `test_high_rank_accumulation.py` | High-rank updates | Sum of low-rank → high-rank |
| `test_noiser_api.py` | API contract | Interface consistency across noisers |

## Key Design Decisions Documented

### 1. Low-Rank Perturbations (`test_low_rank_structure.py`)
EGGROLL generates perturbations as AB^T where A ∈ ℝ^(m×r), B ∈ ℝ^(n×r) with r << min(m,n).
This reduces storage from mn to r(m+n) per layer.

### 2. Antithetic Sampling (`test_antithetic_sampling.py`)
Thread pairs (2k, 2k+1) use the same base noise with opposite signs (±σ).
This is variance reduction: E[f(θ+ε) - f(θ-ε)] cancels common noise.

### 3. Deterministic Noise (`test_noise_determinism.py`)
Noise is generated deterministically via `jax.random.fold_in(key, epoch, thread_id)`.
This enables reconstructing perturbations during updates without storing them.

### 4. High-Rank from Low-Rank (`test_high_rank_accumulation.py`)
Although individual perturbations are rank-r, the weighted sum Σ w_i A_i B_i^T
can have rank up to min(N*r, min(m,n)) - this is the key insight that makes
EGGROLL as expressive as full-rank ES.

## Fixtures (`conftest.py`)

Common test fixtures include:
- `base_key`, `model_key`, `es_key`: PRNG keys
- `small_param`, `medium_param`, `large_param`: Test parameters
- `eggroll_config`, `eggroll_noiser`: Configured noiser instances
- `mlp_setup`: Complete MLP model setup for integration tests
- `make_iterinfo(num_envs, epoch)`: Helper to create population iterinfo

## Adding New Tests

When adding tests, follow these conventions:
1. Test files: `test_<concept>.py`
2. Test classes: `Test<Concept>`
3. Test methods: `test_<specific_behavior>`
4. Include docstrings explaining the paper claim being verified
5. Reference relevant code patterns (CODE: ...) where helpful

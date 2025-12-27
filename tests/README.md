# EGGROLL Test Suite

> **Living documentation** for the EGGROLL codebase, structured as a guide through the [EGGROLL paper](https://www.alphaxiv.org/abs/2511.16652).

This test suite verifies the fundamental claims made in the paper and documents the design decisions in the implementation. Each test file corresponds to a core concept from the paper.

---

## Quick Start

```bash
# From the HyperscaleES directory
uv sync
pytest ./tests -vs

# Run a specific concept
pytest tests/test_low_rank_structure.py -v
```

---

## Guide Through the Paper

### 1. The Core Innovation: Low-Rank Perturbations

**Paper Section**: The key insight of EGGROLL is replacing full-rank Gaussian perturbations with low-rank structured perturbations.

**The Claim**: Instead of sampling ε ∈ ℝ^(m×n) with mn parameters, EGGROLL samples:
```
ε = AB^T   where A ∈ ℝ^(m×r), B ∈ ℝ^(n×r), r << min(m,n)
```

**Test File**: [`test_low_rank_structure.py`](test_low_rank_structure.py)

| Test | Verifies |
|------|----------|
| `test_perturbation_returns_two_factors` | `get_lora_update_params` returns (A, B) not full matrix |
| `test_reconstructed_perturbation_has_low_rank` | SVD rank of AB^T ≤ r |
| `test_rank_parameter_controls_perturbation_rank` | Rank parameter directly controls perturbation rank |
| `test_storage_savings` | r(m+n) << mn storage reduction |

**Code Location**: `hyperscalees/noiser/eggroll.py::get_lora_update_params`

---

### 2. Efficient Forward Pass

**Paper Section**: The computational trick that makes low-rank worthwhile.

**The Claim**: Computing `x @ (W + AB^T)` naively requires forming the mn perturbation. Instead:
```
x @ (W + AB^T)^T = x @ W^T + x @ B @ A^T
```
This reduces cost from O(mn) to O(r(m+n)) per forward pass.

**Test File**: [`test_forward_equivalence.py`](test_forward_equivalence.py)

| Test | Verifies |
|------|----------|
| `test_do_mm_matches_explicit_perturbation` | **Core correctness**: efficient path = explicit path |
| `test_do_Tmm_matches_transposed_computation` | Transposed version also correct |
| `test_antithetic_pairs_bracket_base_output` | (output+ + output-)/2 = unperturbed output |
| `test_vmapped_forward_over_population` | Works correctly when vmapped over population |

**Code Location**: `hyperscalees/noiser/eggroll.py::do_mm`, `do_Tmm`

---

### 3. Antithetic (Mirrored) Sampling

**Paper Section**: Variance reduction via antithetic sampling.

**The Claim**: For each perturbation ε, we also evaluate -ε. Thread pairs (2k, 2k+1) use:
```
thread 2k:   θ + σε
thread 2k+1: θ - σε
```

**Why It Matters**: If f(θ+ε) ≈ f(θ-ε), the contributions cancel, reducing gradient variance without extra compute.

**Test File**: [`test_antithetic_sampling.py`](test_antithetic_sampling.py)

| Test | Verifies |
|------|----------|
| `test_even_odd_thread_pairs_have_opposite_sign` | A_even = -A_odd, B_even = B_odd |
| `test_perturbation_matrices_are_negatives` | Full perturbations are exact negatives |
| `test_antithetic_cancellation_in_mean` | Equal weights → sum = 0 |
| `test_antithetic_pairs_share_base_noise` | Pairs share random seed (thread_id // 2) |

**Code Location**: `hyperscalees/noiser/eggroll.py::get_lora_update_params` (the `thread_id % 2` check)

---

### 4. Deterministic Noise via Key Folding

**Paper Section**: Memory efficiency through reproducible noise.

**The Claim**: Perturbations are generated deterministically from (key, epoch, thread_id). This means we don't need to store perturbations during the forward pass—we regenerate them during the update.

```python
# Noise can be reconstructed from just the iteration info
key = fold_in(fold_in(base_key, true_epoch), true_thread_idx)
```

**Test File**: [`test_noise_determinism.py`](test_noise_determinism.py)

| Test | Verifies |
|------|----------|
| `test_same_inputs_produce_same_noise` | Identical inputs → identical outputs |
| `test_different_epochs_produce_different_noise_when_noise_reuse_nonzero` | Epochs vary noise (when noise_reuse > 0) |
| `test_noise_reuse_zero_means_same_noise_every_epoch` | noise_reuse=0 → same noise always |
| `test_key_folding_is_commutative_in_reproduction` | Call order doesn't matter |

**Code Location**: `hyperscalees/noiser/eggroll.py::get_lora_update_params` (key folding logic)

**⚠️ Gotcha**: `noise_reuse=0` means `true_epoch=0` ALWAYS. Use `noise_reuse=1` if you want different noise each epoch.

---

### 5. High-Rank Updates from Low-Rank Perturbations

**Paper Section**: The theoretical justification for why low-rank doesn't limit expressiveness.

**The Claim**: 
> "Although individual perturbations are low-rank, the expression on the right side is actually high-rank, due to the properties of sums of low-rank matrices."

The gradient estimate is:
```
∇̂ = Σᵢ wᵢ AᵢBᵢ^T
```
Even if each AᵢBᵢ^T is rank-r, the sum can have rank up to min(N×r, min(m,n)).

**Test File**: [`test_high_rank_accumulation.py`](test_high_rank_accumulation.py)

| Test | Verifies |
|------|----------|
| `test_sum_of_rank1_exceeds_rank1` | Sum of rank-1 matrices has rank > 1 |
| `test_accumulated_rank_grows_with_population` | More population → higher rank |
| `test_full_rank_achievable_with_sufficient_population` | Can achieve full rank |
| `test_weighted_sum_respects_fitness_weights` | Higher-weighted perturbations dominate |

**Code Location**: This is emergent from the math—verified by computing SVD ranks.

---

### 6. Fitness Normalization

**Paper Section**: Standard ES practice for stable gradient estimation.

**The Claim**: Raw fitness scores are normalized before computing gradient estimates:
1. Mean subtraction (baseline): centers around zero
2. Variance normalization: scales to unit variance

```python
normalized = (fitness - mean) / sqrt(var + ε)
```

**Test File**: [`test_fitness_normalization.py`](test_fitness_normalization.py)

| Test | Verifies |
|------|----------|
| `test_normalized_scores_have_zero_mean` | Mean ≈ 0 |
| `test_normalized_scores_have_unit_variance` | Variance ≈ 1 |
| `test_normalization_preserves_ordering` | Ranking unchanged |
| `test_constant_scores_handled_gracefully` | No NaN/Inf on constant input |

**Code Location**: `hyperscalees/noiser/eggroll.py::convert_fitnesses`

---

### 7. ES Gradient Estimation & Updates

**Paper Section**: The ES gradient formula and how EGGROLL implements it.

**The Claim**: ES estimates gradients as:
```
∇_θ E[F(θ+σε)] ≈ (1/σ) E[F(θ+σε)·ε]
```

With antithetic sampling and fitness normalization, the update aggregates weighted perturbations.

**Test File**: [`test_update_mechanics.py`](test_update_mechanics.py)

| Test | Verifies |
|------|----------|
| `test_higher_fitness_perturbation_dominates_update` | Update correlates with high-fitness perturbation |
| `test_equal_antithetic_fitnesses_cancel_to_no_update` | Equal fitness → zero update |
| `test_update_magnitude_scales_with_lr` | Larger LR → larger update |
| `test_update_improves_simple_fitness` | ES actually optimizes (scalar task) |
| `test_optimizer_state_updates` | Adam state increments correctly |
| `test_frozen_nonlora_skips_bias_updates` | freeze_nonlora works |

**Code Location**: `hyperscalees/noiser/eggroll.py::_do_update`, `do_updates`

---

### 8. Noiser API Contract

**Design Decision**: All noiser implementations (EggRoll, OpenES, BaseNoiser) follow the same interface, making them interchangeable.

**Test File**: [`test_noiser_api.py`](test_noiser_api.py)

| Test | Verifies |
|------|----------|
| `test_init_noiser_returns_correct_structure` | Returns (frozen_params, noiser_params) |
| `test_do_mm_signature` | do_mm accepts correct arguments |
| `test_do_updates_signature` | do_updates returns (new_noiser_params, new_params) |
| `test_eval_mode_matches_across_noisers` | iterinfo=None → same output for all noisers |

**Code Location**: `hyperscalees/noiser/*.py`

---

## Code Map

```
src/hyperscalees/noiser/
├── base_noiser.py      # Abstract interface
├── eggroll.py          # EGGROLL implementation (low-rank)
├── open_es.py          # OpenES baseline (full-rank)
└── ...

tests/
├── conftest.py                    # Shared fixtures
├── test_low_rank_structure.py     # §1 - Low-rank perturbations
├── test_forward_equivalence.py    # §2 - Efficient forward pass
├── test_antithetic_sampling.py    # §3 - Variance reduction
├── test_noise_determinism.py      # §4 - Key folding
├── test_high_rank_accumulation.py # §5 - High-rank updates
├── test_fitness_normalization.py  # §6 - Score normalization
├── test_update_mechanics.py       # §7 - Gradient estimation
└── test_noiser_api.py             # §8 - API contract
```

---

## Key Fixtures (`conftest.py`)

| Fixture | Description |
|---------|-------------|
| `base_key`, `es_key` | PRNG keys for reproducibility |
| `small_param` | 8×4 matrix for detailed inspection |
| `medium_param` | 64×32 matrix for rank tests |
| `large_param` | 256×128 matrix for scalability |
| `eggroll_config` | Standard config (σ=0.1, lr=0.01, rank=4) |
| `make_iterinfo(num_envs, epoch)` | Helper to create (epochs, thread_ids) arrays |
| `compute_matrix_rank(M)` | SVD-based rank computation |

---

## Adding Tests

Follow these conventions:
1. **File naming**: `test_<paper_concept>.py`
2. **Module docstring**: Include `PAPER CLAIM:` and `DESIGN DECISION:`
3. **Test docstrings**: Explain what's being verified and why it matters
4. **Reference code**: Use `CODE: ...` comments for relevant implementations


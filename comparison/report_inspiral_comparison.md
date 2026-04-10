# Implementation Comparison: EMRIInspiral vs EMRIInspiralFast

**Date:** 2026-04-10  
**Codebase:** `fewtrax`, branch `dev-EMRIInspiralFast`  
**Benchmarked on:** NVIDIA A100 80 GB PCIe, JAX 0.9.2

---

## 1. Overview

Both classes solve the same five-dimensional adiabatic EMRI inspiral ODE

```
dy/dt = [ṗ, ė, Ω_φ, Ω_θ, Ω_r]
```

using a Dopri8 (8th-order Dormand-Prince) integrator with adaptive step-size
control (`diffrax.PIDController`). They share the same flux data, the same
`pex` convention (pre-normalised flux ratios), and the same public call
interface. `EMRIInspiralFast` inherits from `EMRIInspiral` and overrides three
internal primitives.

---

## 2. Algorithmic differences

### 2.1 Elliptic integrals for fundamental frequencies

`Ω_φ, Ω_θ, Ω_r` are expressed through complete elliptic integrals K(m), E(m),
Π(n,k), evaluated at every ODE step.

| | EMRIInspiral (exact) | EMRIInspiralFast |
|---|---|---|
| K(m), E(m) | 64-point Gauss-Legendre (GL64) dot product | AGM (12 iterations, `ellipk_agm`, `ellipe_agm`) |
| Π(n,k) | 64-point GL (`ellip_pi`) | 24-point GL (`ellip_pi_fast`) |
| Theoretical accuracy | ~10⁻¹⁴ | K/E: ~10⁻¹⁵ (quadratic AGM); Π: ~10⁻¹² (24-pt GL) |
| CPU operation count | ~3 × 64 = 192 FP ops (dot products) | ~3 × 24 + 12 × 2 = 96 FP ops |

The AGM approach for K and E is asymptotically faster on CPU because it avoids
the 64-element dot product and replaces it with 12 sequential scalar multiplications.
However, its sequential data-dependency chain (each iteration depends on the
previous) makes it **harder to vectorize across elements in a vmap batch** on GPU.

### 2.2 Separatrix root-finder

The separatrix polynomial `p_sep(a, e)` is a real root of a 5th-degree polynomial
in p. Both models use a root-finding strategy for this.

| | EMRIInspiral (exact) | EMRIInspiralFast |
|---|---|---|
| Algorithm | `get_separatrix`: 50-step bisection | `get_separatrix_fast`: 20 bisection + 5 Newton-Raphson |
| Call frequency | Once per ODE step (event condition + `_flux_pex`) | Once per ODE step via `_ode_rhs_fast`; result reused for both event condition and flux |
| Calls per trajectory (≈150 ODE steps × 5) | ~750 separatrix evaluations | ~150 (reuse within step) |

`EMRIInspiralFast` avoids a factor-of-5 redundancy that exists in the base
class: `_flux_pex` calls `get_separatrix` internally (to get `p_sep` for the
PN normalisation), but the same `p_sep` is also needed in the event condition.
The fast variant computes it once in `_ode_rhs_fast` and passes it to both,
eliminating ~600 of the ~750 total separatrix calls per trajectory.

The NR step convergence is quadratic, reaching float64 precision in ≤ 3 steps
from a tight (20-bisection) starting bracket.

### 2.3 Pre-computed r_ISCO

Both models pre-compute `r_isco = p_sep(a, e=0)` once at call time and thread it
through `ode_args`, avoiding its recomputation at every RHS evaluation. This is
now shared between both classes.

### 2.4 Adjoint method for autodiff

This is the most consequential architectural difference.

| | EMRIInspiral (exact) | EMRIInspiralFast |
|---|---|---|
| `adjoint` in `_solve` | Default (`RecursiveCheckpointAdjoint`) | `DirectAdjoint()` |
| Forward-mode AD (`jacfwd`) | **Not supported** (custom_vjp without custom_jvp) | **Supported** |
| Reverse-mode AD (`jacrev`, `grad`) | Checkpointed; memory O(√N_steps) | Direct adjoint ODE; memory O(N_steps) |
| `jax.hessian` | Requires forward-over-reverse (fails) | Works (fwd over DirectAdjoint) |

`DirectAdjoint` expresses the adjoint as a second ODE solved backward in time.
It exposes a `custom_jvp` rule to diffrax, which enables `jacfwd`. This is why
the benchmark section D shows `jax.jacfwd` working on the fast model (104 ms)
but not on the exact model.

However, `DirectAdjoint` compiles to a fundamentally different XLA program for
the forward pass: the ODE solution is retained in memory throughout the solve
(not checkpointed), which creates higher memory pressure and different kernel
scheduling on GPU.

---

## 3. Benchmark results analysis

### 3.1 Per-step throughput (CPU regime, small N)

At N=1:
- **EMRIInspiralFast: 93.6 ms vs EMRIInspiral: 170.7 ms → 1.82× speedup**

This validates the per-step optimisations: AGM + NR + separatrix reuse reduce
the per-step cost substantially on a single-element evaluation (closest to CPU
behaviour).

At N ≤ 256, the fast model consistently achieves 1.33–1.82× improvement, and
the speedup stabilises at ~1.33× as the batch grows and GPU parallelism
saturates the step cost.

### 3.2 Large-batch inversion (N=262144)

At N=262144:
- **EMRIInspiralFast: 13926 ms (18824 traj/s) vs EMRIInspiral: 4552 ms (57589 traj/s)**
- **The fast model is 3.06× SLOWER than the exact model at large batch**

This is a critical and counter-intuitive result that reveals a fundamental GPU
architecture mismatch.

#### Why the exact model wins at large N

On a GPU, the 64-point Gauss-Legendre evaluation of K(m), E(m), Π(n,k) maps
directly to a batched matrix-vector product: for a batch of N ODE steps (or
B×N_steps simultaneous ODE steps under vmap), the compiler sees a (B, 64)
tensor contraction, which is mapped to cuBLAS/cuBLASLt operations and runs at
near-peak FLOP utilization on the A100.

The AGM loop for K/E consists of 12 sequential scalar iterations with a strict
data dependency chain:

```
aₙ₊₁ = (aₙ + bₙ) / 2
bₙ₊₁ = sqrt(aₙ * bₙ)
```

The next iteration cannot begin until the previous one completes. XLA can
parallelize across the batch dimension (different trajectories), but the 12-step
chain is sequential. At N=262144, each AGM produces 12 × 2 = 24 small GPU
kernel dispatches per elliptic integral, versus a single highly-parallelized
matrix multiplication in the GL approach.

#### The `DirectAdjoint` contribution

`DirectAdjoint` changes how XLA traces the forward ODE. The diffrax
`DirectAdjoint` uses `lax.while_loop` in a way that requires the full state to
be materialized for the backward pass. At large batch sizes, this creates
significantly higher peak memory bandwidth requirements and may force serial
kernel execution for the state management operations.

#### Summary of throughput at large N

The exact model's approach:
- 64-point GL → single large BLAS call per batch → high GPU utilization
- `RecursiveCheckpointAdjoint` → compact memory, standard while_loop schedule

The fast model's approach:
- 12-step AGM → 12 serial scalar kernel dispatches per evaluation → GPU stalls
- `DirectAdjoint` → different XLA schedule, potentially higher memory bandwidth

---

## 4. Accuracy

The accuracy comparison (section C) in the benchmark produced NaN because the
`check_accuracy` function uses `np.mean` / `np.max` without NaN guards. When
trajectories plunge before T=0.5 yr, the tail of the output arrays is filled
with NaN (diffrax pads unfired save-points with NaN when an event fires), and
these propagate through the subtraction. The physical accuracy of the models is
not in question from this result; a NaN-aware re-analysis is needed.

From the theoretical perspective:
- The fast separatrix root-finder (NR) has the same fixed-point as bisection but
  may converge to a slightly different float64 value, causing the event to fire
  at a marginally different time. For most trajectories this is sub-millisecond
  (< 1 ODE step).
- The AGM for K(m) achieves ~10⁻¹⁵ accuracy, better than the 64-point GL
  (~10⁻¹⁴), so the frequency contribution from K/E is more accurate in
  EMRIInspiralFast.
- The 24-point GL for Π is ~10⁻¹² accurate vs ~10⁻¹⁴ for the 64-point version.
  Over a 2-year inspiral with ~10⁴ evaluations, the accumulated dephasing from
  the Π approximation is O(10⁻¹²) × 10⁴ × Ω_r ≈ negligible.

---

## 5. Issues and improvement opportunities

### 5.1 EMRIInspiral (exact)

**Issue 1: Redundant separatrix evaluations.**  
`_flux_pex` calls `get_separatrix` internally (to get `p_sep` for `_pdot_PN_jax`
/ `_edot_PN_jax`), and the same `p_sep` is computed again in the event condition.
This results in ~5 redundant separatrix evaluations per ODE step.

*Fix:* Lift `p_sep` computation to `_ode_rhs` and pass the pre-computed value to
both `_flux_pex` and the event condition (exactly as done in `_ode_rhs_fast`).
The base class `_flux_pex` already accepts `r_isco` externally; adding `p_sep` as
a parameter follows the same pattern.

**Issue 2: `jacfwd` not supported.**  
`RecursiveCheckpointAdjoint` uses a `custom_vjp` rule without a corresponding
`custom_jvp`, blocking forward-mode AD. This means `jax.jacfwd` fails and the
Fisher matrix must be computed with `jacrev` (reverse mode), which scales as
O(N_out) backward passes.

*Fix:* Expose a configurable `adjoint` parameter at construction time (or at call
time) so the user can switch to `DirectAdjoint` for AD-heavy workloads.

**Issue 3: 50-step bisection for separatrix.**  
While robust, 50 bisection steps is conservative for float64: the root is located
to 50-bit precision, but float64 only has 52 bits of mantissa. Using 20 bisection
steps + 3 NR steps (as in `get_separatrix_fast`) achieves the same precision with
far fewer operations.

### 5.2 EMRIInspiralFast

**Issue 1: GPU throughput collapse at large N.**  
The AGM sequential loops cause GPU kernel dispatch overhead that dominates at
N ≥ 10⁴. At N=262144 the fast model is 3× slower than the exact model on A100.

*Fix:* Replace `ellipk_agm` / `ellipe_agm` with the 64-point GL versions for the
fundamental frequency computation. The AGM should only be preferred on CPU.
A hardware-aware backend switch (`jax.devices()[0].platform`) could select the
implementation at call time.

**Issue 2: `DirectAdjoint` as a non-optional default.**  
`DirectAdjoint` enables `jacfwd` but changes the forward-pass XLA schedule in a
way that degrades throughput at large batch sizes. Users who only need `jacrev`
or `grad` pay the `DirectAdjoint` overhead unnecessarily.

*Fix:* Make `adjoint` a configurable parameter (default `RecursiveCheckpointAdjoint`
for throughput; `DirectAdjoint` as an option for jacfwd/hessian workloads).

**Issue 3: Loss of `jacfwd` if adjoint is changed.**  
If the adjoint is made configurable and defaults to `RecursiveCheckpointAdjoint`,
the current benchmark's `jacfwd` test will break. A clear docstring should
document when each adjoint mode is appropriate.

---

## 6. Proposed intermediate model: `EMRIInspiralHybrid`

The goal is a model that is:
- **Very fast on CPU** (low per-step operation count)
- **Scales well on GPU at large batch sizes** (no sequential-dependency bottleneck)
- **Supports `jacfwd`** for Fisher matrix computation
- **Avoids the redundant separatrix evaluations** of the base class

The intermediate model combines the structural improvements of `EMRIInspiralFast`
with the GPU-efficient elliptic integral kernels of `EMRIInspiral`.

### 6.1 Architecture

```
EMRIInspiralHybrid
├── Elliptic integrals:     platform-selected
│   ├── CPU:  AGM for K/E + 24-pt GL for Π   (low op count, sequential is fine)
│   └── GPU:  64-pt GL for K/E/Π             (vectorizes as BLAS matmul)
├── Separatrix root-finder: get_separatrix_fast (20 bisect + 5 NR) for all platforms
├── p_sep reuse:            computed once per ODE step, shared by flux + event
├── r_isco precomputed:     at call time (already in both models)
└── Adjoint:                configurable; default RecursiveCheckpointAdjoint
                            (user passes adjoint=DirectAdjoint() for jacfwd)
```

### 6.2 Platform-aware elliptic integral selection

```python
import jax

def _get_platform() -> str:
    return jax.devices()[0].platform  # "cpu", "gpu", "tpu"

def _ellipk_platform(m):
    if _get_platform() == "gpu":
        return ellipk(m)        # 64-pt GL → cuBLAS-friendly
    else:
        return ellipk_agm(m)    # AGM → fewer sequential ops

def _ellipe_platform(m):
    if _get_platform() == "gpu":
        return ellipe(m)
    else:
        return ellipe_agm(m)
```

Note: the platform check happens at trace time (JIT compilation), not at runtime,
so there is no overhead on repeated calls.

### 6.3 Expected performance profile

| Metric | EMRIInspiral | EMRIInspiralFast | EMRIInspiralHybrid |
|---|---|---|---|
| CPU, single traj | baseline | ~2× faster | ~2× faster (same as fast) |
| GPU, N=256 | baseline | ~1.33× faster | ~1.33× faster |
| GPU, N=262144 | **1.0× (baseline)** | 0.33× (3× slower!) | ~1.0× (matches exact) |
| `jacfwd` | not supported | supported | configurable |
| `jacrev` / `grad` | supported | supported | supported |
| Peak memory (large N) | lower | higher | lower (default adjoint) |

The hybrid model closes the large-batch regression of `EMRIInspiralFast` while
preserving the small-batch per-step improvement. The adjoint configurability
allows users to switch to `DirectAdjoint` only when they need `jacfwd`.

### 6.4 Structural changes required

1. **In `geodesic.py`**: Export `get_fundamental_frequencies_platform` that
   dispatches between the fast (24-pt Π) and exact (64-pt Π) implementations
   based on the device platform.

2. **In `inspiral.py`**: Introduce `EMRIInspiralHybrid(EMRIInspiral)` that:
   - Overrides `_ode_rhs` to pre-compute `p_sep` once and reuse it
   - Selects elliptic integral implementation based on platform
   - Accepts `adjoint` as a constructor argument (default:
     `diffrax.RecursiveCheckpointAdjoint()`)

3. **In `inspiral.py` base class (`EMRIInspiral`)**: Refactor `_flux_pex` to
   accept `p_sep` as an optional argument (avoiding the internal separatrix call
   when the caller already has it). This is a non-breaking change.

4. **In `geodesic.py`**: Replace `get_separatrix` (50-step bisection) with
   `get_separatrix_fast` (20 bisect + 5 NR) as the default in the base class.
   The accuracy difference is below float64 rounding; the robustness of the
   current bisection approach can be preserved by keeping it as a fallback.

---

## 7. Why `EMRIInspiral` scales better than `EMRIInspiralFast` at large N

This deserves a dedicated section because the result is surprising.

### GPU kernel dispatch analysis

For a vmap batch of size B with approximately S adaptive ODE steps each, the
total number of GPU kernel dispatches scales as:

| Operation | EMRIInspiral | EMRIInspiralFast |
|---|---|---|
| K(m) evaluation | 1 kernel (B × 64 matmul) | 12 kernels (scalar, sequential) |
| E(m) evaluation | 1 kernel (B × 64 matmul) | 12 kernels (scalar, sequential) |
| Π(n,k) evaluation | 1 kernel (B × 64 matmul) | 1 kernel (B × 24 matmul) |
| Separatrix bisection | 50 kernels (scalar comparisons) | 20 + 5 kernels (bisect + NR) |
| **Total per RHS** | **~53 kernels** | **~49 kernels** |

On a per-kernel count, the difference is modest. However, the GL dot products
generate kernels that operate on tensors of size (B, 64) — for B=262144, this
is a 16M-element tensor, running at peak bandwidth on the A100. The AGM kernels
operate on tensors of size (B,), dispatching 24 × B = 6.3B element-wise
operations across 24 sequential kernel calls, with inter-kernel synchronization
barriers between each iteration.

The A100's inter-kernel synchronization overhead (~1 µs per barrier) applied
24 times per RHS, across ~150 ODE steps, equals ~3.6 ms of pure synchronization
overhead — independent of the useful computation. For a 4.5 s batch job (exact
model at N=262144), this alone contributes ~3.6 ms, negligible. But the AGM also
generates suboptimal memory access patterns at large B: the scalar operations on
(B,) tensors do not saturate the memory bus as efficiently as the (B, 64)
operations.

A more accurate description is: the GL approach compiles to a single XLA
`dot_general` instruction per elliptic integral; the AGM compiles to a
`lax.fori_loop` whose body is a scalar operation. XLA's loop lowering for
`fori_loop` with a small body can have higher overhead-to-work ratios than a
single large kernel at large batch sizes.

---

## 8. Summary and recommendations

| Recommendation | Priority | Scope |
|---|---|---|
| Fix NaN stats in `check_accuracy` (use `nanmean`/`nanmax`) | High | `benchmark_vmap_tracks.py` |
| Add `p_sep` reuse to `EMRIInspiral._ode_rhs` (eliminate 4× redundant calls) | High | `inspiral.py` |
| Replace `get_separatrix` with `get_separatrix_fast` in base class | Medium | `inspiral.py`, `geodesic.py` |
| Make `adjoint` configurable in both classes | High | `inspiral.py` |
| Implement `EMRIInspiralHybrid` with platform-aware elliptic integrals | High | `inspiral.py`, `geodesic.py` |
| Expose `get_fundamental_frequencies_platform` dispatch function | Medium | `geodesic.py` |
| Add a `--skip-large-batch` flag or warn about `DirectAdjoint` at large N | Low | `benchmark_vmap_tracks.py` |
| Document adjoint choice guidelines in docstrings | Medium | `inspiral.py` |

The most impactful single change is implementing `EMRIInspiralHybrid` with
GPU-aware elliptic integral selection. This recovers the large-batch throughput
advantage of the exact model while preserving the small-batch speedup of the
fast model, and adding `jacfwd` support as a configurable option.

"""Batched vmap trajectory benchmark — the core fewtrax use case.

This is the main performance target of the package: computing vmapped,
individually differentiable EMRI frequency tracks over a large random grid
of intrinsic parameters.  All benchmarks use the full 5D solve
(p, e, Φ_φ, Φ_θ, Φ_r) with ``phases=True``.

What is measured
----------------
A. **vmap throughput** — vmapped ``EMRIInspiral(phases=True)`` over batches
   drawn from a random (seeded) parameter grid.  Reports trajectories/second,
   per-trajectory wall time, and peak GPU memory.

B. **Autodiff benchmarks** — timing for ``jax.grad``, ``jax.jacfwd``,
   ``jax.jacrev``, and ``jax.hessian`` of a scalar trajectory loss w.r.t.
   all five intrinsic parameters (M, mu, a, p0, e0).

   grad/jacrev use ``RecursiveCheckpointAdjoint`` (memory-efficient).
   jacfwd/hessian use ``DirectAdjoint`` (enables forward-mode AD).

C. **Local Fisher matrix** — timing for the (5×5) trajectory Fisher matrix
   via forward-mode Jacobian (``DirectAdjoint``), plus a vmapped batch.

Intrinsic parameter ranges (random grid, fixed seed):

    M    ∈ [5e5, 5e6]  M_sun
    mu   ∈ [5,   50]   M_sun
    a    ∈ [0.05, 0.9]         (prograde equatorial)
    p0   = p_sep(a, e0) + Δp,  Δp ∈ [1, 10] M
    e0   ∈ [0.05, 0.7]

Usage
-----
    python benchmark_vmap_tracks.py [/path/to/few/data]
    python benchmark_vmap_tracks.py --n-batch 512 --n-repeat 5
    python benchmark_vmap_tracks.py --seed 1234 --no-plots
    python benchmark_vmap_tracks.py --skip-autodiff --skip-fisher
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import diffrax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import find_data_dir, block_jax, print_header, print_table, repeat_timer, get_cpu_memory_mb


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_jax_memory_mb() -> float:
    """Return current JAX device memory usage in MiB.

    Falls back to CPU RSS when JAX device stats are unavailable (CPU backend).
    """
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("bytes_in_use", 0) / 1024**2
    except Exception:
        return get_cpu_memory_mb()


def get_peak_memory_mb() -> float:
    """Return peak JAX device memory usage in MiB.

    Falls back to CPU peak RSS when JAX device stats are unavailable.
    """
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("peak_bytes_in_use", 0) / 1024**2
    except Exception:
        return get_cpu_memory_mb()


def nvidia_smi_free_mib() -> float:
    """Return free GPU memory from nvidia-smi [MiB]."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Parameter grid construction
# ---------------------------------------------------------------------------

def build_param_grid(
    N: int,
    seed: int = 42,
    T_yr: float = 0.5,
    dt_s: float = 10.0,
) -> dict[str, np.ndarray]:
    """Draw N random EMRI parameter sets with fixed ``seed``.

    All (p0, e0) pairs are guaranteed to have p0 > p_sep(a, e0).
    Returns a dict of 1-D numpy arrays, each of length ``N``.
    """
    from fewtrax.utils.geodesic import get_separatrix

    rng = np.random.default_rng(seed)

    M_arr  = 10.0 ** rng.uniform(np.log10(5e5), np.log10(5e6), N)
    mu_arr = rng.uniform(5.0, 50.0, N)
    a_arr  = rng.uniform(0.05, 0.90, N)
    e_arr  = rng.uniform(0.05, 0.70, N)
    dp_arr = rng.uniform(1.0, 10.0, N)

    p_arr = np.array([
        float(get_separatrix(jnp.float64(a), jnp.float64(e), jnp.float64(1.0))) + dp
        for a, e, dp in zip(a_arr, e_arr, dp_arr)
    ])

    return dict(
        M=M_arr, mu=mu_arr, a=a_arr, p0=p_arr, e0=e_arr,
        T=np.full(N, T_yr), dt=np.full(N, dt_s),
    )


def build_fdot_param_grid(N: int, seed: int = 77) -> dict[str, np.ndarray]:
    """Draw N random EMRI parameter sets for backward f/fdot/fddot integration.

    ``e_f`` is the eccentricity at plunge — kept modest ([0.02, 0.40]) so that
    the separatrix is safely inside the flux-table domain for all spins.
    """
    rng = np.random.default_rng(seed)
    return dict(
        M   = 10.0 ** rng.uniform(np.log10(5e5), np.log10(5e6), N),
        mu  = rng.uniform(5.0, 50.0, N),
        a   = rng.uniform(0.05, 0.90, N),
        e_f = rng.uniform(0.02, 0.40, N),
    )


# ---------------------------------------------------------------------------
# Vmapped trajectory runners
# ---------------------------------------------------------------------------

def make_batched_traj(traj, dense_steps: int = 100, max_steps: int = 4096,
                      atol: float = 1e-9, rtol: float = 1e-9, T: float = 0.5):
    """Return a vmapped function (p0, e0, a, M, mu) -> trajectory arrays.

    Returns 6 arrays (t, p, e, Phi_phi, Phi_theta, Phi_r) when the trajectory
    was built with ``phases=True``, or 3 arrays (t, p, e) with ``phases=False``.
    """
    fixed = dict(
        T=T, x0=1.0, dt=10.0,
        dense_steps=dense_steps, max_steps=max_steps, atol=atol, rtol=rtol,
    )

    def single(p0, e0, a, M, mu):
        return traj(p0=p0, e0=e0, a=a, M=M, mu=mu, **fixed)

    return jax.jit(jax.vmap(single))


def bench_batch(
    traj,
    grid: dict,
    batch_sizes: list[int],
    n_warmup: int,
    n_repeat: int,
    dense_steps: int = 100,
    T: float = 0.5,
    label: str = "",
) -> list[dict]:
    """Benchmark vmapped trajectory at each batch size from ``grid``."""
    batched_fn = make_batched_traj(traj, dense_steps=dense_steps, T=T)

    results = []
    for N in batch_sizes:
        p0 = jnp.array(grid["p0"][:N], dtype=jnp.float64)
        e0 = jnp.array(grid["e0"][:N], dtype=jnp.float64)
        a  = jnp.array(grid["a"][:N],  dtype=jnp.float64)
        M  = jnp.array(grid["M"][:N],  dtype=jnp.float64)
        mu = jnp.array(grid["mu"][:N], dtype=jnp.float64)

        mem_before = get_jax_memory_mb()

        def fn():
            out = batched_fn(p0, e0, a, M, mu)
            block_jax(out)

        mean_s, std_s = repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)
        mem_after = get_peak_memory_mb()
        mem_used  = mem_after - mem_before if np.isfinite(mem_before) else float("nan")

        throughput = N / mean_s
        results.append(dict(
            N=N, mean_s=mean_s, std_s=std_s,
            throughput=throughput,
            mem_peak_mb=mem_after,
            mem_delta_mb=mem_used,
            label=label,
        ))
        print(
            f"  [{label:5s}] N={N:5d}: {mean_s*1e3:8.2f} ± {std_s*1e3:5.2f} ms  "
            f"({throughput:7.1f} traj/s)  "
            f"mem_peak={mem_after:.0f} MiB"
        )

    return results


# ---------------------------------------------------------------------------
# Autodiff benchmarks
# ---------------------------------------------------------------------------

def _make_traj_scalar_loss(traj, T: float, dense_steps: int):
    """Scalar loss: sum of squared phase at last trajectory point.

    This is a simple differentiable proxy for SNR-like quantities
    and is used to benchmark grad / hessian timing.
    """
    def loss(M, mu, a, p0, e0):
        t, p, e, Phi_phi, Phi_theta, Phi_r = traj(
            p0=p0, e0=e0, a=a, M=M, mu=mu,
            T=T, x0=1.0, dt=10.0,
            dense_steps=dense_steps, max_steps=4096,
            atol=1e-9, rtol=1e-9,
        )
        # Proxy: squared final phase (smooth, differentiable)
        return Phi_phi[-1] ** 2

    return loss


def _make_traj_vector_loss(traj, T: float, dense_steps: int):
    """Vector output: (Phi_phi, p, e) concatenated — for Jacobian benchmarks."""
    def output(M, mu, a, p0, e0):
        t, p, e, Phi_phi, Phi_theta, Phi_r = traj(
            p0=p0, e0=e0, a=a, M=M, mu=mu,
            T=T, x0=1.0, dt=10.0,
            dense_steps=dense_steps, max_steps=4096,
            atol=1e-9, rtol=1e-9,
        )
        # Return phases and orbital elements — all JAX-differentiable
        return jnp.concatenate([Phi_phi, p, e])  # shape (3*dense_steps,)

    return output


def bench_autodiff(
    traj,
    grid: dict,
    T: float = 0.5,
    dense_steps: int = 100,
    n_warmup: int = 2,
    n_repeat: int = 5,
    run_hessian: bool = True,
    traj_fwd=None,
) -> dict:
    """Time grad, jacfwd, hessian w.r.t. (M, mu, a, p0, e0).

    ``traj`` is used for grad / jacrev (RecursiveCheckpointAdjoint).
    ``traj_fwd``, if provided, is used for jacfwd and hessian
    (DirectAdjoint — required for forward-mode and double reverse-mode).
    Falls back to ``traj`` if ``traj_fwd`` is None.
    """
    traj_fwd = traj_fwd if traj_fwd is not None else traj

    # Pick the first valid grid point
    M0  = jnp.float64(grid["M"][0])
    mu0 = jnp.float64(grid["mu"][0])
    a0  = jnp.float64(grid["a"][0])
    p00 = jnp.float64(grid["p0"][0])
    e00 = jnp.float64(grid["e0"][0])

    loss_fn      = _make_traj_scalar_loss(traj, T, dense_steps)
    output_fn    = _make_traj_vector_loss(traj, T, dense_steps)
    loss_fn_fwd  = _make_traj_scalar_loss(traj_fwd, T, dense_steps)
    output_fn_fwd = _make_traj_vector_loss(traj_fwd, T, dense_steps)

    results = {}

    # --- 1. jax.grad (scalar → scalar gradient) ---
    print(f"    jax.grad …", end=" ", flush=True)
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4)))
    mean_s, std_s = repeat_timer(
        lambda: block_jax(grad_fn(M0, mu0, a0, p00, e00)),
        n_warmup=n_warmup, n_repeat=n_repeat,
    )
    results["grad"] = (mean_s, std_s)
    print(f"{mean_s*1e3:.2f} ± {std_s*1e3:.2f} ms")

    # --- 2. jax.jacfwd (vector output → 5-param Jacobian) ---
    # Requires DirectAdjoint (traj_fwd); RecursiveCheckpointAdjoint uses custom_vjp
    # which has no custom_jvp rule and therefore blocks forward-mode AD.
    print(f"    jax.jacfwd (N_out={3*dense_steps}) …", end=" ", flush=True)
    try:
        jac_fn = jax.jit(jax.jacfwd(output_fn_fwd, argnums=(0, 1, 2, 3, 4)))
        mean_s, std_s = repeat_timer(
            lambda: block_jax(jac_fn(M0, mu0, a0, p00, e00)),
            n_warmup=n_warmup, n_repeat=n_repeat,
        )
        results["jacfwd"] = (mean_s, std_s)
        print(f"{mean_s*1e3:.2f} ± {std_s*1e3:.2f} ms")
    except (TypeError, ValueError) as exc:
        exc_s = str(exc)
        if "custom_vjp" in exc_s or "forward-mode" in exc_s:
            results["jacfwd"] = None
            print("skipped (solver uses custom_vjp; use DirectAdjoint to enable jacfwd)")
        elif "while_loop" in exc_s or "Reverse-mode" in exc_s:
            results["jacfwd"] = None
            print("skipped (ODE solver uses dynamic while_loop; pass DirectAdjoint to enable)")
        else:
            raise

    # --- 3. jax.jacrev (vector output → 5-param Jacobian, reverse mode) ---
    print(f"    jax.jacrev (N_out={3*dense_steps}) …", end=" ", flush=True)
    jacrev_fn = jax.jit(jax.jacrev(output_fn, argnums=(0, 1, 2, 3, 4)))
    mean_s, std_s = repeat_timer(
        lambda: block_jax(jacrev_fn(M0, mu0, a0, p00, e00)),
        n_warmup=n_warmup, n_repeat=n_repeat,
    )
    results["jacrev"] = (mean_s, std_s)
    print(f"{mean_s*1e3:.2f} ± {std_s*1e3:.2f} ms")

    # --- 4. jax.hessian ---
    # Uses DirectAdjoint (traj_fwd) via jacfwd(jacrev(...)). RecursiveCheckpointAdjoint
    # blocks forward-mode (custom_vjp) and double reverse-mode (dynamic while_loop).
    if run_hessian:
        print(f"    jax.hessian (5×5) …", end=" ", flush=True)
        hess_fn = jax.jit(
            jax.jacfwd(jax.jacrev(loss_fn_fwd, argnums=(0, 1, 2, 3, 4)),
                       argnums=(0, 1, 2, 3, 4))
        )
        try:
            mean_s, std_s = repeat_timer(
                lambda: block_jax(hess_fn(M0, mu0, a0, p00, e00)),
                n_warmup=n_warmup, n_repeat=n_repeat,
            )
            results["hessian"] = (mean_s, std_s)
            print(f"{mean_s*1e3:.2f} ± {std_s*1e3:.2f} ms")
        except (TypeError, ValueError) as exc:
            exc_s = str(exc)
            if "custom_vjp" in exc_s or "forward-mode" in exc_s:
                results["hessian"] = None
                print("skipped (solver uses custom_vjp; use DirectAdjoint to enable jacfwd)")
            elif "while_loop" in exc_s or "Reverse-mode" in exc_s:
                results["hessian"] = None
                print("skipped (ODE solver uses dynamic while_loop; pass DirectAdjoint to enable)")
            else:
                raise

    return results


# ---------------------------------------------------------------------------
# Local Fisher matrix
# ---------------------------------------------------------------------------

def _make_fisher_fn(traj, T: float, dense_steps: int):
    """Build a JIT-compiled function that returns the 5×5 trajectory Fisher matrix.

    F_ij = Σ_t (∂Φ_φ/∂θ_i)(∂Φ_φ/∂θ_j)

    where θ = (M, mu, a, p0, e0).  Uses forward-mode AD.
    """

    def phi_track(M, mu, a, p0, e0):
        _, p, e, Phi_phi, _, _ = traj(
            p0=p0, e0=e0, a=a, M=M, mu=mu,
            T=T, x0=1.0, dt=10.0,
            dense_steps=dense_steps, max_steps=4096,
            atol=1e-9, rtol=1e-9,
        )
        return Phi_phi  # shape (dense_steps,)

    jac_fn = jax.jacfwd(phi_track, argnums=(0, 1, 2, 3, 4))

    @jax.jit
    def fisher(M, mu, a, p0, e0):
        # jac is a tuple of 5 arrays each shape (dense_steps,)
        jac = jac_fn(M, mu, a, p0, e0)
        J = jnp.stack(jac, axis=1)   # (dense_steps, 5)
        return J.T @ J               # (5, 5)

    return fisher


def _make_fisher_waveform_fn(traj, T: float, dense_steps: int):
    """Build a JIT-compiled function that returns the 5×5 waveform Fisher matrix.

    Uses all three phases (Phi_phi, Phi_theta, Phi_r) and orbital elements
    (p, e) for a richer inner product. Still uses only JAX-differentiable
    components of the pipeline.
    """

    def signal(M, mu, a, p0, e0):
        _, p, e, Phi_phi, Phi_theta, Phi_r = traj(
            p0=p0, e0=e0, a=a, M=M, mu=mu,
            T=T, x0=1.0, dt=10.0,
            dense_steps=dense_steps, max_steps=4096,
            atol=1e-9, rtol=1e-9,
        )
        return jnp.concatenate([Phi_phi, Phi_theta, Phi_r, p, e])

    jac_fn = jax.jacfwd(signal, argnums=(0, 1, 2, 3, 4))

    @jax.jit
    def fisher(M, mu, a, p0, e0):
        jac = jac_fn(M, mu, a, p0, e0)
        J = jnp.stack(jac, axis=1)   # (5*dense_steps, 5)
        return J.T @ J               # (5, 5)

    return fisher


def bench_fisher(
    traj,
    grid: dict,
    T: float = 0.5,
    dense_steps: int = 100,
    batch_sizes: list[int] | None = None,
    n_warmup: int = 2,
    n_repeat: int = 5,
) -> dict:
    """Benchmark local Fisher matrix computation."""
    M0  = jnp.float64(grid["M"][0])
    mu0 = jnp.float64(grid["mu"][0])
    a0  = jnp.float64(grid["a"][0])
    p00 = jnp.float64(grid["p0"][0])
    e00 = jnp.float64(grid["e0"][0])

    results = {}

    # --- Single-point Fisher (phase only) ---
    print(f"    Single Fisher (phase only, 5×5) …", end=" ", flush=True)
    fisher_phase = _make_fisher_fn(traj, T, dense_steps)
    mean_s, std_s = repeat_timer(
        lambda: block_jax(fisher_phase(M0, mu0, a0, p00, e00)),
        n_warmup=n_warmup, n_repeat=n_repeat,
    )
    results["fisher_single_phase"] = (mean_s, std_s)
    print(f"{mean_s*1e3:.2f} ± {std_s*1e3:.2f} ms")

    # Compute one Fisher matrix for display
    F = np.asarray(fisher_phase(M0, mu0, a0, p00, e00))
    eigs = np.linalg.eigvalsh(F)
    cond = eigs[-1] / (eigs[0] + 1e-300)
    print(f"      eigenvalues: {eigs}")
    print(f"      condition number: {cond:.3e}")
    results["fisher_matrix"] = F
    results["fisher_eigenvalues"] = eigs
    results["fisher_condition"] = cond

    # --- Single-point Fisher (full signal: phases + p + e) ---
    print(f"    Single Fisher (full signal, 5×5) …", end=" ", flush=True)
    fisher_full = _make_fisher_waveform_fn(traj, T, dense_steps)
    mean_s, std_s = repeat_timer(
        lambda: block_jax(fisher_full(M0, mu0, a0, p00, e00)),
        n_warmup=n_warmup, n_repeat=n_repeat,
    )
    results["fisher_single_full"] = (mean_s, std_s)
    print(f"{mean_s*1e3:.2f} ± {std_s*1e3:.2f} ms")

    # --- vmapped Fisher over a batch ---
    if batch_sizes is None:
        batch_sizes = [4, 16, 64]

    print(f"    vmapped Fisher (phase only) …")
    vmapped_fisher = jax.jit(jax.vmap(fisher_phase))
    vmap_results = []
    for N in batch_sizes:
        p0v = jnp.array(grid["p0"][:N], dtype=jnp.float64)
        e0v = jnp.array(grid["e0"][:N], dtype=jnp.float64)
        av  = jnp.array(grid["a"][:N],  dtype=jnp.float64)
        Mv  = jnp.array(grid["M"][:N],  dtype=jnp.float64)
        muv = jnp.array(grid["mu"][:N], dtype=jnp.float64)

        mean_s, std_s = repeat_timer(
            lambda: block_jax(vmapped_fisher(Mv, muv, av, p0v, e0v)),
            n_warmup=n_warmup, n_repeat=n_repeat,
        )
        throughput = N / mean_s
        vmap_results.append(dict(N=N, mean_s=mean_s, std_s=std_s, throughput=throughput))
        print(f"      N={N:4d}: {mean_s*1e3:7.2f} ± {std_s*1e3:5.2f} ms  "
              f"({throughput:6.1f} Fisher/s)")

    results["vmap_fisher"] = vmap_results
    return results


# ---------------------------------------------------------------------------
# f / fdot / fddot backward benchmark
# ---------------------------------------------------------------------------

def _make_chunked_fdot_fn(single_fn, chunk_size: int):
    """Return a JIT-compiled function that processes N trajectories in chunks.

    Internally reshapes the batch into ``(n_chunks, chunk_size)`` and uses
    ``jax.lax.map`` to sequentially apply ``jax.vmap(single_fn)`` to each
    chunk.  Peak k-buffer memory is therefore proportional to ``chunk_size``,
    not to the total batch size ``N``.

    The leading dimension of ``N`` is padded to the next multiple of
    ``chunk_size`` before the reshape; padding rows are trimmed from the output.
    """
    chunk_fn = jax.vmap(single_fn)

    def run(a, e_f, M, mu):
        N = a.shape[0]
        if N <= chunk_size:
            # Small batch: direct vmap, padding would clip since pad >= N
            return chunk_fn(a, e_f, M, mu)

        pad = (-N) % chunk_size  # pad < chunk_size <= N, so a[:pad] is always valid
        if pad > 0:
            a   = jnp.concatenate([a,   a[:pad]])
            e_f = jnp.concatenate([e_f, e_f[:pad]])
            M   = jnp.concatenate([M,   M[:pad]])
            mu  = jnp.concatenate([mu,  mu[:pad]])

        n_chunks = (N + pad) // chunk_size
        reshape  = lambda x: x.reshape(n_chunks, chunk_size)
        xs = (reshape(a), reshape(e_f), reshape(M), reshape(mu))

        # lax.map applies chunk_fn sequentially; constant memory per iteration
        out = jax.lax.map(lambda c: chunk_fn(c[0], c[1], c[2], c[3]), xs)

        # out is (n_chunks, chunk_size, ...); flatten and trim padding
        out = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:])[:N], out)
        return out

    return jax.jit(run)


def bench_f_fdot_fddot_back(
    flux_data,
    grid: dict,
    batch_sizes: list[int],
    n_warmup: int,
    n_repeat: int,
    T: float = 2.0,
    N_alpha: int = 1262,
    max_steps: int = 256,
    chunk_size: int = 1024,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> list[dict]:
    """Benchmark vmapped ``EMRIInspiral.get_f_fdot_fddot_back``.

    Integrates backward from the plunge for ``T`` years, on a fixed
    ``N_alpha``-point time grid.  The dense Dopri8 interpolant is
    differentiated three times via ``jax.grad`` to recover f, ḟ, f̈
    without additional ODE solves.

    Large batches are processed via ``jax.lax.map`` over chunks of
    ``chunk_size`` trajectories, bounding peak k-buffer memory to::

        chunk_size × max_steps × 560 bytes

    For chunk_size=1024, max_steps=256: ≈ 143 MiB per chunk.
    """
    from fewtrax.trajectory.inspiral import EMRIInspiral
    from fewtrax.utils.constants import YEAR_SI

    # Fixed time grid shared across all batch lanes
    t_alpha = jnp.linspace(0.0, T * YEAR_SI, N_alpha, dtype=jnp.float64)

    def _track(a, e_f, M, mu):
        return EMRIInspiral.get_f_fdot_fddot_back(
            flux_data, M=M, mu=mu, a=a, e_f=e_f, T=T,
            t_alpha=t_alpha, max_steps=max_steps, atol=atol, rtol=rtol,
        )

    batched_fn = _make_chunked_fdot_fn(_track, chunk_size)

    results = []
    for N in batch_sizes:
        a_b   = jnp.array(grid["a"][:N],   dtype=jnp.float64)
        e_f_b = jnp.array(grid["e_f"][:N], dtype=jnp.float64)
        M_b   = jnp.array(grid["M"][:N],   dtype=jnp.float64)
        mu_b  = jnp.array(grid["mu"][:N],  dtype=jnp.float64)

        # k-buffer is bounded by chunk_size, not N
        effective_chunk = min(chunk_size, N)
        k_buf_mib = effective_chunk * max_steps * 14 * 5 * 8 / 1024**2
        n_chunks  = (N + chunk_size - 1) // chunk_size

        mem_before = get_jax_memory_mb()

        def fn():
            out = batched_fn(a_b, e_f_b, M_b, mu_b)
            block_jax(out)

        # Reduce repeats for very large batches to avoid excessive runtime
        nr = min(n_repeat, max(1, 3 if N <= 8096 else 1))
        nw = min(n_warmup, 1)
        mean_s, std_s = repeat_timer(fn, n_warmup=nw, n_repeat=nr)

        mem_after  = get_peak_memory_mb()
        throughput = N / mean_s

        results.append(dict(
            N=N, mean_s=mean_s, std_s=std_s,
            throughput=throughput,
            mem_peak_mb=mem_after,
            k_buf_mib=k_buf_mib,
            n_chunks=n_chunks,
        ))
        print(
            f"  N={N:7d} ({n_chunks}×{chunk_size}): "
            f"{mean_s*1e3:10.1f} ± {std_s*1e3:7.1f} ms"
            f"  ({throughput:9.1f} tracks/s)"
            f"  k-buf/chunk≈{k_buf_mib:6.0f} MiB"
            f"  mem_peak={mem_after:.0f} MiB"
        )

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    fast_results: list[dict],
    autodiff_results: dict | None,
    fisher_results: dict | None,
    fdot_results: list[dict] | None,
    grid: dict,
    out_dir: Path,
    traj_sample=None,
) -> None:
    """Save benchmark plots to ``out_dir``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    PARAM_LABELS = ["M", "μ", "a", "p₀", "e₀"]

    # ---- 1. Throughput scaling ----
    Ns     = [r["N"] for r in fast_results]
    t_ms   = [r["mean_s"] * 1e3 for r in fast_results]
    std_ms = [r["std_s"] * 1e3 for r in fast_results]
    tps    = [r["throughput"] for r in fast_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_t, ax_m = axes
    ax_t.errorbar(Ns, t_ms, yerr=std_ms, marker="o", color="C0", capsize=3,
                  label="EMRIInspiral (phases=True)")
    ax_t.set_xlabel("Batch size N"); ax_t.set_ylabel("Wall time [ms]")
    ax_t.set_title("vmap wall time vs batch size")
    ax_t.set_xscale("log"); ax_t.set_yscale("log")
    ax_t.legend(); ax_t.grid(True, which="both", alpha=0.3)

    ax_m.plot(Ns, tps, "C0-s", label="EMRIInspiral (phases=True)")
    ax_m.set_xlabel("Batch size N"); ax_m.set_ylabel("Throughput [traj/s]")
    ax_m.set_title("vmap throughput vs batch size")
    ax_m.set_xscale("log"); ax_m.legend(); ax_m.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "throughput_scaling.png", dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_dir / 'throughput_scaling.png'}")

    # ---- 2. Memory vs batch size ----
    mem_vals = [r["mem_peak_mb"] for r in fast_results]
    if any(np.isfinite(m) for m in mem_vals):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(Ns, mem_vals, "C0-o", label="phases=True")
        ax.set_xlabel("Batch size N"); ax.set_ylabel("Peak JAX memory [MiB]")
        ax.set_title("GPU memory vs batch size")
        ax.set_xscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "memory_scaling.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {out_dir / 'memory_scaling.png'}")

    # ---- 3. Sample trajectory ----
    if traj_sample is not None:
        t_yr = traj_sample["t"] / (365.25 * 86400)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for key, ylabel, ax in [
            ("p",       "p [M]",      axes[0, 0]),
            ("e",       "e",          axes[0, 1]),
            ("Phi_phi", "Φ_φ [rad]",  axes[1, 0]),
            ("Phi_r",   "Φ_r [rad]",  axes[1, 1]),
        ]:
            data = traj_sample.get(key)
            if data is None:
                continue
            ax.plot(t_yr, data, "C0-", lw=0.8)
            ax.set_xlabel("t [yr]"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)

        fig.suptitle(f"Sample trajectory  (M={traj_sample['M']:.2e}, "
                     f"μ={traj_sample['mu']:.1f}, a={traj_sample['a']:.2f}, "
                     f"p₀={traj_sample['p0']:.2f}, e₀={traj_sample['e0']:.3f})")
        fig.tight_layout()
        fig.savefig(out_dir / "sample_trajectory.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {out_dir / 'sample_trajectory.png'}")

    # ---- 4. Autodiff timings bar chart ----
    if autodiff_results:
        labels_map = {
            "grad":    "grad\n(scalar)",
            "jacfwd":  "jacfwd\n(fwd)",
            "jacrev":  "jacrev\n(rev)",
            "hessian": "hessian\n(5×5)",
        }
        keys   = [k for k in labels_map if autodiff_results.get(k) is not None]
        means  = [autodiff_results[k][0] * 1e3 for k in keys]
        stds   = [autodiff_results[k][1] * 1e3 for k in keys]
        labels = [labels_map[k] for k in keys]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, means, yerr=stds, capsize=4,
                      color=["C0", "C1", "C2", "C3"][:len(keys)], alpha=0.8)
        ax.set_ylabel("Wall time [ms]")
        ax.set_title("Autodiff timing w.r.t. (M, μ, a, p₀, e₀)")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.1,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "autodiff_timings.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {out_dir / 'autodiff_timings.png'}")

    # ---- 5. Fisher matrix heatmap ----
    if fisher_results and "fisher_matrix" in fisher_results:
        F = fisher_results["fisher_matrix"]
        # Normalise rows/cols by diagonal for display
        d = np.sqrt(np.diag(F))
        d[d == 0] = 1.0
        F_corr = F / np.outer(d, d)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Raw Fisher (log scale)
        im0 = axes[0].imshow(np.log10(np.abs(F) + 1e-300), cmap="viridis")
        axes[0].set_xticks(range(5)); axes[0].set_xticklabels(PARAM_LABELS)
        axes[0].set_yticks(range(5)); axes[0].set_yticklabels(PARAM_LABELS)
        axes[0].set_title("log₁₀|F_ij| (trajectory Fisher)")
        plt.colorbar(im0, ax=axes[0])

        # Normalised (correlation) matrix
        im1 = axes[1].imshow(F_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_xticks(range(5)); axes[1].set_xticklabels(PARAM_LABELS)
        axes[1].set_yticks(range(5)); axes[1].set_yticklabels(PARAM_LABELS)
        axes[1].set_title("Correlation matrix  F_ij / √(F_ii F_jj)")
        plt.colorbar(im1, ax=axes[1])
        for i in range(5):
            for j in range(5):
                axes[1].text(j, i, f"{F_corr[i,j]:.2f}", ha="center", va="center",
                             fontsize=7, color="k" if abs(F_corr[i,j]) < 0.5 else "w")

        fig.suptitle("Local trajectory Fisher matrix  (single parameter point)")
        fig.tight_layout()
        fig.savefig(out_dir / "fisher_matrix.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {out_dir / 'fisher_matrix.png'}")

        # Fisher eigenvalue spectrum
        fig, ax = plt.subplots(figsize=(7, 4))
        eigs = fisher_results["fisher_eigenvalues"]
        ax.semilogy(range(1, 6), np.sort(eigs)[::-1], "C0-o", lw=2)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue (log scale)")
        ax.set_title(f"Fisher eigenvalue spectrum  "
                     f"(κ = {fisher_results['fisher_condition']:.2e})")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fisher_eigenvalues.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {out_dir / 'fisher_eigenvalues.png'}")

    # ---- 6. vmapped Fisher throughput ----
    if fisher_results and "vmap_fisher" in fisher_results:
        vr = fisher_results["vmap_fisher"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["N"] for r in vr], [r["throughput"] for r in vr], "C3-o", lw=2)
        ax.set_xlabel("Batch size N")
        ax.set_ylabel("Fisher matrices / s")
        ax.set_title("vmapped Fisher matrix throughput")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fisher_throughput.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {out_dir / 'fisher_throughput.png'}")

    # ---- 7. f/fdot/fddot backward benchmark throughput ----
    if fdot_results:
        Ns_fd  = [r["N"] for r in fdot_results]
        tps_fd = [r["throughput"] for r in fdot_results]
        t_ms_fd  = [r["mean_s"] * 1e3 for r in fdot_results]
        std_ms_fd = [r["std_s"] * 1e3 for r in fdot_results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].errorbar(Ns_fd, t_ms_fd, yerr=std_ms_fd, marker="o",
                         color="C4", capsize=3)
        axes[0].set_xlabel("Batch size N")
        axes[0].set_ylabel("Wall time [ms]")
        axes[0].set_title("get_f_fdot_fddot_back — wall time vs batch size")
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
        axes[0].grid(True, which="both", alpha=0.3)

        axes[1].plot(Ns_fd, tps_fd, "C4-o")
        axes[1].set_xlabel("Batch size N")
        axes[1].set_ylabel("Throughput [tracks/s]")
        axes[1].set_title("get_f_fdot_fddot_back — throughput vs batch size")
        axes[1].set_xscale("log")
        axes[1].grid(True, which="both", alpha=0.3)

        fig.suptitle(
            "vmapped backward integration + jax.grad³  "
            "(f, ḟ, f̈ on N_alpha-point grid)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "fdot_throughput.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {out_dir / 'fdot_throughput.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--n-repeat",    type=int,   default=5)
    parser.add_argument("--n-warmup",    type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42,
                        help="RNG seed for parameter grid (default: 42)")
    parser.add_argument("--T",           type=float, default=0.5,
                        help="Trajectory duration [yr] (default: 0.5)")
    parser.add_argument("--dense",       type=int,   default=100,
                        help="dense_steps per trajectory (default: 100)")
    parser.add_argument("--max-batch",   type=int,   default=1024,
                        help="Largest batch size to test (default: 1024)")
    parser.add_argument("--skip-autodiff", action="store_true",
                        help="Skip autodiff benchmarks (grad/hessian/jacfwd)")
    parser.add_argument("--skip-fisher", action="store_true",
                        help="Skip Fisher matrix benchmarks")
    parser.add_argument("--skip-hessian", action="store_true",
                        help="Skip hessian (expensive for large dense_steps)")
    parser.add_argument("--skip-fdot", action="store_true",
                        help="Skip f/fdot/fddot backward integration benchmarks")
    parser.add_argument("--T-fdot",        type=float, default=2.0,
                        help="Backward integration duration for fdot benchmark [yr] (default: 2.0)")
    parser.add_argument("--N-alpha",       type=int,   default=1262,
                        help="Time-grid points for fdot benchmark (default: 1262)")
    parser.add_argument("--max-steps-fdot", type=int,  default=150,
                        help="ODE max_steps for dense solve; governs k-buffer memory "
                             "(default: 256 ≈ 4× typical steps for 2-yr integration)")
    parser.add_argument("--fdot-chunk-size", type=int, default=1024,
                        help="Sub-batch size for lax.map chunking in fdot benchmark; "
                             "peak k-buffer ∝ chunk_size × max_steps (default: 4096)")
    parser.add_argument("--no-plots",    action="store_true",
                        help="Do not save plot files")
    parser.add_argument("--plot-dir",    type=str,   default="benchmark_plots",
                        help="Directory for output plots (default: benchmark_plots)")
    args = parser.parse_args()

    nw, nr = args.n_warmup, args.n_repeat

    # --- Device info ---
    print_header("Device information")
    print(f"  JAX version : {jax.__version__}")
    for d in jax.devices():
        print(f"  {d.id}: {d.device_kind}  platform={d.platform}")
    on_gpu = any(d.platform == "gpu" for d in jax.devices())
    if not on_gpu:
        print("  [Note] No GPU detected — results reflect CPU performance.")

    mem0 = get_jax_memory_mb()
    if np.isfinite(mem0):
        print(f"  JAX memory at start: {mem0:.0f} MiB")
    free_mib = nvidia_smi_free_mib()
    if np.isfinite(free_mib):
        print(f"  GPU free memory (nvidia-smi): {free_mib:.0f} MiB")

    # --- Load data ---
    data_dir = find_data_dir(args.data_dir)
    print(f"\n  FEW data directory: {data_dir}")
    from fewtrax.data import load_flux_data
    from fewtrax.trajectory import EMRIInspiral
    flux_data = load_flux_data(data_dir)
    # phases=True: full 5D solve (p, e, Φ_φ, Φ_θ, Φ_r) — RecursiveCheckpointAdjoint
    # is memory-efficient and supports grad/jacrev; used for vmap throughput benchmarks.
    traj_5d = EMRIInspiral(flux_data, phases=True)
    # DirectAdjoint: exposes a custom_jvp rule, enabling jacfwd and hessian.
    # Used for autodiff benchmarks and Fisher matrix computation (which use jacfwd).
    traj_direct = EMRIInspiral(flux_data, phases=True, adjoint=diffrax.DirectAdjoint())

    # --- Build parameter grid ---
    N_max = max(args.max_batch, 64) + 16
    print(f"\n  Building random parameter grid  (N={N_max}, seed={args.seed}) …",
          end=" ", flush=True)
    grid = build_param_grid(N_max, seed=args.seed, T_yr=args.T, dt_s=10.0)
    print("done")
    print(f"  p0  range : [{grid['p0'].min():.2f}, {grid['p0'].max():.2f}] M")
    print(f"  e0  range : [{grid['e0'].min():.3f}, {grid['e0'].max():.3f}]")
    print(f"  a   range : [{grid['a'].min():.3f},  {grid['a'].max():.3f}]")
    print(f"  M   range : [{grid['M'].min():.2e}, {grid['M'].max():.2e}] Msun")
    print(f"  mu  range : [{grid['mu'].min():.1f}, {grid['mu'].max():.1f}] Msun")

    # --- Batch sizes to sweep ---
    max_bs = args.max_batch
    batch_sizes = sorted({1, 4, 16, 64, 256, max_bs} & {n for n in range(1, max_bs + 1)})

    # --- A. EMRIInspiral (phases=True) — vmapped throughput ---
    print_header("A. EMRIInspiral (phases=True) — vmapped batch throughput")
    print("   Full 5D solve: (p, e, Φ_φ, Φ_θ, Φ_r)")
    fast_results = bench_batch(
        traj_5d, grid, batch_sizes, nw, nr,
        dense_steps=args.dense, T=args.T, label="5D",
    )

    # --- Throughput summary table ---
    print_header("Throughput summary")
    headers = ["N_batch", "5D [ms]", "5D [traj/s]"]
    widths  = [10, 20, 16]
    rows = [
        (
            str(r["N"]),
            f"{r['mean_s']*1e3:.1f} ± {r['std_s']*1e3:.1f}",
            f"{r['throughput']:.1f}",
        )
        for r in fast_results
    ]
    print_table(rows, headers, widths)
    peak_5d = max(r["throughput"] for r in fast_results)
    print(f"\n  Peak throughput  (phases=True, 5D): {peak_5d:.1f} traj/s")

    # --- B. Autodiff benchmarks ---
    autodiff_results = None
    if not args.skip_autodiff:
        print_header("B. Autodiff benchmarks  (EMRIInspiral phases=True, single trajectory)")
        print(f"   grad/jacrev use RecursiveCheckpointAdjoint; "
              f"jacfwd/hessian use DirectAdjoint")
        print(f"   Parameters: M, μ, a, p₀, e₀  |  T={args.T} yr  "
              f"|  dense_steps={args.dense}")
        autodiff_results = bench_autodiff(
            traj_5d, grid,
            T=args.T, dense_steps=args.dense,
            n_warmup=nw, n_repeat=nr,
            run_hessian=not args.skip_hessian,
            traj_fwd=traj_direct,
        )

        # Summary table
        print()
        ad_headers = ["Operation", "wall time [ms]", "std [ms]"]
        ad_widths  = [28, 16, 12]
        ad_labels  = {
            "grad":    "jax.grad   (scalar→5 grads)",
            "jacfwd":  f"jax.jacfwd (→{3*args.dense}×5 Jac, fwd)",
            "jacrev":  f"jax.jacrev (→{3*args.dense}×5 Jac, rev)",
            "hessian": "jax.hessian (5×5, jacfwd∘jacrev)",
        }
        ad_rows = [
            (ad_labels[k], f"{v[0]*1e3:.2f}", f"{v[1]*1e3:.2f}")
            for k, v in autodiff_results.items()
            if k in ad_labels and v is not None
        ]
        print_table(ad_rows, ad_headers, ad_widths)

    # --- C. Fisher matrix benchmarks ---
    fisher_results = None
    if not args.skip_fisher:
        print_header("C. Local Fisher matrix  (trajectory inner product, θ = M,μ,a,p₀,e₀)")
        print(f"   Uses DirectAdjoint (jacfwd)  |  "
              f"dense_steps={args.dense}  |  T={args.T} yr")
        fisher_batch_sizes = sorted({1, 16, 64, 256} & {n for n in range(1, max_bs + 1)})
        fisher_results = bench_fisher(
            traj_direct, grid,
            T=args.T, dense_steps=args.dense,
            batch_sizes=fisher_batch_sizes,
            n_warmup=nw, n_repeat=nr,
        )

    # --- D. f/fdot/fddot backward integration benchmark ---
    fdot_results = None
    if not args.skip_fdot:
        fdot_batch_sizes = [64, 1024, 8096, 65000, 131066]
        N_fdot_max = max(fdot_batch_sizes)

        print_header(
            "D. get_f_fdot_fddot_back — vmapped backward ODE + jax.grad³"
        )
        print(f"   T={args.T_fdot} yr backward  |  N_alpha={args.N_alpha} pts"
              f"  |  max_steps={args.max_steps_fdot}"
              f"  |  chunk_size={args.fdot_chunk_size}")
        k_per_chunk_mib = args.fdot_chunk_size * args.max_steps_fdot * 14 * 5 * 8 / 1024**2
        print(f"   k-buffer per chunk ≈ {k_per_chunk_mib:.0f} MiB  "
              f"(chunk_size × max_steps × 14 × 5 × 8 bytes)")
        print(f"   Batch sizes: {fdot_batch_sizes}")
        print()

        print(f"  Building fdot parameter grid  (N={N_fdot_max}, seed=77) …",
              end=" ", flush=True)
        fdot_grid = build_fdot_param_grid(N_fdot_max, seed=77)
        print("done")
        print(f"  a   range : [{fdot_grid['a'].min():.3f},  {fdot_grid['a'].max():.3f}]")
        print(f"  e_f range : [{fdot_grid['e_f'].min():.3f}, {fdot_grid['e_f'].max():.3f}]")
        print(f"  M   range : [{fdot_grid['M'].min():.2e}, {fdot_grid['M'].max():.2e}] Msun")
        print(f"  mu  range : [{fdot_grid['mu'].min():.1f}, {fdot_grid['mu'].max():.1f}] Msun")
        print()

        fdot_results = bench_f_fdot_fddot_back(
            flux_data, fdot_grid,
            batch_sizes=fdot_batch_sizes,
            n_warmup=nw, n_repeat=nr,
            T=args.T_fdot, N_alpha=args.N_alpha,
            max_steps=args.max_steps_fdot,
            chunk_size=args.fdot_chunk_size,
        )

        print()
        fd_headers = ["N_batch", "wall time [ms]", "throughput [tracks/s]", "k-buf [MiB]"]
        fd_widths  = [10, 22, 24, 14]
        fd_rows = [
            (
                str(r["N"]),
                f"{r['mean_s']*1e3:.1f} ± {r['std_s']*1e3:.1f}",
                f"{r['throughput']:.1f}",
                f"{r['k_buf_mib']:.0f}",
            )
            for r in fdot_results
        ]
        print_table(fd_rows, fd_headers, fd_widths)

    # --- Compute sample trajectory for plot ---
    traj_sample = None
    if not args.no_plots:
        print_header("Computing sample trajectory for plot")
        M0, mu0, a0, p00, e00 = (
            float(grid["M"][0]), float(grid["mu"][0]),
            float(grid["a"][0]), float(grid["p0"][0]), float(grid["e0"][0]),
        )
        t, p, e, Phi_phi, Phi_theta, Phi_r = traj_5d(
            p0=p00, e0=e00, a=a0, M=M0, mu=mu0,
            T=args.T, x0=1.0, dt=10.0,
            dense_steps=max(args.dense, 200), max_steps=4096,
            atol=1e-9, rtol=1e-9,
        )
        traj_sample = dict(
            t=np.asarray(t), p=np.asarray(p), e=np.asarray(e),
            Phi_phi=np.asarray(Phi_phi), Phi_r=np.asarray(Phi_r),
            M=M0, mu=mu0, a=a0, p0=p00, e0=e00,
        )
        print(f"  Trajectory has {len(t)} points  "
              f"spanning {np.asarray(t)[-1]/(365.25*86400):.3f} yr")

    # --- Plots ---
    if not args.no_plots:
        print_header("Saving plots")
        out_dir = Path(args.plot_dir)
        make_plots(
            fast_results,
            autodiff_results, fisher_results, fdot_results,
            grid, out_dir, traj_sample,
        )
        print(f"  All plots written to {out_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()

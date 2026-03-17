#!/usr/bin/env python3
"""Profile GPU memory and wall-clock time for vmap-batched fewtrax operations.

This script is NOT a pytest test and will not be collected by the default
test run (filename does not match the ``test_*.py`` pattern).  Run it
directly::

    python tests/profile_vmap_memory.py
    python tests/profile_vmap_memory.py --data-dir /path/to/few/data
    python tests/profile_vmap_memory.py --max-batch 100000 --dense-steps 200

Environment
-----------
FEW_DATA_DIR
    Path to the directory containing ``KerrEccEqFluxData.h5``.

What is measured
----------------
For each operation and batch size the script reports:

bytes_before / bytes_after
    Live device memory immediately before and after the JIT-compiled call
    (after ``jax.block_until_ready``).  The delta equals the size of the
    returned arrays; intermediate buffers are freed by XLA before this
    point.

output_MB
    ``bytes_after - bytes_before``, i.e. the persistent output allocation
    for this batch.  Scales linearly with ``batch_size × dense_steps``.

time_ms / us_per_sample
    Wall-clock time (mean over ``--repeats`` post-JIT calls) and its
    per-sample breakdown.

Note: *peak intermediate* memory (XLA temporaries during the ODE solve)
is not captured here.  For that, wrap a call in ``jax.profiler.trace``
and inspect the resulting profile with TensorBoard.  Peak intermediate
memory is typically 2–5× the output size for trajectory integration.

Operations profiled
-------------------
1. ``vmap(trajectory.__call__)``          — (p, e, phases) arrays
2. ``vmap(trajectory.get_frequency_track)`` — single harmonic (l,m,k,n)
3. ``vmap(jax.grad(loss))``               — gradient of frequency MSE loss
                                             w.r.t. (M, mu, p0, e0)
4. ``AmplitudeInterpolator.evaluate``     — NumPy/CPU baseline, not vmappable;
                                             reported for comparison only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

MB = 1024 ** 2
GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Utility: device memory
# ---------------------------------------------------------------------------

def _memory_stats() -> dict | None:
    """Return memory stats for the first device, or None if unavailable."""
    try:
        return jax.devices()[0].memory_stats()
    except Exception:
        return None


def _bytes_in_use() -> int | None:
    stats = _memory_stats()
    return stats.get("bytes_in_use") if stats else None


def _device_label() -> str:
    devs = jax.devices()
    return str(devs[0]) if devs else "unknown"


def _device_total_gb() -> float | None:
    stats = _memory_stats()
    if stats is None:
        return None
    total = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
    return total / GB if total else None


# ---------------------------------------------------------------------------
# Utility: timing
# ---------------------------------------------------------------------------

def _warmup_and_time(fn: Callable, n_repeats: int = 5) -> tuple[float, int | None, int | None]:
    """JIT-compile fn, then time it.

    Returns
    -------
    mean_ms : float
        Mean wall-clock time per call [ms].
    bytes_before : int or None
        Live device bytes immediately before the timed calls start.
    bytes_after : int or None
        Live device bytes immediately after the last timed call.
    """
    # Warm-up: trigger JIT compilation (not timed)
    result = fn()
    jax.block_until_ready(result)

    bytes_before = _bytes_in_use()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append((time.perf_counter() - t0) * 1e3)

    bytes_after = _bytes_in_use()
    return float(np.mean(times)), bytes_before, bytes_after


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

_COL = 72

def _header(title: str) -> None:
    print(f"\n{'=' * _COL}")
    print(f"  {title}")
    print(f"{'=' * _COL}")


def _print_row(
    batch: int,
    mean_ms: float,
    bytes_before: int | None,
    bytes_after: int | None,
    total_gb: float | None,
) -> None:
    us = mean_ms * 1e3 / batch
    if bytes_before is not None and bytes_after is not None:
        delta_mb = (bytes_after - bytes_before) / MB
        after_mb = bytes_after / MB
        pct = 100 * (bytes_after - bytes_before) / (total_gb * GB) if total_gb else float("nan")
        mem_str = f"  out={delta_mb:+8.1f} MB  live={after_mb:7.1f} MB  ({pct:.2f}% of device)"
    else:
        mem_str = "  (no GPU memory stats)"
    print(f"  batch={batch:>8}  {mean_ms:7.1f} ms  {us:7.2f} µs/sample{mem_str}")


# ---------------------------------------------------------------------------
# Profiling routines
# ---------------------------------------------------------------------------

def profile_trajectory(traj, T, M, mu, dense_steps, batch_sizes, n_repeats, total_gb):
    _header(f"vmap( trajectory.__call__ )   T={T:.1f} yr  dense_steps={dense_steps}")
    print(f"  Output per sample: {5 * dense_steps * 8 / 1024:.1f} kB  "
          f"(5 arrays × {dense_steps} points × float64)")

    def _make_fn(p0_arr, e0_arr):
        @jax.jit
        def _fn():
            return jax.vmap(
                lambda p0, e0: traj(p0=p0, e0=e0, T=T, M=M, mu=mu, dense_steps=dense_steps)
            )(p0_arr, e0_arr)
        return _fn

    rng = np.random.default_rng(42)
    for bs in batch_sizes:
        p0_arr = jnp.array(rng.uniform(8.0, 14.0, bs))
        e0_arr = jnp.array(rng.uniform(0.05, 0.5, bs))
        fn = _make_fn(p0_arr, e0_arr)
        mean_ms, b_before, b_after = _warmup_and_time(fn, n_repeats)
        _print_row(bs, mean_ms, b_before, b_after, total_gb)


def profile_frequency_track(traj, T, M, mu, dense_steps, batch_sizes, n_repeats, total_gb,
                             l=2, m=2, k=0, n=0):
    _header(
        f"vmap( get_frequency_track )   mode=({l},{m},{k},{n})  "
        f"T={T:.1f} yr  dense_steps={dense_steps}"
    )
    print(f"  Output per sample: {2 * dense_steps * 8 / 1024:.1f} kB  "
          f"(t + f arrays × {dense_steps} points × float64)")

    def _make_fn(p0_arr, e0_arr):
        @jax.jit
        def _fn():
            return jax.vmap(
                lambda p0, e0: traj.get_frequency_track(
                    p0=p0, e0=e0, T=T, M=M, mu=mu,
                    l=l, m=m, k=k, n=n, dense_steps=dense_steps,
                )
            )(p0_arr, e0_arr)
        return _fn

    rng = np.random.default_rng(42)
    for bs in batch_sizes:
        p0_arr = jnp.array(rng.uniform(8.0, 14.0, bs))
        e0_arr = jnp.array(rng.uniform(0.05, 0.5, bs))
        fn = _make_fn(p0_arr, e0_arr)
        mean_ms, b_before, b_after = _warmup_and_time(fn, n_repeats)
        _print_row(bs, mean_ms, b_before, b_after, total_gb)


def profile_gradient(traj, T, M, mu, dense_steps, batch_sizes, n_repeats, total_gb,
                     l=2, m=2, k=0, n=0):
    _header(
        f"vmap( grad(loss) )   mode=({l},{m},{k},{n})  "
        f"T={T:.1f} yr  dense_steps={dense_steps}"
    )
    print("  Free params: (M, mu, p0, e0) — 4 scalars.  Spin a held fixed.")
    print("  Loss: frequency MSE residual (σ_f = 1e-5 Hz, STFT bin width).")
    print("  Gradient via diffrax continuous adjoint: O(state_dim) memory,")
    print("  not O(trajectory_length).")

    # Build a synthetic observed track at fixed reference parameters
    t_ref, f_ref = traj.get_frequency_track(
        p0=10.0, e0=0.4, T=T, M=M, mu=mu,
        l=l, m=m, k=k, n=n, dense_steps=dense_steps,
    )
    jax.block_until_ready((t_ref, f_ref))
    t_obs = jnp.array(t_ref)
    sigma_f = 1e-5
    f_obs = jnp.array(np.asarray(f_ref) + np.random.default_rng(0).normal(0, sigma_f, dense_steps))

    def loss(theta):
        """Frequency MSE loss for a single set of parameters."""
        M_, mu_, p0_, e0_ = theta[0], theta[1], theta[2], theta[3]
        t_pred, f_pred = traj.get_frequency_track(
            p0=p0_, e0=e0_, T=T, M=M_, mu=mu_,
            l=l, m=m, k=k, n=n, dense_steps=dense_steps,
        )
        f_interp = jnp.interp(t_obs, t_pred, f_pred)
        return jnp.sum((f_interp - f_obs) ** 2) / sigma_f ** 2

    grad_loss = jax.grad(loss)

    # Time a single gradient call for reference
    theta_ref = jnp.array([M, mu, 10.0, 0.4])
    t0 = time.perf_counter()
    _ = grad_loss(theta_ref)
    jax.block_until_ready(_)
    compile_ms = (time.perf_counter() - t0) * 1e3
    single_ms, _, _ = _warmup_and_time(lambda: grad_loss(theta_ref), n_repeats)
    print(f"\n  Single-sample compile+first call: {compile_ms:.0f} ms")
    print(f"  Single-sample post-JIT:           {single_ms:.2f} ms\n")

    def _make_fn(theta_batch):
        @jax.jit
        def _fn():
            return jax.vmap(grad_loss)(theta_batch)
        return _fn

    rng = np.random.default_rng(7)
    # Gradient is more expensive; limit to batch_sizes <= 10_000
    grad_batch_sizes = [bs for bs in batch_sizes if bs <= 10_000]
    for bs in grad_batch_sizes:
        theta_batch = jnp.array(
            np.column_stack([
                rng.uniform(8e5, 1.2e6, bs),   # M [Msun]
                rng.uniform(8.0, 12.0, bs),     # mu [Msun]
                rng.uniform(8.0, 12.0, bs),     # p0 [M]
                rng.uniform(0.2, 0.6, bs),      # e0
            ])
        )
        fn = _make_fn(theta_batch)
        mean_ms, b_before, b_after = _warmup_and_time(fn, n_repeats)
        _print_row(bs, mean_ms, b_before, b_after, total_gb)

    if any(bs > 10_000 for bs in batch_sizes):
        skipped = [bs for bs in batch_sizes if bs > 10_000]
        print(f"  Skipped batch sizes {skipped} for gradient profiling "
              f"(XLA compile time grows steeply; run manually if needed).")


def profile_amplitude(data_dir: str, dense_steps: int) -> None:
    _header("AmplitudeInterpolator.evaluate   [CPU / NumPy, NOT vmappable]")
    print("  The amplitude interpolator is implemented in NumPy and runs on CPU.")
    print("  It cannot be dispatched to the GPU or wrapped with jax.vmap.")
    print("  Re-implementing it in JAX via interpax is required for GPU batching.")

    try:
        from fewtrax.data import load_amplitude_data
        from fewtrax.amplitude import AmplitudeInterpolator

        amp_data = load_amplitude_data(data_dir)
        amp_interp = AmplitudeInterpolator(amp_data)
        p_test = np.full(dense_steps, 10.0)
        e_test = np.full(dense_steps, 0.4)

        # Warm-up
        _ = amp_interp.evaluate(p_test, e_test)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            amp_interp.evaluate(p_test, e_test)
            times.append((time.perf_counter() - t0) * 1e3)
        single_ms = float(np.mean(times))

        n_modes = amp_interp.n_modes
        print(f"\n  dense_steps={dense_steps}, n_modes={n_modes}")
        print(f"  Single call (CPU): {single_ms:.1f} ms")
        print(f"  Serial cost for batch sizes:")
        for bs in [10, 100, 1_000, 10_000, 100_000]:
            print(f"    batch={bs:>8}: {single_ms * bs / 1e3:8.1f} s  (serial CPU)")

    except Exception as exc:
        print(f"\n  Could not load amplitude data: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile vmap memory and timing for fewtrax on GPU/CPU."
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to FEW data directory (default: auto-detected via "
             "FEW_DATA_DIR env var or ~/.fewtrax/data).",
    )
    parser.add_argument(
        "--max-batch", type=int, default=100_000,
        help="Largest batch size to test (default: 100_000). "
             "Powers of 10 from 1 up to this value are tested.",
    )
    parser.add_argument(
        "--dense-steps", type=int, default=100,
        help="Number of output trajectory points (static shape, default: 100). "
             "Changing this value triggers JAX recompilation.",
    )
    parser.add_argument(
        "--obs-time", type=float, default=1.0,
        help="Observation duration T [years] (default: 1.0).",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Number of timed repetitions per measurement (default: 5).",
    )
    parser.add_argument(
        "--no-gradient", action="store_true",
        help="Skip gradient profiling (faster run for trajectory/freq only).",
    )
    parser.add_argument(
        "--no-amplitude", action="store_true",
        help="Skip amplitude interpolator baseline.",
    )
    args = parser.parse_args()

    # Locate data
    data_dir = args.data_dir
    if data_dir is None:
        env = os.environ.get("FEW_DATA_DIR")
        if env and (Path(env) / "KerrEccEqFluxData.h5").exists():
            data_dir = env
        else:
            try:
                from few.utils.globals import get_file_manager
                fp = get_file_manager().get_file("KerrEccEqFluxData.h5", raise_on_error=False)
                if fp:
                    data_dir = str(Path(fp).parent)
            except Exception:
                pass
        if data_dir is None:
            default = Path.home() / ".fewtrax" / "data"
            if (default / "KerrEccEqFluxData.h5").exists():
                data_dir = str(default)
    if data_dir is None:
        print(
            "ERROR: FEW data directory not found.\n"
            "Set FEW_DATA_DIR or pass --data-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Batch sizes: powers of 10 up to max_batch
    batch_sizes = []
    bs = 1
    while bs <= args.max_batch:
        batch_sizes.append(bs)
        bs *= 10

    # Setup
    from fewtrax.data import load_flux_data
    from fewtrax.trajectory import EMRIInspiral

    print(f"\nDevice : {_device_label()}")
    total_gb = _device_total_gb()
    if total_gb:
        print(f"Memory : {total_gb:.1f} GB")
    else:
        print("Memory : unavailable (CPU build or stats not supported)")
        print("         Timing results will still be reported.")
    print(f"\nLoading flux data from: {data_dir}")
    flux_data = load_flux_data(data_dir)

    M, mu, a = 1e6, 10.0, 0.3
    T = args.obs_time
    dense_steps = args.dense_steps

    traj = EMRIInspiral(flux_data, a=a)

    # Single-call JIT baseline
    print(f"\nJIT-compiling single trajectory call (dense_steps={dense_steps}) ...")
    t0 = time.perf_counter()
    r = traj(p0=10.0, e0=0.4, T=T, M=M, mu=mu, dense_steps=dense_steps)
    jax.block_until_ready(r)
    compile_ms = (time.perf_counter() - t0) * 1e3
    single_ms, _, _ = _warmup_and_time(
        lambda: traj(p0=10.0, e0=0.4, T=T, M=M, mu=mu, dense_steps=dense_steps),
        args.repeats,
    )
    print(f"  compile + first call : {compile_ms:.0f} ms")
    print(f"  post-JIT single call : {single_ms:.2f} ms")

    # Run profiles
    profile_trajectory(traj, T, M, mu, dense_steps, batch_sizes, args.repeats, total_gb)
    profile_frequency_track(traj, T, M, mu, dense_steps, batch_sizes, args.repeats, total_gb)

    if not args.no_gradient:
        profile_gradient(traj, T, M, mu, dense_steps, batch_sizes, args.repeats, total_gb)

    if not args.no_amplitude:
        profile_amplitude(data_dir, dense_steps)

    # Summary table
    _header("Notes on A100 (80 GB) context")
    print("  Output memory scales as: batch × dense_steps × dtype_bytes")
    print(f"    trajectory   : batch × {dense_steps} × 5 × 8 = "
          f"{dense_steps * 5 * 8 / 1024:.1f} kB/sample")
    print(f"    freq_track   : batch × {dense_steps} × 2 × 8 = "
          f"{dense_steps * 2 * 8 / 1024:.1f} kB/sample")
    print()
    print("  Peak *intermediate* memory (XLA temporaries) is typically 2–5×")
    print("  the output size and is NOT captured above.  To measure it, wrap")
    print("  a single call in jax.profiler.trace and inspect with TensorBoard:")
    print()
    print("    with jax.profiler.trace('/tmp/jax-profile', create_perfetto_link=True):")
    print("        result = vmapped_fn(...)")
    print("        jax.block_until_ready(result)")
    print()
    print("  AmplitudeInterpolator is NumPy-only (CPU).  At ~N ms/call, batched")
    print("  amplitude evaluation requires a JAX/interpax re-implementation")
    print("  before GPU vmap is possible for full-waveform generation.")
    print()


if __name__ == "__main__":
    main()

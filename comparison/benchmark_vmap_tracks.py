"""Batched vmap trajectory benchmark — the core fewtrax use case.

This is the main performance target of the package: computing vmapped,
individually differentiable EMRI frequency tracks (or TF-domain tracks)
over a random grid of intrinsic parameters.

What is measured
----------------
A. **Trajectory throughput** — vmapped ``EMRIInspiral`` and ``EMRIInspiralFast``
   over batches of size N drawn from a random (seeded) parameter grid.
   Reports trajectories/second and per-trajectory wall time.

B. **GPU utilisation** — reports JAX device info and peak allocated memory
   (when running on GPU).

C. **Per-trajectory differentiability check** — verifies that ``jax.jacfwd``
   through the vmapped trajectory w.r.t. (p0, e0) produces finite Jacobians
   for all valid points in the batch.

D. (optional, ``--tf``) **TF-domain track benchmark** — for each trajectory in
   a smaller sub-batch, computes analytical WDM TF tracks for the dominant
   (2, 2, 0, 1) and (2, 1, 0, 1) modes via
   :func:`~fewtrax.utils.tf_tracks.build_tf_tracks`.  Reports total time and
   memory footprint per track set.

The random parameter grid is initialised with a fixed seed (``--seed``) for
full reproducibility.  Intrinsic parameter ranges:

    M    ∈ [5e5, 5e6]  M_sun
    mu   ∈ [5,   50]   M_sun
    a    ∈ [0.05, 0.9]         (prograde equatorial)
    p0   = p_sep(a, e0) + Δp,  Δp ∈ [1, 10] M
    e0   ∈ [0.05, 0.7]

Usage
-----
    python benchmark_vmap_tracks.py [/path/to/few/data]
    python benchmark_vmap_tracks.py --n-batch 512 --n-repeat 5
    python benchmark_vmap_tracks.py --tf --tf-batch 64
    python benchmark_vmap_tracks.py --seed 1234
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import find_data_dir, block_jax, print_header, print_table, repeat_timer


# ---------------------------------------------------------------------------
# Parameter grid construction
# ---------------------------------------------------------------------------

def build_param_grid(
    N: int,
    seed: int = 0,
    T_yr: float = 0.5,
    dt_s: float = 10.0,
) -> dict[str, np.ndarray]:
    """Draw N random EMRI parameter sets with fixed ``seed``.

    Returns a dict of 1-D numpy arrays, each of length ``N``.
    All (p0, e0) pairs are guaranteed to have p0 > p_sep(a, e0).
    """
    from fewtrax.utils.geodesic import get_separatrix

    rng = np.random.default_rng(seed)

    M_arr  = 10.0 ** rng.uniform(np.log10(5e5), np.log10(5e6), N)
    mu_arr = rng.uniform(5.0, 50.0, N)
    a_arr  = rng.uniform(0.05, 0.90, N)
    e_arr  = rng.uniform(0.05, 0.70, N)
    dp_arr = rng.uniform(1.0, 10.0, N)  # excess above separatrix [M]

    p_arr = np.array([
        float(get_separatrix(jnp.float64(a), jnp.float64(e), jnp.float64(1.0))) + dp
        for a, e, dp in zip(a_arr, e_arr, dp_arr)
    ])

    return dict(
        M=M_arr, mu=mu_arr, a=a_arr, p0=p_arr, e0=e_arr,
        T=np.full(N, T_yr), dt=np.full(N, dt_s),
    )


# ---------------------------------------------------------------------------
# Vmapped trajectory runners
# ---------------------------------------------------------------------------

def make_batched_traj(traj, dense_steps: int = 100, max_steps: int = 4096,
                      atol: float = 1e-9, rtol: float = 1e-9):
    """Return a vmapped function (p0, e0, a, M, mu) -> (p_arr, e_arr)."""
    fixed = dict(
        T=0.5, x0=1.0, dt=10.0,
        dense_steps=dense_steps, max_steps=max_steps, atol=atol, rtol=rtol,
    )

    def single(p0, e0, a, M, mu):
        _, ps, es, *_ = traj(p0=p0, e0=e0, a=a, M=M, mu=mu, **fixed)
        return ps, es

    return jax.vmap(single)


def bench_batch(
    traj,
    grid: dict,
    batch_sizes: list[int],
    n_warmup: int,
    n_repeat: int,
    dense_steps: int = 100,
    label: str = "",
) -> list[dict]:
    """Benchmark vmapped trajectory at each batch size from ``grid``."""
    batched_fn = make_batched_traj(traj, dense_steps=dense_steps)

    results = []
    for N in batch_sizes:
        p0 = jnp.array(grid["p0"][:N], dtype=jnp.float64)
        e0 = jnp.array(grid["e0"][:N], dtype=jnp.float64)
        a  = jnp.array(grid["a"][:N],  dtype=jnp.float64)
        M  = jnp.array(grid["M"][:N],  dtype=jnp.float64)
        mu = jnp.array(grid["mu"][:N], dtype=jnp.float64)

        def fn():
            out = batched_fn(p0, e0, a, M, mu)
            block_jax(out)

        mean_s, std_s = repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)
        throughput = N / mean_s
        results.append(dict(N=N, mean_s=mean_s, std_s=std_s,
                            throughput=throughput, label=label))
        print(f"  [{label}] N={N:5d}: {mean_s*1e3:7.2f} ± {std_s*1e3:5.2f} ms  "
              f"({throughput:6.1f} traj/s)")

    return results


# ---------------------------------------------------------------------------
# Jacobian differentiability check
# ---------------------------------------------------------------------------

def check_differentiability(traj, grid: dict, N: int = 16) -> bool:
    """Verify jax.jacfwd w.r.t. (p0, e0) produces finite Jacobians."""
    print(f"\n  Checking differentiability (N={N}) …", end=" ", flush=True)

    def _p_final(p0, e0):
        _, ps, *_ = traj(
            p0=p0, e0=e0, a=float(grid["a"][0]), M=float(grid["M"][0]),
            mu=float(grid["mu"][0]), T=0.5, dense_steps=20,
            max_steps=4096, atol=1e-9, rtol=1e-9,
        )
        # return p at last valid point (proxy for the ODE's sensitivity)
        return ps[-1]

    J_fn = jax.jacfwd(_p_final, argnums=(0, 1))
    all_finite = True
    for i in range(min(N, len(grid["p0"]))):
        p0 = jnp.float64(grid["p0"][i])
        e0 = jnp.float64(grid["e0"][i])
        try:
            dp_dp0, dp_de0 = J_fn(p0, e0)
            if not (np.isfinite(float(dp_dp0)) and np.isfinite(float(dp_de0))):
                print(f"\n    Point {i}: non-finite Jacobian ({dp_dp0:.3e}, {dp_de0:.3e})")
                all_finite = False
        except Exception as exc:
            print(f"\n    Point {i}: exception — {exc}")
            all_finite = False

    status = "PASS" if all_finite else "FAIL"
    print(f"{status}")
    return all_finite


# ---------------------------------------------------------------------------
# TF-domain track benchmark (optional)
# ---------------------------------------------------------------------------

def bench_tf_tracks(
    traj,
    grid: dict,
    N_batch: int,
    T_yr: float,
    modes: list[tuple],
    n_repeat: int,
) -> dict:
    """Generate trajectories then compute analytical TF tracks.

    This tests the full pipeline:
      1. Run trajectory (vmapped over N_batch)
      2. For each trajectory, build analytical WDM TF tracks for ``modes``
    """
    from fewtrax.utils.tf_tracks import WDMGrid, build_tf_tracks

    YEAR_SI = 365.25 * 86400.0
    T_s     = T_yr * YEAR_SI
    grid_wdm = WDMGrid(Nf=64, Nt=4096, T=T_s)

    batched_fn = jax.jit(jax.vmap(
        lambda p0, e0, a, M, mu: traj(
            p0=p0, e0=e0, a=a, M=M, mu=mu,
            T=T_yr, x0=1.0, dt=10.0,
            dense_steps=200, max_steps=8192, atol=1e-9, rtol=1e-9,
        )
    ))

    p0_b = jnp.array(grid["p0"][:N_batch], dtype=jnp.float64)
    e0_b = jnp.array(grid["e0"][:N_batch], dtype=jnp.float64)
    a_b  = jnp.array(grid["a"][:N_batch],  dtype=jnp.float64)
    M_b  = jnp.array(grid["M"][:N_batch],  dtype=jnp.float64)
    mu_b = jnp.array(grid["mu"][:N_batch], dtype=jnp.float64)

    times_traj = []
    times_tf   = []
    total_kB   = []

    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = batched_fn(p0_b, e0_b, a_b, M_b, mu_b)
        block_jax(out)
        times_traj.append(time.perf_counter() - t0)

        ts_b, ps_b, es_b = (np.asarray(out[k]) for k in (0, 1, 2))

        t0 = time.perf_counter()
        kB_batch = 0.0
        for i in range(N_batch):
            track_set = build_tf_tracks(
                modes,
                ts_b[i], ps_b[i], es_b[i],
                a=float(a_b[i]), M=float(M_b[i]), mu=float(mu_b[i]),
                grid=grid_wdm, x0=1.0,
            )
            kB_batch += track_set.nbytes / 1024.0
        times_tf.append(time.perf_counter() - t0)
        total_kB.append(kB_batch)

    return dict(
        N_batch=N_batch,
        n_modes=len(modes),
        traj_ms=np.mean(times_traj) * 1e3,
        traj_std_ms=np.std(times_traj) * 1e3,
        tf_ms=np.mean(times_tf) * 1e3,
        tf_std_ms=np.std(times_tf) * 1e3,
        mean_kB_per_set=np.mean(total_kB) / N_batch,
        grid=grid_wdm,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--n-repeat",  type=int, default=5)
    parser.add_argument("--n-warmup",  type=int, default=2)
    parser.add_argument("--seed",      type=int, default=42,
                        help="RNG seed for parameter grid (default: 42)")
    parser.add_argument("--T",         type=float, default=0.5,
                        help="Trajectory duration [yr] (default: 0.5)")
    parser.add_argument("--dense",     type=int, default=100,
                        help="dense_steps per trajectory (default: 100)")
    parser.add_argument("--max-batch", type=int, default=1024,
                        help="Largest batch size to test (default: 1024)")
    parser.add_argument("--tf",        action="store_true",
                        help="Run TF-domain track benchmark")
    parser.add_argument("--tf-batch",  type=int, default=32,
                        help="Batch size for TF benchmark (default: 32)")
    parser.add_argument("--skip-exact", action="store_true",
                        help="Skip the base EMRIInspiral benchmark")
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

    # --- Load data ---
    data_dir = find_data_dir(args.data_dir)
    print(f"\n  FEW data directory: {data_dir}")
    from fewtrax.data import load_flux_data
    from fewtrax.trajectory import EMRIInspiral, EMRIInspiralFast
    flux_data = load_flux_data(data_dir)
    traj_exact = EMRIInspiral(flux_data)
    traj_fast  = EMRIInspiralFast(flux_data)

    # --- Build parameter grid ---
    N_max = max(args.max_batch, args.tf_batch) + 10
    print(f"\n  Building random parameter grid  (N={N_max}, seed={args.seed}) …",
          end=" ", flush=True)
    grid = build_param_grid(N_max, seed=args.seed, T_yr=args.T, dt_s=10.0)
    print("done")
    print(f"  p0  range : [{grid['p0'].min():.2f}, {grid['p0'].max():.2f}] M")
    print(f"  e0  range : [{grid['e0'].min():.3f}, {grid['e0'].max():.3f}]")
    print(f"  a   range : [{grid['a'].min():.3f},  {grid['a'].max():.3f}]")
    print(f"  M   range : [{grid['M'].min():.2e}, {grid['M'].max():.2e}] Msun")

    # --- Batch sizes to sweep ---
    max_bs = args.max_batch
    batch_sizes = sorted({1, 4, 16, 64, 256, max_bs} & {n for n in range(1, max_bs + 1)})

    # --- A. EMRIInspiralFast benchmark ---
    print_header("A. EMRIInspiralFast — vmapped batch throughput")
    fast_results = bench_batch(
        traj_fast, grid, batch_sizes, nw, nr,
        dense_steps=args.dense, label="fast",
    )

    # --- B. EMRIInspiral (exact) benchmark ---
    if not args.skip_exact:
        print_header("B. EMRIInspiral (exact) — vmapped batch throughput")
        exact_results = bench_batch(
            traj_exact, grid, batch_sizes, nw, nr,
            dense_steps=args.dense, label="exact",
        )
    else:
        exact_results = []

    # --- Summary table ---
    print_header("Throughput summary")
    headers = ["N_batch", "fast (traj/s)", "exact (traj/s)", "speedup"]
    widths  = [10, 16, 17, 10]
    rows = []
    for fr in fast_results:
        N = fr["N"]
        er = next((r for r in exact_results if r["N"] == N), None)
        if er:
            sp = f"{fr['throughput']/er['throughput']:.2f}×"
            ex_str = f"{er['throughput']:.1f}"
        else:
            sp = "—"
            ex_str = "—"
        rows.append((str(N), f"{fr['throughput']:.1f}", ex_str, sp))
    print_table(rows, headers, widths)

    peak_fast = max(r["throughput"] for r in fast_results)
    print(f"\n  Peak EMRIInspiralFast throughput: {peak_fast:.1f} traj/s")

    # --- C. Differentiability check ---
    print_header("C. Differentiability check  (jacfwd w.r.t. p0, e0)")
    _ = check_differentiability(traj_fast, grid, N=16)

    # --- D. TF-domain track benchmark ---
    if args.tf:
        print_header("D. TF-domain track benchmark  (analytical WDM tracks)")
        from fewtrax.utils.tf_tracks import WDMGrid
        YEAR_SI = 365.25 * 86400.0

        tf_modes = [(2, 2, 0, 1), (2, 2, 0, 2), (2, 1, 0, 1)]
        print(f"  Modes: {tf_modes}")
        print(f"  Batch: {args.tf_batch}  |  T={args.T} yr  |  "
              f"WDM grid: Nf=64, Nt=4096")

        tf_res = bench_tf_tracks(
            traj_fast, grid,
            N_batch=args.tf_batch,
            T_yr=args.T,
            modes=tf_modes,
            n_repeat=nr,
        )

        print(f"\n  Results (mean over {nr} repeats):")
        print(f"    vmapped trajectory ({args.tf_batch} signals) : "
              f"{tf_res['traj_ms']:.2f} ± {tf_res['traj_std_ms']:.2f} ms")
        print(f"    TF track construction (all modes, all signals) : "
              f"{tf_res['tf_ms']:.2f} ± {tf_res['tf_std_ms']:.2f} ms")
        print(f"    Memory per track set ({tf_res['n_modes']} modes) : "
              f"{tf_res['mean_kB_per_set']:.1f} kB")
        print(f"    Per-signal total (traj + TF) : "
              f"{(tf_res['traj_ms'] + tf_res['tf_ms']) / args.tf_batch:.2f} ms")

        grid_info = tf_res["grid"]
        print(f"\n    WDM grid : {grid_info}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""GPU runtime comparison: fewtrax vs FastEMRIWaveforms on NVIDIA A100.

Designed and tested for an A100-80GB.  JAX will automatically use the GPU
if ``jax[cuda12]`` is installed.  FEW is also run on GPU (via cupy/CUDA)
when available; JAX GPU memory is released before each FEW benchmark to
avoid OOM errors.

What is measured
----------------
A. **Component timings** (per single waveform)
   1. Sparse trajectory only: fewtrax (GPU) vs FEW (GPU/CPU)
   2. Dense trajectory (ODE + amplitude eval): fewtrax (GPU) vs FEW (GPU/CPU)
   3. Mode summation only: fewtrax JAX (GPU) vs numpy reference (CPU)
   4. Full end-to-end: fewtrax (GPU) vs FEW (GPU/CPU)

B. **Observation-time scaling**
   T ∈ {0.05, 0.1, 0.5, 1.0} years (fixed dt = 10 s)

C. **Sample-rate scaling**
   dt ∈ {1, 5, 10, 30, 60} s (fixed T = 0.1 yr)

D. **Mode-summation scaling** (GPU stress test)
   N_modes ∈ {100, 500, 1000, 2000, 4000, 8000} (fixed N_t = 2000)

E. **Batch / vmap scaling** (fewtrax only)
   N_batch ∈ {1, 8, 32, 128, 512, 1024} waveforms evaluated simultaneously
   via ``jax.vmap``.  Tests GPU throughput on the A100's 80 GB.

Usage
-----
    python benchmark_gpu.py [/path/to/few/data] \\
        [--n-repeat 5] [--skip-few] [--skip-vmap] \\
        [--M 1e6] [--mu 10] [--a 0.3] [--p0 10] [--e0 0.4]

Requirements
------------
    pip install "fewtrax[gpu,compare]"   # cuda12 + FastEMRIWaveforms
"""

from __future__ import annotations

import argparse
import gc
import sys
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from utils import (
    find_data_dir,
    PARAMS_DEFAULT,
    repeat_timer,
    block_jax,
    print_header,
    print_table,
)


# ---------------------------------------------------------------------------
# Device / memory helpers
# ---------------------------------------------------------------------------

def describe_devices() -> None:
    devices = jax.devices()
    print(f"\nJAX version : {jax.__version__}")
    print(f"Devices     : {devices}")
    for d in devices:
        print(f"  {d.id}: {d.device_kind}  platform={d.platform}")

    # GPU memory (CUDA only)
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in out.strip().splitlines():
            name, total, free = line.split(", ")
            print(f"  GPU: {name}  total={total} MiB  free={free} MiB")
    except Exception:
        pass


def is_gpu_available() -> bool:
    return any(d.platform == "gpu" for d in jax.devices())


def clear_gpu_memory() -> None:
    """Release JAX GPU caches and cupy memory pools before running FEW.

    FEW uses cupy (not JAX) for GPU operations, so both memory pools must
    be freed to avoid OOM errors when the two frameworks coexist.
    """
    jax.clear_caches()
    gc.collect()
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Trajectory benchmark helpers
# ---------------------------------------------------------------------------

def bench_fewtrax_trajectory_sparse(
    params: dict,
    flux_data,
    dense_steps: int,
    n_warmup: int,
    n_repeat: int,
) -> tuple[float, float]:
    """Time fewtrax EMRIInspiral (ODE only → sparse trajectory output)."""
    from fewtrax.trajectory import EMRIInspiral
    traj = EMRIInspiral(flux_data)

    def fn():
        result = traj(
            p0=params["p0"], e0=params["e0"],
            T=params["T"], dt=params["dt"],
            a=params["a"], x0=params.get("x0", 1.0),
            M=params["M"], mu=params["mu"],
            Phi_phi0=params.get("Phi_phi0", 0.0),
            Phi_theta0=params.get("Phi_theta0", 0.0),
            Phi_r0=params.get("Phi_r0", 0.0),
            dense_steps=dense_steps,
        )
        block_jax(result)

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


def bench_fewtrax_trajectory_dense(
    params: dict,
    wf_gen,
    n_warmup: int,
    n_repeat: int,
) -> tuple[float, float]:
    """Time fewtrax generate_sparse (ODE + Teukolsky amplitude evaluation).

    This covers the complete trajectory pipeline: ODE integration over
    ``dense_steps`` sparse points followed by scipy B-spline amplitude
    evaluation at each point.  The result is the sparse trajectory dict
    used as input to the mode summation step.
    """
    call_kwargs = dict(
        M=params["M"], mu=params["mu"],
        a=params["a"],
        p0=params["p0"], e0=params["e0"],
        x0=params.get("x0", 1.0),
        T=params["T"], dt=params["dt"],
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
    )

    def fn():
        sparse = wf_gen.generate_sparse(**call_kwargs)
        # Block on JAX arrays inside the returned dict
        block_jax(tuple(
            v for v in sparse.values() if hasattr(v, "block_until_ready")
        ))

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


def bench_few_trajectory(
    params: dict,
    n_warmup: int,
    n_repeat: int,
    use_gpu: bool = False,
) -> tuple[float, float]:
    from few.trajectory.inspiral import EMRIInspiral as FEWInspiral
    traj = FEWInspiral(func='KerrEccEqFlux', use_gpu=use_gpu)

    def fn():
        traj(
            params["M"], params["mu"],
            params["a"],
            params["p0"], params["e0"], params.get("x0", 1.0),
            T=params["T"],
            dt=params["dt"],
        )

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


# ---------------------------------------------------------------------------
# Full waveform benchmark helpers
# ---------------------------------------------------------------------------

def bench_fewtrax_waveform(
    params: dict,
    wf_gen,
    n_warmup: int,
    n_repeat: int,
) -> tuple[float, float]:
    call_kwargs = {k: v for k, v in params.items() if k != "label"}

    def fn():
        hp, hx = wf_gen(**call_kwargs)
        block_jax((hp, hx))

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


def bench_few_waveform(
    params: dict,
    n_warmup: int,
    n_repeat: int,
    use_gpu: bool = False,
) -> tuple[float, float]:
    from few.waveform import GenerateEMRIWaveform
    gen = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        use_gpu=use_gpu,
    )
    call_kwargs = dict(
        M=params["M"], mu=params["mu"],
        a=params["a"],
        p0=params["p0"], e0=params["e0"], x0=params.get("x0", 1.0),
        dist=params.get("dist", 1.0),
        qS=params.get("qS", 0.0), phiS=params.get("phiS", 0.0),
        qK=params.get("qK", 0.0), phiK=params.get("phiK", 0.0),
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
        T=params["T"], dt=params["dt"],
    )

    def fn():
        gen(**call_kwargs)

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


# ---------------------------------------------------------------------------
# Mode summation benchmark helpers
# ---------------------------------------------------------------------------

def _make_summation_data(N_t: int, N_modes: int):
    """Build synthetic mode data for summation benchmarks."""
    rng = np.random.default_rng(7)
    teuk = (
        rng.standard_normal((N_t, N_modes))
        + 1j * rng.standard_normal((N_t, N_modes))
    ).astype(np.complex128) * 1e-22
    yp = (rng.standard_normal(N_modes) + 1j * rng.standard_normal(N_modes)).astype(np.complex128)
    yn = (rng.standard_normal(N_modes) + 1j * rng.standard_normal(N_modes)).astype(np.complex128)
    Pp = jnp.linspace(0.0, 200.0 * np.pi, N_t)
    Pt = jnp.linspace(0.0, 120.0 * np.pi, N_t)
    Pr = jnp.linspace(0.0,  40.0 * np.pi, N_t)
    l = jnp.ones(N_modes, dtype=jnp.int32) * 2
    m = jnp.ones(N_modes, dtype=jnp.int32) * 2
    k = jnp.zeros(N_modes, dtype=jnp.int32)
    n = jnp.ones(N_modes, dtype=jnp.int32)
    return jnp.asarray(teuk), jnp.asarray(yp), jnp.asarray(yn), Pp, Pt, Pr, l, m, k, n


def bench_fewtrax_summation(N_t: int, N_modes: int, n_warmup: int, n_repeat: int) -> tuple[float, float]:
    from fewtrax.summation.modes import direct_mode_sum
    data = _make_summation_data(N_t, N_modes)

    def fn():
        h = direct_mode_sum(*data)
        block_jax(h)

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


def bench_numpy_summation(N_t: int, N_modes: int, n_warmup: int, n_repeat: int) -> tuple[float, float]:
    rng = np.random.default_rng(7)
    teuk = (
        rng.standard_normal((N_t, N_modes))
        + 1j * rng.standard_normal((N_t, N_modes))
    ).astype(np.complex128)
    yp = rng.standard_normal(N_modes) + 1j * rng.standard_normal(N_modes)
    yn = rng.standard_normal(N_modes) + 1j * rng.standard_normal(N_modes)
    Pp = np.linspace(0.0, 200.0 * np.pi, N_t)
    Pt = np.linspace(0.0, 120.0 * np.pi, N_t)
    Pr = np.linspace(0.0,  40.0 * np.pi, N_t)
    m_arr = np.ones(N_modes, dtype=int)
    k_arr = np.zeros(N_modes, dtype=int)
    n_arr = np.ones(N_modes, dtype=int)

    def fn():
        phase = (
            m_arr[None, :] * Pp[:, None]
            + k_arr[None, :] * Pt[:, None]
            + n_arr[None, :] * Pr[:, None]
        )
        w1 = np.sum(yp[None, :] * teuk * np.exp(-1j * phase), axis=1)
        w2 = np.sum((-1.0) * yn[None, :] * np.conj(teuk) * np.exp(1j * phase), axis=1)
        return w1 + w2

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


# ---------------------------------------------------------------------------
# Batch / vmap benchmark
# ---------------------------------------------------------------------------

def bench_vmap_waveforms(
    base_params: dict,
    wf_gen,
    flux_data,
    batch_sizes: list[int],
    n_warmup: int,
    n_repeat: int,
) -> list[dict]:
    """Benchmark batched trajectory evaluation via jax.vmap.

    Each batch varies p0 uniformly across [p0-0.5, p0+0.5].
    Only the trajectory step is vmapped (amplitude evaluation is CPU-bound).
    """
    from fewtrax.trajectory import EMRIInspiral
    traj = EMRIInspiral(flux_data)

    fixed_kwargs = dict(
        e0=base_params["e0"],
        T=base_params["T"],
        dt=base_params["dt"],
        a=base_params["a"],
        x0=base_params.get("x0", 1.0),
        M=base_params["M"],
        mu=base_params["mu"],
        dense_steps=100,
    )

    results = []
    for N in batch_sizes:
        p0_batch = jnp.linspace(
            base_params["p0"] - 0.5,
            base_params["p0"] + 0.5,
            N,
            dtype=jnp.float64,
        )

        def fn():
            out = jax.vmap(lambda p0: traj(p0=p0, **fixed_kwargs))(p0_batch)
            block_jax(out)

        mean_t, std_t = repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)
        throughput = N / mean_t
        results.append(dict(
            N=N,
            mean_s=mean_t,
            std_s=std_t,
            throughput=throughput,
        ))
        print(f"  N={N:6d}: {mean_t:.3f} ± {std_t:.3f} s  "
              f"({throughput:.1f} waveforms/s)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", nargs="?", default=None,
                        help="Path to FEW HDF5 data directory")
    parser.add_argument("--n-repeat", type=int, default=5,
                        help="Number of timed repetitions (default: 5)")
    parser.add_argument("--n-warmup", type=int, default=2,
                        help="Number of warm-up calls before timing (default: 2)")
    parser.add_argument("--skip-few", action="store_true",
                        help="Skip FEW benchmarks (use when FEW is not installed)")
    parser.add_argument("--skip-vmap", action="store_true",
                        help="Skip vmap batch benchmark (faster)")

    # Intrinsic parameters
    parser.add_argument("--M", type=float, default=None,
                        help="Primary BH mass [M_sun] (default: 1e6)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Secondary mass [M_sun] (default: 10.0)")
    parser.add_argument("--a", type=float, default=None,
                        help="Dimensionless Kerr spin, |a| < 1 (default: 0.3)")
    parser.add_argument("--p0", type=float, default=None,
                        help="Initial semi-latus rectum [M] (default: 10.0)")
    parser.add_argument("--e0", type=float, default=None,
                        help="Initial eccentricity, 0 ≤ e0 < 1 (default: 0.4)")

    args = parser.parse_args()

    nw, nr = args.n_warmup, args.n_repeat

    describe_devices()
    gpu = is_gpu_available()
    if not gpu:
        print("\n[WARNING] No GPU detected. Results will reflect CPU performance.")
        print("  Install jax[cuda12] and run on an A100 for GPU results.\n")

    # Build parameter dict, applying any CLI overrides
    params = dict(PARAMS_DEFAULT)
    for key, val in [("M", args.M), ("mu", args.mu), ("a", args.a),
                     ("p0", args.p0), ("e0", args.e0)]:
        if val is not None:
            params[key] = val

    print(f"\nBenchmark parameters:")
    print(f"  M={params['M']:.3g} M_sun  mu={params['mu']:.3g} M_sun  "
          f"a={params['a']:.3g}  p0={params['p0']:.3g}  e0={params['e0']:.3g}")
    print(f"  T={params['T']} yr  dt={params['dt']} s")

    data_dir = find_data_dir(args.data_dir)
    print(f"\nFEW data directory: {data_dir}")

    # Build fewtrax objects
    print("\nLoading fewtrax …")
    from fewtrax.data import load_flux_data
    from fewtrax import KerrEccentricEquatorialWaveform
    flux_data = load_flux_data(data_dir)
    wf_gen = KerrEccentricEquatorialWaveform(
        data_dir=data_dir,
        mode_selection_threshold=1e-5,
        dense_steps=100,
    )
    print("  Done.")

    # FEW: use GPU when available (requires cupy / FEW cuda build)
    few_device = "GPU" if gpu else "CPU"
    few_use_gpu = gpu

    # ------------------------------------------------------------------
    # A. Component timings (single waveform)
    # ------------------------------------------------------------------
    print_header("A. Component timings (single waveform, default params)")

    # A1. Sparse trajectory (ODE integration only)
    print("\n  [Trajectory — sparse (ODE only)]")
    ft_ts_mean, ft_ts_std = bench_fewtrax_trajectory_sparse(
        params, flux_data, dense_steps=100, n_warmup=nw, n_repeat=nr,
    )
    print(f"    fewtrax (JAX, {few_device}):  {ft_ts_mean*1e3:.2f} ± {ft_ts_std*1e3:.2f} ms")

    if not args.skip_few:
        print(f"    Clearing JAX GPU memory before FEW …")
        clear_gpu_memory()
        few_ts_mean, few_ts_std = bench_few_trajectory(
            params, nw, nr, use_gpu=few_use_gpu,
        )
        tag = f"FEW ({few_device})"
        print(f"    {tag}:          {few_ts_mean*1e3:.2f} ± {few_ts_std*1e3:.2f} ms")
        print(f"    Speedup:                  {few_ts_mean/ft_ts_mean:.2f}×")

    # A2. Dense trajectory (ODE + amplitude evaluation)
    print("\n  [Trajectory — dense (ODE + amplitude eval)]")
    ft_td_mean, ft_td_std = bench_fewtrax_trajectory_dense(
        params, wf_gen, n_warmup=nw, n_repeat=nr,
    )
    print(f"    fewtrax (JAX traj + CPU amp): {ft_td_mean*1e3:.2f} ± {ft_td_std*1e3:.2f} ms")

    # A3. Mode summation only
    print("\n  [Mode summation  N_t=500, N_modes=200]")
    ft_s_mean, ft_s_std = bench_fewtrax_summation(500, 200, nw, nr)
    np_s_mean, np_s_std = bench_numpy_summation(500, 200, nw, nr)
    print(f"    fewtrax JAX ({few_device}): {ft_s_mean*1e3:.2f} ± {ft_s_std*1e3:.2f} ms")
    print(f"    numpy (CPU ref):    {np_s_mean*1e3:.2f} ± {np_s_std*1e3:.2f} ms")
    print(f"    Speedup:            {np_s_mean/ft_s_mean:.2f}×")

    # A4. Full end-to-end
    print("\n  [Full end-to-end  T=0.1yr, dt=10s]")
    ft_f_mean, ft_f_std = bench_fewtrax_waveform(params, wf_gen, nw, nr)
    print(f"    fewtrax total:      {ft_f_mean:.3f} ± {ft_f_std:.3f} s")
    if not args.skip_few:
        print(f"    Clearing JAX GPU memory before FEW …")
        clear_gpu_memory()
        few_f_mean, few_f_std = bench_few_waveform(
            params, nw, nr, use_gpu=few_use_gpu,
        )
        print(f"    FEW ({few_device}) total: {few_f_mean:.3f} ± {few_f_std:.3f} s")
        print(f"    Speedup:            {few_f_mean/ft_f_mean:.2f}×")

    # ------------------------------------------------------------------
    # B. Observation-time scaling
    # ------------------------------------------------------------------
    print_header("B. Observation-time scaling  (dt = 10 s, fewtrax)")

    T_SWEEP = [0.05, 0.1, 0.5, 1.0]
    headers = ["T [yr]", "N_samples", "fewtrax [s]", f"FEW [{few_device}] [s]", "speedup"]
    widths  = [10, 12, 16, 18, 10]
    rows = []
    for T in T_SWEEP:
        p = dict(params, T=T)
        ft_mean, ft_std = bench_fewtrax_waveform(p, wf_gen, nw, nr)
        N_s = int(T * 365.25 * 86400 / params["dt"])

        if not args.skip_few:
            clear_gpu_memory()
            few_mean, few_std = bench_few_waveform(p, nw, nr, use_gpu=few_use_gpu)
            sp = f"{few_mean/ft_mean:.2f}×"
            few_str = f"{few_mean:.3f} ± {few_std:.3f}"
        else:
            sp = "–"
            few_str = "–"

        rows.append((
            str(T),
            str(N_s),
            f"{ft_mean:.3f} ± {ft_std:.3f}",
            few_str,
            sp,
        ))
    print_table(rows, headers, widths)

    # ------------------------------------------------------------------
    # C. Sample-rate scaling
    # ------------------------------------------------------------------
    print_header("C. Sample-rate scaling  (T = 0.1 yr, fewtrax)")

    DT_SWEEP = [1.0, 5.0, 10.0, 30.0, 60.0]
    rows = []
    for dt in DT_SWEEP:
        p = dict(params, dt=dt)
        ft_mean, ft_std = bench_fewtrax_waveform(p, wf_gen, nw, nr)
        N_s = int(params["T"] * 365.25 * 86400 / dt)

        if not args.skip_few:
            clear_gpu_memory()
            few_mean, few_std = bench_few_waveform(p, nw, nr, use_gpu=few_use_gpu)
            sp = f"{few_mean/ft_mean:.2f}×"
            few_str = f"{few_mean:.3f} ± {few_std:.3f}"
        else:
            sp = "–"
            few_str = "–"

        rows.append((
            f"{dt:.0f}",
            str(N_s),
            f"{ft_mean:.3f} ± {ft_std:.3f}",
            few_str,
            sp,
        ))
    headers = ["dt [s]", "N_samples", "fewtrax [s]", f"FEW [{few_device}] [s]", "speedup"]
    widths  = [10, 12, 16, 18, 10]
    print_table(rows, headers, widths)

    # ------------------------------------------------------------------
    # D. Mode-summation scaling: larger mode counts (GPU-stress test)
    # ------------------------------------------------------------------
    print_header("D. Mode summation scaling on GPU  (fewtrax JAX vs numpy)")

    N_T_BIG  = 2000
    MODES_BIG = [100, 500, 1000, 2000, 4000, 8000]
    headers = ["N_modes", "N_t", "fewtrax JAX [ms]", "numpy [ms]", "speedup"]
    widths  = [10, 8, 20, 16, 10]
    rows = []
    for N_m in MODES_BIG:
        ft_mean, ft_std = bench_fewtrax_summation(N_T_BIG, N_m, nw, nr)
        np_mean, np_std = bench_numpy_summation(N_T_BIG, N_m, nw, nr)
        sp = np_mean / ft_mean if ft_mean > 0 else float("nan")
        rows.append((
            str(N_m),
            str(N_T_BIG),
            f"{ft_mean*1e3:.2f} ± {ft_std*1e3:.2f}",
            f"{np_mean*1e3:.2f} ± {np_std*1e3:.2f}",
            f"{sp:.1f}×",
        ))
    print_table(rows, headers, widths)

    # ------------------------------------------------------------------
    # E. vmap batch benchmark (A100 throughput)
    # ------------------------------------------------------------------
    if not args.skip_vmap:
        print_header("E. Batched trajectory via jax.vmap  (A100 throughput test)")
        print("   Varying p0 ∈ [p0-0.5, p0+0.5],  T=0.1 yr,  dense_steps=100")

        BATCH_SIZES = [1, 8, 32, 128, 512, 1024]
        vmap_results = bench_vmap_waveforms(
            params, wf_gen, flux_data,
            batch_sizes=BATCH_SIZES,
            n_warmup=nw, n_repeat=nr,
        )

        print()
        headers = ["N_batch", "time [s]", "waveforms/s"]
        widths  = [10, 14, 14]
        rows = [
            (str(r["N"]), f"{r['mean_s']:.3f} ± {r['std_s']:.3f}", f"{r['throughput']:.1f}")
            for r in vmap_results
        ]
        print_table(rows, headers, widths)

        peak = max(r["throughput"] for r in vmap_results)
        print(f"\n  Peak throughput: {peak:.1f} waveforms/s")
        print(
            "  Note: throughput is limited by the CPU-bound scipy amplitude step.\n"
            "  The JAX trajectory + summation alone would scale linearly with N_batch."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

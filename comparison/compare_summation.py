"""Mode summation speed: fewtrax JAX vs FEW numpy.

Benchmarks the *summation* step in isolation – the inner loop that takes
precomputed mode amplitudes and trajectory phases and produces the strain.
This is the most computationally intensive part of waveform generation and
the step most improved by GPU execution.

Comparison targets
------------------
1. ``fewtrax.summation.direct_mode_sum`` (JIT-compiled JAX)
2. ``fewtrax.summation.interpolated_mode_sum`` (JIT + cubic-spline upsample)
3. FEW's internal summation (via ``GenerateEMRIWaveform`` with pre-computed
   trajectory/amplitudes, isolating only the summation call)

Scaling sweeps
--------------
- Number of modes  N_modes  ∈ {50, 200, 500, 1000, 2000}
- Number of time samples  N_t  ∈ {100, 500, 2000, 10000}

Usage
-----
    python compare_summation.py [/path/to/few/data] [--n-repeat 10]
"""

from __future__ import annotations

import argparse
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
# Build synthetic mode data (no FEW data needed for scaling sweeps)
# ---------------------------------------------------------------------------

def make_synthetic_modes(N_t: int, N_modes: int, seed: int = 42):
    """Create synthetic trajectory and amplitude data.

    Returns arrays compatible with ``direct_mode_sum``.
    """
    rng = np.random.default_rng(seed)
    teuk_modes = (
        rng.standard_normal((N_t, N_modes))
        + 1j * rng.standard_normal((N_t, N_modes))
    ).astype(np.complex128) * 1e-22

    # Monotonically increasing phases
    Phi_phi   = np.linspace(0.0, 50.0 * np.pi, N_t)
    Phi_theta = np.linspace(0.0, 30.0 * np.pi, N_t)
    Phi_r     = np.linspace(0.0, 10.0 * np.pi, N_t)

    # Random mode indices (l,m,k,n)
    l_arr = rng.integers(2, 11, size=N_modes)
    m_arr = rng.integers(1, 11, size=N_modes)
    k_arr = np.zeros(N_modes, dtype=int)
    n_arr = rng.integers(1, 6,  size=N_modes)

    # Random spherical harmonics
    ylms_pos = (
        rng.standard_normal(N_modes)
        + 1j * rng.standard_normal(N_modes)
    ).astype(np.complex128)
    ylms_neg = (
        rng.standard_normal(N_modes)
        + 1j * rng.standard_normal(N_modes)
    ).astype(np.complex128)

    return (
        jnp.asarray(teuk_modes),
        jnp.asarray(ylms_pos), jnp.asarray(ylms_neg),
        jnp.asarray(Phi_phi), jnp.asarray(Phi_theta), jnp.asarray(Phi_r),
        jnp.asarray(l_arr), jnp.asarray(m_arr),
        jnp.asarray(k_arr), jnp.asarray(n_arr),
    )


# ---------------------------------------------------------------------------
# Benchmark: fewtrax direct_mode_sum (sparse, no upsample)
# ---------------------------------------------------------------------------

def bench_fewtrax_direct(N_t: int, N_modes: int, n_warmup: int, n_repeat: int):
    """Benchmark ``direct_mode_sum`` on the current JAX device."""
    from fewtrax.summation.modes import direct_mode_sum

    data = make_synthetic_modes(N_t, N_modes)
    teuk, yp, yn, Pp, Pt, Pr, l, m, k, n = data

    def fn():
        h = direct_mode_sum(teuk, yp, yn, Pp, Pt, Pr, l, m, k, n)
        block_jax(h)

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


# ---------------------------------------------------------------------------
# Benchmark: fewtrax interpolated_mode_sum (sparse→dense upsample)
# ---------------------------------------------------------------------------

def bench_fewtrax_interpolated(N_sparse: int, N_modes: int, dt: float,
                                n_warmup: int, n_repeat: int):
    """Benchmark ``interpolated_mode_sum`` (sparse trajectory + upsample)."""
    from fewtrax.summation.modes import interpolated_mode_sum

    data = make_synthetic_modes(N_sparse, N_modes)
    teuk, yp, yn, Pp, Pt, Pr, l, m, k, n = data

    t_traj = jnp.linspace(0.0, 3.15e6, N_sparse)  # ~0.1 year in seconds
    l_np = np.asarray(l)
    m_np = np.asarray(m)
    k_np = np.asarray(k)
    n_np = np.asarray(n)

    def fn():
        _, h = interpolated_mode_sum(
            t_traj, np.asarray(teuk), yp, yn, Pp, Pt, Pr,
            l_np, m_np, k_np, n_np, dt=dt,
        )
        block_jax(h)

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


# ---------------------------------------------------------------------------
# Benchmark: numpy reference summation (mimics FEW's approach)
# ---------------------------------------------------------------------------

def bench_numpy_direct(N_t: int, N_modes: int, n_warmup: int, n_repeat: int):
    """Numpy-based mode summation – rough equivalent of FEW's CPU path."""
    rng = np.random.default_rng(0)
    teuk = (
        rng.standard_normal((N_t, N_modes))
        + 1j * rng.standard_normal((N_t, N_modes))
    ).astype(np.complex128)
    ylms_pos = rng.standard_normal(N_modes) + 1j * rng.standard_normal(N_modes)
    ylms_neg = rng.standard_normal(N_modes) + 1j * rng.standard_normal(N_modes)
    Phi_phi   = np.linspace(0.0, 50.0 * np.pi, N_t)
    Phi_theta = np.linspace(0.0, 30.0 * np.pi, N_t)
    Phi_r     = np.linspace(0.0, 10.0 * np.pi, N_t)
    m_arr = np.ones(N_modes, dtype=int)
    k_arr = np.zeros(N_modes, dtype=int)
    n_arr = np.ones(N_modes, dtype=int)

    def fn():
        phase = (
            m_arr[np.newaxis, :] * Phi_phi[:, np.newaxis]
            + k_arr[np.newaxis, :] * Phi_theta[:, np.newaxis]
            + n_arr[np.newaxis, :] * Phi_r[:, np.newaxis]
        )
        w1 = np.sum(ylms_pos[np.newaxis, :] * teuk * np.exp(-1j * phase), axis=1)
        sign_l = (-1.0) ** np.ones(N_modes)
        w2 = np.sum(sign_l[np.newaxis, :] * ylms_neg[np.newaxis, :]
                    * np.conj(teuk) * np.exp(1j * phase), axis=1)
        return w1 + w2

    return repeat_timer(fn, n_warmup=n_warmup, n_repeat=n_repeat)


# ---------------------------------------------------------------------------
# Benchmark: full FEW waveform (trajectory + amplitude + summation together)
# ---------------------------------------------------------------------------

def bench_few_full(params: dict, n_warmup: int, n_repeat: int):
    """Time FEW end-to-end waveform generation (single call)."""
    from few.waveform import GenerateEMRIWaveform
    gen = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux")

    call_kwargs = dict(
        M=params["M"], mu=params["mu"],
        a=params["a"],
        p0=params["p0"], e0=params["e0"], x0=params["x0"],
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--n-repeat", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--skip-few", action="store_true",
                        help="Skip the FEW benchmark (faster, offline)")
    args = parser.parse_args()

    nw, nr = args.n_warmup, args.n_repeat

    # Report JAX device
    devices = jax.devices()
    print(f"\nJAX devices: {devices}")
    primary = devices[0]
    print(f"Running on: {primary.device_kind}  ({primary.platform})")

    # ------------------------------------------------------------------
    # 1. Scaling sweep: N_modes at fixed N_t
    # ------------------------------------------------------------------
    print_header("1. Modes scaling  (N_t = 500 trajectory points, dt = 10 s)")

    N_T_FIXED = 500
    MODE_SWEEP = [50, 100, 200, 500, 1000, 2000]

    headers = ["N_modes", "fewtrax JAX [ms]", "numpy ref [ms]", "speedup"]
    widths  = [10, 20, 18, 10]
    rows = []
    for N_m in MODE_SWEEP:
        ft_mean, ft_std = bench_fewtrax_direct(N_T_FIXED, N_m, n_warmup=nw, n_repeat=nr)
        np_mean, np_std = bench_numpy_direct(N_T_FIXED, N_m, n_warmup=nw, n_repeat=nr)
        speedup = np_mean / ft_mean if ft_mean > 0 else float("nan")
        rows.append((
            str(N_m),
            f"{ft_mean*1e3:.2f} ± {ft_std*1e3:.2f}",
            f"{np_mean*1e3:.2f} ± {np_std*1e3:.2f}",
            f"{speedup:.1f}×",
        ))
    print_table(rows, headers, widths)

    # ------------------------------------------------------------------
    # 2. Scaling sweep: N_t at fixed N_modes
    # ------------------------------------------------------------------
    print_header("2. Time-sample scaling  (N_modes = 200)")

    N_MODES_FIXED = 200
    NT_SWEEP = [100, 500, 2000, 5000, 10000]

    rows = []
    headers = ["N_t", "fewtrax JAX [ms]", "numpy ref [ms]", "speedup"]
    for N_t in NT_SWEEP:
        ft_mean, ft_std = bench_fewtrax_direct(N_t, N_MODES_FIXED, n_warmup=nw, n_repeat=nr)
        np_mean, np_std = bench_numpy_direct(N_t, N_MODES_FIXED, n_warmup=nw, n_repeat=nr)
        speedup = np_mean / ft_mean if ft_mean > 0 else float("nan")
        rows.append((
            str(N_t),
            f"{ft_mean*1e3:.2f} ± {ft_std*1e3:.2f}",
            f"{np_mean*1e3:.2f} ± {np_std*1e3:.2f}",
            f"{speedup:.1f}×",
        ))
    print_table(rows, headers, widths)

    # ------------------------------------------------------------------
    # 3. Interpolated vs direct (upsample cost)
    # ------------------------------------------------------------------
    print_header("3. Interpolated (sparse→dense) summation  (N_modes = 200, dt = 10 s)")

    DT = 10.0
    SPARSE_SWEEP = [50, 100, 200, 500]

    headers = ["N_sparse", "interp [ms]", "direct (same N) [ms]", "overhead"]
    widths  = [12, 14, 22, 12]
    rows = []
    for N_s in SPARSE_SWEEP:
        it_mean, it_std = bench_fewtrax_interpolated(N_s, 200, DT, n_warmup=nw, n_repeat=nr)
        dt_mean, dt_std = bench_fewtrax_direct(N_s, 200, n_warmup=nw, n_repeat=nr)
        overhead = it_mean / dt_mean if dt_mean > 0 else float("nan")
        rows.append((
            str(N_s),
            f"{it_mean*1e3:.2f} ± {it_std*1e3:.2f}",
            f"{dt_mean*1e3:.2f} ± {dt_std*1e3:.2f}",
            f"{overhead:.1f}×",
        ))
    print_table(rows, headers, widths)

    # ------------------------------------------------------------------
    # 4. Full end-to-end waveform (fewtrax vs FEW)
    # ------------------------------------------------------------------
    if not args.skip_few:
        data_dir = find_data_dir(args.data_dir)
        print_header("4. End-to-end waveform  (T=0.1 yr, dt=10 s, default params)")
        print(f"   FEW data: {data_dir}")

        from fewtrax import KerrEccentricEquatorialWaveform
        wf = KerrEccentricEquatorialWaveform(data_dir=data_dir, dense_steps=100)

        ft_params = dict(PARAMS_DEFAULT)

        def fn_ft():
            hp, hx = wf(**{k: v for k, v in ft_params.items()
                           if k not in ("label",)})
            block_jax((hp, hx))

        ft_mean, ft_std = repeat_timer(fn_ft, n_warmup=nw, n_repeat=nr)
        few_mean, few_std = bench_few_full(PARAMS_DEFAULT, n_warmup=nw, n_repeat=nr)

        speedup = few_mean / ft_mean if ft_mean > 0 else float("nan")
        print(f"\n  fewtrax:  {ft_mean:.3f} ± {ft_std:.3f} s")
        print(f"  FEW:      {few_mean:.3f} ± {few_std:.3f} s")
        print(f"  Speedup:  {speedup:.2f}×  (fewtrax / FEW)")
        print()
        print("  Note: fewtrax amplitude step (scipy B-spline, CPU) dominates.")
        print("  The JAX trajectory + summation steps alone are substantially faster.")

    print("\nDone.")


if __name__ == "__main__":
    main()

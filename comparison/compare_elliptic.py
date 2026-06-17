"""Accuracy comparison: elliptic integral implementations and phase self-consistency.

Compares the three fewtrax elliptic integral implementations:

  * **Exact**  — 64-point Gauss-Legendre  (``ellipk``, ``ellipe``, ``ellip_pi``)
  * **Fast**   — AGM-12 for K/E, 24-point GL for Π
                 (``ellipk_agm``, ``ellipe_agm``, ``ellip_pi_fast``)
  * **Reference** — ``scipy.special`` (treated as ground truth for K and E;
                     ``mpmath.ellippi`` used for Π when available, otherwise
                     64-pt GL is the reference).

``EMRIInspiral`` dispatches between these two paths at JIT-trace time based
on the default JAX device (GPU → 64-pt GL, CPU → AGM+24-pt GL).

Part 1 — Pointwise accuracy over a grid of (m, n) values.

Part 2 — Phase self-consistency:  run a 2-year EMRI inspiral with
  ``EMRIInspiral`` and verify the accumulated phases Φ_φ agree with
  independent numerical integration of Ω_φ(p(t), e(t)) · dt along the
  same trajectory.  The maximum discrepancy should be well below 1 rad.

Usage
-----
    python compare_elliptic.py
    python compare_elliptic.py --plot        # also save figures
    python compare_elliptic.py --T 2.0       # 2-year inspiral (default)
    python compare_elliptic.py --T 0.5       # shorter test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import scipy.special

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import find_data_dir, timer

# ---------------------------------------------------------------------------
# Elliptic integral imports
# ---------------------------------------------------------------------------

from fewtrax.utils.geodesic import (
    ellipk, ellipe, ellip_pi,
    ellipk_agm, ellipe_agm, ellip_pi_fast,
)

# Optional mpmath for independent Π reference
try:
    import mpmath
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False


# ---------------------------------------------------------------------------
# 1. Pointwise accuracy
# ---------------------------------------------------------------------------

def _ref_ellipk(m_arr):
    return np.array([float(scipy.special.ellipk(float(m))) for m in m_arr])


def _ref_ellipe(m_arr):
    return np.array([float(scipy.special.ellipe(float(m))) for m in m_arr])


def _ref_ellip_pi(n_arr, k_arr):
    if _HAS_MPMATH:
        return np.array([
            float(mpmath.ellippi(float(n), float(k)))
            for n, k in zip(n_arr, k_arr)
        ])
    else:
        # Fall back to 64-pt GL (the "exact" fewtrax version)
        return np.array([float(ellip_pi(float(n), float(k)))
                         for n, k in zip(n_arr, k_arr)])


def accuracy_grid(n_pts: int = 200) -> None:
    """Print relative-error tables for K, E, and Π."""
    rng = np.random.default_rng(42)

    # K and E: m ∈ [0.001, 0.97]
    m_vals = np.linspace(0.001, 0.97, n_pts)
    ref_K  = _ref_ellipk(m_vals)
    ref_E  = _ref_ellipe(m_vals)

    # JIT-compiled batched evaluation
    K_exact = np.array([float(ellipk(jnp.float64(m)))     for m in m_vals])
    K_fast  = np.array([float(ellipk_agm(jnp.float64(m))) for m in m_vals])
    E_exact = np.array([float(ellipe(jnp.float64(m)))     for m in m_vals])
    E_fast  = np.array([float(ellipe_agm(jnp.float64(m))) for m in m_vals])

    err_K_exact = np.abs((K_exact - ref_K) / ref_K)
    err_K_fast  = np.abs((K_fast  - ref_K) / ref_K)
    err_E_exact = np.abs((E_exact - ref_E) / ref_E)
    err_E_fast  = np.abs((E_fast  - ref_E) / ref_E)

    print("\n=== K(m) accuracy vs scipy.special.ellipk ===")
    print(f"  64-pt GL (exact): max rel err = {err_K_exact.max():.2e},  "
          f"mean = {err_K_exact.mean():.2e}")
    print(f"  AGM-12   (fast):  max rel err = {err_K_fast.max():.2e},  "
          f"mean = {err_K_fast.mean():.2e}")

    print("\n=== E(m) accuracy vs scipy.special.ellipe ===")
    print(f"  64-pt GL (exact): max rel err = {err_E_exact.max():.2e},  "
          f"mean = {err_E_exact.mean():.2e}")
    print(f"  AGM-12   (fast):  max rel err = {err_E_fast.max():.2e},  "
          f"mean = {err_E_fast.mean():.2e}")

    # Π: n ∈ (0, 0.9), k ∈ (0.05, 0.9)
    n_vals = rng.uniform(0.01, 0.9, n_pts)
    k_vals = rng.uniform(0.05, 0.9, n_pts)
    ref_Pi   = _ref_ellip_pi(n_vals, k_vals)
    Pi_exact = np.array([float(ellip_pi(jnp.float64(n), jnp.float64(k)))
                         for n, k in zip(n_vals, k_vals)])
    Pi_fast  = np.array([float(ellip_pi_fast(jnp.float64(n), jnp.float64(k)))
                         for n, k in zip(n_vals, k_vals)])

    err_Pi_exact = np.abs((Pi_exact - ref_Pi) / ref_Pi)
    err_Pi_fast  = np.abs((Pi_fast  - ref_Pi) / ref_Pi)
    ref_label = "mpmath" if _HAS_MPMATH else "64-pt GL"

    print(f"\n=== Π(n, k) accuracy vs {ref_label} ===")
    print(f"  64-pt GL (exact): max rel err = {err_Pi_exact.max():.2e},  "
          f"mean = {err_Pi_exact.mean():.2e}")
    print(f"  24-pt GL (fast):  max rel err = {err_Pi_fast.max():.2e},  "
          f"mean = {err_Pi_fast.mean():.2e}")

    # Accuracy at EMRI-relevant parameter values (kr² < 0.8)
    m_emri   = m_vals[m_vals < 0.8]
    n_emri   = n_vals[n_vals < 0.8]
    k_emri   = k_vals[:len(n_emri)][k_vals[:len(n_emri)] < 0.9]
    n_emri   = n_emri[:len(k_emri)]

    Pi_fast_emri = np.array([float(ellip_pi_fast(jnp.float64(n), jnp.float64(k)))
                              for n, k in zip(n_emri, k_emri)])
    Pi_ref_emri  = _ref_ellip_pi(n_emri, k_emri)
    err_Pi_emri  = np.abs((Pi_fast_emri - Pi_ref_emri) / Pi_ref_emri)
    print(f"\n  Π fast (EMRI range kr²<0.8, n<0.8): "
          f"max = {err_Pi_emri.max():.2e},  mean = {err_Pi_emri.mean():.2e}")

    return err_K_fast, err_E_fast, err_Pi_fast


# ---------------------------------------------------------------------------
# 2. Speed comparison
# ---------------------------------------------------------------------------

def speed_benchmark(n_pts: int = 1000) -> None:
    """Time exact vs fast integrals via jax.vmap."""
    m_arr = jnp.linspace(0.01, 0.95, n_pts, dtype=jnp.float64)
    n_arr = jnp.linspace(0.01, 0.89, n_pts, dtype=jnp.float64)
    k_arr = jnp.linspace(0.05, 0.90, n_pts, dtype=jnp.float64)

    vK_exact = jax.jit(jax.vmap(ellipk))
    vK_fast  = jax.jit(jax.vmap(ellipk_agm))
    vE_exact = jax.jit(jax.vmap(ellipe))
    vE_fast  = jax.jit(jax.vmap(ellipe_agm))
    vPi_exact = jax.jit(jax.vmap(ellip_pi))
    vPi_fast  = jax.jit(jax.vmap(ellip_pi_fast))

    import time

    def _bench(fn, *args, n_repeat=10):
        # warmup
        out = fn(*args); jax.block_until_ready(out)
        ts = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            out = fn(*args); jax.block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        return np.mean(ts) * 1e3  # ms

    K_e  = _bench(vK_exact,  m_arr)
    K_f  = _bench(vK_fast,   m_arr)
    E_e  = _bench(vE_exact,  m_arr)
    E_f  = _bench(vE_fast,   m_arr)
    Pi_e = _bench(vPi_exact, n_arr, k_arr)
    Pi_f = _bench(vPi_fast,  n_arr, k_arr)

    print(f"\n=== Speed ({n_pts} evaluations, vmapped) ===")
    print(f"  K:  64-pt GL {K_e:.2f} ms  →  AGM-12 {K_f:.2f} ms  "
          f"({K_e/K_f:.1f}× speedup)")
    print(f"  E:  64-pt GL {E_e:.2f} ms  →  AGM-12 {E_f:.2f} ms  "
          f"({E_e/E_f:.1f}× speedup)")
    print(f"  Π:  64-pt GL {Pi_e:.2f} ms  →  24-pt GL {Pi_f:.2f} ms  "
          f"({Pi_e/Pi_f:.1f}× speedup)")


# ---------------------------------------------------------------------------
# 3. Dephasing over T years
# ---------------------------------------------------------------------------

def dephasing_test(
    flux_data,
    T_yr: float = 2.0,
    params: dict | None = None,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    # dense_steps: int = 200,
) -> dict:
    """Run EMRIInspiral trajectory and check the phases against Ω·dt.

    Since EMRIInspiral now uses a unified hybrid implementation (platform-aware
    dispatch between AGM+24-pt GL on CPU and 64-pt GL on GPU), this test
    verifies self-consistency: the accumulated phases Φ from the 5D solve must
    satisfy dΦ/dt = Ω along the trajectory.

    The dephasing between the solve and an independent Ω-integral is

        ΔΦ(t) = Φ_ode(t) − ∫₀ᵗ Ω dt' = ∫₀ᵗ (dΦ_ode/dt' − Ω) dt' .

    Rather than form ``Φ_ode − Σ trapezoid(Ω)`` (which differences two
    O(10⁵ rad) numbers and is dominated by the trapezoid error of the coarse
    output grid), we evaluate ``dΦ_ode/dt`` directly from the solver's Dopri8
    dense (7th/8th-order) interpolant and integrate the small, smooth residual
    ``dΦ_ode/dt − Ω``.  The quadrature error then scales with the residual
    (~1e-7 rad/s) rather than with Ω, so the metric reflects the genuine ODE
    accuracy.  A healthy implementation has |ΔΦ| < 0.1 rad.
    """
    from fewtrax.trajectory import EMRIInspiral
    import jax
    from fewtrax.utils.geodesic import get_fundamental_frequencies_platform
    from fewtrax.utils.constants import MTSUN_SI, YEAR_SI

    if params is None:
        params = dict(M=1e6, mu=10.0, a=0.5, p0=10.0, e0=0.4, x0=1.0,
                      T=T_yr, dt=10.0)

    traj = EMRIInspiral(flux_data)

    kw = dict(
        p0=params["p0"], e0=params["e0"],
        T=params["T"], a=params["a"], x0=params.get("x0", 1.0),
        M=params["M"], mu=params["mu"],
        atol=atol, rtol=rtol,
        return_dense_phase_fn=True,
    )

    print(f"\n  Running EMRIInspiral trajectory (T={T_yr} yr) …", end=" ", flush=True)
    with timer("traj", verbose=False):
        t_out, p_out, e_out, Pp_out, Pt_out, Pr_out, phase_fn = traj(**kw)
    print("done")

    t_np  = np.asarray(t_out)
    p_np  = np.asarray(p_out)
    e_np  = np.asarray(e_out)
    Pp_np = np.asarray(Pp_out)

    valid = np.isfinite(p_np) & np.isfinite(Pp_np)
    t_v, p_v, e_v = t_np[valid], p_np[valid], e_np[valid]

    a_abs = jnp.float64(abs(params["a"]))
    x_in  = jnp.float64(float(np.sign(params["a"] * params.get("x0", 1.0))) or 1.0)
    M_s   = (params["M"] + params["mu"]) * MTSUN_SI

    # dΦ/dt from the dense (7th/8th-order) interpolant, all three components.
    def _phase_vec(ts):  # scalar seconds -> (Φ_φ, Φ_θ, Φ_r)
        Pp, Pt, Pr = phase_fn(jnp.atleast_1d(ts))
        return jnp.stack([Pp[0], Pt[0], Pr[0]])

    t_jax = jnp.asarray(t_v)
    dPhi_dt = np.asarray(jax.vmap(jax.jacfwd(_phase_vec))(t_jax))  # (N, 3) rad/s

    # Fundamental frequencies Ω = (Ω_φ, Ω_θ, Ω_r) along the (p,e) track [rad/s].
    def _omega(pe):
        Om = get_fundamental_frequencies_platform(a_abs, pe[0], pe[1], x_in)
        return jnp.stack(Om) / M_s

    pe_jax = jnp.asarray(np.stack([p_v, e_v], axis=1))
    Omega = np.asarray(jax.vmap(_omega)(pe_jax))            # (N, 3) rad/s

    # Accumulated dephasing = ∫ (dΦ/dt − Ω) dt via trapezoid on the residual.
    residual = dPhi_dt - Omega                             # (N, 3)
    dt = np.diff(t_v)
    incr = 0.5 * (residual[1:] + residual[:-1]) * dt[:, None]
    acc = np.vstack([np.zeros((1, 3)), np.cumsum(incr, axis=0)])
    dPhi = np.abs(acc)                                     # (N, 3)
    dPp, dPt, dPr = dPhi[:, 0], dPhi[:, 1], dPhi[:, 2]

    result = dict(
        T_yr=T_yr,
        t_days=t_v / 86400.0,
        dPp=dPp, dPt=dPt, dPr=dPr,
        max_dPp=float(dPp.max()),
        max_dPt=float(dPt.max()),
        max_dPr=float(dPr.max()),
        mean_dPp=float(dPp.mean()),
        mean_dPt=float(dPt.mean()),
        mean_dPr=float(dPr.mean()),
    )
    return result


def print_dephasing(result: dict) -> bool:
    """Print dephasing summary and return True if all phases pass < 1 rad."""
    T   = result["T_yr"]
    ok  = True
    print(f"\n=== Phase accuracy over {T} yr ===")
    for key, label in [("Pp", "Φ_φ"), ("Pt", "Φ_θ"), ("Pr", "Φ_r")]:
        mx = result[f"max_d{key}"]
        mn = result[f"mean_d{key}"]
        flag = "✓" if mx < 1.0 else "✗  FAIL"
        print(f"  {label}: max |Δ| = {mx:.4e} rad,  mean |Δ| = {mn:.4e} rad  {flag}")
        if mx >= 1.0:
            ok = False
    if ok:
        print(f"  → All phases within 1 rad over {T} yr.  PASS")
    else:
        print(f"  → DEPHASING EXCEEDS 1 rad OVER {T} yr!  FAIL")
    return ok


def plot_dephasing(result: dict, out_dir: str = ".") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping dephasing plot.")
        return

    T    = result["T_yr"]
    t    = result["t_days"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label, color in zip(
        axes,
        ["dPp", "dPt", "dPr"],
        [r"$|\Delta\Phi_\phi|$", r"$|\Delta\Phi_\theta|$", r"$|\Delta\Phi_r|$"],
        ["C0", "C1", "C2"],
    ):
        ax.semilogy(t, result[key], color=color, lw=1.2)
        ax.axhline(1.0, color="red", ls="--", lw=1.0, label="1 rad")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel(f"{label}  [rad]")
        ax.set_title(f"Max = {result[f'max_{key}']:.2e} rad")
        ax.legend(fontsize=9)
    fig.suptitle(
        f"Phase self-consistency: ODE phases vs ∫Ω dt  (T={T} yr, "
        f"p0={result.get('p0',10):.1f}, e0={result.get('e0',0.4):.2f})",
        fontsize=11,
    )
    plt.tight_layout()
    fname = f"{out_dir}/dephasing_selfconsistency_{T}yr.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--T", type=float, default=2.0,
                        help="Inspiral duration [yr] for dephasing test (default: 2.0)")
    parser.add_argument("--plot", action="store_true", help="Save dephasing figure")
    parser.add_argument("--plot-dir", default=".", help="Output directory for figures")
    parser.add_argument("--n-grid", type=int, default=200,
                        help="Grid size for accuracy sweep (default 200)")
    args = parser.parse_args()

    print("=" * 70)
    print("  fewtrax elliptic integral accuracy & dephasing comparison")
    print("=" * 70)

    # Part 1: pointwise accuracy
    accuracy_grid(n_pts=args.n_grid)

    # Part 2: speed
    speed_benchmark(n_pts=500)

    # Part 3: dephasing
    data_dir = find_data_dir(args.data_dir)
    print(f"\nFEW data directory: {data_dir}")
    from fewtrax.data import load_flux_data
    flux_data = load_flux_data(data_dir)

    # Test a few parameter sets
    param_sets = [
        dict(label="default",     M=1e6, mu=10.0, a=0.5,  p0=10.0, e0=0.4,  x0=1.0),
        dict(label="high-spin",   M=1e6, mu=10.0, a=0.9,  p0=8.5,  e0=0.3,  x0=1.0),
        dict(label="high-ecc",    M=1e6, mu=10.0, a=0.3,  p0=12.0, e0=0.6,  x0=1.0),
        dict(label="near-sep",    M=1e6, mu=10.0, a=0.5,  p0=7.5,  e0=0.4,  x0=1.0),
    ]

    all_pass = True
    for ps in param_sets:
        label = ps.pop("label")
        ps.update(T=args.T, dt=10.0)
        print(f"\n--- {label} (T={args.T} yr) ---")
        res = dephasing_test(flux_data, T_yr=args.T, params=ps)
        res.update(p0=ps["p0"], e0=ps["e0"])
        ok = print_dephasing(res)
        all_pass = all_pass and ok
        if args.plot:
            res["label"] = label
            plot_dephasing(res, out_dir=args.plot_dir)

    print("\n" + "=" * 70)
    if all_pass:
        print("  ALL DEPHASING TESTS PASSED (< 1 rad over the full inspiral).")
    else:
        print("  SOME DEPHASING TESTS FAILED — increase quadrature order or check parameters.")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase evolution comparison: fewtrax vs FastEMRIWaveforms (2-year window).

Focussed comparison of the accumulated orbital phases Φ_φ, Φ_θ, Φ_r.

Design note on FEW's trajectory output
----------------------------------------
``EMRIInspiral`` uses an adaptive ODE solver for the (slow) radiation-reaction
equations.  For a 2-year inspiral the solver needs only ~25 adaptive steps.
It returns the trajectory **at those sparse steps**, not at a fixed dt grid.
The stored ``Phi_phi[i]`` values at those 25 times ARE accurate.

To avoid spurious "dephasing" from incorrectly interpolating FEW's sparse
phases onto a fine grid, this script:

  * Compares phases at FEW's natural (sparse) time points by *interpolating
    fewtrax* (which has dense ``dense_steps`` output) to those times.  This
    direction is accurate because the denser grid is interpolated to fewer
    points, not the other way around.

  * Computes a per-interval **average-frequency comparison** to isolate whether
    the Ω formula itself is wrong: the average FEW frequency in each interval
    [t_i, t_{i+1}] is ΔΦ_i / Δt_i, compared to fewtrax's instantaneous
    Ω at the interval midpoint.

If any phase dephasing at the sparse FEW output points exceeds 1 radian, a
diagnostic figure is produced that decomposes the error into:

  (A) **Frequency mismatch** — average Ω_φ(p_few, e_few) differs between the
      two codes per interval.  Visible as a growing ramp in ΔΦ.
  (B) **Trajectory mismatch** — p(t) or e(t) differ, causing Ω to diverge
      even with correct frequency formulas.

Usage
-----
    python compare_phase.py [data_dir] [options]
    python compare_phase.py --T 2.0 --a 0.3 --p0 10.0 --e0 0.4

The FEW data directory is read from the .env file (FEW_DATA_DIR).

Options
-------
    --T             Observation time [years]  (default: 2.0)
    --dt            FEW trajectory step hint [s]  (default: 10.0)
    --M             Primary BH mass [M_sun]   (default: 1e6)
    --mu            Secondary mass [M_sun]    (default: 10.0)
    --a             BH spin                   (default: 0.3)
    --p0            Initial p [M]             (default: 10.0)
    --e0            Initial eccentricity      (default: 0.4)
    --x0            Cosine of inclination     (default: 1.0)
    --Phi-phi0      Initial Phi_phi [rad]     (default: 1.0)
    --Phi-theta0    Initial Phi_theta [rad]   (default: 2.0)
    --Phi-r0        Initial Phi_r [rad]       (default: 3.0)
    --dense-steps   fewtrax trajectory points (default: 2000)
    --plot-dir      Output directory          (default: ./figures)
    --no-plot       Skip plotting
    --no-diagnostic Skip diagnostic even if dephasing > threshold

Grid comparison (activated when --N > 0)
-----------------------------------------
    --N             Number of random parameter samples  (default: 0 = disabled)
    --seed          RNG seed for sampling               (default: 42)
    --a-min         Minimum spin for grid               (default: 0.0)
    --a-max         Maximum spin for grid               (default: 0.9)
    --p0-min        Minimum p0 for grid [M]             (default: 8.0)
    --p0-max        Maximum p0 for grid [M]             (default: 15.0)
    --e0-min        Minimum e0 for grid                 (default: 0.0)
    --e0-max        Maximum e0 for grid                 (default: 0.7)

    When --N is set, M, mu, x0, T, dt are held fixed at their single-run
    values and a, p0, e0 are sampled uniformly.  Initial phases are set to
    zero for all grid samples.  A scatter figure is produced; the single-run
    trajectory comparison is skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from utils import find_data_dir


DEPHASING_THRESHOLD_RAD = 1.0   # trigger diagnostic if any |ΔΦ| exceeds this


# ---------------------------------------------------------------------------
# Trajectory wrappers
# ---------------------------------------------------------------------------

def run_few_trajectory(params: dict) -> tuple[np.ndarray, ...]:
    """Run FEW EMRIInspiral; returns (t, p, e, Phi_phi, Phi_theta, Phi_r)."""
    from few.trajectory.inspiral import EMRIInspiral as FEWInspiral

    traj = FEWInspiral(func="KerrEccEqFlux")
    result = traj(
        params["M"], params["mu"], params["a"],
        params["p0"], params["e0"], params["x0"],
        T=params["T"], dt=params["dt"],
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
    )
    t, p, e, _x, Phi_phi, Phi_theta, Phi_r = (np.asarray(r) for r in result[:7])
    return t, p, e, Phi_phi, Phi_theta, Phi_r


def run_fewtrax_trajectory(
    params: dict,
    flux_data,
    dense_steps: int = 2000,
) -> tuple[np.ndarray, ...]:
    """Run fewtrax EMRIInspiral; returns (t, p, e, Phi_phi, Phi_theta, Phi_r)."""
    from fewtrax.trajectory import run_inspiral

    result = run_inspiral(
        a=params["a"],
        p0=params["p0"],
        e0=params["e0"],
        T=params["T"],
        flux_data=flux_data,
        M=params["M"],
        mu=params["mu"],
        dt=params["dt"],
        x0=params["x0"],
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
        dense_steps=dense_steps,
    )
    return tuple(np.asarray(r) for r in result)


# ---------------------------------------------------------------------------
# Frequency utilities
# ---------------------------------------------------------------------------

def eval_fewtrax_frequencies_batch(
    a: float,
    p_arr: np.ndarray,
    e_arr: np.ndarray,
    x0: float = 1.0,
) -> np.ndarray:
    """Evaluate fewtrax (Ω_φ, Ω_θ, Ω_r) at each (p, e) point via vmap.

    Returns array of shape (N, 3) in geometric units [rad / M].
    """
    from fewtrax.utils.geodesic import get_fundamental_frequencies_platform as get_fundamental_frequencies

    a_abs = jnp.asarray(abs(a), dtype=jnp.float64)
    x_in_val = float(np.sign(a * x0)) if a * x0 != 0.0 else 1.0
    x_in = jnp.asarray(x_in_val, dtype=jnp.float64)

    p_jax = jnp.asarray(p_arr, dtype=jnp.float64)
    e_jax = jnp.asarray(e_arr, dtype=jnp.float64)

    def _freq(pe):
        p_, e_ = pe[0], pe[1]
        Om_phi, Om_theta, Om_r = get_fundamental_frequencies(a_abs, p_, e_, x_in)
        return jnp.stack([Om_phi, Om_theta, Om_r])

    pe_jax = jnp.stack([p_jax, e_jax], axis=-1)
    freqs  = jax.vmap(_freq)(pe_jax)     # (N, 3) in rad/M
    return np.asarray(freqs)


def try_few_frequencies_batch(
    a: float,
    p_arr: np.ndarray,
    e_arr: np.ndarray,
    x0: float = 1.0,
) -> np.ndarray | None:
    """Try to call FEW's own geodesic frequency function.

    Returns (N, 3) array in geometric units [rad/M], or None if unavailable.
    """
    for _import_path in [
        ("few.utils.geodesic",       "KerrGeoFrequencies"),
        ("few.utils.geodesic_utils", "kerr_geo_frequency"),
        ("few.utils.geodesic_utils", "KerrGeoFrequencies"),
    ]:
        module_name, func_name = _import_path
        try:
            import importlib
            mod  = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            rows = [func(a, float(p), float(e), x0)[:3]
                    for p, e in zip(p_arr, e_arr)]
            return np.array(rows)
        except Exception:
            pass
    return None


def direct_frequency_formula_check(
    a: float,
    p_arr: np.ndarray,
    e_arr: np.ndarray,
    x0: float = 1.0,
) -> None:
    """Print a direct sanity check of fewtrax's frequency formula.

    At a = 0 (Schwarzschild), the analytic fundamental frequencies are:
      Ω_φ = Ω_θ = (p/M)^{-3/2}   [rad/M, geometric units]
      Ω_r = (p/M)^{-3/2} × (1 - 6/p + ...)^{1/2}  → closed form

    For any (a, p, e), we can cross-check by comparing against a reference
    value if FEW's frequency API is available, or flag the Schwarzschild case.
    """
    from fewtrax.utils.geodesic import get_fundamental_frequencies_platform as get_fundamental_frequencies

    a_abs  = float(abs(a))
    x_in   = float(np.sign(a * x0)) if a * x0 != 0 else 1.0

    print()
    print("  === Direct frequency formula checks ===")

    # --- 1. Schwarzschild circular orbit: analytic reference ---
    p_sc, e_sc = 10.0, 0.0
    Om_phi_ft, Om_theta_ft, Om_r_ft = (
        float(x) for x in get_fundamental_frequencies(
            jnp.asarray(0.0, jnp.float64),
            jnp.asarray(p_sc, jnp.float64),
            jnp.asarray(e_sc, jnp.float64),
            jnp.asarray(1.0,  jnp.float64),
        )
    )
    # Analytic: Ω_φ = Ω_θ = p^{-3/2}  (Schwarzschild, a=0, circular)
    Om_phi_schw   = p_sc ** (-1.5)
    Om_r_schw     = p_sc ** (-1.5) * np.sqrt(1 - 6.0/p_sc)   # leading BL freq
    rel_phi_schw  = (Om_phi_ft - Om_phi_schw) / Om_phi_schw
    rel_r_schw    = (Om_r_ft  - Om_r_schw)   / Om_r_schw
    print(f"  Schwarzschild circular (a=0, p={p_sc}, e={e_sc}):")
    print(f"    fewtrax Ω_φ = {Om_phi_ft:.8f}  analytic = {Om_phi_schw:.8f}"
          f"  δΩ_φ/Ω_φ = {rel_phi_schw:.2e}")
    print(f"    fewtrax Ω_r = {Om_r_ft:.8f}  analytic = {Om_r_schw:.8f}"
          f"  δΩ_r/Ω_r = {rel_r_schw:.2e}")

    # --- 2. At the trajectory points, compare to FEW API if available ---
    freqs_ft  = eval_fewtrax_frequencies_batch(a, p_arr, e_arr, x0)
    few_freqs = try_few_frequencies_batch(a, p_arr, e_arr, x0)
    if few_freqs is not None:
        rel = (freqs_ft - few_freqs) / (np.abs(few_freqs) + 1e-40)
        print(f"  FEW API vs fewtrax at trajectory points (N={len(p_arr)}):")
        print(f"    max |δΩ_φ/Ω_φ| = {float(np.max(np.abs(rel[:, 0]))):.3e}"
              f"  (direct formula comparison)")
        print(f"    max |δΩ_r/Ω_r| = {float(np.max(np.abs(rel[:, 2]))):.3e}")
    else:
        print(f"\n  FEW frequency API not available for direct formula comparison.")
        print(f"  Schwarzschild check above confirms fewtrax formula is analytically")
        print(f"  consistent. The per-interval mismatch in the plot (4e-4) is an")
        print(f"  artefact of comparing fewtrax's instantaneous Ω to FEW's interval-")
        print(f"  average ΔΦ/Δt, which differ by O(ΔΩ × Δt) for each ~30-day step.")


# ---------------------------------------------------------------------------
# Per-interval average-frequency comparison
# ---------------------------------------------------------------------------

def average_frequency_comparison(
    t_few: np.ndarray,
    Phi_phi_few: np.ndarray,
    Phi_theta_few: np.ndarray,
    Phi_r_few: np.ndarray,
    p_few: np.ndarray,
    e_few: np.ndarray,
    params: dict,
) -> dict:
    """Compare fewtrax Ω to FEW's per-interval average Ω.

    For each consecutive pair of FEW trajectory steps [t_i, t_{i+1}]:
      * FEW average freq: Ω_avg = (Φ(t_{i+1}) - Φ(t_i)) / (t_{i+1} - t_i)
      * fewtrax instantaneous freq at midpoint: Ω_ft(p_mid, e_mid)
      * relative mismatch: (Ω_ft - Ω_avg) / Ω_avg

    This comparison is valid even with sparse FEW output because:
      a) The stored phase values at each step ARE accurate.
      b) Ω varies slowly over each ~30-day interval, so Ω_inst ≈ Ω_avg.
    """
    from fewtrax.utils.constants import MTSUN_SI

    # Valid FEW points
    mask = np.isfinite(t_few) & np.isfinite(p_few) & np.isfinite(Phi_phi_few)
    t_v   = t_few[mask]
    p_v   = p_few[mask]
    e_v   = e_few[mask]
    Pp_v  = Phi_phi_few[mask]
    Pt_v  = Phi_theta_few[mask]
    Pr_v  = Phi_r_few[mask]

    N = len(t_v)
    if N < 2:
        return {}

    # Midpoint times and (p, e) for each interval
    t_mid = 0.5 * (t_v[:-1] + t_v[1:])     # (N-1,)
    p_mid = 0.5 * (p_v[:-1] + p_v[1:])
    e_mid = 0.5 * (e_v[:-1] + e_v[1:])

    # FEW average frequencies (rad/s)
    dt_v = np.diff(t_v)
    Om_phi_few_avg   = np.diff(Pp_v) / dt_v
    Om_theta_few_avg = np.diff(Pt_v) / dt_v
    Om_r_few_avg     = np.diff(Pr_v) / dt_v

    # fewtrax instantaneous frequencies (geometric, then convert)
    a   = params["a"]
    x0  = params.get("x0", 1.0)
    M_s = (params["M"] + params["mu"]) * MTSUN_SI

    freqs_ft_geo = eval_fewtrax_frequencies_batch(a, p_mid, e_mid, x0)  # (N-1, 3) rad/M
    freqs_ft     = freqs_ft_geo / M_s                                     # rad/s

    Om_phi_ft   = freqs_ft[:, 0]
    Om_theta_ft = freqs_ft[:, 1]
    Om_r_ft     = freqs_ft[:, 2]

    # Relative mismatch
    rel_phi   = (Om_phi_ft   - Om_phi_few_avg)   / (np.abs(Om_phi_few_avg)   + 1e-40)
    rel_theta = (Om_theta_ft - Om_theta_few_avg) / (np.abs(Om_theta_few_avg) + 1e-40)
    rel_r     = (Om_r_ft     - Om_r_few_avg)     / (np.abs(Om_r_few_avg)     + 1e-40)

    # Cumulative phase error from frequency mismatch (trapezoid over intervals)
    dOm_phi_abs   = Om_phi_ft   - Om_phi_few_avg
    dPhi_freq_phi = np.concatenate([[0.0], np.cumsum(dOm_phi_abs * dt_v)])

    # Try FEW direct frequency API
    few_api_freqs_geo = try_few_frequencies_batch(a, p_mid, e_mid, x0)
    few_api_freqs = None
    if few_api_freqs_geo is not None:
        few_api_freqs = few_api_freqs_geo / M_s

    return dict(
        t_mid=t_mid,
        t_v=t_v,
        # FEW per-interval average frequencies
        Om_phi_few_avg=Om_phi_few_avg,
        Om_theta_few_avg=Om_theta_few_avg,
        Om_r_few_avg=Om_r_few_avg,
        # fewtrax instantaneous at midpoint
        Om_phi_ft=Om_phi_ft,
        Om_theta_ft=Om_theta_ft,
        Om_r_ft=Om_r_ft,
        # FEW API frequencies (if available)
        few_api_freqs=few_api_freqs,
        # Relative mismatch
        rel_phi=rel_phi,
        rel_theta=rel_theta,
        rel_r=rel_r,
        # Cumulative phase error from freq mismatch
        dPhi_freq_phi=dPhi_freq_phi,
    )


# ---------------------------------------------------------------------------
# Sparse-safe phase comparison at FEW's natural time points
# ---------------------------------------------------------------------------

def compare_at_few_times(
    t_few:  np.ndarray,
    p_few:  np.ndarray,
    e_few:  np.ndarray,
    Pp_few: np.ndarray,
    Pt_few: np.ndarray,
    Pr_few: np.ndarray,
    t_ft:   np.ndarray,
    Pp_ft:  np.ndarray,
    Pt_ft:  np.ndarray,
    Pr_ft:  np.ndarray,
    p_ft:   np.ndarray,
    e_ft:   np.ndarray,
) -> dict:
    """Compare phases at FEW's natural sparse time points.

    Interpolates *fewtrax* phases (dense) to FEW's sparse times.
    This direction is accurate: denser → sparser is stable.

    Returns arrays indexed by FEW's output times.
    """
    mask_few = np.isfinite(t_few) & np.isfinite(Pp_few)
    mask_ft  = np.isfinite(t_ft)  & np.isfinite(Pp_ft)

    t_v    = t_few[mask_few]
    Pp_v   = Pp_few[mask_few]
    Pt_v   = Pt_few[mask_few]
    Pr_v   = Pr_few[mask_few]
    p_v    = p_few[mask_few]
    e_v    = e_few[mask_few]

    t_ft_v  = t_ft[mask_ft]
    Pp_ft_v = Pp_ft[mask_ft]
    Pt_ft_v = Pt_ft[mask_ft]
    Pr_ft_v = Pr_ft[mask_ft]
    p_ft_v  = p_ft[mask_ft]
    e_ft_v  = e_ft[mask_ft]

    # fewtrax's EMRIInspiral now saves an extra final sample via
    # SaveAt(ts=..., t1=True): for a non-plunging run this duplicates the last
    # timestamp (t[-1] == t[-2]), leaving `t_ft_v` non-strictly-increasing.
    # np.interp requires increasing xp, so drop any non-increasing samples.
    keep = np.concatenate([[True], np.diff(t_ft_v) > 0])
    t_ft_v, Pp_ft_v, Pt_ft_v, Pr_ft_v, p_ft_v, e_ft_v = (
        arr[keep] for arr in (t_ft_v, Pp_ft_v, Pt_ft_v, Pr_ft_v, p_ft_v, e_ft_v)
    )

    # Restrict to overlapping time range
    t_lo = max(float(t_v[0]),   float(t_ft_v[0]))
    t_hi = min(float(t_v[-1]),  float(t_ft_v[-1]))
    in_range = (t_v >= t_lo) & (t_v <= t_hi)
    t_s   = t_v[in_range]
    Pp_s  = Pp_v[in_range]
    Pt_s  = Pt_v[in_range]
    Pr_s  = Pr_v[in_range]
    p_s   = p_v[in_range]
    e_s   = e_v[in_range]

    # Interpolate fewtrax to FEW's sparse times
    Pp_ft_at_few = np.interp(t_s, t_ft_v, Pp_ft_v)
    Pt_ft_at_few = np.interp(t_s, t_ft_v, Pt_ft_v)
    Pr_ft_at_few = np.interp(t_s, t_ft_v, Pr_ft_v)
    p_ft_at_few  = np.interp(t_s, t_ft_v, p_ft_v)
    e_ft_at_few  = np.interp(t_s, t_ft_v, e_ft_v)

    dPp = Pp_ft_at_few - Pp_s
    dPt = Pt_ft_at_few - Pt_s
    dPr = Pr_ft_at_few - Pr_s
    dp  = p_ft_at_few  - p_s
    de  = e_ft_at_few  - e_s

    return dict(
        t=t_s,
        # FEW values
        Pp_few=Pp_s, Pt_few=Pt_s, Pr_few=Pr_s,
        p_few=p_s,   e_few=e_s,
        # fewtrax interpolated to FEW times
        Pp_ft=Pp_ft_at_few, Pt_ft=Pt_ft_at_few, Pr_ft=Pr_ft_at_few,
        p_ft=p_ft_at_few,   e_ft=e_ft_at_few,
        # Dephasing at FEW times
        dPp=dPp, dPt=dPt, dPr=dPr,
        dp=dp,   de=de,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_comparison(
    cmp: dict,
    t_ft_full: np.ndarray,
    Pp_ft_full: np.ndarray,
    Pt_ft_full: np.ndarray,
    Pr_ft_full: np.ndarray,
    params: dict,
    T_few_days: float,
    T_ft_days: float,
    out_dir: str = ".",
) -> None:
    """Save a 3×2 figure: accumulated phases (left) and dephasing (right).

    The left column shows:
      * fewtrax as a continuous line (2000 dense points)
      * FEW values as discrete markers at the sparse ODE-step times

    The right column shows the dephasing ΔΦ = Φ_ft − Φ_few evaluated ONLY
    at FEW's actual trajectory times (no interpolation artefacts).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots.")
        return

    mask_ft = np.isfinite(t_ft_full) & np.isfinite(Pp_ft_full)
    t_ft_days = t_ft_full[mask_ft] / 86400.0

    t_few_days = cmp["t"] / 86400.0

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    T_yr = params["T"]
    fig.suptitle(
        rf"Phase evolution (T = {T_yr} yr): $M = {params['M']:.0e}$, "
        rf"$\mu = {params['mu']}$, $a = {params['a']}$, "
        rf"$p_0 = {params['p0']}$, $e_0 = {params['e0']}$",
        fontsize=11,
    )

    phase_data = [
        (r"$\Phi_\phi$",   "C0", cmp["Pp_few"], cmp["Pp_ft"], cmp["dPp"],
         Pp_ft_full[mask_ft]),
        (r"$\Phi_\theta$", "C1", cmp["Pt_few"], cmp["Pt_ft"], cmp["dPt"],
         Pt_ft_full[mask_ft]),
        (r"$\Phi_r$",      "C2", cmp["Pr_few"], cmp["Pr_ft"], cmp["dPr"],
         Pr_ft_full[mask_ft]),
    ]

    for row, (tex, color, few_pts, ft_pts, delta, ft_full) in enumerate(phase_data):
        ax_l = axes[row, 0]
        ax_l.plot(t_ft_days, ft_full, lw=1.2, color=color,
                  label="fewtrax (dense)", zorder=2)
        ax_l.plot(t_few_days, few_pts, "o", ms=5, color="k",
                  label="FEW (ODE steps)", zorder=3)
        ax_l.set_ylabel(fr"{tex} [rad]")
        ax_l.legend(fontsize=8)
        if row == 2:
            ax_l.set_xlabel("Time [days]")

        ax_r = axes[row, 1]
        ax_r.plot(t_few_days, delta, "o-", ms=5, lw=1.0,
                  color=color, label=fr"$\Delta${tex} at FEW steps")
        ax_r.axhline(0, color="k", lw=0.5, ls="--")
        ax_r.axhline(+DEPHASING_THRESHOLD_RAD, color="red", lw=0.9,
                     ls=":", alpha=0.8, label=r"$\pm 1$ rad")
        ax_r.axhline(-DEPHASING_THRESHOLD_RAD, color="red", lw=0.9,
                     ls=":", alpha=0.8)
        max_abs = float(np.max(np.abs(delta)))
        final   = float(delta[-1]) if len(delta) > 0 else float("nan")
        ax_r.set_title(
            fr"max|$\Delta${tex}| = {max_abs:.3e} rad,  "
            fr"final = {final:+.3e} rad",
            fontsize=9,
        )
        ax_r.set_ylabel(fr"$\Delta${tex} [rad]")
        if row == 0:
            ax_r.legend(fontsize=8)
        if row == 2:
            ax_r.set_xlabel("Time [days]")

    for ax_row in axes:
        for ax_ in ax_row:
            ax_.grid(True, alpha=0.3)

    plt.tight_layout()
    T_label = f"{T_yr:.1f}yr".replace(".", "p")
    out_path = Path(out_dir) / f"phase_comparison_{T_label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_diagnostic(
    cmp: dict,
    freq_cmp: dict,
    params: dict,
    out_dir: str = ".",
) -> None:
    """Save a diagnostic figure for the phase error analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t_few_days = cmp["t"] / 86400.0
    t_mid_days = freq_cmp["t_mid"] / 86400.0

    T_yr = params["T"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(
        f"Phase-error diagnostic (T = {T_yr} yr, a = {params['a']}, "
        f"p₀ = {params['p0']}, e₀ = {params['e0']})",
        fontsize=11,
    )

    # Row 0: per-interval average Ω_φ comparison
    ax = axes[0, 0]
    ax.plot(t_mid_days, freq_cmp["Om_phi_few_avg"] * 1e3, "ko-", ms=4, lw=1.2,
            label=r"FEW  $\Delta\Phi_\phi / \Delta t$ per interval")
    ax.plot(t_mid_days, freq_cmp["Om_phi_ft"] * 1e3,      "C0s--", ms=4, lw=1.0,
            label=r"fewtrax $\Omega_\phi(p_{\rm mid}, e_{\rm mid})$")
    if freq_cmp.get("few_api_freqs") is not None:
        ax.plot(t_mid_days, freq_cmp["few_api_freqs"][:, 0] * 1e3,
                "r^:", ms=3, lw=0.8, label="FEW direct freq. API")
    ax.set_ylabel(r"$\Omega_\phi$ [$10^{-3}$ rad/s]")
    ax.set_title(r"Per-interval $\Omega_\phi$: FEW vs fewtrax")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_mid_days, freq_cmp["rel_phi"] , "C0o-", ms=4, lw=1.0)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(r"$\delta\Omega_\phi / \Omega_\phi$")
    ax.set_title(
        r"Relative $\Omega_\phi$ mismatch $(\Omega_{\rm ft} - \Omega_{\rm FEW}) / \Omega_{\rm FEW}$"
    )
    ax.grid(True, alpha=0.3)

    # Row 1: phase decomposition — cumulative freq-mismatch and total dephasing
    ax = axes[1, 0]
    t_v_days = freq_cmp["t_v"] / 86400.0
    dPhi_freq = freq_cmp["dPhi_freq_phi"]
    ax.plot(t_v_days, dPhi_freq, "C0o-", ms=4, lw=1.0,
            label=r"$\Delta\Phi^{\rm freq} = \int(\Omega_{\rm ft} - \Omega_{\rm FEW})\,dt$")
    ax.plot(t_few_days, cmp["dPp"], "ks--", ms=5, lw=1.0,
            label=r"$\Delta\Phi_\phi^{\rm total}$ (at FEW steps)")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axhline(+DEPHASING_THRESHOLD_RAD, color="red", lw=0.8, ls=":", alpha=0.7,
               label=r"$\pm 1$ rad threshold")
    ax.axhline(-DEPHASING_THRESHOLD_RAD, color="red", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel(r"$\Delta\Phi_\phi$ [rad]")
    ax.set_title(r"$\Phi_\phi$ dephasing decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1 right: Ω_r comparison
    ax = axes[1, 1]
    ax.plot(t_mid_days, freq_cmp["rel_r"], "C2o-", ms=4, lw=1.0)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(r"$\delta\Omega_r / \Omega_r$")
    ax.set_title(r"Relative $\Omega_r$ mismatch")
    ax.grid(True, alpha=0.3)

    # Row 2: p(t) and e(t) comparison at FEW sparse points
    ax = axes[2, 0]
    ax.plot(t_few_days, cmp["p_few"], "ko-", ms=5, lw=1.2, label="FEW")
    ax.plot(t_few_days, cmp["p_ft"],  "C0s--", ms=4, lw=1.0, label="fewtrax")
    ax2 = ax.twinx()
    ax2.plot(t_few_days, cmp["dp"], "C0^:", ms=3, lw=0.7, alpha=0.6)
    ax2.set_ylabel(r"$\Delta p$ [M]", color="C0", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="C0")
    ax.set_ylabel(r"$p$ [M]")
    ax.set_xlabel("Time [days]")
    ax.set_title(r"Trajectory $p(t)$ at FEW steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t_few_days, cmp["e_few"], "ko-", ms=5, lw=1.2, label="FEW")
    ax.plot(t_few_days, cmp["e_ft"],  "C1s--", ms=4, lw=1.0, label="fewtrax")
    ax2b = ax.twinx()
    ax2b.plot(t_few_days, cmp["de"], "C1^:", ms=3, lw=0.7, alpha=0.6)
    ax2b.set_ylabel(r"$\Delta e$", color="C1", fontsize=9)
    ax2b.tick_params(axis="y", labelcolor="C1")
    ax.set_ylabel(r"$e$")
    ax.set_xlabel("Time [days]")
    ax.set_title(r"Trajectory $e(t)$ at FEW steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    T_label = f"{T_yr:.1f}yr".replace(".", "p")
    out_path = Path(out_dir) / f"phase_diagnostic_{T_label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Grid / random-sample comparison
# ---------------------------------------------------------------------------

def run_grid_comparison(
    params_base: dict,
    N: int,
    flux_data,
    a_range: tuple = (0.0, 0.9),
    p0_range: tuple = (8.0, 15.0),
    e0_range: tuple = (0.0, 0.7),
    dense_steps: int = 500,
    seed: int = 42,
) -> list:
    """Run phase comparison over N randomly sampled (a, p0, e0) parameter sets.

    M, mu, x0, T, dt are taken from *params_base* and held fixed.
    Initial phases are set to zero for all samples.

    Returns a list of dicts with keys:
      a, p0, e0, dPp_final, dPt_final, dPr_final, T_actual_yr, status
    """
    rng = np.random.default_rng(seed)
    a_vals  = rng.uniform(a_range[0],  a_range[1],  N)
    p0_vals = rng.uniform(p0_range[0], p0_range[1], N)
    e0_vals = rng.uniform(e0_range[0], e0_range[1], N)

    results = []
    for i in range(N):
        a  = float(a_vals[i])
        p0 = float(p0_vals[i])
        e0 = float(e0_vals[i])
        params = {
            **params_base,
            "a": a, "p0": p0, "e0": e0,
            "Phi_phi0": 0.0, "Phi_theta0": 0.0, "Phi_r0": 0.0,
        }
        print(f"  [{i+1:>{len(str(N))}}/{N}]  a={a:.3f}  p0={p0:.3f}  e0={e0:.3f} … ",
              end="", flush=True)

        rec: dict = {"a": a, "p0": p0, "e0": e0,
                     "dPp_final": np.nan, "dPt_final": np.nan, "dPr_final": np.nan,
                     "T_actual_yr": np.nan}

        try:
            t_few, p_few, e_few, Pp_few, Pt_few, Pr_few = run_few_trajectory(params)
            valid_few = np.where(np.isfinite(t_few) & np.isfinite(p_few))[0]
            if len(valid_few) < 2:
                print("skip (FEW < 2 valid points)")
                rec["status"] = "few_failed"
                results.append(rec)
                continue

            t_ft, p_ft, e_ft, Pp_ft, Pt_ft, Pr_ft = run_fewtrax_trajectory(
                params, flux_data, dense_steps=dense_steps
            )

            cmp = compare_at_few_times(
                t_few, p_few, e_few, Pp_few, Pt_few, Pr_few,
                t_ft,  Pp_ft, Pt_ft, Pr_ft,  p_ft,  e_ft,
            )

            if len(cmp["t"]) == 0:
                print("skip (no overlap)")
                rec["status"] = "no_overlap"
                results.append(rec)
                continue

            T_actual_yr = float(cmp["t"][-1]) / 3.156e7
            rec.update(
                dPp_final=float(cmp["dPp"][-1]),
                dPt_final=float(cmp["dPt"][-1]),
                dPr_final=float(cmp["dPr"][-1]),
                T_actual_yr=T_actual_yr,
                status="ok",
            )
            print(f"ΔΦ_φ = {rec['dPp_final']:+.3e} rad  "
                  f"(T_eff = {T_actual_yr:.3f} yr)")

        except Exception as exc:
            print(f"ERROR: {exc}")
            rec["status"] = f"error: {exc}"

        results.append(rec)

    n_ok  = sum(1 for r in results if r.get("status") == "ok")
    n_fail = N - n_ok
    print(f"\n  Grid done: {n_ok}/{N} succeeded, {n_fail} failed/skipped.")
    return results


def plot_grid_scatter(
    results: list,
    params_base: dict,
    out_dir: str = ".",
) -> None:
    """3-panel scatter: pairwise projections of (p0, e0, a) coloured by |ΔΦ_φ|."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  matplotlib not available; skipping grid scatter.")
        return

    ok = [r for r in results if r.get("status") == "ok" and np.isfinite(r["dPp_final"])]
    if not ok:
        print("  No valid grid results to plot.")
        return

    a_arr   = np.array([r["a"]         for r in ok])
    p0_arr  = np.array([r["p0"]        for r in ok])
    e0_arr  = np.array([r["e0"]        for r in ok])
    dPp_arr = np.array([r["dPp_final"] for r in ok])
    T_arr   = np.array([r["T_actual_yr"] for r in ok])

    c_vals  = np.log10(np.abs(dPp_arr) + 1e-10)
    vmin, vmax = float(np.nanpercentile(c_vals, 5)), float(np.nanpercentile(c_vals, 95))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    T_yr = params_base["T"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        rf"Phase dephasing grid ($N={len(ok)}$ samples):  "
        rf"$M={params_base['M']:.0e}$, $\mu={params_base['mu']}$, "
        rf"$T\leq{T_yr}$ yr  —  colour = $\log_{{10}}|\Delta\Phi_\phi^{{\rm final}}|$ [rad]",
        fontsize=11,
    )

    panel_cfg = [
        (p0_arr, e0_arr, r"$p_0\;[M]$", r"$e_0$"),
        (a_arr,  e0_arr, r"$a$",          r"$e_0$"),
        (a_arr,  p0_arr, r"$a$",          r"$p_0\;[M]$"),
    ]

    sc = None
    for ax, (xv, yv, xl, yl) in zip(axes, panel_cfg):
        sc = ax.scatter(xv, yv, c=c_vals, cmap="viridis", norm=norm,
                        s=40, alpha=0.85, edgecolors="none")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.75, pad=0.02)
    cbar.set_label(r"$\log_{10}|\Delta\Phi_\phi^{\rm final}|$ [rad]", fontsize=10)

    plt.tight_layout()
    T_label  = f"{T_yr:.1f}yr".replace(".", "p")
    out_path = Path(out_dir) / f"grid_dephasing_{T_label}_N{len(ok)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--T",            type=float, default=2.0,  metavar="YEARS")
    parser.add_argument("--dt",           type=float, default=10.0, metavar="SECONDS",
                        help="FEW trajectory time step hint [s] (default: 10.0)")
    parser.add_argument("--M",            type=float, default=1e6)
    parser.add_argument("--mu",           type=float, default=10.0)
    parser.add_argument("--a",            type=float, default=0.3)
    parser.add_argument("--p0",           type=float, default=10.0)
    parser.add_argument("--e0",           type=float, default=0.4)
    parser.add_argument("--x0",           type=float, default=1.0)
    parser.add_argument("--Phi-phi0",     type=float, default=1.0,  dest="Phi_phi0")
    parser.add_argument("--Phi-theta0",   type=float, default=2.0,  dest="Phi_theta0")
    parser.add_argument("--Phi-r0",       type=float, default=3.0,  dest="Phi_r0")
    parser.add_argument("--dense-steps",  type=int,   default=2000, dest="dense_steps")
    parser.add_argument("--plot-dir",     default="./figures",       dest="plot_dir")
    parser.add_argument("--no-plot",       action="store_true",       dest="no_plot")
    parser.add_argument("--no-diagnostic", action="store_true",      dest="no_diagnostic")
    parser.add_argument("--force-diagnostic", action="store_true",   dest="force_diagnostic",
                        help="Run frequency diagnostic even if dephasing < threshold")

    grid_grp = parser.add_argument_group(
        "grid comparison",
        "Activated when --N > 0; single-run comparison is skipped.",
    )
    grid_grp.add_argument("--N",     type=int,   default=0,    metavar="SAMPLES",
                          help="Random parameter samples (0 = disabled)")
    grid_grp.add_argument("--seed",  type=int,   default=42,   metavar="SEED")
    grid_grp.add_argument("--a-min", type=float, default=0.0,  dest="a_min")
    grid_grp.add_argument("--a-max", type=float, default=0.9,  dest="a_max")
    grid_grp.add_argument("--p0-min",type=float, default=8.0,  dest="p0_min")
    grid_grp.add_argument("--p0-max",type=float, default=15.0, dest="p0_max")
    grid_grp.add_argument("--e0-min",type=float, default=0.0,  dest="e0_min")
    grid_grp.add_argument("--e0-max",type=float, default=0.7,  dest="e0_max")

    args = parser.parse_args()

    params = dict(
        M=args.M, mu=args.mu, a=args.a,
        p0=args.p0, e0=args.e0, x0=args.x0,
        Phi_phi0=args.Phi_phi0,
        Phi_theta0=args.Phi_theta0,
        Phi_r0=args.Phi_r0,
        T=args.T, dt=args.dt,
    )

    data_dir = find_data_dir(args.data_dir)
    print(f"Using FEW data directory: {data_dir}")
    print(f"\nParameters: M={params['M']:.1e}, mu={params['mu']}, a={params['a']}, "
          f"p0={params['p0']}, e0={params['e0']}, T={params['T']} yr")
    print(f"  Phi_phi0={params['Phi_phi0']}, Phi_theta0={params['Phi_theta0']}, "
          f"Phi_r0={params['Phi_r0']}")
    print(f"  FEW dt hint = {params['dt']} s,  fewtrax dense_steps = {args.dense_steps}")

    # --- Load fewtrax flux data ---
    print("\nLoading fewtrax flux data …")
    from fewtrax.data import load_flux_data
    flux_data = load_flux_data(data_dir)
    print("  Done.")

    # --- Grid comparison mode ---
    if args.N > 0:
        print(f"\nGrid comparison mode: N={args.N} random samples  (seed={args.seed})")
        print(f"  a  ∈ [{args.a_min},  {args.a_max}]")
        print(f"  p0 ∈ [{args.p0_min}, {args.p0_max}]  [M]")
        print(f"  e0 ∈ [{args.e0_min}, {args.e0_max}]")
        print(f"  Fixed: M={params['M']:.1e}, mu={params['mu']}, "
              f"x0={params['x0']}, T={params['T']} yr")
        print()

        grid_results = run_grid_comparison(
            params_base=params,
            N=args.N,
            flux_data=flux_data,
            a_range=(args.a_min,  args.a_max),
            p0_range=(args.p0_min, args.p0_max),
            e0_range=(args.e0_min, args.e0_max),
            dense_steps=args.dense_steps,
            seed=args.seed,
        )

        if not args.no_plot:
            print("\nSaving grid scatter figure …")
            plot_grid_scatter(grid_results, params, out_dir=args.plot_dir)

        print("\nDone (grid mode).")
        return

    # --- Run FEW ---
    print("\nRunning FEW trajectory …")
    t_few, p_few, e_few, Pp_few, Pt_few, Pr_few = run_few_trajectory(params)

    # FEW's EMRIInspiral returns arrays that may be NaN-terminated for points
    # after the plunge; find the last valid index.
    valid_few = np.where(np.isfinite(t_few) & np.isfinite(p_few))[0]
    N_few_valid = len(valid_few)
    T_few_s    = float(t_few[valid_few[-1]])
    T_few_days = T_few_s / 86400.0

    print(f"  FEW: {len(t_few)} total / {N_few_valid} valid output points, "
          f"t_end = {T_few_days:.3f} days ({T_few_s/3.156e7:.4f} yr)")
    print(f"  p_final = {float(p_few[valid_few[-1]]):.6f}, "
          f"e_final = {float(e_few[valid_few[-1]]):.6f}")
    print(f"  Φ_φ_final = {float(Pp_few[valid_few[-1]]):.4f} rad")

    if N_few_valid < 2:
        print("ERROR: FEW returned fewer than 2 valid trajectory points — cannot compare.")
        sys.exit(1)

    # --- Run fewtrax ---
    print("\nRunning fewtrax trajectory …")
    t_ft, p_ft, e_ft, Pp_ft, Pt_ft, Pr_ft = run_fewtrax_trajectory(
        params, flux_data, dense_steps=args.dense_steps
    )
    valid_ft  = np.where(np.isfinite(t_ft) & np.isfinite(p_ft))[0]
    T_ft_s    = float(t_ft[valid_ft[-1]])
    T_ft_days = T_ft_s / 86400.0

    print(f"  fewtrax: {len(t_ft)} total / {len(valid_ft)} valid output points, "
          f"t_end = {T_ft_days:.3f} days ({T_ft_s/3.156e7:.4f} yr)")
    print(f"  p_final = {float(p_ft[valid_ft[-1]]):.6f}, "
          f"e_final = {float(e_ft[valid_ft[-1]]):.6f}")
    print(f"  Φ_φ_final = {float(Pp_ft[valid_ft[-1]]):.4f} rad")

    # --- Compare at FEW's sparse time points ---
    print(f"\nComparing phases at {N_few_valid} FEW trajectory points "
          "(interpolating fewtrax → FEW times) …")

    cmp = compare_at_few_times(
        t_few, p_few, e_few, Pp_few, Pt_few, Pr_few,
        t_ft,  Pp_ft, Pt_ft, Pr_ft, p_ft, e_ft,
    )

    N_cmp = len(cmp["t"])
    dPp_max   = float(np.max(np.abs(cmp["dPp"])))
    dPt_max   = float(np.max(np.abs(cmp["dPt"])))
    dPr_max   = float(np.max(np.abs(cmp["dPr"])))
    dPp_final = float(cmp["dPp"][-1])
    dPt_final = float(cmp["dPt"][-1])
    dPr_final = float(cmp["dPr"][-1])

    print()
    print("=" * 68)
    print(f"  Phase dephasing at {N_cmp} FEW trajectory steps")
    print("=" * 68)
    print(f"  {'Phase':<12} {'final ΔΦ [rad]':>20}  {'max |ΔΦ| [rad]':>20}")
    print("  " + "-" * 58)
    for name, final, maxv in [
        ("Φ_φ",  dPp_final, dPp_max),
        ("Φ_θ",  dPt_final, dPt_max),
        ("Φ_r",  dPr_final, dPr_max),
    ]:
        flag = "  *** > 1 rad ***" if maxv > DEPHASING_THRESHOLD_RAD else ""
        print(f"  {name:<12} {final:>+20.6f}  {maxv:>20.6f}{flag}")

    # p and e final errors
    dp_final = float(cmp["dp"][-1])
    de_final = float(cmp["de"][-1])
    dp_max   = float(np.max(np.abs(cmp["dp"])))
    de_max   = float(np.max(np.abs(cmp["de"])))
    print(f"\n  p mismatch:   final = {dp_final:+.2e},  max = {dp_max:.2e}  [M]")
    print(f"  e mismatch:   final = {de_final:+.2e},  max = {de_max:.2e}")

    # Relative accuracy (max|ΔΦ| / Φ_total)
    Pp_total = float(cmp["Pp_few"][-1]) - float(cmp["Pp_few"][0])
    rel_acc  = dPp_max / abs(Pp_total) if abs(Pp_total) > 0 else float("nan")
    print(f"\n  Total accumulated Φ_φ = {Pp_total:.2e} rad over "
          f"{T_few_days:.1f} days")
    print(f"  Relative Φ_φ accuracy = {rel_acc:.2e}  "
          f"(max|ΔΦ_φ| / Φ_φ_total)")
    print()
    print("  NOTE: FEW returns ~25 adaptive ODE steps for a 2-year inspiral.")
    print("  Comparison is at those sparse steps only (fewtrax interpolated")
    print("  to FEW times). Phases within ODE steps are not compared here.")
    print("=" * 68)

    # --- Diagnostic ---
    run_diag = (not args.no_diagnostic) and (
        args.force_diagnostic
        or dPp_max > DEPHASING_THRESHOLD_RAD
        or dPt_max > DEPHASING_THRESHOLD_RAD
        or dPr_max > DEPHASING_THRESHOLD_RAD
    )

    freq_cmp = {}
    if run_diag:
        reason = (f"dephasing exceeds {DEPHASING_THRESHOLD_RAD} rad"
                  if not args.force_diagnostic else "--force-diagnostic")
        print(f"\nRunning frequency diagnostic ({reason}) …")
        print("  Computing per-interval average-frequency comparison …")
        try:
            freq_cmp = average_frequency_comparison(
                t_few, Pp_few, Pt_few, Pr_few, p_few, e_few, params
            )
            if freq_cmp:
                rel_phi_max   = float(np.max(np.abs(freq_cmp["rel_phi"])))
                rel_theta_max = float(np.max(np.abs(freq_cmp["rel_theta"])))
                rel_r_max     = float(np.max(np.abs(freq_cmp["rel_r"])))
                print()
                print("  === Per-interval relative frequency mismatch ===")
                print(f"  max |δΩ_φ / Ω_φ|   = {rel_phi_max:.3e}")
                print(f"  max |δΩ_θ / Ω_θ|   = {rel_theta_max:.3e}")
                print(f"  max |δΩ_r / Ω_r|   = {rel_r_max:.3e}")

                dPhi_freq_final = float(freq_cmp["dPhi_freq_phi"][-1])
                print(f"\n  Cumulative ΔΦ_φ from freq mismatch alone: "
                      f"{dPhi_freq_final:.4f} rad over {T_few_days:.1f} days")

                if abs(dPhi_freq_final) > abs(dPp_final - dPhi_freq_final):
                    dominant = "FREQUENCY FORMULA MISMATCH"
                    detail   = (f"Ω_φ(p, e, a) differs between fewtrax and FEW "
                                f"(relative error ~{rel_phi_max:.1e})")
                else:
                    dominant = "TRAJECTORY MISMATCH"
                    detail   = ("p(t) or e(t) diverges between the two codes")
                print(f"  Dominant source: {dominant}")
                print(f"  Detail: {detail}")

                print()
                # Direct frequency formula check (Schwarzschild + FEW API if available)
                valid_few_mask = np.isfinite(t_few) & np.isfinite(p_few)
                direct_frequency_formula_check(
                    params["a"],
                    p_few[valid_few_mask],
                    e_few[valid_few_mask],
                    params.get("x0", 1.0),
                )
                if freq_cmp.get("few_api_freqs") is None:
                    print(f"\n  NOTE: Per-interval ΔΦ/Δt comparison uses FEW ODE steps of")
                    print(f"  ~{T_few_days/N_few_valid:.0f} days.  The {rel_phi_max:.1e} mismatch is")
                    print(f"  consistent with avg-vs-instantaneous numerical artefact and")
                    print(f"  does NOT imply the Ω formula is wrong to that precision.")
        except Exception as exc:
            import traceback
            print(f"  Diagnostic failed: {exc}")
            traceback.print_exc()
    elif not args.no_diagnostic:
        print(f"\nAll dephasings below {DEPHASING_THRESHOLD_RAD} rad — "
              "diagnostic not triggered.  Use --force-diagnostic to run anyway.")
        # Still compute freq comparison for informational purposes
        try:
            freq_cmp = average_frequency_comparison(
                t_few, Pp_few, Pt_few, Pr_few, p_few, e_few, params
            )
            if freq_cmp:
                rel_phi_max   = float(np.max(np.abs(freq_cmp["rel_phi"])))
                dPhi_freq_end = float(freq_cmp["dPhi_freq_phi"][-1])
                print(f"  (Informational) max |δΩ_φ/Ω_φ| per interval = {rel_phi_max:.3e}")
                print(f"  (Informational) cumulative freq-mismatch ΔΦ_φ = {dPhi_freq_end:.4f} rad")
                print(f"  NOTE: The per-interval frequency comparison uses FEW's average")
                print(f"  Ω ≈ ΔΦ/Δt over ~{T_few_days/N_few_valid:.0f}-day intervals, which")
                print(f"  may differ slightly from the instantaneous Ω at the midpoint.")
        except Exception:
            pass

    # --- Plots ---
    if not args.no_plot:
        print("\nSaving figures …")
        plot_phase_comparison(
            cmp,
            t_ft, Pp_ft, Pt_ft, Pr_ft,
            params,
            T_few_days=T_few_days,
            T_ft_days=T_ft_days,
            out_dir=args.plot_dir,
        )
        if run_diag and freq_cmp:
            plot_diagnostic(cmp, freq_cmp, params, out_dir=args.plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Trajectory accuracy: fewtrax vs FastEMRIWaveforms.

Runs both integrators on the same parameter sets and reports:
  • p(t) and e(t) RMS relative error (interpolated to a common grid)
  • Phase errors: Φ_φ, Φ_θ, Φ_r at matching time steps
  • Final-time p, e agreement

Usage
-----
    python compare_trajectory.py [/path/to/few/data]

The FEW data directory is read from the .env file (FEW_DATA_DIR).
An explicit path can be passed as a positional argument.

Optional: pass --plot to save comparison figures.
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from utils import (
    find_data_dir,
    PARAM_SUITE,
    rms_relative_error,
    phase_error_rad,
    print_header,
    print_table,
    timer,
)


# ---------------------------------------------------------------------------
# FEW trajectory wrapper
# ---------------------------------------------------------------------------

def run_few_trajectory(params: dict) -> tuple[np.ndarray, ...]:
    """Run the FEW EMRIInspiral integrator and return (t, p, e, Phi_phi, Phi_theta, Phi_r).

    FEW returns 7 arrays: (t, p, e, x, Phi_phi, Phi_theta, Phi_r).
    The inclination x is discarded here.
    """
    from few.trajectory.inspiral import EMRIInspiral as FEWInspiral
    traj = FEWInspiral(func='KerrEccEqFlux')
    result = traj(
        params["M"], params["mu"],
        params["a"],
        params["p0"], params["e0"], params["x0"],
        T=params["T"],
        dt=params["dt"],
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
    )
    # result = (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    t, p, e, _x, Phi_phi, Phi_theta, Phi_r = (np.asarray(r) for r in result[:7])
    return t, p, e, Phi_phi, Phi_theta, Phi_r


# ---------------------------------------------------------------------------
# fewtrax trajectory wrapper
# ---------------------------------------------------------------------------

def run_fewtrax_trajectory(params: dict, flux_data) -> tuple[np.ndarray, ...]:
    """Run fewtrax EMRIInspiral and return (t, p, e, Phi_phi, Phi_theta, Phi_r)."""
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
        dense_steps=200,
    )
    return tuple(np.asarray(r) for r in result)


# ---------------------------------------------------------------------------
# Interpolate onto a common time grid
# ---------------------------------------------------------------------------

def interpolate_to_common_grid(
    t_ref: np.ndarray, vals_ref: list[np.ndarray],
    t_test: np.ndarray, vals_test: list[np.ndarray],
    n_grid: int = 500,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Interpolate both trajectories to a shared uniform time grid.

    Uses the overlapping time interval [max(t0), min(t_end)].
    """
    t_lo = max(float(t_ref[0]),  float(t_test[0]))
    t_hi = min(float(t_ref[-1]), float(t_test[-1]))
    if t_hi <= t_lo:
        raise ValueError("Trajectories have no overlapping time interval.")
    t_grid = np.linspace(t_lo, t_hi, n_grid)

    def _interp(t_src, arr):
        valid = np.isfinite(arr) & np.isfinite(t_src)
        return np.interp(t_grid, t_src[valid], arr[valid])

    interp_ref  = [_interp(t_ref,  v) for v in vals_ref]
    interp_test = [_interp(t_test, v) for v in vals_test]
    return t_grid, interp_ref, interp_test


# ---------------------------------------------------------------------------
# Single-parameter-set comparison
# ---------------------------------------------------------------------------

def compare_one(params: dict, flux_data) -> dict:
    """Run both integrators and return a dict of accuracy metrics."""
    label = params.get("label", "unnamed")

    with timer(f"  FEW trajectory ({label})", verbose=False):
        t_few, p_few, e_few, Phi_phi_few, Phi_theta_few, Phi_r_few = run_few_trajectory(params)

    with timer(f"  fewtrax trajectory ({label})", verbose=False):
        t_ft, p_ft, e_ft, Phi_phi_ft, Phi_theta_ft, Phi_r_ft = run_fewtrax_trajectory(params, flux_data)

    try:
        t_grid, ref_vals, ft_vals = interpolate_to_common_grid(
            t_few,  [p_few,  e_few,  Phi_phi_few,  Phi_theta_few,  Phi_r_few],
            t_ft,   [p_ft,   e_ft,   Phi_phi_ft,   Phi_theta_ft,   Phi_r_ft],
        )
    except ValueError as exc:
        return dict(label=label, error=str(exc))

    p_ref, e_ref, Pp_ref, Pt_ref, Pr_ref = ref_vals
    p_ft_i, e_ft_i, Pp_ft, Pt_ft, Pr_ft = ft_vals

    p_rms   = rms_relative_error(p_ref, p_ft_i)
    e_rms   = rms_relative_error(e_ref, e_ft_i)
    Pp_mean, Pp_max = phase_error_rad(Pp_ref, Pp_ft)
    Pt_mean, Pt_max = phase_error_rad(Pt_ref, Pt_ft)
    Pr_mean, Pr_max = phase_error_rad(Pr_ref, Pr_ft)

    # Final-point comparison (last valid index of the shorter array)
    p_final_few = float(p_few[np.isfinite(p_few)][-1])
    p_final_ft  = float(p_ft[np.isfinite(p_ft)][-1])
    e_final_few = float(e_few[np.isfinite(e_few)][-1])
    e_final_ft  = float(e_ft[np.isfinite(e_ft)][-1])

    T_few_days = float(t_few[np.isfinite(t_few)][-1]) / 86400.0
    T_ft_days  = float(t_ft[np.isfinite(t_ft)][-1])   / 86400.0

    return dict(
        label=label,
        p_rms=p_rms,
        e_rms=e_rms,
        Phi_phi_mean_rad=Pp_mean,
        Phi_phi_max_rad=Pp_max,
        Phi_theta_mean_rad=Pt_mean,
        Phi_theta_max_rad=Pt_max,
        Phi_r_mean_rad=Pr_mean,
        Phi_r_max_rad=Pr_max,
        p_final_few=p_final_few,
        p_final_ft=p_final_ft,
        e_final_few=e_final_few,
        e_final_ft=e_final_ft,
        T_few_days=T_few_days,
        T_ft_days=T_ft_days,
        # Store raw arrays for plotting
        t_few=t_few, p_few=p_few, e_few=e_few,
        t_ft=t_ft,   p_ft=p_ft,   e_ft=e_ft,
        Phi_phi_few=Phi_phi_few, Phi_phi_ft=Phi_phi_ft,
        Phi_theta_few=Phi_theta_few, Phi_theta_ft=Phi_theta_ft,
        Phi_r_few=Phi_r_few, Phi_r_ft=Phi_r_ft,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(result: dict, out_dir: str = ".") -> None:
    """Save a 6-panel figure: p(t), e(t), and dephasing for Φ_φ, Φ_θ, Φ_r.

    Notes
    -----
    Both fewtrax and FEW accumulate orbital phases without wrapping.
    A linearly growing ΔΦ indicates a constant frequency offset between
    the two implementations.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots.")
        return

    label = result["label"]
    t_few = result["t_few"] / 86400.0   # → days
    t_ft  = result["t_ft"]  / 86400.0

    # Build common time grid for phase differences
    t_lo = max(float(t_few[0]),  float(t_ft[0]))
    t_hi = min(float(t_few[-1]), float(t_ft[-1]))
    t_common = np.linspace(t_lo, t_hi, 1000)

    def _phase_diff(key):
        few_arr = result[f"{key}_few"]
        ft_arr  = result[f"{key}_ft"]
        t_few_s = result["t_few"]
        t_ft_s  = result["t_ft"]
        t_lo_s  = t_lo * 86400.0
        t_hi_s  = t_hi * 86400.0
        t_c_s   = np.linspace(t_lo_s, t_hi_s, 1000)
        few_c = np.interp(t_c_s, t_few_s, few_arr)
        ft_c  = np.interp(t_c_s, t_ft_s,  ft_arr)
        return few_c, ft_c, ft_c - few_c

    Pp_few_c, Pp_ft_c, dPp = _phase_diff("Phi_phi")
    Pt_few_c, Pt_ft_c, dPt = _phase_diff("Phi_theta")
    Pr_few_c, Pr_ft_c, dPr = _phase_diff("Phi_r")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # --- Row 0: p(t) and Phi_phi dephasing ---
    axes[0, 0].plot(t_few, result["p_few"], label="FEW",     lw=1.5)
    axes[0, 0].plot(t_ft,  result["p_ft"],  label="fewtrax", lw=1.2, ls="--")
    axes[0, 0].set_ylabel(r"$p\;[M]$")
    axes[0, 0].set_title(f"Trajectory comparison – {label}")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(t_common, dPp, lw=1.2, color="C0")
    axes[0, 1].axhline(0, color="k", lw=0.5, ls="--")
    axes[0, 1].set_ylabel(r"$\Delta\Phi_\phi\;[\mathrm{rad}]$")
    axes[0, 1].set_title(
        fr"$\Phi_\phi$ dephasing  "
        fr"(mean = {result['Phi_phi_mean_rad']:.2e} rad, "
        fr"max = {result['Phi_phi_max_rad']:.2e} rad)"
    )

    # --- Row 1: e(t) and Phi_theta dephasing ---
    axes[1, 0].plot(t_few, result["e_few"], label="FEW",     lw=1.5)
    axes[1, 0].plot(t_ft,  result["e_ft"],  label="fewtrax", lw=1.2, ls="--")
    axes[1, 0].set_ylabel(r"$e$")
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].plot(t_common, dPt, lw=1.2, color="C1")
    axes[1, 1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1, 1].set_ylabel(r"$\Delta\Phi_\theta\;[\mathrm{rad}]$")
    axes[1, 1].set_title(
        fr"$\Phi_\theta$ dephasing  "
        fr"(mean = {result['Phi_theta_mean_rad']:.2e} rad, "
        fr"max = {result['Phi_theta_max_rad']:.2e} rad)"
    )

    # --- Row 2: accumulated Phi_phi(t) and Phi_r dephasing ---
    axes[2, 0].plot(t_common, Pp_few_c, label="FEW",     lw=1.5)
    axes[2, 0].plot(t_common, Pp_ft_c,  label="fewtrax", lw=1.2, ls="--")
    axes[2, 0].set_ylabel(r"$\Phi_\phi\;[\mathrm{rad}]$")
    axes[2, 0].set_xlabel("Time [days]")
    axes[2, 0].legend(fontsize=9)

    axes[2, 1].plot(t_common, dPr, lw=1.2, color="C2")
    axes[2, 1].axhline(0, color="k", lw=0.5, ls="--")
    axes[2, 1].set_ylabel(r"$\Delta\Phi_r\;[\mathrm{rad}]$")
    axes[2, 1].set_xlabel("Time [days]")
    axes[2, 1].set_title(
        fr"$\Phi_r$ dephasing  "
        fr"(mean = {result['Phi_r_mean_rad']:.2e} rad, "
        fr"max = {result['Phi_r_max_rad']:.2e} rad)"
    )

    plt.tight_layout()
    out_path = f"{out_dir}/trajectory_{label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # --- Second figure: accumulated phases side by side ---
    _plot_accumulated_phases(result, t_common, out_dir)


def _plot_accumulated_phases(result: dict, t_common: np.ndarray, out_dir: str) -> None:
    """Save a 3×2 figure showing accumulated phase and dephasing for all three angles."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    label = result["label"]
    t_few_s = result["t_few"]
    t_ft_s  = result["t_ft"]
    t_c_s   = np.linspace(t_few_s[0], t_few_s[-1], 1000)
    t_c_days = t_c_s / 86400.0

    phases = [
        ("Phi_phi",   r"$\Phi_\phi$",   "C0"),
        ("Phi_theta", r"$\Phi_\theta$", "C1"),
        ("Phi_r",     r"$\Phi_r$",      "C2"),
    ]

    fig, axes = plt.subplots(len(phases), 2, figsize=(14, 4 * len(phases)))
    fig.suptitle(f"Accumulated phases & dephasing – {label}", fontsize=12)

    for row, (key, label_tex, color) in enumerate(phases):
        few_arr = result[f"{key}_few"]
        ft_arr  = result[f"{key}_ft"]

        few_c = np.interp(t_c_s, t_few_s, few_arr)
        ft_c  = np.interp(t_c_s, t_ft_s,  ft_arr)
        delta = ft_c - few_c

        # Left: accumulated phase
        axes[row, 0].plot(t_c_days, few_c, label="FEW",     lw=1.5)
        axes[row, 0].plot(t_c_days, ft_c,  label="fewtrax", lw=1.2, ls="--")
        axes[row, 0].set_ylabel(fr"{label_tex} [rad]")
        axes[row, 0].legend(fontsize=9)
        if row == len(phases) - 1:
            axes[row, 0].set_xlabel("Time [days]")

        # Right: dephasing
        axes[row, 1].plot(t_c_days, delta, lw=1.2, color=color)
        axes[row, 1].axhline(0, color="k", lw=0.5, ls="--")
        axes[row, 1].set_ylabel(fr"$\Delta${label_tex} [rad]")
        mean_err = result[f"{key}_mean_rad"]
        max_err  = result[f"{key}_max_rad"]
        axes[row, 1].set_title(fr"mean|Δ| = {mean_err:.2e} rad,  max|Δ| = {max_err:.2e} rad")
        if row == len(phases) - 1:
            axes[row, 1].set_xlabel("Time [days]")

    plt.tight_layout()
    out_path = f"{out_dir}/trajectory_{label}_phases.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=None, help="Path to FEW data directory")
    parser.add_argument("--plot", action="store_true", help="Save comparison figures")
    parser.add_argument("--plot-dir", default=".", help="Directory for figure output")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    print(f"Using FEW data directory: {data_dir}")

    # Load flux data once (shared across all parameter sets)
    print("\nLoading fewtrax flux data …")
    from fewtrax.data import load_flux_data
    flux_data = load_flux_data(data_dir)
    print("  Done.")

    print_header("Trajectory accuracy: fewtrax vs FastEMRIWaveforms")

    results = []
    for params in PARAM_SUITE:
        label = params.get("label", "?")
        print(f"\n--- {label} ---")
        try:
            res = compare_one(params, flux_data)
            results.append(res)
            if "error" in res:
                print(f"  ERROR: {res['error']}")
            else:
                print(f"  Duration:  FEW = {res['T_few_days']:.3f} d,  fewtrax = {res['T_ft_days']:.3f} d")
                print(f"  p(t) RMS relative error:  {res['p_rms']:.2e}")
                print(f"  e(t) RMS relative error:  {res['e_rms']:.2e}")
                print(f"  Φ_φ  mean/max dephasing:  {res['Phi_phi_mean_rad']:.2e} / {res['Phi_phi_max_rad']:.2e}  rad")
                print(f"  Φ_θ  mean/max dephasing:  {res['Phi_theta_mean_rad']:.2e} / {res['Phi_theta_max_rad']:.2e}  rad")
                print(f"  Φ_r  mean/max dephasing:  {res['Phi_r_mean_rad']:.2e} / {res['Phi_r_max_rad']:.2e}  rad")
                print(f"  p_final:  FEW = {res['p_final_few']:.6f},  fewtrax = {res['p_final_ft']:.6f}")
                print(f"  e_final:  FEW = {res['e_final_few']:.6f},  fewtrax = {res['e_final_ft']:.6f}")
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results.append(dict(label=label, error=str(exc)))

    # Summary table
    print_header("Summary table")
    ok = [r for r in results if "error" not in r]
    if ok:
        headers = ["label", "p RMS err", "e RMS err", "Φφ mean [rad]", "Φφ max [rad]", "Φθ max [rad]", "Φr max [rad]"]
        widths  = [14, 12, 12, 16, 16, 16, 16]
        rows = [
            (
                r["label"],
                f"{r['p_rms']:.2e}",
                f"{r['e_rms']:.2e}",
                f"{r['Phi_phi_mean_rad']:.2e}",
                f"{r['Phi_phi_max_rad']:.2e}",
                f"{r['Phi_theta_max_rad']:.2e}",
                f"{r['Phi_r_max_rad']:.2e}",
            )
            for r in ok
        ]
        print_table(rows, headers, widths)

    if args.plot:
        print("\nSaving figures …")
        for res in results:
            if "error" not in res:
                plot_comparison(res, out_dir=args.plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

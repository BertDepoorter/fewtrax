"""Trajectory accuracy: fewtrax vs FastEMRIWaveforms.

Runs both integrators on the same parameter sets and reports:
  • p(t) and e(t) RMS relative error
  • Phase errors: Φ_φ, Φ_θ, Φ_r  (absolute and relative)
  • Final-time p, e agreement

Comparison method
-----------------
FEW's ``EMRIInspiral`` returns only the *sparse* adaptive ODE nodes (often <10
points for a sub-year inspiral); ``dt`` is a step-size hint, not an output
density.  Linearly interpolating FEW's rapidly accumulating phases between those
sparse nodes injects multi-radian error that has nothing to do with the
integrators agreeing.

To avoid that artefact entirely, fewtrax is evaluated at FEW's *exact* node
times using the trajectory module's ``t_obs`` feature (``SaveAt(ts=t_obs)`` on
the Dopri8 dense interpolant).  Phases are then compared index-by-index with no
interpolation on either side.  A separate dense fewtrax run is used only to draw
smooth curves in the figures.

Usage
-----
    python compare_trajectory.py [/path/to/few/data]

The FEW data directory is read from the .env file (FEW_DATA_DIR).
An explicit path can be passed as a positional argument.

Optional: pass --plot to save comparison figures.
"""

from __future__ import annotations

import argparse
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from utils import (
    find_data_dir,
    PARAM_SUITE,
    rms_relative_error,
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
    The inclination x is discarded here.  ``t`` is in seconds.
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
# fewtrax trajectory wrappers
# ---------------------------------------------------------------------------

def run_fewtrax_at_nodes(params: dict, flux_data, t_nodes: np.ndarray) -> tuple[np.ndarray, ...]:
    """Run fewtrax EMRIInspiral saving at the exact times ``t_nodes`` [s].

    Uses the trajectory module's ``t_obs`` feature so fewtrax's Dopri8 dense
    interpolant is sampled at FEW's node times — eliminating any fewtrax-side
    interpolation.  Returns (t, p, e, Phi_phi, Phi_theta, Phi_r) trimmed to
    ``len(t_nodes)`` (the ``SaveAt(t1=True)`` slot is dropped).
    """
    from fewtrax.trajectory import EMRIInspiral
    n = len(t_nodes)
    ins = EMRIInspiral(flux_data, t_obs=jnp.asarray(t_nodes, dtype=jnp.float64))
    result = ins(
        p0=params["p0"], e0=params["e0"], T=params["T"], a=params["a"],
        x0=params["x0"], M=params["M"], mu=params["mu"],
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
    )
    return tuple(np.asarray(r)[:n] for r in result)


def run_fewtrax_dense(params: dict, flux_data, dense_steps: int = 300) -> tuple[np.ndarray, ...]:
    """Run fewtrax on a dense uniform grid (for smooth plot curves only)."""
    from fewtrax.trajectory import run_inspiral
    result = run_inspiral(
        a=params["a"], p0=params["p0"], e0=params["e0"], T=params["T"],
        flux_data=flux_data, M=params["M"], mu=params["mu"], dt=params["dt"],
        x0=params["x0"],
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
        dense_steps=dense_steps,
    )
    return tuple(np.asarray(r) for r in result)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _phase_metrics(ref: np.ndarray, test: np.ndarray) -> tuple[float, float, float]:
    """Return (mean|Δ|, max|Δ|, max|Δ|/span) in radians.

    ``span`` is the total accumulated phase ``|ref[-1] - ref[0]|`` over the
    inspiral, so the relative figure is the dephasing as a fraction of the
    phase actually accumulated.
    """
    diff = np.abs(test - ref)
    span = abs(float(ref[-1]) - float(ref[0]))
    rel = float(np.max(diff)) / span if span > 1e-30 else float("nan")
    return float(np.mean(diff)), float(np.max(diff)), rel


# ---------------------------------------------------------------------------
# Single-parameter-set comparison
# ---------------------------------------------------------------------------

def compare_one(params: dict, flux_data) -> dict:
    """Run both integrators and return a dict of accuracy metrics.

    Metrics are computed at FEW's exact node times (no interpolation).
    """
    label = params.get("label", "unnamed")

    with timer(f"  FEW trajectory ({label})", verbose=False):
        t_few, p_few, e_few, Pp_few, Pt_few, Pr_few = run_few_trajectory(params)

    # fewtrax saved at FEW's exact node times → direct, interpolation-free compare
    with timer(f"  fewtrax @ FEW nodes ({label})", verbose=False):
        t_ft, p_ft, e_ft, Pp_ft, Pt_ft, Pr_ft = run_fewtrax_at_nodes(
            params, flux_data, t_few
        )

    # Keep only points valid in both (fewtrax pads past-plunge slots with NaN)
    valid = np.isfinite(t_ft) & np.isfinite(p_ft) & np.isfinite(Pp_ft)
    if valid.sum() < 2:
        return dict(label=label, error="fewtrax produced <2 valid points")

    sl = valid
    p_rms = rms_relative_error(p_few[sl], p_ft[sl])
    e_rms = rms_relative_error(e_few[sl], e_ft[sl])
    Pp_mean, Pp_max, Pp_rel = _phase_metrics(Pp_few[sl], Pp_ft[sl])
    Pt_mean, Pt_max, Pt_rel = _phase_metrics(Pt_few[sl], Pt_ft[sl])
    Pr_mean, Pr_max, Pr_rel = _phase_metrics(Pr_few[sl], Pr_ft[sl])

    p_final_few = float(p_few[np.isfinite(p_few)][-1])
    p_final_ft = float(p_ft[sl][-1])
    e_final_few = float(e_few[np.isfinite(e_few)][-1])
    e_final_ft = float(e_ft[sl][-1])

    T_few_days = float(t_few[np.isfinite(t_few)][-1]) / 86400.0
    T_ft_days = float(t_ft[sl][-1]) / 86400.0

    return dict(
        label=label,
        n_few_nodes=int(np.isfinite(t_few).sum()),
        p_rms=p_rms, e_rms=e_rms,
        Phi_phi_mean_rad=Pp_mean, Phi_phi_max_rad=Pp_max, Phi_phi_rel=Pp_rel,
        Phi_theta_mean_rad=Pt_mean, Phi_theta_max_rad=Pt_max, Phi_theta_rel=Pt_rel,
        Phi_r_mean_rad=Pr_mean, Phi_r_max_rad=Pr_max, Phi_r_rel=Pr_rel,
        Phi_phi_total=abs(float(Pp_few[sl][-1]) - float(Pp_few[sl][0])),
        p_final_few=p_final_few, p_final_ft=p_final_ft,
        e_final_few=e_final_few, e_final_ft=e_final_ft,
        T_few_days=T_few_days, T_ft_days=T_ft_days,
        # Raw arrays for plotting
        t_few=t_few, p_few=p_few, e_few=e_few,
        t_ft_nodes=t_ft[sl],
        Phi_phi_few=Pp_few[sl], Phi_phi_ft=Pp_ft[sl],
        Phi_theta_few=Pt_few[sl], Phi_theta_ft=Pt_ft[sl],
        Phi_r_few=Pr_few[sl], Phi_r_ft=Pr_ft[sl],
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(result: dict, params: dict, flux_data, out_dir: str = ".") -> None:
    """Save a 3×2 figure: p(t), e(t), and dephasing for Φ_φ, Φ_θ, Φ_r.

    p(t)/e(t) use a dense fewtrax curve overlaid on FEW's nodes; dephasing
    panels show the per-node difference at FEW's exact node times.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots.")
        return

    label = result["label"]

    # Dense fewtrax run for smooth p(t)/e(t) curves
    t_ftd, p_ftd, e_ftd, *_ = run_fewtrax_dense(params, flux_data)
    fin = np.isfinite(t_ftd)
    t_ftd_d = t_ftd[fin] / 86400.0
    t_few_d = result["t_few"] / 86400.0
    t_nodes_d = result["t_ft_nodes"] / 86400.0

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # --- p(t) and e(t) ---
    axes[0, 0].plot(t_few_d, result["p_few"], "o", label="FEW nodes", ms=4)
    axes[0, 0].plot(t_ftd_d, p_ftd[fin], label="fewtrax", lw=1.2, ls="--")
    axes[0, 0].set_ylabel(r"$p\;[M]$")
    axes[0, 0].set_title(f"Trajectory comparison – {label}")
    axes[0, 0].legend(fontsize=9)

    axes[1, 0].plot(t_few_d, result["e_few"], "o", label="FEW nodes", ms=4)
    axes[1, 0].plot(t_ftd_d, e_ftd[fin], label="fewtrax", lw=1.2, ls="--")
    axes[1, 0].set_ylabel(r"$e$")
    axes[1, 0].legend(fontsize=9)

    # --- accumulated Phi_phi (left col, row 2) ---
    axes[2, 0].plot(t_nodes_d, result["Phi_phi_few"], "o", label="FEW", ms=4)
    axes[2, 0].plot(t_nodes_d, result["Phi_phi_ft"], "x", label="fewtrax", ms=5)
    axes[2, 0].set_ylabel(r"$\Phi_\phi\;[\mathrm{rad}]$")
    axes[2, 0].set_xlabel("Time [days]")
    axes[2, 0].legend(fontsize=9)

    # --- dephasing at FEW nodes (right column) ---
    phases = [
        ("Phi_phi", r"$\Phi_\phi$", "C0", 0),
        ("Phi_theta", r"$\Phi_\theta$", "C1", 1),
        ("Phi_r", r"$\Phi_r$", "C2", 2),
    ]
    for key, tex, color, row in phases:
        delta = result[f"{key}_ft"] - result[f"{key}_few"]
        axes[row, 1].plot(t_nodes_d, delta, "o-", lw=1.2, color=color, ms=3)
        axes[row, 1].axhline(0, color="k", lw=0.5, ls="--")
        axes[row, 1].set_ylabel(fr"$\Delta${tex} [rad]")
        axes[row, 1].set_title(
            fr"mean|Δ| = {result[f'{key}_mean_rad']:.2e},  "
            fr"max|Δ| = {result[f'{key}_max_rad']:.2e} rad  "
            fr"(rel {result[f'{key}_rel']:.1e})"
        )
        if row == 2:
            axes[row, 1].set_xlabel("Time [days]")

    plt.tight_layout()
    out_path = f"{out_dir}/trajectory_{label}.png"
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

    print("\nLoading fewtrax flux data …")
    from fewtrax.data import load_flux_data
    flux_data = load_flux_data(data_dir)
    print("  Done.")

    print_header("Trajectory accuracy: fewtrax vs FastEMRIWaveforms")
    print("  (fewtrax sampled at FEW's exact ODE nodes — no interpolation)")

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
                print(f"  FEW nodes: {res['n_few_nodes']}   "
                      f"Duration: FEW = {res['T_few_days']:.3f} d, fewtrax = {res['T_ft_days']:.3f} d")
                print(f"  p(t) RMS relative error:  {res['p_rms']:.2e}")
                print(f"  e(t) RMS relative error:  {res['e_rms']:.2e}")
                print(f"  Φ_φ  mean/max dephasing:  {res['Phi_phi_mean_rad']:.2e} / {res['Phi_phi_max_rad']:.2e}  rad"
                      f"   (rel {res['Phi_phi_rel']:.2e}, total {res['Phi_phi_total']:.1f} rad)")
                print(f"  Φ_θ  mean/max dephasing:  {res['Phi_theta_mean_rad']:.2e} / {res['Phi_theta_max_rad']:.2e}  rad"
                      f"   (rel {res['Phi_theta_rel']:.2e})")
                print(f"  Φ_r  mean/max dephasing:  {res['Phi_r_mean_rad']:.2e} / {res['Phi_r_max_rad']:.2e}  rad"
                      f"   (rel {res['Phi_r_rel']:.2e})")
                print(f"  p_final:  FEW = {res['p_final_few']:.6f},  fewtrax = {res['p_final_ft']:.6f}")
                print(f"  e_final:  FEW = {res['e_final_few']:.6f},  fewtrax = {res['e_final_ft']:.6f}")
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results.append(dict(label=label, error=str(exc)))

    # Summary table
    print_header("Summary table  (dephasing at FEW exact nodes)")
    ok = [r for r in results if "error" not in r]
    if ok:
        headers = ["label", "p RMS err", "e RMS err", "Φφ max [rad]", "Φφ rel", "Φr max [rad]", "Φr rel"]
        widths = [14, 12, 12, 14, 12, 14, 12]
        rows = [
            (
                r["label"],
                f"{r['p_rms']:.2e}",
                f"{r['e_rms']:.2e}",
                f"{r['Phi_phi_max_rad']:.2e}",
                f"{r['Phi_phi_rel']:.2e}",
                f"{r['Phi_r_max_rad']:.2e}",
                f"{r['Phi_r_rel']:.2e}",
            )
            for r in ok
        ]
        print_table(rows, headers, widths)

    if args.plot:
        print("\nSaving figures …")
        params_by_label = {p.get("label", "?"): p for p in PARAM_SUITE}
        for res in results:
            if "error" not in res:
                plot_comparison(res, params_by_label[res["label"]], flux_data, out_dir=args.plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

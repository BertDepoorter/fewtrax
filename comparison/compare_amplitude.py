"""Amplitude accuracy: fewtrax bisplev vs FEW multispline.

The amplitude HDF5 file stores bicubic B-spline coefficients that are
evaluated differently in fewtrax and FEW:

* **fewtrax** uses ``scipy.interpolate.bisplev`` with linear interpolation
  between spin-coordinate (z) slices.
* **FEW** uses its own C++ multispline backend (``few.amplitude``).

Both evaluate the same coefficient arrays, so differences arise from
edge handling, z-interpolation order, and numerical precision.

This script
-----------
1. Evaluates fewtrax amplitudes at a grid of (p, e) points for a fixed spin.
2. Calls FEW's amplitude module at the same points.
3. Reports RMS relative error per mode and overall.
4. Saves comparison plots (if ``--plot`` is given).

Usage
-----
    python compare_amplitude.py [/path/to/few/data] [--plot] [--plot-dir .]
"""

from __future__ import annotations

import argparse
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from utils import find_data_dir, print_header, print_table


# ---------------------------------------------------------------------------
# fewtrax amplitude evaluation
# ---------------------------------------------------------------------------

def run_fewtrax_amplitudes(
    a: float,
    p_grid: np.ndarray,
    e_grid: np.ndarray,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate fewtrax amplitudes at all (p, e) grid points.

    Returns
    -------
    amps : np.ndarray, shape (N_pts, N_modes), complex
    l_arr, m_arr, k_arr, n_arr : np.ndarray of int
    """
    from fewtrax.data.loader import load_amplitude_data
    from fewtrax.amplitude.interp import AmplitudeInterpolator

    amp_data = load_amplitude_data(data_dir)
    interp = AmplitudeInterpolator(amp_data)
    amps = interp.evaluate(a, p_grid, e_grid)
    return amps, amp_data.l_arr, amp_data.m_arr, amp_data.k_arr, amp_data.n_arr


# ---------------------------------------------------------------------------
# FEW amplitude evaluation
# ---------------------------------------------------------------------------

def run_few_amplitudes(
    a: float,
    p_grid: np.ndarray,
    e_grid: np.ndarray,
    l_arr: np.ndarray,
    m_arr: np.ndarray,
    n_arr: np.ndarray,
) -> np.ndarray:
    """Evaluate FEW Teukolsky amplitudes at all (p, e) grid points.

    Uses ``few.amplitude.interp2dcubicspline.Interp2DAmplitude`` (the
    multispline backend).  Only modes present in both fewtrax and FEW are
    compared (k=0, equatorial).

    Returns
    -------
    amps_few : np.ndarray, shape (N_pts, N_modes), complex
        NaN where FEW evaluation fails.
    """
    try:
        from few.amplitude.interp2dcubicspline import Interp2DAmplitude
    except ImportError:
        raise ImportError(
            "FEW amplitude module not found.  Install FastEMRIWaveforms."
        )

    N_pts = len(p_grid)
    N_modes = len(l_arr)
    amps_few = np.full((N_pts, N_modes), np.nan, dtype=complex)

    for im, (l, m, n) in enumerate(zip(l_arr, m_arr, n_arr)):
        try:
            amp_fn = Interp2DAmplitude(l, m, 0, n, a, include_minus_m=False)
            for ip, (p, e) in enumerate(zip(p_grid, e_grid)):
                try:
                    amps_few[ip, im] = complex(amp_fn(p, e))
                except Exception:
                    pass
        except Exception:
            pass

    return amps_few


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def amplitude_rms_error(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Per-mode RMS relative error of |amplitude|.

    Parameters
    ----------
    ref, test : shape (N_pts, N_modes)

    Returns
    -------
    rms : shape (N_modes,)
    """
    ref_abs  = np.abs(ref)
    test_abs = np.abs(test)
    valid = np.isfinite(ref_abs) & np.isfinite(test_abs) & (ref_abs > 0)
    rms = np.full(ref.shape[1], np.nan)
    for im in range(ref.shape[1]):
        mask = valid[:, im]
        if mask.sum() < 2:
            continue
        delta = (test_abs[mask, im] - ref_abs[mask, im]) / ref_abs[mask, im]
        rms[im] = float(np.sqrt(np.mean(delta**2)))
    return rms


def phase_rms_error(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Per-mode RMS error in the complex phase [rad].

    Parameters
    ----------
    ref, test : shape (N_pts, N_modes), complex

    Returns
    -------
    rms : shape (N_modes,)
    """
    valid = np.isfinite(ref) & np.isfinite(test) & (np.abs(ref) > 0)
    rms = np.full(ref.shape[1], np.nan)
    for im in range(ref.shape[1]):
        mask = valid[:, im]
        if mask.sum() < 2:
            continue
        phase_ref  = np.angle(ref[mask, im])
        phase_test = np.angle(test[mask, im])
        delta_phi  = np.angle(np.exp(1j * (phase_test - phase_ref)))  # wrapped diff
        rms[im] = float(np.sqrt(np.mean(delta_phi**2)))
    return rms


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mode_amplitudes(
    p_grid: np.ndarray,
    e_grid: np.ndarray,
    amps_ft: np.ndarray,
    amps_few: np.ndarray,
    l_arr: np.ndarray,
    m_arr: np.ndarray,
    n_arr: np.ndarray,
    mode_indices: list[int],
    a: float,
    out_dir: str,
) -> None:
    """Plot |A(p)| and |A(e)| for a selection of modes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots.")
        return

    n_modes_plot = len(mode_indices)
    fig, axes = plt.subplots(n_modes_plot, 2, figsize=(12, 3 * n_modes_plot))
    if n_modes_plot == 1:
        axes = axes[np.newaxis, :]

    for row, im in enumerate(mode_indices):
        l, m, n = int(l_arr[im]), int(m_arr[im]), int(n_arr[im])
        mode_label = f"$\\ell={l},\\,m={m},\\,n={n}$"

        amp_ft  = np.abs(amps_ft[:, im])
        amp_few = np.abs(amps_few[:, im])

        # Left: vs p (first half of grid assumed to vary p)
        axes[row, 0].plot(p_grid, amp_ft,  lw=1.5, label="fewtrax bisplev")
        axes[row, 0].plot(p_grid, amp_few, lw=1.2, ls="--", label="FEW multispline")
        axes[row, 0].set_ylabel(fr"$|A|$  {mode_label}")
        axes[row, 0].set_xlabel(r"$p\;[M]$")
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].set_yscale("log")

        # Right: relative difference
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(amp_few > 0, (amp_ft - amp_few) / amp_few, np.nan)
        axes[row, 1].plot(p_grid, rel, lw=1.2, color="C2")
        axes[row, 1].axhline(0, color="k", lw=0.5, ls="--")
        axes[row, 1].set_ylabel(r"$(|A^{\rm ft}| - |A^{\rm FEW}|)/|A^{\rm FEW}|$")
        axes[row, 1].set_xlabel(r"$p\;[M]$")

    fig.suptitle(fr"Amplitude comparison  ($a={a}$)", fontsize=12)
    plt.tight_layout()
    out_path = f"{out_dir}/amplitude_a{a:.2f}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_rms_per_mode(
    rms_amp: np.ndarray,
    rms_phase: np.ndarray,
    l_arr: np.ndarray,
    m_arr: np.ndarray,
    a: float,
    out_dir: str,
) -> None:
    """Scatter plot of per-mode RMS errors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    valid_amp   = np.isfinite(rms_amp)
    valid_phase = np.isfinite(rms_phase)

    axes[0].scatter(m_arr[valid_amp], rms_amp[valid_amp],
                    c=l_arr[valid_amp], cmap="viridis", s=8, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("$m$")
    axes[0].set_ylabel("RMS relative amplitude error")
    axes[0].set_title(fr"Amplitude error  ($a={a}$)")

    sc = axes[1].scatter(m_arr[valid_phase], rms_phase[valid_phase],
                         c=l_arr[valid_phase], cmap="viridis", s=8, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$m$")
    axes[1].set_ylabel("RMS phase error [rad]")
    axes[1].set_title(fr"Phase error  ($a={a}$)")
    fig.colorbar(sc, ax=axes[1], label="$\\ell$")

    plt.tight_layout()
    out_path = f"{out_dir}/amplitude_rms_a{a:.2f}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--spin", type=float, default=0.3,
                        help="BH spin parameter a (default: 0.3)")
    parser.add_argument("--p-min", type=float, default=7.0)
    parser.add_argument("--p-max", type=float, default=15.0)
    parser.add_argument("--e-fixed", type=float, default=0.3,
                        help="Fixed eccentricity for the p-sweep (default: 0.3)")
    parser.add_argument("--n-grid", type=int, default=30,
                        help="Number of p grid points (default: 30)")
    parser.add_argument("--n-modes-print", type=int, default=10,
                        help="Number of modes to show in summary table")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", default=".")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    print(f"Using FEW data directory: {data_dir}")

    a = args.spin
    p_grid = np.linspace(args.p_min, args.p_max, args.n_grid)
    e_grid = np.full(args.n_grid, args.e_fixed)

    print_header(f"Amplitude comparison: fewtrax bisplev vs FEW multispline  (a={a})")
    print(f"  p grid: {args.p_min:.1f} → {args.p_max:.1f}  ({args.n_grid} points)")
    print(f"  e = {args.e_fixed} (fixed)")

    # --- fewtrax ---
    print("\nEvaluating fewtrax amplitudes …")
    amps_ft, l_arr, m_arr, k_arr, n_arr = run_fewtrax_amplitudes(
        a, p_grid, e_grid, data_dir
    )
    print(f"  Done – {amps_ft.shape[1]} modes evaluated.")

    # --- FEW ---
    print("\nEvaluating FEW amplitudes …")
    try:
        amps_few = run_few_amplitudes(a, p_grid, e_grid, l_arr, m_arr, n_arr)
        have_few = True
        n_valid = int(np.sum(np.any(np.isfinite(amps_few), axis=0)))
        print(f"  Done – {n_valid}/{amps_few.shape[1]} modes returned valid values.")
    except ImportError as exc:
        print(f"  WARNING: {exc}")
        print("  Skipping FEW amplitude comparison; showing fewtrax amplitudes only.")
        have_few = False

    if have_few:
        # --- Metrics ---
        rms_amp   = amplitude_rms_error(amps_few, amps_ft)
        rms_phase = phase_rms_error(amps_few, amps_ft)

        # Overall stats (excluding NaN)
        valid_rms = rms_amp[np.isfinite(rms_amp)]
        print(f"\n  Overall RMS amplitude error: "
              f"median={np.median(valid_rms):.2e}, "
              f"max={np.max(valid_rms):.2e}")

        # Top-N worst modes
        worst = np.argsort(np.where(np.isfinite(rms_amp), rms_amp, 0))[::-1]
        N = min(args.n_modes_print, len(worst))

        print_header(f"Top {N} modes by RMS amplitude error")
        headers = ["(l, m, n)", "RMS |A| err", "RMS phase err [rad]"]
        widths  = [14, 14, 20]
        rows = []
        for im in worst[:N]:
            l, m, n = int(l_arr[im]), int(m_arr[im]), int(n_arr[im])
            rows.append((
                f"({l},{m},{n})",
                f"{rms_amp[im]:.2e}" if np.isfinite(rms_amp[im]) else "NaN",
                f"{rms_phase[im]:.2e}" if np.isfinite(rms_phase[im]) else "NaN",
            ))
        print_table(rows, headers, widths)

        if args.plot:
            print("\nSaving figures …")
            # Plot the 5 modes with largest |A| at the midpoint
            mid = len(p_grid) // 2
            power = np.abs(amps_ft[mid, :])**2
            top5 = list(np.argsort(power)[::-1][:5])
            plot_mode_amplitudes(
                p_grid, e_grid, amps_ft, amps_few,
                l_arr, m_arr, n_arr,
                mode_indices=top5, a=a, out_dir=args.plot_dir,
            )
            plot_rms_per_mode(
                rms_amp, rms_phase, l_arr, m_arr, a=a, out_dir=args.plot_dir,
            )

    else:
        # fewtrax-only plots
        if args.plot:
            print("\nSaving fewtrax-only figures …")
            mid = len(p_grid) // 2
            power = np.abs(amps_ft[mid, :])**2
            top5 = list(np.argsort(power)[::-1][:5])
            dummy_few = np.full_like(amps_ft, np.nan)
            plot_mode_amplitudes(
                p_grid, e_grid, amps_ft, dummy_few,
                l_arr, m_arr, n_arr,
                mode_indices=top5, a=a, out_dir=args.plot_dir,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

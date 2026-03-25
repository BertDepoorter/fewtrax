"""compare_tf.py — Sparse WDM time-frequency track comparison for EMRI waveforms.

Generates a fewtrax EMRI trajectory, computes the dominant harmonic frequency
tracks analytically, and (optionally) validates them against full pywavelet
WDM transforms of single-mode time series.

Usage
-----
# Analytical TF tracks only (fast, no waveform generation)
python compare_tf.py

# Full WDM transform comparison for dominant mode
python compare_tf.py --wdm

# Custom parameters
python compare_tf.py --M 5e5 --mu 10 --a 0.5 --p0 9 --e0 0.3 --T 1.0

# Adjust WDM grid
python compare_tf.py --Nf 128 --Nt 2048

# Save figures
python compare_tf.py --plot --plot-dir ./figures
"""

import os
import sys
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fewtrax(data_dir: str):
    """Load fewtrax flux data and return (EMRIInspiral, FluxData)."""
    from fewtrax.data.loader import load_flux_data
    from fewtrax.trajectory.inspiral import EMRIInspiral

    flux_data = load_flux_data(data_dir)
    inspiral = EMRIInspiral(flux_data)
    return inspiral, flux_data


def run_trajectory(inspiral, M, mu, a, p0, e0, T, dense_steps=2000):
    """Run fewtrax trajectory, return (t, p, e, Phi_phi, Phi_theta, Phi_r)."""
    t, p, e, Phi_phi, Phi_theta, Phi_r = inspiral(
        p0=p0, e0=e0, a=a, T=T, M=M, mu=mu,
        dense_steps=dense_steps,
    )
    return (
        np.array(t), np.array(p), np.array(e),
        np.array(Phi_phi), np.array(Phi_theta), np.array(Phi_r),
    )


def dominant_modes(n_modes: int = 10) -> list:
    """Return the dominant EMRI mode indices (l, m, k, n)."""
    modes = []
    # m=2 dominant, k=0 (equatorial), n=-1..2
    for n in [-1, 0, 1, 2]:
        modes.append((2, 2, 0, n))
    # m=3
    for n in [0, 1]:
        modes.append((3, 3, 0, n))
    # m=4
    modes.append((4, 4, 0, 1))
    # m=2, n=0 harmonics with radial corrections
    modes.append((2, 2, 0, -2))
    modes.append((2, 1, 0, 1))
    modes.append((2, 2, 0, 3))
    return modes[:n_modes]


def print_track_summary(track_set):
    """Print a summary table of TF tracks."""
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title=f"Sparse WDM TF Tracks  [{track_set.grid}]")
    table.add_column("Mode (l,m,k,n)", style="cyan")
    table.add_column("f_min [mHz]", justify="right")
    table.add_column("f_max [mHz]", justify="right")
    table.add_column("df [mHz]", justify="right")
    table.add_column("i_freq range", justify="right")
    table.add_column("Memory [kB]", justify="right")

    for t in track_set.tracks:
        l, m, k, n = t.mode
        f_lo = t.freq_hz.min() * 1e3
        f_hi = t.freq_hz.max() * 1e3
        df = f_hi - f_lo
        i_lo, i_hi = int(t.i_freq.min()), int(t.i_freq.max())
        table.add_row(
            f"({l},{m},{k},{n})",
            f"{f_lo:.3f}",
            f"{f_hi:.3f}",
            f"{df:.3f}",
            f"[{i_lo}–{i_hi}]",
            f"{t.nbytes/1024:.1f}",
        )

    console.print(table)
    console.print(f"[bold]Total:[/bold] {track_set.n_modes} modes, "
                  f"{track_set.nbytes/1024:.1f} kB")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tf_tracks(track_set, title="", ax=None, save_path=None):
    """Plot all TF tracks in the WDM plane."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    grid = track_set.grid
    t_bins = grid.t_bins / (365.25 * 24 * 3600)  # convert to years

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = cm.tab10(np.linspace(0, 1, track_set.n_modes))

    for i, track in enumerate(track_set.tracks):
        l, m, k, n = track.mode
        label = f"({l},{m},{k},{n})"
        f_mhz = track.freq_hz * 1e3
        ax.plot(t_bins, f_mhz, color=colors[i], lw=1.2, label=label)

    # Overlay WDM grid lines (few representative ones)
    delta_F_mhz = grid.delta_F * 1e3
    f_lo_plot = 0.9 * min(t.freq_hz.min() for t in track_set.tracks) * 1e3
    f_hi_plot = 1.1 * max(t.freq_hz.max() for t in track_set.tracks) * 1e3
    n_lines = 0
    for i_bin in range(grid.Nf):
        f_line = i_bin * delta_F_mhz
        if f_lo_plot <= f_line <= f_hi_plot and n_lines < 30:
            ax.axhline(f_line, color="gray", lw=0.3, alpha=0.4)
            n_lines += 1

    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("GW frequency [mHz]")
    ax.set_title(title or "EMRI harmonic TF tracks in WDM plane")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_ylim(max(0, f_lo_plot), f_hi_plot)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_wdm_comparison(track, wave_data, title="", save_path=None):
    """Plot full WDM grid vs analytical track overlay."""
    import matplotlib.pyplot as plt

    grid = track.grid
    t_bins_yr = grid.t_bins / (365.25 * 24 * 3600)
    f_bins_mhz = grid.f_bins * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full WDM power map (zoom around track)
    i_lo = max(0, int(track.i_freq.min()) - 5)
    i_hi = min(grid.Nf - 1, int(track.i_freq.max()) + 5)
    power = wave_data[i_lo:i_hi+1, :] ** 2
    axes[0].pcolormesh(
        t_bins_yr, f_bins_mhz[i_lo:i_hi+1], power,
        cmap="inferno", shading="auto",
    )
    axes[0].plot(t_bins_yr, track.freq_hz * 1e3, "w--", lw=1, label="analytical track")
    axes[0].set_xlabel("Time [yr]")
    axes[0].set_ylabel("GW frequency [mHz]")
    axes[0].set_title("WDM power (zoomed around track)")
    axes[0].legend()

    # Track coefficient power
    if track.coeff is not None:
        axes[1].plot(t_bins_yr, np.abs(track.coeff))
        axes[1].set_xlabel("Time [yr]")
        axes[1].set_ylabel("|WDM coeff|")
        axes[1].set_title("WDM amplitude along track")
    else:
        axes[1].text(0.5, 0.5, "No coefficients\n(analytical track only)",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle(title or f"WDM track  mode={track.mode}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sparse WDM TF tracks for EMRI")
    parser.add_argument("--M", type=float, default=1e6, help="Primary mass [M_sun]")
    parser.add_argument("--mu", type=float, default=10.0, help="Secondary mass [M_sun]")
    parser.add_argument("--a", type=float, default=0.3, help="Kerr spin")
    parser.add_argument("--p0", type=float, default=10.0, help="Initial semi-latus rectum")
    parser.add_argument("--e0", type=float, default=0.4, help="Initial eccentricity")
    parser.add_argument("--T", type=float, default=1.0, help="Observation time [yr]")
    parser.add_argument("--dense-steps", type=int, default=2000,
                        help="Number of fewtrax trajectory points")
    parser.add_argument("--n-modes", type=int, default=10,
                        help="Number of dominant modes to analyse")
    parser.add_argument("--Nf", type=int, default=64, help="WDM frequency bins")
    parser.add_argument("--Nt", type=int, default=4096, help="WDM time bins")
    parser.add_argument("--wdm", action="store_true",
                        help="Run full WDM transform for dominant mode")
    parser.add_argument("--plot", action="store_true", help="Save figures")
    parser.add_argument("--no-plot", action="store_true", help="Skip all plots")
    parser.add_argument("--plot-dir", type=str, default="./figures")
    args = parser.parse_args()

    show_plots = args.plot and not args.no_plot

    # -----------------------------------------------------------------------
    # 1. Data directory
    # -----------------------------------------------------------------------
    data_dir = os.environ.get("FEW_DATA_DIR", None)
    if data_dir is None:
        # Try common defaults
        for candidate in [
            os.path.expanduser("~/.few/data"),
            "/usr/local/share/few/data",
        ]:
            if os.path.isdir(candidate):
                data_dir = candidate
                break
    if data_dir is None or not os.path.isdir(data_dir):
        print("ERROR: FEW_DATA_DIR not set or not found.")
        print("Set FEW_DATA_DIR in .env or export it in your shell.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Trajectory
    # -----------------------------------------------------------------------
    from fewtrax.utils.constants import YEAR_SI

    print(f"\nParameters: M={args.M:.1e} M☉, μ={args.mu} M☉, a={args.a}, "
          f"p0={args.p0}, e0={args.e0}, T={args.T} yr")

    inspiral, _ = load_fewtrax(data_dir)
    t0 = time.perf_counter()
    t, p, e, Phi_phi, Phi_theta, Phi_r = run_trajectory(
        inspiral, args.M, args.mu, args.a, args.p0, args.e0, args.T,
        dense_steps=args.dense_steps,
    )
    elapsed = time.perf_counter() - t0

    valid = np.isfinite(t)
    T_actual = float(t[valid][-1]) / YEAR_SI
    print(f"Trajectory: {valid.sum()} steps, T_actual={T_actual:.3f} yr  "
          f"(computed in {elapsed:.2f}s)")

    # -----------------------------------------------------------------------
    # 3. WDM grid
    # -----------------------------------------------------------------------
    from fewtrax.utils.tf_tracks import WDMGrid, build_tf_tracks

    T_s = float(t[valid][-1])
    grid = WDMGrid(Nf=args.Nf, Nt=args.Nt, T=T_s)
    print(f"\nWDM grid: {grid}")
    print(f"  → total samples N = {grid.Nf * grid.Nt:,}  "
          f"(underlying dt = {T_s / (grid.Nf * grid.Nt):.1f} s)")

    # -----------------------------------------------------------------------
    # 4. Analytical TF tracks
    # -----------------------------------------------------------------------
    modes = dominant_modes(args.n_modes)
    print(f"\nComputing analytical TF tracks for {len(modes)} modes ...")

    t0 = time.perf_counter()
    track_set = build_tf_tracks(
        modes, t, p, e, args.a, args.M, args.mu, grid, x0=1.0
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s")
    print()
    print_track_summary(track_set)

    # -----------------------------------------------------------------------
    # 5. Optional: full WDM transform for dominant mode
    # -----------------------------------------------------------------------
    if args.wdm:
        from fewtrax.utils.tf_tracks import sparse_wdm_track
        from fewtrax.waveform.kerr import KerrEccentricEquatorialWaveform
        from interpax import Interpolator1D
        from pywavelet.transforms.numpy.forward.main import from_time_to_wavelet
        from pywavelet.types import TimeSeries

        dom_mode = (2, 2, 0, 1)
        print(f"\nRunning full WDM transform for dominant mode {dom_mode} ...")

        try:
            wf = KerrEccentricEquatorialWaveform(data_dir)
            sparse = wf.generate_sparse(
                M=args.M, mu=args.mu, a=args.a,
                p0=args.p0, e0=args.e0, T=args.T, dt=10.0,
            )

            # Find dominant mode index
            mode_idx = None
            for i, (l_i, m_i, k_i, n_i) in enumerate(
                zip(sparse["l_arr"], sparse["m_arr"],
                    sparse["k_arr"], sparse["n_arr"])
            ):
                if (l_i, m_i, k_i, n_i) == dom_mode:
                    mode_idx = i
                    break

            if mode_idx is None:
                print(f"  Mode {dom_mode} not in selected set; using first mode instead.")
                mode_idx = 0
                dom_mode = tuple(int(x) for x in (
                    sparse["l_arr"][0], sparse["m_arr"][0],
                    sparse["k_arr"][0], sparse["n_arr"][0],
                ))

            t_wf  = np.array(sparse["t"])
            p_wf  = np.array(sparse["p"])
            e_wf  = np.array(sparse["e"])
            Pp_wf = np.array(sparse["Phi_phi"])
            Pt_wf = np.array(sparse["Phi_theta"])
            Pr_wf = np.array(sparse["Phi_r"])
            teuk_amp = np.array(sparse["teuk_modes"][:, mode_idx])

            t0 = time.perf_counter()
            wdm_track = sparse_wdm_track(
                dom_mode, t_wf, p_wf, e_wf,
                Pp_wf, Pt_wf, Pr_wf, teuk_amp,
                args.a, args.M, args.mu, grid, x0=1.0,
            )
            elapsed = time.perf_counter() - t0
            print(f"  WDM transform done in {elapsed:.2f}s")
            print(f"  {wdm_track}")

            if show_plots:
                os.makedirs(args.plot_dir, exist_ok=True)

                # Rebuild full WDM grid for background image
                valid_m = np.isfinite(t_wf)
                t_v = t_wf[valid_m]
                N = grid.Nf * grid.Nt
                t_dense = np.linspace(float(t_v[0]), float(t_v[-1]), N)

                def _interp(arr):
                    spl = Interpolator1D(t_v, np.asarray(arr)[valid_m],
                                         method="cubic2", extrap=True)
                    return np.array(spl(t_dense))

                l_d, m_d, k_d, n_d = dom_mode
                Pp_d = _interp(Pp_wf)
                Pt_d = _interp(Pt_wf)
                Pr_d = _interp(Pr_wf)
                amp_d = _interp(teuk_amp.real) + 1j * _interp(teuk_amp.imag)
                phase = m_d * Pp_d + k_d * Pt_d + n_d * Pr_d
                h_mode = amp_d * np.exp(-1j * phase)

                ts = TimeSeries(data=np.real(h_mode), time=t_dense)
                wdm_full = from_time_to_wavelet(ts, Nf=grid.Nf, Nt=grid.Nt)

                save_path = os.path.join(
                    args.plot_dir,
                    f"tf_wdm_{int(args.M/1e3)}kM_{dom_mode}.png"
                )
                plot_wdm_comparison(
                    wdm_track, wdm_full.data,
                    title=f"WDM track mode={dom_mode}  M={args.M:.1e} a={args.a}",
                    save_path=save_path,
                )

        except Exception as exc:
            import traceback
            print(f"  WDM step failed: {exc}")
            traceback.print_exc()
            print("  (Run without --wdm for analytical tracks only)")

    # -----------------------------------------------------------------------
    # 6. Plots
    # -----------------------------------------------------------------------
    if show_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        label = f"{int(args.M/1e3)}kM_a{args.a}_p{args.p0}_e{args.e0}_{args.T}yr"
        save_path = os.path.join(args.plot_dir, f"tf_tracks_{label}.png")
        plot_tf_tracks(
            track_set,
            title=f"EMRI TF tracks  M={args.M:.1e} M☉  a={args.a}  "
                  f"p0={args.p0}  e0={args.e0}  T={args.T:.1f} yr",
            save_path=save_path,
        )
    elif not args.no_plot:
        # Interactive display if matplotlib is available
        try:
            import matplotlib
            if matplotlib.get_backend() not in ("agg", "Agg"):
                plot_tf_tracks(track_set,
                               title=f"EMRI TF tracks  a={args.a}  T={args.T:.1f} yr")
        except Exception:
            pass


if __name__ == "__main__":
    main()

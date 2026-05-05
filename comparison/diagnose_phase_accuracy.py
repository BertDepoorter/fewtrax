"""Phase self-consistency diagnostic: EMRIInspiral (phases=True vs phases=False).

For each randomly drawn intrinsic EMRI parameter set this script:

1. Runs ``EMRIInspiral(phases=True)`` (full 5D ODE) and
   ``EMRIInspiral(phases=False)`` (2D (p, e) sub-system only).
2. Identifies trajectories that plunge before T (incomplete trajectories).
3. Computes the final-step inconsistency |Δp(T)| and |Δe(T)| between the
   two solve paths.  Because they share the same 2D ODE kernel, differences
   should be at machine-precision (< 1e-10 relative).
4. Flags trajectories where |Δp|/p₀ exceeds threshold (default 1e-8).
5. Saves a CSV of per-trajectory diagnostics and diagnostic scatter plots
   that map any failing region in (p₀, e₀), (a, e₀), and (M, mu) space.

Previously this script compared ``EMRIInspiral`` (exact, 64-pt GL elliptic
integrals) against ``EMRIInspiralFast`` (AGM + 24-pt GL).  These two variants
have been merged into a single ``EMRIInspiral`` class with platform-aware
dispatch (CPU → AGM+24-pt GL, GPU → 64-pt GL), so the comparison now tests
internal ODE path consistency.

Usage
-----
    python diagnose_phase_accuracy.py                          # 2048 samples
    python diagnose_phase_accuracy.py --N 8192 --seed 7
    python diagnose_phase_accuracy.py --T 1.0 --dense-steps 400
    python diagnose_phase_accuracy.py --no-plots --out results.csv

The script is safe to run on both CPU and GPU; on GPU it vmaps the ODE solves
for throughput.  On CPU, trajectories are evaluated one at a time (vmap still
works, but the batch may be limited by RAM).
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
from utils import find_data_dir, block_jax

# ---------------------------------------------------------------------------
# Threshold and defaults
# ---------------------------------------------------------------------------
P_CONSISTENCY_THRESHOLD: float = 1e-8  # maximum |Δp|/p₀ (relative p inconsistency)


# ---------------------------------------------------------------------------
# Parameter grid (mirroring benchmark_vmap_tracks.py)
# ---------------------------------------------------------------------------

def build_param_grid(N: int, seed: int = 42, T_yr: float = 2.0) -> dict:
    """Draw N random EMRI intrinsic parameter sets.

    All (p₀, e₀) pairs satisfy p₀ > p_sep(a, e₀).
    """
    from fewtrax.utils.geodesic import get_separatrix

    rng = np.random.default_rng(seed)

    M_arr  = 10.0 ** rng.uniform(np.log10(5e5), np.log10(5e6), N)
    mu_arr = rng.uniform(5.0, 50.0, N)
    a_arr  = rng.uniform(0.05, 0.90, N)
    e_arr  = rng.uniform(0.05, 0.70, N)
    dp_arr = rng.uniform(1.0, 10.0, N)

    p_arr = np.array([
        float(get_separatrix(jnp.float64(a), jnp.float64(e), jnp.float64(1.0))) + dp
        for a, e, dp in zip(a_arr, e_arr, dp_arr)
    ])

    # Mass ratio
    q_arr = mu_arr / M_arr

    return dict(
        M=M_arr, mu=mu_arr, a=a_arr, p0=p_arr, e0=e_arr,
        q=q_arr,
        dp=dp_arr,  # distance above separatrix [M]
        T=np.full(N, T_yr),
    )


# ---------------------------------------------------------------------------
# Batched trajectory runner
# ---------------------------------------------------------------------------

def make_batched_traj(traj, dense_steps: int, T: float,
                      max_steps: int = 4096,
                      atol: float = 1e-9, rtol: float = 1e-9):
    """Return a vmapped (p0, e0, a, M, mu) -> (t, p, e, Phi_phi, ...) callable."""
    fixed = dict(T=T, x0=1.0, dt=10.0,
                 dense_steps=dense_steps, max_steps=max_steps,
                 atol=atol, rtol=rtol)

    def single(p0, e0, a, M, mu):
        return traj(p0=p0, e0=e0, a=a, M=M, mu=mu, **fixed)

    return jax.jit(jax.vmap(single))


# ---------------------------------------------------------------------------
# NaN-aware trajectory validity check
# ---------------------------------------------------------------------------

def trajectory_completion_fraction(phi: np.ndarray) -> np.ndarray:
    """For each trajectory (row), return the fraction of non-NaN time steps."""
    assert phi.ndim == 2  # shape (N, dense_steps)
    valid = np.isfinite(phi)
    return valid.sum(axis=1) / phi.shape[1]


def last_valid_phase(phi: np.ndarray) -> np.ndarray:
    """Return the last finite Φ_φ value for each trajectory row."""
    N, S = phi.shape
    result = np.full(N, np.nan)
    for i in range(N):
        idx = np.where(np.isfinite(phi[i]))[0]
        if len(idx) > 0:
            result[i] = phi[i, idx[-1]]
    return result


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(
    traj_5d, traj_2d,
    grid: dict,
    dense_steps: int = 500,
    T: float = 2.0,
    batch_size: int = 256,
    threshold: float = P_CONSISTENCY_THRESHOLD,
) -> dict:
    """Run (p, e) consistency diagnostic between phases=True and phases=False.

    Parameters
    ----------
    traj_5d : EMRIInspiral(phases=True)
        Full 5D ODE trajectory (p, e, Φ_φ, Φ_θ, Φ_r).
    traj_2d : EMRIInspiral(phases=False)
        2D sub-system ODE trajectory (p, e) only.
    grid : dict
        Parameter grid from :func:`build_param_grid`.
    dense_steps : int
        Number of output time steps per trajectory.
    T : float
        Integration time [years].
    batch_size : int
        Chunk size for vmapped evaluation (to avoid OOM on GPU).
    threshold : float
        Maximum allowed relative |Δp|/p₀ between the two solve paths.

    Returns
    -------
    dict
        Per-trajectory diagnostics and aggregate statistics.
    """
    N = len(grid["p0"])
    batched_ref  = make_batched_traj(traj_5d, dense_steps=dense_steps, T=T)
    batched_fast = make_batched_traj(traj_2d, dense_steps=dense_steps, T=T)

    # Output arrays — track final (p, e) from each solve path
    p_ref_end  = np.full(N, np.nan)  # p^5D at last valid step
    p_fast_end = np.full(N, np.nan)  # p^2D at last valid step
    # Re-use phi_* names for plot compatibility; they now hold p-track data
    phi_ref_full  = np.full((N, dense_steps), np.nan)  # p(t) from 5D
    phi_fast_full = np.full((N, dense_steps), np.nan)  # p(t) from 2D
    compl_ref  = np.zeros(N)
    compl_fast = np.zeros(N)
    t_end_ref  = np.full(N, np.nan)
    t_end_fast = np.full(N, np.nan)

    n_batches = (N + batch_size - 1) // batch_size
    print(f"  Processing {N} trajectories in {n_batches} batches of ≤{batch_size} …")

    YEAR_SI = 365.25 * 24 * 3600  # s

    for b in range(n_batches):
        lo = b * batch_size
        hi = min(lo + batch_size, N)
        sl = slice(lo, hi)

        p0 = jnp.array(grid["p0"][sl], dtype=jnp.float64)
        e0 = jnp.array(grid["e0"][sl], dtype=jnp.float64)
        a  = jnp.array(grid["a"][sl],  dtype=jnp.float64)
        M  = jnp.array(grid["M"][sl],  dtype=jnp.float64)
        mu = jnp.array(grid["mu"][sl], dtype=jnp.float64)

        t0 = time.perf_counter()

        out_ref  = batched_ref(p0, e0, a, M, mu)
        out_fast = batched_fast(p0, e0, a, M, mu)
        block_jax(out_ref)
        block_jax(out_fast)

        dt = time.perf_counter() - t0

        # Unpack — phases=True returns (t, p, e, Phi_phi, …); phases=False (t, p, e)
        t_r, p_r, e_r, *_ = (np.asarray(x) for x in out_ref)
        t_f, p_f, e_f, *_ = (np.asarray(x) for x in out_fast)

        # Store p(t) track in the phi_*_full arrays (for plot reuse)
        phi_ref_full[sl]  = p_r
        phi_fast_full[sl] = p_f

        compl_ref[sl]  = trajectory_completion_fraction(p_r)
        compl_fast[sl] = trajectory_completion_fraction(p_f)

        p_ref_end[sl]  = last_valid_phase(p_r)
        p_fast_end[sl] = last_valid_phase(p_f)

        # Last valid time (in years)
        t_yr_r = t_r / YEAR_SI
        t_yr_f = t_f / YEAR_SI
        for i in range(hi - lo):
            idx_r = np.where(np.isfinite(t_yr_r[i]))[0]
            idx_f = np.where(np.isfinite(t_yr_f[i]))[0]
            t_end_ref[lo + i]  = t_yr_r[i, idx_r[-1]] if len(idx_r) > 0 else 0.0
            t_end_fast[lo + i] = t_yr_f[i, idx_f[-1]] if len(idx_f) > 0 else 0.0

        n_done = hi - lo
        print(f"    Batch {b+1}/{n_batches}: [{lo}:{hi}] "
              f"({n_done} traj, {dt:.2f} s, "
              f"{n_done/dt:.0f} traj/s)", flush=True)

    # -----------------------------------------------------------------------
    # Compute per-trajectory p inconsistency between phases=True and =False
    # -----------------------------------------------------------------------
    # Both paths share the same 2D ODE kernel, so |Δp|/p₀ should be ≲ 1e-10.

    delta_phi = np.abs(p_fast_end - p_ref_end) / (np.abs(p_ref_end) + 1e-30)

    # Trajectories that both survived the full window
    both_complete = (compl_ref >= 0.99) & (compl_fast >= 0.99)
    # Trajectories that fail the consistency criterion (among complete ones)
    fails = both_complete & (delta_phi > threshold)

    # RMS of the full p(t) track inconsistency (NaN-aware)
    delta_phi_track_rms = np.sqrt(
        np.nanmean((phi_fast_full - phi_ref_full)**2, axis=1)
    )

    stats = dict(
        # Per-trajectory arrays
        M=grid["M"], mu=grid["mu"], a=grid["a"],
        p0=grid["p0"], e0=grid["e0"],
        q=grid["q"], dp=grid["dp"],

        compl_ref=compl_ref,
        compl_fast=compl_fast,
        t_end_ref=t_end_ref,
        t_end_fast=t_end_fast,

        phi_ref_end=p_ref_end,
        phi_fast_end=p_fast_end,
        delta_phi=delta_phi,
        delta_phi_rms=delta_phi_track_rms,

        phi_ref_full=phi_ref_full,
        phi_fast_full=phi_fast_full,

        both_complete=both_complete,
        fails=fails,

        # Scalar summaries
        N_total=N,
        N_complete=int(both_complete.sum()),
        N_fails=int(fails.sum()),
        threshold=threshold,
    )

    # Print summary
    print()
    print(f"  Total trajectories         : {N}")
    print(f"  Both completed T={T:.1f} yr  : {both_complete.sum()} "
          f"({100*both_complete.mean():.1f} %)")
    print(f"  5D plunged before T        : {(compl_ref < 0.99).sum()}")
    print(f"  2D plunged before T        : {(compl_fast < 0.99).sum()}")
    print()
    if both_complete.sum() > 0:
        dp_bc = delta_phi[both_complete]
        print(f"  p inconsistency |Δp|/p₀ (complete trajectories):")
        print(f"    mean   : {np.nanmean(dp_bc):.3e}")
        print(f"    median : {np.nanmedian(dp_bc):.3e}")
        print(f"    95th%  : {np.nanpercentile(dp_bc, 95):.3e}")
        print(f"    max    : {np.nanmax(dp_bc):.3e}")
        print(f"  Trajectories failing |Δp|/p₀ > {threshold:.0e}: "
              f"{fails.sum()} / {both_complete.sum()} "
              f"({100*fails[both_complete].mean():.1f} %)")

    return stats


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(stats: dict, out_dir: Path, T: float, threshold: float):
    """Generate diagnostic scatter plots and histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out_dir.mkdir(parents=True, exist_ok=True)

    # Common masks
    bc   = stats["both_complete"]    # both complete T
    fail = stats["fails"]            # fail threshold (subset of bc)

    M     = stats["M"]
    mu    = stats["mu"]
    a     = stats["a"]
    p0    = stats["p0"]
    e0    = stats["e0"]
    q     = stats["q"]
    dp    = stats["dp"]
    dphi  = stats["delta_phi"]

    compl_ref  = stats["compl_ref"]
    compl_fast = stats["compl_fast"]

    # ---- 1. Dephasing scatter in (p₀, e₀) plane --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mask, title, c_arr in [
        (axes[0], bc,   "Both complete", dphi[bc]),
        (axes[1], ~bc,  "At least one plunges early",
         np.minimum(compl_ref[~bc], compl_fast[~bc])),
    ]:
        sc = ax.scatter(e0[mask], p0[mask], c=c_arr,
                        cmap="plasma", s=8, alpha=0.7,
                        norm=LogNorm(vmin=max(c_arr.min(), 1e-10), vmax=c_arr.max())
                        if mask.sum() > 0 and c_arr.max() > 0 else None)
        if mask.sum() > 0:
            plt.colorbar(sc, ax=ax, label="dephasing [rad]" if mask.sum() == bc.sum()
                         else "min completion fraction")
        ax.set_xlabel("e₀")
        ax.set_ylabel("p₀ [M]")
        ax.set_title(title)

    # Overlay failing points on left panel
    if fail.sum() > 0:
        axes[0].scatter(e0[fail], p0[fail], marker="x", c="red", s=30,
                        label=f"fails (>{threshold:.0e} rad)")
        axes[0].legend(fontsize=8)

    plt.suptitle(f"Phase accuracy: |ΔΦ_φ| in (p₀, e₀) plane   T={T:.1f} yr",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_p0_e0.png", dpi=150)
    plt.close()

    # ---- 2. Scatter in (a, e₀) plane --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    if bc.sum() > 0:
        sc = ax.scatter(a[bc], e0[bc], c=dphi[bc], cmap="plasma", s=8, alpha=0.7,
                        norm=LogNorm(vmin=max(dphi[bc].min(), 1e-10),
                                     vmax=dphi[bc].max()))
        plt.colorbar(sc, ax=ax, label="|ΔΦ_φ| [rad]")
    if fail.sum() > 0:
        ax.scatter(a[fail], e0[fail], marker="x", c="red", s=30,
                   label=f"fails (>{threshold:.0e} rad)")
        ax.legend(fontsize=8)
    ax.axhline(0.25, color="gray", lw=0.8, ls="--", label="e=0.25 (region boundary)")
    ax.set_xlabel("a (spin)")
    ax.set_ylabel("e₀")
    ax.set_title(f"Phase accuracy in (a, e₀) plane   T={T:.1f} yr")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_a_e0.png", dpi=150)
    plt.close()

    # ---- 3. Scatter in (log M, log mu) plane ------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    if bc.sum() > 0:
        sc = ax.scatter(np.log10(M[bc]), np.log10(mu[bc]), c=dphi[bc],
                        cmap="plasma", s=8, alpha=0.7,
                        norm=LogNorm(vmin=max(dphi[bc].min(), 1e-10),
                                     vmax=dphi[bc].max()))
        plt.colorbar(sc, ax=ax, label="|ΔΦ_φ| [rad]")
    if fail.sum() > 0:
        ax.scatter(np.log10(M[fail]), np.log10(mu[fail]),
                   marker="x", c="red", s=30, label=f"fails (>{threshold:.0e} rad)")
        ax.legend(fontsize=8)
    ax.set_xlabel("log₁₀ M [M☉]")
    ax.set_ylabel("log₁₀ μ [M☉]")
    ax.set_title(f"Phase accuracy in (M, μ) plane   T={T:.1f} yr")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_M_mu.png", dpi=150)
    plt.close()

    # ---- 4. Scatter in (q, dp) plane --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    if bc.sum() > 0:
        sc = ax.scatter(np.log10(q[bc]), dp[bc], c=dphi[bc],
                        cmap="plasma", s=8, alpha=0.7,
                        norm=LogNorm(vmin=max(dphi[bc].min(), 1e-10),
                                     vmax=dphi[bc].max()))
        plt.colorbar(sc, ax=ax, label="|ΔΦ_φ| [rad]")
    if fail.sum() > 0:
        ax.scatter(np.log10(q[fail]), dp[fail],
                   marker="x", c="red", s=30, label=f"fails (>{threshold:.0e} rad)")
        ax.legend(fontsize=8)
    ax.set_xlabel("log₁₀ q = log₁₀(μ/M)")
    ax.set_ylabel("Δp = p₀ − p_sep [M]")
    ax.set_title(f"Phase accuracy in (q, Δp) plane   T={T:.1f} yr")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_q_dp.png", dpi=150)
    plt.close()

    # ---- 5. Dephasing histogram -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if bc.sum() > 0:
        dp_bc = dphi[bc]
        ax = axes[0]
        ax.hist(np.log10(np.maximum(dp_bc, 1e-15)), bins=50, color="steelblue",
                edgecolor="white", lw=0.3)
        ax.axvline(np.log10(threshold), color="red", ls="--",
                   label=f"threshold = {threshold:.0e} rad")
        ax.set_xlabel("log₁₀ |ΔΦ_φ| [rad]")
        ax.set_ylabel("count")
        ax.set_title("Dephasing distribution (complete trajectories)")
        ax.legend()

    ax = axes[1]
    ax.hist(compl_ref,  bins=50, alpha=0.6, color="steelblue", label="exact")
    ax.hist(compl_fast, bins=50, alpha=0.6, color="darkorange", label="fast")
    ax.set_xlabel("completion fraction (fraction of dense_steps that are finite)")
    ax.set_ylabel("count")
    ax.set_title("Trajectory completion rate")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "histograms.png", dpi=150)
    plt.close()

    # ---- 6. Example failing + passing trajectories -------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    def _plot_traj(ax, i, label, color):
        """Plot |ΔΦ_φ|(t) for trajectory i."""
        phi_r = stats["phi_ref_full"][i]
        phi_f = stats["phi_fast_full"][i]
        idx = np.where(np.isfinite(phi_r) & np.isfinite(phi_f))[0]
        if len(idx) == 0:
            ax.text(0.5, 0.5, "no valid data", transform=ax.transAxes, ha="center")
            return
        t_idx = np.linspace(0, T, phi_r.shape[0])
        ax.plot(t_idx[idx], np.abs(phi_f[idx] - phi_r[idx]), color=color, lw=1)
        ax.axhline(threshold, color="red", ls="--", lw=0.8)
        ax.set_xlabel("t [yr]")
        ax.set_ylabel("|ΔΦ_φ| [rad]")
        ax.set_title(
            f"{label}\n"
            f"a={stats['a'][i]:.2f}, e₀={stats['e0'][i]:.2f}, "
            f"p₀={stats['p0'][i]:.1f}\n"
            f"M={stats['M'][i]:.1e}, μ={stats['mu'][i]:.1f}"
        )
        ax.set_yscale("symlog", linthresh=1e-10)

    # Pick up to 3 failing and 3 passing examples
    fail_idx = np.where(fail)[0]
    pass_idx = np.where(bc & ~fail)[0]

    for j, (ax, col) in enumerate(zip(axes[0], ["red", "darkorange", "tomato"])):
        if j < len(fail_idx):
            _plot_traj(ax, fail_idx[j], "FAIL", col)
        else:
            ax.set_visible(False)

    for j, (ax, col) in enumerate(zip(axes[1], ["steelblue", "teal", "slateblue"])):
        if j < len(pass_idx):
            _plot_traj(ax, pass_idx[j], "PASS", col)
        else:
            ax.set_visible(False)

    axes[0, 0].set_title("Failing trajectories\n" + axes[0, 0].get_title()
                          if fail_idx.size > 0 else "No failing trajectories", fontsize=9)
    plt.suptitle(f"|ΔΦ_φ|(t) for sample trajectories  (threshold={threshold:.0e} rad)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "example_trajectories.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(stats: dict, path: Path):
    """Write per-trajectory diagnostics to a CSV file."""
    import csv

    rows = []
    N = stats["N_total"]
    for i in range(N):
        rows.append({
            "idx": i,
            "M":   stats["M"][i],
            "mu":  stats["mu"][i],
            "a":   stats["a"][i],
            "p0":  stats["p0"][i],
            "e0":  stats["e0"][i],
            "q":   stats["q"][i],
            "dp":  stats["dp"][i],
            "compl_ref":   stats["compl_ref"][i],
            "compl_fast":  stats["compl_fast"][i],
            "t_end_ref_yr":  stats["t_end_ref"][i],
            "t_end_fast_yr": stats["t_end_fast"][i],
            "phi_ref_end":   stats["phi_ref_end"][i],
            "phi_fast_end":  stats["phi_fast_end"][i],
            "delta_phi_final": stats["delta_phi"][i],
            "delta_phi_rms":   stats["delta_phi_rms"][i],
            "both_complete": int(stats["both_complete"][i]),
            "fails":         int(stats["fails"][i]),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose EMRIInspiral phases=True vs phases=False consistency."
    )
    p.add_argument("--data-dir", default=None,
                   help="Path to FEW HDF5 data files (overrides FEW_DATA_DIR).")
    p.add_argument("-N", "--N", type=int, default=2048,
                   help="Number of random parameter samples (default: 2048).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--T", type=float, default=2.0,
                   help="Inspiral duration [years] (default: 2.0).")
    p.add_argument("--dense-steps", type=int, default=500,
                   help="Number of trajectory output time steps (default: 500).")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for vmapped evaluation (default: 256).")
    p.add_argument("--threshold", type=float, default=P_CONSISTENCY_THRESHOLD,
                   help=f"Relative p consistency threshold |Δp|/p₀ (default: {P_CONSISTENCY_THRESHOLD}).")
    p.add_argument("--max-steps", type=int, default=4096,
                   help="Max ODE internal steps (default: 4096).")
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation.")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: diagnose_output/results.csv).")
    p.add_argument("--plot-dir", default="diagnose_plots",
                   help="Directory for output plots.")
    return p.parse_args()


def main():
    args = parse_args()
    out_csv = Path(args.out) if args.out else Path("diagnose_output/results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- Device info -------------------------------------------------------
    devices = jax.devices()
    print(f"\n  JAX devices: {[str(d) for d in devices]}")
    print(f"  Running on {devices[0].platform.upper()}")

    # ---- Load data & build trajectories ------------------------------------
    data_dir = find_data_dir(args.data_dir)
    print(f"  FEW data directory: {data_dir}\n")

    from fewtrax.data import load_flux_data
    from fewtrax.trajectory.inspiral import EMRIInspiral

    print("  Loading flux data …", end=" ", flush=True)
    flux_data = load_flux_data(data_dir)
    print("done")

    traj_ref  = EMRIInspiral(flux_data, phases=True)   # full 5D ODE
    traj_fast = EMRIInspiral(flux_data, phases=False)  # 2D (p,e) sub-system

    # ---- Build parameter grid ----------------------------------------------
    print(f"  Building parameter grid (N={args.N}, seed={args.seed}, "
          f"T={args.T} yr) …", end=" ", flush=True)
    grid = build_param_grid(args.N, seed=args.seed, T_yr=args.T)
    print("done")

    print(f"  p₀ range : [{grid['p0'].min():.2f}, {grid['p0'].max():.2f}] M")
    print(f"  e₀ range : [{grid['e0'].min():.3f}, {grid['e0'].max():.3f}]")
    print(f"  a  range : [{grid['a'].min():.3f},  {grid['a'].max():.3f}]")
    print(f"  M  range : [{grid['M'].min():.2e}, {grid['M'].max():.2e}] Msun")
    print(f"  mu range : [{grid['mu'].min():.1f}, {grid['mu'].max():.1f}] Msun")
    print(f"  q  range : [{grid['q'].min():.2e}, {grid['q'].max():.2e}]")
    print()

    # ---- JIT warm-up -------------------------------------------------------
    print("  Warming up JIT …", end=" ", flush=True)
    _wm_r  = jax.jit(jax.vmap(
        lambda p0, e0, a, M, mu: traj_ref(
            p0=p0, e0=e0, a=a, M=M, mu=mu,
            T=args.T, x0=1.0, dt=10.0,
            dense_steps=args.dense_steps, max_steps=args.max_steps,
            atol=args.atol, rtol=args.rtol,
        )
    ))
    _wm_f  = jax.jit(jax.vmap(
        lambda p0, e0, a, M, mu: traj_fast(
            p0=p0, e0=e0, a=a, M=M, mu=mu,
            T=args.T, x0=1.0, dt=10.0,
            dense_steps=args.dense_steps, max_steps=args.max_steps,
            atol=args.atol, rtol=args.rtol,
        )
    ))

    _p1 = jnp.array(grid["p0"][:4], dtype=jnp.float64)
    _e1 = jnp.array(grid["e0"][:4], dtype=jnp.float64)
    _a1 = jnp.array(grid["a"][:4],  dtype=jnp.float64)
    _M1 = jnp.array(grid["M"][:4],  dtype=jnp.float64)
    _m1 = jnp.array(grid["mu"][:4], dtype=jnp.float64)
    block_jax(_wm_r(_p1, _e1, _a1, _M1, _m1))
    block_jax(_wm_f(_p1, _e1, _a1, _M1, _m1))
    print("done\n")

    # ---- Run diagnostic ----------------------------------------------------
    print("=" * 68)
    print("  Phase accuracy diagnostic")
    print("=" * 68)

    t0 = time.perf_counter()
    stats = run_diagnostic(
        traj_ref, traj_fast, grid,
        dense_steps=args.dense_steps,
        T=args.T,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Total diagnostic time: {elapsed:.1f} s")

    # ---- Identify problematic parameter regions ----------------------------
    bc   = stats["both_complete"]
    fail = stats["fails"]

    if fail.sum() > 0:
        print("\n  *** Failing trajectories (|ΔΦ_φ| > threshold) ***")
        print(f"  {'idx':>6}  {'M':>9}  {'mu':>6}  {'a':>6}  {'p0':>7}  "
              f"{'e0':>6}  {'|ΔΦ_φ|':>12}  {'compl_r':>8}  {'compl_f':>8}")
        print("  " + "-" * 78)
        fail_idx = np.where(fail)[0]
        # Sort by dephasing (largest first)
        order = np.argsort(-stats["delta_phi"][fail_idx])
        for i in fail_idx[order]:
            print(
                f"  {i:>6d}  {stats['M'][i]:>9.2e}  {stats['mu'][i]:>6.1f}  "
                f"{stats['a'][i]:>6.3f}  {stats['p0'][i]:>7.3f}  "
                f"{stats['e0'][i]:>6.3f}  {stats['delta_phi'][i]:>12.3e}  "
                f"{stats['compl_ref'][i]:>8.3f}  {stats['compl_fast'][i]:>8.3f}"
            )
    else:
        print(f"\n  All {bc.sum()} complete trajectories pass "
              f"|ΔΦ_φ| < {args.threshold:.0e} rad.")

    # ---- Save CSV ----------------------------------------------------------
    save_csv(stats, out_csv)

    # ---- Plots -------------------------------------------------------------
    if not args.no_plots:
        make_plots(stats, Path(args.plot_dir), T=args.T, threshold=args.threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Waveform accuracy: fewtrax vs FastEMRIWaveforms.

Compares the full h+, h× strain output of fewtrax against the FEW reference
implementation (FastKerrEccentricEquatorialFlux) and reports:

  • Overlap and mismatch (white-noise inner product)
  • Peak strain ratio
  • RMS relative error on h+
  • Number of modes selected by each model

The comparison is run over the PARAM_SUITE defined in utils.py.

Usage
-----
    python compare_waveform.py [/path/to/few/data] [--plot] [--threshold 1e-5]
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
    PARAMS_DEFAULT,
    PARAM_SUITE,
    overlap,
    mismatch,
    rms_relative_error,
    print_header,
    print_table,
    timer,
)


# ---------------------------------------------------------------------------
# FEW waveform wrapper
# ---------------------------------------------------------------------------

def run_few_waveform(params: dict):
    """Generate waveform with FastKerrEccentricEquatorialFlux.

    Returns (hp, hx) as numpy arrays.
    """
    from few.waveform import GenerateEMRIWaveform
    gen = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux")
    h = gen(
        params["M"], params["mu"],
        params["a"],
        params["p0"], params["e0"], params["x0"],
        params.get("dist", 1.0),
        params.get("qS", 0.0), params.get("phiS", 0.0),
        params.get("qK", 0.0), params.get("phiK", 0.0),
        params.get("Phi_phi0", 0.0),
        params.get("Phi_theta0", 0.0),
        params.get("Phi_r0", 0.0),
        T=params["T"],
        dt=params["dt"],
    )
    hp = np.real(np.asarray(h))
    hx = -np.imag(np.asarray(h))
    return hp, hx


# ---------------------------------------------------------------------------
# fewtrax waveform wrapper
# ---------------------------------------------------------------------------

def run_fewtrax_waveform(params: dict, wf_gen, threshold: float = None):
    """Generate waveform with fewtrax.

    Returns (hp, hx, n_modes) where n_modes is the number of modes selected.
    """
    kwargs = {}
    if threshold is not None:
        kwargs["mode_selection_threshold"] = threshold

    hp, hx, sparse = wf_gen(
        M=params["M"], mu=params["mu"], a=params["a"],
        p0=params["p0"], e0=params["e0"], x0=params.get("x0", 1.0),
        dist=params.get("dist", 1.0),
        qS=params.get("qS", 0.0), phiS=params.get("phiS", 0.0),
        qK=params.get("qK", 0.0), phiK=params.get("phiK", 0.0),
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
        T=params["T"], dt=params["dt"],
        return_sparse=True,
        **kwargs,
    )
    n_modes = int(sparse["teuk_modes"].shape[1])
    return np.asarray(hp), np.asarray(hx), n_modes


# ---------------------------------------------------------------------------
# Single-parameter-set comparison
# ---------------------------------------------------------------------------

def compare_one(params: dict, wf_gen, threshold: float) -> dict:
    label = params.get("label", "unnamed")

    import time

    t0 = time.perf_counter()
    try:
        hp_few, hx_few = run_few_waveform(params)
    except Exception as exc:
        return dict(label=label, error=f"FEW failed: {exc}")
    t_few = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        hp_ft, hx_ft, n_modes_ft = run_fewtrax_waveform(params, wf_gen, threshold)
    except Exception as exc:
        return dict(label=label, error=f"fewtrax failed: {exc}")
    t_ft = time.perf_counter() - t0

    ov_hp = overlap(hp_few, hp_ft)
    ov_hx = overlap(hx_few, hx_ft)
    mm_hp = mismatch(hp_few, hp_ft)
    mm_hx = mismatch(hx_few, hx_ft)
    rms_hp = rms_relative_error(hp_few, hp_ft)

    peak_few = float(np.max(np.abs(hp_few)))
    peak_ft  = float(np.max(np.abs(hp_ft)))
    peak_ratio = peak_ft / peak_few if peak_few > 1e-40 else float("nan")

    n_few = len(hp_few)
    n_ft  = len(hp_ft)

    return dict(
        label=label,
        overlap_hp=ov_hp, overlap_hx=ov_hx,
        mismatch_hp=mm_hp, mismatch_hx=mm_hx,
        rms_hp=rms_hp,
        peak_ratio=peak_ratio,
        n_samples_few=n_few, n_samples_ft=n_ft,
        n_modes_ft=n_modes_ft,
        t_few_s=t_few, t_ft_s=t_ft,
        # Store for plotting
        hp_few=hp_few, hx_few=hx_few,
        hp_ft=hp_ft, hx_ft=hx_ft,
        dt=params["dt"],
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(result: dict, out_dir: str = ".") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots.")
        return

    label = result["label"]
    dt = result["dt"]
    n = min(result["n_samples_few"], result["n_samples_ft"])
    t = np.arange(n) * dt / 86400.0  # days

    hp_few = result["hp_few"][:n]
    hp_ft  = result["hp_ft"][:n]
    hx_few = result["hx_few"][:n]
    hx_ft  = result["hx_ft"][:n]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    scale = 1.0 / max(float(np.max(np.abs(hp_few))), 1e-40)

    # h+ comparison
    axes[0].plot(t, hp_few * scale, label="FEW",     lw=1.2, alpha=0.9)
    axes[0].plot(t, hp_ft  * scale, label="fewtrax", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_ylabel(r"$h_+ / h_{+,\max}^{\rm FEW}$")
    axes[0].set_title(
        fr"Waveform comparison – {label}  "
        fr"[overlap $h_+$ = {result['overlap_hp']:.4f}]"
    )
    axes[0].legend(fontsize=9)

    # h× comparison
    axes[1].plot(t, hx_few * scale, label="FEW",     lw=1.2, alpha=0.9)
    axes[1].plot(t, hx_ft  * scale, label="fewtrax", lw=1.0, ls="--", alpha=0.8)
    axes[1].set_ylabel(r"$h_\times / h_{+,\max}^{\rm FEW}$")
    axes[1].legend(fontsize=9)

    # Residual
    residual = (hp_ft - hp_few) * scale
    axes[2].plot(t, residual, lw=0.8, color="C2")
    axes[2].axhline(0, color="k", lw=0.5, ls="--")
    axes[2].set_ylabel(r"$\Delta h_+ / h_{+,\max}^{\rm FEW}$")
    axes[2].set_xlabel("Time [days]")
    axes[2].set_title(
        fr"Residual  (RMS = {result['rms_hp']:.2e},"
        fr"  mismatch = {result['mismatch_hp']:.2e})"
    )

    plt.tight_layout()
    out_path = f"{out_dir}/waveform_{label}.png"
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
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="Mode selection threshold (default: 1e-5)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", default=".")
    parser.add_argument("--single", action="store_true",
                        help="Run only PARAMS_DEFAULT (quick check)")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    print(f"Using FEW data directory: {data_dir}")

    print("\nInitialising fewtrax waveform generator …")
    from fewtrax import KerrEccentricEquatorialWaveform
    wf_gen = KerrEccentricEquatorialWaveform(
        data_dir=data_dir,
        mode_selection_threshold=args.threshold,
        dense_steps=100,
    )
    print("  Done.")

    suite = [PARAMS_DEFAULT] if args.single else PARAM_SUITE
    if args.single:
        suite[0]["label"] = "default"

    print_header(f"Waveform accuracy: fewtrax vs FEW  (threshold={args.threshold:.0e})")

    results = []
    for params in suite:
        label = params.get("label", "?")
        print(f"\n--- {label} ---")
        try:
            res = compare_one(params, wf_gen, args.threshold)
            results.append(res)
            if "error" in res:
                print(f"  ERROR: {res['error']}")
            else:
                print(f"  Samples:    FEW={res['n_samples_few']},  fewtrax={res['n_samples_ft']}")
                print(f"  Modes (fewtrax): {res['n_modes_ft']}")
                print(f"  Overlap h+: {res['overlap_hp']:.6f}   Mismatch: {res['mismatch_hp']:.2e}")
                print(f"  Overlap h×: {res['overlap_hx']:.6f}   Mismatch: {res['mismatch_hx']:.2e}")
                print(f"  Peak strain ratio (ft/FEW): {res['peak_ratio']:.4f}")
                print(f"  h+ RMS relative error: {res['rms_hp']:.2e}")
                print(f"  Wall time:  FEW={res['t_few_s']:.2f}s,  fewtrax={res['t_ft_s']:.2f}s")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append(dict(label=label, error=str(exc)))

    print_header("Summary table")
    ok = [r for r in results if "error" not in r]
    if ok:
        headers = ["label", "overlap h+", "mismatch h+", "peak ratio", "modes", "t_FEW [s]", "t_ft [s]"]
        widths  = [14, 12, 13, 12, 7, 12, 10]
        rows = [
            (
                r["label"],
                f"{r['overlap_hp']:.5f}",
                f"{r['mismatch_hp']:.2e}",
                f"{r['peak_ratio']:.4f}",
                str(r["n_modes_ft"]),
                f"{r['t_few_s']:.2f}",
                f"{r['t_ft_s']:.2f}",
            )
            for r in ok
        ]
        print_table(rows, headers, widths)

    # Pass/fail summary
    print()
    n_pass = sum(1 for r in ok if r["overlap_hp"] > 0.9)
    print(f"Overlap > 0.90: {n_pass}/{len(ok)} parameter sets")

    if args.plot:
        print("\nSaving figures …")
        for res in results:
            if "error" not in res:
                plot_comparison(res, out_dir=args.plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

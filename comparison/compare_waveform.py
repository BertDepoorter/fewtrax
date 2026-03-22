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
    python compare_waveform.py [/path/to/few/data] [--plot] [--all-modes]
         Adds a second comparison pass with all 6993 Teukolsky modes to
         separate dephasing (phase error) from amplitude interpolation error.
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
# Chunked mode summation (avoids the N_dense × N_modes OOM)
# ---------------------------------------------------------------------------

def _interpolated_mode_sum_chunked(
    t_sparse: np.ndarray,
    teuk_sparse: np.ndarray,
    ylms_pos: np.ndarray,
    ylms_neg: np.ndarray,
    Phi_phi_dense: np.ndarray,
    Phi_theta_dense: np.ndarray,
    Phi_r_dense: np.ndarray,
    l_arr: np.ndarray,
    m_arr: np.ndarray,
    k_arr: np.ndarray,
    n_arr: np.ndarray,
    amp_prefactor: float,
    chunk_size: int = 100,
) -> np.ndarray:
    """Sum all modes in chunks to stay within memory.

    Avoids constructing the full (N_dense, N_modes) array which would be
    ~35 GB at 6993 modes × 315 k samples.  Each chunk of ``chunk_size``
    modes is amplitude-interpolated and phase-summed independently, and
    the partial sums are accumulated into a (N_dense,) complex array.

    Parameters
    ----------
    t_sparse : (N_sparse,) float
        Sparse time stamps [s].
    teuk_sparse : (N_sparse, N_modes) complex
        Mode amplitudes at sparse trajectory points.
    ylms_pos, ylms_neg : (N_modes,) complex
        Spherical harmonics.
    Phi_phi_dense, Phi_theta_dense, Phi_r_dense : (N_dense,) float
        Orbital phases at the dense output grid.
    l_arr, m_arr, k_arr, n_arr : (N_modes,) int
        Mode indices.
    amp_prefactor : float
        Physical amplitude prefactor (μ G / c² / d_L).
    chunk_size : int
        Number of modes per chunk.

    Returns
    -------
    h : (N_dense,) complex128
        Full strain h+ − i h×.
    """
    from interpax import Interpolator1D

    N_dense = len(Phi_phi_dense)
    N_modes = teuk_sparse.shape[1]

    # Phase arrays as plain numpy (computed once)
    Pp = np.asarray(Phi_phi_dense)
    Pt = np.asarray(Phi_theta_dense)
    Pr = np.asarray(Phi_r_dense)
    t_dense_np = np.linspace(t_sparse[0], t_sparse[-1], N_dense)

    h = np.zeros(N_dense, dtype=np.complex128)

    for start in range(0, N_modes, chunk_size):
        end = min(start + chunk_size, N_modes)
        nc = end - start

        # Interpolate amplitudes for this chunk
        teuk_chunk = np.zeros((N_dense, nc), dtype=np.complex128)
        for ic in range(nc):
            im = start + ic
            col = teuk_sparse[:, im]
            spl_r = Interpolator1D(t_sparse, col.real, method="cubic2", extrap=True)
            spl_i = Interpolator1D(t_sparse, col.imag, method="cubic2", extrap=True)
            teuk_chunk[:, ic] = (
                np.asarray(spl_r(t_dense_np)) + 1j * np.asarray(spl_i(t_dense_np))
            )

        # Mode indices for this chunk
        m_c = m_arr[start:end]
        k_c = k_arr[start:end]
        n_c = n_arr[start:end]
        l_c = l_arr[start:end]
        yp_c = ylms_pos[start:end]
        yn_c = ylms_neg[start:end]

        # Phase: (N_dense, nc) — compute in numpy to avoid JAX memory
        phase = (
            m_c[np.newaxis, :] * Pp[:, np.newaxis]
            + k_c[np.newaxis, :] * Pt[:, np.newaxis]
            + n_c[np.newaxis, :] * Pr[:, np.newaxis]
        )  # (N_dense, nc)

        exp_neg = np.exp(-1j * phase)          # (N_dense, nc)
        exp_pos = np.exp( 1j * phase)

        # Positive-m contribution
        h += np.sum(yp_c[np.newaxis, :] * teuk_chunk * exp_neg, axis=1)

        # Negative-m contribution (m > 0 only)
        m_pos_mask = (m_c > 0).astype(float)
        sign_l = (-1.0) ** l_c
        h += np.sum(
            m_pos_mask[np.newaxis, :]
            * sign_l[np.newaxis, :]
            * yn_c[np.newaxis, :]
            * np.conj(teuk_chunk)
            * exp_pos,
            axis=1,
        )

    return h * amp_prefactor


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


def run_fewtrax_waveform_all_modes(params: dict, wf_gen) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate fewtrax waveform with ALL Teukolsky modes (threshold=0).

    Uses chunked summation to avoid the ~35 GB (N_dense × N_modes) matrix.
    Returns (hp, hx, n_modes).
    """
    import jax.numpy as jnp
    from fewtrax.utils.constants import G_SI, C_SI, MSUN_SI, GPC_SI
    from fewtrax.utils.harmonics import get_ylms_for_modes
    from fewtrax.waveform.kerr import _get_viewing_angles, _to_ssb_frame
    from interpax import Interpolator1D

    # --- Force threshold=0 to select all modes ---
    old_thresh = wf_gen.mode_selection_threshold
    wf_gen.mode_selection_threshold = 0.0
    sparse = wf_gen.generate_sparse(
        M=params["M"], mu=params["mu"], a=params["a"],
        p0=params["p0"], e0=params["e0"], x0=params.get("x0", 1.0),
        T=params["T"], dt=params["dt"],
        Phi_phi0=params.get("Phi_phi0", 0.0),
        Phi_theta0=params.get("Phi_theta0", 0.0),
        Phi_r0=params.get("Phi_r0", 0.0),
    )
    wf_gen.mode_selection_threshold = old_thresh

    n_modes = sparse["teuk_modes"].shape[1]

    # --- Frame angles ---
    x0 = params.get("x0", 1.0)
    qK_eff = params.get("qK", 0.0)
    phiK_eff = params.get("phiK", 0.0)
    Phi_phi0_eff = params.get("Phi_phi0", 0.0)
    if x0 < 0.0:
        qK_eff = float(np.pi - qK_eff)
        phiK_eff = float(phiK_eff + np.pi)
        Phi_phi0_eff = float(Phi_phi0_eff + np.pi)

    theta_obs, phi_obs = _get_viewing_angles(
        float(params.get("qS", 0.0)), float(params.get("phiS", 0.0)),
        qK_eff, phiK_eff,
    )
    ylms_pos, ylms_neg = get_ylms_for_modes(
        sparse["l_arr"], sparse["m_arr"], theta_obs, phi_obs
    )
    ylms_pos = np.asarray(ylms_pos)
    ylms_neg = np.asarray(ylms_neg)

    # --- Dense phase grid ---
    t_sparse = np.asarray(sparse["t"])
    valid = np.isfinite(t_sparse)
    t_sparse = t_sparse[valid]
    Pp = np.asarray(sparse["Phi_phi"])[valid]
    Pt = np.asarray(sparse["Phi_theta"])[valid]
    Pr = np.asarray(sparse["Phi_r"])[valid]

    dt = params["dt"]
    T_s = float(t_sparse[-1])
    n_dense = max(2, int(np.round(T_s / dt)) + 1)
    t_dense_np = np.linspace(t_sparse[0], t_sparse[-1], n_dense)

    def _interp(arr):
        spl = Interpolator1D(t_sparse, arr, method="cubic2", extrap=True)
        return np.asarray(spl(t_dense_np))

    Phi_phi_d   = _interp(Pp)
    Phi_theta_d = _interp(Pt)
    Phi_r_d     = _interp(Pr)

    # --- Amplitude prefactor ---
    mu  = params["mu"]
    dist = params.get("dist", 1.0)
    amp = mu * MSUN_SI * G_SI / C_SI**2 / (dist * GPC_SI)

    teuk_sparse = np.asarray(sparse["teuk_modes"])[valid, :]

    # --- Chunked summation ---
    h = _interpolated_mode_sum_chunked(
        t_sparse, teuk_sparse,
        ylms_pos, ylms_neg,
        Phi_phi_d, Phi_theta_d, Phi_r_d,
        sparse["l_arr"], sparse["m_arr"], sparse["k_arr"], sparse["n_arr"],
        amp_prefactor=amp,
    )

    # Source-frame sign convention
    h = h * (-1.0)

    hp = np.real(h)
    hx = -np.imag(h)

    hp_j = jnp.asarray(hp)
    hx_j = jnp.asarray(hx)
    hp_j, hx_j = _to_ssb_frame(
        hp_j, hx_j,
        float(params.get("qS", 0.0)), float(params.get("phiS", 0.0)),
        qK_eff, phiK_eff,
    )
    return np.asarray(hp_j), np.asarray(hx_j), n_modes


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
    """Save time-domain and frequency-domain comparison figures."""
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

    # ------------------------------------------------------------------
    # Figure 1: time domain
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    scale = 1.0 / max(float(np.max(np.abs(hp_few))), 1e-40)

    # h+ comparison
    axes[0].plot(t, hp_few * scale, label="FEW",     lw=1.2, alpha=0.9)
    axes[0].plot(t, hp_ft  * scale, label="fewtrax", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_ylabel(r"$h_+ / h_{+,\max}^{\rm FEW}$")
    axes[0].set_title(
        fr"Waveform comparison – {label}  "
        fr"[overlap $h_+$ = {result['overlap_hp']:.4f},  "
        fr"mismatch = {result['mismatch_hp']:.2e}]"
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
        fr"Residual  (RMS = {result['rms_hp']:.2e})"
    )

    plt.tight_layout()
    out_path = f"{out_dir}/waveform_{label}_time.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ------------------------------------------------------------------
    # Figure 1b: first 3 days zoom
    # ------------------------------------------------------------------
    _plot_early_time(
        hp_few=hp_few, hx_few=hx_few,
        hp_ft=hp_ft,   hx_ft=hx_ft,
        t=t, scale=scale, label=label, result=result, out_dir=out_dir,
        zoom_days=3.0,
    )

    # ------------------------------------------------------------------
    # Figure 2: frequency domain
    # ------------------------------------------------------------------
    _plot_frequency_domain(
        hp_few=hp_few, hx_few=hx_few,
        hp_ft=hp_ft,   hx_ft=hx_ft,
        dt=dt, label=label, result=result, out_dir=out_dir,
    )


def _plot_early_time(
    hp_few: np.ndarray,
    hx_few: np.ndarray,
    hp_ft: np.ndarray,
    hx_ft: np.ndarray,
    t: np.ndarray,
    scale: float,
    label: str,
    result: dict,
    out_dir: str,
    zoom_days: float = 3.0,
) -> None:
    """Save a 3-panel zoom of the first ``zoom_days`` days of the waveform."""
    import matplotlib.pyplot as plt

    mask = t <= zoom_days
    if mask.sum() < 2:
        return  # waveform shorter than zoom window

    t_z      = t[mask]
    hp_few_z = hp_few[mask]
    hp_ft_z  = hp_ft[mask]
    hx_few_z = hx_few[mask]
    hx_ft_z  = hx_ft[mask]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    # h+ comparison
    axes[0].plot(t_z, hp_few_z * scale, label="FEW",     lw=1.2, alpha=0.9)
    axes[0].plot(t_z, hp_ft_z  * scale, label="fewtrax", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_ylabel(r"$h_+ / h_{+,\max}^{\rm FEW}$")
    axes[0].set_title(
        fr"Waveform comparison – {label}  (first {zoom_days:.0f} days)  "
        fr"[overlap $h_+$ = {result['overlap_hp']:.4f},  "
        fr"mismatch = {result['mismatch_hp']:.2e}]"
    )
    axes[0].legend(fontsize=9)

    # h× comparison
    axes[1].plot(t_z, hx_few_z * scale, label="FEW",     lw=1.2, alpha=0.9)
    axes[1].plot(t_z, hx_ft_z  * scale, label="fewtrax", lw=1.0, ls="--", alpha=0.8)
    axes[1].set_ylabel(r"$h_\times / h_{+,\max}^{\rm FEW}$")
    axes[1].legend(fontsize=9)

    # Residual
    residual_z = (hp_ft_z - hp_few_z) * scale
    axes[2].plot(t_z, residual_z, lw=0.8, color="C2")
    axes[2].axhline(0, color="k", lw=0.5, ls="--")
    axes[2].set_ylabel(r"$\Delta h_+ / h_{+,\max}^{\rm FEW}$")
    axes[2].set_xlabel("Time [days]")
    axes[2].set_title(fr"Residual over first {zoom_days:.0f} days")

    plt.tight_layout()
    out_path = f"{out_dir}/waveform_{label}_early.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_frequency_domain(
    hp_few: np.ndarray,
    hx_few: np.ndarray,
    hp_ft: np.ndarray,
    hx_ft: np.ndarray,
    dt: float,
    label: str,
    result: dict,
    out_dir: str,
    window: str = "hann",
    zero_pad: bool = True,
    out_suffix: str = "freq",
    title_suffix: str = "",
) -> None:
    """Save a 3-panel frequency-domain comparison figure."""
    import matplotlib.pyplot as plt
    from fewtrax.utils.transforms import to_frequency_domain

    def _fd(arr):
        """FFT with Hann window and zero-padding; returns (f, |h̃|)."""
        f, h_tilde = to_frequency_domain(
            jnp.asarray(arr, dtype=jnp.float64),
            dt=dt,
            window=window,
            zero_pad=zero_pad,
        )
        return np.asarray(f), np.abs(np.asarray(h_tilde))

    f_hp_few, amp_hp_few = _fd(hp_few)
    f_hp_ft,  amp_hp_ft  = _fd(hp_ft)
    f_hx_few, amp_hx_few = _fd(hx_few)
    f_hx_ft,  amp_hx_ft  = _fd(hx_ft)

    # Nyquist and a sensible frequency floor (skip DC and first bin)
    f_nyq = 0.5 / dt
    f_min_plot = max(f_hp_few[1], 1e-5)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    # |h̃+(f)|
    axes[0].loglog(f_hp_few, amp_hp_few, label="FEW",     lw=1.2, alpha=0.9)
    axes[0].loglog(f_hp_ft,  amp_hp_ft,  label="fewtrax", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_xlim(f_min_plot, f_nyq)
    axes[0].set_ylabel(r"$|\tilde{h}_+(f)|$  [strain/Hz]")
    axes[0].set_title(
        fr"Frequency-domain comparison – {label}{title_suffix}  "
        fr"({window} window, zero-pad={zero_pad})"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which="both", ls=":", alpha=0.4)

    # |h̃×(f)|
    axes[1].loglog(f_hx_few, amp_hx_few, label="FEW",     lw=1.2, alpha=0.9)
    axes[1].loglog(f_hx_ft,  amp_hx_ft,  label="fewtrax", lw=1.0, ls="--", alpha=0.8)
    axes[1].set_xlim(f_min_plot, f_nyq)
    axes[1].set_ylabel(r"$|\tilde{h}_\times(f)|$  [strain/Hz]")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which="both", ls=":", alpha=0.4)

    # Amplitude ratio (fewtrax / FEW) for h+
    # Interpolate to a common frequency grid for the ratio
    f_common = f_hp_few  # FEW grid (both should be the same length after zero-pad)
    amp_ft_interp = np.interp(f_common, f_hp_ft, amp_hp_ft)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(amp_hp_few > 1e-50, amp_ft_interp / amp_hp_few, np.nan)

    axes[2].semilogx(f_common, ratio, lw=0.9, color="C2")
    axes[2].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[2].set_xlim(f_min_plot, f_nyq)
    axes[2].set_ylim(0, 2)
    axes[2].set_ylabel(r"$|\tilde{h}_+^{\rm ft}| \,/\, |\tilde{h}_+^{\rm FEW}|$")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_title("Spectral amplitude ratio (fewtrax / FEW)")
    axes[2].grid(True, which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    out_path = f"{out_dir}/waveform_{label}_{out_suffix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# All-modes comparison
# ---------------------------------------------------------------------------

def compare_all_modes_one(params: dict, wf_gen) -> dict:
    """Compare FEW (default mode selection) vs fewtrax (all 6993 modes).

    Separates dephasing from amplitude interpolation error:
      * Dephasing: residual grows linearly with time regardless of N_modes.
      * Amplitude error: residual character changes significantly as modes
        are added.
    """
    import time

    label = params.get("label", "unnamed")

    t0 = time.perf_counter()
    try:
        hp_few, hx_few = run_few_waveform(params)
    except Exception as exc:
        return dict(label=label, error=f"FEW failed: {exc}")
    t_few = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        hp_ft, hx_ft, n_modes_ft = run_fewtrax_waveform_all_modes(params, wf_gen)
    except Exception as exc:
        return dict(label=label, error=f"fewtrax (all modes) failed: {exc}")
    t_ft = time.perf_counter() - t0

    ov_hp = overlap(hp_few, hp_ft)
    ov_hx = overlap(hx_few, hx_ft)
    mm_hp = mismatch(hp_few, hp_ft)
    mm_hx = mismatch(hx_few, hx_ft)
    rms_hp = rms_relative_error(hp_few, hp_ft)
    peak_few = float(np.max(np.abs(hp_few)))
    peak_ft  = float(np.max(np.abs(hp_ft)))
    peak_ratio = peak_ft / peak_few if peak_few > 1e-40 else float("nan")

    return dict(
        label=label,
        overlap_hp=ov_hp, overlap_hx=ov_hx,
        mismatch_hp=mm_hp, mismatch_hx=mm_hx,
        rms_hp=rms_hp,
        peak_ratio=peak_ratio,
        n_samples_few=len(hp_few), n_samples_ft=len(hp_ft),
        n_modes_ft=n_modes_ft,
        t_few_s=t_few, t_ft_s=t_ft,
        hp_few=hp_few, hx_few=hx_few,
        hp_ft=hp_ft, hx_ft=hx_ft,
        dt=params["dt"],
    )


def plot_all_modes_comparison(result: dict, out_dir: str = ".") -> None:
    """Save time-domain comparison figures for the all-modes run.

    Produces two figures:
      * ``waveform_{label}_allmode_time.png``  — full time domain
      * ``waveform_{label}_allmode_early.png`` — first 3 days zoom

    The residual panel is the key diagnostic:
      * Linearly growing residual → dephasing (phase error in trajectory).
      * Oscillatory or amplitude-scaled residual → amplitude interpolation error.
    """
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
    scale = 1.0 / max(float(np.max(np.abs(hp_few))), 1e-40)

    def _make_panel(t_plot, hp_few_p, hp_ft_p, hx_few_p, hx_ft_p, suffix, title_extra=""):
        fig, axes = plt.subplots(3, 1, figsize=(11, 10))

        axes[0].plot(t_plot, hp_few_p * scale, label="FEW (default modes)", lw=1.2, alpha=0.9)
        axes[0].plot(t_plot, hp_ft_p  * scale, label=f"fewtrax ({result['n_modes_ft']} modes)",
                     lw=1.0, ls="--", alpha=0.8)
        axes[0].set_ylabel(r"$h_+ / h_{+,\max}^{\rm FEW}$")
        axes[0].set_title(
            fr"All-modes comparison – {label}{title_extra}  "
            fr"[overlap = {result['overlap_hp']:.4f},  mismatch = {result['mismatch_hp']:.2e}]"
        )
        axes[0].legend(fontsize=9)

        axes[1].plot(t_plot, hx_few_p * scale, label="FEW", lw=1.2, alpha=0.9)
        axes[1].plot(t_plot, hx_ft_p  * scale, label="fewtrax", lw=1.0, ls="--", alpha=0.8)
        axes[1].set_ylabel(r"$h_\times / h_{+,\max}^{\rm FEW}$")
        axes[1].legend(fontsize=9)

        residual = (hp_ft_p - hp_few_p) * scale
        axes[2].plot(t_plot, residual, lw=0.8, color="C2")
        axes[2].axhline(0, color="k", lw=0.5, ls="--")
        axes[2].set_ylabel(r"$\Delta h_+ / h_{+,\max}^{\rm FEW}$")
        axes[2].set_xlabel("Time [days]")
        axes[2].set_title(
            fr"Residual  (RMS = {result['rms_hp']:.2e})  "
            fr"— linearly growing → dephasing; oscillatory → amplitude error"
        )

        plt.tight_layout()
        out_path = f"{out_dir}/waveform_{label}_allmode_{suffix}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    # Full time range
    _make_panel(t, hp_few, hp_ft, hx_few, hx_ft, "time")

    # First 3 days
    zoom_days = 3.0
    mask = t <= zoom_days
    if mask.sum() >= 2:
        _make_panel(
            t[mask], hp_few[mask], hp_ft[mask], hx_few[mask], hx_ft[mask],
            "early",
            title_extra=f"  (first {zoom_days:.0f} days)",
        )

    # Frequency domain
    _plot_frequency_domain(
        hp_few=hp_few, hx_few=hx_few,
        hp_ft=hp_ft,   hx_ft=hx_ft,
        dt=result["dt"], label=label, result=result, out_dir=out_dir,
        out_suffix="allmode_freq",
        title_suffix=f"  ({result['n_modes_ft']} modes)",
    )


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
    parser.add_argument("--all-modes", action="store_true",
                        help="Also run fewtrax with all 6993 modes (chunked summation) "
                             "to separate dephasing from amplitude interpolation error")
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
    n_pass = sum(1 for r in ok if r["overlap_hp"] > 0.99)
    print(f"Overlap > 0.99: {n_pass}/{len(ok)} parameter sets")

    if args.plot:
        print("\nSaving figures …")
        for res in results:
            if "error" not in res:
                plot_comparison(res, out_dir=args.plot_dir)

    # ------------------------------------------------------------------
    # All-modes pass
    # ------------------------------------------------------------------
    if args.all_modes:
        print_header(f"All-modes comparison: fewtrax (6993 modes) vs FEW (default)")
        print("Note: chunked numpy summation — no GPU/JAX memory limit.")
        all_mode_results = []
        for params in suite:
            label = params.get("label", "?")
            print(f"\n--- {label} ---")
            try:
                res = compare_all_modes_one(params, wf_gen)
                all_mode_results.append(res)
                if "error" in res:
                    print(f"  ERROR: {res['error']}")
                else:
                    print(f"  Modes (fewtrax all): {res['n_modes_ft']}")
                    print(f"  Overlap h+: {res['overlap_hp']:.6f}   Mismatch: {res['mismatch_hp']:.2e}")
                    print(f"  Peak strain ratio (ft/FEW): {res['peak_ratio']:.4f}")
                    print(f"  h+ RMS relative error: {res['rms_hp']:.2e}")
                    print(f"  Wall time: FEW={res['t_few_s']:.1f}s  fewtrax={res['t_ft_s']:.1f}s")
            except Exception as exc:
                import traceback
                print(f"  FAILED: {exc}")
                traceback.print_exc()
                all_mode_results.append(dict(label=label, error=str(exc)))

        ok_am = [r for r in all_mode_results if "error" not in r]
        if ok_am:
            print_header("All-modes summary table")
            headers = ["label", "overlap h+", "mismatch h+", "peak ratio", "modes", "t_ft [s]"]
            widths  = [14, 12, 13, 12, 7, 10]
            rows = [
                (
                    r["label"],
                    f"{r['overlap_hp']:.5f}",
                    f"{r['mismatch_hp']:.2e}",
                    f"{r['peak_ratio']:.4f}",
                    str(r["n_modes_ft"]),
                    f"{r['t_ft_s']:.1f}",
                )
                for r in ok_am
            ]
            print_table(rows, headers, widths)

        if args.plot:
            print("\nSaving all-modes figures …")
            for res in all_mode_results:
                if "error" not in res:
                    plot_all_modes_comparison(res, out_dir=args.plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

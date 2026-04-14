"""Direct WDM-domain generation of EMRI harmonic waveforms.

This module implements the direct deposition of EMRI Teukolsky harmonics
onto a WDM (Wilson-Daubechies-Meyer) time-frequency grid without first
constructing a dense time series.

The key formula (see ``research/tf_representation.md`` Sec. 3) is::

    W[q, n] = sum_{lmkn'} [ Y+_lm * A_lmkn'(t_n) * exp(-i Phi_lmkn'(t_n))
                             * K(xi_{nq}, chi_n) + c.c. ]

where ``t_n = n * dT`` is the WDM time-bin centre,
``xi_{nq} = (q*dF - f(t_n)) / dF`` is the normalised frequency offset, and
``chi_n = fdot * dT^2`` is the dimensionless chirp parameter.

The cost is O(Nt * N_modes * Hw) amplitude/phase evaluations (where
Hw = 2*hw+1 ~ 5) compared with O(N_dense * N_modes) for the time-domain
pipeline -- a reduction of N_dense/Nt ~ 1000 at LISA cadence.

API
---
:func:`direct_wdm_sum`
    Full coherent scatter-add over all harmonic modes.  Accepts the same
    sparse-trajectory data returned by
    :meth:`~fewtrax.waveform.KerrEccentricEquatorialWaveform.generate_sparse`.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp

from fewtrax.utils.constants import MTSUN_SI, GPC_SI, G_SI, C_SI, MSUN_SI
from fewtrax.utils.tf_tracks import WDMGrid, direct_wdm_mode


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interp_to_bins(
    t_traj: np.ndarray,
    arr: np.ndarray,
    t_bins: np.ndarray,
) -> np.ndarray:
    """Cubic-spline interpolation of a trajectory array onto WDM time bins.

    Parameters
    ----------
    t_traj : (N_traj,) float
        Sparse trajectory time stamps [s].  May contain NaNs at the end.
    arr : (N_traj, ...) float or complex
        Values to interpolate.  Leading dimension must match *t_traj*.
    t_bins : (Nt,) float
        WDM time-bin centres [s].

    Returns
    -------
    np.ndarray, shape (Nt, ...)
        Interpolated values.
    """
    from interpax import Interpolator1D

    valid = np.isfinite(t_traj)
    t_v = t_traj[valid]
    arr_v = np.asarray(arr)[valid]

    if arr_v.ndim == 1:
        spl = Interpolator1D(t_v, arr_v, method="cubic2", extrap=True)
        return np.asarray(spl(t_bins))

    # 2-D: shape (N_valid, N_cols) -- interpolate column by column
    n_cols = arr_v.shape[1]
    out = np.empty((len(t_bins), n_cols), dtype=arr_v.dtype)
    for ic in range(n_cols):
        spl = Interpolator1D(t_v, arr_v[:, ic], method="cubic2", extrap=True)
        out[:, ic] = np.asarray(spl(t_bins))
    return out


def _complex_interp_to_bins(
    t_traj: np.ndarray,
    arr: np.ndarray,
    t_bins: np.ndarray,
) -> np.ndarray:
    """Like :func:`_interp_to_bins` but for complex arrays (real+imag separately)."""
    re = _interp_to_bins(t_traj, np.real(arr), t_bins)
    im = _interp_to_bins(t_traj, np.imag(arr), t_bins)
    return re + 1j * im


def _fdot_from_freq(f_bins: np.ndarray, delta_T: float) -> np.ndarray:
    """Central-difference frequency derivative at WDM bin centres.

    Parameters
    ----------
    f_bins : (Nt,) float
        Instantaneous frequency at each WDM time bin [Hz].
    delta_T : float
        WDM time-bin width [s].

    Returns
    -------
    (Nt,) float  [Hz/s]
    """
    fdot = np.empty_like(f_bins)
    fdot[1:-1] = (f_bins[2:] - f_bins[:-2]) / (2.0 * delta_T)
    fdot[0]    = (f_bins[1]  - f_bins[0])   / delta_T
    fdot[-1]   = (f_bins[-1] - f_bins[-2])  / delta_T
    return fdot


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def direct_wdm_sum(
    t_traj: np.ndarray,
    teuk_modes: np.ndarray,
    Phi_phi: np.ndarray,
    Phi_theta: np.ndarray,
    Phi_r: np.ndarray,
    l_arr: np.ndarray,
    m_arr: np.ndarray,
    k_arr: np.ndarray,
    n_arr: np.ndarray,
    ylms_pos: np.ndarray,
    ylms_neg: np.ndarray,
    a: float,
    M: float,
    mu: float,
    x0: float,
    grid: WDMGrid,
    dist: float = 1.0,
    hw: int = 2,
    chirp_correction: bool = True,
) -> jnp.ndarray:
    r"""Coherent WDM scatter-add over all EMRI harmonic modes.

    Directly computes the ``Nf x Nt`` WDM strain matrix without constructing a
    dense time series, as described in ``research/tf_representation.md`` Sec. 3.

    The return value is the *complex WDM strain*::

        W[q, n] = W_plus[q, n] - 1j * W_cross[q, n]

    where ``W.real`` is the WDM transform of h_plus and ``-W.imag`` is the WDM
    transform of h_cross, consistent with the time-domain convention in
    :meth:`~fewtrax.waveform.KerrEccentricEquatorialWaveform.__call__`.

    Parameters
    ----------
    t_traj : (N_traj,) float
        Sparse trajectory time stamps [s] (may contain NaNs at end).
    teuk_modes : (N_traj, N_modes) complex
        Teukolsky mode amplitudes at each trajectory point.
    Phi_phi, Phi_theta, Phi_r : (N_traj,) float
        Accumulated orbital phases [rad].
    l_arr, m_arr, k_arr, n_arr : (N_modes,) int
        Mode index arrays.
    ylms_pos : (N_modes,) complex
        Spin-weighted spherical harmonic ``_{-2}Y_{lm}(theta, phi)`` for each mode.
    ylms_neg : (N_modes,) complex
        Spin-weighted spherical harmonic ``_{-2}Y_{l,-m}(theta, phi)`` for each mode.
    a : float
        Dimensionless Kerr spin parameter.
    M : float
        Primary mass [M_sun].
    mu : float
        Secondary mass [M_sun].
    x0 : float
        Inclination sign (+/-1).
    grid : WDMGrid
        Target WDM grid.
    dist : float
        Luminosity distance [Gpc].
    hw : int
        Frequency half-width for deposition (default 2).  The kernel is
        evaluated for bins ``q0 - hw, ..., q0 + hw``.
    chirp_correction : bool
        Apply the leading-order Fresnel chirp correction
        ``exp(-i*pi*chi*xi^2)`` (default True).

    Returns
    -------
    W : jnp.ndarray, shape (Nf, Nt), complex64
        Complex WDM strain.  ``W.real`` is the WDM of h_plus;
        ``-W.imag`` is the WDM of h_cross.

    Notes
    -----
    This function is **not** JIT-compiled as a whole: the interpolation
    step uses numpy/interpax on the CPU, while the scatter-add loop uses JAX.
    The JAX portion (kernel evaluation + scatter) is differentiable with
    respect to the pixel-centre evaluated quantities, but not through the
    interpolation.

    For fully differentiable WDM generation, obtain the interpolated
    quantities with JAX-native interpolators and call :func:`direct_wdm_mode`
    followed by :func:`scatter_wdm` directly.
    """
    t_np = np.asarray(t_traj)

    amp_prefactor = (
        mu * MSUN_SI * G_SI / C_SI ** 2
        / (dist * GPC_SI)
    )

    # WDM pixel-centre times
    t_bins = grid.t_bins       # (Nt,) numpy float64
    dT = grid.delta_T

    # --- Interpolate orbital phases to pixel centres ---
    Phi_phi_bins   = _interp_to_bins(t_np, np.asarray(Phi_phi),   t_bins)  # (Nt,)
    Phi_theta_bins = _interp_to_bins(t_np, np.asarray(Phi_theta), t_bins)
    Phi_r_bins     = _interp_to_bins(t_np, np.asarray(Phi_r),     t_bins)

    # --- Interpolate mode amplitudes to pixel centres ---
    # teuk_modes: (N_traj, N_modes) complex -> (Nt, N_modes) complex
    amp_bins = _complex_interp_to_bins(t_np, np.asarray(teuk_modes), t_bins)

    N_modes = len(m_arr)
    Nf, Nt = grid.Nf, grid.Nt

    # --- Instantaneous frequency via phase derivatives ---
    # f_mkn(t) ~ (m dPhi_phi/dt + k dPhi_theta/dt + n dPhi_r/dt) / (2 pi)
    dPhi_phi_dt   = _fdot_from_freq(Phi_phi_bins,   dT)   # (Nt,) rad/s
    dPhi_theta_dt = _fdot_from_freq(Phi_theta_bins, dT)
    dPhi_r_dt     = _fdot_from_freq(Phi_r_bins,     dT)

    # --- Accumulate WDM grid ---
    W_flat = jnp.zeros(Nf * Nt, dtype=jnp.complex64)

    # Flat scatter index: pixel (q, n) -> q*Nt + n
    j_arr = jnp.arange(Nt, dtype=jnp.int32)    # (Nt,)

    for im in range(N_modes):
        m_i = int(m_arr[im])
        k_i = int(k_arr[im])
        n_i = int(n_arr[im])

        # Instantaneous GW frequency at pixel centres [Hz]
        f_bins_m = (
            m_i * dPhi_phi_dt + k_i * dPhi_theta_dt + n_i * dPhi_r_dt
        ) / (2.0 * np.pi)                              # (Nt,)

        # Frequency derivative [Hz/s]
        fdot_bins = _fdot_from_freq(f_bins_m, dT)     # (Nt,)

        # Accumulated mode phase at pixel centres [rad]
        Phi_mkn = (
            m_i * Phi_phi_bins
            + k_i * Phi_theta_bins
            + n_i * Phi_r_bins
        )                                              # (Nt,)

        # Complex amplitude: Y+ * prefactor * A_lmkn (positive-m term)
        A_pos = (
            jnp.asarray(ylms_pos[im]) * amp_prefactor
            * jnp.asarray(amp_bins[:, im])
        )                                              # (Nt,) complex

        f_jax    = jnp.asarray(f_bins_m,  dtype=jnp.float32)
        fdot_jax = jnp.asarray(fdot_bins, dtype=jnp.float32)
        Phi_jax  = jnp.asarray(Phi_mkn,   dtype=jnp.float64)

        # Positive-m contribution: Y+ * A * exp(-i*Phi) * K
        q_idx, w_pos = direct_wdm_mode(
            A_pos, Phi_jax, f_jax, fdot_jax,
            grid, hw=hw, chirp_correction=chirp_correction,
        )                                              # (Nt, 2hw+1)

        flat_idx = q_idx * Nt + j_arr[:, None]        # (Nt, 2hw+1) int
        W_flat = W_flat.at[flat_idx.ravel()].add(w_pos.ravel())

        # Negative-m contribution (c.c. term): (-1)^l * Y- * A_bar * exp(+i*Phi) * K
        if m_i > 0:
            l_i = int(l_arr[im])
            sign_l = float((-1) ** l_i)
            A_neg = (
                sign_l * jnp.asarray(ylms_neg[im]) * amp_prefactor
                * jnp.conj(jnp.asarray(amp_bins[:, im]))
            )                                          # (Nt,) complex

            # Negative-m mode: exp(+i*Phi), same frequency track
            q_idx_neg, w_neg = direct_wdm_mode(
                A_neg, -Phi_jax, f_jax, fdot_jax,
                grid, hw=hw, chirp_correction=chirp_correction,
            )

            flat_idx_neg = q_idx_neg * Nt + j_arr[:, None]
            W_flat = W_flat.at[flat_idx_neg.ravel()].add(w_neg.ravel())

    # Source-frame sign convention (mirrors the -1 rotation in kerr.py)
    W_flat = W_flat * (-1.0)

    return W_flat.reshape(Nf, Nt)


def scatter_wdm(
    q_idx: jnp.ndarray,
    w: jnp.ndarray,
    grid: WDMGrid,
) -> jnp.ndarray:
    """Scatter per-mode WDM contributions into a flat (Nf*Nt,) buffer.

    Low-level building block for constructing the WDM grid from pre-computed
    :func:`direct_wdm_mode` outputs.  Use this inside ``jax.jit`` for a
    fully JAX-traced WDM generation loop.

    Parameters
    ----------
    q_idx : jnp.ndarray, shape (Nt, 2*hw+1), int32
        Frequency bin indices from :func:`direct_wdm_mode`.
    w : jnp.ndarray, shape (Nt, 2*hw+1), complex64
        Pixel coefficients from :func:`direct_wdm_mode`.
    grid : WDMGrid
        Must match the grid used in :func:`direct_wdm_mode`.

    Returns
    -------
    W_flat : jnp.ndarray, shape (Nf*Nt,), complex64
        Flat WDM buffer.  Reshape to ``(Nf, Nt)`` after accumulating all modes.
    """
    Nt = grid.Nt
    j_arr = jnp.arange(Nt, dtype=jnp.int32)[:, None]
    flat_idx = q_idx * Nt + j_arr                       # (Nt, 2hw+1)
    W_flat = jnp.zeros(grid.Nf * grid.Nt, dtype=jnp.complex64)
    return W_flat.at[flat_idx.ravel()].add(w.ravel())

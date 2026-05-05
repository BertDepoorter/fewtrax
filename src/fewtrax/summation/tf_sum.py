"""Direct TF-domain generation of EMRI harmonic waveforms.

Implements a single coherent scatter-add (:func:`direct_tf_sum`) that works
for any time-frequency basis by dispatching through the grid's ``kernel``
method.  Pass a :class:`~fewtrax.utils.tf_bases.wdm.WDMGrid` or a
:class:`~fewtrax.utils.tf_bases.sft.SFTGrid` (or any custom
:class:`~fewtrax.utils.tf_bases.base.TFGrid` subclass) to select the basis.

The key formula for any basis is::

    W[q, n] = sum_{lmkn'} [ Y+_lm * A_lmkn'(t_n) * exp(-i Phi_lmkn'(t_n))
                             * K(xi_{nq}, chi_n) + c.c. ]

where K is the pixel kernel of the chosen basis, ``t_n = n * dT`` is the
pixel-centre time, ``xi_{nq} = (q*dF - f(t_n)) / dF`` is the normalised
frequency offset, and ``chi_n = fdot * dT**2`` is the dimensionless chirp.

API
---
:func:`direct_tf_sum`
    Coherent TF scatter-add for any :class:`~fewtrax.utils.tf_bases.base.TFGrid`.
:func:`scatter_tf`
    Low-level: scatter pre-computed coefficients into a flat pixel buffer.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp

from fewtrax.utils.constants import MTSUN_SI, GPC_SI, G_SI, C_SI, MSUN_SI
from fewtrax.utils.tf_bases.base import TFGrid, direct_tf_mode
from fewtrax.utils.geodesic import get_fundamental_frequencies_platform


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _interp_to_bins(
    t_traj: np.ndarray,
    arr: np.ndarray,
    t_bins: np.ndarray,
) -> np.ndarray:
    """Cubic-spline interpolation of a trajectory array onto pixel centres."""
    from interpax import Interpolator1D

    valid = np.isfinite(t_traj)
    t_v   = t_traj[valid]
    arr_v = np.asarray(arr)[valid]

    if arr_v.ndim == 1:
        return np.asarray(Interpolator1D(t_v, arr_v, method="cubic2", extrap=True)(t_bins))

    n_cols = arr_v.shape[1]
    out = np.empty((len(t_bins), n_cols), dtype=arr_v.dtype)
    for ic in range(n_cols):
        out[:, ic] = np.asarray(
            Interpolator1D(t_v, arr_v[:, ic], method="cubic2", extrap=True)(t_bins)
        )
    return out


def _complex_interp_to_bins(
    t_traj: np.ndarray,
    arr: np.ndarray,
    t_bins: np.ndarray,
) -> np.ndarray:
    """Cubic-spline interpolation for complex arrays (real and imag separately)."""
    re = _interp_to_bins(t_traj, np.real(arr), t_bins)
    im = _interp_to_bins(t_traj, np.imag(arr), t_bins)
    return re + 1j * im


def _fdot_from_freq(f_bins: np.ndarray, delta_T: float) -> np.ndarray:
    """Central-difference frequency derivative at pixel centres [Hz/s]."""
    fdot = np.empty_like(f_bins)
    fdot[1:-1] = (f_bins[2:] - f_bins[:-2]) / (2.0 * delta_T)
    fdot[0]    = (f_bins[1]  - f_bins[0])   / delta_T
    fdot[-1]   = (f_bins[-1] - f_bins[-2])  / delta_T
    return fdot


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def direct_tf_sum(
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
    grid: TFGrid,
    dist: float = 1.0,
    hw: int = 2,
    p_traj: Optional[np.ndarray] = None,
    e_traj: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    r"""Coherent TF scatter-add over all EMRI harmonic modes for any basis.

    Directly computes the ``Nf × Nt`` TF strain without constructing a dense
    time series.  The pixel kernel is taken from ``grid.kernel(xi, chi)``, so
    passing a :class:`~fewtrax.utils.tf_bases.wdm.WDMGrid` gives the WDM
    waveform and passing a :class:`~fewtrax.utils.tf_bases.sft.SFTGrid` gives
    the SFT waveform.

    The return value is the *complex TF strain*::

        W[q, n] = W_plus[q, n] - 1j * W_cross[q, n]

    where ``W.real`` is the TF transform of h_plus and ``-W.imag`` is the TF
    transform of h_cross.

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
        :math:`{}_{-2}Y_{\ell m}(\theta,\phi)` for each mode.
    ylms_neg : (N_modes,) complex
        :math:`{}_{-2}Y_{\ell,-m}(\theta,\phi)` for each mode.
    a, M, mu, x0 : float
        Kerr spin, primary mass [M☉], secondary mass [M☉], inclination sign.
    grid : TFGrid
        Target grid.  Controls both the pixel layout (``delta_T``,
        ``delta_F``, ``Nf``, ``Nt``) and the kernel (``grid.kernel``).
        Use :class:`~fewtrax.utils.tf_bases.wdm.WDMGrid` for the Meyer/WDM
        basis and :class:`~fewtrax.utils.tf_bases.sft.SFTGrid` for the SFT
        basis.  Grid options (``chirp_correction``, ``use_exact_kernel``) are
        set at grid construction time, not here.
    dist : float
        Luminosity distance [Gpc].
    hw : int
        Frequency half-width for deposition.  For WDM ``hw=2`` is sufficient
        (Meyer window is zero outside one bin).  For SFT ``hw=3`` is
        recommended (sinc has significant tails at ±2 bins).
    p_traj, e_traj : (N_traj,) float, optional
        Sparse orbital elements.  When provided, instantaneous frequencies are
        computed analytically via
        :func:`~fewtrax.utils.geodesic.get_fundamental_frequencies_platform`
        rather than from finite differences of the phases (avoids
        double-differencing bias near plunge).

    Returns
    -------
    W : jnp.ndarray, shape (Nf, Nt), complex64
        Complex TF strain.  ``W.real`` is the TF representation of h_plus;
        ``-W.imag`` is the TF representation of h_cross.
    """
    t_np = np.asarray(t_traj)

    amp_prefactor = mu * MSUN_SI * G_SI / C_SI ** 2 / (dist * GPC_SI)

    t_bins = grid.t_bins
    dT     = grid.delta_T

    # --- Interpolate orbital phases to pixel centres ---
    Phi_phi_bins   = _interp_to_bins(t_np, np.asarray(Phi_phi),   t_bins)
    Phi_theta_bins = _interp_to_bins(t_np, np.asarray(Phi_theta), t_bins)
    Phi_r_bins     = _interp_to_bins(t_np, np.asarray(Phi_r),     t_bins)

    # --- Interpolate mode amplitudes to pixel centres ---
    amp_bins = _complex_interp_to_bins(t_np, np.asarray(teuk_modes), t_bins)

    N_modes = len(m_arr)
    Nf, Nt  = grid.Nf, grid.Nt

    # --- Instantaneous angular frequencies at pixel centres ---
    if p_traj is not None and e_traj is not None:
        p_bins_np = _interp_to_bins(t_np, np.asarray(p_traj), t_bins)
        e_bins_np = _interp_to_bins(t_np, np.asarray(e_traj), t_bins)
        M_s   = (M + mu) * MTSUN_SI
        a_abs = float(np.abs(a))
        ax    = float(a) * float(x0)
        x_in  = float(np.sign(ax)) if ax != 0.0 else 1.0
        Omega_phi_bins, Omega_theta_bins, Omega_r_bins = jax.vmap(
            lambda p_, e_: get_fundamental_frequencies_platform(a_abs, p_, e_, x_in)
        )(jnp.asarray(p_bins_np, dtype=jnp.float64),
          jnp.asarray(e_bins_np, dtype=jnp.float64))
        dPhi_phi_dt   = np.asarray(Omega_phi_bins)   / M_s
        dPhi_theta_dt = np.asarray(Omega_theta_bins) / M_s
        dPhi_r_dt     = np.asarray(Omega_r_bins)     / M_s
    else:
        dPhi_phi_dt   = _fdot_from_freq(Phi_phi_bins,   dT)
        dPhi_theta_dt = _fdot_from_freq(Phi_theta_bins, dT)
        dPhi_r_dt     = _fdot_from_freq(Phi_r_bins,     dT)

    # --- Build per-mode (N_modes, Nt) arrays ---
    m_jax = jnp.asarray(m_arr, dtype=jnp.int32)
    k_jax = jnp.asarray(k_arr, dtype=jnp.int32)
    n_jax = jnp.asarray(n_arr, dtype=jnp.int32)
    l_jax = jnp.asarray(l_arr, dtype=jnp.int32)

    dPhi_phi_j   = jnp.asarray(dPhi_phi_dt,   dtype=jnp.float64)
    dPhi_theta_j = jnp.asarray(dPhi_theta_dt, dtype=jnp.float64)
    dPhi_r_j     = jnp.asarray(dPhi_r_dt,     dtype=jnp.float64)
    Phi_phi_j    = jnp.asarray(Phi_phi_bins,   dtype=jnp.float64)
    Phi_theta_j  = jnp.asarray(Phi_theta_bins, dtype=jnp.float64)
    Phi_r_j      = jnp.asarray(Phi_r_bins,     dtype=jnp.float64)
    amp_bins_j   = jnp.asarray(amp_bins)

    Phi_all = (m_jax[:, None] * Phi_phi_j[None, :]
             + k_jax[:, None] * Phi_theta_j[None, :]
             + n_jax[:, None] * Phi_r_j[None, :])

    f_all = (m_jax[:, None] * dPhi_phi_j[None, :]
           + k_jax[:, None] * dPhi_theta_j[None, :]
           + n_jax[:, None] * dPhi_r_j[None, :]) / (2.0 * jnp.pi)

    fdot_all = jnp.concatenate([
        (f_all[:, 1:2] - f_all[:, 0:1]) / dT,
        (f_all[:, 2:]  - f_all[:, :-2]) / (2.0 * dT),
        (f_all[:, -1:] - f_all[:, -2:-1]) / dT,
    ], axis=1)

    # Positive-m amplitudes  (all modes)
    A_pos_all = (
        jnp.asarray(ylms_pos, dtype=jnp.complex128)[:, None]
        * amp_prefactor
        * amp_bins_j.T
    )

    # Negative-m amplitudes  (m > 0 only, via symmetry)
    m_pos_mask = (m_jax > 0).astype(jnp.float64)
    sign_l_arr = jnp.array([(-1.0) ** int(li) for li in l_arr], dtype=jnp.float64)
    A_neg_all  = (
        (sign_l_arr * m_pos_mask)[:, None]
        * jnp.asarray(ylms_neg, dtype=jnp.complex128)[:, None]
        * amp_prefactor
        * jnp.conj(amp_bins_j.T)
    )

    # --- vmap direct_tf_mode over modes ---
    def _deposit(A_n, Phi_n, f_n, fdot_n):
        return direct_tf_mode(
            A_n, Phi_n,
            f_n.astype(jnp.float32),
            fdot_n.astype(jnp.float32),
            grid, hw=hw,
        )

    q_pos, w_pos = jax.vmap(_deposit)(A_pos_all, Phi_all,  f_all, fdot_all)
    q_neg, w_neg = jax.vmap(_deposit)(A_neg_all, -Phi_all, f_all, fdot_all)

    # --- Scatter-add into flat (Nf*Nt) buffer ---
    j_idx    = jnp.arange(Nt, dtype=jnp.int32)
    flat_pos = (q_pos * Nt + j_idx[None, :, None]).ravel()
    flat_neg = (q_neg * Nt + j_idx[None, :, None]).ravel()

    W_flat = jnp.zeros(Nf * Nt, dtype=jnp.complex64)
    W_flat = W_flat.at[flat_pos].add(w_pos.ravel())
    W_flat = W_flat.at[flat_neg].add(w_neg.ravel())

    return (W_flat * (-1.0)).reshape(Nf, Nt)


# ---------------------------------------------------------------------------
# Low-level scatter helper
# ---------------------------------------------------------------------------

def scatter_tf(
    q_idx: jnp.ndarray,
    w: jnp.ndarray,
    grid: TFGrid,
) -> jnp.ndarray:
    """Scatter per-mode TF contributions into a flat ``(Nf*Nt,)`` buffer.

    Low-level building block for constructing the TF grid from pre-computed
    :func:`~fewtrax.utils.tf_bases.base.direct_tf_mode` outputs.  Works for
    any :class:`~fewtrax.utils.tf_bases.base.TFGrid`.

    Parameters
    ----------
    q_idx : jnp.ndarray, shape (Nt, 2*hw+1), int32
    w : jnp.ndarray, shape (Nt, 2*hw+1), complex64
    grid : TFGrid

    Returns
    -------
    W_flat : jnp.ndarray, shape (Nf*Nt,), complex64
        Reshape to ``(Nf, Nt)`` after accumulating all modes.
    """
    Nt    = grid.Nt
    j_arr = jnp.arange(Nt, dtype=jnp.int32)[:, None]
    W_flat = jnp.zeros(grid.Nf * Nt, dtype=jnp.complex64)
    return W_flat.at[(q_idx * Nt + j_arr).ravel()].add(w.ravel())

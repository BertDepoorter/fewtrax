"""Sparse WDM time-frequency track representation for EMRI harmonics.

Each harmonic mode (l, m, k, n) of an EMRI traces a slowly-chirping frequency
track in the WDM (Wilson-Daubechies-Meyer) time-frequency plane:

    f_{mkn}(t) = [m Ω_φ(t) + k Ω_θ(t) + n Ω_r(t)] / (2π M_s)

where M_s = (M + μ) MTSUN_SI is the total mass in seconds and Ω_i are the
Boyer-Lindquist fundamental frequencies in geometric units [rad/M].

Because EMRI signals evolve slowly, each mode occupies O(Nt) active pixels
in an Nf × Nt WDM grid — a fraction 1/Nf of all pixels.  This allows a
sparse representation of ~tens of kilobytes per mode.

Two representations
-------------------
:func:`analytical_tf_track`
    Purely from trajectory data, no WDM transform needed.  Maps the
    frequency track onto the WDM grid by interpolating fundamental
    frequencies to WDM time bins.  Fastest and smallest footprint.

:func:`sparse_wdm_track`
    Builds the single-mode time series, applies the pywavelet WDM
    transform, and stores only the pixels along the analytical track
    (±``hw`` bins in frequency).  Gives exact WDM coefficients.

Memory estimate (default grid Nf=64, Nt=4096)
----------------------------------------------
- i_freq  int16    : Nt × 2 bytes =   8 kB
- freq_hz float32  : Nt × 4 bytes =  16 kB
- coeff   complex64: Nt × 8 bytes =  32 kB
Total per mode ≈ 56 kB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import jax.numpy as jnp

from fewtrax.utils.constants import MTSUN_SI
from fewtrax.utils.geodesic import get_fundamental_frequencies

# ---------------------------------------------------------------------------
# Re-exports from tf_bases (single authoritative definitions)
# ---------------------------------------------------------------------------
from fewtrax.utils.tf_bases.wdm import (   # noqa: F401
    WDMGrid,
    default_grid,
    meyer_window,
    meyer_kernel,
    _nu,
)
from fewtrax.utils.tf_bases.base import direct_tf_mode  # noqa: F401


# ---------------------------------------------------------------------------
# Sparse TF track
# ---------------------------------------------------------------------------

@dataclass
class TFTrack:
    """Sparse WDM time-frequency track for a single EMRI harmonic mode.

    Attributes
    ----------
    mode : tuple of int
        Mode indices (l, m, k, n).
    grid : WDMGrid
        WDM grid on which the track lives.
    i_freq : np.ndarray, shape (Nt,), int16
        Frequency bin index at each WDM time step.
    freq_hz : np.ndarray, shape (Nt,), float32
        Instantaneous GW frequency [Hz] at each WDM time step.
    coeff : np.ndarray or None, shape (Nt,), complex64
        WDM coefficient along the track (from pywavelet transform).
        None for purely analytical tracks.
    """

    mode: tuple
    grid: WDMGrid
    i_freq: np.ndarray   # int16
    freq_hz: np.ndarray  # float32
    coeff: Optional[np.ndarray] = field(default=None)  # complex64

    @property
    def nbytes(self) -> int:
        """Memory footprint in bytes."""
        n = self.i_freq.nbytes + self.freq_hz.nbytes
        if self.coeff is not None:
            n += self.coeff.nbytes
        return n

    def power(self) -> Optional[np.ndarray]:
        """Power |coeff|² at each time step, or None if coeff is unavailable."""
        return np.abs(self.coeff) ** 2 if self.coeff is not None else None

    def __repr__(self) -> str:
        l, m, k, n = self.mode
        has_coeff = self.coeff is not None
        return (
            f"TFTrack(mode=({l},{m},{k},{n}), "
            f"f=[{self.freq_hz.min()*1e3:.3f},{self.freq_hz.max()*1e3:.3f}]mHz, "
            f"coeff={'yes' if has_coeff else 'no'}, "
            f"{self.nbytes/1024:.1f}kB)"
        )


# ---------------------------------------------------------------------------
# Frequency track computation
# ---------------------------------------------------------------------------

def compute_freq_track(
    m: int,
    k: int,
    n: int,
    t_traj: np.ndarray,
    p_traj: np.ndarray,
    e_traj: np.ndarray,
    a: float,
    M: float,
    mu: float,
    x0: float = 1.0,
) -> np.ndarray:
    """Instantaneous GW frequency of mode (m, k, n) along the trajectory.

    Parameters
    ----------
    m, k, n : int
        Mode harmonic numbers (azimuthal, polar, radial).
    t_traj : array, shape (N_traj,)
        Trajectory time stamps [s].  May contain NaNs at the end if the
        inspiral terminated early.
    p_traj, e_traj : array, shape (N_traj,)
        Semi-latus rectum and eccentricity at each trajectory point.
    a : float
        Kerr spin parameter.
    M, mu : float
        Primary and secondary masses [M_sun].
    x0 : float
        Inclination sign (+1 prograde, -1 retrograde).

    Returns
    -------
    f_mkn : array, shape (N_traj,)
        Instantaneous GW frequency [Hz].  NaN wherever t_traj is NaN.
    """
    M_total_s = (M + mu) * MTSUN_SI  # total mass in seconds

    valid = np.isfinite(t_traj) & np.isfinite(p_traj) & np.isfinite(e_traj)
    f_mkn = np.full_like(t_traj, np.nan, dtype=np.float64)

    a_abs = abs(a)
    x_in = float(np.sign(a * x0)) if a != 0.0 else 1.0

    for i in np.where(valid)[0]:
        Om_phi, Om_theta, Om_r = get_fundamental_frequencies(
            a_abs, float(p_traj[i]), float(e_traj[i]), x_in
        )
        f_mkn[i] = (m * Om_phi + k * Om_theta + n * Om_r) / (
            2.0 * np.pi * M_total_s
        )

    return f_mkn


def compute_freq_track_batch(
    m: int,
    k: int,
    n: int,
    t_traj: np.ndarray,
    p_traj: np.ndarray,
    e_traj: np.ndarray,
    a: float,
    M: float,
    mu: float,
    x0: float = 1.0,
) -> np.ndarray:
    """Vectorised version of :func:`compute_freq_track` using JAX vmap.

    Much faster for long trajectories.  Falls back to the loop-based
    version if JAX is unavailable or if the trajectory contains NaNs at
    unpredictable locations.
    """
    import jax
    import jax.numpy as jnp
    from fewtrax.utils.geodesic import get_fundamental_frequencies as _gff

    M_total_s = (M + mu) * MTSUN_SI
    a_abs = abs(a)
    x_in = float(np.sign(a * x0)) if a != 0.0 else 1.0

    valid = np.isfinite(t_traj) & np.isfinite(p_traj) & np.isfinite(e_traj)

    # Use jnp.where so the shape is always (N_traj,) regardless of how many valid
    # points there are — boolean indexing produces data-dependent shapes that
    # cannot be traced by jax.jit or jax.vmap.
    valid_j = jnp.asarray(valid)
    p_ref   = float(p_traj[valid][0]) if valid.any() else 10.0
    p_safe  = jnp.where(valid_j, jnp.asarray(p_traj, jnp.float64), p_ref)
    e_safe  = jnp.where(valid_j, jnp.asarray(e_traj, jnp.float64), 0.0)

    def _freq_one(p, e):
        Om_phi, Om_theta, Om_r = _gff(a_abs, p, e, x_in)
        return (m * Om_phi + k * Om_theta + n * Om_r) / (2.0 * jnp.pi * M_total_s)

    f_all = jax.vmap(_freq_one)(p_safe, e_safe)
    return np.where(valid, np.array(f_all), np.nan)


# ---------------------------------------------------------------------------
# Analytical TF track (no WDM transform)
# ---------------------------------------------------------------------------

def analytical_tf_track(
    mode: tuple,
    t_traj: np.ndarray,
    p_traj: np.ndarray,
    e_traj: np.ndarray,
    a: float,
    M: float,
    mu: float,
    grid: WDMGrid,
    x0: float = 1.0,
    use_jax: bool = True,
) -> TFTrack:
    """Analytical WDM TF track (no wavelet transform needed).

    Computes the instantaneous GW frequency along the trajectory and
    interpolates it onto the WDM time grid.  Returns a :class:`TFTrack`
    with ``coeff=None``.

    Parameters
    ----------
    mode : (l, m, k, n)
    t_traj, p_traj, e_traj : arrays
        Trajectory arrays (may contain NaNs after plunge).
    a, M, mu : float
        Kerr spin and mass parameters.
    grid : WDMGrid
        Target WDM grid.
    x0 : float
        Inclination sign.
    use_jax : bool
        Use JAX vmap for faster frequency computation (default True).

    Returns
    -------
    TFTrack
    """
    l, m, k, n = mode

    if use_jax:
        try:
            f_mkn = compute_freq_track_batch(m, k, n, t_traj, p_traj, e_traj,
                                             a, M, mu, x0)
        except Exception:
            f_mkn = compute_freq_track(m, k, n, t_traj, p_traj, e_traj,
                                       a, M, mu, x0)
    else:
        f_mkn = compute_freq_track(m, k, n, t_traj, p_traj, e_traj, a, M, mu, x0)

    # Mask NaNs for interpolation
    valid = np.isfinite(f_mkn)
    t_v = t_traj[valid]
    f_v = f_mkn[valid]

    # Interpolate to WDM time bins (extrapolate edge bins with clipping)
    t_bins = grid.t_bins
    f_at_bins = np.interp(t_bins, t_v, f_v,
                          left=f_v[0], right=f_v[-1]).astype(np.float32)

    # Map to nearest frequency bin, clipped to valid range
    i_freq = grid.freq_to_bin(f_at_bins).clip(0, grid.Nf - 1).astype(np.int16)

    return TFTrack(mode=mode, grid=grid, i_freq=i_freq, freq_hz=f_at_bins)


# ---------------------------------------------------------------------------
# WDM transform-based sparse track
# ---------------------------------------------------------------------------

def sparse_wdm_track(
    mode: tuple,
    t_traj: np.ndarray,
    p_traj: np.ndarray,
    e_traj: np.ndarray,
    Phi_phi: np.ndarray,
    Phi_theta: np.ndarray,
    Phi_r: np.ndarray,
    teuk_amp: np.ndarray,
    a: float,
    M: float,
    mu: float,
    grid: WDMGrid,
    x0: float = 1.0,
    hw: int = 1,
    nx: float = 4.0,
    mult: int = 32,
) -> TFTrack:
    """WDM TF track from the pywavelet forward transform.

    Generates the single-mode time series, applies the WDM transform,
    and extracts coefficients along the analytical frequency track (±hw
    frequency bins).

    Parameters
    ----------
    mode : (l, m, k, n)
    t_traj, p_traj, e_traj : arrays, shape (N_traj,)
        Trajectory arrays.
    Phi_phi, Phi_theta, Phi_r : arrays, shape (N_traj,)
        Accumulated orbital phases [rad].
    teuk_amp : array, shape (N_traj,), complex
        Teukolsky mode amplitude A_{lmkn}(t) at each trajectory point.
    a, M, mu : float
    grid : WDMGrid
    x0 : float
    hw : int
        Half-width in frequency bins: store coefficients for
        i_freq ± hw.  hw=0 stores only the peak bin; hw=1 stores ±1.
    nx : float
        pywavelet window parameter (default 4.0).
    mult : int
        pywavelet mult parameter (default 32).

    Returns
    -------
    TFTrack
        ``coeff`` holds the WDM coefficient at the peak frequency bin
        at each time step.  Nearby bins (hw>0) are averaged into coeff.
    """
    from interpax import Interpolator1D
    import jax.numpy as jnp
    from pywavelet.transforms.numpy.forward.main import from_time_to_wavelet
    from pywavelet.types import TimeSeries

    l, m, k, n = mode
    N = grid.Nf * grid.Nt
    dt = grid.T / N  # underlying sample interval

    # Dense time grid for the single-mode waveform
    valid = np.isfinite(t_traj)
    t_v = t_traj[valid]
    t_dense = np.linspace(float(t_v[0]), float(t_v[-1]), N)

    def _interp(arr):
        spl = Interpolator1D(t_v, np.asarray(arr)[valid],
                             method="cubic2", extrap=True)
        return np.array(spl(t_dense))

    Phi_phi_d = _interp(Phi_phi)
    Phi_theta_d = _interp(Phi_theta)
    Phi_r_d = _interp(Phi_r)
    amp_d = (
        _interp(np.real(teuk_amp)) + 1j * _interp(np.imag(teuk_amp))
    )

    # Single-mode strain: h = A(t) * exp(-i Φ_{mkn}(t))
    phase = m * Phi_phi_d + k * Phi_theta_d + n * Phi_r_d
    h_mode = amp_d * np.exp(-1j * phase)

    # WDM transform of real part (pywavelet operates on real signals)
    ts = TimeSeries(data=np.real(h_mode), time=t_dense)
    wdm = from_time_to_wavelet(ts, Nf=grid.Nf, Nt=grid.Nt, nx=nx, mult=mult)
    wave = wdm.data  # shape (Nf, Nt)

    # Analytical track for bin indices
    track = analytical_tf_track(mode, t_traj, p_traj, e_traj, a, M, mu,
                                 grid, x0=x0, use_jax=True)
    i_freq = track.i_freq  # shape (Nt,), int16

    # Extract WDM coefficients along the track (±hw bins)
    j_arr = np.arange(grid.Nt)
    coeff = np.zeros(grid.Nt, dtype=np.complex64)
    for dfi in range(-hw, hw + 1):
        i_arr = (i_freq.astype(int) + dfi).clip(0, grid.Nf - 1)
        coeff += wave[i_arr, j_arr]
    coeff /= (2 * hw + 1)

    return TFTrack(
        mode=mode,
        grid=grid,
        i_freq=i_freq,
        freq_hz=track.freq_hz,
        coeff=coeff.astype(np.complex64),
    )


# ---------------------------------------------------------------------------
# Multi-mode track set
# ---------------------------------------------------------------------------

@dataclass
class TFTrackSet:
    """Collection of sparse TF tracks for multiple EMRI harmonics.

    Parameters
    ----------
    tracks : list of TFTrack
    grid : WDMGrid
    """

    tracks: list
    grid: WDMGrid

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tracks)

    @property
    def n_modes(self) -> int:
        return len(self.tracks)

    def get_mode(self, l: int, m: int, k: int, n: int) -> Optional[TFTrack]:
        for t in self.tracks:
            if t.mode == (l, m, k, n):
                return t
        return None

    def freq_range(self) -> tuple:
        """Overall [f_min, f_max] across all tracks [Hz]."""
        f_lo = min(t.freq_hz.min() for t in self.tracks)
        f_hi = max(t.freq_hz.max() for t in self.tracks)
        return float(f_lo), float(f_hi)

    def __repr__(self) -> str:
        f_lo, f_hi = self.freq_range()
        return (
            f"TFTrackSet({self.n_modes} modes, "
            f"f=[{f_lo*1e3:.3f},{f_hi*1e3:.3f}]mHz, "
            f"{self.nbytes/1024:.1f}kB total)"
        )


def build_tf_tracks(
    mode_list: Sequence[tuple],
    t_traj: np.ndarray,
    p_traj: np.ndarray,
    e_traj: np.ndarray,
    a: float,
    M: float,
    mu: float,
    grid: WDMGrid,
    x0: float = 1.0,
) -> TFTrackSet:
    """Build analytical TF tracks for a list of modes.

    Parameters
    ----------
    mode_list : list of (l, m, k, n)
    t_traj, p_traj, e_traj : arrays
        Trajectory.
    a, M, mu : float
    grid : WDMGrid
    x0 : float

    Returns
    -------
    TFTrackSet
    """
    tracks = [
        analytical_tf_track(mode, t_traj, p_traj, e_traj, a, M, mu,
                             grid, x0=x0)
        for mode in mode_list
    ]
    return TFTrackSet(tracks=tracks, grid=grid)



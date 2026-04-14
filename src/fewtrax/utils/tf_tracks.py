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
# WDM grid descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WDMGrid:
    """Parameters of a WDM time-frequency grid.

    Parameters
    ----------
    Nf : int
        Number of frequency bins.
    Nt : int
        Number of time bins.
    T : float
        Total duration [s].

    Notes
    -----
    Total samples in the underlying time series: N = Nf * Nt.
    Time bin width:       delta_T = T / Nt         [s]
    Frequency bin width:  delta_F = 1 / (2 delta_T) [Hz]
    Nyquist frequency:    f_nyq   = Nf * delta_F    [Hz]
    """

    Nf: int
    Nt: int
    T: float  # observation duration [s]

    @property
    def delta_T(self) -> float:
        """WDM time bin width [s]."""
        return self.T / self.Nt

    @property
    def delta_F(self) -> float:
        """WDM frequency bin width [Hz]."""
        return 1.0 / (2.0 * self.delta_T)

    @property
    def f_nyq(self) -> float:
        """Nyquist frequency of the underlying time series [Hz]."""
        return self.Nf * self.delta_F

    @property
    def t_bins(self) -> np.ndarray:
        """WDM time bin centres [s]."""
        return np.arange(self.Nt, dtype=np.float64) * self.delta_T

    @property
    def f_bins(self) -> np.ndarray:
        """WDM frequency bin centres [Hz]."""
        return np.arange(self.Nf, dtype=np.float64) * self.delta_F

    def freq_to_bin(self, f: np.ndarray) -> np.ndarray:
        """Convert frequencies [Hz] to nearest integer bin indices."""
        return np.round(f / self.delta_F).astype(int)

    def __repr__(self) -> str:
        return (
            f"WDMGrid(Nf={self.Nf}, Nt={self.Nt}, "
            f"T={self.T:.3e}s, "
            f"dT={self.delta_T:.1f}s, "
            f"dF={self.delta_F*1e6:.2f}μHz, "
            f"f_nyq={self.f_nyq*1e3:.2f}mHz)"
        )


def default_grid(T: float, f_max: float = 5e-3, Nt: int = 4096) -> WDMGrid:
    """Construct a WDM grid covering [0, f_max] Hz over duration T.

    Parameters
    ----------
    T : float
        Observation duration [s].
    f_max : float
        Desired maximum frequency coverage [Hz].  The Nyquist frequency
        will be at least f_max; Nf is rounded up to the next power of 2.
    Nt : int
        Number of WDM time bins (power of 2 recommended).

    Returns
    -------
    WDMGrid
    """
    delta_T = T / Nt
    delta_F = 1.0 / (2.0 * delta_T)
    Nf = int(2 ** np.ceil(np.log2(f_max / delta_F)))
    return WDMGrid(Nf=Nf, Nt=Nt, T=T)


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
    p_v = jnp.array(p_traj[valid], dtype=jnp.float64)
    e_v = jnp.array(e_traj[valid], dtype=jnp.float64)

    def _freq_one(p, e):
        Om_phi, Om_theta, Om_r = _gff(a_abs, p, e, x_in)
        return (m * Om_phi + k * Om_theta + n * Om_r) / (2.0 * jnp.pi * M_total_s)

    f_valid = np.array(jax.vmap(_freq_one)(p_v, e_v))

    f_mkn = np.full(len(t_traj), np.nan)
    f_mkn[valid] = f_valid
    return f_mkn


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


# ---------------------------------------------------------------------------
# Meyer WDM kernel  (JAX, JIT-compilable)
# ---------------------------------------------------------------------------

def _nu(x: jnp.ndarray) -> jnp.ndarray:
    """Smooth transition ν: [0,1]→[0,1], ν(0)=0, ν(1)=1 (C∞, all derivatives zero at endpoints).

    Uses the polynomial ν(x) = x⁴(35 − 84x + 70x² − 20x³) which is the
    unique degree-7 solution to ν(0)=0, ν(1)=1, ν'(0)=ν'(1)=ν''(0)=ν''(1)=0.
    This ensures the Meyer window has a smooth, flat transition between the
    passband and the stop band with no Gibbs-like artefacts.
    """
    return x ** 4 * (35.0 - x * (84.0 - x * (70.0 - 20.0 * x)))


def meyer_window(xi: jnp.ndarray) -> jnp.ndarray:
    r"""Real Meyer low-pass window amplitude in normalised frequency ξ = Δ/ΔF.

    Piecewise definition:

    .. math::

        K(\xi) = \begin{cases}
            1 & |\xi| \leq \tfrac{1}{2} \\
            \cos\!\left(\tfrac{\pi}{2}\,\nu(2|\xi|-1)\right) & \tfrac{1}{2} < |\xi| \leq 1 \\
            0 & |\xi| > 1
        \end{cases}

    where :math:`\nu` is the smooth transition function :func:`_nu`.

    The window has compact frequency support (exactly zero beyond one bin width),
    is smooth (C∞), and satisfies the Parseval condition for the WDM orthonormal
    basis.

    Parameters
    ----------
    xi : jnp.ndarray
        Normalised frequency offset(s) Δ/ΔF.  Can be any shape.

    Returns
    -------
    K : jnp.ndarray, same shape as *xi*, float32
        Window amplitude in [0, 1].
    """
    xi_abs = jnp.abs(xi)
    # Transition argument: t ∈ [0,1] for ξ ∈ [0.5, 1]
    t = jnp.clip(2.0 * xi_abs - 1.0, 0.0, 1.0)
    K = jnp.where(
        xi_abs <= 0.5,
        jnp.ones_like(xi_abs),
        jnp.where(
            xi_abs <= 1.0,
            jnp.cos(0.5 * jnp.pi * _nu(t)),
            jnp.zeros_like(xi_abs),
        ),
    )
    return K.astype(jnp.float32)


def meyer_kernel(
    delta_over_dF: jnp.ndarray,
    chi: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    r"""Complex WDM pixel kernel for a (possibly chirped) narrowband signal.

    In the **locally-monochromatic limit** (``chi=None`` or ``chi=0``), a
    signal at instantaneous frequency :math:`f_n` contributes to WDM pixel
    :math:`(n,q)` with amplitude

    .. math::

        w_{nq} \approx A(t_n)\,e^{-i\Phi(t_n)}\cdot K\!\left(\frac{q\Delta F - f_n}{\Delta F}\right)

    where :math:`K` is :func:`meyer_window`.

    With a **chirp correction** (:math:`\chi = \dot f\,\Delta T^2 \neq 0`),
    the Taylor expansion of the phase beyond linear order introduces a
    frequency-dependent complex phase shift.  To leading order in :math:`\chi`
    this is

    .. math::

        K(\xi,\chi) \approx K_0(\xi)\cdot e^{-i\pi\chi\xi^2}

    which is exact at :math:`\chi=0` and captures the dominant
    chirp-induced pixel smearing for :math:`|\chi| \ll 1`.  The correction
    is negligible (< 1 % amplitude error) when :math:`\pi|\chi| < 0.1`,
    i.e. when :math:`\pi|\dot f|\Delta T^2 < 0.1`.

    Parameters
    ----------
    delta_over_dF : jnp.ndarray
        Normalised frequency offset :math:`(q\Delta F - f_n)/\Delta F`.
        Can be any shape.
    chi : jnp.ndarray or None
        Dimensionless chirp parameter :math:`\dot f\,\Delta T^2`.  Must be
        broadcastable with *delta_over_dF*.  If ``None``, the
        locally-monochromatic kernel (real) is returned.

    Returns
    -------
    K : jnp.ndarray, complex64, same shape as *delta_over_dF*
        Complex pixel kernel.  The real part corresponds to :math:`h_+`
        and the imaginary part to :math:`-h_\times`.
    """
    K0 = meyer_window(delta_over_dF)   # real, [0, 1]
    if chi is None:
        return K0.astype(jnp.complex64)
    # Leading-order Fresnel (chirp) correction
    phase_corr = jnp.exp(-1j * jnp.pi * jnp.asarray(chi) * delta_over_dF ** 2)
    return (K0 * phase_corr).astype(jnp.complex64)


# ---------------------------------------------------------------------------
# Direct per-mode WDM deposition  (JAX, JIT-compilable)
# ---------------------------------------------------------------------------

def direct_wdm_mode(
    A_n: jnp.ndarray,
    Phi_n: jnp.ndarray,
    f_n: jnp.ndarray,
    fdot_n: jnp.ndarray,
    grid: WDMGrid,
    hw: int = 2,
    chirp_correction: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Deposit one harmonic mode onto the WDM grid without a time series.

    For each WDM time pixel :math:`n` and the :math:`2h_w+1` adjacent
    frequency bins :math:`q_0 \pm h_w`, computes the complex pixel coefficient

    .. math::

        w_{nq} = A(t_n)\,e^{-i\Phi(t_n)}\cdot
                 K\!\!\left(\frac{q\Delta F - f_n}{\Delta F},\;
                            \dot f_n\,\Delta T^2\right)

    and returns the bin indices and values for a subsequent scatter-add into
    the :math:`N_f\times N_t` WDM grid.

    The function is pure JAX and can be JIT-compiled or differentiated.
    All inputs must be JAX arrays or scalars.

    Parameters
    ----------
    A_n : jnp.ndarray, shape (Nt,), complex
        Complex Teukolsky amplitude at each WDM time-bin centre, including
        the spin-weighted spherical harmonic :math:`Y_{\ell m}` prefactor.
    Phi_n : jnp.ndarray, shape (Nt,)
        Accumulated mode phase :math:`m\Phi_\phi + k\Phi_\theta + n\Phi_r`
        at each WDM time-bin centre [rad].
    f_n : jnp.ndarray, shape (Nt,)
        Instantaneous GW frequency :math:`f_{mkn}(t_n)` [Hz].
    fdot_n : jnp.ndarray, shape (Nt,)
        Frequency derivative :math:`\dot f_{mkn}(t_n)` [Hz s⁻¹].
    grid : WDMGrid
        WDM grid descriptor.
    hw : int
        Half-width in frequency bins.  Contributions are stored for bins
        :math:`q_0 - h_w, \ldots, q_0 + h_w`.  ``hw=2`` is sufficient
        for the Meyer window (zero outside 1 bin width) and provides a
        buffer for the chirp correction.
    chirp_correction : bool
        If True, apply the leading-order Fresnel chirp correction
        :math:`e^{-i\pi\chi\xi^2}`.  Disable to recover the pure
        locally-monochromatic kernel.

    Returns
    -------
    q_idx : jnp.ndarray, shape (Nt, 2*hw+1), int32
        Frequency bin indices (clipped to ``[0, Nf-1]``).
    w : jnp.ndarray, shape (Nt, 2*hw+1), complex64
        Complex WDM coefficients at each (time pixel, frequency neighbour).
    """
    Nt = grid.Nt
    dF = grid.delta_F
    dT = grid.delta_T

    # Central frequency bin for each time pixel
    q0 = jnp.round(f_n / dF).astype(jnp.int32)             # (Nt,)
    dq = jnp.arange(-hw, hw + 1, dtype=jnp.int32)           # (2hw+1,)

    # Neighbouring bin indices, clipped to valid range
    q = q0[:, None] + dq[None, :]                            # (Nt, 2hw+1)
    q_clipped = jnp.clip(q, 0, grid.Nf - 1)

    # Normalised frequency offset  ξ = (q·ΔF − f_n) / ΔF
    xi = (q_clipped.astype(jnp.float32) * dF - f_n[:, None]) / dF  # (Nt, 2hw+1)

    # Chirp parameter  χ = fdot · ΔT²
    chi = (fdot_n * dT ** 2)[:, None] * jnp.ones((1, 2 * hw + 1))  # (Nt, 2hw+1)

    # Kernel amplitude
    K = meyer_kernel(xi, chi if chirp_correction else None)   # (Nt, 2hw+1), complex64

    # Phasor  A(t_n) · exp(−i Φ(t_n))
    phasor = (A_n * jnp.exp(-1j * Phi_n))[:, None]           # (Nt, 1), complex

    w = (phasor * K).astype(jnp.complex64)                   # (Nt, 2hw+1)
    return q_clipped, w

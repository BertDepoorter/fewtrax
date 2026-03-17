"""Fourier transform utilities for gravitational-wave time series.

Functions
---------
to_frequency_domain
    Forward FFT (real or complex input → one-sided spectrum).
to_time_domain
    Inverse FFT (one-sided spectrum → time-domain signal).
"""

from __future__ import annotations

from typing import Optional
import jax.numpy as jnp


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two ≥ n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def to_frequency_domain(
    h: jnp.ndarray,
    dt: float,
    window: Optional[str] = None,
    zero_pad: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Forward FFT of a time-domain gravitational-wave signal.

    Computes

    .. math::

        \tilde{h}(f) = \int_{-\infty}^{+\infty} h(t)\, e^{-2\pi i f t}\, dt

    via the discrete FFT, normalised to physical units (strain / Hz).

    Works for both **real** (e.g. h+, h×) and **complex** (h+ − i h×)
    input arrays.  For real input the one-sided ``rfft`` is used; for
    complex input the two-sided ``fft`` is used and only positive
    frequencies are returned.

    Parameters
    ----------
    h : jnp.ndarray, shape (N,)
        Real or complex time-domain strain.
    dt : float
        Sampling interval [s].
    window : str, optional
        Windowing function applied before the FFT to reduce spectral
        leakage.  Supported values:

        ``"hann"`` (or ``"hanning"``)
            Hann (raised-cosine) window.
        ``"hamming"``
            Hamming window.
        ``"blackman"``
            Blackman window.
        ``"flattop"``
            Flat-top window (best amplitude accuracy).
        ``"tukey"``
            Tukey (cosine-tapered) window with taper fraction 0.1.

        If ``None`` (default) no window is applied.
    zero_pad : bool
        If ``True``, zero-pad the (possibly windowed) signal to the next
        power of two before the FFT.  This increases frequency resolution
        interpolation and avoids wrap-around artefacts, at the cost of a
        longer transform.  Default: ``False``.

    Returns
    -------
    freqs : jnp.ndarray, shape (M,)
        Positive frequencies [Hz], where M = N_fft // 2 + 1.
    h_tilde : jnp.ndarray, shape (M,), complex
        One-sided frequency-domain strain [strain / Hz].

    Notes
    -----
    The normalisation convention matches Parseval's theorem in the form

    .. math::

        \int |h(t)|^2\, dt \approx \int_{0}^{f_{\rm Nyq}} 2 |\tilde{h}(f)|^2\, df

    (factor of 2 for the one-sided spectrum).  When a window is applied
    the amplitude spectrum is divided by the window's coherent power gain
    (the mean of the window coefficients) so that amplitudes remain
    calibrated.
    """
    N = int(h.shape[0])

    # --- Build window ---
    win: Optional[jnp.ndarray] = None
    if window is not None:
        w_key = window.lower()
        n_arr = jnp.arange(N, dtype=jnp.float64)
        if w_key in ("hann", "hanning"):
            win = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * n_arr / (N - 1)))
        elif w_key == "hamming":
            win = 0.54 - 0.46 * jnp.cos(2.0 * jnp.pi * n_arr / (N - 1))
        elif w_key == "blackman":
            win = (0.42
                   - 0.50 * jnp.cos(2.0 * jnp.pi * n_arr / (N - 1))
                   + 0.08 * jnp.cos(4.0 * jnp.pi * n_arr / (N - 1)))
        elif w_key == "flattop":
            a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.27726316, 0.08357895, 0.00694737
            win = (a0
                   - a1 * jnp.cos(2.0 * jnp.pi * n_arr / (N - 1))
                   + a2 * jnp.cos(4.0 * jnp.pi * n_arr / (N - 1))
                   - a3 * jnp.cos(6.0 * jnp.pi * n_arr / (N - 1))
                   + a4 * jnp.cos(8.0 * jnp.pi * n_arr / (N - 1)))
        elif w_key == "tukey":
            alpha = 0.1
            win = _tukey_window(N, alpha)
        else:
            raise ValueError(
                f"Unknown window '{window}'.  Choose from: "
                "hann, hamming, blackman, flattop, tukey."
            )
        # Amplitude correction: divide by the coherent gain (mean of window)
        coherent_gain = jnp.mean(win)
        h = h * win / coherent_gain

    # --- Zero-pad to next power of two ---
    N_fft = _next_power_of_two(N) if zero_pad else N
    if N_fft > N:
        pad = N_fft - N
        h = jnp.pad(h, (0, pad))

    # --- FFT ---
    if jnp.issubdtype(h.dtype, jnp.complexfloating):
        h_full = jnp.fft.fft(h) * dt
        freqs  = jnp.fft.fftfreq(N_fft, d=dt)
        pos    = freqs >= 0
        # jnp boolean index → use integer slicing for JIT-safety
        n_pos = N_fft // 2 + 1
        return freqs[:n_pos], h_full[:n_pos]
    else:
        h_tilde = jnp.fft.rfft(h) * dt
        freqs   = jnp.fft.rfftfreq(N_fft, d=dt)
        return freqs, h_tilde


def to_time_domain(
    h_tilde: jnp.ndarray,
    dt: float,
    N: Optional[int] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Inverse FFT of a one-sided frequency-domain strain.

    Recovers the time-domain signal from a one-sided spectrum produced by
    :func:`to_frequency_domain`.

    Parameters
    ----------
    h_tilde : jnp.ndarray, shape (M,), complex
        One-sided frequency-domain strain (output of :func:`to_frequency_domain`).
    dt : float
        Original sampling interval [s].
    N : int, optional
        Number of output time samples.  If not provided the real-valued
        IFFT length ``2 * (M - 1)`` is used.

    Returns
    -------
    t : jnp.ndarray, shape (N_out,)
        Time stamps [s] starting at 0.
    h : jnp.ndarray, shape (N_out,)
        Reconstructed time-domain signal.
    """
    M = int(h_tilde.shape[0])
    N_fft = 2 * (M - 1) if N is None else N
    h = jnp.fft.irfft(h_tilde / dt, n=N_fft)
    t = jnp.arange(N_fft, dtype=jnp.float64) * dt
    return t, h


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tukey_window(N: int, alpha: float = 0.1) -> jnp.ndarray:
    """Tukey (cosine-tapered) window.

    Parameters
    ----------
    N : int
        Window length.
    alpha : float
        Taper fraction (0 = rectangular, 1 = Hann).
    """
    n_arr = jnp.arange(N, dtype=jnp.float64)
    width = int(alpha * N / 2)
    # Left taper
    left  = 0.5 * (1.0 - jnp.cos(jnp.pi * n_arr / width))
    # Right taper (mirror)
    right = 0.5 * (1.0 - jnp.cos(jnp.pi * (N - 1 - n_arr) / width))
    flat  = jnp.ones(N, dtype=jnp.float64)
    win   = jnp.where(n_arr < width, left,
            jnp.where(n_arr >= N - width, right, flat))
    return win

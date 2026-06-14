"""Short-Time Fourier Transform (SFT) grid and Fresnel kernel.

The SFT divides the full time series into :math:`N_t` non-overlapping segments
of duration :math:`T_{\\rm coh} = \\Delta T` and computes the DFT of each.
For a linearly-chirping tone with instantaneous frequency :math:`f_n` and
derivative :math:`\\dot f_n`, the exact pixel coefficient is

.. math::

    X[n, q] = A(t_n)\\,e^{-i\\Phi(t_n)} \\cdot
              T_{\\rm coh}\\cdot K^{\\rm SFT}(\\xi_{nq},\\, \\chi_n)

where

.. math::

    K^{\\rm SFT}(\\xi, \\chi)
      = \\int_0^1 e^{-2\\pi i\\,\\xi\\,u - \\pi i\\,\\chi\\,u^2}\\,du, \\qquad
    \\xi = \\frac{q\\Delta F - f_n}{\\Delta F}, \\qquad
    \\chi = \\dot f_n\\, T_{\\rm coh}^2 .

Two implementations are provided:

* :func:`sft_kernel` — *approximate* (fast): :math:`\\text{sinc}(\\xi)\\cdot
  e^{-i\\pi\\chi(\\xi^2-1/6)}`.  Accurate to < 1 % for :math:`|\\chi| < 0.3`.
* :func:`sft_kernel_exact` — *exact*: direct Gauss-Legendre quadrature of the
  integral above.  Accurate for :math:`|\\chi| \\lesssim 10` (32 GL points);
  increase ``_GL_N`` for faster-chirping signals near plunge.

The comparison between the two (and against :func:`jax.scipy.special.fresnel`)
can be found in ``comparison/tf_waveform_explorer.ipynb`` § 7.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

from fewtrax.utils.tf_bases.base import TFGrid


# ---------------------------------------------------------------------------
# GL nodes for the exact quadrature (computed once at import time)
# ---------------------------------------------------------------------------

#: Number of Gauss-Legendre quadrature points.
#: GL-64 integrates polynomials exactly up to degree 127.  For oscillatory
#: integrands the effective accuracy degrades when the number of phase cycles
#: over [0, 1] exceeds N/pi ≈ 20 (i.e. |chi| > 20*2 = 40).  For near-plunge
#: segments with longer T_coh, increase this constant.
_GL_N: int = 64

_gl_nodes_m1p1, _gl_weights_m1p1 = np.polynomial.legendre.leggauss(_GL_N)
# Transform from [-1, 1] to [0, 1]
_GL_U: np.ndarray = (_gl_nodes_m1p1 + 1.0) / 2.0   # nodes on [0, 1]
_GL_W: np.ndarray = _gl_weights_m1p1 / 2.0          # weights scaled to [0, 1]

# JAX constants (loaded once; treated as static by jit)
_GL_U_JAX: jnp.ndarray = jnp.asarray(_GL_U, dtype=jnp.float32)
_GL_W_JAX: jnp.ndarray = jnp.asarray(_GL_W, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def sft_kernel(xi: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
    r"""Approximate SFT pixel kernel for a linearly-chirping tone.

    The exact kernel is the integral

    .. math::

        K^{\rm SFT}(\xi, \chi)
          = \int_0^1 e^{-2\pi i\,\xi\, u - \pi i\,\chi\, u^2}\, du .

    For :math:`|\chi| \ll 1`, expanding the Gaussian and evaluating the
    zeroth-order integral gives

    .. math::

        K^{\rm SFT}(\xi, \chi) \approx
        \operatorname{sinc}(\xi)\;
        e^{-i\pi\xi}\;
        e^{-i\pi\chi(\xi^2 - \tfrac{1}{6})}

    where :math:`\operatorname{sinc}(x) = \sin(\pi x)/(\pi x)`.

    The factor :math:`e^{-i\pi\xi}` is the *static phase* arising from the
    one-sided integration window :math:`[0, T_{\rm coh}]` — it is zero only
    for on-bin signals (:math:`\xi = 0`).  It is essential for coherent
    matched filtering; power-only statistics are unaffected by it.

    The :math:`-\tfrac{\chi}{6}` term corrects the constant phase offset of
    the stationary-phase integral to leading order in :math:`\chi`.

    Amplitude error < 1 % for :math:`|\chi| < 0.3`, < 5 % for
    :math:`|\chi| < 0.7`.  Use :func:`sft_kernel_exact` for near-plunge.

    Parameters
    ----------
    xi : jnp.ndarray
        Normalised frequency offset :math:`(q\Delta F - f)/\Delta F`.
    chi : jnp.ndarray
        Dimensionless chirp :math:`\dot f\,T_{\rm coh}^2`.

    Returns
    -------
    jnp.ndarray, complex64, same shape as *xi*.
    """
    xi_f  = jnp.asarray(xi,  dtype=jnp.float32)
    chi_f = jnp.asarray(chi, dtype=jnp.float32)
    K0           = jnp.sinc(xi_f)                                     # real sinc
    static_phase = jnp.exp(-1j * jnp.pi * xi_f)                       # window phase
    chirp_phase  = jnp.exp(-1j * jnp.pi * chi_f * (xi_f**2 - 1.0/6.0))
    return (K0 * static_phase * chirp_phase).astype(jnp.complex64)


def sft_kernel_exact(xi: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
    r"""Exact SFT Fresnel kernel via Gauss-Legendre quadrature.

    Computes

    .. math::

        K^{\rm SFT}(\xi, \chi)
          = \int_0^1 e^{-2\pi i\,\xi\, u - \pi i\,\chi\, u^2}\, du

    directly with :data:`_GL_N`-point Gauss-Legendre quadrature on
    :math:`[0, 1]`.  No series approximation; the integrand is evaluated at
    fixed nodes and the result is a weighted sum.

    Accuracy: the GL-32 rule integrates polynomials exactly up to degree 63.
    For oscillatory integrands the effective accuracy degrades when the number
    of oscillations within the interval exceeds :math:`N/\pi \approx 10`
    (i.e. :math:`|\chi| \gtrsim 10`).  Increase :data:`_GL_N` if near-plunge
    accuracy is required.

    This implementation is fully JAX-traceable and JIT-compilable.

    Parameters
    ----------
    xi : jnp.ndarray
        Normalised frequency offset.
    chi : jnp.ndarray
        Dimensionless chirp.

    Returns
    -------
    jnp.ndarray, complex64, same shape as *xi*.

    See Also
    --------
    sft_kernel : faster approximate version.
    """
    xi_f  = jnp.asarray(xi,  dtype=jnp.float32)
    chi_f = jnp.asarray(chi, dtype=jnp.float32)
    # Broadcast to (..., N_GL)
    u = _GL_U_JAX   # (N,)
    w = _GL_W_JAX   # (N,)
    phase = (
        -2j * jnp.pi * xi_f[..., None] * u
        -  1j * jnp.pi * chi_f[..., None] * u ** 2
    )
    integrand = jnp.exp(phase.astype(jnp.complex64))    # (..., N)
    return (integrand * w).sum(axis=-1).astype(jnp.complex64)


# ---------------------------------------------------------------------------
# SFTGrid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SFTGrid(TFGrid):
    """Non-overlapping Short-Time Fourier Transform grid.

    Parameters
    ----------
    Nf : int
        Number of positive-frequency bins per segment.
        Determines Nyquist: :math:`f_{\\rm nyq} = N_f \\, \\Delta F`.
    Nt : int
        Number of segments (= :math:`T / T_{\\rm coh}`).
    T : float
        Total duration [s].  Must equal :math:`N_t \\cdot T_{\\rm coh}`.
    use_exact_kernel : bool
        Use GL-:data:`_GL_N` quadrature (:func:`sft_kernel_exact`, default
        True) instead of the approximate sinc kernel.  The GL-64 kernel is
        accurate for :math:`|\\chi| \\lesssim 40`; increase :data:`_GL_N`
        for near-plunge segments with very large :math:`|\\chi|`.  The
        approximate sinc kernel (``use_exact_kernel=False``) is accurate
        only for :math:`|\\chi| \\ll 1` — not recommended for typical
        1-day EMRI SFT segments where :math:`\\chi \\sim 0.1`.

    Notes
    -----
    The SFT frequency resolution :math:`\\Delta F = 1/T_{\\rm coh}` is
    *coarser* than WDM's :math:`\\Delta F = 1/(2\\Delta T)` for the same
    time-bin width — equivalently, for the same frequency resolution, the
    SFT has half the time bins of a WDM grid.

    Due to the sinc frequency response, EMRI modes spread power across
    several frequency bins.  Use ``hw >= 3`` in
    :func:`~fewtrax.summation.tf_sum.direct_tf_sum` for accurate deposition
    (vs. ``hw = 2`` for WDM whose Meyer window has compact support).
    """

    Nf: int
    Nt: int
    T: float
    use_exact_kernel: bool = True

    @property
    def delta_T(self) -> float:
        """Segment (coherence) time :math:`T_{\\rm coh}` [s]."""
        return self.T / self.Nt

    @property
    def delta_F(self) -> float:
        r"""Frequency bin width :math:`1/T_{\rm coh}` [Hz].

        Note this is :math:`2\times` the WDM bin width for the same
        :math:`\Delta T`.
        """
        return 1.0 / self.delta_T

    def kernel(self, xi: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        """SFT Fresnel kernel — exact (GL) or approximate (sinc) based on ``use_exact_kernel``."""
        if self.use_exact_kernel:
            return sft_kernel_exact(xi, chi)
        return sft_kernel(xi, chi)

    def __repr__(self) -> str:
        k_label = "exact-GL" if self.use_exact_kernel else "approx"
        return (
            f"SFTGrid(Nf={self.Nf}, Nt={self.Nt}, "
            f"T={self.T:.3e}s, "
            f"T_coh={self.delta_T:.0f}s={self.delta_T/86400:.2f}d, "
            f"dF={self.delta_F*1e6:.2f}μHz, "
            f"f_nyq={self.f_nyq*1e3:.2f}mHz, "
            f"kernel={k_label})"
        )


def default_sft_grid(
    T: float,
    f_max: float = 5e-3,
    T_coh: float = 86400.0,
) -> SFTGrid:
    """Construct an SFT grid with a given coherence time.

    Parameters
    ----------
    T : float
        Total observation duration [s].
    f_max : float
        Maximum frequency to cover [Hz].
    T_coh : float
        Segment length [s].  Default 86400 s (1 day).
    Returns
    -------
    SFTGrid
        The total duration is ceiled to the nearest integer multiple of
        ``T_coh``.
    """
    Nt = max(1, int(round(T / T_coh)))
    T_snap = Nt * T_coh
    dF = 1.0 / T_coh
    Nf = int(np.ceil(f_max / dF))
    return SFTGrid(Nf=Nf, Nt=Nt, T=T_snap)

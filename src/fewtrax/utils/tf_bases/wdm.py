"""Wilson–Daubechies–Meyer (WDM) time-frequency grid and kernel.

The WDM basis is an orthonormal tiling of the time-frequency plane with
Meyer-windowed atoms.  The kernel has compact frequency support (exactly
zero beyond one bin width), is smooth (C∞), and supports a leading-order
Fresnel chirp correction for slowly-chirping signals.

This module is the authoritative definition of :class:`WDMGrid`,
:func:`meyer_window`, and :func:`meyer_kernel`.  All earlier references to
these names in ``fewtrax.utils.tf_tracks`` are now re-exports from here.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

from fewtrax.utils.tf_bases.base import TFGrid


# ---------------------------------------------------------------------------
# Meyer window utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Exact WDM kernel — GL-128 quadrature (computed once at import time)
# ---------------------------------------------------------------------------
#
# Derivation: the WDM pixel coefficient for a locally chirping tone reduces to
#
#   K(ξ, χ) = (1/√|χ|)·exp(−iπ/4·sign(χ))·∫_{-1}^{1} φ_Meyer(ξ')·exp(iπ(ξ'−ξ)²/χ) dξ'
#
# where ξ = (q·ΔF − f)/ΔF and χ = ḟ·ΔT².  The integrand is a smooth function
# of ξ' on the compact support of φ_Meyer, making 128-point GL quadrature
# accurate to <10⁻⁴ for |χ| ≤ 40 (>3 GL points per oscillation cycle).
# For near-plunge segments with |χ| > 40, increase _N_WDM_EXACT.
#
# The result depends only on (ξ, χ); all source parameters feed in through
# those two scalars, so the function is JIT-compiled once for all sources.

_N_WDM_EXACT: int = 128
_WDM_KERNEL_CHI_MIN: float = 1e-3   # SPA fallback below this |χ|

_wdm_gl_nodes_np, _wdm_gl_weights_np = np.polynomial.legendre.leggauss(_N_WDM_EXACT)

# φ_Meyer evaluated at the GL nodes (pure numpy, executed at import time)
def _meyer_window_np(xi: np.ndarray) -> np.ndarray:
    xi_abs = np.abs(xi)
    t = np.clip(2.0 * xi_abs - 1.0, 0.0, 1.0)
    nu_t = t**4 * (35.0 - t * (84.0 - t * (70.0 - 20.0 * t)))
    return np.where(
        xi_abs <= 0.5, 1.0,
        np.where(xi_abs <= 1.0, np.cos(0.5 * np.pi * nu_t), 0.0)
    ).astype(np.float32)

# Precomputed constants on GL nodes over [-1, 1] (the support of φ_Meyer)
_WDM_GL_XI:  jnp.ndarray = jnp.asarray(_wdm_gl_nodes_np,   dtype=jnp.float32)
_WDM_GL_W:   jnp.ndarray = jnp.asarray(_wdm_gl_weights_np, dtype=jnp.float32)
_WDM_GL_PHI: jnp.ndarray = jnp.asarray(
    _meyer_window_np(_wdm_gl_nodes_np), dtype=jnp.float32
)

def _nu(x: jnp.ndarray) -> jnp.ndarray:
    """Smooth transition ν: [0,1]→[0,1] (C∞, all derivatives zero at endpoints).

    Unique degree-7 polynomial satisfying ν(0)=0, ν(1)=1, and zero first
    and second derivatives at both endpoints.
    """
    return x ** 4 * (35.0 - x * (84.0 - x * (70.0 - 20.0 * x)))


def meyer_window(xi: jnp.ndarray) -> jnp.ndarray:
    r"""Real Meyer low-pass window amplitude in normalised frequency ξ = Δ/ΔF.

    .. math::

        K_0(\xi) = \begin{cases}
            1 & |\xi| \leq \tfrac{1}{2} \\
            \cos\!\left(\tfrac{\pi}{2}\,\nu(2|\xi|-1)\right)
              & \tfrac{1}{2} < |\xi| \leq 1 \\
            0 & |\xi| > 1
        \end{cases}

    where :math:`\nu` is :func:`_nu`.  The window has compact frequency
    support and satisfies the WDM Parseval condition.
    """
    xi_abs = jnp.abs(xi)
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


def wdm_kernel_exact(
    delta_over_dF: jnp.ndarray,
    chi: jnp.ndarray,
) -> jnp.ndarray:
    r"""Exact WDM Meyer pixel kernel via 128-point Gauss-Legendre quadrature.

    Evaluates the Fresnel-form integral

    .. math::

        K(\xi, \chi) = \frac{1}{\sqrt{|\chi|}}
            e^{-i\pi\,\mathrm{sign}(\chi)/4}
            \int_{-1}^{1} \varphi_{\rm Meyer}(\xi')\,
            e^{i\pi(\xi'-\xi)^2/\chi}\,d\xi'

    using :data:`_N_WDM_EXACT`-point GL quadrature on :math:`[-1, 1]`.
    For :math:`|\chi| < 10^{-3}` falls back to the SPA limit
    :math:`K = \varphi_{\rm Meyer}(\xi)`.

    Accurate to :math:`< 10^{-4}` for :math:`|\chi| \le 40`
    (≥ 3 GL points per phase-cycle).  For near-plunge segments where
    :math:`|\chi| > 40`, increase :data:`_N_WDM_EXACT`.

    Parameters
    ----------
    delta_over_dF : jnp.ndarray
        Normalised frequency offset :math:`\xi = (q\Delta F - f)/\Delta F`.
    chi : jnp.ndarray
        Dimensionless chirp :math:`\dot{f}\,\Delta T^2`.

    Returns
    -------
    K : jnp.ndarray, complex64
    """
    xi_f  = jnp.asarray(delta_over_dF, dtype=jnp.float32)
    chi_f = jnp.asarray(chi,           dtype=jnp.float32)

    # Replace near-zero chi with 1 to avoid 1/√0; output is selected by jnp.where
    chi_safe = jnp.where(jnp.abs(chi_f) >= _WDM_KERNEL_CHI_MIN,
                         chi_f, jnp.ones_like(chi_f))

    # GL quadrature over ξ' ∈ [-1, 1]; shape: (..., _N_WDM_EXACT)
    dxi      = _WDM_GL_XI - xi_f[..., None]                           # (..., N)
    phase    = jnp.pi * dxi**2 / chi_safe[..., None]                  # (..., N)
    integrand = _WDM_GL_PHI * jnp.exp(1j * phase.astype(jnp.float32)) # (..., N)
    integral  = (integrand * _WDM_GL_W).sum(axis=-1)                   # (...)

    prefactor = (
        jnp.exp(-1j * jnp.pi / 4.0 * jnp.sign(chi_safe))
        / jnp.sqrt(jnp.abs(chi_safe))
    )
    K_exact = (prefactor * integral).astype(jnp.complex64)
    K_spa   = meyer_window(xi_f).astype(jnp.complex64)

    return jnp.where(jnp.abs(chi_f) < _WDM_KERNEL_CHI_MIN, K_spa, K_exact)


def meyer_kernel(
    delta_over_dF: jnp.ndarray,
    chi: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Complex WDM pixel kernel with optional leading-order Fresnel correction.

    In the locally-monochromatic limit (``chi=None``):

    .. math::

        K(\xi) = K_0(\xi)

    With chirp correction (:math:`\chi = \dot f \Delta T^2`):

    .. math::

        K(\xi, \chi) \approx K_0(\xi)\cdot e^{-i\pi\chi\xi^2}

    Parameters
    ----------
    delta_over_dF : jnp.ndarray
        Normalised frequency offset :math:`(q\Delta F - f_n)/\Delta F`.
    chi : jnp.ndarray or None
        Dimensionless chirp.  Pass ``None`` to get the real-valued
        locally-monochromatic kernel.

    Returns
    -------
    K : jnp.ndarray, complex64
    """
    K0 = meyer_window(delta_over_dF)
    if chi is None:
        return K0.astype(jnp.complex64)
    phase_corr = jnp.exp(-1j * jnp.pi * jnp.asarray(chi) * delta_over_dF ** 2)
    return (K0 * phase_corr).astype(jnp.complex64)


# ---------------------------------------------------------------------------
# WDMGrid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WDMGrid(TFGrid):
    """Parameters of a WDM time-frequency grid.

    Parameters
    ----------
    Nf : int
        Number of frequency bins.
    Nt : int
        Number of time bins.
    T : float
        Total duration [s].
    chirp_correction : bool
        Apply the leading-order Fresnel phase
        :math:`e^{-i\\pi\\chi\\xi^2}` to the Meyer window (default True).
        Set to False to recover the real-valued locally-monochromatic kernel
        (useful for debugging or validation against static tones).

    Notes
    -----
    The underlying time series has ``N = Nf * Nt`` samples.

    * Time bin width:       :math:`\\Delta T = T / N_t`
    * Frequency bin width:  :math:`\\Delta F = 1 / (2\\Delta T)`
    * Nyquist frequency:    :math:`f_{\\rm nyq} = N_f\\,\\Delta F`
    """

    Nf: int
    Nt: int
    T: float
    chirp_correction: bool = True
    use_exact_kernel: bool = True

    @property
    def delta_T(self) -> float:
        """WDM time bin width [s]."""
        return self.T / self.Nt

    @property
    def delta_F(self) -> float:
        """WDM frequency bin width [Hz]."""
        return 1.0 / (2.0 * self.delta_T)

    def kernel(self, xi: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        """Meyer kernel dispatched by ``use_exact_kernel`` and ``chirp_correction``.

        * ``use_exact_kernel=True`` (default): :func:`wdm_kernel_exact` —
          128-point GL quadrature, valid for all :math:`|\\chi| \\le 40`.
        * ``use_exact_kernel=False, chirp_correction=True``: leading-order
          approximation :math:`\\varphi(\\xi)\\cdot e^{-i\\pi\\chi\\xi^2}`.
        * ``use_exact_kernel=False, chirp_correction=False``: static SPA
          kernel :math:`\\varphi(\\xi)` (chi ignored).
        """
        if self.use_exact_kernel:
            return wdm_kernel_exact(xi, chi)
        return meyer_kernel(xi, chi if self.chirp_correction else None)

    def __repr__(self) -> str:
        if self.use_exact_kernel:
            k_label = "exact-GL128"
        elif self.chirp_correction:
            k_label = "approx-chirp"
        else:
            k_label = "static-SPA"
        return (
            f"WDMGrid(Nf={self.Nf}, Nt={self.Nt}, "
            f"T={self.T:.3e}s, "
            f"dT={self.delta_T:.1f}s, "
            f"dF={self.delta_F*1e6:.2f}μHz, "
            f"f_nyq={self.f_nyq*1e3:.2f}mHz, "
            f"kernel={k_label})"
        )


def default_grid(
    T: float,
    f_max: float = 5e-3,
    Nt: int = 4096,
    use_exact_kernel: bool = True,
) -> WDMGrid:
    """Construct a WDM grid covering [0, f_max] Hz over duration T.

    Parameters
    ----------
    T : float
        Observation duration [s].
    f_max : float
        Desired maximum frequency [Hz].  Nf is rounded up to the next
        power of 2 so that the Nyquist frequency is at least f_max.
    Nt : int
        Number of WDM time bins (power of 2 recommended).
    use_exact_kernel : bool
        Use :func:`wdm_kernel_exact` (GL-128, default True) instead of the
        leading-order chirp-phase approximation.  Set to False only for
        debugging or performance testing.
    """
    delta_T = T / Nt
    delta_F = 1.0 / (2.0 * delta_T)
    Nf = int(2 ** np.ceil(np.log2(f_max / delta_F)))
    return WDMGrid(Nf=Nf, Nt=Nt, T=T, use_exact_kernel=use_exact_kernel)

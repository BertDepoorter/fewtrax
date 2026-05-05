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

    @property
    def delta_T(self) -> float:
        """WDM time bin width [s]."""
        return self.T / self.Nt

    @property
    def delta_F(self) -> float:
        """WDM frequency bin width [Hz]."""
        return 1.0 / (2.0 * self.delta_T)

    def kernel(self, xi: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        """Meyer kernel, optionally with Fresnel chirp correction."""
        return meyer_kernel(xi, chi if self.chirp_correction else None)

    def __repr__(self) -> str:
        return (
            f"WDMGrid(Nf={self.Nf}, Nt={self.Nt}, "
            f"T={self.T:.3e}s, "
            f"dT={self.delta_T:.1f}s, "
            f"dF={self.delta_F*1e6:.2f}μHz, "
            f"f_nyq={self.f_nyq*1e3:.2f}mHz, "
            f"chirp={self.chirp_correction})"
        )


def default_grid(T: float, f_max: float = 5e-3, Nt: int = 4096) -> WDMGrid:
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
    """
    delta_T = T / Nt
    delta_F = 1.0 / (2.0 * delta_T)
    Nf = int(2 ** np.ceil(np.log2(f_max / delta_F)))
    return WDMGrid(Nf=Nf, Nt=Nt, T=T)

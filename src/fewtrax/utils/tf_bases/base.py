"""Abstract time-frequency grid base class and basis-agnostic mode deposition."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp


class TFGrid(ABC):
    r"""Abstract base for time-frequency grid descriptors.

    Subclasses must provide integer fields ``Nf`` and ``Nt`` (as frozen
    dataclass fields), implement ``delta_T`` and ``delta_F`` as properties,
    and define a ``kernel`` method that maps normalised frequency offsets and
    chirp parameters to complex pixel weights.

    Implemented daughters:
    :class:`~fewtrax.utils.tf_bases.wdm.WDMGrid` (Wilson-Daubechies-Meyer)
    and :class:`~fewtrax.utils.tf_bases.sft.SFTGrid` (Short-Time Fourier
    Transform).
    """

    @property
    @abstractmethod
    def delta_T(self) -> float:
        """Time pixel width [s]."""
        ...

    @property
    @abstractmethod
    def delta_F(self) -> float:
        """Frequency bin width [Hz]."""
        ...

    @abstractmethod
    def kernel(self, xi: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        r"""Complex pixel kernel :math:`K(\xi, \chi)`.

        Parameters
        ----------
        xi : jnp.ndarray
            Normalised frequency offset :math:`(q\,\Delta F - f)/\Delta F`.
        chi : jnp.ndarray
            Dimensionless chirp :math:`\dot f\,\Delta T^2`.  Must be
            broadcastable with *xi*.

        Returns
        -------
        jnp.ndarray, complex64, same shape as *xi*.
        """
        ...

    @property
    def t_bins(self) -> np.ndarray:
        """Pixel-centre times [s], shape ``(Nt,)``."""
        return np.arange(self.Nt, dtype=np.float64) * self.delta_T

    @property
    def f_bins(self) -> np.ndarray:
        """Pixel-centre frequencies [Hz], shape ``(Nf,)``."""
        return np.arange(self.Nf, dtype=np.float64) * self.delta_F

    @property
    def f_nyq(self) -> float:
        """Nyquist frequency [Hz]."""
        return self.Nf * self.delta_F

    def freq_to_bin(self, f: np.ndarray) -> np.ndarray:
        """Round frequencies [Hz] to nearest integer bin index."""
        return np.round(f / self.delta_F).astype(int)


def direct_tf_mode(
    A_n: jnp.ndarray,
    Phi_n: jnp.ndarray,
    f_n: jnp.ndarray,
    fdot_n: jnp.ndarray,
    grid: TFGrid,
    hw: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Basis-agnostic deposition of one harmonic mode onto a TF grid.

    For each time pixel :math:`n` and the :math:`2h_w + 1` adjacent
    frequency bins, computes

    .. math::

        w_{nq} = A(t_n)\,e^{-i\Phi(t_n)}\cdot
                 K\!\!\left(
                     \frac{q\Delta F - f_n}{\Delta F},\;
                     \dot f_n\,\Delta T^2
                 \right)

    where :math:`K` is ``grid.kernel(xi, chi)``.  The function is pure JAX
    and works with any :class:`TFGrid` subclass.

    Parameters
    ----------
    A_n : jnp.ndarray, shape (Nt,), complex
        Complex mode amplitude (including :math:`Y_{\ell m}` prefactor and
        physical units) at each time-pixel centre.
    Phi_n : jnp.ndarray, shape (Nt,)
        Accumulated mode phase [rad].
    f_n : jnp.ndarray, shape (Nt,), float32
        Instantaneous GW frequency [Hz].
    fdot_n : jnp.ndarray, shape (Nt,), float32
        Frequency derivative [Hz s⁻¹].
    grid : TFGrid
        Any concrete grid descriptor.  The kernel is taken from
        ``grid.kernel(xi, chi)``.
    hw : int
        Frequency half-width: bins :math:`q_0 - h_w, \ldots, q_0 + h_w`
        are filled.

    Returns
    -------
    q_clipped : jnp.ndarray, shape (Nt, 2*hw+1), int32
        Frequency bin indices, clipped to ``[0, Nf-1]``.
    w : jnp.ndarray, shape (Nt, 2*hw+1), complex64
        Complex pixel coefficients.
    """
    dF = grid.delta_F
    dT = grid.delta_T

    q0 = jnp.round(f_n / dF).astype(jnp.int32)                       # (Nt,)
    dq = jnp.arange(-hw, hw + 1, dtype=jnp.int32)                     # (2hw+1,)
    q  = q0[:, None] + dq[None, :]                                     # (Nt, 2hw+1)
    q_clipped = jnp.clip(q, 0, grid.Nf - 1)

    xi  = (q_clipped.astype(jnp.float32) * dF - f_n[:, None]) / dF    # (Nt, 2hw+1)
    chi = (fdot_n * dT ** 2)[:, None] * jnp.ones((1, 2 * hw + 1),
                                                   dtype=jnp.float32)  # (Nt, 2hw+1)

    K      = grid.kernel(xi, chi)                                      # (Nt, 2hw+1), complex64
    phasor = (A_n * jnp.exp(-1j * Phi_n))[:, None]                    # (Nt, 1)
    w      = (phasor * K).astype(jnp.complex64)
    return q_clipped, w

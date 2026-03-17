"""Harmonic mode summation for EMRI waveforms.

This module implements the coherent summation of Teukolsky harmonic modes
to produce the gravitational-wave strain:

.. math::

    h_+ - i h_{\\times} =
        \\frac{\\mu}{d_L}
        \\sum_{\\ell, m, k, n}
        A_{\\ell m k n}(t) \\;
        {}_{-2}Y_{\\ell m}(\\theta, \\phi) \\;
        e^{-i \\Phi_{\\ell m k n}(t)}
        + \\text{c.c.}

where the phase is

.. math::

    \\Phi_{\\ell m k n}(t) = m \\Phi_\\phi(t) + k \\Phi_\\theta(t) + n \\Phi_r(t)

and the complex conjugate (c.c.) term accounts for :math:`m < 0` modes.

The summation is performed by :func:`direct_mode_sum`, which is a
pure JAX function that can be JIT-compiled and differentiated:

.. code-block:: python

    import jax
    jit_sum = jax.jit(direct_mode_sum)
    h = jit_sum(teuk_modes, ylms_pos, ylms_neg, Phi_phi, Phi_theta, Phi_r,
                l_arr, m_arr, k_arr, n_arr)

For parameter estimation, the derivative of the strain with respect
to waveform parameters is obtained simply by:

.. code-block:: python

    dh_dp0 = jax.grad(lambda p0: jnp.sum(jnp.abs(
        jit_sum(..., Phi_phi(p0), ...))**2))(p0)

Utilities
---------
:func:`interpolated_mode_sum`
    Upsamples the sparse trajectory to a dense time grid before
    summation using cubic spline interpolation of the phases.
:class:`ModeSum`
    Stateful wrapper that caches the trajectory phases and amplitudes.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

from fewtrax.utils.constants import MTSUN_SI, GPC_SI, YEAR_SI


# ---------------------------------------------------------------------------
# Core summation (pure JAX, fully JIT-compilable)
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=())
def direct_mode_sum(
    teuk_modes: jnp.ndarray,
    ylms_pos: jnp.ndarray,
    ylms_neg: jnp.ndarray,
    Phi_phi: jnp.ndarray,
    Phi_theta: jnp.ndarray,
    Phi_r: jnp.ndarray,
    l_arr: jnp.ndarray,
    m_arr: jnp.ndarray,
    k_arr: jnp.ndarray,
    n_arr: jnp.ndarray,
) -> jnp.ndarray:
    r"""Coherently sum Teukolsky modes to produce the GW strain.

    Computes

    .. math::

        h(t) =
            \sum_{(\ell,m,k,n),\; m \ge 0}
            Y_{+}(\ell,m) \; A_{\ell m k n}(t) \;
            e^{-i(m \Phi_\phi + k \Phi_\theta + n \Phi_r)}
            +
            \sum_{(\ell,m,k,n),\; m > 0}
            (-1)^\ell \; Y_{-}(\ell,m) \;
            \bar{A}_{\ell m k n}(t) \;
            e^{+i(m \Phi_\phi + k \Phi_\theta + n \Phi_r)}

    where :math:`Y_{+}` and :math:`Y_{-}` are the spin-weighted spherical
    harmonics for :math:`+m` and :math:`-m`.

    Parameters
    ----------
    teuk_modes : jnp.ndarray, shape (N_t, N_modes), complex128
        Teukolsky mode amplitudes at each time step.  Assumed to be for
        :math:`m \ge 0` modes only (the negative-m term uses conjugation).
    ylms_pos : jnp.ndarray, shape (N_modes,), complex128
        :math:`{}_{-2}Y_{\ell m}(\theta, \phi)` for each mode.
    ylms_neg : jnp.ndarray, shape (N_modes,), complex128
        :math:`{}_{-2}Y_{\ell, -m}(\theta, \phi)` for each mode.
    Phi_phi, Phi_theta, Phi_r : jnp.ndarray, shape (N_t,)
        Orbital phases.
    l_arr, m_arr, k_arr, n_arr : jnp.ndarray, shape (N_modes,), int
        Mode index arrays.

    Returns
    -------
    jnp.ndarray, shape (N_t,), complex128
        Complex strain :math:`h_+ - i h_{\times}` at each time step.

    Notes
    -----
    The amplitude arrays ``teuk_modes`` are passed as regular (non-JAX
    static) arrays and can originate from numpy.  They will be converted
    to JAX arrays on first call.
    """
    # Ensure JAX types
    teuk_modes = jnp.asarray(teuk_modes)
    l_arr = jnp.asarray(l_arr)
    m_arr = jnp.asarray(m_arr)
    k_arr = jnp.asarray(k_arr)
    n_arr = jnp.asarray(n_arr)

    # Phase: shape (N_t, N_modes)
    phase = (
        m_arr[jnp.newaxis, :] * Phi_phi[:, jnp.newaxis]
        + k_arr[jnp.newaxis, :] * Phi_theta[:, jnp.newaxis]
        + n_arr[jnp.newaxis, :] * Phi_r[:, jnp.newaxis]
    )

    # Positive-m contribution (all modes, including m=0)
    w1 = jnp.sum(
        ylms_pos[jnp.newaxis, :] * teuk_modes * jnp.exp(-1j * phase),
        axis=1,
    )

    # Negative-m contribution (only m > 0 modes, using symmetry)
    m_pos_mask = (m_arr > 0).astype(jnp.float64)
    sign_l = (-1.0) ** l_arr
    w2 = jnp.sum(
        m_pos_mask[jnp.newaxis, :]
        * sign_l[jnp.newaxis, :]
        * ylms_neg[jnp.newaxis, :]
        * jnp.conj(teuk_modes)
        * jnp.exp(1j * phase),
        axis=1,
    )

    return w1 + w2


# ---------------------------------------------------------------------------
# Interpolated mode sum (dense waveform from sparse trajectory)
# ---------------------------------------------------------------------------

def interpolated_mode_sum(
    t_traj: jnp.ndarray,
    teuk_modes: np.ndarray,
    ylms_pos: jnp.ndarray,
    ylms_neg: jnp.ndarray,
    Phi_phi: jnp.ndarray,
    Phi_theta: jnp.ndarray,
    Phi_r: jnp.ndarray,
    l_arr: np.ndarray,
    m_arr: np.ndarray,
    k_arr: np.ndarray,
    n_arr: np.ndarray,
    dt: float = 10.0,
    M: float = 1.0,
    T: float = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Produce a densely sampled waveform by interpolating the phases.

    The trajectory is computed on a sparse grid; this function upsamples
    the phases to a uniformly spaced time grid with step ``dt`` using
    cubic spline interpolation, then calls :func:`direct_mode_sum`.

    Parameters
    ----------
    t_traj : jnp.ndarray, shape (N_traj,)
        Sparse trajectory time stamps [s].
    teuk_modes : array_like, shape (N_traj, N_modes)
        Mode amplitudes at each trajectory point.
    ylms_pos, ylms_neg : jnp.ndarray, shape (N_modes,)
        Spherical harmonics.
    Phi_phi, Phi_theta, Phi_r : jnp.ndarray, shape (N_traj,)
        Orbital phases at trajectory points.
    l_arr, m_arr, k_arr, n_arr : array_like of int
        Mode index arrays.
    dt : float
        Output sampling interval [s].
    M : float
        Primary mass :math:`[M_\\odot]`.
    T : float, optional
        Total duration [years].  If not provided, uses ``t_traj[-1]``.

    Returns
    -------
    t_dense : jnp.ndarray, shape (N_dense,)
        Dense time stamps [s].
    h : jnp.ndarray, shape (N_dense,), complex128
        Waveform strain.
    """
    from interpax import Interpolator1D

    t_np = np.asarray(t_traj)
    # Mask out NaN values (can arise from event termination)
    valid = ~np.isnan(t_np) & ~np.isinf(t_np)
    t_np = t_np[valid]
    N_valid = t_np.size

    if N_valid < 2:
        raise ValueError("Not enough valid trajectory points for interpolation.")

    T_s = float(t_np[-1]) if T is None else T * YEAR_SI

    n_dense = max(2, int(np.round(T_s / dt)) + 1)
    t_dense = jnp.linspace(float(t_np[0]), float(t_np[-1]), n_dense)

    def _interp(arr):
        arr_np = np.asarray(arr)[valid]
        spl = Interpolator1D(t_np, arr_np, method="cubic2", extrap=True)
        return spl(t_dense)

    Phi_phi_d = _interp(Phi_phi)
    Phi_theta_d = _interp(Phi_theta)
    Phi_r_d = _interp(Phi_r)

    # Interpolate amplitude magnitudes and phases separately
    teuk_np = np.asarray(teuk_modes)[valid, :]
    teuk_dense_list = []
    for im in range(teuk_np.shape[1]):
        spl_r = Interpolator1D(t_np, teuk_np[:, im].real, method="cubic2", extrap=True)
        spl_i = Interpolator1D(t_np, teuk_np[:, im].imag, method="cubic2", extrap=True)
        teuk_dense_list.append(spl_r(t_dense) + 1j * spl_i(t_dense))
    teuk_dense = jnp.stack(teuk_dense_list, axis=1)  # (N_dense, N_modes)

    h = direct_mode_sum(
        teuk_dense, ylms_pos, ylms_neg,
        Phi_phi_d, Phi_theta_d, Phi_r_d,
        jnp.asarray(l_arr), jnp.asarray(m_arr),
        jnp.asarray(k_arr), jnp.asarray(n_arr),
    )
    return t_dense, h


# ---------------------------------------------------------------------------
# Frequency-domain transform (re-exported from fewtrax.utils.transforms)
# ---------------------------------------------------------------------------

from fewtrax.utils.transforms import to_frequency_domain, to_time_domain  # noqa: F401


# ---------------------------------------------------------------------------
# Stateful wrapper
# ---------------------------------------------------------------------------

class ModeSum:
    r"""Stateful wrapper that caches trajectory data and evaluates the sum.

    Parameters
    ----------
    l_arr, m_arr, k_arr, n_arr : array_like of int
        Mode index arrays.
    ylms_pos, ylms_neg : array_like, complex
        Spherical harmonics for each mode.
    dist : float
        Luminosity distance [Gpc].
    M : float
        Primary mass [:math:`M_\\odot`].
    mu : float
        Secondary mass [:math:`M_\\odot`].

    Examples
    --------
    >>> summer = ModeSum(l_arr, m_arr, k_arr, n_arr, ylms_pos, ylms_neg,
    ...                  dist=1.0, M=1e6, mu=10.0)
    >>> h = summer(t_traj, teuk_modes, Phi_phi, Phi_theta, Phi_r, dt=10.0)
    """

    def __init__(
        self,
        l_arr: np.ndarray,
        m_arr: np.ndarray,
        k_arr: np.ndarray,
        n_arr: np.ndarray,
        ylms_pos: jnp.ndarray,
        ylms_neg: jnp.ndarray,
        dist: float = 1.0,
        M: float = 1.0,
        mu: float = 1.0,
    ):
        self.l_arr = jnp.asarray(l_arr)
        self.m_arr = jnp.asarray(m_arr)
        self.k_arr = jnp.asarray(k_arr)
        self.n_arr = jnp.asarray(n_arr)
        self.ylms_pos = jnp.asarray(ylms_pos)
        self.ylms_neg = jnp.asarray(ylms_neg)
        self.dist = dist
        self.M = M
        self.mu = mu
        # Amplitude prefactor: mu * G / (c^2 * d_L)
        from fewtrax.utils.constants import G_SI, C_SI, MSUN_SI
        self._amp = (
            mu * MSUN_SI * G_SI / C_SI**2
            / (dist * GPC_SI)
        )

    def __call__(
        self,
        t_traj: jnp.ndarray,
        teuk_modes: np.ndarray,
        Phi_phi: jnp.ndarray,
        Phi_theta: jnp.ndarray,
        Phi_r: jnp.ndarray,
        dt: float = 10.0,
        dense: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Evaluate the gravitational-wave strain.

        Parameters
        ----------
        t_traj : jnp.ndarray, shape (N_traj,)
            Sparse trajectory time stamps [s].
        teuk_modes : np.ndarray, shape (N_traj, N_modes), complex
            Mode amplitudes at trajectory points.
        Phi_phi, Phi_theta, Phi_r : jnp.ndarray, shape (N_traj,)
            Orbital phases.
        dt : float
            Waveform sampling interval [s].
        dense : bool
            If True, upsample to a dense grid.  If False, return the
            sparse trajectory summation.

        Returns
        -------
        t : jnp.ndarray
            Time stamps [s].
        h : jnp.ndarray, complex
            :math:`h_+ - i h_{\times}` in physical units.
        """
        if dense:
            t_out, h = interpolated_mode_sum(
                t_traj, teuk_modes,
                self.ylms_pos, self.ylms_neg,
                Phi_phi, Phi_theta, Phi_r,
                self.l_arr, self.m_arr, self.k_arr, self.n_arr,
                dt=dt, M=self.M,
            )
        else:
            h = direct_mode_sum(
                jnp.asarray(teuk_modes),
                self.ylms_pos, self.ylms_neg,
                Phi_phi, Phi_theta, Phi_r,
                self.l_arr, self.m_arr, self.k_arr, self.n_arr,
            )
            t_out = t_traj

        return t_out, h * self._amp

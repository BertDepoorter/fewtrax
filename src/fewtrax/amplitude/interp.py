"""Teukolsky mode amplitude interpolation from FEW precomputed grids.

The amplitude HDF5 file stores bicubic B-spline coefficients
in ``scipy.interpolate.bisplev`` format (one set per z-slice per mode).
This module evaluates those coefficients at query trajectory points using
``scipy.interpolate.bisplev`` and linear interpolation in the spin
coordinate z, returning complex mode amplitudes as a numpy array.

The result is then passed to the JAX summation module.

Design notes
------------
* The amplitude evaluation is intentionally **numpy-based** rather than
  JAX-based.  The spline coefficient arrays are large (several GB for all
  6993 modes), and precomputing them as JAX arrays on the GPU would exceed
  typical GPU memory budgets.  Instead, the scipy B-spline evaluation runs
  on the CPU and only the resulting (N_traj, N_modes) amplitude array is
  transferred to the GPU for the mode summation.
* Mode selection (thresholding by relative power) is done here to
  reduce the number of modes passed to the summation step.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from fewtrax.utils.coordinates import (
    ALPHA_AMP, BETA_AMP, DELTAPMIN, DELTAPMAX,
)
from fewtrax.data.loader import AmplitudeData, _separatrix_numpy


class AmplitudeInterpolator:
    r"""Evaluate Teukolsky mode amplitudes :math:`A_{\ell mkn}(a, p, e)`.

    Parameters
    ----------
    amp_data : AmplitudeData
        Pre-loaded amplitude data (B-spline coefficients + mode arrays).
    """

    def __init__(self, amp_data: AmplitudeData):
        self.amp_data = amp_data

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _z_of_a(self, a_abs: float) -> float:
        """Spin → z coordinate."""
        chi_max = (1 - (-0.999))**(1/3)
        chi_min = (1 - 0.999)**(1/3)
        chi = (1.0 - a_abs)**(1/3)
        return float(np.clip((chi - chi_min) / (chi_max - chi_min), 0.0, 1.0))

    def _forward_map_A(self, a_abs: float, p: float, e: float):
        """Map (a, p, e) → (u, w, z) in Region A (amplitude coordinates)."""
        pLSO = float(_separatrix_numpy(a_abs, e, 1.0))
        dp = max(p - pLSO - DELTAPMIN, 0.0)
        u_raw = np.log1p(dp / (DELTAPMAX - DELTAPMIN)) / np.log(2.0)
        u = float(np.clip(u_raw**ALPHA_AMP, 0.0, 1.0))
        z = self._z_of_a(a_abs)
        ESEP, EMAX_LOC = 0.25, 0.9
        check = z + u**BETA_AMP * (1.0 - z)
        sgn = np.sign(check)
        Secc = ESEP + (EMAX_LOC - ESEP) * sgn * np.sqrt(sgn * max(check, 0.0))
        Secc = max(Secc, 1e-12)
        w = float(np.clip(e / Secc, 0.0, 1.0))
        return u, w, z

    def _forward_map_B(self, a_abs: float, p: float, e: float):
        """Map (a, p, e) → (u, w, z) in Region B (amplitude coordinates)."""
        pLSO = float(_separatrix_numpy(a_abs, e, 1.0))
        DELTAPMIN_B = 9.001
        PMAX_B = 200.0
        pc = pLSO + DELTAPMIN_B
        u = float(np.clip(
            (pc**-0.5 - p**-0.5) / (pc**-0.5 - (PMAX_B + pc)**-0.5),
            0.0, 1.0,
        ))
        EMAX_B = 0.9
        w = float(np.clip(e / EMAX_B, 0.0, 1.0))
        z = self._z_of_a(a_abs)
        return u, w, z

    def _in_region_A(self, a_abs: float, p: float, e: float) -> bool:
        pLSO = float(_separatrix_numpy(a_abs, e, 1.0))
        return (p - pLSO) <= DELTAPMAX

    # ------------------------------------------------------------------
    # Amplitude evaluation
    # ------------------------------------------------------------------

    def _eval_block(
        self, a_abs: float, p_arr: np.ndarray, e_arr: np.ndarray,
    ) -> np.ndarray:
        """Evaluate amplitudes at all trajectory points.  Returns (N, n_modes)."""
        from scipy.interpolate import bisplev

        ad = self.amp_data
        N = len(p_arr)
        n_modes = ad.n_modes
        amps = np.zeros((N, n_modes), dtype=complex)

        for it in range(N):
            p, e = float(p_arr[it]), float(e_arr[it])
            if not np.isfinite(p) or not np.isfinite(e) or p <= 0:
                continue

            in_A = self._in_region_A(a_abs, p, e)
            if in_A:
                u, w, z = self._forward_map_A(a_abs, p, e)
                coeffs = ad.coeffs_A
                u_kn, w_kn, z_kn = ad.u_knots_A, ad.w_knots_A, ad.z_knots_A
            else:
                if ad.coeffs_B is None:
                    continue
                u, w, z = self._forward_map_B(a_abs, p, e)
                coeffs = ad.coeffs_B
                u_kn, w_kn, z_kn = ad.u_knots_B, ad.w_knots_B, ad.z_knots_B

            n_z = len(z_kn)
            iz = int(np.clip(np.searchsorted(z_kn, z) - 1, 0, n_z - 2))
            z_lo, z_hi = z_kn[iz], z_kn[iz + 1]
            alpha = float(np.clip((z - z_lo) / (z_hi - z_lo + 1e-30), 0.0, 1.0))

            # FEW stores coefficients in (w, u) axis order; bisplev first arg
            # is the query in the first axis direction (w), second is u.
            w_pt = np.array([w])
            u_pt = np.array([u])
            for iz2, fac in [(iz, 1.0 - alpha), (iz + 1, alpha)]:
                if fac < 1e-10:
                    continue
                c_block = coeffs[iz2]   # (n_modes, 2, n_coeffs)
                for im in range(n_modes):
                    vr = float(bisplev(w_pt, u_pt, (w_kn, u_kn, c_block[im, 0], 3, 3)))
                    vi = float(bisplev(w_pt, u_pt, (w_kn, u_kn, c_block[im, 1], 3, 3)))
                    amps[it, im] += complex(vr, vi) * fac

        return amps

    # ------------------------------------------------------------------
    # Convenience properties (forwarded from amp_data)
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        return self.amp_data.n_modes

    @property
    def l_arr(self) -> np.ndarray:
        return self.amp_data.l_arr

    @property
    def m_arr(self) -> np.ndarray:
        return self.amp_data.m_arr

    @property
    def k_arr(self) -> np.ndarray:
        return self.amp_data.k_arr

    @property
    def n_arr(self) -> np.ndarray:
        return self.amp_data.n_arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        a: float,
        p: np.ndarray,
        e: np.ndarray,
        x: float = 1.0,
        specific_modes: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        r"""Evaluate Teukolsky mode amplitudes along a trajectory.

        Parameters
        ----------
        a : float
            Dimensionless BH spin.
        p, e : np.ndarray, shape (N,)
            Semi-latus rectum and eccentricity.
        x : float
            Inclination cosine (±1).
        specific_modes : sequence of int, optional
            Mode indices to evaluate.  Default: all modes.

        Returns
        -------
        amps : np.ndarray, shape (N, n_modes) complex
        """
        a_abs = float(np.abs(a))
        p_arr = np.asarray(p, dtype=float)
        e_arr = np.asarray(e, dtype=float)
        amps = self._eval_block(a_abs, p_arr, e_arr)
        if specific_modes is not None:
            amps = amps[:, specific_modes]
        return amps

    def get_mode_power(
        self,
        a: float,
        p: np.ndarray,
        e: np.ndarray,
        x: float = 1.0,
    ) -> np.ndarray:
        """Return mean |A|² per mode, shape (n_modes,)."""
        amps = self.evaluate(a, p, e, x)
        return np.mean(np.abs(amps)**2, axis=0)

    def select_modes(
        self,
        a: float,
        p: np.ndarray,
        e: np.ndarray,
        threshold: float = 1e-5,
        x: float = 1.0,
    ) -> np.ndarray:
        """Return indices of modes with relative power ≥ ``threshold × max``.

        Returns
        -------
        indices : np.ndarray of int
        """
        power = self.get_mode_power(a, p, e, x)
        if power.max() == 0:
            return np.arange(len(power))
        return np.where(power >= threshold * power.max())[0]

"""Teukolsky mode amplitude interpolation from FEW precomputed grids.

Two backends are available:

:class:`AmplitudeInterpolator`
    Legacy numpy/scipy backend.  Evaluates the raw bisplev B-spline
    coefficients stored in the HDF5 file using ``scipy.interpolate.bisplev``
    in a Python loop over trajectory points and modes.  CPU-only, not
    JAX-traceable, kept for validation and backward compatibility.

:class:`JAXAmplitudeInterpolator`
    Fully JAX-native backend.  Uses :class:`~fewtrax.utils.splines.BatchedTricubicSplineE3`
    splines built by :func:`~fewtrax.data.loader.load_amplitude_data_jax`.
    A single ``einsum`` replaces the ``N_traj × N_modes`` bisplev loop,
    evaluating all modes in parallel.  Supports :func:`jax.jit`,
    :func:`jax.grad`, and :func:`jax.vmap` for batched waveform generation.

Design notes (legacy backend)
------------------------------
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
import jax
import jax.numpy as jnp
import equinox as eqx

from fewtrax.utils.coordinates import (
    ALPHA_AMP, BETA_AMP, DELTAPMIN, DELTAPMAX,
    kerrecceq_forward_map_A, kerrecceq_forward_map_B,
)
from fewtrax.utils.geodesic import get_separatrix_fast
from fewtrax.data.loader import AmplitudeData, AmplitudeDataJAX, _separatrix_numpy


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


# ---------------------------------------------------------------------------
# JAX-native amplitude interpolator
# ---------------------------------------------------------------------------

class JAXAmplitudeInterpolator(eqx.Module):
    r"""Fully JAX-native Teukolsky amplitude evaluator.

    Wraps an :class:`~fewtrax.data.loader.AmplitudeDataJAX` container and
    provides a :meth:`__call__` that evaluates complex mode amplitudes
    :math:`A_{\ell mkn}(a, p, e)` for all modes in a single fused
    ``einsum``, replacing the ``N_{\rm traj} \times N_{\rm modes}``
    ``scipy.bisplev`` Python loop in :class:`AmplitudeInterpolator`.

    All public methods are :func:`jax.jit`-able, :func:`jax.vmap`-able, and
    support :func:`jax.grad` / :func:`jax.jacfwd`.

    Parameters
    ----------
    amp_data : AmplitudeDataJAX
        Pre-built JAX amplitude container; construct with
        :func:`~fewtrax.data.loader.load_amplitude_data_jax`.

    Examples
    --------
    Single-point evaluation::

        interp = JAXAmplitudeInterpolator(amp_data)
        amps = interp(a=0.9, p=10.0, e=0.3)   # (n_modes,) complex128

    Trajectory evaluation (vectorised over time steps)::

        amps_traj = interp.evaluate_trajectory(a=0.9, p=p_arr, e=e_arr)
        # (N_traj, n_modes) complex128

    Batched waveform generation (vmap over initial conditions)::

        batch_eval = jax.vmap(
            lambda a, p0, e0: interp.evaluate_trajectory(a, traj_p(a, p0, e0), ...)
        )
    """

    amp_data: AmplitudeDataJAX

    def __init__(self, amp_data: AmplitudeDataJAX):
        self.amp_data = amp_data

    # ------------------------------------------------------------------
    # Convenience forwarded properties
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        return self.amp_data.n_modes

    @property
    def l_arr(self) -> jnp.ndarray:
        return self.amp_data.l_arr

    @property
    def m_arr(self) -> jnp.ndarray:
        return self.amp_data.m_arr

    @property
    def k_arr(self) -> jnp.ndarray:
        return self.amp_data.k_arr

    @property
    def n_arr(self) -> jnp.ndarray:
        return self.amp_data.n_arr

    # ------------------------------------------------------------------
    # Single-point evaluation
    # ------------------------------------------------------------------

    def __call__(
        self,
        a: float,
        p: float,
        e: float,
        pLSO: Optional[float] = None,
    ) -> jnp.ndarray:
        r"""Evaluate complex amplitudes at a single ``(a, p, e)`` point.

        Parameters
        ----------
        a : float
            Dimensionless BH spin (signed; prograde for a > 0).
        p, e : float
            Semi-latus rectum and eccentricity.
        pLSO : float, optional
            Pre-computed separatrix :math:`p_{\rm sep}(|a|, e)`.  When
            provided the internal separatrix computation is skipped (saves
            ~25 bisection iterations per call — useful inside the ODE where
            pLSO is already known).

        Returns
        -------
        jnp.ndarray, shape ``(n_modes,)`` complex128
            Complex mode amplitudes :math:`A_{\ell mkn}`.
        """
        ad = self.amp_data
        a_abs = jnp.abs(a)
        x_in = jnp.where(a >= 0.0, 1.0, -1.0)

        if pLSO is None:
            pLSO = get_separatrix_fast(a_abs, e, x_in)

        in_A = p <= pLSO + DELTAPMAX

        # Compute coordinates for both regions (JAX always evaluates both
        # branches; jnp.where selects the correct result for each region).
        u_A, w_A, z_A = kerrecceq_forward_map_A(
            a_abs, p, e, pLSO, ALPHA_AMP, BETA_AMP
        )
        u_B, w_B, z_B = kerrecceq_forward_map_B(
            a_abs, p, e, pLSO, is_flux=False
        )

        # Clamp coordinates to [0, 1] (same as extrap=False in interpax)
        u_A = jnp.clip(u_A, 0.0, 1.0)
        w_A = jnp.clip(w_A, 0.0, 1.0)
        z_A = jnp.clip(z_A, 0.0, 1.0)
        u_B = jnp.clip(u_B, 0.0, 1.0)
        w_B = jnp.clip(w_B, 0.0, 1.0)
        z_B = jnp.clip(z_B, 0.0, 1.0)

        # Evaluate Region A splines → (n_modes,)
        amp_A = (
            ad.spline_A_real(u_A, w_A, z_A)
            + 1j * ad.spline_A_imag(u_A, w_A, z_A)
        )

        if ad._has_B:
            amp_B = (
                ad.spline_B_real(u_B, w_B, z_B)
                + 1j * ad.spline_B_imag(u_B, w_B, z_B)
            )
            return jnp.where(in_A, amp_A, amp_B)

        return amp_A

    # ------------------------------------------------------------------
    # Trajectory evaluation
    # ------------------------------------------------------------------

    def evaluate_trajectory(
        self,
        a: float,
        p: jnp.ndarray,
        e: jnp.ndarray,
        pLSO: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        r"""Evaluate amplitudes along a trajectory.

        Uses :func:`jax.vmap` over trajectory points, so the spline
        coefficients are read once and the polynomial evaluation is
        vectorised across all time steps.

        Parameters
        ----------
        a : float
            Dimensionless spin (fixed along a trajectory).
        p, e : jnp.ndarray, shape ``(N_traj,)``
            Semi-latus rectum and eccentricity at each ODE step.
        pLSO : jnp.ndarray, shape ``(N_traj,)``, optional
            Pre-computed separatrix at each step.  When omitted it is
            computed internally via :func:`~fewtrax.utils.geodesic.get_separatrix_fast`.

        Returns
        -------
        jnp.ndarray, shape ``(N_traj, n_modes)`` complex128
            Mode amplitudes at each trajectory point.
        """
        if pLSO is None:
            a_abs = jnp.abs(a)
            x_in = jnp.where(a >= 0.0, 1.0, -1.0)
            pLSO = jax.vmap(
                lambda pi, ei: get_separatrix_fast(a_abs, ei, x_in)
            )(p, e)

        return jax.vmap(
            lambda pi, ei, pli: self(a, pi, ei, pLSO=pli)
        )(p, e, pLSO)

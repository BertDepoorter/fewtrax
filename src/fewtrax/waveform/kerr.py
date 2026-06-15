"""High-level KerrEccentricEquatorial waveform generator.

:class:`KerrEccentricEquatorialWaveform` is the main entry point for
generating EMRI gravitational waveforms in fewtrax.  It mirrors the
API of ``few.waveform.FastKerrEccentricEquatorialFlux`` while using
JAX-accelerated trajectory integration and mode summation.

Pipeline
--------
1. **Data loading**: on construction, the FEW HDF5 data files are read
   once and JAX-compatible interpolators are built.
2. **Trajectory**: :class:`~fewtrax.trajectory.EMRIInspiral` integrates
   the adiabatic ODE for :math:`(p, e, \\Phi_\\phi, \\Phi_\\theta, \\Phi_r)`.
3. **Mode selection**: modes are filtered by their relative power
   contribution (threshold configurable).
4. **Amplitude evaluation**: :class:`~fewtrax.amplitude.JAXAmplitudeInterpolator`
   evaluates :math:`A_{\\ell m k n}` at each trajectory point.
5. **Harmonics**: :func:`~fewtrax.utils.harmonics.get_ylms_for_modes`
   computes :math:`{}_{-2}Y_{\\ell m}(\\theta, \\phi)`.
6. **Summation**: :class:`~fewtrax.summation.ModeSum` upsamples the sparse
   trajectory to the requested ``dt`` and sums the modes.

Frame transformations
---------------------
The waveform is produced in the *source frame* (sky direction
:math:`(\\theta_S, \\phi_S)`, spin direction :math:`(\\theta_K, \\phi_K)`).
The :meth:`__call__` signature matches the FEW convention, including
the sky-location and spin-orientation angles.

JAX compatibility
-----------------
*  :meth:`__call__` is **not** JIT-compiled as a whole (it mixes numpy
   amplitude evaluation with JAX trajectory/summation).  Use
   :meth:`generate_sparse` to obtain the sparse trajectory and
   amplitudes, then call :func:`~fewtrax.summation.direct_mode_sum`
   directly inside ``jax.jit``.
*  The trajectory is fully differentiable with respect to
   :math:`(p_0, e_0, \\Phi_{\\phi 0}, \\Phi_{\\theta 0}, \\Phi_{r 0})`.

Examples
--------
>>> from fewtrax import KerrEccentricEquatorialWaveform
>>> wf = KerrEccentricEquatorialWaveform(data_dir="/path/to/few/data")
>>> hp, hx = wf(
...     M=1e6, mu=10.0, a=0.3,
...     p0=10.0, e0=0.4, x0=1.0,
...     dist=1.0,
...     qS=0.2, phiS=0.2, qK=0.8, phiK=0.8,
...     Phi_phi0=1.0, Phi_theta0=2.0, Phi_r0=3.0,
...     T=0.1, dt=10.0,
... )
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import jax.numpy as jnp

from fewtrax.utils.constants import (
    MTSUN_SI, YEAR_SI, GPC_SI, G_SI, C_SI, MSUN_SI,
)
from fewtrax.utils.harmonics import get_ylms_for_modes
from fewtrax.data.loader import (
    load_flux_data, load_amplitude_data, load_amplitude_data_jax,
    FluxData, AmplitudeData,
)
from fewtrax.trajectory.inspiral import EMRIInspiral
from fewtrax.amplitude.interp import JAXAmplitudeInterpolator
from fewtrax.summation.modes import ModeSum
from fewtrax.summation.tf_sum import direct_tf_sum
from fewtrax.utils.tf_bases.base import TFGrid
from fewtrax.utils.tf_bases.wdm import WDMGrid, default_grid
from fewtrax.utils.tf_bases.sft import SFTGrid, default_sft_grid

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame-transformation helpers  (mirror FEW conventions exactly)
# ---------------------------------------------------------------------------

def _get_viewing_angles(
    qS: float, phiS: float, qK: float, phiK: float
) -> tuple[float, float]:
    """Compute source-frame observer angles from SSB sky/spin angles.

    Mirrors ``GenerateEMRIWaveform._get_viewing_angles`` in FEW.

    Parameters
    ----------
    qS, phiS : float
        Sky location polar/azimuthal angles [rad] in the SSB ecliptic frame.
    qK, phiK : float
        BH spin polar/azimuthal angles [rad] in the SSB ecliptic frame.

    Returns
    -------
    theta : float
        Source-frame polar angle of the observer (angle between source
        direction and spin axis) [rad].
    phi : float
        Source-frame azimuthal angle, fixed to ``-π/2`` by definition.
    """
    R = np.array([
        np.sin(qS) * np.cos(phiS),
        np.sin(qS) * np.sin(phiS),
        np.cos(qS),
    ])
    S = np.array([
        np.sin(qK) * np.cos(phiK),
        np.sin(qK) * np.sin(phiK),
        np.cos(qK),
    ])
    theta = float(np.arccos(np.clip(-np.dot(R, S), -1.0, 1.0)))
    phi = -np.pi / 2.0
    return theta, phi


def _to_ssb_frame(
    hp: jnp.ndarray,
    hx: jnp.ndarray,
    qS: float,
    phiS: float,
    qK: float,
    phiK: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate (h+, h×) from the source frame to the SSB frame.

    Applies a polarization-angle rotation by ``ψ_ldc``.
    Mirrors ``GenerateEMRIWaveform._to_SSB_frame`` in FEW.
    """
    cqS = np.cos(qS)
    sqS = np.sin(qS)
    cqK = np.cos(qK)
    sqK = np.sin(qK)

    up_ldc = cqS * sqK * np.cos(phiS - phiK) - cqK * sqS
    dw_ldc = sqK * np.sin(phiS - phiK)

    psi_ldc = -jnp.arctan2(up_ldc, dw_ldc)
    c2psi = jnp.cos(2.0 * psi_ldc)
    s2psi = jnp.sin(2.0 * psi_ldc)

    hp_new = c2psi * hp - s2psi * hx
    hx_new = s2psi * hp + c2psi * hx
    return hp_new, hx_new


class KerrEccentricEquatorialWaveform:
    r"""EMRI waveform generator for the KerrEccentricEquatorial model.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the FEW HDF5 data files.
    amp_filename : str
        Amplitude data file name.
    mode_selection_threshold : float
        Relative power threshold for mode selection.  Modes whose
        time-averaged power is below this fraction of the dominant mode
        power are discarded.
    dense_steps : int
        Number of sparse trajectory points used in the ODE integration.
    preload_amplitude : bool
        If True, load the amplitude data file at construction time.
        Set to False to defer loading (useful if amplitude data is
        large and the user only wants to compare trajectory outputs).
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        amp_filename: str = "ZNAmps_l10_m10_n55_DS2Outer.h5",
        mode_selection_threshold: float = 1.0e-5,
        dense_steps: int = 100,
        preload_amplitude: bool = True,
    ):
        self.data_dir = data_dir
        self.amp_filename = amp_filename
        self.mode_selection_threshold = mode_selection_threshold
        self.dense_steps = dense_steps

        log.info("Loading flux data …")
        self._flux_data: FluxData = load_flux_data(data_dir)

        self._amp_data: Optional[AmplitudeData] = None
        self._jax_amp_interp: Optional[JAXAmplitudeInterpolator] = None
        if preload_amplitude:
            self._ensure_amp_loaded()

    def _ensure_amp_loaded(self):
        """Load the amplitude data (mode arrays + JAX interpolator) once."""
        if self._amp_data is None:
            log.info("Loading amplitude data …")
            self._amp_data = load_amplitude_data(
                self.data_dir, filename=self.amp_filename
            )
        if self._jax_amp_interp is None:
            log.info("Loading JAX amplitude data …")
            self._jax_amp_interp = JAXAmplitudeInterpolator(
                load_amplitude_data_jax(self.data_dir, filename=self.amp_filename)
            )

    def generate_sparse(
        self,
        M: float,
        mu: float,
        a: float,
        p0: float,
        e0: float,
        x0: float = 1.0,
        T: float = 1.0,
        dt: float = 10.0,
        Phi_phi0: float = 0.0,
        Phi_theta0: float = 0.0,
        Phi_r0: float = 0.0,
        **traj_kwargs,
    ) -> dict:
        r"""Compute the sparse trajectory and mode amplitudes.

        Returns a dictionary with the sparse trajectory outputs and
        amplitude arrays.  This is the differentiable "inner loop";
        the trajectory components are JAX arrays.

        Parameters
        ----------
        M : float
            Primary BH mass [:math:`M_\odot`].
        mu : float
            Secondary mass [:math:`M_\odot`].
        a : float
            Dimensionless spin parameter.
        p0 : float
            Initial semi-latus rectum [:math:`M`].
        e0 : float
            Initial eccentricity.
        x0 : float
            Cosine of inclination (:math:`\pm 1`).
        T : float
            Observation time [years].
        dt : float
            Waveform sampling interval [s] (used for output).
        Phi_phi0, Phi_theta0, Phi_r0 : float
            Initial orbital phases [rad].
        **traj_kwargs
            Extra keyword arguments forwarded to :class:`EMRIInspiral`.

        Returns
        -------
        dict with keys:
            ``t``, ``p``, ``e``, ``Phi_phi``, ``Phi_theta``, ``Phi_r`` :
                Sparse trajectory arrays.
            ``teuk_modes`` : np.ndarray, shape (N_traj, N_modes), complex
                Mode amplitudes at trajectory points.
            ``l_arr``, ``m_arr``, ``k_arr``, ``n_arr`` :
                Mode index arrays.
            ``mode_inds`` : np.ndarray of int
                Selected mode indices.
        """
        self._ensure_amp_loaded()

        # --- Trajectory ---
        # Request the Dopri8 dense-output phase evaluator (7th/8th-order) so the
        # time-domain summation uses the polynomial phases.  Available in both
        # forward and backward fixed-grid mode; only save_at_steps lacks it.
        want_dense_phase = not traj_kwargs.get("save_at_steps", False)
        traj = EMRIInspiral(self._flux_data)
        traj_out = traj(
            p0=p0, e0=e0, T=T, a=a, x0=x0, dt=dt, M=M, mu=mu,
            Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0,
            dense_steps=self.dense_steps,
            return_dense_phase_fn=want_dense_phase,
            **traj_kwargs,
        )
        if want_dense_phase:
            t, p, e, Phi_phi, Phi_theta, Phi_r, phase_fn = traj_out
        else:
            t, p, e, Phi_phi, Phi_theta, Phi_r = traj_out
            phase_fn = None

        # --- Amplitudes (all modes) + mean-power mode selection (JAX-native) ---
        teuk_modes_all = self._jax_amp_interp.evaluate_trajectory(
            jnp.asarray(a, dtype=jnp.float64), p, e
        )  # (N_traj, n_all_modes)
        mode_inds = self._jax_amp_interp.select_modes_from_amps(
            teuk_modes_all, threshold=self.mode_selection_threshold
        )
        log.info("Selected %d modes (threshold=%.1e)", len(mode_inds), self.mode_selection_threshold)
        teuk_modes = teuk_modes_all[:, mode_inds]

        ad = self._amp_data
        return dict(
            t=t, p=p, e=e, Phi_phi=Phi_phi, Phi_theta=Phi_theta, Phi_r=Phi_r,
            phase_fn=phase_fn,
            teuk_modes=teuk_modes,
            l_arr=ad.l_arr[mode_inds],
            m_arr=ad.m_arr[mode_inds],
            k_arr=ad.k_arr[mode_inds],
            n_arr=ad.n_arr[mode_inds],
            mode_inds=mode_inds,
        )

    def __call__(
        self,
        M: float,
        mu: float,
        a: float,
        p0: float,
        e0: float,
        x0: float = 1.0,
        dist: float = 1.0,
        qS: float = 0.0,
        phiS: float = 0.0,
        qK: float = 0.0,
        phiK: float = 0.0,
        Phi_phi0: float = 0.0,
        Phi_theta0: float = 0.0,
        Phi_r0: float = 0.0,
        T: float = 1.0,
        dt: float = 10.0,
        mode_selection_threshold: Optional[float] = None,
        return_sparse: bool = False,
        return_complex: bool = False,
        domain: str = "time",
        grid: Optional[TFGrid] = None,
        hw: int = 2,
        **kwargs,
    ):
        r"""Generate the EMRI gravitational waveform.

        Parameters
        ----------
        M : float
            Primary BH mass [:math:`M_\odot`].
        mu : float
            Secondary mass [:math:`M_\odot`].
        a : float
            Dimensionless spin parameter, :math:`|a| < 1`.
        p0 : float
            Initial semi-latus rectum [:math:`M`].
        e0 : float
            Initial eccentricity, :math:`0 \le e_0 < 1`.
        x0 : float
            Cosine of inclination.  Only :math:`x_0 = \pm 1` are
            currently supported (equatorial orbits).
        dist : float
            Luminosity distance [Gpc].
        qS, phiS : float
            Sky location angles [rad] (polar, azimuthal).
        qK, phiK : float
            Spin orientation angles [rad] (polar, azimuthal).
        Phi_phi0, Phi_theta0, Phi_r0 : float
            Initial orbital phases [rad].
        T : float
            Observation time [years].
        dt : float
            Waveform sampling interval [s].  Used for the time-domain output
            (``domain="time"``); ignored for TF domains.
        mode_selection_threshold : float, optional
            Override the class-level mode selection threshold.
        return_sparse : bool
            If True, also return the sparse trajectory dictionary as a
            last element of the return value.
        return_complex : bool
            If True (``domain="time"`` only), return the complex strain
            ``h = h₊ + i·h×`` instead of the ``(hp, hx)`` tuple.
        domain : {"time", "wdm", "sft"}
            Output domain.

            * ``"time"`` *(default)*: dense time-domain strain arrays.
            * ``"wdm"``: complex :math:`N_f\times N_t` WDM matrix.
            * ``"sft"``: complex :math:`N_f\times N_t` SFT matrix.

        grid : TFGrid or None
            TF grid descriptor, required for ``domain="wdm"`` or
            ``domain="sft"``.  Pass a :class:`~fewtrax.utils.tf_bases.wdm.WDMGrid`
            or a :class:`~fewtrax.utils.tf_bases.sft.SFTGrid`.  If ``None``,
            a default WDM grid is constructed from ``T`` and ``dt`` for
            ``domain="wdm"``, and a default 1-day SFT grid for
            ``domain="sft"``.
        hw : int
            Frequency half-width for TF deposition (default 2).  Use
            ``hw=3`` for SFT due to sinc spectral leakage.

        Returns
        -------
        When ``domain="time"`` and ``return_complex=False``:
            hp, hx : jnp.ndarray, shape (N,)
        When ``domain="time"`` and ``return_complex=True``:
            h : jnp.ndarray, shape (N,), complex
        When ``domain="wdm"``:
            W : jnp.ndarray, shape (Nf, Nt), complex64
        Any of the above optionally followed by *sparse_dict* if
        ``return_sparse=True``.

        Notes
        -----
        The frame transformations from the source frame to the SSB (Solar
        System Barycentre) frame follow the convention in FEW.  The
        angles :math:`(\theta_S, \phi_S)` are the ecliptic co-latitude
        and longitude of the source, and :math:`(\theta_K, \phi_K)` are
        the polar and azimuthal angles of the BH spin in the SSB frame.
        """
        if domain not in ("time", "wdm", "sft"):
            raise ValueError(f"domain must be 'time', 'wdm', or 'sft', got {domain!r}")

        if mode_selection_threshold is not None:
            _old = self.mode_selection_threshold
            self.mode_selection_threshold = mode_selection_threshold

        # --- Retrograde convention (mirrors FEW's waveform.py) ---
        # For retrograde orbits (x0 < 0) FEW flips the spin orientation and
        # shifts the initial azimuthal phase by π so that the source-frame
        # viewing angles and SSB rotation are computed consistently.
        qK_eff = float(qK)
        phiK_eff = float(phiK)
        Phi_phi0_eff = float(Phi_phi0)
        if x0 < 0.0:
            qK_eff = float(np.pi - qK)
            phiK_eff = float(phiK + np.pi)
            Phi_phi0_eff = float(Phi_phi0 + np.pi)

        sparse = self.generate_sparse(
            M=M, mu=mu, a=a, p0=p0, e0=e0, x0=x0, T=T, dt=dt,
            Phi_phi0=Phi_phi0_eff,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            **kwargs,
        )

        if mode_selection_threshold is not None:
            self.mode_selection_threshold = _old

        # --- Source-frame observer angles (FEW convention) ---
        theta_obs, phi_obs = _get_viewing_angles(
            float(qS), float(phiS), qK_eff, phiK_eff
        )

        l_arr = sparse["l_arr"]
        m_arr = sparse["m_arr"]
        k_arr = sparse["k_arr"]
        n_arr = sparse["n_arr"]

        ylms_pos, ylms_neg = get_ylms_for_modes(
            l_arr, m_arr, theta_obs, phi_obs
        )

        # =================================================================
        # TF domain  (WDM or SFT — identical call, grid selects the basis)
        # =================================================================
        if domain in ("wdm", "sft"):
            T_s = float(np.asarray(sparse["t"])[np.isfinite(np.asarray(sparse["t"]))][-1])
            if grid is not None:
                tf_grid = grid
            elif domain == "wdm":
                tf_grid = default_grid(T_s)
            else:
                tf_grid = default_sft_grid(T_s)
            log.info("TF generation (%s): %s", domain, tf_grid)

            W = direct_tf_sum(
                t_traj=np.asarray(sparse["t"]),
                teuk_modes=np.asarray(sparse["teuk_modes"]),
                Phi_phi=np.asarray(sparse["Phi_phi"]),
                Phi_theta=np.asarray(sparse["Phi_theta"]),
                Phi_r=np.asarray(sparse["Phi_r"]),
                l_arr=l_arr,
                m_arr=m_arr,
                k_arr=k_arr,
                n_arr=n_arr,
                ylms_pos=ylms_pos,
                ylms_neg=ylms_neg,
                a=a, M=M, mu=mu, x0=x0,
                grid=tf_grid,
                dist=dist,
                hw=hw,
                p_traj=np.asarray(sparse["p"]),
                e_traj=np.asarray(sparse["e"]),
            )

            # Apply SSB-frame polarisation rotation in WDM domain
            # (a pixel-wise complex rotation by 2ψ_ldc)
            cqS, sqS = np.cos(qS), np.sin(qS)
            cqK_e, sqK_e = np.cos(qK_eff), np.sin(qK_eff)
            up_ldc = cqS * sqK_e * np.cos(phiS - phiK_eff) - cqK_e * sqS
            dw_ldc = sqK_e * np.sin(phiS - phiK_eff)
            psi_ldc = -jnp.arctan2(up_ldc, dw_ldc)
            c2psi = jnp.cos(2.0 * psi_ldc)
            s2psi = jnp.sin(2.0 * psi_ldc)
            # W = W_plus + i W_cross;  rotation: hp' = c2ψ hp - s2ψ hx
            #                                     hx' = s2ψ hp + c2ψ hx
            W_plus  = jnp.real(W)
            W_cross = -jnp.imag(W)
            W_plus_rot  = c2psi * W_plus  - s2psi * W_cross
            W_cross_rot = s2psi * W_plus  + c2psi * W_cross
            W_out = W_plus_rot - 1j * W_cross_rot

            if return_sparse:
                return W_out, sparse
            return W_out

        # =================================================================
        # Time domain
        # =================================================================
        summer = ModeSum(
            l_arr, m_arr, k_arr, n_arr,
            ylms_pos, ylms_neg,
            dist=dist, M=M, mu=mu,
        )

        t_out, h = summer(
            sparse["t"],
            sparse["teuk_modes"],
            sparse["Phi_phi"],
            sparse["Phi_theta"],
            sparse["Phi_r"],
            dt=dt,
            phase_fn=sparse.get("phase_fn"),
        )

        # Source-frame sign convention (FEW rotates by π after summation)
        h = h * (-1.0)

        hp = jnp.real(h)
        hx = -jnp.imag(h)

        # Rotate from source frame to SSB frame
        hp, hx = _to_ssb_frame(hp, hx, float(qS), float(phiS), qK_eff, phiK_eff)

        if return_complex:
            h_out = hp + 1j * hx
            if return_sparse:
                return h_out, sparse
            return h_out

        if return_sparse:
            return hp, hx, sparse
        return hp, hx

    def get_harmonic_track(
        self,
        l: int,
        m: int,
        k: int,
        n: int,
        M: float,
        mu: float,
        a: float,
        p0: float,
        e0: float,
        T: float,
        x0: float = 1.0,
        backward: bool = False,
        e_f: Optional[float] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Compute the instantaneous frequency track of a single harmonic mode.

        The instantaneous GW frequency of mode :math:`(\ell, m, k, n)` is:

        .. math::

            f_{\ell m k n}(t) =
                \frac{1}{2\pi}
                \left|
                    m \Omega_\phi(t) + k \Omega_\theta(t) + n \Omega_r(t)
                \right|

        where :math:`\Omega_i` are the fundamental frequencies in
        Boyer-Lindquist coordinate time evaluated along the trajectory.

        Parameters
        ----------
        l, m, k, n : int
            Mode indices.
        M, mu, a, p0, e0, T, x0 : as for :meth:`__call__`.
        backward : bool
            If ``True``, integrate backward from the separatrix.  The
            returned time axis ``t`` is time before plunge
            (:math:`\tau = T_{\rm plunge} - t`), so ``t[0] = 0`` is at
            plunge and ``t[-1]`` is ``T`` years earlier.  Frequency
            decreases as ``t`` increases.
        e_f : float, optional
            Eccentricity at plunge, required when ``backward=True``.
            Obtain this from a forward run:
            ``e_f = float(e_arr[jnp.isfinite(e_arr)][-1])``.

        Returns
        -------
        t : jnp.ndarray
            Time stamps [s] (forward: from start; backward: time before plunge).
        f : jnp.ndarray
            Instantaneous frequency [Hz], always non-negative.
        """
        traj = EMRIInspiral(self._flux_data)
        return traj.get_frequency_track(
            p0=p0, e0=e0, T=T, M=M, mu=mu, a=a, x0=x0,
            l=l, m=m, k=k, n=n,
            dense_steps=self.dense_steps,
            backward=backward, e_f=e_f,
        )

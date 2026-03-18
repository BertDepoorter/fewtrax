"""EMRI orbital trajectory integration via diffrax.

The trajectory ODE integrates

.. math::

    \\frac{dp}{dt} &= \\left(\\frac{\\partial E}{\\partial p}\\right)^{-1}_{\\rm eff}\\,\\dot{E}_{\\rm GR} + \\cdots \\\\
    \\frac{de}{dt} &= \\cdots \\\\
    \\frac{d\\Phi_\\phi}{dt} &= \\Omega_\\phi(a, p, e) \\\\
    \\frac{d\\Phi_\\theta}{dt} &= \\Omega_\\theta(a, p, e) \\\\
    \\frac{d\\Phi_r}{dt} &= \\Omega_r(a, p, e)

where time is in units of :math:`M`.

The radiation-reaction terms are obtained by interpolating the PN-normalised
:math:`\\dot{E}` and :math:`\\dot{L}` grids from :mod:`fewtrax.data.loader`
and converting them to :math:`\\dot{p}`, :math:`\\dot{e}` via the
Jacobian :math:`\\partial(E,L)/\\partial(p,e)` computed analytically with
:func:`jax.jacfwd`.

Solver: :class:`diffrax.Tsit5` (4th/5th order Runge-Kutta, efficient for
smooth problems) with :class:`diffrax.PIDController` adaptive step-size
control.  The system is stopped when the orbit approaches the separatrix.

The ODE is fully JIT-compilable and differentiable with respect to the
initial conditions :math:`(p_0, e_0)` and the BH spin :math:`a`.

To vmap over a batch of initial conditions::

    from jax import vmap
    batch_traj = vmap(traj)(p0_arr, e0_arr)

Backward integration
--------------------
Setting ``backward=True`` integrates the time-reversed ODE, starting at the
separatrix and moving outward.  This anchors the track to the plunge rather
than to uncertain initial conditions, and is more stable near merger because
the ODE moves *away* from the separatrix.

The backward ODE is simply the negation of the forward ODE:

.. math::

    \\frac{d y}{d\\tau} = -\\frac{d y}{dt}

where :math:`\\tau = T_{\\rm plunge} - t` is time before plunge.  Initial
conditions are set to :math:`(p_{\\rm sep}(e_f, a) + \\epsilon, e_f, 0, 0, 0)`
and the integration runs for duration :math:`T` years.

"""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx

from fewtrax.utils.constants import MTSUN_SI, YEAR_SI
from fewtrax.utils.geodesic import get_separatrix, get_fundamental_frequencies
from fewtrax.utils.coordinates import kerrecceq_forward_map, DELTAPMIN
from fewtrax.data.loader import FluxData, _pdot_PN_jax, _edot_PN_jax

SEPARATRIX_BUFFER: float = 2.0 * DELTAPMIN


class EMRIInspiral(eqx.Module):
    r"""Adiabatic EMRI trajectory integrator (JIT/vmap-compatible).

    Integrates the adiabatic inspiral ODE using :class:`diffrax.Tsit5`
    with adaptive step-size control.  The radiation-reaction force
    (:math:`\dot{p}`, :math:`\dot{e}`) is computed from the FEW flux
    tables using the **pex convention** (matching FEW):

    1. Interpolate the pre-computed ratios
       :math:`r_p = \dot{p}_{\rm GR}/\dot{p}_{\rm PN}` and
       :math:`r_e = \dot{e}_{\rm GR}/\dot{e}_{\rm PN}`.
    2. Multiply by the separatrix-dependent PN functions
       :math:`\dot{p}_{\rm PN}(p, e, r_{\rm ISCO}, p_{\rm sep})` and
       :math:`\dot{e}_{\rm PN}`.

    This avoids the runtime Jacobian inversion that was previously
    needed in the ELQ convention, improving both speed and accuracy.

    Parameters
    ----------
    flux_data : FluxData
        Pre-loaded flux interpolators (pex convention).
    a : float
        Dimensionless BH spin.
    x0 : float
        Inclination cosine (+1 prograde, -1 retrograde).
    """

    flux_data: FluxData
    a: jnp.ndarray
    x0: float

    def __init__(self, flux_data: FluxData, a: float, x0: float = 1.0):
        self.flux_data = flux_data
        self.a = jnp.asarray(a, dtype=jnp.float64)
        self.x0 = float(x0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _x_sign(self):
        """Sign of inclination (±1)."""
        ax = jnp.sign(self.a * self.x0)
        return jnp.where(ax == 0.0, 1.0, ax)

    def _flux_pex(self, p: float, e: float) -> tuple[float, float]:
        r"""Return physical :math:`(\dot{p}, \dot{e})` from the pex spline tables.

        Evaluates the pre-computed pex ratios :math:`\dot{p}/\dot{p}_{\rm PN}`
        and :math:`\dot{e}/\dot{e}_{\rm PN}`, then multiplies by the
        separatrix-dependent PN functions.  Returns **negative** values for
        :math:`\dot{p}` (inspiraling orbit loses semi-latus rectum).
        """
        a_abs = jnp.abs(self.a)
        x_in = self._x_sign()
        u, w, z, in_A = kerrecceq_forward_map(a_abs, p, e, kind="flux")

        rp_A = self.flux_data.pdot_A(u, w, z)
        re_A = self.flux_data.edot_A(u, w, z)
        rp_B = self.flux_data.pdot_B(u, w, z)
        re_B = self.flux_data.edot_B(u, w, z)

        rp = jnp.where(in_A, rp_A, rp_B)
        re = jnp.where(in_A, re_A, re_B)

        # Separatrix-dependent PN functions
        r_isco = get_separatrix(a_abs, jnp.zeros_like(e), x_in)
        p_sep = get_separatrix(a_abs, e, x_in)
        pdot_pn = _pdot_PN_jax(p, e, r_isco, p_sep)
        edot_pn = _edot_PN_jax(p, e, r_isco, p_sep)

        # Negative: orbit is inspiraling (p decreases, e typically decreases)
        pdot = -rp * pdot_pn
        edot = -re * edot_pn
        return pdot, edot

    def _ode_rhs(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        r"""ODE right-hand side: :math:`dy/dt` for :math:`y=(p,e,\Phi_\phi,\Phi_\theta,\Phi_r)`.

        ``args`` carries ``mu_over_M = mu / M`` so the flux-table derivatives
        (normalised to μ=M=1) are properly scaled to the physical mass ratio.
        The orbital phases evolve at the Boyer-Lindquist frequencies and are
        *not* affected by the mass ratio.
        """
        mu_over_M = args
        p, e = y[0], y[1]
        a_abs = jnp.abs(self.a)
        x_in = self._x_sign()

        pdot, edot = self._flux_pex(p, e)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(a_abs, p, e, x_in)
        # pdot/edot come from flux tables normalised to μ=M=1; scale to physical ratio
        return jnp.array([pdot * mu_over_M, edot * mu_over_M,
                          Omega_phi, Omega_theta, Omega_r])

    # ------------------------------------------------------------------
    # JIT-compiled ODE solve (inner loop)
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def _solve(
        self,
        y0: jnp.ndarray,
        t_save: jnp.ndarray,
        T_geo: jnp.ndarray,
        mu_over_M: jnp.ndarray,
        atol: float,
        rtol: float,
        max_steps: int,
    ):
        """JIT-compiled diffrax solve.  All shape-determining args are static."""
        def _event_cond(t, y, args, **kwargs):
            p, e = y[0], y[1]
            p_sep = get_separatrix(jnp.abs(self.a), e, self._x_sign())
            return p < p_sep + SEPARATRIX_BUFFER

        return diffrax.diffeqsolve(
            diffrax.ODETerm(self._ode_rhs),
            diffrax.Tsit5(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            event=diffrax.Event(_event_cond),
            args=mu_over_M,
        )

    @eqx.filter_jit
    def _solve_backward(
        self,
        y0: jnp.ndarray,
        t_save: jnp.ndarray,
        T_geo: jnp.ndarray,
        mu_over_M: jnp.ndarray,
        atol: float,
        rtol: float,
        max_steps: int,
    ):
        r"""JIT-compiled backward diffrax solve.

        Integrates the time-reversed ODE

        .. math::

            \frac{dy}{d\tau} = -\frac{dy}{dt}

        where :math:`\tau = T_{\rm plunge} - t` is time before plunge.
        There is no separatrix event condition: the integration moves
        *away* from the separatrix, so no termination is needed.

        Parameters are identical to :meth:`_solve`.
        """
        return diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -self._ode_rhs(t, y, args)),
            diffrax.Tsit5(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            args=mu_over_M,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        p0: float,
        e0: float,
        T: float,
        dt: float = 10.0,
        M: float = 1e6,
        mu: float = 10.0,
        Phi_phi0: float = 0.0,
        Phi_theta0: float = 0.0,
        Phi_r0: float = 0.0,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        max_steps: int = 4096,
        dense_steps: int = 100,
        backward: bool = False,
        e_f: Optional[float] = None,
    ) -> tuple[jnp.ndarray, ...]:
        r"""Integrate the EMRI inspiral trajectory.

        Parameters
        ----------
        p0, e0 : float
            Initial orbital parameters (forward mode).  Ignored in backward
            mode; the starting point is determined from ``e_f`` instead.
        T : float
            Observation time [years].
        dt : float
            Waveform sampling interval [s] (not the ODE step size).
        M, mu : float
            Primary and secondary masses [:math:`M_\odot`].
        Phi_phi0, Phi_theta0, Phi_r0 : float
            Initial orbital phases [rad].  In backward mode these are the
            phases at plunge (:math:`\tau = 0`); they are often left at zero
            since only the instantaneous frequency matters for track generation.
        atol, rtol : float
            Adaptive solver tolerances.
        max_steps : int
            Maximum number of ODE internal steps.  The Tsit5 solver
            typically uses ≪100 steps for a year-long EMRI inspiral;
            the default of 4096 accommodates edge cases without
            significant runtime cost (execution scales with actual
            step count, not max_steps).
        dense_steps : int
            Number of output trajectory points.
        backward : bool
            If ``True``, integrate the time-reversed ODE starting from the
            separatrix.  The output time axis represents
            :math:`\tau = T_{\rm plunge} - t` (time before plunge), so
            ``t[0] = 0`` is at plunge and ``t[-1]`` is ``T`` years earlier.
            In backward mode ``p`` and ``e`` *increase* along the output
            arrays (moving away from the separatrix), and the phases
            *decrease* (since :math:`d\Phi/d\tau < 0`).
        e_f : float, optional
            Eccentricity at plunge, required when ``backward=True``.
            The corresponding semi-latus rectum is set to
            :math:`p_{\rm sep}(e_f, a) + \epsilon` automatically.
            To obtain a consistent value, run a forward integration first
            and read off ``e`` at the last valid (non-NaN) trajectory point.

        Returns
        -------
        t, p, e, Phi_phi, Phi_theta, Phi_r : jnp.ndarray, shape (dense_steps,)
            Trajectory arrays.  ``t`` is in seconds; phases in radians.
            In backward mode ``t`` is time before plunge (:math:`\tau`).
        """
        M_s = (M + mu) * MTSUN_SI                 # total mass in seconds (matches FEW convention)
        T_s = T * YEAR_SI                         # observation time in seconds
        T_geo = jnp.asarray(T_s / M_s)           # geometric time  (units of M)
        mu_over_M = jnp.asarray(M * mu / (M + mu) ** 2)  # symmetric mass ratio eta (matches FEW)

        t_save = jnp.linspace(jnp.zeros((), jnp.float64), T_geo, dense_steps)

        if backward:
            if e_f is None:
                raise ValueError(
                    "e_f must be provided when backward=True.  "
                    "Run a forward integration first and read off e at the "
                    "last valid trajectory point."
                )
            e_f_ = jnp.asarray(float(e_f), dtype=jnp.float64)
            p_sep = get_separatrix(jnp.abs(self.a), e_f_, self._x_sign())
            p_start = p_sep + SEPARATRIX_BUFFER
            y0 = jnp.array(
                [p_start, e_f_, Phi_phi0, Phi_theta0, Phi_r0], dtype=jnp.float64
            )
            sol = self._solve_backward(y0, t_save, T_geo, mu_over_M, atol, rtol, max_steps)
        else:
            y0 = jnp.array([p0, e0, Phi_phi0, Phi_theta0, Phi_r0], dtype=jnp.float64)
            sol = self._solve(y0, t_save, T_geo, mu_over_M, atol, rtol, max_steps)

        ys = sol.ys          # (dense_steps, 5)
        t_arr = sol.ts       # (dense_steps,)

        t_s = t_arr * M_s
        p_arr = ys[:, 0]
        e_arr = ys[:, 1]
        # Phases are Boyer-Lindquist orbital phases [rad]; no mass-ratio scaling needed
        Phi_phi_arr   = ys[:, 2]
        Phi_theta_arr = ys[:, 3]
        Phi_r_arr     = ys[:, 4]

        return t_s, p_arr, e_arr, Phi_phi_arr, Phi_theta_arr, Phi_r_arr

    def get_frequency_track(
        self,
        p0: float,
        e0: float,
        T: float,
        M: float,
        mu: float,
        l: int, m: int, k: int, n: int,
        dense_steps: int = 100,
        backward: bool = False,
        e_f: Optional[float] = None,
        **kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Compute the instantaneous frequency track of harmonic :math:`(l,m,k,n)`.

        The harmonic frequency is

        .. math::

            f_{mkn}(t) = \frac{|m\,\Omega_\phi + k\,\Omega_\theta + n\,\Omega_r|}{2\pi}

        evaluated along the inspiral trajectory.

        Parameters
        ----------
        p0, e0 : float
            Initial orbital parameters (forward mode; ignored when
            ``backward=True``).
        T, M, mu : float
            Observation time [years], primary and secondary masses.
        l, m, k, n : int
            Harmonic mode indices.
        dense_steps : int
            Number of trajectory output points.
        backward : bool
            If ``True``, integrate backward from the separatrix.  The
            returned time axis is time before plunge (:math:`\tau`), so
            ``t[0] = 0`` is at plunge.  Frequency decreases as ``t``
            increases (moving earlier in the inspiral).
        e_f : float, optional
            Eccentricity at plunge, required when ``backward=True``.

        Returns
        -------
        t : jnp.ndarray, shape (dense_steps,)
            Time [s] (forward: from start; backward: time before plunge).
        f : jnp.ndarray, shape (dense_steps,)
            Instantaneous frequency [Hz], always non-negative.
        """
        t_s, p_arr, e_arr, *_ = self(
            p0=p0, e0=e0, T=T, M=M, mu=mu, dense_steps=dense_steps,
            backward=backward, e_f=e_f, **kwargs
        )
        a_abs = jnp.abs(self.a)
        x_in = self._x_sign()

        def _freq_at_point(pe):
            p_, e_ = pe
            Om_phi, Om_theta, Om_r = get_fundamental_frequencies(a_abs, p_, e_, x_in)
            return jnp.abs(m * Om_phi + k * Om_theta + n * Om_r) / (2.0 * jnp.pi)

        # Physical frequency: convert Mino-time Ω (rad/M) to Hz
        # Ω [rad/M] × (M_s)^{-1} [1/s] / (2π) [already included above]
        M_s = M * MTSUN_SI
        f_arr = jax.vmap(_freq_at_point)(jnp.stack([p_arr, e_arr], axis=1)) / M_s
        return t_s, f_arr


def run_inspiral(
    a: float,
    p0: float,
    e0: float,
    T: float,
    flux_data: FluxData,
    M: float = 1e6,
    mu: float = 10.0,
    dt: float = 10.0,
    x0: float = 1.0,
    Phi_phi0: float = 0.0,
    Phi_theta0: float = 0.0,
    Phi_r0: float = 0.0,
    dense_steps: int = 100,
    backward: bool = False,
    e_f: Optional[float] = None,
    **kwargs,
) -> tuple[jnp.ndarray, ...]:
    r"""Convenience wrapper: integrate an EMRI inspiral trajectory.

    Parameters
    ----------
    a : float
        BH spin parameter.
    p0, e0 : float
        Initial orbital parameters (forward mode; ignored when
        ``backward=True``).
    T : float
        Observation time [years].
    flux_data : FluxData
        Pre-loaded flux data.
    M, mu : float
        Primary and secondary masses [:math:`M_\odot`].
    dt : float
        Waveform sampling interval [s] (not ODE step size).
    x0 : float
        Inclination cosine.
    dense_steps : int
        Number of output trajectory points.
    backward : bool
        If ``True``, integrate the time-reversed ODE from the separatrix.
    e_f : float, optional
        Eccentricity at plunge, required when ``backward=True``.

    Returns
    -------
    t, p, e, Phi_phi, Phi_theta, Phi_r : jnp.ndarray
        Trajectory arrays.  In backward mode ``t`` is time before plunge.
    """
    traj = EMRIInspiral(flux_data, a=a, x0=x0)
    return traj(
        p0=p0, e0=e0, T=T, dt=dt, M=M, mu=mu,
        Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0,
        dense_steps=dense_steps, backward=backward, e_f=e_f, **kwargs,
    )

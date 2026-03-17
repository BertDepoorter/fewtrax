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
from fewtrax.utils.geodesic import (
    get_separatrix,
    get_fundamental_frequencies,
    kerr_geo_energy_equatorial,
    kerr_geo_angular_momentum_equatorial,
)
from fewtrax.utils.coordinates import kerrecceq_forward_map, DELTAPMIN
from fewtrax.data.loader import FluxData, _PN_Edot_jax, _PN_Ldot_jax

SEPARATRIX_BUFFER: float = 2.0 * DELTAPMIN


class EMRIInspiral(eqx.Module):
    r"""Adiabatic EMRI trajectory integrator (JIT/vmap-compatible).

    Integrates the adiabatic inspiral ODE using :class:`diffrax.Tsit5`
    with adaptive step-size control.  The radiation-reaction force
    (:math:`\dot{p}`, :math:`\dot{e}`) is computed from the FEW flux
    tables by:

    1. Interpolating the PN-normalised flux ratios
       :math:`r_E = \dot{E}_{\rm GR}/\dot{E}_{\rm PN}` and
       :math:`r_L = \dot{L}_{\rm GR}/\dot{L}_{\rm PN}`.
    2. Multiplying by the Peters (1964) PN functions.
    3. Applying the Jacobian :math:`\partial(E,L)/\partial(p,e)` via
       :func:`jax.jacfwd` to obtain :math:`(\dot{p}, \dot{e})`.

    Parameters
    ----------
    flux_data : FluxData
        Pre-loaded flux interpolators.
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

    def _flux_EL(self, p: float, e: float) -> tuple[float, float]:
        r"""Return physical :math:`(\dot{E}, \dot{L})` from the spline tables.

        Evaluates the stored PN-normalised ratios and multiplies by the
        leading-order PN functions.  Returns **negative** values
        (energy/angular-momentum is being radiated away).
        """
        a_abs = jnp.abs(self.a)
        x_in = self._x_sign()
        u, w, z, in_A = kerrecceq_forward_map(a_abs, p, e, kind="flux")

        rE_A = self.flux_data.Edot_A(u, w, z)
        rL_A = self.flux_data.Ldot_A(u, w, z)
        rE_B = self.flux_data.Edot_B(u, w, z)
        rL_B = self.flux_data.Ldot_B(u, w, z)

        rE = jnp.where(in_A, rE_A, rE_B)
        rL = jnp.where(in_A, rL_A, rL_B)

        Epn = _PN_Edot_jax(p, e)
        Lpn = _PN_Ldot_jax(p, e)

        # Negative: orbit is inspiraling (losing energy/angular momentum)
        Edot = -rE * Epn
        Ldot = -rL * Lpn
        # For retrograde orbit, Ldot has opposite sign convention
        Ldot = Ldot * x_in
        return Edot, Ldot

    def _ELdot_to_pedot(self, p: float, e: float, Edot: float, Ldot: float):
        r"""Convert :math:`(\dot{E}, \dot{L})` to :math:`(\dot{p}, \dot{e})`.

        Uses :func:`jax.jacfwd` to differentiate the geodesic energy and
        angular momentum functions :math:`E(p,e)`, :math:`L(p,e)`, then
        solves the :math:`2 \times 2` linear system.

        Near-circular limit (``e < 1e-4``): the Jacobian second column
        :math:`\\partial(E,L)/\\partial e \\propto e` vanishes, making the
        system singular.  In that regime we fall back to
        :math:`\\dot{p} = \\dot{E}/(\\partial E/\\partial p)`,
        :math:`\\dot{e} = 0`, which is the physically correct adiabatic
        answer for circular orbits.
        """
        E_CIRC = 1e-4  # threshold below which orbit is treated as circular
        a_abs = jnp.abs(self.a)
        x_in = self._x_sign()

        # Evaluate Jacobian at e_safe to prevent singular matrix.
        # For the circular branch the J columns ∂/∂e → 0, so we clamp e to
        # E_CIRC; the constant floor has zero tangent w.r.t. e for AD purposes.
        e_safe = jnp.maximum(jnp.abs(e), E_CIRC)

        def EL_of_pe(pe):
            p_, e_ = pe[0], pe[1]
            E_ = kerr_geo_energy_equatorial(a_abs, p_, e_, x_in)
            L_ = kerr_geo_angular_momentum_equatorial(a_abs, p_, e_, x_in, E_)
            return jnp.array([E_, L_])

        J = jax.jacfwd(EL_of_pe)(jnp.array([p, e_safe]))   # shape (2, 2)

        # Full Jacobian inversion (valid for e >= E_CIRC)
        rhs = jnp.array([Edot, Ldot])
        pe_dot = jnp.linalg.solve(J, rhs)

        # Circular-orbit fallback: pdot from ∂E/∂p alone, edot = 0
        dEdp = J[0, 0]
        pdot_circ = Edot / jnp.where(jnp.abs(dEdp) > 1e-30, dEdp, 1.0)

        use_circ = e < E_CIRC
        pdot = jnp.where(use_circ, pdot_circ, pe_dot[0])
        edot = jnp.where(use_circ, jnp.zeros_like(e), pe_dot[1])
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

        Edot, Ldot = self._flux_EL(p, e)
        pdot, edot = self._ELdot_to_pedot(p, e, Edot, Ldot)
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
        M_s = M * MTSUN_SI                        # primary mass in seconds
        T_s = T * YEAR_SI                         # observation time in seconds
        T_geo = jnp.asarray(T_s / M_s)           # geometric time  (units of M)
        mu_over_M = jnp.asarray(mu / M)

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

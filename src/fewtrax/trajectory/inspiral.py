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

Solver: :class:`diffrax.Dopri8` (8th-order Dormand-Prince Runge-Kutta,
matching FEW's DOPR853 in order) with :class:`diffrax.PIDController`
adaptive step-size control.  The system is stopped when the orbit approaches
the separatrix.

The ODE is fully JIT-compilable and differentiable with respect to the
initial conditions :math:`(p_0, e_0)`, the BH spin :math:`a`, and the
mass parameters :math:`(M, \\mu)`.

:attr:`EMRIInspiral` stores only the (static) flux data.  All physical
parameters — including ``a`` — are passed at call time, so a single
instance can be vmapped over the full parameter space::

    from jax import vmap
    traj = EMRIInspiral(flux_data)
    batch = vmap(lambda p0, e0, a: traj(p0=p0, e0=e0, a=a, T=1.0, M=1e6, mu=10.0))
    results = batch(p0_arr, e0_arr, a_arr)

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

from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx

from fewtrax.utils.constants import MTSUN_SI, YEAR_SI
from fewtrax.utils.geodesic import (
    get_separatrix_fast,
    get_fundamental_frequencies_platform,
)
from fewtrax.utils.coordinates import (
    kerrecceq_forward_map_fast, DELTAPMIN, min_valid_p,
)
from fewtrax.data.loader import FluxData, _pdot_PN_jax, _edot_PN_jax

SEPARATRIX_BUFFER: float = 2.0 * DELTAPMIN


def _x_sign(a: jnp.ndarray, x0: float) -> jnp.ndarray:
    """Inclination sign (±1), safe at a=0."""
    ax = jnp.sign(a * x0)
    return jnp.where(ax == 0.0, 1.0, ax)


class EMRIInspiral(eqx.Module):
    r"""Adiabatic EMRI trajectory integrator (JIT/vmap-compatible).

    Integrates the adiabatic inspiral ODE using :class:`diffrax.Dopri8`
    (8th-order Dormand-Prince) with adaptive step-size control.  The radiation-reaction force
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

    **Hybrid implementation** — per-step optimisations relative to a naive
    baseline:

    * **Fast separatrix** — :func:`~fewtrax.utils.geodesic.get_separatrix_fast`
      (20-step bisection + 5 Newton-Raphson) replaces the 50-step pure-bisection
      version.  Same float64 accuracy, fewer operations.
    * **p_sep reuse** — the separatrix is computed once per ODE-RHS evaluation
      and passed directly to the flux helper, eliminating the redundant internal
      call that the base-class ``_flux_pex`` used to make.
    * **Platform-aware elliptic integrals** —
      :func:`~fewtrax.utils.geodesic.get_fundamental_frequencies_platform`
      selects 64-point Gauss-Legendre (GPU: maps to cuBLAS contractions) or
      AGM + 24-pt GL (CPU: fewer sequential operations) at JIT-trace time.
    * **Configurable adjoint** — defaults to
      :class:`diffrax.RecursiveCheckpointAdjoint` (low memory, optimal for
      ``jax.grad`` / ``jax.jacrev``).  Pass ``adjoint=diffrax.DirectAdjoint()``
      to enable ``jax.jacfwd`` / ``jax.hessian``.

    ``a`` and ``x0`` are **call-time** arguments rather than constructor
    fields, so a single instance can be vmapped over the full five-parameter
    space :math:`(M, \mu, a, p_0, e_0)` without rebuilding the Module::

        from jax import vmap
        traj = EMRIInspiral(flux_data)
        batch = vmap(lambda p0, e0, a: traj(p0=p0, e0=e0, a=a, T=1.0, M=1e6, mu=10.0))
        results = batch(p0_arr, e0_arr, a_arr)

    Parameters
    ----------
    flux_data : FluxData
        Pre-loaded flux interpolators (pex convention).
    """

    flux_data: FluxData
    t_obs: Optional[jnp.ndarray]
    phases: bool = eqx.field(static=True)
    adjoint: Any = eqx.field(static=True)

    def __init__(
        self,
        flux_data: FluxData,
        t_obs: Optional[jnp.ndarray] = None,
        phases: bool = True,
        adjoint: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        flux_data : FluxData
            Pre-loaded flux interpolators.
        t_obs : array-like, shape (N_obs,), optional
            Physical observation times [s] at which to save the trajectory.
            When provided, the ODE saves at exactly these times instead of a
            uniform ``dense_steps`` grid, so no interpolation is needed when
            comparing to STFT segment centres.  The shape is fixed at
            construction time (required for JIT/vmap).
        phases : bool
            If ``True`` (default), integrate all five state variables
            ``(p, e, Φ_φ, Φ_θ, Φ_r)`` and return the full trajectory.
            If ``False``, integrate only ``(p, e)`` and return
            ``(t, p, e)``; the adjoint and JVP cost are ~2.5× lower,
            which benefits ``jax.grad`` / ``jax.jacfwd`` for frequency-track
            loss functions that do not require the orbital phases.
        adjoint : diffrax adjoint, optional
            Adjoint method for reverse-mode autodiff through the ODE solve.

            * ``None`` (default) → :class:`diffrax.RecursiveCheckpointAdjoint`.
              Memory-efficient checkpointing; supports ``jax.grad`` and
              ``jax.jacrev``.  Recommended for large-batch or memory-constrained
              workloads.
            * :class:`diffrax.DirectAdjoint` → expresses the adjoint as a
              second ODE backward in time; also provides a ``custom_jvp`` rule
              that enables ``jax.jacfwd`` and ``jax.hessian``.  Higher memory
              pressure at large batch sizes.

            Example::

                import diffrax
                traj = EMRIInspiral(flux_data, adjoint=diffrax.DirectAdjoint())
                J = jax.jacfwd(lambda p0: traj(p0=p0, ...))(p0_val)
        """
        self.flux_data = flux_data
        self.t_obs = (
            jnp.asarray(t_obs, dtype=jnp.float64) if t_obs is not None else None
        )
        self.phases = phases
        self.adjoint = (
            diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flux_pex(
        self,
        a: jnp.ndarray,
        x0: float,
        r_isco: jnp.ndarray,
        p: jnp.ndarray,
        e: jnp.ndarray,
        p_sep: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Return physical :math:`(\dot{p}, \dot{e})` from the pex spline tables.

        Evaluates the pre-computed pex ratios :math:`\dot{p}/\dot{p}_{\rm PN}`
        and :math:`\dot{e}/\dot{e}_{\rm PN}`, then multiplies by the
        separatrix-dependent PN functions.  Returns **negative** values for
        :math:`\dot{p}` (inspiraling orbit loses semi-latus rectum).

        ``r_isco = p_sep(a, e=0)`` and ``p_sep = p_sep(a, e)`` are both
        pre-computed by the caller (in ``_ode_rhs``) so that the separatrix
        polynomial is evaluated only once per ODE-RHS call.
        """
        a_abs = jnp.abs(a)
        u, w, z, in_A = kerrecceq_forward_map_fast(a_abs, p, e, p_sep, kind="flux")

        rp_A = self.flux_data.pdot_A(u, w, z)
        re_A = self.flux_data.edot_A(u, w, z)
        rp_B = self.flux_data.pdot_B(u, w, z)
        re_B = self.flux_data.edot_B(u, w, z)

        rp = jnp.where(in_A, rp_A, rp_B)
        re = jnp.where(in_A, re_A, re_B)

        pdot_pn = _pdot_PN_jax(p, e, r_isco, p_sep)
        edot_pn = _edot_PN_jax(p, e, r_isco, p_sep)

        # Negative: orbit is inspiraling (p decreases, e typically decreases)
        return -rp * pdot_pn, -re * edot_pn

    def _ode_rhs(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        r"""ODE right-hand side: :math:`dy/dt` for :math:`y=(p,e,\Phi_\phi,\Phi_\theta,\Phi_r)`.

        ``args`` is a 4-tuple ``(mu_over_M, a, x0, r_isco)`` so that all
        physical parameters flow through diffrax and remain differentiable.
        ``r_isco`` is the spin-only separatrix ``p_sep(a, e=0)``, precomputed
        once per solve.  The orbital phases evolve at the Boyer-Lindquist
        frequencies and are *not* affected by the mass ratio.

        ``p_sep(a, e)`` is computed once here and forwarded to ``_flux_pex``
        so that the separatrix polynomial is not evaluated a second time inside
        the coordinate-map helper.
        """
        mu_over_M, a, x0, r_isco = args
        p, e = y[0], y[1]
        a_abs = jnp.abs(a)
        x_in = _x_sign(a, x0)

        # Compute p_sep once; reuse for both the flux helper and (implicitly)
        # the event condition which is a separate diffrax callback.
        p_sep = get_separatrix_fast(a_abs, e, x_in)
        pdot, edot = self._flux_pex(a, x0, r_isco, p, e, p_sep)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies_platform(
            a_abs, p, e, x_in
        )
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
        ode_args: tuple,
        max_steps: int,
        atol: float = 1e-10,
        rtol: float = 1e-10,
    ):
        """JIT-compiled diffrax solve.  All shape-determining args are static."""
        def _event_cond(t, y, args, **kwargs):
            _, a, x0, _r_isco = args
            p, e = y[0], y[1]
            p_sep = get_separatrix_fast(jnp.abs(a), e, _x_sign(a, x0))
            return p < p_sep + SEPARATRIX_BUFFER

        return diffrax.diffeqsolve(
            diffrax.ODETerm(self._ode_rhs),
            diffrax.Dopri8(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            event=diffrax.Event(_event_cond),
            args=ode_args,
            adjoint=self.adjoint,
            throw=False,
        )

    @eqx.filter_jit
    def _solve_backward(
        self,
        y0: jnp.ndarray,
        t_save: jnp.ndarray,
        T_geo: jnp.ndarray,
        ode_args: tuple,
        max_steps: int,
        atol: float = 1e-10,
        rtol: float = 1e-10,
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
            diffrax.Dopri8(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            args=ode_args,
            adjoint=self.adjoint,
        )

    @eqx.filter_jit
    def _solve_2d(
        self,
        y0: jnp.ndarray,
        t_save: jnp.ndarray,
        T_geo: jnp.ndarray,
        ode_args: tuple,
        max_steps: int,
        atol: float = 1e-10,
        rtol: float = 1e-10,
    ):
        """JIT-compiled 2D diffrax solve: integrates (p, e) only.

        Used when ``phases=False``.  Adjoint and JVP cost are ~2.5× lower
        than the 5D solve because the co-state / tangent vector has only
        two components.
        """
        def _ode_rhs_2d(t, y, args):
            mu_over_M, a, x0, r_isco = args
            p, e = y[0], y[1]
            a_abs = jnp.abs(a)
            x_in = _x_sign(a, x0)
            p_sep = get_separatrix_fast(a_abs, e, x_in)
            pdot, edot = self._flux_pex(a, x0, r_isco, p, e, p_sep)
            return jnp.array([pdot * mu_over_M, edot * mu_over_M])

        def _event_cond(t, y, args, **kwargs):
            _, a, x0, _r_isco = args
            p, e = y[0], y[1]
            p_sep = get_separatrix_fast(jnp.abs(a), e, _x_sign(a, x0))
            return p < p_sep + SEPARATRIX_BUFFER

        return diffrax.diffeqsolve(
            diffrax.ODETerm(_ode_rhs_2d),
            diffrax.Dopri8(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            event=diffrax.Event(_event_cond),
            args=ode_args,
            adjoint=self.adjoint,
            throw=False,
        )

    @eqx.filter_jit
    def _solve_2d_backward(
        self,
        y0: jnp.ndarray,
        t_save: jnp.ndarray,
        T_geo: jnp.ndarray,
        ode_args: tuple,
        max_steps: int,
        atol: float = 1e-10,
        rtol: float = 1e-10,
    ):
        """JIT-compiled backward 2D diffrax solve: integrates (p, e) time-reversed.

        Negates the 2D RHS so p and e *increase* along the output (moving away
        from the separatrix).  No separatrix event is needed: the integration
        always moves outward.
        """
        def _ode_rhs_2d_bwd(t, y, args):
            mu_over_M, a, x0, r_isco = args
            p, e = y[0], y[1]
            a_abs = jnp.abs(a)
            x_in = _x_sign(a, x0)
            p_sep = get_separatrix_fast(a_abs, e, x_in)
            pdot, edot = self._flux_pex(a, x0, r_isco, p, e, p_sep)
            return jnp.array([-pdot * mu_over_M, -edot * mu_over_M])

        return diffrax.diffeqsolve(
            diffrax.ODETerm(_ode_rhs_2d_bwd),
            diffrax.Dopri8(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            args=ode_args,
            adjoint=self.adjoint,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        p0: float,
        e0: float,
        T: float,
        a: float,
        x0: float = 1.0,
        dt: float = 10.0,
        M: float = 1e6,
        mu: float = 10.0,
        Phi_phi0: float = 0.0,
        Phi_theta0: float = 0.0,
        Phi_r0: float = 0.0,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        max_steps: int = 256,
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
        a : float
            Dimensionless BH spin parameter.  Passed as a differentiable
            JAX array so that :func:`jax.grad` / :func:`jax.vmap` work
            across ``a`` without rebuilding the Module.
        x0 : float
            Inclination cosine (+1 prograde, −1 retrograde).
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
            Maximum number of ODE internal steps.  The Dopri8 solver
            typically uses ~100–150 steps for a year-long EMRI inspiral,
            so the default of 256 leaves a safe margin.  Under reverse-mode
            autodiff (``jax.grad``), diffrax allocates adjoint buffers
            scaling with ``max_steps``, so lowering this value directly
            reduces GPU memory use under ``vmap``.  Raise it only for
            pathological cases that actually require more steps.
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
        t, p, e : jnp.ndarray, shape (N_save,)
            When ``phases=False`` (set at construction time).
        t, p, e, Phi_phi, Phi_theta, Phi_r : jnp.ndarray, shape (N_save,)
            When ``phases=True`` (default).  ``t`` is in seconds; phases in
            radians.  In backward mode ``t`` is time before plunge
            (:math:`\tau`).  ``N_save`` equals ``dense_steps`` when
            ``t_obs=None``, or ``len(t_obs)`` when ``t_obs`` was provided at
            construction.
        """
        a_ = jnp.asarray(a, dtype=jnp.float64)
        M_s = (M + mu) * MTSUN_SI                 # total mass in seconds (matches FEW convention)
        T_s = T * YEAR_SI                         # observation time in seconds
        T_geo = jnp.asarray(T_s / M_s)           # geometric time  (units of M)
        mu_over_M = jnp.asarray(M * mu / (M + mu) ** 2)  # symmetric mass ratio eta (matches FEW)
        # Precompute the spin-only separatrix r_isco = p_sep(a, e=0) once and
        # thread it through ``ode_args`` to avoid recomputing at every RHS step.
        r_isco = get_separatrix_fast(
            jnp.abs(a_), jnp.zeros((), dtype=jnp.float64), _x_sign(a_, x0)
        )
        ode_args = (mu_over_M, a_, x0, r_isco)

        if self.t_obs is not None:
            t_save = self.t_obs / M_s   # physical seconds → geometric time
        else:
            t_save = jnp.linspace(jnp.zeros((), jnp.float64), T_geo, dense_steps)

        if backward:
            if e_f is None:
                raise ValueError(
                    "e_f must be provided when backward=True.  "
                    "Run a forward integration first and read off e at the "
                    "last valid trajectory point."
                )
            e_f_ = jnp.asarray(e_f, dtype=jnp.float64)
            p_sep = get_separatrix_fast(jnp.abs(a_), e_f_, _x_sign(a_, x0))
            p_start = p_sep + SEPARATRIX_BUFFER
            if self.phases:
                y0 = jnp.array(
                    [p_start, e_f_, Phi_phi0, Phi_theta0, Phi_r0], dtype=jnp.float64
                )
                sol = self._solve_backward(y0, t_save, T_geo, ode_args, max_steps, atol, rtol)
            else:
                y0 = jnp.array([p_start, e_f_], dtype=jnp.float64)
                sol = self._solve_2d_backward(y0, t_save, T_geo, ode_args, max_steps, atol, rtol)
        else:
            # Only validate p0 against the grid when every input is concrete;
            # under jit/vmap/grad these are tracers and the check is skipped.
            if not any(
                isinstance(x, jax.core.Tracer) for x in (p0, e0, a_, x0)
            ):
                p0_min = float(min_valid_p(a_, e0, x0))
                if float(p0) < p0_min:
                    raise ValueError(
                        f"p0={float(p0):.6g} is outside the flux interpolation "
                        f"grid.  Must be >= {p0_min:.6g} for a={float(a_):.4g}, "
                        f"e0={float(e0):.4g}, x0={float(x0):.4g}."
                    )
            if self.phases:
                y0 = jnp.array([p0, e0, Phi_phi0, Phi_theta0, Phi_r0], dtype=jnp.float64)
                sol = self._solve(y0, t_save, T_geo, ode_args, max_steps, atol, rtol)
            else:
                y0 = jnp.array([p0, e0], dtype=jnp.float64)
                sol = self._solve_2d(y0, t_save, T_geo, ode_args, max_steps, atol, rtol)

        t_arr = sol.ts       # (N_save,)
        ys = sol.ys          # (N_save, 5) or (N_save, 2)
        t_s = t_arr * M_s
        p_arr = ys[:, 0]
        e_arr = ys[:, 1]

        if not self.phases:
            return t_s, p_arr, e_arr

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
        a: float,
        l: int, m: int, k: int, n: int,
        x0: float = 1.0,
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
        a : float
            Dimensionless BH spin.
        l, m, k, n : int
            Harmonic mode indices.
        x0 : float
            Inclination cosine (+1 prograde, −1 retrograde).
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
        a_ = jnp.asarray(a, dtype=jnp.float64)
        t_s, p_arr, e_arr, *_ = self(
            p0=p0, e0=e0, T=T, M=M, mu=mu, a=a_, x0=x0,
            dense_steps=dense_steps, backward=backward, e_f=e_f, **kwargs
        )
        a_abs = jnp.abs(a_)
        x_in = _x_sign(a_, x0)

        def _freq_at_point(pe):
            p_, e_ = pe
            Om_phi, Om_theta, Om_r = get_fundamental_frequencies_platform(a_abs, p_, e_, x_in)
            return jnp.abs(m * Om_phi + k * Om_theta + n * Om_r) / (2.0 * jnp.pi)

        # Physical frequency: convert geometric Ω (rad/M) to Hz.
        # Use total mass M+mu to match the geometric time convention.
        M_s = (M + mu) * MTSUN_SI
        f_arr = jax.vmap(_freq_at_point)(jnp.stack([p_arr, e_arr], axis=1)) / M_s
        return t_s, f_arr

    def get_multi_track(
        self,
        modes: list,
        p0: float,
        e0: float,
        T: float,
        a: float,
        M: float = 1e6,
        mu: float = 10.0,
        x0: float = 1.0,
        dense_steps: int = 200,
        backward: bool = False,
        e_f: Optional[float] = None,
        with_amplitudes: bool = False,
        amp_interp=None,
        return_dict: bool = False,
        **kwargs,
    ):
        r"""Frequency tracks for multiple harmonic modes along one shared trajectory.

        Runs the ODE integrator once and evaluates all mode frequencies
        simultaneously via a single matrix–vector product at each trajectory
        point, which is significantly faster than calling
        :meth:`get_frequency_track` once per mode.

        Parameters
        ----------
        modes : list of tuples
            Each entry is ``(m, k, n)`` or ``(l, m, k, n)``.  The ``l``
            index is only used for amplitude look-up (when
            ``with_amplitudes=True``); it has no effect on frequencies.
        p0, e0 : float
            Initial orbital parameters.
        T : float
            Observation time [years].
        a : float
            Dimensionless BH spin.
        M, mu : float
            Primary and secondary masses [:math:`M_\odot`].
        x0 : float
            Inclination cosine (``+1`` prograde, ``-1`` retrograde).
        dense_steps : int
            Number of output trajectory/frequency points.
        backward : bool
            If ``True``, integrate backward from the separatrix.
        e_f : float, optional
            Eccentricity at plunge; required when ``backward=True``.
        with_amplitudes : bool
            If ``True``, also return the mode amplitudes ``|A_{lmkn}(t)|``
            along the track.  Requires ``amp_interp``.
        amp_interp : AmplitudeInterpolator, optional
            Pre-constructed amplitude interpolator.  Required when
            ``with_amplitudes=True``.
        return_dict : bool
            If ``True``, return a dictionary keyed by the mode tuples
            instead of plain arrays.

        Returns
        -------
        If ``return_dict=False`` (default):
            t : jnp.ndarray, shape (dense_steps,)
                Time [s].
            freqs : jnp.ndarray, shape (N_modes, dense_steps)
                Instantaneous frequency [Hz] for each mode, always
                non-negative.
            amps : jnp.ndarray, shape (N_modes, dense_steps), optional
                ``|A_{lmkn}(t)|`` — only present when
                ``with_amplitudes=True``.

        If ``return_dict=True``:
            dict mapping each mode tuple to ``(t, f)`` or
            ``(t, f, A)`` arrays (``A`` only when
            ``with_amplitudes=True``).
        """
        if not modes:
            raise ValueError("modes must be a non-empty list.")
        if with_amplitudes and amp_interp is None:
            raise ValueError(
                "amp_interp is required when with_amplitudes=True."
            )

        # Normalise mode tuples to (l, m, k, n); use l=None if not given
        norm_modes = []
        for entry in modes:
            if len(entry) == 3:
                norm_modes.append((None, entry[0], entry[1], entry[2]))
            elif len(entry) == 4:
                norm_modes.append(tuple(entry))
            else:
                raise ValueError(
                    f"Each mode must be a (m,k,n) or (l,m,k,n) tuple; got {entry!r}."
                )

        # --- Run ODE once ---
        a_ = jnp.asarray(a, dtype=jnp.float64)
        t_s, p_arr, e_arr, *_ = self(
            p0=p0, e0=e0, T=T, M=M, mu=mu, a=a_, x0=x0,
            dense_steps=dense_steps, backward=backward, e_f=e_f, **kwargs
        )

        a_abs = jnp.abs(a_)
        x_in = _x_sign(a_, x0)
        M_s = (M + mu) * MTSUN_SI

        # Build integer coefficient matrix: shape (N_modes, 3) — columns are m, k, n
        mkn_arr = jnp.array(
            [[lmkn[1], lmkn[2], lmkn[3]] for lmkn in norm_modes],
            dtype=jnp.float64,
        )  # (N_modes, 3)

        # Compute Omegas at every trajectory point: shape (dense_steps, 3)
        def _omegas_at(pe):
            p_, e_ = pe[0], pe[1]
            Om_phi, Om_theta, Om_r = get_fundamental_frequencies_platform(a_abs, p_, e_, x_in)
            return jnp.stack([Om_phi, Om_theta, Om_r])

        Omegas = jax.vmap(_omegas_at)(jnp.stack([p_arr, e_arr], axis=1))  # (dense_steps, 3)

        # All mode frequencies in one matrix multiply: (N_modes, dense_steps)
        freqs = jnp.abs(mkn_arr @ Omegas.T) / (2.0 * jnp.pi * M_s)

        # --- Optional amplitude evaluation ---
        amps = None
        if with_amplitudes:
            import numpy as _np
            p_np = _np.asarray(p_arr)
            e_np = _np.asarray(e_arr)
            a_float = float(jnp.abs(a_))

            # Build list of amplitude mode indices for look-up
            # Each norm_mode has (l, m, k, n); find matching index in amp_interp
            amp_indices = []
            for lmkn in norm_modes:
                l_, m_, k_, n_ = lmkn
                if l_ is None:
                    # Search by (m, k, n) only
                    mask = (
                        (amp_interp.m_arr == m_)
                        & (amp_interp.k_arr == k_)
                        & (amp_interp.n_arr == n_)
                    )
                else:
                    mask = (
                        (amp_interp.l_arr == l_)
                        & (amp_interp.m_arr == m_)
                        & (amp_interp.k_arr == k_)
                        & (amp_interp.n_arr == n_)
                    )
                idx = _np.where(mask)[0]
                amp_indices.append(int(idx[0]) if len(idx) > 0 else None)

            # Filter to modes that exist in the amplitude data
            valid_amp_idx = [i for i in amp_indices if i is not None]
            valid_mode_idx = [j for j, i in enumerate(amp_indices) if i is not None]

            if valid_amp_idx:
                A_raw = amp_interp.evaluate(
                    a_float, p_np, e_np, specific_modes=valid_amp_idx
                )  # (dense_steps, len(valid_amp_idx)), complex
                A_abs = _np.abs(A_raw).T  # (len(valid_amp_idx), dense_steps)
                # Build full amps array (NaN for missing modes)
                amps_np = _np.full(
                    (len(norm_modes), len(t_s)), _np.nan, dtype=_np.float64
                )
                for out_row, mode_row in enumerate(valid_mode_idx):
                    amps_np[mode_row] = A_abs[out_row]
                amps = jnp.array(amps_np)
            else:
                amps = jnp.full((len(norm_modes), len(t_s)), jnp.nan)

        # --- Build output ---
        if return_dict:
            result = {}
            for i, entry in enumerate(modes):
                key = tuple(entry)
                if with_amplitudes:
                    result[key] = (t_s, freqs[i], amps[i])
                else:
                    result[key] = (t_s, freqs[i])
            return result

        if with_amplitudes:
            return t_s, freqs, amps
        return t_s, freqs

    @classmethod
    def get_f_fdot_fddot_back(
        cls,
        flux_data: "FluxData",
        M: float,
        mu: float,
        a: float,
        e_f: float,
        T: float,
        t_alpha: jnp.ndarray,
        x0: float = 1.0,
        max_steps: int = 512,
        atol: float = 1e-10,
        rtol: float = 1e-10,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""Gravitational-wave frequency and its time derivatives on a fixed grid.

        Solves the backward inspiral ODE once with dense (polynomial) output,
        then evaluates f, ḟ, f̈ at every point in ``t_alpha`` by differentiating
        the Dopri8 dense interpolant with ``jax.grad`` — no additional ODE
        solves are required.

        The backward ODE anchors the track to the plunge event rather than to
        uncertain initial conditions, and is numerically more stable near
        merger.  The sign convention follows
        :math:`d\Phi_\phi/d\tau = -\Omega_\phi` (backward time reversal), so

        .. math::

            f        &= -\frac{d\Phi_\phi/d\tau}{2\pi M_s} \\
            \dot{f}  &= +\frac{d^2\Phi_\phi/d\tau^2}{2\pi M_s^2} \\
            \ddot{f} &= -\frac{d^3\Phi_\phi/d\tau^3}{2\pi M_s^3}

        where :math:`\tau` is geometric backward time (:math:`M` units) and
        :math:`M_s = (M + \mu)\,G/c^3` is the total mass in seconds.

        Parameters
        ----------
        flux_data : FluxData
            Pre-loaded flux interpolators.
        M, mu : float
            Primary and secondary masses [:math:`M_\odot`].
        a : float
            Dimensionless BH spin.
        e_f : float
            Eccentricity at plunge (sets the backward-ODE initial condition).
        T : float
            Duration of backward integration [years].  The ODE integrates
            from :math:`\tau = 0` (plunge) to :math:`\tau = T \cdot {\rm yr}`.
        t_alpha : array-like, shape (N,)
            Physical time grid [s].  ``t_alpha[0] = 0`` is the start of the
            observation window; ``t_alpha[-1]`` is the plunge moment.
            Must satisfy ``t_alpha[-1] \le T \cdot {\rm YEAR\_SI}``.
        x0 : float
            Inclination cosine (+1 prograde, −1 retrograde).
        max_steps : int
            Maximum ODE internal steps.
        atol, rtol : float
            Adaptive step-size tolerances.

        Returns
        -------
        f : jnp.ndarray, shape (N,)
            Instantaneous GW frequency [Hz].
        fdot : jnp.ndarray, shape (N,)
            First physical time derivative df/dt [Hz/s].
        fddot : jnp.ndarray, shape (N,)
            Second physical time derivative d²f/dt² [Hz/s²].
        """
        instance = cls(flux_data, phases=True)

        a_ = jnp.asarray(a, dtype=jnp.float64)
        e_f_ = jnp.asarray(e_f, dtype=jnp.float64)
        M_s = jnp.asarray((M + mu) * MTSUN_SI, dtype=jnp.float64)
        T_geo = jnp.asarray(T * YEAR_SI / M_s, dtype=jnp.float64)
        mu_over_M = jnp.asarray(M * mu / (M + mu) ** 2, dtype=jnp.float64)
        r_isco = get_separatrix_fast(
            jnp.abs(a_), jnp.zeros((), jnp.float64), _x_sign(a_, x0)
        )
        ode_args = (mu_over_M, a_, x0, r_isco)

        p_sep = get_separatrix_fast(jnp.abs(a_), e_f_, _x_sign(a_, x0))
        y0 = jnp.array(
            [p_sep + SEPARATRIX_BUFFER, e_f_, 0.0, 0.0, 0.0], dtype=jnp.float64
        )

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -instance._ode_rhs(t, y, args)),
            diffrax.Dopri8(),
            t0=jnp.zeros((), jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(dense=True),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            args=ode_args,
        )

        # Map t_alpha [s] → backward geometric time τ [M]:
        #   τ = 0  at plunge      (t_alpha[-1])
        #   τ = T_geo  at start   (t_alpha[0])
        t_alpha_ = jnp.asarray(t_alpha, dtype=jnp.float64)
        tau_query = (t_alpha_[-1] - t_alpha_) / M_s

        def _phi_phi_at(tau):
            return sol.interpolation.evaluate(tau)[2]

        two_pi_Ms = 2.0 * jnp.pi * M_s

        def _get_derivs(tau):
            d1 = jax.grad(_phi_phi_at)(tau)
            d2 = jax.grad(jax.grad(_phi_phi_at))(tau)
            d3 = jax.grad(jax.grad(jax.grad(_phi_phi_at)))(tau)
            return (
                -d1 / two_pi_Ms,            # f      [Hz]
                d2 / (two_pi_Ms * M_s),     # fdot   [Hz/s]
                -d3 / (two_pi_Ms * M_s**2), # fddot  [Hz/s²]
            )

        f, fdot, fddot = jax.vmap(_get_derivs)(tau_query)
        return f, fdot, fddot


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
    traj = EMRIInspiral(flux_data)
    return traj(
        p0=p0, e0=e0, T=T, a=a, x0=x0, dt=dt, M=M, mu=mu,
        Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0,
        dense_steps=dense_steps, backward=backward, e_f=e_f, **kwargs,
    )

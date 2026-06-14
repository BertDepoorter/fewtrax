"""EMRI orbital trajectory integration via diffrax.

Integrates the adiabatic inspiral ODE for :math:`(p, e, \\Phi_\\phi,
\\Phi_\\theta, \\Phi_r)` (time in units of :math:`M`) with
:class:`diffrax.Dopri8` and adaptive step-size control, stopping when the
orbit reaches the separatrix.  The solve is JIT-compilable and differentiable
in :math:`(p_0, e_0, a, M, \\mu)`.

:class:`EMRIInspiral` stores only the static flux data; all physical
parameters are call-time arguments, so a single instance can be vmapped over
the parameter space::

    from jax import vmap
    traj = EMRIInspiral(flux_data)
    batch = vmap(lambda p0, e0, a: traj(p0=p0, e0=e0, a=a, T=1.0, M=1e6, mu=10.0))
    results = batch(p0_arr, e0_arr, a_arr)

Set ``backward=True`` to integrate the time-reversed ODE from the separatrix
outward (with time axis :math:`\\tau = T_{\\rm plunge} - t`), anchoring the
track to the plunge rather than to uncertain initial conditions.
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
from fewtrax.trajectory.helpers import dense_phase_derivs

SEPARATRIX_BUFFER: float = 2.0 * DELTAPMIN


def _x_sign(a: jnp.ndarray, x0: float) -> jnp.ndarray:
    """Inclination sign (+/-1)."""
    ax = jnp.sign(a * x0)
    return jnp.where(ax == 0.0, 1.0, ax)


class EMRIInspiral(eqx.Module):
    r"""Adiabatic EMRI trajectory integrator (JIT/vmap-compatible).

    Integrates the adiabatic inspiral ODE with :class:`diffrax.Dopri8` and
    adaptive step-size control.  The radiation-reaction force
    (:math:`\dot{p}`, :math:`\dot{e}`) is read from the FEW flux tables in the
    pex convention.  All physical parameters — including ``a`` and ``x0`` — are
    call-time arguments, so a single instance vmaps over the five-parameter
    space :math:`(M, \mu, a, p_0, e_0)` without rebuilding the Module::

        from jax import vmap
        traj = EMRIInspiral(flux_data)
        batch = vmap(lambda p0, e0, a: traj(p0=p0, e0=e0, a=a, T=1.0, M=1e6, mu=10.0))
        results = batch(p0_arr, e0_arr, a_arr)

    The adjoint defaults to :class:`diffrax.RecursiveCheckpointAdjoint`
    (``jax.grad`` / ``jax.jacrev``); pass ``adjoint=diffrax.DirectAdjoint()``
    to enable ``jax.jacfwd`` / ``jax.hessian``. usually jax.jacfwd is the fastest option

    Parameters
    ----------
    flux_data : FluxData
        Pre-loaded flux interpolators (pex convention).
    """

    flux_data: FluxData
    t_obs: Optional[jnp.ndarray]
    adjoint: Any = eqx.field(static=True)

    def __init__(
        self,
        flux_data: FluxData,
        t_obs: Optional[jnp.ndarray] = None,
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

        Values are **negative** (inspiraling orbit).  ``r_isco = p_sep(a, e=0)``
        and ``p_sep = p_sep(a, e)`` are supplied by the caller so the separatrix
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

        ``args`` is the 4-tuple ``(mu_over_M, a, x0, r_isco)`` (all physical
        parameters flow through diffrax and stay differentiable).  ``r_isco`` is
        the spin-only separatrix ``p_sep(a, e=0)``, precomputed once per solve.
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

    # Shared event condition (boolean, vmap-safe — no root-finder required).
    # Returns True when the orbit has reached the separatrix buffer and
    # integration should stop.  Used in all four forward-integration methods.
    @staticmethod
    def _separatrix_event(t, y, args, **kwargs):
        _, a, x0, _ = args
        p, e = y[0], y[1]
        p_sep = get_separatrix_fast(jnp.abs(a), e, _x_sign(a, x0))
        return p < p_sep + SEPARATRIX_BUFFER

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
        """JIT-compiled 5D diffrax solve on a fixed uniform output grid.

        Always retains the Dopri8 dense (7th/8th-order) interpolant on
        ``sol.interpolation`` (``SaveAt(..., dense=True)``), so phases are taken
        from the native polynomial rather than refit with a less accurate cubic
        spline.  ``t1=True`` adds one extra slot capturing the precise stop-time
        (plunge or T_geo), so the final valid state is identifiable without
        searching for NaN boundaries.  Output shape: ``(len(t_save) + 1, 5)``.
        """
        return diffrax.diffeqsolve(
            diffrax.ODETerm(self._ode_rhs),
            diffrax.Dopri8(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_save, t1=True, dense=True),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            event=diffrax.Event(self._separatrix_event),
            args=ode_args,
            adjoint=self.adjoint,
            throw=False,
        )

    @eqx.filter_jit
    def _solve_steps(
        self,
        y0: jnp.ndarray,
        T_geo: jnp.ndarray,
        ode_args: tuple,
        max_steps: int,
        atol: float = 1e-10,
        rtol: float = 1e-10,
    ):
        """JIT-compiled 5D diffrax solve saving at every accepted adaptive step.

        Returns the native ODE nodes rather than a fixed uniform grid.
        ``sol.ts`` and ``sol.ys`` have shape ``(max_steps,)`` /
        ``(max_steps, 5)``, padded with NaNs after the last valid step.  This is
        fully vmap-compatible because ``max_steps`` is static.
        """
        return diffrax.diffeqsolve(
            diffrax.ODETerm(self._ode_rhs),
            diffrax.Dopri8(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(steps=True),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            event=diffrax.Event(self._separatrix_event),
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
        r"""JIT-compiled backward diffrax solve (time-reversed ODE).

        Integrates :math:`dy/d\tau = -dy/dt` with
        :math:`\tau = T_{\rm plunge} - t`.  No separatrix event: the integration
        moves away from the separatrix.  Retains the dense (7th/8th-order)
        interpolant on ``sol.interpolation`` (as :meth:`_solve`) so backward
        waveforms can use the polynomial phases.  Parameters match
        :meth:`_solve`.
        """
        return diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -self._ode_rhs(t, y, args)),
            diffrax.Dopri8(),
            t0=jnp.zeros((), dtype=jnp.float64),
            t1=T_geo,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_save, dense=True),
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
        save_at_steps: bool = False,
        return_dense_phase_fn: bool = False,
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
            Dimensionless BH spin (differentiable / vmappable).
        x0 : float
            Inclination cosine (+1 prograde, −1 retrograde).
        dt : float
            Waveform sampling interval [s] (not the ODE step size).
        M, mu : float
            Primary and secondary masses [:math:`M_\odot`].
        Phi_phi0, Phi_theta0, Phi_r0 : float
            Initial orbital phases [rad] (phases at the backward-mode anchor).
        atol, rtol : float
            Adaptive solver tolerances.
        max_steps : int
            Maximum ODE internal steps.  Under reverse-mode autodiff diffrax
            allocates adjoint buffers scaling with ``max_steps``, so keep it as
            small as safe under ``vmap``.
        dense_steps : int
            Number of output trajectory points.
        backward : bool
            If ``True``, integrate the time-reversed ODE.  The output time axis
            is :math:`\tau` (time before the anchor, ``t[0] = 0`` at the
            anchor); ``p``, ``e`` increase and the phases decrease along the
            arrays.  The start is the separatrix when ``e_f`` is given, otherwise
            ``(p0, e0)`` (see ``e_f``).
        e_f : float, optional
            Backward mode only.  If given, anchor to plunge: start at
            :math:`p = p_{\rm sep}(e_f, a) + \epsilon` (``(p0, e0)`` ignored);
            obtain it from a forward run's last valid ``e``.  If ``None``, the
            backward integration instead starts from the supplied ``(p0, e0)``
            and runs into the past.
        save_at_steps : bool, optional
            If ``True``, return the native adaptive ODE nodes (shape
            ``(max_steps,)``, NaN-padded after the last step) instead of a fixed
            uniform grid.  Recommended for batched (``vmap``) evaluation:
            static shapes and nodes concentrated near plunge.
        return_dense_phase_fn : bool, optional
            If ``True`` (fixed-grid mode, forward or backward), additionally
            return a callable ``phase_fn(t_seconds) -> (Phi_phi, Phi_theta,
            Phi_r)`` as a 7th element, evaluating the phases from the Dopri8
            dense (7th/8th-order) interpolant.  In backward mode ``t_seconds``
            is time before plunge.  Incompatible with ``save_at_steps``.

        Returns
        -------
        t, p, e, Phi_phi, Phi_theta, Phi_r : jnp.ndarray, shape (N_save,)
            ``t`` is in seconds; phases in radians.  In backward mode ``t``
            is time before plunge (:math:`\tau`).  When
            ``return_dense_phase_fn=True`` a 7th element ``phase_fn`` is
            appended (see above).

            ``N_save`` is:

            * ``dense_steps + 1`` when ``save_at_steps=False`` and
              ``t_obs=None`` (the +1 slot captures the precise plunge/end
              time via ``SaveAt(..., t1=True)``).
            * ``len(t_obs)`` when ``t_obs`` was provided at construction.
            * ``max_steps`` when ``save_at_steps=True``.
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

        if return_dense_phase_fn and save_at_steps:
            raise NotImplementedError(
                "return_dense_phase_fn requires a fixed save grid; it is "
                "incompatible with save_at_steps=True.  Use the default "
                "fixed-grid mode (forward or backward)."
            )

        if backward:
            # Two backward-start conventions:
            #   * e_f given      → anchor to plunge: start at the separatrix
            #     p_start = p_sep(e_f) + buffer (the orbit widens going back).
            #   * e_f None        → start from the given (p0, e0) and integrate
            #     backward in time (toward earlier, wider-orbit inspiral).
            if e_f is not None:
                e_start = jnp.asarray(e_f, dtype=jnp.float64)
                p_sep = get_separatrix_fast(jnp.abs(a_), e_start, _x_sign(a_, x0))
                p_start = p_sep + SEPARATRIX_BUFFER
            else:
                if not any(
                    isinstance(x, jax.core.Tracer) for x in (p0, e0, a_, x0)
                ):
                    p0_min = float(min_valid_p(a_, e0, x0))
                    if float(p0) < p0_min:
                        raise ValueError(
                            f"p0={float(p0):.6g} is outside the flux "
                            f"interpolation grid.  Must be >= {p0_min:.6g} for "
                            f"a={float(a_):.4g}, e0={float(e0):.4g}, "
                            f"x0={float(x0):.4g}."
                        )
                p_start = jnp.asarray(p0, dtype=jnp.float64)
                e_start = jnp.asarray(e0, dtype=jnp.float64)
            # Backward integration always uses a uniform save grid
            t_save = (
                self.t_obs / M_s
                if self.t_obs is not None
                else jnp.linspace(jnp.zeros((), jnp.float64), T_geo, dense_steps)
            )
            y0 = jnp.array(
                [p_start, e_start, Phi_phi0, Phi_theta0, Phi_r0], dtype=jnp.float64
            )
            sol = self._solve_backward(y0, t_save, T_geo, ode_args, max_steps, atol, rtol)
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

            y0 = jnp.array([p0, e0, Phi_phi0, Phi_theta0, Phi_r0], dtype=jnp.float64)
            if save_at_steps:
                # Adaptive-nodes path: vmap-friendly, shape (max_steps,)
                sol = self._solve_steps(y0, T_geo, ode_args, max_steps, atol, rtol)
            else:
                # Uniform-grid path: shape (dense_steps+1,) when t_obs is None.
                # _solve always retains the dense interpolant.
                t_save = (
                    self.t_obs / M_s
                    if self.t_obs is not None
                    else jnp.linspace(jnp.zeros((), jnp.float64), T_geo, dense_steps)
                )
                sol = self._solve(y0, t_save, T_geo, ode_args, max_steps, atol, rtol)

        t_arr = sol.ts       # (N_save,)  or (max_steps,)
        ys = sol.ys          # same shape, state dim 5
        t_s = t_arr * M_s
        p_arr = ys[:, 0]
        e_arr = ys[:, 1]

        # Phases are Boyer-Lindquist orbital phases [rad]; no mass-ratio scaling needed
        Phi_phi_arr   = ys[:, 2]
        Phi_theta_arr = ys[:, 3]
        Phi_r_arr     = ys[:, 4]

        if return_dense_phase_fn:
            # Evaluate the orbital phases from the Dopri8 dense (7th/8th-order)
            # interpolant.  The query time [s] maps to the solver's geometric
            # time via tau = t / M_s; this holds for both forward integration
            # (t from start) and backward integration (t = time before plunge),
            # since the solve runs from tau = 0 in either case.
            interp = sol.interpolation

            def phase_fn(t_seconds):
                tau = jnp.asarray(t_seconds, dtype=jnp.float64) / M_s
                phases = jax.vmap(interp.evaluate)(tau)  # (N, 5)
                return phases[:, 2], phases[:, 3], phases[:, 4]

            return (
                t_s, p_arr, e_arr, Phi_phi_arr, Phi_theta_arr, Phi_r_arr, phase_fn
            )

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
        grid: Optional[Any] = None,
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
        amp_interp : JAXAmplitudeInterpolator, optional
            Pre-constructed amplitude interpolator.  Required when
            ``with_amplitudes=True``.
        return_dict : bool
            If ``True``, return a dictionary keyed by the mode tuples
            instead of plain arrays.
        grid : TFGrid, optional
            Any :class:`~fewtrax.utils.tf_bases.base.TFGrid` subclass (WDM, SFT,
            …).  When provided, each mode's track is interpolated onto the grid's
            time-bin centres, quantised to frequency-bin indices, and returned as
            a :class:`~fewtrax.utils.tf_tracks.TFTrackSet`.  Incompatible with
            ``return_dict`` and ``with_amplitudes``.

        Returns
        -------
        If ``grid`` is given:
            :class:`~fewtrax.utils.tf_tracks.TFTrackSet` with one
            :class:`~fewtrax.utils.tf_tracks.TFTrack` per mode (bin indices
            ``i_freq`` and frequencies ``freq_hz`` on the grid's time bins).

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
        if grid is not None and (return_dict or with_amplitudes):
            raise ValueError(
                "grid cannot be combined with return_dict or with_amplitudes."
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

        # --- Optional mapping onto a time-frequency grid (any TFGrid basis) ---
        if grid is not None:
            from fewtrax.utils.tf_tracks import TFTrack, TFTrackSet
            import numpy as _np

            t_np = _np.asarray(t_s)
            freqs_np = _np.asarray(freqs)
            valid = _np.isfinite(t_np)
            t_v = t_np[valid]
            # np.interp needs increasing x; backward tracks already satisfy
            # this (t is time-before-plunge, saved on an increasing grid).
            t_bins = grid.t_bins
            tracks = []
            for i in range(len(norm_modes)):
                f_v = freqs_np[i, valid]
                f_at_bins = _np.interp(
                    t_bins, t_v, f_v, left=f_v[0], right=f_v[-1]
                ).astype(_np.float32)
                i_freq = (
                    grid.freq_to_bin(f_at_bins)
                    .clip(0, grid.Nf - 1)
                    .astype(_np.int16)
                )
                # Always store the normalised (l, m, k, n) key (l may be None)
                tracks.append(
                    TFTrack(mode=norm_modes[i], grid=grid,
                            i_freq=i_freq, freq_hz=f_at_bins)
                )
            return TFTrackSet(tracks=tracks, grid=grid)

        # --- Optional amplitude evaluation ---
        amps = None
        if with_amplitudes:
            import numpy as _np
            l_all = _np.asarray(amp_interp.l_arr)
            m_all = _np.asarray(amp_interp.m_arr)
            k_all = _np.asarray(amp_interp.k_arr)
            n_all = _np.asarray(amp_interp.n_arr)

            # Locate each requested (l, m, k, n) in the amplitude data (l=None
            # matches on (m, k, n) only).
            amp_indices = []
            for lmkn in norm_modes:
                l_, m_, k_, n_ = lmkn
                mask = (m_all == m_) & (k_all == k_) & (n_all == n_)
                if l_ is not None:
                    mask = mask & (l_all == l_)
                idx = _np.where(mask)[0]
                amp_indices.append(int(idx[0]) if len(idx) > 0 else None)

            valid_amp_idx = [i for i in amp_indices if i is not None]
            valid_mode_idx = [j for j, i in enumerate(amp_indices) if i is not None]

            if valid_amp_idx:
                # Evaluate all modes along the track (prograde convention), then
                # keep the requested columns.
                A_all = amp_interp.evaluate_trajectory(jnp.abs(a_), p_arr, e_arr)
                A_abs = _np.abs(_np.asarray(A_all)[:, valid_amp_idx]).T
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
        return_phi_r: bool = False,
    ) -> tuple[jnp.ndarray, ...]:
        r"""Frequency and its time derivatives on a fixed grid (backward solve).

        Solves the backward inspiral ODE once with dense output, then evaluates
        f, ḟ, f̈ at every point in ``t_alpha`` by analytically differentiating
        the Dopri8 dense interpolant (no extra ODE solves; fully vmap-able).

        The conventions are :math:`f = -(d\Phi/d\tau)/(2\pi M_s)`,
        :math:`\dot f = (d^2\Phi/d\tau^2)/(2\pi M_s^2)`,
        :math:`\ddot f = -(d^3\Phi/d\tau^3)/(2\pi M_s^3)`, with :math:`\tau` the
        geometric backward time and :math:`M_s=(M+\mu)\,G/c^3` the total mass in
        seconds.

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
            Duration of backward integration [years].
        t_alpha : array-like, shape (N,)
            Physical time grid [s], increasing.  ``t_alpha[0]`` starts the
            observation window and ``t_alpha[-1]`` is the plunge moment;
            requires ``t_alpha[-1] <= T * YEAR_SI``.
        x0 : float
            Inclination cosine (+1 prograde, −1 retrograde).
        max_steps : int
            Maximum ODE internal steps.
        atol, rtol : float
            Adaptive step-size tolerances.
        return_phi_r : bool
            If ``True``, also return the :math:`\Phi_r`-derived radial frequency
            and its derivatives (``f_r, fdot_r, fddot_r``) as three extra
            arrays.

        Returns
        -------
        f, fdot, fddot : jnp.ndarray, shape (N,)
            :math:`\Phi_\phi`-derived GW frequency [Hz] and its physical time
            derivatives [Hz/s], [Hz/s²].
        f_r, fdot_r, fddot_r : jnp.ndarray, shape (N,), optional
            :math:`\Phi_r`-derived radial frequency and derivatives, only
            present when ``return_phi_r=True``.
        """
        instance = cls(flux_data)

        a_ = jnp.asarray(a, dtype=jnp.float64)
        e_f_ = jnp.asarray(e_f, dtype=jnp.float64)
        # Pure-jnp arithmetic throughout so M, mu may be vmapped tracers.
        M_s = jnp.asarray((M + mu) * MTSUN_SI, dtype=jnp.float64)
        T_geo = jnp.asarray(T * YEAR_SI, dtype=jnp.float64) / M_s
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
        #   τ = 0  at plunge (t_alpha[-1]);  τ = T_geo at start (t_alpha[0])
        t_alpha_ = jnp.asarray(t_alpha, dtype=jnp.float64)
        tau_query = (t_alpha_[-1] - t_alpha_) / M_s

        interp = sol.interpolation
        two_pi_Ms = 2.0 * jnp.pi * M_s

        def _freqs(tau, comp):
            d1, d2, d3 = dense_phase_derivs(interp, tau, comp)
            return (
                -d1 / two_pi_Ms,             # f      [Hz]
                d2 / (two_pi_Ms * M_s),      # fdot   [Hz/s]
                -d3 / (two_pi_Ms * M_s**2),  # fddot  [Hz/s²]
            )

        f, fdot, fddot = jax.vmap(lambda tau: _freqs(tau, 2))(tau_query)
        if return_phi_r:
            f_r, fdot_r, fddot_r = jax.vmap(lambda tau: _freqs(tau, 4))(tau_query)
            return f, fdot, fddot, f_r, fdot_r, fddot_r
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
    r"""Convenience wrapper: build an :class:`EMRIInspiral` and integrate once.

    Takes the same arguments as :meth:`EMRIInspiral.__call__` (plus
    ``flux_data``) and returns ``(t, p, e, Phi_phi, Phi_theta, Phi_r)``; in
    backward mode ``t`` is time before plunge.
    """
    traj = EMRIInspiral(flux_data)
    return traj(
        p0=p0, e0=e0, T=T, a=a, x0=x0, dt=dt, M=M, mu=mu,
        Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0,
        dense_steps=dense_steps, backward=backward, e_f=e_f, **kwargs,
    )


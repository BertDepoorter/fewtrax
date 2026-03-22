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


def _x_sign(a: jnp.ndarray, x0: float) -> jnp.ndarray:
    """Inclination sign (±1), safe at a=0."""
    ax = jnp.sign(a * x0)
    return jnp.where(ax == 0.0, 1.0, ax)


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

    ``a`` and ``x0`` are **call-time** arguments rather than constructor
    fields, so a single instance can be vmapped over the full five-parameter
    space :math:`(M, \mu, a, p_0, e_0)` without rebuilding the Module.

    Parameters
    ----------
    flux_data : FluxData
        Pre-loaded flux interpolators (pex convention).
    """

    flux_data: FluxData

    def __init__(self, flux_data: FluxData):
        self.flux_data = flux_data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flux_pex(
        self, a: jnp.ndarray, x0: float, p: jnp.ndarray, e: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Return physical :math:`(\dot{p}, \dot{e})` from the pex spline tables.

        Evaluates the pre-computed pex ratios :math:`\dot{p}/\dot{p}_{\rm PN}`
        and :math:`\dot{e}/\dot{e}_{\rm PN}`, then multiplies by the
        separatrix-dependent PN functions.  Returns **negative** values for
        :math:`\dot{p}` (inspiraling orbit loses semi-latus rectum).
        """
        a_abs = jnp.abs(a)
        x_in = _x_sign(a, x0)
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

        ``args`` is a 3-tuple ``(mu_over_M, a, x0)`` so that all physical
        parameters flow through diffrax and remain differentiable.
        The orbital phases evolve at the Boyer-Lindquist frequencies and are
        *not* affected by the mass ratio.
        """
        mu_over_M, a, x0 = args
        p, e = y[0], y[1]
        a_abs = jnp.abs(a)
        x_in = _x_sign(a, x0)

        pdot, edot = self._flux_pex(a, x0, p, e)
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
        ode_args: tuple,
        atol: float,
        rtol: float,
        max_steps: int,
    ):
        """JIT-compiled diffrax solve.  All shape-determining args are static."""
        def _event_cond(t, y, args, **kwargs):
            _, a, x0 = args
            p, e = y[0], y[1]
            p_sep = get_separatrix(jnp.abs(a), e, _x_sign(a, x0))
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
            args=ode_args,
        )

    @eqx.filter_jit
    def _solve_backward(
        self,
        y0: jnp.ndarray,
        t_save: jnp.ndarray,
        T_geo: jnp.ndarray,
        ode_args: tuple,
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
            args=ode_args,
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
        a_ = jnp.asarray(a, dtype=jnp.float64)
        M_s = (M + mu) * MTSUN_SI                 # total mass in seconds (matches FEW convention)
        T_s = T * YEAR_SI                         # observation time in seconds
        T_geo = jnp.asarray(T_s / M_s)           # geometric time  (units of M)
        mu_over_M = jnp.asarray(M * mu / (M + mu) ** 2)  # symmetric mass ratio eta (matches FEW)
        ode_args = (mu_over_M, a_, x0)

        t_save = jnp.linspace(jnp.zeros((), jnp.float64), T_geo, dense_steps)

        if backward:
            if e_f is None:
                raise ValueError(
                    "e_f must be provided when backward=True.  "
                    "Run a forward integration first and read off e at the "
                    "last valid trajectory point."
                )
            e_f_ = jnp.asarray(float(e_f), dtype=jnp.float64)
            p_sep = get_separatrix(jnp.abs(a_), e_f_, _x_sign(a_, x0))
            p_start = p_sep + SEPARATRIX_BUFFER
            y0 = jnp.array(
                [p_start, e_f_, Phi_phi0, Phi_theta0, Phi_r0], dtype=jnp.float64
            )
            sol = self._solve_backward(y0, t_save, T_geo, ode_args, atol, rtol, max_steps)
        else:
            y0 = jnp.array([p0, e0, Phi_phi0, Phi_theta0, Phi_r0], dtype=jnp.float64)
            sol = self._solve(y0, t_save, T_geo, ode_args, atol, rtol, max_steps)

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
            Om_phi, Om_theta, Om_r = get_fundamental_frequencies(a_abs, p_, e_, x_in)
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
            Om_phi, Om_theta, Om_r = get_fundamental_frequencies(a_abs, p_, e_, x_in)
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

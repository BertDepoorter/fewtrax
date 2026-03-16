Mathematical Background
=======================

This page summarises the physics and mathematics underlying the EMRI waveform
model implemented in fewtrax.  For full derivations see the references at the
bottom of this page.

Extreme-Mass-Ratio Inspirals
-----------------------------

An **extreme-mass-ratio inspiral** (EMRI) consists of a stellar-mass compact
object (mass :math:`\mu \sim 1\text{–}100\,M_\odot`) orbiting a massive black
hole (mass :math:`M \sim 10^5\text{–}10^7\,M_\odot`) with mass ratio
:math:`\varepsilon \equiv \mu/M \ll 1`.  The large mass ratio means

- the small body follows a geodesic of the Kerr background to leading order,
- radiation-reaction drives a slow, *adiabatic* inspiral on a timescale
  :math:`\mathcal{T}_{\rm insp} \sim M/\varepsilon \gg M`.

The gravitational radiation emitted during the inspiral lies in the millihertz
band detectable by LISA.

Kerr Geodesics
--------------

The background spacetime is described by the Kerr metric with mass :math:`M`
and dimensionless spin :math:`a \in [0,1)`.  Bound equatorial geodesics are
characterised by two orbital parameters:

- :math:`p` — the **semilatus rectum** (in units of :math:`M`)
- :math:`e` — the **eccentricity**

The radial turning points are

.. math::

   r_{\rm min} = \frac{pM}{1+e}, \qquad r_{\rm max} = \frac{pM}{1-e}.

Three fundamental frequencies govern the quasi-periodic motion:

.. math::

   \Omega_\phi = \frac{d\Phi_\phi}{dt}, \qquad
   \Omega_\theta = \frac{d\Phi_\theta}{dt}, \qquad
   \Omega_r = \frac{d\Phi_r}{dt},

where :math:`\Phi_\phi`, :math:`\Phi_\theta`, :math:`\Phi_r` are the
accumulated azimuthal, polar, and radial phases respectively.  For equatorial
orbits :math:`\Omega_\theta = \Omega_\phi`.

The **separatrix** :math:`p_{\rm sep}(a,e)` is the boundary between bound and
plunging orbits.  Inspiral terminates when :math:`p \to p_{\rm sep}`.

Adiabatic Inspiral
------------------

In the adiabatic approximation the orbital parameters evolve according to
orbit-averaged energy and angular-momentum flux balance laws:

.. math::

   \frac{dp}{dt} = f_p(p, e; a), \qquad
   \frac{de}{dt} = f_e(p, e; a),

where the flux functions :math:`f_p` and :math:`f_e` are pre-computed from
Teukolsky perturbation theory and stored in the HDF5 data file
``KerrEccEqFluxData.h5``.

fewtrax integrates these equations using the **Tsit5** explicit Runge–Kutta
solver from `diffrax <https://github.com/patrick-kidger/diffrax>`_ with
adaptive step-size control.  Simultaneously, the three orbital phases
:math:`\Phi_\phi`, :math:`\Phi_\theta`, :math:`\Phi_r` are accumulated.
The result is a *sparse* trajectory of :math:`N_{\rm sparse}` points spanning
the observation time :math:`T`.

Teukolsky Mode Amplitudes
--------------------------

The gravitational-wave strain at a detector is expanded in spin-weighted
spherical harmonics :math:`{}_{-2}Y_{\ell m}(\theta_S, \phi_S)`:

.. math::

   h_+ - i\,h_\times =
   \frac{\mu}{d_L}
   \sum_{\ell m k n}
   A_{\ell m k n}(p, e; a)\;
   {}_{-2}Y_{\ell m}(\theta_S,\phi_S)\;
   e^{i\Phi_{\ell m k n}(t)},

where

- :math:`A_{\ell m k n}` are the Teukolsky mode amplitudes, tabulated in
  ``ZNAmps_l10_m10_n55_DS2Outer.h5`` as a function of :math:`(p, e)` for each
  harmonic index :math:`(\ell, m, k, n)`,
- :math:`d_L` is the luminosity distance,
- :math:`\theta_S, \phi_S` are the sky angles of the source,
- the harmonic phase is

  .. math::

     \Phi_{\ell m k n}(t) =
       m\,\Phi_\phi(t) + k\,\Phi_\theta(t) + n\,\Phi_r(t).

The harmonic indices satisfy :math:`\ell \geq 2`, :math:`|m| \leq \ell`, and
:math:`k,n \in \mathbb{Z}`.

Amplitude Interpolation
------------------------

Because evaluating the Teukolsky amplitudes at arbitrary :math:`(p, e)` is
expensive, fewtrax pre-computes them on a sparse grid during the ODE
integration and then uses **B-spline interpolation** (via
`interpax <https://github.com/f0uriest/interpax>`_) to reconstruct the
amplitudes at arbitrary trajectory points.

Mode Selection
--------------

A single EMRI can contribute thousands of :math:`(\ell, m, k, n)` modes.
fewtrax retains only those modes whose relative power exceeds a threshold
:math:`\epsilon_{\rm thresh}` (controlled by ``mode_selection_threshold``):

.. math::

   \frac{|A_{\ell m k n}|^2}{\max_{\ell' m' k' n'} |A_{\ell' m' k' n'}|^2}
   \geq \epsilon_{\rm thresh}.

The default value :math:`\epsilon_{\rm thresh} = 10^{-5}` retains all modes
that contribute more than :math:`0.001\%` of the peak power, giving an
excellent balance between accuracy and computational cost.

Waveform Synthesis
------------------

Given the sparse trajectory and interpolated amplitudes, fewtrax synthesises
the time-domain strain via:

1. **Interpolation** of :math:`A_{\ell m k n}(t)` and
   :math:`\Phi_{\ell m k n}(t)` from the sparse grid to the full time series
   at cadence :math:`\Delta t`.
2. **Coherent summation** of all selected modes, projecting onto
   :math:`h_+` and :math:`h_\times` via the spin-weighted spherical harmonics
   and the antenna pattern functions.

The complete two-polarisation strain returned by
:class:`~fewtrax.waveform.kerr.KerrEccentricEquatorialWaveform` is

.. math::

   h_+(t) = \text{Re}\!\left[
     \frac{\mu}{d_L}
     \sum_{\ell m k n} A_{\ell m k n}(t)\;
     {}_{-2}Y_{\ell m}(\theta_S,\phi_S)\;
     e^{i\Phi_{\ell m k n}(t)}
   \right],

with an analogous expression for :math:`h_\times`.

JAX Implementation Details
--------------------------

fewtrax leverages the JAX ecosystem for high-performance computing:

- **JIT compilation** (``jax.jit`` / ``equinox.filter_jit``) — the full
  trajectory + summation pipeline is compiled to XLA for CPU/GPU execution.
- **Automatic differentiation** (``jax.grad``, ``jax.jacfwd``) — gradients
  propagate through the ODE solver (via diffrax's adjoint methods) and the
  B-spline interpolation.
- **Vectorisation** (``jax.vmap``) — batched evaluation over populations
  without explicit loops.
- **64-bit precision** — ``jax.config.update("jax_enable_x64", True)`` must
  be set before importing fewtrax to achieve the phase accuracy required for
  EMRI parameter estimation.

References
----------

.. [Katz2021] Katz *et al.*, *Phys. Rev. D* **104**, 064047 (2021).
   `arXiv:2104.04582 <https://arxiv.org/abs/2104.04582>`_

.. [Hughes2021] Hughes *et al.*, *Phys. Rev. D* **103**, 104014 (2021).
   `arXiv:2102.02713 <https://arxiv.org/abs/2102.02713>`_

.. [Drasco2006] Drasco & Hughes, *Phys. Rev. D* **73**, 024027 (2006).
   `arXiv:gr-qc/0509101 <https://arxiv.org/abs/gr-qc/0509101>`_

.. [Barack2018] Barack & Pound, *Rep. Prog. Phys.* **82**, 016904 (2019).
   `arXiv:1805.10385 <https://arxiv.org/abs/1805.10385>`_

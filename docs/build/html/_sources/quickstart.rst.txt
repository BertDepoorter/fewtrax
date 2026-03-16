Quickstart
==========

This page walks you through the most common fewtrax workflows.
For a standalone runnable script see ``examples/quickstart.py``.

Before running any code, enable 64-bit floating-point arithmetic — this is
required for the numerical accuracy that EMRI waveform generation demands:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

1. Building the waveform generator
------------------------------------

.. code-block:: python

   from fewtrax import KerrEccentricEquatorialWaveform

   wf = KerrEccentricEquatorialWaveform(
       data_dir="/path/to/few/data",        # or set FEW_DATA_DIR env var
       mode_selection_threshold=1e-5,       # relative power threshold for mode selection
       dense_steps=100,                     # number of sparse trajectory points
   )

2. Defining source parameters
------------------------------

.. code-block:: python

   params = dict(
       M=1e6,          # primary BH mass          [M_sun]
       mu=10.0,        # secondary (compact-object) mass  [M_sun]
       a=0.3,          # dimensionless Kerr spin parameter
       p0=10.0,        # initial semilatus rectum [M]
       e0=0.4,         # initial eccentricity
       x0=1.0,         # prograde equatorial orbit (must be ±1)
       dist=1.0,       # luminosity distance       [Gpc]
       qS=0.2,         # sky polar angle           [rad]
       phiS=0.2,       # sky azimuthal angle       [rad]
       qK=0.8,         # BH spin polar angle       [rad]
       phiK=0.8,       # BH spin azimuthal angle   [rad]
       Phi_phi0=1.0,   # initial azimuthal phase   [rad]
       Phi_theta0=2.0, # initial polar phase       [rad]
       Phi_r0=3.0,     # initial radial phase      [rad]
       T=0.1,          # observation time          [years]
       dt=10.0,        # sampling interval         [s]
   )

3. Generating a waveform
-------------------------

.. code-block:: python

   hp, hx = wf(**params)
   print(hp.shape)   # (N_samples,)

``hp`` and ``hx`` are JAX arrays containing the plus and cross gravitational-wave
polarisations sampled at intervals of ``dt`` seconds.

4. Frequency-domain representation
------------------------------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from fewtrax.summation.modes import to_frequency_domain

   freqs, h_tilde = to_frequency_domain(hp + 1j * hx, dt=params["dt"])
   f_peak = float(freqs[jnp.argmax(jnp.abs(h_tilde))])
   print(f"Peak frequency: {f_peak * 1e3:.3f} mHz")

5. Harmonic frequency tracks
------------------------------

Each EMRI mode :math:`(\ell, m, k, n)` sweeps in frequency as the orbit
decays.  ``get_harmonic_track`` returns the instantaneous frequency along the
inspiral:

.. code-block:: python

   t_track, f_track = wf.get_harmonic_track(
       l=2, m=2, k=0, n=1,
       M=params["M"], mu=params["mu"],
       a=params["a"], p0=params["p0"],
       e0=params["e0"], T=params["T"],
   )

Typical dominant modes: ``(2,2,0,1)``, ``(2,2,0,2)``, ``(3,2,0,1)``.

6. Automatic differentiation
------------------------------

The entire pipeline — trajectory ODE, amplitude interpolation, and mode
summation — is differentiable through JAX.  Below we compute the gradient of
the total accumulated azimuthal phase with respect to the initial semilatus
rectum :math:`p_0`:

.. code-block:: python

   import jax
   from fewtrax.trajectory import EMRIInspiral

   traj = EMRIInspiral(wf._flux_data, a=params["a"])

   def phase_at_end(p0):
       _, _, _, Phi_phi, _, _ = traj(
           p0=p0, e0=params["e0"], T=0.05,
           M=params["M"], mu=params["mu"],
           dense_steps=20,
       )
       valid = jnp.isfinite(Phi_phi)
       return jnp.sum(jnp.where(valid, Phi_phi, 0.0))

   grad_fn = jax.grad(phase_at_end)
   dPhi_dp0 = grad_fn(jnp.float64(params["p0"]))
   print(f"dΦ_φ/dp₀ = {float(dPhi_dp0):.4f} rad/M")

7. Batch evaluation with ``vmap``
-----------------------------------

Use ``jax.vmap`` to evaluate waveforms for a population of sources in a
single vectorised call:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   p0_values = jnp.linspace(8.0, 12.0, 8)

   def single_waveform(p0):
       return wf(**{**params, "p0": p0})

   hp_batch, hx_batch = jax.vmap(single_waveform)(p0_values)
   print(hp_batch.shape)   # (8, N_samples)

8. Accessing the sparse trajectory
------------------------------------

For diagnostics you can retrieve the sparse orbital trajectory before waveform
synthesis:

.. code-block:: python

   result = wf.generate_sparse(**params)
   # Keys: t, p, e, Phi_phi, Phi_theta, Phi_r, amplitudes, modes
   print(result["p"])   # (dense_steps,) array of semilatus rectum values

Next steps
----------

- See :doc:`examples` for detailed Jupyter notebooks covering each component.
- See :doc:`math_background` for the underlying physics.
- See :doc:`api` for the full API reference.

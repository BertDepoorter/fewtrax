Tutorials
=========

The following Jupyter notebooks walk through each component of fewtrax, with
all code cells pre-executed and output visible inline.  They are rendered here
via `nbsphinx <https://nbsphinx.readthedocs.io>`_.

.. note::

   All notebooks call ``jax.config.update("jax_enable_x64", True)`` at the
   top.  This **must** be done before importing any fewtrax module.

.. toctree::
   :maxdepth: 1
   :caption: Notebook tutorials

   notebooks/01_kerr_geodesics
   notebooks/02_emri_trajectory
   notebooks/03_amplitude_interpolation
   notebooks/04_mode_summation
   notebooks/05_full_waveform
   notebooks/06_jax_features

Notebook descriptions
---------------------

.. list-table::
   :header-rows: 1
   :widths: 12 45

   * - Notebook
     - Description
   * - :doc:`01 — Kerr geodesics <notebooks/01_kerr_geodesics>`
     - Computing Boyer-Lindquist frequencies :math:`\Omega_\phi, \Omega_\theta, \Omega_r`
       and the separatrix :math:`p_{\rm sep}(a, e)` for equatorial orbits.
   * - :doc:`02 — EMRI trajectory <notebooks/02_emri_trajectory>`
     - Integrating the adiabatic inspiral ODE and visualising how :math:`(p, e)`
       and the orbital phases evolve toward the separatrix.
   * - :doc:`03 — Amplitude interpolation <notebooks/03_amplitude_interpolation>`
     - Loading and evaluating the Teukolsky mode amplitude B-splines as a
       function of orbital parameters.
   * - :doc:`04 — Mode summation <notebooks/04_mode_summation>`
     - Coherent harmonic mode summation; choosing which :math:`(l,m,k,n)` modes
       contribute more than a given fractional threshold.
   * - :doc:`05 — Full waveform <notebooks/05_full_waveform>`
     - End-to-end waveform generation, time-domain and frequency-domain analysis,
       and comparison with the FEW reference implementation.
   * - :doc:`06 — JAX features <notebooks/06_jax_features>`
     - JIT compilation, automatic differentiation (``jax.grad``), and batched
       parameter sweeps with ``jax.vmap``.

Running locally
---------------

.. code-block:: bash

   pip install "fewtrax[dev]"   # installs jupyter and matplotlib
   jupyter notebook examples/

Pre-executing notebooks for the docs
-------------------------------------

The docs build uses ``nbsphinx_execute = "never"``, meaning that notebooks
must be **committed with output already saved**.  To regenerate output:

.. code-block:: bash

   # Execute all notebooks in-place (requires FEW data files)
   jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb

   # Then rebuild the docs
   cd docs && make html

This separates the (potentially slow, data-dependent) execution from the
(fast, portable) HTML build.  On ReadTheDocs the notebooks are therefore
rendered from the committed output cells without requiring the FEW HDF5 data
files to be present on the build server.

Standalone quickstart script
-----------------------------

``examples/quickstart.py`` is a self-contained script covering the complete
workflow without Jupyter:

.. code-block:: bash

   export FEW_DATA_DIR=/path/to/few/data
   python examples/quickstart.py

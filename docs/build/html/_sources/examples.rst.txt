Examples
========

The ``examples/`` directory contains Jupyter notebooks and a standalone script
that demonstrate each component of fewtrax.

.. list-table::
   :header-rows: 1
   :widths: 10 40

   * - Notebook
     - Description
   * - ``01_kerr_geodesics.ipynb``
     - Computing Kerr geodesic frequencies and the separatrix
   * - ``02_emri_trajectory.ipynb``
     - Integrating the adiabatic inspiral ODE and visualising the orbital evolution
   * - ``03_amplitude_interpolation.ipynb``
     - Loading and B-spline interpolating Teukolsky mode amplitudes
   * - ``04_mode_summation.ipynb``
     - Coherent harmonic mode summation and mode selection
   * - ``05_full_waveform.ipynb``
     - End-to-end waveform generation, time and frequency domain analysis
   * - ``06_jax_features.ipynb``
     - JIT compilation, automatic differentiation, and ``vmap`` batching

Standalone quickstart script
-----------------------------

``examples/quickstart.py`` is a self-contained script covering the complete
workflow:

.. code-block:: bash

   # Pass the FEW data directory as an argument
   python examples/quickstart.py /path/to/few/data

   # Or set the environment variable
   export FEW_DATA_DIR=/path/to/few/data
   python examples/quickstart.py

The script demonstrates:

1. Building the waveform generator
2. Defining source parameters
3. Generating :math:`h_+` and :math:`h_\times`
4. Computing the frequency-domain strain
5. Extracting harmonic frequency tracks
6. Computing a gradient via ``jax.grad``
7. Plotting all results with matplotlib

Running the notebooks
---------------------

.. code-block:: bash

   pip install "fewtrax[dev]"   # installs jupyter and matplotlib
   jupyter notebook examples/

.. note::

   All notebooks and the quickstart script call
   ``jax.config.update("jax_enable_x64", True)`` at the top.  This must be
   done **before** importing any fewtrax module.

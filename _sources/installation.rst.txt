Installation
============

Requirements
------------

- Python ≥ 3.10
- The FEW HDF5 data files (see :ref:`data-setup` below):

  - ``KerrEccEqFluxData.h5``
  - ``ZNAmps_l10_m10_n55_DS2Outer.h5``

  These are available from the `FastEMRIWaveforms data repository
  <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms>`_.

Installing fewtrax
------------------

**CPU (default)**

.. code-block:: bash

   pip install fewtrax

**GPU (CUDA 12)**

.. code-block:: bash

   pip install "fewtrax[gpu]"

**Development install** (includes tests and Jupyter notebooks)

.. code-block:: bash

   pip install "fewtrax[dev]"

**From source**

.. code-block:: bash

   git clone https://github.com/<your-org>/fewtrax
   cd fewtrax
   pip install -e ".[dev]"

.. _data-setup:

Data setup
----------

fewtrax reads the FEW HDF5 data files at runtime.
Point to them in one of three ways (checked in this order):

1. **Pass** ``data_dir=`` **to the waveform constructor**

   .. code-block:: python

      from fewtrax import KerrEccentricEquatorialWaveform

      wf = KerrEccentricEquatorialWaveform(data_dir="/path/to/few/data")

2. **Set the** ``FEW_DATA_DIR`` **environment variable**

   .. code-block:: bash

      export FEW_DATA_DIR=/path/to/few/data

   We recommend storing personal paths in a ``.env`` file and loading them
   with `python-dotenv <https://github.com/theskumar/python-dotenv>`_:

   .. code-block:: bash

      # .env  (never commit this file)
      FEW_DATA_DIR=/path/to/few/data

   .. code-block:: python

      from dotenv import load_dotenv
      load_dotenv()

3. **Place the files in** ``~/.fewtrax/data/``

   fewtrax will look there automatically as a last resort.

If `FastEMRIWaveforms <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms>`_
is installed, fewtrax also tries its internal file-manager cache.

Verifying the installation
---------------------------

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

   from fewtrax import KerrEccentricEquatorialWaveform
   print("fewtrax imported successfully")

Building the documentation locally
------------------------------------

.. code-block:: bash

   pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme myst-parser
   cd docs
   make html
   open build/html/index.html

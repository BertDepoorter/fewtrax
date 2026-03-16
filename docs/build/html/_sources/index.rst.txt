.. fewtrax documentation master file, created by
   sphinx-quickstart on Sun Mar 15 23:15:47 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


fewtrax
=======

JAX implementation of the **KerrEccentricEquatorial** EMRI waveform model
from `FastEMRIWaveforms <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms>`_,
with full support for JIT compilation, automatic differentiation, and batched
evaluation via ``vmap``.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Background

   math_background

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

.. automodule:: fewtrax
   :members:

"""
fewtrax – JAX implementation of EMRI waveforms.

This package provides a fully differentiable, JIT-compilable, and
vectorisable (vmap) implementation of the EMRI waveform model based on
FastEMRIWaveforms (FEW).  The primary model is
``KerrEccentricEquatorial``: a quasi-adiabatic inspiral in the Kerr
spacetime with eccentric, equatorial orbits.

Modules
-------
fewtrax.utils         – Physical constants, geodesics, harmonics, splines
fewtrax.data          – HDF5 data loading and JAX array construction
fewtrax.trajectory    – ODE-based orbital trajectory (diffrax)
fewtrax.amplitude     – Teukolsky amplitude interpolation
fewtrax.summation     – Harmonic mode summation
fewtrax.waveform      – High-level waveform interface

Quick start
-----------
>>> from fewtrax import KerrEccentricEquatorialWaveform
>>> wf = KerrEccentricEquatorialWaveform(data_dir="/path/to/few/data")
>>> h = wf(M=1e6, mu=10.0, a=0.3, p0=10.0, e0=0.4, T=0.1, dt=10.0)

See ``examples/quickstart.py`` for a complete usage example.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fewtrax")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

from fewtrax.waveform.kerr import KerrEccentricEquatorialWaveform

__all__ = [
    "KerrEccentricEquatorialWaveform",
    "__version__",
]

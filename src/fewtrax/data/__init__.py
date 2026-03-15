"""Data loading utilities for fewtrax.

Provides functions to locate and load the HDF5 data files distributed
with FastEMRIWaveforms (FEW), constructing JAX-compatible interpolators
for in-GPU evaluation during trajectory integration and amplitude
computation.
"""

from fewtrax.data.loader import (
    FluxData,
    AmplitudeData,
    find_few_data_dir,
    load_flux_data,
    load_amplitude_data,
)

__all__ = [
    "FluxData",
    "AmplitudeData",
    "find_few_data_dir",
    "load_flux_data",
    "load_amplitude_data",
]

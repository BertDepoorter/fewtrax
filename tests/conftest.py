"""Shared fixtures for fewtrax tests.

The tests require the FEW data files.  The path is read from:
1. The environment variable ``FEW_DATA_DIR``
2. The FEW package file manager (if FEW is installed)
3. ``~/.fewtrax/data/``

If the data directory cannot be found, tests that require it are skipped.
"""

import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp

# Use 64-bit floats throughout
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Data directory fixture
# ---------------------------------------------------------------------------

def _find_data_dir():
    """Try to locate the FEW data directory."""
    from pathlib import Path

    # Check environment variable
    env = os.environ.get("FEW_DATA_DIR")
    if env and (Path(env) / "KerrEccEqFluxData.h5").exists():
        return env

    # Try FEW's file manager
    try:
        from few.utils.globals import get_file_manager
        fm = get_file_manager()
        fp = fm.get_file("KerrEccEqFluxData.h5", raise_on_error=False)
        if fp is not None:
            return str(Path(fp).parent)
    except Exception:
        pass

    # Fallback default
    default = Path.home() / ".fewtrax" / "data"
    if (default / "KerrEccEqFluxData.h5").exists():
        return str(default)

    return None


@pytest.fixture(scope="session")
def data_dir():
    """Path to FEW data directory (session-scoped)."""
    path = _find_data_dir()
    if path is None:
        pytest.skip(
            "FEW data directory not found. "
            "Set FEW_DATA_DIR or install FastEMRIWaveforms."
        )
    return path


@pytest.fixture(scope="session")
def flux_data(data_dir):
    """Pre-loaded flux data (session-scoped, shared across tests)."""
    from fewtrax.data import load_flux_data
    return load_flux_data(data_dir)


@pytest.fixture(scope="session")
def amp_data(data_dir):
    """Pre-loaded amplitude data (session-scoped)."""
    from fewtrax.data import load_amplitude_data
    return load_amplitude_data(data_dir)


@pytest.fixture(scope="session")
def waveform_gen(data_dir):
    """Pre-built waveform generator (session-scoped)."""
    from fewtrax import KerrEccentricEquatorialWaveform
    return KerrEccentricEquatorialWaveform(data_dir=data_dir)


# ---------------------------------------------------------------------------
# Standard EMRI parameter set
# ---------------------------------------------------------------------------

@pytest.fixture
def emri_params():
    """A standard set of EMRI parameters used across tests."""
    return dict(
        M=1e6,        # primary mass [Msun]
        mu=10.0,      # secondary mass [Msun]
        a=0.3,        # BH spin
        p0=10.0,      # initial semilatus rectum [M]
        e0=0.4,       # initial eccentricity
        x0=1.0,       # prograde equatorial
        T=0.1,        # observation time [years]
        dt=10.0,      # sampling interval [s]
        dist=1.0,     # luminosity distance [Gpc]
        qS=0.2, phiS=0.2,
        qK=0.8, phiK=0.8,
        Phi_phi0=1.0, Phi_theta0=2.0, Phi_r0=3.0,
    )

"""Shared utilities for fewtrax vs FastEMRIWaveforms comparison scripts.

Environment
-----------
Load ``FEW_DATA_DIR`` from the project-root ``.env`` file (via python-dotenv)
or from the shell environment.  Never hardcode paths here.
"""

from __future__ import annotations

import os
import resource
import sys
import time
import contextlib
from pathlib import Path
from typing import Optional

import numpy as np

# Load .env from the repository root (two levels up from this file)
try:
    from dotenv import load_dotenv
    _repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment


# ---------------------------------------------------------------------------
# Data directory discovery
# ---------------------------------------------------------------------------

def find_data_dir(override: Optional[str] = None) -> str:
    """Return path to FEW HDF5 data files.

    Search order:
    1. ``override`` argument
    2. ``FEW_DATA_DIR`` environment variable (populated from ``.env``)
    3. FEW package file manager (if FEW is installed)
    """
    candidates: list[str] = []
    if override is not None:
        candidates.append(override)
    env = os.environ.get("FEW_DATA_DIR")
    if env:
        candidates.append(env)

    for path in candidates:
        if (Path(path) / "KerrEccEqFluxData.h5").exists():
            return path

    # Try FEW package file manager as last resort
    try:
        from few.utils.globals import get_file_manager
        fm = get_file_manager()
        fp = fm.get_file("KerrEccEqFluxData.h5", raise_on_error=False)
        if fp is not None:
            return str(Path(fp).parent)
    except Exception:
        pass

    raise FileNotFoundError(
        "FEW data directory not found.\n"
        "  • Set FEW_DATA_DIR in the project .env file, or\n"
        "  • Export FEW_DATA_DIR in your shell, or\n"
        "  • Pass --data-dir to the script."
    )


# ---------------------------------------------------------------------------
# Standard parameter sets
# ---------------------------------------------------------------------------

PARAMS_DEFAULT = dict(
    M=1e6, mu=10.0, a=0.3,
    p0=10.0, e0=0.4, x0=1.0,
    dist=1.0,
    qS=0.2, phiS=0.2,
    qK=0.8, phiK=0.8,
    Phi_phi0=1.0, Phi_theta0=2.0, Phi_r0=3.0,
    T=1.0, dt=10.0,
)

# Suite covering a range of spins, eccentricities, and mass ratios
PARAM_SUITE = [
    dict(label="low-spin",     M=1e6, mu=10.0, a=0.1,  p0=10.0, e0=0.3,  x0=1.0,
         dist=1.0, qS=0.2, phiS=0.2, qK=0.8, phiK=0.8,
         Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0, T=0.1, dt=10.0),
    dict(label="mid-spin",     M=1e6, mu=10.0, a=0.9,  p0=10.0, e0=0.4,  x0=1.0,
         dist=1.0, qS=0.2, phiS=0.2, qK=0.8, phiK=0.8,
         Phi_phi0=1.0, Phi_theta0=2.0, Phi_r0=3.0, T=0.1, dt=10.0),
    dict(label="high-spin",    M=1e6, mu=10.0, a=0.98,  p0=9.0,  e0=0.3,  x0=1.0,
         dist=1.0, qS=0.5, phiS=1.0, qK=0.5, phiK=1.0,
         Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0, T=0.1, dt=10.0),
    dict(label="near-extreme", M=1e6, mu=10.0, a=0.998,  p0=8.0,  e0=0.2,  x0=1.0,
         dist=1.0, qS=0.3, phiS=0.5, qK=0.3, phiK=0.5,
         Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0, T=0.05, dt=10.0),
    dict(label="high-ecc",     M=1e6, mu=10.0, a=0.5,  p0=12.0, e0=0.75,  x0=1.0,
         dist=1.0, qS=0.4, phiS=1.5, qK=0.4, phiK=1.5,
         Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0, T=0.1, dt=10.0),
    dict(label="low-ecc",      M=1e6, mu=10.0, a=0.3,  p0=11.0, e0=0.3, x0=1.0,
         dist=1.0, qS=0.2, phiS=0.2, qK=0.8, phiK=0.8,
         Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0, T=0.1, dt=10.0),
    dict(label="heavy-mu",     M=5e6, mu=50.0, a=0.5,  p0=10.0, e0=0.4,  x0=1.0,
         dist=2.0, qS=0.2, phiS=0.2, qK=0.8, phiK=0.8,
         Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0, T=0.05, dt=10.0),
]


# ---------------------------------------------------------------------------
# Waveform metrics
# ---------------------------------------------------------------------------

def overlap(h1: np.ndarray, h2: np.ndarray) -> float:
    """Normalised inner product (overlap) in the flat (white-noise) metric.

    Returns a value in [0, 1].  Arrays are trimmed to the shorter length.
    """
    n = min(len(h1), len(h2))
    a = np.asarray(h1[:n], dtype=complex)
    b = np.asarray(h2[:n], dtype=complex)
    norm_a = np.sqrt(np.vdot(a, a).real)
    norm_b = np.sqrt(np.vdot(b, b).real)
    if norm_a < 1e-40 or norm_b < 1e-40:
        return 0.0
    return float(np.abs(np.vdot(a, b)) / (norm_a * norm_b))


def mismatch(h1: np.ndarray, h2: np.ndarray) -> float:
    """Mismatch = 1 − overlap."""
    return 1.0 - overlap(h1, h2)


def rms_relative_error(ref: np.ndarray, test: np.ndarray) -> float:
    """RMS(test − ref) / RMS(ref)."""
    n = min(len(ref), len(test))
    r = np.asarray(ref[:n], dtype=float)
    t = np.asarray(test[:n], dtype=float)
    rms_ref = np.sqrt(np.mean(r**2))
    if rms_ref < 1e-40:
        return float("nan")
    return float(np.sqrt(np.mean((t - r)**2)) / rms_ref)


def phase_error_rad(phi1: np.ndarray, phi2: np.ndarray) -> tuple[float, float]:
    """(mean |Δφ|, max |Δφ|) in radians, on a common time grid."""
    n = min(len(phi1), len(phi2))
    diff = np.abs(np.asarray(phi1[:n]) - np.asarray(phi2[:n]))
    return float(np.mean(diff)), float(np.max(diff))


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timer(label: str = "", verbose: bool = True):
    """Context manager: print wall-clock time of the enclosed block."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  {label}: {elapsed:.3f} s")


def repeat_timer(fn, n_warmup: int = 1, n_repeat: int = 5) -> tuple[float, float]:
    """Return (mean, std) wall-clock seconds over ``n_repeat`` calls.

    ``n_warmup`` calls are made first and excluded from the statistics.
    """
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def get_cpu_memory_mb() -> float:
    """Return current process RSS (resident set size) in MiB.

    Uses ``resource.getrusage`` (macOS/Linux, no extra dependency).
    On macOS ``ru_maxrss`` is in bytes; on Linux it is in KiB.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)          # bytes → MiB
    return ru / 1024                        # KiB → MiB


@contextlib.contextmanager
def memory_tracker(label: str = "", verbose: bool = True):
    """Context manager that reports peak RSS increase of the enclosed block.

    Example::

        with memory_tracker("build waveform"):
            h = gen(...)

    Prints::

        build waveform: ΔRSS +34.2 MiB  (peak 1247.8 MiB)
    """
    mem_before = get_cpu_memory_mb()
    yield
    mem_after = get_cpu_memory_mb()
    delta = mem_after - mem_before
    if verbose:
        print(f"  {label}: ΔRSS {delta:+.1f} MiB  (peak {mem_after:.1f} MiB)")


def block_jax(arrays) -> None:
    """Block until JAX arrays are fully computed (handles tuples/lists)."""
    if hasattr(arrays, "block_until_ready"):
        arrays.block_until_ready()
    elif isinstance(arrays, (tuple, list)):
        for a in arrays:
            block_jax(a)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def print_header(title: str, width: int = 72) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table(rows: list[tuple], headers: list[str], col_widths: list[int]) -> None:
    """Print a fixed-width ASCII table."""
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        cells = [str(v)[:w].ljust(w) for v, w in zip(row, col_widths)]
        print("  ".join(cells))

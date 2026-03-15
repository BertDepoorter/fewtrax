"""HDF5 data loading and JAX interpolator construction.

Flux data (``KerrEccEqFluxData.h5``) is loaded using the **ELQ convention**:
the stored :math:`\\dot{E}` and :math:`\\dot{L}` grids are normalised by the
leading-order Peters (1964) PN expressions, which are smooth and free of
separatrix singularities.  At ODE run-time the stored ratio is multiplied by
the PN function to recover the physical flux, and then the (Ė, L̇) → (ṗ, ė)
Jacobian is applied using JAX automatic differentiation.

Amplitude data (``ZNAmps_l10_m10_n55_DS2Outer.h5``) stores bicubic B-spline
coefficients (``multispline`` format).  These are evaluated at query points
using ``scipy.interpolate.bisplev``, which can be called from the amplitude
interpolation module.  The coefficients themselves are stored in the
:class:`AmplitudeData` container as raw numpy arrays.

Data directory discovery (in order):
1. Explicit ``data_dir`` argument.
2. ``FEW_DATA_DIR`` environment variable.
3. FEW package file-manager cache.
4. ``~/.fewtrax/data/``.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass  # kept for AmplitudeData only
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import equinox as eqx

from fewtrax.utils.splines import CubicSpline3D

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate helpers needed at load time (pure numpy)
# ---------------------------------------------------------------------------

_ALPHA_FLUX = 0.5
_BETA_FLUX = 2.0
_ALPHA_AMP = 1.0 / 3.0
_BETA_AMP = 3.0
_ESEP = 0.25
_EMAX = 0.9
_DELTAPMIN = 0.001
_DELTAPMAX = 9.001


def _separatrix_numpy(a_arr, e_arr, x_arr, tol=1e-12):
    """Scalar/array separatrix via Brent root-finding (numpy)."""
    from scipy.optimize import brentq

    def _poly(p, a, e):
        return (
            a**4 * (-3 - 2*e + e**2)**2
            + p**2 * (-6 - 2*e + p)**2
            - 2*a**2 * (1+e) * p * (14 + 2*e**2 + 3*p - e*p)
        )

    scalar = not hasattr(a_arr, "__len__")
    a_arr = np.atleast_1d(np.asarray(a_arr, dtype=float))
    e_arr = np.atleast_1d(np.asarray(e_arr, dtype=float))
    x_arr = np.atleast_1d(np.asarray(x_arr, dtype=float))
    out = np.empty(len(a_arr))
    for i, (a, e, x) in enumerate(zip(a_arr, e_arr, x_arr)):
        if a == 0.0:
            out[i] = 6.0 + 2.0*e
        elif a * x > 0:
            out[i] = brentq(_poly, 1+e, 6+2*e, args=(a, e), xtol=tol)
        else:
            out[i] = brentq(_poly, 6+2*e, 5+e+4*np.sqrt(1+e), args=(a, e), xtol=tol)
    return out[0] if scalar else out


def _apex_A_numpy(u, w, z, alpha, beta):
    """Region-A inverse map: (u, w, z) → (a, p, e) [numpy]."""
    chi_max = (1 - (-0.999))**(1/3)
    chi_min = (1 - 0.999)**(1/3)
    chi = chi_min + z * (chi_max - chi_min)
    a = 1.0 - chi**3

    check = z + u**beta * (1 - z)
    sgn = np.sign(check)
    Secc = _ESEP + (_EMAX - _ESEP) * sgn * np.sqrt(sgn * check)
    e = Secc * w
    a_abs = np.abs(a)
    x = np.sign(a); x[x == 0] = 1.0
    pLSO = _separatrix_numpy(a_abs, e, x)
    p = (pLSO + _DELTAPMIN) + (_DELTAPMAX - _DELTAPMIN) * (
        np.exp(np.abs(u)**(1.0/alpha) * np.log(2.0)) - 1.0
    )
    return a, p, e


def _apex_B_numpy(U, W, Z):
    """Region-B inverse map: (U, W, Z) → (a, p, e) [numpy]."""
    chi_max = (1 - (-0.999))**(1/3)
    chi_min = (1 - 0.999)**(1/3)
    chi = chi_min + Z * (chi_max - chi_min)
    a = 1.0 - chi**3
    e = W * _EMAX
    a_abs = np.abs(a)
    x = np.sign(a); x[x == 0] = 1.0
    pLSO = _separatrix_numpy(a_abs, e, x)
    DELTAPMIN_B = 9.0
    PMAX_B = 200.0
    p = (DELTAPMIN_B**-0.5 - U * (DELTAPMIN_B**-0.5 - (PMAX_B - pLSO)**-0.5))**-2 + pLSO
    return a, p, e


# ---------------------------------------------------------------------------
# Leading-order PN normalization functions (Peters 1964)
# ---------------------------------------------------------------------------

def _PN_Edot(p, e):
    """Leading-order GW energy flux (Peters 1964), dimensionless."""
    one_me2 = (1.0 - e**2)**1.5
    return (32.0/5.0) * p**(-5) * one_me2 * (1.0 + 73.0/24.0*e**2 + 37.0/96.0*e**4)


def _PN_Ldot(p, e):
    """Leading-order GW angular-momentum flux (Peters 1964), dimensionless."""
    one_me2 = (1.0 - e**2)**1.5
    return (32.0/5.0) * p**(-3.5) * one_me2 * (1.0 + 7.0/8.0*e**2)


# JAX versions used at ODE runtime
def _PN_Edot_jax(p, e):
    one_me2 = (1.0 - e**2)**1.5
    return (32.0/5.0) * p**(-5) * one_me2 * (1.0 + 73.0/24.0*e**2 + 37.0/96.0*e**4)


def _PN_Ldot_jax(p, e):
    one_me2 = (1.0 - e**2)**1.5
    return (32.0/5.0) * p**(-3.5) * one_me2 * (1.0 + 7.0/8.0*e**2)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class FluxData(eqx.Module):
    r"""Container for the Kerr eccentric equatorial flux interpolators.

    Stores the ratio ``Edot_GR / Edot_PN`` and ``Ldot_GR / Ldot_PN``
    (ELQ convention), where the PN normalisation is the leading-order
    Peters (1964) expression.  At ODE run-time the ratio is multiplied by
    the PN function to recover the physical flux; the Jacobian
    (Ė, L̇) → (ṗ, ė) is then applied via JAX automatic differentiation.

    Inherits from :class:`equinox.Module` so it is a valid JAX pytree and
    can be used inside JIT-compiled functions and vmapped code.

    Attributes
    ----------
    Edot_A, Ldot_A : CubicSpline3D
        Energy- and angular-momentum-flux ratio interpolators for Region A.
        Inputs are normalised coordinates (u, w, z) ∈ [0, 1]³.
    Edot_B, Ldot_B : CubicSpline3D
        Same for Region B.
    """

    Edot_A: CubicSpline3D
    Ldot_A: CubicSpline3D
    Edot_B: CubicSpline3D
    Ldot_B: CubicSpline3D


@dataclass
class AmplitudeData:
    r"""Container for Teukolsky mode amplitude data.

    The amplitude HDF5 file stores bicubic B-spline coefficients in the
    ``multispline``/``scipy.bisplev`` format.  They are kept as raw numpy
    arrays and evaluated at query points using ``scipy.interpolate.bisplev``.

    Attributes
    ----------
    coeffs_A : np.ndarray, shape (n_z, n_modes, 2, n_coeffs_A)
        B-spline coefficients for Region A.  Axis 2 indexes [real, imag].
    coeffs_B : np.ndarray or None
        Same for Region B.
    u_knots_A, w_knots_A : np.ndarray
        B-spline knot sequences (including boundary repetitions) for Region A.
    u_knots_B, w_knots_B : np.ndarray or None
        Same for Region B.
    z_knots_A, z_knots_B : np.ndarray
        Spin-grid values used for linear interpolation between z-slices.
    l_arr, m_arr, k_arr, n_arr : np.ndarray of int
        Mode index arrays, shape (n_modes,).
    """

    coeffs_A: np.ndarray
    coeffs_B: Optional[np.ndarray]
    u_knots_A: np.ndarray
    w_knots_A: np.ndarray
    u_knots_B: Optional[np.ndarray]
    w_knots_B: Optional[np.ndarray]
    z_knots_A: np.ndarray
    z_knots_B: Optional[np.ndarray]
    l_arr: np.ndarray
    m_arr: np.ndarray
    k_arr: np.ndarray
    n_arr: np.ndarray

    @property
    def n_modes(self) -> int:
        return len(self.l_arr)


# ---------------------------------------------------------------------------
# Data directory discovery
# ---------------------------------------------------------------------------

def find_few_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    """Locate the FEW data directory.

    Searches (in order): explicit argument → ``FEW_DATA_DIR`` env var →
    FEW package file-manager cache → ``~/.fewtrax/data/``.

    Parameters
    ----------
    data_dir : str or Path, optional
        Explicit data directory override.

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
    """
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    env = os.environ.get("FEW_DATA_DIR")
    if env:
        candidates.append(Path(env))
    try:
        from few.utils.globals import get_file_manager
        fm = get_file_manager()
        fp = fm.get_file("KerrEccEqFluxData.h5", raise_on_error=False)
        if fp is not None:
            candidates.append(Path(fp).parent)
    except Exception:
        pass
    candidates.append(Path.home() / ".fewtrax" / "data")

    for p in candidates:
        if p.is_dir() and (p / "KerrEccEqFluxData.h5").exists():
            log.info("Using FEW data directory: %s", p)
            return p

    raise FileNotFoundError(
        "FEW data directory not found. "
        "Pass data_dir= explicitly or set the FEW_DATA_DIR environment variable."
    )


# ---------------------------------------------------------------------------
# Flux data loader
# ---------------------------------------------------------------------------

def load_flux_data(
    data_dir: Optional[str | Path] = None,
    downsample: Optional[list] = None,
) -> FluxData:
    r"""Load Kerr eccentric equatorial flux data (ELQ convention).

    Reads ``KerrEccEqFluxData.h5`` and builds JAX 3-D cubic-spline
    interpolators for the PN-normalised energy and angular-momentum flux
    ratios.  The normalisation uses the leading-order Peters (1964)
    expressions, which have no separatrix singularity.

    At ODE run-time, the physical fluxes are recovered by multiplying the
    stored ratio by the PN function.  The Jacobian
    (Ė, L̇) → (ṗ, ė) is then applied via JAX automatic differentiation
    inside :class:`~fewtrax.trajectory.EMRIInspiral`.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``KerrEccEqFluxData.h5``.
    downsample : list of two 3-tuples, optional
        ``[(dU_A, dW_A, dZ_A), (dU_B, dW_B, dZ_B)]`` downsampling factors.

    Returns
    -------
    FluxData
    """
    data_path = find_few_data_dir(data_dir)
    fp = data_path / "KerrEccEqFluxData.h5"
    if downsample is None:
        downsample = [(1, 1, 1), (1, 1, 1)]
    ds_A, ds_B = downsample

    with h5py.File(fp, "r") as f:
        rA = f["regionA"]
        NU_A = int(rA.attrs["NU"]); NW_A = int(rA.attrs["NW"]); NZ_A = int(rA.attrs["NZ"])
        u_A = np.linspace(0, 1, NU_A)[::ds_A[0]]
        w_A = np.linspace(0, 1, NW_A)[::ds_A[1]]
        z_A = np.linspace(0, 1, NZ_A)[::ds_A[2]]
        Edot_A = rA["Edot"][()][::ds_A[0], ::ds_A[1], ::ds_A[2]]
        Ldot_A = rA["Ldot"][()][::ds_A[0], ::ds_A[1], ::ds_A[2]]

        rB = f["regionB"]
        NU_B = int(rB.attrs["NU"]); NW_B = int(rB.attrs["NW"]); NZ_B = int(rB.attrs["NZ"])
        u_B = np.linspace(0, 1, NU_B)[::ds_B[0]]
        w_B = np.linspace(0, 1, NW_B)[::ds_B[1]]
        z_B = np.linspace(0, 1, NZ_B)[::ds_B[2]]
        Edot_B = rB["Edot"][()][::ds_B[0], ::ds_B[1], ::ds_B[2]]
        Ldot_B = rB["Ldot"][()][::ds_B[0], ::ds_B[1], ::ds_B[2]]

    log.info(
        "Flux grids — Region A: (%d, %d, %d),  Region B: (%d, %d, %d)",
        u_A.size, w_A.size, z_A.size, u_B.size, w_B.size, z_B.size,
    )

    # Compute PN normalisation on the grid points for each region
    def _pn_normalise(u_g, w_g, z_g, Edot_raw, Ldot_raw, is_A: bool):
        uu, ww, zz = np.meshgrid(u_g, w_g, z_g, indexing="ij")
        uu = uu.ravel(); ww = ww.ravel(); zz = zz.ravel()
        if is_A:
            _, p_g, e_g = _apex_A_numpy(uu, ww, zz, _ALPHA_FLUX, _BETA_FLUX)
        else:
            _, p_g, e_g = _apex_B_numpy(uu, ww, zz)
        Epn = _PN_Edot(p_g, e_g)
        Lpn = _PN_Ldot(p_g, e_g)
        shape = (len(u_g), len(w_g), len(z_g))
        with np.errstate(divide="ignore", invalid="ignore"):
            r_E = np.where(Epn > 0, Edot_raw.ravel() / Epn, 1.0).reshape(shape)
            r_L = np.where(Lpn > 0, Ldot_raw.ravel() / Lpn, 1.0).reshape(shape)
        # Replace any residual NaN/Inf (should not occur with Peters PN)
        r_E = np.where(np.isfinite(r_E), r_E, 1.0)
        r_L = np.where(np.isfinite(r_L), r_L, 1.0)
        return r_E, r_L

    log.info("Normalising Region A flux …")
    rE_A, rL_A = _pn_normalise(u_A, w_A, z_A, Edot_A, Ldot_A, True)
    log.info("Normalising Region B flux …")
    rE_B, rL_B = _pn_normalise(u_B, w_B, z_B, Edot_B, Ldot_B, False)

    return FluxData(
        Edot_A=CubicSpline3D(u_A, w_A, z_A, rE_A),
        Ldot_A=CubicSpline3D(u_A, w_A, z_A, rL_A),
        Edot_B=CubicSpline3D(u_B, w_B, z_B, rE_B),
        Ldot_B=CubicSpline3D(u_B, w_B, z_B, rL_B),
    )


# ---------------------------------------------------------------------------
# Amplitude data loader
# ---------------------------------------------------------------------------

def load_amplitude_data(
    data_dir: Optional[str | Path] = None,
    filename: str = "ZNAmps_l10_m10_n55_DS2Outer.h5",
) -> AmplitudeData:
    r"""Load Teukolsky mode amplitude data.

    Reads the amplitude HDF5 file and returns an :class:`AmplitudeData`
    container holding the raw B-spline coefficients.  The coefficients are
    in ``scipy.interpolate.bisplev`` format and are evaluated at query
    points in :class:`~fewtrax.amplitude.interp.AmplitudeInterpolator`.

    Parameters
    ----------
    data_dir : str or Path, optional
        FEW data directory.
    filename : str
        Amplitude HDF5 file.  Default: ``ZNAmps_l10_m10_n55_DS2Outer.h5``.

    Returns
    -------
    AmplitudeData
    """
    data_path = find_few_data_dir(data_dir)
    fp = data_path / filename

    log.info("Loading amplitude data from %s", fp)

    with h5py.File(fp, "r") as fh:
        lmax = int(fh.attrs.get("lmax", 10))
        mmax = int(fh.attrs.get("mmax", 10))
        nmax = int(fh.attrs.get("nmax", 55))

        rA = fh["regionA"]
        coeffs_A = rA["CoeffsRegionA"][()]   # (n_z, n_modes, 2, n_coeffs)
        u_knots_A = rA["u_knots"][()]
        w_knots_A = rA["w_knots"][()]
        z_knots_A = rA["z_knots"][()]

        has_B = "regionB" in fh
        if has_B:
            rB = fh["regionB"]
            coeffs_B = rB["CoeffsRegionB"][()]
            u_knots_B = rB["u_knots"][()]
            w_knots_B = rB["w_knots"][()]
            z_knots_B = rB["z_knots"][()]
        else:
            coeffs_B = u_knots_B = w_knots_B = z_knots_B = None

    # Generate mode index arrays from lmax, mmax, nmax (equatorial: k=0)
    l_arr, m_arr, k_arr, n_arr = _generate_mode_arrays(lmax, mmax, nmax)
    n_modes_file = coeffs_A.shape[1]
    assert len(l_arr) == n_modes_file, (
        f"Mode count mismatch: generated {len(l_arr)}, file has {n_modes_file}"
    )

    log.info(
        "Amplitude: %d modes, %d z-slices, Region B = %s",
        n_modes_file, len(z_knots_A), has_B,
    )

    return AmplitudeData(
        coeffs_A=coeffs_A,
        coeffs_B=coeffs_B,
        u_knots_A=u_knots_A,
        w_knots_A=w_knots_A,
        u_knots_B=u_knots_B,
        w_knots_B=w_knots_B,
        z_knots_A=z_knots_A,
        z_knots_B=z_knots_B,
        l_arr=l_arr,
        m_arr=m_arr,
        k_arr=k_arr,
        n_arr=n_arr,
    )


def _generate_mode_arrays(lmax: int, mmax: int, nmax: int):
    """Generate (l, m, k, n) index arrays for KerrEccentricEquatorial (k=0).

    Replicates FEW's ``m0sort`` ordering: m=0 modes come first (in ascending
    l, n order), followed by m≠0 modes in their original enumeration order.
    This matches the row ordering of the ``CoeffsRegionA/B`` arrays in the
    HDF5 amplitude file.
    """
    modes = []
    for l in range(2, lmax + 1):
        for m in range(0, min(mmax, l) + 1):
            for n in range(-nmax, nmax + 1):
                modes.append((l, m, 0, n))

    # Apply FEW's m0sort: m=0 modes first, then m≠0 in original order
    m0_modes = [(l, m, k, n) for l, m, k, n in modes if m == 0]
    mpos_modes = [(l, m, k, n) for l, m, k, n in modes if m != 0]
    modes = m0_modes + mpos_modes

    l_arr = np.array([t[0] for t in modes], dtype=np.int32)
    m_arr = np.array([t[1] for t in modes], dtype=np.int32)
    k_arr = np.array([t[2] for t in modes], dtype=np.int32)
    n_arr = np.array([t[3] for t in modes], dtype=np.int32)
    return l_arr, m_arr, k_arr, n_arr

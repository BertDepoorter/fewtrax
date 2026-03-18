"""HDF5 data loading and JAX interpolator construction.

Flux data (``KerrEccEqFluxData.h5``) is loaded using the **pex convention**
(matching FEW's default): the stored :math:`\\dot{E}` and :math:`\\dot{L}`
grids are converted to :math:`\\dot{p}` and :math:`\\dot{e}` at load time
using the analytical Kerr Jacobian, then normalised by the separatrix-dependent
PN functions.  At ODE run-time the stored ratio is multiplied by the PN
function to recover the physical flux directly without any Jacobian inversion.

The leading-order Peters (1964) PN functions (:func:`_PN_Edot_jax`,
:func:`_PN_Ldot_jax`) are still exported for backward compatibility.
The separatrix-dependent pex PN functions (:func:`_pdot_PN_jax`,
:func:`_edot_PN_jax`) are used in the ODE right-hand side.

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


# Separatrix-dependent pex PN normalisation (matches FEW's convention)

def _pdot_PN(p, e, r_isco, p_sep):
    """Leading-order ṗ PN factor with separatrix-dependent denominator (numpy)."""
    denom = (p - r_isco) ** 2 - (p_sep - r_isco) ** 2
    one_me2 = (1.0 - e * e) ** 1.5
    return 8.0 * one_me2 * (8.0 + 7.0 * e * e) / (5.0 * p * denom)


def _edot_PN(p, e, r_isco, p_sep):
    """Leading-order ė PN factor with separatrix-dependent denominator (numpy)."""
    denom = (p - r_isco) ** 2 - (p_sep - r_isco) ** 2
    one_me2 = (1.0 - e * e) ** 1.5
    return one_me2 * (304.0 + 121.0 * e * e) / (15.0 * p * p * denom)


def _pdot_PN_jax(p, e, r_isco, p_sep):
    """Leading-order ṗ PN factor (JAX, ODE runtime)."""
    denom = (p - r_isco) ** 2 - (p_sep - r_isco) ** 2
    one_me2 = (1.0 - e ** 2) ** 1.5
    return 8.0 * one_me2 * (8.0 + 7.0 * e ** 2) / (5.0 * p * denom)


def _edot_PN_jax(p, e, r_isco, p_sep):
    """Leading-order ė PN factor (JAX, ODE runtime)."""
    denom = (p - r_isco) ** 2 - (p_sep - r_isco) ** 2
    one_me2 = (1.0 - e ** 2) ** 1.5
    return one_me2 * (304.0 + 121.0 * e ** 2) / (15.0 * p * p * denom)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class FluxData(eqx.Module):
    r"""Container for the Kerr eccentric equatorial flux interpolators.

    Stores the dimensionless ratios ``pdot_GR / pdot_PN`` and
    ``edot_GR / edot_PN`` (pex convention, matching FEW's default).  The PN
    normalisation uses separatrix-dependent functions that cancel the physical
    pole in the flux near the ISCO, giving smoother interpolants.

    At ODE run-time the physical derivatives are recovered by multiplying the
    stored ratio by the PN function; no Jacobian inversion is required.

    Inherits from :class:`equinox.Module` so it is a valid JAX pytree and
    can be used inside JIT-compiled functions and vmapped code.

    Attributes
    ----------
    pdot_A, edot_A : CubicSpline3D
        Semi-latus rectum and eccentricity rate ratio interpolators for
        Region A.  Inputs are normalised coordinates (u, w, z) ∈ [0, 1]³.
    pdot_B, edot_B : CubicSpline3D
        Same for Region B.
    """

    pdot_A: CubicSpline3D
    edot_A: CubicSpline3D
    pdot_B: CubicSpline3D
    edot_B: CubicSpline3D


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
# Grid densification helper
# ---------------------------------------------------------------------------

def _densify_grid(
    u: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    rp: np.ndarray,
    re: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Densify a flux-ratio grid using E(3) cubic splines (multispline).

    Builds ``multispline.TricubicSpline`` instances with E(3) boundary
    conditions on the original grid, then evaluates them on a ``factor``-times
    denser uniform grid.  The denser grid is then passed to
    ``interpax`` C²-splines, whose boundary-condition error is
    O((h/factor)⁴) ≈ (1/factor)⁴ × smaller than on the original grid.

    Parameters
    ----------
    u, w, z : np.ndarray, 1-D
        Original coordinate axes (uniform, [0, 1]).
    rp, re : np.ndarray, shape (Nu, Nw, Nz)
        Flux ratio arrays on the original grid.
    factor : int
        Densification factor (must be ≥ 2).

    Returns
    -------
    u_d, w_d, z_d, rp_d, re_d : np.ndarray
        Denser grid axes and flux-ratio arrays.
    """
    from multispline.spline import TricubicSpline

    spl_rp = TricubicSpline(u, w, z, rp)
    spl_re = TricubicSpline(u, w, z, re)

    Nu_d = (len(u) - 1) * factor + 1
    Nw_d = (len(w) - 1) * factor + 1
    Nz_d = (len(z) - 1) * factor + 1
    u_d = np.linspace(float(u[0]), float(u[-1]), Nu_d)
    w_d = np.linspace(float(w[0]), float(w[-1]), Nw_d)
    z_d = np.linspace(float(z[0]), float(z[-1]), Nz_d)

    uu, ww, zz = np.meshgrid(u_d, w_d, z_d, indexing="ij")
    shape = (Nu_d, Nw_d, Nz_d)
    rp_d = spl_rp(uu.ravel(), ww.ravel(), zz.ravel()).reshape(shape)
    re_d = spl_re(uu.ravel(), ww.ravel(), zz.ravel()).reshape(shape)
    return u_d, w_d, z_d, rp_d, re_d


# ---------------------------------------------------------------------------
# Flux data loader
# ---------------------------------------------------------------------------

def load_flux_data(
    data_dir: Optional[str | Path] = None,
    downsample: Optional[list] = None,
    densify_factor: int = 1,
) -> FluxData:
    r"""Load Kerr eccentric equatorial flux data in pex convention.

    Reads ``KerrEccEqFluxData.h5``, converts (Ė, L̇) → (ṗ, ė) via the
    analytical Kerr Jacobian, then normalises by the separatrix-dependent PN
    functions to obtain dimensionless ratios ṗ/ṗ_PN and ė/ė_PN.

    By default the pex-ratio grids are densified by ``densify_factor`` (default
    3) using E(3) cubic splines from ``multispline`` before being stored in
    JAX ``interpax`` C² splines.  This eliminates the ~1.6e-6 relative
    interpolation error that would otherwise accumulate as a phase drift of
    ~2.6 rad yr⁻¹, causing the waveform overlap to collapse for T ≳ 0.1 yr.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``KerrEccEqFluxData.h5``.
    downsample : list of two 3-tuples, optional
        ``[(dU_A, dW_A, dZ_A), (dU_B, dW_B, dZ_B)]`` downsampling factors
        applied before densification.  Default ``[(1,1,1),(1,1,1)]`` keeps
        the full grid.
    densify_factor : int, optional
        Grid densification factor applied after ELQ→pex conversion.
        ``densify_factor=3`` (default) inserts 2 extra points between every
        original grid point in each dimension, reducing the C²-vs-E(3)
        boundary-condition interpolation error by ~(1/3)⁴ ≈ 80×.
        Set to 1 to skip densification.

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

    def _pex_normalise(u_g, w_g, z_g, Edot_raw, Ldot_raw, is_A: bool):
        """Convert (Edot, Ldot) grid to pex convention (pdot/pdot_PN, edot/edot_PN).

        Matches FEW's convention exactly:
        - x_g = sign(a_g): prograde for a>0, retrograde for a<0
        - Ldot is multiplied by x_g before the Jacobian (FEW sign convention)
        - PN functions use (|a|, e, x_g) so retrograde ISCO is computed correctly
        """
        from fewtrax.utils.jacobian import ELdot_to_pedot_grid

        uu, ww, zz = np.meshgrid(u_g, w_g, z_g, indexing="ij")
        uu = uu.ravel(); ww = ww.ravel(); zz = zz.ravel()
        if is_A:
            a_g, p_g, e_g = _apex_A_numpy(uu, ww, zz, _ALPHA_FLUX, _BETA_FLUX)
        else:
            a_g, p_g, e_g = _apex_B_numpy(uu, ww, zz)

        # Match FEW: x_g = sign(a_g); prograde for a>0, retrograde for a<0
        x_g = np.sign(a_g)
        x_g[x_g == 0] = 1.0

        # FEW flips Ldot by x_g before passing to the Jacobian
        Ldot_signed = Ldot_raw.ravel() * x_g

        # Step 1: Convert (Edot, Ldot) → (pdot, edot) via analytical Jacobian
        log.info("  Converting ELdot → pedot via JAX Jacobian …")
        pdot_g, edot_g = ELdot_to_pedot_grid(
            np.abs(a_g), p_g, e_g, x_g,
            Edot_raw.ravel(), Ldot_signed,
        )

        # Step 2: Compute separatrix-dependent PN normalisation
        # Uses (|a|, e, x_g) so retrograde ISCO is computed correctly
        r_isco_g = _separatrix_numpy(np.abs(a_g), np.zeros_like(e_g), x_g)
        p_sep_g = _separatrix_numpy(np.abs(a_g), e_g, x_g)

        pdot_pn = _pdot_PN(p_g, e_g, r_isco_g, p_sep_g)
        edot_pn = _edot_PN(p_g, e_g, r_isco_g, p_sep_g)

        shape = (len(u_g), len(w_g), len(z_g))
        with np.errstate(divide="ignore", invalid="ignore"):
            rp = np.where(np.abs(pdot_pn) > 0, pdot_g / pdot_pn, 1.0).reshape(shape)
            re = np.where(np.abs(edot_pn) > 0, edot_g / edot_pn, 0.0).reshape(shape)
        rp = np.where(np.isfinite(rp), rp, 1.0)
        re = np.where(np.isfinite(re), re, 0.0)
        return rp, re

    log.info("Computing pex convention for Region A …")
    rp_A, re_A = _pex_normalise(u_A, w_A, z_A, Edot_A, Ldot_A, True)
    log.info("Computing pex convention for Region B …")
    rp_B, re_B = _pex_normalise(u_B, w_B, z_B, Edot_B, Ldot_B, False)

    if densify_factor > 1:
        log.info(
            "Densifying flux grids by factor %d using E(3) cubic splines …",
            densify_factor,
        )
        u_A, w_A, z_A, rp_A, re_A = _densify_grid(u_A, w_A, z_A, rp_A, re_A, densify_factor)
        u_B, w_B, z_B, rp_B, re_B = _densify_grid(u_B, w_B, z_B, rp_B, re_B, densify_factor)
        log.info(
            "Dense grids — Region A: (%d, %d, %d),  Region B: (%d, %d, %d)",
            u_A.size, w_A.size, z_A.size, u_B.size, w_B.size, z_B.size,
        )

    return FluxData(
        pdot_A=CubicSpline3D(u_A, w_A, z_A, rp_A),
        edot_A=CubicSpline3D(u_A, w_A, z_A, re_A),
        pdot_B=CubicSpline3D(u_B, w_B, z_B, rp_B),
        edot_B=CubicSpline3D(u_B, w_B, z_B, re_B),
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

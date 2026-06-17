"""Unused cubic-spline wrappers, preserved from ``fewtrax.utils.splines``.

These four classes had no call sites anywhere in the package, tests, examples,
or comparison scripts and were moved here to slim the public ``splines`` module.
Only ``TricubicSplineE3`` / ``BatchedTricubicSplineE3`` remain in
``fewtrax/utils/splines.py`` (they back the flux and amplitude tables).

This file is an archive: it is not imported by the package.  To reuse a class,
copy it back into ``fewtrax/utils/splines.py`` (or import directly from here).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from interpax import Interpolator1D, Interpolator2D, Interpolator3D


class CubicSpline1D:
    r"""1-D cubic spline via interpax.

    Parameters
    ----------
    x : array_like, shape (N,)
        Strictly increasing coordinate values.
    f : array_like, shape (N,)
        Function values at ``x``.
    method : str
        Interpolation method passed to interpax.  Default ``"cubic2"``
        (C² natural cubic spline).
    """

    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        method: str = "cubic2",
    ):
        self._interp = Interpolator1D(
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(f, dtype=jnp.float64),
            method=method,
            extrap=False,
        )

    def __call__(self, x: float) -> float:
        """Evaluate the spline at ``x``."""
        return self._interp(x)


class CubicSpline2D:
    r"""2-D bicubic spline via interpax.

    Parameters
    ----------
    x, y : array_like, shape (Nx,), (Ny,)
        Coordinate grids.
    f : array_like, shape (Nx, Ny)
        Function values on the grid.
    method : str
        Interpolation method.  Default ``"cubic2"``.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        f: np.ndarray,
        method: str = "cubic2",
    ):
        self._interp = Interpolator2D(
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(f, dtype=jnp.float64),
            method=method,
            extrap=False,
        )

    def __call__(self, x: float, y: float) -> float:
        """Evaluate the spline at ``(x, y)``."""
        return self._interp(x, y)


class CubicSpline3D:
    r"""3-D tricubic spline via interpax.

    Parameters
    ----------
    x, y, z : array_like
        Coordinate grids.
    f : array_like, shape (Nx, Ny, Nz)
        Function values on the grid.
    method : str
        Interpolation method.  Default ``"cubic2"`` (C² natural spline).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        f: np.ndarray,
        method: str = "cubic2",
    ):
        self._interp = Interpolator3D(
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(z, dtype=jnp.float64),
            jnp.asarray(f, dtype=jnp.float64),
            method=method,
            extrap=False,
        )

    def __call__(self, x: float, y: float, z: float) -> float:
        """Evaluate the spline at ``(x, y, z)``."""
        return self._interp(x, y, z)


class PerModeSpline2D:
    r"""Collection of 2-D splines, one per harmonic mode.

    Stores a separate :class:`CubicSpline2D` for the real and imaginary
    parts of each mode amplitude.  This avoids building a large 3-D
    array when the z-grid (spin) dimension has only a small number of
    points and we want mode-specific sparse interpolation.

    Parameters
    ----------
    x, y : array_like
        Coordinate grids (e.g. u-knots, w-knots).
    f_real : array_like, shape (N_modes, Nx, Ny)
        Real parts of the mode amplitudes on the grid.
    f_imag : array_like, shape (N_modes, Nx, Ny)
        Imaginary parts.
    method : str
        Interpolation method.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        f_real: np.ndarray,
        f_imag: np.ndarray,
        method: str = "cubic2",
    ):
        n_modes = f_real.shape[0]
        self._real = [
            CubicSpline2D(x, y, f_real[i], method=method)
            for i in range(n_modes)
        ]
        self._imag = [
            CubicSpline2D(x, y, f_imag[i], method=method)
            for i in range(n_modes)
        ]
        self.n_modes = n_modes

    def __call__(self, x: float, y: float) -> jnp.ndarray:
        """Return complex amplitudes for all modes at ``(x, y)``."""
        real = jnp.stack([spl(x, y) for spl in self._real])
        imag = jnp.stack([spl(x, y) for spl in self._imag])
        return real + 1j * imag

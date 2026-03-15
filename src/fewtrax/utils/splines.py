"""JAX-compatible cubic spline utilities.

Thin wrappers around ``interpax`` providing a consistent interface for
1-D, 2-D, and 3-D cubic spline interpolation that is fully compatible
with JAX JIT compilation, ``vmap``, and automatic differentiation.

The interpax library (https://github.com/f0uriest/interpax) stores spline
coefficients as JAX arrays in an equinox ``Module``, enabling all JAX
transforms.

Usage example
-------------
>>> from fewtrax.utils.splines import CubicSpline3D
>>> import numpy as np, jax.numpy as jnp
>>> x = np.linspace(0, 1, 20)
>>> y = np.linspace(0, 1, 25)
>>> z = np.linspace(0, 1, 15)
>>> f = np.random.randn(20, 25, 15)
>>> spl = CubicSpline3D(x, y, z, f)
>>> val = spl(0.3, 0.7, 0.5)   # scalar query
>>> # batch query via vmap
>>> import jax
>>> vals = jax.vmap(spl)(jnp.linspace(0, 1, 100),
...                      jnp.ones(100) * 0.5,
...                      jnp.ones(100) * 0.5)
"""

from __future__ import annotations

from typing import Optional
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

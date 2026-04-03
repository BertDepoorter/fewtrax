"""JAX-compatible cubic spline utilities.

Thin wrappers around ``interpax`` providing a consistent interface for
1-D, 2-D, and 3-D cubic spline interpolation that is fully compatible
with JAX JIT compilation, ``vmap``, and automatic differentiation.

The interpax library (https://github.com/f0uriest/interpax) stores spline
coefficients as JAX arrays in an equinox ``Module``, enabling all JAX
transforms.

:class:`TricubicSplineE3` provides an alternative 3-D evaluator that
accepts pre-computed coefficients from ``multispline.TricubicSpline``,
which uses E(3) (not-a-knot) boundary conditions matching FEW's convention.
The coefficient computation happens once at construction time via multispline;
evaluation is pure JAX and fully differentiable.

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
import equinox as eqx
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


class TricubicSplineE3(eqx.Module):
    r"""3-D tricubic spline with E(3) boundary conditions, JAX-differentiable.

    Uses piecewise cubic polynomial coefficients computed by
    ``multispline.TricubicSpline`` (E(3) / not-a-knot end conditions),
    matching FEW's interpolation convention exactly.  Only the *evaluation*
    is JAX-traced; the expensive coefficient computation is done once at
    construction time via multispline/numpy and stored as static arrays.

    All JAX transforms — :func:`jax.jit`, :func:`jax.grad`,
    :func:`jax.jacfwd`, :func:`jax.vmap` — work through :meth:`__call__`.

    Parameters
    ----------
    u, w, z : array_like, 1-D, strictly increasing
        Uniform coordinate axes (each must have constant spacing).
    coeffs : array_like, shape ``(Nu-1, Nw-1, Nz-1, 4, 4, 4)``
        Piecewise cubic polynomial coefficients extracted from a
        ``multispline.TricubicSpline`` instance (see :meth:`from_multispline`).
        The expected convention is that for cell ``(i, j, k)`` the value is

        .. math::

            f(u, w, z) = \sum_{a,b,c=0}^{3}
                C_{i,j,k,a,b,c}\,
                (u - u_i)^a\,(w - w_j)^b\,(z - z_k)^c

        where :math:`u_i = u_0 + i \cdot \Delta u` and coordinates are in
        the same physical units as the axis arrays (not normalised to
        ``[0, 1]``).  Verify the layout of your multispline build against
        this convention before use.

    Notes
    -----
    To build from grid values directly, use the :meth:`from_multispline`
    class method, which calls ``multispline.TricubicSpline`` internally::

        spl = TricubicSplineE3.from_multispline(u, w, z, values)

    To supply pre-extracted coefficients::

        from multispline.spline import TricubicSpline
        ms = TricubicSpline(u, w, z, values)
        coeffs = np.asarray(ms.c)          # verify shape and convention
        spl = TricubicSplineE3(u, w, z, coeffs)
    """

    # Coefficient array — the only traced pytree leaf.
    coeffs: jnp.ndarray  # (Nu-1, Nw-1, Nz-1, 4, 4, 4)

    # Grid parameters stored as plain floats/ints.
    # Floats are JAX array leaves; ints are marked static so they can be
    # used in concrete Python arithmetic inside jit (e.g. as clip bounds).
    _u0: float
    _du: float
    _w0: float
    _dw: float
    _z0: float
    _dz: float
    _Nu_m1: int = eqx.field(static=True)
    _Nw_m1: int = eqx.field(static=True)
    _Nz_m1: int = eqx.field(static=True)

    def __init__(
        self,
        u: np.ndarray,
        w: np.ndarray,
        z: np.ndarray,
        coeffs: np.ndarray,
    ):
        u = np.asarray(u, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        self.coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
        self._u0 = float(u[0])
        self._du = float(u[1] - u[0])
        self._w0 = float(w[0])
        self._dw = float(w[1] - w[0])
        self._z0 = float(z[0])
        self._dz = float(z[1] - z[0])
        self._Nu_m1 = len(u) - 1
        self._Nw_m1 = len(w) - 1
        self._Nz_m1 = len(z) - 1

    @classmethod
    def from_multispline(
        cls,
        u: np.ndarray,
        w: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
    ) -> "TricubicSplineE3":
        """Build from grid values using ``multispline.TricubicSpline``.

        Parameters
        ----------
        u, w, z : array_like, 1-D
            Uniform coordinate axes.
        values : array_like, shape ``(Nu, Nw, Nz)``
            Function values on the grid.

        Returns
        -------
        TricubicSplineE3
        """
        from multispline.spline import TricubicSpline

        ms = TricubicSpline(
            np.asarray(u, dtype=np.float64),
            np.asarray(w, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        # ms.coefficients has shape (Nu-1, Nw-1, 64*(Nz-1)) with the last dim
        # encoding [z_cell, mx, my, mz] in C order (mx slowest, mz fastest).
        raw = np.asarray(ms.coefficients)  # (Nu-1, Nw-1, 64*(Nz-1))
        Nu_m1, Nw_m1 = raw.shape[0], raw.shape[1]
        Nz_m1 = raw.shape[2] // 64
        coeffs = raw.reshape(Nu_m1, Nw_m1, Nz_m1, 4, 4, 4)
        return cls(u, w, z, coeffs)

    def __call__(self, u: float, w: float, z: float) -> float:
        """Evaluate the spline at ``(u, w, z)``.

        Parameters
        ----------
        u, w, z : float
            Query coordinates.  Values outside the grid are clamped to the
            nearest boundary cell (same behaviour as ``extrap=False`` in
            interpax).

        Returns
        -------
        float
            Interpolated value.
        """
        # --- cell indices (clipped to valid range) ---
        iu = jnp.clip(
            jnp.floor((u - self._u0) / self._du).astype(jnp.int32),
            0, self._Nu_m1 - 1,
        )
        iw = jnp.clip(
            jnp.floor((w - self._w0) / self._dw).astype(jnp.int32),
            0, self._Nw_m1 - 1,
        )
        iz = jnp.clip(
            jnp.floor((z - self._z0) / self._dz).astype(jnp.int32),
            0, self._Nz_m1 - 1,
        )

        # --- local coordinates normalised to [0,1] within each cell ---
        # multispline stores coefficients in the normalised frame t=(x-x_i)/dx
        tu = (u - (self._u0 + iu * self._du)) / self._du
        tw = (w - (self._w0 + iw * self._dw)) / self._dw
        tz = (z - (self._z0 + iz * self._dz)) / self._dz

        # --- retrieve the 4×4×4 coefficient block for this cell ---
        c = self.coeffs[iu, iw, iz]  # (4, 4, 4)

        # --- evaluate tricubic polynomial via einsum ---
        # f = Σ_{a,b,c} c[a,b,c] * tu^a * tw^b * tz^c
        ptu = jnp.array([1.0, tu, tu * tu, tu * tu * tu])
        ptw = jnp.array([1.0, tw, tw * tw, tw * tw * tw])
        ptz = jnp.array([1.0, tz, tz * tz, tz * tz * tz])
        return jnp.einsum("abc,a,b,c->", c, ptu, ptw, ptz)

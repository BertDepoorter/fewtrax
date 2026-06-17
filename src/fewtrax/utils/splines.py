"""JAX-differentiable tricubic spline tables for the flux/amplitude grids.

:class:`TricubicSplineE3` is a 3-D evaluator that accepts pre-computed
coefficients from ``multispline.TricubicSpline`` (E(3) / not-a-knot boundary
conditions matching FEW's convention).  Coefficients are built once at
construction; evaluation is pure JAX and fully differentiable.

:class:`BatchedTricubicSplineE3` extends this to a **batched channel axis**
(e.g. one spline per Teukolsky mode sharing a common ``(u, w, z)`` grid):
one cell lookup per query, then a single ``einsum`` over all channels,
replacing the ``N_traj x N_modes`` ``scipy.bisplev`` Python loop.
JIT / vmap / grad compatible.

(Simple 1-D/2-D/3-D interpax wrappers that were unused live in
``legacy/splines_unused.py``.)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import equinox as eqx


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


class BatchedTricubicSplineE3(eqx.Module):
    r"""Batched tricubic E(3) spline sharing one ``(u, w, z)`` grid.

    Stores a coefficient tensor of shape
    ``(B, Nu-1, Nw-1, Nz-1, 4, 4, 4)`` where ``B`` is the leading batch
    axis (e.g. Teukolsky mode index).  A single cell lookup is performed
    for the query point; the polynomial basis is then contracted against
    every batch element in one ``einsum``.  The output of
    :meth:`__call__` has shape ``(B,)``.

    This replaces FEW's ``O(N_traj × N_modes)`` sequential
    ``scipy.interpolate.bisplev`` CPU calls: evaluating one trajectory point
    on ``B`` modes is a single fused kernel here, fully compatible with
    :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

    Parameters
    ----------
    u, w, z : array_like, 1-D, strictly increasing, uniform spacing
        Coordinate axes shared across the batch.
    coeffs : array_like, shape ``(B, Nu-1, Nw-1, Nz-1, 4, 4, 4)``
        Piecewise cubic polynomial coefficients per batch element,
        following the same normalised-cell convention as
        :class:`TricubicSplineE3`.

    Memory notes
    ------------
    For ``B = 400`` modes on a ``Nu = 33, Nw = 10, Nz = 11`` grid the
    coefficient tensor occupies ``400 × 32 × 9 × 10 × 64 × 8 B ≈ 590 MB``.
    Mode selection via power thresholding (see
    :meth:`~fewtrax.amplitude.interp.JAXAmplitudeInterpolator.select_modes`)
    is the recommended way to keep this in range on a single GPU.

    See Also
    --------
    TricubicSplineE3
        Scalar (non-batched) counterpart.
    """

    coeffs: jnp.ndarray  # (B, Nu-1, Nw-1, Nz-1, 4, 4, 4)

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
        coeffs = np.asarray(coeffs, dtype=np.float64)

        Nu_m1, Nw_m1, Nz_m1 = len(u) - 1, len(w) - 1, len(z) - 1
        expected = (coeffs.shape[0], Nu_m1, Nw_m1, Nz_m1, 4, 4, 4)
        if coeffs.shape != expected:
            raise ValueError(
                f"coeffs shape {coeffs.shape} does not match expected "
                f"{expected} for grid ({len(u)}, {len(w)}, {len(z)})."
            )

        # Verify uniform spacing (TricubicSplineE3 assumption)
        for name, ax in (("u", u), ("w", w), ("z", z)):
            d = np.diff(ax)
            if not np.allclose(d, d[0], rtol=1e-10, atol=1e-12):
                raise ValueError(f"{name} axis is not uniformly spaced.")

        self.coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
        self._u0 = float(u[0])
        self._du = float(u[1] - u[0])
        self._w0 = float(w[0])
        self._dw = float(w[1] - w[0])
        self._z0 = float(z[0])
        self._dz = float(z[1] - z[0])
        self._Nu_m1 = Nu_m1
        self._Nw_m1 = Nw_m1
        self._Nz_m1 = Nz_m1

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_grid_values(
        cls,
        u: np.ndarray,
        w: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
    ) -> "BatchedTricubicSplineE3":
        r"""Build from stacked grid values using ``multispline.TricubicSpline``.

        One :class:`multispline.TricubicSpline` is fit per batch element at
        load time (numpy/CPU); the coefficients are then stacked and
        uploaded to the JAX device as a single tensor.

        Parameters
        ----------
        u, w, z : array_like, 1-D, uniform
            Shared coordinate axes.
        values : array_like, shape ``(B, Nu, Nw, Nz)``
            Function values on the grid for each batch element.

        Returns
        -------
        BatchedTricubicSplineE3
        """
        from multispline.spline import TricubicSpline

        u_np = np.asarray(u, dtype=np.float64)
        w_np = np.asarray(w, dtype=np.float64)
        z_np = np.asarray(z, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 4:
            raise ValueError(
                f"values must have shape (B, Nu, Nw, Nz); got {values.shape}."
            )
        B, Nu, Nw, Nz = values.shape
        if (Nu, Nw, Nz) != (len(u_np), len(w_np), len(z_np)):
            raise ValueError(
                f"values grid {(Nu, Nw, Nz)} does not match axes "
                f"{(len(u_np), len(w_np), len(z_np))}."
            )

        Nu_m1, Nw_m1, Nz_m1 = Nu - 1, Nw - 1, Nz - 1
        coeffs = np.empty(
            (B, Nu_m1, Nw_m1, Nz_m1, 4, 4, 4), dtype=np.float64
        )
        for b in range(B):
            ms = TricubicSpline(u_np, w_np, z_np, values[b])
            raw = np.asarray(ms.coefficients)  # (Nu-1, Nw-1, 64*(Nz-1))
            coeffs[b] = raw.reshape(Nu_m1, Nw_m1, Nz_m1, 4, 4, 4)

        return cls(u_np, w_np, z_np, coeffs)

    @classmethod
    def from_complex_grid_values(
        cls,
        u: np.ndarray,
        w: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
    ) -> tuple["BatchedTricubicSplineE3", "BatchedTricubicSplineE3"]:
        r"""Build a pair of real/imag batched splines from complex grid values.

        Convenience wrapper for complex amplitude grids: the real and
        imaginary parts are fit independently (as they must be, since
        ``multispline`` is real-valued) and returned as two separate
        :class:`BatchedTricubicSplineE3` instances.

        Parameters
        ----------
        u, w, z : array_like, 1-D, uniform
        values : array_like, shape ``(B, Nu, Nw, Nz)`` complex

        Returns
        -------
        real_spline, imag_spline : BatchedTricubicSplineE3
            Use ``real_spline(u, w, z) + 1j * imag_spline(u, w, z)`` to
            recover the complex amplitudes.
        """
        values = np.asarray(values)
        if not np.iscomplexobj(values):
            raise ValueError("values must be a complex array.")
        real = cls.from_grid_values(u, w, z, values.real)
        imag = cls.from_grid_values(u, w, z, values.imag)
        return real, imag

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, u: float, w: float, z: float) -> jnp.ndarray:
        r"""Evaluate all batch elements at the scalar query ``(u, w, z)``.

        Parameters
        ----------
        u, w, z : float
            Scalar query coordinates.  Out-of-grid points are clamped to
            the nearest boundary cell (matching :class:`TricubicSplineE3`).
            Use :func:`jax.vmap` over any of the arguments to query many
            points at once.

        Returns
        -------
        jnp.ndarray, shape ``(B,)``
            Interpolated values for every batch element.
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

        # --- local coordinates in the normalised cell frame ---
        tu = (u - (self._u0 + iu * self._du)) / self._du
        tw = (w - (self._w0 + iw * self._dw)) / self._dw
        tz = (z - (self._z0 + iz * self._dz)) / self._dz

        # --- gather the per-batch (4, 4, 4) coefficient blocks ---
        # Advanced indexing on the three cell axes; shape (B, 4, 4, 4).
        c = self.coeffs[:, iu, iw, iz]

        # --- polynomial basis ---
        ptu = jnp.array([1.0, tu, tu * tu, tu * tu * tu])
        ptw = jnp.array([1.0, tw, tw * tw, tw * tw * tw])
        ptz = jnp.array([1.0, tz, tz * tz, tz * tz * tz])

        # f_B = Σ_{a,b,c} c[B,a,b,c] * tu^a * tw^b * tz^c
        return jnp.einsum("Babc,a,b,c->B", c, ptu, ptw, ptz)

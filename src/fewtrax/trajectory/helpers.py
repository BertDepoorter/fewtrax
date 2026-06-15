"""Helper utilities for the trajectory module."""

from __future__ import annotations

import jax.numpy as jnp


def dense_phase_derivs(interp, tau: jnp.ndarray, comp: int):
    r"""First three :math:`\tau`-derivatives of a dense-output component.

    Differentiates the Dopri8 local interpolant analytically.  Each step is a
    degree-6 polynomial of the rescaled time :math:`s = (\tau - t_0)/h`,
    :math:`y(\tau) = y_0 + \sum_i c_i(s)\,k_i` with
    :math:`c_i(s) = s\cdot{\rm polyval}({\rm eval\_coeffs}_i, s)`, so the
    derivatives follow from :func:`jnp.polyder` and a chain-rule factor
    :math:`h^{-n}`.

    Parameters
    ----------
    interp : diffrax DenseInterpolation
        ``sol.interpolation`` from a ``SaveAt(dense=True)`` solve.
    tau : jnp.ndarray
        Scalar query time in geometric units (vmap-compatible).
    comp : int
        State-vector component index (``2`` for :math:`\Phi_\phi`,
        ``4`` for :math:`\Phi_r`).

    Returns
    -------
    d1, d2, d3 : jnp.ndarray
        First, second and third derivatives w.r.t. ``tau``.
    """
    li = interp._get_local_interpolation(tau, True)
    h = li.t1 - li.t0
    s = (tau - li.t0) / h
    eval_coeffs = jnp.asarray(li.eval_coeffs, dtype=h.dtype)  # (14, 6)
    # c_i(s) = s * polyval(eval_coeffs_i, s): pad to degree-6 polyval form
    val_coeffs = jnp.concatenate(
        [eval_coeffs, jnp.zeros((eval_coeffs.shape[0], 1), eval_coeffs.dtype)],
        axis=1,
    )
    poly = li.k[:, comp] @ val_coeffs  # scalar degree-6 polynomial (polyval form)
    d1 = jnp.polyval(jnp.polyder(poly, 1), s) / h
    d2 = jnp.polyval(jnp.polyder(poly, 2), s) / h ** 2
    d3 = jnp.polyval(jnp.polyder(poly, 3), s) / h ** 3
    return d1, d2, d3

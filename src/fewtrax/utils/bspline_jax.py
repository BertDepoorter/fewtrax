"""JAX-native cubic B-spline (bisplev) evaluation primitives.

Implements the Piegl-Tiller triangular algorithm (NURBS Book, Algorithm A2.2)
for the 4 non-zero cubic B-spline basis functions, and a batched 2-D
tensor-product evaluator that replaces ``scipy.interpolate.bisplev`` for the
FEW amplitude data.

The key function is :func:`eval_bisplev_batched`, which evaluates all
``n_modes`` complex amplitudes at a single ``(w, u)`` query point in a single
``jnp.einsum`` call, replacing the nested Python loop over modes and
trajectory points in :class:`~fewtrax.amplitude.interp.AmplitudeInterpolator`.

All functions are fully compatible with ``jax.jit``, ``jax.vmap``, and
``jax.grad``.

Coefficient convention
----------------------
The layout matches ``scipy.interpolate.bisplev``:

    ``c_flat = bisplev_tck[2]``  →  shape ``(N1 * N2,)``

where ``N1 = len(t1) - 4`` and ``N2 = len(t2) - 4`` (cubic degree).  The
coefficients are stored in C (row-major) order::

    c_flat[i * N2 + j]  =  coefficient for  B_i(x) * B_j(y)

with ``x`` (first bisplev argument) indexed by ``i`` and ``y`` (second
argument) by ``j``.  After reshaping to ``(N1, N2)``, the first axis
corresponds to the first knot vector / bisplev argument.

In the FEW amplitude HDF5 file the first argument is always ``w`` (eccentricity
coordinate) and the second is ``u`` (semi-latus rectum coordinate)::

    bisplev(w_query, u_query, (w_knots, u_knots, c_flat, 3, 3))

so :func:`eval_bisplev_batched` should be called as
``eval_bisplev_batched(w, u, coeffs, t1=w_knots, t2=u_knots)``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

_K: int = 3  # cubic B-spline degree (fixed)


# ---------------------------------------------------------------------------
# Span finding
# ---------------------------------------------------------------------------

def _find_span(x: float, t: jnp.ndarray, n: int) -> jnp.ndarray:
    """Return the knot-span index l such that ``t[l] <= x < t[l+1]``.

    Parameters
    ----------
    x : float
        Query coordinate.
    t : (M,) float64
        Full clamped knot vector (non-decreasing; first/last value repeated
        ``_K + 1`` times).
    n : int
        Number of B-spline basis functions = ``len(t) - _K - 1`` (static).

    Returns
    -------
    l : int32 scalar
        Span index, clamped to ``[_K, n - 1]`` so that the 4-element gather
        ``t[l-2 : l+4]`` is always in-bounds.
    """
    l = jnp.searchsorted(t, x, side="right") - 1
    return jnp.clip(l, _K, n - 1).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Cubic basis functions
# ---------------------------------------------------------------------------

def _bspline_basis_cubic(
    x: float,
    t: jnp.ndarray,
    l: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the 4 non-zero cubic B-spline basis functions at ``x``.

    Implements NURBS Book Algorithm A2.2 (Piegl & Tiller), fully unrolled for
    degree ``_K = 3`` with no Python loops at JAX-trace time.

    Safe division (``0/0 → 0``) handles coincident knots at the clamped
    boundary.

    Parameters
    ----------
    x : float
        Query coordinate.
    t : (M,) float64
        Full clamped knot vector.
    l : int32
        Knot-span index from :func:`_find_span`.  Satisfies
        ``t[l] <= x < t[l+1]`` and ``l ∈ [_K, n-1]``.

    Returns
    -------
    N : (4,) float64
        ``N[j] = B_{l-3+j, 3}(x)`` for ``j = 0, 1, 2, 3``.
        Sum of elements is 1 (partition of unity).
    """
    # Gather 6 consecutive knot values: t[l-2], ..., t[l+3]
    t6 = lax.dynamic_slice(t, (l - 2,), (6,))
    # Indices into t6:  t[l+k] = t6[k + 2]
    #   t[l-2] = t6[0],  t[l-1] = t6[1],  t[l] = t6[2]
    #   t[l+1] = t6[3],  t[l+2] = t6[4],  t[l+3] = t6[5]

    # left[j] = x - t[l+1-j],  right[j] = t[l+j] - x,  for j = 1, 2, 3
    l1 = x - t6[2]   # x - t[l]
    l2 = x - t6[1]   # x - t[l-1]
    l3 = x - t6[0]   # x - t[l-2]
    r1 = t6[3] - x   # t[l+1] - x
    r2 = t6[4] - x   # t[l+2] - x
    r3 = t6[5] - x   # t[l+3] - x

    def _div(a: float, b: float) -> float:
        """Safe division: a/b = 0 when b == 0 (handles coincident knots)."""
        return jnp.where(b == 0.0, 0.0, a / b)

    # Degree-0 seed
    N0 = 1.0

    # ---- j = 1 -------------------------------------------------------
    d1 = r1 + l1          # t[l+1] - t[l]
    tmp = _div(N0, d1)
    n0_1 = r1 * tmp        # N[0] after j = 1
    n1_1 = l1 * tmp        # N[1] after j = 1  (= saved)

    # ---- j = 2 -------------------------------------------------------
    tmp = _div(n0_1, r1 + l2)   # right[1] + left[2] = t[l+1] - t[l-1]
    n0_2 = r1 * tmp
    saved = l2 * tmp

    tmp = _div(n1_1, r2 + l1)   # right[2] + left[1] = t[l+2] - t[l]
    n1_2 = saved + r2 * tmp
    n2_2 = l1 * tmp              # N[2] after j = 2  (= saved)

    # ---- j = 3 -------------------------------------------------------
    tmp = _div(n0_2, r1 + l3)   # right[1] + left[3] = t[l+1] - t[l-2]
    n0_3 = r1 * tmp
    saved = l3 * tmp

    tmp = _div(n1_2, r2 + l2)   # right[2] + left[2] = t[l+2] - t[l-1]
    n1_3 = saved + r2 * tmp
    saved = l2 * tmp

    tmp = _div(n2_2, r3 + l1)   # right[3] + left[1] = t[l+3] - t[l]
    n2_3 = saved + r3 * tmp
    n3_3 = l1 * tmp              # N[3] after j = 3  (= saved)

    return jnp.array([n0_3, n1_3, n2_3, n3_3])


# ---------------------------------------------------------------------------
# 2-D batched evaluator
# ---------------------------------------------------------------------------

def eval_bisplev_batched(
    w: float,
    u: float,
    coeffs: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a batched 2-D cubic B-spline at a scalar ``(w, u)`` point.

    Matches ``scipy.interpolate.bisplev(w, u, (t1, t2, c_flat, 3, 3))``
    exactly for every mode, but evaluates all modes in a single
    ``jnp.einsum`` call.

    Parameters
    ----------
    w : float
        Query coordinate for the first bisplev argument (w / eccentricity).
    u : float
        Query coordinate for the second bisplev argument (u / semi-latus).
    coeffs : (n_modes, N1, N2) float64
        Pre-shaped B-spline coefficient matrices.  ``coeffs[m]`` is the
        ``(N1, N2)`` matrix for mode ``m``, where::

            coeffs[m, i, j]  =  c_flat_m[i * N2 + j]
                              =  coefficient for  B_i(w) * B_j(u)

        ``N1 = len(t1) - 4``,  ``N2 = len(t2) - 4``.
    t1 : (len_t1,) float64
        Clamped knot vector for the first axis (w / ``w_knots`` in HDF5).
    t2 : (len_t2,) float64
        Clamped knot vector for the second axis (u / ``u_knots`` in HDF5).

    Returns
    -------
    (n_modes,) float64
        Spline values for all modes at the query point ``(w, u)``.
    """
    n1 = coeffs.shape[1]   # N1_coeffs  (static at JIT time)
    n2 = coeffs.shape[2]   # N2_coeffs  (static at JIT time)
    n_modes = coeffs.shape[0]

    l1 = _find_span(w, t1, n1)
    l2 = _find_span(u, t2, n2)

    B1 = _bspline_basis_cubic(w, t1, l1)   # (4,)  — w basis
    B2 = _bspline_basis_cubic(u, t2, l2)   # (4,)  — u basis

    # Gather the (n_modes, 4, 4) coefficient block centred on (l1, l2).
    # l1 ∈ [3, n1-1]  →  l1-3 ∈ [0, n1-4], l1-3+4 = l1+1 ∈ [4, n1]  ✓
    # l2 ∈ [3, n2-1]  →  l2-3 ∈ [0, n2-4], l2-3+4 = l2+1 ∈ [4, n2]  ✓
    # All start indices must share the same int dtype for dynamic_slice.
    block = lax.dynamic_slice(
        coeffs,
        jnp.array([0, l1 - 3, l2 - 3], dtype=jnp.int32),
        (n_modes, 4, 4),
    )   # (n_modes, 4, 4)

    return jnp.einsum("mij,i,j->m", block, B1, B2)

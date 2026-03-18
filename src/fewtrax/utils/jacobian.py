"""Analytical Jacobian utilities for (Ė, L̇) ↔ (ṗ, ė) conversion.

Provides JAX-differentiable conversion between the two radiation-reaction
conventions used in EMRI trajectory codes:

* **ELQ convention** (fewtrax legacy): store PN-normalised (Ė/Ė_PN, L̇/L̇_PN);
  apply the Jacobian ∂(E,L)/∂(p,e) at ODE run-time.
* **pex convention** (FEW): pre-compute (ṗ/ṗ_PN, ė/ė_PN) at grid points using
  the analytical Jacobian at load time; at ODE run-time just evaluate splines
  multiplied by the PN functions.

Public API
----------
:func:`ELdot_to_pedot_jax`
    (Ė, L̇) → (ṗ, ė) using JAX ``jacfwd`` Jacobian (JIT-able).
:func:`pedot_to_ELdot_jax`
    (ṗ, ė) → (Ė, L̇) using JAX ``jacfwd`` Jacobian (JIT-able).
:func:`ELdot_to_pedot_grid`
    Vectorised (Ė, L̇) → (ṗ, ė) for full flux grids at data-loading time.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from fewtrax.utils.geodesic import (
    kerr_geo_energy_equatorial,
    kerr_geo_angular_momentum_equatorial,
)


# ---------------------------------------------------------------------------
# JAX Jacobian helpers (JIT-able)
# ---------------------------------------------------------------------------

def _EL_of_pe(a_abs: jnp.ndarray, x: jnp.ndarray, pe: jnp.ndarray) -> jnp.ndarray:
    """Return [E, L] as a JAX array given [p, e]."""
    E = kerr_geo_energy_equatorial(a_abs, pe[0], pe[1], x)
    L = kerr_geo_angular_momentum_equatorial(a_abs, pe[0], pe[1], x, E)
    return jnp.array([E, L])


def ELdot_to_pedot_jax(
    a: jnp.ndarray,
    p: jnp.ndarray,
    e: jnp.ndarray,
    x: jnp.ndarray,
    Edot: jnp.ndarray,
    Ldot: jnp.ndarray,
    e_min: float = 1e-4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Convert (Ė, L̇) to (ṗ, ė) using the JAX ``jacfwd`` Jacobian.

    Solves the 2×2 linear system :math:`J \cdot (\dot{p}, \dot{e})^T = (\dot{E}, \dot{L})^T`
    where :math:`J = \partial(E, L)/\partial(p, e)` is computed via
    forward-mode automatic differentiation.

    Parameters
    ----------
    a, p, e, x : JAX scalars
        Spin, semi-latus rectum, eccentricity, inclination sign.
    Edot, Ldot : JAX scalars
        Physical energy and angular-momentum time derivatives.
    e_min : float
        Minimum eccentricity clamped before computing the Jacobian to avoid
        the singular circular-orbit limit.  Default is 1e-4.

    Returns
    -------
    pdot, edot : JAX scalars
    """
    a_abs = jnp.abs(a)
    e_safe = jnp.maximum(jnp.abs(e), e_min)

    J = jax.jacfwd(lambda pe: _EL_of_pe(a_abs, x, pe))(
        jnp.array([p, e_safe])
    )  # shape (2, 2)

    rhs = jnp.array([Edot, Ldot])
    pe_dot = jnp.linalg.solve(J, rhs)

    # Circular-orbit fallback: use ∂E/∂p alone, set edot = 0
    dEdp = J[0, 0]
    pdot_circ = Edot / jnp.where(jnp.abs(dEdp) > 1e-30, dEdp, 1.0)

    use_circ = jnp.abs(e) < e_min
    pdot = jnp.where(use_circ, pdot_circ, pe_dot[0])
    edot = jnp.where(use_circ, jnp.zeros_like(e), pe_dot[1])
    return pdot, edot


def pedot_to_ELdot_jax(
    a: jnp.ndarray,
    p: jnp.ndarray,
    e: jnp.ndarray,
    x: jnp.ndarray,
    pdot: jnp.ndarray,
    edot: jnp.ndarray,
    e_min: float = 1e-4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Convert (ṗ, ė) to (Ė, L̇) using the forward Jacobian (JAX, JIT-able).

    Applies the Jacobian :math:`J = \partial(E, L)/\partial(p, e)` directly:
    :math:`(\dot{E}, \dot{L}) = J \cdot (\dot{p}, \dot{e})`.

    Parameters
    ----------
    a, p, e, x : JAX scalars
        Spin, semi-latus rectum, eccentricity, inclination sign.
    pdot, edot : JAX scalars
        Physical semi-latus rectum and eccentricity time derivatives.
    e_min : float
        Minimum eccentricity clamped to avoid the circular-orbit singularity.

    Returns
    -------
    Edot, Ldot : JAX scalars
    """
    a_abs = jnp.abs(a)
    e_safe = jnp.maximum(jnp.abs(e), e_min)

    J = jax.jacfwd(lambda pe: _EL_of_pe(a_abs, x, pe))(
        jnp.array([p, e_safe])
    )  # shape (2, 2)

    pe_dot = jnp.array([pdot, edot])
    EL_dot = J @ pe_dot
    return EL_dot[0], EL_dot[1]


# ---------------------------------------------------------------------------
# Grid-level (load-time) conversion: ELdot → pedot over full numpy arrays
# ---------------------------------------------------------------------------

def ELdot_to_pedot_grid(
    a_arr: np.ndarray,
    p_arr: np.ndarray,
    e_arr: np.ndarray,
    x_arr: np.ndarray,
    Edot_arr: np.ndarray,
    Ldot_arr: np.ndarray,
    e_min: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flattened (Ė, L̇) arrays to (ṗ, ė) for an entire flux grid.

    Applies :func:`ELdot_to_pedot_jax` point-by-point using JAX ``vmap``
    for efficiency.  Intended for use at data-loading time to pre-compute the
    pex-convention flux tables.

    Parameters
    ----------
    a_arr, p_arr, e_arr, x_arr : np.ndarray, shape (N,)
        Flattened grid of (a, p, e, x) values.
    Edot_arr, Ldot_arr : np.ndarray, shape (N,)
        Physical flux derivatives at each grid point.
    e_min : float
        Minimum eccentricity for the circular-orbit clamping.

    Returns
    -------
    pdot_arr, edot_arr : np.ndarray, shape (N,)
    """
    a_jax = jnp.asarray(a_arr, dtype=jnp.float64)
    p_jax = jnp.asarray(p_arr, dtype=jnp.float64)
    e_jax = jnp.asarray(e_arr, dtype=jnp.float64)
    x_jax = jnp.asarray(x_arr, dtype=jnp.float64)
    Edot_jax = jnp.asarray(Edot_arr, dtype=jnp.float64)
    Ldot_jax = jnp.asarray(Ldot_arr, dtype=jnp.float64)

    def _single(a, p, e, x, Ed, Ld):
        return ELdot_to_pedot_jax(a, p, e, x, Ed, Ld, e_min=e_min)

    pdot_jax, edot_jax = jax.vmap(_single)(
        a_jax, p_jax, e_jax, x_jax, Edot_jax, Ldot_jax
    )
    return np.asarray(pdot_jax), np.asarray(edot_jax)

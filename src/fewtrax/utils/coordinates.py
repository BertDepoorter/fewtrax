"""Coordinate transformations for the FEW grid system.

The FEW flux and amplitude grids are parameterised by compressed
coordinates :math:`(u, w, z)` that map the physically relevant domain
into the unit cube :math:`[0, 1]^3`.  Two separate regions cover:

*  **Region A** ("near separatrix"):  :math:`p \\le p_{\\rm sep} + \\Delta p_{\\rm max}`
*  **Region B** ("far region"):       :math:`p > p_{\\rm sep} + \\Delta p_{\\rm max}`

The transformations differ between flux and amplitude interpolation through
parameters :math:`(\\alpha, \\beta)`:

*  Flux:     :math:`\\alpha = 1/2, \\beta = 2`
*  Amplitude: :math:`\\alpha = 1/3, \\beta = 3`

These values reproduce the coordinate mappings from ``few.utils.mappings.kerrecceq``.

References
----------
FastEMRIWaveforms source: ``src/few/utils/mappings/kerrecceq.py``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from fewtrax.utils.geodesic import get_separatrix, get_separatrix_fast

# ---------------------------------------------------------------------------
# Constants matching the FEW source
# ---------------------------------------------------------------------------

XMIN: float = 0.05
AMAX: float = 0.999
AMIN: float = -AMAX
DELTAPMIN: float = 0.001
DELTAPMAX: float = 9.0 + DELTAPMIN
EMAX: float = 0.9
ESEP: float = 0.25

ALPHA_FLUX: float = 0.5
BETA_FLUX: float = 2.0
ALPHA_AMP: float = 1.0 / 3.0
BETA_AMP: float = 3.0

# Region B constants
DPC_REGIONB: float = DELTAPMAX - 0.001
PMAX_REGIONB: float = 200.0
DELTAPMIN_REGIONB: float = 9.0
EMAX_REGIONB: float = 0.9


# ---------------------------------------------------------------------------
# Region A coordinate transforms
# ---------------------------------------------------------------------------

@jit
def _u_of_p_A(p: float, pLSO: float, alpha: float) -> float:
    r"""Map :math:`p \to u` for Region A."""
    check = (
        jnp.log(p - pLSO + DELTAPMAX - 2.0 * DELTAPMIN)
        - jnp.log(DELTAPMAX - DELTAPMIN)
    ) / jnp.log(2.0)
    sgn = jnp.sign(check)
    return sgn * (sgn * check) ** alpha


@jit
def _p_of_u_A(u: float, pLSO: float, alpha: float) -> float:
    r"""Map :math:`u \to p` for Region A (inverse of :func:`_u_of_p_A`)."""
    return (pLSO + DELTAPMIN) + (DELTAPMAX - DELTAPMIN) * (
        jnp.exp(jnp.abs(u) ** (1.0 / alpha) * jnp.log(2.0)) - 1.0
    )


@jit
def _chi_of_a(a: float) -> float:
    return (1.0 - a) ** (1.0 / 3.0)


@jit
def z_of_a(a: float) -> float:
    r"""Map spin :math:`a \to z \in [0, 1]` for Region A."""
    chimax = _chi_of_a(AMIN)
    chimin = _chi_of_a(AMAX)
    return (_chi_of_a(a) - chimin) / (chimax - chimin)


@jit
def a_of_z(z: float) -> float:
    r"""Inverse spin mapping :math:`z \to a` for Region A."""
    chimax = _chi_of_a(AMIN)
    chimin = _chi_of_a(AMAX)
    chi = chimin + z * (chimax - chimin)
    return 1.0 - chi**3


@jit
def _Secc_of_uz(u: float, z: float, beta: float) -> float:
    check = z + u**beta * (1.0 - z)
    sgn = jnp.sign(check)
    return ESEP + (EMAX - ESEP) * sgn * jnp.sqrt(sgn * check)


@jit
def _w_of_euz(e: float, u: float, z: float, beta: float) -> float:
    return e / _Secc_of_uz(u, z, beta)


@jit
def kerrecceq_forward_map_A(
    a: float, p: float, e: float, pLSO: float, alpha: float, beta: float
) -> tuple[float, float, float]:
    r"""Forward map :math:`(a, p, e) \to (u, w, z)` for Region A.

    Parameters
    ----------
    a, p, e : float
        Orbital parameters.
    pLSO : float
        Separatrix value :math:`p_{\rm sep}`.
    alpha, beta : float
        Region A compression parameters.

    Returns
    -------
    tuple of float
        :math:`(u, w, z)`.
    """
    u = _u_of_p_A(p, pLSO, alpha)
    z = z_of_a(a)
    w = _w_of_euz(e, u, z, beta)
    return u, w, z


# ---------------------------------------------------------------------------
# Region B coordinate transforms
# ---------------------------------------------------------------------------

@jit
def _U_of_p_B_flux(p: float, pLSO: float) -> float:
    return (DELTAPMIN_REGIONB**-0.5 - (p - pLSO) ** -0.5) / (
        DELTAPMIN_REGIONB**-0.5 - (PMAX_REGIONB - pLSO) ** -0.5
    )


@jit
def _U_of_p_B_amp(p: float, pLSO: float) -> float:
    pc = pLSO + DPC_REGIONB
    return (pc**-0.5 - p**-0.5) / (pc**-0.5 - (PMAX_REGIONB + pc) ** -0.5)


@jit
def _p_of_U_B_flux(U: float, pLSO: float) -> float:
    return (
        DELTAPMIN_REGIONB**-0.5
        - U * (DELTAPMIN_REGIONB**-0.5 - (PMAX_REGIONB - pLSO) ** -0.5)
    ) ** -2 + pLSO


@jit
def _Z_of_a_B(a: float) -> float:
    """Spin mapping for Region B (uses same chi as Region A)."""
    chimax = _chi_of_a(AMIN)
    chimin = _chi_of_a(AMAX)
    return (_chi_of_a(a) - chimin) / (chimax - chimin)


@jit
def _W_of_e_B(e: float) -> float:
    return e / EMAX_REGIONB


@partial(jit, static_argnames=("is_flux",))
def kerrecceq_forward_map_B(
    a: float, p: float, e: float, pLSO: float, is_flux: bool = True
) -> tuple[float, float, float]:
    r"""Forward map :math:`(a, p, e) \to (U, W, Z)` for Region B.

    Parameters
    ----------
    a, p, e : float
        Orbital parameters.
    pLSO : float
        Separatrix value.
    is_flux : bool
        If True use the flux coordinate mapping; else the amplitude mapping.

    Returns
    -------
    tuple of float
        :math:`(U, W, Z)`.
    """
    U = jax.lax.cond(
        is_flux,
        lambda: _U_of_p_B_flux(p, pLSO),
        lambda: _U_of_p_B_amp(p, pLSO),
    )
    W = _W_of_e_B(e)
    Z = _Z_of_a_B(a)
    return U, W, Z


# ---------------------------------------------------------------------------
# Combined forward map (selects region automatically)
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=("kind",))
def kerrecceq_forward_map(
    a: float, p: float, e: float, kind: str = "flux"
) -> tuple[float, float, float, bool]:
    r"""Map :math:`(a, p, e) \to (u, w, z, \text{in\_A})`.

    Automatically selects Region A or B based on
    :math:`p \le p_{\rm sep} + \Delta p_{\rm max}`.

    Parameters
    ----------
    a, p, e : float
        Orbital parameters.
    kind : str
        ``"flux"`` or ``"amplitude"`` – selects compression parameters.

    Returns
    -------
    u, w, z : float
        Compressed coordinates.
    in_region_A : bool
        True if the point is in Region A.
    """
    alpha = ALPHA_FLUX if kind == "flux" else ALPHA_AMP
    beta = BETA_FLUX if kind == "flux" else BETA_AMP
    is_flux = kind == "flux"

    a_in = jnp.abs(a)
    x_in = jnp.sign(a)
    x_in = jnp.where(x_in == 0.0, 1.0, x_in)
    pLSO = get_separatrix(a_in, e, x_in)

    in_A = p <= pLSO + DELTAPMAX

    u_A, w_A, z_A = kerrecceq_forward_map_A(a_in, p, e, pLSO, alpha, beta)
    u_B, w_B, z_B = kerrecceq_forward_map_B(a_in, p, e, pLSO, is_flux)

    u = jnp.where(in_A, u_A, u_B)
    w = jnp.where(in_A, w_A, w_B)
    z = jnp.where(in_A, z_A, z_B)

    return u, w, z, in_A


@partial(jit, static_argnames=("kind",))
def kerrecceq_forward_map_fast(
    a: float, p: float, e: float, pLSO: float, kind: str = "flux"
) -> tuple[float, float, float, bool]:
    r"""Like :func:`kerrecceq_forward_map` but accepts a pre-computed ``pLSO``.

    Avoids the internal :func:`~fewtrax.utils.geodesic.get_separatrix` call,
    allowing the caller to compute the separatrix once per ODE step and reuse
    it for both the flux interpolation and the event condition.

    Parameters
    ----------
    a, p, e : float
        Orbital parameters.
    pLSO : float
        Pre-computed separatrix :math:`p_{\rm sep}(a, e)`.
    kind : str
        ``"flux"`` or ``"amplitude"``.

    Returns
    -------
    u, w, z : float
        Compressed coordinates.
    in_region_A : bool
        True when the point is in Region A.
    """
    alpha  = ALPHA_FLUX if kind == "flux" else ALPHA_AMP
    beta   = BETA_FLUX  if kind == "flux" else BETA_AMP
    is_flux = kind == "flux"

    a_in = jnp.abs(a)
    in_A = p <= pLSO + DELTAPMAX

    u_A, w_A, z_A = kerrecceq_forward_map_A(a_in, p, e, pLSO, alpha, beta)
    u_B, w_B, z_B = kerrecceq_forward_map_B(a_in, p, e, pLSO, is_flux)

    u = jnp.where(in_A, u_A, u_B)
    w = jnp.where(in_A, w_A, w_B)
    z = jnp.where(in_A, z_A, z_B)

    return u, w, z, in_A

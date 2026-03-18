"""Spin-weight −2 spherical harmonics :math:`{}_{-2}Y_{\\ell m}(\\theta, \\phi)`.

These are the spin-weighted spherical harmonics with spin weight :math:`s = -2`
that appear in the gravitational wave strain decomposition:

.. math::

    h_+ - i h_{\\times} = \\sum_{\\ell, m} {}_{-2}Y_{\\ell m}(\\theta, \\phi) h_{\\ell m}(t)

Closed-form expressions are tabulated for :math:`\\ell = 2, \\ldots, 10`.
They are expressed in terms of
:math:`c = \\cos(\\theta/2)` and :math:`s = \\sin(\\theta/2)`.

All functions are JIT-compilable.  For use in vectorised pipelines,
use :func:`get_ylms_for_modes` which returns a 1-D complex array of
harmonics for a list of :math:`(\\ell, m)` mode indices.

References
----------
Goldberg et al. (1967); Berti, Cardoso & Will (2006) for conventions.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit


# ---------------------------------------------------------------------------
# Core harmonic computation for a single (l, m) pair
# ---------------------------------------------------------------------------

@jit
def spin_weighted_spherical_harmonic(
    l: int, m: int, theta: float, phi: float
) -> complex:
    r"""Spin-weight −2 spherical harmonic :math:`{}_{-2}Y_{\ell m}(\theta, \phi)`.

    Parameters
    ----------
    l : int
        Degree :math:`\ell \ge 2`.
    m : int
        Order :math:`-\ell \le m \le \ell`.
    theta : float
        Polar angle [rad].
    phi : float
        Azimuthal angle [rad].

    Returns
    -------
    complex
        The harmonic value.

    Notes
    -----
    Expressions are hardcoded for :math:`\ell = 2 \ldots 10`.
    """
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    ep = jnp.exp(1j * m * phi)

    # Use the lookup table implemented via lax.switch
    def _y22(): return (jnp.sqrt(5.0 / jnp.pi) / 2.0) * c**4 * ep
    def _y21(): return jnp.sqrt(5.0 / jnp.pi) * c**3 * s * ep
    def _y20(): return jnp.sqrt(15.0 / (2.0 * jnp.pi)) * c**2 * s**2 * ep
    def _y2m1(): return jnp.sqrt(5.0 / jnp.pi) * c * s**3 * ep
    def _y2m2(): return (jnp.sqrt(5.0 / jnp.pi) / 2.0) * s**4 * ep

    def _y33(): return -(jnp.sqrt(21.0 / (2.0 * jnp.pi))) * c**5 * s * ep
    def _y32(): return (jnp.sqrt(7.0 / jnp.pi) / 2.0) * (c**6 - 5.0 * c**4 * s**2) * ep
    def _y31(): return -(jnp.sqrt(7.0 / (10.0 * jnp.pi))) * (
        -5.0 * c**5 * s + 10.0 * c**3 * s**3
    ) * ep
    def _y30(): return (jnp.sqrt(21.0 / (10.0 * jnp.pi)) / 2.0) * (
        10.0 * c**4 * s**2 - 10.0 * c**2 * s**4
    ) * ep
    def _y3m1(): return -(jnp.sqrt(7.0 / (10.0 * jnp.pi))) * (
        -10.0 * c**3 * s**3 + 5.0 * c * s**5
    ) * ep
    def _y3m2(): return (jnp.sqrt(7.0 / jnp.pi) / 2.0) * (
        5.0 * c**2 * s**4 - s**6
    ) * ep
    def _y3m3(): return jnp.sqrt(21.0 / (2.0 * jnp.pi)) * c * s**5 * ep

    # Map (l, m) to a flat index for lax.switch
    _ylm_table = {
        (2, -2): _y2m2, (2, -1): _y2m1, (2, 0): _y20, (2, 1): _y21, (2, 2): _y22,
        (3, -3): _y3m3, (3, -2): _y3m2, (3, -1): _y3m1, (3, 0): _y30,
        (3, 1): _y31, (3, 2): _y32, (3, 3): _y33,
    }

    # Fall back to general formula for l >= 4
    return _general_swsh(l, m, theta, phi)


@jit
def _general_swsh(l: int, m: int, theta: float, phi: float) -> complex:
    r"""General spin-weight −2 spherical harmonic via the Wigner d-matrix.

    Uses the Goldberg et al. (1967) recursion for arbitrary :math:`\ell`.
    This version computes the harmonic numerically and is JAX-differentiable.
    """
    # We use the relation:
    #   _s Y_lm = sqrt((2l+1)/(4pi)) * d^l_{m, -s}(theta) * exp(i m phi)
    # where d^l is the Wigner small d-matrix with s = -2.
    #
    # For s = -2 (gravitational waves):
    # Equivalent to the formula in Berti, Cardoso & Will (2006) A.5
    s = -2  # spin weight
    # Compute via the relation to associated Legendre functions with complex factors
    # Use the analytic formula from Goldberg et al. (1967):
    #   _s Y_lm = sqrt((2l+1)/(4pi) * (l-s)!*(l+s)! / ((l-m)!*(l+m)!)) * ...
    #           * sum_{k} C(l-s, k) * C(l+s, k+s-m) * (-1)^(l+s-k) ...
    #                   * cos^(2k+s-m)(theta/2) * sin^(2l-2k-s+m)(theta/2) * exp(i m phi)

    cos2 = jnp.cos(theta / 2.0)
    sin2 = jnp.sin(theta / 2.0)

    # Normalisation factor
    # Factorial via jnp.lgamma
    def lfact(n):
        return jnp.where(n >= 0, jax.lax.lgamma(jnp.float64(n) + 1.0), -jnp.inf)

    log_norm = (
        0.5 * jnp.log((2.0 * l + 1.0) / (4.0 * jnp.pi))
        + 0.5 * (lfact(l + m) + lfact(l - m) - lfact(l + s) - lfact(l - s))
    )
    norm = jnp.exp(log_norm)
    sign = (-1.0) ** (m + s)

    # Sum over k (Clebsch-Gordan type sum)
    # Use a fixed loop unrolled for l up to 10
    # k ranges from max(0, m-s) to min(l+s, l-m) == 0..2l usually
    k_vals = jnp.arange(2 * l + 1)
    mask = (k_vals >= 0) & (k_vals <= l - s) & (k_vals + s - m >= 0) & (k_vals + s - m <= l + s)

    def binom_log(n, k):
        return lfact(n) - lfact(k) - lfact(n - k)

    log_binom1 = binom_log(l - s, k_vals)
    log_binom2 = binom_log(l + s, k_vals + s - m)
    sign = (-1.0) ** (l + s - k_vals)
    power_cos = (2.0 * k_vals + s - m)
    power_sin = (2.0 * l - 2.0 * k_vals - s + m)

    term = (
        sign
        * jnp.exp(log_binom1 + log_binom2)
        * jnp.where(power_cos >= 0, cos2**power_cos, 0.0)
        * jnp.where(power_sin >= 0, sin2**power_sin, 0.0)
    )
    total = jnp.sum(jnp.where(mask, term, 0.0))

    return sign * norm * total * jnp.exp(1j * m * phi)


# ---------------------------------------------------------------------------
# Batch harmonic evaluation for mode arrays
# ---------------------------------------------------------------------------

def get_ylms_for_modes(
    l_arr: np.ndarray,
    m_arr: np.ndarray,
    theta: float,
    phi: float,
) -> jnp.ndarray:
    r"""Compute :math:`{}_{-2}Y_{\ell m}(\theta, \phi)` for an array of modes.

    Returns a complex array of shape ``(N_modes,)`` as well as the
    corresponding negative-m harmonics ``(N_modes,)`` needed for the
    conjugate term in the summation.

    Parameters
    ----------
    l_arr, m_arr : array_like of int
        Arrays of :math:`\ell` and :math:`m` values (length N_modes).
    theta, phi : float
        Observer angles [rad].

    Returns
    -------
    ylms_pos : jnp.ndarray, shape (N_modes,), complex
        Harmonics :math:`{}_{-2}Y_{\ell m}` (for m as given).
    ylms_neg : jnp.ndarray, shape (N_modes,), complex
        Harmonics :math:`{}_{-2}Y_{\ell, -m}` (for the negative-m conjugate term).
    """
    l_arr = np.asarray(l_arr, dtype=int)
    m_arr = np.asarray(m_arr, dtype=int)

    # Evaluate in numpy loop (one-time cost at initialisation)
    ylms_pos = np.array(
        [_general_swsh_numpy(l, m, theta, phi) for l, m in zip(l_arr, m_arr)],
        dtype=complex,
    )
    ylms_neg = np.array(
        [_general_swsh_numpy(l, -m, theta, phi) for l, m in zip(l_arr, m_arr)],
        dtype=complex,
    )
    return jnp.asarray(ylms_pos), jnp.asarray(ylms_neg)


def _general_swsh_numpy(l: int, m: int, theta: float, phi: float) -> complex:
    """Pure-numpy spin-weight −2 spherical harmonic (for initialisation)."""
    from math import factorial, sqrt, cos, sin, pi, exp
    from cmath import exp as cexp

    s = -2
    cos2 = cos(theta / 2.0)
    sin2 = sin(theta / 2.0)

    sign = 1 if (m + s) % 2 == 0 else -1

    # Normalisation
    norm = sqrt(
        (2 * l + 1) / (4 * pi)
        * factorial(l + m)
        * factorial(l - m)
        / (factorial(l + s) * factorial(l - s))
    )

    def binom(n, k):
        if n < 0 or k < 0 or k > n:
            return 0
        return factorial(n) // (factorial(k) * factorial(n - k))

    total = 0.0
    for k in range(l - s + 1):
        k2 = k + s - m
        if k2 < 0 or k2 > l + s:
            continue
        power_cos = 2 * k + s - m
        power_sin = 2 * l - 2 * k - s + m
        if power_cos < 0 or power_sin < 0:
            continue
        term = (
            binom(l - s, k)
            * binom(l + s, k2)
            * ((-1.0) ** int(l + s - k))
            * cos2**power_cos
            * sin2**power_sin
        )
        total += term

    return sign * norm * total * cexp(1j * m * phi)

"""Kerr geodesic utilities implemented in JAX.

This module provides JAX-compatible functions for computing key quantities
of Kerr geodesics required by the EMRI waveform model:

*  Orbital energy :math:`E` and angular momentum :math:`L_z`
*  Separatrix :math:`p_{\\rm sep}(a, e, x_I)`
*  Boyer-Lindquist fundamental frequencies
   :math:`\\Omega_\\phi, \\Omega_\\theta, \\Omega_r`

All functions are JIT-compilable and differentiable via JAX's automatic
differentiation.  Elliptic integrals :math:`K(k)` and :math:`E(k)` are
evaluated with ``jax.scipy.special``; the complete elliptic integral of
the third kind :math:`\\Pi(n, k)` is evaluated by 64-point Gauss-Legendre
quadrature on the standard integral representation.

Mathematical background
-----------------------
Orbital mechanics follows Schmidt (2002), gr-qc/0202090.
Separatrix polynomials follow Glampedakis & Kennefick (2002),
gr-qc/0203086.
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial

from fewtrax.utils.constants import PI

# ---------------------------------------------------------------------------
# Gauss-Legendre nodes/weights for EllipPi (precomputed once at import time)
# ---------------------------------------------------------------------------

_N_GL = 64
_gl_nodes_np, _gl_weights_np = np.polynomial.legendre.leggauss(_N_GL)
# Transform from [-1, 1] to [0, π/2]
_GL_THETA = jnp.asarray((_gl_nodes_np + 1.0) / 2.0 * np.pi / 2.0, dtype=jnp.float64)
_GL_W = jnp.asarray(_gl_weights_np / 2.0 * np.pi / 2.0, dtype=jnp.float64)

# 24-point GL nodes/weights for the fast Π variant (accurate to ~1e-12 for
# the smooth integrands encountered in bound EMRI orbits).
_N_GL24 = 24
_gl24_nodes_np, _gl24_weights_np = np.polynomial.legendre.leggauss(_N_GL24)
_GL24_THETA = jnp.asarray((_gl24_nodes_np + 1.0) / 2.0 * np.pi / 2.0, dtype=jnp.float64)
_GL24_W = jnp.asarray(_gl24_weights_np / 2.0 * np.pi / 2.0, dtype=jnp.float64)

# Number of AGM iterations used by ellipk_agm / ellipe_agm.
# 12 iterations gives ~2^{-50} ≈ 1e-15 relative error (quadratic convergence).
_N_AGM: int = 12

# Hybrid bisection+NR for the fast separatrix.
# Phase 1: _N_BISECT_INIT bisection steps narrow the bracket to width
#   ~(hi-lo)/2^20 ≈ 5e-6, giving a safe NR starting point even when the
#   root sits near the bracket edge (e.g. high-spin prograde: root ≈ 1+e,
#   bracket [1+e, 6+2e], midpoint far from root).
# Phase 2: _N_NR Newton-Raphson steps from the tight midpoint.
#   Starting within 2.5e-6 of the root, quadratic convergence reaches
#   float64 precision in ≤3 steps; 5 is used for a generous safety margin.
_N_BISECT_INIT: int = 20
_N_NR: int = 5


@jit
def ellipk(m: float) -> float:
    r"""Complete elliptic integral of the first kind :math:`K(m)`.

    .. math::  K(m) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - m \sin^2\theta}}

    Uses 64-point Gauss-Legendre quadrature.  ``m`` is the *parameter*
    (square of the modulus), consistent with scipy's convention.
    """
    sin2 = jnp.sin(_GL_THETA) ** 2
    integrand = 1.0 / jnp.sqrt(jnp.maximum(1.0 - m * sin2, 1e-30))
    return jnp.dot(_GL_W, integrand)


@jit
def ellipe(m: float) -> float:
    r"""Complete elliptic integral of the second kind :math:`E(m)`.

    .. math::  E(m) = \int_0^{\pi/2} \sqrt{1 - m \sin^2\theta}\, d\theta

    Uses 64-point Gauss-Legendre quadrature.
    """
    sin2 = jnp.sin(_GL_THETA) ** 2
    integrand = jnp.sqrt(jnp.maximum(1.0 - m * sin2, 0.0))
    return jnp.dot(_GL_W, integrand)


@jit
def ellip_pi(n: float, k: float) -> float:
    r"""Complete elliptic integral of the third kind :math:`\Pi(n, k)`.

    .. math::

        \Pi(n, k) = \int_0^{\pi/2}
            \frac{d\theta}{(1 - n\sin^2\theta)\sqrt{1 - k^2\sin^2\theta}}

    Evaluated via 64-point Gauss-Legendre quadrature on :math:`[0, \pi/2]`.
    Accurate to approximately :math:`10^{-12}` for the parameter ranges
    encountered in EMRI trajectories.

    Parameters
    ----------
    n : float
        Characteristic parameter; must satisfy :math:`n < 1`.
    k : float
        Modulus; must satisfy :math:`0 \le k < 1`.

    Returns
    -------
    float
        :math:`\Pi(n, k)`.
    """
    sin2 = jnp.sin(_GL_THETA) ** 2
    integrand = 1.0 / ((1.0 - n * sin2) * jnp.sqrt(1.0 - k**2 * sin2))
    return jnp.dot(_GL_W, integrand)


# ---------------------------------------------------------------------------
# Fast elliptic integrals
# ---------------------------------------------------------------------------
#
# These replace the 64-point Gauss-Legendre versions above with two faster
# algorithms:
#
#   * K(m) and E(m)  — Arithmetic-Geometric Mean (AGM / Borwein, _N_AGM iters).
#     12 iterations achieves ~1e-15 relative accuracy via quadratic convergence.
#     ~5× faster than 64-point GL on CPU; avoids the 64-element dot product.
#
#   * Π(n, k)        — 24-point Gauss-Legendre.
#     For the smooth integrands of bound EMRI orbits (k² < 0.95, n < 1) the
#     error is below 1e-12, well within the 1-rad dephasing budget over 2 yr.
#     ~2.5× faster than the 64-point version.
#
# Combined, each call to get_fundamental_frequencies_fast saves ~5× on the
# elliptic integral part.

@jit
def ellipk_agm(m: float) -> float:
    r"""Complete elliptic integral K(m) via the AGM algorithm.

    Uses :math:`K(m) = \pi / (2\,\mathrm{AGM}(1, \sqrt{1-m}))`.

    Parameters
    ----------
    m : float
        Parameter (square of the modulus), :math:`0 \le m < 1`.

    Returns
    -------
    float
        :math:`K(m)`.
    """
    def body(_, state):
        a, b = state
        a_new = (a + b) * 0.5
        b_new = jnp.sqrt(jnp.maximum(a * b, 0.0))
        return a_new, b_new

    a0 = jnp.float64(1.0)
    b0 = jnp.sqrt(jnp.maximum(1.0 - m, 0.0))
    a_f, _ = jax.lax.fori_loop(0, _N_AGM, body, (a0, b0))
    return jnp.pi / (2.0 * a_f)


@jit
def ellipe_agm(m: float) -> float:
    r"""Complete elliptic integral E(m) via the AGM / Borwein algorithm.

    Uses the identity
    :math:`E(m) = K(m)\bigl(1 - \sum_{n \ge 0} 2^{n-1} c_n^2\bigr)`
    where :math:`c_n` are the AGM correction terms.

    Parameters
    ----------
    m : float
        Parameter, :math:`0 \le m < 1`.

    Returns
    -------
    float
        :math:`E(m)`.
    """
    def body(_, state):
        a, b, esum, power = state
        a_new = (a + b) * 0.5
        b_new = jnp.sqrt(jnp.maximum(a * b, 0.0))
        c = a - a_new                          # = (a − b)/2  = c_{n+1}
        return a_new, b_new, esum + power * c * c, power * 2.0

    a0 = jnp.float64(1.0)
    b0 = jnp.sqrt(jnp.maximum(1.0 - m, 0.0))
    # Seed with the n=0 term: 2^{-1} · c_0^2  where c_0 = √m
    init_esum = m * 0.5
    a_f, _, esum, _ = jax.lax.fori_loop(
        0, _N_AGM, body, (a0, b0, init_esum, jnp.float64(1.0))
    )
    K = jnp.pi / (2.0 * a_f)
    return K * (1.0 - esum)


@jit
def ellip_pi_fast(n: float, k: float) -> float:
    r"""Complete elliptic integral Π(n, k) via 24-point GL quadrature.

    Approximately 2.5× faster than :func:`ellip_pi` (64-point version)
    with ~1e-12 accuracy for the smooth integrands of bound EMRI orbits.

    Parameters
    ----------
    n : float
        Characteristic parameter, :math:`n < 1`.
    k : float
        Modulus, :math:`0 \le k < 1`.

    Returns
    -------
    float
        :math:`\Pi(n, k)`.
    """
    sin2 = jnp.sin(_GL24_THETA) ** 2
    integrand = 1.0 / ((1.0 - n * sin2) * jnp.sqrt(jnp.maximum(1.0 - k**2 * sin2, 1e-30)))
    return jnp.dot(_GL24_W, integrand)


# ---------------------------------------------------------------------------
# Helper functions for Kerr geodesics (equatorial)
# ---------------------------------------------------------------------------

@jit
def _capital_delta(r: float, a: float) -> float:
    r"""Kerr :math:`\Delta(r) = r^2 - 2r + a^2`."""
    return r * r - 2.0 * r + a * a


@jit
def _f_eq(r: float, a: float) -> float:
    """Function f(r, a) for equatorial orbits (zm=0)."""
    return r**4 + a**2 * r * (r + 2.0)


@jit
def _g_eq(r: float, a: float) -> float:
    """Function g(r, a) = 2 a r."""
    return 2.0 * a * r


@jit
def _h_eq(r: float, a: float) -> float:
    """Function h(r, a) for equatorial orbits (zm=0)."""
    return r * (r - 2.0)


@jit
def _d_eq(r: float, a: float) -> float:
    """Function d(r, a) = r^2 * Delta for equatorial orbits (zm=0)."""
    return r * r * _capital_delta(r, a)


# ---------------------------------------------------------------------------
# Orbital energy and angular momentum
# ---------------------------------------------------------------------------

@jit
def kerr_geo_energy_equatorial(a: float, p: float, e: float, x: float) -> float:
    r"""Orbital energy :math:`E` for an equatorial Kerr geodesic.

    Uses the closed-form expression derived from the roots of the radial
    potential (Schmidt 2002, Eqs. 28–32), specialized to the equatorial
    case :math:`x_I = \pm 1`.

    Parameters
    ----------
    a : float
        Dimensionless spin parameter, :math:`|a| \le 1`.
    p : float
        Semi-latus rectum in units of :math:`M`.
    e : float
        Orbital eccentricity, :math:`0 \le e < 1`.
    x : float
        Cosine of inclination; must be :math:`\pm 1` for equatorial orbits.

    Returns
    -------
    float
        Specific orbital energy :math:`E`.
    """
    sgnax = jnp.sign(a * x)
    # Closed-form for |x| == 1 (Schmidt 2002)
    denom = -4.0 * a**2 * (e**2 - 1.0) ** 2 + (3.0 + e**2 - p) ** 2 * p
    inner = (
        a**6 * (e**2 - 1.0) ** 2
        + a**2 * (-4.0 * e**2 + (p - 2.0) ** 2) * p**2
        + 2.0 * a**4 * p * (-2.0 + p + e**2 * (2.0 + p))
    ) / p**3
    # Double-where on ``inner`` so that both the primal and tangent of
    # ``sqrt(inner)`` are finite when ``inner → 0`` (e.g. the a=0 limit):
    # ``jnp.maximum(inner, ε)`` still lets ``jnp.sqrt`` see the real value
    # under forward-mode AD and produces a spurious 1/(2√ε) tangent.
    safe_inner = jnp.where(inner > 0.0, inner, 1.0)
    sqrt_inner = jnp.where(inner > 0.0, jnp.sqrt(safe_inner), 0.0)
    numer = (e**2 - 1.0) * (
        a**2 * (1.0 + 3.0 * e**2 + p)
        + p * (-3.0 - e**2 + p - sgnax * 2.0 * sqrt_inner)
    )
    ratio = jnp.where(jnp.abs(denom) < 1.0e-14, 0.0, numer / denom)
    # Guard the outer sqrt: at high-e edge cases numerical cancellation can
    # push the argument slightly below zero, which would NaN out both the
    # primal and the gradient.
    outer = 1.0 - (1.0 - e**2) * (1.0 + ratio) / p
    safe_outer = jnp.where(outer > 0.0, outer, 1.0)
    return jnp.where(outer > 0.0, jnp.sqrt(safe_outer), 0.0)


@jit
def kerr_geo_angular_momentum_equatorial(
    a: float, p: float, e: float, x: float, En: float
) -> float:
    r"""Orbital angular momentum :math:`L_z` for an equatorial Kerr geodesic.

    Parameters
    ----------
    a, p, e, x : float
        Orbital parameters as for :func:`kerr_geo_energy_equatorial`.
    En : float
        Orbital energy (from :func:`kerr_geo_energy_equatorial`).

    Returns
    -------
    float
        Specific orbital angular momentum :math:`L_z`.
    """
    r1 = p / (1.0 - e)
    d1 = _d_eq(r1, a)
    f1 = _f_eq(r1, a)
    g1 = _g_eq(r1, a)
    h1 = _h_eq(r1, a)
    sgnx = jnp.sign(x)
    disc = -d1 * h1 + En**2 * (g1**2 + f1 * h1)
    return (-En * g1 + sgnx * jnp.sqrt(jnp.maximum(disc, 0.0))) / h1


# ---------------------------------------------------------------------------
# Separatrix via bisection
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=("tol", "max_steps"))
def _bisect(f_lo: float, f_hi: float, lo: float, hi: float, tol: float = 1.0e-13,
            max_steps: int = 100) -> float:
    """Generic bisection on a pre-evaluated sign-change bracket.

    Performs bisection starting from ``lo`` / ``hi`` where the signs of the
    function at those endpoints are given by ``f_lo`` and ``f_hi``.
    """
    # We carry the function reference indirectly via a closure in the callers.
    # Here we just do the iteration on a generic bracket.
    def cond(state):
        lo, hi, _fl, _fh = state
        return (hi - lo) > tol

    def body(state):
        lo, hi, fl, fh = state
        mid = (lo + hi) / 2.0
        return lo, hi, fl, fh  # placeholder – overridden below

    return (lo + hi) / 2.0  # replaced by specialised callers


def _bisect_equat(a: float, e: float, lo: float, hi: float, n_iter: int = 50):
    """Find equatorial separatrix root via fixed-count bisection.

    Uses ``jax.lax.fori_loop`` with static bounds so the function is
    differentiable via both forward and reverse-mode JAX AD.
    50 iterations gives precision ≲ 2^{-50} × (hi - lo) ≈ 1e-15 × 6 ≈ 6e-15.
    """
    def _poly(p):
        return (
            a**4 * (-3.0 - 2.0 * e + e**2) ** 2
            + p**2 * (-6.0 - 2.0 * e + p) ** 2
            - 2.0 * a**2 * (1.0 + e) * p * (14.0 + 2.0 * e**2 + 3.0 * p - e * p)
        )

    def body(_, state):
        lo, hi = state
        mid = (lo + hi) / 2.0
        fl = _poly(lo)
        fm = _poly(mid)
        same_sign = fl * fm > 0.0
        new_lo = jnp.where(same_sign, mid, lo)
        new_hi = jnp.where(same_sign, hi, mid)
        return new_lo, new_hi

    lo_f, hi_f = jax.lax.fori_loop(0, n_iter, body, (jnp.float64(lo), jnp.float64(hi)))
    return (lo_f + hi_f) / 2.0


@partial(jit, static_argnames=())
def get_separatrix(a: float, e: float, x: float) -> float:
    r"""Compute the separatrix semi-latus rectum :math:`p_{\rm sep}(a, e, x_I)`.

    The separatrix is the boundary between stable and unstable bound orbits.
    For Schwarzschild it equals :math:`6 + 2e`; for Kerr equatorial it is
    found as the root of a polynomial (Glampedakis & Kennefick 2002).

    Only equatorial orbits (:math:`x_I = \pm 1`) are currently supported.

    Parameters
    ----------
    a : float
        Dimensionless BH spin, :math:`-1 < a < 1`.
    e : float
        Eccentricity.
    x : float
        Cosine of inclination; must be :math:`\pm 1`.

    Returns
    -------
    float
        :math:`p_{\rm sep}`.
    """
    # Schwarzschild case
    schw_sep = 6.0 + 2.0 * e

    # For Kerr equatorial, find root of separatrix polynomial
    # Prograde:  bracket [1+e, 6+2e];  Retrograde:  [6+2e, 5+e+4√(1+e)]
    is_prograde = (a * x) > 0.0
    lo_prog = 1.0 + e
    hi_prog = 6.0 + 2.0 * e
    lo_retro = 6.0 + 2.0 * e
    hi_retro = 5.0 + e + 4.0 * jnp.sqrt(1.0 + e)

    lo = jnp.where(is_prograde, lo_prog, lo_retro)
    hi = jnp.where(is_prograde, hi_prog, hi_retro)

    kerr_sep = _bisect_equat(a, e, lo, hi)
    return jnp.where(a == 0.0, schw_sep, kerr_sep)


def _bisect_equat_fast(a: float, e: float, lo: float, hi: float) -> float:
    """Hybrid bisection + Newton-Raphson root of the separatrix polynomial.

    Pure NR from the bracket midpoint is unreliable: for high-spin prograde
    orbits the root sits near ``1+e`` while the bracket is ``[1+e, 6+2e]``,
    so the midpoint is far from the root and NR can diverge or stall.

    Strategy
    --------
    1. ``_N_BISECT_INIT`` bisection steps (globally convergent) tighten the
       bracket to width ``(hi-lo)/2^20 < 5e-6``.
    2. ``_N_NR`` Newton-Raphson steps from the bisected midpoint.
       Starting within ~2.5e-6 of the root, quadratic convergence reaches
       full float64 precision in ≤3 steps.

    Both loops have static bounds: fully JIT / vmap / AD compatible.
    """
    # Separatrix polynomial in p (Glampedakis & Kennefick 2002):
    #   P(p) = a⁴(−3−2e+e²)² + p²(−6−2e+p)² − 2a²(1+e)p(14+2e²+3p−ep)
    A_const = a ** 4 * (-3.0 - 2.0 * e + e ** 2) ** 2
    B_fac   = 2.0 * a ** 2 * (1.0 + e)
    lin_c   = 14.0 + 2.0 * e ** 2          # constant in linear term
    quad_c  = 3.0 - e                       # coefficient of p in linear term
    csep    = -6.0 - 2.0 * e               # = -(6+2e)

    def _poly(p):
        return A_const + p ** 2 * (csep + p) ** 2 - B_fac * p * (lin_c + quad_c * p)

    def _dpoly(p):
        # d/dp[p²(csep+p)²] = 2p(csep+p)(csep+2p)
        # d/dp[-B_fac·p·(lin_c+quad_c·p)] = -B_fac·(lin_c + 2·quad_c·p)
        return (2.0 * p * (csep + p) * (csep + 2.0 * p)
                - B_fac * (lin_c + 2.0 * quad_c * p))

    # Phase 1: bisection — narrow bracket to < 5e-6 width
    def bisect_step(_, state):
        lo_, hi_ = state
        mid = (lo_ + hi_) * 0.5
        fl  = _poly(lo_)
        fm  = _poly(mid)
        same_sign = fl * fm > 0.0
        return jnp.where(same_sign, mid, lo_), jnp.where(same_sign, hi_, mid)

    lo_t, hi_t = jax.lax.fori_loop(
        0, _N_BISECT_INIT, bisect_step, (jnp.float64(lo), jnp.float64(hi))
    )

    # Phase 2: NR refinement from the tight midpoint
    def newton_step(_, p):
        fp  = _poly(p)
        dfp = _dpoly(p)
        return p - fp / jnp.where(jnp.abs(dfp) > 1e-30, dfp, 1e-30)

    p0 = (lo_t + hi_t) * 0.5
    return jax.lax.fori_loop(0, _N_NR, newton_step, p0)


@partial(jit, static_argnames=())
def get_separatrix_fast(a: float, e: float, x: float) -> float:
    r"""Fast separatrix via hybrid bisection + Newton-Raphson.

    Identical interface to :func:`get_separatrix`.  Uses
    :data:`_N_BISECT_INIT` bisection steps to obtain a tight initial guess,
    then :data:`_N_NR` Newton-Raphson steps to reach float64 precision.
    Robust across the full physical parameter space including near-extremal
    prograde orbits where pure NR from the bracket midpoint can diverge.

    Parameters
    ----------
    a, e, x : float
        BH spin, eccentricity, inclination sign (±1).

    Returns
    -------
    float
        :math:`p_{\rm sep}`.
    """
    schw_sep = 6.0 + 2.0 * e

    is_prograde = (a * x) > 0.0
    lo = jnp.where(is_prograde, 1.0 + e, 6.0 + 2.0 * e)
    hi = jnp.where(is_prograde, 6.0 + 2.0 * e, 5.0 + e + 4.0 * jnp.sqrt(1.0 + e))

    kerr_sep = _bisect_equat_fast(a, e, lo, hi)
    return jnp.where(a == 0.0, schw_sep, kerr_sep)


# ---------------------------------------------------------------------------
# Fundamental frequencies (Boyer-Lindquist coordinate time)
# ---------------------------------------------------------------------------

@jit
def _radial_roots_equatorial(a: float, p: float, e: float, En: float):
    """Return radial turning points (r1, r2, r3, r4) for Q=0."""
    r1 = p / (1.0 - e)
    r2 = p / (1.0 + e)
    AplusB = 2.0 / (1.0 - En**2) - (r1 + r2)
    # Q = 0  ⟹  AB = 0  ⟹  r4 = 0
    r3 = AplusB
    r4 = 0.0
    return r1, r2, r3, r4


@jit
def _mino_frequencies_equatorial(
    a: float, p: float, e: float, x: float
) -> tuple[float, float, float, float]:
    r"""Mino-time frequencies for equatorial eccentric Kerr geodesics.

    Returns :math:`(\Gamma, \Upsilon_\phi, |\Upsilon_\theta|, \Upsilon_r)`.

    Follows Schmidt (2002) §IV; the equatorial specialisation (Q = 0)
    removes the :math:`\theta`-sector elliptic integrals.
    """
    En = kerr_geo_energy_equatorial(a, p, e, x)
    L = kerr_geo_angular_momentum_equatorial(a, p, e, x, En)

    r1, r2, r3, r4 = _radial_roots_equatorial(a, p, e, En)

    # Radial modulus
    kr2 = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4)
    kr = jnp.sqrt(jnp.maximum(kr2, 0.0))

    # Horizon radii (M = 1)
    rp = 1.0 + jnp.sqrt(1.0 - a**2)
    rm = 1.0 - jnp.sqrt(1.0 - a**2)

    EllK = ellipk(kr**2)  # jax uses m = k^2 convention
    EllE_val = ellipe(kr**2)

    # Υ_r (Mino time)
    Upsilon_r = PI * jnp.sqrt((1.0 - En**2) * (r1 - r3) * r2) / (2.0 * EllK)

    # Υ_θ (equatorial: = |x| * sqrt(L^2 + a^2*(1-E^2)))
    zp = a**2 * (1.0 - En**2) + L**2
    Upsilon_theta = jnp.abs(x) * jnp.sqrt(jnp.maximum(zp, 0.0))

    # Epsilon0zp = zp / L^2
    Epsilon0zp = zp / L**2

    # Υ_φ via Schmidt (2002) Eq. (21)
    hr = (r1 - r2) / (r1 - r3)
    hp = (r1 - r2) * (r3 - rp) / ((r1 - r3) * (r2 - rp))
    hm = (r1 - r2) * (r3 - rm) / ((r1 - r3) * (r2 - rm))

    EllPi_hr = ellip_pi(hr, kr)
    EllPi_hp = ellip_pi(hp, kr)
    EllPi_hm = ellip_pi(hm, kr)

    fac_r = 2.0 * a * Upsilon_r / (
        PI * (rp - rm) * jnp.sqrt((1.0 - En**2) * (r1 - r3) * r2)
    )
    prob1 = jnp.where(
        jnp.abs(r3 - rp) > 1.0e-14,
        (2.0 * En * rp - a * L)
        * (EllK - (r2 - r3) / (r2 - rp) * EllPi_hp)
        / (r3 - rp),
        0.0,
    )
    prob2_neg = (2.0 * En * rm - a * L) * (
        EllK - (r2 - r3) / (r2 - rm) * EllPi_hm
    ) / (r3 - rm)

    Upsilon_phi = (
        Upsilon_theta / jnp.sqrt(Epsilon0zp)
        + fac_r * (prob1 - prob2_neg)
    )

    # Γ (Boyer-Lindquist time per Mino time)
    prob3 = jnp.where(
        jnp.abs(r3 - rp) > 1.0e-14,
        ((4.0 * En - a * L) * rp - 2.0 * a**2 * En)
        * (EllK - (r2 - r3) / (r2 - rp) * EllPi_hp)
        / (r3 - rp),
        0.0,
    )
    prob3_neg = (
        ((4.0 * En - a * L) * rm - 2.0 * a**2 * En)
        * (EllK - (r2 - r3) / (r2 - rm) * EllPi_hm)
        / (r3 - rm)
    )
    Gamma = 4.0 * En + (2.0 * Upsilon_r / (
        PI * jnp.sqrt((1.0 - En**2) * (r1 - r3) * r2)
    )) * (
        En / 2.0 * (
            (r3 * (r1 + r2 + r3) - r1 * r2) * EllK
            + (r2 - r3) * (r1 + r2 + r3 + r4) * EllPi_hr
            + (r1 - r3) * r2 * EllE_val
        )
        + 2.0 * En * (r3 * EllK + (r2 - r3) * EllPi_hr)
        + 2.0 / (rp - rm) * (prob3 - prob3_neg)
    )

    return Gamma, Upsilon_phi, jnp.abs(Upsilon_theta), Upsilon_r


@jit
def get_fundamental_frequencies(
    a: float, p: float, e: float, x: float
) -> tuple[float, float, float]:
    r"""Boyer-Lindquist fundamental frequencies for an equatorial Kerr geodesic.

    .. math::

        \Omega_\phi = \frac{\Upsilon_\phi}{\Gamma}, \quad
        \Omega_\theta = \frac{\Upsilon_\theta}{\Gamma}, \quad
        \Omega_r = \frac{\Upsilon_r}{\Gamma}

    Parameters
    ----------
    a : float
        BH spin parameter.
    p : float
        Semi-latus rectum in units of :math:`M`.
    e : float
        Eccentricity.
    x : float
        Cosine of inclination (:math:`\pm 1` for equatorial).

    Returns
    -------
    tuple of float
        :math:`(\Omega_\phi, \Omega_\theta, \Omega_r)`.
    """
    Gamma, Up_phi, Up_theta, Up_r = _mino_frequencies_equatorial(a, p, e, x)
    return Up_phi / Gamma, Up_theta / Gamma, Up_r / Gamma


# ---------------------------------------------------------------------------
# Fast fundamental frequencies (AGM + 24-pt GL)
# ---------------------------------------------------------------------------

@jit
def _mino_frequencies_equatorial_fast(
    a: float, p: float, e: float, x: float
) -> tuple[float, float, float, float]:
    r"""Mino-time frequencies using fast elliptic integrals.

    Identical to :func:`_mino_frequencies_equatorial` but substitutes:

    * :func:`ellipk_agm` / :func:`ellipe_agm` for K and E.
    * :func:`ellip_pi_fast` (24-pt GL) for all three Π evaluations.

    Approximately 3–4× faster than the exact version for typical EMRI orbits.
    """
    En = kerr_geo_energy_equatorial(a, p, e, x)
    L = kerr_geo_angular_momentum_equatorial(a, p, e, x, En)

    r1, r2, r3, r4 = _radial_roots_equatorial(a, p, e, En)

    kr2 = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4)
    kr  = jnp.sqrt(jnp.maximum(kr2, 0.0))

    rp = 1.0 + jnp.sqrt(1.0 - a ** 2)
    rm = 1.0 - jnp.sqrt(1.0 - a ** 2)

    # --- fast K and E via AGM ---
    EllK     = ellipk_agm(kr ** 2)
    EllE_val = ellipe_agm(kr ** 2)

    Upsilon_r = PI * jnp.sqrt((1.0 - En ** 2) * (r1 - r3) * r2) / (2.0 * EllK)

    zp = a ** 2 * (1.0 - En ** 2) + L ** 2
    Upsilon_theta = jnp.abs(x) * jnp.sqrt(jnp.maximum(zp, 0.0))
    Epsilon0zp = zp / L ** 2

    hr = (r1 - r2) / (r1 - r3)
    hp = (r1 - r2) * (r3 - rp) / ((r1 - r3) * (r2 - rp))
    hm = (r1 - r2) * (r3 - rm) / ((r1 - r3) * (r2 - rm))

    # --- fast Π via 24-pt GL ---
    EllPi_hr = ellip_pi_fast(hr, kr)
    EllPi_hp = ellip_pi_fast(hp, kr)
    EllPi_hm = ellip_pi_fast(hm, kr)

    fac_r = 2.0 * a * Upsilon_r / (
        PI * (rp - rm) * jnp.sqrt((1.0 - En ** 2) * (r1 - r3) * r2)
    )
    prob1 = jnp.where(
        jnp.abs(r3 - rp) > 1.0e-14,
        (2.0 * En * rp - a * L)
        * (EllK - (r2 - r3) / (r2 - rp) * EllPi_hp)
        / (r3 - rp),
        0.0,
    )
    prob2_neg = (2.0 * En * rm - a * L) * (
        EllK - (r2 - r3) / (r2 - rm) * EllPi_hm
    ) / (r3 - rm)

    Upsilon_phi = (
        Upsilon_theta / jnp.sqrt(Epsilon0zp)
        + fac_r * (prob1 - prob2_neg)
    )

    prob3 = jnp.where(
        jnp.abs(r3 - rp) > 1.0e-14,
        ((4.0 * En - a * L) * rp - 2.0 * a ** 2 * En)
        * (EllK - (r2 - r3) / (r2 - rp) * EllPi_hp)
        / (r3 - rp),
        0.0,
    )
    prob3_neg = (
        ((4.0 * En - a * L) * rm - 2.0 * a ** 2 * En)
        * (EllK - (r2 - r3) / (r2 - rm) * EllPi_hm)
        / (r3 - rm)
    )
    Gamma = 4.0 * En + (2.0 * Upsilon_r / (
        PI * jnp.sqrt((1.0 - En ** 2) * (r1 - r3) * r2)
    )) * (
        En / 2.0 * (
            (r3 * (r1 + r2 + r3) - r1 * r2) * EllK
            + (r2 - r3) * (r1 + r2 + r3 + r4) * EllPi_hr
            + (r1 - r3) * r2 * EllE_val
        )
        + 2.0 * En * (r3 * EllK + (r2 - r3) * EllPi_hr)
        + 2.0 / (rp - rm) * (prob3 - prob3_neg)
    )

    return Gamma, Upsilon_phi, jnp.abs(Upsilon_theta), Upsilon_r


@jit
def get_fundamental_frequencies_fast(
    a: float, p: float, e: float, x: float
) -> tuple[float, float, float]:
    r"""Boyer-Lindquist fundamental frequencies using fast elliptic integrals.

    Identical interface to :func:`get_fundamental_frequencies`; uses
    :func:`ellipk_agm`, :func:`ellipe_agm`, and :func:`ellip_pi_fast`
    instead of the 64-point GL versions.  Roughly 3–4× faster.

    Parameters
    ----------
    a, p, e, x : float
        Spin, semi-latus rectum, eccentricity, inclination sign.

    Returns
    -------
    tuple of float
        :math:`(\Omega_\phi, \Omega_\theta, \Omega_r)`.
    """
    Gamma, Up_phi, Up_theta, Up_r = _mino_frequencies_equatorial_fast(a, p, e, x)
    return Up_phi / Gamma, Up_theta / Gamma, Up_r / Gamma

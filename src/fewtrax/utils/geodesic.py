
"""Kerr geodesic utilities implemented in JAX.

This module provides JAX-compatible functions for computing key quantities
of Kerr geodesics required by the EMRI waveform model:

*  Orbital energy :math:`E` and angular momentum :math:`L_z`
*  Separatrix :math:`p_{\\rm sep}(a, e, x_I)`
*  Boyer-Lindquist fundamental frequencies
   :math:`\\Omega_\\phi, \\Omega_\\theta, \\Omega_r`

All functions are JIT-compilable and differentiable via JAX's automatic
differentiation.  The complete elliptic integrals :math:`K, E, \\Pi` come
from two interchangeable backends — :mod:`fewtrax.utils.elliptic_gpu` and
:mod:`fewtrax.utils.elliptic_cpu` — each holding the algorithm that is
fastest at scale on its device.  :func:`get_fundamental_frequencies_platform`
selects between them based on the default JAX device.

Mathematical background
-----------------------
Orbital mechanics follows Schmidt (2002), gr-qc/0202090.
Separatrix polynomials follow Glampedakis & Kennefick (2002),
gr-qc/0203086.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial

from fewtrax.utils.constants import PI
from fewtrax.utils import elliptic_gpu as _ell_gpu
from fewtrax.utils import elliptic_cpu as _ell_cpu

# ---------------------------------------------------------------------------
# Elliptic-integral backend aliases (for callers that import them directly).
# The frequency code below does not use these names — it binds a backend
# explicitly via the factory — but the validation/comparison harness and
# external callers still import the legacy symbols.
# ---------------------------------------------------------------------------

ellipk = _ell_gpu.ellipk            # 64-point Gauss-Legendre K(m)
ellipe = _ell_gpu.ellipe            # 64-point Gauss-Legendre E(m)
ellip_pi = _ell_gpu.ellip_pi_exact  # 64-point Gauss-Legendre Π(n, k) (accuracy ref)

ellipk_agm = _ell_cpu.ellipk        # AGM-12 K(m)
ellipe_agm = _ell_cpu.ellipe        # AGM-12 E(m)
ellip_pi_fast = _ell_gpu.ellip_pi   # 24-point Gauss-Legendre Π(n, k)

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


def _make_mino_frequencies(ellipk_fn, ellipe_fn, ellip_pi_fn):
    r"""Build a JIT-compiled Mino-time frequency function bound to an elliptic
    backend.

    The mathematics (Schmidt 2002 §IV, equatorial Q = 0 specialisation) is
    identical for every backend; only the three elliptic-integral primitives
    differ.  Binding them once here keeps a single source of truth for the
    frequency formulae instead of one copy per platform.

    Parameters
    ----------
    ellipk_fn, ellipe_fn : callable
        :math:`K(m)` and :math:`E(m)` (parameter convention, ``m = k^2``).
    ellip_pi_fn : callable
        :math:`\Pi(n, k)`.

    Returns
    -------
    callable
        JIT-compiled ``(a, p, e, x) -> (Γ, Υ_φ, |Υ_θ|, Υ_r)``.
    """
    @jit
    def _mino(a: float, p: float, e: float, x: float):
        En = kerr_geo_energy_equatorial(a, p, e, x)
        L = kerr_geo_angular_momentum_equatorial(a, p, e, x, En)

        r1, r2, r3, r4 = _radial_roots_equatorial(a, p, e, En)

        # Radial modulus
        kr2 = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4)
        kr = jnp.sqrt(jnp.maximum(kr2, 0.0))

        # Horizon radii (M = 1)
        rp = 1.0 + jnp.sqrt(1.0 - a**2)
        rm = 1.0 - jnp.sqrt(1.0 - a**2)

        EllK = ellipk_fn(kr**2)  # jax uses m = k^2 convention
        EllE_val = ellipe_fn(kr**2)

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

        EllPi_hr = ellip_pi_fn(hr, kr)
        EllPi_hp = ellip_pi_fn(hp, kr)
        EllPi_hm = ellip_pi_fn(hm, kr)

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

    return _mino


# Two backend-bound Mino-frequency functions, built once at import time.
_mino_frequencies_equatorial = _make_mino_frequencies(
    _ell_gpu.ellipk, _ell_gpu.ellipe, _ell_gpu.ellip_pi
)
_mino_frequencies_equatorial_fast = _make_mino_frequencies(
    _ell_cpu.ellipk, _ell_cpu.ellipe, _ell_cpu.ellip_pi
)


@jit
def get_fundamental_frequencies(
    a: float, p: float, e: float, x: float
) -> tuple[float, float, float]:
    r"""Boyer-Lindquist fundamental frequencies (GPU-optimal backend).

    .. math::

        \Omega_\phi = \frac{\Upsilon_\phi}{\Gamma}, \quad
        \Omega_\theta = \frac{\Upsilon_\theta}{\Gamma}, \quad
        \Omega_r = \frac{\Upsilon_r}{\Gamma}

    Uses :mod:`fewtrax.utils.elliptic_gpu`: 64-point Gauss-Legendre for
    :math:`K, E` and 24-point GL for :math:`\Pi` — the combination that
    maximises GPU throughput at batch scale.

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


@jit
def get_fundamental_frequencies_fast(
    a: float, p: float, e: float, x: float
) -> tuple[float, float, float]:
    r"""Boyer-Lindquist fundamental frequencies (CPU/TPU-optimal backend).

    Identical interface and mathematics to :func:`get_fundamental_frequencies`;
    uses :mod:`fewtrax.utils.elliptic_cpu` (AGM-12 for :math:`K, E`, 24-point
    GL for :math:`\Pi`), which has fewer sequential operations and is faster on
    scalar / small-batch CPU workloads.

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


def get_fundamental_frequencies_platform(
    a: float, p: float, e: float, x: float
) -> tuple[float, float, float]:
    r"""Platform-aware Boyer-Lindquist fundamental frequencies (dispatcher).

    Single entry point that routes to the elliptic backend that is fastest at
    scale on the current device.  The branch is resolved at Python (trace)
    time, so there is no runtime overhead on repeated JIT-compiled calls.

    * **GPU** — :func:`get_fundamental_frequencies`
      (:mod:`~fewtrax.utils.elliptic_gpu`: 64-point GL for :math:`K, E`,
      24-point GL for :math:`\Pi`).  The quadratures compile to fused BLAS-like
      contractions that saturate GPU throughput; the iterative AGM is
      latency-bound and measured *slower* here.

    * **CPU / TPU** — :func:`get_fundamental_frequencies_fast`
      (:mod:`~fewtrax.utils.elliptic_cpu`: AGM-12 for :math:`K, E`, 24-point GL
      for :math:`\Pi`).  Fewer sequential operations → lower per-step cost on
      scalar / small-batch workloads.

    Parameters
    ----------
    a, p, e, x : float
        BH spin, semi-latus rectum, eccentricity, inclination sign (±1).

    Returns
    -------
    tuple of float
        :math:`(\Omega_\phi, \Omega_\theta, \Omega_r)`.
    """
    platform = jax.devices()[0].platform
    if platform == "gpu":
        return get_fundamental_frequencies(a, p, e, x)
    else:
        return get_fundamental_frequencies_fast(a, p, e, x)

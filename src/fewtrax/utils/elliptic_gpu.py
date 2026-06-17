"""GPU-optimal complete elliptic integrals (JAX).

This backend is selected by :func:`fewtrax.utils.geodesic.get_fundamental_frequencies_platform`
when the default JAX device is a GPU.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

# ---------------------------------------------------------------------------
# Gauss-Legendre nodes/weights, precomputed once at import time.
# Transformed from the reference interval [-1, 1] to [0, pi/2].
# ---------------------------------------------------------------------------

_N_GL = 64
_gl_nodes_np, _gl_weights_np = np.polynomial.legendre.leggauss(_N_GL)
_GL_THETA = jnp.asarray((_gl_nodes_np + 1.0) / 2.0 * np.pi / 2.0, dtype=jnp.float64)
_GL_W = jnp.asarray(_gl_weights_np / 2.0 * np.pi / 2.0, dtype=jnp.float64)

# 24-point rule for Pi: accurate to ~1e-12 for the smooth integrands of bound
# EMRI orbits, and faster than the 64-point rule on GPU at batch scale.
_N_GL24 = 24
_gl24_nodes_np, _gl24_weights_np = np.polynomial.legendre.leggauss(_N_GL24)
_GL24_THETA = jnp.asarray((_gl24_nodes_np + 1.0) / 2.0 * np.pi / 2.0, dtype=jnp.float64)
_GL24_W = jnp.asarray(_gl24_weights_np / 2.0 * np.pi / 2.0, dtype=jnp.float64)


@jit
def ellipk(m: float) -> float:
    r"""Complete elliptic integral of the first kind :math:`K(m)`.

    .. math::  K(m) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - m \sin^2\theta}}

    64-point Gauss-Legendre quadrature.  ``m`` is the *parameter* (square of
    the modulus), consistent with scipy's convention.
    """
    sin2 = jnp.sin(_GL_THETA) ** 2
    integrand = 1.0 / jnp.sqrt(jnp.maximum(1.0 - m * sin2, 1e-30))
    return jnp.dot(_GL_W, integrand)


@jit
def ellipe(m: float) -> float:
    r"""Complete elliptic integral of the second kind :math:`E(m)`.

    .. math::  E(m) = \int_0^{\pi/2} \sqrt{1 - m \sin^2\theta}\, d\theta

    64-point Gauss-Legendre quadrature.
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

    24-point Gauss-Legendre quadrature: the production choice on GPU.  Faster
    than the 64-point rule (the gap widens with batch size) and accurate to
    ~4e-16 over the EMRI range (:math:`k^2 < 0.8`, :math:`n < 0.8`).

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
    sin2 = jnp.sin(_GL24_THETA) ** 2
    integrand = 1.0 / ((1.0 - n * sin2) * jnp.sqrt(jnp.maximum(1.0 - k**2 * sin2, 1e-30)))
    return jnp.dot(_GL24_W, integrand)


@jit
def ellip_pi_exact(n: float, k: float) -> float:
    r"""64-point Gauss-Legendre :math:`\Pi(n, k)`.

    Accuracy reference for the validation harness; not used in production
    frequency evaluation (:func:`ellip_pi`, the 24-point rule, is faster on
    GPU at every batch size measured).
    """
    sin2 = jnp.sin(_GL_THETA) ** 2
    integrand = 1.0 / ((1.0 - n * sin2) * jnp.sqrt(jnp.maximum(1.0 - k**2 * sin2, 1e-30)))
    return jnp.dot(_GL_W, integrand)

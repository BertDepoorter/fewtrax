"""CPU/TPU-optimal complete elliptic integrals (JAX).

This backend is selected by :func:`fewtrax.utils.geodesic.get_fundamental_frequencies_platform`
when the default JAX device is not a GPU (CPU or TPU).

On scalar / small-batch workloads the cost is dominated by per-call latency
rather than throughput, so algorithms with fewer arithmetic operations win:
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

# Number of AGM iterations for ellipk / ellipe.
# 12 iterations gives ~2^{-50} ≈ 1e-15 relative error (quadratic convergence).
_N_AGM: int = 12

# 24-point Gauss-Legendre nodes/weights for Π, transformed to [0, pi/2].
_N_GL24 = 24
_gl24_nodes_np, _gl24_weights_np = np.polynomial.legendre.leggauss(_N_GL24)
_GL24_THETA = jnp.asarray((_gl24_nodes_np + 1.0) / 2.0 * np.pi / 2.0, dtype=jnp.float64)
_GL24_W = jnp.asarray(_gl24_weights_np / 2.0 * np.pi / 2.0, dtype=jnp.float64)


@jit
def ellipk(m: float) -> float:
    r"""Complete elliptic integral :math:`K(m)` via the AGM algorithm.

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
def ellipe(m: float) -> float:
    r"""Complete elliptic integral :math:`E(m)` via the AGM / Borwein algorithm.

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
def ellip_pi(n: float, k: float) -> float:
    r"""Complete elliptic integral :math:`\Pi(n, k)` via 24-point GL quadrature.

    .. math::

        \Pi(n, k) = \int_0^{\pi/2}
            \frac{d\theta}{(1 - n\sin^2\theta)\sqrt{1 - k^2\sin^2\theta}}

    ~2.5x faster than the 64-point rule with ~1e-12 accuracy for the smooth
    integrands of bound EMRI orbits.

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

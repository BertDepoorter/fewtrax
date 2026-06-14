r"""Spin-weight −2 spherical harmonics :math:`{}_{-2}Y_{\ell m}(\theta, \phi)`.

These appear in the strain decomposition
:math:`h_+ - i h_\times = \sum_{\ell m} {}_{-2}Y_{\ell m}(\theta, \phi)\, h_{\ell m}(t)`.

Closed forms for :math:`\ell = 2 \ldots 10` are hardcoded in :data:`_YLM_TABLE`
(from FastEMRIWaveforms ``few/utils/ylm.py``; Goldberg et al. 1967 convention),
written in JAX so they are JIT-compilable and differentiable in :math:`(\theta,
\phi)`; :math:`(\ell, m)` are static. Keys are ``l<ell>_m<m>``, negative orders
``neg`` (e.g. ``l2_m2``, ``l2_mneg2``); entries are functions of ``(c, s, phi)``
with ``c = cos(theta/2)``, ``s = sin(theta/2)``.
"""

from __future__ import annotations

from math import sqrt, pi as PI

import numpy as np
import jax.numpy as jnp


def _mode_key(l: int, m: int) -> str:
    """Return the table key ``l<ell>_m<m>`` (negative m written ``neg``)."""
    return f"l{int(l)}_m{'neg' if m < 0 else ''}{abs(int(m))}"


# ---------------------------------------------------------------------------
# Hardcoded closed-form table, keyed by ``l<ell>_m<m>`` (e.g. l2_m2, l2_mneg2).
# Each entry is a function of (c, s, phi) with c = cos(theta/2), s = sin(theta/2);
# (l, m) are static, (theta, phi) may be JAX tracers -> JIT + differentiable.
# ---------------------------------------------------------------------------

_YLM_TABLE = {
    "l2_mneg2": lambda c, s, phi: (sqrt(5./PI)*s**4)/(2.*jnp.exp(2.*1j*phi)),
    "l2_mneg1": lambda c, s, phi: (sqrt(5./PI)*c*s**3)/jnp.exp(1j*phi),
    "l2_m0": lambda c, s, phi: sqrt(15./(2.*PI))*c**2*s**2,
    "l2_m1": lambda c, s, phi: jnp.exp(1j*phi)*sqrt(5./PI)*c**3*s,
    "l2_m2": lambda c, s, phi: (jnp.exp(2.*1j*phi)*sqrt(5./PI)*c**4)/2.,
    "l3_mneg3": lambda c, s, phi: (sqrt(21./(2.*PI))*c*s**5)/jnp.exp(3.*1j*phi),
    "l3_mneg2": lambda c, s, phi: (sqrt(7./PI)*(5.*c**2*s**4 - s**6))/(2.*jnp.exp(2.*1j*phi)),
    "l3_mneg1": lambda c, s, phi: -((sqrt(7./(10.*PI))*(-10.*c**3*s**3 + 5.*c*s**5))/jnp.exp(1j*phi)),
    "l3_m0": lambda c, s, phi: (sqrt(21./(10.*PI))*(10.*c**4*s**2 - 10.*c**2*s**4))/2.,
    "l3_m1": lambda c, s, phi: -(jnp.exp(1j*phi)*sqrt(7./(10.*PI))*(-5.*c**5*s + 10.*c**3*s**3)),
    "l3_m2": lambda c, s, phi: (jnp.exp(2.*1j*phi)*sqrt(7./PI)*(c**6 - 5.*c**4*s**2))/2.,
    "l3_m3": lambda c, s, phi: -(jnp.exp(3.*1j*phi)*sqrt(21./(2.*PI))*c**5*s),
    "l4_mneg4": lambda c, s, phi: (3.*sqrt(7./PI)*c**2*s**6)/jnp.exp(4.*1j*phi),
    "l4_mneg3": lambda c, s, phi: (-3.*sqrt(7./(2.*PI))*(-6.*c**3*s**5 + 2.*c*s**7))/(2.*jnp.exp(3.*1j*phi)),
    "l4_mneg2": lambda c, s, phi: (3.*(15.*c**4*s**4 - 12.*c**2*s**6 + s**8))/(2.*jnp.exp(2.*1j*phi)*sqrt(PI)),
    "l4_mneg1": lambda c, s, phi: (-3.*(-20.*c**5*s**3 + 30.*c**3*s**5 - 6.*c*s**7))/(2.*jnp.exp(1j*phi)*sqrt(2.*PI)),
    "l4_m0": lambda c, s, phi: (3.*(15.*c**6*s**2 - 40.*c**4*s**4 + 15.*c**2*s**6))/sqrt(10.*PI),
    "l4_m1": lambda c, s, phi: (-3.*jnp.exp(1j*phi)*(-6.*c**7*s + 30.*c**5*s**3 - 20.*c**3*s**5))/(2.*sqrt(2.*PI)),
    "l4_m2": lambda c, s, phi: (3.*jnp.exp(2.*1j*phi)*(c**8 - 12.*c**6*s**2 + 15.*c**4*s**4))/(2.*sqrt(PI)),
    "l4_m3": lambda c, s, phi: (-3.*jnp.exp(3.*1j*phi)*sqrt(7./(2.*PI))*(2.*c**7*s - 6.*c**5*s**3))/2.,
    "l4_m4": lambda c, s, phi: 3.*jnp.exp(4.*1j*phi)*sqrt(7./PI)*c**6*s**2,
    "l5_mneg5": lambda c, s, phi: (sqrt(330./PI)*c**3*s**7)/jnp.exp(5.*1j*phi),
    "l5_mneg4": lambda c, s, phi: (sqrt(33./PI)*(7.*c**4*s**6 - 3.*c**2*s**8))/jnp.exp(4.*1j*phi),
    "l5_mneg3": lambda c, s, phi: -((sqrt(22./(3.*PI))*(-21.*c**5*s**5 + 21.*c**3*s**7 - 3.*c*s**9))/jnp.exp(3.*1j*phi)),
    "l5_mneg2": lambda c, s, phi: (sqrt(11./PI)*(35.*c**6*s**4 - 63.*c**4*s**6 + 21.*c**2*s**8 - s**10))/(2.*jnp.exp(2.*1j*phi)),
    "l5_mneg1": lambda c, s, phi: -((sqrt(11./(7.*PI))*(-35.*c**7*s**3 + 105.*c**5*s**5 - 63.*c**3*s**7 + 7.*c*s**9))/jnp.exp(1j*phi)),
    "l5_m0": lambda c, s, phi: sqrt(55./(42.*PI))*(21.*c**8*s**2 - 105.*c**6*s**4 + 105.*c**4*s**6 - 21.*c**2*s**8),
    "l5_m1": lambda c, s, phi: -(jnp.exp(1j*phi)*sqrt(11./(7.*PI))*(-7.*c**9*s + 63.*c**7*s**3 - 105.*c**5*s**5 + 35.*c**3*s**7)),
    "l5_m2": lambda c, s, phi: (jnp.exp(2.*1j*phi)*sqrt(11./PI)*(c**10 - 21.*c**8*s**2 + 63.*c**6*s**4 - 35.*c**4*s**6))/2.,
    "l5_m3": lambda c, s, phi: -(jnp.exp(3.*1j*phi)*sqrt(22./(3.*PI))*(3.*c**9*s - 21.*c**7*s**3 + 21.*c**5*s**5)),
    "l5_m4": lambda c, s, phi: jnp.exp(4.*1j*phi)*sqrt(33./PI)*(3.*c**8*s**2 - 7.*c**6*s**4),
    "l5_m5": lambda c, s, phi: -(jnp.exp(5.*1j*phi)*sqrt(330./PI)*c**7*s**3),
    "l6_mneg6": lambda c, s, phi: (3.*sqrt(715./PI)*c**4*s**8)/(2.*jnp.exp(6.*1j*phi)),
    "l6_mneg5": lambda c, s, phi: -(sqrt(2145./PI)*(-8.*c**5*s**7 + 4.*c**3*s**9))/(4.*jnp.exp(5.*1j*phi)),
    "l6_mneg4": lambda c, s, phi: (sqrt(195./(2.*PI))*(28.*c**6*s**6 - 32.*c**4*s**8 + 6.*c**2*s**10))/(2.*jnp.exp(4.*1j*phi)),
    "l6_mneg3": lambda c, s, phi: (-3.*sqrt(13./PI)*(-56.*c**7*s**5 + 112.*c**5*s**7 - 48.*c**3*s**9 + 4.*c*s**11))/(4.*jnp.exp(3.*1j*phi)),
    "l6_mneg2": lambda c, s, phi: (sqrt(13./PI)*(70.*c**8*s**4 - 224.*c**6*s**6 + 168.*c**4*s**8 - 32.*c**2*s**10 + s**12))/(2.*jnp.exp(2.*1j*phi)),
    "l6_mneg1": lambda c, s, phi: -(sqrt(65./(2.*PI))*(-56.*c**9*s**3 + 280.*c**7*s**5 - 336.*c**5*s**7 + 112.*c**3*s**9 - 8.*c*s**11))/(4.*jnp.exp(1j*phi)),
    "l6_m0": lambda c, s, phi: (sqrt(195./(7.*PI))*(28.*c**10*s**2 - 224.*c**8*s**4 + 420.*c**6*s**6 - 224.*c**4*s**8 + 28.*c**2*s**10))/4.,
    "l6_m1": lambda c, s, phi: -(jnp.exp(1j*phi)*sqrt(65./(2.*PI))*(-8.*c**11*s + 112.*c**9*s**3 - 336.*c**7*s**5 + 280.*c**5*s**7 - 56.*c**3*s**9))/4.,
    "l6_m2": lambda c, s, phi: (jnp.exp(2.*1j*phi)*sqrt(13./PI)*(c**12 - 32.*c**10*s**2 + 168.*c**8*s**4 - 224.*c**6*s**6 + 70.*c**4*s**8))/2.,
    "l6_m3": lambda c, s, phi: (-3.*jnp.exp(3.*1j*phi)*sqrt(13./PI)*(4.*c**11*s - 48.*c**9*s**3 + 112.*c**7*s**5 - 56.*c**5*s**7))/4.,
    "l6_m4": lambda c, s, phi: (jnp.exp(4.*1j*phi)*sqrt(195./(2.*PI))*(6.*c**10*s**2 - 32.*c**8*s**4 + 28.*c**6*s**6))/2.,
    "l6_m5": lambda c, s, phi: -(jnp.exp(5.*1j*phi)*sqrt(2145./PI)*(4.*c**9*s**3 - 8.*c**7*s**5))/4.,
    "l6_m6": lambda c, s, phi: (3.*jnp.exp(6.*1j*phi)*sqrt(715./PI)*c**8*s**4)/2.,
    "l7_mneg7": lambda c, s, phi: (sqrt(15015./(2.*PI))*c**5*s**9)/jnp.exp(7.*1j*phi),
    "l7_mneg6": lambda c, s, phi: (sqrt(2145./PI)*(9.*c**6*s**8 - 5.*c**4*s**10))/(2.*jnp.exp(6.*1j*phi)),
    "l7_mneg5": lambda c, s, phi: -((sqrt(165./(2.*PI))*(-36.*c**7*s**7 + 45.*c**5*s**9 - 10.*c**3*s**11))/jnp.exp(5.*1j*phi)),
    "l7_mneg4": lambda c, s, phi: (sqrt(165./(2.*PI))*(84.*c**8*s**6 - 180.*c**6*s**8 + 90.*c**4*s**10 - 10.*c**2*s**12))/(2.*jnp.exp(4.*1j*phi)),
    "l7_mneg3": lambda c, s, phi: -((sqrt(15./(2.*PI))*(-126.*c**9*s**5 + 420.*c**7*s**7 - 360.*c**5*s**9 + 90.*c**3*s**11 - 5.*c*s**13))/jnp.exp(3.*1j*phi)),
    "l7_mneg2": lambda c, s, phi: (sqrt(15./PI)*(126.*c**10*s**4 - 630.*c**8*s**6 + 840.*c**6*s**8 - 360.*c**4*s**10 + 45.*c**2*s**12 - s**14))/(2.*jnp.exp(2.*1j*phi)),
    "l7_mneg1": lambda c, s, phi: -((sqrt(5./(2.*PI))*(-84.*c**11*s**3 + 630.*c**9*s**5 - 1260.*c**7*s**7 + 840.*c**5*s**9 - 180.*c**3*s**11 + 9.*c*s**13))/jnp.exp(1j*phi)),
    "l7_m0": lambda c, s, phi: (sqrt(35./PI)*(36.*c**12*s**2 - 420.*c**10*s**4 + 1260.*c**8*s**6 - 1260.*c**6*s**8 + 420.*c**4*s**10 - 36.*c**2*s**12))/4.,
    "l7_m1": lambda c, s, phi: -(jnp.exp(1j*phi)*sqrt(5./(2.*PI))*(-9.*c**13*s + 180.*c**11*s**3 - 840.*c**9*s**5 + 1260.*c**7*s**7 - 630.*c**5*s**9 + 84.*c**3*s**11)),
    "l7_m2": lambda c, s, phi: (jnp.exp(2.*1j*phi)*sqrt(15./PI)*(c**14 - 45.*c**12*s**2 + 360.*c**10*s**4 - 840.*c**8*s**6 + 630.*c**6*s**8 - 126.*c**4*s**10))/2.,
    "l7_m3": lambda c, s, phi: -(jnp.exp(3.*1j*phi)*sqrt(15./(2.*PI))*(5.*c**13*s - 90.*c**11*s**3 + 360.*c**9*s**5 - 420.*c**7*s**7 + 126.*c**5*s**9)),
    "l7_m4": lambda c, s, phi: (jnp.exp(4.*1j*phi)*sqrt(165./(2.*PI))*(10.*c**12*s**2 - 90.*c**10*s**4 + 180.*c**8*s**6 - 84.*c**6*s**8))/2.,
    "l7_m5": lambda c, s, phi: -(jnp.exp(5.*1j*phi)*sqrt(165./(2.*PI))*(10.*c**11*s**3 - 45.*c**9*s**5 + 36.*c**7*s**7)),
    "l7_m6": lambda c, s, phi: (jnp.exp(6.*1j*phi)*sqrt(2145./PI)*(5.*c**10*s**4 - 9.*c**8*s**6))/2.,
    "l7_m7": lambda c, s, phi: -(jnp.exp(7.*1j*phi)*sqrt(15015./(2.*PI))*c**9*s**5),
    "l8_mneg8": lambda c, s, phi: (sqrt(34034./PI)*c**6*s**10)/jnp.exp(8.*1j*phi),
    "l8_mneg7": lambda c, s, phi: -(sqrt(17017./(2.*PI))*(-10.*c**7*s**9 + 6.*c**5*s**11))/(2.*jnp.exp(7.*1j*phi)),
    "l8_mneg6": lambda c, s, phi: (sqrt(17017./(15.*PI))*(45.*c**8*s**8 - 60.*c**6*s**10 + 15.*c**4*s**12))/(2.*jnp.exp(6.*1j*phi)),
    "l8_mneg5": lambda c, s, phi: -(sqrt(2431./(10.*PI))*(-120.*c**9*s**7 + 270.*c**7*s**9 - 150.*c**5*s**11 + 20.*c**3*s**13))/(2.*jnp.exp(5.*1j*phi)),
    "l8_mneg4": lambda c, s, phi: (sqrt(187./(10.*PI))*(210.*c**10*s**6 - 720.*c**8*s**8 + 675.*c**6*s**10 - 200.*c**4*s**12 + 15.*c**2*s**14))/jnp.exp(4.*1j*phi),
    "l8_mneg3": lambda c, s, phi: -(sqrt(187./(6.*PI))*(-252.*c**11*s**5 + 1260.*c**9*s**7 - 1800.*c**7*s**9 + 900.*c**5*s**11 - 150.*c**3*s**13 + 6.*c*s**15))/(2.*jnp.exp(3.*1j*phi)),
    "l8_mneg2": lambda c, s, phi: (sqrt(17./PI)*(210.*c**12*s**4 - 1512.*c**10*s**6 + 3150.*c**8*s**8 - 2400.*c**6*s**10 + 675.*c**4*s**12 - 60.*c**2*s**14 + s**16))/(2.*jnp.exp(2.*1j*phi)),
    "l8_mneg1": lambda c, s, phi: -(sqrt(119./(10.*PI))*(-120.*c**13*s**3 + 1260.*c**11*s**5 - 3780.*c**9*s**7 + 4200.*c**7*s**9 - 1800.*c**5*s**11 + 270.*c**3*s**13 - 10.*c*s**15))/(2.*jnp.exp(1j*phi)),
    "l8_m0": lambda c, s, phi: (sqrt(119./(5.*PI))*(45.*c**14*s**2 - 720.*c**12*s**4 + 3150.*c**10*s**6 - 5040.*c**8*s**8 + 3150.*c**6*s**10 - 720.*c**4*s**12 + 45.*c**2*s**14))/3.,
    "l8_m1": lambda c, s, phi: -(jnp.exp(1j*phi)*sqrt(119./(10.*PI))*(-10.*c**15*s + 270.*c**13*s**3 - 1800.*c**11*s**5 + 4200.*c**9*s**7 - 3780.*c**7*s**9 + 1260.*c**5*s**11 - 120.*c**3*s**13))/2.,
    "l8_m2": lambda c, s, phi: (jnp.exp(2.*1j*phi)*sqrt(17./PI)*(c**16 - 60.*c**14*s**2 + 675.*c**12*s**4 - 2400.*c**10*s**6 + 3150.*c**8*s**8 - 1512.*c**6*s**10 + 210.*c**4*s**12))/2.,
    "l8_m3": lambda c, s, phi: -(jnp.exp(3.*1j*phi)*sqrt(187./(6.*PI))*(6.*c**15*s - 150.*c**13*s**3 + 900.*c**11*s**5 - 1800.*c**9*s**7 + 1260.*c**7*s**9 - 252.*c**5*s**11))/2.,
    "l8_m4": lambda c, s, phi: jnp.exp(4.*1j*phi)*sqrt(187./(10.*PI))*(15.*c**14*s**2 - 200.*c**12*s**4 + 675.*c**10*s**6 - 720.*c**8*s**8 + 210.*c**6*s**10),
    "l8_m5": lambda c, s, phi: -(jnp.exp(5.*1j*phi)*sqrt(2431./(10.*PI))*(20.*c**13*s**3 - 150.*c**11*s**5 + 270.*c**9*s**7 - 120.*c**7*s**9))/2.,
    "l8_m6": lambda c, s, phi: (jnp.exp(6.*1j*phi)*sqrt(17017./(15.*PI))*(15.*c**12*s**4 - 60.*c**10*s**6 + 45.*c**8*s**8))/2.,
    "l8_m7": lambda c, s, phi: -(jnp.exp(7.*1j*phi)*sqrt(17017./(2.*PI))*(6.*c**11*s**5 - 10.*c**9*s**7))/2.,
    "l8_m8": lambda c, s, phi: jnp.exp(8.*1j*phi)*sqrt(34034./PI)*c**10*s**6,
    "l9_mneg9": lambda c, s, phi: (6.*sqrt(4199./PI)*c**7*s**11)/jnp.exp(9.*1j*phi),
    "l9_mneg8": lambda c, s, phi: (sqrt(8398./PI)*(11.*c**8*s**10 - 7.*c**6*s**12))/jnp.exp(8.*1j*phi),
    "l9_mneg7": lambda c, s, phi: (-2.*sqrt(247./PI)*(-55.*c**9*s**9 + 77.*c**7*s**11 - 21.*c**5*s**13))/jnp.exp(7.*1j*phi),
    "l9_mneg6": lambda c, s, phi: (sqrt(741./PI)*(165.*c**10*s**8 - 385.*c**8*s**10 + 231.*c**6*s**12 - 35.*c**4*s**14))/(2.*jnp.exp(6.*1j*phi)),
    "l9_mneg5": lambda c, s, phi: -((sqrt(247./(5.*PI))*(-330.*c**11*s**7 + 1155.*c**9*s**9 - 1155.*c**7*s**11 + 385.*c**5*s**13 - 35.*c**3*s**15))/jnp.exp(5.*1j*phi)),
    "l9_mneg4": lambda c, s, phi: (sqrt(247./(14.*PI))*(462.*c**12*s**6 - 2310.*c**10*s**8 + 3465.*c**8*s**10 - 1925.*c**6*s**12 + 385.*c**4*s**14 - 21.*c**2*s**16))/jnp.exp(4.*1j*phi),
    "l9_mneg3": lambda c, s, phi: -((sqrt(57./(7.*PI))*(-462.*c**13*s**5 + 3234.*c**11*s**7 - 6930.*c**9*s**9 + 5775.*c**7*s**11 - 1925.*c**5*s**13 + 231.*c**3*s**15 - 7.*c*s**17))/jnp.exp(3.*1j*phi)),
    "l9_mneg2": lambda c, s, phi: (sqrt(19./PI)*(330.*c**14*s**4 - 3234.*c**12*s**6 + 9702.*c**10*s**8 - 11550.*c**8*s**10 + 5775.*c**6*s**12 - 1155.*c**4*s**14 + 77.*c**2*s**16 - s**18))/(2.*jnp.exp(2.*1j*phi)),
    "l9_mneg1": lambda c, s, phi: -((sqrt(38./(11.*PI))*(-165.*c**15*s**3 + 2310.*c**13*s**5 - 9702.*c**11*s**7 + 16170.*c**9*s**9 - 11550.*c**7*s**11 + 3465.*c**5*s**13 - 385.*c**3*s**15 + 11.*c*s**17))/jnp.exp(1j*phi)),
    "l9_m0": lambda c, s, phi: 3.*sqrt(19./(55.*PI))*(55.*c**16*s**2 - 1155.*c**14*s**4 + 6930.*c**12*s**6 - 16170.*c**10*s**8 + 16170.*c**8*s**10 - 6930.*c**6*s**12 + 1155.*c**4*s**14 - 55.*c**2*s**16),
    "l9_m1": lambda c, s, phi: -(jnp.exp(1j*phi)*sqrt(38./(11.*PI))*(-11.*c**17*s + 385.*c**15*s**3 - 3465.*c**13*s**5 + 11550.*c**11*s**7 - 16170.*c**9*s**9 + 9702.*c**7*s**11 - 2310.*c**5*s**13 + 165.*c**3*s**15)),
    "l9_m2": lambda c, s, phi: (jnp.exp(2.*1j*phi)*sqrt(19./PI)*(c**18 - 77.*c**16*s**2 + 1155.*c**14*s**4 - 5775.*c**12*s**6 + 11550.*c**10*s**8 - 9702.*c**8*s**10 + 3234.*c**6*s**12 - 330.*c**4*s**14))/2.,
    "l9_m3": lambda c, s, phi: -(jnp.exp(3.*1j*phi)*sqrt(57./(7.*PI))*(7.*c**17*s - 231.*c**15*s**3 + 1925.*c**13*s**5 - 5775.*c**11*s**7 + 6930.*c**9*s**9 - 3234.*c**7*s**11 + 462.*c**5*s**13)),
    "l9_m4": lambda c, s, phi: jnp.exp(4.*1j*phi)*sqrt(247./(14.*PI))*(21.*c**16*s**2 - 385.*c**14*s**4 + 1925.*c**12*s**6 - 3465.*c**10*s**8 + 2310.*c**8*s**10 - 462.*c**6*s**12),
    "l9_m5": lambda c, s, phi: -(jnp.exp(5.*1j*phi)*sqrt(247./(5.*PI))*(35.*c**15*s**3 - 385.*c**13*s**5 + 1155.*c**11*s**7 - 1155.*c**9*s**9 + 330.*c**7*s**11)),
    "l9_m6": lambda c, s, phi: (jnp.exp(6.*1j*phi)*sqrt(741./PI)*(35.*c**14*s**4 - 231.*c**12*s**6 + 385.*c**10*s**8 - 165.*c**8*s**10))/2.,
    "l9_m7": lambda c, s, phi: -2.*jnp.exp(7.*1j*phi)*sqrt(247./PI)*(21.*c**13*s**5 - 77.*c**11*s**7 + 55.*c**9*s**9),
    "l9_m8": lambda c, s, phi: jnp.exp(8.*1j*phi)*sqrt(8398./PI)*(7.*c**12*s**6 - 11.*c**10*s**8),
    "l9_m9": lambda c, s, phi: -6.*jnp.exp(9.*1j*phi)*sqrt(4199./PI)*c**11*s**7,
    "l10_mneg10": lambda c, s, phi: (3.*sqrt(146965./(2.*PI))*c**8*s**12)/jnp.exp(10.*1j*phi),
    "l10_mneg9": lambda c, s, phi: (-3.*sqrt(29393./(2.*PI))*(-12.*c**9*s**11 + 8.*c**7*s**13))/(2.*jnp.exp(9.*1j*phi)),
    "l10_mneg8": lambda c, s, phi: (3.*sqrt(1547./PI)*(66.*c**10*s**10 - 96.*c**8*s**12 + 28.*c**6*s**14))/(2.*jnp.exp(8.*1j*phi)),
    "l10_mneg7": lambda c, s, phi: -(sqrt(4641./(2.*PI))*(-220.*c**11*s**9 + 528.*c**9*s**11 - 336.*c**7*s**13 + 56.*c**5*s**15))/(2.*jnp.exp(7.*1j*phi)),
    "l10_mneg6": lambda c, s, phi: (sqrt(273./(2.*PI))*(495.*c**12*s**8 - 1760.*c**10*s**10 + 1848.*c**8*s**12 - 672.*c**6*s**14 + 70.*c**4*s**16))/jnp.exp(6.*1j*phi),
    "l10_mneg5": lambda c, s, phi: -(sqrt(1365./(2.*PI))*(-792.*c**13*s**7 + 3960.*c**11*s**9 - 6160.*c**9*s**11 + 3696.*c**7*s**13 - 840.*c**5*s**15 + 56.*c**3*s**17))/(4.*jnp.exp(5.*1j*phi)),
    "l10_mneg4": lambda c, s, phi: (sqrt(273./PI)*(924.*c**14*s**6 - 6336.*c**12*s**8 + 13860.*c**10*s**10 - 12320.*c**8*s**12 + 4620.*c**6*s**14 - 672.*c**4*s**16 + 28.*c**2*s**18))/(4.*jnp.exp(4.*1j*phi)),
    "l10_mneg3": lambda c, s, phi: -(sqrt(273./(2.*PI))*(-792.*c**15*s**5 + 7392.*c**13*s**7 - 22176.*c**11*s**9 + 27720.*c**9*s**11 - 15400.*c**7*s**13 + 3696.*c**5*s**15 - 336.*c**3*s**17 + 8.*c*s**19))/(4.*jnp.exp(3.*1j*phi)),
    "l10_mneg2": lambda c, s, phi: (sqrt(21./PI)*(495.*c**16*s**4 - 6336.*c**14*s**6 + 25872.*c**12*s**8 - 44352.*c**10*s**10 + 34650.*c**8*s**12 - 12320.*c**6*s**14 + 1848.*c**4*s**16 - 96.*c**2*s**18 + s**20))/(2.*jnp.exp(2.*1j*phi)),
    "l10_mneg1": lambda c, s, phi: (-3.*sqrt(7./PI)*(-220.*c**17*s**3 + 3960.*c**15*s**5 - 22176.*c**13*s**7 + 51744.*c**11*s**9 - 55440.*c**9*s**11 + 27720.*c**7*s**13 - 6160.*c**5*s**15 + 528.*c**3*s**17 - 12.*c*s**19))/(4.*jnp.exp(1j*phi)),
    "l10_m0": lambda c, s, phi: (3.*sqrt(35./(22.*PI))*(66.*c**18*s**2 - 1760.*c**16*s**4 + 13860.*c**14*s**6 - 44352.*c**12*s**8 + 64680.*c**10*s**10 - 44352.*c**8*s**12 + 13860.*c**6*s**14 - 1760.*c**4*s**16 + 66.*c**2*s**18))/2.,
    "l10_m1": lambda c, s, phi: (-3.*jnp.exp(1j*phi)*sqrt(7./PI)*(-12.*c**19*s + 528.*c**17*s**3 - 6160.*c**15*s**5 + 27720.*c**13*s**7 - 55440.*c**11*s**9 + 51744.*c**9*s**11 - 22176.*c**7*s**13 + 3960.*c**5*s**15 - 220.*c**3*s**17))/4.,
    "l10_m2": lambda c, s, phi: (jnp.exp(2.*1j*phi)*sqrt(21./PI)*(c**20 - 96.*c**18*s**2 + 1848.*c**16*s**4 - 12320.*c**14*s**6 + 34650.*c**12*s**8 - 44352.*c**10*s**10 + 25872.*c**8*s**12 - 6336.*c**6*s**14 + 495.*c**4*s**16))/2.,
    "l10_m3": lambda c, s, phi: -(jnp.exp(3.*1j*phi)*sqrt(273./(2.*PI))*(8.*c**19*s - 336.*c**17*s**3 + 3696.*c**15*s**5 - 15400.*c**13*s**7 + 27720.*c**11*s**9 - 22176.*c**9*s**11 + 7392.*c**7*s**13 - 792.*c**5*s**15))/4.,
    "l10_m4": lambda c, s, phi: (jnp.exp(4.*1j*phi)*sqrt(273./PI)*(28.*c**18*s**2 - 672.*c**16*s**4 + 4620.*c**14*s**6 - 12320.*c**12*s**8 + 13860.*c**10*s**10 - 6336.*c**8*s**12 + 924.*c**6*s**14))/4.,
    "l10_m5": lambda c, s, phi: -(jnp.exp(5.*1j*phi)*sqrt(1365./(2.*PI))*(56.*c**17*s**3 - 840.*c**15*s**5 + 3696.*c**13*s**7 - 6160.*c**11*s**9 + 3960.*c**9*s**11 - 792.*c**7*s**13))/4.,
    "l10_m6": lambda c, s, phi: jnp.exp(6.*1j*phi)*sqrt(273./(2.*PI))*(70.*c**16*s**4 - 672.*c**14*s**6 + 1848.*c**12*s**8 - 1760.*c**10*s**10 + 495.*c**8*s**12),
    "l10_m7": lambda c, s, phi: -(jnp.exp(7.*1j*phi)*sqrt(4641./(2.*PI))*(56.*c**15*s**5 - 336.*c**13*s**7 + 528.*c**11*s**9 - 220.*c**9*s**11))/2.,
    "l10_m8": lambda c, s, phi: (3.*jnp.exp(8.*1j*phi)*sqrt(1547./PI)*(28.*c**14*s**6 - 96.*c**12*s**8 + 66.*c**10*s**10))/2.,
    "l10_m9": lambda c, s, phi: (-3.*jnp.exp(9.*1j*phi)*sqrt(29393./(2.*PI))*(8.*c**13*s**7 - 12.*c**11*s**9))/2.,
    "l10_m10": lambda c, s, phi: 3.*jnp.exp(10.*1j*phi)*sqrt(146965./(2.*PI))*c**12*s**8,
}


def _swsh_jax(l: int, m: int, theta, phi):
    r"""Spin-weight −2 harmonic :math:`{}_{-2}Y_{\ell m}(\theta, \phi)`.

    ``l, m`` are static ints (:math:`2 \le \ell \le 10`, :math:`|m| \le \ell`);
    ``theta, phi`` (rad) may be JAX tracers, so the result is JIT/``grad`` friendly.
    """
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    key = _mode_key(l, m)
    try:
        return _YLM_TABLE[key](c, s, phi)
    except KeyError:
        raise ValueError(
            f"(l, m)=({l}, {m}) not supported; need 2 <= l <= 10 and |m| <= l."
        ) from None


def spin_weighted_spherical_harmonic(l: int, m: int, theta, phi):
    r"""Public single-mode SWSH :math:`{}_{-2}Y_{\ell m}(\theta, \phi)`.

    Wrapper around :func:`_swsh_jax`; to ``jax.jit`` directly, mark ``l, m`` static.
    """
    return _swsh_jax(l, m, theta, phi)


# ---------------------------------------------------------------------------
# Batch evaluation for mode arrays (deduplicated over unique (l, m))
# ---------------------------------------------------------------------------


def get_ylms_for_modes(l_arr, m_arr, theta, phi):
    r"""Batched :math:`{}_{-2}Y_{\ell m}(\theta, \phi)` for arrays of modes.

    Each distinct :math:`(\ell, m)` is evaluated once then gathered, so cost
    scales with the number of *unique* harmonics, not the mode count. Static
    ``l_arr, m_arr`` (int arrays); ``theta, phi`` (rad) may be JAX tracers.

    Returns ``(ylms_pos, ylms_neg)``, both ``(N_modes,)`` complex128, holding
    :math:`{}_{-2}Y_{\ell m}` and :math:`{}_{-2}Y_{\ell, -m}` (conjugate term).
    """
    l_np = np.asarray(l_arr, dtype=int)
    m_np = np.asarray(m_arr, dtype=int)

    # Unique (l, m) -> inverse index mapping each mode to its harmonic slot.
    keys, inv = np.unique(np.stack([l_np, m_np], axis=1), axis=0,
                          return_inverse=True)
    inv = np.asarray(inv).reshape(-1)

    uniq_pos = jnp.stack(
        [_swsh_jax(int(l), int(m), theta, phi) for l, m in keys]
    ).astype(jnp.complex128)
    uniq_neg = jnp.stack(
        [_swsh_jax(int(l), int(-m), theta, phi) for l, m in keys]
    ).astype(jnp.complex128)

    inv_j = jnp.asarray(inv)
    return uniq_pos[inv_j], uniq_neg[inv_j]

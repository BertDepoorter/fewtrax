"""Physical and mathematical constants for EMRI waveform computation.

All values are drawn from the :mod:`lisaconstants` package to ensure
consistency with the wider LISA data-analysis ecosystem.  Geometric units
(G = c = 1, mass measured in seconds) are used internally by the trajectory
integrator; the conversions are provided here.

References
----------
lisaconstants: https://pypi.org/project/lisaconstants/ 
"""

import math
import lisaconstants as _lc

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------

PI: float = math.pi
"""The mathematical constant π."""

# ---------------------------------------------------------------------------
# Fundamental SI constants
# ---------------------------------------------------------------------------

G_SI: float = _lc.GRAVITATIONAL_CONSTANT
"""Newtonian gravitational constant [m³ kg⁻¹ s⁻²]."""

C_SI: float = _lc.SPEED_OF_LIGHT
"""Speed of light in vacuum [m s⁻¹]."""

# ---------------------------------------------------------------------------
# Astronomical units
# ---------------------------------------------------------------------------

MSUN_SI: float = _lc.SOLAR_MASS
"""Solar mass [kg]."""

PC_SI: float = _lc.PARSEC
"""Parsec [m]."""

GPC_SI: float = 1.0e9 * PC_SI
"""Gigaparsec [m]."""

YEAR_SI: float = _lc.ASTRONOMICAL_YEAR
"""Astronomical (sidereal) year [s]."""

# ---------------------------------------------------------------------------
# Geometric unit conversions  (G = c = 1, mass unit = solar mass)
# ---------------------------------------------------------------------------

MTSUN_SI: float = _lc.GM_SUN / C_SI**3
"""Geometric time unit: G M_☉ / c³  [s].

For a source of mass M (in solar masses) the time unit is
    t_geo = M * MTSUN_SI   [s].

Computed from :data:`lisaconstants.GM_SUN` to avoid rounding errors in
the product G × M_☉.
"""

MRSUN_SI: float = _lc.GM_SUN / C_SI**2
"""Geometric length unit: G M_☉ / c²  [m]."""

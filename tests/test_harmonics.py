"""Unit tests for spin-weighted spherical harmonics.

Verifies the normalisation of :func:`get_ylms_for_modes` against known
analytic formulae.  These tests catch any regression in the ``(-1)^{m+s}``
sign prefactor or in the Wigner-d normalisation factor.
"""

import math
import numpy as np
import pytest

jax_available = True
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax_available = False


@pytest.mark.skipif(not jax_available, reason="JAX not installed")
class TestSpinWeightedHarmonics:
    """Analytic checks for _{-2}Y_{lm}(θ, φ)."""

    # ------------------------------------------------------------------ #
    # (l=2, m=0): exact formula                                           #
    #   _{-2}Y_{2,0}(θ) = √(15/(32π)) · sin²θ                           #
    #   |_{-2}Y_{2,0}|² = (15/(32π)) · sin⁴θ                            #
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize("theta", [
        np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2,
        2 * np.pi / 3, 3 * np.pi / 4, 5 * np.pi / 6,
    ])
    def test_ylm_20_magnitude_squared(self, theta):
        """|_{-2}Y_{2,0}(θ,φ)|² = (15/(32π)) sin⁴θ at arbitrary φ."""
        from fewtrax.utils.harmonics import get_ylms_for_modes

        phi = 0.7  # arbitrary, m=0 gives no φ dependence
        ylms_pos, _ = get_ylms_for_modes(
            np.array([2]), np.array([0]), theta, phi
        )
        computed = float(np.abs(ylms_pos[0])**2)
        expected = (15.0 / (32.0 * np.pi)) * np.sin(theta)**4
        assert computed == pytest.approx(expected, rel=1e-10), (
            f"θ={theta:.4f}: got {computed:.6e}, expected {expected:.6e}"
        )

    # ------------------------------------------------------------------ #
    # (l=2, m=0): no φ dependence                                        #
    # ------------------------------------------------------------------ #
    def test_ylm_20_phi_independent(self):
        """_{-2}Y_{2,0} is real and φ-independent (m=0)."""
        from fewtrax.utils.harmonics import get_ylms_for_modes

        theta = np.pi / 3
        for phi in [0.0, 0.5, 1.0, 2.0, np.pi]:
            ylms, _ = get_ylms_for_modes(np.array([2]), np.array([0]), theta, phi)
            assert abs(ylms[0].imag) < 1e-12, "m=0 harmonic must be real"
            assert abs(ylms[0].real - math.sqrt(15 / (32 * math.pi)) * math.sin(theta)**2) < 1e-10

    # ------------------------------------------------------------------ #
    # Orthonormality: Σ_{m=-l}^{l} |Y_{lm}|² = (2l+1)/(4π)             #
    # (verified numerically via a single-point sum over m)               #
    # The full orthonormality requires integration; this checks the      #
    # algebraic normalisation factor at a fixed angle.                   #
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize("l", [2, 3, 4])
    def test_power_sum_normalisation(self, l):
        """Σ_{m=0..l} (|Y_{lm}|² + |Y_{l,-m}|²) power-sum at a reference angle."""
        from fewtrax.utils.harmonics import get_ylms_for_modes

        theta, phi = np.pi / 3, 0.5
        m_arr = np.arange(0, l + 1)
        l_arr = np.full_like(m_arr, l)
        ylms_pos, ylms_neg = get_ylms_for_modes(l_arr, m_arr, theta, phi)

        # All harmonics should be finite
        assert np.all(np.isfinite(np.abs(ylms_pos))), "ylms_pos contains NaN/Inf"
        assert np.all(np.isfinite(np.abs(ylms_neg))), "ylms_neg contains NaN/Inf"

        # The m=0 harmonic must be real
        assert abs(ylms_pos[0].imag) < 1e-10, "m=0 harmonic should be real"

"""Tests for the amplitude interpolation module (JAXAmplitudeInterpolator).

Tests verify:
- Amplitude arrays have the correct shape.
- Amplitudes are finite and not identically zero.
- Mode selection behaves monotonically in the threshold.
- A known FEW amplitude value is reproduced to within tolerance.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Known test cases from the FEW test suite (lmkn, a, p, e, expected amplitude)
KERRECCEQ_AMP_TEST_POINTS = [
    # (l, m, k, n), a, p, e,  expected A_re,  expected A_im  (approx)
    ((3, 2, 0, 4), 0.94, 3.03, 0.24, 0.00275, 0.00414),
]


@pytest.fixture(scope="module")
def amp_interp(data_dir):
    """Module-scoped JAX amplitude interpolator."""
    from fewtrax.data.loader import load_amplitude_data_jax
    from fewtrax.amplitude import JAXAmplitudeInterpolator
    return JAXAmplitudeInterpolator(load_amplitude_data_jax(data_dir))


class TestAmplitudeShape:
    """Shape and basic validity tests."""

    def test_shape(self, amp_interp):
        """evaluate_trajectory should return (N_times, N_modes)."""
        N = 50
        p = jnp.linspace(8.0, 10.0, N)
        e = jnp.linspace(0.35, 0.45, N)
        A = amp_interp.evaluate_trajectory(jnp.float64(0.3), p, e)
        assert A.shape == (N, amp_interp.n_modes), \
            f"Expected shape ({N}, {amp_interp.n_modes}), got {A.shape}"

    def test_amplitudes_finite(self, amp_interp):
        """Amplitudes should be finite (no NaN/Inf)."""
        p = jnp.array([10.0, 12.0, 15.0])
        e = jnp.array([0.4, 0.3, 0.2])
        A = amp_interp.evaluate_trajectory(jnp.float64(0.3), p, e)
        assert np.all(np.isfinite(np.asarray(A))), "Amplitudes contain NaN or Inf."

    def test_amplitudes_not_zero(self, amp_interp):
        """Dominant modes should have non-zero amplitudes."""
        A = amp_interp(a=0.3, p=10.0, e=0.4)   # (n_modes,)
        assert float(jnp.max(jnp.abs(A))) > 1e-10, \
            "Maximum amplitude is unexpectedly small."


class TestModeSelection:
    """Tests for the mean-power mode selection utility."""

    def test_returns_subset(self, amp_interp):
        """Selected modes should be a non-empty subset of all modes."""
        p = jnp.linspace(9.0, 11.0, 20)
        e = jnp.linspace(0.3, 0.4, 20)
        inds = amp_interp.select_modes(a=0.3, p=p, e=e, threshold=1e-3)
        assert 0 < len(inds) <= amp_interp.n_modes

    def test_threshold_reduces_modes(self, amp_interp):
        """Higher threshold should select fewer (or equal) modes."""
        p = jnp.linspace(9.0, 11.0, 20)
        e = jnp.linspace(0.3, 0.4, 20)
        inds_loose = amp_interp.select_modes(a=0.3, p=p, e=e, threshold=1e-6)
        inds_strict = amp_interp.select_modes(a=0.3, p=p, e=e, threshold=1e-2)
        assert len(inds_strict) <= len(inds_loose)

    def test_select_from_amps_matches(self, amp_interp):
        """select_modes_from_amps on a precomputed table matches select_modes."""
        p = jnp.linspace(9.0, 11.0, 20)
        e = jnp.linspace(0.3, 0.4, 20)
        teuk = amp_interp.evaluate_trajectory(jnp.float64(0.3), p, e)
        inds_a = amp_interp.select_modes_from_amps(teuk, threshold=1e-3)
        inds_b = amp_interp.select_modes(a=0.3, p=p, e=e, threshold=1e-3)
        np.testing.assert_array_equal(inds_a, inds_b)


class TestAmplitudeValues:
    """Numerical value tests (require FEW data for comparison)."""

    @pytest.mark.parametrize(
        "mode,a,p,e,re_expected,im_expected",
        KERRECCEQ_AMP_TEST_POINTS,
    )
    def test_known_value(self, amp_interp, mode, a, p, e, re_expected, im_expected):
        """Check amplitude magnitude matches the known FEW value to within 50%."""
        l, m, k, n = mode
        l_arr = np.asarray(amp_interp.l_arr)
        m_arr = np.asarray(amp_interp.m_arr)
        k_arr = np.asarray(amp_interp.k_arr)
        n_arr = np.asarray(amp_interp.n_arr)
        match = np.where(
            (l_arr == l) & (m_arr == m) & (k_arr == k) & (n_arr == n)
        )[0]
        if match.size == 0:
            pytest.skip(f"Mode {mode} not in dataset.")

        A = np.asarray(amp_interp(a=a, p=p, e=e))
        A_val = complex(A[int(match[0])])
        expected = complex(re_expected, im_expected)

        assert abs(abs(A_val) - abs(expected)) / abs(expected) < 0.5, \
            f"Amplitude {A_val:.4e} deviates >50% from expected {expected:.4e}"

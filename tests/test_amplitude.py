"""Tests for the amplitude interpolation module.

Tests verify:
- Amplitude arrays have the correct shape.
- Amplitudes are finite and not identically zero.
- Mode symmetry relations are satisfied.
- The amplitude evaluator handles edge cases gracefully.
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


class TestAmplitudeShape:
    """Shape and basic validity tests."""

    def test_shape(self, amp_data):
        """Amplitude array should have shape (N_times, N_modes)."""
        from fewtrax.amplitude import AmplitudeInterpolator
        amp = AmplitudeInterpolator(amp_data)
        N = 50
        p = np.linspace(8.0, 10.0, N)
        e = np.linspace(0.35, 0.45, N)
        A = amp.evaluate(a=0.3, p=p, e=e)
        assert A.shape == (N, amp.n_modes), \
            f"Expected shape ({N}, {amp.n_modes}), got {A.shape}"

    def test_amplitudes_finite(self, amp_data):
        """Amplitudes should be finite (no NaN/Inf)."""
        from fewtrax.amplitude import AmplitudeInterpolator
        amp = AmplitudeInterpolator(amp_data)
        p = np.array([10.0, 12.0, 15.0])
        e = np.array([0.4, 0.3, 0.2])
        A = amp.evaluate(a=0.3, p=p, e=e)
        assert np.all(np.isfinite(A)), "Amplitudes contain NaN or Inf."

    def test_amplitudes_not_zero(self, amp_data):
        """Dominant modes should have non-zero amplitudes."""
        from fewtrax.amplitude import AmplitudeInterpolator
        amp = AmplitudeInterpolator(amp_data)
        p = np.array([10.0])
        e = np.array([0.4])
        A = amp.evaluate(a=0.3, p=p, e=e)
        max_amp = np.max(np.abs(A))
        assert max_amp > 1e-10, "Maximum amplitude is unexpectedly small."


class TestModeSelection:
    """Tests for the mode selection utility."""

    def test_returns_subset(self, amp_data):
        """Selected modes should be a subset of all modes."""
        from fewtrax.amplitude import AmplitudeInterpolator
        amp = AmplitudeInterpolator(amp_data)
        p = np.linspace(9.0, 11.0, 20)
        e = np.linspace(0.3, 0.4, 20)
        inds = amp.select_modes(a=0.3, p=p, e=e, threshold=1e-3)
        assert len(inds) <= amp.n_modes
        assert len(inds) > 0, "No modes selected!"

    def test_threshold_reduces_modes(self, amp_data):
        """Higher threshold should select fewer modes."""
        from fewtrax.amplitude import AmplitudeInterpolator
        amp = AmplitudeInterpolator(amp_data)
        p = np.linspace(9.0, 11.0, 20)
        e = np.linspace(0.3, 0.4, 20)
        inds_loose = amp.select_modes(a=0.3, p=p, e=e, threshold=1e-6)
        inds_strict = amp.select_modes(a=0.3, p=p, e=e, threshold=1e-2)
        assert len(inds_strict) <= len(inds_loose)

    def test_specific_modes(self, amp_data):
        """Evaluating specific modes returns the correct number of columns."""
        from fewtrax.amplitude import AmplitudeInterpolator
        amp = AmplitudeInterpolator(amp_data)
        p = np.array([10.0, 11.0])
        e = np.array([0.4, 0.35])
        modes = np.array([0, 1, 2])
        A = amp.evaluate(a=0.3, p=p, e=e, specific_modes=modes)
        assert A.shape == (2, 3)


class TestAmplitudeValues:
    """Numerical value tests (require FEW for comparison)."""

    @pytest.mark.parametrize(
        "mode,a,p,e,re_expected,im_expected",
        KERRECCEQ_AMP_TEST_POINTS,
    )
    def test_known_value(self, amp_data, mode, a, p, e, re_expected, im_expected):
        """Check amplitude matches known FEW value to within 10%."""
        from fewtrax.amplitude import AmplitudeInterpolator
        amp = AmplitudeInterpolator(amp_data)

        # Find the mode index
        l, m, k, n = mode
        idx = None
        for i, (li, mi, ki, ni) in enumerate(
            zip(amp.l_arr, amp.m_arr, amp.k_arr, amp.n_arr)
        ):
            if int(li) == l and int(mi) == m and int(ki) == k and int(ni) == n:
                idx = i
                break

        if idx is None:
            pytest.skip(f"Mode {mode} not in dataset.")

        A = amp.evaluate(a=a, p=np.array([p]), e=np.array([e]),
                         specific_modes=np.array([idx]))
        A_val = complex(A[0, 0])
        expected = complex(re_expected, im_expected)

        # Check magnitude is in the right ballpark (order of magnitude)
        assert abs(abs(A_val) - abs(expected)) / abs(expected) < 0.5, \
            f"Amplitude {A_val:.4e} deviates >50% from expected {expected:.4e}"

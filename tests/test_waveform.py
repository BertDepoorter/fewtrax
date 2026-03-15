"""End-to-end waveform generation tests.

Tests verify:
- Waveform output shape and finiteness.
- Physical units are correct (strain ≈ 10^-16 to 10^-22 at 1 Gpc for typical EMRI).
- hp and hx are real-valued arrays.
- The waveform changes with parameter variations in the expected direction.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestWaveformBasic:
    """Basic correctness tests for the full waveform pipeline."""

    def test_shape_and_dtype(self, waveform_gen, emri_params):
        """hp and hx should be real 1-D arrays."""
        hp, hx = waveform_gen(**emri_params)
        assert hp.ndim == 1
        assert hx.ndim == 1
        assert hp.dtype in (jnp.float32, jnp.float64)

    def test_hp_hx_real(self, waveform_gen, emri_params):
        """hp and hx should be real (imaginary part = 0)."""
        hp, hx = waveform_gen(**emri_params)
        assert jnp.all(jnp.isreal(hp)), "hp has imaginary component."
        assert jnp.all(jnp.isreal(hx)), "hx has imaginary component."

    def test_finite(self, waveform_gen, emri_params):
        """All waveform samples should be finite."""
        hp, hx = waveform_gen(**emri_params)
        assert jnp.all(jnp.isfinite(hp)), "hp contains NaN/Inf."
        assert jnp.all(jnp.isfinite(hx)), "hx contains NaN/Inf."

    def test_strain_magnitude(self, waveform_gen, emri_params):
        """Strain should be in a physically reasonable range."""
        hp, hx = waveform_gen(**emri_params)
        max_h = float(jnp.max(jnp.abs(hp)))
        # Typical: ~1e-17 to 1e-20 at 1 Gpc for M=1e6, mu=10
        assert 1e-25 < max_h < 1e-14, \
            f"Strain |h|_max = {max_h:.2e} is outside physical range."

    def test_not_identically_zero(self, waveform_gen, emri_params):
        """Waveform should not be identically zero."""
        hp, hx = waveform_gen(**emri_params)
        assert jnp.any(jnp.abs(hp) > 0.0), "hp is identically zero."
        assert jnp.any(jnp.abs(hx) > 0.0), "hx is identically zero."


class TestWaveformParameterDependence:
    """Test that waveforms change correctly with parameter variation."""

    def test_distance_scaling(self, waveform_gen, emri_params):
        """Strain should scale as 1/distance."""
        p1 = dict(emri_params, dist=1.0)
        p2 = dict(emri_params, dist=2.0)
        hp1, _ = waveform_gen(**p1)
        hp2, _ = waveform_gen(**p2)
        ratio = float(jnp.max(jnp.abs(hp1))) / float(jnp.max(jnp.abs(hp2)))
        assert abs(ratio - 2.0) / 2.0 < 0.05, \
            f"Expected 2:1 strain ratio, got {ratio:.2f}."

    def test_mass_ratio_scaling(self, waveform_gen, emri_params):
        """Strain should scale approximately linearly with mass ratio mu/M."""
        p1 = dict(emri_params, mu=10.0)
        p2 = dict(emri_params, mu=20.0)
        hp1, _ = waveform_gen(**p1)
        hp2, _ = waveform_gen(**p2)
        ratio = float(jnp.max(jnp.abs(hp2))) / float(jnp.max(jnp.abs(hp1)))
        assert 1.5 < ratio < 2.5, \
            f"Expected ~2:1 mu scaling, got {ratio:.2f}."

    def test_spin_nonzero(self, waveform_gen, emri_params):
        """Kerr and Schwarzschild waveforms should differ."""
        p_kerr = dict(emri_params, a=0.5)
        p_schw = dict(emri_params, a=0.0)
        hp_k, _ = waveform_gen(**p_kerr)
        hp_s, _ = waveform_gen(**p_schw)
        diff = float(jnp.mean(jnp.abs(hp_k - hp_s[:hp_k.size])))
        assert diff > 0.0, "Kerr and Schwarzschild waveforms are identical."


class TestWaveformHarmonicTrack:
    """Test the harmonic track utility."""

    def test_track_shape(self, waveform_gen, emri_params):
        """Harmonic track should be a 1-D array of frequencies."""
        t, f = waveform_gen.get_harmonic_track(
            l=2, m=2, k=0, n=1,
            M=emri_params["M"], mu=emri_params["mu"],
            a=emri_params["a"], p0=emri_params["p0"],
            e0=emri_params["e0"], T=emri_params["T"],
        )
        assert t.ndim == 1
        assert f.ndim == 1
        assert f.shape == t.shape

    def test_track_positive_frequencies(self, waveform_gen, emri_params):
        """Frequency track should be non-negative."""
        _, f = waveform_gen.get_harmonic_track(
            l=2, m=2, k=0, n=1,
            M=emri_params["M"], mu=emri_params["mu"],
            a=emri_params["a"], p0=emri_params["p0"],
            e0=emri_params["e0"], T=emri_params["T"],
        )
        assert jnp.all(f >= 0.0), "Frequency track has negative values."

    def test_track_frequency_increases(self, waveform_gen, emri_params):
        """Dominant mode frequency should increase as the orbit tightens."""
        _, f = waveform_gen.get_harmonic_track(
            l=2, m=2, k=0, n=0,
            M=emri_params["M"], mu=emri_params["mu"],
            a=emri_params["a"], p0=emri_params["p0"],
            e0=emri_params["e0"], T=emri_params["T"],
        )
        f_np = np.asarray(f)
        valid = np.isfinite(f_np)
        if valid.sum() > 2:
            # Frequency should trend upward (chirp)
            first_half = float(np.mean(f_np[valid][:valid.sum() // 2]))
            second_half = float(np.mean(f_np[valid][valid.sum() // 2:]))
            assert second_half >= first_half * 0.99, \
                "Frequency should not decrease significantly during inspiral."


class TestWaveformFrequencyDomain:
    """Test the frequency-domain transform of the waveform."""

    def test_fourier_transform(self, waveform_gen, emri_params):
        """Frequency-domain waveform should be obtainable via FFT."""
        from fewtrax.summation.modes import to_frequency_domain
        hp, _ = waveform_gen(**emri_params)
        dt = emri_params["dt"]
        freqs, h_tilde = to_frequency_domain(hp, dt=dt)
        assert freqs.shape[0] == hp.shape[0] // 2 + 1
        assert jnp.all(jnp.isfinite(h_tilde))

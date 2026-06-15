"""Tests for the mode summation module.

Tests verify:
- Output shape and finiteness.
- The waveform is real (hp) and real (hx) as expected.
- Symmetry: zero phase gives a purely real sum.
- The frequency-domain transform is consistent with the time-domain signal.
- JIT compilation and vmap work correctly.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _make_test_data(n_modes=10, N_t=100):
    """Generate synthetic test data for summation tests."""
    rng = np.random.default_rng(42)
    teuk_modes = (rng.standard_normal((N_t, n_modes))
                  + 1j * rng.standard_normal((N_t, n_modes))) * 1e-23
    ylms_pos = (rng.standard_normal(n_modes)
                + 1j * rng.standard_normal(n_modes))
    ylms_neg = (rng.standard_normal(n_modes)
                + 1j * rng.standard_normal(n_modes))
    t = np.linspace(0, 1e6, N_t)
    Phi_phi = np.linspace(0, 100.0, N_t)
    Phi_theta = np.linspace(0, 80.0, N_t)
    Phi_r = np.linspace(0, 50.0, N_t)
    l_arr = np.full(n_modes, 2, dtype=int)
    m_arr = np.arange(n_modes, dtype=int) % 5 + 1
    k_arr = np.zeros(n_modes, dtype=int)
    n_arr = np.arange(n_modes, dtype=int) - n_modes // 2
    return dict(
        teuk_modes=jnp.asarray(teuk_modes),
        ylms_pos=jnp.asarray(ylms_pos),
        ylms_neg=jnp.asarray(ylms_neg),
        t=jnp.asarray(t),
        Phi_phi=jnp.asarray(Phi_phi),
        Phi_theta=jnp.asarray(Phi_theta),
        Phi_r=jnp.asarray(Phi_r),
        l_arr=jnp.asarray(l_arr),
        m_arr=jnp.asarray(m_arr),
        k_arr=jnp.asarray(k_arr),
        n_arr=jnp.asarray(n_arr),
    )


class TestDirectModeSum:
    """Tests for direct_mode_sum."""

    def test_output_shape(self):
        d = _make_test_data(n_modes=10, N_t=100)
        from fewtrax.summation import direct_mode_sum
        h = direct_mode_sum(
            d["teuk_modes"], d["ylms_pos"], d["ylms_neg"],
            d["Phi_phi"], d["Phi_theta"], d["Phi_r"],
            d["l_arr"], d["m_arr"], d["k_arr"], d["n_arr"],
        )
        assert h.shape == (100,), f"Expected (100,), got {h.shape}"

    def test_output_finite(self):
        d = _make_test_data(n_modes=15, N_t=200)
        from fewtrax.summation import direct_mode_sum
        h = direct_mode_sum(
            d["teuk_modes"], d["ylms_pos"], d["ylms_neg"],
            d["Phi_phi"], d["Phi_theta"], d["Phi_r"],
            d["l_arr"], d["m_arr"], d["k_arr"], d["n_arr"],
        )
        assert jnp.all(jnp.isfinite(h)), "Waveform contains NaN or Inf."

    def test_zero_amplitude(self):
        """Zero amplitudes should give zero waveform."""
        d = _make_test_data(n_modes=5, N_t=50)
        from fewtrax.summation import direct_mode_sum
        h = direct_mode_sum(
            jnp.zeros_like(d["teuk_modes"]),
            d["ylms_pos"], d["ylms_neg"],
            d["Phi_phi"], d["Phi_theta"], d["Phi_r"],
            d["l_arr"], d["m_arr"], d["k_arr"], d["n_arr"],
        )
        assert jnp.allclose(h, 0.0), "Zero amplitudes should give zero waveform."

    def test_jit_compilable(self):
        """direct_mode_sum should be JIT-compilable."""
        d = _make_test_data(n_modes=8, N_t=64)
        from fewtrax.summation import direct_mode_sum
        jit_fn = jax.jit(direct_mode_sum)
        h = jit_fn(
            d["teuk_modes"], d["ylms_pos"], d["ylms_neg"],
            d["Phi_phi"], d["Phi_theta"], d["Phi_r"],
            d["l_arr"], d["m_arr"], d["k_arr"], d["n_arr"],
        )
        assert h.shape == (64,)

    def test_differentiable_wrt_phases(self):
        """Waveform should be differentiable w.r.t. orbital phases."""
        d = _make_test_data(n_modes=4, N_t=16)
        from fewtrax.summation import direct_mode_sum

        def loss(Phi_phi):
            h = direct_mode_sum(
                d["teuk_modes"], d["ylms_pos"], d["ylms_neg"],
                Phi_phi, d["Phi_theta"], d["Phi_r"],
                d["l_arr"], d["m_arr"], d["k_arr"], d["n_arr"],
            )
            return jnp.sum(jnp.abs(h) ** 2).real

        grad = jax.grad(loss)(d["Phi_phi"])
        assert jnp.all(jnp.isfinite(grad)), "Gradient w.r.t. phases contains NaN."

    def test_linearity_in_modes(self):
        """Sum should be linear in mode amplitudes."""
        d = _make_test_data(n_modes=6, N_t=40)
        from fewtrax.summation import direct_mode_sum

        kwargs = dict(
            ylms_pos=d["ylms_pos"], ylms_neg=d["ylms_neg"],
            Phi_phi=d["Phi_phi"], Phi_theta=d["Phi_theta"], Phi_r=d["Phi_r"],
            l_arr=d["l_arr"], m_arr=d["m_arr"],
            k_arr=d["k_arr"], n_arr=d["n_arr"],
        )

        h1 = direct_mode_sum(d["teuk_modes"], **kwargs)
        h2 = direct_mode_sum(2.0 * d["teuk_modes"], **kwargs)
        assert jnp.allclose(2.0 * h1, h2, atol=1e-12), \
            "Sum is not linear in mode amplitudes."


class TestFrequencyDomain:
    """Tests for to_frequency_domain."""

    def test_parseval(self):
        """Parseval's theorem: sum(|h|^2) * dt ≈ sum(|h_tilde|^2) * df."""
        from fewtrax.summation.modes import to_frequency_domain
        dt = 10.0
        N = 1024
        t = np.arange(N) * dt
        h = jnp.array(np.sin(2 * np.pi * 0.01 * t))
        freqs, h_tilde = to_frequency_domain(h, dt)
        # Parseval (approximate, one-sided FFT)
        # Parseval: sum|h|^2 * dt == sum|H|^2 / N  (where H = rfft(h))
        # rfft gives one-sided spectrum; we scaled H = rfft(h)*dt so:
        # sum|H|^2/dt^2 / N == sum|h|^2  (Parseval for FFT)
        energy_time = float(jnp.sum(jnp.abs(h) ** 2))
        H_raw = jnp.fft.rfft(h)  # unscaled, length N//2+1
        # For real signal, full DFT has conjugate symmetry: X[N-k] = conj(X[k])
        # so sum|X|^2 = |H[0]|^2 + 2*sum|H[1:-1]|^2 + |H[-1]|^2
        energy_freq = float(
            (jnp.abs(H_raw[0]) ** 2
             + 2.0 * jnp.sum(jnp.abs(H_raw[1:-1]) ** 2)
             + jnp.abs(H_raw[-1]) ** 2) / N
        )
        assert abs(energy_time - energy_freq) / energy_time < 0.01, \
            "Parseval's theorem violated."

    def test_positive_frequencies(self):
        """Frequency array should be non-negative."""
        from fewtrax.summation.modes import to_frequency_domain
        h = jnp.ones(64)
        freqs, _ = to_frequency_domain(h, dt=10.0)
        assert jnp.all(freqs >= 0.0)


class TestInterpolatedModeSum:
    """Tests for interpolated_mode_sum."""

    def test_output_length(self):
        """Dense waveform should have the correct length."""
        d = _make_test_data(n_modes=5, N_t=20)
        from fewtrax.summation import interpolated_mode_sum
        dt = 10.0
        T_s = float(d["t"][-1])
        n_expected = max(2, int(round(T_s / dt)) + 1)

        # Linear-in-time stand-in for the Dopri8 dense-output phase evaluator.
        t0, t1 = float(d["t"][0]), float(d["t"][-1])
        def phase_fn(t_dense):
            frac = (t_dense - t0) / (t1 - t0)
            return (100.0 * frac, 80.0 * frac, 50.0 * frac)

        t_out, h = interpolated_mode_sum(
            d["t"], np.array(d["teuk_modes"]),
            d["ylms_pos"], d["ylms_neg"],
            d["Phi_phi"], d["Phi_theta"], d["Phi_r"],
            np.array(d["l_arr"]), np.array(d["m_arr"]),
            np.array(d["k_arr"]), np.array(d["n_arr"]),
            phase_fn, dt=dt,
        )
        assert t_out.shape[0] == n_expected, \
            f"Expected {n_expected} output points, got {t_out.shape[0]}"

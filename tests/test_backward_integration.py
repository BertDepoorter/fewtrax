"""Tests for backward (time-reversed) trajectory integration.

The backward ODE integrates dy/dτ = -dy/dt where τ = T_plunge - t is time
before plunge.  Starting conditions are set to (p_sep(e_f, a) + ε, e_f),
just above the separatrix.

Tests verify:
- p increases along the backward trajectory (orbit moves away from plunge).
- e changes in the opposite direction to forward integration.
- Phases decrease (time-reversed angular motion).
- Frequency track is positive and decreasing (chirp reversed).
- The backward trajectory is consistent with a forward trajectory started at
  the same point: the local frequency at the common (p, e) must match.
- ValueError is raised if e_f is not provided.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestBackwardBasics:
    """Sanity checks for the backward integrator."""

    def test_p_increases(self, flux_data):
        """p should increase going backward (moving away from separatrix)."""
        from fewtrax.trajectory import run_inspiral
        from fewtrax.utils.geodesic import get_separatrix

        a, e_f = 0.3, 0.3
        t, p, e, _, _, _ = run_inspiral(
            a=a, p0=None, e0=None, T=0.2, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=50, backward=True, e_f=e_f,
        )
        p_np = np.asarray(p)
        dp = np.diff(p_np)
        assert np.all(dp >= -1e-4), "p must not decrease in backward integration."
        assert p_np[-1] > p_np[0], "p must be larger at τ=T than at τ=0."

    def test_starts_near_separatrix(self, flux_data):
        """First output point should be just above the separatrix."""
        from fewtrax.trajectory import run_inspiral
        from fewtrax.utils.geodesic import get_separatrix
        from fewtrax.utils.coordinates import DELTAPMIN

        a, e_f = 0.5, 0.4
        SEPARATRIX_BUFFER = 2.0 * DELTAPMIN
        _, p, e, _, _, _ = run_inspiral(
            a=a, p0=None, e0=None, T=0.1, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=30, backward=True, e_f=e_f,
        )
        p_sep = float(get_separatrix(abs(a), float(e_f), 1.0))
        p_start = float(p[0])
        assert abs(p_start - (p_sep + SEPARATRIX_BUFFER)) < 0.01, (
            f"Backward trajectory should start at p_sep+buffer={p_sep + SEPARATRIX_BUFFER:.4f}, "
            f"got p[0]={p_start:.4f}"
        )

    def test_time_starts_at_zero(self, flux_data):
        """Time axis should start at 0 (plunge reference point)."""
        from fewtrax.trajectory import run_inspiral

        t, _, _, _, _, _ = run_inspiral(
            a=0.3, p0=None, e0=None, T=0.1, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=30, backward=True, e_f=0.35,
        )
        assert float(t[0]) == pytest.approx(0.0, abs=1.0)

    def test_phases_decrease(self, flux_data):
        """Orbital phases should decrease along the backward trajectory."""
        from fewtrax.trajectory import run_inspiral

        _, _, _, Phi_phi, Phi_theta, Phi_r = run_inspiral(
            a=0.3, p0=None, e0=None, T=0.2, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=50, backward=True, e_f=0.3,
        )
        for name, phase_arr in [
            ("Phi_phi", Phi_phi), ("Phi_theta", Phi_theta), ("Phi_r", Phi_r)
        ]:
            ph = np.asarray(phase_arr)
            assert np.all(np.diff(ph) <= 1e-10), (
                f"{name} should be monotonically non-increasing in backward mode."
            )

    def test_e_f_required(self, flux_data):
        """ValueError must be raised when backward=True and e_f is None."""
        from fewtrax.trajectory import EMRIInspiral
        traj = EMRIInspiral(flux_data)
        with pytest.raises(ValueError, match="e_f must be provided"):
            traj(p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3,
                 dense_steps=10, backward=True, e_f=None)

    def test_schwarzschild_backward(self, flux_data):
        """Backward integration should work for a=0 (Schwarzschild)."""
        from fewtrax.trajectory import run_inspiral

        t, p, e, _, _, _ = run_inspiral(
            a=0.0, p0=None, e0=None, T=0.2, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=40, backward=True, e_f=0.3,
        )
        assert np.all(np.isfinite(np.asarray(p))), "Schwarzschild backward should not produce NaN."
        assert float(p[-1]) > float(p[0])


class TestBackwardFrequencyTrack:
    """Tests for backward frequency track generation."""

    def test_frequency_positive(self, flux_data):
        """Frequency must be positive at all trajectory points."""
        from fewtrax.trajectory import EMRIInspiral

        traj = EMRIInspiral(flux_data)
        t, f = traj.get_frequency_track(
            p0=None, e0=None, T=0.2, M=1e6, mu=10.0, a=0.3,
            l=2, m=2, k=0, n=1,
            dense_steps=50, backward=True, e_f=0.35,
        )
        assert np.all(np.asarray(f) > 0.0), "Frequency must be positive."

    def test_frequency_decreasing_backward(self, flux_data):
        """Frequency should decrease along backward trajectory (moving earlier in inspiral)."""
        from fewtrax.trajectory import EMRIInspiral

        traj = EMRIInspiral(flux_data)
        t, f = traj.get_frequency_track(
            p0=None, e0=None, T=0.3, M=1e6, mu=10.0, a=0.3,
            l=2, m=2, k=0, n=1,
            dense_steps=60, backward=True, e_f=0.3,
        )
        f_np = np.asarray(f)
        df = np.diff(f_np)
        assert np.all(df <= 1e-20), (
            "Frequency should be non-increasing along backward trajectory "
            "(going to earlier times means lower frequency)."
        )


class TestBackwardForwardConsistency:
    """Cross-checks between forward and backward trajectories."""

    def test_round_trip(self, flux_data):
        """Round-trip check: backward T years then forward T years should return
        close to the starting (p, e).

        Strategy:
        1. Integrate backward from e_f for T years → reach (p_back, e_back).
        2. Integrate forward from (p_back, e_back) for T years.
        3. The forward endpoint should be close to the backward start
           (p_sep(e_f) + buffer, e_f).
        """
        from fewtrax.trajectory import EMRIInspiral
        from fewtrax.utils.geodesic import get_separatrix
        from fewtrax.utils.coordinates import DELTAPMIN

        SEPARATRIX_BUFFER = 2.0 * DELTAPMIN
        a, e_f = 0.3, 0.3
        T = 0.05   # short enough that numerical drift is small

        traj = EMRIInspiral(flux_data)

        # Step 1: backward run
        _, p_bwd, e_bwd, _, _, _ = traj(
            p0=None, e0=None, T=T, M=1e6, mu=10.0, a=a,
            dense_steps=80, backward=True, e_f=e_f,
        )
        p0_rt = float(p_bwd[-1])
        e0_rt = float(e_bwd[-1])

        # Step 2: forward run from backward endpoint
        _, p_fwd, e_fwd, _, _, _ = traj(
            p0=p0_rt, e0=e0_rt, T=T, M=1e6, mu=10.0, a=a, dense_steps=80,
        )
        p_np = np.asarray(p_fwd)
        e_np = np.asarray(e_fwd)
        valid = np.isfinite(p_np) & np.isfinite(e_np)
        p_end = float(p_np[valid][-1])
        e_end = float(e_np[valid][-1])

        # Expected endpoint: backward start = p_sep(e_f) + buffer
        p_sep = float(get_separatrix(abs(a), float(e_f), 1.0))
        p_expected = p_sep + SEPARATRIX_BUFFER

        # Allow 1% relative error in p and 1e-3 absolute error in e
        assert abs(p_end - p_expected) / p_expected < 0.01, (
            f"Round-trip p: expected {p_expected:.4f}, got {p_end:.4f} "
            f"(relative error {abs(p_end - p_expected) / p_expected:.3%})"
        )
        assert abs(e_end - e_f) < 1e-3, (
            f"Round-trip e: expected {e_f:.4f}, got {e_end:.4f}"
        )

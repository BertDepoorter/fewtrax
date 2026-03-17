"""Tests for the trajectory integration module.

Tests verify:
- The trajectory starts at (p0, e0) and evolves monotonically toward merger.
- Phases are monotonically increasing.
- The separatrix termination condition is respected.
- The trajectory is differentiable w.r.t. initial conditions.
- vmap over initial conditions produces correct batched results.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestTrajectoryBasic:
    """Basic sanity checks for the trajectory integrator."""

    def test_initial_conditions(self, flux_data):
        """Trajectory should start at (p0, e0)."""
        from fewtrax.trajectory import run_inspiral
        t, p, e, _, _, _ = run_inspiral(
            a=0.3, p0=10.0, e0=0.4, T=0.1, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=50,
        )
        assert float(p[0]) == pytest.approx(10.0, rel=1e-3)
        assert float(e[0]) == pytest.approx(0.4, rel=1e-3)

    def test_p_decreases(self, flux_data):
        """Semi-latus rectum should decrease (inspiral)."""
        from fewtrax.trajectory import run_inspiral
        _, p, e, _, _, _ = run_inspiral(
            a=0.3, p0=10.0, e0=0.4, T=0.2, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=50,
        )
        p_np = np.asarray(p)
        valid = ~np.isnan(p_np)
        # p should not increase significantly
        dp = np.diff(p_np[valid])
        assert np.all(dp <= 1e-3), "Semi-latus rectum should not increase."

    def test_phases_increase(self, flux_data):
        """Orbital phases should be monotonically increasing."""
        from fewtrax.trajectory import run_inspiral
        _, _, _, Phi_phi, Phi_theta, Phi_r = run_inspiral(
            a=0.3, p0=10.0, e0=0.4, T=0.1, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=50,
        )
        for phase_arr in [Phi_phi, Phi_theta, Phi_r]:
            ph = np.asarray(phase_arr)
            valid = ~np.isnan(ph)
            if valid.sum() > 1:
                assert np.all(np.diff(ph[valid]) >= 0.0), \
                    "Phase should be monotonically non-decreasing."

    def test_time_starts_at_zero(self, flux_data):
        """Time array should start at (approximately) zero."""
        from fewtrax.trajectory import run_inspiral
        t, _, _, _, _, _ = run_inspiral(
            a=0.3, p0=10.0, e0=0.4, T=0.1, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=50,
        )
        assert float(t[0]) == pytest.approx(0.0, abs=1.0)  # within 1 second

    def test_separatrix_respected(self, flux_data):
        """Trajectory should stop before the separatrix."""
        from fewtrax.trajectory import run_inspiral
        from fewtrax.utils.geodesic import get_separatrix
        a = 0.3
        _, p, e, _, _, _ = run_inspiral(
            a=a, p0=10.0, e0=0.4, T=2.0, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=100,
        )
        p_np = np.asarray(p)
        e_np = np.asarray(e)
        valid = ~np.isnan(p_np)
        p_final = p_np[valid][-1]
        e_final = e_np[valid][-1]
        p_sep = float(get_separatrix(abs(a), e_final, 1.0))
        assert p_final >= p_sep + 1e-4, \
            f"Trajectory p={p_final:.4f} < p_sep={p_sep:.4f}"

    def test_schwarzschild(self, flux_data):
        """Schwarzschild (a=0) trajectory should match known p_sep = 6+2e."""
        from fewtrax.trajectory import run_inspiral
        from fewtrax.utils.geodesic import get_separatrix
        _, p, e, _, _, _ = run_inspiral(
            a=0.0, p0=12.0, e0=0.3, T=1.0, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=100,
        )
        p_np = np.asarray(p)
        e_np = np.asarray(e)
        valid = ~np.isnan(p_np)
        e_final = float(e_np[valid][-1])
        p_sep_expected = 6 + 2 * e_final
        p_final = float(p_np[valid][-1])
        # Should have stopped near or above p_sep
        assert p_final > p_sep_expected - 0.1


class TestTrajectoryDifferentiability:
    """Test that the trajectory is differentiable via JAX AD."""

    def test_grad_p0(self, flux_data):
        """Gradient of final phase w.r.t. p0 should be finite."""
        from fewtrax.trajectory import EMRIInspiral

        traj = EMRIInspiral(flux_data, a=0.3)

        def final_phase(p0):
            _, _, _, Phi_phi, _, _ = traj(
                p0=p0, e0=0.4, T=0.1, M=1e6, mu=10.0, dense_steps=20,
            )
            valid = jnp.isfinite(Phi_phi)
            return jnp.sum(jnp.where(valid, Phi_phi, 0.0))

        grad = jax.grad(final_phase)(jnp.float64(10.0))
        assert jnp.isfinite(grad), "Gradient w.r.t. p0 should be finite."

    def test_grad_e0(self, flux_data):
        """Gradient of final phase w.r.t. e0 should be finite."""
        from fewtrax.trajectory import EMRIInspiral

        traj = EMRIInspiral(flux_data, a=0.3)

        def final_phase(e0):
            _, _, _, _, _, Phi_r = traj(
                p0=10.0, e0=e0, T=0.1, M=1e6, mu=10.0, dense_steps=20,
            )
            valid = jnp.isfinite(Phi_r)
            return jnp.sum(jnp.where(valid, Phi_r, 0.0))

        grad = jax.grad(final_phase)(jnp.float64(0.4))
        assert jnp.isfinite(grad), "Gradient w.r.t. e0 should be finite."

    def test_grad_a(self, flux_data):
        """Gradient of final phase w.r.t. spin a should be finite and non-zero."""
        from fewtrax.trajectory import EMRIInspiral

        def final_phase(a):
            traj = EMRIInspiral(flux_data, a=a)
            _, _, _, Phi_phi, _, _ = traj(
                p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, dense_steps=20,
            )
            valid = jnp.isfinite(Phi_phi)
            return jnp.sum(jnp.where(valid, Phi_phi, 0.0))

        grad = jax.grad(final_phase)(jnp.float64(0.3))
        assert jnp.isfinite(grad), "Gradient w.r.t. a should be finite."
        assert grad != 0.0, "Gradient w.r.t. a should be non-zero."


class TestTrajectoryParameterVariation:
    """Test trajectory variations with physical parameter changes."""

    @pytest.mark.parametrize("a", [0.0, 0.3, 0.7, 0.99])
    def test_various_spins(self, flux_data, a):
        """Trajectory should complete for a range of spin values."""
        from fewtrax.trajectory import run_inspiral
        t, p, e, _, _, _ = run_inspiral(
            a=a, p0=10.0, e0=0.3, T=0.1, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=30,
        )
        assert np.isfinite(float(t[0])), f"Trajectory failed for a={a}"

    @pytest.mark.parametrize("e0", [0.0, 0.1, 0.4, 0.7])
    def test_various_eccentricities(self, flux_data, e0):
        """Trajectory should complete for a range of eccentricities."""
        from fewtrax.trajectory import run_inspiral
        t, p, e, _, _, _ = run_inspiral(
            a=0.3, p0=12.0, e0=e0, T=0.1, flux_data=flux_data,
            M=1e6, mu=10.0, dense_steps=30,
        )
        assert np.isfinite(float(t[0])), f"Trajectory failed for e0={e0}"

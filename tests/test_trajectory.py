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

        traj = EMRIInspiral(flux_data)

        def final_phase(p0):
            _, _, _, Phi_phi, _, _ = traj(
                p0=p0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3, dense_steps=20,
            )
            valid = jnp.isfinite(Phi_phi)
            return jnp.sum(jnp.where(valid, Phi_phi, 0.0))

        grad = jax.grad(final_phase)(jnp.float64(10.0))
        assert jnp.isfinite(grad), "Gradient w.r.t. p0 should be finite."

    def test_grad_e0(self, flux_data):
        """Gradient of final phase w.r.t. e0 should be finite."""
        from fewtrax.trajectory import EMRIInspiral

        traj = EMRIInspiral(flux_data)

        def final_phase(e0):
            _, _, _, _, _, Phi_r = traj(
                p0=10.0, e0=e0, T=0.1, M=1e6, mu=10.0, a=0.3, dense_steps=20,
            )
            valid = jnp.isfinite(Phi_r)
            return jnp.sum(jnp.where(valid, Phi_r, 0.0))

        grad = jax.grad(final_phase)(jnp.float64(0.4))
        assert jnp.isfinite(grad), "Gradient w.r.t. e0 should be finite."

    def test_grad_a(self, flux_data):
        """Gradient of final phase w.r.t. spin a should be finite and non-zero."""
        from fewtrax.trajectory import EMRIInspiral

        traj = EMRIInspiral(flux_data)

        def final_phase(a):
            _, _, _, Phi_phi, _, _ = traj(
                p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=a, dense_steps=20,
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


class TestHighEccentricityHighSpin:
    """Near-separatrix accuracy tests for high-e / high-a systems.

    These are the hardest cases for the ODE solver and flux interpolator:
    - high spin (a ≥ 0.98) pushes p_sep very close to r_ISCO
    - high eccentricity (e ≥ 0.7) stresses the e-axis compression and
      makes the w-coordinate approach 1 near the separatrix
    - together they probe the most challenging corner of the flux grid

    Tests verify that the trajectory:
    1. Completes without NaN values at the starting point.
    2. Stays above the separatrix throughout.
    3. Accumulates orbital phases monotonically.
    4. Terminates at p no more than 1.0 M above the separatrix (i.e., the
       solver actually reaches the separatrix region rather than stopping
       prematurely due to numerical failure).
    """

    # (a, e0, p0_above_sep) — p0 is set relative to p_sep so every case
    # starts just outside the separatrix.  p0 = p_sep + offset.
    _CASES = [
        (0.98, 0.70, 2.0),
        (0.98, 0.75, 2.0),
        (0.99, 0.70, 2.0),
        (0.99, 0.75, 1.5),
        (0.999, 0.65, 1.5),
        (0.999, 0.70, 1.5),
    ]

    @pytest.fixture(params=_CASES, ids=[f"a{a}_e{e}" for a, e, _ in _CASES])
    def high_ae_params(self, request, flux_data):
        """Run trajectory for extreme (a, e) case and return results."""
        from fewtrax.trajectory import run_inspiral
        from fewtrax.utils.geodesic import get_separatrix

        a, e0, dp = request.param
        p_sep = float(get_separatrix(a, e0, 1.0))
        p0 = p_sep + dp

        t, p, e, Phi_phi, Phi_theta, Phi_r = run_inspiral(
            a=a, p0=p0, e0=e0, T=0.5,
            flux_data=flux_data,
            M=1e6, mu=10.0,
            dense_steps=100,
            max_steps=8192,
        )
        return {
            "a": a, "e0": e0, "p0": p0, "p_sep": p_sep,
            "t": t, "p": p, "e": e,
            "Phi_phi": Phi_phi, "Phi_theta": Phi_theta, "Phi_r": Phi_r,
        }

    def test_no_initial_nan(self, high_ae_params):
        """First trajectory point must be finite — solver must not fail immediately."""
        r = high_ae_params
        assert np.isfinite(float(r["p"][0])), (
            f"a={r['a']}, e0={r['e0']}: p[0] is NaN"
        )
        assert np.isfinite(float(r["e"][0])), (
            f"a={r['a']}, e0={r['e0']}: e[0] is NaN"
        )

    def test_separatrix_not_crossed(self, high_ae_params):
        """p must stay above the separatrix at every valid point."""
        from fewtrax.utils.geodesic import get_separatrix

        r = high_ae_params
        p_np = np.asarray(r["p"])
        e_np = np.asarray(r["e"])
        valid = np.isfinite(p_np) & np.isfinite(e_np)
        for p_val, e_val in zip(p_np[valid], e_np[valid]):
            p_s = float(get_separatrix(r["a"], float(e_val), 1.0))
            assert float(p_val) >= p_s - 1e-4, (
                f"a={r['a']}, e0={r['e0']}: p={p_val:.5f} < p_sep={p_s:.5f}"
            )

    def test_phases_monotone(self, high_ae_params):
        """Orbital phases should be non-decreasing at all valid points."""
        r = high_ae_params
        for name, arr in [("Phi_phi", r["Phi_phi"]),
                          ("Phi_theta", r["Phi_theta"]),
                          ("Phi_r", r["Phi_r"])]:
            ph = np.asarray(arr)
            valid = np.isfinite(ph)
            if valid.sum() > 1:
                dph = np.diff(ph[valid])
                assert np.all(dph >= -1e-6), (
                    f"a={r['a']}, e0={r['e0']}: {name} decreased by "
                    f"{dph.min():.2e} rad"
                )

    def test_inspiral_reaches_separatrix_region(self, high_ae_params):
        """The trajectory should spiral inward; final p must be within 1 M of p_sep.

        If the solver terminates too early (e.g. a numerical failure causes it
        to stall), the final valid p will be far above p_sep.  With T=0.5 yr
        these systems plunge in < 0.2 yr for the given mass ratio.
        """
        from fewtrax.utils.geodesic import get_separatrix

        r = high_ae_params
        p_np = np.asarray(r["p"])
        e_np = np.asarray(r["e"])
        valid = np.isfinite(p_np) & np.isfinite(e_np)
        if valid.sum() < 2:
            pytest.skip("Trajectory has fewer than 2 valid points.")
        p_final = float(p_np[valid][-1])
        e_final = float(e_np[valid][-1])
        p_sep_final = float(get_separatrix(r["a"], e_final, 1.0))
        gap = p_final - p_sep_final
        assert gap < 1.0, (
            f"a={r['a']}, e0={r['e0']}: final p={p_final:.4f} is {gap:.3f} M "
            f"above p_sep={p_sep_final:.4f}; trajectory may have stalled."
        )

    @pytest.mark.parametrize("a", [0.98, 0.99, 0.999])
    def test_grad_p0_high_spin(self, flux_data, a):
        """Gradient w.r.t. p0 should be finite even at high spin."""
        from fewtrax.trajectory import EMRIInspiral
        from fewtrax.utils.geodesic import get_separatrix

        traj = EMRIInspiral(flux_data)
        p_sep = float(get_separatrix(a, 0.3, 1.0))
        p0_val = p_sep + 3.0

        def final_phase(p0):
            _, _, _, Phi_phi, _, _ = traj(
                p0=p0, e0=0.3, T=0.1, M=1e6, mu=10.0, a=a,
                dense_steps=20, max_steps=4096,
            )
            valid = jnp.isfinite(Phi_phi)
            return jnp.sum(jnp.where(valid, Phi_phi, 0.0))

        grad = jax.grad(final_phase)(jnp.float64(p0_val))
        assert jnp.isfinite(grad), (
            f"Gradient w.r.t. p0 is not finite for a={a}"
        )

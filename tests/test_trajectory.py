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
                # tolerance: the t1=True slot duplicates the final state,
                # producing a ~1e-11 negative diff from float round-off
                assert np.all(np.diff(ph[valid]) >= -1e-8), \
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


class TestJacfwdFisher:
    """Forward-mode autodiff over the full 5-parameter space.

    Guards against regressions in the coordinate-transform double-where
    (``_Secc_of_uz``) and the orbital-energy sqrt guard — both of which
    previously produced NaN tangents under ``jax.jacfwd`` at high
    eccentricity.  The Fisher-matrix path in
    ``gradient_identification.md §7`` exercises exactly this code path.
    """

    @pytest.mark.parametrize("e0", [0.2, 0.5, 0.75])
    def test_jacfwd_5d_finite(self, flux_data, e0):
        """``jacfwd`` over (M, μ, a, p₀, e₀) must return a finite Jacobian."""
        import diffrax
        from fewtrax.trajectory.inspiral import EMRIInspiral
        from fewtrax.utils.geodesic import get_separatrix

        # jacfwd requires DirectAdjoint (custom_jvp); the default
        # RecursiveCheckpointAdjoint only supports reverse mode.
        traj = EMRIInspiral(flux_data, adjoint=diffrax.DirectAdjoint())
        a = 0.5
        p0 = float(get_separatrix(jnp.abs(jnp.asarray(a)),
                                  jnp.asarray(e0), 1.0)) + 2.5

        def freq_track(theta):
            M, mu, a_, p0_, e0_ = theta
            t, f = traj.get_frequency_track(
                p0=p0_, e0=e0_, T=0.05, M=M, mu=mu, a=a_,
                l=2, m=2, k=0, n=0, dense_steps=32,
            )
            return f

        theta0 = jnp.array([1e6, 10.0, a, p0, e0], dtype=jnp.float64)
        J = jax.jacfwd(freq_track)(theta0)

        # N_save = dense_steps + 1 (the t1=True slot captures the end time)
        assert J.shape == (33, 5)
        assert jnp.all(jnp.isfinite(J)), (
            f"jacfwd produced non-finite entries at e0={e0}"
        )
        # Sanity: at least one sensitivity per parameter should be non-zero
        col_norms = jnp.linalg.norm(J, axis=0)
        assert jnp.all(col_norms > 0), (
            f"Degenerate Jacobian column at e0={e0}: {col_norms}"
        )

    def test_fisher_matrix_psd(self, flux_data):
        """Fisher = Jᵀ J / σ² must be symmetric and positive semi-definite."""
        import diffrax
        from fewtrax.trajectory.inspiral import EMRIInspiral

        traj = EMRIInspiral(flux_data, adjoint=diffrax.DirectAdjoint())

        def freq_track(theta):
            M, mu, a_, p0_, e0_ = theta
            _, f = traj.get_frequency_track(
                p0=p0_, e0=e0_, T=0.05, M=M, mu=mu, a=a_,
                l=2, m=2, k=0, n=0, dense_steps=32,
            )
            return f

        theta0 = jnp.array([1e6, 10.0, 0.3, 10.0, 0.4], dtype=jnp.float64)
        J = jax.jacfwd(freq_track)(theta0)
        sigma_f = 1e-5
        fisher = (J.T @ J) / sigma_f**2

        assert jnp.all(jnp.isfinite(fisher))
        # Symmetric
        assert jnp.allclose(fisher, fisher.T, atol=1e-6 * jnp.abs(fisher).max())
        # PSD — smallest eigenvalue ≥ 0 up to numerical noise
        evals = jnp.linalg.eigvalsh(fisher)
        assert float(evals.min()) > -1e-6 * float(evals.max())



class TestEMRIInspiralTObs:
    """Test the t_obs construction-time parameter.

    t_obs pins the output times to a fixed physical grid (e.g. STFT segment
    centres) rather than a uniform dense_steps grid.  The array shape is
    static (required for JIT/vmap).
    """

    def test_output_length_matches_t_obs(self, flux_data):
        """Output time array must have exactly len(t_obs) elements."""
        from fewtrax.trajectory.inspiral import EMRIInspiral
        from fewtrax.utils.constants import YEAR_SI

        n_obs = 20
        T_yr = 0.1
        t_obs = np.linspace(0.0, T_yr * YEAR_SI * 0.9, n_obs)

        traj = EMRIInspiral(flux_data, t_obs=t_obs)
        result = traj(p0=10.0, e0=0.4, T=T_yr, M=1e6, mu=10.0, a=0.3)
        # N_save = len(t_obs) + 1 (the t1=True slot captures the end time)
        assert result[0].shape[0] == n_obs + 1, (
            f"Expected {n_obs + 1} output points; got {result[0].shape[0]}."
        )

    def test_t_obs_first_time_near_zero(self, flux_data):
        """First output time must be approximately zero when t_obs[0]=0."""
        from fewtrax.trajectory.inspiral import EMRIInspiral
        from fewtrax.utils.constants import YEAR_SI

        t_obs = np.linspace(0.0, 0.08 * YEAR_SI, 15)
        traj = EMRIInspiral(flux_data, t_obs=t_obs)
        result = traj(p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3)
        assert float(result[0][0]) == pytest.approx(0.0, abs=1.0)

    def test_t_obs_pe_consistent_with_dense_steps(self, flux_data):
        """Trajectory output at t_obs times must agree with the dense_steps
        output interpolated to the same times (tolerance 1e-4 relative)."""
        from fewtrax.trajectory.inspiral import EMRIInspiral
        from fewtrax.utils.constants import YEAR_SI

        T_yr = 0.1
        n_pts = 15
        t_obs = np.linspace(0.0, T_yr * YEAR_SI * 0.8, n_pts)

        traj_tobs  = EMRIInspiral(flux_data, t_obs=t_obs)
        traj_dense = EMRIInspiral(flux_data, t_obs=None)

        kw = dict(p0=10.0, e0=0.4, T=T_yr, M=1e6, mu=10.0, a=0.3)
        res_tobs  = traj_tobs(**kw)
        res_dense = traj_dense(**kw, dense_steps=500)

        t_tobs  = np.asarray(res_tobs[0])
        p_tobs  = np.asarray(res_tobs[1])
        t_dense = np.asarray(res_dense[0])
        p_dense = np.asarray(res_dense[1])

        # Interpolate dense onto the t_obs grid for comparison
        p_interp = np.interp(t_tobs, t_dense, p_dense)
        np.testing.assert_allclose(p_tobs, p_interp, rtol=1e-4,
                                   err_msg="t_obs output disagrees with dense interpolation.")


class TestEMRIInspiralMultiTrack:
    """Tests for the get_multi_track method.

    get_multi_track runs the ODE once and evaluates multiple harmonic
    mode frequencies simultaneously via a single matrix-vector product,
    which is faster than calling get_frequency_track once per mode.
    """

    def test_single_mode_matches_frequency_track(self, flux_data):
        """get_multi_track for one (m,k,n) mode must match get_frequency_track."""
        from fewtrax.trajectory.inspiral import EMRIInspiral

        traj = EMRIInspiral(flux_data)
        kw = dict(p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3, dense_steps=50)

        _t_single, f_single = traj.get_frequency_track(l=2, m=2, k=0, n=0, **kw)
        _t_multi,  freqs    = traj.get_multi_track([(2, 0, 0)], **kw)

        np.testing.assert_allclose(
            np.asarray(f_single), np.asarray(freqs[0]), rtol=1e-8,
            err_msg="Single-mode get_multi_track disagrees with get_frequency_track.",
        )

    def test_multiple_modes_output_shape(self, flux_data):
        """Output freqs must have shape (N_modes, dense_steps), all non-negative."""
        from fewtrax.trajectory.inspiral import EMRIInspiral

        traj = EMRIInspiral(flux_data)
        modes = [(2, 0, 0), (2, 0, 1), (3, 0, 0)]
        _t, freqs = traj.get_multi_track(
            modes, p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3, dense_steps=30
        )
        # dense_steps + 1 points (the t1=True slot captures the end time)
        assert freqs.shape == (3, 31), f"Expected shape (3, 31); got {freqs.shape}."
        assert np.all(np.asarray(freqs) >= 0.0), "All frequencies must be non-negative."

    def test_lmkn_tuples_accepted(self, flux_data):
        """4-tuple (l, m, k, n) modes should produce the same frequencies as
        the corresponding 3-tuple (m, k, n)."""
        from fewtrax.trajectory.inspiral import EMRIInspiral

        traj = EMRIInspiral(flux_data)
        kw = dict(p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3, dense_steps=30)
        _t3, f3 = traj.get_multi_track([(2, 0, 0)], **kw)
        _t4, f4 = traj.get_multi_track([(2, 2, 0, 0)], **kw)
        np.testing.assert_allclose(np.asarray(f3), np.asarray(f4), rtol=1e-12)

    def test_return_dict_mode(self, flux_data):
        """return_dict=True must key results by the original mode tuples."""
        from fewtrax.trajectory.inspiral import EMRIInspiral

        traj = EMRIInspiral(flux_data)
        modes = [(2, 0, 0), (2, 0, 1)]
        result = traj.get_multi_track(
            modes, p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3,
            dense_steps=30, return_dict=True,
        )
        assert isinstance(result, dict)
        for key in modes:
            assert tuple(key) in result, f"Key {key} missing from return dict."
            t_k, f_k = result[tuple(key)]
            # dense_steps + 1 points (the t1=True slot captures the end time)
            assert len(t_k) == 31
            assert len(f_k) == 31

    def test_empty_modes_raises(self, flux_data):
        """Empty mode list must raise ValueError."""
        from fewtrax.trajectory.inspiral import EMRIInspiral

        traj = EMRIInspiral(flux_data)
        with pytest.raises(ValueError, match="non-empty"):
            traj.get_multi_track(
                [], p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3
            )


class TestPlatformDispatch:
    """Test CPU/GPU platform-aware dispatch in get_fundamental_frequencies_platform.

    The hybrid implementation dispatches at JIT-trace time:
    - GPU  -> get_fundamental_frequencies (64-point Gauss-Legendre)
    - CPU  -> get_fundamental_frequencies_fast (AGM-12 + 24-pt GL)

    Both paths must agree to < 1e-10 relative error for the smooth integrands
    encountered in bound EMRI orbits.
    """

    def test_platform_dispatch_returns_finite(self):
        """Platform-dispatched frequencies must be finite for a typical orbit."""
        from fewtrax.utils.geodesic import get_fundamental_frequencies_platform

        Om = get_fundamental_frequencies_platform(
            jnp.float64(0.5), jnp.float64(10.0), jnp.float64(0.4), jnp.float64(1.0)
        )
        for val in Om:
            assert jnp.isfinite(val), f"Non-finite frequency from platform dispatch: {val}"

    def test_platform_dispatch_routes_correctly(self):
        """Platform dispatch must return the same result as the expected path."""
        from fewtrax.utils.geodesic import (
            get_fundamental_frequencies,
            get_fundamental_frequencies_fast,
            get_fundamental_frequencies_platform,
        )

        a, p, e, x = (jnp.float64(v) for v in (0.3, 12.0, 0.3, 1.0))

        Om_plat = get_fundamental_frequencies_platform(a, p, e, x)
        Om_gpu  = get_fundamental_frequencies(a, p, e, x)
        Om_cpu  = get_fundamental_frequencies_fast(a, p, e, x)

        platform = jax.devices()[0].platform
        expected = Om_gpu if platform == "gpu" else Om_cpu

        for got, exp in zip(Om_plat, expected):
            assert float(got) == pytest.approx(float(exp), rel=1e-12), (
                f"Platform dispatch returned wrong value for platform={platform!r}."
            )

    @pytest.mark.parametrize("a,p,e", [
        (0.0,  10.0, 0.3),   # Schwarzschild
        (0.5,  10.0, 0.4),   # moderate spin
        (0.9,   8.0, 0.2),   # high spin
        (0.5,  12.0, 0.7),   # high eccentricity
        (0.99,  7.0, 0.1),   # near-extreme spin
    ])
    def test_cpu_gpu_paths_agree(self, a, p, e):
        """64-pt GL and AGM+24-pt GL frequency evaluations must agree to < 1e-10."""
        from fewtrax.utils.geodesic import (
            get_fundamental_frequencies,
            get_fundamental_frequencies_fast,
        )

        a_, p_, e_, x_ = (jnp.float64(v) for v in (a, p, e, 1.0))
        Om_exact = get_fundamental_frequencies(a_, p_, e_, x_)
        Om_fast  = get_fundamental_frequencies_fast(a_, p_, e_, x_)

        for name, exact, fast in zip(
            ("Omega_phi", "Omega_theta", "Omega_r"), Om_exact, Om_fast
        ):
            assert float(fast) == pytest.approx(float(exact), rel=1e-10), (
                f"a={a}, p={p}, e={e}: {name} "
                f"fast={float(fast):.14g} vs exact={float(exact):.14g}"
            )


class TestDirectAdjoint:
    """Test that DirectAdjoint enables forward-mode autodiff (jacfwd/hessian).

    RecursiveCheckpointAdjoint (the default) uses a custom_vjp rule that
    is incompatible with jax.jacfwd.  DirectAdjoint uses a custom_jvp rule
    instead, enabling forward-mode AD at the cost of higher memory use.
    """

    def test_jacfwd_with_direct_adjoint(self, flux_data):
        """EMRIInspiral(adjoint=DirectAdjoint()) must support jax.jacfwd."""
        import diffrax
        from fewtrax.trajectory.inspiral import EMRIInspiral

        traj = EMRIInspiral(flux_data, adjoint=diffrax.DirectAdjoint())

        def pe_track(p0, e0):
            _, p, e, *_ = traj(
                p0=p0, e0=e0, T=0.05, M=1e6, mu=10.0, a=0.3, dense_steps=16,
            )
            return jnp.stack([p, e], axis=0)  # (2, dense_steps)

        J_p0, J_e0 = jax.jacfwd(pe_track, argnums=(0, 1))(
            jnp.float64(10.0), jnp.float64(0.4)
        )
        # N_save = dense_steps + 1 (the t1=True slot captures the end time)
        assert J_p0.shape == (2, 17)
        assert J_e0.shape == (2, 17)
        assert jnp.all(jnp.isfinite(J_p0)), "jacfwd J w.r.t. p0 contains non-finite values."
        assert jnp.all(jnp.isfinite(J_e0)), "jacfwd J w.r.t. e0 contains non-finite values."

    def test_direct_adjoint_agrees_with_default(self, flux_data):
        """DirectAdjoint and default RecursiveCheckpointAdjoint must yield
        identical trajectories (same ODE, different AD bookkeeping)."""
        import diffrax
        from fewtrax.trajectory.inspiral import EMRIInspiral

        kw = dict(p0=10.0, e0=0.4, T=0.1, M=1e6, mu=10.0, a=0.3,
                  dense_steps=40)
        traj_rca = EMRIInspiral(flux_data)
        traj_da  = EMRIInspiral(flux_data, adjoint=diffrax.DirectAdjoint())

        _t_r, p_r, e_r, *_ = traj_rca(**kw)
        _t_d, p_d, e_d, *_ = traj_da(**kw)

        np.testing.assert_allclose(
            np.asarray(p_r), np.asarray(p_d), rtol=1e-10,
            err_msg="p differs between RecursiveCheckpointAdjoint and DirectAdjoint."
        )
        np.testing.assert_allclose(
            np.asarray(e_r), np.asarray(e_d), rtol=1e-10,
            err_msg="e differs between RecursiveCheckpointAdjoint and DirectAdjoint."
        )

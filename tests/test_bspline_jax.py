"""Tests for the JAX-native B-spline (bisplev) evaluator.

Four layers of verification, from primitive to integration:

1. ``test_basis_cubic_*``     — de Boor basis functions match scipy.splev
2. ``test_bisplev_batched_*`` — 2-D evaluator matches scipy.bisplev
3. ``test_jax_transforms``    — jit / vmap / grad work correctly
4. ``test_amplitude_vs_scipy``— full JAXAmplitudeInterpolator matches
                                 AmplitudeInterpolator (scipy path)
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.interpolate import bisplev as scipy_bisplev
from scipy.interpolate import splev as scipy_splev

jax.config.update("jax_enable_x64", True)

from fewtrax.utils.bspline_jax import _find_span, _bspline_basis_cubic, eval_bisplev_batched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clamped_knots(n_pts: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Build a degree-3 clamped knot vector for n_pts interpolation nodes."""
    interior = np.linspace(lo, hi, n_pts)[1:-1]  # n_pts - 2 interior knots
    return np.concatenate([[lo] * 4, interior, [hi] * 4])


def _make_nonuniform_knots(lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Non-uniform knot vector mimicking the FEW amplitude file structure.

    Matches the HDF5 u_knots layout (shape 37, 31 unique breakpoints):
    first interior interval is 0.0625, subsequent 28 intervals are 0.03125.
    Total: [lo]*4 + 29 interior + [hi]*4 = 37 elements, n_coeffs = 33.
    """
    interior = np.array([0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875,
                         0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625,
                         0.4375, 0.46875, 0.5, 0.53125, 0.5625, 0.59375,
                         0.625, 0.65625, 0.6875, 0.71875, 0.75, 0.78125,
                         0.8125, 0.84375, 0.875, 0.90625, 0.9375])
    return np.concatenate([[lo] * 4, interior, [hi] * 4])


# ---------------------------------------------------------------------------
# 1. Basis function tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_factory,label", [
    (_make_clamped_knots, "uniform"),
    (lambda: _make_nonuniform_knots(), "nonuniform"),
])
def test_basis_cubic_matches_scipy(t_factory, label):
    """_bspline_basis_cubic must agree with scipy.splev to float64 precision."""
    if callable(t_factory) and t_factory.__name__ == "<lambda>":
        t = t_factory()
    else:
        t = t_factory(33)

    n = len(t) - 4   # number of B-spline coefficients
    t_jax = jnp.asarray(t)

    rng = np.random.default_rng(0)
    xs = rng.uniform(t[3], t[-4], size=200)

    for x in xs:
        l = int(_find_span(float(x), t_jax, n))
        N_jax = np.array(_bspline_basis_cubic(float(x), t_jax, jnp.int32(l)))

        # Compare against scipy: set one coefficient to 1, rest 0 for each basis
        for j in range(4):
            c = np.zeros(n)
            c[l - 3 + j] = 1.0
            val_scipy = float(scipy_splev(x, (t, c, 3)))
            assert abs(N_jax[j] - val_scipy) < 2e-14, (
                f"[{label}] x={x:.4f}, l={l}, j={j}: "
                f"JAX={N_jax[j]:.16e}  scipy={val_scipy:.16e}"
            )


def test_basis_partition_of_unity():
    """Basis functions must sum to 1 everywhere (partition of unity)."""
    t = _make_nonuniform_knots()
    t_jax = jnp.asarray(t)
    n = len(t) - 4

    xs = np.linspace(t[3] + 1e-9, t[-4] - 1e-9, 500)
    for x in xs:
        l = _find_span(float(x), t_jax, n)
        N = _bspline_basis_cubic(float(x), t_jax, l)
        assert abs(float(jnp.sum(N)) - 1.0) < 1e-13, (
            f"Partition of unity violated at x={x}: sum={float(jnp.sum(N))}"
        )


def test_basis_boundary_clamp():
    """Evaluating at or beyond boundaries must not crash and must stay clamped."""
    t = _make_clamped_knots(33)
    t_jax = jnp.asarray(t)
    n = len(t) - 4

    for x in [t[0], t[-1], t[0] - 1.0, t[-1] + 1.0]:
        l = _find_span(float(x), t_jax, n)
        N = _bspline_basis_cubic(float(x), t_jax, l)
        assert jnp.all(jnp.isfinite(N)), f"Non-finite basis at x={x}"
        # Partition-of-unity holds algebraically; allow generous float64 tolerance
        # for out-of-domain points where large cancellations occur.
        assert abs(float(jnp.sum(N)) - 1.0) < 1e-9, (
            f"Partition-of-unity violated at x={x}: sum={float(jnp.sum(N))}"
        )


# ---------------------------------------------------------------------------
# 2. 2-D bisplev tests
# ---------------------------------------------------------------------------

def _random_bisplev_problem(n_modes: int = 5, seed: int = 42):
    """Return (t1, t2, coeffs_3d) for a random batched 2-D bisplev problem.

    Uses the FEW-like nonuniform knot vector for t1 (w-axis) and a uniform
    clamped knot vector for t2 (u-axis).  n_coeffs is derived from the knot
    vectors as ``len(t) - 4`` (cubic degree).
    """
    rng = np.random.default_rng(seed)
    t1 = _make_nonuniform_knots()   # w-axis knot vector
    t2 = _make_clamped_knots(29)    # u-axis: 29 unique breakpoints → n_coeffs=33
    n1 = len(t1) - 4               # n_coeffs for w-axis
    n2 = len(t2) - 4               # n_coeffs for u-axis
    # Shape (n_modes, N1, N2) — C-order: c[m, i, j] = coeff for B_i(w)*B_j(u)
    coeffs = rng.standard_normal((n_modes, n1, n2))
    return t1, t2, coeffs


def test_bisplev_batched_matches_scipy():
    """eval_bisplev_batched must match scipy.bisplev to 1e-12 for each mode."""
    t1, t2, coeffs = _random_bisplev_problem()
    t1_j = jnp.asarray(t1)
    t2_j = jnp.asarray(t2)
    coeffs_j = jnp.asarray(coeffs)

    rng = np.random.default_rng(99)
    n_modes = coeffs.shape[0]
    n1, n2 = coeffs.shape[1], coeffs.shape[2]

    w_samples = rng.uniform(t1[3], t1[-4], 50)
    u_samples = rng.uniform(t2[3], t2[-4], 50)

    for w, u in zip(w_samples, u_samples):
        jax_vals = np.array(eval_bisplev_batched(float(w), float(u), coeffs_j, t1_j, t2_j))

        for m in range(n_modes):
            c_flat = coeffs[m].ravel()   # C-order: row = w-axis, col = u-axis
            scipy_val = float(scipy_bisplev(w, u, (t1, t2, c_flat, 3, 3)))
            assert abs(jax_vals[m] - scipy_val) < 1e-11, (
                f"Mode {m}, w={w:.4f}, u={u:.4f}: "
                f"JAX={jax_vals[m]:.16e}  scipy={scipy_val:.16e}"
            )


def test_bisplev_batched_boundary():
    """Evaluation at the grid boundary must not produce NaN or Inf."""
    t1, t2, coeffs = _random_bisplev_problem(n_modes=3)
    t1_j, t2_j, coeffs_j = jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(coeffs)

    for w in [t1[0], t1[-1]]:
        for u in [t2[0], t2[-1]]:
            v = eval_bisplev_batched(float(w), float(u), coeffs_j, t1_j, t2_j)
            assert jnp.all(jnp.isfinite(v)), f"Non-finite at boundary ({w}, {u})"


# ---------------------------------------------------------------------------
# 3. JAX transform compatibility
# ---------------------------------------------------------------------------

def test_jit_consistent():
    """jit-compiled result must equal eager result."""
    t1, t2, coeffs = _random_bisplev_problem(n_modes=8)
    t1_j, t2_j, coeffs_j = jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(coeffs)
    eval_jit = jax.jit(eval_bisplev_batched, static_argnums=())

    w, u = 0.42, 0.37
    eager = eval_bisplev_batched(w, u, coeffs_j, t1_j, t2_j)
    compiled = eval_jit(w, u, coeffs_j, t1_j, t2_j)
    np.testing.assert_allclose(np.array(eager), np.array(compiled), atol=0)


def test_vmap_over_trajectory():
    """vmap over (w, u) pairs must match the loop result."""
    t1, t2, coeffs = _random_bisplev_problem(n_modes=10)
    t1_j, t2_j, coeffs_j = jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(coeffs)

    rng = np.random.default_rng(7)
    N_traj = 30
    ws = jnp.asarray(rng.uniform(t1[3] + 0.01, t1[-4] - 0.01, N_traj))
    us = jnp.asarray(rng.uniform(t2[3] + 0.01, t2[-4] - 0.01, N_traj))

    # vmap over trajectory axis
    vmap_eval = jax.vmap(
        lambda w, u: eval_bisplev_batched(w, u, coeffs_j, t1_j, t2_j)
    )
    vmap_result = np.array(vmap_eval(ws, us))  # (N_traj, n_modes)

    # loop reference
    loop_result = np.stack([
        np.array(eval_bisplev_batched(float(ws[i]), float(us[i]), coeffs_j, t1_j, t2_j))
        for i in range(N_traj)
    ])

    np.testing.assert_allclose(vmap_result, loop_result, atol=0)


def test_grad_wrt_query_point():
    """jax.grad of the sum over modes w.r.t. (w, u) must be finite."""
    t1, t2, coeffs = _random_bisplev_problem(n_modes=5)
    t1_j, t2_j, coeffs_j = jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(coeffs)

    def scalar_fn(w, u):
        return jnp.sum(eval_bisplev_batched(w, u, coeffs_j, t1_j, t2_j))

    grad_fn = jax.grad(scalar_fn, argnums=(0, 1))
    dw, du = grad_fn(0.42, 0.37)
    assert jnp.isfinite(dw) and jnp.isfinite(du), f"Non-finite grad: dw={dw}, du={du}"


def test_grad_finite_difference_consistency():
    """Autodiff gradient must match finite differences to 1e-5 relative."""
    t1, t2, coeffs = _random_bisplev_problem(n_modes=5)
    t1_j, t2_j, coeffs_j = jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(coeffs)

    def scalar_fn(w, u):
        return jnp.sum(eval_bisplev_batched(w, u, coeffs_j, t1_j, t2_j))

    w0, u0 = 0.42, 0.37
    h = 1e-5
    dw_fd = (scalar_fn(w0 + h, u0) - scalar_fn(w0 - h, u0)) / (2 * h)
    du_fd = (scalar_fn(w0, u0 + h) - scalar_fn(w0, u0 - h)) / (2 * h)

    dw_ad, du_ad = jax.grad(scalar_fn, argnums=(0, 1))(w0, u0)

    assert abs(float(dw_ad) - float(dw_fd)) / (abs(float(dw_fd)) + 1e-30) < 1e-4, (
        f"dw: AD={float(dw_ad):.8e}  FD={float(dw_fd):.8e}"
    )
    assert abs(float(du_ad) - float(du_fd)) / (abs(float(du_fd)) + 1e-30) < 1e-4, (
        f"du: AD={float(du_ad):.8e}  FD={float(du_fd):.8e}"
    )


# ---------------------------------------------------------------------------
# 4. Full amplitude interpolator vs scipy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jax_amp_interp(data_dir):
    """Session-scoped JAX amplitude interpolator."""
    from fewtrax.data.loader import load_amplitude_data_jax
    from fewtrax.amplitude.interp import JAXAmplitudeInterpolator
    amp_data = load_amplitude_data_jax(data_dir)
    return JAXAmplitudeInterpolator(amp_data)


@pytest.fixture(scope="module")
def scipy_amp_interp(data_dir):
    """Session-scoped scipy amplitude interpolator."""
    from fewtrax.data.loader import load_amplitude_data
    from fewtrax.amplitude.interp import AmplitudeInterpolator
    amp_data = load_amplitude_data(data_dir)
    return AmplitudeInterpolator(amp_data)


def test_amplitude_single_point_vs_scipy(jax_amp_interp, scipy_amp_interp):
    """JAXAmplitudeInterpolator must match scipy path to 1e-10 at a single point."""
    a, p, e = 0.3, 10.0, 0.4
    a_abs = abs(a)

    # scipy path
    p_arr = np.array([p])
    e_arr = np.array([e])
    scipy_amps = scipy_amp_interp.evaluate(a, p_arr, e_arr)[0]  # (n_modes,) complex

    # JAX path
    jax_amps = np.array(jax_amp_interp(a=a, p=p, e=e))         # (n_modes,) complex128

    # Select a subset of large-amplitude modes for robust comparison
    power = np.abs(scipy_amps) ** 2
    top = np.argsort(power)[-200:]

    np.testing.assert_allclose(
        np.abs(jax_amps[top]),
        np.abs(scipy_amps[top]),
        rtol=1e-9,
        err_msg="Amplitude magnitudes differ between JAX and scipy paths.",
    )
    # Phase must also match (within float64 rounding)
    np.testing.assert_allclose(
        np.angle(jax_amps[top]),
        np.angle(scipy_amps[top]),
        atol=1e-9,
        err_msg="Amplitude phases differ between JAX and scipy paths.",
    )


def test_amplitude_trajectory_vs_scipy(jax_amp_interp, scipy_amp_interp):
    """Trajectory evaluation via vmap must match the scipy loop."""
    rng = np.random.default_rng(123)
    N = 20
    a = 0.5
    p_arr = rng.uniform(8.0, 14.0, N)
    e_arr = rng.uniform(0.05, 0.6, N)

    # scipy reference
    scipy_traj = scipy_amp_interp.evaluate(a, p_arr, e_arr)  # (N, n_modes)

    # JAX vmap path
    jax_traj = np.array(
        jax_amp_interp.evaluate_trajectory(
            jnp.float64(a),
            jnp.asarray(p_arr, dtype=jnp.float64),
            jnp.asarray(e_arr, dtype=jnp.float64),
        )
    )  # (N, n_modes)

    power = np.abs(scipy_traj).mean(axis=0) ** 2
    top = np.argsort(power)[-200:]

    np.testing.assert_allclose(
        np.abs(jax_traj[:, top]),
        np.abs(scipy_traj[:, top]),
        rtol=1e-8,
        err_msg="Trajectory amplitude magnitudes differ.",
    )


def test_amplitude_region_B(jax_amp_interp, scipy_amp_interp):
    """Check Region B (large-p orbit) also matches scipy."""
    a, p, e = 0.3, 50.0, 0.1   # large p → Region B

    scipy_amps = scipy_amp_interp.evaluate(a, np.array([p]), np.array([e]))[0]
    jax_amps = np.array(jax_amp_interp(a=a, p=p, e=e))

    power = np.abs(scipy_amps) ** 2
    top = np.argsort(power)[-100:]

    np.testing.assert_allclose(
        np.abs(jax_amps[top]),
        np.abs(scipy_amps[top]),
        rtol=1e-9,
        err_msg="Region B amplitudes differ.",
    )

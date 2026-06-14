'''
File to implement the DOPR 7th order polynomial that we interpolate to gt frequency tracks. 

Note: this comes from the notebook that Federico Fantocolli created and shared. 
'''

import jax
import jax.numpy as jnp
import numpy as np

from scipy.interpolate import CubicSpline as _CubicSpline

# Mode-batched interp + direct mode sum.
# v3: cubic spline interp (amplitudes AND phases) + factorised exp [ACTIVE]

# v3 matches FEW's InterpolatedModeSum accuracy (not-a-knot cubic splines)
# while keeping the exp-factorisation speedup from v2.

from scipy.interpolate import CubicSpline as _CubicSpline

def _fit_spline(t_sparse_np, y_sparse_np):
    """Fit a not-a-knot cubic spline and return scipy's (4, n-1, ...) coefficient array."""
    cs = _CubicSpline(t_sparse_np, y_sparse_np)
    return jnp.asarray(cs.c)   # (4, n_sparse-1, *y_shape[1:])


@jax.jit
def _bracket(t_traj, t_dense):
    """Bracket indices and normalised weights: inds, w = (t-t_i)/(t_{i+1}-t_i)."""
    inds = jnp.clip(
        jnp.searchsorted(t_traj, t_dense, side="right") - 1,
        0, t_traj.size - 2,
    )
    w = (t_dense - t_traj[inds]) / (t_traj[inds + 1] - t_traj[inds])
    return inds, w


@jax.jit
def _eval_cubic(spline_c, inds, dx):
    """Evaluate cubic splines at dense points via Horner's method.

    spline_c : (4, n_sparse-1, n_splines)  scipy cs.c layout:
                 c[0]=cubic, c[1]=quad, c[2]=linear, c[3]=constant
    inds     : (n_dense,) int  -- bracket indices
    dx       : (n_dense,) float -- time offset within bracket (t - t_i)
    returns  : (n_dense, n_splines)
    """
    result = spline_c[0, inds]                              # d (cubic coeff)
    result = spline_c[1, inds] + dx[:, None] * result      # c + dx*d
    result = spline_c[2, inds] + dx[:, None] * result      # b + dx*(c+dx*d)
    result = spline_c[3, inds] + dx[:, None] * result      # a + dx*(...)
    return result                                            # (n_dense, n_splines)

# =============================================================================
# v3 -- cubic spline interp (amplitudes + phases) + factorised exp    [ACTIVE]
# =============================================================================
# Cubic splines (not-a-knot, same as FEW's InterpolatedModeSum) are fitted
# once to the 67 sparse trajectory points -- negligible CPU time.
# Evaluation uses Horner's method: each step gathers from a tiny
# (66, batch) coefficient array before writing the (n_t_dense, batch) result.
# The exp factorisation from v2 is retained.

@jax.jit
def _interp_modesum_batch_cubic(
    inds, dx,            # (n_t_dense,) -- bracket index and time offset
    spline_coeffs_b,     # (4, n_sparse-1, batch) complex128 -- tiny coefficient batch
    ylm_pos_b, coeff_neg_b,
    E_phi,               # (n_uniq_m, n_t_dense) -- pre-factored m-phase exp table
    E_theta,             # (n_uniq_k, n_t_dense) -- pre-factored k-phase exp table
    E_r,                 # (n_uniq_n, n_t_dense) -- pre-factored n-phase exp table
    m_inv_b, k_inv_b, n_inv_b,   # (batch,) int32 -- indices into exp tables
):
    # Cubic Horner evaluation of mode amplitudes -- no linear weight w.
    # scipy c layout: c[0]=cubic, c[1]=quad, c[2]=linear, c[3]=constant
    teuk_dense = spline_coeffs_b[0, inds]
    teuk_dense = spline_coeffs_b[1, inds] + dx[:, None] * teuk_dense
    teuk_dense = spline_coeffs_b[2, inds] + dx[:, None] * teuk_dense
    teuk_dense = spline_coeffs_b[3, inds] + dx[:, None] * teuk_dense  # (n_t, batch)

    # Phase factor by gather + multiply -- no exp call here.
    E  = (E_phi[m_inv_b] * E_theta[k_inv_b] * E_r[n_inv_b]).T  # (n_t, batch)

    T  = teuk_dense * E
    w1 = T @ ylm_pos_b
    w2 = jnp.conj(T) @ coeff_neg_b
    return w1 + w2


@jax.jit
def _eval_bezier_phases_on_dense(t_dense, knot_t, coeff_3d):
    """DOPR853 degree-7 Bezier evaluation -- matches FEW's InterpolatedModeSum kernel.

    `coeff_3d` is FEW's `few_traj.integrator_spline_phase_coeff` with shape
    `(N_intervals, 3, 8)`. Axis-1 ordering is [Phi_phi, Phi_theta, Phi_r],
    axis-2 holds the 8 Bezier coefficients per interval, identical to the
    DOPR853 native dense-output polynomial used inside FEW.

    Formula (from interpolate.cu line 642):
        s  = (t - knot_t[i]) / (knot_t[i+1] - knot_t[i])
        s1 = 1 - s
        P(s) = c0 + s*(c1 + s1*(c2 + s*(c3 + s1*(c4 + s*(c5 + s1*(c6 + s*c7))))))
    """
    inds = jnp.clip(
        jnp.searchsorted(knot_t, t_dense, side="right") - 1,
        0, knot_t.size - 2,
    )
    s  = (t_dense - knot_t[inds]) / (knot_t[inds + 1] - knot_t[inds])
    s1 = 1.0 - s
    # Gather all 8 coeffs for all 3 phases at every dense time -- (n_t, 3, 8)
    c = coeff_3d[inds]
    # Horner-Bezier evaluation along the coefficient axis (inside-out)
    out = c[..., 7]
    out = c[..., 6] + s[:,  None] * out
    out = c[..., 5] + s1[:, None] * out
    out = c[..., 4] + s[:,  None] * out
    out = c[..., 3] + s1[:, None] * out
    out = c[..., 2] + s[:,  None] * out
    out = c[..., 1] + s1[:, None] * out
    out = c[..., 0] + s[:,  None] * out   # (n_t, 3)
    return out[:, 0], out[:, 1], out[:, 2]


def Interp_and_modesum_WFbuilder(
    time, teuk_modes, ylms, ls, ms, ks, ns,
    Phi_phi_tr, Phi_theta_tr, Phi_r_tr,
    t_dense,
    sum_batch_size=100,
    phase_spline_t=None,
    phase_spline_coeff=None,
):
    """Mode-batched fused interp + direct mode sum.

    Amplitudes: not-a-knot cubic spline (matches FEW's amp interpolation).
    Phases:
       - If `phase_spline_t` and `phase_spline_coeff` are provided, use FEW's
         DOPR853 degree-7 Bezier dense output. This matches FEW's
         `InterpolatedModeSum` to machine precision.  Pass:
              phase_spline_t     = few_traj.integrator_spline_t        # (N+1,)
              phase_spline_coeff = few_traj.integrator_spline_phase_coeff  # (N, 3, 8)
         right after calling few_traj(...) with the same (m1, m2, a, p0, e0, x0, T).
       - Otherwise, use a not-a-knot cubic spline on Phi_*_tr -- faster to
         construct but the phase between sparse knots drifts ~1e-3 rad
         relative to DOPR853, which is what causes the empirical "2i factor"
         observed when comparing against FEW's GenerateEMRIWaveform.

    All the speed optimisations from v3 are retained:
       - cubic spline for amplitudes (gather + Horner inside the JIT kernel)
       - exp factorisation: only ~20 complex exp() calls instead of n_m * n_t
       - mode-batched GEMV reductions
    """
    n_m  = int(ls.shape[0])
    t_np = np.array(time)

    # Fit cubic spline to amplitudes once (FEW also uses cubic for amps).
    spline_teuk = _fit_spline(t_np, np.array(teuk_modes))   # (4, n_sparse-1, n_m)

    # Bracket indices and actual time offset for amplitude eval.
    inds, w  = _bracket(time, t_dense)
    h_sparse = time[1:] - time[:-1]
    dx       = w * h_sparse[inds]

    # ------------------------------------------------------------------
    # Phase interpolation: either FEW DOPR853 Bezier (preferred) or cubic.
    # ------------------------------------------------------------------
    if phase_spline_t is not None and phase_spline_coeff is not None:
        knot_t_jax  = jnp.asarray(np.array(phase_spline_t),     dtype=jnp.float64)
        coeff_3d_jax = jnp.asarray(np.array(phase_spline_coeff), dtype=jnp.float64)
        Phi_phi_d, Phi_theta_d, Phi_r_d = _eval_bezier_phases_on_dense(
            t_dense, knot_t_jax, coeff_3d_jax,
        )
    else:
        # cubic spline fallback (slightly less accurate, no FEW state needed)
        phases_np = np.stack([np.array(Phi_phi_tr),
                              np.array(Phi_theta_tr),
                              np.array(Phi_r_tr)], axis=1)
        spline_phases = _fit_spline(t_np, phases_np)
        phases_dense = _eval_cubic(spline_phases, inds, dx)
        Phi_phi_d, Phi_theta_d, Phi_r_d = (
            phases_dense[:, 0], phases_dense[:, 1], phases_dense[:, 2]
        )

    # Ylm coefficient vectors (unchanged).
    ylm_pos   = ylms[:n_m]
    ylm_neg   = ylms[n_m:2 * n_m]
    coeff_neg = ((ms > 0) * ((-1.0) ** ls)).astype(ylm_neg.dtype) * ylm_neg

    # Exp factorisation (one exp per unique quantum number, not per mode).
    unique_ms, m_inv = np.unique(np.array(ms), return_inverse=True)
    unique_ks, k_inv = np.unique(np.array(ks), return_inverse=True)
    unique_ns, n_inv = np.unique(np.array(ns), return_inverse=True)

    E_phi   = jnp.exp(-1j * jnp.asarray(unique_ms)[:, None] * Phi_phi_d[None, :])
    E_theta = jnp.exp(-1j * jnp.asarray(unique_ks)[:, None] * Phi_theta_d[None, :])
    E_r     = jnp.exp(-1j * jnp.asarray(unique_ns)[:, None] * Phi_r_d[None, :])

    m_inv_j = jnp.asarray(m_inv, dtype=jnp.int32)
    k_inv_j = jnp.asarray(k_inv, dtype=jnp.int32)
    n_inv_j = jnp.asarray(n_inv, dtype=jnp.int32)

    out_wf = jnp.zeros(t_dense.shape[0], dtype=jnp.complex128)

    for i in range(0, n_m, sum_batch_size):
        i_end = min(i + sum_batch_size, n_m)
        out_wf = out_wf + _interp_modesum_batch_cubic(
            inds, dx, spline_teuk[:, :, i:i_end],
            ylm_pos[i:i_end], coeff_neg[i:i_end],
            E_phi, E_theta, E_r,
            m_inv_j[i:i_end], k_inv_j[i:i_end], n_inv_j[i:i_end],
        )

    return np.asarray(out_wf)



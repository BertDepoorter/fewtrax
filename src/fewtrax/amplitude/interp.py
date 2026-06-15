"""Teukolsky mode amplitude interpolation from FEW precomputed grids.

:class:`JAXAmplitudeInterpolator` is a fully JAX-native evaluator of the complex
mode amplitudes :math:`A_{\\ell mkn}(a, p, e)`.  It wraps the
:class:`~fewtrax.utils.splines.BatchedTricubicSplineE3` coefficients built by
:func:`~fewtrax.data.loader.load_amplitude_data_jax` and evaluates all modes in
a single fused ``einsum`` — the JAX analogue of FEW's per-point
``scipy.bisplev`` loop — with full :func:`jax.jit` / :func:`jax.grad` /
:func:`jax.vmap` support.  It also carries the FEW-2.0.0-exact mode selector
:meth:`JAXAmplitudeInterpolator.select_modes_few`.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx

from fewtrax.utils.coordinates import (
    ALPHA_AMP, BETA_AMP, DELTAPMAX,
    kerrecceq_forward_map_A, kerrecceq_forward_map_B,
)
from fewtrax.utils.geodesic import get_separatrix_fast
from fewtrax.utils.bspline_jax import eval_bisplev_batched
from fewtrax.data.loader import AmplitudeDataJAX


class JAXAmplitudeInterpolator(eqx.Module):
    r"""Fully JAX-native Teukolsky amplitude evaluator.

    Wraps an :class:`~fewtrax.data.loader.AmplitudeDataJAX` container and
    evaluates the complex mode amplitudes :math:`A_{\ell mkn}(a, p, e)` for all
    modes in a single fused ``einsum``.  All public methods are jit/vmap-able
    and support :func:`jax.grad` / :func:`jax.jacfwd`.

    Parameters
    ----------
    amp_data : AmplitudeDataJAX
        Pre-built JAX amplitude container; construct with
        :func:`~fewtrax.data.loader.load_amplitude_data_jax`.

    Examples
    --------
    ::

        interp = JAXAmplitudeInterpolator(amp_data)
        amps = interp(a=0.9, p=10.0, e=0.3)                   # (n_modes,)
        amps_traj = interp.evaluate_trajectory(0.9, p_arr, e_arr)  # (N, n_modes)
    """

    amp_data: AmplitudeDataJAX

    def __init__(self, amp_data: AmplitudeDataJAX):
        self.amp_data = amp_data

    # ------------------------------------------------------------------
    # Convenience forwarded properties
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        return self.amp_data.n_modes

    @property
    def l_arr(self) -> jnp.ndarray:
        return self.amp_data.l_arr

    @property
    def m_arr(self) -> jnp.ndarray:
        return self.amp_data.m_arr

    @property
    def k_arr(self) -> jnp.ndarray:
        return self.amp_data.k_arr

    @property
    def n_arr(self) -> jnp.ndarray:
        return self.amp_data.n_arr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_region(
        w: float,
        u: float,
        z: float,
        coeffs: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        z_knots: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate one region's complex amplitudes at a single (w, u, z) point.

        Linear interpolation in ``z`` between the two surrounding z-slices
        (matching FEW), with
        :func:`~fewtrax.utils.bspline_jax.eval_bisplev_batched` for the 2-D
        B-spline at each slice.  ``coeffs`` is ``(Nz, n_modes, 2, N1, N2)``,
        ``t1``/``t2`` the w/u knot vectors, ``z_knots`` the slice positions;
        returns ``(n_modes,)`` complex128.
        """
        Nz = coeffs.shape[0]
        n_modes = coeffs.shape[1]
        N1 = coeffs.shape[3]   # static at JIT time
        N2 = coeffs.shape[4]   # static at JIT time

        # Find the z interval
        iz = jnp.clip(
            jnp.searchsorted(z_knots, z, side="right") - 1,
            0, Nz - 2,
        ).astype(jnp.int32)
        alpha = (z - z_knots[iz]) / (z_knots[iz + 1] - z_knots[iz])

        # Gather the (n_modes, 2, N1, N2) slabs at iz and iz+1.
        # All start indices must share the same int dtype for dynamic_slice.
        c_lo = lax.dynamic_slice(
            coeffs,
            jnp.array([iz,     0, 0, 0, 0], dtype=jnp.int32),
            (1, n_modes, 2, N1, N2),
        )[0]
        c_hi = lax.dynamic_slice(
            coeffs,
            jnp.array([iz + 1, 0, 0, 0, 0], dtype=jnp.int32),
            (1, n_modes, 2, N1, N2),
        )[0]

        # 2-D bisplev at each z-slice, linear-interpolate
        def _eval2d(c_slice):
            # c_slice: (n_modes, 2, N1, N2) — split real/imag
            re = eval_bisplev_batched(w, u, c_slice[:, 0, :, :], t1, t2)
            im = eval_bisplev_batched(w, u, c_slice[:, 1, :, :], t1, t2)
            return re + 1j * im

        return (1.0 - alpha) * _eval2d(c_lo) + alpha * _eval2d(c_hi)

    # ------------------------------------------------------------------
    # Single-point evaluation
    # ------------------------------------------------------------------

    def __call__(
        self,
        a: float,
        p: float,
        e: float,
        pLSO: Optional[float] = None,
    ) -> jnp.ndarray:
        r"""Evaluate complex amplitudes at a single ``(a, p, e)`` point.

        Parameters
        ----------
        a : float
            Dimensionless BH spin (signed; prograde for a > 0).
        p, e : float
            Semi-latus rectum and eccentricity.
        pLSO : float, optional
            Pre-computed separatrix :math:`p_{\rm sep}(|a|, e)`.  When
            provided the internal separatrix computation is skipped.

        Returns
        -------
        jnp.ndarray, shape ``(n_modes,)`` complex128
            Complex mode amplitudes :math:`A_{\ell mkn}`.
        """
        ad = self.amp_data
        a_abs = jnp.abs(a)
        x_in = jnp.where(a >= 0.0, 1.0, -1.0)

        if pLSO is None:
            pLSO = get_separatrix_fast(a_abs, e, x_in)

        in_A = p <= pLSO + DELTAPMAX

        # Compute and clamp coordinates for both regions
        u_A, w_A, z_A = kerrecceq_forward_map_A(
            a_abs, p, e, pLSO, ALPHA_AMP, BETA_AMP
        )
        u_A = jnp.clip(u_A, 0.0, 1.0)
        w_A = jnp.clip(w_A, 0.0, 1.0)
        z_A = jnp.clip(z_A, 0.0, 1.0)

        amp_A = self._eval_region(
            w_A, u_A, z_A,
            ad.coeffs_A, ad.t1_A, ad.t2_A, ad.z_knots_A,
        )

        if ad._has_B:
            u_B, w_B, z_B = kerrecceq_forward_map_B(
                a_abs, p, e, pLSO, is_flux=False
            )
            u_B = jnp.clip(u_B, 0.0, 1.0)
            w_B = jnp.clip(w_B, 0.0, 1.0)
            z_B = jnp.clip(z_B, 0.0, 1.0)

            amp_B = self._eval_region(
                w_B, u_B, z_B,
                ad.coeffs_B, ad.t1_B, ad.t2_B, ad.z_knots_B,
            )
            return jnp.where(in_A, amp_A, amp_B)

        return amp_A

    # ------------------------------------------------------------------
    # FEW-compatible mode selection (JAX-native, one jitted kernel)
    # ------------------------------------------------------------------

    def select_modes_few(
        self,
        a: float,
        p: jnp.ndarray,
        e: jnp.ndarray,
        ylms_pos: jnp.ndarray,
        ylms_neg: jnp.ndarray,
        threshold: float = 1e-5,
        include_minus_mkn: bool = True,
        teuk_modes: Optional[jnp.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""FEW-2.0.0-exact mode selection (Ylm-weighted, per-point cumulative cutoff).

        Reproduces ``few.utils.modeselector.ModeSelector.__call__`` (2.0.0
        threshold path) one-to-one: build the Ylm-weighted power of every
        mode and its ``-mkn`` partner, apply ``cumsum < total·(1−threshold)``
        per trajectory point, and keep the union across points.

        Parameters
        ----------
        a : float
            Dimensionless BH spin.
        p, e : (N_traj,) jnp.ndarray
            Sparse trajectory (no Δt weighting in 2.0.0).
        ylms_pos, ylms_neg : (n_modes,) complex
            :math:`Y_{l_i, +m_i}` and :math:`Y_{l_i, -m_i}` for each mode in
            ``self.amp_data`` order, e.g. ``ylm_gen(l_arr, ±m_arr, θ, φ)``
            (the ``m_i = 0`` entries of ``ylms_neg`` are unused).
        threshold : float
            FEW ``mode_selection_threshold``.
        include_minus_mkn : bool
            Accepted only for API parity; ignored (2.0.0 always pairs ±mkn).
        teuk_modes : (N_traj, n_modes) complex, optional
            Pre-computed amplitudes to score, in ``self.amp_data.l_arr`` order.
            Supplying FEW's own amplitudes makes the selection byte-for-byte
            reproducible against ``ModeSelector``.

        Returns
        -------
        keep_inds : (n_keep,) np.ndarray int
            Sorted-ascending indices into ``self.amp_data``'s mode arrays.
        teuk_all : (N_traj, n_modes) np.ndarray complex128
            Amplitudes on the sparse trajectory; ``teuk_all[:, keep_inds]``
            feeds the mode summation without a second amplitude pass.
        """
        del include_minus_mkn  # FEW 2.0.0 threshold path ignores this flag

        m_np = np.asarray(self.amp_data.m_arr)
        num_m0 = int(np.sum(m_np == 0))                       # static
        num_m_zero_up = int(m_np.size)                        # = n_modes (static)
        num_m_1_up = num_m_zero_up - num_m0                   # static

        yp_j = jnp.asarray(ylms_pos, dtype=jnp.complex128)
        yn_j = jnp.asarray(ylms_neg, dtype=jnp.complex128)

        #if teuk_modes is None:
        #    a_j = jnp.asarray(a, dtype=jnp.float64)
        #    p_j = jnp.asarray(p, dtype=jnp.float64)
        #    e_j = jnp.asarray(e, dtype=jnp.float64)
        #    teuk_all_j, inds_sort, keep_mask = _few_eval_and_score_jit(
        #        self, a_j, p_j, e_j, yp_j, yn_j, num_m0, float(threshold),
        #    )
        #    teuk_all = np.asarray(teuk_all_j)
        #else:
        teuk_in_j = jnp.asarray(teuk_modes, dtype=jnp.complex128)
        inds_sort, keep_mask = _few_score_modes_jit(teuk_in_j, yp_j, yn_j, num_m0, float(threshold))
        teuk_all = np.asarray(teuk_modes)

        # Collect kept indices across all time steps (FEW's union step),
        # remap partner indices to their +m positions, deduplicate.
        inds_sort_np = np.asarray(inds_sort)
        keep_mask_np = np.asarray(keep_mask)
        temp = inds_sort_np[keep_mask_np]                     # 1-D

        # This reorders the modes accordingly to FEW positive and negative
        # Mode ordering is [m=0 modes, m>0 modes], so the -m partner of a m>0 mode is at index
        # index - num_m_1_up.  The m=0 modes have no partners and are left unchanged.
        temp_remapped = np.where(
            temp < num_m_zero_up,
            temp,
            temp - num_m_1_up,
        )

        # Keep the unique modes
        keep_inds = np.unique(temp_remapped)                  # sorted ascending
        return keep_inds, teuk_all

    def evaluate_trajectory(
        self,
        a: float,
        p: jnp.ndarray,
        e: jnp.ndarray,
        pLSO: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        r"""Evaluate amplitudes along a trajectory.

        Uses :func:`jax.vmap` over trajectory points, so the spline
        coefficients are read once and the polynomial evaluation is
        vectorised across all time steps.

        Parameters
        ----------
        a : float
            Dimensionless spin (fixed along a trajectory).
        p, e : jnp.ndarray, shape ``(N_traj,)``
            Semi-latus rectum and eccentricity at each ODE step.
        pLSO : jnp.ndarray, shape ``(N_traj,)``, optional
            Pre-computed separatrix at each step.  When omitted it is
            computed internally via :func:`~fewtrax.utils.geodesic.get_separatrix_fast`.

        Returns
        -------
        jnp.ndarray, shape ``(N_traj, n_modes)`` complex128
            Mode amplitudes at each trajectory point.
        """
        if pLSO is None:
            a_abs = jnp.abs(a)
            x_in = jnp.where(a >= 0.0, 1.0, -1.0)
            pLSO = jax.vmap(
                lambda pi, ei: get_separatrix_fast(a_abs, ei, x_in)
            )(p, e)

        return jax.vmap(
            lambda pi, ei, pli: self(a, pi, ei, pLSO=pli)
        )(p, e, pLSO)

    # ------------------------------------------------------------------
    # Mode selection (mean-power threshold)
    # ------------------------------------------------------------------

    @staticmethod
    def select_modes_from_amps(
        teuk_modes: jnp.ndarray, threshold: float = 1e-5,
    ) -> np.ndarray:
        """Indices of modes whose mean :math:`|A|^2` is >= ``threshold * max``.

        Operates on already-evaluated ``(N_traj, n_modes)`` amplitudes, so the
        trajectory amplitude pass is not repeated.  Returns a sorted numpy
        index array.
        """
        power = np.asarray(jnp.mean(jnp.abs(teuk_modes) ** 2, axis=0))
        if power.max() == 0:
            return np.arange(power.size)
        return np.where(power >= threshold * power.max())[0]

    def select_modes(
        self,
        a: float,
        p: jnp.ndarray,
        e: jnp.ndarray,
        threshold: float = 1e-5,
    ) -> np.ndarray:
        """Evaluate the trajectory amplitudes and threshold by mean power.

        Convenience wrapper around :meth:`evaluate_trajectory` +
        :meth:`select_modes_from_amps`; see the latter for the threshold rule.
        """
        teuk = self.evaluate_trajectory(
            jnp.asarray(a, dtype=jnp.float64),
            jnp.asarray(p, dtype=jnp.float64),
            jnp.asarray(e, dtype=jnp.float64),
        )
        return self.select_modes_from_amps(teuk, threshold)


# ---------------------------------------------------------------------------
# Module-level jit kernels for FEW-compatible mode selection (FEW 2.0.0)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _few_score_modes_jit(
    teuk_all: jnp.ndarray,
    ylms_pos: jnp.ndarray,
    ylms_neg: jnp.ndarray,
    num_m0: int,
    threshold: float,
):
    """FEW-2.0.0 scoring of (already evaluated) mode amplitudes.

    Builds the extended Ylm-weighted power array

        P[t, j] = |A[t, j] · Y_+m[j]|²            for j ∈ [0, n_m)
        P[t, n_m + k] = |A*[t, n_m0 + k] · Y_-m[n_m0 + k]|²   for k ∈ [0, n_m_pos)

    where ``num_m0`` is the count of m = 0 modes (so the m > 0 modes occupy
    the contiguous slice ``[num_m0, n_m)`` thanks to FEW's ``m0sort``
    ordering, which fewtrax replicates in ``_generate_mode_arrays``).

    Each row is argsorted descending; the per-row cumulative cutoff
    ``cumsum < total · (1 − threshold)`` is applied with the same
    "first-overshoot-kept" rule as FEW.  The caller takes the union of
    kept indices across rows in numpy because the kept count varies per
    row.

    Returns
    -------
    inds_sort : (N_traj, n_m + n_m_pos) int  -- descending order per row
    keep_mask : (N_traj, n_m + n_m_pos) bool -- FEW's per-row threshold mask
    """
    # +m column block — all modes paired with Y_{l,+m}.
    positive_amps = jnp.abs(teuk_all * ylms_pos[None, :]) ** 2                # (n_t, n_m)
    # -m partner block — only m > 0 modes (contiguous in m0sort order).
    teuk_neg = teuk_all[:, num_m0:]                                # (n_t, n_m_pos)
    ylms_neg_harm = ylms_neg[num_m0:]                                   # (n_m_pos,)
    negative_amps = jnp.abs(jnp.conj(teuk_neg) * ylms_neg_harm[None, :]) ** 2

    # it contains the power at each time step for all the modes
    power = jnp.concatenate([positive_amps, negative_amps], axis=1)            # (n_t, n_ext)

    # Sorting the modes in each time step depending on their total power
    inds_sort = jnp.argsort(power, axis=1)[:, ::-1]                # (n_t, n_ext)
    sorted_power = jnp.take_along_axis(power, inds_sort, axis=1)
    cumsum = jnp.cumsum(sorted_power, axis=1)

    # Per-row cumulative cutoff:  inds_keep[:, 0]=True; inds_keep[:, i] =
    # cumsum[:, i-1] < cumsum[:, -1] · (1 − threshold).
    budget = cumsum[:, -1:] * (1.0 - threshold)
    head = jnp.ones((cumsum.shape[0], 1), dtype=bool)
    keep_mask = jnp.concatenate([head, cumsum[:, :-1] < budget], axis=1)

    return inds_sort, keep_mask


@eqx.filter_jit
def _few_eval_and_score_jit(
    interp: JAXAmplitudeInterpolator,
    a: jnp.ndarray,
    p: jnp.ndarray,
    e: jnp.ndarray,
    ylms_pos: jnp.ndarray,
    ylms_neg: jnp.ndarray,
    num_m0: int,
    threshold: float,
):
    """Evaluate amplitudes on the trajectory then score (single GPU kernel)."""
    teuk_all = interp.evaluate_trajectory(a, p, e)           # (n_t, n_modes)
    inds_sort, keep_mask = _few_score_modes_jit(
        teuk_all, ylms_pos, ylms_neg, num_m0, threshold,
    )
    return teuk_all, inds_sort, keep_mask

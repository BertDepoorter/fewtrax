"""Teukolsky mode amplitude interpolation from FEW precomputed grids.

Two backends are available:

:class:`AmplitudeInterpolator`
    Legacy numpy/scipy backend.  Evaluates the raw bisplev B-spline
    coefficients stored in the HDF5 file using ``scipy.interpolate.bisplev``
    in a Python loop over trajectory points and modes.  CPU-only, not
    JAX-traceable, kept for validation and backward compatibility.

:class:`JAXAmplitudeInterpolator`
    Fully JAX-native backend.  Uses :class:`~fewtrax.utils.splines.BatchedTricubicSplineE3`
    splines built by :func:`~fewtrax.data.loader.load_amplitude_data_jax`.
    A single ``einsum`` replaces the ``N_traj × N_modes`` bisplev loop,
    evaluating all modes in parallel.  Supports :func:`jax.jit`,
    :func:`jax.grad`, and :func:`jax.vmap` for batched waveform generation.

Design notes (legacy backend)
------------------------------
* The amplitude evaluation is intentionally **numpy-based** rather than
  JAX-based.  The spline coefficient arrays are large (several GB for all
  6993 modes), and precomputing them as JAX arrays on the GPU would exceed
  typical GPU memory budgets.  Instead, the scipy B-spline evaluation runs
  on the CPU and only the resulting (N_traj, N_modes) amplitude array is
  transferred to the GPU for the mode summation.
* Mode selection (thresholding by relative power) is done here to
  reduce the number of modes passed to the summation step.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx

from fewtrax.utils.coordinates import (
    ALPHA_AMP, BETA_AMP, DELTAPMIN, DELTAPMAX,
    kerrecceq_forward_map_A, kerrecceq_forward_map_B,
)
from fewtrax.utils.geodesic import get_separatrix_fast
from fewtrax.utils.bspline_jax import eval_bisplev_batched
from fewtrax.data.loader import AmplitudeData, AmplitudeDataJAX, _separatrix_numpy


class AmplitudeInterpolator:
    r"""Evaluate Teukolsky mode amplitudes :math:`A_{\ell mkn}(a, p, e)`.

    Parameters
    ----------
    amp_data : AmplitudeData
        Pre-loaded amplitude data (B-spline coefficients + mode arrays).
    """

    def __init__(self, amp_data: AmplitudeData):
        self.amp_data = amp_data

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _z_of_a(self, a_abs: float) -> float:
        """Spin → z coordinate."""
        chi_max = (1 - (-0.999))**(1/3)
        chi_min = (1 - 0.999)**(1/3)
        chi = (1.0 - a_abs)**(1/3)
        return float(np.clip((chi - chi_min) / (chi_max - chi_min), 0.0, 1.0))

    def _forward_map_A(self, a_abs: float, p: float, e: float):
        """Map (a, p, e) → (u, w, z) in Region A (amplitude coordinates)."""
        pLSO = float(_separatrix_numpy(a_abs, e, 1.0))
        dp = max(p - pLSO - DELTAPMIN, 0.0)
        u_raw = np.log1p(dp / (DELTAPMAX - DELTAPMIN)) / np.log(2.0)
        u = float(np.clip(u_raw**ALPHA_AMP, 0.0, 1.0))
        z = self._z_of_a(a_abs)
        ESEP, EMAX_LOC = 0.25, 0.9
        check = z + u**BETA_AMP * (1.0 - z)
        sgn = np.sign(check)
        Secc = ESEP + (EMAX_LOC - ESEP) * sgn * np.sqrt(sgn * max(check, 0.0))
        Secc = max(Secc, 1e-12)
        w = float(np.clip(e / Secc, 0.0, 1.0))
        return u, w, z

    def _forward_map_B(self, a_abs: float, p: float, e: float):
        """Map (a, p, e) → (u, w, z) in Region B (amplitude coordinates)."""
        pLSO = float(_separatrix_numpy(a_abs, e, 1.0))
        # Match FEW's `pc = pLSO + DPC_REGIONB` where DPC_REGIONB = 9.0
        # (i.e. DELTAPMAX - DELTAPMIN, not DELTAPMAX).
        DELTAPMIN_B = 9.0
        PMAX_B = 200.0
        pc = pLSO + DELTAPMIN_B
        u = float(np.clip(
            (pc**-0.5 - p**-0.5) / (pc**-0.5 - (PMAX_B + pc)**-0.5),
            0.0, 1.0,
        ))
        EMAX_B = 0.9
        w = float(np.clip(e / EMAX_B, 0.0, 1.0))
        z = self._z_of_a(a_abs)
        return u, w, z

    def _in_region_A(self, a_abs: float, p: float, e: float) -> bool:
        pLSO = float(_separatrix_numpy(a_abs, e, 1.0))
        return (p - pLSO) <= DELTAPMAX

    # ------------------------------------------------------------------
    # Amplitude evaluation
    # ------------------------------------------------------------------

    def _eval_block(
        self, a_abs: float, p_arr: np.ndarray, e_arr: np.ndarray,
    ) -> np.ndarray:
        """Evaluate amplitudes at all trajectory points.  Returns (N, n_modes)."""
        from scipy.interpolate import bisplev

        ad = self.amp_data
        N = len(p_arr)
        n_modes = ad.n_modes
        amps = np.zeros((N, n_modes), dtype=complex)

        for it in range(N):
            p, e = float(p_arr[it]), float(e_arr[it])
            if not np.isfinite(p) or not np.isfinite(e) or p <= 0:
                continue

            in_A = self._in_region_A(a_abs, p, e)
            if in_A:
                u, w, z = self._forward_map_A(a_abs, p, e)
                coeffs = ad.coeffs_A
                u_kn, w_kn, z_kn = ad.u_knots_A, ad.w_knots_A, ad.z_knots_A
            else:
                if ad.coeffs_B is None:
                    continue
                u, w, z = self._forward_map_B(a_abs, p, e)
                coeffs = ad.coeffs_B
                u_kn, w_kn, z_kn = ad.u_knots_B, ad.w_knots_B, ad.z_knots_B

            n_z = len(z_kn)
            iz = int(np.clip(np.searchsorted(z_kn, z) - 1, 0, n_z - 2))
            z_lo, z_hi = z_kn[iz], z_kn[iz + 1]
            alpha = float(np.clip((z - z_lo) / (z_hi - z_lo + 1e-30), 0.0, 1.0))

            # FEW stores coefficients in (w, u) axis order; bisplev first arg
            # is the query in the first axis direction (w), second is u.
            w_pt = np.array([w])
            u_pt = np.array([u])
            for iz2, fac in [(iz, 1.0 - alpha), (iz + 1, alpha)]:
                if fac < 1e-10:
                    continue
                c_block = coeffs[iz2]   # (n_modes, 2, n_coeffs)
                for im in range(n_modes):
                    vr = float(bisplev(w_pt, u_pt, (w_kn, u_kn, c_block[im, 0], 3, 3)))
                    vi = float(bisplev(w_pt, u_pt, (w_kn, u_kn, c_block[im, 1], 3, 3)))
                    amps[it, im] += complex(vr, vi) * fac

        return amps

    # ------------------------------------------------------------------
    # Convenience properties (forwarded from amp_data)
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        return self.amp_data.n_modes

    @property
    def l_arr(self) -> np.ndarray:
        return self.amp_data.l_arr

    @property
    def m_arr(self) -> np.ndarray:
        return self.amp_data.m_arr

    @property
    def k_arr(self) -> np.ndarray:
        return self.amp_data.k_arr

    @property
    def n_arr(self) -> np.ndarray:
        return self.amp_data.n_arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        a: float,
        p: np.ndarray,
        e: np.ndarray,
        x: float = 1.0,
        specific_modes: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        r"""Evaluate Teukolsky mode amplitudes along a trajectory.

        Parameters
        ----------
        a : float
            Dimensionless BH spin.
        p, e : np.ndarray, shape (N,)
            Semi-latus rectum and eccentricity.
        x : float
            Inclination cosine (±1).
        specific_modes : sequence of int, optional
            Mode indices to evaluate.  Default: all modes.

        Returns
        -------
        amps : np.ndarray, shape (N, n_modes) complex
        """
        a_abs = float(np.abs(a))
        p_arr = np.asarray(p, dtype=float)
        e_arr = np.asarray(e, dtype=float)
        amps = self._eval_block(a_abs, p_arr, e_arr)
        if specific_modes is not None:
            amps = amps[:, specific_modes]
        return amps

    def get_mode_power(
        self,
        a: float,
        p: np.ndarray,
        e: np.ndarray,
        x: float = 1.0,
    ) -> np.ndarray:
        """Return mean |A|² per mode, shape (n_modes,)."""
        amps = self.evaluate(a, p, e, x)
        return np.mean(np.abs(amps)**2, axis=0)

    def select_modes(
        self,
        a: float,
        p: np.ndarray,
        e: np.ndarray,
        threshold: float = 1e-5,
        x: float = 1.0,
    ) -> np.ndarray:
        """Return indices of modes with relative power ≥ ``threshold × max``.

        Returns
        -------
        indices : np.ndarray of int
        """
        power = self.get_mode_power(a, p, e, x)
        if power.max() == 0:
            return np.arange(len(power))
        return np.where(power >= threshold * power.max())[0]


# ---------------------------------------------------------------------------
# JAX-native amplitude interpolator
# ---------------------------------------------------------------------------

class JAXAmplitudeInterpolator(eqx.Module):
    r"""Fully JAX-native Teukolsky amplitude evaluator.

    Wraps an :class:`~fewtrax.data.loader.AmplitudeDataJAX` container and
    provides a :meth:`__call__` that evaluates complex mode amplitudes
    :math:`A_{\ell mkn}(a, p, e)` for all modes in a single fused
    ``einsum``, replacing the ``N_{\rm traj} \times N_{\rm modes}``
    ``scipy.bisplev`` Python loop in :class:`AmplitudeInterpolator`.

    All public methods are :func:`jax.jit`-able, :func:`jax.vmap`-able, and
    support :func:`jax.grad` / :func:`jax.jacfwd`.

    Parameters
    ----------
    amp_data : AmplitudeDataJAX
        Pre-built JAX amplitude container; construct with
        :func:`~fewtrax.data.loader.load_amplitude_data_jax`.

    Examples
    --------
    Single-point evaluation::

        interp = JAXAmplitudeInterpolator(amp_data)
        amps = interp(a=0.9, p=10.0, e=0.3)   # (n_modes,) complex128

    Trajectory evaluation (vectorised over time steps)::

        amps_traj = interp.evaluate_trajectory(a=0.9, p=p_arr, e=e_arr)
        # (N_traj, n_modes) complex128

    Batched waveform generation (vmap over initial conditions)::

        batch_eval = jax.vmap(
            lambda a, p0, e0: interp.evaluate_trajectory(a, traj_p(a, p0, e0), ...)
        )
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

        Uses linear interpolation in ``z`` between the two surrounding
        z-slices (matching FEW's ``AmplitudeInterpolator`` exactly), and
        :func:`~fewtrax.utils.bspline_jax.eval_bisplev_batched` for the 2-D
        B-spline evaluation at each slice.

        Parameters
        ----------
        w, u, z : float
            Normalised amplitude coordinates.
        coeffs : (Nz, n_modes, 2, N1, N2) float64
        t1 : (len_t1,) float64  — w-axis knot vector.
        t2 : (len_t2,) float64  — u-axis knot vector.
        z_knots : (Nz,) float64 — z-slice positions.

        Returns
        -------
        (n_modes,) complex128
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
    # Trajectory evaluation
    # ------------------------------------------------------------------

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
        r"""FEW-compatible mode selection performed entirely on the GPU.

        Reproduces ``few.utils.modeselector.ModeSelector.__call__`` (default
        threshold path) of FEW **2.0.0** one-to-one.  The 2.0.0 algorithm
        is different from earlier FEW releases — there is no Δt weighting,
        and the cumulative cutoff is applied per trajectory point with the
        final kept set being the union across all time steps:

        1. Build the extended power array of shape
           ``(N_traj, n_m + n_m_pos)`` by concatenating the m≥0 column
           block ``|A · Y_{l,+m}|²`` with the m>0 column block
           ``|A* · Y_{l,-m}|²`` (one column per ``(-m, -k, -n)`` partner).
        2. For each row (time step) sort descending and apply
           ``cumsum < total · (1 − threshold)``.  The first mode that
           pushes the cumsum past the budget is retained.
        3. The final kept set is the union of all per-time-step kept
           indices.  Partner indices land in ``[n_m, n_m + n_m_pos)`` and
           are remapped to their +m positions ``index − n_m_pos`` (FEW's
           CUDA kernel keeps +m and -m together by construction).

        The whole power-build + per-row argsort + cumsum runs in one
        ``eqx.filter_jit`` kernel and yields the ``(N_traj, n_modes)``
        amplitude table as a by-product, so the caller does not need a
        second amplitude pass.

        Parameters
        ----------
        a : float
            Dimensionless BH spin.
        p, e : (N_traj,) jnp.ndarray
            Sparse trajectory.  ``t`` is *not* required — FEW 2.0.0 does
            not weight by Δt.
        ylms_pos : (n_modes,) complex
            Spin-weighted spherical harmonic :math:`Y_{l_i, +m_i}` for each
            mode in ``self.amp_data.l_arr`` / ``m_arr``.  Easiest source::

                ylms_pos = ylm_gen(amp_data.l_arr,  amp_data.m_arr, θ, φ)

        ylms_neg : (n_modes,) complex
            :math:`Y_{l_i, -m_i}` for each mode.  The ``m_i = 0`` entries
            are unused (FEW does not pair m=0 modes with a -m partner).
            Easiest source::

                ylms_neg = ylm_gen(amp_data.l_arr, -amp_data.m_arr, θ, φ)

        threshold : float
            FEW mode-selection threshold (``mode_selection_threshold``).
        include_minus_mkn : bool
            Accepted only for API parity.  FEW 2.0.0 always builds the
            extended power including the -mkn partner in the threshold
            path; setting this to ``False`` does not change the result.
        teuk_modes : (N_traj, n_modes) complex, optional
            Pre-computed amplitudes to score, in the same mode order as
            ``self.amp_data.l_arr``.  When supplied the JAX amplitude pass
            is skipped, so the selection becomes byte-for-byte
            reproducible against FEW (whose amplitudes come from
            scipy/cupy ``bisplev``).  Use this to feed
            ``model.amplitude_generator(a, p, e, x0)`` when you need to
            match ``ModeSelector`` exactly; omit it for the all-JAX fast
            path.

        Returns
        -------
        keep_inds : (n_keep,) np.ndarray int
            Sorted-ascending indices into the mode arrays of
            ``self.amp_data``.
        teuk_all : (N_traj, n_modes) np.ndarray complex128
            All amplitudes evaluated on the sparse trajectory.  Use
            ``teuk_all[:, keep_inds]`` to feed the mode-summation step
            without re-evaluating the amplitudes.
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

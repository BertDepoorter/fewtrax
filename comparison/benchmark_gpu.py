"""GPU benchmark: fewtrax vmap + autodiff for EMRI frequency-track identification.

This script targets the regime where fewtrax offers the strongest advantage over
CPU-serial alternatives: large-batch trajectory evaluation via ``jax.vmap``
combined with end-to-end automatic differentiation through the ODE.

Scientific context (see gradient_identification.md)
----------------------------------------------------
Semi-coherent STFT searches produce candidate chirping tracks in the LISA
time-frequency plane.  Each track approximates the instantaneous GW frequency
of one Teukolsky harmonic mode (l, m, k, n):

    f_mkn(t; θ) = |m Ω_φ(t) + k Ω_θ(t) + n Ω_r(t)| / 2π M_s

Recovering θ = (M, μ, a, p₀, e₀) requires minimising

    L(θ) = Σ_t [f_pred(t; θ) - f_obs(t)]²

The GPU excels at:
  - vmap over N ≫ n_CPU_cores parameter sets (Section A)
  - jax.value_and_grad(L) for iterative optimisation (Section B)
  - jax.jacfwd of f_mkn for Fisher-matrix computation (Section C)
  - vmap(jacfwd) over N parameter sets — batched Fisher grid (Section D)
  - vmap(value_and_grad) over N starting points — multi-start descent (Section E)
  - vmap over candidate mode numbers (m, k, n) — mode identification (Section F)

No single-trajectory vs FEW comparison is included: the GPU advantage for a
single sequential ODE is minimal.  The genuine gain emerges at N ≳ 64 and
whenever gradients are needed.

Usage
-----
    python benchmark_gpu.py [/path/to/few/data]
    python benchmark_gpu.py --n-repeat 5 --T 0.5 --dense 100
    python benchmark_gpu.py --skip-fisher --skip-multistart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import find_data_dir, block_jax, print_header, print_table, repeat_timer

from fewtrax.utils.constants import MTSUN_SI
from fewtrax.utils.geodesic import get_separatrix_fast, get_fundamental_frequencies_fast


# ---------------------------------------------------------------------------
# Device / memory helpers
# ---------------------------------------------------------------------------

def describe_devices() -> None:
    devices = jax.devices()
    print(f"\n  JAX version : {jax.__version__}")
    for d in devices:
        print(f"  {d.id}: {d.device_kind}  platform={d.platform}")
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in out.strip().splitlines():
            name, free = line.split(", ")
            print(f"  GPU free memory (nvidia-smi): {free} MiB")
    except Exception:
        pass


def get_peak_mem_mib() -> float:
    try:
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1024**2
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Random parameter grid
# ---------------------------------------------------------------------------

def build_param_grid(N: int, seed: int = 42, T_yr: float = 0.5) -> dict:
    """Draw N random EMRI parameter sets with p0 well inside the grid.

    Uses Δp ∈ [3, 10] M so no trajectory plunges during T years for the
    mass ranges sampled.  All a > 0 (prograde equatorial).
    """
    rng = np.random.default_rng(seed)
    M_arr  = 10.0 ** rng.uniform(np.log10(5e5), np.log10(5e6), N)
    mu_arr = rng.uniform(5.0, 50.0, N)
    a_arr  = rng.uniform(0.05, 0.90, N)
    e_arr  = rng.uniform(0.05, 0.70, N)
    dp_arr = rng.uniform(3.0, 10.0, N)
    p_arr  = np.array([
        float(get_separatrix_fast(jnp.float64(a), jnp.float64(e), jnp.float64(1.0))) + dp
        for a, e, dp in zip(a_arr, e_arr, dp_arr)
    ])
    return dict(M=M_arr, mu=mu_arr, a=a_arr, p0=p_arr, e0=e_arr)


# ---------------------------------------------------------------------------
# Differentiable frequency-track primitives
# ---------------------------------------------------------------------------
#
# The physical observable for gradient_identification.md is the instantaneous
# GW frequency of harmonic (m, k, n):
#
#     f_mkn(t; θ) = |m Ω_φ + k Ω_θ + n Ω_r| / (2π M_s)
#
# This function is fully differentiable w.r.t. θ = (M, μ, a, p₀, e₀) via
# jax.jacfwd / jax.grad, since:
#   (1) diffrax propagates derivatives through the ODE automatically,
#   (2) get_fundamental_frequencies_fast is pure JAX arithmetic.

def make_freq_track(traj, m_mode: int, k_mode: int, n_mode: int,
                    T: float, dense_steps: int):
    """Return differentiable (M, mu, a, p0, e0) -> frequency track [Hz].

    The inner jax.vmap evaluates Ω(p, e) at each of the dense_steps saved
    trajectory points.  This composes correctly with an outer jax.vmap (over
    parameter batches) and with jax.jacfwd / jax.grad.
    """
    m = jnp.int32(m_mode)
    k = jnp.int32(k_mode)
    n = jnp.int32(n_mode)

    def fn(M, mu, a, p0, e0):
        t_s, p_arr, e_arr, _, _, _ = traj(
            p0=p0, e0=e0, T=T, a=a, M=M, mu=mu,
            dense_steps=dense_steps, x0=jnp.float64(1.0),
        )
        a_abs = jnp.abs(a)
        M_s   = (M + mu) * MTSUN_SI

        def _f_at_pe(pe):
            p_, e_ = pe
            Om_phi, Om_theta, Om_r = get_fundamental_frequencies_fast(
                a_abs, p_, e_, jnp.float64(1.0)
            )
            return jnp.abs(m * Om_phi + k * Om_theta + n * Om_r) / (2.0 * jnp.pi * M_s)

        return jax.vmap(_f_at_pe)(jnp.stack([p_arr, e_arr], axis=1))  # (dense_steps,)

    return fn


def make_loss(freq_track_fn, f_obs: jnp.ndarray):
    """Scalar MSE loss  L(θ) = mean((f_pred(θ) - f_obs)²).

    Differentiable via jax.grad (reverse-mode) or jax.value_and_grad.
    The gradient ∇_θ L is used directly by gradient-descent optimisers
    (Adam, L-BFGS) and by HMC/NUTS samplers.
    """
    def loss(M, mu, a, p0, e0):
        f_pred = freq_track_fn(M, mu, a, p0, e0)
        return jnp.mean((f_pred - f_obs) ** 2)
    return loss


def make_freq_of_theta(freq_track_fn):
    """Wrap (M, mu, a, p0, e0) -> f as a single-array interface for jacfwd/vmap."""
    def fn(theta):
        M_, mu_, a_, p0_, e0_ = theta
        return freq_track_fn(M_, mu_, a_, p0_, e0_)
    return fn


def make_loss_of_theta(loss_fn):
    """Wrap scalar loss as a single-array interface for vmap(grad)."""
    def fn(theta):
        M_, mu_, a_, p0_, e0_ = theta
        return loss_fn(M_, mu_, a_, p0_, e0_)
    return fn


# ---------------------------------------------------------------------------
# A. vmap trajectory throughput
# ---------------------------------------------------------------------------

def bench_vmap_throughput(traj, grid, batch_sizes, T, dense_steps, nw, nr):
    """Vmapped EMRIInspiralFast over growing batch sizes.

    Measures raw trajectory throughput: the foundation on which all
    autodiff and optimisation sections build.
    """
    results = []
    for N in batch_sizes:
        M  = jnp.array(grid["M"][:N],  dtype=jnp.float64)
        mu = jnp.array(grid["mu"][:N], dtype=jnp.float64)
        a  = jnp.array(grid["a"][:N],  dtype=jnp.float64)
        p0 = jnp.array(grid["p0"][:N], dtype=jnp.float64)
        e0 = jnp.array(grid["e0"][:N], dtype=jnp.float64)

        def single(M_, mu_, a_, p0_, e0_):
            return traj(p0=p0_, e0=e0_, T=T, a=a_, M=M_, mu=mu_,
                        dense_steps=dense_steps, x0=jnp.float64(1.0))

        batched = jax.jit(jax.vmap(single))

        def fn():
            block_jax(batched(M, mu, a, p0, e0))

        mean_s, std_s = repeat_timer(fn, n_warmup=nw, n_repeat=nr)
        mem_mib   = get_peak_mem_mib()
        throughput = N / mean_s
        results.append(dict(N=N, mean_s=mean_s, std_s=std_s,
                             throughput=throughput, mem_mib=mem_mib))
        print(f"  N={N:5d}: {mean_s*1e3:8.2f} ± {std_s*1e3:5.2f} ms  "
              f"({throughput:7.1f} traj/s)  mem_peak={mem_mib:.0f} MiB")
    return results


# ---------------------------------------------------------------------------
# B. Autodiff overhead — gradient of scalar loss w.r.t. (M, μ, a, p₀, e₀)
# ---------------------------------------------------------------------------

def bench_autodiff(loss_fn, theta_ref, nw, nr):
    """Compare forward eval, value_and_grad, and jacfwd costs.

    The gradient ∇L is used at every step of gradient-based optimisers
    (Adam, L-BFGS) and HMC leapfrog steps.  The overhead factor vs a
    plain forward eval quantifies the cost of adding autodiff.
    """
    M, mu, a, p0, e0 = theta_ref

    # --- forward only ---
    fwd_jit = jax.jit(loss_fn)
    def fwd_fn():
        block_jax(fwd_jit(M, mu, a, p0, e0))
    fwd_mean, fwd_std = repeat_timer(fwd_fn, n_warmup=nw, n_repeat=nr)
    print(f"  Forward eval:              {fwd_mean*1e3:7.2f} ± {fwd_std*1e3:5.2f} ms")

    # --- value + reverse-mode gradient (adjoint through ODE) ---
    vg_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4)))
    def vg_fn():
        block_jax(vg_jit(M, mu, a, p0, e0))
    vg_mean, vg_std = repeat_timer(vg_fn, n_warmup=nw, n_repeat=nr)
    overhead = vg_mean / fwd_mean
    print(f"  value_and_grad (reverse):  {vg_mean*1e3:7.2f} ± {vg_std*1e3:5.2f} ms  "
          f"({overhead:.2f}× fwd)")

    # --- forward-mode gradient via jacfwd (5 JVPs) ---
    # jacfwd(scalar_loss) = grad, but forces forward-mode instead of adjoint.
    # For 5 parameters, forward mode costs 5 JVPs ≈ 5× forward pass.
    jf_jit = jax.jit(jax.jacfwd(loss_fn, argnums=(0, 1, 2, 3, 4)))
    def jf_fn():
        block_jax(jf_jit(M, mu, a, p0, e0))
    jf_mean, jf_std = repeat_timer(jf_fn, n_warmup=nw, n_repeat=nr)
    overhead_jf = jf_mean / fwd_mean
    print(f"  jacfwd (forward, 5 JVPs):  {jf_mean*1e3:7.2f} ± {jf_std*1e3:5.2f} ms  "
          f"({overhead_jf:.2f}× fwd)")

    return dict(fwd_ms=fwd_mean*1e3, vg_ms=vg_mean*1e3, jf_ms=jf_mean*1e3,
                overhead_rev=overhead, overhead_fwd=overhead_jf)


# ---------------------------------------------------------------------------
# C. Fisher matrix — jax.jacfwd of frequency track
# ---------------------------------------------------------------------------
#
# The Cramér-Rao Fisher matrix for a track observation is
#
#     F_ij = Σ_t (∂f/∂θ_i)(∂f/∂θ_j) / σ_f²
#
# where σ_f = 1/T_seg (STFT bin width, typically ~10⁻⁵ Hz for a week).
# jax.jacfwd gives the (dense_steps × 5) Jacobian ∂f/∂θ via 5 JVPs, which
# is more efficient than jacrev (dense_steps VJPs) for dense_steps ≫ 5.

def bench_fisher_single(freq_of_theta, theta_ref_arr, dense_steps, sigma_f, nw, nr):
    """Benchmark Fisher matrix for a single parameter set."""

    @jax.jit
    def fisher_fn(theta):
        J = jax.jacfwd(freq_of_theta)(theta)   # (dense_steps, 5)
        return J.T @ J / sigma_f**2             # (5, 5)

    def fn():
        block_jax(fisher_fn(theta_ref_arr))
    mean_s, std_s = repeat_timer(fn, n_warmup=nw, n_repeat=nr)
    print(f"  Single Fisher (5×5, jacfwd): {mean_s*1e3:7.2f} ± {std_s*1e3:5.2f} ms")

    # Print eigenvalues for interpretability
    F = np.asarray(fisher_fn(theta_ref_arr))
    eigs = np.linalg.eigvalsh(F)
    cond = eigs[-1] / max(eigs[0], 1e-300)
    print(f"    Eigenvalues: {eigs}")
    print(f"    Condition number: {cond:.2e}")
    print(f"    Cramér-Rao σ(M)={np.sqrt(1/max(F[0,0],1e-300)):.3e} M_sun  "
          f"σ(a)={np.sqrt(1/max(F[2,2],1e-300)):.3e}")

    return dict(mean_ms=mean_s*1e3, std_ms=std_s*1e3,
                fisher_matrix=F, eigenvalues=eigs, condition=cond)


def bench_fisher_batched(freq_of_theta, grid, batch_sizes, sigma_f, nw, nr):
    """Benchmark vmap(jacfwd) — N Fisher matrices simultaneously on GPU.

    This is the key use case for parameter-space surveys: computing the
    Fisher information over a grid of (M, μ, a, p₀, e₀) values.
    """
    @jax.jit
    def single_fisher(theta):
        J = jax.jacfwd(freq_of_theta)(theta)
        return J.T @ J / sigma_f**2

    batched_fisher = jax.jit(jax.vmap(single_fisher))

    results = []
    for N in batch_sizes:
        thetas = jnp.array(
            np.stack([grid["M"][:N], grid["mu"][:N], grid["a"][:N],
                      grid["p0"][:N], grid["e0"][:N]], axis=1),
            dtype=jnp.float64,
        )  # (N, 5)

        def fn():
            block_jax(batched_fisher(thetas))

        mean_s, std_s = repeat_timer(fn, n_warmup=nw, n_repeat=nr)
        throughput = N / mean_s
        mem_mib = get_peak_mem_mib()
        results.append(dict(N=N, mean_s=mean_s, std_s=std_s,
                             throughput=throughput, mem_mib=mem_mib))
        print(f"  N={N:5d}: {mean_s*1e3:8.2f} ± {std_s*1e3:5.2f} ms  "
              f"({throughput:7.1f} Fisher/s)  mem_peak={mem_mib:.0f} MiB")
    return results


# ---------------------------------------------------------------------------
# D. Multi-start gradient descent — vmap(value_and_grad) over N starts
# ---------------------------------------------------------------------------
#
# Multi-start optimisation (gradient_identification.md §5.1) runs N gradient-
# descent chains from different θ initialisations in parallel.  One GPU step
# computes N (loss, grad) pairs simultaneously, compared to N sequential CPU
# calls.  The GPU advantage scales with N.

def bench_multistart_grad(loss_of_theta, theta_ref_arr, grid, batch_sizes, nw, nr):
    """Benchmark vmap(value_and_grad) — N gradient evaluations in parallel.

    Each starting point is a perturbation of theta_ref drawn from the grid,
    mimicking a multi-start optimisation initialised from a coarse parameter scan.
    """
    batched_vg = jax.jit(jax.vmap(jax.value_and_grad(loss_of_theta)))

    results = []
    for N in batch_sizes:
        thetas = jnp.array(
            np.stack([grid["M"][:N], grid["mu"][:N], grid["a"][:N],
                      grid["p0"][:N], grid["e0"][:N]], axis=1),
            dtype=jnp.float64,
        )  # (N, 5)

        def fn():
            vals, grads = batched_vg(thetas)
            block_jax((vals, grads))

        mean_s, std_s = repeat_timer(fn, n_warmup=nw, n_repeat=nr)
        throughput = N / mean_s
        results.append(dict(N=N, mean_s=mean_s, std_s=std_s, throughput=throughput))
        print(f"  N={N:5d}: {mean_s*1e3:8.2f} ± {std_s*1e3:5.2f} ms  "
              f"({throughput:7.1f} grad-evals/s)")
    return results


# ---------------------------------------------------------------------------
# E. Mode identification — vmap over candidate (m, k, n) combinations
# ---------------------------------------------------------------------------
#
# Mode numbers are integers (not differentiable), so identification uses
# brute force: evaluate the loss for every candidate (m, k, n) combination
# and select the best fit.  With jax.vmap this is embarrassingly parallel.
# (gradient_identification.md §6.1)

def bench_mode_identification(traj, theta_ref, f_obs_ref, T, dense_steps, nw, nr):
    """Benchmark vmap over candidate mode sets for a fixed parameter vector.

    Candidate set: |m| ≤ 4, k = 0 (equatorial), |n| ≤ 3 → ~36 combinations.
    Evaluates all losses simultaneously on GPU.
    """
    M, mu, a, p0, e0 = theta_ref

    # Build candidate (m, k, n) grid — equatorial modes (k=0), |m|≤4, |n|≤3
    candidates = [
        (m, 0, n)
        for m in range(1, 5)   # m=0 gives zero frequency; m<0 equivalent by conjugate
        for n in range(-3, 4)
    ]
    m_arr = jnp.array([c[0] for c in candidates], dtype=jnp.int32)
    k_arr = jnp.array([c[1] for c in candidates], dtype=jnp.int32)
    n_arr = jnp.array([c[2] for c in candidates], dtype=jnp.int32)
    N_cand = len(candidates)
    print(f"  Candidate modes: {N_cand}")

    # Run the trajectory once to get (p, e) arrays; reuse across mode evaluations
    t_s, p_arr, e_arr, _, _, _ = traj(
        p0=jnp.float64(p0), e0=jnp.float64(e0),
        T=T, a=jnp.float64(a),
        M=jnp.float64(M), mu=jnp.float64(mu),
        dense_steps=dense_steps, x0=jnp.float64(1.0),
    )
    pe = jnp.stack([p_arr, e_arr], axis=1)  # (dense_steps, 2)
    M_s = (M + mu) * MTSUN_SI
    a_abs = float(abs(a))

    # Pre-compute (Ω_φ, Ω_θ, Ω_r) at each trajectory point — shared across modes
    def _omegas(pe_):
        return get_fundamental_frequencies_fast(
            jnp.float64(a_abs), pe_[0], pe_[1], jnp.float64(1.0)
        )
    omegas = jax.vmap(_omegas)(pe)   # 3-tuple of (dense_steps,)
    Om_phi, Om_theta, Om_r = omegas  # each (dense_steps,)

    # f_mkn for a single candidate mode
    def loss_for_mode(m, k, n):
        f_pred = jnp.abs(m * Om_phi + k * Om_theta + n * Om_r) / (2.0 * jnp.pi * M_s)
        return jnp.mean((f_pred - f_obs_ref) ** 2)

    # vmap over candidate modes using index into pre-built arrays
    def loss_for_idx(idx):
        return loss_for_mode(m_arr[idx], k_arr[idx], n_arr[idx])

    idx_arr = jnp.arange(N_cand, dtype=jnp.int32)
    all_losses_fn = jax.jit(jax.vmap(loss_for_idx))

    def fn():
        block_jax(all_losses_fn(idx_arr))

    mean_s, std_s = repeat_timer(fn, n_warmup=nw, n_repeat=nr)
    throughput = N_cand / mean_s
    print(f"  vmap over {N_cand} modes: {mean_s*1e3:.3f} ± {std_s*1e3:.3f} ms  "
          f"({throughput:.1f} mode-evals/s)")

    # Show the minimum-loss mode (self-consistency check: should match the ref mode)
    losses = np.asarray(all_losses_fn(idx_arr))
    best = int(np.argmin(losses))
    print(f"  Best-fit mode: m={candidates[best][0]} k={candidates[best][1]} "
          f"n={candidates[best][2]}  loss={losses[best]:.3e}")

    return dict(mean_ms=mean_s*1e3, std_ms=std_s*1e3,
                candidates=candidates, losses=losses, best_mode=candidates[best])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--n-repeat",  type=int,   default=5)
    parser.add_argument("--n-warmup",  type=int,   default=2)
    parser.add_argument("--n-grid",    type=int,   default=1024,
                        help="Size of random parameter grid (default: 1024)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--T",         type=float, default=0.5,
                        help="Observation time [yr] (default: 0.5)")
    parser.add_argument("--dense",     type=int,   default=100,
                        help="Trajectory output points (default: 100)")
    parser.add_argument("--m-mode",    type=int,   default=2,
                        help="Harmonic index m for frequency track (default: 2)")
    parser.add_argument("--k-mode",    type=int,   default=0)
    parser.add_argument("--n-mode",    type=int,   default=0)
    parser.add_argument("--sigma-f",   type=float, default=1e-5,
                        help="Frequency noise σ_f [Hz] (default: 1e-5, ≈ 1/week STFT)")
    parser.add_argument("--skip-autodiff",   action="store_true")
    parser.add_argument("--skip-fisher",     action="store_true")
    parser.add_argument("--skip-multistart", action="store_true")
    parser.add_argument("--skip-mode-id",    action="store_true")
    args = parser.parse_args()

    nw, nr = args.n_warmup, args.n_repeat
    T      = args.T
    dense  = args.dense

    print_header("Device information")
    describe_devices()

    data_dir = find_data_dir(args.data_dir)
    print(f"\n  FEW data directory: {data_dir}")

    # Load fewtrax
    print("\n  Loading fewtrax …", end=" ", flush=True)
    from fewtrax.data import load_flux_data
    from fewtrax.trajectory import EMRIInspiralFast
    flux_data = load_flux_data(data_dir)
    traj = EMRIInspiralFast(flux_data)
    print("done")

    # Build parameter grid
    print(f"\n  Building random parameter grid  (N={args.n_grid}, seed={args.seed}) …",
          end=" ", flush=True)
    grid = build_param_grid(args.n_grid, seed=args.seed, T_yr=T)
    print(" done")
    print(f"  p0 range : [{grid['p0'].min():.2f}, {grid['p0'].max():.2f}] M")
    print(f"  e0 range : [{grid['e0'].min():.3f}, {grid['e0'].max():.3f}]")
    print(f"  a  range : [{grid['a'].min():.3f},  {grid['a'].max():.3f}]")
    print(f"  M  range : [{grid['M'].min():.2e}, {grid['M'].max():.2e}] Msun")
    print(f"  mu range : [{grid['mu'].min():.1f}, {grid['mu'].max():.1f}] Msun")

    # Reference parameters (used for autodiff / Fisher / mode-ID sections)
    # Pick a mid-grid point for reproducibility
    i_ref = args.n_grid // 2
    theta_ref = (
        float(grid["M"][i_ref]),
        float(grid["mu"][i_ref]),
        float(grid["a"][i_ref]),
        float(grid["p0"][i_ref]),
        float(grid["e0"][i_ref]),
    )
    theta_ref_arr = jnp.array(theta_ref, dtype=jnp.float64)
    print(f"\n  Reference θ: M={theta_ref[0]:.3e}  μ={theta_ref[1]:.2f}  "
          f"a={theta_ref[2]:.3f}  p₀={theta_ref[3]:.3f}  e₀={theta_ref[4]:.3f}")
    print(f"  Frequency track: mode (m={args.m_mode}, k={args.k_mode}, n={args.n_mode})")
    print(f"  σ_f = {args.sigma_f:.1e} Hz  (STFT noise model)")

    # Build differentiable frequency track and loss for the reference mode
    freq_track_fn = make_freq_track(traj, args.m_mode, args.k_mode, args.n_mode,
                                    T, dense)
    freq_of_theta  = make_freq_of_theta(freq_track_fn)

    # Synthetic "observed" frequency track: reference parameters + noise
    print("\n  Generating synthetic f_obs from reference parameters …", end=" ")
    rng = np.random.default_rng(999)
    f_clean = np.asarray(jax.jit(freq_track_fn)(*theta_ref))
    f_obs   = jnp.array(f_clean + rng.normal(0.0, args.sigma_f, dense), dtype=jnp.float64)
    print(f"done  (f̄ = {float(jnp.mean(f_obs))*1e3:.3f} mHz)")

    loss_fn       = make_loss(freq_track_fn, f_obs)
    loss_of_theta = make_loss_of_theta(loss_fn)

    # ------------------------------------------------------------------
    # A. vmap trajectory throughput
    # ------------------------------------------------------------------
    print_header("A. vmap trajectory throughput  (EMRIInspiralFast)")
    print(f"  T={T} yr,  dense_steps={dense},  n_warmup={nw},  n_repeat={nr}")

    batch_sizes_A = [1, 4, 16, 64, 256, 1024]
    traj_results  = bench_vmap_throughput(traj, grid, batch_sizes_A, T, dense, nw, nr)

    rows_A = [
        (str(r["N"]),
         f"{r['mean_s']*1e3:.2f} ± {r['std_s']*1e3:.2f}",
         f"{r['throughput']:.1f}",
         f"{r['mem_mib']:.0f}")
        for r in traj_results
    ]
    print()
    print_table(rows_A,
                ["N_batch", "time [ms]", "traj/s", "mem_peak [MiB]"],
                [10, 22, 12, 16])

    # ------------------------------------------------------------------
    # B. Autodiff overhead — scalar loss gradient
    # ------------------------------------------------------------------
    print_header("B. Autodiff overhead — ∇L(θ) for frequency-track MSE loss")
    print(f"  Loss = mean((f_mkn(θ) - f_obs)²)  with {dense} track points")
    print(f"  Single parameter set: M={theta_ref[0]:.2e}  μ={theta_ref[1]:.1f}  "
          f"a={theta_ref[2]:.3f}")

    if not args.skip_autodiff:
        autodiff_results = bench_autodiff(loss_fn, theta_ref, nw, nr)

        print()
        print("  Overhead summary:")
        print(f"    Reverse-mode (adjoint):  {autodiff_results['overhead_rev']:.2f}× forward eval")
        print(f"    Forward-mode (5 JVPs):   {autodiff_results['overhead_fwd']:.2f}× forward eval")
        print(f"  → For gradient descent, prefer reverse-mode (value_and_grad).")
        print(f"  → For Fisher matrix computation, prefer forward-mode (jacfwd).")
    else:
        print("  [skipped]")
        autodiff_results = {}

    # ------------------------------------------------------------------
    # C. Fisher matrix — jacfwd of frequency track
    # ------------------------------------------------------------------
    print_header("C. Fisher matrix  F_ij = Σ_t (∂f/∂θ_i)(∂f/∂θ_j) / σ_f²")
    print(f"  σ_f = {args.sigma_f:.1e} Hz,  {dense} track points")

    fisher_single_result = {}
    fisher_batch_result  = []
    if not args.skip_fisher:
        print("\n  [Single parameter set]")
        fisher_single_result = bench_fisher_single(
            freq_of_theta, theta_ref_arr, dense, args.sigma_f, nw, nr
        )

        print("\n  [Batched: vmap(jacfwd) over N parameter sets]")
        batch_sizes_C = [1, 4, 16, 64, 256]
        fisher_batch_result = bench_fisher_batched(
            freq_of_theta, grid, batch_sizes_C, args.sigma_f, nw, nr
        )

        rows_C = [
            (str(r["N"]),
             f"{r['mean_s']*1e3:.2f} ± {r['std_s']*1e3:.2f}",
             f"{r['throughput']:.1f}",
             f"{r['mem_mib']:.0f}")
            for r in fisher_batch_result
        ]
        print()
        print_table(rows_C,
                    ["N_batch", "time [ms]", "Fisher/s", "mem_peak [MiB]"],
                    [10, 22, 12, 16])
    else:
        print("  [skipped]")

    # ------------------------------------------------------------------
    # D. Multi-start gradient descent — vmap(value_and_grad)
    # ------------------------------------------------------------------
    print_header("D. Multi-start gradient descent  — vmap(value_and_grad(L))")
    print(f"  N starting points drawn from parameter grid.")
    print(f"  Each call returns N (loss, grad) pairs — one GPU kernel.")

    multistart_results = []
    if not args.skip_multistart:
        batch_sizes_D = [1, 4, 16, 64, 256]
        multistart_results = bench_multistart_grad(
            loss_of_theta, theta_ref_arr, grid, batch_sizes_D, nw, nr
        )

        rows_D = [
            (str(r["N"]),
             f"{r['mean_s']*1e3:.2f} ± {r['std_s']*1e3:.2f}",
             f"{r['throughput']:.1f}")
            for r in multistart_results
        ]
        print()
        print_table(rows_D,
                    ["N_starts", "time [ms]", "grad-evals/s"],
                    [12, 22, 16])
    else:
        print("  [skipped]")

    # ------------------------------------------------------------------
    # E. Mode identification — vmap over candidate (m, k, n) combinations
    # ------------------------------------------------------------------
    print_header("E. Mode identification  — vmap over candidate (m, k, n) modes")
    print(f"  Fixed θ = θ_ref.  Evaluate MSE loss for each candidate mode.")
    print(f"  Ground-truth mode: (m={args.m_mode}, k={args.k_mode}, n={args.n_mode})")

    if not args.skip_mode_id:
        mode_id_result = bench_mode_identification(
            traj, theta_ref, f_obs, T, dense, nw, nr
        )
        print(f"  Self-consistency: ground-truth mode recovered = "
              f"{mode_id_result['best_mode'] == (args.m_mode, args.k_mode, args.n_mode)}")
    else:
        print("  [skipped]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_header("Summary")
    print(f"  T = {T} yr,  dense_steps = {dense},  mode (m,k,n) = "
          f"({args.m_mode},{args.k_mode},{args.n_mode}),  σ_f = {args.sigma_f:.0e} Hz\n")

    if traj_results:
        peak = max(r["throughput"] for r in traj_results)
        print(f"  A. Peak vmap trajectory throughput:   {peak:.1f} traj/s")
    if autodiff_results:
        print(f"  B. value_and_grad overhead:           {autodiff_results['overhead_rev']:.2f}× forward eval")
        print(f"     jacfwd overhead:                   {autodiff_results['overhead_fwd']:.2f}× forward eval")
    if fisher_single_result:
        print(f"  C. Single Fisher matrix (5×5):        {fisher_single_result['mean_ms']:.2f} ms")
    if fisher_batch_result:
        peak_F = max(r["throughput"] for r in fisher_batch_result)
        print(f"     Peak batched Fisher throughput:    {peak_F:.1f} Fisher/s")
    if multistart_results:
        peak_G = max(r["throughput"] for r in multistart_results)
        print(f"  D. Peak multi-start grad throughput:  {peak_G:.1f} grad-evals/s")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""fewtrax quickstart example.

This script demonstrates the complete workflow for generating an EMRI
gravitational waveform with fewtrax:

1. Load the FEW data files
2. Generate the waveform (h+, h×)
3. Plot the time-domain waveform and frequency-domain spectrum
4. Compute harmonic frequency tracks

Run this script with the FEW data directory as an argument:

    python quickstart.py /path/to/few/data

or set the FEW_DATA_DIR environment variable.
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Activate 64-bit floats for accuracy
jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FEW_DATA_DIR")

if DATA_DIR is None:
    print("Usage: python quickstart.py /path/to/few/data")
    print("  or set the FEW_DATA_DIR environment variable.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build the waveform generator
# ─────────────────────────────────────────────────────────────────────────────

from fewtrax import KerrEccentricEquatorialWaveform

print("Initialising fewtrax waveform generator …")
wf = KerrEccentricEquatorialWaveform(
    data_dir=DATA_DIR,
    mode_selection_threshold=1e-5,   # relative power threshold for mode selection
    dense_steps=100,                 # number of sparse trajectory points
)
print(f"  ✓ Loaded flux and amplitude data from {DATA_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EMRI source parameters
# ─────────────────────────────────────────────────────────────────────────────

params = dict(
    M=1e6,          # primary BH mass   [M_sun]
    mu=10.0,        # secondary mass     [M_sun]
    a=0.3,          # dimensionless spin
    p0=10.0,        # initial semilatus rectum  [M]
    e0=0.4,         # initial eccentricity
    x0=1.0,         # prograde equatorial orbit
    dist=1.0,       # luminosity distance  [Gpc]
    qS=0.2,         # sky polar angle      [rad]
    phiS=0.2,       # sky azimuthal angle  [rad]
    qK=0.8,         # BH spin polar angle  [rad]
    phiK=0.8,       # BH spin azimuthal    [rad]
    Phi_phi0=1.0,   # initial azimuthal phase  [rad]
    Phi_theta0=2.0, # initial polar phase      [rad]
    Phi_r0=3.0,     # initial radial phase     [rad]
    T=0.1,          # observation time  [years]
    dt=10.0,        # sampling interval [s]
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate the waveform
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating waveform …")
hp, hx = wf(**params)

N = hp.shape[0]
dt = params["dt"]
T_s = N * dt
t = np.arange(N) * dt

print(f"  ✓ Generated {N} samples spanning {T_s/3600:.2f} hours")
print(f"  |h+|_max = {float(jnp.max(jnp.abs(hp))):.3e}")
print(f"  |h×|_max = {float(jnp.max(jnp.abs(hx))):.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Frequency-domain representation
# ─────────────────────────────────────────────────────────────────────────────

from fewtrax.summation.modes import to_frequency_domain

freqs, h_tilde = to_frequency_domain(hp + 1j * hx, dt=dt)
print(f"\nFrequency domain:")
f_peak = float(freqs[jnp.argmax(jnp.abs(h_tilde))])
print(f"  Peak frequency: {f_peak*1000:.3f} mHz")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Harmonic frequency tracks
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing frequency tracks for dominant modes …")
tracks = {}
for (l, m, k, n) in [(2, 2, 0, 1), (2, 2, 0, 2), (3, 2, 0, 1)]:
    t_track, f_track = wf.get_harmonic_track(
        l=l, m=m, k=k, n=n,
        M=params["M"], mu=params["mu"],
        a=params["a"], p0=params["p0"],
        e0=params["e0"], T=params["T"],
    )
    tracks[(l, m, k, n)] = (t_track, f_track)
    f_arr = np.asarray(f_track)
    valid = np.isfinite(f_arr)
    if valid.any():
        print(f"  Mode ({l},{m},{k},{n}):  "
              f"f = {1000*float(f_arr[valid][0]):.3f} → "
              f"{1000*float(f_arr[valid][-1]):.3f} mHz")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Automatic differentiation example
# ─────────────────────────────────────────────────────────────────────────────

print("\nAutomatic differentiation demo …")

from fewtrax.data import load_flux_data
from fewtrax.trajectory import EMRIInspiral

flux_data = wf._flux_data
traj = EMRIInspiral(flux_data, a=params["a"])


def phase_at_end(p0):
    """Total azimuthal phase accumulated during the inspiral."""
    _, _, _, Phi_phi, _, _ = traj(
        p0=p0, e0=params["e0"], T=0.05,
        M=params["M"], mu=params["mu"],
        dense_steps=20,
    )
    valid = jnp.isfinite(Phi_phi)
    return jnp.sum(jnp.where(valid, Phi_phi, 0.0))


grad_fn = jax.grad(phase_at_end)
dPhi_dp0 = grad_fn(jnp.float64(params["p0"]))
print(f"  dΦ_φ/dp₀ = {float(dPhi_dp0):.4f} rad/M")

# ─────────────────────────────────────────────────────────────────────────────
# 7. (Optional) Plot
# ─────────────────────────────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Time domain
    t_days = t / 86400.0
    axes[0].plot(t_days, np.asarray(hp) * 1e20, label=r"$h_+$", lw=0.8)
    axes[0].plot(t_days, np.asarray(hx) * 1e20, label=r"$h_\times$", lw=0.8, alpha=0.7)
    axes[0].set_xlabel("Time [days]")
    axes[0].set_ylabel(r"$h \times 10^{20}$")
    axes[0].set_title(
        fr"EMRI waveform: $M={params['M']:.0e}M_\odot$, "
        fr"$\mu={params['mu']}M_\odot$, $a={params['a']}$, "
        fr"$p_0={params['p0']}$, $e_0={params['e0']}$"
    )
    axes[0].legend()

    # Frequency domain
    f_mHz = np.asarray(freqs) * 1000
    h_tilde_np = np.asarray(jnp.abs(h_tilde))
    axes[1].semilogy(f_mHz, h_tilde_np)
    axes[1].set_xlabel("Frequency [mHz]")
    axes[1].set_ylabel(r"$|\tilde{h}|$ [strain/Hz]")
    axes[1].set_title("Frequency-domain strain")
    axes[1].set_xlim([0, 20])

    # Harmonic tracks
    colours = ["C0", "C1", "C2"]
    for i, ((l, m, k, n), (t_track, f_track)) in enumerate(tracks.items()):
        t_d = np.asarray(t_track) / 86400.0
        f_mHz_track = np.asarray(f_track) * 1000
        valid = np.isfinite(f_mHz_track)
        axes[2].plot(
            t_d[valid], f_mHz_track[valid],
            colour=colours[i],
            label=fr"$(\ell,m,k,n) = ({l},{m},{k},{n})$",
        )
    axes[2].set_xlabel("Time [days]")
    axes[2].set_ylabel("Instantaneous frequency [mHz]")
    axes[2].set_title("Harmonic frequency tracks")
    axes[2].legend()

    plt.tight_layout()
    out_path = "fewtrax_quickstart.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    plt.show()

except ImportError:
    print("\n(matplotlib not installed; skipping plots)")

print("\nDone.")

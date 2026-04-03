## Tutorial

### 1. Generating a waveform and its frequency-domain representation

```python
import numpy as np
import jax.numpy as jnp
from fewtrax.summation.modes import to_frequency_domain

hp, hx = wf(**params)

# Time axis
N = hp.shape[0]
t = np.arange(N) * params["dt"]          # seconds

# Frequency domain
freqs, h_tilde = to_frequency_domain(hp + 1j * hx, dt=params["dt"])
f_peak = float(freqs[jnp.argmax(jnp.abs(h_tilde))])
print(f"Peak frequency: {f_peak * 1e3:.3f} mHz")
```

### 2. Harmonic frequency tracks

Each EMRI mode `(l, m, k, n)` sweeps in frequency as the orbit decays.
`get_harmonic_track` returns the instantaneous frequency along the inspiral:

```python
t_track, f_track = wf.get_harmonic_track(
    l=2, m=2, k=0, n=1,
    M=params["M"], mu=params["mu"],
    a=params["a"], p0=params["p0"],
    e0=params["e0"], T=params["T"],
)
```

Typical dominant modes to inspect: `(2,2,0,1)`, `(2,2,0,2)`, `(3,2,0,1)`.

### 3. Automatic differentiation

The trajectory and mode summation are fully differentiable. Below, we compute
the gradient of the total accumulated azimuthal phase with respect to the
initial semilatus rectum `p0`:

```python
import jax
from fewtrax.trajectory import EMRIInspiral

traj = EMRIInspiral(wf._flux_data, a=params["a"])

def phase_at_end(p0):
    _, _, _, Phi_phi, _, _ = traj(
        p0=p0, e0=params["e0"], T=0.05,
        M=params["M"], mu=params["mu"],
        dense_steps=20,
    )
    valid = jnp.isfinite(Phi_phi)
    return jnp.sum(jnp.where(valid, Phi_phi, 0.0))

grad_fn = jax.grad(phase_at_end)
dPhi_dp0 = grad_fn(jnp.float64(params["p0"]))
print(f"dΦ_φ/dp₀ = {float(dPhi_dp0):.4f} rad/M")
```

### 4. Batch evaluation with `vmap`

```python
import jax
import jax.numpy as jnp

p0_values = jnp.linspace(8.0, 12.0, 8)

# vmap over p0; all other params are fixed
def single_waveform(p0):
    return wf(**{**params, "p0": p0})

hp_batch, hx_batch = jax.vmap(single_waveform)(p0_values)
print(hp_batch.shape)  # (8, N_samples)
```

### 5. Accessing sparse trajectory data

For diagnostics or downstream analysis you can retrieve the sparse orbital
trajectory before waveform synthesis:

```python
result = wf.generate_sparse(**params)
# result keys: t, p, e, Phi_phi, Phi_theta, Phi_r, amplitudes, modes
print(result["p"])   # (dense_steps,) array of semilatus rectum values
```

The full runnable script including plots is in [`examples/quickstart.py`](examples/quickstart.py).

## API overview

| Class / function | Description |
|---|---|
| `KerrEccentricEquatorialWaveform` | Top-level waveform generator |
| `EMRIInspiral` | ODE-based adiabatic inspiral integrator |
| `AmplitudeInterpolator` | B-spline Teukolsky mode amplitude evaluator |
| `ModeSum` | Coherent harmonic mode summation |
| `load_flux_data` / `load_amplitude_data` | HDF5 data loaders |
| `to_frequency_domain` | FFT helper (returns positive-frequency half) |
| `spin_weighted_spherical_harmonic` | ₋₂Y_ℓm at arbitrary angles |
| `get_separatrix` | Separatrix p(a, e) for Kerr equatorial orbits |
| `get_fundamental_frequencies` | Ω_φ, Ω_θ, Ω_r along the inspiral |

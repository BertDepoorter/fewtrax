# fewtrax

JAX implementation of the **KerrEccentricEquatorial** EMRI waveform model from [FastEMRIWaveforms (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms), with full support for JIT compilation, automatic differentiation, and batched evaluation via `vmap`.

## Features

- **JIT compilation** – waveform generation via `jax.jit` and `equinox.filter_jit`
- **Automatic differentiation** – gradients of any output w.r.t. source parameters
- **Vectorisation** – batch over populations with `jax.vmap`
- **ODE trajectory** – adiabatic inspiral via [diffrax](https://github.com/patrick-kidger/diffrax) (Tsit5, adaptive step-size)
- **Mode amplitude interpolation** – Teukolsky mode amplitudes from FEW HDF5 files
- **GPU-ready** – install the `gpu` extra to run on CUDA hardware

## Requirements

- Python ≥ 3.10
- FEW HDF5 data files (`KerrEccEqFluxData.h5` and `ZNAmps_l10_m10_n55_DS2Outer.h5`), available from the [FEW data repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)

## Installation

```bash
# CPU
pip install fewtrax

# GPU (CUDA 12)
pip install "fewtrax[gpu]"

# Development (tests + notebooks)
pip install "fewtrax[dev]"
```

Or from source:

```bash
git clone https://github.com/<your-org>/fewtrax
cd fewtrax
pip install -e ".[dev]"
```

## Data setup

fewtrax reads the FEW HDF5 data files. Point to them in one of three ways (checked in this order):

1. Pass `data_dir=` to `KerrEccentricEquatorialWaveform`
2. Set the environment variable `FEW_DATA_DIR`
3. Place the files in `~/.fewtrax/data/`

If `FastEMRIWaveforms` is installed, fewtrax will also try its internal file-manager cache.

## Quickstart

```python
import jax
jax.config.update("jax_enable_x64", True)  # required for numerical accuracy

from fewtrax import KerrEccentricEquatorialWaveform

wf = KerrEccentricEquatorialWaveform(
    data_dir="/path/to/few/data",
    mode_selection_threshold=1e-5,  # keep modes with relative power > threshold
    dense_steps=100,                # trajectory resolution
)

params = dict(
    M=1e6,          # primary BH mass        [M_sun]
    mu=10.0,        # secondary mass          [M_sun]
    a=0.3,          # dimensionless spin
    p0=10.0,        # initial semilatus rectum [M]
    e0=0.4,         # initial eccentricity
    x0=1.0,         # prograde equatorial orbit (must be ±1)
    dist=1.0,       # luminosity distance      [Gpc]
    qS=0.2,         # sky polar angle          [rad]
    phiS=0.2,       # sky azimuthal angle      [rad]
    qK=0.8,         # BH spin polar angle      [rad]
    phiK=0.8,       # BH spin azimuthal        [rad]
    Phi_phi0=1.0,   # initial azimuthal phase  [rad]
    Phi_theta0=2.0, # initial polar phase      [rad]
    Phi_r0=3.0,     # initial radial phase     [rad]
    T=0.1,          # observation time         [years]
    dt=10.0,        # sampling interval        [s]
)

hp, hx = wf(**params)
print(hp.shape)   # (N_samples,)
```

## Tutorials

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

## Running tests

```bash
pytest                      # unit tests
pytest --cov=fewtrax        # with coverage report
```

To run the comparison suite against FastEMRIWaveforms:

```bash
pip install "fewtrax[compare]"
pytest tests/test_compare_few.py
```

## Building documentation

The source code uses NumPy-style docstrings throughout. The steps below set up
a Sphinx-based site and deploy it to **GitHub Pages**.

### 1. Install Sphinx and extensions

```bash
pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme myst-parser
```

### 2. Initialise the docs directory

```bash
sphinx-quickstart docs \
  --sep \
  --project fewtrax \
  --author "<your name>" \
  --release 0.1.0 \
  --language en
```

### 3. Configure `docs/source/conf.py`

Add the following to the generated `conf.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath("../../src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # NumPy & Google docstrings
    "sphinx.ext.mathjax",           # LaTeX math in docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",                  # Markdown source files
]

html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
```

### 4. Write `docs/source/index.rst`

```rst
fewtrax
=======

.. toctree::
   :maxdepth: 2

   api
   tutorials

.. automodule:: fewtrax
   :members:
```

Create `docs/source/api.rst`:

```rst
API Reference
=============

.. automodule:: fewtrax.waveform.kerr
   :members:

.. automodule:: fewtrax.trajectory.inspiral
   :members:

.. automodule:: fewtrax.amplitude.interp
   :members:

.. automodule:: fewtrax.summation.modes
   :members:

.. automodule:: fewtrax.data.loader
   :members:

.. automodule:: fewtrax.utils.geodesic
   :members:

.. automodule:: fewtrax.utils.harmonics
   :members:
```

### 5. Build locally

```bash
cd docs
make html
open build/html/index.html
```

### 6. Deploy to GitHub Pages via GitHub Actions

Create `.github/workflows/docs.yml`:

```yaml
name: Build and deploy docs

on:
  push:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install package and doc dependencies
        run: |
          pip install -e .
          pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme myst-parser

      - name: Build HTML docs
        run: |
          cd docs
          make html

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/build/html
```

In your repository settings → **Pages**, set the source branch to `gh-pages`.
The docs will be available at `https://<your-org>.github.io/fewtrax/`.

### Alternative: ReadTheDocs

Add a `.readthedocs.yaml` at the repository root:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs/source/conf.py

python:
  install:
    - method: pip
      path: .
    - requirements: docs/requirements.txt
```

Create `docs/requirements.txt`:

```
sphinx
sphinx-autodoc-typehints
sphinx-rtd-theme
myst-parser
```

Then connect the repository on [readthedocs.org](https://readthedocs.org) and
the docs will build automatically on every push to `main`.

## Citation

If you use fewtrax in your research, please cite the original FEW paper:

```bibtex
@article{Katz2021,
  author  = {Katz, Michael L. and Chua, Alvin J. K. and Speri, Lorenzo and
             Warburton, Niels and Hughes, Scott A.},
  title   = {Fast extreme-mass-ratio-inspiral waveforms: New tools for
             millihertz gravitational-wave data analysis},
  journal = {Phys. Rev. D},
  volume  = {104},
  pages   = {064047},
  year    = {2021},
  doi     = {10.1103/PhysRevD.104.064047},
}
```

## License

MIT © 2024 – see [LICENSE](LICENSE).

# fewtrax
[![Doc badge](https://img.shields.io/badge/Docs-master-brightgreen)](https://bertdepoorter.github.io/fewtrax)
JAX implementation of the **KerrEccentricEquatorial** EMRI waveform model from [FastEMRIWaveforms (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms), with full support for JIT compilation, automatic differentiation, and batched evaluation via `vmap`.

## Features

- **JIT compilation** – waveform generation via `jax.jit` and `equinox.filter_jit`
- **Automatic differentiation** – gradients of any output w.r.t. source parameters
- **Vectorisation** – batch over populations with `jax.vmap`
- **ODE trajectory** – adiabatic inspiral via [diffrax](https://github.com/patrick-kidger/diffrax) (Dopri8, adaptive step-size through PIDcontroller)
- **Mode amplitude interpolation** – Teukolsky mode amplitudes from FEW HDF5 files
- **GPU-ready** – install the `gpu` extra to run on CUDA hardware. No manual kernels required, only smart use of JIT and vmap.

## Requirements

- Python ≥ 3.10
- FEW HDF5 data files (`KerrEccEqFluxData.h5` and `ZNAmps_l10_m10_n55_DS2Outer.h5`), available from the [FEW data repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)

## Installation
[NOT YET PUBLISHED] Install from PyPI:
```bash
# CPU
pip install fewtrax

# GPU (CUDA 12)
pip install "fewtrax[gpu]"

# Development (tests + notebooks)
pip install "fewtrax[dev]"
```
The package has not yet been published to PyPI. To install and run locally, clone the repository and install in editable mode for now:
Or from source:

```bash
git clone https://github.com/BertDepoorter/fewtrax
cd fewtrax
pip install -e ".[dev]"
```

## Data setup

fewtrax reads the FEW KerrEccentricEquatorial data files. Point to them in one of three ways (checked in this order):

1. Pass `data_dir=` to `KerrEccentricEquatorialWaveform`
2. Set the environment variable `FEW_DATA_DIR` in a .env folder
3. Place the files in `~/.fewtrax/data/`. 

If `FastEMRIWaveforms` is installed (default when you install with the `comparison` extra), fewtrax will also try its internal file-manager cache. It is highly adviced to install FEW as well to compare the results of both waveform models. 

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

## Quickstart

```python
import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # required for numerical accuracy

from fewtrax import KerrEccentricEquatorialWaveform
from fewtrax.summation.modes import to_frequency_domain

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

# Time axis
N = hp.shape[0]
t = np.arange(N) * params["dt"]          # seconds

# Frequency domain
freqs, h_tilde = to_frequency_domain(hp + 1j * hx, dt=params["dt"])
f_peak = float(freqs[jnp.argmax(jnp.abs(h_tilde))])
print(f"Peak frequency: {f_peak * 1e3:.3f} mHz")
```
A more extensive introduction is available [here](./TUTORIAL.md). 

## Building documentation

The source code uses NumPy-style docstrings throughout. You can build the documentation locally or consult [https://bertdepoorter.github.io/fewtrax/](https://bertdepoorter.github.io/fewtrax/)

### Building local docs with Sphinx 
If you install with the tag `[docs]`, these packages are included and the following command is obsolete. 
```bash
pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme myst-parser
```
Build the docs locally with:
```bash
cd docs
make html
open build/html/index.html
```
The HTML file can be opened in a web browser where you can view the docs. 


## Citation

If you use fewtrax in your research, please cite the original FEW papers:

```bibtex
@article{Chapman-bird2025,
  title = {Efficient waveforms for asymmetric-mass eccentric equatorial inspirals into rapidly spinning black holes},
  author = {Chapman-Bird, Christian E. A. and Speri, Lorenzo and Nasipak, Zachary and Burke, Ollie and Katz, Michael L. and Santini, Alessandro and Kejriwal, Shubham and Lynch, Philip and Mathews, Josh and Khalvati, Hassan and Thompson, Jonathan E. and Isoyama, Soichiro and Hughes, Scott A. and Warburton, Niels and Chua, Alvin J. K. and Pigou, Maxime},
  journal = {Phys. Rev. D},
  volume = {112},
  issue = {10},
  pages = {104023},
  numpages = {59},
  year = {2025},
  month = {Nov},
  publisher = {American Physical Society},
  doi = {10.1103/scbp-75pf},
  url = {https://link.aps.org/doi/10.1103/scbp-75pf}
}

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

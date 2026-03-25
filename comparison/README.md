# comparison/

Scripts that compare **fewtrax** against the reference
[FastEMRIWaveforms (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)
implementation.

## Setup

Install the comparison dependencies:

```bash
pip install "fewtrax[compare]"   # adds FastEMRIWaveforms + python-dotenv + matplotlib
```

Create a `.env` file in the repository root (already git-ignored):

```
FEW_DATA_DIR=/path/to/few/data
```

All scripts load this variable automatically via `python-dotenv`.
You can also pass the path as a positional argument or export `FEW_DATA_DIR`
in your shell.

## Scripts

| Script | What it measures |
|---|---|
| `compare_phase.py` | Dedicated 2-year phase-evolution comparison; decomposes error into frequency vs trajectory |
| `compare_trajectory.py` | p(t), e(t), phase accuracy vs FEW; saves trajectory + accumulated-phase figures |
| `compare_waveform.py` | h+, h× overlap / mismatch vs FEW; optional all-modes pass |
| `compare_amplitude.py` | Teukolsky amplitude accuracy (fewtrax `bisplev` vs FEW multispline) |
| `compare_summation.py` | Mode summation speed (JAX vs numpy, mode/time scaling sweeps, interpolated vs direct) |
| `compare_tf.py` | Sparse WDM time-frequency tracks; analytical track map + optional pywavelet WDM transform |
| `benchmark_gpu.py` | Full GPU runtime comparison on A100 |

`utils.py` contains shared helpers (data-dir discovery, metrics, timing, table printing).

## Running

```bash
cd comparison

# Phase evolution over 2 years (default params: a=0.3, p0=10, e0=0.4)
# Produces phase_comparison_2p0yr.png in ./figures
python compare_phase.py --T 2.0 --plot-dir ./figures

# Force the full frequency diagnostic even if dephasing < 1 rad
python compare_phase.py --T 2.0 --force-diagnostic --plot-dir ./figures

# Custom parameters
python compare_phase.py --T 2.0 --a 0.7 --p0 9.0 --e0 0.3

# Trajectory accuracy (all parameter sets, with plots)
# Produces trajectory_<label>.png and trajectory_<label>_phases.png per parameter set
python compare_trajectory.py --plot --plot-dir ./figures

# Waveform accuracy (quick single-params check)
python compare_waveform.py --single

# Waveform accuracy (full suite, custom mode threshold)
python compare_waveform.py --threshold 1e-4 --plot --plot-dir ./figures

# Waveform accuracy with all 6993 modes to separate dephasing from amplitude error
python compare_waveform.py --all-modes

# Amplitude accuracy (fewtrax bisplev vs FEW multispline, spin a=0.5)
python compare_amplitude.py --spin 0.5 --plot --plot-dir ./figures

# Summation speed (no FEW data needed for scaling sweeps)
python compare_summation.py --skip-few

# Summation speed including full end-to-end comparison
python compare_summation.py

# Analytical WDM TF tracks (no waveform generation; fast)
python compare_tf.py --T 1.0 --no-plot

# With figures saved to ./figures
python compare_tf.py --T 1.0 --plot --plot-dir ./figures

# Full WDM transform of dominant mode (requires amplitude data)
python compare_tf.py --T 1.0 --wdm --plot --plot-dir ./figures

# Custom WDM grid (more time bins = finer time resolution per pixel)
python compare_tf.py --Nf 128 --Nt 8192 --T 2.0

# Full GPU benchmark on A100
python benchmark_gpu.py --n-repeat 10

# GPU benchmark without vmap batch test
python benchmark_gpu.py --skip-vmap
```

## GPU requirements (`benchmark_gpu.py`)

```bash
pip install "fewtrax[gpu,compare]"   # jax[cuda12] + FEW
python benchmark_gpu.py
```

The script reports:
- **A** – Single-waveform component timings (trajectory, summation, end-to-end)
- **B** – Observation-time scaling (T = 0.05 … 1 yr)
- **C** – Sample-rate scaling (dt = 1 … 60 s)
- **D** – Mode-summation GPU stress test (up to 8000 modes × 2000 time steps)
- **E** – `jax.vmap` batch throughput (up to N = 1024 simultaneous waveforms); skip with `--skip-vmap`

## Script details

### `compare_phase.py`

Dedicated phase-evolution comparison for long observation windows (default: T = 2 yr).

**Key design choices:**

FEW's `EMRIInspiral` uses an adaptive ODE solver whose natural step size for a 2-year EMRI
inspiral is ~25–30 steps (one step per ~30 days).  Naively interpolating these 25 sparse
phase values to a fine grid introduces huge artefacts (hundreds of radians) because the
phase changes by ~17 000 rad between consecutive steps.  This script avoids that by:

1. Running fewtrax with `dense_steps=2000` (one point per ~8.7 hours).
2. Interpolating *fewtrax* (dense) to FEW's sparse times — not the other way around.
3. Reporting dephasing only at the 25 FEW ODE-step times where both codes are accurate.

**Results for default parameters** (M=1e6, mu=10, a=0.3, p0=10, e0=0.4, T=2yr):

- max |ΔΦ_φ| = **0.067 rad** at FEW ODE steps  (threshold: 1 rad)
- relative accuracy = 1.5 × 10⁻⁷  (max|ΔΦ_φ| / Φ_φ_total)
- fewtrax Ω_φ formula validated to machine precision against the Schwarzschild analytic value

**Diagnostic flags:**

| Flag | Effect |
|---|---|
| `--T` | Observation time [years] (default 2.0) |
| `--force-diagnostic` | Run the frequency/trajectory decomposition even if dephasing < 1 rad |
| `--no-diagnostic` | Skip all diagnostics |
| `--no-plot` | Skip figure output |

### `compare_trajectory.py`

Runs the FEW `EMRIInspiral` integrator and fewtrax side-by-side over the full
`PARAM_SUITE` and reports p(t) / e(t) RMS relative error and mean/max dephasing
for all three orbital phases (Φ_φ, Φ_θ, Φ_r).  With `--plot` two figures are
saved per parameter set:

- `trajectory_<label>.png` – p(t), e(t) and dephasing panels
- `trajectory_<label>_phases.png` – accumulated phase and dephasing for all three angles

### `compare_waveform.py`

Compares the full h+, h× strain.  Key flags:

| Flag | Default | Effect |
|---|---|---|
| `--threshold` | `1e-5` | Mode selection threshold for fewtrax |
| `--single` | off | Run only the default parameter set (fast check) |
| `--all-modes` | off | Second pass with all 6993 Teukolsky modes (chunked to avoid OOM) to isolate dephasing from amplitude interpolation error |

### `compare_amplitude.py`

Sweeps a (p, e) grid at a fixed spin and compares Teukolsky mode amplitudes
between fewtrax (`scipy.interpolate.bisplev`) and FEW's C++ multispline backend.
Reports per-mode RMS relative error on |A| and the complex phase.  Key flags:

| Flag | Default | Effect |
|---|---|---|
| `--spin` | `0.3` | BH spin parameter a |
| `--p-min` / `--p-max` | `7.0` / `15.0` | Semi-latus rectum range |
| `--e-fixed` | `0.3` | Fixed eccentricity for the sweep |
| `--n-grid` | `30` | Number of p grid points |
| `--n-modes-print` | `10` | Modes shown in summary table |

### `compare_tf.py`

Constructs sparse WDM (Wilson-Daubechies-Meyer) time-frequency tracks for the
dominant EMRI harmonic modes.

**Key design:**

Each harmonic `(l,m,k,n)` traces a slowly-chirping frequency track:
```
f_{mkn}(t) = [m Ω_φ(t) + k Ω_θ(t) + n Ω_r(t)] / (2π M_s)
```
Because EMRI signals evolve slowly, each mode occupies only O(Nt) pixels in the
Nf×Nt WDM grid — a fraction 1/Nf of all pixels.

**Two representations:**

1. **Analytical track** — frequency bins computed directly from the trajectory,
   no wavelet transform needed.  Per mode: ~24 kB (Nt=4096).
2. **WDM transform track** (`--wdm`) — pywavelet forward transform of the
   single-mode time series; stores WDM coefficients along the track.
   Per mode: ~56 kB (Nt=4096, with complex64 coefficients).

**Key flags:**

| Flag | Default | Effect |
|---|---|---|
| `--Nf` | `64` | WDM frequency bins (Nyquist = Nf × ΔF) |
| `--Nt` | `4096` | WDM time bins (ΔT = T/Nt ≈ 2 hr for T=1 yr) |
| `--n-modes` | `10` | Number of dominant modes to analyse |
| `--wdm` | off | Also run pywavelet WDM transform for dominant mode |

**Expected results** (default: M=1e6, μ=10, a=0.3, p0=10, e0=0.4, T=1 yr):

- 10 analytical tracks computed in < 0.2 s
- Frequency range: 0.5–5 mHz (dominant modes)
- Each mode spans 2–4 frequency bins per time step (slowly chirping)
- WDM transform of dominant mode: < 2 s; track coefficient array: 32 kB

### `compare_summation.py`

Benchmarks the summation step in isolation.  Runs three sweeps:

1. **Modes scaling** – N_modes ∈ {50, 100, 200, 500, 1000, 2000} at fixed N_t = 500
2. **Time-sample scaling** – N_t ∈ {100, 500, 2000, 5000, 10000} at fixed N_modes = 200
3. **Interpolated vs direct** – sparse→dense upsample overhead for N_sparse ∈ {50, 100, 200, 500}
4. **End-to-end** (optional, requires FEW) – full fewtrax vs FEW waveform timing

Pass `--skip-few` to run sweeps 1–3 without a FEW installation.

## Notes on expected results

- **Overlap** of the full waveform vs FEW should be > 0.90 for all parameter sets.
  Differences arise from the interpax cubic splines used by fewtrax (rather than
  FEW's pre-computed multispline B-spline tables).
- **Trajectory** p(t) RMS relative error is typically < 1 %.
- **Amplitude** RMS relative error (median over modes) is typically < 1 % for
  moderate spins; larger discrepancies appear near domain boundaries where the
  two backends handle edge cases differently.
- **Summation speedup** on an A100 vs a single CPU core is typically
  10–100× depending on mode count and sample count.
- The **amplitude evaluation** step (scipy B-spline, CPU) currently dominates
  the wall-clock time of a single fewtrax waveform; the JAX trajectory and
  summation steps are sub-second even for 1-year observations.

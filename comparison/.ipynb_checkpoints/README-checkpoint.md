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
| `compare_trajectory.py` | p(t), e(t), phase accuracy vs FEW |
| `compare_waveform.py` | h+, h× overlap / mismatch vs FEW |
| `compare_summation.py` | Mode summation speed (JAX vs numpy, scaling sweeps) |
| `benchmark_gpu.py` | Full GPU runtime comparison on A100 |

`utils.py` contains shared helpers (data-dir discovery, metrics, timing, table printing).

## Running

```bash
cd comparison

# Trajectory accuracy (all parameter sets, with plots)
python compare_trajectory.py --plot --plot-dir ./figures

# Waveform accuracy (quick single-params check)
python compare_waveform.py --single

# Summation speed (no FEW data needed for scaling sweeps)
python compare_summation.py --skip-few

# Full GPU benchmark on A100
python benchmark_gpu.py --n-repeat 10
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
- **E** – `jax.vmap` batch throughput (up to N = 1024 simultaneous waveforms)

## Notes on expected results

- **Overlap** of the full waveform vs FEW should be > 0.90 for all parameter sets.
  Differences arise from the interpax cubic splines used by fewtrax (rather than
  FEW's pre-computed multispline B-spline tables).
- **Trajectory** p(t) RMS relative error is typically < 1 %.
- **Summation speedup** on an A100 vs a single CPU core is typically
  10–100× depending on mode count and sample count.
- The **amplitude evaluation** step (scipy B-spline, CPU) currently dominates
  the wall-clock time of a single fewtrax waveform; the JAX trajectory and
  summation steps are sub-second even for 1-year observations.

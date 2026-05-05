========================================================================
  Device information
========================================================================

  JAX version : 0.9.2
  0: NVIDIA A100 80GB PCIe  platform=gpu
  GPU free memory (nvidia-smi): 19854 MiB

  FEW data directory: /data/leuven/367/vsc36785/LISA/FastEMRIWaveforms/data

  Loading fewtrax … done

  Building random parameter grid  (N=16384, seed=42) …  done
  p0 range : [5.43, 16.89] M
  e0 range : [0.050, 0.700]
  a  range : [0.050,  0.900]
  M  range : [5.00e+05, 5.00e+06] Msun
  mu range : [5.0, 50.0] Msun

  Reference θ: M=6.500e+05  μ=21.85  a=0.216  p₀=12.606  e₀=0.591
  Frequency track: mode (m=2, k=0, n=0)
  σ_f = 1.0e-05 Hz  (STFT noise model)

  Generating synthetic f_obs from reference parameters … done  (f̄ = 1.486 mHz)

========================================================================
  A. vmap trajectory throughput  (EMRIInspiral)
========================================================================
  T=0.5 yr,  dense_steps=100,  n_warmup=2,  n_repeat=5
  N=    1:    49.48 ±  0.13 ms  (   20.2 traj/s)  mem_peak=832 MiB
  N=   64:   249.40 ±  0.34 ms  (  256.6 traj/s)  mem_peak=832 MiB
  N=  256:   263.00 ±  0.59 ms  (  973.4 traj/s)  mem_peak=832 MiB
  N= 1024:   275.59 ±  0.16 ms  ( 3715.7 traj/s)  mem_peak=832 MiB
  N= 4096:   292.81 ±  0.16 ms  (13988.7 traj/s)  mem_peak=832 MiB
  N=16384:   385.95 ±  0.47 ms  (42451.4 traj/s)  mem_peak=840 MiB

N_batch     time [ms]               traj/s        mem_peak [MiB]  
----------  ----------------------  ------------  ----------------
1           49.48 ± 0.13            20.2          832             
64          249.40 ± 0.34           256.6         832             
256         263.00 ± 0.59           973.4         832             
1024        275.59 ± 0.16           3715.7        832             
4096        292.81 ± 0.16           13988.7       832             
16384       385.95 ± 0.47           42451.4       840             

========================================================================
  B. Autodiff overhead — ∇L(θ) for frequency-track MSE loss
========================================================================
  Loss = mean((f_mkn(θ) - f_obs)²)  with 100 track points
  Single parameter set: M=6.50e+05  μ=21.8  a=0.216
  Forward eval:                56.66 ±  0.14 ms
  value_and_grad (reverse):   225.76 ±  0.09 ms  (3.98× fwd)
  jacfwd (forward, 5 JVPs):    79.19 ±  0.05 ms  (1.40× fwd)

  Overhead summary:
    Reverse-mode (adjoint):  3.98× forward eval
    Forward-mode (5 JVPs):   1.40× forward eval
  → For gradient descent, prefer reverse-mode (value_and_grad).
  → For Fisher matrix computation, prefer forward-mode (jacfwd).

========================================================================
  C. Fisher matrix  F_ij = Σ_t (∂f/∂θ_i)(∂f/∂θ_j) / σ_f²
========================================================================
  σ_f = 1.0e-05 Hz,  100 track points

  [Single parameter set]
  Single Fisher (5×5, jacfwd):   78.71 ±  0.04 ms
    Eigenvalues: [5.76678318e-19 5.02034365e-07 1.39571459e+00 1.06446563e+04
 1.06561891e+07]
    Condition number: 1.85e+25
    Cramér-Rao σ(M)=3.219e+02 M_sun  σ(a)=4.263e-03

  [Batched: vmap(jacfwd) over N parameter sets]
  N=    1:    67.68 ±  0.06 ms  (   14.8 Fisher/s)  mem_peak=1312 MiB
  N=   64:   483.79 ±  0.10 ms  (  132.3 Fisher/s)  mem_peak=1546 MiB
  N=  256:   532.89 ±  0.14 ms  (  480.4 Fisher/s)  mem_peak=1688 MiB
  N= 1024:   682.07 ±  0.81 ms  ( 1501.3 Fisher/s)  mem_peak=1938 MiB
  N= 4096:  1408.91 ±  0.21 ms  ( 2907.2 Fisher/s)  mem_peak=2800 MiB
  N=16384:  4332.59 ±  0.41 ms  ( 3781.6 Fisher/s)  mem_peak=5419 MiB

N_batch     time [ms]               Fisher/s      mem_peak [MiB]  
----------  ----------------------  ------------  ----------------
1           67.68 ± 0.06            14.8          1312            
64          483.79 ± 0.10           132.3         1546            
256         532.89 ± 0.14           480.4         1688            
1024        682.07 ± 0.81           1501.3        1938            
4096        1408.91 ± 0.21          2907.2        2800            
16384       4332.59 ± 0.41          3781.6        5419            

========================================================================
  D. Multi-start gradient descent  — vmap(value_and_grad(L))
========================================================================
  N starting points drawn from parameter grid.
  Each call returns N (loss, grad) pairs — one GPU kernel.
  N=    1:   196.28 ±  0.52 ms  (    5.1 grad-evals/s)
  N=   64:  1412.43 ±  1.34 ms  (   45.3 grad-evals/s)
  N=  256:  1484.55 ±  0.66 ms  (  172.4 grad-evals/s)
  N= 1024:  1577.74 ±  1.29 ms  (  649.0 grad-evals/s)
  N= 4096:  1724.30 ±  0.62 ms  ( 2375.5 grad-evals/s)

N_starts      time [ms]               grad-evals/s    
------------  ----------------------  ----------------
1             196.28 ± 0.52           5.1             
64            1412.43 ± 1.34          45.3            
256           1484.55 ± 0.66          172.4           
1024          1577.74 ± 1.29          649.0           
4096          1724.30 ± 0.62          2375.5          

========================================================================
  E. Mode identification  — vmap over candidate (m, k, n) modes
========================================================================
  Fixed θ = θ_ref.  Evaluate MSE loss for each candidate mode.
  Ground-truth mode: (m=2, k=0, n=0)
  Candidate modes: 28
  vmap over 28 modes: 0.063 ± 0.005 ms  (441399.5 mode-evals/s)
  Best-fit mode: m=2 k=0 n=0  loss=8.041e-11
  Self-consistency: ground-truth mode recovered = True

========================================================================
  Summary
========================================================================
  T = 0.5 yr,  dense_steps = 100,  mode (m,k,n) = (2,0,0),  σ_f = 1e-05 Hz

  A. Peak vmap trajectory throughput:   42451.4 traj/s
  B. value_and_grad overhead:           3.98× forward eval
     jacfwd overhead:                   1.40× forward eval
  C. Single Fisher matrix (5×5):        78.71 ms
     Peak batched Fisher throughput:    3781.6 Fisher/s
  D. Peak multi-start grad throughput:  2375.5 grad-evals/s

Done.

# CLI Tools

CAST ships four command-line binaries for hardware profiling and benchmarking.
All require `LLVM_CONFIG` to be set (see the README).

## profile_hw — Hardware Profiler

Measures the memory/compute crossover point by sweeping gate kernels across a
range of arithmetic intensities and fitting a two-segment piecewise roofline
model.

### Quick Start

```sh
# Profile CPU (all precisions, auto-detect statevector size)
cargo run --bin profile_hw --release

# Profile CPU + CUDA
cargo run --bin profile_hw --features cuda --release

# Save profiles to JSON for later use
cargo run --bin profile_hw --release -- --save-profiles profiles/

# CUDA only, F64, explicit 28-qubit statevector, 60s budget
cargo run --bin profile_hw --features cuda --release -- \
      --backend cuda --precision f64 -n 28 --budget 60 \
      --save-profiles profiles/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {cpu,cuda,all}` | `all` | Which backend(s) to profile |
| `--precision {f32,f64,all}` | `all` | Which precision(s) to test |
| `-n, --n-qubits <N>` | auto | Statevector size; auto-detected from free memory if omitted |
| `--budget <SECS>` | `30` | Wall-time budget per profile run |
| `--threads <N>` | all cores | CPU thread count (also via `CAST_NUM_THREADS`) |
| `--save-profiles <DIR>` | — | Save each profile as `<backend>_<precision>.json` |

### Output

The profiler prints a progress bar during sweeping and reports:

- **Peak BW** (GiB/s) — measured memory bandwidth
- **Peak Compute** (GFLOPs/s) — measured compute throughput
- **BW slope** — GFLOPs/s per unit arithmetic intensity (memory-bound regime)
- **Crossover AI** — the arithmetic intensity where memory-bound meets compute-bound
- **R²** — goodness of fit for the roofline model

### Choosing Statevector Size

The profiling statevector should match the simulation workload size for
representative cache/TLB behavior.  For density-matrix simulation of n
physical qubits, the statevector has 2n virtual qubits — use `-n 2N`.

If omitted, profile_hw auto-detects available memory and picks the largest
feasible n (clamped to 30).

**Why size matters for profile accuracy:**  The roofline parameters —
especially effective memory bandwidth — depend on working set size because
of cache effects.  A 28-qubit statevector (4 GiB) can partially fit in GPU
L2 cache (modern GPUs have 48-96 MB L2), which raises effective bandwidth
for low-AI gates and shifts the crossover AI downward compared to a
30-qubit statevector (16 GiB) that streams entirely from HBM.

At large scales (30+ SV qubits), the statevector far exceeds any cache and
the roofline stabilises — profiles at 30 and 32 qubits will agree closely.
But in the transition zone (24-30 qubits), cache effects can meaningfully
shift the crossover, so the profile qubit count should match the target
simulation size for best results.

### Saved Profiles

Profiles are saved as JSON and can be loaded by `bench_fusion`,
`bench_ablation`, and `bench_noisy_qft` via `--profile <path>` to skip
re-profiling.

---

## bench_fusion — Fusion Strategy Benchmark

Loads OpenQASM circuit files, applies several fusion strategies, then times
execution on the selected backend.

### Quick Start

```sh
# CPU backend, single circuit
cargo run --bin bench_fusion --release -- \
      --backend cpu examples/journal_examples/qft-cx-30.qasm

# CUDA backend with cached profile
cargo run --bin bench_fusion --features cuda --release -- \
      --backend cuda --profile profiles/cuda_f64.json \
      examples/journal_examples/ala-30.qasm

# All 30-qubit circuits
cargo run --bin bench_fusion --features cuda --release -- \
      --backend cuda examples/journal_examples/*-30*.qasm
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {cpu,cuda}` | `cuda` | Simulation backend |
| `--profile <PATH>` | — | Cached HardwareProfile JSON (skips profiling) |
| `--profile-qubits <N>` | `30` | SV qubits for profiling |
| `--profile-budget <S>` | `20` | Profiling time budget (seconds) |
| `--max-size <N>` | `4` | Max gate size for hw-adaptive fusion |
| `--bench-budget <S>` | `5` | Time budget per benchmark run (seconds) |
| `--force-dense` | off | Disable sparsity-aware codegen (ztol=0) |
| `--fusion-log` | off | Print per-decision fusion log for hw-adaptive |
| Files (positional) | required | One or more `.qasm` files |

### Fusion Configs Compared

| Config | Description |
|--------|-------------|
| `no-fusion` | `size_only(1)` — Phase 1 canonicalization only |
| `default` | `size_only(3)` — fuse up to 3-qubit gates |
| `aggressive` | `size_only(4)` — fuse up to 4-qubit gates |
| `hw-adaptive` | Roofline-guided fusion up to `--max-size` |

### Output

A table per circuit with columns: Config, Gates, Depth, Cold-Start (kernel
compilation time), GPU/Exec time (with ± stddev), and iteration count.

---

## bench_ablation — Dense vs Sparse Kernel Ablation

For each input circuit, applies the selected fusion strategy, then runs the
fused circuit twice — once with sparsity-aware kernels (default) and once
with dense kernels (ztol=0). Reports per-gate sparsity statistics and the
speedup from sparsity exploitation.

### Quick Start

```sh
# CUDA backend, hw-adaptive fusion
cargo run --bin bench_ablation --features cuda --release -- \
      --backend cuda --profile profiles/cuda_f64.json \
      examples/journal_examples/*-30.qasm

# No fusion (measure raw per-gate sparsity benefit)
cargo run --bin bench_ablation --features cuda --release -- \
      --backend cuda --profile profiles/cuda_f64.json \
      --fusion no-fusion examples/journal_examples/mexp-20.qasm
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {cpu,cuda}` | `cuda` | Simulation backend |
| `--fusion {no-fusion,default,aggressive,hw-adaptive}` | `hw-adaptive` | Fusion strategy |
| `--profile <PATH>` | — | Cached HardwareProfile JSON |
| `--profile-qubits <N>` | `30` | SV qubits for profiling |
| `--profile-budget <S>` | `20` | Profiling time budget (seconds) |
| `--max-size <N>` | `4` | Max gate size for hw-adaptive fusion |
| `--bench-budget <S>` | `5` | Time budget per benchmark run (seconds) |
| Files (positional) | required | One or more `.qasm` files |

### Output

A table per circuit with columns: Qubits, Gates, Depth, mean AI,
AI range, Sparse time, Dense time, Speedup (dense/sparse), JIT compile
times for both modes, and JIT ratio.

---

## bench_noisy_qft — Noisy QFT Benchmark

Builds an n-qubit QFT circuit with depolarizing noise, lifts it to a
density-matrix circuit on 2n virtual qubits, then benchmarks execution under
multiple fusion strategies.

### Quick Start

```sh
# Default: 14-qubit QFT, CPU only, 20s budget per config
cargo run --bin bench_noisy_qft --release

# Smaller circuit for a quick run
cargo run --bin bench_noisy_qft --release -- -n 8

# With CUDA
cargo run --bin bench_noisy_qft --features cuda --release -- -n 10

# Custom noise and budget
cargo run --bin bench_noisy_qft --release -- --noise-p 0.01 --bench-budget 10

# With per-backend cached profiles
cargo run --bin bench_noisy_qft --features cuda --release -- \
      --cpu-profile profiles/cpu_f64.json --cuda-profile profiles/cuda_f64.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --n-qubits <N>` | `14` | Physical qubits in the QFT (DM SV has 2N qubits) |
| `--noise-p <P>` | `0.005` | Depolarizing error probability per gate |
| `--bench-budget <S>` | `20` | Time budget per benchmark run (seconds) |
| `--max-size <N>` | `6` | Max virtual-qubit gate size for fusion |
| `--cpu-profile <PATH>` | — | Cached CPU HardwareProfile JSON |
| `--cuda-profile <PATH>` | — | Cached CUDA HardwareProfile JSON |
| `--profile-budget <S>` | `20` | Profiling budget (seconds) |
| `--profile-qubits <N>` | auto (= 2n) | SV qubits for profiling |

### Fusion Configs Compared

| Config | Description |
|--------|-------------|
| `unfused` | Raw DM circuit, no fusion at all |
| `fused(4)` | Fuse up to 4-qubit DM gates (2 physical qubits) |
| `fused(6)` | Fuse up to 6-qubit DM gates (3 physical qubits) |
| `hw-adaptive` | Roofline-guided fusion up to `--max-size` |

DM gates always have even virtual-qubit counts, so fusion steps by 2.

### Memory Requirements

The density-matrix statevector for n physical qubits has 2^(2n) complex F64
amplitudes:

| n | SV qubits | Memory |
|---|-----------|--------|
| 8 | 16 | 1 MiB |
| 10 | 20 | 16 MiB |
| 12 | 24 | 256 MiB |
| 14 | 28 | 4 GiB |
| 16 | 32 | 64 GiB |

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `LLVM_CONFIG` | build.rs | **Required.** Path to `llvm-config` binary. |
| `CXX` | build.rs | C++17 compiler (default: `c++`) |
| `CUDA_PATH` | build.rs | CUDA toolkit root (auto-detected if unset) |
| `CAST_NUM_THREADS` | cpu::get_num_threads | CPU simulation threads (default: all cores) |

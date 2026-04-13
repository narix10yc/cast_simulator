# CLI Tools

CAST ships two command-line binaries for hardware profiling and benchmarking.
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
the roofline stabilizes — profiles at 30 and 32 qubits will agree closely.
But in the transition zone (24-30 qubits), cache effects can meaningfully
shift the crossover, so the profile qubit count should match the target
simulation size for best results.

### Saved Profiles

Profiles are saved as JSON and can be loaded by `bench` via
`--profile <path>` to skip re-profiling.

### Merging Multiple Runs

Because peak memory bandwidth has ±4–5% run-to-run variance on typical
GPUs, any single profile is noisy. Profile multiple times and merge for
a more robust consensus profile:

```sh
# Five independent runs
for i in 1 2 3 4 5; do
    cargo run --bin profile_hw --features cuda --release -- \
        --backend cuda --precision f64 -n 30 --budget 180 \
        --save-profiles tmp/profiles_runs/run${i}/
done

# Merge into one profile
cargo run --bin profile_hw --features cuda --release -- \
    --merge tmp/profiles_runs/run{1,2,3,4,5}/cuda_f64.json \
    --merge-out profiles/cuda_f64.json
```

See [docs/hardware_profiling.md](hardware_profiling.md) for a full
discussion of run-to-run variance and the merge workflow.

---

## bench — Fusion Strategy Benchmark

Loads OpenQASM circuit files, applies each requested fusion strategy, then
times execution on the selected backend via `Simulator::bench`.
Compile and execute phases are reported separately; the execute phase is
sampled adaptively within `--bench-budget` wall seconds.

### Quick Start

```sh
# Default: all three fusion modes × one circuit, CUDA backend, cached profile
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json \
      examples/journal_examples/qft-cx-30.qasm

# All 30-qubit circuits, default fusion modes
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json \
      examples/journal_examples/*-30*.qasm

# CPU backend, only hardware-adaptive fusion
cargo run --bin bench --release -- \
      --backend cpu --fusion hw-adaptive \
      examples/journal_examples/qft-cx-30.qasm

# Two selected modes, comma-separated
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json \
      --fusion none,hw-adaptive \
      examples/journal_examples/mexp-17.qasm

# Sparse-vs-dense ablation: run twice with and without --force-dense,
# then compare rows (identical other args).
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json --fusion hw-adaptive \
      examples/journal_examples/mexp-17.qasm
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json --fusion hw-adaptive --force-dense \
      examples/journal_examples/mexp-17.qasm
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {cpu,cuda}` | `cuda` | Simulation backend |
| `--fusion <MODES>` | `none,size-only,hw-adaptive` | Fusion modes to benchmark (CSV or repeated flag) |
| `--profile <PATH>` | — | Cached HardwareProfile JSON (skips profiling) |
| `--profile-qubits <N>` | `30` | SV qubits for profiling |
| `--profile-budget <S>` | `20` | Profiling time budget (seconds) |
| `--max-size <N>` | `4` | Max gate size for `size-only` and `hw-adaptive` fusion |
| `--bench-budget <S>` | `5` | Exec-phase time budget per benchmark row (seconds) |
| `--force-dense` | off | Disable sparsity-aware codegen (ztol=0) |
| `--fusion-log` | off | Print per-decision fusion log for each hw-adaptive row |
| Files (positional) | required | One or more `.qasm` files |

### Fusion Modes

| Mode | Description |
|------|-------------|
| `none` | No fusion (`size_only(1)`, Phase 1 canonicalization only) |
| `size-only` | Size-gated fusion up to `--max-size` qubits |
| `hw-adaptive` | Roofline-guided fusion up to `--max-size` qubits |

### Output

One row per `(circuit, fusion mode)` pair. Columns:

- **Circuit** — QASM file stem
- **Fusion** — fusion mode label (`none`, `size-only`, `hw-adaptive`)
- **Gates** — post-fusion gate count
- **Depth** — post-fusion circuit depth (row count)
- **Compile** — cold-start wall time to generate and finalize all kernels
- **Exec time** — adaptively-sampled execution time (wall on CPU, summed
  CUDA event times on GPU), printed as `mean ± stddev (N iters)`

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `LLVM_CONFIG` | build.rs | **Required.** Path to `llvm-config` binary. |
| `CXX` | build.rs | C++17 compiler (default: `c++`) |
| `CUDA_PATH` | build.rs | CUDA toolkit root (auto-detected if unset) |
| `CAST_NUM_THREADS` | cpu::get_num_threads | CPU simulation threads (default: all cores) |

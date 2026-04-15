# Hardware Profiling Workflow

The adaptive roofline profiler in [`src/profile.rs`](../src/profile.rs) has
one inherent limitation: any single run gives a noisy snapshot of peak
memory bandwidth. On the RTX 5090 we measured ±4–5% run-to-run variance
in `peak_bw_gib_s`, driven by CUDA driver state, thermal behaviour, and
concurrent workload. Peak compute is much more stable (±0.5%), and the
crossover AI is insensitive at the ~2% level for most circuits, but the
bandwidth noise still feeds back into the fitted roofline.

## The `--merge` workflow

The profiler supports pooling raw sweep samples from multiple independent
runs into one consensus profile. This is strictly better than any single
run: the piecewise-roofline fit sees 5× the data points, so the median
used to estimate `bw_slope` becomes substantially more robust.

### Step 1: run the profiler N times

```sh
# Five independent F64 runs, each with a 180s budget.
for i in 1 2 3 4 5; do
    target/release/profile_hw \
        --backend cuda --precision f64 \
        -n 30 --budget 180 \
        --save-profiles tmp/profiles_runs/run${i}/
done
```

Each run completes the seed phase (7 probes at AI = 1, 2, 4, …, 64) and
one or more refinement rounds, producing 8–14 samples with the adaptive
fit loop stopping when `R² >= 0.95` stabilises (`R2_DELTA = 0.01`). With a
180s budget the refinement loop has plenty of headroom — runs typically
finish in 65–85s.

### Step 2: merge the runs

```sh
target/release/profile_hw \
    --merge \
      tmp/profiles_runs/run1/cuda_f64.json,\
      tmp/profiles_runs/run2/cuda_f64.json,\
      tmp/profiles_runs/run3/cuda_f64.json,\
      tmp/profiles_runs/run4/cuda_f64.json,\
      tmp/profiles_runs/run5/cuda_f64.json \
    --merge-out profiles/cuda_f64.json
```

The merge mode:

1. Loads each input `HardwareProfile` JSON.
2. Validates that all inputs share the same `ProfileConfig` (same backend,
   device, precision, and `n_qubits`). Fails with a clear error if any input
   was measured on different hardware.
3. Concatenates every input's `raw: Vec<SweepEntry>` into one pooled vector.
4. Passes the pooled samples to [`profile::fit_from_samples`](../src/profile.rs),
   which runs the same piecewise-roofline fit that `adaptive_sweep` uses
   internally — just with more data.
5. Emits one consensus `HardwareProfile` JSON suitable for
   [`bench --profile`](tools.md#bench--fusion-strategy-benchmark).

The merge step is pure CPU; it takes milliseconds regardless of how many
runs you pool.

## Observed variance (5 RTX 5090 runs)

### CUDA F64 — individual runs

| Run | Samples | R² | Peak BW (GiB/s) | Peak compute (GFLOP/s) | Crossover AI |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.999 | 1640.3 | 827.1 | 8.52 |
| 2 | 9 | 0.998 | 1695.1 | 822.7 | 8.40 |
| 3 | 10 | 0.984 | 1542.5 | 822.7 | 8.73 |
| 4 | 14 | — | 1680.5 | 827.5 | 8.94 |
| 5 | 9 | — | 1705.8 | 822.6 | 8.24 |
| **range** | **6** | — | **1542–1706** | **822.6–827.5** | **8.24–8.94** |
| **CoV** | — | — | **3.9 %** | **0.3 %** | **3.2 %** |
| **merged (50 pts)** | **50** | — | **1705.8** | **827.5** | **8.60** |

### CUDA F32 — individual runs

| Run | Samples | R² | Peak BW (GiB/s) | Peak compute (GFLOP/s) | Crossover AI |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 1.000 | 1583.6 | 17234.0 | 90.94 |
| 2 | 10 | 1.000 | 1488.9 | 18118.8 | 95.30 |
| 3 | 10 | 0.999 | 1640.0 | 17752.9 | 91.53 |
| 4 | 10 | — | 1585.2 | 17244.1 | 90.99 |
| 5 | 10 | — | 1702.0 | 17783.6 | 91.31 |
| **range** | — | — | **1488–1702** | **17234–18119** | **90.9–95.3** |
| **CoV** | — | — | **4.7 %** | **2.0 %** | **1.9 %** |
| **merged (50 pts)** | **50** | — | **1702.0** | **18118.8** | **95.29** |

### Interpretation

- **Peak BW noise is a GPU scheduling artifact.** CUDA kernels compete for
  memory controllers with whatever else is running on the system, and the
  peak probe (at the lowest AI) saturates the memory subsystem. Thermal
  throttling can also kick in during long sweeps. The merged profile's
  peak BW is the max across runs, which is the correct statistical
  estimator of the true hardware ceiling (1792 GiB/s nominal for RTX 5090).

- **Peak compute is stable** because it's limited by on-chip ALU
  throughput and is insensitive to memory behaviour.

- **The merged crossover AI is tighter** because `bw_slope` is a median
  over the lower half of 50 pooled samples instead of 4–5 from a single
  run. A wrong crossover AI by >20% would push `hw-adaptive` fusion to
  make different decisions; within ±2%, it does not.

## Using the merged profile

The promoted `profiles/cuda_f64.json` and `profiles/cuda_f32.json` can be
consumed by any `bench` invocation:

```sh
target/release/bench \
    --profile profiles/cuda_f64.json \
    --fusion hw-adaptive \
    examples/journal_examples/mexp-17.qasm
```

Verification: re-running the 30q FP64 ablation with the merged profile
produces identical fusion decisions (same gate counts per circuit, same
depth) and identical exec times (within ±0.1%) as with the pre-merge
profile. The merged profile is strictly better for downstream fusion
decisions because the fit is more robust, without introducing regression
on any existing benchmark circuit.

See [`docs/replication_ablation_30q.md`](replication_ablation_30q.md) for
the full ablation table.

## When to re-profile

Re-run the full 5-run merge workflow when:

- Moving to new hardware (different GPU, different CPU, different CUDA
  driver version). The `ProfileConfig.device` field tracks this, but the
  numbers drift dramatically across hardware generations.
- Changing the CAST kernel generator in a way that affects per-gate
  throughput at a given AI. (The fusion cost model uses the roofline to
  predict fused-gate throughput; changes to the kernel generator's code
  path invalidate the assumption.)
- After significant thermal events (throttling, ambient temperature
  shifts). The merge workflow will catch this because outlier runs
  contribute only their lowest observations to the peak.

The `profiles/` directory is gitignored: each user re-profiles for their
own hardware. Keep the raw per-run files in `tmp/profiles_runs/` if you
want to diff subsequent merges.

## CPU profiling notes

CPU profiling uses the same adaptive roofline sweep as CUDA, but with
wall-clock timing (`Instant::now`) instead of CUDA events.

Key difference: at 30 qubits the CPU profiler's max probed AI (64, from
`MAX_GATE_QUBITS=6`) does not reach the compute-bound regime on a
32-core Threadripper 7970X. The fitted "peak compute" is the highest
bandwidth-limited throughput, not the true FLOP ceiling. The roofline R²
is moderate for AVX2 (0.62-0.99 depending on run) because the transition
region is noisy on CPU. AVX-512 R² is negative — the transition region
behaves differently at wider SIMD widths.

For CPU benchmarks, AVX2 (256-bit) is recommended: it matches AVX-512
performance on bandwidth-bound 30q workloads and produces a more stable
roofline fit. The `bench` binary accepts `--simd 256` to override the
default (native auto-detect).

### Implications for hw-adaptive fusion on CPU

The CPU crossover AI (~45-52 on the Threadripper 7970X at 30q) is much
higher than the GPU's (~8.6). This limits the discriminative power of
hw-adaptive fusion:

| Max fused gate size | Max AI (dense) | GPU cost (÷8.6) | CPU cost (÷51.7) |
|---:|---:|---:|---:|
| 4q | ~32 | 3.7 | 1.0 (memory-bound) |
| 5q | ~64 | 7.4 | 1.2 |
| 6q | ~128 | 14.9 | 2.5 |

At `--max-size 4`, all fused gates remain below the CPU crossover, so
hw-adaptive and size-only produce identical results. At `--max-size 6`,
dense 6-qubit gates cross the threshold (cost = 2.5), enabling
hw-adaptive to reject some unprofitable fusions. In practice, this
matters only on circuits with dense fused gates (qvc-30: 5.8% gain).

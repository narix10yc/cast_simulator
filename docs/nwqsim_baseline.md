# NWQ-Sim Baseline Setup & Exploration Notes

## Overview

**NWQ-Sim** (PNNL, Ang Li et al.) is a unified HPC quantum simulation framework
that subsumes the earlier standalone **SV-Sim** (SC'21) and **DM-Sim** (SC'20,
Best Paper nominee) projects.

- **Repo:** https://github.com/pnnl/NWQ-Sim
- **Legacy repos:** https://github.com/pnnl/SV-Sim, https://github.com/pnnl/DM-Sim
  (both marked "Merged in NWQSim")
- **Key papers:**
  - DM-Sim: SC'20 (IEEE), Best Paper nominee
  - SV-Sim: SC'21 (ACM)
  - NWQ-Sim: SC'23 poster, arXiv:2401.06861

### Core Contributions

| Component | Technique | Focus |
|-----------|-----------|-------|
| SV-Sim | PGAS (Partitioned Global Address Space) via NVSHMEM/MPI | Multi-node/multi-GPU distributed state-vector |
| DM-Sim | BSP (Bulk Synchronous Parallel) on GPU clusters | Density-matrix with IBM device noise models |
| NWQ-Sim | Unified wrapper + QASM frontend | Also includes TN-Sim, STAB-Sim |

### Comparison to CAST

| Aspect | CAST | NWQ-Sim |
|--------|------|---------|
| Core idea | JIT-compiled, sparsity-aware kernels via LLVM IR; roofline-guided fusion | Hand-written kernels; distributed scaling |
| Fusion | Agglomerative with hardware-adaptive cost model | Basic 2-gate fusion (greedy) |
| Sparsity | Per-gate sparsity detection, zero-skip in generated code | Dense matrix-vector multiply |
| Backends | Single-node CPU (SIMD) + single GPU (CUDA) | CPU, OpenMP, MPI, CUDA, NVSHMEM multi-GPU, HIP/AMD |
| Scale target | Single-node performance (deep circuits) | Multi-node HPC clusters (wide circuits) |
| Noise | KrausChannel + density-matrix via superoperator | Device JSON noise profiles (IBM models) |
| Input | OpenQASM 2.0 | OpenQASM 2/3, Q#/QIR, Qiskit, XACC |

**Key differentiator:** CAST competes on *per-node kernel efficiency* — JIT
specialization and fusion should outperform NWQ-Sim's generic kernels on the
same hardware. NWQ-Sim's strength is *distributed scaling*, which is orthogonal.

## Build

### Prerequisites

- CMake 3.20+, C++17 compiler
- CUDA toolkit (for GPU backend)
- OpenMP (for multi-threaded CPU backend)
- Optional: MPI, NVSHMEM (multi-node), NLopt (VQE)

### Build Commands

```bash
cd external/NWQ-Sim
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH=120 \
         -DNWQSIM_ENABLE_HIP=OFF \
         -DNWQSIM_ENABLE_VQE=OFF
make -j32
```

- Set `-DCUDA_ARCH=<compute_capability>` for your GPU (e.g., 70=Volta, 80=Ampere, 89=Ada, 120=Blackwell).
- VQE and HIP disabled — not needed for benchmarking.

### Our Build Environment

- GPU: 2x NVIDIA RTX 5090 (sm_120, Blackwell)
- CUDA: 13.2
- CPU: 64 cores
- GCC: 13.3, CMake: 3.28

## Usage

### CLI

```bash
# State-vector simulation
./build/qasm/nwq_qasm --backend <BACKEND> --sim sv --shots <N> -q <circuit.qasm>

# Density-matrix simulation (requires --device noise profile)
./build/qasm/nwq_qasm --backend <BACKEND> --sim dm --device <device.json> -q <circuit.qasm>
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `--backend CPU\|OpenMP\|NVGPU` | Simulation backend |
| `--sim sv\|dm` | State-vector or density-matrix |
| `--shots <N>` | Number of measurement shots |
| `--verbose` | **Enable timing output** (required to get sim time) |
| `--disable_fusion` | Disable gate fusion |
| `--metrics` | Circuit metrics (depth, gate counts, densities) |
| `--hw_threads <N>` | Set OpenMP thread count |
| `--device <path>` | Device noise profile JSON (required for DM-Sim) |
| `--basis` | Decompose to backend basis gates |

### Timing Output Format

With `--verbose`, prints:
```
n_qubits:<N>, n_gates:<original>, sim_gates:<after_fusion>, ncpus:<N>, comp:<ms>, comm:<ms>, sim:<ms>, mem:<MB>, mem_per_cpu:<MB>
```

- `comp` = compute time (ms)
- `sim` = total simulation time (ms), equals comp for single-node
- `ncpus` always reports 1 (display bug in OpenMP backend, actual threading works)

## Exploration Results (2026-03-21)

### SV-Sim Performance on Bundled Circuits

All benchmarks: fusion enabled, 1 shot.

| Circuit | Qubits | Gates | Fused Gates | CPU (ms) | OpenMP 32T (ms) | GPU (ms) |
|---------|--------|-------|-------------|----------|-----------------|----------|
| qft_n4 | 4 | 13 | 8 | 0.01 | 0.007 | 3.9 |
| bv_n14 | 14 | 42 | 38 | 1.6 | 1.2 | 3.5 |
| qec_n17 | 17 | 41 | 36 | 18.5 | 15.5 | 3.5 |
| qaoa_n20 | 20 | 101 | 57 | 153.6 | 122.5 | 4.3 |
| knn_n25 | 25 | 231 | 75 | 11555 | 9864 | 64.0 |

### SV-Sim GPU on CAST Benchmark Circuits (30 qubits)

Hardware: NVIDIA RTX 5090 (sm_120), CUDA 13.2. Single GPU, 1 shot.

| Circuit | Gates | Fused Gates | Fusion ON (ms) | Fusion OFF (ms) | Fusion Speedup |
|---------|-------|-------------|----------------|-----------------|----------------|
| ala-30 | 811 | 430 | 10,623 | 19,519 | 1.84x |
| hea-30 | 742 | 453 | 11,443 | 18,204 | 1.59x |
| hes-30 | 628 | 203 | 5,185 | 15,368 | 2.96x |
| icmp-30 | 411 | 166 | 4,204 | 9,959 | 2.37x |
| iqp-30 | 1,710 | 358 | 9,046 | 41,206 | 4.56x |
| qft-cp-30 | 601 | 451 | 11,400 | 15,024 | 1.32x |
| qft-cx-30 | 2,193 | 802 | 19,600 | 52,845 | 2.70x |
| qvc-30 | 4,081 | 503 | 12,643 | 97,689 | 7.73x |
| rqc-30 | 901 | 224 | 5,673 | 21,782 | 3.84x |

### CAST vs NWQ-Sim Head-to-Head (GPU, 30 qubits, F64)

Hardware: NVIDIA RTX 5090 (sm_120). CAST uses hw-adaptive fusion (best config),
NWQ-Sim uses its built-in fusion. Both use F64 precision.

CAST profile: BW=1644.0 GiB/s, Compute=831.3 GFLOPs/s, Crossover AI=8.7

| Circuit | NWQ-Sim (ms) | CAST (ms) | Speedup | CAST Gates | NWQ-Sim Fused Gates |
|---------|-------------|-----------|---------|------------|---------------------|
| ala-30 | 10,623 | 3,100 | 3.4x | 90 | 430 |
| hea-30 | 11,443 | 4,880 | 2.3x | 197 | 453 |
| hes-30 | 5,185 | 1,760 | 2.9x | 90 | 203 |
| icmp-30 | 4,204 | 625 | 6.7x | 27 | 166 |
| iqp-30 | 9,046 | 2,270 | 4.0x | 135 | 358 |
| qft-cp-30 | 11,400 | 1,680 | 6.8x | 112 | 451 |
| qft-cx-30 | 19,600 | 2,040 | 9.6x | 112 | 802 |
| qvc-30 | 12,643 | 9,260 | 1.4x | 250 | 503 |
| rqc-30 | 5,673 | 2,660 | 2.1x | 92 | 224 |

**CAST is 1.4–9.6x faster** (geometric mean ~3.5x) across all circuits.

### CAST Full Results (all fusion configs)

| Circuit | no-fusion | default | aggressive | hw-adaptive |
|---------|-----------|---------|------------|-------------|
| ala-30 | 6.18 s (270g) | 3.39 s (145g) | 3.10 s (90g) | 3.10 s (90g) |
| hea-30 | 8.25 s (441g) | 5.51 s (246g) | 5.01 s (195g) | 4.88 s (197g) |
| hes-30 | 2.79 s (201g) | 1.99 s (112g) | 1.79 s (89g) | 1.76 s (90g) |
| icmp-30 | 2.60 s (137g) | 659 ms (29g) | 625 ms (27g) | 625 ms (27g) |
| iqp-30 | 4.10 s (336g) | 2.72 s (175g) | 2.27 s (134g) | 2.27 s (135g) |
| qft-cx-30 | 5.80 s (450g) | 2.84 s (224g) | 2.04 s (112g) | 2.04 s (112g) |
| qvc-30 | 10.0 s (437g) | 9.25 s (250g) | 10.4 s (190g) | 9.26 s (250g) |
| rqc-30 | 4.17 s (197g) | 2.97 s (114g) | 2.90 s (86g) | 2.66 s (92g) |

### Observations

1. **CAST consistently outperforms NWQ-Sim** — the combination of JIT-compiled
   sparsity-aware kernels and aggressive fusion yields 1.4–9.6x speedup.

2. **Fusion is the key differentiator.** CAST reduces gate counts far more
   aggressively (e.g., icmp: 411→27 vs NWQ-Sim 411→166). This compounds with
   per-gate efficiency gains from sparsity-aware codegen.

3. **qvc-30 is the weakest case** (1.4x). This circuit has many gates (4081) and
   aggressive fusion actually hurts CAST (10.4s vs 9.25s default), suggesting the
   cost model correctly avoids over-fusion here. NWQ-Sim's fusion is also effective
   on this circuit (4081→503).

4. **qft-cx-30 is the strongest case** (9.6x). NWQ-Sim fuses 2193→802 gates but
   still takes 19.6s. CAST fuses to 112 gates and finishes in 2.04s.

5. **NWQ-Sim CPU backends are impractical** at 30 qubits — single-threaded CPU
   and OpenMP both timeout (>2-5 minutes). GPU has ~3.5ms launch overhead.

6. **DM-Sim only supports IBM basis gates** ({X, ID, DELAY, SX, RZ}). Direct
   DM-Sim comparison requires transpilation.

### Single-Gate Kernel Performance (no fusion, GPU, F64)

Measures per-gate kernel execution time in isolation. Circuit: CX chain
`cx(0,1), cx(1,2), ..., cx(n-2,n-1)` repeated 3 passes. Each consecutive CX
shares a qubit with the next, preventing any fusion or merging by either
simulator.

**Verification:** CAST gate counts exactly match input counts at all qubit sizes
(57, 63, 69, 75, 81, 87), confirming zero fusion/merging. `--force-dense` was
used for CAST to disable sparsity-aware codegen (verified: CX timing identical
with and without `--force-dense` since CX is a permutation matrix — already
bandwidth-bound).

| Qubits | SV Size | NWQ-Sim (ms/gate) | CAST dense (ms/gate) | CAST Speedup |
|--------|---------|-------------------|---------------------|--------------|
| 20 | 16 MiB | 0.084 | 0.0087 | 9.6x |
| 22 | 64 MiB | 0.172 | 0.023 | 7.4x |
| 24 | 256 MiB | 0.453 | 0.183 | 2.5x |
| 26 | 1 GiB | 1.65 | 0.737 | 2.2x |
| 28 | 4 GiB | 6.58 | 2.95 | 2.2x |
| 30 | 16 GiB | 26.4 | 11.8 | 2.2x |

**At large qubit counts (26–30) where kernels are bandwidth-bound, CAST's
JIT-compiled dense kernels are ~2.2x faster per gate than NWQ-Sim's hand-written
kernels.** At smaller counts (20–22), CAST's advantage is larger (7–10x), likely
due to lower per-kernel launch overhead via persistent-grid work-stealing.

Both simulators scale ~4x per 2 added qubits, consistent with bandwidth-bound
behavior (state vector doubles per qubit → 2x more data per gate).

## Next Steps

- [ ] Extend comparison to 32 and 34 qubit circuits
- [ ] Document fusion configs and optimization parity for paper

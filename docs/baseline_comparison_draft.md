# Draft: Experimental Evaluation Sections

*For the ACM TQC journal revision. Addresses Reviewer 2 (breadth of
comparison), Reviewer 3 (apples-to-apples fairness, documented
configurations, dense vs sparse ablation).*

---

## 8.3 Comparison with State-of-the-Art Simulators

To evaluate CAST against widely-used quantum simulation frameworks, we
benchmark eight 30-qubit circuits from the paper's standard set against
four external simulators on a single NVIDIA RTX 5090 GPU (Blackwell
SM 120, 32 GiB, CUDA 13.2). The circuits span representative structures:
variational ansatze (ala, hea, hes), random circuits (rqc), structured
entanglement (iqp, qft-cx), quantum volume (qvc), and integer comparison
(icmp). All experiments use the adaptive fusion mode for CAST
(`hw-adaptive`, max fused gate size 4).

### Baseline Configuration

To ensure a fair comparison, we give each baseline its best available
fusion configuration. We tested each simulator's fusion parameter at
multiple settings and report the minimum per-circuit time. We use each
tool's own internal timing metric where available, stripping Python
framework overhead that is not part of the simulation engine.

| Simulator | Version | Precision | Fusion | Timing metric |
|-----------|---------|-----------|--------|---------------|
| **CAST** | 0.1.0 | F32 / F64 | hw-adaptive (max 4q) | CUDA-event kernel time |
| **Qiskit-Aer** | 0.17.2 | F32 / F64 | best of {off, max 2q, max 5q} | C++ `metadata['time_taken']` |
| **Qibo** | 0.3.2 (qibojit+cupy) | F32 / F64 | none (no built-in fusion) | Python wall-clock |
| **QSim** | 0.22.0 (cuStateVec) | F32 only | built-in (max 4q) | C++ `simu time` (verbosity=2) |

Qiskit-Aer's fusion aggregates gates into up to N-qubit unitaries,
controlled by `fusion_max_qubit`. We tested N in {off, 2, 3, 4, 5}
and found the optimal setting varies by circuit and precision: at F64,
the default (max 5q) is best on 6 of 8 circuits; at F32, max 2q is
best on 5 of 8 circuits because the CPU-side cost of computing 3-5
qubit fused unitaries (32x32 complex matrix multiplication) exceeds
the GPU-side savings when the state vector is smaller. We report the
best per-circuit result for Qiskit-Aer throughout.

QSim delegates GPU execution to NVIDIA's cuStateVec library and applies
its `MultiQubitGateFuser` with `max_fused_gate_size=4`. Qibo performs
no gate fusion but benefits from qibojit's JIT-compiled cupy kernels.
All simulators use a single GPU; multi-GPU and distributed modes are
disabled for parity. Warm-up iterations absorb JIT compilation and
CUDA module loading before timing begins.

### FP64 Results

At double precision, CAST is the only simulator that generates
JIT-compiled, sparsity-aware fused kernels. QSim is omitted (FP32 only)
and Qulacs is omitted (CPU only, no GPU wheel for the current CUDA
toolkit).

| Circuit | CAST | Qiskit-Aer | Qibo | Speedup |
|---------|------|------------|------|---------|
| ala     | 3.09 s | 18.0 s  | 15.5 s | 5.0x |
| hea     | 4.81 s | 20.5 s  | 11.8 s | 2.5x |
| hes     | 1.75 s | 14.6 s  | 9.80 s | 5.6x |
| icmp    | 0.61 s | 14.4 s  | 7.55 s | **12.3x** |
| iqp     | 2.24 s | 14.8 s  | 31.1 s | 6.6x |
| qft-cx  | 2.04 s | 14.7 s  | 40.0 s | 7.2x |
| qvc     | 9.27 s | 23.6 s  | 77.5 s | 2.5x |
| rqc     | 2.63 s | 15.2 s  | 16.3 s | 5.8x |

Qiskit-Aer values use the best of {fusion off, fusion max 5q} per circuit;
hea uses fusion off (20.5 s vs 23.3 s with fusion on).

CAST outperforms the best baseline on every circuit, with speedups
ranging from 2.5x (hea, qvc) to 12.3x (icmp). The advantage is most
pronounced on structured circuits where CAST's roofline-guided fusion
aggressively reduces the number of state-vector passes: icmp-30 is fused
from 411 input gates to 27 fused kernels. On denser variational circuits
(hea, qvc), fusion opportunities are more limited and the gap narrows,
though CAST still leads by 2.5x.

### FP32 Results

Single-precision experiments introduce QSim backed by NVIDIA's
cuStateVec library — a hand-optimized, closed-source implementation
tuned for NVIDIA GPUs. To isolate kernel quality from Python/framework
overhead, we report cuStateVec's self-reported `simu time` (the C++
gate-application loop measured via `std::chrono`), excluding cuStateVec
handle creation (~3 s) and Cirq's numpy state-vector copy (~3 s).

| Circuit | CAST | Qiskit-Aer* | Qibo | cuStateVec | CAST / best |
|---------|------|-------------|------|------------|-------------|
| ala     | 1.03 s | 13.8 s  | 7.82 s | 1.20 s  | 1.2x |
| hea     | 2.21 s | 13.8 s  | 6.03 s | 2.03 s  | 0.9x |
| hes     | 0.88 s |  8.21 s | 4.95 s | 1.37 s  | 1.5x |
| icmp    | 0.31 s |  8.58 s | 3.79 s | 1.55 s  | **5.0x** |
| iqp     | 1.11 s | 10.8 s  | 15.6 s | 1.64 s  | 1.5x |
| qft-cx  | 1.00 s |  9.52 s | 20.4 s | 2.03 s  | 2.0x |
| qvc     | 2.16 s | 20.3 s  | 39.2 s | 1.91 s  | 0.9x |
| rqc     | 0.98 s | 11.3 s  | 8.29 s | 1.06 s  | 1.1x |

\* Qiskit-Aer uses `fusion_max_qubit=2` on {ala, hes, iqp, qvc, rqc},
default (max 5q) on {icmp, qft-cx}, and fusion off on {hea} — whichever
gives the lowest time per circuit.

Against cuStateVec, CAST wins six of eight circuits, with the largest
margin on icmp (5.0x) and qft-cx (2.0x) — circuits whose gate
structure enables aggressive fusion with high sparsity. cuStateVec wins
narrowly on hea and qvc (both ~1.1x), the two densest circuits in the
set. Profiling with NVIDIA Nsight Systems reveals that the gap on these
circuits is primarily a fusion-count difference (CAST produces 195 fused
gates vs. cuStateVec's 180 for hea-30), not a per-kernel efficiency
gap: at the individual-kernel level, CAST's JIT-generated kernels match
cuStateVec's hand-tuned kernels within 1% on median execution time
(11.4 ms vs. 11.3 ms per gate at 30 qubits).

---

## 8.4 Sparsity-Aware Code Generation Ablation

CAST's LLVM-based code generator classifies each entry of the fused-gate
matrix as zero, +/-1, or a general constant at compile time. Zero entries
are elided entirely (no multiply-accumulate emitted); +/-1 entries are
folded into sign flips (no multiply). To isolate the effect, we run each
circuit with sparsity-aware classification enabled (default, `ztol=1e-12`
for F64, `1e-6` for F32) and disabled (`--force-dense`, `ztol=0`).

### F64 — Sparse vs. Dense

| Circuit | Sparse | Dense | Overhead |
|---------|--------|-------|----------|
| ala     | 3.08 s | 3.10 s  | +0.6% |
| hea     | 4.81 s | 4.82 s  | +0.2% |
| hes     | 1.75 s | 1.75 s  | +0.0% |
| icmp    | 0.61 s | 0.83 s  | **+35.6%** |
| iqp     | 2.24 s | 2.48 s  | **+10.7%** |
| qft-cx  | 2.04 s | 2.27 s  | **+11.3%** |
| qvc     | 9.23 s | 9.27 s  | +0.4% |
| rqc     | 2.62 s | 2.81 s  | +7.3% |

### F32 — Sparse vs. Dense

| Circuit | Sparse | Dense | Overhead |
|---------|--------|-------|----------|
| ala     | 1.03 s | 1.03 s  | +0.0% |
| hea     | 2.21 s | 2.21 s  | +0.0% |
| hes     | 0.88 s | 0.88 s  | +0.1% |
| icmp    | 0.31 s | 0.31 s  | -0.3% |
| iqp     | 1.11 s | 1.20 s  | **+8.1%** |
| qft-cx  | 1.00 s | 1.04 s  | +4.5% |
| qvc     | 2.17 s | 2.17 s  | +0.0% |
| rqc     | 0.98 s | 0.98 s  | +0.2% |

The benefit of sparsity-aware codegen depends on the circuit structure.
Circuits built from controlled-phase (cp) and diagonal gates — icmp,
iqp, qft-cx — contain fused matrices with structurally zero entries
that the classifier exploits. icmp-30 benefits the most (35.6% at F64),
because the 7-qubit fused cp-chain produces a matrix that is 75% zero.
Dense variational circuits (ala, hea, qvc), whose fused matrices have
no structural zeros, show negligible difference (<1%).

The effect is more pronounced at F64 than F32 because F64 kernels
operate further from the memory-bandwidth ceiling. At F32, the state
vector is half the size; kernel execution is already dominated by DRAM
transfer latency, so eliminating a few FP operations has little
measurable impact. At F64, the additional floating-point instructions
compete with the memory subsystem for pipeline cycles, making the
zero-skip savings visible.

---

## 8.5 Baseline Fusion Parity

To address the concern about apples-to-apples comparisons, we run each
baseline simulator with its built-in fusion both *enabled* and
*disabled*, documenting the per-circuit impact. This reveals whether the
speedups in Section 8.3 stem from CAST's fusion or from its kernel
efficiency. CAST's own fusion ablation (none / size-only / hw-adaptive)
is reported in Section 8.2.

For Qiskit-Aer, fusion is controlled via the `fusion_enable` and
`fusion_max_qubit` constructor parameters (confirmed disabled via
`metadata['fusion']['enabled'] = False`; confirmed max-qubit via
`metadata['fusion']['max_fused_qubits']`). For QSim,
`max_fused_gate_size=1` disables fusion (each gate applied
individually). Qibo does not implement gate fusion.

### Qiskit-Aer Fusion Tuning (F32)

| Circuit | Off | max 2q | max 5q (default) | Best |
|---------|-----|--------|------------------|------|
| ala     | 18.0 s | **13.8 s** | 23.8 s | max 2q |
| hea     | **13.8 s** | 15.8 s | 34.5 s | off |
| hes     | 9.28 s | **8.21 s** | 10.3 s | max 2q |
| icmp    | 8.64 s | 9.07 s | **8.58 s** | max 5q |
| iqp     | 16.1 s | **10.8 s** | 12.4 s | max 2q |
| qft-cx  | 22.5 s | 15.2 s | **9.52 s** | max 5q |
| qvc     | 69.2 s | **20.3 s** | 33.7 s | max 2q |
| rqc     | 15.1 s | **11.3 s** | 16.7 s | max 2q |

The optimal Qiskit-Aer configuration varies by circuit: max 2q wins on
5 of 8 circuits, max 5q (the default) wins on 2, and no fusion wins on
1. The root cause is that Aer's 3-5 qubit fusion computes dense fused
unitaries (up to 32x32 complex matrices) on the CPU. At F32, where GPU
kernels are already fast, this CPU-side overhead often exceeds the
GPU-side savings from fewer kernel launches. The effect reverses at F64
(not shown), where longer GPU kernel times make the launch-count
reduction worthwhile.

### QSim (cuStateVec): Fusion ON vs OFF (F32)

| Circuit | Fused (max 4q) | Unfused | Δ |
|---------|---------------|---------|------|
| ala     | 1.20 s | 3.05 s | +155% |
| hea     | 2.03 s | 4.98 s | +145% |
| hes     | 1.37 s | 2.27 s | +66% |
| icmp    | 1.55 s | 1.55 s | +0% |
| iqp     | 1.64 s | 3.80 s | +132% |
| qft-cx  | 2.03 s | 5.08 s | +151% |
| qvc     | 1.91 s | 4.94 s | +159% |
| rqc     | 1.06 s | 2.23 s | +110% |

cuStateVec's fusion is uniformly beneficial (1.7-2.6x on all circuits
except icmp), because the cuStateVec library handles fused gates of any
size with minimal overhead and the fused-matrix computation in QSim's
C++ layer is efficient.

### Discussion

The fusion parity data reveals a key design trade-off. Naive fusion
heuristics — such as "always fuse up to 5 qubits" — can degrade
performance when the CPU-side unitary computation dominates. This is
precisely the scenario CAST's roofline cost model is designed to handle:
the `benefit()` function (Algorithm 3) evaluates each candidate fusion
against the roofline model and only accepts it when the predicted
bandwidth savings outweigh the increased arithmetic intensity of the
larger fused kernel. In practice, CAST's adaptive policy accepts all
profitable fusions and rejects none on these circuits — avoiding both
the under-fusion of conservative heuristics and the over-fusion that
hurts Qiskit-Aer at single precision.

The circuit-dependent variation in Qiskit-Aer's best fusion setting
(5 of 8 circuits prefer max 2q, not the default max 5q) underscores
the difficulty of selecting a single global fusion parameter and
motivates CAST's per-gate, per-backend cost-model approach.

---

*Reproduction artifacts: `external/baselines/sweep.py` orchestrates the
full sweep; `external/baselines/run_baseline.py --no-fusion` runs each
baseline without fusion. Qiskit-Aer fusion tuning uses
`AerSimulator(fusion_max_qubit=N)`. Raw timing data for all experiments
are stored in `external/baselines/results/` as JSONL files alongside the
corresponding ablation text logs. CAST's `--force-dense` flag disables
sparsity classification for the ablation in Section 8.4.*

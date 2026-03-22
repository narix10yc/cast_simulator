# F32 vs F64 Precision in Statevector Simulation

A systematic empirical study of when single-precision (F32) arithmetic suffices
for statevector quantum circuit simulation, and when double-precision (F64)
is necessary. All experiments use CAST's LLVM JIT kernel pipeline on CPU,
comparing F32 and F64 results for the same circuit with F64 as ground truth.

## Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **1 − Fidelity** | `1 − \|⟨ψ_f32\|ψ_f64⟩\|²` | Global state-level accuracy. The fundamental metric. |
| **TVD** | `0.5 × Σ\|p_i^f32 − p_i^f64\|` | Measurement distribution agreement. Relevant for sampling. |
| **MaxAbsErr** | `max\|p_i^f32 − p_i^f64\|` | Worst-case single-outcome probability error. |
| **NormErr** | `\|1 − ‖ψ_f32‖²\|` | Accumulated norm drift (both backends accumulate in F64). |

## Experiment 1: Circuit Families

Five circuit families tested at 8–24 qubits. Depth scales naturally with
each circuit type.

| Qubits | Circuit | Gates | 1−Fidelity | TVD | MaxAbsErr | NormErr |
|--------|---------|-------|-----------|-----|-----------|---------|
| 8 | QFT | 40 | 2.384e-7 | 1.192e-7 | 9.31e-10 | 2.384e-7 |
| 8 | Random | 96 | 2.669e-7 | 1.931e-7 | 1.44e-8 | 2.669e-7 |
| 8 | Grover-16 | 1000 | 2.384e-7 | 1.192e-7 | 3.73e-9 | 2.384e-7 |
| 8 | VQE | 120 | 2.661e-7 | 1.962e-7 | 1.35e-8 | 2.661e-7 |
| 12 | QFT | 84 | 2.384e-7 | 1.192e-7 | 5.82e-11 | 2.384e-7 |
| 12 | Random | 216 | 4.035e-7 | 3.076e-7 | 9.57e-9 | 4.035e-7 |
| 12 | Grover-50 | 4712 | 2.384e-7 | 1.192e-7 | 5.82e-11 | 2.384e-7 |
| 12 | VQE | 276 | 3.334e-7 | 2.934e-7 | 3.73e-9 | 3.334e-7 |
| 16 | QFT | 144 | 2.384e-7 | 1.192e-7 | 3.64e-12 | 2.384e-7 |
| 16 | Random | 384 | 1.910e-7 | 3.307e-7 | 8.79e-10 | 1.910e-7 |
| 16 | Grover-50 | 6316 | 2.384e-7 | 1.192e-7 | 3.64e-12 | 2.384e-7 |
| 16 | VQE | 496 | 3.212e-7 | 3.643e-7 | 1.94e-10 | 3.212e-7 |
| 20 | QFT | 220 | 2.384e-7 | 1.192e-7 | 2.27e-13 | 2.384e-7 |
| 20 | Random | 600 | 1.035e-7 | 3.818e-7 | 3.97e-10 | 1.035e-7 |
| 20 | Grover-50 | 7920 | 2.384e-7 | 1.192e-7 | 2.27e-13 | 2.384e-7 |
| 20 | VQE | 780 | 8.714e-7 | 5.795e-7 | 2.47e-11 | 8.714e-7 |
| 24 | QFT | 312 | 2.384e-7 | 1.192e-7 | 1.42e-14 | 2.384e-7 |
| 24 | Random | 864 | 3.340e-7 | 4.124e-7 | 2.31e-10 | 3.340e-7 |
| 24 | Grover-50 | 9524 | 2.384e-7 | 1.192e-7 | 1.42e-14 | 2.384e-7 |
| 24 | VQE | 1128 | 1.120e-6 | 7.226e-7 | 2.99e-12 | 1.120e-6 |

### Key Observation 1: Sparse-gate circuits are immune to precision loss

QFT and Grover use exclusively sparse gates (H, X, CZ, CP, SWAP) whose
matrix entries are exactly representable in F32 (0, ±1, ±1/√2, phase
rotations). CAST's sparsity-aware JIT **skips zero multiplications entirely**
and folds ±1 multiplications into sign flips, leaving only the non-trivial
entries to be computed in floating-point.

Result: 1 − Fidelity = 2.384e-7 = 4·ε_mach(F32) **regardless of circuit
depth or qubit count**. This is not error accumulation — it is a constant
per-element rounding floor from the 1/√2 entries in H gates. Grover with
9524 gates shows the same error as QFT with 40 gates.

### Key Observation 2: MaxAbsErr decreases with qubit count

At fixed circuit family and scaling, MaxAbsErr drops as ~1/2^n. This is
because individual probabilities scale as ~1/2^n, and the absolute rounding
error per probability scales proportionally. **F32 maintains constant
relative precision per probability across all qubit counts.**

## Experiment 2: Depth Scaling (Random Circuits)

Random circuits with alternating U3 + CX layers, varying depth from 10 to
5000 at 12, 16, and 20 qubits.

| Qubits | Depth | Gates | 1−Fidelity | TVD | MaxAbsErr |
|--------|-------|-------|-----------|-----|-----------|
| 12 | 10 | 90 | 2.97e-7 | 2.06e-7 | 9.10e-9 |
| 12 | 50 | 450 | 7.45e-7 | 5.16e-7 | 8.18e-9 |
| 12 | 200 | 1800 | 1.69e-6 | 1.07e-6 | 4.16e-8 |
| 12 | 1000 | 9000 | 9.69e-7 | 1.78e-6 | 4.91e-8 |
| 12 | 5000 | 45000 | 4.39e-6 | 3.77e-6 | 1.69e-7 |
| 16 | 10 | 120 | 6.44e-8 | 1.72e-7 | 6.84e-10 |
| 16 | 50 | 600 | 3.15e-8 | 4.09e-7 | 1.63e-9 |
| 16 | 200 | 2400 | 1.32e-6 | 1.04e-6 | 1.28e-8 |
| 16 | 1000 | 12000 | 1.58e-6 | 1.93e-6 | 9.87e-9 |
| 16 | 5000 | 60000 | 1.65e-6 | 3.24e-6 | 5.36e-8 |
| 20 | 10 | 150 | 9.95e-8 | 1.94e-7 | 1.19e-9 |
| 20 | 50 | 750 | 3.77e-7 | 4.22e-7 | 1.36e-9 |
| 20 | 200 | 3000 | 4.14e-7 | 7.93e-7 | 1.01e-9 |
| 20 | 1000 | 15000 | 6.09e-7 | 1.67e-6 | 2.20e-9 |
| 20 | 5000 | 75000 | 9.66e-7 | 3.77e-6 | 6.81e-9 |

### Key Observation 3: Infidelity grows sub-linearly with depth

1 − Fidelity scales approximately as **√D** (random walk), not D. At 20
qubits, going from depth 10 (1−F ≈ 10^-7) to depth 5000 (1−F ≈ 10^-6)
is only a 10× increase over a 500× depth increase. Even at 75000 gates,
infidelity stays below 10^-5.

### Key Observation 4: TVD grows as ~√D

TVD is the more relevant metric for sampling-based algorithms. At 20 qubits:
- Depth 10: TVD = 1.9e-7
- Depth 5000: TVD = 3.8e-6

This means F32 measurement distributions differ from F64 by at most ~4 ppm
in total variation, even for extremely deep circuits. For any practical
sampling task, this is indistinguishable.

### Key Observation 5: Infidelity does NOT grow with qubit count

At fixed depth (100 layers), infidelity at 12 qubits (1.77e-7) is comparable
to 20 qubits (1.41e-7). The per-gate error is per-amplitude relative, and
as amplitudes shrink (more qubits), absolute errors shrink proportionally,
leaving fidelity unchanged.

## Experiment 3: Per-Gate Error Isolation

Repeated application of a single gate type on qubit 0, 20-qubit SV.

| Gate | Repeats | 1−Fidelity | TVD | MaxAbsErr |
|------|---------|-----------|-----|-----------|
| H | 100 | 2.384e-7 | 1.192e-7 | 2.384e-7 |
| H | 1000 | 2.384e-7 | 1.192e-7 | 2.384e-7 |
| H | 10000 | 2.384e-7 | 1.192e-7 | 2.384e-7 |
| U3(1.23,4.56,7.89) | 100 | 1.77e-6 | 8.94e-7 | 1.78e-6 |
| U3(1.23,4.56,7.89) | 1000 | 1.67e-5 | 8.37e-6 | 1.66e-5 |
| U3(1.23,4.56,7.89) | 10000 | 1.66e-4 | 8.30e-5 | 1.63e-4 |
| CX | 100 | 0 | 0 | 0 |
| CX | 5000 | 0 | 0 | 0 |

### Key Observation 6: CAST's sparsity optimization eliminates rounding for sparse gates

**H gate**: constant 1−F = 2.384e-7 regardless of repetition count. The JIT
kernel folds the ±1/√2 entries as immediates; the only rounding is the
representation of 1/√2 itself, which is a one-time constant error.

**CX gate**: **literally zero error** at any repetition count. CX is a
permutation matrix (entries are 0 or 1) — CAST's zero-skip and one-fold
optimizations mean no floating-point arithmetic is performed at all.

**U3 gate**: error grows linearly with repetitions at rate ~1.7e-8 per gate.
This is consistent with ~2·ε_mach(F32) per dense multiply-add, confirming
the theoretical per-gate error model.

### Implication

The error behavior is **bimodal**: sparse gates contribute zero accumulated
error, while dense gates contribute ~2·ε_mach per gate. The total circuit
error depends on the **number of dense gate applications**, not the total
gate count. Sparsity-aware compilation (as in CAST) fundamentally changes
the precision calculus.

## Experiment 4: Fusion Impact

Same random circuit, unfused vs fused with `size_only(3)`.

| Config | Qubits | Gates before | Gates after | 1−Fidelity | TVD |
|--------|--------|-------------|-------------|-----------|-----|
| unfused | 12 | 216 | 216 | 4.04e-7 | 3.08e-7 |
| fused(3) | 12 | 216 | 6 | 4.83e-8 | 7.72e-8 |
| unfused | 16 | 384 | 384 | 1.91e-7 | 3.31e-7 |
| fused(3) | 16 | 384 | 8 | 6.46e-9 | 7.71e-8 |
| unfused | 20 | 600 | 600 | 1.04e-7 | 3.82e-7 |
| fused(3) | 20 | 600 | 10 | 9.21e-8 | 9.29e-8 |

### Key Observation 7: Fusion improves F32 precision

Fusing 216 gates into 6 reduces infidelity by ~8× (12 qubits) and TVD by
~4×. Each fused gate is a denser matrix (more nonzeros), but there are far
fewer kernel applications. Since rounding error accumulates per kernel
application, **fewer applications = less error**, even though each
application is denser.

This means CAST's fusion optimization provides a double benefit: faster
execution AND better F32 precision.

## Experiment 5: Qubit Scaling at Fixed Depth

Random circuit, depth=100 layers, varying qubit count from 6 to 24.

| Qubits | Gates | 1−Fidelity | TVD | MaxAbsErr |
|--------|-------|-----------|-----|-----------|
| 6 | 450 | 6.31e-7 | 5.02e-7 | 3.18e-7 |
| 8 | 600 | 1.19e-7 | 3.34e-7 | 6.49e-8 |
| 10 | 750 | 2.03e-8 | 4.64e-7 | 1.39e-8 |
| 12 | 900 | 1.77e-7 | 5.47e-7 | 2.27e-8 |
| 14 | 1050 | 6.74e-7 | 5.11e-7 | 6.77e-9 |
| 16 | 1200 | 1.28e-7 | 5.28e-7 | 2.28e-9 |
| 18 | 1350 | 2.08e-7 | 6.31e-7 | 5.47e-10 |
| 20 | 1500 | 1.41e-7 | 5.79e-7 | 7.76e-10 |
| 22 | 1650 | 6.15e-7 | 5.64e-7 | 6.62e-10 |
| 24 | 1800 | 1.18e-6 | 7.96e-7 | 6.24e-10 |

### Key Observation 8: Fidelity is independent of qubit count

1 − Fidelity fluctuates around 10^-7 to 10^-6 from 6 to 24 qubits with no
systematic trend. This confirms that per-amplitude rounding is **relative**,
not absolute — adding qubits doesn't degrade precision.

MaxAbsErr, however, drops by ~500× from 6 to 24 qubits. This reflects the
shrinking probability scale (1/2^n), not improving relative accuracy.

## Summary of Findings

### When F32 suffices

1. **All circuits using only standard gates** (H, X, Y, Z, S, T, CX, CZ,
   SWAP, CCX, CP): zero accumulated error from CAST's sparsity-aware JIT.
   F32 is as accurate as F64 for these circuits.

2. **Sampling-based algorithms** (VQE, QAOA, variational): TVD stays below
   10^-5 even at depth 5000. Measurement distributions are statistically
   identical in F32 and F64.

3. **Fused circuits**: fusion reduces both execution time and F32 error.
   With CAST's hw-adaptive fusion, F32 precision improves further.

4. **Any qubit count**: fidelity does not degrade with more qubits at
   fixed circuit depth.

### When F64 is necessary

1. **Circuits with many dense (non-Clifford+T) gates and no fusion**: U3-heavy
   circuits accumulate ~2·ε_mach ≈ 1.2×10^-7 infidelity per dense gate.
   At 10^4 dense gates, infidelity reaches ~10^-4 (4 significant digits).
   At 10^5 dense gates, infidelity reaches ~10^-3 (3 significant digits).

2. **Reference simulations for hardware benchmarking**: if the simulation is
   used to compute reference fidelities against noisy hardware (where hardware
   error rates are ~10^-3 to 10^-2), the simulator error must be well below
   the hardware error. F64 provides ~10^-15 infidelity — safely below any
   hardware noise floor.

3. **Algorithms relying on small amplitude differences**: phase estimation,
   quantum chemistry with high-precision energy differences, or any task
   where the signal of interest is within 4–5 digits of the noise floor.

### The role of sparsity-aware compilation

CAST's JIT kernels classify each matrix entry as zero (skip), ±1 (fold), or
general (multiply). This classification is the key differentiator:

- **Standard quantum gates** (H, X, CX, etc.) have matrices with only 0 and
  ±1 entries (or ±1/√2 for H). Rounding error comes only from the non-trivial
  entries, and does not accumulate across gate applications.

- **Dense gates** (U3 with arbitrary angles, fused multi-qubit gates) have
  all-nonzero matrices. Each application contributes ~2^k · ε_mach per-amplitude
  error for a k-qubit gate.

A simulator without sparsity optimization would show linear error growth even
for standard gates, making F32 impractical at moderate depths. CAST's approach
makes F32 viable for a much wider range of circuits.

### Practical recommendation

| Use case | Recommended | Rationale |
|----------|------------|-----------|
| Standard gate circuits (QFT, Grover, Clifford+T) | **F32** | Zero accumulated error from sparsity-aware JIT |
| Variational algorithms (VQE, QAOA) | **F32** | TVD < 10^-5 at practical depths |
| Random/dense circuits, depth < 1000 | **F32** | 1−F < 10^-5 |
| Dense circuits, depth > 10000 | **F64** | Accumulated error exceeds 4 sig. digits |
| Hardware benchmarking reference | **F64** | Must be below hardware error floor |
| Density-matrix simulation | **F32 or F64** | Superoperators preserve gate sparsity (S = U⊗conj(U) has nnz² out of 16^k entries); same precision rules as SV mode apply |

### Memory and performance trade-off

On an RTX 5090 (32 GiB VRAM):
- F64: max 30 qubits (16 GiB statevector)
- F32: max 31 qubits (8 GiB statevector)

The extra qubit is a modest gain. The primary benefit of F32 is ~2× faster
gate kernel execution (halved memory bandwidth), which is significant for
large-scale benchmarking where the precision requirements are met.

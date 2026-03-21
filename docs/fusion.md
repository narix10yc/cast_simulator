# Gate Fusion

The fusion optimizer reduces circuit depth and kernel count by merging
compatible gates into larger fused gates.  Fewer, denser gates mean fewer
kernel launches and better utilization of memory bandwidth — up to the point
where the fused gate becomes compute-bound.

## Entry Point

```rust
use cast::{fusion, cost_model::FusionConfig, CircuitGraph};

let config = FusionConfig::size_only(4);  // fuse up to 4-qubit gates
fusion::optimize(&mut graph, &config);
```

`optimize()` runs both phases in sequence and calls `squeeze()` after each to
compact the circuit graph.

## Phase 1: Size-2 Canonicalization

**Function:** `apply_size_two_fusion(cg)`

This phase performs two local rewrites that never increase gate size beyond 2
qubits:

### Step A — Absorb single-qubit gates

For each single-qubit unitary gate at position `(row, qubit)`:

1. **Forward scan:** walk later rows on the same qubit.  If the next gate is
   unitary, fuse the pair (matrix product) and place the result at the later
   position.  Stop at channel gates — they act as causal barriers.

2. **Backward scan** (fallback): if no forward target was found, try fusing
   with the nearest earlier unitary on the same qubit.

Repeat until no more absorptions occur, then `squeeze()`.

### Step B — Merge adjacent 2-qubit gates

Scan consecutive row pairs.  If both rows contain 2-qubit unitary gates on
the same pair of qubits, fuse them via matrix product.

Repeat until stable, then `squeeze()`.

**Result:** after Phase 1, the circuit has no "standalone" single-qubit gates
that could have been absorbed, and no redundant back-to-back 2-qubit gates on
the same wires.

## Phase 2: Agglomerative Fusion

**Function:** `apply_gate_fusion(cg, config, cdd_size)`

For each candidate fused-gate size from 3 up to `config.size_max`, the
optimizer repeatedly sweeps the circuit attempting to grow gates:

### Single fusion attempt: `start_fusion()`

1. **Seed** — pick a gate at `(start_row, start_qubit)`.  Skip if it's a
   channel gate or if `start_qubit` isn't the gate's lowest qubit (avoids
   re-processing the same multi-qubit gate from different qubit slots).

2. **Same-row sweep** — absorb gates to the right in the same row, as long as
   the union qubit count stays ≤ `cdd_size`.  Fused via tensor product (gates
   in the same row act on disjoint qubits).

3. **Cross-row loop** — advance one row at a time.  In each subsequent row,
   look for a gate that overlaps the current product's qubit set and whose
   union stays within the size budget.  Fuse via matrix product (composition).
   Restart the row scan on each successful absorption.  Stop when a full row
   pass yields no progress.

4. **Cost check** — compute the benefit ratio:

   ```
   benefit = Σ(old gate costs) / (new gate cost + ε) − 1
   ```

   Accept the fusion if `benefit >= config.benefit_margin`.  Otherwise roll
   back by restoring the original gates.

After each full sweep of `apply_gate_fusion`, call `squeeze()` and repeat
until no fusions occurred.

### Channel gate handling

Channel (noise) gates are **never** fused.  Every code path that considers a
candidate gate checks `gate.is_unitary()` and skips non-unitary gates.
Channel gates in the graph act as barriers that prevent fusion across them.

## Cost Models

The cost model decides whether a proposed fusion is beneficial.  Two
implementations are provided:

### Size-Only (`FusionConfig::size_only(max_size)`)

Binary model: gates within the qubit budget cost `1e-10` (effectively free),
larger gates cost `1.0`.  Always accepts fusions that stay within the budget
(`benefit_margin = 0`).  No hardware awareness.

Useful presets:
- `FusionConfig::default()` = `size_only(3)`
- `FusionConfig::aggressive()` = `size_only(4)`

### Hardware-Adaptive (`FusionConfig::hardware_adaptive(profile, max_size)`)

Roofline-based model using a measured `HardwareProfile`:

```
cost(gate) = max(1.0,  AI(gate) / crossover_AI)
```

- **Memory-bound** gates (AI < crossover): cost = 1.0.  Fusing always helps
  because the fused kernel has the same memory traffic but fewer launches.

- **Compute-bound** gates (AI ≥ crossover): cost > 1.0.  Fusing becomes
  increasingly expensive as the matrix gets denser.

- Gates exceeding `max_size` qubits: cost = infinity (never fused).

The crossover AI comes from the roofline profile and represents the hardware's
memory bandwidth / compute throughput balance point.

**Profile matching:** the hardware-adaptive model is only as good as its
profile.  Two conditions must hold for accurate cost decisions:

1. **Backend match** — a CPU profile must be used for CPU simulation and a
   CUDA profile for CUDA.  CPU and GPU have fundamentally different
   bandwidth/compute ratios and crossover points.

2. **Working-set size match** — the profile should be measured at the same
   statevector qubit count as the target simulation.  Effective memory
   bandwidth varies with working-set size due to cache effects (see
   [Choosing Statevector Size](tools.md#choosing-statevector-size) in the
   tools documentation).  A mismatched profile can cause the optimizer to
   accept fusions that are actually compute-bound (crossover too high) or
   reject beneficial fusions (crossover too low).

## Effective Qubit Counting

For density-matrix simulation, channel gates act on 2k virtual qubits for k
physical qubits.  The cost model uses `effective_n_qubits()` which returns
`2 * n_qubits` for channel gates and `n_qubits` for unitary gates.  This
ensures the size budget correctly accounts for the expanded gate dimensions.

## Density-Matrix Circuits

When fusing a DM circuit:

- All DM unitary gates have **even** virtual-qubit counts (2k for k physical
  qubits).  So `size_only(3)` and `size_only(2)` behave identically — use
  even size limits (4, 6) for meaningful comparisons.

- `size_only(1)` is **not** the same as "unfused" — Phase 1 always runs and
  absorbs single-qubit gates.  For a true baseline, skip `optimize()` entirely.

- Max practical fusion size for DM gates is 6 virtual qubits (3 physical
  qubits).  Size 8 puts too much pressure on kernel generation and JIT.

## Compile Time vs Execution Time Scaling

JIT compilation cost and kernel execution cost scale differently with
statevector size, and this has practical implications for when fusion pays off.

**Compile time** depends on the circuit structure — number of gates and the
matrix size of each fused gate.  It is independent of the statevector qubit
count.  A 3-qubit fused gate generates the same LLVM IR whether the
statevector has 28 or 40 qubits.

**Execution time** is proportional to the statevector size.  Each kernel
touches all 2^n amplitudes, so every additional qubit doubles the per-gate
cost.

At small statevector sizes (e.g. 28-qubit DM for 14 physical qubits), compile
time can dominate total wall time — the CUDA hw-adaptive config compiles in
~2.6 s but executes in ~360 ms, a 7x compile/exec ratio.  Aggressive fusion
(fused(6)) is even worse: 8.9 s compile for 465 ms execution, making it a net
loss in single-shot mode.

The crossover where execution overtakes compilation is around **30-32
statevector qubits** (15-16 physical qubits for DM simulation).  Beyond that
point compile cost becomes negligible and the fusion speedups translate
directly to wall-time savings:

| SV qubits | Compile/Exec ratio | Fusion wall-time impact |
|-----------|--------------------:|------------------------|
| 28        | ~7x                 | Compile dominates; aggressive fusion hurts total time |
| 30        | ~2x                 | Roughly balanced; moderate fusion breaks even |
| 32        | ~0.5x               | Execution dominates; fusion speedups fully realized |
| 36+       | negligible          | Compile is noise; optimize purely for kernel throughput |

**Implication for the cost model:** the current roofline-based cost model
optimizes kernel execution throughput, which is the right objective for
large-scale simulations (30+ SV qubits) where compile cost is amortized.  For
few-shot workloads at smaller scales, total wall time (compile + exec) may be
the better metric.

## Public API

```rust
// Full optimization pipeline.
pub fn optimize(cg: &mut CircuitGraph, config: &FusionConfig);

// Individual phases (called by optimize):
pub fn apply_size_two_fusion(cg: &mut CircuitGraph);
pub fn apply_gate_fusion(cg: &mut CircuitGraph, config: &FusionConfig, cdd_size: usize);
```

# Trajectory Simulation Refactor

## Motivation

Trajectory (Monte Carlo wavefunction) simulation exists to produce measurement
samples from noisy circuits — not to return quantum states. The current
implementation returns the last trajectory's statevector, which is not useful.
Refactoring around measurement-as-output simplifies the API, removes backend
dependence from the result type, and opens the door to performance
optimizations.

## Current problems

1. **Returns a meaningless state** — `SimulationResult<B>` contains the last
   trajectory's statevector. Earlier trajectories are overwritten.

2. **Compile time misreported** — `compile_time_s` only covers noiseless
   kernels. Noise-branch kernel compilation (~8.7 s on a 28-qubit noisy QFT)
   happens inside `execute_trajectory` and is not reported.

3. **Massive kernel duplication** — every depolarizing noise gate compiles 4
   branch kernels independently. For 812 identical depolarizing gates, 3,248
   kernels are compiled when 4 would suffice.

4. **Sequential trajectories** — trajectories are independent but executed one
   after another.

5. **No fusion** — the simulator does not apply fusion to the circuit in
   trajectory mode (fusion is available but the user must configure it; noisy
   gates act as barriers so only noiseless segments benefit).

## Design

### Measurement model

All measurements happen at the end of the circuit (deferred measurement
principle). The caller specifies which qubits to measure:

```rust
SimulationMode::Trajectory {
    n_samples: u64,             // total measurement samples requested
    measured_qubits: Vec<u32>,  // which qubits to measure
    seed: Option<u64>,
    time_budget_s: Option<f64>, // wall-time budget for branch exploration
    max_ensemble: Option<usize>, // max concurrent statevectors (memory limit)
}
```

Measurement samples from the Born-rule distribution: probability of outcome
bitstring `b` is `Σ_{j matching b on measured qubits} |ψ_j|²`. Samples are
batch-generated from explored branches, not one-per-trajectory.

### Result types

Trajectory results are backend-independent — no `QuantumState<B>`:

```rust
/// A single explored noise branch and its contribution to the histogram.
pub struct ExploredBranch {
    /// Probability weight of this branch (product of noise-branch
    /// probabilities along the path).
    pub weight: f64,
    /// Noise branch index chosen at each noisy gate in circuit order.
    pub noise_path: Vec<usize>,
    /// Number of measurement samples generated from this branch's
    /// final statevector.
    pub n_samples: u64,
}

/// Result of a trajectory / ensemble simulation.
pub struct TrajectoryResult {
    /// Measurement histogram: bitstring → count.
    pub histogram: HashMap<u64, u64>,
    /// Total measurement samples generated.
    pub n_samples: u64,
    /// Branches explored, sorted by weight descending.
    pub branches: Vec<ExploredBranch>,
    /// Sum of explored branch weights (≤ 1.0). The gap
    /// `1.0 − explored_weight` bounds the approximation error.
    pub explored_weight: f64,
    /// Kernel compilation wall time (seconds), including noise-branch
    /// kernels.
    pub compile_time_s: f64,
    /// Wall time spent on circuit simulation (all branches).
    pub simulation_time_s: f64,
    /// Wall time spent on measurement sampling.
    pub sampling_time_s: f64,
}
```

`SimulationResult<B>` changes for trajectory mode — the `state` field becomes
`None` (or the enum is restructured so trajectory results don't carry a state):

```rust
pub struct SimulationResult<B: Backend> {
    /// Final quantum state (None for trajectory mode).
    pub state: Option<QuantumState<B>>,
    /// Execution time statistics.
    pub timing: TimingStats,
    /// Kernel compilation wall time (seconds), including noise-branch kernels.
    pub compile_time_s: f64,
    /// Trajectory / ensemble simulation data (only for Trajectory mode).
    pub trajectory_data: Option<TrajectoryResult>,
}
```

### Measurement implementation (CPU)

After the final gate in each trajectory:

1. Compute marginal probabilities over the measured qubits by summing
   `|ψ_j|²` over non-measured qubit indices.
2. Sample one bitstring from the marginal distribution.
3. Record the bitstring in `TrajectoryResult::measurement`.

For k measured qubits, the marginal distribution has 2^k entries. When k is
small (typical), this is cheap. When k equals n_qubits (measure all), the
marginal is just the full probability vector.

### Measurement implementation (CUDA)

TODO — options include:
- Download probabilities to host and sample there.
- Implement a device-side marginal + sampling kernel.

For now, the CPU-side download-and-sample approach is acceptable since the
measurement cost is small relative to the circuit execution.

### Dead gate elimination

Because `QuantumCircuit` does not carry measurement instructions, the
simulator has no way to eliminate dead gates during fusion. Trajectory mode
changes this: the caller explicitly declares which qubits are measured. Gates
that cannot influence any measured qubit are dead code and can be removed
before kernel compilation.

**Algorithm** — backward liveness analysis on the circuit gate list:

1. Initialize a live-qubit set `L = { measured_qubits }`.
2. Walk the gate list in reverse order. For each gate `g`:
   - If **any** of `g.qubits()` is in `L`, the gate is **live**. Add **all**
     of `g.qubits()` to `L` (a multi-qubit gate entangles its targets, so if
     one output is observed, all inputs matter).
   - Otherwise the gate is **dead** — mark for removal.
3. Remove dead gates from the list before building the `CircuitGraph`.

**Properties:**
- Runs in O(G · k) where G = gate count, k = max gate width. Negligible
  compared to kernel compilation.
- Preserves causal ordering — removing a gate never makes a previously
  reachable gate unreachable.
- Applies to both noiseless and noisy gates. A dead noisy gate (noise on an
  unmeasured qubit with no downstream entanglement to a measured qubit) is
  eliminated along with its noise-branch kernels.
- Composable with fusion: eliminate first, then fuse the surviving gates.

**Example:** 28-qubit noisy QFT measuring only qubits 0–3. In QFT, qubit `q`
interacts with qubits `q+1, ..., n-1` via controlled-phase gates. Backward
analysis from qubits 0–3 would pull in their CP partners, but gates acting
only on high-index qubits with no path to qubits 0–3 are dead. The exact
savings depend on circuit topology, but for QFT the entanglement structure
means most qubits are reachable — the bigger wins come from circuits with
more locality (e.g., nearest-neighbor ansätze where distant qubits are
independent of the measured subset).

### Noisy gate fusion for trajectory mode

In StateVector and DensityMatrix modes, noisy gates are fusion barriers —
the `is_unitary()` guard prevents them from participating in merges. This is
correct because DM superoperators mix unitary and noise structure in ways
that don't compose simply.

In trajectory mode, however, at each noisy gate exactly one branch is
chosen: the applied operation is always a unitary `U_noise · U_gate`. This
means **noisy gates can be fused in trajectory mode** — the fusion algorithm
is identical to the noiseless case, with two additions:

1. **Branch tracking.** The fused gate carries the Cartesian product of
   its constituents' noise branches. Fusing gate A (branches `[(p_i, V_i)]`)
   with gate B (branches `[(q_j, W_j)]`) produces a fused gate with
   `|A.branches| × |B.branches|` branches:

   ```
   weight:  p_i × q_j
   unitary: (W_j · U_B) · (V_i · U_A)   // full composed unitary for branch (i,j)
   ```

   Noiseless gates contribute 1 branch (weight=1, operator=identity), so
   fusing noiseless with noisy is just the noisy gate's branch set composed
   with the noiseless unitary.

2. **Cost model uses dominant-branch AI.** The dominant branch (all noise
   operators = identity) has composed unitary = `U_B · U_A` — exactly the
   noiseless fusion product. Its AI equals the noiseless fused gate's AI.
   Since the dominant branch is what runs in the vast majority of
   trajectories (or ensemble members), this is the right value for the
   cost model. **The existing fusion algorithm and cost model work unchanged
   for trajectory mode.**

**Branch count bounds:**

- A k-qubit quantum channel has at most 4^k Kraus operators.
- In practice, the branch count is the product of constituent branch counts.
  For a size-4 fused gate with 2 noiseless and 2 depolarizing (4-branch)
  gates: 1 × 1 × 4 × 4 = 16 branches.
- With the standard fusion size limit of 4 qubits, the theoretical max is
  4^4 = 256 branches per fused gate. Practical counts are much lower when
  most constituents are noiseless.

**Implications for ensemble simulation:**

When a fused noisy gate is encountered during ensemble execution:
- Each ensemble member branches into `n_branches` new members.
- With low noise rates, the dominant (all-identity) branch carries almost all
  the weight, so pruning is highly effective — most non-dominant branches are
  immediately discarded.

### Ensemble simulation with batch sampling

The core idea: **separate circuit simulation from measurement sampling.**

Circuit simulation (expensive, O(G × 2^n) per branch) determines the final
statevector for a specific noise path. Measurement sampling (cheap,
O(2^n + N) per branch) batch-generates N bitstrings from a final statevector.
For 1 million requested samples, we do *not* simulate 1 million trajectories.
We deterministically explore the dominant noise branches, then batch-sample
from each branch's final statevector proportional to its weight.

This sits between full DM and stochastic trajectory simulation:

| Method             | Memory      | Measurement error              |
|--------------------|-------------|--------------------------------|
| Full DM            | O(4^n)      | None (exact)                   |
| Ensemble (M SVs)   | O(M · 2^n)  | Bounded by unexplored weight   |
| Stochastic traj.   | O(2^n)      | O(1/√N) sampling noise         |

#### Algorithm

**Phase 1 — Deterministic branch exploration (time-budgeted):**

1. Start with ensemble `E = [(1.0, |0⟩)]`.
2. For each gate in the circuit:
   - **Noiseless gate U**: apply U to every `|ψ_i⟩` in the ensemble.
     Ensemble size unchanged.
   - **Noisy gate** with branches `[(p_j, U_j)]`: replace each member
     `(w_i, |ψ_i⟩)` with B new members `(w_i · p_j, U_j · U · |ψ_i⟩)`.
     Ensemble grows by factor B.
   - **Prune** if `|E| > M`: sort by weight descending, keep the top M
     members. Track cumulative discarded weight.
3. After the circuit, we have `E = [(w_i, |ψ_i⟩)]` — each member is a
   fully-simulated noise path with its exact final statevector.

If a `time_budget_s` is given and time remains after the first pass, explore
additional branches that were previously pruned (re-simulate from the
branching point). More time → more branches → smaller unexplored weight →
more accurate histogram.

**Phase 2 — Batch measurement sampling (cheap):**

4. For each ensemble member `(w_i, |ψ_i⟩)`:
   - Compute marginal probabilities over `measured_qubits`.
   - Allocate `n_i = round(w_i / W_total × n_samples)` samples to this
     branch (where `W_total = Σ w_i`).
   - Batch-sample `n_i` bitstrings from the marginal distribution.
   - Record `(weight, noise_path, n_samples)` in `ExploredBranch`.
5. Merge all samples into a single histogram.

**Cost comparison:**

| Approach | Circuit simulation cost | Samples generated |
|----------|------------------------|-------------------|
| N independent trajectories | N × G × O(2^n) | N |
| Ensemble (M branches) | M × G × O(2^n) | unlimited (batch) |

For 1M samples with M=10 branches: ensemble does 10 circuit simulations
instead of 1M. The measurement sampling phase generates all 1M samples
from 10 precomputed statevectors at negligible cost.

#### Properties

- **No redundant prefix computation.** Before the first noisy gate, only one
  statevector is simulated regardless of ensemble size.
- **Deterministic where it reaches.** Explored branches are exact — no
  sampling noise in the circuit simulation. The only stochasticity is in
  the final measurement sampling.
- **Anytime refinement.** The result improves monotonically with more
  time/branches. `explored_weight` tells the caller how complete the
  approximation is.
- **Efficient at low noise rates.** With p=0.005 depolarizing noise, the
  "no error" branch carries weight (1−p)^G. For 420 noiseless + 812 noisy
  gates, the dominant branch alone has weight 0.995^812 ≈ 1.7%. But the
  top few branches collectively dominate: with M=15 ensemble members, the
  covered weight is much higher than 15 independent stochastic trajectories
  would provide.

#### Pruning strategy

**Greedy pruning**: keep the M highest-weight members, discard the rest.
Track cumulative discarded weight as `1.0 − explored_weight`.

This prioritises dominant branches, which is optimal for batch sampling:
the branches that contribute most to the measurement distribution are
explored exactly, while low-weight branches (rare error paths) are
approximated or omitted.

#### When to use

The ensemble mode is most beneficial when:
- Full DM doesn't fit (n > 15 physical qubits for F64 on a 32 GiB GPU).
- Memory holds M ≫ 1 statevectors (e.g., 28-qubit F32 SV is 2 GiB;
  a 32 GiB machine fits ~15 copies).
- Noise rate is low, so the probability mass concentrates in few branches.
- A large number of samples is requested (amortises simulation cost).
- Circuit has long noiseless segments (shared prefix is simulated once).

When M = 1, this degenerates to a single deterministic trajectory (always
picking the highest-probability noise branch). When M ≥ B^G (all branches
kept), it reproduces the exact density matrix result.

## Refactor steps

### Step 1: Restructure result types

- Make `SimulationResult.state` optional (`Option<QuantumState<B>>`), set to
  `None` for trajectory mode.
- Add `measured_qubits: Vec<u32>` to `SimulationMode::Trajectory`.
- Update `TrajectoryResult` to include `measurement: u64`.

### Step 2: Implement measurement sampling

- Write `marginal_probabilities(sv, measured_qubits) -> Vec<f64>` for CPU.
  Sums `|ψ_j|²` over non-measured qubit indices to produce a 2^k probability
  table (k = number of measured qubits).
- Write `batch_sample(probs, n_samples, rng) -> HashMap<u64, u64>` that
  draws `n_samples` bitstrings from the marginal distribution and returns a
  histogram.
- Call at the end of each explored branch instead of retaining the
  statevector state.

### Step 3: Dead gate elimination

- Implement `eliminate_dead_gates(gates, measured_qubits)` as described above.
- Call it inside `Simulator::run()` for trajectory mode, before building the
  `CircuitGraph` (so fusion and compilation only process live gates).
- The function takes `&[Arc<QuantumGate>]` and returns a filtered
  `Vec<Arc<QuantumGate>>`.

### Step 4: Fix compile time reporting

- Include noise-branch kernel compilation time in `compile_time_s`.
- Move the noise-branch pre-compilation out of `execute_trajectory` and into
  the compile phase alongside noiseless kernels (or at least time it and add
  to the reported value).

### Step 5: Deduplicate noise-branch kernels

- Noise gates with identical composed unitaries (e.g., all single-qubit
  depolarizing gates) should share compiled kernels.
- Key by `(gate_matrix, noise_branch_matrix, qubits)` or by the composed
  matrix directly.
- This reduces 3,248 kernel compilations to ~4 for a uniform depolarizing
  circuit.

### Step 6: Noisy gate fusion for trajectory mode

- Lift the `is_unitary()` barrier in `apply_size_two_fusion` and
  `apply_gate_fusion` when running in trajectory mode. The fusion functions
  need a flag or mode parameter to allow noisy gate participation.
- When fusing two noisy gates, compute the Cartesian product of their
  branch sets. Each fused branch stores the composed unitary and the
  product weight.
- When fusing a noiseless gate with a noisy gate, compose the noiseless
  unitary into each branch of the noisy gate.
- Cost model: use the dominant branch (all-identity noise) AI, which is
  the same as the noiseless fusion product's AI. No cost model changes
  needed.
- Update `QuantumGate` (or a wrapper) to support the fused branch
  representation.

### Step 7: Ensemble simulation with batch sampling

- Replace the old stochastic trajectory loop with the deterministic
  ensemble simulation described above.
- Implement Phase 1 (branch exploration with pruning) and Phase 2 (batch
  measurement sampling) as separate internal methods.
- Ensemble branching reuses the same compiled noise-branch kernels from
  step 5.
- For noiseless gates, apply the kernel to every ensemble member (loop
  over SVs).
- For noisy gates, branch each member: clone SV, apply the composed
  `U_noise · U_gate` kernel, update weight.
- Prune after each noisy gate if ensemble exceeds `max_ensemble`.
- After circuit completes, batch-sample from each surviving member's
  final SV proportional to its weight.
- Populate `TrajectoryResult` with histogram, branches, explored_weight,
  and timing breakdown.

### Step 8 (future): Parallelize ensemble members

- CPU: apply each gate to ensemble members in parallel (independent SVs,
  no shared mutable state).
- CUDA: run multiple statevectors concurrently using separate streams or
  batched kernel launches.
- Enabled by the ensemble design — members are independent during gate
  application.

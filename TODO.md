# Journal Revision TODO (ACM TQC, due 2026-04-27)

## Code Tasks

- [ ] NWQSim/SV-Sim/DM-Sim benchmarks — build Ang Li et al. simulators, run head-to-head on same circuits (R2)
- [ ] Benchmark reproducibility scripts — wrap bench binaries in scripts that produce paper tables, document fusion configs/flags/threading per baseline (R3)
- [ ] Wider qubit range — extend benchmarks beyond 30-34 qubits (R2)
- [ ] Dense vs sparse ablation — run benchmarks with --force-dense, compare against default sparsity-aware kernels (R3)
- [ ] Open-source preparation — license, contribution guide, public README (R1)

## Paper Tasks

- [ ] Formal cost model description — document benefit() internals, calibration procedure, fitted parameters (R1, R3)
- [ ] Gate swapping legality rules — algorithmic specification, correctness constraints, interaction with fusion (R3)
- [ ] Mid-circuit measurement scaling — analyze O(4^m) cost, discuss MBQC limitations (R2)
- [ ] Baseline fairness documentation — fusion on/off, optimization flags, kernel choices, threading per tool and circuit (R3)
- [ ] Table 1 — add diff between conference and journal paper (R1)
- [ ] Figure 4 — explain symbols, fix "Tranditional" typo (R1)
- [ ] Figure 12 — add legend for U1/H1/S1 vs U3/H3/S3 (R3)
- [ ] Writing fixes — duplicate ref 33, poor sentences, terminology precision (R1, R3)

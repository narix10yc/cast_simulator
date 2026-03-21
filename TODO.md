# Journal Revision TODO (ACM TQC, due 2026-04-27)

## Code Tasks

- [x] Fusion decision logging — instrument fusion optimizer to log accept/reject with cost/AI values (R1, R3)
- [x] `--force-dense` flag — set ztol=0 to disable sparsity optimization (R3)
- [x] NWQSim/SV-Sim benchmarks — built NWQ-Sim, head-to-head GPU on all 30q circuits (CAST 1.4-9.6x faster), single-gate kernel comparison (2.2x at scale). Results in `docs/nwqsim_baseline.md` (R2)
- [x] `cp` gate support — QASM parser handles controlled-phase; qft-cp-30 now works (6.8x vs NWQ-Sim)
- [x] Dense vs sparse ablation tool — `bench_ablation` binary, verified on mexp-14/17/20 (R3)
- [ ] Dense vs sparse ablation data — run full ablation on all benchmark circuits across fusion strategies (R3)
- [ ] Benchmark reproducibility scripts — wrap bench binaries in scripts that produce paper tables, document fusion configs/flags/threading per baseline (R3)
- [ ] Wider qubit range — extend benchmarks beyond 30-34 qubits; 5090 max 32q/GPU for F64 (R2)
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

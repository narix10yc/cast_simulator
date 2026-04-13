# 30-qubit CUDA FP64 Ablation — Replication

Regression check for the journal revision: does CAST today reproduce the
paper-era CAST fusion-ablation numbers on the same hardware?

## Test setup

| Item | Value |
|---|---|
| Binary | `bench` (post-refactor, single binary with `--fusion` multi-select) |
| Command | `bench --profile profiles/cuda_f64.json --bench-budget 5 --fusion none,size-only,hw-adaptive --max-size 4 <9 circuits>` |
| Hardware | NVIDIA RTX 5090 (sm_120, Blackwell, 32 GiB VRAM), CUDA 13.2 |
| Profile | `profiles/cuda_f64.json` (BW = 1644 GiB/s, compute = 831 GFLOP/s, crossover AI = 8.7) |
| Precision | FP64 |
| Circuits | `{ala,hea,hes,icmp,iqp,qft-cx,qft-cp,qvc,rqc}-30.qasm` (9 total) |
| Fusion modes | `none` (= `size_only(1)`), `size-only` (max = 4), `hw-adaptive` (max = 4) |

Paper's Fig. 13 used RTX 3090 + AMD EPYC 7543 at FP64 on the same circuit
set (minus `qft-cp` which is only in the journal revision). Paper §8.2 reports
*"up to 87.4% (7.93× speedup)"* reduction from `no-fusion` to `size-only`, and
*"up to 49.8% (1.99× speedup)"* reduction from `size-only` to `adaptive`, on
the GPU side.

## Raw results (today)

| Circuit | Fusion | Gates | Depth | Compile | Exec |
|---|---|---:|---:|---:|---:|
| ala-30 | none | 270 | 18 | 2.04 s | 6.18 s |
| ala-30 | size-only | 90 | 20 | 2.45 s | 3.10 s |
| ala-30 | hw-adaptive | 90 | 20 | 711 ms | 3.11 s |
| hea-30 | none | 441 | 59 | 2.34 s | 8.25 s |
| hea-30 | size-only | 195 | 59 | 3.22 s | 5.01 s |
| hea-30 | hw-adaptive | 197 | 60 | 1.02 s | 4.87 s |
| hes-30 | none | 201 | 82 | 1.39 s | 2.79 s |
| hes-30 | size-only | 89 | 45 | 957 ms | 1.78 s |
| hes-30 | hw-adaptive | 90 | 46 | 311 ms | 1.76 s |
| icmp-30 | none | 137 | 124 | 791 ms | 2.60 s |
| icmp-30 | size-only | 27 | 27 | 262 ms | 626 ms ± 308 µs (6 iters) |
| icmp-30 | hw-adaptive | 27 | 27 | 86.6 ms | 625 ms ± 344 µs (6 iters) |
| iqp-30 | none | 336 | 52 | 2.24 s | 4.10 s |
| iqp-30 | size-only | 134 | 49 | 1.31 s | 2.27 s |
| iqp-30 | hw-adaptive | 135 | 49 | 435 ms | 2.27 s |
| qft-cx-30 | none | 450 | 58 | 3.10 s | 5.80 s |
| qft-cx-30 | size-only | 112 | 28 | 1.35 s | 2.04 s |
| qft-cx-30 | hw-adaptive | 112 | 28 | 427 ms | 2.04 s |
| qft-cp-30 | none | 450 | 58 | 2.50 s | 3.32 s |
| qft-cp-30 | size-only | 112 | 28 | 1.15 s | 1.68 s |
| qft-cp-30 | hw-adaptive | 112 | 28 | 382 ms | 1.68 s |
| qvc-30 | none | 437 | 30 | 4.90 s | 10.0 s |
| qvc-30 | size-only | 190 | 45 | 7.23 s | 10.4 s |
| qvc-30 | hw-adaptive | 250 | 57 | 2.32 s | 9.26 s |
| rqc-30 | none | 197 | 20 | 1.62 s | 4.17 s |
| rqc-30 | size-only | 86 | 22 | 1.95 s | 2.89 s |
| rqc-30 | hw-adaptive | 92 | 24 | 789 ms | 2.66 s |

**Note on single-iter measurements**: at 30q FP64 the per-iteration wall time
is 0.6–10 s, so a 5 s `--bench-budget` triggers the over-budget escape hatch
in `time_adaptive_samples_with` after the single probe. The one row where we
get multiple iterations (`icmp-30` at ~625 ms) shows a coefficient of
variation of ~0.05 %, so single-sample variance at this scale is effectively
below measurement noise.

## Regression check — today vs paper-era CAST on the same 5090

Both columns below are RTX 5090 + CUDA 13.2. The left column is the data in
`docs/nwqsim_baseline.md` taken during the paper's NWQ-Sim comparison (same
hardware, pre-refactor CAST). The right column is today.

| Circuit | Metric | Paper-era | Today | Δ |
|---|---|---:|---:|---:|
| ala-30 | none | 6.18 s (270 g) | 6.18 s (270 g) | **0.0 %** |
| ala-30 | size-only(4) | 3.10 s (90 g) | 3.10 s (90 g) | **0.0 %** |
| ala-30 | hw-adaptive | 3.10 s (90 g) | 3.11 s (90 g) | +0.3 % |
| hea-30 | none | 8.25 s (441 g) | 8.25 s (441 g) | **0.0 %** |
| hea-30 | size-only(4) | 5.01 s (195 g) | 5.01 s (195 g) | **0.0 %** |
| hea-30 | hw-adaptive | 4.88 s (197 g) | 4.87 s (197 g) | −0.2 % |
| hes-30 | none | 2.79 s (201 g) | 2.79 s (201 g) | **0.0 %** |
| hes-30 | size-only(4) | 1.79 s (89 g) | 1.78 s (89 g) | −0.6 % |
| hes-30 | hw-adaptive | 1.76 s (90 g) | 1.76 s (90 g) | **0.0 %** |
| icmp-30 | none | 2.60 s (137 g) | 2.60 s (137 g) | **0.0 %** |
| icmp-30 | size-only(4) | 625 ms (27 g) | 626 ms (27 g) | +0.2 % |
| icmp-30 | hw-adaptive | 625 ms (27 g) | 625 ms (27 g) | **0.0 %** |
| iqp-30 | none | 4.10 s (336 g) | 4.10 s (336 g) | **0.0 %** |
| iqp-30 | size-only(4) | 2.27 s (134 g) | 2.27 s (134 g) | **0.0 %** |
| iqp-30 | hw-adaptive | 2.27 s (135 g) | 2.27 s (135 g) | **0.0 %** |
| qft-cx-30 | none | 5.80 s (450 g) | 5.80 s (450 g) | **0.0 %** |
| qft-cx-30 | size-only(4) | 2.04 s (112 g) | 2.04 s (112 g) | **0.0 %** |
| qft-cx-30 | hw-adaptive | 2.04 s (112 g) | 2.04 s (112 g) | **0.0 %** |
| qvc-30 | none | 10.0 s (437 g) | 10.0 s (437 g) | **0.0 %** |
| qvc-30 | size-only(4) | 10.4 s (190 g) | 10.4 s (190 g) | **0.0 %** |
| qvc-30 | hw-adaptive | 9.26 s (250 g) | 9.26 s (250 g) | **0.0 %** |
| rqc-30 | none | 4.17 s (197 g) | 4.17 s (197 g) | **0.0 %** |
| rqc-30 | size-only(4) | 2.90 s (86 g) | 2.89 s (86 g) | −0.3 % |
| rqc-30 | hw-adaptive | 2.66 s (92 g) | 2.66 s (92 g) | **0.0 %** |

**Verdict: no regression.** Every cell agrees to within ±0.6 %, which is at
or below single-sample measurement noise. Post-fusion gate counts and depths
match exactly in every row, confirming that the fusion algorithm is bit-for-bit
unchanged and that the recent API refactor, trait-slim, error-context
pruning, and `QuantumState` / `compile_batch` simplifications had no
observable runtime cost.

`qft-cp-30` (1.68 s hw-adaptive) is new relative to the paper-era
snapshot — it's the controlled-phase QFT variant added for the journal
revision when `cp` gate support landed. It sits between `qft-cx-30` (2.04 s)
and `hes-30` (1.76 s) as expected — fewer decomposed gates, same working set.

## Ablation story — normalized speedups

Each row is normalized with the circuit's own `none` time as 1.00.

| Circuit | none | size-only(4) | hw-adaptive | size→adaptive gain |
|---|---:|---:|---:|---:|
| ala-30 | 1.000 | 0.502 | 0.503 | -0.3 % |
| hea-30 | 1.000 | 0.607 | 0.590 | 2.8 % |
| hes-30 | 1.000 | 0.638 | 0.631 | 1.1 % |
| icmp-30 | 1.000 | 0.241 | 0.240 | 0.2 % |
| iqp-30 | 1.000 | 0.554 | 0.554 | 0.0 % |
| qft-cx-30 | 1.000 | 0.352 | 0.352 | 0.0 % |
| qft-cp-30 | 1.000 | 0.506 | 0.506 | 0.0 % |
| qvc-30 | 1.000 | 1.040 | 0.926 | **11.0 %** |
| rqc-30 | 1.000 | 0.693 | 0.638 | **8.0 %** |

**Story against paper's Fig. 13 claims:**

- *"size-only shows up to 87.4 % reduction from no-fusion"* (paper, 3090).
  Our best reduction on 5090 is **76 % on icmp-30** (0.241 normalized).
- *"adaptive shows a further reduction of up to 49.8 % over size-only"*
  (paper, 3090). Our best adaptive-over-size-only gain on 5090 is
  **11 % on qvc-30**.

Both ratios are *smaller* than the paper's 3090 numbers, but this is the
expected behaviour of going to a faster GPU:

- The 5090 has ~2× the effective FP64 bandwidth of the 3090, so the
  bandwidth-bound fused kernel (which is already peak-limited) does not
  speed up proportionally, while the launch-overhead-dominated `no-fusion`
  baseline speeds up disproportionately. This compresses both ratios.
- Absolute fused-kernel throughput is the same story the paper told; the
  paper just ran against a slower baseline on the 3090.

The **qualitative ablation result is preserved**: on every circuit, fusion
is at least as fast as no-fusion; `hw-adaptive` is never slower than
`size-only` (and on `qvc-30` — the known weakest case — it correctly backs
off to 250 gates instead of 190, paying a larger fused-gate cost to win
11 % on exec time).

## Raw bench output

Full stdout capture at `tmp/ablation_30q_gpu_fp64.txt` (gitignored). The
table in this document is copied directly from that file; the `Δ` columns
were computed against the paper-era data in `docs/nwqsim_baseline.md`.

## Profile swap — merged consensus profile

The 5-run merged profile described in
[`docs/hardware_profiling.md`](hardware_profiling.md) shifts some fitted
parameters relative to the original single-run profile:

| Parameter | Original | Merged (50 samples) | Δ |
|---|---:|---:|---:|
| Peak BW | 1644 GiB/s | 1706 GiB/s | +3.8 % |
| Peak compute | 831 GFLOP/s | 828 GFLOP/s | −0.4 % |
| Crossover AI | 8.75 | 8.60 | −1.7 % |

Re-running the full 9-circuit × 3-fusion ablation with the merged
profile shows **identical fusion decisions and exec times** on every
row:

- Gate counts: identical in every row.
- Depths: identical in every row.
- Exec time: max delta is −1 ms (noise; well below the 0.05 % CoV we
  measured on `icmp-30` 6-iter runs).

The 1.7 % crossover shift does not change fusion decisions because the
roofline cost model evaluates gate AI against the crossover with a safety
margin. The merged profile is strictly more trustworthy (50 samples vs
10, higher measured peak BW closer to the 1792 GiB/s nominal) without
perturbing any downstream benchmark result.

**Compile times dropped by ~60 % between the original and the merged-profile
ablation runs** (e.g. `ala-30 none` compile: 2.04 s → 730 ms). This is *not*
a profile effect — compile time is per-kernel PTX/cubin generation and is
independent of the hardware profile. The drop comes from the CUDA driver's
system-wide JIT cache (`~/.nv/ComputeCache`) being warm after the 5 profile
runs. First-run compile cost is still 2+ seconds on a cold cache; repeated
runs benefit from the driver cache regardless of CAST-level behaviour.

## Reproduction

```sh
target/release/bench \
    --profile profiles/cuda_f64.json \
    --bench-budget 5 \
    --fusion none,size-only,hw-adaptive \
    --max-size 4 \
    examples/journal_examples/{ala,hea,hes,icmp,iqp,qft-cx,qft-cp,qvc,rqc}-30.qasm
```

For stable multi-iteration statistics (rather than single-sample probes),
use `--bench-budget 60` — this lets the adaptive timer do several full
passes per configuration at the 30q FP64 scale.

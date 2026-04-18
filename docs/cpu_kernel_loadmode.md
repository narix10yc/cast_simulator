# CPU Kernel Load Mode: Mega vs Tiled

## The problem

Each CPU kernel must load `vec_size` scalars from the statevector into
native SIMD registers, rearrange them into `LK` amplitude vector pairs
(re/im), apply the matrix-vector product, then scatter the results back.
The load and store phases dominate when the matrix is sparse (most entries
folded away by InstCombine), so their cost matters.

Two rearrangement strategies exist:

| Mode | Loads | Rearrangement |
|------|-------|---------------|
| **Mega** | 1 wide `<vec_size x scalar>` load (LLVM legalizes to `vec_size/S` native loads) | `2*LK` ShuffleVector ops on the wide vector |
| **Tiled** | `vec_size/S` explicit native-width chunk loads | `4*LK*S` scalar extract+insert ops |

The number of memory loads is identical (`vec_size/S` native-width
reads either way).  The difference is purely in the data rearrangement
cost:

- **Mega** relies on LLVM's shuffle lowering to turn wide-vector
  ShuffleVector into native shuffle/blend/permute instructions.  The
  cost depends on ISA support (e.g. AVX-512 `vperm*` can do 16-lane
  shuffles in one instruction; NEON has no cross-register shuffle).

- **Tiled** uses scalar extract+insert which is portable and predictable,
  but scales linearly with `S` (the number of SIMD lanes).

Phase 3 (store) mirrors the load: Mega uses a merge-tree of
ShuffleVectors + interleave; Tiled uses the inverse scatter.

## Why not decide automatically

The optimal mode depends on factors that vary across ISAs and
microarchitectures:

1. **Legalized shuffle cost** — how many native instructions does LLVM
   emit for a ShuffleVector on a `<vec_size x scalar>` vector when
   `vec_size > S`?  This is ISA-specific (AVX-512 has rich permute
   support; NEON/SVE do not).

2. **Scalar extract/insert cost** — varies from 1 uop (x86 with AVX-512)
   to 2-3 uops (NEON lane moves) per operation.

3. **Register pressure** — Mega's wide vector occupies multiple physical
   registers simultaneously; Tiled works on one chunk at a time.

4. **Interaction with Block matvec** — Block mode's volatile
   retire/reload may shift the register-pressure balance.

A fixed heuristic (the previous `S <= 8` threshold) cannot capture
these tradeoffs across all targets.  The right approach is empirical:
benchmark both modes on the target hardware and configure accordingly,
similar to how the hardware roofline profile drives adaptive fusion
decisions.

## Current default

`CAST_CPU_LOADMODE=mega` (the default when unset).

Mega is the conservative default: it produces correct code on all ISAs,
and when `vec_size == S` (no lo-bits — the common case for small gates)
the wide load IS a native load and the shuffles are single-instruction.

## Configuration

Set the `CAST_CPU_LOADMODE` environment variable:

```sh
# Default: wide load + LLVM shuffle lowering
CAST_CPU_LOADMODE=mega cargo run --bin bench_compile --release -- ...

# Alternative: native-width chunk loads + scalar gather
CAST_CPU_LOADMODE=tiled cargo run --bin bench_compile --release -- ...
```

The value is read once at first kernel generation and cached for the
process lifetime.

## Relevant parameters

For a given kernel, the load cost depends on:

| Symbol | Meaning | Source |
|--------|---------|--------|
| `simd_s` | log2 of native SIMD lanes | `get_simd_s(simd_width, precision)` |
| `S` | SIMD lanes (scalars per native register) | `2^simd_s` |
| `vec_size` | Total scalars in the Phase 1 load region | `2^sep_bit` from `BitLayout(qubits, simd_s)` |
| `LK` | Number of lo-partitions (output rows per hi) | `2^lk` from `BitLayout(qubits, simd_s)` |
| `vec_size/S` | Number of native loads (same for both modes) | — |

The rearrangement cost per hi-combination (load side only):

| Mode | Operation count |
|------|----------------|
| Mega | `2*LK` ShuffleVector (each on `<vec_size x scalar>`, legalized) |
| Tiled | `4*LK*S` scalar extract + insert |

S values for each configuration:

| Precision | W128 | W256 | W512 |
|-----------|------|------|------|
| F64       | 2    | 4    | 8    |
| F32       | 4    | 8    | 16   |

## Future work

Add a load-mode benchmark sweep to `bench_compile` that measures both
modes across the (precision, SIMD width, k, qubit placement) parameter
space.  The results can inform a per-target default or an auto-selection
heuristic integrated into the hardware profile, analogous to the
roofline-based fusion cost model.

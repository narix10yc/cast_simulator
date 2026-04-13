# CAST Simulator

CAST is a Rust-first quantum circuit simulator with CPU and optional CUDA backends.

## Status

The old standalone C++/CMake codebase has been removed.

- Main development flow is `cargo`-based.
- New simulator features should be implemented in Rust.
- The C++ sources under `src/cpp/` remain as a bridge for the current LLVM/CUDA backends.

## Repository Layout

- `src/`: Rust library code
- `src/bin/`: Rust command-line tools and experiments
- `tests/`: end-to-end Rust integration tests
- `src/cpp/`: C++ FFI/JIT layer used by the current Rust build
- `scripts/build_llvm.sh`: helper for building a local LLVM install

## Requirements

For the current Rust build:

- Rust toolchain
- A C++17 compiler available as `c++`, or specified explicitly via `CXX=...`
- A working LLVM installation with `llvm-config`

For building LLVM locally with the helper script:

- `cmake`
- `ninja`

For CUDA support:

- CUDA toolkit and driver
- An LLVM build that includes the `NVPTX` target

## LLVM Setup

The Rust build requires:

```sh
export LLVM_CONFIG=/path/to/llvm-config
```

The build script reads `LLVM_CONFIG` directly from the environment. If it is not set, `cargo build`, `cargo check`, and `cargo test` fail immediately.

You can either use an existing LLVM install or build one locally with [`scripts/build_llvm.sh`](scripts/build_llvm.sh).

### Option 1: Use an Existing LLVM Install

If you already have a suitable LLVM install:

```sh
export LLVM_CONFIG=/path/to/bin/llvm-config
```

For CUDA builds, make sure that LLVM was built with `NVPTX` enabled.

### Option 2: Build LLVM Locally

Download and unpack a recent `llvm-project-${version}.src` release somewhere under `~/llvm/`, then run:

```sh
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src
```

That default mode is aimed at the current Rust workflow:

- release build only
- targets `Native;NVPTX`
- install layout under the chosen LLVM root: `release-build/` and `release-install/`

Useful variants:

```sh
# CPU-only LLVM, no NVPTX
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src --cpu-only

# Verify an existing install against the requested target layout
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src --verify-targets

# Remove old release-build/ and debug-build/ directories after a successful install
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src --clean

# Also build a debug LLVM install
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src --with-debug

# Include clang/lld/lldb in the release install
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src --with-clang-tools
```

After a successful build, set:

```sh
export LLVM_CONFIG=~/llvm/${version}/release-install/bin/llvm-config
```

The helper supports two maintenance-oriented modes:

- `--verify-targets` checks that `release-install/bin/llvm-config` exists and that the install contains the targets implied by your flags, such as `NVPTX` in the default mode.
- `--clean` removes only `release-build/` and `debug-build/`. It does not remove `release-install/`, `debug-install/`, or the unpacked LLVM source tree.
- The helper intentionally accepts only the new long-form options. Old alias flags such as `-minimal` and `-native-only` are no longer supported.

## Build Process

The current build is Cargo-driven, and it compiles the bridge layer through [`build.rs`](build.rs):

1. Cargo starts the Rust build.
2. [`build.rs`](build.rs) chooses the C++ compiler:
   `CXX` if you set it, otherwise plain `c++`.
3. [`build.rs`](build.rs) runs `llvm-config` from `LLVM_CONFIG` to obtain:
   - C++ compile flags
   - LLVM link flags
   - LLVM library selections
4. [`build.rs`](build.rs) compiles the bridge sources under `src/cpp/` into static archives and links them into the Rust crate.
5. With the `cuda` feature enabled, [`build.rs`](build.rs) also builds the CUDA bridge and links CUDA-specific dependencies.

Important details:

- `LLVM_CONFIG` selects the LLVM installation.
- `LLVM_CONFIG` does not select the C++ compiler.
- On macOS, the build script automatically adds a Homebrew library search path when LLVM requires `zstd`, so no manual `LIBRARY_PATH` workaround is needed on this machine.

## Rust Build And Test

For the current machine, the normal flow is:

```sh
source ~/.zshrc
cargo check
cargo test
```

If you want to use a non-default C++ compiler for the bridge layer:

```sh
source ~/.zshrc
export CXX=/path/to/your/c++
cargo test
```

## Hardware Profiling

The adaptive roofline profiler (`profile_hw`) measures the memory/compute
crossover point for gate simulation kernels.  It sweeps gate kernels across a
range of arithmetic intensities, fits a two-segment piecewise roofline model,
and outputs a `HardwareProfile` with peak bandwidth, peak compute, the
memory-bound slope, and the crossover AI — plus raw sweep data for plotting.

The crossover AI determines when gate fusion helps vs. hurts: fusing gates
whose combined AI stays below the crossover is free (memory-bound); fusing past
it pushes into the compute-bound regime with diminishing returns.

### Usage

```sh
cargo run --bin profile_hw --release -- --help

# CPU only (all precisions)
cargo run --bin profile_hw --release

# CPU + CUDA, all precisions
cargo run --bin profile_hw --features cuda --release

# CUDA only, F32, 60s budget, 30-qubit statevector, save profiles
cargo run --bin profile_hw --features cuda --release -- \
      --backend cuda --precision f32 -n 30 --budget 60 \
      --save-profiles profiles/

# Override CPU thread count
CAST_NUM_THREADS=32 cargo run --bin profile_hw --release
```

### Cached Profiles

Saved profiles are JSON files in `profiles/` (gitignored) and can be loaded by
`bench` via `--profile profiles/cuda_f64.json` to skip re-profiling:

```sh
cargo run --bin bench --features cuda --release -- \
      --backend cuda --profile profiles/cuda_f64.json \
      examples/journal_examples/qft-cx-30.qasm
```

### Key Source Files

- [`src/profile.rs`](src/profile.rs) — Adaptive sweep engine, `measure()`, `measure_cpu()`, `measure_cuda()`
- [`src/cost_model.rs`](src/cost_model.rs) — `HardwareProfile`, `HardwareAdaptiveCostModel`, `FusionConfig`
- [`src/bin/profile_hw.rs`](src/bin/profile_hw.rs) — CLI for running profiles
- [`src/bin/bench.rs`](src/bin/bench.rs) — Fusion benchmark, consumes profiles

## CUDA Build And Test

CUDA support is behind the Cargo feature `cuda`.

Environment variables used by the build:

- `CUDA_PATH`: optional CUDA toolkit root; if unset, the build probes common locations
- `NVJITLINK_LIB`: optional directory containing `libnvJitLink`

Build or test with CUDA enabled:

```sh
cargo test --features cuda --test cpu_cuda_compare -- --ignored
```

Relevant entry points:

- [`tests/cpu_cuda_compare.rs`](tests/cpu_cuda_compare.rs)

The CUDA comparison tests are `#[ignore]` by default because they require a CUDA-capable GPU. You can override the SM target for the tests with:

```sh
CUDA_SM=80 cargo test --features cuda --test cpu_cuda_compare -- --ignored
```

## Citing

If you find this work useful, please consider citing:

```bibtex
@article{lu2025versatile,
  title={Versatile Cross-platform Compilation Toolchain for Schr$\backslash$" odinger-style Quantum Circuit Simulation},
  author={Lu, Yuncheng and Liang, Shuang and Fan, Hongxiang and Guo, Ce and Luk, Wayne and Kelly, Paul HJ},
  journal={arXiv preprint arXiv:2503.19894},
  year={2025}
}
```

## License

CAST is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for the
full text and [NOTICE](NOTICE) for attribution and third-party dependency
information.

CAST links against LLVM (Apache 2.0 with LLVM Exceptions) and, optionally,
the NVIDIA CUDA Toolkit at build time. Neither is redistributed with this
repository. The `external/` directory contains third-party assets used for
benchmarking and reference only; each retains its own license.

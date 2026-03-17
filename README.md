# CAST Simulator

CAST is a Rust-first quantum circuit simulator with CPU and optional CUDA backends.

## Status

The migration from the older C++-centric codebase to Rust is nearly complete.

- New features and extensions should be implemented in Rust.
- The C++ sources under `src/cpp/` are a temporary bridge for the current JIT backends.
- The legacy CMake/C++ development flow is being retired and will eventually be removed.

Today, the main developer workflow is `cargo`-based.

## Repository Layout

- `src/`: Rust library code
- `src/bin/`: Rust command-line tools and experiments
- `tests/`: end-to-end Rust integration tests
- `src/cpp/`: temporary C++ FFI/JIT layer used by the current Rust build
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

# Also build a debug LLVM install for legacy/debug CMake workflows
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src --with-debug

# Include clang/lld/lldb in the release install
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src --with-clang-tools
```

After a successful build, set:

```sh
export LLVM_CONFIG=~/llvm/${version}/release-install/bin/llvm-config
```

The helper also prints legacy CMake variables for the remaining transitional pieces:

```sh
export CAST_LLVM_ROOT=~/llvm/${version}/release-install
export CAST_DEV_LLVM_ROOT=~/llvm/${version}
```

Only the old CMake-based flow still cares about `CAST_LLVM_ROOT` and `CAST_DEV_LLVM_ROOT`. The Rust build path uses `LLVM_CONFIG`.

The helper supports two maintenance-oriented modes:

- `--verify-targets` checks that `release-install/bin/llvm-config` exists and that the install contains the targets implied by your flags, such as `NVPTX` in the default mode.
- `--clean` removes only `release-build/` and `debug-build/`. It does not remove `release-install/`, `debug-install/`, or the unpacked LLVM source tree.
- The helper intentionally accepts only the new long-form options. Old alias flags such as `-minimal` and `-native-only` are no longer supported.

## Build Process

The current build is Cargo-driven, but it still compiles a temporary C++ bridge through [`build.rs`](build.rs):

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

If you want to use a non-default C++ compiler for the temporary bridge layer:

```sh
source ~/.zshrc
export CXX=/path/to/your/c++
cargo test
```

## CPU Tools

The main CPU profiling and crossover sweep lives in [`src/bin/cpu_crossover.rs`](src/bin/cpu_crossover.rs).

Examples:

```sh
cargo run --bin cpu_crossover --release -- --help
cargo run --bin cpu_crossover --release -- --n-qubits 30 --threads 10 --budget-secs 120
```

There is also a small scratch binary in [`src/bin/scratch.rs`](src/bin/scratch.rs):

```sh
cargo run --bin scratch
```

## CUDA Build And Test

CUDA support is behind the Cargo feature `cuda`.

Environment variables used by the build:

- `CUDA_PATH`: optional CUDA toolkit root; if unset, the build probes common locations
- `NVJITLINK_LIB`: optional directory containing `libnvJitLink`

Build or test with CUDA enabled:

```sh
cargo test --features cuda --test cpu_cuda_compare -- --ignored
cargo run --features cuda --bin cuda_crossover --release -- --help
```

Relevant entry points:

- [`tests/cpu_cuda_compare.rs`](tests/cpu_cuda_compare.rs)
- [`src/bin/cuda_crossover.rs`](src/bin/cuda_crossover.rs)

The CUDA comparison tests are `#[ignore]` by default because they require a CUDA-capable GPU. You can override the SM target for the tests with:

```sh
CUDA_SM=80 cargo test --features cuda --test cpu_cuda_compare -- --ignored
```

For the crossover binary, use `--sm`:

```sh
cargo run --features cuda --bin cuda_crossover --release -- --sm 80
```

## Transitional Notes

- The old README sections about CMake demos, Python bindings, and C++-only developer flows are intentionally removed from the main path.
- The remaining CMake variables and dual-install LLVM layout are kept only because some transitional code still expects them.
- As the migration finishes, the expectation is that LLVM and backend integration remain, but new simulator functionality is added at the Rust layer only.

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

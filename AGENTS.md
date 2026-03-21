# Build Notes

- Primary workflow is Cargo-based.
- The Rust build compiles the C++ bridge in `src/cpp/` through `build.rs`.
- `LLVM_CONFIG` selects the LLVM installation used for build flags and linking.
- The C++ compiler is `CXX` if set, otherwise plain `c++`.

# Developer Setup

- Each developer must point `LLVM_CONFIG` at a locally installed `llvm-config`.
- Example:

```sh
export LLVM_CONFIG=/path/to/llvm/bin/llvm-config
```

- For CUDA builds, that LLVM install must include the `NVPTX` target.
- If a developer wants a non-default C++ compiler for the bridge layer, set:

```sh
export CXX=/path/to/c++
```

# Quick Build

```sh
cargo check
cargo test
```

- If `LLVM_CONFIG` is set in the developer's shell startup files, source those first.
- Otherwise export `LLVM_CONFIG` explicitly for the current shell before running Cargo.

# LLVM Helper

- Local LLVM helper: `scripts/build_llvm.sh`
- Default mode builds release LLVM with `Native;NVPTX`
- Useful maintenance:
  - `scripts/build_llvm.sh <llvm-src-dir> --verify-targets`
  - `scripts/build_llvm.sh <llvm-src-dir> --clean`
  - `scripts/build_llvm.sh <llvm-src-dir> --with-clang-tools`

- Typical local build flow:

```sh
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src
export LLVM_CONFIG=~/llvm/${version}/release-install/bin/llvm-config
```

# Notes

- `build.rs` already handles the macOS/Homebrew `zstd` link path automatically, so developers should not need to add a manual `LIBRARY_PATH` workaround for that case.
- Editor settings such as VS Code `rust-analyzer`, `clangd`, or per-machine LLVM paths are local environment concerns and should not be treated as repository-wide defaults.

# Further Documentation

- **Architecture, data flow, kernel details:** see [docs/architecture.md](docs/architecture.md)
- **CLI tools (profile_hw, bench_fusion, bench_ablation, bench_noisy_qft):** see [docs/tools.md](docs/tools.md)
- **Fusion algorithm and cost model:** see [docs/fusion.md](docs/fusion.md)
- **Noisy simulation and density matrices:** see [docs/noise.md](docs/noise.md)
- **NWQ-Sim baseline comparison:** see [docs/nwqsim_baseline.md](docs/nwqsim_baseline.md)

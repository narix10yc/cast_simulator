# Build Notes

- Primary workflow is Cargo-based.
- The Rust build still compiles the temporary C++ bridge in `src/cpp/` through `build.rs`.
- `LLVM_CONFIG` selects the LLVM installation used for build flags and linking.
- The C++ compiler is `CXX` if set, otherwise plain `c++`.

# Local Machine Setup

- Current LLVM install: `/Users/yc/llvm/22.1.1/release-install`
- Current `llvm-config`: `/Users/yc/llvm/22.1.1/release-install/bin/llvm-config`
- `~/.zshrc` exports `LLVM_CONFIG` to that path.

# Quick Build

```sh
source ~/.zshrc
cargo check
cargo test
```

# LLVM Helper

- Local LLVM helper: `scripts/build_llvm.sh`
- Default mode builds release LLVM with `Native;NVPTX`
- Useful maintenance:
  - `scripts/build_llvm.sh <llvm-src-dir> --verify-targets`
  - `scripts/build_llvm.sh <llvm-src-dir> --clean`

# Current Notes

- `build.rs` already handles the macOS/Homebrew `zstd` link path automatically, so no manual `LIBRARY_PATH` workaround is needed on this machine.
- VS Code workspace settings were updated locally so rust-analyzer uses the same LLVM 22 install.

# Rust Refactor Plan

This branch prepares a staged migration rather than a one-shot rewrite.

## Target layout

```text
Cargo.toml
src/
bin/
cpp/
  include/
  src/
  tests/
```

## Ownership split

- Top-level Rust code will own core semantics: data structures, IR, planning, optimization, and CPU execution.
- `cpp/` will own platform-facing integrations: LLVM, CUDA, and stable C++ wrappers.
- The language boundary should be a narrow C ABI, not direct template-heavy interop.

## Header split

- `cpp/include/cast/` remains the compatibility include root for now.
- `cpp/include/cast/Detail/` is the new home for non-stable internal headers.
- Public API review should keep user-facing includes out of `Detail/`.

## Current to future mapping

- `cpp/src/Core` -> top-level Rust library modules and later Rust crates if needed
- `cpp/src/CPU` -> top-level Rust CPU implementation
- `cpp/src/CUDA` -> `cpp/src/CUDA`
- `cpp/src/FPGA` -> `cpp/src/FPGA`
- `cpp/src/OpenQASM` -> keep in C++ for now
- `cpp/include/cast/Core` -> future stable API wrappers in `cpp/include/cast/api`
- `cpp/include/cast/CUDA` -> `cpp/include/cast/cuda`
- `cpp/include/cast/CPU` -> wrapper layer after Rust CPU interfaces settle

## Migration order

1. Prepare tree and docs without moving working code.
2. Introduce top-level Cargo package and first Rust core module.
3. Move pure semantic types before runtime integration.
4. Move planner and optimizer.
5. Move CPU backend.
6. Shrink C++ to wrappers around LLVM and CUDA.

## Rule for this branch

Until the Rust API boundary is reviewed, existing CMake targets stay in place under `cpp/`.

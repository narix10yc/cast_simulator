# Build Notes

- Primary workflow is Cargo-based.
- The Rust build compiles the C++ bridge in `src/cpp/` through `build.rs`.
- `LLVM_CONFIG` selects the LLVM installation used for build flags and linking.
- The C++ compiler is `CXX` if set, otherwise plain `c++`.

## Developer Setup

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

- Query local memory to see if `LLVM_CONFIG` path is specified there.

## Code Format & Style

### Section separators

Use section separators in files with distinct logical sections:

```rust
// ---------------------------------------------------------------------------
// Section name
// ---------------------------------------------------------------------------
```

Do not add excessive separators — only where they genuinely clarify structure (e.g. separating types from implementation, or grouping related functions).

### Function arguments

When a function takes a closure or function argument, put it as the **last parameter**. This enables trailing-closure style at call sites.

### Error handling

- Use `anyhow` for error propagation. The `anyhow::` namespace should often be preserved. For example, `anyhow::Result<T>` -- not `use anyhow::Result`, which shadows `std::Result`.
- `use anyhow::Context;` is allowed -- grants methods such as `with_context`. The macros `anyhow::bail!` and `anyhow::ensure!` should keep the `anyhow::` namespace.

## General

- Use US English spelling in all code, comments, and documentation (e.g. "color" not "colour", "serialize" not "serialise", "labeled" not "labelled").
- Prefer dispatch methods that delegate to helpers over large match blocks with inline logic.
- Keep closures short; hoist complex logic into named functions.
- Think twice before adding helper functions. Can this helper function be inline? Is this helper function already defined elsewhere?

# TODO

No active TODO items are currently tracked here.

## Recently Completed

- C++ FFI cleanup and internal API migration (`91e8c52`)
  - Renamed internal C++ headers from `.h` to `.hpp`.
  - Moved CPU internals into `cast::cpu`.
  - Moved CUDA internals into `cast::cuda`.
  - Added C++-native internal types in `src/cpp/internal/types.hpp`.
  - Kept Rust-facing C ABI headers and exported `cast_*` functions stable.
  - Added lightweight ABI guardrails at the FFI boundary.


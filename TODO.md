# C++ FFI Cleanup and Internal API Migration

## Goal

Keep the Rust-facing C ABI stable while making the C++ side read like C++.

The public boundary remains:

- `src/cpp/include/ffi_types.h`
- `src/cpp/include/ffi_cpu.h`
- `src/cpp/include/ffi_cuda.h`
- `extern "C"` functions implemented in `cpu.cpp`, `cuda.cpp`, and `cuda_exec.cpp`

Everything behind that boundary should move toward C++ naming, namespaces, and
internal types. The end state is that internal C++ headers do not use the C ABI
headers as their native data model.

## Non-goals

- Do not change exported `cast_*` FFI function names.
- Do not namespace declarations inside `ffi_*.h`; those headers must remain C-compatible.
- Do not change Rust FFI declarations unless the ABI itself intentionally changes.
- Do not introduce generated bindings as part of this cleanup.

## Commit 1: Rename Internal C++ Headers `.h` -> `.hpp`

**Status:** Complete.

Mechanical-only commit. No namespace, type, or behavior changes.

**11 files renamed** with `git mv`:

| Old | New |
|---|---|
| `src/cpp/internal/err_buf.h` | `src/cpp/internal/err_buf.hpp` |
| `src/cpp/cpu/cpu_gen.h` | `src/cpp/cpu/cpu_gen.hpp` |
| `src/cpp/cpu/cpu_jit.h` | `src/cpp/cpu/cpu_jit.hpp` |
| `src/cpp/cpu/internal/util.h` | `src/cpp/cpu/internal/util.hpp` |
| `src/cpp/cpu/internal/bit_layout.h` | `src/cpp/cpu/internal/bit_layout.hpp` |
| `src/cpp/cpu/internal/shuffle_masks.h` | `src/cpp/cpu/internal/shuffle_masks.hpp` |
| `src/cpp/cpu/internal/matrix_data.h` | `src/cpp/cpu/internal/matrix_data.hpp` |
| `src/cpp/cpu/internal/kernel_codegen.h` | `src/cpp/cpu/internal/kernel_codegen.hpp` |
| `src/cpp/cuda/cuda_gen.h` | `src/cpp/cuda/cuda_gen.hpp` |
| `src/cpp/cuda/cuda_jit.h` | `src/cpp/cuda/cuda_jit.hpp` |
| `src/cpp/cuda/cuda_util.h` | `src/cpp/cuda/cuda_util.hpp` |

**FFI headers stay `.h`:**

- `src/cpp/include/ffi_types.h`
- `src/cpp/include/ffi_cpu.h`
- `src/cpp/include/ffi_cuda.h`

**Also update:**

- Include guards in renamed headers (`_H` -> `_HPP`).
- All `#include` directives that reference the renamed headers.
- `build.rs` source tracking arrays. These arrays drive `cargo:rerun-if-changed`,
  so stale header paths can cause missed rebuilds even if compilation appears to work.
- Stale Rust doc comment in `src/cpu/kernel.rs`:
  `Matches cast_cpu_launch_args_t in cpu.h` -> `ffi_cpu.h`.

**Verification:**

```sh
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo check
```

## Commit 2: CPU Namespace and Internal Symbol Migration

**Status:** Complete.

Move CPU implementation names into an idiomatic C++ namespace.

**Namespace migration:**

- `cast_cpu_detail` -> `cast::cpu`

Use C++17 nested namespace syntax:

```cpp
namespace cast::cpu {
// ...
} // namespace cast::cpu
```

**Struct renames:**

- `CastCpuGeneratedKernel` -> `cast::cpu::GeneratedKernel`

**Internal function renames:**

- `cast_cpu_generate_kernel_ir` -> `cast::cpu::generate_kernel_ir`
- `cast_cpu_jit_create` -> `cast::cpu::jit_create`
- `cast_cpu_optimize_kernel_ir` -> `cast::cpu::optimize_kernel_ir`
- `cast_cpu_jit_compile_kernel` -> `cast::cpu::jit_compile_kernel`

**Boundary rule:**

- `cpu.cpp` keeps the exported `extern "C"` names unchanged.
- Avoid `using namespace cast::cpu;` in `cpu.cpp`; prefer explicit
  `cast::cpu::...` at the boundary so ABI code remains visually distinct from
  internal C++ code.

**Verification:**

```sh
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo check
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo test
```

## Commit 3: CUDA Namespace and Internal Symbol Migration

**Status:** Complete.

Move CUDA implementation names into `cast::cuda`.

CUDA currently has less internal namespacing than CPU, so keep this separate
from the CPU migration for easier review.

**Introduce namespace:**

- `cast::cuda`

**Struct renames:**

- `CastCudaGeneratedKernel` -> `cast::cuda::GeneratedKernel`
- `CastCudaCompiledKernel` -> `cast::cuda::CompiledKernel`

**Internal function renames:**

- `cast_cuda_generate_kernel_ir` -> `cast::cuda::generate_kernel_ir`
- `cast_cuda_optimize_kernel_ir` -> `cast::cuda::optimize_kernel_ir`
- `cast_cuda_compile_kernel` -> `cast::cuda::compile_kernel`

**Boundary rule:**

- `cuda.cpp` keeps `cast_cuda_compile_gate_ptx` and `cast_cuda_str_free`
  exported with unchanged C linkage.
- `cuda_exec.cpp` keeps all CUDA driver/runtime `extern "C"` names unchanged.
- Avoid broad `using namespace cast::cuda;` in boundary files.

**Verification:**

```sh
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo check
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo test
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo check --features cuda
```

If CUDA tooling or hardware is unavailable locally, record that `--features cuda`
was skipped and why.

## Commit 4: Introduce C++ Internal Types and Boundary Conversions

**Status:** Complete.

This is the actual C -> C++ data-model migration.

Add C++-native internal types, likely under a shared internal header such as:

- `src/cpp/internal/types.hpp`
- or separate CPU/CUDA type headers if that keeps dependencies cleaner

Candidate internal types:

- `cast::Precision`
- `cast::Complex64`
- `cast::cpu::SimdWidth`
- `cast::cpu::MatrixLoadMode`
- `cast::cpu::KernelGenSpec`
- `cast::cpu::KernelMetadata`
- `cast::cpu::GeneratedKernel`
- `cast::cpu::CompiledKernelRecord` or similar
- `cast::cuda::KernelGenSpec`
- `cast::cuda::GeneratedKernel`
- `cast::cuda::CompiledKernel`

Add conversion helpers at the ABI boundary, not deep in internal code:

- FFI CPU spec -> `cast::cpu::KernelGenSpec`
- FFI CUDA spec -> `cast::cuda::KernelGenSpec`
- FFI complex matrix span -> internal complex view/span
- Internal CPU compiled record -> `cast_cpu_jit_kernel_record_t`

After this commit, internal generator/JIT/codegen headers should prefer C++
types and should not include `ffi_cpu.h` or `ffi_cuda.h` unless they are truly
part of the boundary.

Expected dependency direction:

```text
Rust FFI declarations
        |
        v
ffi_*.h  <---- included by boundary .cpp files only
        |
        v
boundary conversion code
        |
        v
internal C++ .hpp/.cpp APIs
```

**Acceptance checks:**

- `cpu_gen.hpp`, `cpu_jit.hpp`, `cuda_gen.hpp`, and `cuda_jit.hpp` no longer
  expose C ABI structs in their public internal signatures.
- `src/cpp/cpu/internal/*.hpp` no longer includes `ffi_cpu.h` except for a
  deliberate, documented exception.
- `src/cpp/cuda/*.hpp` no longer includes `ffi_cuda.h` except for a deliberate,
  documented exception.
- `extern "C"` functions still translate all errors to return codes and
  `err_buf`; no C++ exception crosses into Rust.

**Verification:**

```sh
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo check
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo test
LLVM_CONFIG=$HOME/llvm/22.1.0/release-install/bin/llvm-config cargo check --features cuda
```

## Optional Commit 5: ABI Guardrails

**Status:** Complete.

Add lightweight compile-time checks near the FFI boundary.

Possible checks:

- `static_assert(sizeof(cast_complex64_t) == 16)`
- `static_assert(alignof(cast_complex64_t) == alignof(double))`
- enum value checks for precision, SIMD width, and matrix load mode
- field-offset checks for structs shared with Rust, where useful

This commit should remain small and should not refactor behavior.

## Final Cleanup Checklist

- `rg '#include ".*\.h"' src/cpp` shows only FFI headers or intentional C/system headers.
- `rg 'cast_cpu_detail|CastCpu|CastCuda' src/cpp` returns no stale internal names.
- `rg 'cast_cpu_generate|cast_cpu_jit_|cast_cuda_generate|cast_cuda_compile' src/cpp`
  returns only exported FFI names or comments that intentionally mention old names.
- Rust FFI modules still mirror `ffi_*.h`.
- `cargo test` passes for the CPU path.
- CUDA build path is checked with `--features cuda` when local tooling permits.

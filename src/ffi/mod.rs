//! Raw FFI bindings to the C++ backend libraries.
//!
//! Each submodule mirrors the corresponding header in `src/cpp/include/`:
//! - [`cpu`] — `ffi_cpu.h` (LLVM JIT kernel pipeline)
//! - [`cuda`] — `ffi_cuda.h` (NVPTX codegen, CUDA driver, device kernels)

pub(crate) mod cpu;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;

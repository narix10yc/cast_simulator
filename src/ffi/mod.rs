//! Raw FFI bindings to the C++ backend libraries.
//!
//! Each submodule mirrors the corresponding header in `src/cpp/include/`:
//! - [`cpu`] — `cast_cpu.h` (LLVM JIT kernel pipeline)
//! - [`cuda`] — `cast_cuda.h` (NVPTX codegen, CUDA driver, device kernels)

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

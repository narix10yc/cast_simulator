/// Floating-point precision used in the CUDA kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaPrecision {
    F32 = 0,
    F64 = 1,
}

/// Configuration passed to the kernel generator for each gate.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaKernelGenSpec {
    pub precision: CudaPrecision,
    /// Threshold below which a matrix element is treated as exactly 0.
    pub ztol: f64,
    /// Threshold within which a matrix element is treated as exactly ±1.
    pub otol: f64,
    /// CUDA compute capability major version (e.g. 8 for sm_86).
    pub sm_major: u32,
    /// CUDA compute capability minor version (e.g. 6 for sm_86).
    pub sm_minor: u32,
}

impl CudaKernelGenSpec {
    /// Single-precision defaults with auto-detected SM version.
    /// Falls back to sm_86 if device query fails (e.g. no `cuda` feature).
    pub fn f32() -> Self {
        let (sm_major, sm_minor) = super::device_sm().unwrap_or((8, 6));
        Self {
            precision: CudaPrecision::F32,
            ztol: 1e-6,
            otol: 1e-6,
            sm_major,
            sm_minor,
        }
    }

    /// Double-precision defaults with auto-detected SM version.
    /// Falls back to sm_86 if device query fails (e.g. no `cuda` feature).
    pub fn f64() -> Self {
        let (sm_major, sm_minor) = super::device_sm().unwrap_or((8, 6));
        Self {
            precision: CudaPrecision::F64,
            ztol: 1e-12,
            otol: 1e-12,
            sm_major,
            sm_minor,
        }
    }
}

impl CudaPrecision {
    /// Size of one real scalar in bytes (4 for F32, 8 for F64).
    pub fn scalar_bytes(self) -> usize {
        match self {
            CudaPrecision::F32 => 4,
            CudaPrecision::F64 => 8,
        }
    }
}

/// Opaque handle returned by [`super::CudaKernelManager::generate`], used to
/// identify a compiled kernel owned by a [`super::CudaKernelManager`].
pub type CudaKernelId = u64;

pub(super) const ERR_BUF_LEN: usize = 1024;

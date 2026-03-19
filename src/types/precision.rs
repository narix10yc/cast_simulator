/// Floating-point precision used for the statevector and generated kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F64,
}

impl Precision {
    /// Size of one real scalar in bytes (4 for f32, 8 for f64).
    pub fn scalar_bytes(self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F64 => 8,
        }
    }
}

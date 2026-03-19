use std::ffi::c_char;
use std::fmt;

use super::error_from_buf;
use super::types::{CudaPrecision, ERR_BUF_LEN};

mod ffi {
    use std::ffi::c_char;

    unsafe extern "C" {
        pub fn cast_cuda_sv_alloc(
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> u64;
        pub fn cast_cuda_sv_free(dptr: u64);
        pub fn cast_cuda_sv_zero(
            dptr: u64,
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_sv_upload(
            dptr: u64,
            host_data: *const f64,
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_sv_download(
            dptr: u64,
            host_data: *mut f64,
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
    }
}

/// A statevector allocated in GPU device memory.
///
/// Amplitudes are stored as interleaved `(re, im)` scalars in the precision
/// specified at construction. The host API always uses `f64` regardless of
/// device precision; narrowing to `f32` happens inside the C++ layer.
///
/// The underlying `CUdeviceptr` is owned directly by this struct; device
/// memory is freed on drop.
pub struct CudaStatevector {
    dptr: u64, // CUdeviceptr
    n_qubits: u32,
    precision: CudaPrecision,
    n_elements: usize, // 2 * 2^n_qubits (interleaved re/im scalars)
}

impl Drop for CudaStatevector {
    fn drop(&mut self) {
        unsafe { ffi::cast_cuda_sv_free(self.dptr) };
    }
}

impl fmt::Debug for CudaStatevector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaStatevector")
            .field("n_qubits", &self.n_qubits)
            .field("precision", &self.precision)
            .finish()
    }
}

impl CudaStatevector {
    /// Allocates a device statevector for `2^n_qubits` complex amplitudes.
    pub fn new(n_qubits: u32, precision: CudaPrecision) -> anyhow::Result<Self> {
        let n_elements = 2usize << n_qubits; // 2 * 2^n_qubits
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let dptr = unsafe {
            ffi::cast_cuda_sv_alloc(
                n_elements,
                precision as u8,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if dptr == 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }
        Ok(Self {
            dptr,
            n_qubits,
            precision,
            n_elements,
        })
    }

    pub fn n_qubits(&self) -> u32 {
        self.n_qubits
    }

    pub fn precision(&self) -> CudaPrecision {
        self.precision
    }

    /// Number of complex amplitudes: `2^n_qubits`.
    pub fn len(&self) -> usize {
        1 << self.n_qubits
    }

    pub fn is_empty(&self) -> bool {
        false // a statevector always has at least 1 amplitude
    }

    /// Returns the raw `CUdeviceptr` (for passing to the exec session).
    pub(super) fn dptr(&self) -> u64 {
        self.dptr
    }

    /// Sets the device statevector to the `|0⟩` computational basis state.
    pub fn zero(&mut self) -> anyhow::Result<()> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cuda_sv_zero(
                self.dptr,
                self.n_elements,
                self.precision as u8,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!(error_from_buf(&err_buf)))
        }
    }

    /// Uploads amplitudes from a host slice of `(re, im)` pairs.
    ///
    /// The slice length must equal `2^n_qubits`.
    pub fn upload(&mut self, data: &[(f64, f64)]) -> anyhow::Result<()> {
        if data.len() != self.len() {
            anyhow::bail!(
                "upload: expected {} amplitudes, got {}",
                self.len(),
                data.len()
            );
        }
        let flat: Vec<f64> = data.iter().flat_map(|&(re, im)| [re, im]).collect();
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cuda_sv_upload(
                self.dptr,
                flat.as_ptr(),
                flat.len(),
                self.precision as u8,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!(error_from_buf(&err_buf)))
        }
    }

    /// Downloads all amplitudes to the host as `(re, im)` pairs.
    pub fn download(&self) -> anyhow::Result<Vec<(f64, f64)>> {
        let mut flat = vec![0.0f64; self.n_elements];
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cuda_sv_download(
                self.dptr,
                flat.as_mut_ptr(),
                flat.len(),
                self.precision as u8,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }
        Ok(flat.chunks_exact(2).map(|c| (c[0], c[1])).collect())
    }
}

#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_BIT_LAYOUT_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_BIT_LAYOUT_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cast::cpu {

// Qubits are partitioned into three groups relative to the SIMD register
// (simd_s = log2 of lanes):
//   simd_bits — non-target positions forming the SIMD lanes.
//   lo_bits   — target qubits inside the SIMD range (shuffle within one load).
//   hi_bits   — target qubits above the SIMD range (one aligned load each).
// Positions at or above simd_s are adjusted +1 to account for the implicit
// zero bit the statevector layout inserts there.  sep_bit is the first bit
// above the SIMD region.
//
// BitLayout is the authoritative kernel shape; every size constant used by
// the emitters is derived via the accessors.
struct BitLayout {
  std::vector<unsigned> simd_bits;
  std::vector<unsigned> lo_bits;
  std::vector<unsigned> hi_bits;
  unsigned sep_bit = 0;

  unsigned k() const { return lo_bits.size() + hi_bits.size(); }
  unsigned lk() const { return lo_bits.size(); }
  unsigned hk() const { return hi_bits.size(); }
  unsigned s() const { return simd_bits.size(); }

  unsigned K() const { return 1u << k(); }
  unsigned LK() const { return 1u << lk(); }
  unsigned HK() const { return 1u << hk(); }
  unsigned S() const { return 1u << s(); }
  unsigned vec_size() const { return 1u << sep_bit; } // Phase-1 amp vector width
};

BitLayout compute_bit_layout(const uint32_t *qubits, size_t n_qubits, unsigned simd_s);

// Pointer-segment mask/shift pair used by emit_sv_base_ptr to scatter
// task_id bits into the non-hi statevector address dimensions.
struct PtrSegment {
  uint64_t src_mask;
  unsigned dst_shift;
};

std::vector<PtrSegment> compute_hi_ptr_segments(const std::vector<unsigned> &hi_bits,
                                                unsigned sep_bit);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_BIT_LAYOUT_HPP

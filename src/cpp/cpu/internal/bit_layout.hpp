#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_BIT_LAYOUT_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_BIT_LAYOUT_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cast::cpu {

// Qubits are partitioned into three groups relative to the SIMD register
// (simdS = log2 of lanes):
//   simdBits — non-target positions forming the SIMD lanes.
//   loBits   — target qubits inside the SIMD range (shuffle within one load).
//   hiBits   — target qubits above the SIMD range (one aligned load each).
// Positions at or above simdS are adjusted +1 to account for the implicit
// zero bit the statevector layout inserts there.  sepBit is the first bit
// above the SIMD region.
//
// BitLayout is the authoritative kernel shape; every size constant used by
// the emitters is derived via the accessors.
struct BitLayout {
  std::vector<unsigned> simdBits;
  std::vector<unsigned> loBits;
  std::vector<unsigned> hiBits;
  unsigned sepBit = 0;

  unsigned k() const { return loBits.size() + hiBits.size(); }
  unsigned lk() const { return loBits.size(); }
  unsigned hk() const { return hiBits.size(); }
  unsigned s() const { return simdBits.size(); }

  unsigned K() const { return 1u << k(); }
  unsigned LK() const { return 1u << lk(); }
  unsigned HK() const { return 1u << hk(); }
  unsigned S() const { return 1u << s(); }
  unsigned vecSize() const { return 1u << sepBit; } // Phase-1 amp vector width
};

BitLayout computeBitLayout(const uint32_t *qubits, size_t nQubits, unsigned simdS);

// Pointer-segment mask/shift pair used by emit_sv_base_ptr to scatter
// taskId bits into the non-hi statevector address dimensions.
struct PtrSegment {
  uint64_t srcMask;
  unsigned dstShift;
};

std::vector<PtrSegment> computeHiPtrSegments(const std::vector<unsigned> &hiBits, unsigned sepBit);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_BIT_LAYOUT_HPP

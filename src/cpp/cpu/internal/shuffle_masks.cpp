#include "shuffle_masks.hpp"

#include <cstdint>
#include <utility>

namespace cast::cpu {

namespace {

// Software pdep — used for split-mask precomputation only; not hot.
uint32_t pdep32(uint32_t src, uint32_t mask, unsigned nbits) {
  uint32_t out = 0;
  unsigned srcBit = 0;
  for (unsigned dstBit = 0; dstBit < nbits; ++dstBit) {
    if (!(mask & (uint32_t(1) << dstBit)))
      continue;
    if (src & (uint32_t(1) << srcBit))
      out |= uint32_t(1) << dstBit;
    ++srcBit;
  }
  return out;
}

// Split masks — for each lo-partition (li) and lane (si), index of the
// re/im scalar in the full <vecSize × scalar> interleaved load.  pdep
// places `li`'s bits into lo positions and `si`'s bits into simd-lane
// positions; `| (1 << simdS)` flips the re/im bit.
void computeSplitMasks(const BitLayout &layout, unsigned s, unsigned simdS, ShuffleMasks &out) {
  const unsigned LK = 1u << layout.loBits.size();
  out.reSplit.resize(LK * s);
  out.imSplit.resize(LK * s);

  uint32_t pdepMaskS = 0;
  const unsigned pdepNbitsS = layout.simdBits.empty() ? 0u : layout.simdBits.back() + 1u;
  for (auto bit : layout.simdBits)
    pdepMaskS |= uint32_t(1) << bit;

  uint32_t pdepMaskL = 0;
  const unsigned pdepNbitsL = layout.loBits.empty() ? 0u : layout.loBits.back() + 1u;
  for (auto bit : layout.loBits)
    pdepMaskL |= uint32_t(1) << bit;

  for (unsigned li = 0; li < LK; ++li) {
    for (unsigned si = 0; si < s; ++si) {
      out.reSplit[li * s + si] =
          static_cast<int>(pdep32(li, pdepMaskL, pdepNbitsL) | pdep32(si, pdepMaskS, pdepNbitsS));
      out.imSplit[li * s + si] = out.reSplit[li * s + si] | (1 << simdS);
    }
  }
}

// Merge masks — one round per loBit, each doubling vector width by
// stable-merging the current vector with its `+ (1 << loBit)` twin.
// Phase 3 consumes these to reassemble LK split vectors into one.
void computeMergeMasks(const BitLayout &layout, unsigned s, ShuffleMasks &out) {
  std::vector<int> current(out.reSplit.begin(), out.reSplit.begin() + s);
  for (unsigned loBit : layout.loBits) {
    const int half = static_cast<int>(current.size());

    std::vector<int> rhs(half);
    for (int i = 0; i < half; ++i)
      rhs[i] = current[i] | (1 << loBit);

    out.merge.emplace_back(half * 2);
    auto &mask = out.merge.back();
    std::vector<int> merged(half * 2);
    int il = 0, ir = 0, im = 0;
    while (il < half || ir < half) {
      const bool takeLeft = (ir == half) || (il < half && current[il] < rhs[ir]);
      if (takeLeft) {
        mask[im] = il;
        merged[im++] = current[il++];
      } else {
        mask[im] = ir + half;
        merged[im++] = rhs[ir++];
      }
    }
    current = std::move(merged);
  }
}

// Re-im interleave mask — final `[re0, im0, re1, im1, ...]` permutation
// applied to produce the interleaved store output.
void computeReimMergeMask(unsigned s, unsigned simdS, unsigned vecSize, ShuffleMasks &out) {
  out.reimMerge.reserve(vecSize);
  for (unsigned pairIdx = 0; pairIdx < (vecSize >> simdS >> 1); ++pairIdx) {
    for (unsigned i = 0; i < s; ++i)
      out.reimMerge.push_back(static_cast<int>(s * pairIdx + i));
    for (unsigned i = 0; i < s; ++i)
      out.reimMerge.push_back(static_cast<int>(s * pairIdx + i + (vecSize >> 1)));
  }
}

} // namespace

ShuffleMasks computeShuffleMasks(const BitLayout &layout, unsigned s, unsigned simdS,
                                 unsigned vecSize) {
  ShuffleMasks out;
  computeSplitMasks(layout, s, simdS, out);
  computeMergeMasks(layout, s, out);
  computeReimMergeMask(s, simdS, vecSize, out);
  return out;
}

} // namespace cast::cpu

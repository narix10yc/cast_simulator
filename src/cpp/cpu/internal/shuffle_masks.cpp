#include "shuffle_masks.h"

#include <cstdint>
#include <utility>

namespace cast_cpu_detail {

namespace {

// Software pdep — used for split-mask precomputation only; not hot.
uint32_t pdep32(uint32_t src, uint32_t mask, unsigned nbits) {
  uint32_t out = 0;
  unsigned src_bit = 0;
  for (unsigned dst_bit = 0; dst_bit < nbits; ++dst_bit) {
    if (!(mask & (uint32_t(1) << dst_bit)))
      continue;
    if (src & (uint32_t(1) << src_bit))
      out |= uint32_t(1) << dst_bit;
    ++src_bit;
  }
  return out;
}

// Split masks — for each lo-partition (li) and lane (si), index of the
// re/im scalar in the full <vec_size × scalar> interleaved load.  pdep
// places `li`'s bits into lo positions and `si`'s bits into simd-lane
// positions; `| (1 << simd_s)` flips the re/im bit.
void compute_split_masks(const BitLayout &layout, unsigned s, unsigned simd_s, ShuffleMasks &out) {
  const unsigned LK = 1u << layout.lo_bits.size();
  out.re_split.resize(LK * s);
  out.im_split.resize(LK * s);

  uint32_t pdep_mask_s = 0;
  const unsigned pdep_nbits_s = layout.simd_bits.empty() ? 0u : layout.simd_bits.back() + 1u;
  for (auto bit : layout.simd_bits)
    pdep_mask_s |= uint32_t(1) << bit;

  uint32_t pdep_mask_l = 0;
  const unsigned pdep_nbits_l = layout.lo_bits.empty() ? 0u : layout.lo_bits.back() + 1u;
  for (auto bit : layout.lo_bits)
    pdep_mask_l |= uint32_t(1) << bit;

  for (unsigned li = 0; li < LK; ++li) {
    for (unsigned si = 0; si < s; ++si) {
      out.re_split[li * s + si] = static_cast<int>(pdep32(li, pdep_mask_l, pdep_nbits_l) |
                                                   pdep32(si, pdep_mask_s, pdep_nbits_s));
      out.im_split[li * s + si] = out.re_split[li * s + si] | (1 << simd_s);
    }
  }
}

// Merge masks — one round per lo_bit, each doubling vector width by
// stable-merging the current vector with its `+ (1 << lo_bit)` twin.
// Phase 3 consumes these to reassemble LK split vectors into one.
void compute_merge_masks(const BitLayout &layout, unsigned s, ShuffleMasks &out) {
  std::vector<int> current(out.re_split.begin(), out.re_split.begin() + s);
  for (unsigned lo_bit : layout.lo_bits) {
    const int half = static_cast<int>(current.size());

    std::vector<int> rhs(half);
    for (int i = 0; i < half; ++i)
      rhs[i] = current[i] | (1 << lo_bit);

    out.merge.emplace_back(half * 2);
    auto &mask = out.merge.back();
    std::vector<int> merged(half * 2);
    int il = 0, ir = 0, im = 0;
    while (il < half || ir < half) {
      const bool take_left = (ir == half) || (il < half && current[il] < rhs[ir]);
      if (take_left) {
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
void compute_reim_merge_mask(unsigned s, unsigned simd_s, unsigned vec_size, ShuffleMasks &out) {
  out.reim_merge.reserve(vec_size);
  for (unsigned pair_idx = 0; pair_idx < (vec_size >> simd_s >> 1); ++pair_idx) {
    for (unsigned i = 0; i < s; ++i)
      out.reim_merge.push_back(static_cast<int>(s * pair_idx + i));
    for (unsigned i = 0; i < s; ++i)
      out.reim_merge.push_back(static_cast<int>(s * pair_idx + i + (vec_size >> 1)));
  }
}

} // namespace

ShuffleMasks compute_shuffle_masks(const BitLayout &layout, unsigned s, unsigned simd_s,
                                   unsigned vec_size) {
  ShuffleMasks out;
  compute_split_masks(layout, s, simd_s, out);
  compute_merge_masks(layout, s, out);
  compute_reim_merge_mask(s, simd_s, vec_size, out);
  return out;
}

} // namespace cast_cpu_detail

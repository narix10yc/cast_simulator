#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_H
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_H

#include "bit_layout.h"

#include <vector>

namespace cast_cpu_detail {

// Precomputed shufflevector indices used by Phase 1 (load → split) and
// Phase 3 (merge → store).  `re_split`/`im_split` flatten as (li, si) with
// stride `s`; `merge` holds one mask per lo_bit for the reassembly round.
struct ShuffleMasks {
  std::vector<int> re_split;
  std::vector<int> im_split;
  std::vector<std::vector<int>> merge;
  std::vector<int> reim_merge;
};

ShuffleMasks compute_shuffle_masks(const BitLayout &layout, unsigned s, unsigned simd_s,
                                   unsigned vec_size);

} // namespace cast_cpu_detail

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_H

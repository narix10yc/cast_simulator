#include "cpu_gen.h"

#include "cpu_util.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

enum ScalarKind : uint8_t {
  SK_Unknown = 0,
  SK_Runtime,
  SK_Zero,
  SK_One,
  SK_MinusOne,
};

struct LaunchArgs {
  llvm::Value* p_sv = nullptr;
  llvm::Value* ctr_begin = nullptr;
  llvm::Value* ctr_end = nullptr;
  llvm::Value* p_mat = nullptr;
};

struct MatrixView {
  const cast_cpu_complex64_t* data = nullptr;
  uint32_t edge_size = 0;

  double re(size_t idx) const { return data[idx].re; }
  double im(size_t idx) const { return data[idx].im; }
};

struct IRMatData {
  llvm::Value* re_elm = nullptr;
  llvm::Value* im_elm = nullptr;
  llvm::Value* re_vec = nullptr;
  llvm::Value* im_vec = nullptr;
  ScalarKind re_flag = SK_Unknown;
  ScalarKind im_flag = SK_Unknown;
};

// Validation and simd_s helpers live in cpu_util.h; use them via the
// cast_cpu_detail namespace.

uint32_t pdep32(uint32_t src, uint32_t mask, unsigned nbits = 32) {
  uint32_t out = 0;
  unsigned src_bit = 0;
  for (unsigned dst_bit = 0; dst_bit < nbits; ++dst_bit) {
    if ((mask & (uint32_t(1) << dst_bit)) == 0) {
      continue;
    }
    if (src & (uint32_t(1) << src_bit)) {
      out |= uint32_t(1) << dst_bit;
    }
    ++src_bit;
  }
  return out;
}

llvm::Value* gen_mul_add(llvm::IRBuilder<>& builder,
                         llvm::Value* a,
                         llvm::Value* b,
                         llvm::Value* c,
                         ScalarKind a_kind,
                         const llvm::Twine& name = "") {
  switch (a_kind) {
  case SK_Runtime:
    if (c != nullptr) {
      return builder.CreateIntrinsic(
          a->getType(), llvm::Intrinsic::fmuladd, {a, b, c}, nullptr, name);
    }
    return builder.CreateFMul(a, b, name);
  case SK_One:
    return (c != nullptr) ? builder.CreateFAdd(b, c, name) : b;
  case SK_MinusOne:
    return (c != nullptr) ? builder.CreateFSub(c, b, name)
                          : builder.CreateFNeg(b, name);
  case SK_Zero:
    return c;
  default:
    llvm_unreachable("unsupported scalar kind");
  }
}

llvm::Value* gen_neg_mul_add(llvm::IRBuilder<>& builder,
                             llvm::Value* a,
                             llvm::Value* b,
                             llvm::Value* c,
                             ScalarKind a_kind,
                             const llvm::Twine& name = "") {
  switch (a_kind) {
  case SK_One:
    return (c != nullptr) ? builder.CreateFSub(c, b, name)
                          : builder.CreateFNeg(b, name);
  case SK_MinusOne:
    return (c != nullptr) ? builder.CreateFAdd(b, c, name) : b;
  case SK_Zero:
    return c;
  case SK_Runtime: {
    auto* a_neg = builder.CreateFNeg(a, "a.neg");
    if (c != nullptr) {
      return builder.CreateIntrinsic(a_neg->getType(),
                                     llvm::Intrinsic::fmuladd,
                                     {a_neg, b, c},
                                     nullptr,
                                     name);
    }
    return builder.CreateFMul(a_neg, b, name);
  }
  default:
    llvm_unreachable("unsupported scalar kind");
  }
}

std::vector<IRMatData> init_matrix_data(llvm::IRBuilder<>& builder,
                                        const cast_cpu_kernel_gen_spec_t& spec,
                                        const MatrixView& matrix,
                                        llvm::Value* p_mat_arg,
                                        unsigned simd_s) {
  const unsigned k = matrix.edge_size;
  const unsigned kk = k * k;
  std::vector<IRMatData> out(kk);

  for (unsigned i = 0; i < kk; ++i) {
    const double re = matrix.re(i);
    const double im = matrix.im(i);
    out[i].re_flag = (std::abs(re) < spec.ztol)         ? SK_Zero
                     : (std::abs(re - 1.0) < spec.otol) ? SK_One
                     : (std::abs(re + 1.0) < spec.otol) ? SK_MinusOne
                                                        : SK_Runtime;
    out[i].im_flag = (std::abs(im) < spec.ztol)         ? SK_Zero
                     : (std::abs(im - 1.0) < spec.otol) ? SK_One
                     : (std::abs(im + 1.0) < spec.otol) ? SK_MinusOne
                                                        : SK_Runtime;
  }

  auto* scalar_ty = (spec.precision == CAST_CPU_PRECISION_F32)
                        ? builder.getFloatTy()
                        : builder.getDoubleTy();
  auto ec = llvm::ElementCount::getFixed(1u << simd_s);

  for (unsigned i = 0; i < kk; ++i) {
    auto& entry = out[i];
    if (spec.mode == CAST_CPU_MATRIX_LOAD_IMM_VALUE) {
      if (entry.re_flag == SK_Runtime) {
        auto* c = llvm::ConstantFP::get(scalar_ty, matrix.re(i));
        entry.re_elm = c;
        entry.re_vec = llvm::ConstantVector::getSplat(ec, c);
      }
      if (entry.im_flag == SK_Runtime) {
        auto* c = llvm::ConstantFP::get(scalar_ty, matrix.im(i));
        entry.im_elm = c;
        entry.im_vec = llvm::ConstantVector::getSplat(ec, c);
      }
      continue;
    }

    if (entry.re_flag == SK_Runtime) {
      auto* ptr = builder.CreateConstGEP1_32(
          scalar_ty, p_mat_arg, static_cast<unsigned>(2 * i), "re.mat.ptr");
      entry.re_elm = builder.CreateLoad(scalar_ty, ptr, "re.mat");
      entry.re_vec = builder.CreateVectorSplat(ec, entry.re_elm, "re.mat.vec");
    }
    if (entry.im_flag == SK_Runtime) {
      auto* ptr = builder.CreateConstGEP1_32(
          scalar_ty, p_mat_arg, static_cast<unsigned>(2 * i + 1), "im.mat.ptr");
      entry.im_elm = builder.CreateLoad(scalar_ty, ptr, "im.mat");
      entry.im_vec = builder.CreateVectorSplat(ec, entry.im_elm, "im.mat.vec");
    }
  }

  return out;
}

} // namespace

llvm::Expected<llvm::Function*>
cast_cpu_generate_kernel_ir(const cast_cpu_kernel_gen_spec_t& spec,
                            const cast_cpu_complex64_t* matrix,
                            size_t matrix_len,
                            const uint32_t* qubits,
                            size_t n_qubits,
                            llvm::StringRef func_name,
                            llvm::Module& module) {
  if (!cast_cpu_detail::is_valid_precision(spec.precision)) {
    return llvm::createStringError("invalid precision");
  }
  if (!cast_cpu_detail::is_valid_simd_width(spec.simd_width)) {
    return llvm::createStringError("invalid SIMD width");
  }
  if (!cast_cpu_detail::is_valid_mode(spec.mode)) {
    return llvm::createStringError("invalid matrix load mode");
  }
  if (spec.ztol < 0.0 || spec.otol < 0.0) {
    return llvm::createStringError("tolerances must be non-negative");
  }
  if (matrix == nullptr) {
    return llvm::createStringError("matrix must not be null");
  }
  if (qubits == nullptr && n_qubits != 0) {
    return llvm::createStringError("qubits must not be null");
  }
  for (size_t i = 1; i < n_qubits; ++i) {
    if (qubits[i - 1] >= qubits[i]) {
      return llvm::createStringError("qubits must be strictly ascending");
    }
  }

  size_t expected_len = 0;
  if (!cast_cpu_detail::expected_matrix_len(n_qubits, &expected_len) ||
      expected_len != matrix_len) {
    return llvm::createStringError(
        "matrix length does not match the target qubit count");
  }

  const unsigned simd_s =
      cast_cpu_detail::get_simd_s(spec.simd_width, spec.precision);
  // simd_s is in [1,4] for all valid (precision, simd_width) pairs;
  // the validation above guarantees we never reach here with simd_s == 0.
  assert(simd_s > 0 && simd_s <= 4);
  const unsigned s = 1u << simd_s;
  const unsigned k = static_cast<unsigned>(n_qubits);
  const unsigned K = 1u << k;
  MatrixView mat_view{matrix, static_cast<uint32_t>(K)};

  auto& ctx = module.getContext();
  llvm::IRBuilder<> builder(ctx);
  auto* scalar_ty = (spec.precision == CAST_CPU_PRECISION_F32)
                        ? builder.getFloatTy()
                        : builder.getDoubleTy();

  auto* launch_ty = llvm::StructType::get(builder.getPtrTy(),
                                          builder.getInt64Ty(),
                                          builder.getInt64Ty(),
                                          builder.getPtrTy());
  auto* func_ty =
      llvm::FunctionType::get(builder.getVoidTy(), {builder.getPtrTy()}, false);
  auto* func = llvm::Function::Create(
      func_ty, llvm::Function::ExternalLinkage, func_name, module);

  auto* entry_bb = llvm::BasicBlock::Create(ctx, "entry", func);
  auto* loop_bb = llvm::BasicBlock::Create(ctx, "loop", func);
  auto* loop_body_bb = llvm::BasicBlock::Create(ctx, "loop.body", func);
  auto* ret_bb = llvm::BasicBlock::Create(ctx, "ret", func);

  LaunchArgs args;
  builder.SetInsertPoint(entry_bb);
  auto* launch_arg = func->getArg(0);
  args.p_sv = builder.CreateLoad(
      builder.getPtrTy(),
      builder.CreateStructGEP(launch_ty, launch_arg, 0, "launch.sv.ptr"),
      "sv");
  args.ctr_begin = builder.CreateLoad(
      builder.getInt64Ty(),
      builder.CreateStructGEP(launch_ty, launch_arg, 1, "launch.ctr.begin.ptr"),
      "ctr.begin");
  args.ctr_end = builder.CreateLoad(
      builder.getInt64Ty(),
      builder.CreateStructGEP(launch_ty, launch_arg, 2, "launch.ctr.end.ptr"),
      "ctr.end");
  args.p_mat = builder.CreateLoad(
      builder.getPtrTy(),
      builder.CreateStructGEP(launch_ty, launch_arg, 3, "launch.mat.ptr"),
      "mat");

  auto mat_data = init_matrix_data(builder, spec, mat_view, args.p_mat, simd_s);

  unsigned sep_bit = 0;
  std::vector<unsigned> simd_bits;
  std::vector<unsigned> lo_bits;
  std::vector<unsigned> hi_bits;
  {
    unsigned q = 0;
    size_t qi = 0;
    while (simd_bits.size() != simd_s) {
      if (qi < n_qubits && qubits[qi] == q) {
        lo_bits.push_back(q);
        ++qi;
      } else {
        simd_bits.push_back(q);
      }
      ++q;
    }
    while (qi < n_qubits) {
      hi_bits.push_back(qubits[qi++]);
    }
    // The SIMD layout inserts a zero bit at position simd_s (via
    // insert_zero_to_bit), so every qubit index at or above simd_s is shifted
    // up by one in the flat scalar buffer.  Adjust all collected bit positions
    // accordingly before they are used to construct GEP offsets and shuffle
    // masks.
    for (auto& bit : lo_bits) {
      if (bit >= simd_s) {
        ++bit;
      }
    }
    for (auto& bit : simd_bits) {
      if (bit >= simd_s) {
        ++bit;
      }
    }
    for (auto& bit : hi_bits) {
      if (bit >= simd_s) {
        ++bit;
      }
    }
    sep_bit = (simd_s == 0) ? 0 : (simd_bits.back() + 1);
    if (sep_bit == simd_s) {
      ++sep_bit;
    }
  }

  const unsigned vec_size = 1u << sep_bit;
  auto* vec_ty = llvm::VectorType::get(scalar_ty, vec_size, false);
  const unsigned lk = lo_bits.size();
  const unsigned LK = 1u << lk;
  const unsigned hk = hi_bits.size();
  const unsigned HK = 1u << hk;

  builder.CreateBr(loop_bb);
  builder.SetInsertPoint(loop_bb);
  auto* task_id = builder.CreatePHI(builder.getInt64Ty(), 2, "taskid");
  task_id->addIncoming(args.ctr_begin, entry_bb);
  auto* cond = builder.CreateICmpSLT(task_id, args.ctr_end, "cond");
  builder.CreateCondBr(cond, loop_body_bb, ret_bb);

  builder.SetInsertPoint(loop_body_bb);
  llvm::Value* ptr_sv_begin = nullptr;
  if (hi_bits.empty()) {
    ptr_sv_begin =
        builder.CreateGEP(vec_ty, args.p_sv, task_id, "ptr.sv.begin");
  } else {
    llvm::Value* idx_start = builder.getInt64(0);
    uint64_t mask = 0;
    const auto highest_q = hi_bits.back();
    unsigned q_idx = 0;
    unsigned counter_q = 0;
    for (unsigned q = sep_bit; q <= highest_q; ++q) {
      if (q < hi_bits[q_idx]) {
        mask |= (uint64_t(1) << counter_q++);
        continue;
      }
      ++q_idx;
      if (mask == 0) {
        continue;
      }
      auto* tmp = builder.CreateAnd(task_id, mask, "tmp.taskid");
      tmp = builder.CreateShl(tmp, q_idx - 1, "tmp.taskid.shl");
      idx_start = builder.CreateAdd(idx_start, tmp, "idx.start.part");
      mask = 0;
    }
    mask = ~((uint64_t(1) << (highest_q - sep_bit - hk + 1)) - 1);
    auto* tmp = builder.CreateAnd(task_id, mask, "tmp.taskid");
    tmp = builder.CreateShl(tmp, hk, "tmp.taskid.shl");
    idx_start = builder.CreateAdd(idx_start, tmp, "idx.start");
    ptr_sv_begin =
        builder.CreateGEP(vec_ty, args.p_sv, idx_start, "ptr.sv.begin");
  }

  std::vector<int> re_split_masks(LK * s);
  std::vector<int> im_split_masks(LK * s);
  {
    uint32_t pdep_mask_s = 0;
    unsigned pdep_nbits_s = simd_bits.empty() ? 0 : (simd_bits.back() + 1);
    for (auto bit : simd_bits) {
      pdep_mask_s |= uint32_t(1) << bit;
    }
    uint32_t pdep_mask_l = 0;
    unsigned pdep_nbits_l = lo_bits.empty() ? 0 : (lo_bits.back() + 1);
    for (auto bit : lo_bits) {
      pdep_mask_l |= uint32_t(1) << bit;
    }
    for (unsigned li = 0; li < LK; ++li) {
      for (unsigned si = 0; si < s; ++si) {
        re_split_masks[li * s + si] =
            static_cast<int>(pdep32(li, pdep_mask_l, pdep_nbits_l) |
                             pdep32(si, pdep_mask_s, pdep_nbits_s));
        im_split_masks[li * s + si] =
            re_split_masks[li * s + si] | (1 << simd_s);
      }
    }
  }

  std::vector<llvm::Value*> re_amps(K);
  std::vector<llvm::Value*> im_amps(K);
  std::vector<llvm::Value*> p_svs(HK);
  for (unsigned hi = 0; hi < HK; ++hi) {
    uint64_t idx_shift = 0;
    for (unsigned hbit = 0; hbit < hk; ++hbit) {
      if (hi & (1u << hbit)) {
        idx_shift += uint64_t(1) << hi_bits[hbit];
      }
    }
    idx_shift >>= sep_bit;
    p_svs[hi] = builder.CreateConstGEP1_64(
        vec_ty, ptr_sv_begin, idx_shift, "ptr.sv.hi");
    auto* amp_full = builder.CreateLoad(vec_ty, p_svs[hi], "sv.full");
    for (unsigned li = 0; li < LK; ++li) {
      re_amps[hi * LK + li] = builder.CreateShuffleVector(
          amp_full,
          llvm::ArrayRef<int>(re_split_masks.data() + li * s, s),
          "re");
      im_amps[hi * LK + li] = builder.CreateShuffleVector(
          amp_full,
          llvm::ArrayRef<int>(im_split_masks.data() + li * s, s),
          "im");
    }
  }

  std::vector<std::vector<int>> merge_masks;
  std::vector<int> reim_merge_mask;
  {
    std::vector<int> arr0(LK * s), arr1(LK * s), arr2(LK * s);
    std::vector<int>* cache_lhs = &arr0;
    std::vector<int>* cache_rhs = &arr1;
    std::vector<int>* cache_combined = &arr2;
    std::memcpy(arr0.data(), re_split_masks.data(), s * sizeof(int));
    if (LK > 1) {
      std::memcpy(arr1.data(), re_split_masks.data() + s, s * sizeof(int));
    }
    unsigned round_idx = 0;
    while (round_idx < lk) {
      const int cached_len = static_cast<int>(s << round_idx);
      merge_masks.emplace_back(cached_len << 1);
      auto& mask = merge_masks.back();

      int idx_l = 0;
      int idx_r = 0;
      for (int idx = 0; idx < (cached_len << 1); ++idx) {
        if (idx_l == cached_len) {
          while (idx_r < cached_len) {
            mask[idx] = idx_r + cached_len;
            (*cache_combined)[idx++] = (*cache_rhs)[idx_r++];
          }
          break;
        }
        if (idx_r == cached_len) {
          while (idx_l < cached_len) {
            mask[idx] = idx_l;
            (*cache_combined)[idx++] = (*cache_lhs)[idx_l++];
          }
          break;
        }
        if ((*cache_lhs)[idx_l] < (*cache_rhs)[idx_r]) {
          mask[idx] = idx_l;
          (*cache_combined)[idx] = (*cache_lhs)[idx_l++];
        } else {
          mask[idx] = idx_r + cached_len;
          (*cache_combined)[idx] = (*cache_rhs)[idx_r++];
        }
      }

      if (++round_idx == lk) {
        break;
      }
      cache_lhs = cache_combined;
      if (cache_lhs == &arr2) {
        cache_rhs = &arr0;
        cache_combined = &arr1;
      } else if (cache_lhs == &arr1) {
        cache_rhs = &arr2;
        cache_combined = &arr0;
      } else {
        cache_rhs = &arr1;
        cache_combined = &arr2;
      }
      for (int i = 0; i < (cached_len << 1); ++i) {
        (*cache_rhs)[i] = (*cache_lhs)[i] | (1 << lo_bits[round_idx]);
      }
    }

    reim_merge_mask.reserve(vec_size);
    for (unsigned pair_idx = 0; pair_idx < (vec_size >> simd_s >> 1);
         ++pair_idx) {
      for (unsigned i = 0; i < s; ++i) {
        reim_merge_mask.push_back(static_cast<int>(s * pair_idx + i));
      }
      for (unsigned i = 0; i < s; ++i) {
        reim_merge_mask.push_back(
            static_cast<int>(s * pair_idx + i + (vec_size >> 1)));
      }
    }
  }

  std::vector<llvm::Value*> updated_re_amps(LK);
  std::vector<llvm::Value*> updated_im_amps(LK);
  for (unsigned hi = 0; hi < HK; ++hi) {
    std::fill(updated_re_amps.begin(), updated_re_amps.end(), nullptr);
    std::fill(updated_im_amps.begin(), updated_im_amps.end(), nullptr);
    for (unsigned li = 0; li < LK; ++li) {
      const unsigned r = hi * LK + li;
      for (unsigned c = 0; c < K; ++c) {
        const auto& entry = mat_data[r * K + c];
        updated_re_amps[li] = gen_mul_add(builder,
                                          entry.re_vec,
                                          re_amps[c],
                                          updated_re_amps[li],
                                          entry.re_flag,
                                          "new.re");
        updated_re_amps[li] = gen_neg_mul_add(builder,
                                              entry.im_vec,
                                              im_amps[c],
                                              updated_re_amps[li],
                                              entry.im_flag,
                                              "new.re");

        updated_im_amps[li] = gen_mul_add(builder,
                                          entry.re_vec,
                                          im_amps[c],
                                          updated_im_amps[li],
                                          entry.re_flag,
                                          "new.im");
        updated_im_amps[li] = gen_mul_add(builder,
                                          entry.im_vec,
                                          re_amps[c],
                                          updated_im_amps[li],
                                          entry.im_flag,
                                          "new.im");
      }
    }

    auto* zero_vec = llvm::ConstantAggregateZero::get(
        llvm::VectorType::get(scalar_ty, s, false));
    for (auto& value : updated_re_amps) {
      if (value == nullptr) {
        value = zero_vec;
      }
    }
    for (auto& value : updated_im_amps) {
      if (value == nullptr) {
        value = zero_vec;
      }
    }

    for (unsigned merge_idx = 0; merge_idx < lk; ++merge_idx) {
      for (unsigned pair_idx = 0; pair_idx < (LK >> merge_idx >> 1);
           ++pair_idx) {
        const unsigned idx_l = pair_idx << merge_idx << 1;
        const unsigned idx_r = idx_l | (1u << merge_idx);
        updated_re_amps[idx_l] =
            builder.CreateShuffleVector(updated_re_amps[idx_l],
                                        updated_re_amps[idx_r],
                                        merge_masks[merge_idx],
                                        "re.merged");
        updated_im_amps[idx_l] =
            builder.CreateShuffleVector(updated_im_amps[idx_l],
                                        updated_im_amps[idx_r],
                                        merge_masks[merge_idx],
                                        "im.merged");
      }
    }

    auto* merged = builder.CreateShuffleVector(
        updated_re_amps[0], updated_im_amps[0], reim_merge_mask, "amp.merged");
    builder.CreateStore(merged, p_svs[hi]);
  }

  auto* task_id_next =
      builder.CreateAdd(task_id, builder.getInt64(1), "taskid.next");
  task_id->addIncoming(task_id_next, loop_body_bb);
  builder.CreateBr(loop_bb);

  builder.SetInsertPoint(ret_bb);
  builder.CreateRetVoid();

  return func;
}

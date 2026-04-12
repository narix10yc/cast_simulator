#include "cuda_gen.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>

#include <cassert>
#include <cmath>
#include <vector>

using namespace llvm;

namespace {

// ── ScalarKind ─────────────────────────────────────────────────────────────

enum ScalarKind {
  SK_Zero = 0,
  SK_One = 1,
  SK_MinusOne = -1,
  SK_ImmValue = 2,
  SK_Runtime = 3,
};

// ── Helper types ────────────────────────────────────────────────────────────

struct KernelArgs {
  Argument *p_sv;     // ptr to statevector
  Argument *p_mat;    // ptr to matrix (unused in ImmValue mode)
  Argument *p_combos; // i64: total number of amplitude combos
};

struct IRMatData {
  Value *re_val;
  Value *im_val;
  ScalarKind re_kind;
  ScalarKind im_kind;
};

// ── emit_opt_fmul ──────────────────────────────────────────────────────────

static Value *emit_opt_fmul(Value *a, Value *b, ScalarKind aKind, IRBuilder<> &B) {
  switch (aKind) {
  case SK_Runtime:
  case SK_ImmValue:
    assert(a);
    return B.CreateFMul(a, b);
  case SK_One:
    return b;
  case SK_MinusOne:
    return B.CreateFNeg(b);
  case SK_Zero:
    return nullptr;
  default:
    llvm_unreachable("Unknown ScalarKind");
  }
}

// ── build_matrix_data ──────────────────────────────────────────────────────

/// Classifies each matrix element as zero, ±1, or general, and returns the
/// corresponding LLVM constant value alongside its ScalarKind tag.
static std::vector<IRMatData> build_matrix_data(IRBuilder<> &B,
                                                const cast_cuda_kernel_gen_spec_t &spec,
                                                const cast_cuda_complex64_t *matrix,
                                                unsigned n_qubits) {
  const auto K = 1U << n_qubits;
  const auto KK = K * K;

  const auto z_tol = spec.ztol / static_cast<double>(K);
  const auto o_tol = spec.otol / static_cast<double>(K);

  const bool fp32 = (spec.precision == CAST_CUDA_PRECISION_F32);
  Type *scalar_ty = fp32 ? B.getFloatTy() : B.getDoubleTy();
  auto *zero_val = ConstantFP::get(scalar_ty, 0.0);
  auto *one_val = ConstantFP::get(scalar_ty, 1.0);
  auto *minus_one_val = ConstantFP::get(scalar_ty, -1.0);

  // Classify kind and build LLVM constant for a single scalar.
  auto classify = [&](double v) -> std::pair<Value *, ScalarKind> {
    if (spec.ztol > 0.0 && std::abs(v) < z_tol)
      return {zero_val, SK_Zero};
    if (spec.otol > 0.0 && std::abs(v - 1.0) < o_tol)
      return {one_val, SK_One};
    if (spec.otol > 0.0 && std::abs(v + 1.0) < o_tol)
      return {minus_one_val, SK_MinusOne};
    return {ConstantFP::get(scalar_ty, fp32 ? static_cast<double>(static_cast<float>(v)) : v),
            SK_ImmValue};
  };

  std::vector<IRMatData> data(KK);
  for (unsigned i = 0; i < KK; ++i) {
    auto [re_val, re_kind] = classify(matrix[i].re);
    auto [im_val, im_kind] = classify(matrix[i].im);
    data[i] = {re_val, im_val, re_kind, im_kind};
  }
  return data;
}

// ── create_kernel_function ───────────────────────────────────────────────────

static Function *create_kernel_function(IRBuilder<> &B, Module &M, const std::string &func_name,
                                        KernelArgs &args) {
  auto params = std::array<Type *, 3>{
      B.getPtrTy(),  // sv
      B.getPtrTy(),  // mat
      B.getInt64Ty() // combos
  };
  auto *fty = FunctionType::get(B.getVoidTy(), params, false);
  auto *func = Function::Create(fty, Function::ExternalLinkage, func_name, M);

  args.p_sv = func->getArg(0);
  args.p_sv->setName("p.sv");
  args.p_mat = func->getArg(1);
  args.p_mat->setName("p.mat");
  args.p_combos = func->getArg(2);
  args.p_combos->setName("p.combos");

  // Mark as a CUDA (PTX) kernel via nvvm.annotations metadata.
  auto *mdString = MDString::get(M.getContext(), "kernel");
  auto *mdOne = ConstantAsMetadata::get(B.getInt32(1));
  auto *md = MDNode::get(M.getContext(), {ValueAsMetadata::get(func), mdString, mdOne});
  M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md);
  return func;
}

// ── emit_combo_offset ──────────────────────────────────────────────────────
//
// Maps a combo counter to the base statevector amplitude index (in units of
// scalars) using a compile-time-known PDEP-like bit scatter.
//
// Example: target qubits [2, 4, 5], counter xxxhgfedcba
//   result (in scalar units) = hgfed00c0ba * 2
//
// The returned Value is the byte-GEP index into the scalar array (×2 shift
// for re/im is applied by the caller).

static Value *emit_combo_offset(IRBuilder<> &B, Value *counter_v, const uint32_t *qubits,
                                size_t n_qubits) {
  assert(n_qubits > 0);

  auto *offset = static_cast<Value *>(B.getInt64(0ULL));
  counter_v = B.CreateZExt(counter_v, B.getInt64Ty(), "i64.counter");

  const int k = static_cast<int>(n_qubits);
  const int highest_q = static_cast<int>(qubits[n_qubits - 1]);

  uint64_t mask = 0ULL;
  int q_idx = 0;
  int counter_q = 0;

  Value *tmp_counter = nullptr;

  for (int q = 0; q <= highest_q; q++) {
    if (q < static_cast<int>(qubits[q_idx])) {
      mask |= (1ULL << counter_q++);
      continue;
    }
    ++q_idx;
    if (mask == 0)
      continue;

    tmp_counter = B.CreateAnd(counter_v, mask, "tmpCounter");
    tmp_counter = B.CreateShl(tmp_counter, (q_idx - 1), "tmpCounter");
    offset = B.CreateAdd(offset, tmp_counter, "tmpIdx");
    mask = 0ULL;
  }

  mask = ~((1ULL << (highest_q - k + 1)) - 1);
  tmp_counter = B.CreateAnd(counter_v, mask, "tmpCounter");
  tmp_counter = B.CreateShl(tmp_counter, k, "tmpCounter");
  offset = B.CreateAdd(offset, tmp_counter, "offset");

  return offset;
}

// ── emit_matvec ─────────────────────────────────────────────────────────────
//
// Emits straight-line code that:
//   1. Loads all K amplitude pairs (re, im) from sv[sv_ptr + 2*delta{i}]
//   2. Accumulates M*v with constant (ImmValue) matrix entries
//   3. Stores the updated amplitudes back in place
//
// sv_ptr already points to the base amplitude for this combo (offset has been
// applied by the caller).

static void emit_matvec(IRBuilder<> &B, const uint32_t *qubits, size_t n_qubits,
                        const std::vector<IRMatData> &mat_data, Value *sv_ptr, Type *scalar_ty) {
  B.setFastMathFlags(FastMathFlags::getFast());

  const unsigned k = static_cast<unsigned>(n_qubits);
  const unsigned K = 1u << k;

  // Vector type for complex pair (re, im) — enables ld.global.v2 / st.global.v2
  // in the NVPTX backend, doubling memory-transaction utilization for the
  // interleaved [re₀, im₀, re₁, im₁, …] layout.
  auto *vec2_ty = FixedVectorType::get(scalar_ty, 2);

  std::vector<Value *> amp_ptrs(K);  // base pointer per amplitude (re position)
  std::vector<Value *> re_amps(K), im_amps(K);

  // Compute the offset for each of the K amplitude slots in the combo.
  // Amplitude i corresponds to setting bits qubits[b] iff bit b of i is set.
  for (unsigned i = 0; i < K; ++i) {
    uint64_t delta = 0;
    for (unsigned b = 0; b < k; ++b)
      if (i & (1u << b))
        delta |= (1ull << qubits[b]);

    uint64_t off2 = 2ull * delta;
    amp_ptrs[i] = B.CreateConstGEP1_64(scalar_ty, sv_ptr, off2, "amp.ptr");
    auto *pair = B.CreateLoad(vec2_ty, amp_ptrs[i], "amp.pair");
    re_amps[i] = B.CreateExtractElement(pair, (uint64_t)0, "re.amp");
    im_amps[i] = B.CreateExtractElement(pair, (uint64_t)1, "im.amp");
  }

  // For each output row r:  new[r] = sum_c  M[r,c] * old[c]
  for (unsigned r = 0; r < K; ++r) {
    auto *acc_re0 = static_cast<Value *>(ConstantFP::get(scalar_ty, 0.0));
    auto *acc_re1 = static_cast<Value *>(ConstantFP::get(scalar_ty, 0.0));
    auto *acc_im = static_cast<Value *>(ConstantFP::get(scalar_ty, 0.0));

    for (unsigned c = 0; c < K; ++c) {
      const auto &md = mat_data[r * K + c];
      if (md.re_kind == SK_Zero && md.im_kind == SK_Zero)
        continue;

      // Re(new) = Re(M)*Re(old) - Im(M)*Im(old)
      if (auto *t0 = emit_opt_fmul(md.re_val, re_amps[c], md.re_kind, B))
        acc_re0 = B.CreateFAdd(acc_re0, t0);
      if (auto *t1 = emit_opt_fmul(md.im_val, im_amps[c], md.im_kind, B))
        acc_re1 = B.CreateFAdd(acc_re1, t1);

      // Im(new) = Re(M)*Im(old) + Im(M)*Re(old)
      if (auto *t2 = emit_opt_fmul(md.re_val, im_amps[c], md.re_kind, B))
        acc_im = B.CreateFAdd(acc_im, t2);
      if (auto *t3 = emit_opt_fmul(md.im_val, re_amps[c], md.im_kind, B))
        acc_im = B.CreateFAdd(acc_im, t3);
    }

    auto *new_re = B.CreateFSub(acc_re0, acc_re1, "new.re");
    Value *out = PoisonValue::get(vec2_ty);
    out = B.CreateInsertElement(out, new_re, (uint64_t)0, "out.re");
    out = B.CreateInsertElement(out, acc_im, (uint64_t)1, "out.im");
    B.CreateStore(out, amp_ptrs[r]);
  }
}

// ── emit_persistent_grid_loop ────────────────────────────────────────────────
//
// Wraps emit_matvec inside a persistent-grid loop:
//   for (combo = global_tid; combo < p.combos; combo += stride) { ... }

static void emit_persistent_grid_loop(IRBuilder<> &B, const uint32_t *qubits, size_t n_qubits,
                                      const std::vector<IRMatData> &mat_data, Value *sv_root,
                                      Value *combos_v, Type *scalar_ty) {
  auto *func = B.GetInsertBlock()->getParent();
  auto &C = B.getContext();

  // NVPTX thread-index intrinsics.
  auto *tid = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  auto *ntid = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ntid_x, {});
  auto *ctaid = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
  auto *nctaid = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_nctaid_x, {});

  // global_tid = ctaid * ntid + tid
  auto *global_tid = B.CreateAdd(B.CreateMul(ctaid, ntid), tid, "global.tid");
  global_tid = B.CreateIntCast(global_tid, B.getInt64Ty(), /*isSigned=*/true, "global.tid");

  // stride = nctaid * ntid  (total thread count)
  auto *stride = B.CreateMul(nctaid, ntid, "combo.stride");
  stride = B.CreateIntCast(stride, B.getInt64Ty(), true, "combo.stride");

  // Persistent-grid combo loop.
  auto *loop_bb = BasicBlock::Create(C, "cmb.chk", func);
  auto *loop_body_bb = BasicBlock::Create(C, "cmb.body", func);
  auto *loop_inc_bb = BasicBlock::Create(C, "cmb.inc", func);
  auto *loop_done_bb = BasicBlock::Create(C, "cmb.done", func);

  auto *pre = B.GetInsertBlock();
  B.CreateBr(loop_bb);

  B.SetInsertPoint(loop_bb);
  auto *combo_id = B.CreatePHI(B.getInt64Ty(), 2, "combo");
  combo_id->addIncoming(global_tid, pre);
  B.CreateCondBr(B.CreateICmpULT(combo_id, combos_v), loop_body_bb, loop_done_bb);

  B.SetInsertPoint(loop_body_bb);
  {
    auto *sv_base = emit_combo_offset(B, combo_id, qubits, n_qubits);
    sv_base = B.CreateShl(sv_base, 1, "sv.base.idx"); // ×2 for re/im interleave
    sv_base = B.CreateGEP(scalar_ty, sv_root, sv_base, "sv.base");
    emit_matvec(B, qubits, n_qubits, mat_data, sv_base, scalar_ty);
    B.CreateBr(loop_inc_bb);
  }

  B.SetInsertPoint(loop_inc_bb);
  {
    auto *combo_next = B.CreateAdd(combo_id, stride, "combo.next");
    combo_id->addIncoming(combo_next, loop_inc_bb);
    B.CreateBr(loop_bb);
  }

  B.SetInsertPoint(loop_done_bb);
}

} // end anonymous namespace

// ── Public entry point ───────────────────────────────────────────────────────

llvm::Expected<llvm::Function *> cast_cuda_generate_kernel_ir(
    const cast_cuda_kernel_gen_spec_t &spec, const cast_cuda_complex64_t *matrix, size_t matrix_len,
    const uint32_t *qubits, size_t n_qubits, llvm::StringRef func_name, llvm::Module &module) {
  if (matrix == nullptr)
    return llvm::createStringError("matrix pointer must not be null");
  if (qubits == nullptr || n_qubits == 0)
    return llvm::createStringError("qubits must not be null/empty");

  const unsigned K = 1U << n_qubits;
  if (matrix_len != static_cast<size_t>(K) * K)
    return llvm::createStringError("matrix_len must equal (2^n_qubits)^2");

  if (spec.precision != CAST_CUDA_PRECISION_F32 && spec.precision != CAST_CUDA_PRECISION_F64)
    return llvm::createStringError("spec.precision must be F32 or F64");

  auto &ctx = module.getContext();
  IRBuilder<> B(ctx);

  Type *scalar_ty = (spec.precision == CAST_CUDA_PRECISION_F32) ? B.getFloatTy() : B.getDoubleTy();

  KernelArgs args;
  auto *func = create_kernel_function(B, module, func_name.str(), args);

  auto *entry_bb = BasicBlock::Create(ctx, "entry", func);
  B.SetInsertPoint(entry_bb);

  auto mat_data = build_matrix_data(B, spec, matrix, static_cast<unsigned>(n_qubits));

  emit_persistent_grid_loop(B, qubits, n_qubits, mat_data, args.p_sv, args.p_combos, scalar_ty);

  B.CreateRetVoid();

  std::string err_info;
  llvm::raw_string_ostream rso(err_info);
  if (llvm::verifyFunction(*func, &rso))
    return llvm::createStringError("Function verification failed: " + rso.str());

  // LLVM >= 21 requires PTX_Kernel calling convention (not just nvvm.annotations).
  func->setCallingConv(llvm::CallingConv::PTX_Kernel);
  return func;
}

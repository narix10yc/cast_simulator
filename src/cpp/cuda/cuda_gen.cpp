#include "cuda_gen.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>

#include <cassert>
#include <cmath>
#include <vector>

using namespace llvm;

namespace cast::cuda {

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
  Value *reVal;
  Value *imVal;
  ScalarKind reKind;
  ScalarKind imKind;
};

// ── emitOptFmul ──────────────────────────────────────────────────────────

static Value *emitOptFmul(Value *a, Value *b, ScalarKind aKind, IRBuilder<> &B) {
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

// ── buildMatrixData ──────────────────────────────────────────────────────

/// Classifies each matrix element as zero, ±1, or general, and returns the
/// corresponding LLVM constant value alongside its ScalarKind tag.
static std::vector<IRMatData> buildMatrixData(IRBuilder<> &B, const cast::cuda::KernelGenSpec &spec,
                                              const cast::Complex64 *matrix, unsigned nQubits) {
  const auto K = 1U << nQubits;
  const auto KK = K * K;

  // Scale tolerances by 1/K so that the classification becomes stricter
  // as the gate dimension grows.  Each output amplitude accumulates K
  // products; an element classified as zero contributes no error, while
  // a mis-classified zero introduces error proportional to |element|.
  // Dividing by K keeps the worst-case accumulated round-off from zero
  // classification constant regardless of gate size.
  //
  // Note: the CPU codegen (cpu/kernel.rs) applies ztol/otol directly
  // without this 1/K scaling.  The asymmetry is intentional — the CUDA
  // path is more conservative (classifies fewer elements as zero/±1) to
  // avoid accumulation artifacts in large fused gates, at the cost of a
  // few extra FP ops that the GPU easily absorbs.
  const auto zTol = spec.ztol / static_cast<double>(K);
  const auto oTol = spec.otol / static_cast<double>(K);

  const bool fp32 = (spec.precision == cast::Precision::F32);
  Type *scalarTy = fp32 ? B.getFloatTy() : B.getDoubleTy();
  auto *zeroVal = ConstantFP::get(scalarTy, 0.0);
  auto *oneVal = ConstantFP::get(scalarTy, 1.0);
  auto *minusOneVal = ConstantFP::get(scalarTy, -1.0);

  // Classify kind and build LLVM constant for a single scalar.
  auto classify = [&](double v) -> std::pair<Value *, ScalarKind> {
    if (spec.ztol > 0.0 && std::abs(v) < zTol)
      return {zeroVal, SK_Zero};
    if (spec.otol > 0.0 && std::abs(v - 1.0) < oTol)
      return {oneVal, SK_One};
    if (spec.otol > 0.0 && std::abs(v + 1.0) < oTol)
      return {minusOneVal, SK_MinusOne};
    return {ConstantFP::get(scalarTy, fp32 ? static_cast<double>(static_cast<float>(v)) : v),
            SK_ImmValue};
  };

  std::vector<IRMatData> data(KK);
  for (unsigned i = 0; i < KK; ++i) {
    auto [reVal, reKind] = classify(matrix[i].re);
    auto [imVal, imKind] = classify(matrix[i].im);
    data[i] = {reVal, imVal, reKind, imKind};
  }
  return data;
}

// ── createKernelFunction ───────────────────────────────────────────────────

static Function *createKernelFunction(IRBuilder<> &B, Module &M, const std::string &funcName,
                                      KernelArgs &args) {
  auto params = std::array<Type *, 3>{
      B.getPtrTy(),  // sv
      B.getPtrTy(),  // mat
      B.getInt64Ty() // combos
  };
  auto *fty = FunctionType::get(B.getVoidTy(), params, false);
  auto *func = Function::Create(fty, Function::ExternalLinkage, funcName, M);

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

// ── emitComboOffset ──────────────────────────────────────────────────────
//
// Maps a linear combo index to the statevector amplitude index using a
// PDEP-like bit scatter.  Target qubits mark positions that must be ZERO
// in the base address (they're filled in later by emitMatvec's delta
// loop); all other address bits come from the combo counter.
//
// Algorithm: scan qubit positions 0..highestQ.  Non-target positions
// consume counter bits in order.  Each time we reach a target position,
// the accumulated counter bits are shifted left by the number of target
// qubits seen so far (inserting zeros at the target positions).
// After the loop, remaining high counter bits are shifted by k (total
// target qubits).
//
// Worked example — target qubits [2, 4, 5], k=3, combo counter = 0b110_01:
//
//   Step 1: q=0,1 are non-target → consume counter bits b0,b1 (mask=0b11)
//           q=2 is target #0 → emit (counter & 0b11) << 0 = 0b01
//           Insert zero at position 2.
//   Step 2: q=3 is non-target → consume counter bit b2 (mask=0b100)
//           q=4 is target #1 → emit (counter & 0b100) << 1 = 0b1000
//           Insert zero at position 4.
//   Step 3: q=5 is target #2 → mask=0, nothing to emit.
//   Final:  high bits of counter (bits ≥ highestQ-k+1 = 3) → shift << 3.
//           counter bits b3,b4 = 0b11 → shifted << 3 = 0b11_000_000
//
//   Result: 0b11_0_0_1000_0_01 = positions [..., -, -, 1, 0, 0, 0, -, 0, 1]
//           where - are the target-qubit zeros (positions 2, 4, 5).
//
// The returned Value is the element index into the scalar array.  The
// caller applies ×2 for re/im interleave.

static Value *emitComboOffset(IRBuilder<> &B, Value *counterV, const uint32_t *qubits,
                              size_t nQubits) {
  assert(nQubits > 0);

  auto *offset = static_cast<Value *>(B.getInt64(0ULL));
  counterV = B.CreateZExt(counterV, B.getInt64Ty());

  const int k = static_cast<int>(nQubits);
  const int highestQ = static_cast<int>(qubits[nQubits - 1]);

  uint64_t mask = 0ULL;
  int qIdx = 0;
  int counterQ = 0;

  Value *tmpCounter = nullptr;

  for (int q = 0; q <= highestQ; q++) {
    if (q < static_cast<int>(qubits[qIdx])) {
      mask |= (1ULL << counterQ++);
      continue;
    }
    ++qIdx;
    if (mask == 0)
      continue;

    tmpCounter = B.CreateAnd(counterV, mask);
    tmpCounter = B.CreateShl(tmpCounter, (qIdx - 1));
    offset = B.CreateAdd(offset, tmpCounter);
    mask = 0ULL;
  }

  mask = ~((1ULL << (highestQ - k + 1)) - 1);
  tmpCounter = B.CreateAnd(counterV, mask);
  tmpCounter = B.CreateShl(tmpCounter, k);
  offset = B.CreateAdd(offset, tmpCounter);

  return offset;
}

// ── emitMatvec ─────────────────────────────────────────────────────────────
//
// Emits straight-line code that:
//   1. Loads all K amplitude pairs (re, im) from sv[svPtr + 2*delta{i}]
//   2. Accumulates M*v with constant (ImmValue) matrix entries
//   3. Stores the updated amplitudes back in place
//
// svPtr already points to the base amplitude for this combo (offset has been
// applied by the caller).

static void emitMatvec(IRBuilder<> &B, const uint32_t *qubits, size_t nQubits,
                       const std::vector<IRMatData> &matData, Value *svPtr, Type *scalarTy) {
  B.setFastMathFlags(FastMathFlags::getFast());

  const auto k = static_cast<unsigned>(nQubits);
  const unsigned K = 1u << k;

  // Vector type for complex pair (re, im) — enables ld.global.v2 / st.global.v2
  // in the NVPTX backend, doubling memory-transaction utilization for the
  // interleaved [re₀, im₀, re₁, im₁, …] layout.
  auto *vec2Ty = FixedVectorType::get(scalarTy, 2);

  std::vector<Value *> ampPtrs(K); // base pointer per amplitude (re position)
  std::vector<Value *> reAmps(K);
  std::vector<Value *> imAmps(K);

  // Compute the offset for each of the K amplitude slots in the combo.
  // Amplitude i corresponds to setting bits qubits[b] iff bit b of i is set.
  for (unsigned i = 0; i < K; ++i) {
    uint64_t delta = 0;
    for (unsigned b = 0; b < k; ++b)
      if (i & (1u << b))
        delta |= (1ull << qubits[b]);

    uint64_t const off2 = 2ull * delta;
    ampPtrs[i] = B.CreateConstGEP1_64(scalarTy, svPtr, off2);
    auto *pair = B.CreateLoad(vec2Ty, ampPtrs[i]);
    reAmps[i] = B.CreateExtractElement(pair, (uint64_t)0);
    imAmps[i] = B.CreateExtractElement(pair, (uint64_t)1);
  }

  // For each output row r:  new[r] = sum_c  M[r,c] * old[c]
  for (unsigned r = 0; r < K; ++r) {
    auto *accRe0 = static_cast<Value *>(ConstantFP::get(scalarTy, 0.0));
    auto *accRe1 = static_cast<Value *>(ConstantFP::get(scalarTy, 0.0));
    auto *accIm = static_cast<Value *>(ConstantFP::get(scalarTy, 0.0));

    for (unsigned c = 0; c < K; ++c) {
      const auto &md = matData[r * K + c];
      if (md.reKind == SK_Zero && md.imKind == SK_Zero)
        continue;

      // Re(new) = Re(M)*Re(old) - Im(M)*Im(old)
      if (auto *t0 = emitOptFmul(md.reVal, reAmps[c], md.reKind, B))
        accRe0 = B.CreateFAdd(accRe0, t0);
      if (auto *t1 = emitOptFmul(md.imVal, imAmps[c], md.imKind, B))
        accRe1 = B.CreateFAdd(accRe1, t1);

      // Im(new) = Re(M)*Im(old) + Im(M)*Re(old)
      if (auto *t2 = emitOptFmul(md.reVal, imAmps[c], md.reKind, B))
        accIm = B.CreateFAdd(accIm, t2);
      if (auto *t3 = emitOptFmul(md.imVal, reAmps[c], md.imKind, B))
        accIm = B.CreateFAdd(accIm, t3);
    }

    auto *newRe = B.CreateFSub(accRe0, accRe1);
    Value *out = PoisonValue::get(vec2Ty);
    out = B.CreateInsertElement(out, newRe, (uint64_t)0);
    out = B.CreateInsertElement(out, accIm, (uint64_t)1);
    B.CreateStore(out, ampPtrs[r]);
  }
}

// ── emitPersistentGridLoop ────────────────────────────────────────────────
//
// Wraps emitMatvec inside a persistent-grid loop:
//   for (combo = globalTid; combo < p.combos; combo += stride) { ... }

static void emitPersistentGridLoop(IRBuilder<> &B, const uint32_t *qubits, size_t nQubits,
                                   const std::vector<IRMatData> &matData, Value *svRoot,
                                   Value *combosV, Type *scalarTy) {
  auto *func = B.GetInsertBlock()->getParent();
  auto &C = B.getContext();

  // NVPTX thread-index intrinsics.
  auto *tid = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  auto *ntid = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ntid_x, {});
  auto *ctaid = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
  auto *nctaid = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_nctaid_x, {});

  // globalTid = ctaid * ntid + tid
  auto *globalTid = B.CreateAdd(B.CreateMul(ctaid, ntid), tid);
  globalTid = B.CreateIntCast(globalTid, B.getInt64Ty(), /*isSigned=*/true);

  // stride = nctaid * ntid  (total thread count)
  auto *stride = B.CreateMul(nctaid, ntid);
  stride = B.CreateIntCast(stride, B.getInt64Ty(), true);

  // Persistent-grid combo loop.
  auto *loopBb = BasicBlock::Create(C, "cmb.chk", func);
  auto *loopBodyBb = BasicBlock::Create(C, "cmb.body", func);
  auto *loopIncBb = BasicBlock::Create(C, "cmb.inc", func);
  auto *loopDoneBb = BasicBlock::Create(C, "cmb.done", func);

  auto *pre = B.GetInsertBlock();
  B.CreateBr(loopBb);

  B.SetInsertPoint(loopBb);
  auto *comboId = B.CreatePHI(B.getInt64Ty(), 2);
  comboId->addIncoming(globalTid, pre);
  B.CreateCondBr(B.CreateICmpULT(comboId, combosV), loopBodyBb, loopDoneBb);

  B.SetInsertPoint(loopBodyBb);
  {
    auto *svBase = emitComboOffset(B, comboId, qubits, nQubits);
    svBase = B.CreateShl(svBase, 1); // ×2 for re/im interleave
    svBase = B.CreateGEP(scalarTy, svRoot, svBase);
    emitMatvec(B, qubits, nQubits, matData, svBase, scalarTy);
    B.CreateBr(loopIncBb);
  }

  B.SetInsertPoint(loopIncBb);
  {
    auto *comboNext = B.CreateAdd(comboId, stride);
    comboId->addIncoming(comboNext, loopIncBb);
    B.CreateBr(loopBb);
  }

  B.SetInsertPoint(loopDoneBb);
}

} // end anonymous namespace

// ── Public entry point ───────────────────────────────────────────────────────

llvm::Expected<llvm::Function *> generateKernelIr(const KernelGenSpec &spec,
                                                  const cast::Complex64 *matrix, size_t matrixLen,
                                                  const uint32_t *qubits, size_t nQubits,
                                                  llvm::StringRef funcName, llvm::Module &module) {
  if (matrix == nullptr)
    return llvm::createStringError("matrix pointer must not be null");
  if (qubits == nullptr || nQubits == 0)
    return llvm::createStringError("qubits must not be null/empty");

  const unsigned K = 1U << nQubits;
  if (matrixLen != static_cast<size_t>(K) * K)
    return llvm::createStringError("matrixLen must equal (2^nQubits)^2");

  if (!cast::isValidPrecision(spec.precision))
    return llvm::createStringError("spec.precision must be F32 or F64");

  auto &ctx = module.getContext();
  IRBuilder<> B(ctx);

  Type *scalarTy = (spec.precision == cast::Precision::F32) ? B.getFloatTy() : B.getDoubleTy();

  KernelArgs args;
  auto *func = createKernelFunction(B, module, funcName.str(), args);

  auto *entryBb = BasicBlock::Create(ctx, "entry", func);
  B.SetInsertPoint(entryBb);

  auto matData = buildMatrixData(B, spec, matrix, static_cast<unsigned>(nQubits));

  emitPersistentGridLoop(B, qubits, nQubits, matData, args.p_sv, args.p_combos, scalarTy);

  B.CreateRetVoid();

  std::string errInfo;
  llvm::raw_string_ostream rso(errInfo);
  if (llvm::verifyFunction(*func, &rso))
    return llvm::createStringError("Function verification failed: " + rso.str());

  // LLVM >= 21 requires PTX_Kernel calling convention (not just nvvm.annotations).
  func->setCallingConv(llvm::CallingConv::PTX_Kernel);
  return func;
}

} // namespace cast::cuda

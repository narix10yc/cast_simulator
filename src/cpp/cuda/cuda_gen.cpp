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

struct IRArgsCUDA {
  Argument *pSvArg;  // ptr to statevector
  Argument *pMatArg; // ptr to matrix (unused in ImmValue mode)
  Argument *pCombos; // i64: total number of amplitude combos
};

struct IRMatDataCUDA {
  Value *reVal;
  Value *imVal;
  ScalarKind reKind;
  ScalarKind imKind;
};

// ── genOptFMul ──────────────────────────────────────────────────────────────

static Value *genOptFMul(Value *a, Value *b, ScalarKind aKind, IRBuilder<> &B) {
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

// ── getMatDataCUDA ──────────────────────────────────────────────────────────

static std::vector<IRMatDataCUDA> getMatDataCUDA(IRBuilder<> &B,
                                                 const cast_cuda_kernel_gen_spec_t &spec,
                                                 const cast_cuda_complex64_t *matrix,
                                                 unsigned nQubits) {
  const auto K = 1U << nQubits;
  const auto KK = K * K;

  const auto zTol = spec.ztol / static_cast<double>(K);
  const auto oTol = spec.otol / static_cast<double>(K);

  auto data = std::vector<IRMatDataCUDA>(KK);

  const bool fp32 = (spec.precision == CAST_CUDA_PRECISION_F32);

  // Classify kinds
  for (unsigned r = 0; r < K; ++r) {
    for (unsigned c = 0; c < K; ++c) {
      unsigned idx = r * K + c;
      double re = matrix[idx].re;
      double im = matrix[idx].im;

      // reKind
      if (spec.ztol > 0.0 && std::abs(re) < zTol)
        data[idx].reKind = SK_Zero;
      else if (spec.otol > 0.0 && std::abs(re - 1.0) < oTol)
        data[idx].reKind = SK_One;
      else if (spec.otol > 0.0 && std::abs(re + 1.0) < oTol)
        data[idx].reKind = SK_MinusOne;
      else
        data[idx].reKind = SK_ImmValue;

      // imKind
      if (spec.ztol > 0.0 && std::abs(im) < zTol)
        data[idx].imKind = SK_Zero;
      else if (spec.otol > 0.0 && std::abs(im - 1.0) < oTol)
        data[idx].imKind = SK_One;
      else if (spec.otol > 0.0 && std::abs(im + 1.0) < oTol)
        data[idx].imKind = SK_MinusOne;
      else
        data[idx].imKind = SK_ImmValue;
    }
  }

  // Build LLVM constant values
  Type *scalarTy = fp32 ? B.getFloatTy() : B.getDoubleTy();
  auto zeroVal = ConstantFP::get(scalarTy, 0.0);
  auto oneVal = ConstantFP::get(scalarTy, 1.0);
  auto minusOneVal = ConstantFP::get(scalarTy, -1.0);

  for (unsigned r = 0; r < K; ++r) {
    for (unsigned c = 0; c < K; ++c) {
      unsigned idx = r * K + c;
      double re = matrix[idx].re;
      double im = matrix[idx].im;

      switch (data[idx].reKind) {
      case SK_Zero:
        data[idx].reVal = zeroVal;
        break;
      case SK_One:
        data[idx].reVal = oneVal;
        break;
      case SK_MinusOne:
        data[idx].reVal = minusOneVal;
        break;
      case SK_ImmValue:
        data[idx].reVal =
            ConstantFP::get(scalarTy, fp32 ? static_cast<double>(static_cast<float>(re)) : re);
        break;
      default:
        break;
      }

      switch (data[idx].imKind) {
      case SK_Zero:
        data[idx].imVal = zeroVal;
        break;
      case SK_One:
        data[idx].imVal = oneVal;
        break;
      case SK_MinusOne:
        data[idx].imVal = minusOneVal;
        break;
      case SK_ImmValue:
        data[idx].imVal =
            ConstantFP::get(scalarTy, fp32 ? static_cast<double>(static_cast<float>(im)) : im);
        break;
      default:
        break;
      }
    }
  }

  return data;
}

// ── getFunctionDeclarationCUDA ───────────────────────────────────────────────

static Function *getFunctionDeclarationCUDA(IRBuilder<> &B, Module &M, const std::string &funcName,
                                            IRArgsCUDA &args) {
  auto params = std::array<Type *, 3>{
      B.getPtrTy(),  // sv
      B.getPtrTy(),  // mat
      B.getInt64Ty() // combos
  };
  auto *fty = FunctionType::get(B.getVoidTy(), params, false);
  auto *func = Function::Create(fty, Function::ExternalLinkage, funcName, M);

  args.pSvArg = func->getArg(0);
  args.pSvArg->setName("p.sv");
  args.pMatArg = func->getArg(1);
  args.pMatArg->setName("p.mat");
  args.pCombos = func->getArg(2);
  args.pCombos->setName("p.combos");

  // Mark as a CUDA (PTX) kernel via nvvm.annotations metadata.
  auto *mdString = MDString::get(M.getContext(), "kernel");
  auto *mdOne = ConstantAsMetadata::get(B.getInt32(1));
  auto *md = MDNode::get(M.getContext(), {ValueAsMetadata::get(func), mdString, mdOne});
  M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md);
  return func;
}

// ── buildOffset ─────────────────────────────────────────────────────────────
//
// Maps a combo counter to the base statevector amplitude index (in units of
// scalars) using a compile-time-known PDEP-like bit scatter.
//
// Example: target qubits [2, 4, 5], counter xxxhgfedcba
//   result (in scalar units) = hgfed00c0ba * 2
//
// The returned Value is the byte-GEP index into the scalar array (×2 shift
// for re/im is applied by the caller).

static Value *buildOffset(IRBuilder<> &B, Value *counterV, const uint32_t *qubits,
                          size_t n_qubits) {
  assert(n_qubits > 0);

  auto *offset = static_cast<Value *>(B.getInt64(0ULL));
  counterV = B.CreateZExt(counterV, B.getInt64Ty(), "i64.counter");

  const int k = static_cast<int>(n_qubits);
  const int highestQ = static_cast<int>(qubits[n_qubits - 1]);

  uint64_t mask = 0ULL;
  int qIdx = 0;
  int counterQ = 0;

  Value *tmpCounterV = nullptr;

  for (int q = 0; q <= highestQ; q++) {
    if (q < static_cast<int>(qubits[qIdx])) {
      mask |= (1ULL << counterQ++);
      continue;
    }
    ++qIdx;
    if (mask == 0)
      continue;

    tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
    tmpCounterV = B.CreateShl(tmpCounterV, (qIdx - 1), "tmpCounter");
    offset = B.CreateAdd(offset, tmpCounterV, "tmpIdx");
    mask = 0ULL;
  }

  mask = ~((1ULL << (highestQ - k + 1)) - 1);
  tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
  tmpCounterV = B.CreateShl(tmpCounterV, k, "tmpCounter");
  offset = B.CreateAdd(offset, tmpCounterV, "offset");

  return offset;
}

// ── genMatrixVectorMultiply_InlineImm ────────────────────────────────────────
//
// Emits straight-line code that:
//   1. Loads all K amplitude pairs (re, im) from sv[svPtrV + 2*delta{i}]
//   2. Accumulates M*v with constant (ImmValue) matrix entries
//   3. Stores the updated amplitudes back in place
//
// svPtrV already points to the base amplitude for this combo (offset has been
// applied by the caller).

static void genMatrixVectorMultiply_InlineImm(IRBuilder<> &B, const uint32_t *qubits,
                                              size_t n_qubits,
                                              const std::vector<IRMatDataCUDA> &matData,
                                              Value *svPtrV, Type *scalarTy) {
  B.setFastMathFlags(FastMathFlags::getFast());

  const unsigned k = static_cast<unsigned>(n_qubits);
  const unsigned K = 1u << k;

  std::vector<Value *> reAmpPtrs(K), imAmpPtrs(K);
  std::vector<Value *> reAmps(K), imAmps(K);

  // Compute the offset for each of the K amplitude slots in the combo.
  // Amplitude i corresponds to setting bits qubits[b] iff bit b of i is set.
  for (unsigned i = 0; i < K; ++i) {
    uint64_t delta = 0;
    for (unsigned b = 0; b < k; ++b)
      if (i & (1u << b))
        delta |= (1ull << qubits[b]);

    uint64_t off2 = 2ull * delta;
    reAmpPtrs[i] = B.CreateConstGEP1_64(scalarTy, svPtrV, off2, "re.ptr");
    imAmpPtrs[i] = B.CreateConstGEP1_64(scalarTy, svPtrV, off2 + 1, "im.ptr");
    reAmps[i] = B.CreateLoad(scalarTy, reAmpPtrs[i], "re.amp");
    imAmps[i] = B.CreateLoad(scalarTy, imAmpPtrs[i], "im.amp");
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
      if (auto *t0 = genOptFMul(md.reVal, reAmps[c], md.reKind, B))
        accRe0 = B.CreateFAdd(accRe0, t0);
      if (auto *t1 = genOptFMul(md.imVal, imAmps[c], md.imKind, B))
        accRe1 = B.CreateFAdd(accRe1, t1);

      // Im(new) = Re(M)*Im(old) + Im(M)*Re(old)
      if (auto *t2 = genOptFMul(md.reVal, imAmps[c], md.reKind, B))
        accIm = B.CreateFAdd(accIm, t2);
      if (auto *t3 = genOptFMul(md.imVal, reAmps[c], md.imKind, B))
        accIm = B.CreateFAdd(accIm, t3);
    }

    auto *newRe = B.CreateFSub(accRe0, accRe1, "new.re");
    B.CreateStore(newRe, reAmpPtrs[r]);
    B.CreateStore(accIm, imAmpPtrs[r]);
  }
}

// ── genMatVecMul_Imm ─────────────────────────────────────────────────────────
//
// Wraps genMatrixVectorMultiply_InlineImm inside a persistent-grid loop:
//   for (combo = global_tid; combo < p.combos; combo += stride) { ... }

static void genMatVecMul_Imm(IRBuilder<> &B, const uint32_t *qubits, size_t n_qubits,
                             const std::vector<IRMatDataCUDA> &matData, Value *svRoot,
                             Type *scalarTy) {
  auto *func = B.GetInsertBlock()->getParent();
  auto &C = B.getContext();

  // Retrieve p.combos argument by name.
  Argument *combosV = nullptr;
  for (auto &A : func->args()) {
    if (A.getName() == "p.combos") {
      combosV = &A;
      break;
    }
  }
  assert(combosV && "Missing kernel arg 'p.combos'");

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
  auto *cmbChk = BasicBlock::Create(C, "cmb.chk", func);
  auto *cmbBody = BasicBlock::Create(C, "cmb.body", func);
  auto *cmbInc = BasicBlock::Create(C, "cmb.inc", func);
  auto *cmbDone = BasicBlock::Create(C, "cmb.done", func);

  auto *pre = B.GetInsertBlock();
  B.CreateBr(cmbChk);

  B.SetInsertPoint(cmbChk);
  auto *comboid = B.CreatePHI(B.getInt64Ty(), 2, "combo");
  comboid->addIncoming(global_tid, pre);
  B.CreateCondBr(B.CreateICmpULT(comboid, combosV), cmbBody, cmbDone);

  B.SetInsertPoint(cmbBody);
  {
    auto *svBase = buildOffset(B, comboid, qubits, n_qubits);
    svBase = B.CreateShl(svBase, 1, "sv.base.idx"); // ×2 for re/im interleave
    svBase = B.CreateGEP(scalarTy, svRoot, svBase, "sv.base");
    genMatrixVectorMultiply_InlineImm(B, qubits, n_qubits, matData, svBase, scalarTy);
    B.CreateBr(cmbInc);
  }

  B.SetInsertPoint(cmbInc);
  {
    auto *nextCombo = B.CreateAdd(comboid, stride, "combo.next");
    comboid->addIncoming(nextCombo, cmbInc);
    B.CreateBr(cmbChk);
  }

  B.SetInsertPoint(cmbDone);
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

  Type *scalarTy = (spec.precision == CAST_CUDA_PRECISION_F32) ? B.getFloatTy() : B.getDoubleTy();

  IRArgsCUDA args;
  auto *func = getFunctionDeclarationCUDA(B, module, func_name.str(), args);

  auto *entryBB = BasicBlock::Create(ctx, "entry", func);
  B.SetInsertPoint(entryBB);

  auto matData = getMatDataCUDA(B, spec, matrix, static_cast<unsigned>(n_qubits));

  genMatVecMul_Imm(B, qubits, n_qubits, matData, args.pSvArg, scalarTy);

  B.CreateRetVoid();

  std::string errInfo;
  llvm::raw_string_ostream rso(errInfo);
  if (llvm::verifyFunction(*func, &rso))
    return llvm::createStringError("Function verification failed: " + rso.str());

  // LLVM >= 21 requires PTX_Kernel calling convention (not just nvvm.annotations).
  func->setCallingConv(llvm::CallingConv::PTX_Kernel);
  return func;
}

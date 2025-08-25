#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/Core/KernelGenInternal.h"

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Verifier.h"

#include "utils/Formats.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include <numeric>

#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "codegen-cuda"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

static FixedVectorType* getVec2Ty(Type* scalarTy) {
  return FixedVectorType::get(scalarTy, 2); // length = 2, non-scalable
}

/// Bit-cast a scalar* to a vec2* **without** changing the address-space.
static Value* bitCastPtrToVec2(IRBuilder<>& B, Value* ptr, Type* scalarTy) {
  auto* vec2Ty = getVec2Ty(scalarTy);
  unsigned AS = llvm::cast<PointerType>(ptr->getType())->getAddressSpace();
  return B.CreateBitCast(ptr, PointerType::get(vec2Ty, AS), "as_vec2");
}

namespace {

Value* genOptFMul(Value* a, Value* b, ScalarKind aKind, IRBuilder<>& B) {
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
    return nullptr;
  }
}

struct IRArgsCUDA {
  Argument* pSvArg;  // ptr to statevector
  Argument* pMatArg; // ptr to matrix
  Argument* pCombos;
};

// reVal and imVal will be llvm constants if matrixLoadMode is UseMatImmValues
// otherwise, they will be loaded from the matrix pointer
struct IRMatDataCUDA {
  Value* reVal;
  Value* imVal;
  ScalarKind reKind;
  ScalarKind imKind;
};

std::vector<IRMatDataCUDA> getMatDataCUDA(IRBuilder<>& B,
                                          const CUDAKernelGenConfig& config,
                                          const ComplexSquareMatrix& matrix,
                                          int nQubits) {
  const unsigned K = 1U << nQubits;
  const unsigned KK = K * K;

  // Scale the tolerances by matrix size
  const double zTol = config.zeroTol / K;
  const double oTol = config.oneTol / K;

  std::vector<IRMatDataCUDA> data(KK);

  // reKind and imKind
  if (config.forceDenseKernel) {
    if (config.matrixLoadMode == CUDAMatrixLoadMode::UseMatImmValues) {
      // forceDenseKernel and UseMatImmValues: everything is an immediate value
      for (unsigned i = 0; i < KK; ++i) {
        data[i].reKind = SK_ImmValue;
        data[i].imKind = SK_ImmValue;
      }
    } else {
      // forceDenseKernel and not UseMatImmValues:
      // everything is runtime value, i.e. loaded from the matrix pointer
      for (unsigned i = 0; i < KK; ++i) {
        data[i].reKind = SK_Runtime;
        data[i].imKind = SK_Runtime;
      }
    }
  } else {
    // not forceDenseKernel: determine reKind and imKind based on the matrix
    // elements
    for (unsigned r = 0; r < K; ++r) {
      for (unsigned c = 0; c < K; ++c) {
        unsigned idx = r * K + c;
        auto elem = matrix.rc(r, c);

        // reKind
        if (std::abs(elem.real()) < zTol)
          data[idx].reKind = SK_Zero;
        else if (std::abs(elem.real() - 1.0) < oTol)
          data[idx].reKind = SK_One;
        else if (std::abs(elem.real() + 1.0) < oTol)
          data[idx].reKind = SK_MinusOne;
        else if (config.matrixLoadMode == CUDAMatrixLoadMode::UseMatImmValues)
          data[idx].reKind = SK_ImmValue;
        else
          data[idx].reKind = SK_Runtime;

        // imKind
        if (std::abs(elem.imag()) < zTol)
          data[idx].imKind = SK_Zero;
        else if (std::abs(elem.imag() - 1.0) < oTol)
          data[idx].imKind = SK_One;
        else if (std::abs(elem.imag() + 1.0) < oTol)
          data[idx].imKind = SK_MinusOne;
        else if (config.matrixLoadMode == CUDAMatrixLoadMode::UseMatImmValues)
          data[idx].imKind = SK_ImmValue;
        else
          data[idx].imKind = SK_Runtime;
      }
    }
  }

  // reVal and imVal
  auto zeroVal = ConstantFP::get(
      B.getContext(),
      (config.precision == Precision::F32) ? APFloat(0.0f) : APFloat(0.0));
  auto oneVal = ConstantFP::get(
      B.getContext(),
      (config.precision == Precision::F32) ? APFloat(1.0f) : APFloat(1.0));
  auto minusOneVal = ConstantFP::get(
      B.getContext(),
      (config.precision == Precision::F32) ? APFloat(-1.0f) : APFloat(-1.0));

  for (unsigned i = 0; i < KK; ++i) {
    switch (data[i].reKind) {
    case SK_Runtime:
      break;
    case SK_Zero:
      data[i].reVal = zeroVal;
      break;
    case SK_One:
      data[i].reVal = oneVal;
      break;
    case SK_MinusOne:
      data[i].reVal = minusOneVal;
      break;
    case SK_ImmValue: {
      auto re = matrix.reData()[i];
      data[i].reVal = ConstantFP::get(B.getContext(),
                                      (config.precision == Precision::F32)
                                          ? APFloat(static_cast<float>(re))
                                          : APFloat(static_cast<double>(re)));
      break;
    }
    default:
      assert(false && "Unknown ScalarKind for reVal");
    }

    switch (data[i].imKind) {
    case SK_Runtime:
      break;
    case SK_Zero:
      data[i].imVal = zeroVal;
      break;
    case SK_One:
      data[i].imVal = oneVal;
      break;
    case SK_MinusOne:
      data[i].imVal = minusOneVal;
      break;
    case SK_ImmValue: {
      auto im = matrix.imData()[i];
      data[i].imVal = ConstantFP::get(B.getContext(),
                                      (config.precision == Precision::F32)
                                          ? APFloat(static_cast<float>(im))
                                          : APFloat(static_cast<double>(im)));
      break;
    }
    default:
      assert(false && "Unknown ScalarKind for imVal");
    }
  }
  return data;
}

static uint64_t
hashMatrixImm(const ComplexSquareMatrix& M, unsigned N, bool f32) {
  // 2*N*N scalars (re,im)
  const size_t count = 2ull * N * N;
  uint64_t h = 1469598103934665603ull; // FNV-1a (ok for cache keys)

  if (f32) {
    const float* p = reinterpret_cast<const float*>(M.reData());
    const float* q = reinterpret_cast<const float*>(M.imData());
    for (size_t i = 0; i < count / 2; i++) {
      h ^= std::bit_cast<uint32_t>(p[i]);
      h *= 1099511628211ull;
    }
    for (size_t i = 0; i < count / 2; i++) {
      h ^= std::bit_cast<uint32_t>(q[i]);
      h *= 1099511628211ull;
    }
  } else {
    const double* p = reinterpret_cast<const double*>(M.reData());
    const double* q = reinterpret_cast<const double*>(M.imData());
    for (size_t i = 0; i < count / 2; i++) {
      h ^= std::bit_cast<uint64_t>(p[i]);
      h *= 1099511628211ull;
    }
    for (size_t i = 0; i < count / 2; i++) {
      h ^= std::bit_cast<uint64_t>(q[i]);
      h *= 1099511628211ull;
    }
  }
  return h;
}

Function* getFunctionDeclarationCUDA(IRBuilder<>& B,
                                     Module& M,
                                     const std::string& funcName,
                                     const CUDAKernelGenConfig& config,
                                     IRArgsCUDA& args) {
  const bool needsMatArg =
      (config.matrixLoadMode == CUDAMatrixLoadMode::LoadInDefaultMemSpace);

  // FunctionType *fty = needsMatArg
  //                         ? FunctionType::get(
  //                               B.getVoidTy(), {B.getPtrTy(), B.getPtrTy()},
  //                               false)
  //                         : FunctionType::get(
  //                               B.getVoidTy(), {B.getPtrTy()}, false);
  SmallVector<Type*, 3> params;
  params.push_back(B.getPtrTy()); // p.sv
  if (needsMatArg)
    params.push_back(B.getPtrTy()); // p.mat
  params.push_back(B.getInt32Ty()); // p.combos
  FunctionType* fty = FunctionType::get(B.getVoidTy(), params, false);

  auto* func = Function::Create(fty, Function::ExternalLinkage, funcName, M);

  args.pSvArg = func->getArg(0);
  args.pSvArg->setName("p.sv");

  if (needsMatArg) {
    args.pMatArg = func->getArg(1);
    args.pMatArg->setName("p.mat");
    args.pCombos = func->getArg(2);
    args.pCombos->setName("p.combos");
  } else {
    args.pMatArg = nullptr;
    args.pCombos = func->getArg(1);
    args.pCombos->setName("p.combos");
  }

  // mark as kernel (unchanged)
  auto* mdString = MDString::get(M.getContext(), "kernel");
  auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
  auto* md = MDNode::get(M.getContext(),
                         {ValueAsMetadata::get(func), mdString, mdOne});
  M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md);
  return func;
}

Value* getGlobalTidCUDA(IRBuilder<>& B) {
  // thread index
  auto* tidV = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, nullptr, "tid");
  // gridSize (number of threads in each block)
  auto* gridSizeV = B.CreateIntrinsic(B.getInt32Ty(),
                                      Intrinsic::nvvm_read_ptx_sreg_ntid_x,
                                      {},
                                      nullptr,
                                      "blockSize");
  // block index
  auto* bidV = B.CreateIntrinsic(B.getInt32Ty(),
                                 Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
                                 {},
                                 nullptr,
                                 "bid");
  // global tid = grid size * block index + thread index
  auto* globalTidV = B.CreateMul(bidV, gridSizeV);
  globalTidV = B.CreateAdd(globalTidV, tidV, "counter.i32");
  globalTidV = B.CreateIntCast(globalTidV, B.getInt64Ty(), true, "global.tid");
  return globalTidV;
}

void attachNoUnrollMetadata(IRBuilder<>& B, BasicBlock* latchBB) {
  auto* latchTerm = latchBB->getTerminator();
  if (!latchTerm)
    return; // safety

  LLVMContext& ctx = B.getContext();
  auto* noUnrollMD =
      MDNode::get(ctx, {MDString::get(ctx, "llvm.loop.unroll.disable")});
  latchTerm->setMetadata("llvm.loop", noUnrollMD);
}

/**
 * Create a global array [2*K*K x scalarTy] to store the real/imag parts
 * from matData. That way, we can do a run-time IR loop to read them,
 * instead of unrolling a for-loop in C++.
 */
GlobalVariable*
createGlobalMatrixArray_NoUnroll(Module& M,
                                 Type* scalarTy,
                                 const std::vector<IRMatDataCUDA>& matData,
                                 unsigned K,
                                 const std::string& globalName) {
  unsigned totalElems = 2U * K * K;
  ArrayType* arrTy = ArrayType::get(scalarTy, totalElems);

  // Build a ConstantArray with the imm real/imag values.
  std::vector<Constant*> constVals;
  constVals.reserve(totalElems);
  for (unsigned i = 0; i < K * K; i++) {
    const auto& md = matData[i];
    ConstantFP* cRe = dyn_cast_or_null<ConstantFP>(md.reVal);
    ConstantFP* cIm = dyn_cast_or_null<ConstantFP>(md.imVal);
    double fallbackRe = 0.0, fallbackIm = 0.0;

    if (md.reKind == SK_ImmValue && cRe) {
      constVals.push_back(cRe);
    } else if (md.reKind == SK_One) {
      constVals.push_back(ConstantFP::get(scalarTy, 1.0));
    } else if (md.reKind == SK_MinusOne) {
      constVals.push_back(ConstantFP::get(scalarTy, -1.0));
    } else if (md.reKind == SK_Zero) {
      constVals.push_back(ConstantFP::get(scalarTy, 0.0));
    } else {
      constVals.push_back(ConstantFP::get(scalarTy, fallbackRe));
    }

    if (md.imKind == SK_ImmValue && cIm) {
      constVals.push_back(cIm);
    } else if (md.imKind == SK_One) {
      constVals.push_back(ConstantFP::get(scalarTy, 1.0));
    } else if (md.imKind == SK_MinusOne) {
      constVals.push_back(ConstantFP::get(scalarTy, -1.0));
    } else if (md.imKind == SK_Zero) {
      constVals.push_back(ConstantFP::get(scalarTy, 0.0));
    } else {
      constVals.push_back(ConstantFP::get(scalarTy, fallbackIm));
    }
  }

  auto* arrInit = ConstantArray::get(arrTy, constVals);

  auto* gVar = new GlobalVariable(M,
                                  arrTy,
                                  /*isConstant=*/true,
                                  GlobalValue::PrivateLinkage,
                                  arrInit,
                                  globalName);
  gVar->setAlignment(MaybeAlign(8));
  return gVar;
}

void genMatrixVectorMultiply(IRBuilder<>& B,
                             const CUDAKernelGenConfig& config,
                             const ComplexSquareMatrix& matrix,
                             const QuantumGate::TargetQubitsType& qubits,
                             const std::vector<IRMatDataCUDA>& matData,
                             Value* svPtrV,
                             Type* scalarTy) {
  unsigned k = qubits.size();
  unsigned K = 1u << k;

  LLVMContext& ctx = B.getContext();
  BasicBlock* curBB = B.GetInsertBlock();
  Function* func = curBB->getParent();
  Module& M = *func->getParent();

  GlobalVariable* gMatImmediate = createGlobalMatrixArray_NoUnroll(
      M, scalarTy, matData, K, "gMatImmediate");

  ArrayType* arrTy = ArrayType::get(scalarTy, K);
  AllocaInst* reAmpsAlloca = B.CreateAlloca(arrTy, nullptr, "reAmps");
  AllocaInst* imAmpsAlloca = B.CreateAlloca(arrTy, nullptr, "imAmps");

  BasicBlock* entryBB = curBB;
  BasicBlock* loadCheckBB = BasicBlock::Create(ctx, "load.check", func);
  BasicBlock* loadBodyBB = BasicBlock::Create(ctx, "load.body", func);
  BasicBlock* loadIncBB = BasicBlock::Create(ctx, "load.inc", func);
  BasicBlock* loadExitBB = BasicBlock::Create(ctx, "load.exit", func);

  B.CreateBr(loadCheckBB);
  B.SetInsertPoint(loadCheckBB);
  PHINode* iPHI = B.CreatePHI(B.getInt32Ty(), 2, "i");
  iPHI->addIncoming(ConstantInt::get(B.getInt32Ty(), 0), entryBB);
  Value* condLoad = B.CreateICmpSLT(iPHI, ConstantInt::get(B.getInt32Ty(), K));
  B.CreateCondBr(condLoad, loadBodyBB, loadExitBB);

  B.SetInsertPoint(loadBodyBB);
  {
    // compute delta
    Value* deltaVal = ConstantInt::get(B.getInt64Ty(), 0);
    for (unsigned b = 0; b < k; b++) {
      Value* maskB = ConstantInt::get(B.getInt32Ty(), 1u << b);
      Value* test = B.CreateAnd(iPHI, maskB);
      Value* cond = B.CreateICmpNE(test, ConstantInt::get(B.getInt32Ty(), 0));
      Value* shiftVal = ConstantInt::get(B.getInt64Ty(), 1ULL << qubits[b]);
      Value* orVal = B.CreateOr(deltaVal, shiftVal);
      deltaVal = B.CreateSelect(cond, orVal, deltaVal);
    }

    // rePtr = svPtrV + 2*deltaVal
    // imPtr = svPtrV + 2*deltaVal+1
    Value* twoDelta =
        B.CreateMul(ConstantInt::get(B.getInt64Ty(), 2), deltaVal);
    Value* rePtr = B.CreateGEP(scalarTy, svPtrV, twoDelta, "rePtr");
    Value* imPtr =
        B.CreateGEP(scalarTy,
                    svPtrV,
                    B.CreateAdd(twoDelta, ConstantInt::get(B.getInt64Ty(), 1)),
                    "imPtr");

    // load oldRe, oldIm
    Value* oldRe = B.CreateLoad(scalarTy, rePtr, "oldRe");
    Value* oldIm = B.CreateLoad(scalarTy, imPtr, "oldIm");

    // store into reAmpsAlloca[i], imAmpsAlloca[i]
    Value* reSlot = B.CreateGEP(
        arrTy, reAmpsAlloca, {ConstantInt::get(B.getInt32Ty(), 0), iPHI});
    Value* imSlot = B.CreateGEP(
        arrTy, imAmpsAlloca, {ConstantInt::get(B.getInt32Ty(), 0), iPHI});
    B.CreateStore(oldRe, reSlot);
    B.CreateStore(oldIm, imSlot);
  }
  B.CreateBr(loadIncBB);

  B.SetInsertPoint(loadIncBB);
  {
    Value* iNext = B.CreateAdd(iPHI, ConstantInt::get(B.getInt32Ty(), 1));
    iPHI->addIncoming(iNext, loadIncBB);
    B.CreateBr(loadCheckBB);
  }

  B.SetInsertPoint(loadExitBB);
  attachNoUnrollMetadata(B,
                         loadIncBB); // disable unroll for amplitude-load loop
  // Outer r-loop scaffolding
  BasicBlock* outerCheckBB = BasicBlock::Create(ctx, "outer.check", func);
  BasicBlock* outerBodyBB = BasicBlock::Create(ctx, "outer.body", func);
  BasicBlock* outerIncBB = BasicBlock::Create(ctx, "outer.inc", func);
  BasicBlock* outerExitBB = BasicBlock::Create(ctx, "outer.exit", func);

  B.CreateBr(outerCheckBB);
  B.SetInsertPoint(outerCheckBB);
  PHINode* rPHI = B.CreatePHI(B.getInt32Ty(), 2, "r");
  rPHI->addIncoming(ConstantInt::get(B.getInt32Ty(), 0), loadExitBB);
  Value* condOuter = B.CreateICmpSLT(rPHI, ConstantInt::get(B.getInt32Ty(), K));
  B.CreateCondBr(condOuter, outerBodyBB, outerExitBB);

  B.SetInsertPoint(outerBodyBB);
  {
    // partial sums in local allocas
    AllocaInst* reAmp0A = B.CreateAlloca(scalarTy, nullptr, "reAmp0A");
    AllocaInst* reAmp1A = B.CreateAlloca(scalarTy, nullptr, "reAmp1A");
    AllocaInst* imAmpA = B.CreateAlloca(scalarTy, nullptr, "imAmpA");

    Value* zeroVal = ConstantFP::get(scalarTy, 0.0);
    B.CreateStore(zeroVal, reAmp0A);
    B.CreateStore(zeroVal, reAmp1A);
    B.CreateStore(zeroVal, imAmpA);

    // Inner c-loop scaffolding
    BasicBlock* innerCheckBB = BasicBlock::Create(ctx, "inner.check", func);
    BasicBlock* innerBodyBB = BasicBlock::Create(ctx, "inner.body", func);
    BasicBlock* innerIncBB = BasicBlock::Create(ctx, "inner.inc", func);
    BasicBlock* innerExitBB = BasicBlock::Create(ctx, "inner.exit", func);

    B.CreateBr(innerCheckBB);
    B.SetInsertPoint(innerCheckBB);
    PHINode* cPHI = B.CreatePHI(B.getInt32Ty(), 2, "c");
    cPHI->addIncoming(ConstantInt::get(B.getInt32Ty(), 0), outerBodyBB);
    Value* condInner =
        B.CreateICmpSLT(cPHI, ConstantInt::get(B.getInt32Ty(), K));
    B.CreateCondBr(condInner, innerBodyBB, innerExitBB);

    // innerBody: read M[r,c], oldAmp[c], accumulate
    B.SetInsertPoint(innerBodyBB);
    {
      // linear index = r*K + c => 2*(r*K + c) => re/im
      Value* r64 = B.CreateIntCast(rPHI, B.getInt64Ty(), false);
      Value* c64 = B.CreateIntCast(cPHI, B.getInt64Ty(), false);
      Value* bigK = ConstantInt::get(B.getInt64Ty(), K);
      Value* rK = B.CreateMul(r64, bigK);
      Value* rc = B.CreateAdd(rK, c64);
      Value* base2 = B.CreateMul(ConstantInt::get(B.getInt64Ty(), 2), rc);

      // GEP into gMatImmediate => real, imag
      // global array type is [2*K*K x scalarTy]
      auto* arrTy2 = llvm::cast<ArrayType>(gMatImmediate->getValueType());

      // rePtr => gMatImmediate[0, base2]
      // imPtr => gMatImmediate[0, base2+1]
      Value* idxRe = base2;
      Value* idxIm = B.CreateAdd(base2, ConstantInt::get(B.getInt64Ty(), 1));
      Value* rePtr =
          B.CreateGEP(arrTy2,
                      gMatImmediate,
                      {ConstantInt::get(B.getInt32Ty(), 0),
                       B.CreateIntCast(idxRe, B.getInt32Ty(), false)},
                      "matRePtr");
      Value* imPtr =
          B.CreateGEP(arrTy2,
                      gMatImmediate,
                      {ConstantInt::get(B.getInt32Ty(), 0),
                       B.CreateIntCast(idxIm, B.getInt32Ty(), false)},
                      "matImPtr");

      Value* matRe = B.CreateLoad(scalarTy, rePtr, "matRe");
      Value* matIm = B.CreateLoad(scalarTy, imPtr, "matIm");

      // oldAmp[c]
      Value* oldReSlot = B.CreateGEP(
          arrTy, reAmpsAlloca, {ConstantInt::get(B.getInt32Ty(), 0), cPHI});
      Value* oldImSlot = B.CreateGEP(
          arrTy, imAmpsAlloca, {ConstantInt::get(B.getInt32Ty(), 0), cPHI});
      Value* oldRe = B.CreateLoad(scalarTy, oldReSlot, "oldRe");
      Value* oldIm = B.CreateLoad(scalarTy, oldImSlot, "oldIm");

      // partial sums
      Value* reA0 = B.CreateLoad(scalarTy, reAmp0A);
      Value* reA1 = B.CreateLoad(scalarTy, reAmp1A);
      Value* imA = B.CreateLoad(scalarTy, imAmpA);

      // reA0 += matRe * oldRe
      Value* addRe0 = B.CreateFAdd(reA0, B.CreateFMul(matRe, oldRe));
      B.CreateStore(addRe0, reAmp0A);

      // reA1 += matIm * oldIm
      Value* addRe1 = B.CreateFAdd(reA1, B.CreateFMul(matIm, oldIm));
      B.CreateStore(addRe1, reAmp1A);

      // imA += matRe*oldIm + matIm*oldRe
      Value* cross =
          B.CreateFAdd(B.CreateFMul(matRe, oldIm), B.CreateFMul(matIm, oldRe));
      Value* addIm = B.CreateFAdd(imA, cross);
      B.CreateStore(addIm, imAmpA);
    }
    B.CreateBr(innerIncBB);

    B.SetInsertPoint(innerIncBB);
    {
      Value* cNext = B.CreateAdd(cPHI, ConstantInt::get(B.getInt32Ty(), 1));
      cPHI->addIncoming(cNext, innerIncBB);
      B.CreateBr(innerCheckBB);
    }

    B.SetInsertPoint(innerExitBB);
    attachNoUnrollMetadata(B, innerIncBB);

    // finalize amplitude: newReAmp = reA0 - reA1, newImAmp = imA
    Value* reA0 = B.CreateLoad(scalarTy, reAmp0A);
    Value* reA1 = B.CreateLoad(scalarTy, reAmp1A);
    Value* imA = B.CreateLoad(scalarTy, imAmpA);
    Value* newReAmp = B.CreateFSub(reA0, reA1);
    Value* newImAmp = imA;

    // store back to row r
    Value* deltaVal = ConstantInt::get(B.getInt64Ty(), 0);
    for (unsigned b = 0; b < k; b++) {
      Value* maskB = ConstantInt::get(B.getInt32Ty(), 1u << b);
      Value* test = B.CreateAnd(rPHI, maskB);
      Value* cond = B.CreateICmpNE(test, ConstantInt::get(B.getInt32Ty(), 0));
      Value* shiftVal = ConstantInt::get(B.getInt64Ty(), 1ULL << qubits[b]);
      Value* orVal = B.CreateOr(deltaVal, shiftVal);
      deltaVal = B.CreateSelect(cond, orVal, deltaVal);
    }
    Value* twoDelta =
        B.CreateMul(ConstantInt::get(B.getInt64Ty(), 2), deltaVal);
    Value* outRePtr = B.CreateGEP(scalarTy, svPtrV, twoDelta);
    Value* outImPtr =
        B.CreateGEP(scalarTy,
                    svPtrV,
                    B.CreateAdd(twoDelta, ConstantInt::get(B.getInt64Ty(), 1)));
    B.CreateStore(newReAmp, outRePtr);
    B.CreateStore(newImAmp, outImPtr);
  }
  B.CreateBr(outerIncBB);

  B.SetInsertPoint(outerIncBB);
  {
    Value* rNext = B.CreateAdd(rPHI, ConstantInt::get(B.getInt32Ty(), 1));
    rPHI->addIncoming(rNext, outerIncBB);
    B.CreateBr(outerCheckBB);
  }

  B.SetInsertPoint(outerExitBB);
  attachNoUnrollMetadata(B, outerIncBB);
  // IR now has run-time loops for loading amplitudes,
  // for (r,c) partial sums, and unrolling has been disabled.
}

// Packed kind table: 1 byte per complex (low nibble=reKind, high nibble=imKind)
// Codes: 0:Zero, 1:+1, 2:-1, 3:ImmValue
static GlobalVariable*
createPackedKindTable(Module& M,
                      unsigned N,
                      const std::vector<IRMatDataCUDA>& matData,
                      StringRef name,
                      unsigned addrSpace) {
  auto& C = M.getContext();
  auto* arrTy = ArrayType::get(Type::getInt8Ty(C), N * N);

  auto code = [](ScalarKind sk) -> uint8_t {
    switch (sk) {
    case SK_Zero:
      return 0;
    case SK_One:
      return 1;
    case SK_MinusOne:
      return 2;
    default:
      return 3;
    }
  };

  std::vector<Constant*> init;
  init.reserve(N * N);
  for (unsigned i = 0; i < N * N; ++i) {
    uint8_t packed = (uint8_t(code(matData[i].imKind)) << 4) |
                     uint8_t(code(matData[i].reKind));
    init.push_back(ConstantInt::get(Type::getInt8Ty(C), packed));
  }

  auto* CA = ConstantArray::get(arrTy, init);
  auto* gv = new GlobalVariable(M,
                                arrTy,
                                /*isConst=*/true,
                                GlobalValue::PrivateLinkage,
                                CA,
                                name,
                                /*Before=*/nullptr,
                                GlobalValue::NotThreadLocal,
                                /*AS=*/addrSpace);
  gv->setAlignment(MaybeAlign(16));
  return gv;
}

// used to avoid recomputing the “pdep” style mapping in inner loops
static GlobalVariable* createDeltaLUT(Module& M,
                                      unsigned k,
                                      llvm::ArrayRef<int> qubits,
                                      StringRef name,
                                      unsigned addrSpace) {
  const unsigned N = 1u << k;
  auto& C = M.getContext();
  auto* arrTy = ArrayType::get(Type::getInt64Ty(C), N);

  std::vector<Constant*> init;
  init.reserve(N);
  for (unsigned c = 0; c < N; ++c) {
    uint64_t delta = 0;
    for (unsigned b = 0; b < k; ++b)
      if (c & (1u << b))
        delta |= (1ull << qubits[b]);
    init.push_back(ConstantInt::get(Type::getInt64Ty(C), delta));
  }

  auto* CA = ConstantArray::get(arrTy, init);
  auto* gv = new GlobalVariable(M,
                                arrTy,
                                /*isConst=*/true,
                                GlobalValue::PrivateLinkage,
                                CA,
                                name,
                                /*Before=*/nullptr,
                                GlobalValue::NotThreadLocal,
                                /*AS=*/addrSpace);
  gv->setAlignment(MaybeAlign(16));
  return gv;
}

GlobalVariable* createGlobalMatrixArray_SharedTiledImm(
    Module& M,
    Type* scalarTy,
    unsigned N,
    const std::vector<IRMatDataCUDA>& matData,
    const std::string& globalName,
    unsigned addrSpace) {
  // The array type: [2*N*N x scalarTy].
  unsigned totalElems = 2 * N * N;
  ArrayType* arrTy = ArrayType::get(scalarTy, totalElems);

  // -------------------------------- For Debug
  // ------------------------------------- #include <iostream> std::cerr
  // << "matData for " << globalName << ":\n";
  // for (unsigned i = 0; i < N*N; /* i++) */) {
  // const auto &md = matData[i];
  // ConstantFP *cRe = dyn_cast_or_null<ConstantFP>(md.reVal);
  // ConstantFP *cIm = dyn_cast_or_null<ConstantFP>(md.imVal);
  // double reVal = (md.reKind == SK_ImmValue && cRe) ?
  //     cRe->getValueAPF().convertToDouble() :
  //     (md.reKind == SK_One) ? 1.0 :
  //     (md.reKind == SK_MinusOne) ? -1.0 : 0.0;
  // double imVal = (md.imKind == SK_ImmValue && cIm) ?
  //     cIm->getValueAPF().convertToDouble() :
  //     (md.imKind == SK_One) ? 1.0 :
  //     (md.imKind == SK_MinusOne) ? -1.0 : 0.0;
  // std::cerr << "matData[" << i << "] = (" << reVal << ", " << imVal << ")\n";
  // break;
  // }
  // --------------------------------------------------------------------------------

  std::vector<Constant*> initVals;
  initVals.reserve(totalElems);

  // Fill them from matData. We assume matData.size() == (N*N).
  // For each index i => matData[i] => reVal, imVal
  // Place them in positions [2*i], [2*i + 1].
  for (unsigned i = 0; i < N * N; i++) {
    const auto& md = matData[i];

    // Attempt to cast reVal, imVal to ConstantFP if SK_ImmValue
    ConstantFP* cRe = dyn_cast_or_null<ConstantFP>(md.reVal);
    ConstantFP* cIm = dyn_cast_or_null<ConstantFP>(md.imVal);

    // Fallback
    if (md.reKind == SK_ImmValue && cRe) {
      initVals.push_back(cRe);
    } else if (md.reKind == SK_One) {
      initVals.push_back(ConstantFP::get(scalarTy, 1.0));
    } else if (md.reKind == SK_MinusOne) {
      initVals.push_back(ConstantFP::get(scalarTy, -1.0));
    } else if (md.reKind == SK_Zero) {
      initVals.push_back(ConstantFP::get(scalarTy, 0.0));
    } else {
      // e.g. SK_General => fallback to 0.0 or handle differently
      initVals.push_back(ConstantFP::get(scalarTy, 0.0));
    }

    if (md.imKind == SK_ImmValue && cIm) {
      initVals.push_back(cIm);
    } else if (md.imKind == SK_One) {
      initVals.push_back(ConstantFP::get(scalarTy, 1.0));
    } else if (md.imKind == SK_MinusOne) {
      initVals.push_back(ConstantFP::get(scalarTy, -1.0));
    } else if (md.imKind == SK_Zero) {
      initVals.push_back(ConstantFP::get(scalarTy, 0.0));
    } else {
      initVals.push_back(ConstantFP::get(scalarTy, 0.0));
    }
  }

  // auto* gVar =
  // new GlobalVariable(M,
  //                    arrTy,
  //                    /*isConstant=*/true,
  //                    GlobalValue::PrivateLinkage, // or InternalLinkage
  //                    arrInit,
  //                    globalName);
  // gVar->setAlignment(MaybeAlign(8));

  Constant* arrInit = ConstantArray::get(arrTy, initVals);
  auto* gVar =
      new GlobalVariable(M,
                         arrTy,
                         /*isConstant=*/true,
                         GlobalValue::PrivateLinkage,
                         arrInit,
                         globalName,
                         /*Before=*/nullptr,
                         GlobalValue::NotThreadLocal,
                         /*AddressSpace=*/addrSpace // <— use AS=4 or AS=1
      );
  gVar->setAlignment(MaybeAlign(16));
  return gVar;
}

Value* helper_getBlockIdx(IRBuilder<>& B) {
  return B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
}

/// @brief This mimics a runtime PDEP operation with compile-time known mask.
/// Given a counter t, find the offset index idx such that in round t
/// of the iteration, amplitude index starts at idx. The returned idx is in the
/// unit of [real, imag], that is, <2 x ScalarType>.
Value* buildOffset(IRBuilder<>& B,
                   Value* counterV,
                   const QuantumGate::TargetQubitsType& qubits) {
  /*
   Example: with target qubits 2, 4, 5
   counter: xxxhgfedcba
   pbex mask: 11111001011
   idxStart: hgfed00c0ba (in unit of <2 x scalarTy>)
   hgfed00c0ba = (xxxhgfedcba & 00000000011) << 0
               + (xxxhgfedcba & 00000000100) << 1
               + (xxxhgfedcba & 11111111000) << 3
   We build this segment by segment. For [2, 4, 5], there are 3 segments:
   [0, 2), [3, 4), [5, ),
   corresponding to masks 00000000011, 00000000100, 11111111000
  */
  assert(!qubits.empty());

  Value* offset = B.getInt64(0ULL);
  counterV = B.CreateZExt(counterV, B.getInt64Ty(), "i64.counter");

  Value* tmpCounterV;
  uint64_t mask = 0ULL;

  int k = qubits.size();
  int highestQ = qubits.back();
  int qIdx = 0;
  int counterQ = 0;

  for (int q = 0; q <= highestQ; q++) {
    if (q < qubits[qIdx]) {
      mask |= (1ULL << counterQ++);
      continue;
    }
    ++qIdx;
    if (mask == 0)
      continue;

    tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
    tmpCounterV = B.CreateShl(tmpCounterV, (qIdx - 1), "tmpCounter");
    offset = B.CreateAdd(offset, tmpCounterV, "tmpIdx");

    LLVM_DEBUG(std::cerr << " (globalThreadIdx & " << utils::fmt_0b(mask, 32)
                         << ") << " << (qIdx - 1) << "\n";);
    mask = 0ULL;
  }

  mask = ~((1ULL << (highestQ - k + 1)) - 1);
  LLVM_DEBUG(std::cerr << " (globalThreadIdx & " << utils::fmt_0b(mask, 32)
                       << ") << " << (k) << "\n";);

  tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
  tmpCounterV = B.CreateShl(tmpCounterV, k, "tmpCounter");
  offset = B.CreateAdd(offset, tmpCounterV, "offset");

  return offset;
}

static Value* warpBroadcastF32(IRBuilder<>& B, Value* valF32, Value* srcLane) {
  auto& C = B.getContext();
  Value* mask = ConstantInt::get(Type::getInt32Ty(C), -1);

  auto shfl = Intrinsic::getOrInsertDeclaration(
      B.GetInsertBlock()->getModule(), Intrinsic::nvvm_shfl_sync_idx_i32);

  Value* valI32 = B.CreateBitCast(valF32, Type::getInt32Ty(C));
  Value* resI32 = B.CreateCall(shfl, {mask, valI32, srcLane, B.getInt32(31)});
  return B.CreateBitCast(resI32, valF32->getType());
}

static void genMatrixVectorMultiply_ImmRowPerLane(
    IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const QuantumGate::TargetQubitsType& qubits,
    const std::vector<IRMatDataCUDA>& matData,
    Value* svPtrV, // base of full statevector
    Type* scalarTy) {
  using namespace llvm;

  // Fast-math
  FastMathFlags FMF;
  FMF.setNoNaNs();
  FMF.setNoInfs();
  FMF.setNoSignedZeros();
  FMF.setAllowContract(true);
  FMF.setAllowReassoc(true);
  B.setFastMathFlags(FMF);

  const unsigned k = qubits.size();
  const unsigned N = 1u << k;

  Function* func = B.GetInsertBlock()->getParent();
  Module& M = *func->getParent();
  LLVMContext& CTX = B.getContext();
  const size_t elem = (config.precision == Precision::F32) ? 4 : 8;
  const size_t bytesM = size_t(2) * N * N * elem;
  const unsigned AS_M = 1;
  GlobalVariable* gMatImm = createGlobalMatrixArray_SharedTiledImm(
      M,
      scalarTy,
      N,
      matData,
      (Twine("gMatImm_") + func->getName()).str(),
      AS_M);
  auto* arrTyM = llvm::cast<ArrayType>(gMatImm->getValueType());

  // Autotune warps: min(8, ceil(K/32)), at least 1
  const unsigned warps = std::min(8u, std::max(1u, (N + 31u) / 32u));
  func->addFnAttr("cast.kstyle", "imm-shared-warp");
  func->addFnAttr("cast.warps", std::to_string(warps));

  // Build the same "immediate + kind" tables as in the tiled path
  // const size_t elem    = (config.precision == Precision::F32) ? 4 : 8;
  // const size_t bytesM  = size_t(2) * N * N * elem;
  // const unsigned AS_M  = (bytesM <= 64*1024) ? 4 : 1;  // const or global
  // GlobalVariable *gMatImm =
  //     createGlobalMatrixArray_SharedTiledImm(M, scalarTy, N, matData,
  //                                            "gMatImm_RowLane", AS_M);
  // // Kind table: 1 byte / complex; const if fits in 64 KiB
  // const size_t bytesKind = size_t(N) * N; // 1 byte per complex
  // const unsigned AS_KIND = (bytesKind <= 64 * 1024) ? 4 : 1;
  // auto *gMatKindPacked = createPackedKindTable(
  //     M, N, matData, "gMatKind_RowLane", AS_KIND);
  // auto *kindArrTy = cast<ArrayType>(gMatKindPacked->getValueType());

  // // Δ‑LUT for generic addressing (const if fits in 64 KiB)
  // GlobalVariable *gDelta = nullptr;
  // ArrayType *deltaArrTy = nullptr;
  // if (!config.assumeContiguousTargets) {
  //     const size_t deltaBytes = size_t(N) * sizeof(uint64_t);
  //     const unsigned AS_DELTA = (deltaBytes <= 64 * 1024) ? 4 : 1;
  //     gDelta = createDeltaLUT(M, k, qubits, "gDelta_RowLane", AS_DELTA);
  //     deltaArrTy = cast<ArrayType>(gDelta->getValueType());
  // }

  // auto mapKind = [](ScalarKind sk)->unsigned {
  //     switch (sk) { case SK_Zero: return 0; case SK_One: return 1;
  //                   case SK_MinusOne:return 2; default: return 3; }
  // };
  // ArrayType *kindArrTy = ArrayType::get(B.getInt8Ty(), 2 * N * N);
  // std::vector<Constant*> kinds; kinds.reserve(2*N*N);
  // for (unsigned i=0;i<N*N;++i) {
  //     kinds.push_back(ConstantInt::get(B.getInt8Ty(),
  //     mapKind(matData[i].reKind)));
  //     kinds.push_back(ConstantInt::get(B.getInt8Ty(),
  //     mapKind(matData[i].imKind)));
  // }
  // auto *gMatKind = new GlobalVariable(
  //     M, kindArrTy, true, GlobalValue::PrivateLinkage,
  //     ConstantArray::get(kindArrTy, kinds), "gMatKind_RowLane",
  //     nullptr, GlobalValue::NotThreadLocal, (bytesM<=64*1024?4:1));
  // gMatKind->setAlignment(MaybeAlign(16));

  const size_t bytesKind = size_t(N) * N;
  const unsigned AS_KIND = (bytesKind <= 64 * 1024) ? 4 : 1;
  auto* gMatKindPacked = createPackedKindTable(
      M, N, matData, (Twine("gMatKind_") + func->getName()).str(), AS_KIND);
  auto* kindArrTy = llvm::cast<ArrayType>(gMatKindPacked->getValueType());

  // Δ‑LUT for generic addressing (only if targets are not contiguous)
  GlobalVariable* gDelta = nullptr;
  ArrayType* deltaArrTy = nullptr;
  if (!config.assumeContiguousTargets) {
    const size_t deltaBytes = size_t(N) * sizeof(uint64_t);
    const unsigned AS_DELTA = (deltaBytes <= 64 * 1024) ? 4 : 1;
    gDelta = createDeltaLUT(
        M, k, qubits, (Twine("gDelta_") + func->getName()).str(), AS_DELTA);
    deltaArrTy = llvm::cast<ArrayType>(gDelta->getValueType());
  }

  // auto *arrTyM = llvm::cast<ArrayType>(gMatImm->getValueType());

  // Thread indices and warp bookkeeping
  Value* tid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  Value* ntid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ntid_x, {});
  Value* lane32 = B.CreateAnd(tid32, B.getInt32(31), "lane");
  Value* warpInBlock = B.CreateAShr(tid32, B.getInt32(5), "warp.in.block");
  Value* warpsPerBlck = B.CreateAShr(ntid32, B.getInt32(5), "warps.per.block");

  // Starting row and stride across rows handled by this lane
  Value* r0 =
      B.CreateAdd(B.CreateZExt(lane32, B.getInt64Ty()),
                  B.CreateShl(B.CreateZExt(warpInBlock, B.getInt64Ty()), 5),
                  "r0");
  Value* rStep =
      B.CreateShl(B.CreateZExt(warpsPerBlck, B.getInt64Ty()), 5, "r.step");

  // Persistent-grid combos loop:
  // We keep gridDim.x a multiple of tilesPerGate on the host.
  const unsigned TILE = std::min(256u, N);
  const unsigned tilesPerGate = (N + TILE - 1) / TILE;

  // Read ctaid.x and gridDim.x
  Value* bid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
  Value* gridX = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_nctaid_x, {});

  // comboSlot = ctaid.x / tilesPerGate
  // comboStride = gridDim.x / tilesPerGate (integral; launcher guarantees)
  Value* comboSlot =
      B.CreateUDiv(bid32, B.getInt32(tilesPerGate), "combo.slot");
  Value* comboStride =
      B.CreateUDiv(gridX, B.getInt32(tilesPerGate), "combo.stride");

  // Fetch p.combos (uint32 arg named exactly "p.combos")
  Argument* combosV = nullptr;
  for (auto& A : func->args()) {
    if (A.getName() == "p.combos") {
      combosV = &A;
      break;
    }
  }
  assert(combosV && "Missing kernel argument 'p.combos' (uint32)");

  BasicBlock* cmbChk = BasicBlock::Create(CTX, "cmb.chk", func);
  BasicBlock* cmbBody = BasicBlock::Create(CTX, "cmb.body", func);
  BasicBlock* cmbInc = BasicBlock::Create(CTX, "cmb.inc", func);
  BasicBlock* cmbDone = BasicBlock::Create(CTX, "cmb.done", func);

  BasicBlock* cmbPre = B.GetInsertBlock();
  B.CreateBr(cmbChk);
  B.SetInsertPoint(cmbChk);
  PHINode* comboPhi = B.CreatePHI(B.getInt32Ty(), 2, "combo");
  comboPhi->addIncoming(comboSlot, cmbPre); // correct predecessor
  B.CreateCondBr(B.CreateICmpULT(comboPhi, combosV), cmbBody, cmbDone);

  // cmb.body
  B.SetInsertPoint(cmbBody);

  // svBaseCombo: per-combo base pointer into full SV
  Value* svBaseCombo = nullptr;
  if (config.assumeContiguousTargets) {
    Value* c64 = B.CreateZExt(comboPhi, B.getInt64Ty());
    Value* base2 = B.CreateShl(c64, k + 1); // <<(k+1) for (re,im)
    svBaseCombo = B.CreateGEP(scalarTy, svPtrV, base2, "sv.base.combo");
  } else {
    Value* c64 = B.CreateZExt(comboPhi, B.getInt64Ty());
    Value* off = buildOffset(B, c64, qubits);
    Value* base2 = B.CreateShl(off, 1);
    svBaseCombo = B.CreateGEP(scalarTy, svPtrV, base2, "sv.base.combo");
  }

  // Outer loop over rows for this combo: for (r=r0; r<N; r+=rStep)
  BasicBlock* rChk = BasicBlock::Create(CTX, "r.chk", func);
  BasicBlock* rBody = BasicBlock::Create(CTX, "r.body", func);
  BasicBlock* rInc = BasicBlock::Create(CTX, "r.inc", func);
  BasicBlock* rExit = BasicBlock::Create(CTX, "r.exit", func);

  B.CreateBr(rChk);
  B.SetInsertPoint(rChk);
  PHINode* rPHI = B.CreatePHI(B.getInt64Ty(), 2, "r");
  rPHI->addIncoming(r0, cmbBody);
  B.CreateCondBr(B.CreateICmpULT(rPHI, B.getInt64(N)), rBody, rExit);

  // Body: compute one output row 'rPHI' in this lane
  B.SetInsertPoint(rBody);

  // Inner loop over columns with loop-carried accumulators
  BasicBlock* cChk = BasicBlock::Create(CTX, "c.chk", func);
  BasicBlock* cBody = BasicBlock::Create(CTX, "c.body", func);
  BasicBlock* cInc = BasicBlock::Create(CTX, "c.inc", func);
  BasicBlock* cEnd = BasicBlock::Create(CTX, "c.end", func);

  B.CreateBr(cChk);
  B.SetInsertPoint(cChk);

  PHINode* cPHI = B.CreatePHI(B.getInt64Ty(), 2, "c");
  PHINode* accRe0 = B.CreatePHI(scalarTy, 2, "accRe0"); // Σ matRe * xRe
  PHINode* accRe1 = B.CreatePHI(scalarTy, 2, "accRe1"); // Σ matIm * xIm
  PHINode* accIm =
      B.CreatePHI(scalarTy, 2, "accIm"); // Σ (matRe*xIm + matIm*xRe)

  cPHI->addIncoming(B.getInt64(0), rBody);
  accRe0->addIncoming(ConstantFP::get(scalarTy, 0.0), rBody);
  accRe1->addIncoming(ConstantFP::get(scalarTy, 0.0), rBody);
  accIm->addIncoming(ConstantFP::get(scalarTy, 0.0), rBody);

  B.CreateCondBr(B.CreateICmpULT(cPHI, B.getInt64(N)), cBody, cEnd);

  // c.body
  B.SetInsertPoint(cBody);
  {
    // idx2 = 2*(r*N + c)
    Value* lin = B.CreateAdd(B.CreateMul(rPHI, B.getInt64(N)), cPHI, "lin");
    Value* idx2 = B.CreateMul(lin, B.getInt64(2), "idx2");

    // Load kinds
    auto toI32 = [&](Value* v) {
      return B.CreateIntCast(v, B.getInt32Ty(), false);
    };
    Value* lin32 = toI32(lin);
    Value* kByte = B.CreateLoad(
        B.getInt8Ty(),
        B.CreateGEP(kindArrTy, gMatKindPacked, {B.getInt32(0), lin32}));
    Value* kByte32 = B.CreateZExt(kByte, B.getInt32Ty());
    Value* kRe = B.CreateAnd(kByte32, B.getInt32(0xF));
    Value* kIm =
        B.CreateAnd(B.CreateLShr(kByte32, B.getInt32(4)), B.getInt32(0xF));

    // If either part is "imm", load the pair from gMatImm
    Value* mPairLd = nullptr;
    Value* mPairUndef = nullptr;
    BasicBlock* ldYes = BasicBlock::Create(CTX, "m.ld.y", func);
    BasicBlock* ldNo = BasicBlock::Create(CTX, "m.ld.n", func);
    BasicBlock* ldEnd = BasicBlock::Create(CTX, "m.ld.end", func);
    Value* needLoad = B.CreateOr(B.CreateICmpEQ(kRe, B.getInt32(3)),
                                 B.CreateICmpEQ(kIm, B.getInt32(3)));
    B.CreateCondBr(needLoad, ldYes, ldNo);

    B.SetInsertPoint(ldYes);
    Value* mPairPtr = bitCastPtrToVec2(
        B,
        B.CreateGEP(arrTyM, gMatImm, {B.getInt32(0), toI32(idx2)}),
        scalarTy);
    mPairLd = B.CreateLoad(getVec2Ty(scalarTy), mPairPtr, "m.pair");
    B.CreateBr(ldEnd);

    B.SetInsertPoint(ldNo);
    mPairUndef = UndefValue::get(getVec2Ty(scalarTy));
    B.CreateBr(ldEnd);

    B.SetInsertPoint(ldEnd);
    PHINode* mPair = B.CreatePHI(getVec2Ty(scalarTy), 2, "m.pair.phi");
    mPair->addIncoming(mPairLd, ldYes);
    mPair->addIncoming(mPairUndef, ldNo);

    // materialize (mRe, mIm) from kind
    Constant* c0 = ConstantFP::get(scalarTy, 0.0);
    Constant* c1 = ConstantFP::get(scalarTy, 1.0);
    Constant* cN = ConstantFP::get(scalarTy, -1.0);
    auto selK = [&](Value* k, Value* loaded) {
      Value* isZ = B.CreateICmpEQ(k, B.getInt32(0));
      Value* isP1 = B.CreateICmpEQ(k, B.getInt32(1));
      Value* isM1 = B.CreateICmpEQ(k, B.getInt32(2));
      Value* v = B.CreateSelect(isP1, c1, loaded);
      v = B.CreateSelect(isM1, cN, v);
      v = B.CreateSelect(isZ, c0, v);
      return v;
    };
    Value* loadedRe = B.CreateExtractElement(mPair, B.getInt32(0));
    Value* loadedIm = B.CreateExtractElement(mPair, B.getInt32(1));
    Value* mRe = selK(kRe, loadedRe);
    Value* mIm = selK(kIm, loadedIm);

    // Map logical column -> physical delta in SV (generic or contiguous)
    // Value *delta = B.getInt64(0);
    // if (config.assumeContiguousTargets) {
    //     delta = cPHI;
    // } else {
    //     for (unsigned b = 0; b < k; ++b) {
    //         Value *mask = B.getInt64(1ULL << b);
    //         Value *bit  = B.CreateAnd(cPHI, mask);
    //         Value *cond = B.CreateICmpNE(bit, B.getInt64(0));
    //         Value *shift= B.getInt64(1ULL << qubits[b]);
    //         delta = B.CreateSelect(cond, B.CreateOr(delta, shift), delta);
    //     }
    // }

    Value* delta = nullptr;
    if (config.assumeContiguousTargets) {
      delta = cPHI; // logical==physical
    } else {
      Value* c32 = B.CreateIntCast(cPHI, B.getInt32Ty(), false);
      Value* dPtr = B.CreateGEP(deltaArrTy, gDelta, {B.getInt32(0), c32});
      delta = B.CreateLoad(B.getInt64Ty(), dPtr, "delta.c");
    }
    // Load x[c] from svBaseCombo (vectorized)
    // Value *xPairPtr = bitCastPtrToVec2(
    //     B, B.CreateGEP(scalarTy, svBaseCombo, B.CreateShl(delta, 1)),
    //     scalarTy);
    // Value *xPair = B.CreateLoad(getVec2Ty(scalarTy), xPairPtr, "x.pair");
    // Value *xRe = B.CreateExtractElement(xPair, B.getInt32(0));
    // Value *xIm = B.CreateExtractElement(xPair, B.getInt32(1));

    Value* xRe = UndefValue::get(scalarTy);
    Value* xIm = UndefValue::get(scalarTy);

    if (config.precision == Precision::F32) {
      // Only lane 0 loads; everyone else gets it via warp broadcast.
      Value* isLane0 = B.CreateICmpEQ(lane32, B.getInt32(0));
      BasicBlock* ldYes = BasicBlock::Create(CTX, "x.ld.y", func);
      BasicBlock* ldNo = BasicBlock::Create(CTX, "x.ld.n", func);
      BasicBlock* ldEnd = BasicBlock::Create(CTX, "x.ld.end", func);
      B.CreateCondBr(isLane0, ldYes, ldNo);

      B.SetInsertPoint(ldYes);
      Value* xPairPtr = bitCastPtrToVec2(
          B,
          B.CreateGEP(scalarTy, svBaseCombo, B.CreateShl(delta, 1)),
          scalarTy);
      Value* xPairLd =
          B.CreateLoad(getVec2Ty(scalarTy), xPairPtr, "x.pair.lane0");
      Value* xRe0 = B.CreateExtractElement(xPairLd, B.getInt32(0));
      Value* xIm0 = B.CreateExtractElement(xPairLd, B.getInt32(1));
      B.CreateBr(ldEnd);

      B.SetInsertPoint(ldNo);
      Value* xReU = UndefValue::get(scalarTy);
      Value* xImU = UndefValue::get(scalarTy);
      B.CreateBr(ldEnd);

      B.SetInsertPoint(ldEnd);
      PHINode* xRePhi = B.CreatePHI(scalarTy, 2);
      PHINode* xImPhi = B.CreatePHI(scalarTy, 2);
      xRePhi->addIncoming(xRe0, ldYes);
      xRePhi->addIncoming(xReU, ldNo);
      xImPhi->addIncoming(xIm0, ldYes);
      xImPhi->addIncoming(xImU, ldNo);

      Value* srcLane = B.getInt32(0);
      xRe = warpBroadcastF32(B, xRePhi, srcLane);
      xIm = warpBroadcastF32(B, xImPhi, srcLane);
    } else {
      // F64 path: leave as-is (or split to two I32 shfls if desired later)
      Value* xPairPtr = bitCastPtrToVec2(
          B,
          B.CreateGEP(scalarTy, svBaseCombo, B.CreateShl(delta, 1)),
          scalarTy);
      Value* xPair = B.CreateLoad(getVec2Ty(scalarTy), xPairPtr, "x.pair");
      xRe = B.CreateExtractElement(xPair, B.getInt32(0));
      xIm = B.CreateExtractElement(xPair, B.getInt32(1));
    }

    // Complex MAC → next accumulators
    Value* accRe0n = B.CreateFAdd(accRe0, B.CreateFMul(mRe, xRe));
    Value* accRe1n = B.CreateFAdd(accRe1, B.CreateFMul(mIm, xIm));
    Value* accImn = B.CreateFAdd(
        accIm, B.CreateFAdd(B.CreateFMul(mRe, xIm), B.CreateFMul(mIm, xRe)));

    B.CreateBr(cInc);

    // c.inc
    B.SetInsertPoint(cInc);
    Value* cNext = B.CreateAdd(cPHI, B.getInt64(1));
    cPHI->addIncoming(cNext, cInc);
    accRe0->addIncoming(accRe0n, cInc);
    accRe1->addIncoming(accRe1n, cInc);
    accIm->addIncoming(accImn, cInc);
    B.CreateBr(cChk);
  }

  // c.end
  B.SetInsertPoint(cEnd);
  Value* newRe = B.CreateFSub(accRe0, accRe1);
  Value* newIm = accIm;

  // Store y[r] into svBaseCombo
  Value* rDelta = nullptr;
  if (config.assumeContiguousTargets) {
    rDelta = rPHI;
  } else {
    Value* r32 = B.CreateIntCast(rPHI, B.getInt32Ty(), false);
    Value* dPtr = B.CreateGEP(deltaArrTy, gDelta, {B.getInt32(0), r32});
    rDelta = B.CreateLoad(B.getInt64Ty(), dPtr, "delta.r");
  }
  Value* dstPair = UndefValue::get(getVec2Ty(scalarTy));
  dstPair = B.CreateInsertElement(dstPair, newRe, B.getInt32(0));
  dstPair = B.CreateInsertElement(dstPair, newIm, B.getInt32(1));
  Value* dstPtr = bitCastPtrToVec2(
      B, B.CreateGEP(scalarTy, svBaseCombo, B.CreateShl(rDelta, 1)), scalarTy);
  B.CreateStore(dstPair, dstPtr);

  // Advance r
  B.CreateBr(rInc);
  B.SetInsertPoint(rInc);
  Value* rNext = B.CreateAdd(rPHI, rStep);
  rPHI->addIncoming(rNext, rInc);
  B.CreateBr(rChk);

  // end rows for this combo
  B.SetInsertPoint(rExit);
  B.CreateBr(cmbInc);

  // next combo
  B.SetInsertPoint(cmbInc);
  Value* comboNext = B.CreateAdd(comboPhi, comboStride);
  comboPhi->addIncoming(comboNext, cmbInc);
  B.CreateBr(cmbChk);

  // done
  B.SetInsertPoint(cmbDone);
}

void genMatrixVectorMultiply_SharedTiled(
    IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const ComplexSquareMatrix& /*matrix*/,
    const QuantumGate::TargetQubitsType& qubits,
    const std::vector<IRMatDataCUDA>& matData,
    Value* svPtrV, // **base of full statevector** (no combo skew)
    Type* scalarTy) {
  using namespace llvm;

  // Fast math flags
  FastMathFlags FMF;
  FMF.setNoNaNs();
  FMF.setNoInfs();
  FMF.setNoSignedZeros();
  FMF.setAllowContract(true);
  FMF.setAllowReassoc(true);
  B.setFastMathFlags(FMF);

#ifndef NDEBUG
  // The immediate+shared path expects no SK_Runtime entries
  for (const auto& md : matData) {
    assert(
        md.reKind != SK_Runtime && md.imKind != SK_Runtime &&
        "imm-shared path cannot handle SK_Runtime; use pointer path instead");
  }
#endif

  // Parameters
  const unsigned k = qubits.size();
  const unsigned N = 1u << k;
  const unsigned TILE =
      std::min((config.precision == Precision::F64) ? 128u : 256u, N);

  Function* func = B.GetInsertBlock()->getParent();
  if (!func->hasFnAttribute("cast.tile"))
    func->addFnAttr("cast.tile", std::to_string(TILE));
  Module& M = *func->getParent();
  LLVMContext& CTX = B.getContext();

  // Fetch p.combos (added to the function prototype and named "p.combos")
  Argument* combosV = nullptr;
  for (auto& A : func->args()) {
    if (A.getName() == "p.combos") {
      combosV = &A;
      break;
    }
  }
  assert(combosV && "Missing kernel argument 'p.combos' (uint32)");

  // Matrix immediates + "kind" table
  const size_t elem = (config.precision == Precision::F32) ? 4 : 8;
  const size_t bytesM = size_t(2) * N * N * elem;   // complex (re,im)
  const size_t CONST0 = 64 * 1024;                  // 64 KiB
  const unsigned AS_M = (bytesM <= CONST0) ? 4 : 1; // const (4) or global (1)

  GlobalVariable* gMatImm = createGlobalMatrixArray_SharedTiledImm(
      M, scalarTy, N, matData, "gMatImmSharedTiled", AS_M);
  auto* arrTyM = llvm::cast<ArrayType>(gMatImm->getValueType());
  Value* gMatBase = B.CreateInBoundsGEP(
      arrTyM, gMatImm, {B.getInt32(0), B.getInt64(0)}, "gMat.base");

  // Kind table encodes re/im kind in 2 i8 values per complex
  // auto mapKind = [](ScalarKind sk) -> unsigned {
  //     switch (sk) {
  //         case SK_Zero: return 0;
  //         case SK_One: return 1;
  //         case SK_MinusOne: return 2;
  //         case SK_ImmValue: return 3;
  //         default: return 3;
  //     }
  // };
  // ArrayType *kindArrTy = ArrayType::get(B.getInt8Ty(), 2 * N * N);
  // std::vector<Constant *> kindInit; kindInit.reserve(2 * N * N);
  // for (unsigned i = 0; i < N * N; ++i) {
  //     kindInit.push_back(ConstantInt::get(B.getInt8Ty(),
  //     mapKind(matData[i].reKind)));
  //     kindInit.push_back(ConstantInt::get(B.getInt8Ty(),
  //     mapKind(matData[i].imKind)));
  // }
  // const size_t bytesKind = size_t(2) * N * N;
  // const unsigned AS_KIND = (bytesKind <= CONST0) ? 4 : 1;

  // auto *kindCA = ConstantArray::get(kindArrTy, kindInit);
  // auto *gMatKind = new GlobalVariable(
  //     M, kindArrTy, /*isConst=*/true, GlobalValue::PrivateLinkage, kindCA,
  //     Twine("gMatKind_") + func->getName(), /*Before=*/nullptr,
  //     GlobalValue::NotThreadLocal, AS_KIND);
  // gMatKind->setAlignment(MaybeAlign(16));

  const size_t bytesKind = size_t(N) * N; // 1 byte per complex
  const unsigned AS_KIND = (bytesKind <= CONST0) ? 4 : 1;
  auto* gMatKindPacked = createPackedKindTable(
      M, N, matData, (Twine("gMatKind_") + func->getName()).str(), AS_KIND);
  auto* kindArrTy = llvm::cast<ArrayType>(gMatKindPacked->getValueType());

  // Δ‑LUT for generic addressing (only if targets are not contiguous)
  GlobalVariable* gDelta = nullptr;
  ArrayType* deltaArrTy = nullptr;
  if (!config.assumeContiguousTargets) {
    const size_t deltaBytes = size_t(N) * sizeof(uint64_t);
    const unsigned AS_DELTA = (deltaBytes <= CONST0) ? 4 : 1;
    gDelta = createDeltaLUT(
        M, k, qubits, (Twine("gDelta_") + func->getName()).str(), AS_DELTA);
    deltaArrTy = llvm::cast<ArrayType>(gDelta->getValueType());
  }

  // Shared tile for statevector
  ArrayType* smVecTy = ArrayType::get(scalarTy, 2 * TILE);
  auto* smVecGV = new GlobalVariable(M,
                                     smVecTy,
                                     /*constant=*/false,
                                     GlobalValue::PrivateLinkage,
                                     UndefValue::get(smVecTy),
                                     (Twine("tileX_") + func->getName()).str(),
                                     nullptr,
                                     GlobalValue::NotThreadLocal,
                                     /*AS=*/3);
  Value* smVecBase =
      B.CreateGEP(smVecTy, smVecGV, {B.getInt32(0), B.getInt32(0)}, "smX.base");

  // Thread & grid indices
  Value* tid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  Value* bid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
  Value* gridX = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_nctaid_x, {}); // gridDim.x

  // Only the first TILE threads per CTA act within a tile
  Value* tid32L = (TILE == 1)
                      ? tid32
                      : B.CreateURem(tid32, B.getInt32(TILE), "tid.in.tile");
  Value* tid64L = B.CreateZExt(tid32L, B.getInt64Ty(), "tid64L");
  Value* tidInTile = (TILE == 1)
                         ? ConstantInt::getTrue(CTX)
                         : B.CreateICmpULT(tid32, B.getInt32(TILE), "tid<tile");

  // Row‑tile decomposition for this CTA
  const unsigned tilesPerGate = (N + TILE - 1) / TILE;
  Value* tilesPerGateV = B.getInt32(tilesPerGate);
  Value* rowTileIdx32 = B.CreateURem(bid32, tilesPerGateV, "rowTileIdx");
  Value* rowBase64 = B.CreateMul(
      B.CreateZExt(rowTileIdx32, B.getInt64Ty()), B.getInt64(TILE), "row.base");
  Value* row64 = B.CreateAdd(rowBase64, tid64L, "row");

  // Guard row in-range
  BasicBlock* rowOK = BasicBlock::Create(CTX, "row.ok", func);
  BasicBlock* rowEnd = BasicBlock::Create(CTX, "row.end", func);
  B.CreateCondBr(B.CreateICmpULT(row64, B.getInt64(N)), rowOK, rowEnd);
  B.SetInsertPoint(rowOK);

  // Persistent-grid combos loop:
  // comboSlot = ctaid.x / tilesPerGate
  // comboStride = gridDim.x / tilesPerGate (assume divisible; launcher
  // enforces)
  Value* comboSlot = B.CreateUDiv(bid32, tilesPerGateV, "combo.slot");
  Value* comboStride = B.CreateUDiv(gridX, tilesPerGateV, "combo.stride");

  BasicBlock* cmbChk = BasicBlock::Create(CTX, "cmb.chk", func);
  BasicBlock* cmbBody = BasicBlock::Create(CTX, "cmb.body", func);
  BasicBlock* cmbInc = BasicBlock::Create(CTX, "cmb.inc", func);
  BasicBlock* cmbDone = BasicBlock::Create(CTX, "cmb.done", func);

  B.CreateBr(cmbChk);
  B.SetInsertPoint(cmbChk);
  PHINode* comboPhi = B.CreatePHI(B.getInt32Ty(), 2, "combo");
  comboPhi->addIncoming(comboSlot, rowOK); // first iteration enters from rowOK
  B.CreateCondBr(B.CreateICmpULT(comboPhi, combosV), cmbBody, cmbDone);

  // cmb.body
  B.SetInsertPoint(cmbBody);

  // Compute per‑combo base SV pointer (svBaseCombo)
  Value* svBaseCombo = nullptr;
  {
    if (config.assumeContiguousTargets) {
      // base2 = (combo << (k+1)) (x2 for (re,im))
      Value* c64 = B.CreateZExt(comboPhi, B.getInt64Ty());
      Value* base2 = B.CreateShl(c64, k + 1);
      svBaseCombo = B.CreateGEP(scalarTy, svPtrV, base2, "sv.base.combo");
    } else {
      // off = buildOffset(combo, qubits) ; base2 = off << 1
      Value* c64 = B.CreateZExt(comboPhi, B.getInt64Ty());
      Value* off = buildOffset(B, c64, qubits);
      Value* base2 = B.CreateShl(off, 1);
      svBaseCombo = B.CreateGEP(scalarTy, svPtrV, base2, "sv.base.combo");
    }
  }

  // Tile loop scaffolding (with loop-carried accumulators)
  BasicBlock* tileChk = BasicBlock::Create(CTX, "tile.chk", func);
  BasicBlock* tileBody = BasicBlock::Create(CTX, "tile.body", func);
  BasicBlock* tileInc = BasicBlock::Create(CTX, "tile.inc", func);
  BasicBlock* tileDone = BasicBlock::Create(CTX, "tile.done", func);

  B.CreateBr(tileChk);
  B.SetInsertPoint(tileChk);

  // Loop-carried accumulators start at 0.0 for each combo iteration
  PHINode* accReLC = B.CreatePHI(scalarTy, /*numPreds=*/2, "accRe.lc");
  PHINode* accImLC = B.CreatePHI(scalarTy, /*numPreds=*/2, "accIm.lc");
  accReLC->addIncoming(ConstantFP::get(scalarTy, 0.0), cmbBody);
  accImLC->addIncoming(ConstantFP::get(scalarTy, 0.0), cmbBody);

  PHINode* col0Phi = B.CreatePHI(B.getInt64Ty(), 2, "col0");
  col0Phi->addIncoming(B.getInt64(0), cmbBody);

  B.CreateCondBr(B.CreateICmpULT(col0Phi, B.getInt64(N)), tileBody, tileDone);

  // tile.body
  B.SetInsertPoint(tileBody);

  // This tile's running sums start from loop-carried values
  Value* accReTile = accReLC;
  Value* accImTile = accImLC;

  // (1) Load |x⟩ tile into shared (only tid < TILE)
  {
    BasicBlock* ldChk = BasicBlock::Create(CTX, "ld.chk", func);
    BasicBlock* ldYes = BasicBlock::Create(CTX, "ld.y", func);
    BasicBlock* ldZero = BasicBlock::Create(CTX, "ld.zero", func);
    BasicBlock* ldSkip = BasicBlock::Create(CTX, "ld.skip", func);
    BasicBlock* ldEnd = BasicBlock::Create(CTX, "ld.end", func);

    B.CreateCondBr(tidInTile, ldChk, ldSkip);

    // in-range check for column we own in this tile
    B.SetInsertPoint(ldChk);
    Value* colIdx = B.CreateAdd(col0Phi, tid64L, "colIdx");
    Value* inRange = B.CreateICmpULT(colIdx, B.getInt64(N), "col.in.range");
    B.CreateCondBr(inRange, ldYes, ldZero);

    // load from statevector into shared
    B.SetInsertPoint(ldYes);
    {
      // Value *off2;
      // if (config.assumeContiguousTargets) {
      //     off2 = B.CreateShl(colIdx, 1); // ×2 for (re,im)
      // } else {
      //     Value *delta = B.getInt64(0);
      //     for (unsigned b = 0; b < k; ++b) {
      //         Value *mask = B.getInt64(1ULL << b);
      //         Value *bit  = B.CreateAnd(colIdx, mask);
      //         Value *cond = B.CreateICmpNE(bit, B.getInt64(0));
      //         Value *shift= B.getInt64(1ULL << qubits[b]);
      //         delta = B.CreateSelect(cond, B.CreateOr(delta, shift), delta);
      //     }
      //     off2 = B.CreateShl(delta, 1);
      // }
      Value* off2 = nullptr;
      if (config.assumeContiguousTargets) {
        off2 = B.CreateShl(colIdx, 1); // ×2 for (re,im)
      } else {
        Value* c32 = B.CreateIntCast(colIdx, B.getInt32Ty(), false);
        Value* dPtr = B.CreateGEP(deltaArrTy, gDelta, {B.getInt32(0), c32});
        Value* delta = B.CreateLoad(B.getInt64Ty(), dPtr);
        off2 = B.CreateShl(delta, 1);
      }

      Value* srcPairPtr = bitCastPtrToVec2(
          B, B.CreateGEP(scalarTy, svBaseCombo, off2), scalarTy);
      Value* xPairLd = B.CreateLoad(getVec2Ty(scalarTy), srcPairPtr, "x.pair");
      Value* dstPairPtr = bitCastPtrToVec2(
          B,
          B.CreateGEP(scalarTy, smVecBase, B.CreateMul(tid64L, B.getInt64(2))),
          scalarTy);
      B.CreateStore(xPairLd, dstPairPtr);
    }
    B.CreateBr(ldEnd);

    // out-of-range → write zeros
    B.SetInsertPoint(ldZero);
    {
      Constant* z = ConstantFP::get(scalarTy, 0.0);
      Value* idxS = B.CreateMul(tid64L, B.getInt64(2));
      B.CreateStore(z, B.CreateGEP(scalarTy, smVecBase, idxS));
      B.CreateStore(
          z,
          B.CreateGEP(scalarTy, smVecBase, B.CreateAdd(idxS, B.getInt64(1))));
    }
    B.CreateBr(ldEnd);

    // skip entirely if tid ≥ TILE
    B.SetInsertPoint(ldSkip);
    B.CreateBr(ldEnd);

    B.SetInsertPoint(ldEnd);
  }

  // __syncthreads()
  B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_barrier0));

  // (2) Dot product over this tile (skip zeros; ±1 materialized without loads)
  for (unsigned t = 0; t < TILE; ++t) {
    Value* col = B.CreateAdd(col0Phi, B.getInt64(t), "col");
    Value* inR = B.CreateICmpULT(col, B.getInt64(N), "in.rng");
    Value* actv = B.CreateAnd(inR, tidInTile, "active");

    BasicBlock* yesBB = BasicBlock::Create(CTX, "dot.y", func);
    BasicBlock* noBB = BasicBlock::Create(CTX, "dot.n", func);
    BasicBlock* endBB = BasicBlock::Create(CTX, "dot.end", func);
    BasicBlock* contBB = BasicBlock::Create(CTX, "dot.cont", func);

    PHINode* accReAfter = nullptr;
    PHINode* accImAfter = nullptr;

    B.CreateCondBr(actv, yesBB, noBB);

    // active lane path
    B.SetInsertPoint(yesBB);
    {
      // idx2 = 2 * (row*N + col)
      Value* lin = B.CreateAdd(B.CreateMul(row64, B.getInt64(N)), col, "lin");
      Value* idx2 = B.CreateMul(lin, B.getInt64(2), "idx2");

      // load kinds (i8) for this element
      // auto *i32Ty = B.getInt32Ty();
      // auto toI32 = [&](Value *v64) { return B.CreateIntCast(v64, i32Ty,
      // false); }; Value *kRePtr = B.CreateGEP(kindArrTy, gMatKind,
      // {B.getInt32(0), toI32(idx2)}); Value *kImPtr = B.CreateGEP(kindArrTy,
      // gMatKind,
      //                             {B.getInt32(0), B.CreateAdd(toI32(idx2),
      //                             B.getInt32(1))});
      // Value *kRe = B.CreateZExt(B.CreateLoad(B.getInt8Ty(), kRePtr), i32Ty);
      // Value *kIm = B.CreateZExt(B.CreateLoad(B.getInt8Ty(), kImPtr), i32Ty);

      auto* i32Ty = B.getInt32Ty();
      auto toI32 = [&](Value* v64) {
        return B.CreateIntCast(v64, i32Ty, false);
      };

      Value* lin32 = toI32(lin);
      Value* kByte = B.CreateLoad(
          B.getInt8Ty(),
          B.CreateGEP(kindArrTy, gMatKindPacked, {B.getInt32(0), lin32}));
      Value* kByte32 = B.CreateZExt(kByte, i32Ty);
      Value* kRe = B.CreateAnd(kByte32, B.getInt32(0xF));
      Value* kIm =
          B.CreateAnd(B.CreateLShr(kByte32, B.getInt32(4)), B.getInt32(0xF));

      // if both zero → skip computation
      Value* bothZero = B.CreateAnd(B.CreateICmpEQ(kRe, B.getInt32(0)),
                                    B.CreateICmpEQ(kIm, B.getInt32(0)));
      BasicBlock* skipBB = BasicBlock::Create(CTX, "dot.skip", func);
      BasicBlock* nzBB = BasicBlock::Create(CTX, "dot.nz", func);
      B.CreateCondBr(bothZero, skipBB, nzBB);

      // Non-zero path: maybe load imm from gMatImm; materialize re/im via kind
      B.SetInsertPoint(nzBB);
      Value* needLoad = B.CreateOr(B.CreateICmpEQ(kRe, B.getInt32(3)),
                                   B.CreateICmpEQ(kIm, B.getInt32(3)));
      Value* mPairLd = nullptr;
      Value* mPairUndef = nullptr;
      BasicBlock* ldYes = BasicBlock::Create(CTX, "m.ld.y", func);
      BasicBlock* ldNo = BasicBlock::Create(CTX, "m.ld.n", func);
      BasicBlock* ldEnd = BasicBlock::Create(CTX, "m.ld.end", func);
      B.CreateCondBr(needLoad, ldYes, ldNo);

      B.SetInsertPoint(ldYes);
      Value* mPairPtr =
          bitCastPtrToVec2(B, B.CreateGEP(scalarTy, gMatBase, idx2), scalarTy);
      mPairLd = B.CreateLoad(getVec2Ty(scalarTy), mPairPtr, "mPair");
      B.CreateBr(ldEnd);

      B.SetInsertPoint(ldNo);
      mPairUndef = UndefValue::get(getVec2Ty(scalarTy));
      B.CreateBr(ldEnd);

      // join load/no-load
      B.SetInsertPoint(ldEnd);
      PHINode* mPairPhi = B.CreatePHI(getVec2Ty(scalarTy), 2, "mPair.phi");
      mPairPhi->addIncoming(mPairLd, ldYes);
      mPairPhi->addIncoming(mPairUndef, ldNo);

      // materialize re/im from kind
      Constant* c0 = ConstantFP::get(scalarTy, 0.0);
      Constant* c1 = ConstantFP::get(scalarTy, 1.0);
      Constant* cN = ConstantFP::get(scalarTy, -1.0);
      auto selK = [&](Value* k, Value* loadedLane) {
        Value* isZ = B.CreateICmpEQ(k, B.getInt32(0));
        Value* isP1 = B.CreateICmpEQ(k, B.getInt32(1));
        Value* isM1 = B.CreateICmpEQ(k, B.getInt32(2));
        Value* v = B.CreateSelect(isP1, c1, loadedLane);
        v = B.CreateSelect(isM1, cN, v);
        v = B.CreateSelect(isZ, c0, v);
        return v;
      };

      Value* loadedRe = B.CreateExtractElement(mPairPhi, B.getInt32(0));
      Value* loadedIm = B.CreateExtractElement(mPairPhi, B.getInt32(1));
      Value* mRe = selK(kRe, loadedRe);
      Value* mIm = selK(kIm, loadedIm);

      // x[col] from shared
      Value* xPairPtr = bitCastPtrToVec2(
          B,
          B.CreateGEP(
              scalarTy, smVecBase, B.CreateMul(B.getInt64(t), B.getInt64(2))),
          scalarTy);
      Value* xPair = B.CreateLoad(getVec2Ty(scalarTy), xPairPtr, "x.pair.t");
      Value* xRe = B.CreateExtractElement(xPair, B.getInt32(0));
      Value* xIm = B.CreateExtractElement(xPair, B.getInt32(1));

      // complex MAC into per-tile accumulators
      Value* pRe = B.CreateFSub(B.CreateFMul(mRe, xRe), B.CreateFMul(mIm, xIm));
      Value* pIm = B.CreateFAdd(B.CreateFMul(mRe, xIm), B.CreateFMul(mIm, xRe));
      Value* accReOut = B.CreateFAdd(accReTile, pRe);
      Value* accImOut = B.CreateFAdd(accImTile, pIm);
      B.CreateBr(contBB);

      // skip path (bothZero)
      B.SetInsertPoint(skipBB);
      B.CreateBr(contBB);

      // join nz/skip
      B.SetInsertPoint(contBB);
      accReAfter = B.CreatePHI(scalarTy, 2, "accRe.after");
      accImAfter = B.CreatePHI(scalarTy, 2, "accIm.after");
      accReAfter->addIncoming(accReOut, /*pred*/ ldEnd);
      accReAfter->addIncoming(accReTile, /*pred*/ skipBB);
      accImAfter->addIncoming(accImOut, /*pred*/ ldEnd);
      accImAfter->addIncoming(accImTile, /*pred*/ skipBB);

      B.CreateBr(endBB);
    }

    // inactive lane (tid ≥ TILE or col out-of-range)
    B.SetInsertPoint(noBB);
    B.CreateBr(endBB);

    // stitch active / inactive
    B.SetInsertPoint(endBB);
    PHINode* accRePhi = B.CreatePHI(scalarTy, 2, "accRe.phi");
    PHINode* accImPhi = B.CreatePHI(scalarTy, 2, "accIm.phi");
    accRePhi->addIncoming(accReAfter, /*pred*/ contBB);
    accRePhi->addIncoming(accReTile, /*pred*/ noBB);
    accImPhi->addIncoming(accImAfter, /*pred*/ contBB);
    accImPhi->addIncoming(accImTile, /*pred*/ noBB);

    accReTile = accRePhi;
    accImTile = accImPhi;
  }

  // __syncthreads() before advancing tile
  B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_barrier0));

  // (3) Next tile: advance and feed loop-carried PHIs
  B.CreateBr(tileInc);
  B.SetInsertPoint(tileInc);
  Value* nextCol0 = B.CreateAdd(col0Phi, B.getInt64(TILE), "col0.next");
  col0Phi->addIncoming(nextCol0, tileInc);
  accReLC->addIncoming(accReTile, tileInc);
  accImLC->addIncoming(accImTile, tileInc);
  B.CreateBr(tileChk);

  // tile.done
  B.SetInsertPoint(tileDone);

  // (4) Store result (only tid < TILE); use loop-carried sums which dominate
  // exit
  BasicBlock* stYes = BasicBlock::Create(CTX, "st.y", func);
  BasicBlock* stNo = BasicBlock::Create(CTX, "st.n", func);
  B.CreateCondBr(tidInTile, stYes, stNo);

  B.SetInsertPoint(stYes);
  {
    Value* off2 = nullptr;
    if (config.assumeContiguousTargets) {
      off2 = B.CreateShl(row64, 1);
    } else {
      Value* r32 = B.CreateIntCast(row64, B.getInt32Ty(), false);
      Value* dPtr = B.CreateGEP(deltaArrTy, gDelta, {B.getInt32(0), r32});
      Value* delta = B.CreateLoad(B.getInt64Ty(), dPtr);
      off2 = B.CreateShl(delta, 1);
    }

    Value* dstPairPtr =
        bitCastPtrToVec2(B, B.CreateGEP(scalarTy, svBaseCombo, off2), scalarTy);
    Value* resPair = UndefValue::get(getVec2Ty(scalarTy));
    resPair = B.CreateInsertElement(resPair, accReLC, B.getInt32(0));
    resPair = B.CreateInsertElement(resPair, accImLC, B.getInt32(1));
    B.CreateStore(resPair, dstPairPtr);
    B.CreateBr(stNo);
  }

  B.SetInsertPoint(stNo);

  // Advance to next combo
  B.CreateBr(cmbInc);
  B.SetInsertPoint(cmbInc);
  Value* comboNext = B.CreateAdd(comboPhi, comboStride);
  comboPhi->addIncoming(comboNext, cmbInc);
  B.CreateBr(cmbChk);

  // cmb.done
  B.SetInsertPoint(cmbDone);
  B.CreateBr(rowEnd);

  // row.end
  B.SetInsertPoint(rowEnd);
}

void genMatrixVectorMultiplyFromPointer_SharedTiled(
    IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const ComplexSquareMatrix& gateMat,
    const QuantumGate::TargetQubitsType& qubits,
    Value* matBasePtr, // AS 1 – run-time matrix
    Value* svPtrV,     // AS 0 – state-vector
    Type* scalarTy) {
  using namespace llvm;

  /*────────── Parameters & types ──────────*/
  const unsigned k = qubits.size();
  const unsigned N = 1u << k;
  const unsigned TILE =
      std::min((config.precision == Precision::F64) ? 128u : 256u, N);

  Module& M = *B.GetInsertBlock()->getModule();
  LLVMContext& CTX = M.getContext();

  /* Shared memory tile for statevector */
  ArrayType* smVecTy = ArrayType::get(scalarTy, 2 * TILE);
  auto* smVecGV = new GlobalVariable(M,
                                     smVecTy,
                                     /*constant=*/false,
                                     GlobalValue::PrivateLinkage,
                                     UndefValue::get(smVecTy),
                                     "TileX",
                                     nullptr,
                                     GlobalValue::NotThreadLocal,
                                     /*AddressSpace=*/3);
  smVecGV->setAlignment(MaybeAlign(8));
  Value* smVecBase =
      B.CreateGEP(smVecTy, smVecGV, {B.getInt32(0), B.getInt32(0)});

  /* Thread & grid indices */
  Value* tid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  Value* bid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
  const unsigned tilesPerGate = (N + TILE - 1) / TILE;
  Value* tilesPerGateV = B.getInt32(tilesPerGate);

  /* packed logical row index (Tile-aware mapping) */
  Value* rowTileIdx32 = B.CreateURem(bid32, tilesPerGateV, "rowTileIdx");
  Value* rowBase64 = B.CreateMul(
      B.CreateZExt(rowTileIdx32, B.getInt64Ty()), B.getInt64(TILE), "rowBase");
  Value* row64 =
      B.CreateAdd(rowBase64, B.CreateZExt(tid32, B.getInt64Ty()), "row64");

  /* Guard against over-run */
  Function* func = B.GetInsertBlock()->getParent();
  if (!func->hasFnAttribute("cast.kstyle"))
    func->addFnAttr("cast.kstyle", "ptr-shared");
  if (!func->hasFnAttribute("cast.tile"))
    func->addFnAttr("cast.tile", std::to_string(TILE));

  BasicBlock* rowOK = BasicBlock::Create(CTX, "row.ok", func);
  BasicBlock* rowEnd = BasicBlock::Create(CTX, "row.end", func);
  B.CreateCondBr(B.CreateICmpULT(row64, B.getInt64(N)), rowOK, rowEnd);

  B.SetInsertPoint(rowOK);

  /* Local accumulators – registers */
  Value* accRe = ConstantFP::get(scalarTy, 0.0);
  Value* accIm = ConstantFP::get(scalarTy, 0.0);

  /* Outer loop over column tiles */
  BasicBlock* tileChk = BasicBlock::Create(CTX, "tile.chk", func);
  BasicBlock* tileBody = BasicBlock::Create(CTX, "tile.body", func);
  BasicBlock* tileInc = BasicBlock::Create(CTX, "tile.inc", func);
  BasicBlock* tileDone = BasicBlock::Create(CTX, "tile.done", func);

  B.CreateBr(tileChk);
  B.SetInsertPoint(tileChk);
  PHINode* col0Phi = B.CreatePHI(B.getInt64Ty(), /*numPreds=*/2, "col0");
  col0Phi->addIncoming(B.getInt64(0), rowOK);
  B.CreateCondBr(B.CreateICmpULT(col0Phi, B.getInt64(N)), tileBody, tileDone);

  /* tileBody */
  B.SetInsertPoint(tileBody);

  /* 1. Load TILE amplitudes of |x〉 into shared memory */
  {
    Value* tid64 = B.CreateZExt(tid32, B.getInt64Ty());
    Value* colIdx = B.CreateAdd(col0Phi, tid64);
    Value* inRange = B.CreateICmpULT(colIdx, B.getInt64(N));

    BasicBlock* ldYes = BasicBlock::Create(CTX, "ld.y", func);
    BasicBlock* ldNo = BasicBlock::Create(CTX, "ld.n", func);
    BasicBlock* ldEnd = BasicBlock::Create(CTX, "ld.end", func);

    B.CreateCondBr(inRange, ldYes, ldNo);

    /* in-range */
    B.SetInsertPoint(ldYes);
    {
      /* map logical column -> physical delta in SV */
      Value* delta = B.getInt64(0);
      for (unsigned b = 0; b < k; ++b) {
        Value* mask = B.getInt64(1ULL << b);
        Value* bitLog = B.CreateAnd(colIdx, mask);
        Value* cond = B.CreateICmpNE(bitLog, B.getInt64(0));
        Value* shift = B.getInt64(1ULL << qubits[b]);
        delta = B.CreateSelect(cond, B.CreateOr(delta, shift), delta);
      }

      Value* off2 = B.CreateMul(delta, B.getInt64(2));
      Value* srcRe = B.CreateGEP(scalarTy, svPtrV, off2);
      Value* srcIm =
          B.CreateGEP(scalarTy, svPtrV, B.CreateAdd(off2, B.getInt64(1)));
      Value* idxS = B.CreateMul(tid64, B.getInt64(2));
      B.CreateStore(B.CreateLoad(scalarTy, srcRe),
                    B.CreateGEP(scalarTy, smVecBase, idxS));
      B.CreateStore(
          B.CreateLoad(scalarTy, srcIm),
          B.CreateGEP(scalarTy, smVecBase, B.CreateAdd(idxS, B.getInt64(1))));
      B.CreateBr(ldEnd);
    }

    /* out-of-range => 0 */
    B.SetInsertPoint(ldNo);
    {
      Constant* zero = ConstantFP::get(scalarTy, 0.0);
      Value* idxS =
          B.CreateMul(B.CreateZExt(tid32, B.getInt64Ty()), B.getInt64(2));
      B.CreateStore(zero, B.CreateGEP(scalarTy, smVecBase, idxS));
      B.CreateStore(
          zero,
          B.CreateGEP(scalarTy, smVecBase, B.CreateAdd(idxS, B.getInt64(1))));
      B.CreateBr(ldEnd);
    }

    B.SetInsertPoint(ldEnd);
  }

  /* 2. __syncthreads() so every thread sees |x〉 tile */
  B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_barrier0));

  /* 3. Dot-product over this tile */
  for (unsigned t = 0; t < TILE; ++t) {
    Value* col = B.CreateAdd(col0Phi, B.getInt64(t));
    Value* inRng = B.CreateICmpULT(col, B.getInt64(N));

    BasicBlock* yesBB = BasicBlock::Create(CTX, "dot.y", func);
    BasicBlock* noBB = BasicBlock::Create(CTX, "dot.n", func);
    BasicBlock* endBB = BasicBlock::Create(CTX, "dot.end", func);
    B.CreateCondBr(inRng, yesBB, noBB);

    /* in-range */
    B.SetInsertPoint(yesBB);
    {
      /* matrix element: matBasePtr[(row,col)] */
      Value* lin = B.CreateAdd(B.CreateMul(row64, B.getInt64(N)), col);
      Value* idx2 = B.CreateMul(lin, B.getInt64(2));
      Value* mRe =
          B.CreateLoad(scalarTy, B.CreateGEP(scalarTy, matBasePtr, idx2));
      Value* mIm = B.CreateLoad(
          scalarTy,
          B.CreateGEP(scalarTy, matBasePtr, B.CreateAdd(idx2, B.getInt64(1))));

      /* vector element from shared mem */
      Value* offX = B.CreateMul(B.getInt64(t), B.getInt64(2));
      Value* xRe =
          B.CreateLoad(scalarTy, B.CreateGEP(scalarTy, smVecBase, offX));
      Value* xIm = B.CreateLoad(
          scalarTy,
          B.CreateGEP(scalarTy, smVecBase, B.CreateAdd(offX, B.getInt64(1))));

      /* accumulate */
      Value* pRe = B.CreateFSub(B.CreateFMul(mRe, xRe), B.CreateFMul(mIm, xIm));
      Value* pIm = B.CreateFAdd(B.CreateFMul(mRe, xIm), B.CreateFMul(mIm, xRe));
      accRe = B.CreateFAdd(accRe, pRe);
      accIm = B.CreateFAdd(accIm, pIm);
      B.CreateBr(endBB);
    }

    /* out-of-range */
    B.SetInsertPoint(noBB);
    B.CreateBr(endBB);

    B.SetInsertPoint(endBB);
  }

  /* 4. Barrier before next tile iteration */
  B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_barrier0));

  /* 5. Advance to next tile */
  B.CreateBr(tileInc);
  B.SetInsertPoint(tileInc);
  Value* nextCol0 = B.CreateAdd(col0Phi, B.getInt64(TILE));
  col0Phi->addIncoming(nextCol0, tileInc);
  B.CreateBr(tileChk);

  /* after last tile */
  B.SetInsertPoint(tileDone);

  /* rebuild physical delta for storing result */
  Value* delta = B.getInt64(0);
  for (unsigned b = 0; b < k; ++b) {
    Value* mask = B.getInt64(1ULL << b);
    Value* bitLog = B.CreateAnd(row64, mask);
    Value* cond = B.CreateICmpNE(bitLog, B.getInt64(0));
    Value* shift = B.getInt64(1ULL << qubits[b]);
    delta = B.CreateSelect(cond, B.CreateOr(delta, shift), delta);
  }

  Value* off2 = B.CreateMul(delta, B.getInt64(2));
  B.CreateStore(accRe, B.CreateGEP(scalarTy, svPtrV, off2));
  B.CreateStore(
      accIm, B.CreateGEP(scalarTy, svPtrV, B.CreateAdd(off2, B.getInt64(1))));

  B.CreateBr(rowEnd);
  B.SetInsertPoint(rowEnd);
}

static void
genMatrixVectorMultiply_InlineImm(IRBuilder<>& B,
                                  const CUDAKernelGenConfig& config,
                                  const ComplexSquareMatrix& /*matrix*/,
                                  const QuantumGate::TargetQubitsType& qubits,
                                  const std::vector<IRMatDataCUDA>& matData,
                                  Value* svPtrV,
                                  Type* scalarTy) {
  llvm::FastMathFlags FMF;
  FMF.setNoNaNs();
  FMF.setNoInfs();
  FMF.setNoSignedZeros();
  FMF.setAllowContract(true);
  FMF.setAllowReassoc(true);
  B.setFastMathFlags(FMF);

  const unsigned k = qubits.size();
  const unsigned K = 1u << k;

  // Precompute per-amplitude pointers and load them once.
  std::vector<Value*> reAmpPtrs(K), imAmpPtrs(K);
  std::vector<Value*> reAmps(K), imAmps(K);

  for (unsigned i = 0; i < K; ++i) {
    uint64_t off2;
    if (config.assumeContiguousTargets) {
      off2 = 2ull * i;
    } else {
      uint64_t delta = 0;
      for (unsigned b = 0; b < k; ++b)
        if (i & (1u << b))
          delta |= (1ull << qubits[b]);
      off2 = 2ull * delta;
    }
    reAmpPtrs[i] = B.CreateConstGEP1_64(scalarTy, svPtrV, off2);
    imAmpPtrs[i] = B.CreateConstGEP1_64(scalarTy, svPtrV, off2 + 1ull);
    reAmps[i] = B.CreateLoad(scalarTy, reAmpPtrs[i]);
    imAmps[i] = B.CreateLoad(scalarTy, imAmpPtrs[i]);
  }

  // For each output row r: new = M[r,*] · old
  for (unsigned r = 0; r < K; ++r) {
    Value* accRe0 = ConstantFP::get(scalarTy, 0.0); // Σ matRe * oldRe
    Value* accRe1 = ConstantFP::get(scalarTy, 0.0); // Σ matIm * oldIm
    Value* accIm =
        ConstantFP::get(scalarTy, 0.0); // Σ matRe*oldIm + matIm*oldRe

    for (unsigned c = 0; c < K; ++c) {
      const auto& md = matData[r * K + c];
      if (md.reKind == SK_Zero && md.imKind == SK_Zero)
        continue;

      // Re(new)
      if (Value* t0 = genOptFMul(md.reVal, reAmps[c], md.reKind, B))
        accRe0 = B.CreateFAdd(accRe0, t0);
      if (Value* t1 = genOptFMul(md.imVal, imAmps[c], md.imKind, B))
        accRe1 = B.CreateFAdd(accRe1, t1);

      // Im(new)
      if (Value* t2 = genOptFMul(md.reVal, imAmps[c], md.reKind, B))
        accIm = B.CreateFAdd(accIm, t2);
      if (Value* t3 = genOptFMul(md.imVal, reAmps[c], md.imKind, B))
        accIm = B.CreateFAdd(accIm, t3);
    }

    Value* newRe = B.CreateFSub(accRe0, accRe1);
    B.CreateStore(newRe, reAmpPtrs[r]);
    B.CreateStore(accIm, imAmpPtrs[r]);
  }
}

static void genMatrixVectorMultiply_InlineImm_LanePerCombo(
    IRBuilder<>& B,
    const CUDAKernelGenConfig& cfg,
    const ComplexSquareMatrix& matrix,
    const QuantumGate::TargetQubitsType& qLSB,
    const std::vector<IRMatDataCUDA>& matData,
    Value* svRoot,
    Type* scalarTy) {
  Function* func = B.GetInsertBlock()->getParent();
  LLVMContext& C = B.getContext();

  // p.combos
  Argument* combosV = nullptr;
  for (auto& A : func->args())
    if (A.getName() == "p.combos") {
      combosV = &A;
      break;
    }
  assert(combosV && "Missing kernel arg 'p.combos'");

  // lane/warp & grid
  Value* tid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  Value* ntid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ntid_x, {});
  Value* cta = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
  Value* gridX = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_nctaid_x, {});

  Value* lane = B.CreateAnd(tid32, B.getInt32(31), "lane");
  Value* warpIn = B.CreateAShr(tid32, B.getInt32(5), "warp.in.cta");
  Value* warpsPerCTA = B.CreateAShr(ntid32, B.getInt32(5), "warps.per.cta");

  // start = (cta*warpsPerCTA + warpIn)*32 + lane
  Value* start32 = B.CreateAdd(
      B.CreateMul(B.CreateAdd(B.CreateMul(cta, warpsPerCTA), warpIn),
                  B.getInt32(32)),
      lane,
      "combo.start");

  // stride = gridDim.x * warpsPerCTA * 32
  Value* stride32 = B.CreateMul(
      B.CreateMul(gridX, warpsPerCTA), B.getInt32(32), "combo.stride");

  // combo loop
  auto* cmbChk = BasicBlock::Create(C, "cmb.chk", func);
  auto* cmbBody = BasicBlock::Create(C, "cmb.body", func);
  auto* cmbInc = BasicBlock::Create(C, "cmb.inc", func);
  auto* cmbDone = BasicBlock::Create(C, "cmb.done", func);

  BasicBlock* pre = B.GetInsertBlock();
  B.CreateBr(cmbChk);
  B.SetInsertPoint(cmbChk);
  auto* comboPhi = B.CreatePHI(B.getInt32Ty(), 2, "combo");
  comboPhi->addIncoming(start32, pre);
  B.CreateCondBr(B.CreateICmpULT(comboPhi, combosV), cmbBody, cmbDone);

  B.SetInsertPoint(cmbBody);
  {
    Value* idx64 = B.CreateZExt(comboPhi, B.getInt64Ty());
    unsigned k = (unsigned)qLSB.size();
    Value* base2 = cfg.assumeContiguousTargets
                       ? B.CreateShl(idx64, k + 1) // ×2 for (re,im)
                       : B.CreateShl(buildOffset(B, idx64, qLSB), 1);
    Value* svComboBase = B.CreateGEP(scalarTy, svRoot, base2, "sv.combo.base");

    // Reuse straight-line multiply on this combo:
    genMatrixVectorMultiply_InlineImm(
        B, cfg, matrix, qLSB, matData, svComboBase, scalarTy);

    B.CreateBr(cmbInc);
  }

  B.SetInsertPoint(cmbInc);
  {
    Value* nextCombo = B.CreateAdd(comboPhi, stride32);
    comboPhi->addIncoming(nextCombo, cmbInc);
    B.CreateBr(cmbChk);
  }

  B.SetInsertPoint(cmbDone);
}

void genMatrixVectorMultiplyFromPointer(
    IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const ComplexSquareMatrix& gateMat,
    const QuantumGate::TargetQubitsType& qubits,
    Value* matBasePtr, // pointer to global memory (AS 1)
    Value* svPtrV,     // pointer to state vector
    Type* scalarTy) {
  unsigned k = qubits.size();
  unsigned K = 1u << k;

  LLVMContext& ctx = B.getContext();
  Function* func = B.GetInsertBlock()->getParent();

  // Get thread index
  Value* tid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, nullptr, "tid");
  Value* row64 = B.CreateZExt(tid32, B.getInt64Ty(), "row");

  // Guard: skip if row >= K
  BasicBlock* rowOkBB = BasicBlock::Create(ctx, "rowOk", func);
  BasicBlock* checkEndBB = BasicBlock::Create(ctx, "rowCheck.end", func);
  Value* condRowOk = B.CreateICmpULT(row64, B.getInt64(K));
  B.CreateCondBr(condRowOk, rowOkBB, checkEndBB);

  B.SetInsertPoint(rowOkBB);

  // Allocate arrays for input amplitudes
  ArrayType* arrTy = ArrayType::get(scalarTy, K);
  AllocaInst* reAmpsAlloca = B.CreateAlloca(arrTy, nullptr, "reAmps");
  AllocaInst* imAmpsAlloca = B.CreateAlloca(arrTy, nullptr, "imAmps");

  // Load input state vector amplitudes
  BasicBlock* loadCheckBB = BasicBlock::Create(ctx, "load.check", func);
  BasicBlock* loadBodyBB = BasicBlock::Create(ctx, "load.body", func);
  BasicBlock* loadIncBB = BasicBlock::Create(ctx, "load.inc", func);
  BasicBlock* loadExitBB = BasicBlock::Create(ctx, "load.exit", func);

  B.CreateBr(loadCheckBB);
  B.SetInsertPoint(loadCheckBB);
  PHINode* iPHI = B.CreatePHI(B.getInt32Ty(), 2, "i");
  iPHI->addIncoming(ConstantInt::get(B.getInt32Ty(), 0), rowOkBB);
  Value* condLoad = B.CreateICmpSLT(iPHI, ConstantInt::get(B.getInt32Ty(), K));
  B.CreateCondBr(condLoad, loadBodyBB, loadExitBB);

  B.SetInsertPoint(loadBodyBB);
  {
    Value* deltaVal = ConstantInt::get(B.getInt64Ty(), 0);
    for (unsigned b = 0; b < k; b++) {
      Value* maskB = ConstantInt::get(B.getInt32Ty(), 1u << b);
      Value* test = B.CreateAnd(iPHI, maskB);
      Value* cond = B.CreateICmpNE(test, ConstantInt::get(B.getInt32Ty(), 0));
      Value* shiftVal = ConstantInt::get(B.getInt64Ty(), 1ULL << qubits[b]);
      Value* orVal = B.CreateOr(deltaVal, shiftVal);
      deltaVal = B.CreateSelect(cond, orVal, deltaVal);
    }

    Value* twoDelta =
        B.CreateMul(ConstantInt::get(B.getInt64Ty(), 2), deltaVal);
    Value* rePtr = B.CreateGEP(scalarTy, svPtrV, twoDelta, "rePtr");
    Value* imPtr =
        B.CreateGEP(scalarTy,
                    svPtrV,
                    B.CreateAdd(twoDelta, ConstantInt::get(B.getInt64Ty(), 1)),
                    "imPtr");

    Value* oldRe = B.CreateLoad(scalarTy, rePtr, "oldRe");
    Value* oldIm = B.CreateLoad(scalarTy, imPtr, "oldIm");

    Value* reSlot = B.CreateGEP(
        arrTy, reAmpsAlloca, {ConstantInt::get(B.getInt32Ty(), 0), iPHI});
    Value* imSlot = B.CreateGEP(
        arrTy, imAmpsAlloca, {ConstantInt::get(B.getInt32Ty(), 0), iPHI});
    B.CreateStore(oldRe, reSlot);
    B.CreateStore(oldIm, imSlot);
  }
  B.CreateBr(loadIncBB);

  B.SetInsertPoint(loadIncBB);
  {
    Value* iNext = B.CreateAdd(iPHI, ConstantInt::get(B.getInt32Ty(), 1));
    iPHI->addIncoming(iNext, loadIncBB);
    B.CreateBr(loadCheckBB);
  }

  B.SetInsertPoint(loadExitBB);
  attachNoUnrollMetadata(B, loadIncBB);

  // Compute output amplitude for row
  AllocaInst* reAmp0A = B.CreateAlloca(scalarTy, nullptr, "reAmp0A");
  AllocaInst* reAmp1A = B.CreateAlloca(scalarTy, nullptr, "reAmp1A");
  AllocaInst* imAmpA = B.CreateAlloca(scalarTy, nullptr, "imAmpA");
  Value* zeroVal = ConstantFP::get(scalarTy, 0.0);
  B.CreateStore(zeroVal, reAmp0A);
  B.CreateStore(zeroVal, reAmp1A);
  B.CreateStore(zeroVal, imAmpA);

  // Inner loop over columns
  BasicBlock* innerCheckBB = BasicBlock::Create(ctx, "inner.check", func);
  BasicBlock* innerBodyBB = BasicBlock::Create(ctx, "inner.body", func);
  BasicBlock* innerIncBB = BasicBlock::Create(ctx, "inner.inc", func);
  BasicBlock* innerExitBB = BasicBlock::Create(ctx, "inner.exit", func);

  B.CreateBr(innerCheckBB);
  B.SetInsertPoint(innerCheckBB);
  PHINode* cPHI = B.CreatePHI(B.getInt32Ty(), 2, "c");
  cPHI->addIncoming(ConstantInt::get(B.getInt32Ty(), 0), loadExitBB);
  Value* condInner = B.CreateICmpSLT(cPHI, ConstantInt::get(B.getInt32Ty(), K));
  B.CreateCondBr(condInner, innerBodyBB, innerExitBB);

  B.SetInsertPoint(innerBodyBB);
  {
    Value* r64 = B.CreateIntCast(row64, B.getInt64Ty(), false);
    Value* c64 = B.CreateIntCast(cPHI, B.getInt64Ty(), false);
    Value* bigK = ConstantInt::get(B.getInt64Ty(), K);
    Value* rK = B.CreateMul(r64, bigK);
    Value* rc = B.CreateAdd(rK, c64);
    Value* base2 = B.CreateMul(ConstantInt::get(B.getInt64Ty(), 2), rc);

    Value* rePtr = B.CreateGEP(scalarTy, matBasePtr, base2, "matRePtr");
    Value* imPtr =
        B.CreateGEP(scalarTy,
                    matBasePtr,
                    B.CreateAdd(base2, ConstantInt::get(B.getInt64Ty(), 1)),
                    "matImPtr");
    Value* matRe = B.CreateLoad(scalarTy, rePtr, "matRe");
    Value* matIm = B.CreateLoad(scalarTy, imPtr, "matIm");

    Value* oldReSlot = B.CreateGEP(
        arrTy, reAmpsAlloca, {ConstantInt::get(B.getInt32Ty(), 0), cPHI});
    Value* oldImSlot = B.CreateGEP(
        arrTy, imAmpsAlloca, {ConstantInt::get(B.getInt32Ty(), 0), cPHI});
    Value* oldRe = B.CreateLoad(scalarTy, oldReSlot, "oldRe");
    Value* oldIm = B.CreateLoad(scalarTy, oldImSlot, "oldIm");

    Value* reA0 = B.CreateLoad(scalarTy, reAmp0A);
    Value* reA1 = B.CreateLoad(scalarTy, reAmp1A);
    Value* imA = B.CreateLoad(scalarTy, imAmpA);

    Value* addRe0 = B.CreateFAdd(reA0, B.CreateFMul(matRe, oldRe));
    B.CreateStore(addRe0, reAmp0A);

    Value* addRe1 = B.CreateFAdd(reA1, B.CreateFMul(matIm, oldIm));
    B.CreateStore(addRe1, reAmp1A);

    Value* cross =
        B.CreateFAdd(B.CreateFMul(matRe, oldIm), B.CreateFMul(matIm, oldRe));
    Value* addIm = B.CreateFAdd(imA, cross);
    B.CreateStore(addIm, imAmpA);
  }
  B.CreateBr(innerIncBB);

  B.SetInsertPoint(innerIncBB);
  {
    Value* cNext = B.CreateAdd(cPHI, ConstantInt::get(B.getInt32Ty(), 1));
    cPHI->addIncoming(cNext, innerIncBB);
    B.CreateBr(innerCheckBB);
  }

  B.SetInsertPoint(innerExitBB);
  attachNoUnrollMetadata(B, innerIncBB);

  Value* reA0 = B.CreateLoad(scalarTy, reAmp0A);
  Value* reA1 = B.CreateLoad(scalarTy, reAmp1A);
  Value* imA = B.CreateLoad(scalarTy, imAmpA);
  Value* newReAmp = B.CreateFSub(reA0, reA1);
  Value* newImAmp = imA;

  // Store back to state vector
  Value* deltaVal = ConstantInt::get(B.getInt64Ty(), 0);
  for (unsigned b = 0; b < k; b++) {
    Value* maskB = ConstantInt::get(B.getInt64Ty(), 1ULL << b); // Fix: i64
    Value* test = B.CreateAnd(row64, maskB);
    Value* cond =
        B.CreateICmpNE(test, ConstantInt::get(B.getInt64Ty(), 0)); // Fix: i64
    Value* shiftVal = ConstantInt::get(B.getInt64Ty(), 1ULL << qubits[b]);
    Value* orVal = B.CreateOr(deltaVal, shiftVal);
    deltaVal = B.CreateSelect(cond, orVal, deltaVal);
  }
  Value* twoDelta = B.CreateMul(ConstantInt::get(B.getInt64Ty(), 2), deltaVal);
  Value* outRePtr = B.CreateGEP(scalarTy, svPtrV, twoDelta);
  Value* outImPtr =
      B.CreateGEP(scalarTy,
                  svPtrV,
                  B.CreateAdd(twoDelta, ConstantInt::get(B.getInt64Ty(), 1)));
  B.CreateStore(newReAmp, outRePtr);
  B.CreateStore(newImAmp, outImPtr);

  B.CreateBr(checkEndBB);
  B.SetInsertPoint(checkEndBB);
}

void genMatrixVectorMultiplyFromConst(
    IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const ComplexSquareMatrix& gateMat,
    const QuantumGate::TargetQubitsType& qubits,
    GlobalVariable* gConstMat,
    Value* svPtrV,
    Value* matPtrV,
    Type* scalarTy,
    const std::vector<IRMatDataCUDA>& matData) {
  llvm::FastMathFlags FMF;
  FMF.setNoNaNs();
  FMF.setNoInfs();
  FMF.setNoSignedZeros();
  FMF.setAllowContract(true);
  FMF.setAllowReassoc(true);
  B.setFastMathFlags(FMF);

  unsigned k = qubits.size();
  unsigned K = 1u << k;

  std::cerr << "USECONST!\n";

  std::vector<Value*> reAmpPtrs(K), imAmpPtrs(K);
  std::vector<Value*> reAmps(K), imAmps(K);

  // Load statevector amplitudes
  for (unsigned i = 0; i < K; i++) {
    uint64_t delta = 0;
    for (unsigned b = 0; b < k; b++) {
      if (i & (1u << b))
        delta |= (1ull << qubits[b]);
    }
    reAmpPtrs[i] = B.CreateConstGEP1_64(
        scalarTy, svPtrV, 2 * delta, "reAmpPtr." + std::to_string(i));
    imAmpPtrs[i] = B.CreateConstGEP1_64(
        scalarTy, svPtrV, 2 * delta + 1, "imAmpPtr." + std::to_string(i));

    reAmps[i] =
        B.CreateLoad(scalarTy, reAmpPtrs[i], "oldRe." + std::to_string(i));
    imAmps[i] =
        B.CreateLoad(scalarTy, imAmpPtrs[i], "oldIm." + std::to_string(i));
  }

  auto* arrTy = llvm::cast<ArrayType>(gConstMat->getValueType());
  for (unsigned r = 0; r < K; ++r) {
    Value* accRe0 = ConstantFP::get(scalarTy, 0.0); // sum(matRe * oldRe)
    Value* accRe1 = ConstantFP::get(scalarTy, 0.0); // sum(matIm * oldIm)
    Value* accIm =
        ConstantFP::get(scalarTy, 0.0); // sum(matRe*oldIm + matIm*oldRe)

    for (unsigned c = 0; c < K; ++c) {
      const IRMatDataCUDA& md = matData[r * K + c];

      // Skip entries that are exactly zero (within tolerance)
      if (md.reKind == SK_Zero && md.imKind == SK_Zero)
        continue; // no IR for this (r,c)

      // We only touch constant memory if at least one part is "runtime".
      Value* matRe = nullptr;
      Value* matIm = nullptr;

      if (md.reKind == SK_Runtime || md.imKind == SK_Runtime) {
        uint64_t baseIndex = 2ull * (r * K + c);
        Value* idx2 = B.getInt64(baseIndex);
        Value* baseScalarPtr =
            B.CreateGEP(arrTy, gConstMat, {B.getInt32(0), idx2});
        Value* pairPtr = bitCastPtrToVec2(B, baseScalarPtr, scalarTy);
        Value* pair = B.CreateLoad(getVec2Ty(scalarTy), pairPtr, "mPair");

        if (md.reKind == SK_Runtime)
          matRe = B.CreateExtractElement(pair, (uint64_t)0);
        else
          matRe = md.reVal; // 1, -1 or immediate

        if (md.imKind == SK_Runtime)
          matIm = B.CreateExtractElement(pair, 1);
        else
          matIm = md.imVal; // 1, -1 or immediate
      } else {
        // Both parts are compile-time constants (0, ±1, or immediate)
        matRe = md.reVal;
        matIm = md.imVal;
      }

      Value* oldRe = reAmps[c];
      Value* oldIm = imAmps[c];

      // Re(new) = Σ ( matRe*oldRe ) - Σ ( matIm*oldIm )
      if (Value* t0 = genOptFMul(matRe, oldRe, md.reKind, B))
        accRe0 = B.CreateFAdd(accRe0, t0);
      if (Value* t1 = genOptFMul(matIm, oldIm, md.imKind, B))
        accRe1 = B.CreateFAdd(accRe1, t1);

      // Im(new) = Σ ( matRe*oldIm + matIm*oldRe )
      if (Value* t2 = genOptFMul(matRe, oldIm, md.reKind, B))
        accIm = B.CreateFAdd(accIm, t2);
      if (Value* t3 = genOptFMul(matIm, oldRe, md.imKind, B))
        accIm = B.CreateFAdd(accIm, t3);
    }

    Value* newReAmp = B.CreateFSub(accRe0, accRe1);
    B.CreateStore(newReAmp, reAmpPtrs[r]);
    B.CreateStore(accIm, imAmpPtrs[r]);
  }
}

GlobalVariable*
getOrCreateConstMatGlobal(Module& M, Type* arrTy, StringRef globalName) {
  // Check if already exists
  if (auto* gv = M.getNamedGlobal(globalName))
    return gv;

  // Create new global in address space 4
  auto* gConstMat =
      new GlobalVariable(M,
                         arrTy,
                         /* isConstant */ true,
                         GlobalValue::ExternalLinkage,
                         UndefValue::get(arrTy), // Initialized later
                         globalName,
                         nullptr,
                         GlobalValue::NotThreadLocal,
                         /* addressSpace */ 4);
  gConstMat->setAlignment(MaybeAlign(8));
  return gConstMat;
}

} // end of anonymous namespace

// Function *CUDAKernelManager::gen_(
//     const CUDAKernelGenConfig &config,
//     const ComplexSquareMatrix &matrix,
//     const QuantumGate::TargetQubitsType &qubits,
//     const std::string &funcName
// ) {
//     const unsigned k  = qubits.size();
//     const unsigned K  = 1ULL << k;
//     const unsigned KK = K * K;

//     CUDAKernelGenConfig cfg = config;
//     // if (cfg.matrixLoadMode == CUDAMatrixLoadMode::UseMatImmValues && k <=
//     3)
//     //     cfg.matrixLoadMode = CUDAMatrixLoadMode::LoadInConstMemSpace;

//     auto &llvmContextModulePair = createNewLLVMContextModulePair(funcName +
//     "Module"); IRBuilder<> B(*llvmContextModulePair.llvmContext);

//     assert(config.precision != Precision::Unknown);
//     Type *scalarTy = (cfg.precision == Precision::F32) ? B.getFloatTy() :
//     B.getDoubleTy();

//     IRArgsCUDA args;
//     auto *func = getFunctionDeclarationCUDA(
//         B, *llvmContextModulePair.llvmModule, funcName, config, args
//     );

//     auto *entryBB = BasicBlock::Create(*llvmContextModulePair.llvmContext,
//     "entry", func); B.SetInsertPoint(entryBB);

//     Value *svPtrV;
//     {
//         // auto* counterV = helper_getBlockIdx(B);
//         // auto* offset   = buildOffset(B, counterV, qubits);
//         // auto* idxStartV =
//         //     B.CreateShl(offset, 1, "twice.offset"); // x2 because (re,im)
//         // svPtrV = B.CreateGEP(scalarTy, args.pSvArg, idxStartV, "sv.ptr");

//         if (cfg.matrixLoadMode == CUDAMatrixLoadMode::UseMatImmValues &&
//             cfg.assumeContiguousTargets) {
//             // Compute combosIdx = ctaid.x / tilesPerGate (tilesPerGate is
//             compile-time const here) const unsigned N = 1u << k; const
//             unsigned TILE = std::min(256u, N); const unsigned tilesPerGate =
//             (N + TILE - 1u) / TILE;

//             auto *cta = B.CreateIntrinsic(B.getInt32Ty(),
//             Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}); auto *combosIdx32 =
//             B.CreateUDiv(cta, B.getInt32(tilesPerGate)); auto *combosIdx64 =
//             B.CreateZExt(combosIdx32, B.getInt64Ty());

//             // base2 = (combosIdx << k) * 2 == combosIdx << (k+1)
//             auto *base2 = B.CreateShl(combosIdx64, k + 1, "base2");
//             svPtrV = B.CreateGEP(scalarTy, args.pSvArg, base2, "sv.ptr");
//         } else {
//             const unsigned N = 1u << k;
//             const unsigned TILE = std::min(256u, N);
//             const unsigned tilesPerGate = (N + TILE - 1u) / TILE;

//             auto *cta = B.CreateIntrinsic(B.getInt32Ty(),
//             Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}); auto *combosIdx32 =
//             B.CreateUDiv(cta, B.getInt32(tilesPerGate)); auto *combosIdx64 =
//             B.CreateZExt(combosIdx32, B.getInt64Ty());

//             auto *offset = buildOffset(B, combosIdx64, qubits);
//             auto *idxStartV = B.CreateShl(offset, 1, "twice.offset"); // x2
//             because (re,im) svPtrV = B.CreateGEP(scalarTy, args.pSvArg,
//             idxStartV, "sv.ptr");
//         }
//     }
//     svPtrV = args.pSvArg;

//     auto matData = getMatDataCUDA(B, cfg, matrix, k);

//     switch (cfg.matrixLoadMode) {
//         // case CUDAMatrixLoadMode::UseMatImmValues: {
//         //     // // heuristic: inline up to K<=16 (k<=4) for f32, K<=8
//         (k<=3) for f64
//         //     // const bool inlineOK =
//         //     //     (cfg.precision == Precision::F32) ? (k <= 4) : (k <=
//         3);
//         //     // if (inlineOK) {
//         //     //     genMatrixVectorMultiply_InlineImm(B, cfg, matrix,
//         qubits, matData, svPtrV, scalarTy);
//         //     //     func->addFnAttr("cast.kstyle", "imm-inline");
//         //     // } else {
//         //     //     genMatrixVectorMultiply_SharedTiled(B, cfg, matrix,
//         qubits, matData, svPtrV, scalarTy);
//         //     //     func->addFnAttr("cast.kstyle", "imm-shared");
//         //     // }
//         //     // break;

//         //     std::vector<int> qLSB = qubits;
//         //     if (cfg.assumeContiguousTargets) {
//         //         qLSB.resize(k);
//         //         std::iota(qLSB.begin(), qLSB.end(), 0);
//         //     }

//         //     const bool inlineOK = (cfg.precision == Precision::F32) ? (k
//         <= 4) : (k <= 3);
//         //     if (inlineOK) {
//         //       genMatrixVectorMultiply_ImmRowPerLane(B, cfg, qLSB, matData,
//         svPtrV, scalarTy);
//         //       // genMatrixVectorMultiply_InlineImm(B, cfg, matrix, qLSB,
//         matData, svPtrV, scalarTy);
//         //       // func->addFnAttr("cast.kstyle", "imm-inline");
//         //     } else {
//         //         genMatrixVectorMultiply_SharedTiled(B, cfg, matrix, qLSB,
//         matData, svPtrV, scalarTy);
//         //         func->addFnAttr("cast.kstyle", "imm-shared");
//         //         // genMatrixVectorMultiply_SharedTiled_WarpPerRow(B, cfg,
//         matrix, qLSB, matData, svPtrV, scalarTy);
//         //     }
//         //     break;
//         // }

//         case CUDAMatrixLoadMode::UseMatImmValues: {
//             std::vector<int> qLSB = qubits;
//             if (cfg.assumeContiguousTargets) {
//                 qLSB.resize(k);
//                 std::iota(qLSB.begin(), qLSB.end(), 0); // {0..k-1}
//             }
//             // Heuristic: warp-per-row for small/medium K; shared-tiled for
//             bigger K
//             // const bool useWPR = (cfg.precision == Precision::F32) ? (k <=
//             6) : (k <= 5);
//             // if (useWPR) {
//             //     genMatrixVectorMultiply_ImmRowPerLane(B, cfg, qLSB,
//             matData, svPtrV, scalarTy);
//             //     // The callee sets cast.kstyle="imm-shared-warp" +
//             cast.warps
//             // } else {
//             //     genMatrixVectorMultiply_SharedTiled(B, cfg, matrix, qLSB,
//             matData, svPtrV, scalarTy);
//             //     func->addFnAttr("cast.kstyle", "imm-shared");
//             // }
//             const unsigned N = 1u << k;
//             if ((cfg.precision == Precision::F32 ? (N <= 8) : (N <= 4))) {
//                 // Tiny gates → straight-line inline path (best warp
//                 efficiency and least CF) genMatrixVectorMultiply_InlineImm(B,
//                 cfg, matrix, qLSB, matData, svPtrV, scalarTy);
//                 func->addFnAttr("cast.kstyle", "imm-inline");
//             } else if (N >= 32 && N <= 64) {
//                 // Row-per-lane only when we have ≥1 full warp of rows
//                 genMatrixVectorMultiply_ImmRowPerLane(B, cfg, qLSB, matData,
//                 svPtrV, scalarTy);
//                 // cast.kstyle set by callee
//             } else {
//                 // Mid/large → shared tiled
//                 genMatrixVectorMultiply_SharedTiled(B, cfg, matrix, qLSB,
//                 matData, svPtrV, scalarTy); func->addFnAttr("cast.kstyle",
//                 "imm-shared");
//             }
//             break;
//         }

//         // case CUDAMatrixLoadMode::UseMatImmValues: {
//         //     // TODO: why number of qubits here
//         //     if (k < config.enableTilingGateSize) {
//         //         genMatrixVectorMultiply(
//         //             B, config, matrix, qubits, matData, svPtrV, scalarTy);
//         //     } else {
//         //         genMatrixVectorMultiply_SharedTiled(
//         //             B, config, matrix, qubits, matData, svPtrV, scalarTy);
//         //     }
//         //     break;
//         // }

//         case CUDAMatrixLoadMode::LoadInDefaultMemSpace: {
//             // This path loads matrix from pMatArg (args.pMatArg) in address
//             space 1 Value *matBasePtr = args.pMatArg;

//             // Cast from address space 0 to 1:
//             matBasePtr = B.CreateAddrSpaceCast(
//                 args.pMatArg,
//                 PointerType::get(scalarTy, /*AS=*/1),
//                 "matGlobalPtr"
//             );

//             // Generate the IR that loops over the matrix elements
//             // (K*K complex elements) and loads from matBasePtr + offset
//             if (k < config.enableTilingGateSize) {
//                 genMatrixVectorMultiplyFromPointer(
//                     B, config, matrix, qubits, matBasePtr, svPtrV, scalarTy
//                 );
//             } else {
//                 genMatrixVectorMultiplyFromPointer_SharedTiled(
//                     B, config, matrix, qubits, matBasePtr, svPtrV, scalarTy
//                 );
//             }
//             break;
//         }

//         case CUDAMatrixLoadMode::LoadInConstMemSpace: {
//             // This path loads the matrix from pointer in address space 4
//             // (constant memory space)
//             auto *arrTy = ArrayType::get(scalarTy, 2U * KK);
//             auto *gConstMat = getOrCreateConstMatGlobal(
//                 *llvmContextModulePair.llvmModule, arrTy, "gConstMatShared"
//             );

//             // Initialize with matrix values if available
//             std::vector<Constant *> constElems;
//             constElems.reserve(2U * KK);
//             for (unsigned r = 0; r < K; ++r) {
//                 for (unsigned c = 0; c < K; ++c) {
//                     auto cplx = matrix.rc(r, c);
//                     constElems.push_back(ConstantFP::get(scalarTy,
//                     cplx.real()));
//                     constElems.push_back(ConstantFP::get(scalarTy,
//                     cplx.imag()));
//                 }
//             }
//             gConstMat->setInitializer(ConstantArray::get(arrTy, constElems));

//             genMatrixVectorMultiplyFromConst(
//                 B, config, matrix, qubits, gConstMat, svPtrV, args.pMatArg,
//                 scalarTy, matData
//             );
//             func->addFnAttr("cast.kstyle", "const-small");
//             break;
//         }
//     }

//     B.CreateRetVoid();
//     // LLVM_DEBUG(func->dump());
//     return func;
// }

cast::MaybeError<cast::CUDAKernelManager::KernelPair>
CUDAKernelManager::genCUDAGateVariants_(const CUDAKernelGenConfig& config,
                                        ConstQuantumGatePtr gate,
                                        const std::string& baseName) {
  auto buildOne = [&](bool assumeContiguous,
                      const std::string& name) -> llvm::Function* {
    CUDAKernelGenConfig cfg = config;
    cfg.assumeContiguousTargets = assumeContiguous;

    llvm::Function* fn = nullptr;

    if (auto* stdQuGate = llvm::dyn_cast<const StandardQuantumGate>(gate.get());
        stdQuGate && stdQuGate->noiseChannel() == nullptr) {
      const auto scalarGM = stdQuGate->getScalarGM();
      assert(scalarGM && "Only supporting scalar GM for now");

      auto q = stdQuGate->qubits();
      std::vector<int> qArg = q;
      if (assumeContiguous) {
        qArg.resize(qArg.size());
        std::iota(qArg.begin(), qArg.end(), 0); // {0..k-1}
      }
      fn = gen_(cfg, scalarGM->matrix(), qArg, name);
    } else {
      // super-op path
      auto superopGate = gate->getSuperopGate();
      assert(superopGate && "Superop gate should not be null");
      const auto scalarGM = superopGate->getMatrix();
      assert(scalarGM && "superop gate matrix should not be null");

      auto qs = superopGate->qubits();
      auto nQ = superopGate->nQubits();

      std::vector<int> both = qs;
      for (auto q : qs)
        both.push_back(q + nQ);

      std::vector<int> qArg = both;
      if (assumeContiguous) {
        qArg.resize(qArg.size());
        std::iota(qArg.begin(), qArg.end(), 0);
      }
      fn = gen_(cfg, scalarGM->matrix(), qArg, name);
    }

    if (fn)
      fn->addFnAttr("cast.addr", assumeContiguous ? "lsb" : "generic");
    return fn;
  };

  const std::string nameLSB = baseName + "_lsb";
  const std::string nameGEN = baseName + "_gen";

  llvm::Function* fLSB = buildOne(/*assumeContiguous=*/true, nameLSB);
  llvm::Function* fGEN = buildOne(/*assumeContiguous=*/false, nameGEN);

  if (!fLSB || !fGEN) {
    return cast::makeError<KernelPair>(
        "Failed to generate one or both variants for gate '" + baseName + "'");
  }

  CUDAKernelInfo::CUDATuple cuTuple;

  auto makeKI = [&](llvm::Function* f) -> KernelInfoPtr {
    auto ki = std::make_unique<CUDAKernelInfo>(CUDAKernelInfo::PTXStringType(),
                                               config.precision,
                                               f->getName().str(),
                                               gate,
                                               CUDAKernelInfo::CUDATuple{},
                                               gate->opCount(config.zeroTol));

    // if (f->hasFnAttribute("cast.kstyle")) {
    //     auto style = f->getFnAttribute("cast.kstyle").getValueAsString();
    //     // inline (or const-small) => 1 thread/block
    //     if (style == "imm-inline" || style == "const-small") {
    //         ki->oneThreadPerBlock = true;
    //     }
    //     // WPR shared => ask for small fixed #warps (default 4)
    //     else if (style == "imm-shared-warp") {
    //         ki->oneThreadPerBlock = false;
    //         if (f->hasFnAttribute("cast.warps")) {
    //             auto s =
    //             f->getFnAttribute("cast.warps").getValueAsString().str();
    //             ki->warpsPerCTA = std::max(1u, (unsigned)std::stoi(s));
    //         } else {
    //             ki->warpsPerCTA = 4; // sane default
    //         }
    //     }
    // }
    if (f->hasFnAttribute("cast.kstyle")) {
      auto style = f->getFnAttribute("cast.kstyle").getValueAsString();
      if (style == "imm-inline" || style == "const-small") {
        ki->oneThreadPerBlock = true;
      } else if (style == "imm-inline-warp" || style == "imm-shared-warp") {
        ki->oneThreadPerBlock = false;
        if (f->hasFnAttribute("cast.warps")) {
          auto s = f->getFnAttribute("cast.warps").getValueAsString().str();
          ki->warpsPerCTA = std::max(1u, (unsigned)std::stoi(s));
        } else {
          ki->warpsPerCTA = 4;
        }
      }
    }

    return ki;
  };

  KernelPair out{makeKI(fLSB), makeKI(fGEN)};
  return out;
}

Function* CUDAKernelManager::gen_(const CUDAKernelGenConfig& config,
                                  const ComplexSquareMatrix& matrix,
                                  const QuantumGate::TargetQubitsType& qubits,
                                  const std::string& funcName) {
  const unsigned k = qubits.size();
  const unsigned K = 1u << k;
  const unsigned KK = K * K;

  CUDAKernelGenConfig cfg = config;

  auto& llvmContextModulePair =
      createNewLLVMContextModulePair(funcName + "Module");
  IRBuilder<> B(*llvmContextModulePair.llvmContext);

  assert(cfg.precision != Precision::Unknown);
  Type* scalarTy =
      (cfg.precision == Precision::F32) ? B.getFloatTy() : B.getDoubleTy();

  // Create the kernel function and plumb the arguments (p.sv, [p.mat],
  // p.combos)
  IRArgsCUDA args;
  auto* func = getFunctionDeclarationCUDA(
      B, *llvmContextModulePair.llvmModule, funcName, cfg, args);

  auto* entryBB =
      BasicBlock::Create(*llvmContextModulePair.llvmContext, "entry", func);
  B.SetInsertPoint(entryBB);

  // Helper: compute the base pointer (into the full statevector) for this CTA's
  // combo. We map comboSlot := ctaid.x / tilesPerGate, where tilesPerGate is
  // the number of row-tiles per gate for the chosen kernel style (or 1 when
  // there is no row tiling).
  auto comboBasePtr =
      [&](unsigned tilesPerGate,
          const QuantumGate::TargetQubitsType& qsForOffset) -> Value* {
    // ctaid.x
    auto* cta = B.CreateIntrinsic(
        B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
    Value* idx32 =
        (tilesPerGate == 1) ? cta : B.CreateUDiv(cta, B.getInt32(tilesPerGate));
    Value* idx64 = B.CreateZExt(idx32, B.getInt64Ty());

    Value* base2 = nullptr;
    if (cfg.assumeContiguousTargets) {
      // combo << (k+1)  (×2 for (re,im))
      base2 = B.CreateShl(idx64, k + 1);
    } else {
      // Generic mapping: pdep-style using target-bit mask
      Value* off =
          buildOffset(B, idx64, qsForOffset); // qsForOffset must be ascending
      base2 = B.CreateShl(off, 1);
    }
    return B.CreateGEP(scalarTy, args.pSvArg, base2, "sv.combo.base");
  };

  // For kernels that need compile-time matrix analysis
  auto matData = getMatDataCUDA(B, cfg, matrix, k);

  // Build a "local view" of qubits: for LSB kernels we want {0..k-1},
  // otherwise we keep the provided (ascending) physical positions.
  std::vector<int> qLSB = qubits;
  if (cfg.assumeContiguousTargets) {
    qLSB.resize(k);
    std::iota(qLSB.begin(), qLSB.end(), 0); // {0..k-1}
  }

  switch (cfg.matrixLoadMode) {
  case CUDAMatrixLoadMode::UseMatImmValues: {
    // const unsigned N = 1u << k;

    // // Tiny gates -> straight-line inline path per combo
    // const bool inlineOK = (cfg.precision == Precision::F32) ? (N <= 8) : (N
    // <= 4); if (inlineOK) {
    //     // No row-tiling; one CTA per combo ⇒ tilesPerGate = 1
    //     Value *svComboBase = comboBasePtr(/*tilesPerGate=*/1, qLSB);
    //     genMatrixVectorMultiply_InlineImm(
    //         B, cfg, matrix, qLSB, matData, svComboBase, scalarTy
    //     );
    //     func->addFnAttr("cast.kstyle", "imm-inline");
    //     break;

    //     // Argument *combosV = nullptr;
    //     // for (auto &A : func->args()) if (A.getName() == "p.combos") {
    //     combosV = &A; break; }
    //     // assert(combosV && "Missing kernel arg 'p.combos'");

    //     // auto *cta   = B.CreateIntrinsic(B.getInt32Ty(),
    //     Intrinsic::nvvm_read_ptx_sreg_ctaid_x,   {});
    //     // auto *gridX = B.CreateIntrinsic(B.getInt32Ty(),
    //     Intrinsic::nvvm_read_ptx_sreg_nctaid_x, {});

    //     // // combo loop scaffolding
    //     // auto *cmbChk  = BasicBlock::Create(B.getContext(), "cmb.chk",
    //     func);
    //     // auto *cmbBody = BasicBlock::Create(B.getContext(), "cmb.body",
    //     func);
    //     // auto *cmbInc  = BasicBlock::Create(B.getContext(), "cmb.inc",
    //     func);
    //     // auto *cmbDone = BasicBlock::Create(B.getContext(), "cmb.done",
    //     func);

    //     // B.CreateBr(cmbChk);
    //     // B.SetInsertPoint(cmbChk);
    //     // auto *comboPhi = B.CreatePHI(B.getInt32Ty(), 2, "combo");
    //     // comboPhi->addIncoming(cta, entryBB);
    //     // B.CreateCondBr(B.CreateICmpULT(comboPhi, combosV), cmbBody,
    //     cmbDone);

    //     // B.SetInsertPoint(cmbBody);
    //     // {
    //     //   Value *idx64 = B.CreateZExt(comboPhi, B.getInt64Ty());
    //     //   Value *base2 = cfg.assumeContiguousTargets
    //     //                 ? B.CreateShl(idx64, k + 1)
    //     //                 : B.CreateShl(buildOffset(B, idx64, qLSB), 1);
    //     //   Value *svComboBase = B.CreateGEP(scalarTy, args.pSvArg, base2,
    //     "sv.combo.base");
    //     //   genMatrixVectorMultiply_InlineImm(B, cfg, matrix, qLSB, matData,
    //     svComboBase, scalarTy);
    //     //   B.CreateBr(cmbInc);
    //     // }

    //     // B.SetInsertPoint(cmbInc);
    //     // {
    //     //   Value *nextCombo = B.CreateAdd(comboPhi, gridX);
    //     //   comboPhi->addIncoming(nextCombo, cmbInc);
    //     //   B.CreateBr(cmbChk);
    //     // }

    //     // B.SetInsertPoint(cmbDone);
    //     // func->addFnAttr("cast.kstyle", "imm-inline");
    // }

    // // Medium (one warp handles a row each): the kernel loops over combos
    // internally if (N >= 32 && N <= 64) {
    //     // Pass the full SV base; the callee reads p.combos and iterates
    //     combos genMatrixVectorMultiply_ImmRowPerLane(
    //         B, cfg, qLSB, matData, /*svBase=*/args.pSvArg, scalarTy
    //     );
    //     // callee sets "cast.kstyle" and "cast.warps"
    //     break;
    // }

    // // Large → shared tiled (kernel iterates combos internally)
    // genMatrixVectorMultiply_SharedTiled(
    //     B, cfg, matrix, qLSB, matData, /*svBase=*/args.pSvArg, scalarTy
    // );
    // func->addFnAttr("cast.kstyle", "imm-shared");
    // break;

    const unsigned N = 1u << k;
    const bool inlineOK = (N <= 8);

    // Build {0..k-1} for LSB kernels
    std::vector<int> qLSB = qubits;
    if (cfg.assumeContiguousTargets) {
      qLSB.resize(k);
      std::iota(qLSB.begin(), qLSB.end(), 0);
    }

    if (inlineOK) {
      // lane-per-combo persistent inline
      genMatrixVectorMultiply_InlineImm_LanePerCombo(
          B, cfg, matrix, qLSB, matData, /*svRoot=*/args.pSvArg, scalarTy);
      func->addFnAttr("cast.kstyle", "imm-inline-warp");
      func->addFnAttr("cast.warps",
                      "4"); // default: 4 warps/CTA → blockDim.x=128
      break;
    }
    if (N >= 32 && N <= 64) {
      genMatrixVectorMultiply_ImmRowPerLane(
          B, cfg, qLSB, matData, /*svBase=*/args.pSvArg, scalarTy);
      // callee sets cast.kstyle and cast.warps
    } else {
      genMatrixVectorMultiply_SharedTiled(
          B, cfg, matrix, qLSB, matData, /*svBase=*/args.pSvArg, scalarTy);
      func->addFnAttr("cast.kstyle", "imm-shared");
    }
    break;
  }
  case CUDAMatrixLoadMode::LoadInDefaultMemSpace: {
    // Matrix provided at runtime in global memory (AS=1)
    Value* matBasePtr = B.CreateAddrSpaceCast(
        args.pMatArg, PointerType::get(scalarTy, /*AS=*/1), "matGlobalPtr");

    const unsigned N = 1u << k;
    const unsigned TILE =
        std::min((cfg.precision == Precision::F64) ? 128u : 256u, N);

    if (k < cfg.enableTilingGateSize) {
      // Non-tiled pointer kernel computes one combo per CTA -> tilesPerGate=1
      Value* svComboBase = comboBasePtr(/*tilesPerGate=*/1, qLSB);
      genMatrixVectorMultiplyFromPointer(
          B, cfg, matrix, qLSB, matBasePtr, svComboBase, scalarTy);
    } else {
      // Shared-tiled pointer kernel; there are tilesPerGate row tiles per gate.
      const unsigned tilesPerGate = (N + TILE - 1u) / TILE;
      Value* svComboBase = comboBasePtr(tilesPerGate, qLSB);
      genMatrixVectorMultiplyFromPointer_SharedTiled(
          B, cfg, matrix, qLSB, matBasePtr, svComboBase, scalarTy);
      if (!func->hasFnAttribute("cast.tile"))
        func->addFnAttr("cast.tile", std::to_string(TILE));
    }
    break;
  }
  case CUDAMatrixLoadMode::LoadInConstMemSpace: {
    // Build/lookup constant-memory global (AS=4) and initialize with matrix
    // values
    auto* arrTy = ArrayType::get(scalarTy, 2u * KK);
    auto* gConstMat = getOrCreateConstMatGlobal(
        *llvmContextModulePair.llvmModule, arrTy, "gConstMatShared");

    std::vector<Constant*> constElems;
    constElems.reserve(2u * KK);
    for (unsigned r = 0; r < K; ++r) {
      for (unsigned c = 0; c < K; ++c) {
        auto cplx = matrix.rc(r, c);
        constElems.push_back(ConstantFP::get(scalarTy, cplx.real()));
        constElems.push_back(ConstantFP::get(scalarTy, cplx.imag()));
      }
    }
    gConstMat->setInitializer(ConstantArray::get(arrTy, constElems));

    // Const path is non-persistent wrt combos -> one CTA per combo
    Value* svComboBase = comboBasePtr(/*tilesPerGate=*/1, qLSB);
    genMatrixVectorMultiplyFromConst(B,
                                     cfg,
                                     matrix,
                                     qLSB,
                                     gConstMat,
                                     svComboBase,
                                     args.pMatArg,
                                     scalarTy,
                                     matData);
    func->addFnAttr("cast.kstyle", "const-small");
    break;
  }
  }

  B.CreateRetVoid();
  return func;
}

cast::KernelInfoCompiled
CUDAKernelManager::getOrBuildKernel_(const CUDAKernelGenConfig& baseCfg,
                                     const ComplexSquareMatrix& M,
                                     llvm::ArrayRef<int> qubits,
                                     bool assumeContiguous,
                                     const std::string& nameHint) {
  const unsigned k = qubits.size();
  const bool imm =
      (baseCfg.matrixLoadMode == CUDAMatrixLoadMode::UseMatImmValues);
  const uint64_t matHash =
      imm ? hashMatrixImm(M, 1u << k, baseCfg.precision == Precision::F32)
          : 0ull;

  KernelKey key{
      k, baseCfg.precision, baseCfg.matrixLoadMode, assumeContiguous, matHash};
  if (auto it = kernelCache_.find(key); it != kernelCache_.end())
    return it->second;

  // Build
  CUDAKernelGenConfig cfg = baseCfg;
  cfg.assumeContiguousTargets = assumeContiguous;

  std::vector<int> qArg(qubits.begin(), qubits.end());
  if (assumeContiguous) {
    qArg.resize(k);
    std::iota(qArg.begin(), qArg.end(), 0); // {0..k-1}
  }

  // Distinguish names so both can live in the module
  std::string finalName = nameHint + (assumeContiguous ? "_lsb" : "_gen");
  llvm::Function* fn = gen_(cfg, M, qArg, finalName);

  KernelInfoCompiled out{fn, fn->getName().str()};
  kernelCache_.emplace(key, out);
  return out;
}

MaybeError<CUDAKernelManager::KernelInfoPtr>
CUDAKernelManager::genCUDAGate_(const CUDAKernelGenConfig& config,
                                ConstQuantumGatePtr gate,
                                const std::string& funcName) {
  auto* stdQuGate = llvm::dyn_cast<const StandardQuantumGate>(gate.get());
  llvm::Function* func = nullptr;

  if (stdQuGate != nullptr && stdQuGate->noiseChannel() == nullptr) {
    // a normal gate, no noise channel
    const auto scalarGM = stdQuGate->getScalarGM();
    assert(scalarGM != nullptr && "Only supporting scalar GM for now");
    func = gen_(config, scalarGM->matrix(), stdQuGate->qubits(), funcName);
  } else {
    // super op gates are treated as normal gates with twice the number of
    // qubits
    auto superopGate = gate->getSuperopGate();
    assert(superopGate != nullptr && "Superop gate should not be null");
    const auto scalarGM = superopGate->getMatrix();
    assert(scalarGM != nullptr && "superop gate matrix should not be null");

    auto qubits = superopGate->qubits();
    auto nQubits = superopGate->nQubits();

    // for (const auto& q : qubits)
    //     qubits.push_back(q + nQubits);

    auto qs = superopGate->qubits();
    auto nQ = superopGate->nQubits();
    std::vector<int> both = qs;
    for (auto q : qs)
      both.push_back(q + nQ);

    func = gen_(config, scalarGM->matrix(), both, funcName);
    // func = gen_(config, scalarGM->matrix(), qubits, funcName);
  }

  if (func == nullptr) {
    std::ostringstream oss;
    oss << "Failed to generate kernel for gate " << (void*)(gate.get())
        << " with name " << funcName;
    return cast::makeError<KernelInfoPtr>(oss.str());
  }

  CUDAKernelInfo::CUDATuple cuTuple;
  auto ki = std::make_unique<CUDAKernelInfo>(CUDAKernelInfo::PTXStringType(),
                                             config.precision,
                                             func->getName().str(),
                                             gate,
                                             cuTuple,
                                             gate->opCount(config.zeroTol));

  // if (func->hasFnAttribute("cast.kstyle")) {
  //     auto style = func->getFnAttribute("cast.kstyle").getValueAsString();
  //     ki->oneThreadPerBlock = (style == "imm-inline" || style ==
  //     "const-small");
  // } else {
  //     ki->oneThreadPerBlock = false;
  // }

  if (func->hasFnAttribute("cast.kstyle")) {
    auto style = func->getFnAttribute("cast.kstyle").getValueAsString();
    if (style == "imm-inline" || style == "const-small") {
      ki->oneThreadPerBlock = true;
    } else if (style == "imm-inline-warp" || style == "imm-shared-warp") {
      ki->oneThreadPerBlock = false;
      if (func->hasFnAttribute("cast.warps")) {
        auto s = func->getFnAttribute("cast.warps").getValueAsString().str();
        ki->warpsPerCTA = std::max(1u, (unsigned)std::stoi(s));
      } else {
        ki->warpsPerCTA = 4;
      }
    } else {
      ki->oneThreadPerBlock = false;
    }
  } else {
    ki->oneThreadPerBlock = false;
  }

  if (func->hasFnAttribute("cast.tile")) {
    auto t = func->getFnAttribute("cast.tile").getValueAsString();
    ki->tileSize = static_cast<unsigned>(std::stoul(std::string(t)));
  } else {
    ki->tileSize = 0;
  }

  return ki;
}

MaybeError<void>
CUDAKernelManager::genStandaloneGate(const CUDAKernelGenConfig& config,
                                     ConstQuantumGatePtr gate,
                                     const std::string& _funcName) {
  std::string funcName(_funcName);
  if (funcName.empty())
    funcName = "kernel_" + std::to_string(standaloneKernels_.size());

  // check for name conflicts
  for (const auto& kernel : standaloneKernels_) {
    if (kernel->llvmFuncName == funcName) {
      return cast::makeError<void>("Kernel with name '" + funcName +
                                   "' already exists.");
    }
  }

  auto result = genCUDAGate_(config, gate, funcName);
  if (!result) {
    return cast::makeError<void>("Err: " + result.takeError());
  }

  standaloneKernels_.emplace_back(result.takeValue());
  return {}; // success
}

MaybeError<void>
CUDAKernelManager::genGraphGates(const CUDAKernelGenConfig& config,
                                 const ir::CircuitGraphNode& graph,
                                 const std::string& graphName) {
  assert(graph.checkConsistency());

  if (graphKernels_.contains(graphName)) {
    std::ostringstream oss;
    oss << "Graph with name '" << graphName
        << "' already has generated kernels. Please use a different name.";
    return cast::makeError<void>(oss.str());
  }

  auto mangledGraphName = internal::mangleGraphName(graphName);
  auto allGates = graph.getAllGatesShared();
  int order = 0;

  std::vector<KernelInfoPtr> kernels;
  kernels.reserve(2 * allGates.size());

  for (const auto& gate : allGates) {
    // auto name = mangledGraphName + "_" + std::to_string(order++) + "_" +
    //             std::to_string(graph.gateId(gate));
    // auto result = genCUDAGate_(config, gate, name);
    // if (!result) {
    //     std::ostringstream oss;
    //     oss << "Failed to generate kernel for gate " << (void*)(gate.get())
    //         << ": " << result.takeError() << "\n";
    //     return cast::makeError<void>(oss.str());
    // }
    // kernels.emplace_back(result.takeValue());

    // auto vr = genCUDAGateVariants_(config, gate, name);
    // if (!vr) {
    //     std::ostringstream oss;
    //     oss << "Failed to generate variants for gate " << (void*)(gate.get())
    //         << ": " << vr.takeError() << "\n";
    //     return cast::makeError<void>(oss.str());
    // }
    // auto pair = vr.takeValue();
    // kernels.emplace_back(std::move(pair.lsb)); // <base>_lsb
    // kernels.emplace_back(std::move(pair.gen)); // <base>_gen

    const std::string base = mangledGraphName + "_" + std::to_string(order++) +
                             "_" + std::to_string(graph.gateId(gate));

    auto vr = genCUDAGateVariants_(config, gate, base);
    if (!vr) {
      std::ostringstream oss;
      oss << "Failed to generate variants for gate " << (void*)gate.get()
          << ": " << vr.takeError() << "\n";
      return cast::makeError<void>(oss.str());
    }
    auto pair = vr.takeValue();
    kernels.emplace_back(std::move(pair.lsb)); // <base>_lsb
    kernels.emplace_back(std::move(pair.gen)); // <base>_gen
  }

  // Store the generated kernels in the map
  graphKernels_[graphName] = std::move(kernels);
  return {}; // success
}

void CUDAKernelManager::rebuildOrderedKernelIndex_() {
  orderedKernels_.clear();
  orderedKernels_.reserve(llvmContextModulePairs.size());

  auto findByName = [&](const std::string& name) -> CUDAKernelInfo* {
    for (auto& k : standaloneKernels_) {
      if (k->llvmFuncName == name)
        return k.get();
    }
    for (auto& kv : graphKernels_) {
      for (auto& k : kv.second) {
        if (k->llvmFuncName == name)
          return k.get();
      }
    }
    return nullptr;
  };

  for (size_t i = 0; i < llvmContextModulePairs.size(); ++i) {
    llvm::Module* M = llvmContextModulePairs[i].llvmModule.get();
    std::string modName = M->getName().str(); // e.g. "<funcName>Module"

    std::string funcName = modName;
    if (funcName.size() >= 6 &&
        funcName.compare(funcName.size() - 6, 6, "Module") == 0) {
      funcName.resize(funcName.size() - 6);
    }

    CUDAKernelInfo* ki = findByName(funcName);

    // As a fallback, scan defined functions in the module and match by name.
    if (!ki) {
      for (auto& F : M->functions()) {
        if (!F.isDeclaration()) {
          ki = findByName(F.getName().str());
          if (ki)
            break;
        }
      }
    }

    if (!ki) {
      std::ostringstream oss;
      oss << "Internal error: cannot match module '" << modName
          << "' to any CUDAKernelInfo.";
      throw std::runtime_error(oss.str());
    }

    orderedKernels_.push_back(ki);
  }
}

#undef DEBUG_TYPE

#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/Core/KernelGenInternal.h"

#include "utils/Formats.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
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
  if (config.zeroTol <= 0.0 && config.oneTol <= 0.0) {
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

Function* getFunctionDeclarationCUDA(IRBuilder<>& B,
                                     Module& M,
                                     const std::string& funcName,
                                     const CUDAKernelGenConfig& config,
                                     IRArgsCUDA& args) {
  const bool needsMatArg =
      (config.matrixLoadMode == CUDAMatrixLoadMode::LoadInDefaultMemSpace);

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

  // mark as kernel
  auto* mdString = MDString::get(M.getContext(), "kernel");
  auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
  auto* md = MDNode::get(M.getContext(),
                         {ValueAsMetadata::get(func), mdString, mdOne});
  M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md);
  return func;
}

Value* getGlobalTid(IRBuilder<>& B) {
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

// Get block index x (blockIdx.x)
Value* getBid(IRBuilder<>& B) {
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

static void
genMatrixVectorMultiply_InlineImm(IRBuilder<>& B,
                                  const CUDAKernelGenConfig& config,
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
    uint64_t delta = 0;
    for (unsigned b = 0; b < k; ++b)
      if (i & (1u << b))
        delta |= (1ull << qubits[b]);
    off2 = 2ull * delta;
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

static void genMatVecMul_Imm(IRBuilder<>& B,
                             const CUDAKernelGenConfig& config,
                             const QuantumGate::TargetQubitsType& qubits,
                             const std::vector<IRMatDataCUDA>& matData,
                             Value* svRoot,
                             Type* scalarTy) {
  Function* func = B.GetInsertBlock()->getParent();
  LLVMContext& C = B.getContext();

  // p.combos
  Argument* combosV = nullptr;
  for (auto& A : func->args()) {
    if (A.getName() == "p.combos") {
      combosV = &A;
      break;
    }
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

  // start = (cta * warpsPerCTA + warpIn) * 32 + lane
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
    Value* base2 = B.CreateShl(buildOffset(B, idx64, qubits), 1);
    Value* svComboBase = B.CreateGEP(scalarTy, svRoot, base2, "sv.combo.base");

    // Reuse straight-line multiply on this combo:
    genMatrixVectorMultiply_InlineImm(
        B, config, qubits, matData, svComboBase, scalarTy);

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

Function* CUDAKernelManager::gen_(const CUDAKernelGenConfig& config,
                                  const ComplexSquareMatrix& matrix,
                                  const QuantumGate::TargetQubitsType& qubits,
                                  const std::string& funcName) {
  const unsigned k = qubits.size();
  const unsigned K = 1u << k;
  const unsigned KK = K * K;

  auto& llvmContextModulePair =
      createNewLLVMContextModulePair(funcName + "Module");
  IRBuilder<> B(*llvmContextModulePair.llvmContext);

  assert(config.precision != Precision::Unknown);
  Type* scalarTy =
      (config.precision == Precision::F32) ? B.getFloatTy() : B.getDoubleTy();

  IRArgsCUDA args;
  auto* func = getFunctionDeclarationCUDA(
      B, *llvmContextModulePair.llvmModule, funcName, config, args);

  auto* entryBB =
      BasicBlock::Create(*llvmContextModulePair.llvmContext, "entry", func);
  B.SetInsertPoint(entryBB);

  // Helper: compute the base pointer (into the full statevector) for this CTA's
  // combo. We map comboSlot := ctaid.x / tilesPerGate, where tilesPerGate is
  // the number of row-tiles per gate for the chosen kernel style (or 1 when
  // there is no row tiling).
  const auto computeComboBasePtr =
      [&](const QuantumGate::TargetQubitsType& qubits) -> Value* {
    // ctaid.x
    auto* cta = B.CreateIntrinsic(
        B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
    auto* idx64 = B.CreateZExt(cta, B.getInt64Ty());
    Value* base2 = nullptr;

    // Generic mapping: pdep-style using target-bit mask
    // qsForOffset must be ascending
    Value* off = buildOffset(B, idx64, qubits);
    base2 = B.CreateShl(off, 1);
    return B.CreateGEP(scalarTy, args.pSvArg, base2, "sv.combo.base");
  };

  // For kernels that need compile-time matrix analysis
  auto matData = getMatDataCUDA(B, config, matrix, k);

  switch (config.matrixLoadMode) {
  case CUDAMatrixLoadMode::UseMatImmValues: {
    // lane-per-combo persistent inline
    genMatVecMul_Imm(B, config, qubits, matData, args.pSvArg, scalarTy);
    break;
  }
  case CUDAMatrixLoadMode::LoadInDefaultMemSpace: {
    // Matrix provided at runtime in global memory (AS=1)
    Value* matBasePtr = B.CreateAddrSpaceCast(
        args.pMatArg, PointerType::get(scalarTy, /*AS=*/1), "matGlobalPtr");

    Value* svComboBase = computeComboBasePtr(qubits);
    genMatrixVectorMultiplyFromPointer(
        B, config, matrix, qubits, matBasePtr, svComboBase, scalarTy);

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
    Value* svComboBase = computeComboBasePtr(qubits);
    genMatrixVectorMultiplyFromConst(B,
                                     config,
                                     matrix,
                                     qubits,
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
  // func->print(errs());
  return func;
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
    for (const auto& q : qubits)
      qubits.push_back(q + nQubits);

    func = gen_(config, scalarGM->matrix(), qubits, funcName);
  }

  if (func == nullptr) {
    std::ostringstream oss;
    oss << "Failed to generate kernel for gate " << (void*)(gate.get())
        << " with name " << funcName;
    return cast::makeError<KernelInfoPtr>(oss.str());
  }

  // std::string ptxString;
  // std::vector<uint8_t> cubinData;
  // ConstQuantumGatePtr gate;
  // Precision precision;
  // llvm::LLVMContext* llvmContext;
  // llvm::Module* llvmModule;
  // std::string llvmFuncName;
  auto ki = std::make_unique<CUDAKernelInfo>(std::string(),
                                             std::vector<uint8_t>(),
                                             gate,
                                             config.precision,
                                             &func->getContext(),
                                             func->getParent(),
                                             func->getName().str());

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
  kernels.reserve(allGates.size());

  for (const auto& gate : allGates) {
    auto name = mangledGraphName + "_" + std::to_string(order++) + "_" +
                std::to_string(graph.gateId(gate));
    auto result = genCUDAGate_(config, gate, name);
    if (!result) {
      std::ostringstream oss;
      oss << "Failed to generate kernel for gate " << (void*)(gate.get())
          << ": " << result.takeError() << "\n";
      return cast::makeError<void>(oss.str());
    }
    kernels.emplace_back(result.takeValue());
  }

  // Store the generated kernels in the map
  graphKernels_[graphName] = std::move(kernels);
  return {}; // success
}

#undef DEBUG_TYPE

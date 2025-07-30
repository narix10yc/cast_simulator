#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/Core/KernelGenInternal.h"

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Verifier.h"

#include "utils/Formats.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

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
    } // switch reKind

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
    } // switch imKind
  }

  return data;
}

// Function* getFunctionDeclarationCUDA(
//     IRBuilder<>& B, Module& M, const std::string& funcName,
//     const CUDAKernelGenConfig& config, IRArgsCUDA& args) {
//   /*
//       Address space:
//       0: Generic;
//       1: Global;
//       2: Internal Use;
//       3: Shared;
//       4: Constant (often 64KB)
//       5: Local;

//       For a reference see https://llvm.org/docs/NVPTXUsage.html#id32
//   */

//   FunctionType *fty = (config.matrixLoadMode ==
//   CUDAKernelGenConfig::LoadInConstMemSpace)
//                       ? FunctionType::get(B.getVoidTy(), { B.getPtrTy() },
//                       false) : FunctionType::get(B.getVoidTy(), {
//                       B.getPtrTy(),  B.getPtrTy() }, false);
//   auto* func = Function::Create(
//     fty,
//     Function::ExternalLinkage,
//     funcName,
//     M
//   );
//   if (funcName.empty())
//     func->setName("ptx_kernel_");
//   else
//     func->setName(funcName);
//
//   args.pSvArg = func->getArg(0);
//   args.pSvArg->setName("p.sv");
//   if (config.matrixLoadMode != CUDAKernelGenConfig::LoadInConstMemSpace) {
//     args.pMatArg = func->getArg(1);
//     args.pMatArg->setName("p.mat");
//   } else {
//     args.pMatArg = nullptr;
//   }

//   // mark this function as a kernel
//   auto* mdString = MDString::get(M.getContext(), "kernel");
//   auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
//   auto* kernelMetadata = MDNode::get(
//     M.getContext(),
//     { ValueAsMetadata::get(func), mdString, mdOne });
//   M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(kernelMetadata);

//   return func;
// }

Function* getFunctionDeclarationCUDA(IRBuilder<>& B,
                                     Module& M,
                                     const std::string& funcName,
                                     const CUDAKernelGenConfig& config,
                                     IRArgsCUDA& args) {
  // Always include pSvArg and pMatArg
  auto* fty =
      FunctionType::get(B.getVoidTy(), {B.getPtrTy(), B.getPtrTy()}, false);
  auto* func = Function::Create(fty, Function::ExternalLinkage, funcName, M);
  if (funcName.empty())
    func->setName("ptx_kernel_");
  else
    func->setName(funcName);

  args.pSvArg = func->getArg(0);
  args.pSvArg->setName("p.sv");
  args.pMatArg = func->getArg(1);
  args.pMatArg->setName("p.mat");

  // Mark as kernel
  auto* mdString = MDString::get(M.getContext(), "kernel");
  auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
  auto* kernelMetadata = MDNode::get(
      M.getContext(), {ValueAsMetadata::get(func), mdString, mdOne});
  M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(kernelMetadata);

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
 *  Create a global array [2*K*K x scalarTy] to store the real/imag parts
 *    from matData. That way, we can do a run-time IR loop to read them,
 *    instead of unrolling a for-loop in C++.
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
    // If md.reVal is a ConstantFP, we can cast directly:
    ConstantFP* cRe = dyn_cast_or_null<ConstantFP>(md.reVal);
    ConstantFP* cIm = dyn_cast_or_null<ConstantFP>(md.imVal);

    double fallbackRe = 0.0, fallbackIm = 0.0;
    // If not SK_ImmValue, fallback to 0 or do more logic
    if (md.reKind == SK_ImmValue && cRe) {
      constVals.push_back(cRe);
    } else if (md.reKind == SK_One) {
      constVals.push_back(ConstantFP::get(scalarTy, 1.0));
    } else if (md.reKind == SK_MinusOne) {
      constVals.push_back(ConstantFP::get(scalarTy, -1.0));
    } else if (md.reKind == SK_Zero) {
      constVals.push_back(ConstantFP::get(scalarTy, 0.0));
    } else {
      // e.g. SK_General => fallback or store 0.0
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
      // fallback
      constVals.push_back(ConstantFP::get(scalarTy, fallbackIm));
    }
  }

  auto* arrInit = ConstantArray::get(arrTy, constVals);

  // Create a private global
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

    // Inner loop c in [0..K)
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

    // innerBody: read M[r,c] from the global array, read oldAmp[c], do
    // partial sums
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
      auto arrTy2 = llvm::cast<ArrayType>(gMatImmediate->getValueType());
      // rePtr => gMatImmediate[0, base2], imPtr => gMatImmediate[0, base2+1]
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

      // imA  += matRe*oldIm + matIm*oldRe
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

GlobalVariable* createGlobalMatrixArray_SharedTiledImm(
    Module& M,
    Type* scalarTy,
    unsigned N,
    const std::vector<IRMatDataCUDA>& matData,
    const std::string& globalName) {
  // The array type: [2*N*N x scalarTy].
  unsigned totalElems = 2 * N * N;
  ArrayType* arrTy = ArrayType::get(scalarTy, totalElems);

  // --------------------------------------- For Debug
  // --------------------------------------------- #include <iostream>
  // std::cerr
  // << "matData for " << globalName << ":\n"; for (unsigned i = 0; i < N*N;
  // i++) {
  //     const auto &md = matData[i];
  //     ConstantFP *cRe = dyn_cast_or_null<ConstantFP>(md.reVal);
  //     ConstantFP *cIm = dyn_cast_or_null<ConstantFP>(md.imVal);
  //     double reVal = (md.reKind == SK_ImmValue && cRe) ?
  //     cRe->getValueAPF().convertToDouble() :
  //                    (md.reKind == SK_One) ? 1.0 :
  //                    (md.reKind == SK_MinusOne) ? -1.0 : 0.0;
  //     double imVal = (md.imKind == SK_ImmValue && cIm) ?
  //     cIm->getValueAPF().convertToDouble() :
  //                    (md.imKind == SK_One) ? 1.0 :
  //                    (md.imKind == SK_MinusOne) ? -1.0 : 0.0;
  //     std::cerr << "matData[" << i << "] = (" << reVal << ", " << imVal <<
  //     ")\n";
  // }
  // -----------------------------------------------------------------------------------------------

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

  Constant* arrInit = ConstantArray::get(arrTy, initVals);

  auto* gVar =
      new GlobalVariable(M,
                         arrTy,
                         /*isConstant=*/true,
                         GlobalValue::PrivateLinkage, // or InternalLinkage
                         arrInit,
                         globalName);
  gVar->setAlignment(MaybeAlign(8));
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
  counter:   xxxhgfedcba
  pbex mask: 11111001011
  idxStart:  hgfed00c0ba (in unit of <2 x scalarTy>)

  hgfed00c0ba = (xxxhgfedcba & 00000000011) << 0
              + (xxxhgfedcba & 00000000100) << 1
              + (xxxhgfedcba & 11111111000) << 3

  We build this segment by segment. For [2, 4, 5], there are 3 segments:
    [0, 2),      [3, 4),      [5, ),
  corresponding to masks
    00000000011, 00000000100, 11111111000
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
      mask |= (1 << counterQ++);
      continue;
    }
    ++qIdx;
    if (mask == 0)
      continue;
    tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
    tmpCounterV = B.CreateShl(tmpCounterV, (qIdx - 1), "tmpCounter");
    offset = B.CreateAdd(offset, tmpCounterV, "tmpIdx");
    LLVM_DEBUG(std::cerr << "  (globalThreadIdx & " << utils::fmt_0b(mask, 32) << ") << "
              << (qIdx - 1) << "\n";);
    mask = 0ULL;
  }
  mask = ~((1ULL << (highestQ - k + 1)) - 1);
  LLVM_DEBUG(std::cerr << "  (globalThreadIdx & " << utils::fmt_0b(mask, 32) << ") << "
            << (k) << "\n";);

  tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
  tmpCounterV = B.CreateShl(tmpCounterV, k, "tmpCounter");
  offset = B.CreateAdd(offset, tmpCounterV, "offset");
  return offset;
}

void genMatrixVectorMultiply_SharedTiled(
    IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const ComplexSquareMatrix& matrix,
    const QuantumGate::TargetQubitsType& qubits,
    const std::vector<IRMatDataCUDA>& matData,
    Value* svPtrV,
    Type* scalarTy) {
  using namespace llvm;

  const unsigned k = qubits.size();
  const unsigned N = 1u << k;
  // const unsigned TILE  = 256;
  const unsigned TILE = std::min(256u, N);

  /*────────── Module‑level allocations ──────────*/
  Function* func = B.GetInsertBlock()->getParent();
  Module& M = *func->getParent();

  /* constant gate matrix in global memory */
  GlobalVariable* gMatImm = createGlobalMatrixArray_SharedTiledImm(
      M, scalarTy, N, matData, "gMatImmSharedTiled");

  /* shared‑memory tile for vector (addr‑space 3) */
  ArrayType* smVecTy = ArrayType::get(scalarTy, 2 * TILE);
  auto* smVecGV = new GlobalVariable(M,
                                     smVecTy,
                                     false,
                                     GlobalValue::ExternalLinkage,
                                     UndefValue::get(smVecTy),
                                     "tileX",
                                     nullptr,
                                     GlobalValue::NotThreadLocal,
                                     /*AS=*/3);
  Value* smVecBase =
      B.CreateGEP(smVecTy, smVecGV, {B.getInt32(0), B.getInt32(0)});

  /*────────── Thread & grid indices ──────────*/
  Value* tid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  Value* bid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});

  const unsigned tilesPerGate = (N + TILE - 1) / TILE;
  Value* tilesPerGateV = B.getInt32(tilesPerGate);

  /* rowTileIdx = bid32 % tilesPerGate */
  Value* rowTileIdx32 = B.CreateURem(bid32, tilesPerGateV, "rowTileIdx");
  Value* rowBase64 = B.CreateMul(
      B.CreateZExt(rowTileIdx32, B.getInt64Ty()), B.getInt64(TILE), "rowBase");
  Value* row64 = B.CreateAdd(rowBase64,
                             B.CreateZExt(tid32, B.getInt64Ty()),
                             "row64"); // <-- fixed

  /* guard : if row64 >= K skip the thread */
  BasicBlock* rowOK = BasicBlock::Create(B.getContext(), "row.ok", func);
  BasicBlock* rowEnd = BasicBlock::Create(B.getContext(), "row.end", func);
  B.CreateCondBr(B.CreateICmpULT(row64, B.getInt64(N)), rowOK, rowEnd);
  B.SetInsertPoint(rowOK);

  // Local accumulators in registers
  Value* accRe = ConstantFP::get(scalarTy, 0.0);
  Value* accIm = ConstantFP::get(scalarTy, 0.0);

  // Loop over column tiles
  const unsigned nTiles = (N + TILE - 1) / TILE;

  auto* tileChk = BasicBlock::Create(B.getContext(), "tile.chk", func);
  auto* tileBody = BasicBlock::Create(B.getContext(), "tile.body", func);
  auto* tileInc = BasicBlock::Create(B.getContext(), "tile.inc", func);
  auto* tileDone = BasicBlock::Create(B.getContext(), "tile.done", func);

  B.CreateBr(tileChk);
  B.SetInsertPoint(tileChk);

  PHINode* col0Phi = B.CreatePHI(B.getInt64Ty(), 2, "col0");
  col0Phi->addIncoming(B.getInt64(0), rowOK);

  B.CreateCondBr(B.CreateICmpULT(col0Phi, B.getInt64(N)), tileBody, tileDone);

  /*────────────── tileBody ─────────────*/
  B.SetInsertPoint(tileBody);

  /* load TILE amplitudes of X into shared memory */
  {
    Value* tid64 = B.CreateZExt(tid32, B.getInt64Ty());
    Value* colIdx = B.CreateAdd(col0Phi, tid64);
    Value* inRange = B.CreateICmpULT(colIdx, B.getInt64(N));

    BasicBlock* ldYes = BasicBlock::Create(B.getContext(), "ld.y", func);
    BasicBlock* ldNo = BasicBlock::Create(B.getContext(), "ld.n", func);
    BasicBlock* ldEnd = BasicBlock::Create(B.getContext(), "ld.end", func);

    B.CreateCondBr(inRange, ldYes, ldNo);

    B.SetInsertPoint(ldYes);
    {
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

    B.SetInsertPoint(ldNo);
    {
      Constant* zero = ConstantFP::get(scalarTy, 0.0);
      Value* idxS = B.CreateMul(tid64, B.getInt64(2));
      B.CreateStore(zero, B.CreateGEP(scalarTy, smVecBase, idxS));
      B.CreateStore(
          zero,
          B.CreateGEP(scalarTy, smVecBase, B.CreateAdd(idxS, B.getInt64(1))));
      B.CreateBr(ldEnd);
    }
    B.SetInsertPoint(ldEnd);
  }

  // __syncthreads()
  B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::nvvm_barrier0));

  /* dot‑product over this tile */
  for (unsigned t = 0; t < TILE; ++t) {
    Value* col = B.CreateAdd(col0Phi, B.getInt64(t));
    Value* inRange = B.CreateICmpULT(col, B.getInt64(N));

    auto* dotYes = BasicBlock::Create(B.getContext(), "dot.y", func);
    auto* dotNo = BasicBlock::Create(B.getContext(), "dot.n", func);
    auto* dotEnd = BasicBlock::Create(B.getContext(), "dot.end", func);

    B.CreateCondBr(inRange, dotYes, dotNo);

    B.SetInsertPoint(dotYes);
    {
      /* load matrix element from global memory */
      Value* lin = B.CreateAdd(B.CreateMul(row64, B.getInt64(N)), col);
      Value* idx2 = B.CreateMul(lin, B.getInt64(2));
      Value* mRePtr = B.CreateGEP(scalarTy, gMatImm, idx2);
      Value* mImPtr =
          B.CreateGEP(scalarTy, gMatImm, B.CreateAdd(idx2, B.getInt64(1)));
      Value* mRe = B.CreateLoad(scalarTy, mRePtr);
      Value* mIm = B.CreateLoad(scalarTy, mImPtr);

      /* load vector element from shared memory */
      Value* offX = B.CreateMul(B.getInt64(t), B.getInt64(2));
      Value* xRe =
          B.CreateLoad(scalarTy, B.CreateGEP(scalarTy, smVecBase, offX));
      Value* xIm = B.CreateLoad(
          scalarTy,
          B.CreateGEP(scalarTy, smVecBase, B.CreateAdd(offX, B.getInt64(1))));

      /* complex multiplication and accumulation */
      Value* pRe = B.CreateFSub(B.CreateFMul(mRe, xRe), B.CreateFMul(mIm, xIm));
      Value* pIm = B.CreateFAdd(B.CreateFMul(mRe, xIm), B.CreateFMul(mIm, xRe));
      accRe = B.CreateFAdd(accRe, pRe);
      accIm = B.CreateFAdd(accIm, pIm);
      B.CreateBr(dotEnd);
    }

    B.SetInsertPoint(dotNo);
    {
      /* skip contribution if out of range */
      B.CreateBr(dotEnd);
    }

    B.SetInsertPoint(dotEnd);
  }

  // __syncthreads()
  B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::nvvm_barrier0));

  /* advance to next tile */
  B.CreateBr(tileInc);
  B.SetInsertPoint(tileInc);
  Value* nextCol0 = B.CreateAdd(col0Phi, B.getInt64(TILE));
  col0Phi->addIncoming(nextCol0, tileInc);
  B.CreateBr(tileChk);

  /*────────────── tileDone ──────────────*/
  B.SetInsertPoint(tileDone);

  /* store the result amplitude */
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

void genMatrixVectorMultiplyFromPointer_SharedTiled(
    IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const ComplexSquareMatrix& gateMat,
    const QuantumGate::TargetQubitsType& qubits,
    Value* matBasePtr, // AS 1  – run‑time matrix
    Value* svPtrV,     // AS 0  – state‑vector
    Type* scalarTy) {
  using namespace llvm;

  /*────────── Parameters & types ──────────*/
  const unsigned k = qubits.size();
  const unsigned N = 1u << k;
  const unsigned TILE = std::min(256u, N); // matches constant path

  Module& M = *B.GetInsertBlock()->getModule();
  LLVMContext& CTX = M.getContext();

  /*────────── Shared memory tile for |x〉 ──────────*/
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

  /*────────── Thread & grid indices ──────────*/
  Value* tid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  Value* bid32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});

  const unsigned tilesPerGate = (N + TILE - 1) / TILE;
  Value* tilesPerGateV = B.getInt32(tilesPerGate);

  /* packed logical row index (Tile‑aware mapping) */
  Value* rowTileIdx32 = B.CreateURem(bid32, tilesPerGateV, "rowTileIdx");
  Value* rowBase64 = B.CreateMul(
      B.CreateZExt(rowTileIdx32, B.getInt64Ty()), B.getInt64(TILE), "rowBase");
  Value* row64 =
      B.CreateAdd(rowBase64, B.CreateZExt(tid32, B.getInt64Ty()), "row64");

  /*────────── Guard against over‑run ──────────*/
  Function* func = B.GetInsertBlock()->getParent();
  BasicBlock* rowOK = BasicBlock::Create(CTX, "row.ok", func);
  BasicBlock* rowEnd = BasicBlock::Create(CTX, "row.end", func);
  B.CreateCondBr(B.CreateICmpULT(row64, B.getInt64(N)), rowOK, rowEnd);
  B.SetInsertPoint(rowOK);

  /*────────── Local accumulators – registers ──────────*/
  Value* accRe = ConstantFP::get(scalarTy, 0.0);
  Value* accIm = ConstantFP::get(scalarTy, 0.0);

  /* =========  Outer loop over column tiles  ========= */
  BasicBlock* tileChk = BasicBlock::Create(CTX, "tile.chk", func);
  BasicBlock* tileBody = BasicBlock::Create(CTX, "tile.body", func);
  BasicBlock* tileInc = BasicBlock::Create(CTX, "tile.inc", func);
  BasicBlock* tileDone = BasicBlock::Create(CTX, "tile.done", func);

  B.CreateBr(tileChk);
  B.SetInsertPoint(tileChk);

  PHINode* col0Phi = B.CreatePHI(B.getInt64Ty(), /*numPreds=*/2, "col0");
  col0Phi->addIncoming(B.getInt64(0), rowOK);

  B.CreateCondBr(B.CreateICmpULT(col0Phi, B.getInt64(N)), tileBody, tileDone);

  /*────────────── tileBody ─────────────*/
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

    /* in‑range */
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
    /* out‑of‑range => 0 */
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
  B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::nvvm_barrier0));

  /* 3. Dot‑product over this tile */
  for (unsigned t = 0; t < TILE; ++t) {
    Value* col = B.CreateAdd(col0Phi, B.getInt64(t));
    Value* inRng = B.CreateICmpULT(col, B.getInt64(N));

    BasicBlock* yesBB = BasicBlock::Create(CTX, "dot.y", func);
    BasicBlock* noBB = BasicBlock::Create(CTX, "dot.n", func);
    BasicBlock* endBB = BasicBlock::Create(CTX, "dot.end", func);
    B.CreateCondBr(inRng, yesBB, noBB);

    /* --- in‑range --- */
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
    /* --- out‑of‑range --- */
    B.SetInsertPoint(noBB);
    B.CreateBr(endBB);
    B.SetInsertPoint(endBB);
  }

  /* 4. Barrier before next tile iteration */
  B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::nvvm_barrier0));

  /* 5. Advance to next tile */
  B.CreateBr(tileInc);
  B.SetInsertPoint(tileInc);
  Value* nextCol0 = B.CreateAdd(col0Phi, B.getInt64(TILE));
  col0Phi->addIncoming(nextCol0, tileInc);
  B.CreateBr(tileChk);

  /*────────────── after last tile ─────────────*/
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
    Type* scalarTy) {
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

  // Matrix-vector multiplication
  for (unsigned r = 0; r < K; r++) {
    Value* updatedReAmp0 = ConstantFP::get(scalarTy, 0.0);
    Value* updatedReAmp1 = ConstantFP::get(scalarTy, 0.0);
    Value* updatedImAmp = ConstantFP::get(scalarTy, 0.0);

    for (unsigned c = 0; c < K; c++) {
      uint64_t baseIndex = 2ull * (r * K + c);
      auto* idxReal = B.getInt64(baseIndex);
      auto* idxImag = B.getInt64(baseIndex + 1);

      // Use constant memory (AS 4)
      auto* realPtr = B.CreateGEP(
          gConstMat->getValueType(), gConstMat, idxReal, "constElemPtrRe");
      auto* imagPtr = B.CreateGEP(
          gConstMat->getValueType(), gConstMat, idxImag, "constElemPtrIm");

      auto* matRe = B.CreateLoad(scalarTy, realPtr, "matRe");
      auto* matIm = B.CreateLoad(scalarTy, imagPtr, "matIm");
      auto* oldRe = reAmps[c];
      auto* oldIm = imAmps[c];

      updatedReAmp0 = B.CreateFAdd(updatedReAmp0, B.CreateFMul(matRe, oldRe));
      updatedReAmp1 = B.CreateFAdd(updatedReAmp1, B.CreateFMul(matIm, oldIm));
      updatedImAmp = B.CreateFAdd(
          updatedImAmp,
          B.CreateFAdd(B.CreateFMul(matRe, oldIm), B.CreateFMul(matIm, oldRe)));
    }

    auto* newReAmp = B.CreateFSub(updatedReAmp0, updatedReAmp1);
    auto* newImAmp = updatedImAmp;
    B.CreateStore(newReAmp, reAmpPtrs[r]);
    B.CreateStore(newImAmp, imAmpPtrs[r]);
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
  const unsigned K = 1ULL << k;
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

  Value* svPtrV;
  {
    auto* counterV = (config.matrixLoadMode == CUDAMatrixLoadMode::LoadInConstMemSpace) ? getGlobalTidCUDA(B) : helper_getBlockIdx(B);
    auto* offset = buildOffset(B, counterV, qubits);
    auto* idxStartV = B.CreateShl(offset, 1, "twice.offset"); // x2 because (re,im)
    svPtrV = B.CreateGEP(scalarTy, args.pSvArg, idxStartV, "sv.ptr");
  }
  entryBB->dump();

  auto matData = getMatDataCUDA(B, config, matrix, k);

  switch (config.matrixLoadMode) {
  case CUDAMatrixLoadMode::UseMatImmValues: {
    // TODO: why number of qubits here
    if (k < config.enableTilingGateSize) {
      genMatrixVectorMultiply(
          B, config, matrix, qubits, matData, svPtrV, scalarTy);
    } else {
      genMatrixVectorMultiply_SharedTiled(
          B, config, matrix, qubits, matData, svPtrV, scalarTy);
    }
    break;
  } // case CUDAMatrixLoadMode::UseMatImmValues
  case CUDAMatrixLoadMode::LoadInDefaultMemSpace: {
    // This path loads matrix from pMatArg (args.pMatArg) in address space 1
    Value* matBasePtr = args.pMatArg;
    // Cast from address space 0 to 1:
    matBasePtr = B.CreateAddrSpaceCast(
        args.pMatArg, PointerType::get(scalarTy, /*AS=*/1), "matGlobalPtr");

    // Generate the IR that loops over the matrix elements
    // (K*K complex elements) and loads from matBasePtr + offset
    if (k < config.enableTilingGateSize) {
      genMatrixVectorMultiplyFromPointer(
          B, config, matrix, qubits, matBasePtr, svPtrV, scalarTy);
    } else {
      genMatrixVectorMultiplyFromPointer_SharedTiled(
          B, config, matrix, qubits, matBasePtr, svPtrV, scalarTy);
    }
    break;
  } // case CUDAMatrixLoadMode::LoadInDefaultMemSpace
  case CUDAMatrixLoadMode::LoadInConstMemSpace: {
    // This path loads the matrix from pointer in address space 4
    // (constant memory space)
    auto* arrTy = ArrayType::get(scalarTy, 2U * KK);
    auto* gConstMat = getOrCreateConstMatGlobal(
        *llvmContextModulePair.llvmModule, arrTy, "gConstMatShared");

    // Initialize with matrix values if available
    std::vector<Constant*> constElems;
    constElems.reserve(2U * KK);
    for (unsigned r = 0; r < K; ++r) {
      for (unsigned c = 0; c < K; ++c) {
        auto cplx = matrix.rc(r, c);
        constElems.push_back(ConstantFP::get(scalarTy, cplx.real()));
        constElems.push_back(ConstantFP::get(scalarTy, cplx.imag()));
      }
    }
    gConstMat->setInitializer(ConstantArray::get(arrTy, constElems));

    genMatrixVectorMultiplyFromConst(
        B, config, matrix, qubits, gConstMat, svPtrV, args.pMatArg, scalarTy);
    break;
  } // case CUDAMatrixLoadMode::LoadInConstMemSpace
  } // switch (config.matrixLoadMode)

  B.CreateRetVoid();
  LLVM_DEBUG(func->dump());

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

  CUDAKernelInfo::CUDATuple cuTuple;
  return std::make_unique<CUDAKernelInfo>(
      CUDAKernelInfo::PTXStringType(), // ptxString
      config.precision,
      func->getName().str(),
      gate,
      cuTuple,
      gate->opCount(config.zeroTol));
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

// CUDAKernelManager& CUDAKernelManager::genCUDAGatesFromCircuitGraph(
//     const CUDAKernelGenConfig& config,
//     const CircuitGraph& graph,
//     const std::string& graphName,
//     int nQubits) {
//   const auto allBlocks = graph.getAllBlocks();
//   const auto mangledName = internal::mangleGraphName(graphName);
//   for (const auto& block : allBlocks) {
//     genCUDAGate(config,
//                 block->quantumGate,
//                 mangledName + std::to_string(block->id),
//                 nQubits);
//   }
//   return *this;
// }

#undef DEBUG_TYPE
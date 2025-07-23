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
  case SK_General:
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

struct IRMatDataCUDA {
  Value* reVal;
  Value* imVal;
  ScalarKind reKind;
  ScalarKind imKind;
};

std::vector<IRMatDataCUDA> getMatDataCUDA(IRBuilder<>& B,
                                          const GateMatrix& gateMatrix,
                                          const CUDAKernelGenConfig& config) {
  const int k = gateMatrix.nQubits();
  const unsigned K = 1 << k;
  const unsigned KK = K * K;

  std::vector<IRMatDataCUDA> data(KK);

  const double zTol = config.zeroTol / K;
  const double oTol = config.oneTol / K;
  const auto* cMat = gateMatrix.getConstantMatrix();
  // assert(cMat && "Parametrized matrices codegen not implemented yet");
  if (cMat == nullptr || config.forceDenseKernel) {
    for (unsigned i = 0; i < KK; i++) {
      data[i].reKind = SK_General;
      data[i].imKind = SK_General;
    }
  } else {
    for (unsigned i = 0; i < KK; i++) {
      if (cMat == nullptr || config.forceDenseKernel) {
        data[i].reKind = SK_General;
        data[i].imKind = SK_General;
        continue;
      }
      auto real = cMat->data()[i].real();
      auto imag = cMat->data()[i].imag();

      if (std::abs(real) < zTol)
        data[i].reKind = SK_Zero;
      else if (std::abs(real - 1.0) < oTol)
        data[i].reKind = SK_One;
      else if (std::abs(real + 1.0) < oTol)
        data[i].reKind = SK_MinusOne;
      else if (config.matrixLoadMode == CUDAKernelGenConfig::UseMatImmValues) {
        data[i].reKind = SK_ImmValue;
        data[i].reVal = ConstantFP::get(
            B.getContext(),
            (config.precision == 32) ? APFloat(static_cast<float>(real))
                                     : APFloat(static_cast<double>(real)));
      } else
        data[i].reKind = SK_General;

      if (std::abs(imag) < zTol)
        data[i].imKind = SK_Zero;
      else if (std::abs(imag - 1.0) < oTol)
        data[i].imKind = SK_One;
      else if (std::abs(imag + 1.0) < oTol)
        data[i].imKind = SK_MinusOne;
      else if (config.matrixLoadMode == CUDAKernelGenConfig::UseMatImmValues) {
        data[i].imKind = SK_ImmValue;
        data[i].imVal =
            ConstantFP::get(B.getContext(),
                            (config.precision == Precision::F32)
                                ? APFloat(static_cast<float>(imag))
                                : APFloat(static_cast<double>(imag)));
      } else
        data[i].imKind = SK_General;
    }
  }
  return data;
}

// Function* getFunctionDeclarationCUDA(
//     IRBuilder<>& B, llvm::Module& M, const std::string& funcName,
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
// \
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
                                     llvm::Module& M,
                                     const std::string& funcName,
                                     const CUDAKernelGenConfig& config,
                                     IRArgsCUDA& args) {
  // Always include pSvArg and pMatArg
  FunctionType* fty =
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
  auto* globalTidV = B.CreateMul(bidV, gridSizeV);
  globalTidV = B.CreateAdd(globalTidV, tidV, "counter.i32");
  globalTidV = B.CreateIntCast(globalTidV, B.getInt64Ty(), true, "global.tid");
  return globalTidV;
}
} // anonymous namespace

static void attachNoUnrollMetadata(llvm::IRBuilder<>& B,
                                   llvm::BasicBlock* latchBB) {
  using namespace llvm;
  Instruction* latchTerm = latchBB->getTerminator();
  if (!latchTerm)
    return; // safety
  LLVMContext& ctx = B.getContext();
  MDNode* noUnrollMD =
      MDNode::get(ctx, {MDString::get(ctx, "llvm.loop.unroll.disable")});
  latchTerm->setMetadata("llvm.loop", noUnrollMD);
}

/**
 *  Create a global array [2*K*K x scalarTy] to store the real/imag parts
 *    from matData. That way, we can do a run-time IR loop to read them,
 *    instead of unrolling a for-loop in C++.
 */
static llvm::GlobalVariable*
createGlobalMatrixArray_NoUnroll(llvm::Module& M,
                                 llvm::Type* scalarTy,
                                 const std::vector<IRMatDataCUDA>& matData,
                                 unsigned K,
                                 const std::string& globalName) {
  using namespace llvm;

  unsigned totalElems = 2 * K * K;
  ArrayType* arrTy = ArrayType::get(scalarTy, totalElems);

  // Build a ConstantArray with the imm real/imag values.
  std::vector<Constant*> constVals;
  constVals.reserve(totalElems);

  for (unsigned i = 0; i < K * K; i++) {
    const auto& md = matData[i];
    // If md.reVal is a ConstantFP, we can cast directly:
    ConstantFP* cRe = llvm::dyn_cast_or_null<ConstantFP>(md.reVal);
    ConstantFP* cIm = llvm::dyn_cast_or_null<ConstantFP>(md.imVal);

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

  Constant* arrInit = ConstantArray::get(arrTy, constVals);

  // Create a private global
  auto* gVar = new GlobalVariable(M,
                                  arrTy,
                                  /*isConstant=*/true,
                                  llvm::GlobalValue::PrivateLinkage,
                                  arrInit,
                                  globalName);
  gVar->setAlignment(llvm::MaybeAlign(8));
  return gVar;
}

void genMatrixVectorMultiply(llvm::IRBuilder<>& B,
                             const CUDAKernelGenConfig& config,
                             const GateMatrix& gateMat,
                             llvm::ArrayRef<int> qubits,
                             const std::vector<IRMatDataCUDA>& matData,
                             llvm::Value* svPtrV,
                             llvm::Type* scalarTy) {
  using namespace llvm;

  unsigned k = gateMat.nQubits();
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

    // innerBody: read M[r,c] from the global array, read oldAmp[c], do partial
    // sums
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

static GlobalVariable* createGlobalMatrixArray_SharedTiledImm(
    Module& M,
    Type* scalarTy,
    unsigned N,
    const std::vector<IRMatDataCUDA>& matData,
    const std::string& globalName) {
  // The array type: [2*N*N x scalarTy].
  unsigned totalElems = 2 * N * N;
  ArrayType* arrTy = ArrayType::get(scalarTy, totalElems);

  // --------------------------------------- For Debug
  // --------------------------------------------- #include <iostream> std::cerr
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

Value* getBlockIdxCUDA(IRBuilder<>& B) {
  return B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
}
Value* buildBitExtractOffset(IRBuilder<>& B,
                             Value* counterV,
                             ArrayRef<int> tgtQubits,
                             int nQubits) {
  Value* idxStart = B.getInt64(0);
  Value* blockIdx = B.CreateZExt(getBlockIdxCUDA(B), B.getInt64Ty());
  int bitPos = 0;
  for (int q = 0; q < nQubits; q++) {
    if (std::find(tgtQubits.begin(), tgtQubits.end(), q) == tgtQubits.end()) {
      Value* bit = B.CreateAnd(blockIdx, B.getInt64(1ULL << bitPos));
      bit = B.CreateLShr(bit, bitPos);
      bit = B.CreateShl(bit, q);
      idxStart = B.CreateOr(idxStart, bit);
      bitPos++;
    }
  }
  return idxStart;
}

Value* buildBitExtractOffsetConst(IRBuilder<>& B,
                                  Value* counterV,
                                  ArrayRef<int> tgtQubits,
                                  int nQubits) {
  Value* idxStart = B.getInt64(0);
  int bitPos = 0;
  for (int q = 0; q < nQubits; q++) {
    if (std::find(tgtQubits.begin(), tgtQubits.end(), q) == tgtQubits.end()) {
      Value* bit = B.CreateAnd(counterV, B.getInt64(1ULL << bitPos));
      bit = B.CreateLShr(bit, bitPos);
      bit = B.CreateShl(bit, q);
      idxStart = B.CreateOr(idxStart, bit);
      bitPos++;
    }
  }
  return idxStart;
}

void genMatrixVectorMultiply_SharedTiled(
    llvm::IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const GateMatrix& gateMat,
    llvm::ArrayRef<int> qubits,
    const std::vector<IRMatDataCUDA>& matData,
    llvm::Value* svPtrV,
    llvm::Type* scalarTy) {
  using namespace llvm;

  const unsigned k = gateMat.nQubits();
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

  // Value *bDim32 = B.CreateIntrinsic(B.getInt32Ty(),
  //                     Intrinsic::nvvm_read_ptx_sreg_ntid_x,{});

  // Value *ctr64 = B.CreateAdd(
  //                  B.CreateMul(B.CreateZExt(bid32 , B.getInt64Ty()),
  //                               B.CreateZExt(bDim32, B.getInt64Ty())),
  //                  B.CreateZExt(tid32 , B.getInt64Ty()),
  //                  "threadCtr");

  // /* packed logical row index */
  // Value *row64 = B.getInt64(0);
  // for (unsigned b = 0; b < k; ++b) {
  //     Value *mask      = B.getInt64(1ULL << qubits[b]);
  //     Value *bit       = B.CreateAnd(ctr64, mask);
  //     Value *bitToLSB  = B.CreateLShr(bit, qubits[b]);
  //     Value *bitPacked = B.CreateShl (bitToLSB,  b);
  //     row64            = B.CreateOr(row64, bitPacked);
  // }

  // /* guard: skip if row64 >= N */
  // BasicBlock *rowOK  = BasicBlock::Create(B.getContext(), "row.ok", func);
  // BasicBlock *rowEnd = BasicBlock::Create(B.getContext(), "row.end", func);
  // B.CreateCondBr(B.CreateICmpULT(row64, B.getInt64(N)), rowOK, rowEnd);
  // B.SetInsertPoint(rowOK);

  /*────────── Local accumulators in registers ──────────*/
  Value* accRe = ConstantFP::get(scalarTy, 0.0);
  Value* accIm = ConstantFP::get(scalarTy, 0.0);

  /* ========= Loop over column tiles ========*/
  const unsigned nTiles = (N + TILE - 1) / TILE;

  BasicBlock* tileChk = BasicBlock::Create(B.getContext(), "tile.chk", func);
  BasicBlock* tileBody = BasicBlock::Create(B.getContext(), "tile.body", func);
  BasicBlock* tileInc = BasicBlock::Create(B.getContext(), "tile.inc", func);
  BasicBlock* tileDone = BasicBlock::Create(B.getContext(), "tile.done", func);

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

  /* __syncthreads() */
  B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::nvvm_barrier0));

  /* dot‑product over this tile */
  for (unsigned t = 0; t < TILE; ++t) {
    Value* col = B.CreateAdd(col0Phi, B.getInt64(t));
    Value* inRange = B.CreateICmpULT(col, B.getInt64(N));

    BasicBlock* dotYes = BasicBlock::Create(B.getContext(), "dot.y", func);
    BasicBlock* dotNo = BasicBlock::Create(B.getContext(), "dot.n", func);
    BasicBlock* dotEnd = BasicBlock::Create(B.getContext(), "dot.end", func);

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
    llvm::IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const GateMatrix& gateMat,
    llvm::ArrayRef<int> qubits,
    llvm::Value* matBasePtr, // AS 1  – run‑time matrix
    llvm::Value* svPtrV,     // AS 0  – state‑vector
    llvm::Type* scalarTy) {
  using namespace llvm;

  /*────────── Parameters & types ──────────*/
  const unsigned k = gateMat.nQubits();
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

  /* 1.  Load TILE amplitudes of |x〉 into shared memory */
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
    llvm::IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const GateMatrix& gateMat,
    llvm::ArrayRef<int> qubits,
    llvm::Value* matBasePtr, // pointer to global memory (AS 1)
    llvm::Value* svPtrV,     // pointer to state vector
    llvm::Type* scalarTy) {
  using namespace llvm;

  unsigned k = gateMat.nQubits();
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

void genMatrixVectorMultiplyFromConst(llvm::IRBuilder<>& B,
                                      const CUDAKernelGenConfig& config,
                                      const GateMatrix& gateMat,
                                      const llvm::ArrayRef<int> qubits,
                                      llvm::GlobalVariable* gConstMat,
                                      llvm::Value* svPtrV,
                                      llvm::Value* matPtrV,
                                      llvm::Type* scalarTy) {
  bool isSymbolic = (gateMat.getConstantMatrix() == nullptr);
  unsigned k = gateMat.nQubits();
  unsigned K = 1u << k;

  std::vector<llvm::Value*> reAmpPtrs(K), imAmpPtrs(K);
  std::vector<llvm::Value*> reAmps(K), imAmps(K);

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
    llvm::Value* updatedReAmp0 = llvm::ConstantFP::get(scalarTy, 0.0);
    llvm::Value* updatedReAmp1 = llvm::ConstantFP::get(scalarTy, 0.0);
    llvm::Value* updatedImAmp = llvm::ConstantFP::get(scalarTy, 0.0);

    for (unsigned c = 0; c < K; c++) {
      uint64_t baseIndex = 2ull * (r * K + c);
      llvm::Value* idxReal = B.getInt64(baseIndex);
      llvm::Value* idxImag = B.getInt64(baseIndex + 1);

      llvm::Value* realPtr;
      llvm::Value* imagPtr;
      if (isSymbolic) {
        // Use runtime matrix from global memory (AS 1)
        realPtr = B.CreateGEP(scalarTy, matPtrV, idxReal, "matElemPtrRe");
        imagPtr = B.CreateGEP(scalarTy, matPtrV, idxImag, "matElemPtrIm");
      } else {
        // Use constant memory (AS 4)
        realPtr = B.CreateGEP(gConstMat->getValueType(),
                              gConstMat,
                              {B.getInt32(0), idxReal},
                              "constElemPtrRe");
        imagPtr = B.CreateGEP(gConstMat->getValueType(),
                              gConstMat,
                              {B.getInt32(0), idxImag},
                              "constElemPtrIm");
      }

      llvm::Value* matRe = B.CreateLoad(scalarTy, realPtr, "matRe");
      llvm::Value* matIm = B.CreateLoad(scalarTy, imagPtr, "matIm");
      llvm::Value* oldRe = reAmps[c];
      llvm::Value* oldIm = imAmps[c];

      updatedReAmp0 = B.CreateFAdd(updatedReAmp0, B.CreateFMul(matRe, oldRe));
      updatedReAmp1 = B.CreateFAdd(updatedReAmp1, B.CreateFMul(matIm, oldIm));
      updatedImAmp = B.CreateFAdd(
          updatedImAmp,
          B.CreateFAdd(B.CreateFMul(matRe, oldIm), B.CreateFMul(matIm, oldRe)));
    }

    llvm::Value* newReAmp = B.CreateFSub(updatedReAmp0, updatedReAmp1);
    llvm::Value* newImAmp = updatedImAmp;
    B.CreateStore(newReAmp, reAmpPtrs[r]);
    B.CreateStore(newImAmp, imAmpPtrs[r]);
  }
}

static GlobalVariable*
getOrCreateConstMatGlobal(Module& M, Type* arrTy, StringRef globalName) {
  // Check if already exists
  if (auto* gv = M.getNamedGlobal(globalName))
    return gv;

  // Create new global in address space 4
  auto* gConstMat =
      new GlobalVariable(M,
                         arrTy,
                         /*isConstant*/ true,
                         GlobalValue::ExternalLinkage,
                         UndefValue::get(arrTy), // Initialized later
                         globalName,
                         nullptr,
                         GlobalValue::NotThreadLocal,
                         4);
  gConstMat->setAlignment(MaybeAlign(8));
  return gConstMat;
}

CUDAKernelManager&
CUDAKernelManager::genCUDAGate(const CUDAKernelGenConfig& config,
                               std::shared_ptr<QuantumGate> gate,
                               const std::string& funcName,
                               int nQubits) {
  const unsigned k = gate->qubits.size();
  const unsigned K = 1ULL << k;
  const unsigned KK = K * K;

  LLVM_DEBUG(
      std::cerr << CYAN("=== DEBUG genGPUKernel '" << funcName << "' ===\n");
      utils::printArray(std::cerr << "Matrix on qubits ", gate->qubits) << "\n";
      gate->gateMatrix.printCMat(std::cerr) << "\n";);

  auto& llvmContextModulePair =
      createNewLLVMContextModulePair(funcName + "Module");

  IRBuilder<> B(*llvmContextModulePair.llvmContext);
  assert(config.precision != Precision::Unknown);
  Type* scalarTy = (config.precision == Precision::F32) ? B.getFloatTy() : B.getDoubleTy();

  IRArgsCUDA args;
  auto* func = getFunctionDeclarationCUDA(
      B, *llvmContextModulePair.llvmModule, funcName, config, args);

  BasicBlock* entryBB =
      BasicBlock::Create(*llvmContextModulePair.llvmContext, "entry", func);
  B.SetInsertPoint(entryBB);
  // get global tid
  auto* counterV = getGlobalTidCUDA(B);

  // Value* svPtrV = args.pSvArg;
  Value* idxStartV =
      (config.matrixLoadMode == CUDAMatrixLoadMode::LoadInConstMemSpace)
          ? buildBitExtractOffsetConst(
                B, counterV /*globalTid*/, gate->qubits, nQubits)
          : buildBitExtractOffset(
                B, counterV /*globalTid*/, gate->qubits, nQubits);
  idxStartV = B.CreateShl(idxStartV, 1); // ×2 because (re,im)
  Value* svPtrV = B.CreateGEP(scalarTy, args.pSvArg, idxStartV);

  // Value* svPtrV;
  // {
  //   // Value* twoTimesCounter = B.CreateMul(counterV, B.getInt64(2),
  //   "times2");
  //   // svPtrV = B.CreateGEP(scalarTy, args.pSvArg, twoTimesCounter,
  //   "sv.ptr"); Value* idxStartV = B.getInt64(0ULL); Value* tmpCounterV;
  //   uint64_t mask = 0ULL;
  //   int highestQ = gate->qubits.back();
  //   int qIdx = 0;
  //   int counterQ = 0;
  //   for (int q = 0; q <= highestQ; q++) {
  //     if (q < gate->qubits[qIdx]) {
  //       mask |= (1 << counterQ++);
  //       continue;
  //     }
  //     // q == qubits[qIdx];
  //     ++qIdx;
  //     if (mask == 0)
  //       continue;
  //     tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
  //     tmpCounterV = B.CreateShl(tmpCounterV, (qIdx - 1), "tmpCounter");
  //     idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
  //     // std::cerr << "  (globalThreadIdx & " << utils::as0b(mask, 32) << ")
  //     <<
  //     // " << (qIdx - 1) << "\n";
  //     mask = 0ULL;
  //   }
  //   mask = ~((1ULL << (gate->qubits.back() - k + 1)) - 1);
  //   // std::cerr << "  (globalThreadIdx & " << utils::as0b(mask, 32) << ") <<
  //   "
  //   // << (k) << "\n";

  //   tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
  //   tmpCounterV = B.CreateShl(tmpCounterV, k, "tmpCounter");
  //   idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "idxStart");
  //   idxStartV = B.CreateShl(idxStartV, 1, "idxStart");
  //   svPtrV = B.CreateGEP(scalarTy, args.pSvArg, idxStartV, "sv.ptr");
  // }

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
  // the pointer of sv start in this thread
  auto matData = getMatDataCUDA(B, gate->gateMatrix, config);

  switch (config.matrixLoadMode) {
  case CUDAMatrixLoadMode::UseMatImmValues: {
    const auto* gateCMat = gate->gateMatrix.getConstantMatrix();
    assert(gateCMat &&
           "Runtime matrix incompatible with UseMatImmValues load mode");

    if (gate->nQubits() <= 0) {
      genMatrixVectorMultiply(
          B, config, gate->gateMatrix, gate->qubits, matData, svPtrV, scalarTy);
    } else {
      // genMatrixVectorMultiply(B, config, gate->gateMatrix, gate->qubits,
      // matData, svPtrV, scalarTy);
      genMatrixVectorMultiply_SharedTiled(
          B, config, gate->gateMatrix, gate->qubits, matData, svPtrV, scalarTy);
    }
    break;
  }
  case CUDAKernelGenConfig::LoadInDefaultMemSpace: {
    // This path loads matrix from pMatArg (args.pMatArg) in address space 1
    Value* matBasePtr = args.pMatArg;
    // Cast from address space 0 to 1:
    matBasePtr = B.CreateAddrSpaceCast(
        args.pMatArg, PointerType::get(scalarTy, /*AS=*/1), "matGlobalPtr");

    // Generate the IR that loops over the matrix elements
    // (K*K complex elements) and loads from matBasePtr + offset
    if (gate->nQubits() <= 0) {
      genMatrixVectorMultiplyFromPointer(B,
                                         config,
                                         gate->gateMatrix,
                                         gate->qubits,
                                         matBasePtr,
                                         svPtrV,
                                         scalarTy);
    } else {
      // genMatrixVectorMultiplyFromPointer(B, config, gate->gateMatrix,
      // gate->qubits, matBasePtr, svPtrV, scalarTy);
      genMatrixVectorMultiplyFromPointer_SharedTiled(B,
                                                     config,
                                                     gate->gateMatrix,
                                                     gate->qubits,
                                                     matBasePtr,
                                                     svPtrV,
                                                     scalarTy);
    }
    break;
  }
  case CUDAKernelGenConfig::LoadInConstMemSpace: {
    // std::cerr << "***USE CONSTMEMSPACE\n";
    // This path loads the matrix from pointer in address space 4 (constant
    // mem).
    unsigned totalElems = 2 * (K * K);
    llvm::ArrayType* arrTy = llvm::ArrayType::get(scalarTy, totalElems);
    auto* gConstMat = getOrCreateConstMatGlobal(
        *llvmContextModulePair.llvmModule, arrTy, "gConstMatShared");

    // Initialize with matrix values if available
    if (auto cMat = gate->gateMatrix.getConstantMatrix()) {
      std::vector<llvm::Constant*> constElems;
      for (unsigned i = 0; i < K * K; i++) {
        constElems.push_back(ConstantFP::get(scalarTy, cMat->data()[i].real()));
        constElems.push_back(ConstantFP::get(scalarTy, cMat->data()[i].imag()));
      }
      gConstMat->setInitializer(ConstantArray::get(arrTy, constElems));
    } else {
      // For symbolic gates, need to initialize with proper values
      // should initialize with the actual Rz matrix values

      std::vector<llvm::Constant*> constElems;
      for (unsigned i = 0; i < K * K; i++) {
        double real = (i / K == i % K) ? 1.0 : 0.0;
        double imag = 0.0;
        constElems.push_back(ConstantFP::get(scalarTy, real));
        constElems.push_back(ConstantFP::get(scalarTy, imag));
      }
      gConstMat->setInitializer(ConstantArray::get(arrTy, constElems));
    }

    genMatrixVectorMultiplyFromConst(B,
                                     config,
                                     gate->gateMatrix,
                                     gate->qubits,
                                     gConstMat,
                                     svPtrV,
                                     args.pMatArg,
                                     scalarTy);
    break;
  }
  }

  B.CreateRetVoid();
  LLVM_DEBUG(func->dump());

  CUDAKernelInfo::CUDATuple cudaTuple;

  this->_cudaKernels.emplace_back(
      CUDAKernelInfo::PTXStringType(), // empty ptxString
      config.precision,
      func->getName().str(),
      gate,
      cudaTuple,
      gate->opCount(config.zeroTol));
  return *this;
}

CUDAKernelManager& CUDAKernelManager::genCUDAGatesFromCircuitGraph(
    const CUDAKernelGenConfig& config,
    const CircuitGraph& graph,
    const std::string& graphName,
    int nQubits) {
  const auto allBlocks = graph.getAllBlocks();
  const auto mangledName = internal::mangleGraphName(graphName);
  for (const auto& block : allBlocks) {
    genCUDAGate(config,
                block->quantumGate,
                mangledName + std::to_string(block->id),
                nQubits);
  }
  return *this;
}

// Attempt to implement kernel sharing for gates that meet the conditions
static constexpr double SIMILARITY_THRESHOLD = 0.85;
static constexpr unsigned MAX_GATES_PER_KERNEL = 4;
static constexpr unsigned MAX_QUBITS_PER_GATE = 8;

double qubitSimilarity(const QuantumGate& A, const QuantumGate& B) {
  // Convert each qubit list to a std::set.
  std::set<int> setA(A.qubits.begin(), A.qubits.end());
  std::set<int> setB(B.qubits.begin(), B.qubits.end());

  // Compute intersection
  std::vector<int> interAB;
  std::set_intersection(setA.begin(),
                        setA.end(),
                        setB.begin(),
                        setB.end(),
                        std::back_inserter(interAB));

  // Compute union
  std::vector<int> unionAB;
  std::set_union(setA.begin(),
                 setA.end(),
                 setB.begin(),
                 setB.end(),
                 std::back_inserter(unionAB));

  if (unionAB.empty()) {
    return 0.0;
  }
  return double(interAB.size()) / double(unionAB.size());
}

static Function* getMultiKernelDeclaration(IRBuilder<>& B,
                                           Module& M,
                                           const std::string& funcName) {
  FunctionType* fty = FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false);
  auto* func = Function::Create(fty, Function::ExternalLinkage, funcName, M);

  // Mark as a kernel
  auto& ctx = M.getContext();
  auto* mdString = MDString::get(ctx, "kernel");
  auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
  auto* kernelMetadata =
      MDNode::get(ctx, {ValueAsMetadata::get(func), mdString, mdOne});
  M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(kernelMetadata);

  // Name the argument
  func->getArg(0)->setName("p.sv");
  return func;
}

/**
 * Only group gates that pass threshold for qubitSimilarity,
 * MAX_QUBITS_PER_GATE, and cap the group at MAX_GATES_PER_KERNEL.
 */
std::vector<std::vector<std::shared_ptr<QuantumGate>>>
groupGatesByOverlapAndSize(
    const std::vector<std::shared_ptr<QuantumGate>>& allGates) {
  std::vector<std::vector<std::shared_ptr<QuantumGate>>> groups;
  groups.reserve(allGates.size());

  for (auto& g : allGates) {
    // also skip if gate is too large
    if (g->nQubits() > MAX_QUBITS_PER_GATE) {
      // put it alone in its own group
      groups.push_back({g});
      continue;
    }

    bool placed = false;
    for (auto& grp : groups) {
      if (grp.size() >= MAX_GATES_PER_KERNEL)
        continue; // group full
      // check similarity vs group[0]
      double sim = qubitSimilarity(*g, *grp.front());
      if (sim >= SIMILARITY_THRESHOLD) {
        grp.push_back(g);
        placed = true;
        break;
      }
    }
    if (!placed) {
      groups.emplace_back();
      groups.back().push_back(g);
    }
  }

  return groups;
}

CUDAKernelManager& CUDAKernelManager::genCUDAGateMulti(
    const CUDAKernelGenConfig& config,
    const std::vector<std::shared_ptr<QuantumGate>>& gateList,
    const std::string& funcName) {
  // 1) create new module
  auto& cmp = createNewLLVMContextModulePair(funcName + "_Module");
  IRBuilder<> B(*cmp.llvmContext);
  Module& M = *cmp.llvmModule;

  // 2) Build a function: “__global__ void kernel(double *pSv)”
  FunctionType* fty = FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false);
  Function* func = Function::Create(fty,
                                    Function::ExternalLinkage,
                                    funcName,
                                    &M // place it in module M
  );

  // Mark as a kernel via metadata
  {
    auto& ctx = M.getContext();
    auto* mdString = MDString::get(ctx, "kernel");
    auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
    auto* mdNode =
        MDNode::get(ctx, {ValueAsMetadata::get(func), mdString, mdOne});
    M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(mdNode);
  }

  // Name the single argument
  Argument* pSvArg = func->getArg(0);
  pSvArg->setName("p.sv");

  // 3) Create the “entry” basic block and set insertion point
  BasicBlock* entryBB = BasicBlock::Create(B.getContext(), "entry", func);
  B.SetInsertPoint(entryBB);

  // 4) Get global thread ID in 64-bit
  Value* threadIdx64 = getGlobalTidCUDA(B); // e.g. 0..(some large)

  // track total ops
  size_t totalOps = 0;

  // 5) For each gate, do:
  for (auto& g : gateList) {
    // (A) Create the bit-mask pointer. This replicates the single-gate
    // “idxStartV” logic.
    const auto& qubits = g->qubits;
    int k = (int)qubits.size();
    int highestQ = qubits.back();

    // need the scalar type for the GEP
    Type* scalarTy =
        (config.precision == 32) ? B.getFloatTy() : B.getDoubleTy();

    // Build idxStartV
    Value* idxStartV = B.getInt64(0);
    {
      Value* tmpCounterV;
      uint64_t mask = 0ULL;
      int qIdx = 0;
      int counterQ = 0;

      // for each integer q in [0..highestQ], build partial mask
      for (int q = 0; q <= highestQ; q++) {
        if (q < qubits[qIdx]) {
          // accumulate bits
          mask |= (1ULL << (counterQ++));
          continue;
        }
        // else q == qubits[qIdx]
        ++qIdx;
        if (mask != 0ULL) {
          tmpCounterV =
              B.CreateAnd(threadIdx64, B.getInt64(mask), "tmpCounter");
          // shift by (qIdx - 1)
          tmpCounterV =
              B.CreateShl(tmpCounterV, B.getInt64(qIdx - 1), "tmpCounter");
          idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
          mask = 0ULL;
        }
        if (qIdx >= k) {
          break; // done if we've processed all gate qubits
        }
      }

      // Now handle bits above the last qubit
      // mask = ~((1ULL << (gate->qubits.back() - k + 1)) - 1);
      mask = ~((1ULL << (highestQ - k + 1)) - 1ULL);

      tmpCounterV = B.CreateAnd(threadIdx64, B.getInt64(mask), "tmpCounter");
      tmpCounterV = B.CreateShl(tmpCounterV, B.getInt64(k), "tmpCounter");
      idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "idxStart");
    }

    // multiply by 2 (for real+imag)
    idxStartV = B.CreateShl(idxStartV, 1, "idxStart");

    // Now do GEP
    Value* maskedSvPtr =
        B.CreateGEP(scalarTy, pSvArg, idxStartV, "maskedSvPtr");

    // (B) Get matrix data
    auto matData = getMatDataCUDA(B, g->gateMatrix, config);

    // (C) Call genMatrixVectorMultiply_SharedTiled with this masked pointer
    genMatrixVectorMultiply_SharedTiled(
        B,
        config,
        g->gateMatrix,
        g->qubits,
        matData,
        maskedSvPtr, // use the bit-masked pointer
        scalarTy);

    // track ops
    totalOps += g->opCount(config.zeroTol);
  }

  // 6) return void
  B.CreateRetVoid();

  // 7) verify
  if (verifyFunction(*func, &errs())) {
    errs() << "[ERROR] multi function invalid.\n";
  }

  // 8) store kernel info
  CUDAKernelInfo::CUDATuple emptyCT;
  _cudaKernels.emplace_back(CUDAKernelInfo::PTXStringType(),
                            config.precision,
                            func->getName().str(),
                            nullptr, // no single gate
                            emptyCT,
                            totalOps);

  return *this;
}

CUDAKernelManager& CUDAKernelManager::genCUDAGatesFromCircuitGraphMulti(
    const CUDAKernelGenConfig& config,
    const CircuitGraph& graph,
    const std::string& graphName) {
  // gather all gates
  std::vector<std::shared_ptr<QuantumGate>> allGates;
  for (auto& block : graph.getAllBlocks()) {
    allGates.push_back(block->quantumGate);
  }

  // group them
  auto grouped = groupGatesByOverlapAndSize(allGates);

  // for each group => build a single multi kernel
  for (size_t i = 0; i < grouped.size(); i++) {
    std::string fnName = graphName + "_multi_" + std::to_string(i);
    genCUDAGateMulti(config, grouped[i], fnName);
  }

  return *this;
}

void CUDAKernelManager::dumpPTX(const std::string& kernelName,
                                llvm::raw_ostream& os) {
  // First check if we have already compiled to PTX
  for (auto& kernelInfo : standaloneKernels_) {
    if (kernelInfo.llvmFuncName == kernelName) {
      if (!kernelInfo.ptxString.empty()) {
        os << "=== PTX for kernel '" << kernelName << "' ===\n";
        os << kernelInfo.ptxString << "\n";
        return;
      }
      break;
    }
  }

  // If not found in compiled kernels, check the modules
  for (auto& cmp : llvmContextModulePairs) {
    auto& M = *cmp.llvmModule;
    if (auto* F = M.getFunction(kernelName)) {
      os << "=== Generating PTX for kernel '" << kernelName << "' ===\n";

      // Initialize LLVM targets
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmPrinters();
      llvm::InitializeAllAsmParsers();

      // Configure for NVPTX
      std::string error;
      auto targetTriple = "nvptx64-nvidia-cuda";
      auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
      if (!target) {
        os << "Error getting NVPTX target: " << error << "\n";
        return;
      }

      // Set target options
      llvm::TargetOptions opt;
      auto RM = std::optional<llvm::Reloc::Model>();
      auto targetMachine =
          target->createTargetMachine(targetTriple, "sm_70", "+ptx60", opt, RM);

      // Set up output stream
      llvm::SmallString<0> ptxCode;
      llvm::raw_svector_ostream ptxStream(ptxCode);

      // Use legacy pass manager
      llvm::legacy::PassManager pass;

      // Version-agnostic file type selection
      auto fileType =
#if LLVM_VERSION_MAJOR >= 10
          llvm::CodeGenFileType::AssemblyFile;
#else
          llvm::CGFT_AssemblyFile;
#endif

      if (targetMachine->addPassesToEmitFile(
              pass, ptxStream, nullptr, fileType)) {
        os << "Failed to generate PTX\n";
        return;
      }

      pass.run(M);

      os << ptxCode.str() << "\n";
      return;
    }
  }

  os << "No kernel found with name '" << kernelName << "'\n";
}

#undef DEBUG_TYPE
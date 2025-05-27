#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/Verifier.h>
#include "cast/LegacyQuantumGate.h"
#include "cast/LegacyCircuitGraph.h"

#include "simulation/KernelManager.h"
#include "simulation/KernelGenInternal.h"

#include "utils/utils.h"
#include "utils/iocolor.h"
#include "utils/Formats.h"

#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/Support/CodeGen.h>

#define DEBUG_TYPE "codegen-cuda"
#include <llvm/Support/Debug.h>
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
  Argument* pSvArg;       // ptr to statevector
  Argument* pMatArg;      // ptr to matrix
};

struct IRMatDataCUDA {
  Value* reVal;
  Value* imVal;
  ScalarKind reKind;
  ScalarKind imKind;
};

std::vector<IRMatDataCUDA> getMatDataCUDA(
    IRBuilder<>& B, const LegacyGateMatrix& gateMatrix,
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
        data[i].reVal = ConstantFP::get(B.getContext(), 
          (config.precision == 32) ? APFloat(static_cast<float>(real))
                                  : APFloat(static_cast<double>(real)));
      }
      else
        data[i].reKind = SK_General;

      if (std::abs(imag) < zTol)
        data[i].imKind = SK_Zero;
      else if (std::abs(imag - 1.0) < oTol)
        data[i].imKind = SK_One;
      else if (std::abs(imag + 1.0) < oTol)
        data[i].imKind = SK_MinusOne;
      else if (config.matrixLoadMode == CUDAKernelGenConfig::UseMatImmValues) {
        data[i].imKind = SK_ImmValue;
        data[i].imVal = ConstantFP::get(B.getContext(),
          (config.precision == 32) ? APFloat(static_cast<float>(imag))
                                  : APFloat(static_cast<double>(imag)));
      }
      else
        data[i].imKind = SK_General;
      }
    }
    return data;
  }

Function* getFunctionDeclarationCUDA(
    IRBuilder<>& B, llvm::Module& M, const std::string& funcName,
    const CUDAKernelGenConfig& config, IRArgsCUDA& args) {
  /*
      Address space:
      0: Generic;
      1: Global;
      2: Internal Use;
      3: Shared;
      4: Constant (often 64KB)
      5: Local;

      For a reference see https://llvm.org/docs/NVPTXUsage.html#id32
  */

  FunctionType *fty = (config.matrixLoadMode == CUDAKernelGenConfig::LoadInConstMemSpace)
                      ? FunctionType::get(B.getVoidTy(), { B.getPtrTy() }, false)
                      : FunctionType::get(B.getVoidTy(), { B.getPtrTy(),  B.getPtrTy() }, false);
  auto* func = Function::Create(
    fty,
    Function::ExternalLinkage,
    funcName,
    M
  );
  if (funcName.empty())
    func->setName("ptx_kernel_");
  else
    func->setName(funcName);
\
  args.pSvArg = func->getArg(0);
  args.pSvArg->setName("p.sv");
  if (config.matrixLoadMode != CUDAKernelGenConfig::LoadInConstMemSpace) {
    args.pMatArg = func->getArg(1);
    args.pMatArg->setName("p.mat");
  } else {
    args.pMatArg = nullptr;
  }

  // mark this function as a kernel
  auto* mdString = MDString::get(M.getContext(), "kernel");
  auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
  auto* kernelMetadata = MDNode::get(
    M.getContext(),
    { ValueAsMetadata::get(func), mdString, mdOne });
  M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(kernelMetadata);

  return func;
}

Value* getGlobalTidCUDA(IRBuilder<>& B) {
  // thread index
  auto* tidV = B.CreateIntrinsic(
    B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x,
    {}, nullptr, "tid");
  // gridSize (number of threads in each block)
  auto* gridSizeV = B.CreateIntrinsic(
    B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ntid_x,
    {}, nullptr, "blockSize");
  // block index
  auto* bidV = B.CreateIntrinsic(
    B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
    {}, nullptr, "bid");
  auto* globalTidV = B.CreateMul(bidV, gridSizeV);
  globalTidV = B.CreateAdd(globalTidV, tidV, "counter.i32");
  globalTidV = B.CreateIntCast(globalTidV, B.getInt64Ty(), true, "global.tid");
  return globalTidV;
}
} // anonymous namespace

static void attachNoUnrollMetadata(llvm::IRBuilder<>& B, llvm::BasicBlock* latchBB)
{
    using namespace llvm;
    Instruction* latchTerm = latchBB->getTerminator();
    if (!latchTerm) return; // safety
    LLVMContext &ctx = B.getContext();
    MDNode* noUnrollMD = MDNode::get(ctx, {
        MDString::get(ctx, "llvm.loop.unroll.disable")
    });
    latchTerm->setMetadata("llvm.loop", noUnrollMD);
}

/**
 *  Create a global array [2*K*K x scalarTy] to store the real/imag parts 
 *    from matData. That way, we can do a run-time IR loop to read them, 
 *    instead of unrolling a for-loop in C++.
 */
static llvm::GlobalVariable* createGlobalMatrixArray_NoUnroll(
    llvm::Module &M,
    llvm::Type   *scalarTy,
    const std::vector<IRMatDataCUDA> &matData,
    unsigned K,
    const std::string &globalName
)
{
    using namespace llvm;

    unsigned totalElems = 2 * K * K;
    ArrayType* arrTy    = ArrayType::get(scalarTy, totalElems);

    // Build a ConstantArray with the imm real/imag values.
    std::vector<Constant*> constVals;
    constVals.reserve(totalElems);

    for (unsigned i = 0; i < K*K; i++) {
        const auto &md = matData[i];
        // If md.reVal is a ConstantFP, we can cast directly:
        ConstantFP *cRe = llvm::dyn_cast_or_null<ConstantFP>(md.reVal);
        ConstantFP *cIm = llvm::dyn_cast_or_null<ConstantFP>(md.imVal);

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
    auto* gVar = new GlobalVariable(
        M,
        arrTy,
        /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage,
        arrInit,
        globalName
    );
    gVar->setAlignment(llvm::MaybeAlign(8));
    return gVar;
}


void genMatrixVectorMultiply(
    llvm::IRBuilder<>&            B,
    const CUDAKernelGenConfig&    config,
    const LegacyGateMatrix&             gateMat,
    llvm::ArrayRef<int>           qubits,
    const std::vector<IRMatDataCUDA>& matData,
    llvm::Value*                  svPtrV,
    llvm::Type*                   scalarTy
)
{
    using namespace llvm;

    unsigned k = gateMat.nQubits();
    unsigned K = 1u << k;
    LLVMContext &ctx = B.getContext();

    BasicBlock *curBB = B.GetInsertBlock();
    Function   *func  = curBB->getParent();
    Module     &M     = *func->getParent();

    GlobalVariable* gMatImmediate = createGlobalMatrixArray_NoUnroll(
        M, scalarTy, matData, K, "gMatImmediate"
    );

    ArrayType *arrTy = ArrayType::get(scalarTy, K);
    AllocaInst* reAmpsAlloca = B.CreateAlloca(arrTy, nullptr, "reAmps");
    AllocaInst* imAmpsAlloca = B.CreateAlloca(arrTy, nullptr, "imAmps");

    BasicBlock* entryBB     = curBB;
    BasicBlock* loadCheckBB = BasicBlock::Create(ctx, "load.check", func);
    BasicBlock* loadBodyBB  = BasicBlock::Create(ctx, "load.body",  func);
    BasicBlock* loadIncBB   = BasicBlock::Create(ctx, "load.inc",   func);
    BasicBlock* loadExitBB  = BasicBlock::Create(ctx, "load.exit",  func);

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
            Value* test  = B.CreateAnd(iPHI, maskB);
            Value* cond  = B.CreateICmpNE(test, ConstantInt::get(B.getInt32Ty(), 0));
            Value* shiftVal = ConstantInt::get(B.getInt64Ty(), 1ULL << qubits[b]);
            Value* orVal = B.CreateOr(deltaVal, shiftVal);
            deltaVal = B.CreateSelect(cond, orVal, deltaVal);
        }
        // rePtr = svPtrV + 2*deltaVal
        // imPtr = svPtrV + 2*deltaVal+1
        Value* twoDelta = B.CreateMul(ConstantInt::get(B.getInt64Ty(), 2), deltaVal);
        Value* rePtr = B.CreateGEP(scalarTy, svPtrV, twoDelta, "rePtr");
        Value* imPtr = B.CreateGEP(
            scalarTy, svPtrV,
            B.CreateAdd(twoDelta, ConstantInt::get(B.getInt64Ty(), 1)),
            "imPtr"
        );
        // load oldRe, oldIm
        Value* oldRe = B.CreateLoad(scalarTy, rePtr, "oldRe");
        Value* oldIm = B.CreateLoad(scalarTy, imPtr, "oldIm");

        // store into reAmpsAlloca[i], imAmpsAlloca[i]
        Value* reSlot = B.CreateGEP(arrTy, reAmpsAlloca,
                         { ConstantInt::get(B.getInt32Ty(),0), iPHI });
        Value* imSlot = B.CreateGEP(arrTy, imAmpsAlloca,
                         { ConstantInt::get(B.getInt32Ty(),0), iPHI });
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
    attachNoUnrollMetadata(B, loadIncBB); // disable unroll for amplitude-load loop

    BasicBlock* outerCheckBB = BasicBlock::Create(ctx, "outer.check", func);
    BasicBlock* outerBodyBB  = BasicBlock::Create(ctx, "outer.body",  func);
    BasicBlock* outerIncBB   = BasicBlock::Create(ctx, "outer.inc",   func);
    BasicBlock* outerExitBB  = BasicBlock::Create(ctx, "outer.exit",  func);

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
        AllocaInst* imAmpA  = B.CreateAlloca(scalarTy, nullptr, "imAmpA");

        Value* zeroVal = ConstantFP::get(scalarTy, 0.0);
        B.CreateStore(zeroVal, reAmp0A);
        B.CreateStore(zeroVal, reAmp1A);
        B.CreateStore(zeroVal, imAmpA);

        // Inner loop c in [0..K)
        BasicBlock* innerCheckBB = BasicBlock::Create(ctx, "inner.check", func);
        BasicBlock* innerBodyBB  = BasicBlock::Create(ctx, "inner.body",  func);
        BasicBlock* innerIncBB   = BasicBlock::Create(ctx, "inner.inc",   func);
        BasicBlock* innerExitBB  = BasicBlock::Create(ctx, "inner.exit",  func);

        B.CreateBr(innerCheckBB);
        B.SetInsertPoint(innerCheckBB);
        PHINode* cPHI = B.CreatePHI(B.getInt32Ty(), 2, "c");
        cPHI->addIncoming(ConstantInt::get(B.getInt32Ty(), 0), outerBodyBB);

        Value* condInner = B.CreateICmpSLT(cPHI, ConstantInt::get(B.getInt32Ty(), K));
        B.CreateCondBr(condInner, innerBodyBB, innerExitBB);

        // innerBody: read M[r,c] from the global array, read oldAmp[c], do partial sums
        B.SetInsertPoint(innerBodyBB);
        {
            // linear index = r*K + c => 2*(r*K + c) => re/im
            Value* r64 = B.CreateIntCast(rPHI, B.getInt64Ty(), false);
            Value* c64 = B.CreateIntCast(cPHI, B.getInt64Ty(), false);
            Value* bigK = ConstantInt::get(B.getInt64Ty(), K);
            Value* rK   = B.CreateMul(r64, bigK);
            Value* rc   = B.CreateAdd(rK, c64);
            Value* base2= B.CreateMul(ConstantInt::get(B.getInt64Ty(), 2), rc);

            // GEP into gMatImmediate => real, imag
            // global array type is [2*K*K x scalarTy]
            auto arrTy2 = llvm::cast<ArrayType>(gMatImmediate->getValueType());
            // rePtr => gMatImmediate[0, base2], imPtr => gMatImmediate[0, base2+1]
            Value* idxRe = base2;
            Value* idxIm = B.CreateAdd(base2, ConstantInt::get(B.getInt64Ty(), 1));

            Value* rePtr = B.CreateGEP(
                arrTy2, gMatImmediate,
                { ConstantInt::get(B.getInt32Ty(),0),
                  B.CreateIntCast(idxRe, B.getInt32Ty(), false) },
                "matRePtr"
            );
            Value* imPtr = B.CreateGEP(
                arrTy2, gMatImmediate,
                { ConstantInt::get(B.getInt32Ty(),0),
                  B.CreateIntCast(idxIm, B.getInt32Ty(), false) },
                "matImPtr"
            );

            Value* matRe = B.CreateLoad(scalarTy, rePtr, "matRe");
            Value* matIm = B.CreateLoad(scalarTy, imPtr, "matIm");

            // oldAmp[c]
            Value* oldReSlot = B.CreateGEP(
                arrTy, reAmpsAlloca,
                { ConstantInt::get(B.getInt32Ty(),0), cPHI }
            );
            Value* oldImSlot = B.CreateGEP(
                arrTy, imAmpsAlloca,
                { ConstantInt::get(B.getInt32Ty(),0), cPHI }
            );
            Value* oldRe = B.CreateLoad(scalarTy, oldReSlot, "oldRe");
            Value* oldIm = B.CreateLoad(scalarTy, oldImSlot, "oldIm");

            // partial sums
            Value* reA0 = B.CreateLoad(scalarTy, reAmp0A);
            Value* reA1 = B.CreateLoad(scalarTy, reAmp1A);
            Value* imA  = B.CreateLoad(scalarTy, imAmpA);

            // reA0 += matRe * oldRe
            Value* addRe0 = B.CreateFAdd(reA0, B.CreateFMul(matRe, oldRe));
            B.CreateStore(addRe0, reAmp0A);

            // reA1 += matIm * oldIm
            Value* addRe1 = B.CreateFAdd(reA1, B.CreateFMul(matIm, oldIm));
            B.CreateStore(addRe1, reAmp1A);

            // imA  += matRe*oldIm + matIm*oldRe
            Value* cross = B.CreateFAdd(
                B.CreateFMul(matRe, oldIm),
                B.CreateFMul(matIm, oldRe)
            );
            Value* addIm = B.CreateFAdd(imA, cross);
            B.CreateStore(addIm, imAmpA);
        }
        B.CreateBr(innerIncBB);

        B.SetInsertPoint(innerIncBB);
        {
            Value* cNext = B.CreateAdd(cPHI, ConstantInt::get(B.getInt32Ty(),1));
            cPHI->addIncoming(cNext, innerIncBB);
            B.CreateBr(innerCheckBB);
        }

        B.SetInsertPoint(innerExitBB);
        attachNoUnrollMetadata(B, innerIncBB);

        // finalize amplitude: newReAmp = reA0 - reA1, newImAmp = imA
        Value* reA0 = B.CreateLoad(scalarTy, reAmp0A);
        Value* reA1 = B.CreateLoad(scalarTy, reAmp1A);
        Value* imA  = B.CreateLoad(scalarTy, imAmpA);

        Value* newReAmp = B.CreateFSub(reA0, reA1);
        Value* newImAmp = imA;

        // store back to row r
        Value* deltaVal = ConstantInt::get(B.getInt64Ty(), 0);
        for (unsigned b = 0; b < k; b++) {
            Value* maskB = ConstantInt::get(B.getInt32Ty(), 1u << b);
            Value* test  = B.CreateAnd(rPHI, maskB);
            Value* cond  = B.CreateICmpNE(test, ConstantInt::get(B.getInt32Ty(), 0));
            Value* shiftVal = ConstantInt::get(B.getInt64Ty(), 1ULL << qubits[b]);
            Value* orVal = B.CreateOr(deltaVal, shiftVal);
            deltaVal = B.CreateSelect(cond, orVal, deltaVal);
        }
        Value* twoDelta = B.CreateMul(ConstantInt::get(B.getInt64Ty(),2), deltaVal);
        Value* outRePtr = B.CreateGEP(scalarTy, svPtrV, twoDelta);
        Value* outImPtr = B.CreateGEP(
            scalarTy, svPtrV,
            B.CreateAdd(twoDelta, ConstantInt::get(B.getInt64Ty(),1))
        );

        B.CreateStore(newReAmp, outRePtr);
        B.CreateStore(newImAmp, outImPtr);
    }
    B.CreateBr(outerIncBB);

    B.SetInsertPoint(outerIncBB);
    {
        Value* rNext = B.CreateAdd(rPHI, ConstantInt::get(B.getInt32Ty(),1));
        rPHI->addIncoming(rNext, outerIncBB);
        B.CreateBr(outerCheckBB);
    }

    B.SetInsertPoint(outerExitBB);
    attachNoUnrollMetadata(B, outerIncBB);

    // IR now has run-time loops for loading amplitudes, 
    // for (r,c) partial sums, and unrolling has been disabled.
}

static GlobalVariable* createGlobalMatrixArray_SharedTiledImm(
    Module &M,
    Type* scalarTy,
    unsigned N,
    const std::vector<IRMatDataCUDA> &matData,
    const std::string &globalName
)
{
    // The array type: [2*N*N x scalarTy].
    unsigned totalElems = 2 * N * N;
    ArrayType *arrTy = ArrayType::get(scalarTy, totalElems);

    std::vector<Constant*> initVals;
    initVals.reserve(totalElems);

    // Fill them from matData. We assume matData.size() == (N*N).
    // For each index i => matData[i] => reVal, imVal
    // We'll place them in positions [2*i], [2*i + 1].
    for (unsigned i = 0; i < N*N; i++) {
        const auto &md = matData[i];
        // Attempt to cast reVal, imVal to ConstantFP if SK_ImmValue
        ConstantFP *cRe = dyn_cast_or_null<ConstantFP>(md.reVal);
        ConstantFP *cIm = dyn_cast_or_null<ConstantFP>(md.imVal);

        // Fallback
        if (md.reKind == SK_ImmValue && cRe) {
            initVals.push_back(cRe);
        }
        else if (md.reKind == SK_One) {
            initVals.push_back(ConstantFP::get(scalarTy, 1.0));
        }
        else if (md.reKind == SK_MinusOne) {
            initVals.push_back(ConstantFP::get(scalarTy, -1.0));
        }
        else if (md.reKind == SK_Zero) {
            initVals.push_back(ConstantFP::get(scalarTy, 0.0));
        }
        else {
            // e.g. SK_General => fallback to 0.0 or handle differently
            initVals.push_back(ConstantFP::get(scalarTy, 0.0));
        }

        if (md.imKind == SK_ImmValue && cIm) {
            initVals.push_back(cIm);
        }
        else if (md.imKind == SK_One) {
            initVals.push_back(ConstantFP::get(scalarTy, 1.0));
        }
        else if (md.imKind == SK_MinusOne) {
            initVals.push_back(ConstantFP::get(scalarTy, -1.0));
        }
        else if (md.imKind == SK_Zero) {
            initVals.push_back(ConstantFP::get(scalarTy, 0.0));
        }
        else {
            initVals.push_back(ConstantFP::get(scalarTy, 0.0));
        }
    }

    Constant* arrInit = ConstantArray::get(arrTy, initVals);

    auto* gVar = new GlobalVariable(
        M,
        arrTy,
        /*isConstant=*/true,
        GlobalValue::PrivateLinkage, // or InternalLinkage
        arrInit,
        globalName
    );
    gVar->setAlignment(MaybeAlign(8));
    return gVar;
}

void genMatrixVectorMultiply_SharedTiled(
    IRBuilder<> &B,
    const CUDAKernelGenConfig &config,
    const LegacyGateMatrix &gateMat,
    llvm::ArrayRef<int> qubits,
    const std::vector<IRMatDataCUDA> &matData,
    Value *svPtrV,
    Type *scalarTy
)
{
    using namespace llvm;

    // Dimension
    unsigned k = gateMat.nQubits();
    unsigned N = (1u << k);

    // Global matrix array
    Function *func = B.GetInsertBlock()->getParent();
    Module   &M    = *func->getParent();
    GlobalVariable *gMatImm = createGlobalMatrixArray_SharedTiledImm(
        M, scalarTy, N, matData, "gMatImmSharedTiled"
    );

    // Create static shared arrays tileM, tileX
    constexpr unsigned TILE_SIZE = 256;
    ArrayType *tileArrTy = ArrayType::get(scalarTy, 2ULL * TILE_SIZE);

    auto* tileM_GV = new GlobalVariable(
        M, tileArrTy, false,
        GlobalValue::PrivateLinkage,
        UndefValue::get(tileArrTy),
        "TileM",
        nullptr,
        GlobalValue::NotThreadLocal,
        /*AddressSpace=*/3
    );
    tileM_GV->setAlignment(MaybeAlign(8));

    auto* tileX_GV = new GlobalVariable(
        M, tileArrTy, false,
        GlobalValue::PrivateLinkage,
        UndefValue::get(tileArrTy),
        "TileX",
        nullptr,
        GlobalValue::NotThreadLocal,
        3
    );
    tileX_GV->setAlignment(MaybeAlign(8));

    Value *zero32 = B.getInt32(0);
    Value *tileMBase = B.CreateGEP(tileArrTy, tileM_GV, { zero32, zero32 }, "tileM.base");
    Value *tileXBase = B.CreateGEP(tileArrTy, tileX_GV, { zero32, zero32 }, "tileX.base");

    //
    Value *row64 = getGlobalTidCUDA(B);
    Function *f = B.GetInsertBlock()->getParent();

    BasicBlock *checkEndBB = BasicBlock::Create(B.getContext(), "rowCheck.end", f);
    BasicBlock *rowOkBB    = BasicBlock::Create(B.getContext(), "rowOk", f);

    Value *condRowOk = B.CreateICmpULT(row64, B.getInt64(N), "condRowOk");
    B.CreateCondBr(condRowOk, rowOkBB, checkEndBB);

    // Create them in the *entry* block of the function
    IRBuilder<> entryBuilder(&f->getEntryBlock(), f->getEntryBlock().begin());
    AllocaInst *sumReAlloca = entryBuilder.CreateAlloca(scalarTy, nullptr, "sumReA");
    AllocaInst *sumImAlloca = entryBuilder.CreateAlloca(scalarTy, nullptr, "sumImA");

    // Initialize them to 0.0 in the entry block
    entryBuilder.CreateStore(ConstantFP::get(scalarTy, 0.0), sumReAlloca);
    entryBuilder.CreateStore(ConstantFP::get(scalarTy, 0.0), sumImAlloca);

    B.SetInsertPoint(rowOkBB);

    // 5) Outer loop over cStart
    BasicBlock *loopCheckBB = BasicBlock::Create(B.getContext(), "cStart.check", f);
    BasicBlock *loopBodyBB  = BasicBlock::Create(B.getContext(), "cStart.body",  f);
    BasicBlock *loopIncBB   = BasicBlock::Create(B.getContext(), "cStart.inc",   f);
    BasicBlock *loopExitBB  = BasicBlock::Create(B.getContext(), "cStart.exit",  f);

    B.CreateBr(loopCheckBB);
    B.SetInsertPoint(loopCheckBB);

    PHINode *cStartPHI = B.CreatePHI(B.getInt64Ty(), 2, "cStart");
    cStartPHI->addIncoming(B.getInt64(0), rowOkBB);

    Value *condC = B.CreateICmpSLT(cStartPHI, B.getInt64(N), "condC");
    B.CreateCondBr(condC, loopBodyBB, loopExitBB);

    B.SetInsertPoint(loopBodyBB);
    Value *cStartVal = cStartPHI;

    auto barrierFuncName = Intrinsic::getName(Intrinsic::nvvm_barrier0);
    auto barrierFuncTy = Intrinsic::getType(M.getContext(), Intrinsic::nvvm_barrier0);
    auto barrierFunc = M.getOrInsertFunction(barrierFuncName, barrierFuncTy);

    Value *threadIdx32 = B.CreateIntrinsic(
        B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});

    // cVal = cStartVal + threadIdx.x
    Value *cVal = B.CreateAdd(cStartVal, B.CreateZExt(threadIdx32, B.getInt64Ty()), "cVal");

    // Check inRange
    BasicBlock *thenBB = BasicBlock::Create(B.getContext(), "load.inRange", f);
    BasicBlock *elseBB = BasicBlock::Create(B.getContext(), "load.outRange", f);
    BasicBlock *contBB = BasicBlock::Create(B.getContext(), "load.cont", f);
    Value *condInRange = B.CreateICmpULT(cVal, B.getInt64(N));
    B.CreateCondBr(condInRange, thenBB, elseBB);

    // thenBB => load from global => store to tileM, tileX
    B.SetInsertPoint(thenBB);
    {
        Value *rowTimesN = B.CreateMul(row64, B.getInt64(N));
        Value *matIdx    = B.CreateAdd(rowTimesN, cVal);
        Value *matIdx2   = B.CreateMul(matIdx, B.getInt64(2));

        auto arrTy2 = llvm::cast<ArrayType>(gMatImm->getValueType());
        Value *rePtr = B.CreateGEP(
            arrTy2, gMatImm,
            { B.getInt32(0), B.CreateIntCast(matIdx2, B.getInt32Ty(), false) }
        );
        Value *imPtr = B.CreateGEP(
            arrTy2, gMatImm,
            { B.getInt32(0),
              B.CreateAdd(B.CreateIntCast(matIdx2, B.getInt32Ty(), false),
                          B.getInt32(1)) }
        );
        Value *mReVal = B.CreateLoad(scalarTy, rePtr);
        Value *mImVal = B.CreateLoad(scalarTy, imPtr);

        Value *i64ThreadIdx = B.CreateZExt(threadIdx32, B.getInt64Ty());
        Value *tileOff = B.CreateMul(i64ThreadIdx, B.getInt64(2));
        B.CreateStore(mReVal, B.CreateGEP(scalarTy, tileMBase, tileOff));
        B.CreateStore(mImVal, B.CreateGEP(scalarTy, tileMBase, B.CreateAdd(tileOff, B.getInt64(1))));

        // X[cVal]
        Value *xIdx2 = B.CreateMul(cVal, B.getInt64(2));
        Value *xRePtr = B.CreateGEP(scalarTy, svPtrV, xIdx2);
        Value *xImPtr = B.CreateGEP(scalarTy, svPtrV, B.CreateAdd(xIdx2, B.getInt64(1)));
        Value *xReVal = B.CreateLoad(scalarTy, xRePtr);
        Value *xImVal = B.CreateLoad(scalarTy, xImPtr);
        B.CreateStore(xReVal, B.CreateGEP(scalarTy, tileXBase, tileOff));
        B.CreateStore(xImVal, B.CreateGEP(scalarTy, tileXBase, B.CreateAdd(tileOff, B.getInt64(1))));
        B.CreateBr(contBB);
    }

    // elseBB => store zeros
    B.SetInsertPoint(elseBB);
    {
        Value *i64ThreadIdx = B.CreateZExt(threadIdx32, B.getInt64Ty());
        Value *tileOff = B.CreateMul(i64ThreadIdx, B.getInt64(2));
        Value *zeroVal = ConstantFP::get(scalarTy, 0.0);
        B.CreateStore(zeroVal, B.CreateGEP(scalarTy, tileMBase, tileOff));
        B.CreateStore(zeroVal, B.CreateGEP(scalarTy, tileMBase, B.CreateAdd(tileOff, B.getInt64(1))));
        B.CreateStore(zeroVal, B.CreateGEP(scalarTy, tileXBase, tileOff));
        B.CreateStore(zeroVal, B.CreateGEP(scalarTy, tileXBase, B.CreateAdd(tileOff, B.getInt64(1))));
        B.CreateBr(contBB);
    }

    // barrier
    B.SetInsertPoint(contBB);
    B.CreateCall(barrierFunc);

    // Dot-product => for j in [0..TILE_SIZE)
    BasicBlock *dotCheckBB = BasicBlock::Create(B.getContext(), "dot.check", f);
    BasicBlock *dotBodyBB  = BasicBlock::Create(B.getContext(), "dot.body",  f);
    BasicBlock *dotIncBB   = BasicBlock::Create(B.getContext(), "dot.inc",   f);
    BasicBlock *dotExitBB  = BasicBlock::Create(B.getContext(), "dot.exit",  f);

    B.CreateBr(dotCheckBB);
    B.SetInsertPoint(dotCheckBB);

    PHINode *jPHI = B.CreatePHI(B.getInt64Ty(), 2, "j");
    jPHI->addIncoming(B.getInt64(0), contBB);

    Value *condJ = B.CreateICmpSLT(jPHI, B.getInt64(TILE_SIZE));
    B.CreateCondBr(condJ, dotBodyBB, dotExitBB);

    // dotBodyBB
    B.SetInsertPoint(dotBodyBB);
    {
        // load partial sums from alloca each iteration
        Value *oldRe = B.CreateLoad(scalarTy, sumReAlloca, "oldReSum");
        Value *oldIm = B.CreateLoad(scalarTy, sumImAlloca, "oldImSum");

        Value *off2 = B.CreateMul(jPHI, B.getInt64(2));
        Value *mRePtr = B.CreateGEP(scalarTy, tileMBase, off2);
        Value *mImPtr = B.CreateGEP(scalarTy, tileMBase, B.CreateAdd(off2, B.getInt64(1)));
        Value *xRePtr = B.CreateGEP(scalarTy, tileXBase, off2);
        Value *xImPtr = B.CreateGEP(scalarTy, tileXBase, B.CreateAdd(off2, B.getInt64(1)));

        Value *Mre = B.CreateLoad(scalarTy, mRePtr, "Mre");
        Value *Mim = B.CreateLoad(scalarTy, mImPtr, "Mim");
        Value *Xre = B.CreateLoad(scalarTy, xRePtr, "Xre");
        Value *Xim = B.CreateLoad(scalarTy, xImPtr, "Xim");

        // partial product
        Value *prodRe = B.CreateFSub(B.CreateFMul(Mre, Xre), B.CreateFMul(Mim, Xim));
        Value *prodIm = B.CreateFAdd(B.CreateFMul(Mre, Xim), B.CreateFMul(Mim, Xre));

        // new partial sums
        Value *sumReNew = B.CreateFAdd(oldRe, prodRe, "sumReNew");
        Value *sumImNew = B.CreateFAdd(oldIm, prodIm, "sumImNew");

        // store them back
        B.CreateStore(sumReNew, sumReAlloca);
        B.CreateStore(sumImNew, sumImAlloca);

        Value *jNext = B.CreateAdd(jPHI, B.getInt64(1));
        B.CreateBr(dotIncBB);

        // dotIncBB
        B.SetInsertPoint(dotIncBB);
        jPHI->addIncoming(jNext, dotIncBB);
        B.CreateBr(dotCheckBB);
    }

    // dotExitBB
    B.SetInsertPoint(dotExitBB);
    B.CreateCall(barrierFunc);

    B.CreateBr(loopIncBB);

    // loopIncBB => next cStart = cStart + TILE_SIZE
    B.SetInsertPoint(loopIncBB);
    Value *cNextVal = B.CreateAdd(cStartVal, B.getInt64(TILE_SIZE));
    cStartPHI->addIncoming(cNextVal, loopIncBB);
    B.CreateBr(loopCheckBB);

    // loopExitBB => now have final partial sums in sumReAlloca / sumImAlloca
    B.SetInsertPoint(loopExitBB);

    // read them out
    Value *finalSumRe = B.CreateLoad(scalarTy, sumReAlloca);
    Value *finalSumIm = B.CreateLoad(scalarTy, sumImAlloca);

    // store to row
    Value *rowTimes2 = B.CreateMul(row64, B.getInt64(2));
    Value *outRePtr = B.CreateGEP(scalarTy, svPtrV, rowTimes2);
    Value *outImPtr = B.CreateGEP(scalarTy, svPtrV, B.CreateAdd(rowTimes2, B.getInt64(1)));

    B.CreateStore(finalSumRe, outRePtr);
    B.CreateStore(finalSumIm, outImPtr);

    B.CreateBr(checkEndBB);
    B.SetInsertPoint(checkEndBB);
}


void genMatrixVectorMultiplyFromPointer_SharedTiled(
    llvm::IRBuilder<>           &B,
    const CUDAKernelGenConfig   &config,
    const LegacyGateMatrix            &gateMat,
    llvm::ArrayRef<int>          qubits,
    llvm::Value                 *matBasePtr,  // pointer to global memory
    llvm::Value                 *svPtrV,      // pointer to statevector
    llvm::Type                  *scalarTy)
{
  using namespace llvm;

  // 1) Get dimension
  unsigned k = gateMat.nQubits();
  unsigned N = (1u << k);  // dimension

  constexpr unsigned TILE_SIZE = 256;  // this should match GPU block size

  Module *mod = B.GetInsertBlock()->getModule();

  // 2) Create *static* shared arrays for M & X chunk in AS=3
  ArrayType *tileArrTy = ArrayType::get(scalarTy, 2ULL * TILE_SIZE);

  // a) create or reuse a global for tileM
  auto* tileM_GV = new GlobalVariable(
      *mod,
      tileArrTy,
      /*isConstant=*/false,
      GlobalValue::PrivateLinkage,
      UndefValue::get(tileArrTy),
      "TileM",
      nullptr,
      GlobalValue::NotThreadLocal,
      /*AddressSpace=*/3 // shared mem
  );
  tileM_GV->setAlignment(MaybeAlign(8));

  // b) create or reuse a global for tileX
  auto* tileX_GV = new GlobalVariable(
      *mod,
      tileArrTy,
      /*isConstant=*/false,
      GlobalValue::PrivateLinkage,
      UndefValue::get(tileArrTy),
      "TileX",
      nullptr,
      GlobalValue::NotThreadLocal,
      3 // shared
  );
  tileX_GV->setAlignment(MaybeAlign(8));

  // Take pointer to the first element of each array
  // GEP: tileMBase = &tileM_GV[0][0];
  auto *zero32 = B.getInt32(0);
  auto *tileMBase = B.CreateGEP(
      tileArrTy,
      tileM_GV,
      { zero32, zero32 },
      "tileM.base"
  );
  auto *tileXBase = B.CreateGEP(
      tileArrTy,
      tileX_GV,
      { zero32, zero32 },
      "tileX.base"
  );

  // 3) Get thread row index: row = blockIdx.x * blockDim.x + threadIdx.x
  //    in 64-bit
  Value *row64 = getGlobalTidCUDA(B); // e.g. 0..N-1
  // If row >= N => skip
  BasicBlock *entryBB = B.GetInsertBlock();
  Function   *func    = entryBB->getParent();

  BasicBlock *checkEndBB = BasicBlock::Create(B.getContext(), "rowCheck.end", func);
  BasicBlock *rowOkBB    = BasicBlock::Create(B.getContext(), "rowOk",        func);

  Value *condRowOk = B.CreateICmpULT(row64, B.getInt64(N), "condRowOk");
  B.CreateCondBr(condRowOk, rowOkBB, checkEndBB);

  B.SetInsertPoint(rowOkBB);

  // keep partial sum in local Values
  Value *sumRe = ConstantFP::get(scalarTy, 0.0);
  Value *sumIm = ConstantFP::get(scalarTy, 0.0);

  // We want a for-loop over cStart in [0..N..TILE_SIZE]
  BasicBlock *loopCheckBB = BasicBlock::Create(B.getContext(), "cStart.check", func);
  BasicBlock *loopBodyBB  = BasicBlock::Create(B.getContext(), "cStart.body",  func);
  BasicBlock *loopIncBB   = BasicBlock::Create(B.getContext(), "cStart.inc",   func);
  BasicBlock *loopExitBB  = BasicBlock::Create(B.getContext(), "cStart.exit",  func);

  // Jump to loopCheck
  B.CreateBr(loopCheckBB);
  B.SetInsertPoint(loopCheckBB);

  // PHI node: cStartPHI
  PHINode* cStartPHI = B.CreatePHI(B.getInt64Ty(), 2, "cStart");
  cStartPHI->addIncoming(B.getInt64(0), rowOkBB);

  // condition: cStartPHI < N
  Value *condC = B.CreateICmpSLT(cStartPHI, B.getInt64(N), "condC");
  B.CreateCondBr(condC, loopBodyBB, loopExitBB);

  // loopBody
  B.SetInsertPoint(loopBodyBB);
  Value *cStartVal = cStartPHI;

  // 4) Load chunk of columns from global memory => shared tile
  // barrier function
  Module *m = B.GetInsertBlock()->getModule();
  auto barrierFuncName = Intrinsic::getName(Intrinsic::nvvm_barrier0);
  auto barrierFuncTy = Intrinsic::getType(m->getContext(), Intrinsic::nvvm_barrier0);
  auto barrierFunc = m->getOrInsertFunction(barrierFuncName, barrierFuncTy);

  // iVal = threadIdx.x
  Value *threadIdx32 = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  // iVal in [0..(blockDim.x-1)]
  // assume blockDim.x == TILE_SIZE for simplicity

  // c = cStartVal + iVal
  Value *cVal = B.CreateAdd(cStartVal, B.CreateZExt(threadIdx32, B.getInt64Ty()), "cVal");
  // Check if c < N
  Value *condCInRange = B.CreateICmpULT(cVal, B.getInt64(N));

  // if in range, load from global
  BasicBlock *thenBB = BasicBlock::Create(B.getContext(), "load.inRange", func);
  BasicBlock *elseBB = BasicBlock::Create(B.getContext(), "load.outRange", func);
  BasicBlock *contBB = BasicBlock::Create(B.getContext(), "load.done",    func);

  B.CreateCondBr(condCInRange, thenBB, elseBB);

  // thenBB: do loads
  B.SetInsertPoint(thenBB);
  {
      // M[row,c]
      Value *matIdx = B.CreateAdd(
          B.CreateMul(row64, B.getInt64(N)), // row*N
          cVal,                              // + c
          "matLinearIdx"
      );
      Value *matIdx2 = B.CreateMul(matIdx, B.getInt64(2), "matLinearIdx2");

      // GEP matBasePtr + matIdx2 => re
      Value *rePtr = B.CreateGEP(scalarTy, matBasePtr, matIdx2, "matRePtr");
      // GEP +1 => im
      Value *imPtr = B.CreateGEP(scalarTy, matBasePtr,
                                  B.CreateAdd(matIdx2, B.getInt64(1)), "matImPtr");

      Value *mReVal = B.CreateLoad(scalarTy, rePtr, "mReVal");
      Value *mImVal = B.CreateLoad(scalarTy, imPtr, "mImVal");

      // store into tileM
      // tileM[2*threadIdx.x + 0] = mReVal
      Value *i64ThreadIdx = B.CreateZExt(threadIdx32, B.getInt64Ty());
      Value *tileOff = B.CreateMul(i64ThreadIdx, B.getInt64(2));
      B.CreateStore(mReVal, B.CreateGEP(scalarTy, tileMBase, tileOff));
      B.CreateStore(mImVal, B.CreateGEP(scalarTy, tileMBase,
                      B.CreateAdd(tileOff, B.getInt64(1))));

      // X[c]
      Value *xIdx2 = B.CreateMul(cVal, B.getInt64(2));
      Value *xRePtr = B.CreateGEP(scalarTy, svPtrV, xIdx2); // Xre
      Value *xImPtr = B.CreateGEP(scalarTy, svPtrV,
                          B.CreateAdd(xIdx2, B.getInt64(1))); // Xim

      Value *xReVal = B.CreateLoad(scalarTy, xRePtr, "xReVal");
      Value *xImVal = B.CreateLoad(scalarTy, xImPtr, "xImVal");

      // store into tileX
      B.CreateStore(xReVal, B.CreateGEP(scalarTy, tileXBase, tileOff));
      B.CreateStore(xImVal, B.CreateGEP(scalarTy, tileXBase,
                      B.CreateAdd(tileOff, B.getInt64(1))));

      B.CreateBr(contBB);
  }

  // elseBB: store 0
  B.SetInsertPoint(elseBB);
  {
      Value *i64ThreadIdx = B.CreateZExt(threadIdx32, B.getInt64Ty());
      Value *tileOff = B.CreateMul(i64ThreadIdx, B.getInt64(2));
      Value *zeroVal = ConstantFP::get(scalarTy, 0.0);
      // store zero into tileM
      B.CreateStore(zeroVal, B.CreateGEP(scalarTy, tileMBase, tileOff));
      B.CreateStore(zeroVal, B.CreateGEP(scalarTy, tileMBase,
                          B.CreateAdd(tileOff, B.getInt64(1))));
      // store zero into tileX
      B.CreateStore(zeroVal, B.CreateGEP(scalarTy, tileXBase, tileOff));
      B.CreateStore(zeroVal, B.CreateGEP(scalarTy, tileXBase,
                          B.CreateAdd(tileOff, B.getInt64(1))));
      B.CreateBr(contBB);
  }

  B.SetInsertPoint(contBB);

  // barrier
  B.CreateCall(barrierFunc);

  // 5) Dot-product: sum over j in [0..TILE_SIZE)
  BasicBlock *dotCheckBB = BasicBlock::Create(B.getContext(), "dot.check", func);
  BasicBlock *dotBodyBB  = BasicBlock::Create(B.getContext(), "dot.body",  func);
  BasicBlock *dotIncBB   = BasicBlock::Create(B.getContext(), "dot.inc",   func);
  BasicBlock *dotExitBB  = BasicBlock::Create(B.getContext(), "dot.exit",  func);

  // store the partial sums in PHI nodes
  B.CreateBr(dotCheckBB);
  B.SetInsertPoint(dotCheckBB);
  PHINode *jPHI     = B.CreatePHI(B.getInt64Ty(), 2, "j");
  PHINode *sumRePHI = B.CreatePHI(scalarTy, 2, "sumRePHI");
  PHINode *sumImPHI = B.CreatePHI(scalarTy, 2, "sumImPHI");

  jPHI->addIncoming(B.getInt64(0), contBB);
  sumRePHI->addIncoming(sumRe, contBB);
  sumImPHI->addIncoming(sumIm, contBB);

  Value *condJ = B.CreateICmpSLT(jPHI, B.getInt64(TILE_SIZE), "condJ");
  B.CreateCondBr(condJ, dotBodyBB, dotExitBB);

  B.SetInsertPoint(dotBodyBB);
  {
      // Mre, Mim from tileM[2*j+0..1]
      // Xre, Xim from tileX[2*j+0..1]
      Value *off2 = B.CreateMul(jPHI, B.getInt64(2));
      Value *mRePtr = B.CreateGEP(scalarTy, tileMBase, off2);
      Value *mImPtr = B.CreateGEP(scalarTy, tileMBase,
                          B.CreateAdd(off2, B.getInt64(1)));

      Value *xRePtr = B.CreateGEP(scalarTy, tileXBase, off2);
      Value *xImPtr = B.CreateGEP(scalarTy, tileXBase,
                          B.CreateAdd(off2, B.getInt64(1)));

      Value *Mre = B.CreateLoad(scalarTy, mRePtr, "Mre");
      Value *Mim = B.CreateLoad(scalarTy, mImPtr, "Mim");
      Value *Xre = B.CreateLoad(scalarTy, xRePtr, "Xre");
      Value *Xim = B.CreateLoad(scalarTy, xImPtr, "Xim");

      // (Mre + iMim)*(Xre + iXim)
      // = (Mre*Xre - Mim*Xim) + i(Mre*Xim + Mim*Xre)
      Value *prodRe = B.CreateFSub(B.CreateFMul(Mre, Xre),
                                    B.CreateFMul(Mim, Xim));
      Value *prodIm = B.CreateFAdd(B.CreateFMul(Mre, Xim),
                                    B.CreateFMul(Mim, Xre));

      // partial sums
      Value *newSumRe = B.CreateFAdd(sumRePHI, prodRe);
      Value *newSumIm = B.CreateFAdd(sumImPHI, prodIm);

      Value *nextJ = B.CreateAdd(jPHI, B.getInt64(1));
      B.CreateBr(dotIncBB);

      // dotIncBB
      B.SetInsertPoint(dotIncBB);
      jPHI    ->addIncoming(nextJ,   dotIncBB);
      sumRePHI->addIncoming(newSumRe,dotIncBB);
      sumImPHI->addIncoming(newSumIm,dotIncBB);

      B.CreateBr(dotCheckBB);
  }

  // dotExitBB
  B.SetInsertPoint(dotExitBB);
  Value *finalRe = sumRePHI;
  Value *finalIm = sumImPHI;

  // barrier again if needed
  B.CreateCall(barrierFunc);

  // Accumulate into sumRe, sumIm
  Value *sumReNow = B.CreateFAdd(sumRe, finalRe, "sumReNow");
  Value *sumImNow = B.CreateFAdd(sumIm, finalIm, "sumImNow");

  // store them back into local sumRe / sumIm variables
  sumRe = sumReNow;
  sumIm = sumImNow;

  // 6) end of chunk => next cStart = cStart + TILE_SIZE
  B.CreateBr(loopIncBB);

  // loopIncBB
  B.SetInsertPoint(loopIncBB);
  Value *addVal = B.CreateAdd(cStartVal, B.getInt64(TILE_SIZE));
  cStartPHI->addIncoming(addVal, loopIncBB);
  B.CreateBr(loopCheckBB);

  // loopExitBB
  B.SetInsertPoint(loopExitBB);

  // 7) Store final amplitude Y[row] = (sumRe, sumIm)
  // row*2 => re, row*2+1 => im
  Value *rowTimes2 = B.CreateMul(row64, B.getInt64(2));
  Value *rePtr = B.CreateGEP(scalarTy, svPtrV, rowTimes2);
  Value *imPtr = B.CreateGEP(scalarTy, svPtrV, 
                    B.CreateAdd(rowTimes2, B.getInt64(1)));

  B.CreateStore(sumRe, rePtr);
  B.CreateStore(sumIm, imPtr);

  B.CreateBr(checkEndBB);

  // checkEndBB
  B.SetInsertPoint(checkEndBB);
}


void genMatrixVectorMultiplyFromConst(
    llvm::IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const LegacyGateMatrix& gateMat,
    const llvm::ArrayRef<int> qubits,
    llvm::GlobalVariable* gConstMat,
    llvm::Value* svPtrV,
    llvm::Type* scalarTy)
{
    unsigned k = gateMat.nQubits();
    unsigned K = 1u << k;

    std::vector<llvm::Value*> reAmpPtrs(K), imAmpPtrs(K);
    std::vector<llvm::Value*> reAmps(K), imAmps(K);

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

        reAmps[i] = B.CreateLoad(scalarTy, reAmpPtrs[i], "oldRe." + std::to_string(i));
        imAmps[i] = B.CreateLoad(scalarTy, imAmpPtrs[i], "oldIm." + std::to_string(i));
    }

    // For each row r, compute the new amplitude
    for (unsigned r = 0; r < K; r++) {
        llvm::Value* updatedReAmp0 = nullptr; 
        llvm::Value* updatedReAmp1 = nullptr;
        llvm::Value* updatedImAmp = nullptr;

        llvm::Value* zeroVal = llvm::ConstantFP::get(scalarTy, 0.0);
        updatedReAmp0 = zeroVal;
        updatedReAmp1 = zeroVal;
        updatedImAmp = zeroVal;

        for (unsigned c = 0; c < K; c++) {
            // Calculate indices for real and imaginary parts
            uint64_t baseIndex = 2ull * (r * K + c);
            llvm::Value* idxReal = B.getInt64(baseIndex);
            llvm::Value* idxImag = B.getInt64(baseIndex + 1);

            // Get pointers to matrix elements in constant memory
            llvm::Value* realPtr = B.CreateGEP(
                gConstMat->getValueType(),
                gConstMat,
                {B.getInt32(0), idxReal},
                "constElemPtrRe");
            llvm::Value* imagPtr = B.CreateGEP(
                gConstMat->getValueType(),
                gConstMat,
                {B.getInt32(0), idxImag},
                "constElemPtrIm");

            // Load matrix elements
            llvm::Value* matRe = B.CreateLoad(scalarTy, realPtr, "matRe");
            llvm::Value* matIm = B.CreateLoad(scalarTy, imagPtr, "matIm");

            llvm::Value* oldRe = reAmps[c];
            llvm::Value* oldIm = imAmps[c];

            // Matrix-vector multiplication calculations
            llvm::Value* mulReRe = B.CreateFMul(matRe, oldRe);
            updatedReAmp0 = B.CreateFAdd(updatedReAmp0, mulReRe);

            llvm::Value* mulImIm = B.CreateFMul(matIm, oldIm);
            updatedReAmp1 = B.CreateFAdd(updatedReAmp1, mulImIm);

            llvm::Value* mulReIm = B.CreateFMul(matRe, oldIm);
            llvm::Value* mulImRe = B.CreateFMul(matIm, oldRe);
            llvm::Value* sumIm = B.CreateFAdd(mulReIm, mulImRe);
            updatedImAmp = B.CreateFAdd(updatedImAmp, sumIm);
        }

        // Calculate final new amplitudes
        llvm::Value* newReAmp = B.CreateFSub(updatedReAmp0, updatedReAmp1);
        llvm::Value* newImAmp = updatedImAmp;

        // Store results back to statevector
        B.CreateStore(newReAmp, reAmpPtrs[r]);
        B.CreateStore(newImAmp, imAmpPtrs[r]);
    }
}

static GlobalVariable* getOrCreateConstMatGlobal(
    Module &M, 
    Type *arrTy, 
    StringRef globalName) 
{
    // Check if already exists
    if (auto *gv = M.getNamedGlobal(globalName))
        return gv;
    
    // Create new global in address space 4
    auto* gConstMat = new GlobalVariable(
        M,
        arrTy,
        /*isConstant*/ true,
        GlobalValue::ExternalLinkage,
        UndefValue::get(arrTy), // Initialized later
        globalName,
        nullptr,
        GlobalValue::NotThreadLocal,
        4
    );
    gConstMat->setAlignment(MaybeAlign(8));
    return gConstMat;
}

CUDAKernelManager& CUDAKernelManager::genCUDAGate(
    const CUDAKernelGenConfig& config,
    std::shared_ptr<LegacyQuantumGate> gate, const std::string& funcName) {
  const unsigned k = gate->qubits.size();
  const unsigned K = 1ULL << k;
  const unsigned KK = K * K;

  LLVM_DEBUG(
    std::cerr << CYAN("=== DEBUG genGPUKernel '" << funcName << "' ===\n");
    utils::printArray(std::cerr << "Matrix on qubits ", gate->qubits) << "\n";
    gate->gateMatrix.printCMat(std::cerr) << "\n";
  );

  auto& llvmContextModulePair = 
    createNewLLVMContextModulePair(funcName + "Module");

  IRBuilder<> B(*llvmContextModulePair.llvmContext);
  assert(config.precision == 32 || config.precision == 64);
  Type* scalarTy = (config.precision == 32) ? B.getFloatTy() : B.getDoubleTy();
    
  IRArgsCUDA args;
  auto* func = getFunctionDeclarationCUDA(
    B, *llvmContextModulePair.llvmModule, funcName, config, args);

  BasicBlock* entryBB = BasicBlock::Create(
    *llvmContextModulePair.llvmContext, "entry", func);
  B.SetInsertPoint(entryBB);
  // get global tid
  auto* counterV = getGlobalTidCUDA(B);

  Value* svPtrV;
  {
    // Value* twoTimesCounter = B.CreateMul(counterV, B.getInt64(2), "times2");
    // svPtrV = B.CreateGEP(scalarTy, args.pSvArg, twoTimesCounter, "sv.ptr");
    Value* idxStartV = B.getInt64(0ULL);
    Value* tmpCounterV;
    uint64_t mask = 0ULL;
    int highestQ = gate->qubits.back();
    int qIdx = 0;
    int counterQ = 0;
    for (int q = 0; q <= highestQ; q++) {
      if (q < gate->qubits[qIdx]) {
        mask |= (1 << counterQ++);
        continue;
      }
      // q == qubits[qIdx];
      ++qIdx;
      if (mask == 0)
        continue;
      tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
      tmpCounterV = B.CreateShl(tmpCounterV, (qIdx - 1), "tmpCounter");
      idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
      // std::cerr << "  (globalThreadIdx & " << utils::as0b(mask, 32) << ") <<
      // " << (qIdx - 1) << "\n";
      mask = 0ULL;
    }
    mask = ~((1ULL << (gate->qubits.back() - k + 1)) - 1);
    // std::cerr << "  (globalThreadIdx & " << utils::as0b(mask, 32) << ") << "
    // << (k) << "\n";

    tmpCounterV = B.CreateAnd(counterV, mask, "tmpCounter");
    tmpCounterV = B.CreateShl(tmpCounterV, k, "tmpCounter");
    idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "idxStart");
    idxStartV = B.CreateShl(idxStartV, 1, "idxStart");
    svPtrV = B.CreateGEP(scalarTy, args.pSvArg, idxStartV, "sv.ptr");
  }

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
    case CUDAKernelGenConfig::UseMatImmValues: {
      const auto* gateCMat = gate->gateMatrix.getConstantMatrix();
      assert(gateCMat && "Runtime matrix incompatible with UseMatImmValues load mode");
      // genMatrixVectorMultiply(
      //     B, config, gate->gateMatrix, gate->qubits, matData, svPtrV, scalarTy);
      genMatrixVectorMultiply_SharedTiled(
          B, config, gate->gateMatrix, gate->qubits, matData, svPtrV, scalarTy);
      break;
    }
    case CUDAKernelGenConfig::LoadInDefaultMemSpace: {
      // This path loads matrix from pMatArg (args.pMatArg) in address space 1
      Value* matBasePtr = args.pMatArg;
      // Cast from address space 0 to 1:
      matBasePtr = B.CreateAddrSpaceCast(args.pMatArg,
            PointerType::get(scalarTy, /*AS=*/1), "matGlobalPtr");

      // Generate the IR that loops over the matrix elements
      // (K*K complex elements) and loads from matBasePtr + offset
      // genMatrixVectorMultiplyFromPointer(
      //     B, config, gate->gateMatrix, gate->qubits, matBasePtr, svPtrV, scalarTy);
      genMatrixVectorMultiplyFromPointer_SharedTiled(
          B, config, gate->gateMatrix, gate->qubits, matBasePtr, svPtrV, scalarTy);
      break;
    }
    case CUDAKernelGenConfig::LoadInConstMemSpace: {
      std::cerr << "***USE CONSTMEMSPACE\n";
      // This path loads the matrix from pointer in address space 4 (constant mem).
      unsigned totalElems = 2 * (K * K);
      llvm::ArrayType *arrTy = llvm::ArrayType::get(scalarTy, totalElems);
      auto *gConstMat = getOrCreateConstMatGlobal(
          *llvmContextModulePair.llvmModule,
          arrTy,
          "gConstMatShared");

      // Initialize with matrix values if available
      if (auto cMat = gate->gateMatrix.getConstantMatrix()) {
          std::vector<llvm::Constant*> constElems;
          for (unsigned i = 0; i < K*K; i++) {
              constElems.push_back(ConstantFP::get(scalarTy, cMat->data()[i].real()));
              constElems.push_back(ConstantFP::get(scalarTy, cMat->data()[i].imag()));
          }
          gConstMat->setInitializer(ConstantArray::get(arrTy, constElems));
      } else {
          // For symbolic gates, need to initialize with proper values
          // should initialize with the actual Rz matrix values
          std::vector<llvm::Constant*> constElems;
          for (unsigned i = 0; i < K*K; i++) {
              double real = (i/K == i%K) ? 1.0 : 0.0;
              double imag = 0.0;
              constElems.push_back(ConstantFP::get(scalarTy, real));
              constElems.push_back(ConstantFP::get(scalarTy, imag));
          }
          gConstMat->setInitializer(ConstantArray::get(arrTy, constElems));
      }

      genMatrixVectorMultiplyFromConst(
          B, config, gate->gateMatrix, gate->qubits, 
          gConstMat, svPtrV, scalarTy);
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

CUDAKernelManager& CUDAKernelManager::genCUDAGatesFromLegacyCircuitGraph(
    const CUDAKernelGenConfig& config,
    const LegacyCircuitGraph& graph, const std::string& graphName) {
  const auto allBlocks = graph.getAllBlocks();
  const auto mangledName = internal::mangleGraphName(graphName);
  for (const auto& block : allBlocks) {
    genCUDAGate(
      config, block->quantumGate, mangledName + std::to_string(block->id));
  }
  return *this;
}


// Attempt to implement kernel sharing for gates that meet the conditions
static constexpr double SIMILARITY_THRESHOLD = 0.85;
static constexpr unsigned MAX_GATES_PER_KERNEL = 4;
static constexpr unsigned MAX_QUBITS_PER_GATE = 8;

double qubitSimilarity(const LegacyQuantumGate &A, const LegacyQuantumGate &B) {
    // Convert each qubit list to a std::set.
    std::set<int> setA(A.qubits.begin(), A.qubits.end());
    std::set<int> setB(B.qubits.begin(), B.qubits.end());

    // Compute intersection
    std::vector<int> interAB;
    std::set_intersection(
        setA.begin(), setA.end(),
        setB.begin(), setB.end(),
        std::back_inserter(interAB)
    );

    // Compute union
    std::vector<int> unionAB;
    std::set_union(
        setA.begin(), setA.end(),
        setB.begin(), setB.end(),
        std::back_inserter(unionAB)
    );

    if (unionAB.empty()) {
        return 0.0;
    }
    return double(interAB.size()) / double(unionAB.size());
}

static Function* getMultiKernelDeclaration(
    IRBuilder<>& B, 
    Module& M,
    const std::string& funcName
) {
    FunctionType* fty = FunctionType::get(B.getVoidTy(), { B.getPtrTy() }, false);
    auto* func = Function::Create(
        fty,
        Function::ExternalLinkage,
        funcName,
        M
    );

    // Mark as a kernel
    auto& ctx = M.getContext();
    auto* mdString = MDString::get(ctx, "kernel");
    auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
    auto* kernelMetadata = MDNode::get(ctx, {
        ValueAsMetadata::get(func), mdString, mdOne
    });
    M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(kernelMetadata);

    // Name the argument
    func->getArg(0)->setName("p.sv");
    return func;
}

/**
 * Only group gates that pass threshold for qubitSimilarity, MAX_QUBITS_PER_GATE, and cap the group at MAX_GATES_PER_KERNEL.
 */
std::vector<std::vector<std::shared_ptr<LegacyQuantumGate>>>
groupGatesByOverlapAndSize(const std::vector<std::shared_ptr<LegacyQuantumGate>>& allGates)
{
  std::vector<std::vector<std::shared_ptr<LegacyQuantumGate>>> groups;
  groups.reserve(allGates.size());

  for (auto &g : allGates) {
    // also skip if gate is too large
    if (g->nQubits() > MAX_QUBITS_PER_GATE) {
      // put it alone in its own group
      groups.push_back({ g });
      continue;
    }

    bool placed = false;
    for (auto &grp : groups) {
      if (grp.size() >= MAX_GATES_PER_KERNEL) continue; // group full
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
    const std::vector<std::shared_ptr<LegacyQuantumGate>>& gateList,
    const std::string& funcName
)
{
  // 1) create new module
  auto& cmp = createNewLLVMContextModulePair(funcName + "_Module");
  IRBuilder<> B(*cmp.llvmContext);
  Module& M = *cmp.llvmModule;

  // 2) Build a function: “__global__ void kernel(double *pSv)”
  FunctionType* fty = FunctionType::get(B.getVoidTy(), { B.getPtrTy() }, false);
  Function* func = Function::Create(
      fty,
      Function::ExternalLinkage,
      funcName,
      &M  // place it in module M
  );

  // Mark as a kernel via metadata
  {
    auto& ctx = M.getContext();
    auto* mdString = MDString::get(ctx, "kernel");
    auto* mdOne    = ConstantAsMetadata::get(B.getInt32(1));
    auto* mdNode   = MDNode::get(ctx, {
        ValueAsMetadata::get(func), mdString, mdOne
    });
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
  for (auto &g : gateList) {
    // (A) Create the bit-mask pointer. This replicates the single-gate “idxStartV” logic.
    const auto& qubits = g->qubits;
    int k = (int) qubits.size();
    int highestQ = qubits.back();

    // need the scalar type for the GEP
    Type* scalarTy = (config.precision == 32) ? B.getFloatTy() : B.getDoubleTy();

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
          tmpCounterV = B.CreateAnd(threadIdx64, B.getInt64(mask), "tmpCounter");
          // shift by (qIdx - 1)
          tmpCounterV = B.CreateShl(tmpCounterV, B.getInt64(qIdx - 1), "tmpCounter");
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
    Value* maskedSvPtr = B.CreateGEP(scalarTy, pSvArg, idxStartV, "maskedSvPtr");

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
        scalarTy
    );

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
  _cudaKernels.emplace_back(
    CUDAKernelInfo::PTXStringType(),
    config.precision,
    func->getName().str(),
    nullptr, // no single gate
    emptyCT,
    totalOps
  );

  return *this;
}

CUDAKernelManager& CUDAKernelManager::genCUDAGatesFromLegacyCircuitGraphMulti(
    const CUDAKernelGenConfig& config,
    const LegacyCircuitGraph& graph,
    const std::string& graphName)
{
  // gather all gates
  std::vector<std::shared_ptr<LegacyQuantumGate>> allGates;
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

void CUDAKernelManager::dumpPTX(const std::string &kernelName, llvm::raw_ostream &os) {
  // First check if we have already compiled to PTX
  for (auto &kernelInfo : _cudaKernels) {
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
  for (auto &cmp : llvmContextModulePairs) {
    auto &M = *cmp.llvmModule;
    if (auto *F = M.getFunction(kernelName)) {
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
      auto targetMachine = target->createTargetMachine(
          targetTriple, "sm_70", "+ptx60", opt, RM);

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

      if (targetMachine->addPassesToEmitFile(pass, ptxStream,
                                           nullptr, fileType)) {
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
#include <llvm/IR/IntrinsicsNVPTX.h>

#include "cast/QuantumGate.h"
#include "cast/CircuitGraph.h"

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
    IRBuilder<>& B, const GateMatrix& gateMatrix,
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

void genMatrixVectorMultiply(
    llvm::IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const GateMatrix& gateMat,
    const llvm::ArrayRef<int> qubits,
    const std::vector<IRMatDataCUDA>& matData,
    llvm::Value* svPtrV,
    llvm::Type* scalarTy)
{
    // For k-qubit gate, dimension = 1 << k
    unsigned k = gateMat.nQubits();
    unsigned K = 1u << k;

    // Load old amplitudes from svPtrV
    std::vector<llvm::Value*> reAmpPtrs(K), imAmpPtrs(K);
    std::vector<llvm::Value*> reAmps(K), imAmps(K);

    for (unsigned i = 0; i < K; i++) {
        uint64_t delta = 0;
        for (unsigned b = 0; b < k; b++) {
            if (i & (1ull << b)) {
                delta |= (1ull << qubits[b]);
            }
        }
        // Real pointer = svPtrV + (2 * delta)
        reAmpPtrs[i] = B.CreateConstGEP1_64(scalarTy, svPtrV, 2*delta, "reAmpPtr"+std::to_string(i));
        imAmpPtrs[i] = B.CreateConstGEP1_64(scalarTy, svPtrV, 2*delta+1, "imAmpPtr"+std::to_string(i));

        reAmps[i] = B.CreateLoad(scalarTy, reAmpPtrs[i], "oldRe."+std::to_string(i));
        imAmps[i] = B.CreateLoad(scalarTy, imAmpPtrs[i], "oldIm."+std::to_string(i));
    }

    // For each row r in [0..K-1], accumulate
    for (unsigned r = 0; r < K; ++r) {
        // build partial sums for real and imaginary
        llvm::Value* zeroVal = llvm::ConstantFP::get(scalarTy, 0.0);
        llvm::Value* updatedReAmp0 = zeroVal; // sum( matRe[r,c]* oldRe[c] )
        llvm::Value* updatedReAmp1 = zeroVal; // sum( matIm[r,c]* oldIm[c] )
        llvm::Value* updatedImAmp  = zeroVal; // sum of cross terms

        for (unsigned c = 0; c < K; ++c) {
            // Gather matrix data for row r, col c
            const auto& md = matData[r*K + c];

            llvm::Value* oldRe = reAmps[c];
            llvm::Value* oldIm = imAmps[c];

            // updatedReAmp0 += matReVal*oldRe
            llvm::Value* mulReRe = genOptFMul(md.reVal, oldRe, md.reKind, B);
            if (mulReRe) {
                updatedReAmp0 = B.CreateFAdd(updatedReAmp0, mulReRe);
            }

            // updatedReAmp1 += matImVal*oldIm
            llvm::Value* mulImIm = genOptFMul(md.imVal, oldIm, md.imKind, B);
            if (mulImIm) {
                updatedReAmp1 = B.CreateFAdd(updatedReAmp1, mulImIm);
            }

            // updatedImAmp += (matReVal*oldIm + matImVal*oldRe)
            llvm::Value* sumIm = zeroVal;

            llvm::Value* mulReIm = genOptFMul(md.reVal, oldIm, md.reKind, B);
            if (mulReIm) {
                sumIm = B.CreateFAdd(sumIm, mulReIm);
            }
            llvm::Value* mulImRe = genOptFMul(md.imVal, oldRe, md.imKind, B);
            if (mulImRe) {
                sumIm = B.CreateFAdd(sumIm, mulImRe);
            }
            updatedImAmp = B.CreateFAdd(updatedImAmp, sumIm);
        }

        // final real = updatedReAmp0 - updatedReAmp1
        llvm::Value* newReAmp = B.CreateFSub(updatedReAmp0, updatedReAmp1);
        llvm::Value* newImAmp = updatedImAmp;

        // store back
        B.CreateStore(newReAmp, reAmpPtrs[r]);
        B.CreateStore(newImAmp, imAmpPtrs[r]);
    }
}

void genMatrixVectorMultiplyFromPointer(
    llvm::IRBuilder<>& B,
    const CUDAKernelGenConfig& config,
    const GateMatrix& gateMat,
    const llvm::ArrayRef<int> qubits,
    llvm::Value* matBasePtr,
    llvm::Value* svPtrV,
    llvm::Type* scalarTy)
{
    unsigned k = gateMat.nQubits();
    unsigned K = 1u << k;

    std::vector<llvm::Value*> reAmpPtrs(K), imAmpPtrs(K);
    std::vector<llvm::Value*> reAmps(K),    imAmps(K);

    for (unsigned i = 0; i < K; i++) {
        uint64_t delta = 0;
        for (unsigned b = 0; b < k; b++) {
            if (i & (1u << b))
                delta |= (1ull << qubits[b]);
        }
        // pointer arithmetic for real, imag
        reAmpPtrs[i] = B.CreateConstGEP1_64(
            scalarTy, svPtrV, 2 * delta, "reAmpPtr." + std::to_string(i));
        imAmpPtrs[i] = B.CreateConstGEP1_64(
            scalarTy, svPtrV, 2 * delta + 1, "imAmpPtr." + std::to_string(i));

        reAmps[i] = B.CreateLoad(scalarTy, reAmpPtrs[i], "oldRe." + std::to_string(i));
        imAmps[i] = B.CreateLoad(scalarTy, imAmpPtrs[i], "oldIm." + std::to_string(i));
    }

    // For each row r, compute the new amplitude
    for (unsigned r = 0; r < K; ++r) {
        llvm::Value* updatedReAmp0 = nullptr; 
        llvm::Value* updatedReAmp1 = nullptr;
        llvm::Value* updatedImAmp  = nullptr;

        llvm::Value* zeroVal = llvm::ConstantFP::get(scalarTy, 0.0);
        updatedReAmp0 = zeroVal;
        updatedReAmp1 = zeroVal;
        updatedImAmp  = zeroVal;

        for (unsigned c = 0; c < K; ++c) {
            // read the real & imag parts from matBasePtr
            uint64_t baseIndex = 2ull * (r * K + c);
            llvm::Value* idxReal = B.getInt64(baseIndex + 0);
            llvm::Value* idxImag = B.getInt64(baseIndex + 1);

            llvm::Value* realPtr = B.CreateGEP(scalarTy, matBasePtr, idxReal, "matRePtr");
            llvm::Value* imagPtr = B.CreateGEP(scalarTy, matBasePtr, idxImag, "matImPtr");

            llvm::Value* matRe = B.CreateLoad(scalarTy, realPtr, "matRe");
            llvm::Value* matIm = B.CreateLoad(scalarTy, imagPtr, "matIm");

            llvm::Value* oldRe = reAmps[c];
            llvm::Value* oldIm = imAmps[c];

            // updatedReAmp0 += matRe*oldRe
            llvm::Value* mulReRe = B.CreateFMul(matRe, oldRe);
            updatedReAmp0 = B.CreateFAdd(updatedReAmp0, mulReRe);

            // updatedReAmp1 += matIm*oldIm
            llvm::Value* mulImIm = B.CreateFMul(matIm, oldIm);
            updatedReAmp1 = B.CreateFAdd(updatedReAmp1, mulImIm);

            // updatedImAmp += matRe*oldIm + matIm*oldRe
            llvm::Value* mulReIm = B.CreateFMul(matRe, oldIm);
            llvm::Value* mulImRe = B.CreateFMul(matIm, oldRe);
            llvm::Value* sumIm   = B.CreateFAdd(mulReIm, mulImRe);
            updatedImAmp  = B.CreateFAdd(updatedImAmp, sumIm);
        }

        // final real = updatedReAmp0 - updatedReAmp1
        llvm::Value* newReAmp = B.CreateFSub(updatedReAmp0, updatedReAmp1);
        llvm::Value* newImAmp = updatedImAmp;

        // store them
        B.CreateStore(newReAmp, reAmpPtrs[r]);
        B.CreateStore(newImAmp, imAmpPtrs[r]);
    }
}

void genMatrixVectorMultiplyFromPointer_SharedTiled(
    llvm::IRBuilder<>           &B,
    const CUDAKernelGenConfig   &config,
    const GateMatrix            &gateMat,
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

  // rowOkBB
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
  auto barrierFunc = Intrinsic::getOrInsertDeclaration(
      m, Intrinsic::nvvm_barrier0
  );

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
    const GateMatrix& gateMat,
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
    std::shared_ptr<QuantumGate> gate, const std::string& funcName) {
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
  // unsigned asArg = llvm::cast<llvm::PointerType>(args.pMatArg->getType())->getAddressSpace();

  switch (config.matrixLoadMode) {
    case CUDAKernelGenConfig::UseMatImmValues: {
      const auto* gateCMat = gate->gateMatrix.getConstantMatrix();
      assert(gateCMat && "Runtime matrix incompatible with UseMatImmValues load mode");
      genMatrixVectorMultiply(
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

CUDAKernelManager& CUDAKernelManager::genCUDAGatesFromCircuitGraph(
    const CUDAKernelGenConfig& config,
    const CircuitGraph& graph, const std::string& graphName) {
  const auto allBlocks = graph.getAllBlocks();
  const auto mangledName = internal::mangleGraphName(graphName);
  for (const auto& block : allBlocks) {
    genCUDAGate(
      config, block->quantumGate, mangledName + std::to_string(block->id));
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
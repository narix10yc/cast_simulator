#define DEBUG_TYPE "codegen-cpu"
#include <llvm/Support/Debug.h>
// #define LLVM_DEBUG(X) X
#include <llvm/IR/IntrinsicsX86.h>

#include "cast/QuantumGate.h"
#include "cast/CircuitGraph.h"
#include "simulation/KernelManager.h"
#include "simulation/KernelGenInternal.h"

#include "utils/iocolor.h"
#include "utils/utils.h"
#include "utils/Formats.h"
#include "utils/PODVector.h"

#include <cmath>

using namespace llvm;
using namespace cast;

namespace {

struct CPUArgs {
  Argument* pSvArg;       // ptr to statevector
  Argument* ctrBeginArg;  // counter begin
  Argument* ctrEndArg;    // counter end
  Argument* pMatArg;      // ptr to matrix
};

Function* cpuGetFunctionDeclaration(
    IRBuilder<>& B, Module& M, const std::string& funcName,
    const CPUKernelGenConfig& config, CPUArgs& args) {
  SmallVector<Type*> argType {
    B.getPtrTy(), B.getInt64Ty(), B.getInt64Ty(), B.getPtrTy()
  };

  auto* funcTy = FunctionType::get(B.getVoidTy(), argType, false);
  auto* func = Function::Create(funcTy, Function::ExternalLinkage, funcName, M);

  args.pSvArg = func->getArg(0);
  args.pSvArg->setName("p.sv.arg");
  args.ctrBeginArg = func->getArg(1);
  args.ctrBeginArg->setName("ctr.begin");
  args.ctrEndArg = func->getArg(2);
  args.ctrEndArg->setName("ctr.end");
  args.pMatArg = func->getArg(3);
  args.pMatArg->setName("pmat");

  return func;
}

struct IRMatDataCUDA {
  Value* reElemVal;
  Value* imElemVal;
  Value* reVecVal;
  Value* imVecVal;
  ScalarKind reKind;
  ScalarKind imKind;
};

inline std::vector<IRMatDataCUDA> getMatrixData(
    IRBuilder<>& B, const LegacyGateMatrix& gateMatrix,
    const CPUKernelGenConfig& config) {
  const int k = gateMatrix.nQubits();
  const unsigned K = 1 << k;
  const unsigned KK = K * K;

  std::vector<IRMatDataCUDA> data(KK);

  const auto* cMat = gateMatrix.getConstantMatrix();
  // assert(cMat && "Parametrized matrices codegen not implemented yet");

  // Have compile-time known matrix
  if (cMat != nullptr && !config.forceDenseKernel) {
    const double zTol = config.zeroTol / K;
    const double oTol = config.oneTol / K;

    for (unsigned i = 0; i < KK; i++) {
			auto real = cMat->data()[i].real();
			auto imag = cMat->data()[i].imag();

			if (std::abs(real) < zTol)
				data[i].reKind = SK_Zero;
			else if (std::abs(real - 1.0) < oTol)
				data[i].reKind = SK_One;
			else if (std::abs(real + 1.0) < oTol)
				data[i].reKind = SK_MinusOne;
			else if (config.matrixLoadMode == CPUKernelGenConfig::UseMatImmValues) {
				data[i].reKind = SK_ImmValue;
				data[i].reElemVal = ConstantFP::get(B.getContext(), APFloat(real));
			}
			else
				data[i].reKind = SK_General;

			if (std::abs(imag) < zTol)
				data[i].imKind = SK_Zero;
			else if (std::abs(imag - 1.0) < oTol)
				data[i].imKind = SK_One;
			else if (std::abs(imag + 1.0) < oTol)
				data[i].imKind = SK_MinusOne;
			else if (config.matrixLoadMode == CPUKernelGenConfig::UseMatImmValues) {
				data[i].imKind = SK_ImmValue;
				data[i].imElemVal = ConstantFP::get(B.getContext(), APFloat(imag));
			}
			else
				data[i].imKind = SK_General;
		}
  } else { // runtime / param matrix - no compile-time numeric classification possible
			for (unsigned i = 0; i < KK; i++) {
				data[i].reKind = SK_General;
				data[i].imKind = SK_General;
    }
	}
  return data;
}

} // anonymous namespace

CPUKernelManager& CPUKernelManager::genCPUGate(
    const CPUKernelGenConfig& config,
    std::shared_ptr<QuantumGate> gate, const std::string& funcName) {
  const unsigned s = config.simd_s;
  const unsigned S = 1ULL << s;
  const unsigned k = gate->qubits.size();
  const unsigned K = 1ULL << k;
  const unsigned KK = K * K;

  LLVM_DEBUG(
    std::cerr << CYAN("=== DEBUG genCPUKernel funcName " << funcName << " ===\n");
    utils::printArray(std::cerr << "Matrix on qubits ", gate->qubits) << "\n";
    gate->gateMatrix.printCMat(std::cerr) << "\n";
  );

  auto& llvmContextModulePair = createNewLLVMContextModulePair(funcName + "Module");

  IRBuilder<> B(*llvmContextModulePair.llvmContext);
  assert(config.precision == 32 || config.precision == 64);
  Type* scalarTy = (config.precision == 32) ? B.getFloatTy() : B.getDoubleTy();
  
  // init function
  CPUArgs args;
  Function* func = cpuGetFunctionDeclaration(
    B, *llvmContextModulePair.llvmModule, funcName, config, args);

  // init matrix
  auto matrixData = getMatrixData(B, gate->gateMatrix, config);

	bool haveConstantMatrix = (gate->gateMatrix.getConstantMatrix() != nullptr); // TODO: change this to be less redundant

  // init basic blocks
  BasicBlock* entryBB = BasicBlock::Create(B.getContext(), "entry", func);
  BasicBlock* loopBB = BasicBlock::Create(B.getContext(), "loop", func);
  BasicBlock* loopBodyBB = BasicBlock::Create(B.getContext(), "loop.body", func);
  BasicBlock* retBB = BasicBlock::Create(B.getContext(), "ret", func);

  // split qubits
  int sepBit;
  SmallVector<int, 6U> simdBits, hiBits, loBits;
  { /* split qubits */
  unsigned q = 0;
  auto qubitsIt = gate->qubits.begin();
  const auto qubitsEnd = gate->qubits.end();
  while (simdBits.size() != s) {
    if (qubitsIt != qubitsEnd && *qubitsIt == q) {
      loBits.push_back(q);
      ++qubitsIt;
    } else {
      simdBits.push_back(q);
    }
    ++q;
  }
  while (qubitsIt != qubitsEnd) {
      hiBits.push_back(*qubitsIt);
      ++qubitsIt;
  }

  for (auto& b : loBits) {
    if (b >= s)
      ++b;
  }
  for (auto& b : simdBits) {
    if (b >= s)
      ++b;
  }
  for (auto& b : hiBits) {
    if (b >= s)
      ++b;
  }

  sepBit = (s == 0) ? 0 : simdBits.back() + 1;
  if (sepBit == s)
    ++sepBit;
  }

  const unsigned vecSize = 1U << sepBit;
  auto* vecType = VectorType::get(scalarTy, vecSize, false);

  const unsigned lk = loBits.size();
  const unsigned LK = 1 << lk;
  const unsigned hk = hiBits.size();
  const unsigned HK = 1 << hk;

  // debug print qubit splits
  LLVM_DEBUG(
    dbgs() << CYAN("-- qubit split done\n");
    utils::printArray(std::cerr << "- lower bits:  ", loBits) << "\n";
    utils::printArray(std::cerr << "- higher bits: ", hiBits) << "\n";
    utils::printArray(std::cerr << "- simd bits:   ", simdBits) << "\n";
    dbgs() << "- reImBit (simd_s): " << s << "\n";
    dbgs() << "- sepBit:           " << sepBit << "\n";
    dbgs() << "- vecSize:          " << vecSize << "\n";
  );
  
  B.SetInsertPoint(entryBB);
  // load matrix (if needed)
  switch (config.matrixLoadMode) {
  case CPUKernelGenConfig::UseMatImmValues: {
    // Imm values are LLVM Constant. They will not appear as instructions in 
    // the entry block.
		if (!haveConstantMatrix) {
      llvm_unreachable("Asked to use immediate values but we have no cMat");
    }
    for (unsigned i = 0, n = matrixData.size(); i < n; i++) {
      if (matrixData[i].reElemVal) {
        matrixData[i].reVecVal = B.CreateVectorSplat(
          S, matrixData[i].reElemVal, "mat.re." + std::to_string(i) + ".vec");
      }
      if (matrixData[i].imElemVal) {
        matrixData[i].imVecVal = B.CreateVectorSplat(
          S, matrixData[i].imElemVal, "mat.im." + std::to_string(i) + ".vec");
      }
    }
    break;
  }
  case CPUKernelGenConfig::StackLoadMatElems: {
    for (unsigned i = 0, n = matrixData.size(); i < n; i++) {
      if (matrixData[i].reKind == SK_General) {
        auto* ptrV = B.CreateConstGEP1_32(
          scalarTy, args.pMatArg, 2 * i, "p.mat.re." + std::to_string(i));
        matrixData[i].reElemVal = B.CreateLoad(
          scalarTy, ptrV, "mat.re." + std::to_string(i));
        matrixData[i].reVecVal = B.CreateVectorSplat(
          S, matrixData[i].reElemVal, "mat.re." + std::to_string(i) + ".vec");
      }
      if (matrixData[i].imKind == SK_General) {
        auto* ptrV = B.CreateConstGEP1_32(
          scalarTy, args.pMatArg, 2 * i + 1, "p.mat.im." + std::to_string(i));
        matrixData[i].imElemVal = B.CreateLoad(
          scalarTy, ptrV, "mat.im." + std::to_string(i));
        matrixData[i].imVecVal = B.CreateVectorSplat(
          S, matrixData[i].imElemVal, "mat.im." + std::to_string(i) + ".vec");
      }
    }
    break;
  }
  default: {
    llvm_unreachable(
      "Unrecognized matrixLoadMode - Only supporting UseMatImmValues and StackLoadMatElems modes");
    break;
  }
  }

  // entryBB->print(errs());

  B.CreateBr(loopBB);

  // loop entry: set up counter
  B.SetInsertPoint(loopBB);
  PHINode* taskIdV = B.CreatePHI(B.getInt64Ty(), 2, "taskid");
  taskIdV->addIncoming(args.ctrBeginArg, entryBB);
  Value* cond = B.CreateICmpSLT(taskIdV, args.ctrEndArg, "cond");
  B.CreateCondBr(cond, loopBodyBB, retBB);

  // loop body
  B.SetInsertPoint(loopBodyBB);
  
  // the start pointer in the SV based on taskID
  Value* ptrSvBeginV = nullptr;
  if (hiBits.empty()) {
    ptrSvBeginV = B.CreateGEP(vecType, args.pSvArg, taskIdV, "ptr.sv.begin");
  } else {
    // the shift from args.pSvArg in the unit of vecTypeX2
    Value* idxStartV = B.getInt64(0ULL);
    Value* tmpCounterV;
    uint64_t mask = 0ULL;
    int highestQ = hiBits.back();
    int qIdx = 0;
    int counterQ = 0;
    for (int q = sepBit; q <= highestQ; q++) {
      if (q < hiBits[qIdx]) {
        mask |= (1 << counterQ++);
        continue;
      }
      ++qIdx;
      if (mask == 0)
        continue;
      tmpCounterV = B.CreateAnd(taskIdV, mask, "tmp.taskid");
      tmpCounterV = B.CreateShl(tmpCounterV, (qIdx - 1), "tmp.taskid");
      idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "tmp.idx.begin");
      LLVM_DEBUG(
        std::cerr << "  (taskID & " << utils::fmt_0b(mask, highestQ) << ") << "
                  << (qIdx - 1) << "\n";
      );
      mask = 0ULL;
    }
    mask = ~((1ULL << (highestQ - sepBit - hk + 1)) - 1);
    LLVM_DEBUG(
      std::cerr << "  (taskID & " << utils::fmt_0b(mask, 16) << ") << "
                << hk << "\n";
    );

    tmpCounterV = B.CreateAnd(taskIdV, mask, "tmp.taskid");
    tmpCounterV = B.CreateShl(tmpCounterV, hk, "tmp.taskid");
    idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "idx.begin");
    ptrSvBeginV = B.CreateGEP(vecType, args.pSvArg, idxStartV, "ptr.sv.begin");
  }

  /* load amplitude registers
    There are a total of 2K size-S amplitude registers (K real and K imag).
    In Alt Format, we load HK size-(2*S*LK) LLVM registers. i.e. Loop over
    higher qubits.
    There are two stages of shuffling (splits)
    - Stage 1:
      Each size-(2*S*LK) reg is shuffled into 2 size-(S*LK) regs, the reFull
      and imFull. There are a total of (2 * HK) reFull and imFull each.
    - Stage 2:
      Each reFull (resp. imFull) is shuffled into LK size-S regs, the reAmps
      (resp. imAmps) vector.
  */

  // [re/im]SplitMasks are like flattened LK*S matrices
  SmallVector<int> reSplitMasks;
  SmallVector<int> imSplitMasks;
  {
  unsigned pdepMaskS = 0U;
  int pdepNbitsS = simdBits.empty() ? 0 : simdBits.back() + 1;
  for (const auto& b : simdBits)
    pdepMaskS |= (1U << b);
  unsigned pdepMaskL = 0U;
  int pdepNbitsL = loBits.empty() ? 0 : loBits.back() + 1;
  for (const auto& b : loBits)
    pdepMaskL |= (1U << b);
  reSplitMasks.resize_for_overwrite(LK * S);
  imSplitMasks.resize_for_overwrite(LK * S);
  // TODO: Optimize this double-loop
  for (unsigned li = 0; li < LK; li++) {
    for (unsigned si = 0; si < S; si++) {
      reSplitMasks[li * S + si] = utils::pdep32(li, pdepMaskL, pdepNbitsL) |
                                  utils::pdep32(si, pdepMaskS, pdepNbitsS);
      imSplitMasks[li * S + si] = reSplitMasks[li * S + si] | (1 << s);
    }
  }
  LLVM_DEBUG(
    std::cerr << "- reSplitMasks: [";
    for (const auto& e : reSplitMasks)
      std::cerr << utils::fmt_0b(e, sepBit + 1) << ",";
    std::cerr << "]\n";
    std::cerr << "- imSplitMasks: [";
    for (const auto& e : imSplitMasks)
      std::cerr << utils::fmt_0b(e, sepBit + 1) << ",";
    std::cerr << "]\n";
  );
  } // end init [re/im]SplitMasks

  // load vectors
  SmallVector<Value*> reAmps; // real amplitudes
  SmallVector<Value*> imAmps; // imag amplitudes
  reAmps.resize_for_overwrite(K);
  imAmps.resize_for_overwrite(K);

  SmallVector<Value*> pSvs;
  pSvs.resize_for_overwrite(HK);
  for (unsigned hi = 0; hi < HK; hi++) {
    uint64_t idxShift = 0ULL;
    for (unsigned hBit = 0; hBit < hk; hBit++) {
      if (hi & (1 << hBit))
        idxShift += 1ULL << hiBits[hBit];
    }
    idxShift >>= sepBit;
    LLVM_DEBUG(
      std::cerr << "hi = " << hi << ": idxShift = "
                << utils::fmt_0b(idxShift, hiBits.empty() ? 1 : hiBits.back())
                << "\n";
    );
    pSvs[hi] = B.CreateConstGEP1_64(
      vecType, ptrSvBeginV, idxShift, "ptr.sv.hi." + std::to_string(hi));
    auto* ampFull = B.CreateLoad(
      vecType, pSvs[hi], "sv.full.hi." + std::to_string(hi));

    for (unsigned li = 0; li < LK; li++) {
      reAmps[hi * LK + li] = B.CreateShuffleVector(
        ampFull, ArrayRef<int>(reSplitMasks.data() + li * S, S),
        "re." + std::to_string(hi) + "." + std::to_string(li));
      imAmps[hi * LK + li] = B.CreateShuffleVector(
        ampFull, ArrayRef<int>(imSplitMasks.data() + li * S, S),
        "im." + std::to_string(hi) + "." + std::to_string(li));
    }
  }

  std::vector<std::vector<int>> mergeMasks;
  mergeMasks.reserve(lk);
  std::vector<int> reimMergeMask;
  reimMergeMask.reserve(vecSize);
  {
  int idxL, idxR;
  unsigned lCached; // length of cached array
  std::vector<int> arr0(LK * S), arr1(LK * S), arr2(LK * S);
  std::vector<int>& cacheLHS = arr0, &cacheRHS = arr1, &cacheCombined = arr2;
  std::memcpy(arr0.data(), reSplitMasks.data(),     S * sizeof(int));
  std::memcpy(arr1.data(), reSplitMasks.data() + S, S * sizeof(int));
  int roundIdx = 0;
  while (roundIdx < lk) {
    LLVM_DEBUG(
      std::cerr << "Round " << roundIdx << ": ";
      utils::printArray(std::cerr, llvm::ArrayRef(cacheLHS)) << " and ";
      utils::printArray(std::cerr, llvm::ArrayRef(cacheRHS)) << "\n";
    );

    lCached = S << roundIdx;
    mergeMasks.emplace_back(lCached << 1);
    auto& mask = mergeMasks.back();
    
    idxL = 0; idxR = 0;
    for (int idxCombined = 0; idxCombined < (lCached << 1); idxCombined++) {
      if (idxL == lCached) {
        // append cacheRHS[idxR:] to cacheCombined
        while (idxR < lCached) {
          mask[idxCombined] = idxR + lCached;
          cacheCombined[idxCombined++] = cacheRHS[idxR++];
        }
        break;
      }
      if (idxR == lCached) {
        // append cacheLHS[idxL:] to cacheCombined
        while (idxL < lCached) {
          mask[idxCombined] = idxL;
          cacheCombined[idxCombined++] = cacheLHS[idxL++];
        }
        break;
      }
      if (cacheLHS[idxL] < cacheRHS[idxR]) {
        mask[idxCombined] = idxL;
        cacheCombined[idxCombined] = cacheLHS[idxL];
        ++idxL;
      }
      else {
        mask[idxCombined] = idxR + lCached;
        cacheCombined[idxCombined] = cacheRHS[idxR];
        ++idxR;
      }
    }
    LLVM_DEBUG(
      utils::printArray(std::cerr << "  Cache Combined: ", llvm::ArrayRef(cacheCombined)) << "\n";
      utils::printArray(std::cerr << "  Mask: ", llvm::ArrayRef(mask)) << "\n";
    );
    // rotate the assignments of
    // (cacheLHS, cacheRHS, cacheCombined) with (arr0, arr1, arr2)
    if (++roundIdx == lk)
      break;
    cacheLHS = cacheCombined;
    if (cacheLHS == arr2) {
      cacheRHS = arr0;
      cacheCombined = arr1;
    } else if (cacheLHS == arr1) {
      cacheRHS = arr2;
      cacheCombined = arr0;
    } else {
      assert(cacheLHS == arr0);
      cacheRHS = arr1;
      cacheCombined = arr2;
    }
    for (int i = 0; i < (lCached << 1); i++) {
      assert((cacheLHS[i] & (1 << loBits[roundIdx])) == 0);
      cacheRHS[i] = cacheLHS[i] | (1 << loBits[roundIdx]);
    }
  } // end while

  // init reimMergeMask
  for (int pairIdx = 0; pairIdx < (vecSize >> s >> 1); pairIdx++) {
    for (int i = 0; i < S; i++)
      reimMergeMask.push_back(S * pairIdx + i);
    for (int i = 0; i < S; i++)
      reimMergeMask.push_back(S * pairIdx + i + (vecSize >> 1));
  }
  LLVM_DEBUG(
    utils::printArray(
      std::cerr << "reimMergeMask: ",llvm::ArrayRef(reimMergeMask)) << "\n";
    std::cerr << CYAN("- Merged masks initiated\n");
  );
  }

  std::vector<Value*> updatedReAmps(LK);
  std::vector<Value*> updatedImAmps(LK);
  for (unsigned hi = 0; hi < HK; hi++) {
    // mat-vec mul
    for (auto& v : updatedReAmps)
      v = nullptr;
    for (auto& v : updatedImAmps)
      v = nullptr;
    for (unsigned li = 0; li < LK; li++) {
      unsigned r = hi * LK + li; // row
      for (unsigned c = 0; c < K; c++) { // column
        // updatedReAmps = sum of reAmps * reMats - imAmps * imMats
        const auto& matrixEntry = matrixData[r * K + c];
        updatedReAmps[li] = internal::genMulAdd(B,
          matrixEntry.reVecVal, reAmps[c], updatedReAmps[li],
          matrixEntry.reKind, 
          "new.re." + std::to_string(hi) + "." + std::to_string(li) + ".");
        updatedReAmps[li] = internal::genNegMulAdd(B,
          matrixEntry.imVecVal, imAmps[c], updatedReAmps[li],
          matrixEntry.imKind, 
          "new.re." + std::to_string(hi) + "." + std::to_string(li) + ".");

        // updatedImAmps = sum of reAmps * imMats + imAmps * reMats
        updatedImAmps[li] = internal::genMulAdd(B,
          matrixEntry.reVecVal, imAmps[c], updatedImAmps[li],
          matrixEntry.reKind, 
          "new.im." + std::to_string(hi) + "." + std::to_string(li) + ".");
        updatedImAmps[li] = internal::genMulAdd(B,
          matrixEntry.imVecVal, reAmps[c], updatedImAmps[li],
          matrixEntry.imKind, 
          "new.im." + std::to_string(hi) + "." + std::to_string(li) + ".");
      }
    }

    /* Safety check: sometimes in testing, we use matrices such that some rows
     * are full of zeros. These matrices do not represent valid quantum gates,
     * and will cause normal IR generation to fail.
     */
    {
      bool valid = true;
      for (auto& v : updatedReAmps) {
        if (v == nullptr) {
          v = ConstantAggregateZero::get(VectorType::get(scalarTy, S, false));
          valid = false;
        }
      }
      for (auto& v : updatedImAmps) {
        if (v == nullptr) {
          v = ConstantAggregateZero::get(VectorType::get(scalarTy, S, false));
          valid = false;
        }
      }
      if (!valid) {
        std::cerr << BOLDYELLOW("Warning: ")
                  << "Updated amplitudes are left in invalid states after "
        "matrix-vector multiplication. This could mean the input matrix is "
        "invalid (some rows are full of zeros).\n";
      }
    }

    /* Merge
     * Merge example
     * Round 0: (xxx0, xxx1) => xxx0
     * Round 1: (xx00, xx10) => xx00
     * Round 2: (x000, x100) => x000
     */
    assert((1 << mergeMasks.size()) == updatedReAmps.size());
    for (unsigned mergeIdx = 0; mergeIdx < lk; mergeIdx++) {
      for (unsigned pairIdx = 0; pairIdx < (LK >> mergeIdx >> 1); pairIdx++) {
        unsigned idxL = pairIdx << mergeIdx << 1;
        unsigned idxR = idxL | (1 << mergeIdx);
        LLVM_DEBUG(
          dbgs() << "(mergeIdx, pairIdx) = ("
                << mergeIdx << ", " << pairIdx << "): (idxL, idxR) = ("
                << idxL << ", " << idxR << ")\n";
        );
        updatedReAmps[idxL] = B.CreateShuffleVector(
          updatedReAmps[idxL], updatedReAmps[idxR], mergeMasks[mergeIdx],
          "re.merged." + std::to_string(mergeIdx) + "." + std::to_string(pairIdx));
        updatedImAmps[idxL] = B.CreateShuffleVector(
          updatedImAmps[idxL], updatedImAmps[idxR], mergeMasks[mergeIdx],
          "im.merged." + std::to_string(mergeIdx) + "." + std::to_string(pairIdx));
      }
    }

    // store
    auto* merged = B.CreateShuffleVector(
      updatedReAmps[0], updatedImAmps[0], reimMergeMask,
      "amp.merged.hi." + std::to_string(hi));
    B.CreateStore(merged, pSvs[hi]);
  }

  // loopBodyBB->print(errs());

  // increment counter and return
  auto* taskIdNextV = B.CreateAdd(taskIdV, B.getInt64(1), "taskid.next");
  taskIdV->addIncoming(taskIdNextV, loopBodyBB);
  B.CreateBr(loopBB);

  B.SetInsertPoint(retBB);
  B.CreateRetVoid();

  // append the newly generated kernel
  this->_kernels.emplace_back(
    std::function<CPU_KERNEL_TYPE>(),
    config.precision,
    func->getName().str(),
    gate,
    config.simd_s,
    gate->opCount(config.zeroTol), // TODO: zeroTol here is different from zTol used in sigMat
    lk);
  return *this;
}

CPUKernelManager& CPUKernelManager::genCPUGatesFromCircuitGraph(
    const CPUKernelGenConfig& config, const CircuitGraph& graph,
    const std::string& graphName) {
  const auto allBlocks = graph.getAllBlocks();
  const auto mangledName = internal::mangleGraphName(graphName);
  for (const auto& block : allBlocks) {
    genCPUGate(
      config, block->quantumGate, mangledName + std::to_string(block->id));
  }
  return *this;
}

// CPUKernelManager& CPUKernelManager::genCPUMeasure(
//     const CPUKernelGenConfig& config, int q, const std::string& funcName) {
//   assert(0 && "Not Implemented");
//   return *this;
// }

void CPUKernelManager::dumpIR(const std::string &funcName, llvm::raw_ostream &os) {
  // We haven't called initJIT() yet, so the modules should still be in
  // llvmContextModulePairs.
  for (auto &cmp : llvmContextModulePairs) {
    auto &M = *cmp.llvmModule;
    if (auto *F = M.getFunction(funcName)) {
      os << "=== Dumping IR for function '" << funcName << "' ===\n";
      M.print(os, nullptr);
      return;
    }
  }
  os << "No module found containing function '" << funcName << "'\n";
}

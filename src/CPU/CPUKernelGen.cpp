#include "llvm/IR/IntrinsicsX86.h"

#include "cast/CPU/CPUKernelManager.h"
#include "cast/Core/KernelGenInternal.h"

#include "utils/Formats.h"
#include "utils/PrintSpan.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include <cmath>

#define DEBUG_TYPE "codegen-cpu"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;

std::atomic<int> cast::CPUKernelManager::_standaloneKernelCounter = 0;

// Top-level entry
MaybeError<CPUKernelManager::KernelInfoPtr>
CPUKernelManager::_genCPUGate(const CPUKernelGenConfig& config,
                              ConstQuantumGatePtr gate,
                              const std::string& funcName) {
  auto* stdQuGate = llvm::dyn_cast<const StandardQuantumGate>(gate.get());
  llvm::Function* func = nullptr;
  if (stdQuGate != nullptr && stdQuGate->noiseChannel() == nullptr) {
    // a normal gate, no noise channel
    const auto scalarGM = stdQuGate->getScalarGM();
    assert(scalarGM != nullptr && "Only supporting scalar GM for now");
    func = _gen(config, scalarGM->matrix(), stdQuGate->qubits(), funcName);
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
    func = _gen(config, scalarGM->matrix(), qubits, funcName);
  }

  if (func == nullptr) {
    std::ostringstream oss;
    oss << "Failed to generate kernel for gate " << (void*)(gate.get())
        << " with name " << funcName;
    return cast::makeError<KernelInfoPtr>(oss.str());
  }

  return std::make_unique<CPUKernelInfo>(
      std::function<CPU_KERNEL_TYPE>(), // empty executable
      config.precision,
      func->getName().str(),
      config.matrixLoadMode,
      gate,
      config.simdWidth,
      gate->opCount(config.zeroTol) // TODO: zeroTol here is different from zTol
                                    // used in sigMat
  );
}

MaybeError<void>
CPUKernelManager::genStandaloneGate(const CPUKernelGenConfig& config,
                                    ConstQuantumGatePtr gate,
                                    const std::string& _funcName) {
  std::string funcName(_funcName);
  if (funcName.empty())
    funcName = "kernel_" + std::to_string(_standaloneKernelCounter++);
  // check for name conflicts
  for (const auto& kernel : _standaloneKernels) {
    if (kernel->llvmFuncName == funcName) {
      return cast::makeError<void>("Kernel with name '" + funcName +
                                   "' already exists.");
    }
  }

  auto result = _genCPUGate(config, gate, funcName);
  if (!result) {
    return cast::makeError<void>("Err: " + result.takeError());
  }
  _standaloneKernels.emplace_back(result.takeValue());
  return {}; // success
}

MaybeError<void>
CPUKernelManager::genGraphGates(const CPUKernelGenConfig& config,
                                const ir::CircuitGraphNode& graph,
                                const std::string& graphName) {
  assert(graph.checkConsistency());

  if (_graphKernels.contains(graphName)) {
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
    auto result = _genCPUGate(config, gate, name);
    if (!result) {
      std::ostringstream oss;
      oss << "Failed to generate kernel for gate " << (void*)(gate.get())
          << ": " << result.takeError() << "\n";
      return cast::makeError<void>(oss.str());
    }
    kernels.emplace_back(result.takeValue());
  }
  // Store the generated kernels in the map
  _graphKernels[graphName] = std::move(kernels);

  return {}; // success
}

namespace {

struct CPUArgs {
  llvm::Value* pSvArg;      // ptr to statevector
  llvm::Value* ctrBeginArg; // counter begin
  llvm::Value* ctrEndArg;   // counter end
  llvm::Value* pMatArg;     // ptr to matrix
};

struct IRMatData {
  llvm::Value* reElmVal; // element of the real entry
  llvm::Value* imElmVal; // element of the imag entry
  llvm::Value* reVecVal; // vector of the real entry
  llvm::Value* imVecVal; // vector of the imag entry
  ScalarKind reFlag;     // flag for the real entry
  ScalarKind imFlag;     // flag for the imag entry

  // Default constructor needed here
  IRMatData()
      : reElmVal(nullptr), imElmVal(nullptr), reVecVal(nullptr),
        imVecVal(nullptr), reFlag(SK_Unknown), imFlag(SK_Unknown) {}
}; // IRMatData

/// @brief This function must be called after function declaration and that the
/// IRBuilder is set to the entry block.
std::vector<IRMatData> initMatrixData(llvm::IRBuilder<>& B,
                                      const CPUKernelGenConfig& config,
                                      const ComplexSquareMatrix& mat,
                                      llvm::Value* pMatArg) {
  const unsigned K = mat.edgeSize();
  const unsigned KK = K * K;

  std::vector<IRMatData> matData(KK);

  // TODO: decide whether to scale it by gate size
  double zTol = config.zeroTol;
  double oTol = config.oneTol;
  assert(zTol >= 0.0 && oTol >= 0.0 &&
         "Zero and one tolerances must be non-negative");
  // Step 1: set the flags
  if (zTol == 0.0 && oTol == 0.0) {
    for (unsigned i = 0; i < KK; ++i) {
      matData[i].reFlag = SK_General;
      matData[i].imFlag = SK_General;
    }
  } else {
    for (unsigned i = 0; i < KK; ++i) {
      auto re = mat.reData()[i];
      auto im = mat.imData()[i];
      if (std::abs(re) < zTol)
        matData[i].reFlag = SK_Zero;
      else if (std::abs(re - 1.0) < oTol)
        matData[i].reFlag = SK_One;
      else if (std::abs(re + 1.0) < oTol)
        matData[i].reFlag = SK_MinusOne;
      else
        matData[i].reFlag = SK_General;

      if (std::abs(im) < zTol)
        matData[i].imFlag = SK_Zero;
      else if (std::abs(im - 1.0) < oTol)
        matData[i].imFlag = SK_One;
      else if (std::abs(im + 1.0) < oTol)
        matData[i].imFlag = SK_MinusOne;
      else
        matData[i].imFlag = SK_General;
    }
  }

  // Step 2: set the values, either as imm values (llvm::Constant) or as
  // run-time loaded values
  assert(config.precision != Precision::Unknown);
  auto* ty =
      (config.precision == Precision::F32) ? B.getFloatTy() : B.getDoubleTy();
  auto ec = llvm::ElementCount::getFixed(1 << config.get_simd_s());
  switch (config.matrixLoadMode) {
  case CPUMatrixLoadMode::UseMatImmValues: {
    for (unsigned i = 0; i < KK; ++i) {
      // We only initialize the values for general entries. Values in other
      // cases (SK_Zero, SK_One, SK_MinusOne) are left as null. This is okay as
      // in the main loop we will only use these imm values if the flag is
      // SK_General.
      auto& d = matData[i];
      if (d.reFlag == SK_General) {
        auto* reConstantVal = llvm::ConstantFP::get(ty, mat.reData()[i]);
        d.reElmVal = reConstantVal;
        d.reVecVal = llvm::ConstantVector::getSplat(ec, reConstantVal);
      }
      if (d.imFlag == SK_General) {
        auto* imConstantVal = llvm::ConstantFP::get(ty, mat.imData()[i]);
        d.imElmVal = imConstantVal;
        d.imVecVal = llvm::ConstantVector::getSplat(ec, imConstantVal);
      }
    }
    break;
  }
  case CPUMatrixLoadMode::StackLoadMatElems: {
    // In StackLoadMatElems mode, the matrix elements are assumed to alternate
    // between one real and one imag value. That is, real[0], imag[0], real[1],
    // imag[1], ..., real[K*K-1], imag[K*K-1].
    for (unsigned i = 0; i < KK; ++i) {
      auto& d = matData[i];
      if (d.reFlag == SK_General) {
        auto* ptrV = B.CreateConstGEP1_32(
            ty, pMatArg, 2 * i, "re.mat.elem." + llvm::Twine(i));
        d.reElmVal = B.CreateLoad(ty, ptrV, "re.mat.elem." + llvm::Twine(i));
        d.reVecVal = B.CreateVectorSplat(
            ec, d.reElmVal, "re.mat.vec." + std::to_string(i));
      }
      if (d.imFlag == SK_General) {
        auto* ptrV = B.CreateConstGEP1_32(
            ty, pMatArg, 2 * i + 1, "im.mat.elem." + llvm::Twine(i));
        d.imElmVal = B.CreateLoad(ty, ptrV, "im.mat.elem." + llvm::Twine(i));
        d.imVecVal = B.CreateVectorSplat(
            ec, d.imElmVal, "im.mat.vec." + std::to_string(i));
      }
    }
    break;
  }
  default: {
    assert(false && "Unsupported matrix load mode");
    break;
  }
  } // switch (config.matrixLoadMode)

  return matData;
}

char flagToChar(ScalarKind flag) {
  switch (flag) {
  case SK_General:
    return 'X';
  case SK_Zero:
    return '0';
  case SK_One:
    return '+';
  case SK_MinusOne:
    return '-';
  default:
    return '?';
  }
}

void debugPrintMatData(std::ostream& os,
                       const std::vector<IRMatData>& matData) {
  unsigned KK = matData.size();
  unsigned K = std::sqrt(KK);
  for (unsigned r = 0; r < K; ++r) {
    if (r != 0) {
      os << "\n";
    }
    for (unsigned c = 0; c < K; ++c) {
      os.put(flagToChar(matData[r * K + c].reFlag));
      os.put(':');
      os.put(flagToChar(matData[r * K + c].imFlag));
      if (r != K - 1 || c != K - 1) {
        os << ", ";
      }
    }
  }
}

} // end of anonymous namespace

llvm::Function*
CPUKernelManager::_gen(const CPUKernelGenConfig& config,
                       const ComplexSquareMatrix& mat,
                       const QuantumGate::TargetQubitsType& qubits,
                       const std::string& funcName) {
  const unsigned s = config.get_simd_s();
  const unsigned S = 1ULL << s;
  const unsigned k = qubits.size();
  const unsigned K = 1ULL << k;
  const unsigned KK = K * K;
  assert(K == mat.edgeSize() && "matrix size mismatch");

  auto& llvmContextModulePair =
      createNewLLVMContextModulePair(funcName + "Module");
  auto& llvmContext = *llvmContextModulePair.llvmContext;
  auto& llvmModule = *llvmContextModulePair.llvmModule;

  llvm::IRBuilder<> B(llvmContext);
  assert(config.precision != Precision::Unknown);
  auto* scalarTy =
      (config.precision == Precision::F32) ? B.getFloatTy() : B.getDoubleTy();

  // Create function declaration
  CPUArgs args;
  llvm::Function* func;
  llvm::BasicBlock *entryBB, *loopBB, *loopBodyBB, *retBB;
  { // start of function declaration
    auto* funcTy = llvm::FunctionType::get(
        /* return type */ B.getVoidTy(),
        /* arg type */ {B.getPtrTy()},
        /* isVarArg */ false);
    func = llvm::Function::Create(
        funcTy, llvm::Function::ExternalLinkage, funcName, llvmModule);
    entryBB = llvm::BasicBlock::Create(B.getContext(), "entry", func);
    loopBB = llvm::BasicBlock::Create(B.getContext(), "loop", func);
    loopBodyBB = llvm::BasicBlock::Create(B.getContext(), "loop.body", func);
    retBB = llvm::BasicBlock::Create(B.getContext(), "ret", func);

    B.SetInsertPoint(entryBB);
    // get arguments
    auto* funcArg = func->getArg(0);
    funcArg->setName("func.arg");
    llvm::Value* tmp;
    tmp = B.CreateConstGEP1_32(B.getPtrTy(), funcArg, 0, "p.p.sv.arg");
    args.pSvArg = B.CreateLoad(B.getPtrTy(), tmp, "p.sv.arg");
    tmp = B.CreateConstGEP1_32(B.getPtrTy(), funcArg, 1, "p.p.ctr.begin");
    tmp = B.CreateLoad(B.getPtrTy(), tmp, "p.ctr.begin");
    args.ctrBeginArg = B.CreateLoad(B.getInt64Ty(), tmp, "ctr.begin");
    tmp = B.CreateConstGEP1_32(B.getPtrTy(), funcArg, 2, "p.p.ctr.end");
    tmp = B.CreateLoad(B.getPtrTy(), tmp, "p.ctr.end");
    args.ctrEndArg = B.CreateLoad(B.getInt64Ty(), tmp, "ctr.end");
    tmp = B.CreateConstGEP1_32(B.getPtrTy(), funcArg, 3, "p.p.mat.arg");
    args.pMatArg = B.CreateLoad(B.getPtrTy(), tmp, "p.mat.arg");
  } // end of function declaration

  // initialize matrix data
  B.SetInsertPoint(entryBB);
  auto matData = initMatrixData(B, config, mat, args.pMatArg);

  LLVM_DEBUG(debugPrintMatData(std::cerr, matData););

  // split qubits
  int sepBit;
  llvm::SmallVector<int, 6U> simdBits, hiBits, loBits;
  { // start of qubit splitting
    unsigned q = 0;
    auto qubitsIt = qubits.begin();
    const auto qubitsEnd = qubits.end();
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
  } // end of qubit splitting
  const unsigned vecSize = 1U << sepBit;
  auto* vecType = llvm::VectorType::get(scalarTy, vecSize, false);

  const unsigned lk = loBits.size();
  const unsigned LK = 1 << lk;
  const unsigned hk = hiBits.size();
  const unsigned HK = 1 << hk;

  // debug print qubit splits
  LLVM_DEBUG(std::cerr << CYAN("-- qubit split done\n");
             utils::printArray(std::cerr << "- lower bits:  ", loBits) << "\n";
             utils::printArray(std::cerr << "- higher bits: ", hiBits) << "\n";
             utils::printArray(std::cerr << "- simd bits:   ", simdBits)
             << "\n";
             std::cerr << "- reImBit (simd_s): " << s << "\n";
             std::cerr << "- sepBit:           " << sepBit << "\n";
             std::cerr << "- vecSize:          " << vecSize << "\n";);

  // entryBB->print(llvm::errs());

  // set up counter and loop structure
  B.CreateBr(loopBB);
  B.SetInsertPoint(loopBB);
  auto* taskIdV = B.CreatePHI(B.getInt64Ty(), 2, "taskid");
  taskIdV->addIncoming(args.ctrBeginArg, entryBB);
  auto* cond = B.CreateICmpSLT(taskIdV, args.ctrEndArg, "cond");
  B.CreateCondBr(cond, loopBodyBB, retBB);

  // loop body
  B.SetInsertPoint(loopBodyBB);

  // the start pointer in the SV based on taskID
  llvm::Value* ptrSvBeginV = nullptr;
  if (hiBits.empty()) {
    ptrSvBeginV = B.CreateGEP(vecType, args.pSvArg, taskIdV, "ptr.sv.begin");
  } else {
    // the shift from args.pSvArg in the unit of vecTypeX2
    llvm::Value* idxStartV = B.getInt64(0ULL);
    llvm::Value* tmpCounterV;
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
      LLVM_DEBUG(std::cerr << "  (taskID & " << utils::fmt_0b(mask, highestQ)
                           << ") << " << (qIdx - 1) << "\n";);
      mask = 0ULL;
    }
    mask = ~((1ULL << (highestQ - sepBit - hk + 1)) - 1);
    LLVM_DEBUG(std::cerr << "  (taskID & " << utils::fmt_0b(mask, 16) << ") << "
                         << hk << "\n";);

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
  llvm::SmallVector<int> reSplitMasks;
  llvm::SmallVector<int> imSplitMasks;
  { // begin init [re/im]SplitMasks
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
    LLVM_DEBUG(std::cerr << "- reSplitMasks: [";
               for (const auto& e
                    : reSplitMasks) std::cerr
               << utils::fmt_0b(e, sepBit + 1) << ",";
               std::cerr << "]\n";
               std::cerr << "- imSplitMasks: [";
               for (const auto& e
                    : imSplitMasks) std::cerr
               << utils::fmt_0b(e, sepBit + 1) << ",";
               std::cerr << "]\n";);
  } // end init [re/im]SplitMasks

  // load vectors
  llvm::SmallVector<llvm::Value*> reAmps; // real amplitudes
  llvm::SmallVector<llvm::Value*> imAmps; // imag amplitudes
  reAmps.resize_for_overwrite(K);
  imAmps.resize_for_overwrite(K);

  llvm::SmallVector<llvm::Value*> pSvs;
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
                  << "\n";);
    pSvs[hi] = B.CreateConstGEP1_64(
        vecType, ptrSvBeginV, idxShift, "ptr.sv.hi." + std::to_string(hi));
    auto* ampFull =
        B.CreateLoad(vecType, pSvs[hi], "sv.full.hi." + std::to_string(hi));

    for (unsigned li = 0; li < LK; li++) {
      reAmps[hi * LK + li] = B.CreateShuffleVector(
          ampFull,
          llvm::ArrayRef<int>(reSplitMasks.data() + li * S, S),
          "re." + std::to_string(hi) + "." + std::to_string(li));
      imAmps[hi * LK + li] = B.CreateShuffleVector(
          ampFull,
          llvm::ArrayRef<int>(imSplitMasks.data() + li * S, S),
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
    std::vector<int>&cacheLHS = arr0, &cacheRHS = arr1, &cacheCombined = arr2;
    std::memcpy(arr0.data(), reSplitMasks.data(), S * sizeof(int));
    std::memcpy(arr1.data(), reSplitMasks.data() + S, S * sizeof(int));
    int roundIdx = 0;
    while (roundIdx < lk) {
      LLVM_DEBUG(
          std::cerr << "Round " << roundIdx << ": ";
          utils::printArray(std::cerr, llvm::ArrayRef(cacheLHS)) << " and ";
          utils::printArray(std::cerr, llvm::ArrayRef(cacheRHS)) << "\n";);

      lCached = S << roundIdx;
      mergeMasks.emplace_back(lCached << 1);
      auto& mask = mergeMasks.back();

      idxL = 0;
      idxR = 0;
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
        } else {
          mask[idxCombined] = idxR + lCached;
          cacheCombined[idxCombined] = cacheRHS[idxR];
          ++idxR;
        }
      }
      LLVM_DEBUG(
          utils::printArray(std::cerr << "  Cache Combined: ",
                            llvm::ArrayRef(cacheCombined))
              << "\n";
          utils::printArray(std::cerr << "  Mask: ", llvm::ArrayRef(mask))
          << "\n";);
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
    LLVM_DEBUG(utils::printArray(std::cerr << "reimMergeMask: ",
                                 llvm::ArrayRef(reimMergeMask))
                   << "\n";
               std::cerr << CYAN("- Merged masks initiated\n"););
  }

  std::vector<llvm::Value*> updatedReAmps(LK);
  std::vector<llvm::Value*> updatedImAmps(LK);
  for (unsigned hi = 0; hi < HK; hi++) {
    // mat-vec mul
    for (auto& v : updatedReAmps)
      v = nullptr;
    for (auto& v : updatedImAmps)
      v = nullptr;
    for (unsigned li = 0; li < LK; li++) {
      unsigned r = hi * LK + li;         // row
      for (unsigned c = 0; c < K; c++) { // column
        // updatedReAmps = sum of reAmps * reMats - imAmps * imMats
        const auto& matrixEntry = matData[r * K + c];
        updatedReAmps[li] = internal::genMulAdd(
            B,
            matrixEntry.reVecVal,
            reAmps[c],
            updatedReAmps[li],
            matrixEntry.reFlag,
            "new.re." + std::to_string(hi) + "." + std::to_string(li) + ".");
        updatedReAmps[li] = internal::genNegMulAdd(
            B,
            matrixEntry.imVecVal,
            imAmps[c],
            updatedReAmps[li],
            matrixEntry.imFlag,
            "new.re." + std::to_string(hi) + "." + std::to_string(li) + ".");

        // updatedImAmps = sum of reAmps * imMats + imAmps * reMats
        updatedImAmps[li] = internal::genMulAdd(
            B,
            matrixEntry.reVecVal,
            imAmps[c],
            updatedImAmps[li],
            matrixEntry.reFlag,
            "new.im." + std::to_string(hi) + "." + std::to_string(li) + ".");
        updatedImAmps[li] = internal::genMulAdd(
            B,
            matrixEntry.imVecVal,
            reAmps[c],
            updatedImAmps[li],
            matrixEntry.imFlag,
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
          v = llvm::ConstantAggregateZero::get(
              llvm::VectorType::get(scalarTy, S, false));
          valid = false;
        }
      }
      for (auto& v : updatedImAmps) {
        if (v == nullptr) {
          v = llvm::ConstantAggregateZero::get(
              llvm::VectorType::get(scalarTy, S, false));
          valid = false;
        }
      }
      if (!valid) {
        std::cerr << BOLDYELLOW("Warning: ")
                  << "Updated amplitudes are left in invalid states after "
                     "matrix-vector multiplication. This could mean the input "
                     "matrix is "
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
        LLVM_DEBUG(std::cerr << "(mergeIdx, pairIdx) = (" << mergeIdx << ", "
                             << pairIdx << "): (idxL, idxR) = (" << idxL << ", "
                             << idxR << ")\n";);
        updatedReAmps[idxL] =
            B.CreateShuffleVector(updatedReAmps[idxL],
                                  updatedReAmps[idxR],
                                  mergeMasks[mergeIdx],
                                  "re.merged." + std::to_string(mergeIdx) +
                                      "." + std::to_string(pairIdx));
        updatedImAmps[idxL] =
            B.CreateShuffleVector(updatedImAmps[idxL],
                                  updatedImAmps[idxR],
                                  mergeMasks[mergeIdx],
                                  "im.merged." + std::to_string(mergeIdx) +
                                      "." + std::to_string(pairIdx));
      }
    }

    // store
    auto* merged = B.CreateShuffleVector(updatedReAmps[0],
                                         updatedImAmps[0],
                                         reimMergeMask,
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

  // func->print(llvm::errs());
  return func;
}
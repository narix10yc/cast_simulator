#include <llvm/IR/IntrinsicsNVPTX.h>

#include "cast/QuantumGate.h"
#include "cast/CircuitGraph.h"

#include "simulation/KernelManager.h"
#include "simulation/KernelGenInternal.h"

#include "utils/utils.h"
#include "utils/iocolor.h"
#include "utils/Formats.h"

#define DEBUG_TYPE "codegen-cuda"
#include <llvm/Support/Debug.h>
// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

namespace {
Value* genOptFMul(Value* a, Value* b, ScalarKind aKind, IRBuilder<>& B) {
  switch (aKind) {
  case SK_General:
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
  auto* func = Function::Create(
    FunctionType::get(
      // returns void
      B.getVoidTy(),
      // takes (void*, void*)
      { B.getPtrTy(), B.getPtrTy()},  
      // not variadic
      false
    ),
    Function::ExternalLinkage,
    funcName,
    M
  );
  if (funcName == "")
    func->setName("ptx_kernel_");
  else
    func->setName(funcName);

  args.pSvArg = func->getArg(0);
  args.pSvArg->setName("p.sv");
  args.pMatArg = func->getArg(1);
  args.pMatArg->setName("p.mat");

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

  // load amplitudes. For a k-qubit gate (with K = 1 << K), we local K real amps 
  // and K imag amplitudes. So every iteration updates 2*K scalar elements.
  std::vector<Value*> reAmpPtrs(K), imAmpPtrs(K), reAmps(K), imAmps(K);
  for (uint64_t i = 0; i < K; i++) {
    uint64_t delta = 0ULL;
    for (int b = 0; b < k; b++) {
      if (i & (1 << b))
        delta |= (1 << gate->qubits[b]);
    }
    // std::cerr << "amp idx " << utils::as0b(i, k) << ", delta = " <<
    // utils::as0b(delta, 32) << "\n";

    reAmpPtrs[i] = B.CreateConstGEP1_64(
      scalarTy, svPtrV, 2 * delta, "amp.re.ptr." + std::to_string(i));
    reAmps[i] = B.CreateLoad(
      scalarTy, reAmpPtrs[i], "amp.re." + std::to_string(i));
    imAmpPtrs[i] = B.CreateConstGEP1_64(
      scalarTy, svPtrV, 2 * delta + 1, "amp.im.ptr." + std::to_string(i));
    imAmps[i] = B.CreateLoad(
      scalarTy, imAmpPtrs[i], "amp.im." + std::to_string(i));
  }

  const auto* gateCMat = gate->gateMatrix.getConstantMatrix();
  // assert(gateCMat && "Only supporting constant matrix for now");

  // This loop updates reAmpPtrs[r] and imAmpPtrs[r].
  // Calculated by the complex inner product of the r-th row of matrix and 
  // the complex vector (reAmps + i * imAmps)
  for (unsigned r = 0; r < K; ++r) {
    // matrix-vector multiplication
    // updatedReAmp = sum(matRe_i * ampRe_i) - sum(matIm_i * ampIm_i)
    // updatedImAmp = sum(matRe_i * ampIm_i) + sum(matIm_i * ampRe_i)

    auto& md = matData[r * K];
    // updatedReAmp0 collects sum(matRe_i * ampRe_i)
    Value* updatedReAmp0 = internal::genMulAdd(
      B, md.reVal, reAmps[0], nullptr, md.reKind);

    // updatedReAmp1 collects sum(matIm_i * ampIm_i)
    Value* updatedReAmp1 = internal::genMulAdd(
      B, md.imVal, imAmps[0], nullptr, md.imKind);

    // updatedImAmp equals to sum(matRe_i * ampIm_i) + sum(matIm_i * ampRe_i)
    Value* updatedImAmp = internal::genMulAdd(
      B, md.reVal, imAmps[0], nullptr, md.reKind);
    updatedImAmp = internal::genMulAdd(
      B, md.imVal, reAmps[0], updatedImAmp, md.imKind);

    for (unsigned c = 1; c < K; ++c) {
      md = matData[r * K + c];
      updatedReAmp0 = internal::genMulAdd(
        B, md.reVal, reAmps[c], updatedReAmp0, md.reKind);
      updatedReAmp1 = internal::genMulAdd(
        B, md.imVal, imAmps[c], updatedReAmp1, md.imKind);
      updatedImAmp = internal::genMulAdd(
        B, md.reVal, imAmps[c], updatedImAmp, md.reKind);
      updatedImAmp = internal::genMulAdd(
        B, md.imVal, reAmps[c], updatedImAmp, md.imKind);
    }
    
    Value* updatedReAmp = nullptr;
    if (updatedReAmp0 && updatedReAmp1)
      updatedReAmp = B.CreateFSub(updatedReAmp0, updatedReAmp1);
    else if (updatedReAmp0)
      updatedReAmp = updatedReAmp0;
    else if (updatedReAmp1)
      updatedReAmp = B.CreateFNeg(updatedReAmp1);
    else
      llvm_unreachable("updatedReAmp should not be zero");

    B.CreateStore(updatedReAmp, reAmpPtrs[r]);
    B.CreateStore(updatedImAmp, imAmpPtrs[r]);
  }

  B.CreateRetVoid();
  LLVM_DEBUG(func->dump());

  // append the newly generated kernel
  this->_cudaKernels.emplace_back(
    CUDAKernelInfo::PTXStringType(), // empty ptxString
    config.precision,
    func->getName().str(),
    gate,
    CUDAKernelInfo::CUDATuple() // default constructor of CUDATuple
  );
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


#undef DEBUG_TYPE
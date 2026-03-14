#include "cast/Core/ScalarKind.h"

#include "cast/CUDA/CUDAKernelManager.h"

#include "utils/Formats.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Target/TargetMachine.h>

#define DEBUG_TYPE "codegen-cuda"
#include <llvm/Support/Debug.h>
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
  const auto K = 1U << nQubits;
  const auto KK = K * K;

  // Scale the tolerances by matrix size
  const auto zTol = config.zeroTol / K;
  const auto oTol = config.oneTol / K;

  auto data = std::vector<IRMatDataCUDA>(KK);

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
      (config.precision == Precision::FP32) ? APFloat(0.0f) : APFloat(0.0));
  auto oneVal = ConstantFP::get(
      B.getContext(),
      (config.precision == Precision::FP32) ? APFloat(1.0f) : APFloat(1.0));
  auto minusOneVal = ConstantFP::get(
      B.getContext(),
      (config.precision == Precision::FP32) ? APFloat(-1.0f) : APFloat(-1.0));

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
                                      (config.precision == Precision::FP32)
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
                                      (config.precision == Precision::FP32)
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
                                     IRArgsCUDA& args) {
  auto params = std::array<Type*, 3>{
      B.getPtrTy(),  // sv
      B.getPtrTy(),  // mat
      B.getInt64Ty() // combos
  };
  auto* fty = FunctionType::get(B.getVoidTy(), params, false);
  auto* func = Function::Create(fty, Function::ExternalLinkage, funcName, M);

  args.pSvArg = func->getArg(0);
  args.pSvArg->setName("p.sv");
  args.pMatArg = func->getArg(1);
  args.pMatArg->setName("p.mat");
  args.pCombos = func->getArg(2);
  args.pCombos->setName("p.combos");

  // mark as CUDA (PTX) kernel
  auto* mdString = MDString::get(M.getContext(), "kernel");
  auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
  auto* md = MDNode::get(M.getContext(),
                         {ValueAsMetadata::get(func), mdString, mdOne});
  M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md);
  return func;
}

/* This mimics a runtime PDEP operation with compile-time known mask.
Given a counter t, find the offset index idx such that in round t of the
iteration, amplitude index starts at idx. The returned idx is in the unit of
[real, imag], i.e. <2 x ScalarType>.
- Example: with target qubits 2, 4, 5
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
Value* buildOffset(IRBuilder<>& B,
                   Value* counterV,
                   const QuantumGate::TargetQubitsType& qubits) {

  assert(!qubits.empty());

  auto* offset = static_cast<Value*>(B.getInt64(0ULL));
  counterV = B.CreateZExt(counterV, B.getInt64Ty(), "i64.counter");

  auto* tmpCounterV = static_cast<Value*>(nullptr);
  uint64_t mask = 0ULL;

  const auto k = static_cast<int>(qubits.size());
  const auto highestQ = static_cast<int>(qubits.back());
  auto qIdx = 0;
  auto counterQ = 0;

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

static void
genMatrixVectorMultiply_InlineImm(IRBuilder<>& B,
                                  const QuantumGate::TargetQubitsType& qubits,
                                  const std::vector<IRMatDataCUDA>& matData,
                                  Value* svPtrV,
                                  Type* scalarTy) {
  B.setFastMathFlags(FastMathFlags::getFast());

  const auto k = static_cast<unsigned>(qubits.size());
  const auto K = 1u << k;

  // Precompute per-amplitude pointers and load them once.
  auto reAmpPtrs = std::vector<Value*>(K);
  auto imAmpPtrs = std::vector<Value*>(K);
  auto reAmps = std::vector<Value*>(K);
  auto imAmps = std::vector<Value*>(K);

  for (unsigned i = 0; i < K; ++i) {
    auto off2 = uint64_t{0};
    auto delta = uint64_t{0};
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
    auto* accRe0 = static_cast<Value*>(ConstantFP::get(scalarTy, 0.0));
    auto* accRe1 = static_cast<Value*>(ConstantFP::get(scalarTy, 0.0));
    auto* accIm = static_cast<Value*>(ConstantFP::get(scalarTy, 0.0));

    for (unsigned c = 0; c < K; ++c) {
      const auto& md = matData[r * K + c];
      if (md.reKind == SK_Zero && md.imKind == SK_Zero)
        continue;

      // Re(new)
      if (auto* t0 = genOptFMul(md.reVal, reAmps[c], md.reKind, B))
        accRe0 = B.CreateFAdd(accRe0, t0);
      if (auto* t1 = genOptFMul(md.imVal, imAmps[c], md.imKind, B))
        accRe1 = B.CreateFAdd(accRe1, t1);

      // Im(new)
      if (auto* t2 = genOptFMul(md.reVal, imAmps[c], md.reKind, B))
        accIm = B.CreateFAdd(accIm, t2);
      if (auto* t3 = genOptFMul(md.imVal, reAmps[c], md.imKind, B))
        accIm = B.CreateFAdd(accIm, t3);
    }

    auto* newRe = B.CreateFSub(accRe0, accRe1);
    B.CreateStore(newRe, reAmpPtrs[r]);
    B.CreateStore(accIm, imAmpPtrs[r]);
  }
}

static void genMatVecMul_Imm(IRBuilder<>& B,
                             const QuantumGate::TargetQubitsType& qubits,
                             const std::vector<IRMatDataCUDA>& matData,
                             Value* svRoot,
                             Type* scalarTy) {
  auto* func = B.GetInsertBlock()->getParent();
  auto& C = B.getContext();

  // total number of combos: equals to 2^{n-k}
  // may exceed 32 bits, so we need Int64Ty
  auto* combosV = static_cast<Argument*>(nullptr);
  for (auto& A : func->args()) {
    if (A.getName() == "p.combos") {
      combosV = &A;
      break;
    }
  }
  assert(combosV && "Missing kernel arg 'p.combos'");
  assert(combosV->getType()->isIntegerTy(64));

  // nvptx intrinsics
  auto* tid = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_tid_x, {});
  auto* ntid = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ntid_x, {});
  auto* ctaid = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
  auto* nctaid = B.CreateIntrinsic(
      B.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_nctaid_x, {});

  // global.tid = ctaid * ntid + tid
  auto* global_tid = B.CreateAdd(B.CreateMul(ctaid, ntid), tid, "global.tid");
  global_tid = B.CreateIntCast(global_tid, B.getInt64Ty(), true, "global.tid");

  // stride = nctaid * ntid = <total number of threads>
  auto* stride = B.CreateMul(nctaid, ntid, "combo.stride");
  stride = B.CreateIntCast(stride, B.getInt64Ty(), true, "combo.stride");

  // Each thread processes multiple combos in a strided loop:
  // for (combo = global_tid; combo < p.combos; combo += stride) { ... }

  // combo loop
  auto* cmbChk = BasicBlock::Create(C, "cmb.chk", func);
  auto* cmbBody = BasicBlock::Create(C, "cmb.body", func);
  auto* cmbInc = BasicBlock::Create(C, "cmb.inc", func);
  auto* cmbDone = BasicBlock::Create(C, "cmb.done", func);

  auto* pre = B.GetInsertBlock();
  B.CreateBr(cmbChk);
  B.SetInsertPoint(cmbChk);
  auto* comboid = B.CreatePHI(B.getInt64Ty(), 2, "combo");
  comboid->addIncoming(global_tid, pre);
  B.CreateCondBr(B.CreateICmpULT(comboid, combosV), cmbBody, cmbDone);

  B.SetInsertPoint(cmbBody);
  {
    auto* svBase = buildOffset(B, comboid, qubits);
    svBase = B.CreateShl(svBase, 1); // for real/imag
    svBase = B.CreateGEP(scalarTy, svRoot, svBase, "sv.base");

    // Reuse straight-line multiply on this combo:
    genMatrixVectorMultiply_InlineImm(B, qubits, matData, svBase, scalarTy);

    B.CreateBr(cmbInc);
  }

  B.SetInsertPoint(cmbInc);
  {
    auto* nextCombo = B.CreateAdd(comboid, stride);
    comboid->addIncoming(nextCombo, cmbInc);
    B.CreateBr(cmbChk);
  }

  B.SetInsertPoint(cmbDone);
}

} // end of anonymous namespace

llvm::Expected<llvm::Function*>
CUDAKernelManager::gen_(const CUDAKernelGenConfig& config,
                        const ComplexSquareMatrix& matrix,
                        const QuantumGate::TargetQubitsType& qubits,
                        const std::string& funcName,
                        llvm::Module& llvmModule) {
  // number of target qubits
  const auto k = static_cast<unsigned>(qubits.size());

  auto& llvmContext = llvmModule.getContext();
  IRBuilder<> B(llvmContext);

  if (config.precision == Precision::Unknown) {
    return llvm::createStringError("KernelGenConfig must specify a Precision");
  }
  auto* scalarTy =
      (config.precision == Precision::FP32) ? B.getFloatTy() : B.getDoubleTy();

  IRArgsCUDA args;
  auto* func = getFunctionDeclarationCUDA(B, llvmModule, funcName, args);

  auto* entryBB = BasicBlock::Create(llvmContext, "entry", func);
  B.SetInsertPoint(entryBB);

  auto matData = getMatDataCUDA(B, config, matrix, k);

  switch (config.matrixLoadMode) {
  case CUDAMatrixLoadMode::UseMatImmValues: {
    // lane-per-combo persistent inline
    genMatVecMul_Imm(B, qubits, matData, args.pSvArg, scalarTy);
    break;
  }
  default:
    return llvm::createStringError("Unsupported CUDAMatrixLoadMode");
  }

  B.CreateRetVoid();
  auto errInfo = std::string{};
  llvm::raw_string_ostream rso(errInfo);
  if (llvm::verifyFunction(*func, &rso)) {
    return llvm::createStringError("Function verification failed: " +
                                   rso.str());
  }

  // We must use calling conventions for PTX kernels with LLVM >= 21. That is,
  // define ptx_kernel void @foo(...) { ... }
  // Prior to LLVM 21, attaching the attribute was sufficient.
  // Check LLVM docs:
  // https://releases.llvm.org/20.1.0/docs/NVPTXUsage.html#marking-functions-as-kernels
  // versus
  // https://releases.llvm.org/21.1.0/docs/NVPTXUsage.html#marking-functions-as-kernels
  // FIXME: Remove function metadata when we drop support for LLVM < 21.
  func->setCallingConv(llvm::CallingConv::PTX_Kernel);
  return func;
}

llvm::Expected<CudaKernel*>
CUDAKernelManager::genCUDAGate_(const CUDAKernelGenConfig& config,
                                ConstQuantumGatePtr gate,
                                const std::string& funcName,
                                Pool& pool) {
  // This is a freshly created kernel so we don't need to lock it
  auto kernel = std::make_unique<CudaKernel>(funcName);

  auto* stdQuGate = llvm::dyn_cast<const StandardQuantumGate>(gate.get());
  auto* func = static_cast<llvm::Function*>(nullptr);

  if (stdQuGate != nullptr && stdQuGate->noiseChannel() == nullptr) {
    // a normal gate, no noise channel
    const auto scalarGM = stdQuGate->getScalarGM();
    if (scalarGM == nullptr) {
      return llvm::createStringError("Only supporting scalar GM for now");
    }

    if (auto expectedF = gen_(config,
                              scalarGM->matrix(),
                              stdQuGate->qubits(),
                              funcName,
                              *kernel->llvmModule)) {
      func = *expectedF;
    } else {
      return expectedF.takeError();
    }
  } else {
    // super op gates are treated as normal gates with twice the number of
    // qubits
    auto superopGate = gate->getSuperopGate();
    assert(superopGate != nullptr && "Superop gate should not be null");
    const auto scalarGM = superopGate->getMatrix();
    assert(scalarGM != nullptr && "superop gate matrix should not be null");

    auto qubits = superopGate->qubits();
    const auto nQubits = superopGate->nQubits();
    for (const auto& q : qubits)
      qubits.push_back(q + nQubits);

    if (auto expectedF = gen_(config,
                              scalarGM->matrix(),
                              qubits,
                              funcName,
                              *kernel->llvmModule)) {
      func = *expectedF;
    } else {
      return expectedF.takeError();
    }
  }

  if (func == nullptr) {
    return llvm::createStringError(funcName + " generates a null function");
  }

  // update the kernel info
  kernel->gate = gate;
  kernel->llvmFunc = func;
  kernel->precision = config.precision;

  pool.push_back(std::move(kernel));
  return pool.back().get();
}

llvm::Expected<CudaKernel*>
CUDAKernelManager::genGate(const CUDAKernelGenConfig& config,
                           ConstQuantumGatePtr gate,
                           const std::string& funcName_) {
  auto funcName = std::string(funcName_);
  auto& pool = getDefaultPool();
  if (funcName.empty())
    funcName = "k" + std::to_string(pool.size());

  // check for name conflicts
  for (const auto& kernel : pool) {
    if (kernel->llvmFunc->getName() == funcName) {
      return llvm::createStringError("Kernel with name '" + funcName +
                                     "' already exists.");
    }
  }

  auto eKernel = genCUDAGate_(config, gate, funcName, pool);
  if (!eKernel) {
    return llvm::joinErrors(
        llvm::createStringError("Failed to generate gate '" + funcName + "'"),
        eKernel.takeError());
  }

  // enqueue for compilation
  enqueueForCompilation(*eKernel);
  return *eKernel;
}

llvm::Error CUDAKernelManager::genGraphGates(const CUDAKernelGenConfig& config,
                                             const ir::CircuitGraphNode& graph,
                                             const std::string& poolName) {
  assert(graph.checkConsistency());

  if (kernelPools_.contains(poolName)) {
    return llvm::createStringError("Already exists a pool named '" + poolName +
                                   "'");
  }

  kernelPools_.emplace(poolName, Pool());
  auto& pool = kernelPools_.at(poolName);
  auto allGates = graph.getAllGatesShared();
  auto order = 0;

  for (const auto& gate : allGates) {
    auto name = poolName + "_" + std::to_string(order++) + "_" +
                std::to_string(graph.gateId(gate));

    // genCUDAGate_ will put the new kernel into pool
    auto eKernel = genCUDAGate_(config, gate, name, pool);
    if (!eKernel) {
      return llvm::joinErrors(
          llvm::createStringError("Failed to generate kernel for gate " + name),
          eKernel.takeError());
    }

    enqueueForCompilation(*eKernel);
  }
  return llvm::Error::success();
}

#undef DEBUG_TYPE

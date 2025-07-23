#include "cast/CPU/CPUKernelManager.h"
#include "cast/Legacy/CircuitGraph.h"

using namespace cast;

// Allocates memory for the gate matrix to be used in invoking CPU kernels
// @return a pointer to the allocated memory. Remember to free it after use.
static void* mallocGatePointer(const cast::QuantumGate* gate,
                               Precision precision) {
  void* p = nullptr;
  auto* standardQuGate = llvm::dyn_cast<cast::StandardQuantumGate>(gate);
  if (standardQuGate != nullptr) {
    auto* gateMatrix = standardQuGate->gateMatrix().get();
    assert(gateMatrix != nullptr && "Empty gate matrix");
    auto* scalarGM = llvm::dyn_cast<cast::ScalarGateMatrix>(gateMatrix);
    assert(scalarGM != nullptr && "Only ScalarGateMatrix is supported for now");
    const auto& mat = scalarGM->matrix();
    auto edgeSize = mat.edgeSize();
    if (precision == Precision::F32) {
      p = std::malloc(2 * edgeSize * edgeSize * sizeof(float));
      float* pp = reinterpret_cast<float*>(p);
      for (unsigned r = 0; r < edgeSize; ++r) {
        for (unsigned c = 0; c < edgeSize; ++c) {
          auto idx = r * edgeSize + c;
          pp[2 * idx] = static_cast<float>(mat.reData()[idx]);
          pp[2 * idx + 1] = static_cast<float>(mat.imData()[idx]);
        }
      }
    } else {
      assert(precision == Precision::F64 && "Unsupported precision");
      p = std::malloc(2 * edgeSize * edgeSize * sizeof(double));
      double* pp = reinterpret_cast<double*>(p);
      for (unsigned r = 0; r < edgeSize; ++r) {
        for (unsigned c = 0; c < edgeSize; ++c) {
          auto idx = r * edgeSize + c;
          pp[2 * idx] = mat.reData()[idx];
          pp[2 * idx + 1] = mat.imData()[idx];
        }
      }
    }
  } else {
    assert(false && "Only StandardQuantumGate is supported for now");
  }

  assert(p != nullptr && "Failed to allocate memory for gate matrix");
  return p;
}

MaybeError<void> CPUKernelManager::applyCPUKernel(void* sv,
                                                  int nQubits,
                                                  const CPUKernelInfo& kernel,
                                                  int nThreads) const {
  if (!isJITed()) {
    return cast::makeError<void>(
        "Must initialize JIT session before applying CPU kernel.");
  }
  if (kernel.executable == nullptr) {
    return cast::makeError<void>("Kernel executable not available.");
  }
  int simd_s = cast::get_simd_s(kernel.simdWidth, kernel.precision);
  int tmp = nQubits - kernel.gate->nQubits() - simd_s;
  if (tmp < 0) {
    std::ostringstream oss;
    oss << "Invalid number of qubits for the kernel '" << kernel.llvmFuncName
        << "'. This kernel must act on statevectors with at least "
        << (kernel.gate->nQubits() + simd_s) << "qubits.";
    return cast::makeError<void>(oss.str());
  }
  uint64_t nTasks = 1ULL << tmp;
  void* pMat = nullptr;
  if (kernel.matrixLoadMode == CPUMatrixLoadMode::StackLoadMatElems) {
    pMat = mallocGatePointer(kernel.gate.get(), kernel.precision);
    if (pMat == nullptr) {
      return cast::makeError<void>(
          "Failed to allocate memory for gate matrix.");
    }
  }

  std::vector<std::thread> threads;
  threads.reserve(nThreads);
  const uint64_t nTasksPerThread = nTasks / nThreads;

  // Thread t will access tasks from taskIds[t] to taskIds[t + 1]
  uint64_t taskIds[nThreads + 1];
  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx)
    taskIds[tIdx] = nTasksPerThread * tIdx;
  taskIds[nThreads] = nTasks; // Last thread will handle the rest

  void** argvs[nThreads];
  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx) {
    argvs[tIdx] = new void*[4];
    argvs[tIdx][0] = sv;                   // state vector
    argvs[tIdx][1] = &(taskIds[tIdx]);     // taskID begin
    argvs[tIdx][2] = &(taskIds[tIdx + 1]); // taskID end
    argvs[tIdx][3] = pMat;                 // matrix pointer
  }

  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx)
    threads.emplace_back(kernel.executable, argvs[tIdx]);
  for (auto& t : threads)
    t.join();

  // clean up
  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx)
    delete[] argvs[tIdx];
  std::free(pMat);
  return {}; // success
}

MaybeError<void>
CPUKernelManager::applyCPUKernel(void* sv,
                                 int nQubits,
                                 const std::string& llvmFuncName,
                                 int nThreads) const {
  const auto* kernel = getKernelByName(llvmFuncName);
  if (kernel == nullptr) {
    return cast::makeError<void>("Kernel not found: " + llvmFuncName);
  }
  auto rst = applyCPUKernel(sv, nQubits, *kernel, nThreads);
  if (!rst) {
    return cast::makeError<void>(rst.takeError());
  }
  return {}; // success
}

MaybeError<void> CPUKernelManager::applyCPUKernelsFromGraph(
    void* sv, int nQubits, const std::string& graphName, int nThreads) const {
  if (!isJITed()) {
    return cast::makeError<void>(
        "Must initialize JIT session before applying CPU kernel.");
  }
  if (!graphKernels_.contains(graphName)) {
    return cast::makeError<void>("Graph not found: " + graphName);
  }
  const auto& kernels = graphKernels_.at(graphName);
  for (const auto& kernel : kernels) {
    if (kernel->executable == nullptr) {
      std::ostringstream oss;
      oss << "Kernel '" << kernel->llvmFuncName << "' has no executable.";
      return cast::makeError<void>(oss.str());
    }
  }

  for (const auto& kernel : kernels) {
    auto result = applyCPUKernel(sv, nQubits, *kernel, nThreads);
    if (!result) {
      std::ostringstream oss;
      oss << "Failed to apply kernel '" << kernel->llvmFuncName
          << "': " << result.takeError();
      return cast::makeError<void>(oss.str());
    }
  }

  return {}; // success
}

// std::vector<CPUKernelInfo*>
// CPUKernelManager::collectCPUKernelsFromLegacyCircuitGraph(
//     const std::string& graphName) {
//   assert(isJITed() && "Must initialize JIT session "
//                       "before calling
//                       KernelManager::collectCPUGraphKernels");
//   std::vector<CPUKernelInfo*> kernelInfos;
//   const auto mangledName = internal::mangleGraphName(graphName);
//   for (auto& kernel : _kernels) {
//     if (kernel.llvmFuncName.starts_with(mangledName)) {
//       ensureExecutable(kernel);
//       kernelInfos.push_back(&kernel);
//     }
//   }
//   return kernelInfos;
// }

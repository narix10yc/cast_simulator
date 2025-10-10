#include "cast/CPU/CPUKernelManager.h"

using namespace cast;

// Allocates memory for the gate matrix to be used in invoking CPU kernels
// @return a raw pointer to the allocated memory. Remember to free it after use.
// TODO: refactor this function
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
    if (precision == Precision::FP32) {
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
      assert(precision == Precision::FP64 && "Unsupported precision");
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

llvm::Error CPUKernelManager::applyCPUKernel(void* sv,
                                             int nQubits,
                                             CPUKernelInfo& kernel,
                                             int nThreads) {
  if (kernel.executable == nullptr) {
    return llvm::createStringError(
        "Kernel executable not available. Did you call initJIT()?");
  }
  if (nThreads <= 0) {
    return llvm::createStringError("Invalid number of threads: " +
                                   llvm::Twine(nThreads));
  }

  const int simd_s = cast::get_simd_s(kernel.simdWidth, kernel.precision);
  const int tmp = nQubits - kernel.gate->nQubits() - simd_s;
  if (tmp < 0) {
    return llvm::createStringError(
        "Invalid number of qubits for the kernel '" +
        llvm::Twine(kernel.llvmFuncName) +
        "'. This kernel must act on statevectors with at least " +
        llvm::Twine(kernel.gate->nQubits() + simd_s) + " qubits.");
  }
  uint64_t nTasks = 1ULL << tmp;
  void* pMat = nullptr;
  if (kernel.matrixLoadMode == CPUMatrixLoadMode::StackLoadMatElems) {
    pMat = mallocGatePointer(kernel.gate.get(), kernel.precision);
    if (pMat == nullptr) {
      return llvm::createStringError(
          "Failed to allocate memory for gate matrix.");
    }
  }

  std::vector<std::thread> threads;
  threads.reserve(nThreads);
  const uint64_t nTasksPerThread = nTasks / nThreads;

  // Thread t will access tasks from taskIds[t] to taskIds[t + 1]
  std::unique_ptr<uint64_t[]> taskIds(new uint64_t[nThreads + 1]);
  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx)
    taskIds[tIdx] = nTasksPerThread * tIdx;
  taskIds[nThreads] = nTasks; // Last thread will handle the rest

  std::unique_ptr<void*[]> argvs(new void*[4 * nThreads]);
  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx) {
    argvs[tIdx * 4] = sv;                       // state vector
    argvs[tIdx * 4 + 1] = &(taskIds[tIdx]);     // taskID begin
    argvs[tIdx * 4 + 2] = &(taskIds[tIdx + 1]); // taskID end
    argvs[tIdx * 4 + 3] = pMat;                 // matrix pointer
  }

  kernel.tpExecStart = std::chrono::steady_clock::now();

  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx)
    threads.emplace_back(kernel.executable, &argvs[tIdx * 4]);
  for (auto& t : threads)
    t.join();

  kernel.tpExecFinish = std::chrono::steady_clock::now();

  // clean up
  std::free(pMat);
  return llvm::Error::success();
}

llvm::Error CPUKernelManager::applyCPUKernel(void* sv,
                                             int nQubits,
                                             const std::string& llvmFuncName,
                                             int nThreads) {
  auto* kernel = getKernelByName(llvmFuncName);
  if (kernel == nullptr)
    return llvm::createStringError("Kernel not found: " + llvmFuncName);

  if (auto e = applyCPUKernel(sv, nQubits, *kernel, nThreads))
    return e;
  return llvm::Error::success();
}

llvm::Error CPUKernelManager::applyCPUKernelsFromGraph(
    void* sv, int nQubits, const std::string& graphName, int nThreads) {
  if (!kernelPools_.contains(graphName)) {
    return llvm::createStringError("Graph not found: " + graphName);
  }
  const auto& kernels = kernelPools_.at(graphName);
  for (const auto& kernel : kernels.iter_kernels()) {
    if (kernel->executable == nullptr) {
      return llvm::createStringError("Kernel '" + kernel->llvmFuncName +
                                     "' has no executable.");
    }
  }

  for (const auto& kernel : kernels.iter_kernels()) {
    if (auto e = applyCPUKernel(sv, nQubits, *kernel, nThreads))
      return e;
  }

  return llvm::Error::success();
}

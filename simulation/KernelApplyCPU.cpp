#include "simulation/KernelManager.h"
#include "cast/CircuitGraph.h"

using namespace cast;

void CPUKernelManager::applyCPUKernel(
    void* sv, int nQubits, const std::string& funcName) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  for (auto& kernel : _kernels) {
    if (kernel.llvmFuncName == funcName) {
      applyCPUKernel(sv, nQubits, kernel);
      return;
    }
  }
  llvm_unreachable("KernelManager::applyCPUKernel: kernel not found by name");
}

void CPUKernelManager::applyCPUKernel(
    void* sv, int nQubits, CPUKernelInfo& kernel) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  ensureExecutable(kernel);
  int tmp = nQubits - kernel.gate->nQubits() - kernel.simd_s;
  assert(tmp > 0);
  uint64_t idxEnd = 1ULL << tmp;
  const void* pMat = nullptr;
  if (kernel.gate->gateMatrix.getConstantMatrix() != nullptr)
    pMat = kernel.gate->gateMatrix.getConstantMatrix()->data();
  kernel.executable(sv, 0ULL, idxEnd, pMat);
}

void CPUKernelManager::applyCPUKernel(
    void* sv, 
    int nQubits,
    const std::string& funcName,
    const void* pMatArg
) {
  assert(isJITed() && "Must initialize JIT session before calling applyCPUKernel");
  for (auto& kernel : _kernels) {
    if (kernel.llvmFuncName == funcName) {
      // Found the right kernel
      applyCPUKernel(sv, nQubits, kernel, pMatArg);
      return;
    }
  }
  llvm_unreachable("KernelManager::applyCPUKernel(pMatArg): kernel not found by name");
}

void CPUKernelManager::applyCPUKernel(
    void* sv, 
    int nQubits,
    CPUKernelInfo& kernel,
    const void* pMatArg
) {
  assert(isJITed() && "Must initialize JIT session before calling applyCPUKernel");
  ensureExecutable(kernel);
  int tmp = nQubits - kernel.gate->nQubits() - kernel.simd_s;
  assert(tmp >= 0 && "Something’s off with qubit count");
  uint64_t idxEnd = 1ULL << tmp;
  kernel.executable(sv, 0ULL, idxEnd, pMatArg);
}

void CPUKernelManager::applyCPUKernelMultithread(
    void* sv, int nQubits, CPUKernelInfo& kernel, int nThreads) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  ensureExecutable(kernel);
  int tmp = nQubits - kernel.gate->nQubits() - kernel.simd_s;
  assert(tmp > 0);
  uint64_t nTasks = 1ULL << tmp;
  const void* pMat = nullptr;
  if (kernel.gate->gateMatrix.getConstantMatrix() != nullptr)
    pMat = kernel.gate->gateMatrix.getConstantMatrix()->data();

  std::vector<std::thread> threads;
  threads.reserve(nThreads);
  const uint64_t nTasksPerThread = nTasks / nThreads;

  for (unsigned tIdx = 0; tIdx < nThreads - 1; ++tIdx) {
    threads.emplace_back(kernel.executable, sv,
      nTasksPerThread * tIdx, nTasksPerThread * (tIdx + 1), pMat);
  }
  threads.emplace_back(kernel.executable, sv,
    nTasksPerThread * (nThreads - 1), nTasks, pMat);
  for (auto& t : threads)
    t.join();
}

void CPUKernelManager::applyCPUKernelMultithread(
    void* sv, int nQubits, const std::string& funcName, int nThreads) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  for (auto& kernel : _kernels) {
    if (kernel.llvmFuncName == funcName) {
      applyCPUKernelMultithread(sv, nQubits, kernel, nThreads);
      return;
    }
  }
  llvm_unreachable("KernelManager::applyCPUKernelMultithread: "
                   "kernel not found by name");
}

std::vector<CPUKernelInfo*>
CPUKernelManager::collectCPUKernelsFromCircuitGraph(const std::string& graphName) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::collectCPUGraphKernels");
  std::vector<CPUKernelInfo*> kernelInfos;
  const auto mangledName = internal::mangleGraphName(graphName);
  for (auto& kernel : _kernels) {
    if (kernel.llvmFuncName.starts_with(mangledName)) {
      ensureExecutable(kernel);
      kernelInfos.push_back(&kernel);
    }
  }
  return kernelInfos;
}


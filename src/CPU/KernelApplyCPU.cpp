#include "cast/CPU/KernelManagerCPU.h"
#include "cast/Legacy/CircuitGraph.h"

using namespace cast;

static void* mallocGatePointer(cast::QuantumGate* gate, int precision) {
  void* p = nullptr;
  auto* standardQuGate = llvm::dyn_cast<cast::StandardQuantumGate>(gate);
  if (standardQuGate != nullptr) {
    auto* gateMatrix = standardQuGate->gateMatrix().get();
    assert(gateMatrix != nullptr && "Empty gate matrix");
    auto* scalarGM = llvm::dyn_cast<cast::ScalarGateMatrix>(gateMatrix);
    assert(scalarGM != nullptr && "Only ScalarGateMatrix is supported for now");
    const auto& mat = scalarGM->matrix();
    auto edgeSize = mat.edgeSize();
    if (precision == 32) {
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
      assert(precision == 64 && "Unsupported precision");
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
  }
  else {
    assert(false && "Only StandardQuantumGate is supported for now");
  }

  assert(p != nullptr && "Failed to allocate memory for gate matrix");
  return p;
}

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
  assert(tmp >= 0);
  uint64_t idxEnd = 1ULL << tmp;
  void* pMat = nullptr;
  if (kernel.matrixLoadMode == MatrixLoadMode::StackLoadMatElems)
    pMat = mallocGatePointer(kernel.gate.get(), kernel.precision);
  
  uint64_t taskIDBegin = 0ULL;
  uint64_t taskIDEnd = idxEnd;
  void* args[4];
  args[0] = sv; // state vector
  args[1] = &taskIDBegin; // taskID begin
  args[2] = &taskIDEnd; // taskID end
  args[3] = pMat; // matrix pointer
  kernel.executable(args);
  // std::cerr << "Single-thread: passes in argument "
  //           << (void*)(args) << " with "
  //           << "sv = " << (void*)sv << ", "
  //           << "arg[0] = " << *args << ", "
  //           << "arg[1] = " << *(uint64_t*)args[1] << ", "
  //           << "arg[2] = " << *(uint64_t*)args[2] << ", "
  //           << "arg[3] = " << args[3] << "\n";
  std::free(pMat);
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
    const void* pMatArg) {
  assert(isJITed() && "Must initialize JIT session before calling applyCPUKernel");
  ensureExecutable(kernel);
  int tmp = nQubits - kernel.gate->nQubits() - kernel.simd_s;
  assert(tmp >= 0 && "Something's off with qubit count");
  uint64_t idxBegin = 0ULL;
  uint64_t idxEnd = 1ULL << tmp;
  void* argv[4];
  argv[0] = sv; // state vector pointer
  argv[1] = static_cast<void*>(&idxBegin); // taskID begin
  argv[2] = static_cast<void*>(&idxEnd); // taskID end
  argv[3] = const_cast<void*>(pMatArg); // matrix pointer
  kernel.executable(argv);
}

void CPUKernelManager::applyCPUKernelMultithread(
    void* sv, int nQubits, CPUKernelInfo& kernel, int nThreads) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  ensureExecutable(kernel);
  int tmp = nQubits - kernel.gate->nQubits() - kernel.simd_s;
  assert(tmp >= 0);
  uint64_t nTasks = 1ULL << tmp;
  void* pMat = nullptr;
  if (kernel.matrixLoadMode == MatrixLoadMode::StackLoadMatElems)
    pMat = mallocGatePointer(kernel.gate.get(), kernel.precision);

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
    argvs[tIdx][0] = sv; // state vector
    argvs[tIdx][1] = &(taskIds[tIdx]); // taskID begin
    argvs[tIdx][2] = &(taskIds[tIdx + 1]); // taskID end
    argvs[tIdx][3] = pMat; // matrix pointer

    // void** arg = argvs[tIdx];
    // std::cerr << "Thread " << tIdx << " passes in argument "
    //           << (void*)(arg) << " with "
    //           << "sv = " << (void*)sv << ", "
    //           << "arg[0] = " << *arg << ", "
    //           << "arg[1] = " << *(uint64_t*)arg[1] << ", "
    //           << "arg[2] = " << *(uint64_t*)arg[2] << ", "
    //           << "arg[3] = " << arg[3] << "\n";
  }

  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx)
    threads.emplace_back(kernel.executable, argvs[tIdx]);
  for (auto& t : threads)
    t.join();

  // clean up
  for (unsigned tIdx = 0; tIdx < nThreads; ++tIdx)
    delete[] argvs[tIdx];
  std::free(pMat);
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

// std::vector<CPUKernelInfo*>
// CPUKernelManager::collectCPUKernelsFromLegacyCircuitGraph(
//     const std::string& graphName) {
//   assert(isJITed() && "Must initialize JIT session "
//                       "before calling KernelManager::collectCPUGraphKernels");
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


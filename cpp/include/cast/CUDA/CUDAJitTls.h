#ifndef CAST_CUDA_CUDAJITTLS_H
#define CAST_CUDA_CUDAJITTLS_H

#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Target/TargetMachine.h"
#include <llvm/Passes/PassBuilder.h>

#include <memory>

namespace cast {

// It is a fatal error if the NVPTX target machine cannot be created.
class CUDAJitTls {
  std::unique_ptr<llvm::TargetMachine> targetMachine_;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassBuilder PB;
  llvm::ModulePassManager MPM;

public:
  CUDAJitTls();

  CUDAJitTls(const CUDAJitTls&) = delete;
  CUDAJitTls& operator=(const CUDAJitTls&) = delete;
  CUDAJitTls(CUDAJitTls&&) = delete;
  CUDAJitTls& operator=(CUDAJitTls&&) = delete;

  ~CUDAJitTls() = default;

  llvm::TargetMachine& getTargetMachine() { return *targetMachine_; }

  const llvm::TargetMachine& getTargetMachine() const {
    return *targetMachine_;
  }

  // Run optimization passes on the given module.
  void runOnModule(llvm::Module& module, llvm::OptimizationLevel optLevel);

}; // class CUDAJitTls

} // namespace cast

#endif // CAST_CUDA_CUDAJITTLS_H
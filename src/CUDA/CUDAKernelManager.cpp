#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"

#include "utils/Formats.h"
#include "utils/InfoLogger.h"
#include "utils/iocolor.h"

#include <atomic>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

#include <nvJitLink.h>

#define DEBUG_TYPE "kernel-mgr-cuda"
#include <llvm/Support/Debug.h>

// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

void CUDAKernelGenConfig::displayInfo(utils::InfoLogger logger) const {
  logger.put("Precision", precision)
      .put("Zero Tolerance", zeroTol)
      .put("One Tolerance", oneTol)
      .put("Matrix Load Mode", [&](std::ostream& os) {
        switch (matrixLoadMode) {
        case CUDAMatrixLoadMode::UseMatImmValues:
          os << "UseMatImmValues";
          break;
        case CUDAMatrixLoadMode::LoadInDefaultMemSpace:
          os << "LoadInDefaultMemSpace";
          break;
        case CUDAMatrixLoadMode::LoadInConstMemSpace:
          os << "LoadInConstMemSpace";
          break;
        default:
          os << "Unknown";
          break;
        }
      });
}

void CudaKernel::displayInfo(utils::InfoLogger logger) const {
  auto status = this->status.load(std::memory_order_acquire);
  switch (status) {
  case Status::Empty:
    logger.put("Status", "Empty");
    return;
  case Pending:
    logger.put("Status", "Pending");
    return;
  case Status::Compiling:
    logger.put("Status", "Compiling");
    return;
  case Status::Ready:
    logger.put("Status", "Ready");
    break; // continue to display more info
  }
  logger.put("Kernel Name", llvmFunc->getName().str())
      .put("Precision ", precision)
      .put("PTX Size  ", utils::fmt_mem(ptxString.size()))
      .put("Cubin Size", utils::fmt_mem(cubinData.size()));
}

void CUDAKernelManager::displayInfo(utils::InfoLogger logger) const {
  logger.put("Num Worker Threads", tPool.getNumWorkers());
}

namespace {
class raw_pwrite_vector_ostream : public llvm::raw_pwrite_stream {
  std::string& out;

  void write_impl(const char* Ptr, size_t Size) override {
    out.append(Ptr, Size);
  }

  void pwrite_impl(const char* Ptr, size_t Size, uint64_t Offset) override {
    if (out.size() < Offset + Size)
      out.resize(static_cast<size_t>(Offset + Size));
    std::memcpy(out.data() + Offset, Ptr, Size);
  }

  uint64_t current_pos() const override { return out.size(); }

public:
  explicit raw_pwrite_vector_ostream(std::string& str) : out(str) {
    SetUnbuffered();
  }
  ~raw_pwrite_vector_ostream() override { flush(); }
};
} // namespace

static inline llvm::OptimizationLevel wrapLLVMOptLevel(int llvmOptLevel) {
  if (llvmOptLevel == 0)
    return llvm::OptimizationLevel::O0;
  if (llvmOptLevel == 1 || llvmOptLevel < 0)
    return llvm::OptimizationLevel::O1;
  if (llvmOptLevel == 2)
    return llvm::OptimizationLevel::O2;
  // llvmOptLevel >= 3
  return llvm::OptimizationLevel::O3;
}

static llvm::Error
optimizeLLVMIR_work(int llvmOptLevel, CudaKernel& kernel, CUDAJitTls& jitTls) {
  if (kernel.llvmModule == nullptr)
    return llvm::createStringError("LLVM module is null");

  auto optLevel = wrapLLVMOptLevel(llvmOptLevel);
  jitTls.runOnModule(*kernel.llvmModule, optLevel);

  std::string errLog;
  llvm::raw_string_ostream rso(errLog);
  if (llvm::verifyModule(*kernel.llvmModule, &rso))
    return llvm::createStringError("Module verification failed: " + errLog);

  return llvm::Error::success();
}

static llvm::Error compileLLVMIRToPTX_work(CudaKernel& kernel,
                                           CUDAJitTls& jitTls) {
  if (kernel.llvmModule == nullptr)
    return llvm::createStringError("LLVM module is null");

  auto& TM = jitTls.getTargetMachine();

  kernel.llvmModule->setTargetTriple(TM.getTargetTriple());
  kernel.llvmModule->setDataLayout(TM.createDataLayout());
  std::string errorStr;
  llvm::raw_string_ostream sstream(errorStr);
  if (llvm::verifyModule(*kernel.llvmModule, &sstream))
    return llvm::createStringError("Module verification failed: " + errorStr);

  raw_pwrite_vector_ostream vecStream(kernel.ptxString);
  legacy::PassManager PM;
  // this function returns false on success
  if (TM.addPassesToEmitFile(
          PM, vecStream, nullptr, CodeGenFileType::AssemblyFile)) {
    return llvm::createStringError("LLVM target machine can't emit PTX");
  }

  PM.run(*kernel.llvmModule);

  return llvm::Error::success();
}

static llvm::Error compileToCubin_work(int cuOptLevel, CudaKernel& kernel) {
  if (kernel.ptxString.empty())
    return llvm::createStringError("PTX is empty");

  std::vector<const char*> options;
  int major, minor;
  cast::getCudaComputeCapability(major, minor);
  std::string archOption = "-arch=sm_" + std::to_string(10 * major + minor);
  options.push_back(archOption.c_str());

  nvJitLinkHandle handle;
  NVJITLINK_CHECK(handle,
                  nvJitLinkCreate(&handle, options.size(), options.data()));

  NVJITLINK_CHECK(handle,
                  nvJitLinkAddData(handle,
                                   NVJITLINK_INPUT_PTX,
                                   kernel.ptxString.data(),
                                   kernel.ptxString.size(),
                                   nullptr));

  NVJITLINK_CHECK(handle, nvJitLinkComplete(handle));
  size_t cubinSize = 0;
  NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
  void* cubinOut = std::malloc(cubinSize);
  NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubin(handle, cubinOut));

  kernel.cubinData.assign(static_cast<uint8_t*>(cubinOut),
                          static_cast<uint8_t*>(cubinOut) + cubinSize);

  std::free(cubinOut);
  NVJITLINK_CHECK(handle, nvJitLinkDestroy(&handle));

  return llvm::Error::success();
}

void CUDAKernelManager::enqueueForCompilation(CudaKernel* kernel) {
  using Status = CudaKernel::Status;
  kernel->status.store(Status::Pending, std::memory_order_release);

  tPool.enqueueMayErr([this, kernel]() -> llvm::Error {
    auto& tls = tPool.getTLS();
    llvm::Error err = llvm::Error::success();

    kernel->status.store(Status::Compiling, std::memory_order_release);

    err =
        llvm::joinErrors(optimizeLLVMIR_work(1, *kernel, tls), std::move(err));
    err =
        llvm::joinErrors(compileLLVMIRToPTX_work(*kernel, tls), std::move(err));
    err = llvm::joinErrors(compileToCubin_work(1, *kernel), std::move(err));

    kernel->status.store(Status::Ready, std::memory_order_release);
    return err;
  });
}

void CUDAKernelManager::syncCompilation() { tPool.sync(); }

void CUDAKernelManager::syncKernelExecution() { tPool.sync(); }

#undef DEBUG_TYPE
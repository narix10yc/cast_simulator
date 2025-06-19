#include "llvm/Support/TargetSelect.h"

#include "llvm/Object/ObjectFile.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/FormattedStream.h"

#include "cast/CPU/CPUKernelManager.h"
#include "utils/iocolor.h"
#include "utils/TaskDispatcher.h"

#include <cassert>

using namespace cast;
using namespace llvm;

std::ostream& CPUKernelGenConfig::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== CPU Kernel Gen Config ===\n")
     << "simd_s:     " << simd_s << "\n"
     << "precision:  " << precision << "\n"
     << "amp format: ";
  switch (this->ampFormat) {
    case AltFormat:
      os << "AltFormat\n"; break;
    case SepFormat:
      os << "SepFormat\n"; break;
    default:
      assert(0 && "Unreachable");
  }

  os << "useFMA:         " << useFMA << "\n"
     << "useFMS:         " << useFMS << "\n"
     << "usePDEP:        " << usePDEP << "\n"
     << "zeroTolerance:  " << zeroTol << "\n"
     << "oneTolerance:   " << oneTol << "\n"
     << "matrixLoadMode: ";
  switch (this->matrixLoadMode) {
    case MatrixLoadMode::UseMatImmValues:
      os << "UseMatImmValues\n"; break;
    case MatrixLoadMode::StackLoadMatElems:
      os << "StackLoadMatElems\n"; break;
    case MatrixLoadMode::StackLoadMatVecs:
      os << "StackLoadMatVecs\n"; break;
  }

  os << CYAN("================================\n");
  return os;
}

void CPUKernelManager::ensureAllExecutable(int nThreads, bool progressBar) {
  assert(nThreads > 0);
  if (nThreads == 1) {
    for (auto& kernel : _kernels)
      ensureExecutable(kernel);
    return;
  }

  // multi-thread compile
  utils::TaskDispatcher dispatcher(nThreads);
  for (auto& kernel : _kernels) {
	  dispatcher.enqueue([this, &kernel]() {
      ensureExecutable(kernel);
	  });
  }
  if (progressBar)
    std::cerr << "Ensure All Executables...\n";
  dispatcher.sync(progressBar);
}

MaybeError<void> CPUKernelManager::initJIT(
    int nThreads, OptimizationLevel optLevel, bool useLazyJIT, int verbose) {
  if (nThreads <= 0) {
    return cast::makeError<void>(
      "Invalid number of threads: " + std::to_string(nThreads));
  }
  if (isJITed()) {
    return cast::makeError<void>("JIT has already been initialized.");
  }

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  applyLLVMOptimization(nThreads, optLevel, /* progressBar */ verbose > 0);

  if (useLazyJIT) {
    // lazy JIT engine
    orc::LLLazyJITBuilder jitBuilder;
    /// It seems not matter the concurrency we set here.
    /// As long as we set it, we can invoke multiple lookup. We control the 
    /// actual number of threads via our custom TaskDispatcher
    jitBuilder.setNumCompileThreads(nThreads);
    auto lazyJIT = cantFail(jitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      auto err = lazyJIT->addLazyIRModule(
        orc::ThreadSafeModule(std::move(mod), std::move(ctx)));
      if (err) {
        return cast::makeError<void>(
            "Failed to add lazy IR module: " + llvm::toString(std::move(err)));
      }
    }
    this->llvmJIT = std::move(lazyJIT);
    ensureAllExecutable(nThreads, /* progressBar */ verbose > 0);
  } else {
    // eager JIT engine
    orc::LLJITBuilder eagerJitBuilder;
    eagerJitBuilder.setNumCompileThreads(nThreads);
    auto eagerJIT = cantFail(eagerJitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      auto err = eagerJIT->addIRModule(
        orc::ThreadSafeModule(std::move(mod), std::move(ctx)));
      if (err) {
        return cast::makeError<void>(
            "Failed to add IR module: " + llvm::toString(std::move(err)));
      }
    }
    this->llvmJIT = std::move(eagerJIT);
    // eager compile all kernels
    ensureAllExecutable(nThreads, /* progressBar */ verbose > 0);
  }
  this->llvmContextModulePairs.clear();
  return {}; // success
}

void CPUKernelManager::dumpIR(const std::string& funcName,
                              llvm::raw_ostream& os) {
  assert(isJITed() == false && "Only supports un-JITed kernels");

  for (const auto& ctxModPair : llvmContextModulePairs) {
    if (auto* func = ctxModPair.llvmModule->getFunction(funcName)) {
      func->print(os, nullptr);
      return;
    }
  }
  std::cerr << RED("[Err] ") << "In CPUKernelManager::dumpIR: "
            << "Function " << funcName << " not found.\n";
}


/// Dump the native assembly of a JIT-compiled kernel using LLVM's disassembler API.
/// This does not use disk I/O. The output will be written to the given std::ostream.
/// Note: This is a simplified example and may need adaptation for your JIT setup.
// void CPUKernelManager::dumpAsm(const std::string& funcName, llvm::raw_ostream& os) {
//   using namespace llvm;

//   assert(llvmJIT && "JIT must be initialized");

//   // Find the symbol address
//   auto symOrErr = llvmJIT->lookup(funcName);
//   if (!symOrErr) {
//     os << RED("[Err] ") << "In CPUKernelManager::dumpAsm: funcName "
//        << funcName << " not found in JIT.\n";
//     return;
//   }
//   uint64_t funcAddr = symOrErr.get().getValue();

//   // Get the object buffer from the JIT's ObjectLinkingLayer
//   auto& objLayer = llvmJIT->getObjLinkingLayer();
//   bool found = false;

//   for (auto it = objLayer.get)
//   objLayer.forEachObject([&](orc::MaterializationResponsibility&,
//                              const MemoryBufferRef& objBuffer) {
//     auto objOrErr = object::ObjectFile::createObjectFile(objBuffer);
//     if (!objOrErr)
//       return true; // continue

//     auto& obj = **objOrErr;
//     for (const auto& sym : obj.symbols()) {
//       auto addrOrErr = sym.getAddress();
//       if (!addrOrErr)
//         continue;
//       if (*addrOrErr == funcAddr) {
//         // Found the symbol in this object
//         auto nameOrErr = sym.getName();
//         if (!nameOrErr)
//           continue;

//         std::string tripleName = obj.makeTriple().getTriple();
//         std::string error;
//         const Target* target = TargetRegistry::lookupTarget(tripleName, error);
//         if (!target) {
//           os << "Target not found: " << error << "\n";
//           found = true;
//           return false;
//         }

//         // Set up disassembler
//         std::string cpu = "generic";
//         SubtargetFeatures features;
//         auto sti = target->createMCSubtargetInfo(tripleName, cpu, features.getString());
//         auto mri = target->createMCRegInfo(tripleName);
//         auto asmInfo = target->createMCAsmInfo(*mri, tripleName);
//         auto mcii = target->createMCInstrInfo();

//         // Modern MCContext construction
//         auto fileInfo = std::make_unique<MCObjectFileInfo>();
//         auto ctx = std::make_unique<MCContext>(asmInfo.get(), mri.get(), fileInfo.get());
//         fileInfo->InitMCObjectFileInfo(Triple(tripleName), /*PIC=*/false, *ctx, /*LargeCodeModel=*/false);

//         auto disAsm = target->createMCDisassembler(*sti, *ctx);
//         auto ip = target->createMCInstPrinter(
//             Triple(tripleName), asmInfo->getAssemblerDialect(), *asmInfo, *mcii, *mri);

//         if (!disAsm || !ip) {
//           os << "Failed to create disassembler or printer.\n";
//           found = true;
//           return false;
//         }

//         // Find section containing the symbol
//         for (const auto& sec : obj.sections()) {
//           uint64_t secAddr = sec.getAddress();
//           uint64_t secSize = sec.getSize();
//           if (funcAddr >= secAddr && funcAddr < secAddr + secSize) {
//             auto secDataOrErr = sec.getContents();
//             if (!secDataOrErr) {
//               os << "Failed to get section contents.\n";
//               found = true;
//               return false;
//             }
//             StringRef secData = *secDataOrErr;
//             uint64_t offset = funcAddr - secAddr;
//             if (offset >= secSize) {
//               os << "Offset out of section bounds.\n";
//               found = true;
//               return false;
//             }
//             ArrayRef<uint8_t> bytes(
//                 reinterpret_cast<const uint8_t*>(secData.data()) + offset,
//                 secSize - offset);

//             uint64_t index = 0;
//             uint64_t absAddr = funcAddr;
//             raw_os_ostream llvm_os(os);
//             while (index < bytes.size()) {
//               MCInst inst;
//               uint64_t size = 0;
//               if (disAsm->getInstruction(inst, size, bytes.slice(index), absAddr, nulls(), nulls())) {
//                 llvm_os << format_hex(absAddr, 10) << ":\t";
//                 ip->printInst(&inst, llvm_os, "", *sti);
//                 llvm_os << "\n";
//                 absAddr += size;
//                 index += size;
//               } else {
//                 llvm_os << format_hex(absAddr, 10) << ":\t<unknown>\n";
//                 break;
//               }
//             }
//             found = true;
//             return false;
//           }
//         }
//       }
//     }
//     return !found; // continue if not found
//   });

//   if (!found) {
//     os << "Could not find native code for function " << funcName << ".\n";
//   }
// }


#undef DEBUG_TYPE
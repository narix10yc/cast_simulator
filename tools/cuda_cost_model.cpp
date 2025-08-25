#include "cast/CUDA/CUDACostModel.h"
#include "cast/CUDA/CUDAKernelManager.h"

#include "utils/iocolor.h"
#include "llvm/Support/CommandLine.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

namespace cl = llvm::cl;
using namespace cast;

cl::opt<std::string>
    ArgOutputFilename("o", cl::desc("Output file name"), cl::Required);

cl::opt<bool>
    ArgOverwriteMode("overwrite",
                     cl::desc("Overwrite the output file with new results"),
                     cl::init(false));

cl::opt<int> ArgNQubits("nqubits", cl::desc("Number of qubits"), cl::init(28));

cl::opt<int> ArgBlockSize("block-size",
                          cl::desc("CUDA block size (power of two ≤ 1024)"),
                          cl::init(256));

cl::opt<int> ArgWorkerThreads(
    "worker-threads",
    cl::desc("CPU threads used for JIT compilation (if applicable)"),
    cl::init(0));

cl::opt<int> ArgNTests("N",
                       cl::desc("Number of benchmarked kernels"),
                       cl::Prefix,
                       cl::Required);

cl::opt<bool> ArgF32("f32",
                     cl::desc("Enable single-precision (f32) kernels"),
                     cl::init(false));

cl::opt<bool> ArgF64("f64",
                     cl::desc("Enable double-precision (f64) kernels"),
                     cl::init(true));

// ---- helpers ----------------------------------------------------------------

static bool validateBlockSize(int blk) {
  if (blk < 32 || blk > 1024 || (blk & (blk - 1)) != 0) {
    std::cerr << BOLDRED("[Error]: ")
              << "Block size must be a power-of-two in [32, 1024].\n";
    return false;
  }
  return true;
}

static void ensureCsvHeader(std::ofstream& ofs) {
  ofs << CUDAPerformanceCache::Item::CSV_TITLE << '\n';
}

static CUDADeviceInfo getDeviceInfo(int device, bool verbose) {
  CUDADeviceInfo dev{};
  dev.device = device;

  cudaDeviceProp props{};
  cudaError_t st = cudaGetDeviceProperties(&props, device);
  if (st != cudaSuccess) {
    std::cerr << BOLDRED("[Error]: ") << "cudaGetDeviceProperties(" << device
              << ") failed: " << cudaGetErrorString(st) << "\n";
    // Provide safe fallbacks to avoid UB
    dev.warpSize = 32;
    dev.maxThreadsPerSM = 2048;
    dev.smCount = 1;
    return dev;
  }

  dev.warpSize = props.warpSize;
  dev.maxThreadsPerSM = props.maxThreadsPerMultiProcessor;
  dev.smCount = props.multiProcessorCount;

  if (verbose) {
    std::cerr << BOLDCYAN("[Info]: ") << "GPU " << device << " = " << props.name
              << " | SMs=" << dev.smCount << " | warp=" << dev.warpSize
              << " | maxThreads/SM=" << dev.maxThreadsPerSM << "\n";
  }
  return dev;
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv);

  if (!validateBlockSize(ArgBlockSize))
    return 1;

  std::ifstream inFile;
  std::ofstream outFile;

  if (ArgOverwriteMode)
    outFile.open(ArgOutputFilename, std::ios::out | std::ios::trunc);
  else
    outFile.open(ArgOutputFilename, std::ios::app);

  inFile.open(ArgOutputFilename, std::ios::in);
  if (!outFile || !inFile) {
    std::cerr << BOLDRED("[Error]: ") << "Unable to open '" << ArgOutputFilename
              << "'.\n";
    return 1;
  }
  if (inFile.peek() == std::ifstream::traits_type::eof())
    ensureCsvHeader(outFile);
  inFile.close();

  // Build device info (measurement-based; we read real device caps once)
  int devId = 0;
  (void)cudaGetDevice(&devId); // if not set, default to 0
  const bool verboseDevice = true;
  CUDADeviceInfo dev = getDeviceInfo(devId, verboseDevice);

  CUDAPerformanceCache cache;
  CUDAKernelGenConfig cfg;
  cfg.blockSize = ArgBlockSize;

  std::cerr << BOLDCYAN("[Info]: ") << "Benchmarking " << ArgNTests
            << " kernels on " << ArgNQubits << " qubits – block "
            << ArgBlockSize << ", worker threads " << ArgWorkerThreads << ".\n";

  // Each precision is measured independently and appended to the CSV
  if (ArgF32) {
    std::cerr << BOLDCYAN("[Info]: ") << "  • Single precision (f32)\n";
    cfg.precision = Precision::F32;
    cache.runExperiments(cfg,
                         dev,
                         ArgNQubits,
                         ArgNTests,
                         /*verbose=*/1);
  }
  if (ArgF64) {
    std::cerr << BOLDCYAN("[Info]: ") << "  • Double precision (f64)\n";
    cfg.precision = Precision::F64;
    cache.runExperiments(cfg,
                         dev,
                         ArgNQubits,
                         ArgNTests,
                         /*verbose=*/1);
  }

  cache.writeResults(outFile);
  outFile.close();

  std::cerr << BOLDCYAN("[Info]: ") << "Results appended to '"
            << ArgOutputFilename << "'.\n";
  return 0;
}

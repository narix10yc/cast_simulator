/*
 * demo_cuda_bwmat.cpp
 * CUDA Bandwidth Matrix
 */

// g++ -O2 driver_p2p_copy.cpp -o driver_p2p_copy -lcuda
// (or) nvcc -O2 driver_p2p_copy.cpp -o driver_p2p_copy -lcuda

#include "cast/CUDA/Config.h"

#include "utils/Formats.h"
#include "utils/iocolor.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cuda.h>
#include <iostream>
#include <memory>
#include <vector>

#include <llvm/Support/CommandLine.h>

namespace cl = llvm::cl;

using namespace cast;

#pragma region Command line arguments
// clang-format off

static cl::OptionCategory Category("Demo CUDA Bandwidth Matrix Options");

static cl::opt<int> ArgSize("size", cl::cat(Category),
   cl::desc("Size of the buffer in MiB (default: 1024 MiB)"),
   cl::init(1024));

static cl::opt<int> ArgIterations("iterations", cl::cat(Category),
   cl::desc("Number of iterations to average bandwidth over (default: 10)"),
   cl::init(10));

static cl::opt<int> ArgWarmup("warmup", cl::cat(Category),
   cl::desc("Number of warm-up iterations (default: 2)"),
   cl::init(2));

static cl::opt<int> ArgVerbose("verbose", cl::cat(Category),
   cl::desc("Log verbosity level (default: 1)"),
   cl::init(1));

// clang-format on

static llvm::Error checkClArgs() {
  if (ArgSize <= 0)
    return llvm::createStringError("Buffer size must be positive");

  if (ArgIterations <= 0)
    return llvm::createStringError("Number of iterations must be positive");

  if (ArgWarmup < 0)
    return llvm::createStringError("Number of warm-ups must be non-negative");

  return llvm::Error::success();
}

#pragma endregion

static std::ostream& logerr() { return std::cerr << BOLDRED("[Err] "); }

static void
printBwMatrix(const char* title, const std::vector<double>& data, int ndev) {
  std::cerr << title << "\n";
  std::cerr << "    ";
  for (int j = 0; j < ndev; ++j)
    std::cerr << "GPU:" << j << " ";
  std::cerr << "\n";
  for (int i = 0; i < ndev; ++i) {
    std::cerr << "GPU:" << i << " ";
    for (int j = 0; j < ndev; ++j) {
      if (i == j) {
        std::cerr << "  -  ";
      } else {
        std::cerr << " " << utils::fmt_mem(data[i * ndev + j]) << " ";
      }
    }
    std::cerr << "\n";
  }
}

class BWMatrix {

public:
  BWMatrix(int ndev)
      : ndev(ndev), h2d_uni(ndev * ndev), d2h_uni(ndev * ndev),
        d2d_uni(ndev * ndev), h2d_bi(ndev * ndev), d2d_bi(ndev * ndev) {}

  // number of devices
  int ndev;

  // Uni-directional host -> device
  std::vector<double> h2d_uni;
  // Uni-directional device -> host
  std::vector<double> d2h_uni;
  // Uni-directional device -> device
  std::vector<double> d2d_uni;

  // Bi-directional host <-> device
  std::vector<double> h2d_bi;
  // Bi-directional device <-> device
  std::vector<double> d2d_bi;
};

// Uni-directional device to device. Okay to have the same srcCtx and dstCtx, in
// which case it's just a device-local copy.
// Return the bandwidth in bytes per second (Bps)
// Uni-directional device to device.
// - If srcCtx == dstCtx: device-local D2D copy (cuMemcpyDtoDAsync).
// - Else: peer copy (cuMemcpyPeerAsync).
// Returns bandwidth in bytes per second (Bps).
static double runUniD2D(CUcontext srcCtx,
                        CUdeviceptr srcPtr,
                        CUcontext dstCtx,
                        CUdeviceptr dstPtr,
                        size_t bytes) {
  // We enqueue work on a stream that lives in dstCtx.
  CU_CHECK(cuCtxSetCurrent(dstCtx));

  CUstream s;
  CU_CHECK(cuStreamCreate(&s, CU_STREAM_DEFAULT));

  CUevent e0, e1;
  CU_CHECK(cuEventCreate(&e0, CU_EVENT_DEFAULT));
  CU_CHECK(cuEventCreate(&e1, CU_EVENT_DEFAULT));

  // Warm-up: enqueue then sync stream to settle clocks/routing.
  for (int _ = 0; _ < ArgWarmup; ++_) {
    if (srcCtx == dstCtx) {
      CU_CHECK(cuMemcpyDtoDAsync(dstPtr, srcPtr, bytes, s));
    } else {
      CU_CHECK(cuMemcpyPeerAsync(dstPtr, dstCtx, srcPtr, srcCtx, bytes, s));
    }
  }
  CU_CHECK(cuStreamSynchronize(s));

  // Timed runs: record events on the same stream that carries the copies.
  CU_CHECK(cuEventRecord(e0, s));
  for (int _ = 0; _ < ArgIterations; ++_) {
    if (srcCtx == dstCtx) {
      CU_CHECK(cuMemcpyDtoDAsync(dstPtr, srcPtr, bytes, s));
    } else {
      CU_CHECK(cuMemcpyPeerAsync(dstPtr, dstCtx, srcPtr, srcCtx, bytes, s));
    }
  }
  CU_CHECK(cuEventRecord(e1, s));
  CU_CHECK(cuEventSynchronize(e1));

  float ms = 0.0f;
  CU_CHECK(cuEventElapsedTime(&ms, e0, e1));

  CU_CHECK(cuEventDestroy(e0));
  CU_CHECK(cuEventDestroy(e1));
  CU_CHECK(cuStreamDestroy(s));

  // ms is total time for ArgIterations copies
  return (double)bytes * (double)ArgIterations / ((double)ms * 1e-3); // B/s
}

// Bi-directional device to device. Returns 0 if srcCtx == dstCtx.
// Return the bandwidth in bytes per second (Bps)
static double runBiD2D(CUcontext aCtx,
                       CUdeviceptr aSrcPtr,
                       CUdeviceptr aDstPtr,
                       CUcontext bCtx,
                       CUdeviceptr bSrcPtr,
                       CUdeviceptr bDstPtr,
                       size_t bytes) {
  if (aCtx == bCtx)
    return 0.0;

  // Create stream+events in each context
  CU_CHECK(cuCtxSetCurrent(aCtx));
  CUstream sA;
  CUevent eA0, eA1;
  CU_CHECK(cuStreamCreate(&sA, CU_STREAM_DEFAULT));
  CU_CHECK(cuEventCreate(&eA0, CU_EVENT_DEFAULT));
  CU_CHECK(cuEventCreate(&eA1, CU_EVENT_DEFAULT));

  CU_CHECK(cuCtxSetCurrent(bCtx));
  CUstream sB;
  CUevent eB0, eB1;
  CU_CHECK(cuStreamCreate(&sB, CU_STREAM_DEFAULT));
  CU_CHECK(cuEventCreate(&eB0, CU_EVENT_DEFAULT));
  CU_CHECK(cuEventCreate(&eB1, CU_EVENT_DEFAULT));

  // Warmup
  for (int w = 0; w < ArgWarmup; ++w) {
    CU_CHECK(cuCtxSetCurrent(aCtx));
    CU_CHECK(cuMemcpyPeerAsync(bDstPtr, bCtx, aSrcPtr, aCtx, bytes, sA));
    CU_CHECK(cuCtxSetCurrent(bCtx));
    CU_CHECK(cuMemcpyPeerAsync(aDstPtr, aCtx, bSrcPtr, bCtx, bytes, sB));
  }
  CU_CHECK(cuCtxSetCurrent(aCtx));
  CU_CHECK(cuStreamSynchronize(sA));
  CU_CHECK(cuCtxSetCurrent(bCtx));
  CU_CHECK(cuStreamSynchronize(sB));

  // Timed
  CU_CHECK(cuCtxSetCurrent(aCtx));
  CU_CHECK(cuEventRecord(eA0, sA));
  CU_CHECK(cuCtxSetCurrent(bCtx));
  CU_CHECK(cuEventRecord(eB0, sB));

  for (int it = 0; it < ArgIterations; ++it) {
    CU_CHECK(cuCtxSetCurrent(aCtx));
    CU_CHECK(cuMemcpyPeerAsync(bDstPtr, bCtx, aSrcPtr, aCtx, bytes, sA));
    CU_CHECK(cuCtxSetCurrent(bCtx));
    CU_CHECK(cuMemcpyPeerAsync(aDstPtr, aCtx, bSrcPtr, bCtx, bytes, sB));
  }

  CU_CHECK(cuCtxSetCurrent(aCtx));
  CU_CHECK(cuEventRecord(eA1, sA));
  CU_CHECK(cuCtxSetCurrent(bCtx));
  CU_CHECK(cuEventRecord(eB1, sB));

  CU_CHECK(cuEventSynchronize(eA1));
  CU_CHECK(cuEventSynchronize(eB1));

  float msA = 0.0f, msB = 0.0f;
  CU_CHECK(cuEventElapsedTime(&msA, eA0, eA1));
  CU_CHECK(cuEventElapsedTime(&msB, eB0, eB1));

  float ms = (msA > msB) ? msA : msB;
  double agg_Bps =
      (2.0 * (double)bytes * (double)ArgIterations) / ((double)ms * 1e-3);

  // Cleanup (destroy in owning contexts ideally; often works as-is if current
  // matches)
  CU_CHECK(cuCtxSetCurrent(aCtx));
  CU_CHECK(cuEventDestroy(eA0));
  CU_CHECK(cuEventDestroy(eA1));
  CU_CHECK(cuStreamDestroy(sA));

  CU_CHECK(cuCtxSetCurrent(bCtx));
  CU_CHECK(cuEventDestroy(eB0));
  CU_CHECK(cuEventDestroy(eB1));
  CU_CHECK(cuStreamDestroy(sB));

  return agg_Bps;
}

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(Category);
  cl::ParseCommandLineOptions(argc, argv);
  if (auto e = checkClArgs()) {
    logerr() << "Invalid command line arguments: "
             << llvm::toString(std::move(e)) << "\n";
    return 1;
  }

  CU_CHECK(cuInit(0));

  // number of devices
  int ndev = 0;
  CU_CHECK(cuDeviceGetCount(&ndev));
  if (ndev <= 0) {
    logerr() << "No CUDA devices found.\n";
    return 1;
  }
  std::vector<CUdevice> devices(ndev);
  std::vector<CUcontext> contexts(ndev);
  for (int i = 0; i < ndev; ++i) {
    CU_CHECK(cuDeviceGet(&devices[i], i));
    CU_CHECK(cuCtxCreate(&contexts[i], nullptr, 0, devices[i]));
  }

  // Check peer access capability
  std::vector<int> peerMatrix(ndev * ndev);
  for (int i = 0; i < ndev; ++i) {
    for (int j = 0; j < ndev; ++j) {
      if (i == j)
        continue;
      int can;
      CU_CHECK(cuDeviceCanAccessPeer(&can, devices[i], devices[j]));
      peerMatrix[i * ndev + j] = can;
    }
  }
  // Display peer access matrix
  std::cerr << "Peer Access Matrix:\n";
  std::cerr << "    ";
  for (int j = 0; j < ndev; ++j)
    std::cerr << "GPU:" << j << " ";
  std::cerr << "\n";
  for (int i = 0; i < ndev; ++i) {
    std::cerr << "GPU:" << i << " ";
    for (int j = 0; j < ndev; ++j) {
      if (i == j)
        std::cerr << "  -  ";
      else
        std::cerr << "  " << (peerMatrix[i * ndev + j] ? "Y" : "N") << "  ";
    }
    std::cerr << "\n";
  }

  // Enable peer access
  for (int i = 0; i < ndev; ++i) {
    for (int j = 0; j < ndev; ++j) {
      if (i == j)
        continue;
      if (peerMatrix[i * ndev + j]) {
        CU_CHECK(cuCtxSetCurrent(contexts[i]));
        CUresult e = cuCtxEnablePeerAccess(contexts[j], 0);
        if (e != CUDA_SUCCESS && e != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
          CU_CHECK(e);
      }
    }
  }

  const size_t bytes = ArgSize * 1024 * 1024; // Convert MiB to bytes

  // GPU receive & send ptrs
  std::vector<CUdeviceptr> dRecvPtrs(ndev);
  std::vector<CUdeviceptr> dSendPtrs(ndev);
  for (int i = 0; i < ndev; ++i) {
    CU_CHECK(cuCtxSetCurrent(contexts[i]));
    CUdeviceptr d;
    CU_CHECK(cuMemAlloc(&d, bytes));
    dRecvPtrs[i] = d;
    CU_CHECK(cuMemAlloc(&d, bytes));
    dSendPtrs[i] = d;
  }

  BWMatrix bwmat(ndev);

  for (int i = 0; i < ndev; ++i) {
    for (int j = 0; j < ndev; ++j) {
      double bw;

      bw = runUniD2D(
          contexts[i], dSendPtrs[i], contexts[j], dRecvPtrs[j], bytes);
      if (ArgVerbose >= 1) {
        std::cerr << "Uni D2D GPU:" << i << " -> GPU:" << j << " : "
                  << utils::fmt_mem(bw) << " per sec\n";
      }
      bwmat.d2d_uni[i * ndev + j] = bw;

      if (i != j) {
        // bi-directional only for different devices
        bw = runBiD2D(contexts[i],
                      dSendPtrs[i],
                      dRecvPtrs[i],
                      contexts[j],
                      dSendPtrs[j],
                      dRecvPtrs[j],
                      bytes);
        if (ArgVerbose >= 1) {
          std::cerr << "Bi D2D GPU:" << i << " -> GPU:" << j << " : "
                    << utils::fmt_mem(bw) << " per sec\n";
        }
        bwmat.d2d_bi[i * ndev + j] = bw;
      }
    }
  }

  // clean up
  for (int i = 0; i < ndev; ++i) {
    CU_CHECK(cuCtxSetCurrent(contexts[i]));
    CU_CHECK(cuMemFree(dRecvPtrs[i]));
    CU_CHECK(cuMemFree(dSendPtrs[i]));
    CU_CHECK(cuCtxDestroy(contexts[i]));
  }

  printBwMatrix("Uni-directional D2D Bandwidth Matrix:", bwmat.d2d_uni, ndev);
  printBwMatrix("Bi-directional D2D Bandwidth Matrix:", bwmat.d2d_bi, ndev);

  return 0;
}

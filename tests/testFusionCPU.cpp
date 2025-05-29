#include "cast/Legacy/CircuitGraph.h"
#include "cast/Legacy/Parser.h"
#include "cast/Fusion.h"
#include "cast/Core/KernelManager.h"
#include "tests/TestKit.h"
#include "cast/CPU/StatevectorCPU.h"

#include <filesystem>

namespace fs = std::filesystem;

using namespace cast::legacy;
using namespace utils;

template<unsigned simd_s>
static void f() {
  cast::test::TestSuite suite("Fusion CPU (s = " + std::to_string(simd_s) + ")");

  cast::CPUKernelManager kernelMgrBeforeFusion;
  cast::CPUKernelManager kernelMgrAfterFusion;

  cast::CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = simd_s;

  auto fusionConfig = cast::FusionConfig::Default;
  cast::NaiveCostModel costModel(4, -1, 0);

  std::cerr << "Test Dir: " << TEST_DIR << "\n";
  fs::path circuitDir = fs::path(TEST_DIR) / "circuits";
  if (!fs::exists(circuitDir) || !fs::is_directory(circuitDir)) {
    std::cerr << BOLDRED("Error: ") << "No circuit directory found\n";
    return;
  }
  for (const auto& p : fs::directory_iterator(circuitDir)) {
    if (!p.is_regular_file())
      continue;

    Parser parser(p.path().c_str());
    auto qc = parser.parseQuantumCircuit();
    CircuitGraph graph;
    qc.toLegacyCircuitGraph(graph);
    auto allBlocks = graph.getAllBlocks();
    for (const auto& block : allBlocks) {
      kernelMgrBeforeFusion.genCPUGate(
        kernelGenConfig, block->quantumGate,
        "beforeFusion" + std::to_string(block->id));
    }

    applyGateFusion(fusionConfig, &costModel, graph);
    allBlocks = graph.getAllBlocks();
    for (const auto& block : allBlocks) {
      kernelMgrAfterFusion.genCPUGate(
        kernelGenConfig, block->quantumGate,
        "afterFusion" + std::to_string(block->id));
    }

    kernelMgrBeforeFusion.initJIT();
    kernelMgrAfterFusion.initJIT();

    utils::StatevectorCPU<double> sv0(graph.nQubits, simd_s);
    utils::StatevectorCPU<double> sv1(graph.nQubits, simd_s);
    sv0.randomize();
    sv1 = sv0;

    for (const auto& k : kernelMgrBeforeFusion.kernels()) {
      kernelMgrBeforeFusion.applyCPUKernel(
        sv0.data(), sv0.nQubits(), k.llvmFuncName);
    }

    for (const auto& k : kernelMgrAfterFusion.kernels()) {
      kernelMgrAfterFusion.applyCPUKernel(
        sv1.data(), sv1.nQubits(), k.llvmFuncName);
    }

    suite.assertClose(utils::fidelity(sv0, sv1), 1.0,
      p.path().filename(), GET_INFO(), 1e-8);
  }

  suite.displayResult();
}

void cast::test::test_fusionCPU() {
  f<1>();
  f<2>();
}
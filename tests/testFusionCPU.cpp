#include "simulation/KernelManager.h"
#include "tests/TestKit.h"
#include "utils/statevector.h"

#include "saot/CircuitGraph.h"
#include "saot/Parser.h"

#include <filesystem>
#include <saot/Fusion.h>
namespace fs = std::filesystem;

using namespace saot;
using namespace utils;

template<unsigned simd_s>
static void internal() {
  test::TestSuite suite("Fusion CPU (s = " + std::to_string(simd_s) + ")");

  KernelManager kernelMgrBeforeFusion;
  KernelManager kernelMgrAfterFusion;

  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = simd_s;

  CPUFusionConfig fusionConfig = CPUFusionConfig::Default;
  NaiveCostModel costModel(4, -1, 0);

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
    qc.toCircuitGraph(graph);
    auto allBlocks = graph.getAllBlocks();
    for (const auto& block : allBlocks) {
      kernelMgrBeforeFusion.genCPUKernel(
        kernelGenConfig, *block->quantumGate,
        "beforeFusion" + std::to_string(block->id));
    }

    graph.print(std::cerr, 2) << "\n";
    applyCPUGateFusion(fusionConfig, &costModel, graph);
    graph.print(std::cerr, 2);
    allBlocks = graph.getAllBlocks();
    for (const auto& block : allBlocks) {
      kernelMgrAfterFusion.genCPUKernel(
        kernelGenConfig, *block->quantumGate,
        "afterFusion" + std::to_string(block->id));
    }

    kernelMgrBeforeFusion.initJIT();
    kernelMgrAfterFusion.initJIT();

    utils::StatevectorAlt<double> sv0(graph.nQubits, simd_s);
    utils::StatevectorAlt<double> sv1(graph.nQubits, simd_s);
    sv0.randomize();
    sv1 = sv0;

    for (const auto& k : kernelMgrBeforeFusion.kernels())
      kernelMgrBeforeFusion.applyCPUKernel(sv0.data, sv0.nQubits, k.llvmFuncName);

    for (const auto& k : kernelMgrAfterFusion.kernels())
      kernelMgrAfterFusion.applyCPUKernel(sv1.data, sv1.nQubits, k.llvmFuncName);

    suite.assertClose(utils::fidelity(sv0, sv1), 1.0,
      p.path().filename(), GET_INFO(), 1e-8);
  }

  suite.displayResult();
}

void test::test_fusionCPU() {
  internal<1>();
  internal<2>();
}
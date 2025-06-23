#include "cast/Core/AST/Parser.h"
#include "cast/Transform/Transform.h"
#include "cast/Fusion.h"
#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"
#include "tests/TestKit.h"

#include <filesystem>

namespace fs = std::filesystem;

using namespace cast;

template<unsigned simd_s>
static void f() {
  cast::test::TestSuite suite("Fusion CPU (s = " + std::to_string(simd_s) + ")");

  cast::CPUKernelManager kernelMgrBeforeFusion;
  cast::CPUKernelManager kernelMgrAfterFusion;

  cast::CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = simd_s;

  auto fusionConfig = cast::FusionConfig::Default;
  cast::NaiveCostModel costModel(2, -1, 0);

  std::cerr << "Test Dir: " << TEST_DIR << "\n";
  fs::path circuitDir = fs::path(TEST_DIR) / "circuits";
  if (!fs::exists(circuitDir) || !fs::is_directory(circuitDir)) {
    std::cerr << BOLDRED("Error: ") << "No circuit directory found\n";
    return;
  }
  for (const auto& p : fs::directory_iterator(circuitDir)) {
    if (!p.is_regular_file())
      continue;
    
    ast::ASTContext astCtx;
    astCtx.loadFromFile(p.path().string().c_str());
    ast::Parser parser(astCtx);
    auto* astRoot = parser.parse();
    auto* astCircuit = astRoot->lookupCircuit();
    auto circuitNode = transform::cvtAstCircuitToIrCircuit(*astCircuit, astCtx);
    auto circuitGraphs = circuitNode->getAllCircuitGraphs();
    assert(circuitGraphs.size() == 1 && "Expected exactly one circuit graph");
    auto& graph = *circuitGraphs[0];

    auto allGates = graph.getAllGatesShared();
    std::cerr << "Before fusion: " << allGates.size() << " gates\n";
    std::cerr << "nqubits = " << graph.nQubits() << "\n";
    graph.visualize(std::cerr);
    for (const auto& gate : allGates) {
      kernelMgrBeforeFusion.genStandaloneGate(
        kernelGenConfig, gate,
        "beforeFusion" + std::to_string(graph.gateId(gate))
      ).consumeError(); // ignore possible error
    }

    cast::applyGateFusion(fusionConfig, &costModel, graph);
    allGates = graph.getAllGatesShared();
    std::cerr << "After fusion: " << allGates.size() << " gates\n";
    graph.visualize(std::cerr);
    for (const auto& gate : allGates) {
      kernelMgrAfterFusion.genStandaloneGate(
        kernelGenConfig, gate,
        "afterFusion" + std::to_string(graph.gateId(gate))
      ).consumeError(); // ignore possible error
    }

    kernelMgrBeforeFusion.initJIT().consumeError(); // ignore possible error
    kernelMgrAfterFusion.initJIT().consumeError(); // ignore possible error

    cast::CPUStatevector<double> sv0(graph.nQubits(), simd_s);
    cast::CPUStatevector<double> sv1(graph.nQubits(), simd_s);
    sv0.randomize();
    sv1 = sv0;
    
    for (const auto& k : kernelMgrBeforeFusion.getAllStandaloneKernels()) {
      kernelMgrBeforeFusion.applyCPUKernel(
        sv0.data(), sv0.nQubits(), *k).consumeError();
    }

    for (const auto& k : kernelMgrAfterFusion.getAllStandaloneKernels()) {
      kernelMgrAfterFusion.applyCPUKernel(
        sv1.data(), sv1.nQubits(), *k).consumeError();
    }

    suite.assertClose(sv0.norm(), 1.0,
      p.path().filename().string() + " no-fuse norm", GET_INFO());
    suite.assertClose(sv1.norm(), 1.0,
      p.path().filename().string() + " fuse norm", GET_INFO());

    suite.assertClose(cast::fidelity(sv0, sv1), 1.0,
      p.path().filename().string() + " fidelity", GET_INFO());
  }

  suite.displayResult();
}

void cast::test::test_fusionCPU() {
  f<1>();
  // f<2>();
}
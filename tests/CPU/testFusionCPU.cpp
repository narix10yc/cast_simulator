#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUOptimizer.h"
#include "cast/CPU/CPUStatevector.h"
#include "cast/Core/AST/Parser.h"
#include "cast/Transform/Transform.h"
#include "tests/TestKit.h"

#include <filesystem>

namespace fs = std::filesystem;

using namespace cast;

template <CPUSimdWidth SimdWidth> static void f() {
  cast::test::TestSuite suite("Fusion CPU (s = " + std::to_string(SimdWidth) +
                              ")");

  cast::CPUKernelManager km;
  cast::CPUKernelGenConfig genCfg(SimdWidth, cast::Precision::F64);

  cast::CPUOptimizer opt;
  opt.disableCFO().setSizeOnlyFusionConfig(3);

  std::cerr << "Test Dir: " << TEST_DIR << "\n";
  fs::path circuitDir = fs::path(TEST_DIR) / "circuits";
  if (!fs::exists(circuitDir) || !fs::is_directory(circuitDir)) {
    std::cerr << BOLDRED("Error: ") << "No circuit directory found\n";
    return;
  }
  for (const auto& p : fs::directory_iterator(circuitDir)) {
    if (!p.is_regular_file())
      continue;

    auto circuit =
        llvm::cantFail(cast::parseCircuitFromQASMFile(p.path().string()));

    auto graphs = circuit->getAllCircuitGraphs();
    assert(graphs.size() == 1 &&
           "Expected exactly one circuit graph in the test circuit");

    // generate gates before fusion
    llvm::cantFail(km.genGraphGates(genCfg, *graphs[0], "graphBeforeFusion"));

    opt.run(*circuit, nullptr);
    graphs = circuit->getAllCircuitGraphs();

    // generate gates after fusion
    llvm::cantFail(km.genGraphGates(genCfg, *graphs[0], "graphAfterFusion"));
    llvm::cantFail(km.initJIT());

    cast::CPUStatevector<double> sv0(graphs[0]->nQubits(), SimdWidth);
    cast::CPUStatevector<double> sv1(graphs[0]->nQubits(), SimdWidth);
    // sv0.randomize();
    sv0.initialize();
    sv1 = sv0;

    llvm::cantFail(km.applyCPUKernelsFromGraph(
        sv0.data(), sv0.nQubits(), "graphBeforeFusion"));
    llvm::cantFail(km.applyCPUKernelsFromGraph(
        sv1.data(), sv1.nQubits(), "graphAfterFusion"));

    suite.assertCloseF64(sv0.norm(),
                         1.0,
                         p.path().filename().string() + " no-fuse norm",
                         GET_INFO());
    suite.assertCloseF64(sv1.norm(),
                         1.0,
                         p.path().filename().string() + " fuse norm",
                         GET_INFO());

    suite.assertCloseF64(cast::fidelity(sv0, sv1),
                         1.0,
                         p.path().filename().string() + " fidelity",
                         GET_INFO());
  }

  suite.displayResult();
}

void cast::test::test_fusionCPU() {
  f<W128>();
  f<W256>();
}
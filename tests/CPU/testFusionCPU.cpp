#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUOptimizer.h"
#include "cast/CPU/CPUStatevector.h"
#include "cast/Core/AST/Parser.h"
#include "cast/Transform/Transform.h"
#include "tests/TestKit.h"

#include <filesystem>

namespace fs = std::filesystem;

using namespace cast;

template <CPUSimdWidth SimdWidth> static bool f() {
  cast::test::TestSuite suite(
      "CPU fusion equivalence (SIMD=" + std::to_string(SimdWidth) + ")");

  cast::CPUKernelManager km;
  cast::CPUKernelGenConfig genCfg(SimdWidth, cast::Precision::FP64);

  cast::CPUOptimizer opt;
  opt.enableCFO(false).setSizeOnlyFusionConfig(3);

  std::cerr << "Test Dir: " << TEST_DIR << "\n";
  fs::path circuitDir = fs::path(TEST_DIR) / "circuits";
  if (!fs::exists(circuitDir) || !fs::is_directory(circuitDir)) {
    suite.assertFalse("Test circuit directory exists", GET_INFO());
    return suite.displayResult();
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
    CHECK(suite, km.genGraphGates(genCfg, *graphs[0], "graphBeforeFusion"));

    opt.run(*circuit, nullptr);
    graphs = circuit->getAllCircuitGraphs();

    // generate gates after fusion
    llvm::cantFail(km.genGraphGates(genCfg, *graphs[0], "graphAfterFusion"));
    llvm::cantFail(km.compileAllPools());

    cast::CPUStatevector<double> sv0(graphs[0]->nQubits(), SimdWidth);
    cast::CPUStatevector<double> sv1(graphs[0]->nQubits(), SimdWidth);
    // sv0.randomize();
    sv0.initialize();
    sv1 = sv0;

    CHECK(suite,
          km.applyCPUKernelsFromGraph(
              sv0.data(), sv0.nQubits(), "graphBeforeFusion"));
    CHECK(suite,
          km.applyCPUKernelsFromGraph(
              sv1.data(), sv1.nQubits(), "graphAfterFusion"));

    suite.assertCloseFP64(sv0.norm(),
                          1.0,
                          p.path().filename().string() +
                              ": no-fusion execution preserves norm",
                          GET_INFO());
    suite.assertCloseFP64(sv1.norm(),
                          1.0,
                          p.path().filename().string() +
                              ": fused execution preserves norm",
                          GET_INFO());

    suite.assertCloseFP64(cast::fidelity(sv0, sv1),
                          1.0,
                          p.path().filename().string() +
                              ": fused and unfused executions are equivalent",
                          GET_INFO());
  }

  return suite.displayResult();
}

bool cast::test::test_fusionCPU() {
  const bool ok128 = f<W128>();
  const bool ok256 = f<W256>();
  return ok128 && ok256;
}

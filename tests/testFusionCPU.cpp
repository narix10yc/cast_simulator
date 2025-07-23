#include "cast/Core/AST/Parser.h"
#include "cast/Transform/Transform.h"
#include "cast/CPU/CPUOptimizerBuilder.h"
#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"
#include "tests/TestKit.h"

#include <filesystem>

namespace fs = std::filesystem;

using namespace cast;

template<CPUSimdWidth SimdWidth>
static void f() {
  cast::test::TestSuite suite(
    "Fusion CPU (s = " + std::to_string(SimdWidth) + ")"
  );

  cast::CPUKernelManager kernelMgr;
  cast::CPUKernelGenConfig kernelGenConfig(SimdWidth, cast::Precision::F64);

  cast::CPUOptimizerBuilder optBuilder;
  optBuilder.disableCFO()
            .setSizeOnlyFusion(3)
            .setNThreads(1)
            .setPrecision(kernelGenConfig.precision);
  auto optOrError = optBuilder.build();
  if (!optOrError) {
    suite.assertFalse(optOrError.takeError(), GET_INFO());
  }
  auto opt = optOrError.takeValue();

  std::cerr << "Test Dir: " << TEST_DIR << "\n";
  fs::path circuitDir = fs::path(TEST_DIR) / "circuits";
  if (!fs::exists(circuitDir) || !fs::is_directory(circuitDir)) {
    std::cerr << BOLDRED("Error: ") << "No circuit directory found\n";
    return;
  }
  for (const auto& p : fs::directory_iterator(circuitDir)) {
    if (!p.is_regular_file())
      continue;
  
    auto circuitOrErr = cast::parseCircuitFromQASMFile(p.path().string());
    if (!circuitOrErr) {
      suite.assertFalse(circuitOrErr.takeError(), GET_INFO());
      continue;
    }
    auto circuit = circuitOrErr.takeValue();

    auto circuitGraphs = circuit.getAllCircuitGraphs();
    assert(circuitGraphs.size() == 1 &&
      "Expected exactly one circuit graph in the test circuit");

    // generate gates before fusion
    kernelMgr.genGraphGates(
      kernelGenConfig, *circuitGraphs[0], "graphBeforeFusion"
    ).consumeError();

    opt.run(circuit, /* verbose */ 1);
    circuitGraphs = circuit.getAllCircuitGraphs();

    // generate gates after fusion
    kernelMgr.genGraphGates(
      kernelGenConfig, *circuitGraphs[0], "graphAfterFusion"
    ).consumeError();

    kernelMgr.initJIT().consumeError(); // ignore possible error

    cast::CPUStatevector<double> sv0(circuitGraphs[0]->nQubits(), SimdWidth);
    cast::CPUStatevector<double> sv1(circuitGraphs[0]->nQubits(), SimdWidth);
    // sv0.randomize();
    sv0.initialize();
    sv1 = sv0;
    
    kernelMgr.applyCPUKernelsFromGraph(
      sv0.data(), sv0.nQubits(), "graphBeforeFusion"
    ).consumeError();
    kernelMgr.applyCPUKernelsFromGraph(
      sv1.data(), sv1.nQubits(), "graphAfterFusion"
    ).consumeError();

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
  f<W128>();
  // f<W256>();
}
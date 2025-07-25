#include "tplt.h"
#include "timeit/timeit.h"
#include "utils/CSVParsable.h"

#include "cast/CPU/CPUKernelManager.h"

using namespace cast;

struct Item : utils::CSVParsable<Item> {
  std::string method;
  int targetQubit;
  double memSpd;

  Item(const std::string& method, int targetQubit, double memSpd)
      : method(method), targetQubit(targetQubit), memSpd(memSpd) {};

  CSV_DATA_FIELD(method, targetQubit, memSpd);
};

constexpr int NQUBITS = 28;

// Calculate memory speed in GB/s
template<typename ScalarType>
static double calculateMemSpd(int nQubits, double time) {
  return (static_cast<double>(1ULL << nQubits) * sizeof(ScalarType) * 2) / (time * 1e9);
}

int main() {
  std::vector<Item> items;
  items.reserve(NQUBITS);

  timeit::Timer timer(3, /* verbose */ 0);
  timeit::TimingResult tr;

  CPUKernelGenConfig config;
  config.precision = Precision::F64;
  config.matrixLoadMode = MatrixLoadMode::StackLoadMatElems;
  config.zeroTol = 0.0;
  config.oneTol = 0.0;

  CPUKernelManager kernelMgr;
  for (int k = 0; k < NQUBITS; ++k) {
    if (!kernelMgr.genStandaloneGate(
      config,
      StandardQuantumGate::RandomUnitary(k),
      "test_gate_" + std::to_string(k)
    )) {
      std::cerr << "Failed to generate kernel for qubit " << k << "\n";
      return 1;
    }
  }
  if (!kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false)) {
    std::cerr << "Failed to initialize JIT\n";
    return 1;
  }

  auto sv = std::make_unique<double[]>((2ULL << NQUBITS) + 8);
  std::uintptr_t raw_addr = reinterpret_cast<std::uintptr_t>(sv.get());
  std::size_t misalignment = raw_addr % 64;
  std::size_t offset = misalignment == 0 ? 0 : (64 - misalignment) / sizeof(double);

  double* data = sv.get() + offset;
  auto mat = std::make_unique<double[]>(8); // 2x2 complex matrix

  double memSpd;
  for (int k = 0; k < NQUBITS; ++k) {
    tr = timer.timeit([&]() {
      tplt::applySingleQubit<double>(
        data, data + (1ULL << NQUBITS), mat.get(), NQUBITS, k
      );
    });
    memSpd = calculateMemSpd<double>(NQUBITS, tr.min);
    items.emplace_back("general", k, memSpd);
    items.back().write(std::cerr);
    std::cerr << "\n";

    tr = timer.timeit([&]() {
      tplt::applySingleQubitTemplateSwitch<double>(
        data, data + (1ULL << NQUBITS), mat.get(), NQUBITS, k
      );
    });
    memSpd = calculateMemSpd<double>(NQUBITS, tr.min);
    items.emplace_back("template", k, memSpd);
    items.back().write(std::cerr);
    std::cerr << "\n";

    tr = timer.timeit([&]() {
      kernelMgr.applyCPUKernel(
        data, NQUBITS, "test_gate_" + std::to_string(k), 1
      ).consumeError();
    });
    memSpd = calculateMemSpd<double>(NQUBITS, tr.min);
    items.emplace_back("cast", k, memSpd);
    items.back().write(std::cerr);
    std::cerr << "\n";
  }

  auto& os = std::cerr;
  os << Item::CSV_TITLE << "\n";
  for (const auto& item : items) {
    item.write(os);
    os << "\n";
  }

  return 0;
}
#include "cast/FPGA/FPGAGateCategory.h"
#include "openqasm/parser.h"
#include "cast/Transform/Transform.h"

using namespace cast;

int main(int argc, char** argv) {
  openqasm::Parser parser(argv[1]);
  auto root = parser.parse();

  ast::ASTContext astCtx;
  auto circuitStmt = transform::cvtQasmCircuitToAstCircuit(*root, astCtx);
  auto irCircuit = transform::cvtAstCircuitToIrCircuit(*circuitStmt, astCtx);
  auto* graph = irCircuit->getAllCircuitGraphs()[0];

  auto allGates = graph->getAllGates();
  std::cerr << "A total of " << allGates.size() << " gates found.\n";
  int nNonComp = 0, nRealOnly = 0, nUnitaryPerm = 0, nSingleQubit = 0;
  for (const auto& gate : allGates) {
    auto cate = fpga::getFPGAGateCategory(gate);
    if (cate.is(fpga::FPGAGateCategory::NonComp)) {
      ++nNonComp;
    }
    if (cate.is(fpga::FPGAGateCategory::RealOnly)) {
      ++nRealOnly;
    }
    if (cate.is(fpga::FPGAGateCategory::UnitaryPerm)) {
      ++nUnitaryPerm;
    }
    if (cate.is(fpga::FPGAGateCategory::SingleQubit)) {
      ++nSingleQubit;
    }
  }

  std::cerr << "Found " << nNonComp << " Non-Computational gates.\n";
  std::cerr << "Found " << nRealOnly << " Real Only gates.\n";
  std::cerr << "Found " << nUnitaryPerm << " Unitary Permutation gates.\n";
  std::cerr << "Found " << nSingleQubit << " Single Qubit gates.\n";
  
  return 0;
}
#include "saot/NewParser.h"
#include "saot/ast.h"
#include "saot/QuantumGate.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"

#include "saot/Polynomial.h"
#include "simulation/jit.h"

using namespace saot;

using namespace saot::ast;
using namespace simulation;

using namespace llvm;

int main(int argc, char** argv) {
    std::vector<std::pair<int, double>> varValues {
        {0, 1.1}, {1, 0.4}, {2, 0.1}, {3, -0.3}, {4, -0.9}, {5, 1.9}};


    assert(argc > 1);

    parse::Parser parser(argv[1]);
    auto qc = parser.parseQuantumCircuit();
    std::cerr << "Recovered:\n";

    std::ofstream file(std::string(argv[1]) + ".rec");
    qc.print(file);

    auto graph = qc.toCircuitGraph();
    graph.print(std::cerr) << "\n";

    applyCPUGateFusion(CPUFusionConfig::Default, graph);
    graph.print(std::cerr) << "\n";

    auto& fusedGate = graph.getAllBlocks()[0]->quantumGate;

    fusedGate->gateMatrix.printParametrizedMatrix(std::cerr) << "\n";
    fusedGate->simplifyGateMatrix();
    fusedGate->gateMatrix.printParametrizedMatrix(std::cerr) << "\n";


    // for (auto& P : fusedGate->gateMatrix.pData())
    //     P.simplify(varValues);
    // fusedGate->gateMatrix.printMatrix(std::cerr);

    IRGenerator G;
    Function* func = G.generatePrepareParameter(graph);
    G.dumpToStderr();

    G.applyLLVMOptimization(OptimizationLevel::O2);
    G.dumpToStderr();

    // // JIT
    // jit::JitEngine jitEngine(G);
    // jitEngine.dumpNativeAssembly(llvm::errs());


    return 0;
}
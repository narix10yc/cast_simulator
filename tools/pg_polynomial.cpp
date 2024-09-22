#include "quench/parser.h"
#include "quench/QuantumGate.h"
#include "quench/CircuitGraph.h"

#include "saot/Polynomial.h"

using namespace saot;

using namespace quench::ast;
using namespace quench::quantum_gate;
using namespace quench::circuit_graph;

int main(int argc, char** argv) {
    auto mat1 = GateMatrix::FromParameters("u1q", std::vector<GateParameter>{GateParameter(0), GateParameter(1), GateParameter(2)});
    auto mat2 = GateMatrix::FromParameters("u1q", std::vector<GateParameter>{GateParameter(3), GateParameter(4), GateParameter(5)});

    mat1.printMatrix(std::cerr) << "\n";

    QuantumGate gate1(mat1, {1});
    QuantumGate gate2(mat2, {2});

    gate1.lmatmul(gate2).gateMatrix.printMatrix(std::cerr);

    // assert(argc > 1);

    // Parser parser(argv[1]);
    // auto* root = parser.parse();
    // std::cerr << "Recovered:\n";

    // std::ofstream file(std::string(argv[1]) + ".rec");
    // root->print(file);

    // auto graph = root->toCircuitGraph();
    // graph.updateFusionConfig(FusionConfig::Default());
    // graph.greedyGateFusion();

    // graph.getAllBlocks()[0]->quantumGate->displayInfo(std::cerr);
    // graph.displayInfo(std::cerr, 3);

    Monomial m1;
    m1.insertMulTerm(VariableSumNode::Cosine({0, 2}, 1.2));
    m1.insertExpiVar(3);

    Polynomial p;
    p.insertMonomial(m1);

    p.print(std::cerr) << "\n";

    p.simplify({{0, 0.77}, {2, 1.14}, {3, 4.0}});
    p.print(std::cerr) << "\n";

    return 0;
}
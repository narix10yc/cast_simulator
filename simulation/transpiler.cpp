#include "simulation/transpiler.h"
#include "simulation/utils.h"

#include <queue>
#include <tuple>
#include <sstream>

using namespace simulation;
using namespace simulation::transpile;
using namespace qch::ast;

void CircuitGraph::removeGateNode(GateNode* node) {
    assert(node != nullptr);

    for (auto& data : node->dataVector) {
        auto q = data.qubit;
        auto* left = data.leftNode;
        auto* right = data.rightNode;
        connectTwoNodes(left, right, q);
        // update left and right entries
        if (leftEntry[q] == node)
            leftEntry[q] = right;
        if (rightEntry[q] == node)
            rightEntry[q] = left;
    }

    allNodes.erase(node);
    delete(node);
}

unsigned CircuitGraph::connectTwoNodes(GateNode* left, GateNode* right, int q) {
    if (left == nullptr && right == nullptr)
        return 0;
    
    if (left == nullptr) {
        if (q < 0) {
            for (auto& dataR : right->dataVector)
                dataR.leftNode = nullptr;
            return right->nqubits;
        }
        for (auto& dataR : right->dataVector) {
            if (dataR.qubit == q) {
                dataR.leftNode = nullptr;
                return 1;
            }
        }
        return 0;
    }

    if (right == nullptr) {
        if (q < 0) {
            for (auto& dataL : left->dataVector)
                dataL.rightNode = nullptr;
            return left->nqubits;
        }
        for (auto& dataL : left->dataVector) {
            if (dataL.qubit == q) {
                dataL.rightNode = nullptr;
                return 1;
            }
        }
        return 0;
    }

    // both left and right are non-null
    // connect along specific qubit
    if (q >= 0) {
        int idxL = -1, idxR = -1;
        for (size_t i = 0; i < left->nqubits; i++) {
            if (left->dataVector[i].qubit == q) {
                idxL = i;
                break;
            }
        }
        if (idxL < 0) { return 0; }
        for (size_t i = 0; i < right->nqubits; i++) {
            if (right->dataVector[i].qubit == q) {
                idxR = i;
                break;
            }
        }
        if (idxR < 0) { return 0; }
        left->dataVector[idxL].rightNode = right;
        right->dataVector[idxR].leftNode = left;
        return 1;
    }

    // connect along all qubits
    unsigned nConnected = 0;
    for (auto& dataL : left->dataVector) {
        for (auto& dataR : right->dataVector) {
            if (dataL.qubit == dataR.qubit) {
                dataL.rightNode = right;
                dataR.leftNode = left;
                nConnected++;
                break;
            }
        }
    }
    return nConnected;
}

GateNode* CircuitGraph::addSingleQubitGate(const U3Gate& u3) {
    unsigned k = u3.k;
    if (k >= nqubits)
        nqubits = k + 1;
    if (k >= leftEntry.size()) {
        leftEntry.resize(k + 1, nullptr);
        rightEntry.resize(k + 1, nullptr);
    }

    // create node
    auto* node = new GateNode(1, count);
    count++;
    allNodes.insert(node);

    node->dataVector[0].qubit = k;
    for (size_t i = 0; i < 4; i++)
        node->matrix.data[i] = Complex<>(u3.mat.real[i], u3.mat.imag[i]);

    // update graph
    if (leftEntry[k] == nullptr) {
        leftEntry[k] = node;
    } else {
        connectTwoNodes(rightEntry[k], node);
    }
    rightEntry[k] = node;
    return node;
}

GateNode* CircuitGraph::addTwoQubitGate(const U2qGate& u2q) {
    unsigned k = u2q.k;
    unsigned l = u2q.l;
    unsigned tmp = (k > l) ? u2q.k : u2q.l;
    if (tmp >= nqubits)
        nqubits = tmp + 1;
    if (tmp >= leftEntry.size()) {
        leftEntry.resize(tmp + 1, nullptr);
        rightEntry.resize(tmp + 1, nullptr);
    }

    // create node
    auto* node = new GateNode(2, count);
    count++;
    allNodes.insert(node);

    node->dataVector[0].qubit = k;
    node->dataVector[1].qubit = l;
    for (size_t i = 0; i < 16; i++)
        node->matrix.data[i] = Complex<>(u2q.mat.real[i], u2q.mat.imag[i]);
    
    // update graph 
    for (auto& data : node->dataVector) {
        unsigned q = data.qubit;
        if (leftEntry[q] == nullptr) {
            leftEntry[q] = node;
        } else {
            connectTwoNodes(rightEntry[q], node, q);
        }
        rightEntry[q] = node;
    }

    return node;
}

CircuitGraph CircuitGraph::FromQch(const RootNode& root) {
    CircuitGraph graph;

    auto circuit = dynamic_cast<CircuitStmt*>(root.getStmtPtr(0));
    for (size_t i = 0; i < circuit->countStmts(); i++) {
        auto gateApply = dynamic_cast<GateApplyStmt*>(circuit->getStmtPtr(i));
        if (gateApply == nullptr)
            continue;
        if (gateApply->name == "u3") {
            auto u3 = U3Gate(ComplexMatrix2<>::FromEulerAngles(
                                gateApply->parameters[0], 
                                gateApply->parameters[1],
                                gateApply->parameters[2]),
                            gateApply->qubits[0]);
            graph.addSingleQubitGate(u3);
        } else if (gateApply->name == "cx") {
            auto u2q = U2qGate(gateApply->qubits[0], gateApply->qubits[1],
                        {{1,0,0,0, 0,0,0,1, 0,0,1,0, 0,1,0,0}, {}});
            if (u2q.k < u2q.l)
                u2q.swapTargetQubits();
            graph.addTwoQubitGate(u2q);
        }
    }
    return graph;
}

unsigned CircuitGraph::absorbNeighbouringSingleQubitGates(GateNode* node) {
    assert(node != nullptr && node->nqubits < 3);

    unsigned nFused = 0;
    if (node->nqubits == 1) {
        GateNode* left = node->dataVector[0].leftNode;
        if (left != nullptr && left->nqubits == 1) {
            node->matrix = node->matrix.matmul(left->matrix);
            removeGateNode(left);
            nFused += 1;
        }
        GateNode* right = node->dataVector[0].rightNode;
        if (right != nullptr && right->nqubits == 1) {
            node->matrix = right->matrix.matmul(node->matrix);
            removeGateNode(right);
            nFused += 1;
        }
        return nFused;
    }

    GateNode* leftK = node->dataVector[0].leftNode;
    if (leftK != nullptr && leftK->nqubits == 1) {
        // R @ (L otimes I)
        node->matrix = (node->matrix).matmul(leftK->matrix.rightKronI());
        removeGateNode(leftK);
        nFused += 1;
    }
    GateNode* leftL = node->dataVector[1].leftNode;
    if (leftL != nullptr && leftL->nqubits == 1) {
        // R @ (I otimes L)
        node->matrix = node->matrix.matmul(leftL->matrix.leftKronI());
        removeGateNode(leftL);
        nFused += 1;
    }
    GateNode* rightK = node->dataVector[0].rightNode;
    if (rightK != nullptr && rightK->nqubits == 1) {
        // (R otimes I) @ L
        node->matrix = rightK->matrix.rightKronI().matmul(node->matrix);
        removeGateNode(rightK);
        nFused += 1;
    }
    GateNode* rightL = node->dataVector[1].rightNode;
    if (rightL != nullptr && rightL->nqubits == 1) {
        // (I otimes R) @ L
        node->matrix = rightL->matrix.leftKronI().matmul(node->matrix);
        removeGateNode(rightL);
        nFused += 1;
    }
    return nFused;
}

unsigned CircuitGraph::absorbNeighbouringTwoQubitGates(GateNode* node) {
    assert(node != nullptr && node->nqubits < 3);

    if (node->nqubits == 1)
        return 0;
    
    GateNode* left = node->dataVector[0].leftNode;
    if (left == nullptr || left != node->dataVector[1].leftNode)
        return 0;
    
    if (left->dataVector[0].qubit == node->dataVector[0].qubit) {
        node->matrix = node->matrix.matmul(left->matrix);
    } else {
        node->matrix = node->matrix.matmul(left->matrix.swapTargetQubits());
    }
    
    removeGateNode(left);
    return 1;
}

void CircuitGraph::transpileForCPU() {
    // step 1: absorb single-qubit gates
    assert(sanityCheck(std::cerr));

    bool update;
    do {
        update = false;
        for (GateNode* node : allNodes) {
            if (absorbNeighbouringSingleQubitGates(node) > 0) {
                update = true;
                break;
            }
        }
    } while (update);
    std::cerr << "-- Fusion step 1 finished! "
              << allNodes.size() << " nodes remaining\n";

    // step 2: fuse two-qubit gates
    do {
        update = false;
        for (GateNode* node : allNodes) {
            if (absorbNeighbouringTwoQubitGates(node) > 0) {
                update = true;
                break;
            }
        }
    } while (update);
    std::cerr << "-- Fusion step 2 finished! "
              << allNodes.size() << " nodes remaining\n";
    
    assert(sanityCheck(std::cerr));
}

RootNode CircuitGraph::toQch() const {
    RootNode root;
    auto circuit = std::make_unique<CircuitStmt>("transpiled");

    for (GateNode* node : allNodes) {
        std::string name = (node->nqubits == 1) ? "u3" : "u2q";
        auto gateApply = std::make_unique<GateApplyStmt>(name);
        for (auto p : node->matrix.data) {
            gateApply->addParameter(utils::approximate(p.real));
            gateApply->addParameter(utils::approximate(p.imag));
        }
        for (auto& data : node->dataVector) {
            gateApply->addTargetQubit(data.qubit);
        }

        circuit->addStmt(std::move(gateApply));
    }
    root.addStmt(std::move(circuit));
    return root;
}

bool CircuitGraph::sanityCheck(std::ostream& os) const {
    bool success = true;
    const std::string ERR = "\033[31mSanity check error:\033[0m\n  ";
    const std::string CYAN = "\033[36m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string RESET = "\033[0m";

    os << "== " << CYAN << "Circuit Graph Sanity Check" << RESET << " ==\n";
    // number of qubits
    if (leftEntry.size() != rightEntry.size()) {
        success = false;
        os << ERR << "leftEntry and rightEntry size does not match!\n";
    } else {
        os << "leftEntry and rightEntry size = " << leftEntry.size() << "\n";
        unsigned nqubits = 0;
        for (size_t i = 0; i < leftEntry.size(); i++) {
            auto* left = leftEntry[i];
            auto* right = rightEntry[i];
            if (left != nullptr && right != nullptr)
                nqubits += 1;
            else if (left == nullptr && right == nullptr)
                continue;
            else {
                os << ERR << "leftEntry and rightEntry "
                   << "does not match at qubit " << i << "\n";
                success = false;
            }
        }
        os << "Number of qubits = " << nqubits << "\n";
    }

    // check connection
    bool connectionSuccess = true;
    for (const auto& node : allNodes) {
        if (node->nqubits != node->dataVector.size()) {
            os << ERR << "Node " << node->id << " has unmatched nqubits\n";
            connectionSuccess = false;
        }
        else if (node->nqubits == 2) {
            auto k = node->dataVector[0].qubit;
            auto l = node->dataVector[1].qubit;
            if (k <= l) {
                os << ERR << "Node " << node->id << " acts on two qubits, but "
                   << "k = " << k << " and " << "l = " << l << "\n";
                connectionSuccess = false;
            }
        }
        for (const auto& data : node->dataVector) {
            auto q = data.qubit;
            auto* left = data.leftNode;
            auto* right = data.rightNode;
            if (left == nullptr && leftEntry[q] != node) {
                os << ERR << "Node " << node->id << " (along qubit " << q << ")"
                   << " has false left connection boundary\n";
                connectionSuccess = false;
            }
            if (right == nullptr && rightEntry[q] != node) {
                os << ERR << "Node " << node->id << " (along qubit " << q << ")"
                   << " has false right connection boundary\n";
                connectionSuccess = false;
            }
            if (left != nullptr && !left->actsOnQubit(q)) {
                os << ERR << "Node " << node->id << " (along qubit " << q << ")"
                   << " has false left connection\n";
                connectionSuccess = false;
            }
            if (right != nullptr && !right->actsOnQubit(q)) {
                os << ERR << "Node " << node->id << " (along qubit " << q << ")"
                   << " has false right connection\n";
                connectionSuccess = false;
            }
        }
    }
    // check number of nodes via connection
    std::set<GateNode*> foundNodes;
    std::queue<GateNode*> searchQueue;
    for (auto* entryNode : leftEntry) {
        if (entryNode != nullptr)
            searchQueue.push(entryNode);
    }

    while (!searchQueue.empty()) {
        auto* node = searchQueue.front();
        searchQueue.pop();
        if (foundNodes.count(node) == 0)
            foundNodes.insert(node);
        else continue;

        for (auto& data : node->dataVector) {
            if (data.rightNode != nullptr)
                searchQueue.push(data.rightNode);
        }
    }
    if (foundNodes.size() != allNodes.size()) {
        os << ERR << "number of nodes mismatch!"
           << " foundNodes.size() = " << foundNodes.size()
           << " allNodes.size() = " << allNodes.size() << "\n";
        connectionSuccess = false;
    }

    if (connectionSuccess)
        os << "Connection checked! (" << allNodes.size() << " nodes)\n";
    else 
        success = false;
    

    if (success)
        os << "== " << CYAN << "Sanity Check " << GREEN << "Success" << RESET << " ==\n";
    else
        os << "== " << CYAN << "Sanity Check " << RED << "Failed" << RESET << " ==\n";
    return success;
}

CircuitGraph::Tile CircuitGraph::getTile() const {
    std::vector<std::vector<std::pair<int, int>>> tile;
    auto appendOneLine = [&, nqubits=nqubits]() {
        tile.push_back(std::vector<std::pair<int, int>>(nqubits, {-1, 0}));
    };
    
    auto lastEmptyLine = [&](unsigned c) -> unsigned {
        if (tile[tile.size() - 1][c].first != -1) {
            // last line is occupied
            appendOneLine();
            return tile.size() - 1;
        }
        for (int i = tile.size() - 2; i >= 0; i--) 
            // find the first occupied one, bottom up
            if (tile[i][c].first != -1)
                return i + 1;
        return 0;
    };

    appendOneLine();
    for (const auto* node : allNodes) {
        if (node == nullptr)
            continue;
        unsigned l = 0;
        std::vector<unsigned> qubits = node->qubits();
        for (auto q : qubits) {
            unsigned tmp = lastEmptyLine(q);
            if (tmp > l)
                l = tmp;
        }
        for (auto q : qubits) {
            tile[l][q].first = node->id;
            tile[l][q].second = node->nqubits;
        }
    }
    return { tile };
}

void CircuitGraph::draw(std::ostream& os) const {
    std::vector<std::vector<std::pair<int, bool>>> tile;

    auto appendOneLine = [&, nqubits=nqubits]() {
        tile.push_back(std::vector<std::pair<int, bool>>(nqubits, {-1, false}));
    };
    auto lastEmptyLine = [&](unsigned c) -> unsigned {
        if (tile[tile.size() - 1][c].first != -1) {
            // last line is occupied
            appendOneLine();
            return tile.size() - 1;
        }
        for (int i = tile.size() - 2; i >= 0; i--) 
            // find the first occupied one, bottom up
            if (tile[i][c].first != -1)
                return i + 1;
        return 0;
    };

    unsigned largestID = 0;
    appendOneLine();
    for (const auto* node : allNodes) {
        if (node == nullptr)
            continue;
        unsigned l = 0;
        std::vector<unsigned> qubits = node->qubits();
        for (auto q : qubits) {
            unsigned tmp = lastEmptyLine(q);
            if (tmp > l)
                l = tmp;
        }
        if (qubits.size() > 1) {
            for (auto q : qubits) {
                tile[l][q].first = node->id;
                tile[l][q].second = true;
            }
        } else {
            tile[l][qubits[0]].first = node->id;
        }
        if (node->id > largestID)
            largestID = node->id;
    }

    unsigned w = std::log10(largestID) + 1;
    if (w % 2 == 0)
        w++;
    std::string whiteSpace(w/2+1, ' ');

    for (unsigned l = 0; l < tile.size(); l++) {
        for (unsigned q = 0; q < tile[0].size(); q++) {
            auto id = tile[l][q].first;
            auto isMultiQubit = tile[l][q].second;
            if (id == -1) {
                os << whiteSpace << "|" << whiteSpace << " ";
                continue;
            }
            os << ((isMultiQubit) ? " " : "(")
               << std::setw(w) << std::setfill(' ') << id
               << ((isMultiQubit) ? "  " : ") ");
        }
        os << "\n";
    }
     
}

void CircuitGraph::Tile::toTikZ(std::ostream& os) const {
    const double nodeSize = 0.4;
    const double nodeSeparation = 1.0; 

    int nrows = rows.size();
    int ncols = rows[0].size();

    const auto coord = [](double x, double y) -> std::string {
        std::stringstream ss;
        ss << "(" << x << "," << y << ")";
        return ss.str();
    };

    for (size_t c = 0; c < ncols; c++) {
        double yPos = -nodeSeparation * c;
        os << "\\draw[thick] " << coord(-nodeSeparation, yPos);
        for (size_t r = 0; r < nrows; r++) {
            double xPos = nodeSeparation * r;
            auto node = rows[r][c];
            int id = node.first;
            int nqubits = node.second;
            if (id < 0)
                continue;
            if (nqubits == 1) {
                os << " -- " << coord(xPos - nodeSize, yPos) << " "
                   << coord(xPos, yPos) << " circle[radius=" << nodeSize << "] "
                   << "node {$" << id << "$} "
                   << coord(xPos + nodeSize, yPos);
            } else {
                os << " -- " << coord(xPos - nodeSize, yPos) << " "
                   << coord(xPos-nodeSize, yPos-nodeSize) << " rectangle "
                   << coord(xPos+nodeSize, yPos+nodeSize) << " "
                   << coord(xPos, yPos) << " node {$" << id << "$} "
                   << coord(xPos+nodeSize, yPos);      
            }
        }
        os << " -- " << coord(nodeSeparation * nrows, yPos) << ";\n";
    }
    
}

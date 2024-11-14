#include "saot/FPGAInst.h"
#include "saot/CircuitGraph.h"

using namespace saot;
using namespace saot::fpga;

std::ostream& MInstEXT::print(std::ostream& os) const {
    os << "EXT ";
    utils::printVector(flags, os);
    return os;
}

std::ostream& GInstSQ::print(std::ostream& os) const {
    os << "SQ<id=" << block->id << "> ";
    for (const auto& data : block->dataVector)
        os << data.qubit << " ";
    return os;
}

std::ostream& GInstUP::print(std::ostream& os) const {
    os << "UP<id=" << block->id << "> ";
    for (const auto& data : block->dataVector)
        os << data.qubit << " ";
    return os;
}

Instruction::CostKind Instruction::getCostKind(const FPGACostConfig& config) const {
    if (mInst->getKind() == MOp_EXT) {
        auto extInst = dynamic_cast<const MInstEXT&>(*mInst);
        int n = std::min(static_cast<int>(extInst.flags.size()),
                         config.numLocalQubitsForTwiceExtMemOpTime);
        for (int i = 0; i < n; i++) {
            if (extInst.flags[i] < config.localQubitSignificanceForTwiceExtMemOpTime)
                return CK_TwiceExtMemTime;
        }
        return CK_ExtMemTime;
    }

    if (gInst->isNull()) {
        assert(!mInst->isNull());
        return CK_NonExtMemTime;
    }
    
    if (gInst->getKind() == GOp_UP)
        return CK_UPGate;
    assert(gInst->getKind() == GOp_SQ);

    if (gInst->blockKind.is(FPGAGateCategory::fpgaRealOnly))
        return CK_RealOnlySQGate;
    return CK_GeneralSQGate;
}

// helper methods to saot::fpga::genInstruction
namespace {

enum QubitKind : int {
    QK_Unknown = -1,

    QK_Local = 0,
    QK_Row = 1,
    QK_Col = 2,
    QK_Depth = 3,
    QK_OffChip = 4,
};

struct QubitStatus {
    QubitKind kind;
    // the index of this qubit among all qubits with the same kind
    int kindIdx;

    QubitStatus() : kind(QK_Unknown), kindIdx(0) {}
    QubitStatus(QubitKind kind, int kindIdx) : kind(kind), kindIdx(kindIdx) {}

    std::ostream& print(std::ostream& os) const {
        os << "(";
        switch (kind) {
            case QK_Local: os << "loc"; break;
            case QK_Row: os << "row"; break;
            case QK_Col: os << "col"; break;
            case QK_Depth: os << "dep"; break;
            case QK_OffChip: os << "ext"; break;
            case QK_Unknown: os << "unknown"; break;
            default: break;
        }
        os << ", " << kindIdx << ")";
        return os;
    }
};

// 0, 1, 2, 4
int getNumberOfFullSwapCycles(int kindIdx) {
    return (1 << kindIdx) >> 1;
}

class InstGenState {
private:
    enum available_block_kind_t {
        ABK_OnChipLocalSQ,      // on-chip local single-qubit
        ABK_OnChipNonLocalSQ,   // on-chip non-local single-qubit
        ABK_OffChipSQ,          // off-chip single-qubit
        ABK_UnitaryPerm,        // unitary permutation
        ABK_NonComp,            // non-computational
        ABK_NotInited,          // not initialized
    };

    struct available_blocks_t {
        GateBlock* block;
        FPGAGateCategory blockKind;

        available_blocks_t(GateBlock* block, FPGAGateCategory blockKind)
            : block(block), blockKind(blockKind) {}
        
        available_block_kind_t getABK(const std::vector<QubitStatus>& qubitStatuses) const {
            if (blockKind.is(FPGAGateCategory::fpgaNonComp))
                return ABK_NonComp;
            if (blockKind.is(FPGAGateCategory::fpgaUnitaryPerm))
                return ABK_UnitaryPerm;
            // single-qubit block
            assert(blockKind.is(FPGAGateCategory::fpgaSingleQubit));
            assert(block->dataVector.size() == 1);
            int q = block->dataVector[0].qubit;
            if (qubitStatuses[q].kind == QK_OffChip)
                return ABK_OffChipSQ;
            if (qubitStatuses[q].kind == QK_Local)
                return ABK_OnChipLocalSQ;
            assert(qubitStatuses[q].kind == QK_Row || qubitStatuses[q].kind == QK_Col);
            return ABK_OnChipNonLocalSQ;
        }
    };

    void init(const CircuitGraph& graph) {
        // initialize qubit statuses
        std::vector<int> priorities(nqubits);
        for (int i = 0; i < nqubits; ++i)
            priorities[i] = i;
        assignQubitStatuses(priorities);

        // initialize node state
        int row = 0;
        for (auto it = graph.tile().begin(); it != graph.tile().end(); it++, row++) {
            for (unsigned q = 0; q < nqubits; q++)
                tileBlocks[nqubits * row + q] = (*it)[q];
        }
        // initialize unlockedRowIndices
        for (unsigned q = 0; q < nqubits; q++) {
            for (row = 0; row < nrows; row++) {
                if (tileBlocks[nqubits * row + q] != nullptr)
                    break;
            }
            unlockedRowIndices[q] = row;
        }
        // initialize availables
        for (unsigned q = 0; q < nqubits; q++) {
            row = unlockedRowIndices[q];
            if (row >= nrows)
                continue;
            auto* cddBlock = tileBlocks[nqubits * row + q];
            assert(cddBlock);
            if (std::find_if(availables.begin(), availables.end(),
                    [&cddBlock](const available_blocks_t& avail) {
                        return avail.block == cddBlock;
                    }) != availables.end()) {
                continue;
            }
            
            bool acceptFlag = true;
            for (const auto& bData : cddBlock->dataVector) {
                if (unlockedRowIndices[bData.qubit] < row) {
                    acceptFlag = false;
                    break;
                }
            }
            if (acceptFlag)
                availables.emplace_back(cddBlock, getBlockKind(cddBlock));
        }
    }
public:
    const FPGAInstGenConfig& config;
    int nrows;
    int nqubits;
    std::vector<QubitStatus> qubitStatuses;
    std::vector<GateBlock*> tileBlocks;
    // unlockedRowIndices[q] gives the index of the last unlocked row in wire q
    std::vector<int> unlockedRowIndices;
    std::vector<available_blocks_t> availables;

    InstGenState(const CircuitGraph& graph, const FPGAInstGenConfig& config)
            : config(config),
              nrows(graph.tile().size()),
              nqubits(graph.nqubits),
              qubitStatuses(graph.nqubits),
              tileBlocks(graph.tile().size() * nqubits),
              unlockedRowIndices(nqubits),
              availables() { init(graph); }

    std::ostream& printQubitStatuses(std::ostream& os) const {
        auto it = qubitStatuses.cbegin();
        it->print(os << "0:");
        int i = 1;
        while (++it != qubitStatuses.cend())
            it->print(os << ", " << i++ << ":");
        return os << "\n";
    }

    FPGAGateCategory getBlockKind(GateBlock* block) const {
        return getFPGAGateCategory(*block->quantumGate, config.tolerances);
    }
    // popBlock: pop a block from \p availables. Update \p availables accordingly.
    void popBlock(GateBlock* block) {
        auto it = std::find_if(availables.begin(), availables.end(),
            [&block](const available_blocks_t& avail) {
                return avail.block == block;
            });
        assert(it != availables.end());
        availables.erase(it);

        // grab next availables
        std::vector<GateBlock*> candidateBlocks;
        for (const auto& data : block->dataVector) {
            const auto& qubit = data.qubit;

            GateBlock* cddBlock = nullptr;
            for (auto& updatedRow = ++unlockedRowIndices[qubit]; updatedRow < nrows; ++updatedRow) {
                auto idx = nqubits * updatedRow + qubit;
                cddBlock = tileBlocks[idx];
                if (cddBlock)
                    break;
            }
            if (cddBlock &&
                    std::find(candidateBlocks.begin(), candidateBlocks.end(), cddBlock) == candidateBlocks.end())
                candidateBlocks.push_back(cddBlock);
        }
        for (const auto& b : candidateBlocks) {
            bool insertFlag = true;
            auto row = unlockedRowIndices[b->dataVector[0].qubit];
            for (const auto& data : b->dataVector) {
                if (unlockedRowIndices[data.qubit] != row) {
                    insertFlag = false;
                    break;
                }
            }
            if (insertFlag)
                availables.emplace_back(b, getBlockKind(b));
        }
    }

    void assignQubitStatuses(const std::vector<int>& priorities) {
        assert(utils::isPermutation(priorities));
        int nOnChipQubits = config.getNOnChipQubits();

        int q;
        if (nqubits <= config.nLocalQubits) {
            for (q = 0; q < nqubits; q++)
                qubitStatuses[priorities[q]] = QubitStatus(QK_Local, q);
            return;
        }

        // local
        for (q = 0; q < config.nLocalQubits; q++)
            qubitStatuses[priorities[q]] = QubitStatus(QK_Local, q);

        // row and col
        int kindIdx = 0;
        q = config.nLocalQubits;
        int nQubitsAvailable = std::min(nqubits, nOnChipQubits);
        while (true) {
            if (q >= nQubitsAvailable)
                break;
            qubitStatuses[priorities[q]] = QubitStatus(QK_Row, kindIdx);
            ++q;
            if (q >= nQubitsAvailable)
                break;
            qubitStatuses[priorities[q]] = QubitStatus(QK_Col, kindIdx);
            ++q;
            ++kindIdx;
        }

        // off-chip
        for (q = 0; q < nqubits - nOnChipQubits; q++)
            qubitStatuses[priorities[nOnChipQubits + q]] = QubitStatus(QK_OffChip, q);
    }

    GateBlock* findBlockWithABK(available_block_kind_t abk) const {
        for (const auto& candidate : availables) {
            if (candidate.getABK(qubitStatuses) == abk)
                return candidate.block;
        }
        return nullptr;
    }

    std::vector<Instruction> generate() {
        std::vector<Instruction> instructions;
        // The minimum indices at which we can insert mem / gate instructions
        int vacantMemIdx = 0;
        int vacantGateIdx = 0;
        int sqGateBarrierIdx = 0; // single-qubit gate

        // This method will update vacantMemIdx = idx + 1
        const auto writeMemInst = [&](int idx, std::unique_ptr<MemoryInst> inst) {
            if (idx < instructions.size()) {
                assert(instructions[idx].mInst->isNull());
                instructions[idx].setMInst(std::move(inst));
            } else {
                assert(idx == instructions.size());
                instructions.emplace_back(std::move(inst), nullptr);
            }
            vacantMemIdx = idx + 1;
        };

        const auto generateFullSwap = [&](int localQ, int nonLocalQ) {
            assert(qubitStatuses[localQ].kind == QK_Local);
            assert(qubitStatuses[nonLocalQ].kind != QK_Local);
            const int fullSwapQIdx = qubitStatuses[nonLocalQ].kindIdx;
            const int nFSCycles = getNumberOfFullSwapCycles(fullSwapQIdx);
            const int shuffleSwapQIdx = qubitStatuses[localQ].kindIdx;

            int insertIdx = std::max(vacantMemIdx, sqGateBarrierIdx);
            // full swaps
            for (int cycle = 0; cycle < nFSCycles; cycle++) {
                if (qubitStatuses[nonLocalQ].kind == QK_Row)
                    writeMemInst(insertIdx++, std::make_unique<MInstFSR>(fullSwapQIdx, cycle));
                else
                    writeMemInst(insertIdx++, std::make_unique<MInstFSC>(fullSwapQIdx, cycle));
            }
            // shuffle swap
            if (qubitStatuses[nonLocalQ].kind == QK_Row)
                writeMemInst(insertIdx++, std::make_unique<MInstSSR>(shuffleSwapQIdx));
            else
                writeMemInst(insertIdx++, std::make_unique<MInstSSC>(shuffleSwapQIdx));

            // swap qubit statuses
            if (fullSwapQIdx != 0) {
                // permute nonLocalQ -> kind[0] -> localQ
                auto it = std::find_if(qubitStatuses.begin(), qubitStatuses.end(),
                    [kind=qubitStatuses[nonLocalQ].kind](const QubitStatus& S) {
                        return S.kind == kind && S.kindIdx == 0;
                    });
                assert(it != qubitStatuses.end());
                auto tmp = *it;
                *it = qubitStatuses[nonLocalQ];
                qubitStatuses[nonLocalQ] = qubitStatuses[localQ];
                qubitStatuses[localQ] = tmp;
            }
            else {
                // swap nonLocalQ and localQ
                auto tmp = qubitStatuses[localQ];
                qubitStatuses[localQ] = qubitStatuses[nonLocalQ];
                qubitStatuses[nonLocalQ] = tmp;
            }
        };

        const auto generateUPBlock = [&](GateBlock* b) {
            popBlock(b);

            if (vacantGateIdx == instructions.size()) {
                instructions.emplace_back(nullptr, std::make_unique<GInstUP>(b, getBlockKind(b)));
            } else {
                auto& inst = instructions[vacantGateIdx];
                assert(inst.gInst->isNull());
                inst.setGInst(std::make_unique<GInstUP>(b, getBlockKind(b)));
            }
            ++vacantGateIdx;
        };

        const auto generateLocalSQBlock = [&](GateBlock* b) {
            popBlock(b);
            assert(b->quantumGate->qubits.size() == 1 &&
                "SQ Block has more than 1 target qubits?");
            auto qubit = b->quantumGate->qubits[0];
            assert(qubitStatuses[qubit].kind == QK_Local);

            instructions.emplace_back(nullptr, std::make_unique<GInstSQ>(b, getBlockKind(b)));
            vacantGateIdx = instructions.size();
            sqGateBarrierIdx = vacantGateIdx;
        };

        const auto generateNonLocalSQBlock = [&](GateBlock* b) {
            assert(b->quantumGate->qubits.size() == 1 &&
                "SQ Block has more than 1 target qubits?");
            auto qubit = b->quantumGate->qubits[0];
            assert(qubitStatuses[qubit].kind != QK_Local);
            // TODO: the ideal case is after full swap, there is a local SQ
            // block. However, we need deeper search since potentially many
            // UP gates are to be applied together with full swap insts.

            // For now, we always use the first (least significant) local qubit.
            for (int localQ = 0; localQ < nqubits; localQ++) {
                if (qubitStatuses[localQ].kind == QK_Local) {
                    generateFullSwap(localQ, qubit);
                    break;
                }
            }
            generateLocalSQBlock(b);
        };

        const auto insertExtMemInst = [&](const std::vector<int>& priorities) {
            int insertPosition = std::max(vacantMemIdx, sqGateBarrierIdx);
            instructions.insert(instructions.cbegin() + insertPosition,
                Instruction(std::make_unique<MInstEXT>(priorities), nullptr));
            ++insertPosition;
            ++vacantMemIdx;
            if (vacantMemIdx < insertPosition)
                vacantMemIdx = insertPosition;
            // we don't have to increment sqGateBarrierIdx as its use case is
            // always in sync with vacantMemIdx
            if (sqGateBarrierIdx == insertPosition - 1)
                ++sqGateBarrierIdx;
        };

        // reassign qubit statuses (on-chip / off-chip) based on available blocks
        // this function will call updateAvailables()
        const auto generateOnChipReassignment = [&]() {
            std::vector<int> priorities;
            priorities.reserve(nqubits);

            auto availablesCopy(availables);
            // prioritize assigning SQ gates as local
            while (!availablesCopy.empty()) {
                auto it = std::find_if(availablesCopy.begin(), availablesCopy.end(),
                [](const available_blocks_t& avail) {
                    return avail.blockKind.is(FPGAGateCategory::fpgaSingleQubit);
                });
                if (it == availablesCopy.end())
                    break;
                assert(it->block->nqubits() == 1);
                int q = it->block->dataVector[0].qubit;
                utils::pushBackIfNotInVector(priorities, q);
                availablesCopy.erase(it);
            }
            // no SQ gates, prioritize UP gates
            for (const auto& avail : availablesCopy) {
                for (const auto& data : avail.block->dataVector)
                    utils::pushBackIfNotInVector(priorities, data.qubit);
            }
            // fill up priorities vector
            int startQubit = priorities.empty() ? 0 : priorities[0];
            for (int q = 0; q < nqubits; q++)
                utils::pushBackIfNotInVector(priorities, (q+startQubit) % nqubits);

            // update qubitStatuses
            assignQubitStatuses(priorities);
            insertExtMemInst(priorities);
        };

        while (!availables.empty()) {
            // TODO: handle non-comp gates (omit them for now)
            bool nonCompFlag = false;
            for (const auto& avail : availables) {
                if (avail.blockKind.is(FPGAGateCategory::fpgaNonComp)) {
                    // std::cerr << "Ignored block " << avail.block->id << " because it is non-comp\n";
                    popBlock(avail.block);
                    nonCompFlag = true;
                    break;
                }
            }
            if (nonCompFlag)
                continue;
                            
            if (!config.selectiveGenerationMode) {
                auto& avail = availables[0];
                auto abk = avail.getABK(qubitStatuses);
                if (abk == ABK_OffChipSQ) {
                    std::vector<int> priorities(nqubits);
                    assert(avail.block->dataVector.size() == 1);
                    int q = avail.block->dataVector[0].qubit;
                    priorities[0] = q;
                    for (int i = 1; i < nqubits; i++)
                        priorities[i] = (i <= q) ? (i - 1) : i;
                    assignQubitStatuses(priorities);
                    insertExtMemInst(priorities);
                }

                abk = avail.getABK(qubitStatuses);
                if (abk == ABK_OnChipLocalSQ)
                    generateLocalSQBlock(avail.block);
                else if (abk == ABK_UnitaryPerm)
                    generateUPBlock(avail.block);
                else if (abk == ABK_OnChipNonLocalSQ)
                    generateNonLocalSQBlock(avail.block);
                else
                    assert(false && "Unreachable");
                continue;
            }
            
            // TODO: optimize this traversal
            if (auto* b = findBlockWithABK(ABK_OnChipLocalSQ))
                generateLocalSQBlock(b);
            else if (auto* b = findBlockWithABK(ABK_UnitaryPerm))
                generateUPBlock(b);
            else if (auto* b = findBlockWithABK(ABK_OnChipNonLocalSQ))
                generateNonLocalSQBlock(b);
            else // no onChipBlock
                generateOnChipReassignment();
        }
        return instructions;
    }
};

} // anonymous namespace

std::vector<Instruction> saot::fpga::genInstruction(
        const CircuitGraph& graph, const FPGAInstGenConfig& config) {
    InstGenState state(graph, config);

    return state.generate();
}


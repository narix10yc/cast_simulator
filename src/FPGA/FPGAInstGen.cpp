#include "cast/FPGA/FPGAInstGen.h"
#include "cast/Core/IRNode.h"
#include "cast/FPGA/FPGAInst.h"
#include "utils/PrintSpan.h"
#include "utils/utils.h"

using namespace cast;
using namespace cast::fpga;

std::ostream& MInstEXT::print(std::ostream& os) const {
  os << "EXT ";
  utils::printSpan(os, std::span(flags));
  return os;
}

std::ostream& GInstSQ::print(std::ostream& os) const {
  os << "SQ<gate@" << (void*)(gate.get()) << "> ";
  utils::printSpan(os, std::span(gate->qubits()));
  return os;
}

std::ostream& GInstUP::print(std::ostream& os) const {
  os << "UP<gate@" << (void*)(gate.get()) << "> ";
  utils::printSpan(os, std::span(gate->qubits()));
  return os;
}

Instruction::CostKind
Instruction::getCostKind(const FPGACostConfig& config) const {
  if (mInst_->getKind() == MOp_EXT) {
    auto extInst = static_cast<const MInstEXT&>(*mInst_);
    if (extInst.flags[0] < config.lowestQIdxForTwiceExtTime)
      return CK_TwiceExtMemTime;
    return CK_ExtMemTime;
  }

  if (gInst_->isNull()) {
    assert(!mInst_->isNull());
    return CK_NonExtMemTime;
  }

  if (gInst_->getKind() == GOp_UP)
    return CK_UPGate;
  assert(gInst_->getKind() == GOp_SQ);

  if (gInst_->blockKind.is(FPGAGateCategory::RealOnly))
    return CK_RealOnlySQGate;
  return CK_GeneralSQGate;
}

// helper methods to cast::fpga::genInstruction
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
    case QK_Local:
      os << "loc";
      break;
    case QK_Row:
      os << "row";
      break;
    case QK_Col:
      os << "col";
      break;
    case QK_Depth:
      os << "dep";
      break;
    case QK_OffChip:
      os << "ext";
      break;
    case QK_Unknown:
      os << "unknown";
      break;
    default:
      break;
    }
    os << ", " << kindIdx << ")";
    return os;
  }
};

// 0, 1, 2, 4
int getNumberOfFullSwapCycles(int kindIdx) { return (1 << kindIdx) >> 1; }

class InstGenState {
private:
  enum available_block_kind_t {
    ABK_OnChipLocalSQ,    // on-chip local single-qubit
    ABK_OnChipNonLocalSQ, // on-chip non-local single-qubit
    ABK_OffChipSQ,        // off-chip single-qubit
    ABK_UnitaryPerm,      // unitary permutation
    ABK_NonComp,          // non-computational
    ABK_NotInited,        // not initialized
  };

  struct available_blocks_t {
    QuantumGatePtr gate;
    FPGAGateCategory gateKind;

    available_blocks_t(QuantumGatePtr gate, FPGAGateCategory gateKind)
        : gate(gate), gateKind(gateKind) {}

    available_block_kind_t
    getABK(const std::vector<QubitStatus>& qubitStatuses) const {
      if (gateKind.is(FPGAGateCategory::NonComp))
        return ABK_NonComp;
      if (gateKind.is(FPGAGateCategory::UnitaryPerm))
        return ABK_UnitaryPerm;
      // single-qubit block
      assert(gateKind.is(FPGAGateCategory::SingleQubit));
      assert(gate->nQubits() == 1);
      int q = gate->qubits()[0];
      if (qubitStatuses[q].kind == QK_OffChip)
        return ABK_OffChipSQ;
      if (qubitStatuses[q].kind == QK_Local)
        return ABK_OnChipLocalSQ;
      assert(qubitStatuses[q].kind == QK_Row ||
             qubitStatuses[q].kind == QK_Col);
      return ABK_OnChipNonLocalSQ;
    }
  };

  void init(const ir::CircuitGraphNode& graph) {
    // initialize qubit statuses
    std::vector<int> priorities(nQubits);
    for (int i = 0; i < nQubits; ++i)
      priorities[i] = i;
    assignQubitStatuses(priorities);

    // initialize node state
    int row = 0;
    for (auto it = graph.tile().begin(), end = graph.tile().end(); it != end;
         it++, row++) {
      for (int q = 0; q < nQubits; q++)
        tileBlocks[nQubits * row + q] = graph.lookup((*it)[q]);
    }
    // initialize unlockedRowIndices
    for (int q = 0; q < nQubits; q++) {
      for (row = 0; row < nRows; row++) {
        if (tileBlocks[nQubits * row + q] != nullptr)
          break;
      }
      unlockedRowIndices[q] = row;
    }
    // initialize availables
    for (int q = 0; q < nQubits; q++) {
      row = unlockedRowIndices[q];
      if (row >= nRows)
        continue;
      auto cddGate = tileBlocks[nQubits * row + q];
      assert(cddGate != nullptr);
      if (std::ranges::find_if(availables,
                               [&cddGate](const available_blocks_t& avail) {
                                 return avail.gate.get() == cddGate.get();
                               }) != availables.end()) {
        continue;
      }

      bool acceptFlag = true;
      for (const auto& qubit : cddGate->qubits()) {
        if (unlockedRowIndices[qubit] < row) {
          acceptFlag = false;
          break;
        }
      }
      if (acceptFlag)
        availables.emplace_back(cddGate, getBlockKind(cddGate));
    }
  }

public:
  const ir::CircuitGraphNode& graph;
  const FPGAInstGenConfig& config;
  int nRows;
  int nQubits;
  std::vector<QubitStatus> qubitStatuses;
  std::vector<QuantumGatePtr> tileBlocks;
  // unlockedRowIndices[q] gives the index of the last unlocked row in wire q
  std::vector<int> unlockedRowIndices;
  std::vector<available_blocks_t> availables;

  InstGenState(const ir::CircuitGraphNode& graph,
               const FPGAInstGenConfig& config)
      : graph(graph), config(config), nRows(graph.tile().size()),
        nQubits(graph.nQubits()), qubitStatuses(graph.nQubits()),
        tileBlocks(graph.tile().size() * nQubits), unlockedRowIndices(nQubits),
        availables() {
    init(graph);
  }

  std::ostream& printQubitStatuses(std::ostream& os) const {
    auto it = qubitStatuses.cbegin();
    it->print(os << "0:");
    int i = 1;
    while (++it != qubitStatuses.cend())
      it->print(os << ", " << i++ << ":");
    return os << "\n";
  }

  FPGAGateCategory getBlockKind(QuantumGatePtr gate) const {
    return getFPGAGateCategory(gate.get(), config.tolerance);
  }

  // popGate: pop a gate from \p availables. Update \p availables
  // accordingly.
  void popGate(QuantumGatePtr gate) {
    auto it = std::find_if(availables.begin(),
                           availables.end(),
                           [&gate](const available_blocks_t& avail) {
                             return avail.gate == gate;
                           });
    assert(it != availables.end());
    availables.erase(it);

    // grab next availables
    std::vector<QuantumGatePtr> cddGates;
    for (const auto& qubit : gate->qubits()) {
      QuantumGatePtr cddBlock = nullptr;
      for (auto& updatedRow = ++unlockedRowIndices[qubit]; updatedRow < nRows;
           ++updatedRow) {
        auto idx = nQubits * updatedRow + qubit;
        cddBlock = tileBlocks[idx];
        if (cddBlock)
          break;
      }
      if (cddBlock && std::find(cddGates.begin(), cddGates.end(), cddBlock) ==
                          cddGates.end())
        cddGates.push_back(cddBlock);
    }
    for (const auto& g : cddGates) {
      bool insertFlag = true;
      auto row = unlockedRowIndices[g->qubits()[0]];
      for (const auto& qubit : g->qubits()) {
        if (unlockedRowIndices[qubit] != row) {
          insertFlag = false;
          break;
        }
      }
      if (insertFlag)
        availables.emplace_back(g, getBlockKind(g));
    }
  }

  void assignQubitStatuses(const std::vector<int>& priorities) {
    // assert(utils::isPermutation(priorities));
    int nOnChipQubits = config.getNOnChipQubits();

    int q;
    if (nQubits <= config.nLocalQubits) {
      for (q = 0; q < nQubits; q++)
        qubitStatuses[priorities[q]] = QubitStatus(QK_Local, q);
      return;
    }

    // local
    for (q = 0; q < config.nLocalQubits; q++)
      qubitStatuses[priorities[q]] = QubitStatus(QK_Local, q);

    // row and col
    int kindIdx = 0;
    q = config.nLocalQubits;
    int nQubitsAvailable = std::min(nQubits, nOnChipQubits);
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
    for (q = 0; q < nQubits - nOnChipQubits; q++)
      qubitStatuses[priorities[nOnChipQubits + q]] = QubitStatus(QK_OffChip, q);
  }

  QuantumGatePtr findBlockWithABK(available_block_kind_t abk) const {
    for (const auto& candidate : availables) {
      if (candidate.getABK(qubitStatuses) == abk)
        return candidate.gate;
    }
    return nullptr;
  }

  std::vector<Instruction> generate() {
    std::vector<Instruction> instructions;
    // The minimum indices at which we can insert mem / gate instructions
    unsigned vacantMemIdx = 0;
    unsigned vacantGateIdx = 0;
    unsigned sqGateBarrierIdx = 0; // single-qubit gate

    // This method will update vacantMemIdx = idx + 1
    const auto writeMemInst = [&](unsigned idx,
                                  std::unique_ptr<MemoryInst> inst) {
      if (idx < instructions.size()) {
        assert(instructions[idx].mInst_->isNull());
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
          writeMemInst(insertIdx++,
                       std::make_unique<MInstFSR>(fullSwapQIdx, cycle));
        else
          writeMemInst(insertIdx++,
                       std::make_unique<MInstFSC>(fullSwapQIdx, cycle));
      }
      // shuffle swap
      if (qubitStatuses[nonLocalQ].kind == QK_Row)
        writeMemInst(insertIdx++, std::make_unique<MInstSSR>(shuffleSwapQIdx));
      else
        writeMemInst(insertIdx++, std::make_unique<MInstSSC>(shuffleSwapQIdx));

      // swap qubit statuses
      if (fullSwapQIdx != 0) {
        // permute nonLocalQ -> kind[0] -> localQ
        auto it = std::find_if(
            qubitStatuses.begin(),
            qubitStatuses.end(),
            [kind = qubitStatuses[nonLocalQ].kind](const QubitStatus& S) {
              return S.kind == kind && S.kindIdx == 0;
            });
        assert(it != qubitStatuses.end());
        auto tmp = *it;
        *it = qubitStatuses[nonLocalQ];
        qubitStatuses[nonLocalQ] = qubitStatuses[localQ];
        qubitStatuses[localQ] = tmp;
      } else {
        // swap nonLocalQ and localQ
        auto tmp = qubitStatuses[localQ];
        qubitStatuses[localQ] = qubitStatuses[nonLocalQ];
        qubitStatuses[nonLocalQ] = tmp;
      }
    };

    bool upFusionFlag = false;
    const auto generateUPBlock = [&](QuantumGatePtr g) {
      popGate(g);
      QuantumGatePtr lastUPGate = nullptr;
      assert(vacantGateIdx >= 0);
      if (config.maxUpSize > 0 && !instructions.empty() &&
          instructions[vacantGateIdx - 1].gInst_->getKind() == GOp_UP) {
        lastUPGate = instructions[vacantGateIdx - 1].gInst_->gate;
        // check fusion condition
        auto candidateQubits = lastUPGate->qubits();
        for (const auto& q : g->qubits())
          utils::push_back_if_not_present(candidateQubits, q);
        // accept fusion
        if (candidateQubits.size() <= static_cast<unsigned>(config.maxUpSize)) {
          auto gate = cast::matmul(g.get(), lastUPGate.get());
          instructions[vacantGateIdx - 1].setGInst(
              std::make_unique<GInstUP>(gate, FPGAGateCategory::NonComp));
          // std::cerr << "InstGen Time Fusion Accepted\n";
          upFusionFlag = true;
          return;
        }
      }
      upFusionFlag = false;
      if (vacantGateIdx == instructions.size()) {
        instructions.emplace_back(
            nullptr, std::make_unique<GInstUP>(g, getBlockKind(g)));
      } else {
        auto& inst = instructions[vacantGateIdx];
        assert(inst.gInst_->isNull());
        inst.setGInst(std::make_unique<GInstUP>(g, getBlockKind(g)));
      }
      ++vacantGateIdx;
    };

    const auto generateLocalSQBlock = [&](QuantumGatePtr gate) {
      popGate(gate);
      assert(gate->nQubits() == 1 && "SQ Gate has more than 1 target qubits?");
      auto qubit = gate->qubits()[0];
      assert(qubitStatuses[qubit].kind == QK_Local);

      instructions.emplace_back(
          nullptr, std::make_unique<GInstSQ>(gate, getBlockKind(gate)));
      vacantGateIdx = instructions.size();
      sqGateBarrierIdx = vacantGateIdx;
    };

    const auto generateNonLocalSQBlock = [&](QuantumGatePtr b) {
      assert(b->nQubits() == 1 && "SQ Block has more than 1 target qubits?");
      auto q = b->qubits()[0];
      assert(qubitStatuses[q].kind != QK_Local);
      // TODO: the ideal case is after full swap, there is a local SQ
      // block. However, we need deeper search since potentially many
      // UP gates are to be applied together with full swap insts.

      // For now, we always use the first (least significant) local qubit.
      for (int localQ = 0; localQ < nQubits; localQ++) {
        if (qubitStatuses[localQ].kind == QK_Local) {
          generateFullSwap(localQ, q);
          break;
        }
      }
      generateLocalSQBlock(b);
    };

    const auto insertExtMemInst = [&](const std::vector<int>& priorities) {
      int insertPosition = std::max(vacantMemIdx, sqGateBarrierIdx);
      instructions.insert(
          instructions.cbegin() + insertPosition,
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
      priorities.reserve(nQubits);
      priorities.push_back(nQubits >> 1);

      auto availablesCopy(availables);
      // prioritize assigning SQ gates as local
      while (!availablesCopy.empty()) {
        auto it = std::ranges::find_if(
            availablesCopy, [](const available_blocks_t& avail) {
              return avail.gateKind.is(FPGAGateCategory::SingleQubit);
            });
        if (it == availablesCopy.end())
          break;
        assert(it->gate->nQubits() == 1);
        int q = it->gate->qubits()[0];
        utils::push_back_if_not_present(priorities, q);
        availablesCopy.erase(it);
      }
      // no SQ gates, prioritize UP gates
      for (const auto& avail : availablesCopy) {
        for (const auto& q : avail.gate->qubits())
          utils::push_back_if_not_present(priorities, q);
      }
      // to diminish external memory access overhead
      // int numToSort = std::min(static_cast<int>(priorities.size()),
      // config.nLocalQubits); if (numToSort == 0) {
      //     priorities.push_back(nQubits - 1);
      // }
      // else if (numToSort == 1) {
      //     int tmp = priorities[0];
      //     if (tmp != nQubits - 1) {
      //         priorities[0] = nQubits - 1;
      //         priorities.push_back(tmp);
      //     }
      // }
      // else {
      //     std::sort(priorities.begin(), priorities.begin() + numToSort,
      //     std::greater<>());
      // }

      // fill up priorities vector
      int startQubit = priorities.empty() ? (nQubits >> 1) : priorities[0];
      for (int q = 0; q < nQubits; q++)
        utils::push_back_if_not_present(priorities, (q + startQubit) % nQubits);

      // update qubitStatuses
      assignQubitStatuses(priorities);
      insertExtMemInst(priorities);
    };

    while (!availables.empty()) {
      // TODO: handle non-comp gates (omit them for now)
      bool nonCompFlag = false;
      for (const auto& avail : availables) {
        if (avail.gateKind.is(FPGAGateCategory::NonComp)) {
          // std::cerr << "Ignored block " << avail.gate->id << " because it
          // is non-comp\n";
          popGate(avail.gate);
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
          std::vector<int> priorities(nQubits);
          assert(avail.gate->nQubits() == 1);
          int q = avail.gate->qubits()[0];
          priorities[0] = q;
          for (int i = 1; i < nQubits; i++)
            priorities[i] = (i <= q) ? (i - 1) : i;
          assignQubitStatuses(priorities);
          insertExtMemInst(priorities);
        }

        abk = avail.getABK(qubitStatuses);
        if (abk == ABK_OnChipLocalSQ)
          generateLocalSQBlock(avail.gate);
        else if (abk == ABK_UnitaryPerm)
          generateUPBlock(avail.gate);
        else if (abk == ABK_OnChipNonLocalSQ)
          generateNonLocalSQBlock(avail.gate);
        else
          assert(false && "Unreachable");
        continue;
      }

      if (upFusionFlag) {
        if (auto b = findBlockWithABK(ABK_UnitaryPerm)) {
          generateUPBlock(b);
          continue;
        }
      }
      // TODO: optimize this traversal
      if (auto b = findBlockWithABK(ABK_OnChipLocalSQ)) {
        generateLocalSQBlock(b);
        continue;
      }
      if (auto b = findBlockWithABK(ABK_UnitaryPerm)) {
        generateUPBlock(b);
      } else if (auto b = findBlockWithABK(ABK_OnChipNonLocalSQ))
        generateNonLocalSQBlock(b);
      else // no onChipBlock
        generateOnChipReassignment();
    }
    return instructions;
  }
};

} // anonymous namespace

std::vector<Instruction>
cast::fpga::genInstruction(const ir::CircuitGraphNode& graph,
                           const FPGAInstGenConfig& config) {
  InstGenState state(graph, config);

  return state.generate();
}

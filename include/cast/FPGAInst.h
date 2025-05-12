#ifndef CAST_FPGAINST_H
#define CAST_FPGAINST_H

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "cast/FPGAConfig.h"

namespace cast {
class CircuitGraph;
class LegacyQuantumGate;
class GateBlock;
} // namespace cast

namespace cast::fpga {

// @param upTol: tolerance of the absolute values of complex entries in the
// matrix smaller than (or equal to) which can be considered zero;
// @param reOnlyTol: tolerance of the absolute value of imaginary value of
// each entry smaller than (or equal to) which can be considered zero;
FPGAGateCategory
getFPGAGateCategory(const LegacyQuantumGate& gate,
                    const FPGAGateCategoryTolerance &tolerances);

enum GInstKind : int {
  GOp_NUL = 0,

  GOp_SQ = 1, // Single Qubit
  GOp_UP = 2, // Unitary Permutation
};

enum MInstKind : int {
  MOp_NUL = 0,

  MOp_SSR = 1, // Shuffle Swap Row
  MOp_SSC = 2, // Shuffle Swap Col
  MOp_FSR = 3, // Full Swap Row
  MOp_FSC = 4, // Full Swap Col
  MOp_EXT = 5, // External Memory Swap
};

class MemoryInst {
private:
  MInstKind mKind;

public:
  MemoryInst(MInstKind mKind) : mKind(mKind) {}

  virtual ~MemoryInst() = default;

  MInstKind getKind() const { return mKind; }

  bool isNull() { return getKind() == MOp_NUL; }
  virtual std::ostream& print(std::ostream& os) const {
    assert(false && "Calling from base class");
    return os;
  }
};

// Null (NUL)
class MInstNUL : public MemoryInst {
public:
  MInstNUL() : MemoryInst(MOp_NUL) {}

  std::ostream& print(std::ostream& os) const override {
    return os << "NUL" << std::string(12, ' ');
  }
};

// Shuffle Swap Row (SSR)
class MInstSSR : public MemoryInst {
public:
  int qIdx;
  MInstSSR(int qIdx) : MemoryInst(MOp_SSR), qIdx(qIdx) {}

  std::ostream& print(std::ostream& os) const override {
    return os << "SSR " << qIdx << std::string(10, ' ');
  }
};

// Shuffle Swap Col (SSC)
class MInstSSC : public MemoryInst {
public:
  int qIdx;
  MInstSSC(int qIdx) : MemoryInst(MOp_SSC), qIdx(qIdx) {}

  std::ostream& print(std::ostream& os) const override {
    return os << "SSC " << qIdx << std::string(10, ' ');
  }
};

// Full Swap Row (FSR)
class MInstFSR : public MemoryInst {
public:
  int qIdx;
  int cycle;

  MInstFSR(int qIdx, int cycle)
      : MemoryInst(MOp_FSR), qIdx(qIdx), cycle(cycle) {}

  std::ostream& print(std::ostream& os) const override {
    return os << "FSR <cycle=" << cycle << "> " << qIdx;
  }
};

// Full Swap Col (FSC)
class MInstFSC : public MemoryInst {
public:
  int qIdx;
  int cycle;

  MInstFSC(int qIdx, int cycle)
      : MemoryInst(MOp_FSC), qIdx(qIdx), cycle(cycle) {}

  std::ostream& print(std::ostream& os) const override {
    return os << "FSC <cycle=" << cycle << "> " << qIdx;
  }
};

// External (EXT)
class MInstEXT : public MemoryInst {
public:
  std::vector<int> flags;

  MInstEXT(std::initializer_list<int> flags)
      : MemoryInst(MOp_EXT), flags(flags) {}
  MInstEXT(const std::vector<int>& flags) : MemoryInst(MOp_EXT), flags(flags) {}

  std::ostream& print(std::ostream& os) const override;
};

class GateInst {
private:
  GInstKind gKind;

public:
  GateBlock* block;
  FPGAGateCategory blockKind;

  GateInst(GInstKind gKind)
      : gKind(gKind), block(nullptr), blockKind(FPGAGateCategory::General) {}

  GateInst(GInstKind gKind, GateBlock* block, FPGAGateCategory blockKind)
      : gKind(gKind), block(block), blockKind(blockKind) {}

  virtual ~GateInst() = default;

  GInstKind getKind() const { return gKind; }

  bool isNull() const { return getKind() == GOp_NUL; }
  virtual std::ostream& print(std::ostream& os) const {
    assert(false && "Calling from base class");
    return os;
  }
};

class GInstNUL : public GateInst {
public:
  GInstNUL() : GateInst(GOp_NUL) {}

  std::ostream& print(std::ostream& os) const override { return os << "NUL"; }
};

// Single Qubit Gate (SQ)
class GInstSQ : public GateInst {
public:
  GInstSQ(GateBlock* block, FPGAGateCategory blockKind)
      : GateInst(GOp_SQ, block, blockKind) {}

  std::ostream& print(std::ostream& os) const override;
};

// Unitary Permutation Gate (UP)
class GInstUP : public GateInst {
public:
  GInstUP(GateBlock* block, FPGAGateCategory blockKind)
      : GateInst(GOp_UP, block, blockKind) {}

  std::ostream& print(std::ostream& os) const override;
};

struct FPGACostConfig {
  // If the lowest-significant loaded-in qubit has qubit index less than this
  // value, external memory access takes twice the time (default to 7)
  int lowestQIdxForTwiceExtTime;
};

class Instruction {
public:
  enum CostKind {
    CK_GeneralSQGate,  // SQ gate
    CK_RealOnlySQGate, // SQ real-only gate
    CK_UPGate,         // UP gate
    CK_NonExtMemTime,  // FSR, FSC, SSR, SSC with no gate inst
    CK_ExtMemTime,     // EXT mem inst
    CK_TwiceExtMemTime // twice EXT mem inst
  };

public:
  std::unique_ptr<MemoryInst> mInst;
  std::unique_ptr<GateInst> gInst;

  Instruction(
      std::unique_ptr<MemoryInst> _mInst, std::unique_ptr<GateInst> _gInst) {
    setMInst(std::move(_mInst));
    setGInst(std::move(_gInst));
  }

  std::ostream& print(std::ostream& os) const {
    mInst->print(os) << " : ";
    gInst->print(os) << "\n";
    return os;
  }

  void setMInst(std::unique_ptr<MemoryInst> inst) {
    if (inst) {
      mInst = std::move(inst);
      return;
    }
    mInst = std::make_unique<MInstNUL>();
  }

  void setGInst(std::unique_ptr<GateInst> inst) {
    if (inst) {
      gInst = std::move(inst);
      return;
    }
    gInst = std::make_unique<GInstNUL>();
  }

  CostKind getCostKind(const FPGACostConfig&) const;
};

// top-level function to generate FPGA instructions from a CircuitGraph
std::vector<Instruction> genInstruction(
    const CircuitGraph&, const FPGAInstGenConfig&);

}; // namespace cast::fpga

#endif // CAST_FPGAINST_H
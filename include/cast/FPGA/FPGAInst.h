#ifndef CAST_FPGA_FPGAINST_H
#define CAST_FPGA_FPGAINST_H

#include "cast/IR/IRNode.h"
#include "cast/FPGA/FPGAGateCategory.h"

namespace cast::fpga {

// Gate Instruction Kind
enum GInstKind : int {
  GOp_NUL = 0,

  GOp_SQ = 1, // Single Qubit
  GOp_UP = 2, // Unitary Permutation
};

// Memory Instruction Kind
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

  MInstEXT(const std::vector<int>& flags)
    : MemoryInst(MOp_EXT), flags(flags) {}

  std::ostream& print(std::ostream& os) const override;
};

class GateInst {
private:
  GInstKind gKind;
public:
  ConstQuantumGatePtr gate;
  FPGAGateCategory blockKind;

  GateInst(GInstKind gKind)
      : gKind(gKind), gate(nullptr), blockKind(FPGAGateCategory::General) {}

  GateInst(GInstKind gKind, ConstQuantumGatePtr gate, FPGAGateCategory blockKind)
      : gKind(gKind), gate(gate), blockKind(blockKind) {}

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
  GInstSQ(ConstQuantumGatePtr gate, FPGAGateCategory blockKind)
    : GateInst(GOp_SQ, gate, blockKind) {}

  std::ostream& print(std::ostream& os) const override;
};

// Unitary Permutation Gate (UP)
class GInstUP : public GateInst {
public:
  GInstUP(ConstQuantumGatePtr gate, FPGAGateCategory blockKind)
    : GateInst(GOp_UP, gate, blockKind) {}

  std::ostream& print(std::ostream& os) const override;
};

struct FPGACostConfig {
  // If the least-significant loaded-in qubit has qubit index less than this
  // value, external memory access takes twice the time (default to 7)
  int lowestQIdxForTwiceExtTime = 7;
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

private:
  std::unique_ptr<MemoryInst> _mInst;
  std::unique_ptr<GateInst> _gInst;
public:
  Instruction(std::unique_ptr<MemoryInst> _mInst,
              std::unique_ptr<GateInst> _gInst) {
    setMInst(std::move(_mInst));
    setGInst(std::move(_gInst));
  }

  std::ostream& print(std::ostream& os) const {
    _mInst->print(os) << " : ";
    _gInst->print(os) << "\n";
    return os;
  }

  /// Get the memory instruction.
  const MemoryInst* getMInst() const { return _mInst.get(); }

  /// Get the gate instruction.
  const GateInst* getGInst() const { return _gInst.get(); }

  /// Set the memory instruction.
  /// @param inst Mem instruction. Could be nullptr, in which case it will be
  /// set to an MInstNul
  void setMInst(std::unique_ptr<MemoryInst> inst) {
    if (inst) {
      _mInst = std::move(inst);
      return;
    }
    _mInst = std::make_unique<MInstNUL>();
  }

  /// @brief Set the gate instruction.
  /// @param inst Gate instruction. Could be nullptr, in which case it will be 
  /// set to a GInstNUL.
  void setGInst(std::unique_ptr<GateInst> inst) {
    if (inst) {
      _gInst = std::move(inst);
      return;
    }
    _gInst = std::make_unique<GInstNUL>();
  }

  CostKind getCostKind(const FPGACostConfig&) const;
};

}; // namespace cast::fpga

#endif // CAST_FPGA_FPGAINST_H
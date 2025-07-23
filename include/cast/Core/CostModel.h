#ifndef CAST_CORE_COSTMODEL_H
#define CAST_CORE_COSTMODEL_H

#include "cast/Core/Precision.h"
#include "cast/Core/QuantumGate.h"
#include <string>

namespace cast {

// An abstract class for all cost models.
class CostModel {
public:
  enum CostModelKind {
    CM_Base,
    CM_SizeOnly, // Size only cost model
    CM_Constant, // Constant cost model. Every gate takes the same time.
    CM_CPU,      // CPU cost model
    CM_CUDA,
    CM_End
  };

protected:
  CostModelKind _kind;

public:
  explicit CostModel(CostModelKind kind) : _kind(kind) {}

  virtual ~CostModel() = default;

  CostModelKind getKind() const { return _kind; }

  // The time it takes to update 1 GiB of memory, in seconds.
  virtual double computeGiBTime(const QuantumGate* gate) const = 0;

  virtual std::ostream& displayInfo(std::ostream& os, int verbose) const {
    return os << "CostModel::displayInfo() not implemented";
  }
};

/// @brief \c SizeOnlyCostModel is based on the size and operation count of
/// fused gates.
class SizeOnlyCostModel : public CostModel {
  int maxSize;
  int maxOp;
  double zeroTol;

public:
  SizeOnlyCostModel(int maxSize, int maxOp, double zeroTol)
      : CostModel(CM_SizeOnly), maxSize(maxSize), maxOp(maxOp),
        zeroTol(zeroTol) {}

  double computeGiBTime(const QuantumGate* gate) const override;

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;

  static bool classof(const CostModel* model) {
    return model->getKind() == CM_SizeOnly;
  }
};

class ConstantCostModel : public CostModel {
public:
  ConstantCostModel() : CostModel(CM_Constant) {}

  double computeGiBTime(const QuantumGate* gate) const override { return 1.0; }

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;

  static bool classof(const CostModel* model) {
    return model->getKind() == CM_Constant;
  }
};

} // namespace cast

#endif // CAST_CORE_COSTMODEL_H
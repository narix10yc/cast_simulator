#ifndef CAST_CORE_FUSIONCONFIG_H
#define CAST_CORE_FUSIONCONFIG_H

#include "cast/Core/CostModel.h"
#include "utils/InfoLogger.h"

namespace cast {

class FusionConfig {
protected:
  enum FusionConfigKind {
    FC_Base,     // Base fusion config
    FC_SizeOnly, // Size only fusion config
    FC_CPU,      // CPU fusion config
    FC_CUDA,     // CUDA fusion config
    FC_End
  };

  FusionConfigKind kind_;
  std::unique_ptr<CostModel> costModel_ = nullptr;

public:
  explicit FusionConfig(FusionConfigKind kind) : kind_(kind) {}

  virtual ~FusionConfig() = default;

  FusionConfigKind getKind() const { return kind_; }

  void setCostModel(std::unique_ptr<CostModel> cm) {
    costModel_ = std::move(cm);
  }

  CostModel* getCostModel() { return costModel_.get(); }

  const CostModel* getCostModel() const { return costModel_.get(); }

  // Zero tolerance. This mainly affects calculating the operation count.
  double zeroTol = 1e-8;

  // Swapping tolerance. Set to 0.0 or negative to disable.
  double swapTol = 0.0;

  // The range of sizes in fusion algorithms.
  // The lower limit effectively controls turning on/off agglomerative scheme
  // We pose an upper limit this because sometimes even though the cost model
  // predicts a benefit, kernel generation may be too expensive.
  int sizeMin = 2;
  int sizeMax = 6;

  /// How much benefit do we recognize as significant. For example, if set to
  /// 0.1, then we accept fusion whenever costModel predicts >10% improvement
  double benefitMargin = 0.0;

  bool enableMultiTraverse = true;

  // Set the aggresiveness level of fusion.
  // level <= 0: sizeMax = 4. Disables swapping analysis.
  // level == 1: sizeMax = 5. Disables swapping analysis.
  // level == 2: sizeMax = 6 (default). Disables swapping analysis.
  // level >= 3: sizeMax = 7. Enables swapping analysis.
  void setAggresiveness(int level) {
    // Swapping analysis is enabled only when level >= 3
    swapTol = 0.0;
    if (level <= 0)
      sizeMax = 4;
    else if (level == 1)
      sizeMax = 5;
    else if (level == 2)
      sizeMax = 6;
    else { // level >= 3
      sizeMax = 7;
      swapTol = 1e-8;
    }
  }

  virtual void displayInfo(utils::InfoLogger logger) const {
    if (costModel_) {
      logger.put("CostModel");
      costModel_->displayInfo(logger.indent());
    } else {
      logger.put("CostModel", "None");
    }
    logger.put("Zero Tolerance ", zeroTol)
        .put("Swap Tolerance ", swapTol)
        .put("Size Range     ",
             [=, this](std::ostream& os) {
               os << "[" << sizeMin << ", " << sizeMax << "]";
             })
        .put("Benefit Margin ", benefitMargin)
        .put("Multi-Traverse ", enableMultiTraverse);
  }
};

class SizeOnlyFusionConfig : public FusionConfig {
public:
  SizeOnlyFusionConfig(int size) : FusionConfig(FC_SizeOnly) {
    sizeMin = 2;
    sizeMax = size;
  }

  void displayInfo(utils::InfoLogger logger) const override {
    logger.put("SizeOnlyFusionConfig").put("size", sizeMin);
    FusionConfig::displayInfo(logger.indent(logger.verbose - 1));
  }

  static bool classof(const FusionConfig* config) {
    return config->getKind() == FC_SizeOnly;
  }
}; // SizeOnlyFusionConfig

} // end namespace cast

#endif // CAST_CORE_FUSIONCONFIG_H
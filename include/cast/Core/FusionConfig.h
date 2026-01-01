#ifndef CAST_CORE_FUSIONCONFIG_H
#define CAST_CORE_FUSIONCONFIG_H

#include "cast/Core/CostModel.h"
#include "utils/InfoLogger.h"

namespace cast {

enum class FusionOptLevel {
  Mild = 0,      // sizeMax = 5
  Balanced = 1,  // sizeMax = 6
  Aggressive = 2 // sizeMax = 7
};

} // namespace cast

#include "utils/CSVParsable.h"

namespace utils {

template <> struct CSVField<cast::FusionOptLevel> {
  static void parse(std::string_view token, cast::FusionOptLevel& field) {
    if (token == "mild")
      field = cast::FusionOptLevel::Mild;
    else if (token == "balanced")
      field = cast::FusionOptLevel::Balanced;
    else if (token == "aggressive")
      field = cast::FusionOptLevel::Aggressive;
    else {
      assert(false && "Unknown FusionOptLevel");
    }
  }

  static void write(std::ostream& os, const cast::FusionOptLevel& value) {
    switch (value) {
    case cast::FusionOptLevel::Mild:
      os << "mild";
      break;
    case cast::FusionOptLevel::Balanced:
      os << "balanced";
      break;
    case cast::FusionOptLevel::Aggressive:
      os << "aggressive";
      break;
    default:
      assert(false && "Unknown FusionOptLevel");
      break;
    }
  }
};

} // namespace utils

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

  // Set fusion optimization level.
  // Mild: sizeMax = 5
  // Balanced: sizeMax = 6 (default)
  // Aggresive: sizeMax = 7
  void setOptLevel(FusionOptLevel level) {
    switch (level) {
    case FusionOptLevel::Mild:
      sizeMax = 5;
      break;
    case FusionOptLevel::Balanced:
      sizeMax = 6;
      break;
    case FusionOptLevel::Aggressive:
      sizeMax = 7;
      break;
    default:
      assert(false && "Unknown FusionOptLevel");
      sizeMax = 6;
      break;
    }
  }

  virtual void displayInfo(utils::InfoLogger logger) const {
    if (costModel_) {
      logger.put("CostModel");
      logger.indent([&](auto& l) { costModel_->displayInfo(l); });
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
    logger.put("SizeOnlyFusionConfig").put("size", sizeMax);
  }

  static bool classof(const FusionConfig* config) {
    return config->getKind() == FC_SizeOnly;
  }
}; // SizeOnlyFusionConfig

} // end namespace cast

#endif // CAST_CORE_FUSIONCONFIG_H
#ifndef CAST_CORE_FUSIONCONFIG_H
#define CAST_CORE_FUSIONCONFIG_H

#include "cast/Core/CostModel.h"

constexpr int GLOBAL_MAX_K = 7;

namespace cast {

class FusionConfig {
public:
  enum FusionConfigKind {
    FC_Base,          // Base fusion config
    FC_SizeOnly,      // Size only fusion config
    FC_CPU,           // CPU fusion config
    FC_CUDA,          // CUDA fusion config
    FC_End
  };
protected:
  FusionConfigKind _kind;
public:
  explicit FusionConfig(FusionConfigKind kind) : _kind(kind) {}

  virtual ~FusionConfig() = default;

  FusionConfigKind getKind() const { return _kind; }

  std::unique_ptr<CostModel> costModel = nullptr;

  // Zero tolerance. This mainly affects calculating the operation count.
  double zeroTol = 1e-8;

  // Swapping tolerance. Set to 0.0 or negative to disable.
  double swapTol = 1e-8;
  
  // The range of sizes in fusion algorithms.
  // The lower limit effectively controls turning on/off agglomerative scheme
  // We pose an upper limit this because sometimes even though the cost model
  // predicts a benefit, kernel generation may be too expensive.
  int sizeMin = 2;
  int sizeMax = GLOBAL_MAX_K;
  
  /// How much benefit do we recognize as significant. For example, if set to
  /// 0.1, then we accept fusion whenever costModel predicts >10% improvement
  double benefitMargin = 0.1;

  bool enableMultiTraverse = true;
  bool enableFusionCFOPass = true;

  virtual std::ostream& displayInfo(std::ostream& os, int verbose) const = 0;
};

class SizeOnlyFusionConfig : public FusionConfig {
public:
  SizeOnlyFusionConfig(int size) : FusionConfig(FC_SizeOnly) {
    sizeMin = size;
    sizeMax = size;
  }

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override {
    return os << "SizeOnlyFusionConfig with size " << sizeMin << "\n";
  }

  static bool classof(const FusionConfig* config) {
    return config->getKind() == FC_SizeOnly;
  }
}; // SizeOnlyFusionConfig

} // end namespace cast

#endif // CAST_CORE_FUSIONCONFIG_H
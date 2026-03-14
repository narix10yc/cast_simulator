#ifndef CAST_CORE_PLANNER_H
#define CAST_CORE_PLANNER_H

#include "cast/Core/CostModel.h"

#include "utils/InfoLogger.h"

#include <llvm/Support/Error.h>

#include <string>
#include <variant>

namespace cast {

class Planner {

  /* Device */

  struct DeviceCPU {
    int nThreads;
  };
  struct DeviceCUDA {
    int nGpus;
  };

  using DeviceSelection = std::variant<std::monostate, DeviceCPU, DeviceCUDA>;
  DeviceSelection device_;

  void setDeviceCPU(int nThreads) { device_ = DeviceCPU{nThreads}; }
  void setDeviceCUDA(int nGpus) { device_ = DeviceCUDA{nGpus}; }

  /* Cost Model */

  std::unique_ptr<CostModel> costModel_{};

public:
  /// Set the target device for planning. This function silently fails upon
  /// invalid inputs.
  /// Valid device name formats:
  /// - "cpu" : use all available CPU threads
  /// - "cpu:<nthreads>" or "cpu:<nthreads>t" : use specified number of CPU
  /// threads. For example, "cpu:4" or "cpu:4t"
  /// - "cuda" or "gpu" : use the first available GPU
  void setDevice(const std::string& deviceName);

  void setCostModel(std::unique_ptr<CostModel> model) {
    costModel_ = std::move(model);
  }

  /// Validate the configuration of the planner.
  llvm::Error validate() const;

  void displayInfo(utils::InfoLogger logger) const;

}; // class Planner

} // namespace cast

#endif // CAST_CORE_PLANNER_H
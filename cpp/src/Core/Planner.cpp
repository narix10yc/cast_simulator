#include "cast/Core/Planner.h"
#include "cast/CPU/Config.h"

#include <algorithm>
#include <variant>

using namespace cast;

void Planner::displayInfo(utils::InfoLogger logger) const {
  logger.put("Planner Info")
      .put("Device",
           std::visit(
               [](auto&& d) -> std::string {
                 using T = std::decay_t<decltype(d)>;
                 if constexpr (std::is_same_v<T, std::monostate>)
                   return "Not Set";
                 else if constexpr (std::is_same_v<T, DeviceCPU>)
                   return "CPU (" + std::to_string(d.nThreads) + " threads)";
                 else if constexpr (std::is_same_v<T, DeviceCUDA>)
                   return "CUDA with " + std::to_string(d.nGpus) + " GPUs";
                 else
                   return "Unknown Device";
               },
               device_))
      .put("Cost Model", (costModel_ ? "Set" : "Not Set"));
}

llvm::Error Planner::validate() const {
  if (std::holds_alternative<std::monostate>(device_)) {
    return llvm::createStringError(
        "Device is not set. Call `setDevice` first.");
  }

  if (costModel_ == nullptr) {
    return llvm::createStringError(
        "Cost model is not set. Call `setCostModel` first.");
  }
  // TODO
  return llvm::Error::success();
}

void Planner::setDevice(const std::string& deviceName) {
  auto name = deviceName;
  std::ranges::transform(name, name.begin(), ::tolower);
  const auto* p = name.data();
  // cpus
  if (name.starts_with("cpu")) {
    p += 3;
    if (*p++ != ':') {
      // cpu with default threads
      setDeviceCPU(cast::get_cpu_num_threads());
      return;
    }
    // cpu with specified threads
    // `cpu:<nthreads>` or `cpu:<nthreads>t`
    int nThreads = 0;
    while (*p && std::isdigit(*p)) {
      nThreads = nThreads * 10 + (*p - '0');
      ++p;
    }
    if (nThreads <= 0) {
      // invalid thread count, sliently fail
      return;
    }
    // check for the tail: either nothing or 't'
    if (*p != '\0' && *p != 't') {
      // invalid format, silently fail
      return;
    }

    // good format
    setDeviceCPU(nThreads);

    return;
  }
  // cuda / gpu : they are the same for now
  assert(false && "CUDA device not implemented yet");
  if (name.starts_with("cuda"))
    p += 4;
  else if (name.starts_with("gpu"))
    p += 3;

  if (p == name.data()) {
    // unknown device
    return;
  }
}
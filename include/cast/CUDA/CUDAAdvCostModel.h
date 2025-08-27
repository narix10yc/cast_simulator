#ifndef CAST_CUDA_CUDAADVCOSTMODEL_H
#define CAST_CUDA_CUDAADVCOSTMODEL_H

#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"
#include "cast/Core/CostModel.h"
#include "cast/Core/Precision.h"
#include "cast/Core/QuantumGate.h"
#include "utils/CSVParsable.h"
#include "llvm/Passes/OptimizationLevel.h"

#include <fstream>
#include <map>
#include <utility>

namespace cast {

enum class AccessPattern : int {
  Contiguous = 0, // low min-qubit index
  Semi = 1,
  Strided = 2, // high min-qubit index
  Unknown = -1
};

} // namespace cast

namespace utils {

template <> struct CSVField<cast::AccessPattern> {
  static void parse(std::string_view token, cast::AccessPattern& field) {
    if (token == "contiguous")
      field = cast::AccessPattern::Contiguous;
    else if (token == "semi")
      field = cast::AccessPattern::Semi;
    else if (token == "strided")
      field = cast::AccessPattern::Strided;
    else
      field = cast::AccessPattern::Unknown;
  }

  static void write(std::ostream& os, const cast::AccessPattern& value) {
    switch (value) {
    case cast::AccessPattern::Contiguous:
      os << "contiguous";
      break;
    case cast::AccessPattern::Semi:
      os << "semi";
      break;
    case cast::AccessPattern::Strided:
      os << "strided";
      break;
    case cast::AccessPattern::Unknown:
    default:
      os << "unknown";
      break;
    }
  }
};

} // namespace utils

namespace cast {

struct CUDADeviceInfo {
  int device = 0;
  int warpSize = 32;
  int maxThreadsPerSM = 2048;
  int smCount = 0;
};

class CUDAAdvPerfCache {
  struct Item : utils::CSVParsable<Item> {
    // gate/kernel shape
    int nQubitsGate = 0; // k
    double opCount = 0.0;
    Precision precision = Precision::Unknown;
    AccessPattern pattern = AccessPattern::Contiguous;

    // launch & resources (for information / occupancy curves)
    int gridX, gridY, gridZ;
    int blockX, blockY, blockZ;
    int regsPerThread = 0;
    int smemPerBlock = 0;   // bytes
    double occupancy = 0.0; // [0.0, 1.0]

    // measured & derived perf
    double time_s = 0.0;      // best-of-N seconds
    double bytes = 0.0;       // modeled bytes
    double flops = 0.0;       // modeled flops
    double Bach_GBs = 0.0;    // bytes/time (GB/s, decimal)
    double Pach_GFLOPs = 0.0; // flops/time (GF/s)

    // clang-format off
    CSV_DATA_FIELD(nQubitsGate, opCount, precision, pattern,
                   gridX, gridY, gridZ,
                   blockX, blockY, blockZ,
                   regsPerThread,
                   smemPerBlock, occupancy, time_s, bytes, flops, Bach_GBs,
                   Pach_GFLOPs);
    // clang-format on
  };

  std::vector<Item> items_;

  // Calibration runs: generate, JIT, measure
  void runPreliminaryExperiments(const CUDAKernelGenConfig& cfg,
                                 int nQubits,
                                 int nRunsHint,
                                 std::vector<int>& weights,
                                 int verbose);

public:
  CUDAAdvPerfCache() = default;

  std::span<const Item> items() const { return items_; }

  void runExperiments(const CUDAKernelGenConfig& cfg,
                      const CUDADeviceInfo& dev,
                      int nQubits,
                      int nRuns,
                      int verbose);

  void writeResults(std::ostream& os) const;

  bool loadFromFile(const std::string& fileName) {
    std::ifstream ifs(fileName);
    if (!ifs.is_open()) {
      return false;
    }
    std::string line;
    std::getline(ifs, line); // Read header
    if (line != Item::CSV_TITLE) {
      return false;
    }

    while (std::getline(ifs, line)) {
      items_.emplace_back();
      items_.back().parse(line);
    }
    return true;
  }
};

// ----------------------------- query & model -----------------------------
struct CUDAAdvCostQuery {
  int nQubits = 0;
  Precision precision = Precision::Unknown;

  // variant/path choices (the generator may expose only "Global"):
  bool considerSmemTranspose = false;
  bool considerTensorCores = false;
  bool forceGlobalVariant = false;
  bool forceSmemTransposeVariant = false;

  // execution semantics:
  int nLaunches = 1;
  double coverageFraction = 1.0; // 0..1 fraction of state touched
  double bytesFixup = 0.0;       // extra bytes for explicit transposes
};

class CUDAAdvCostModel : public CostModel {
public:
  // ---- Backward-compat penalty knob set (optional) ----
  struct Params {
    double launchOH = 3.0e-6;
    double blkAlpha = 0.25;
    double occAlpha = 0.0;
    double coalAlpha = 0.0;
    double sizeBeta = 0.0;
  } params;

  // Construct from an empty performance cache (to be populated via
  // runExperiments)
  explicit CUDAAdvCostModel(std::unique_ptr<CUDAAdvPerfCache> cache,
                            double zeroTol = 1e-8);

  ~CUDAAdvCostModel() override = default;

  // --------- Primary roofline API (recommended in new code) ----------
  // Predict wall-time (seconds) for a gate or fused block
  double computeTime(const QuantumGate* gate, const CUDAAdvCostQuery& q) const;

  // --------- Backward-compat API (used by existing passes) ----------
  // seconds per GiB memory for this gate shape (roofline-based)
  double computeGiBTime(const QuantumGate* gate) const override;

  // very-cheap predictor (kept for compatibility; uses anchor scaling)
  double computeGiBTimeStage1(const QuantumGate* g) const;

  // refine with probed occupancy (optional)
  struct SkeletonStats {
    int regsPerThread = 0;
    size_t staticSmem = 0;
    double occupancy = 1.0;
  };
  double refineWithProbe(const QuantumGate* g,
                         int blockSize,
                         const SkeletonStats& sk) const;

  // query context (back-compat)
  void setQueryBlockSize(int blk) { queryBlockSize_ = blk; }
  void setQueryPrecision(Precision p) { queryPrecision_ = p; }
  void setPenaltyParams(const Params& p) { params = p; }

  // in-model probing toggles (back-compat; safe no-ops if unused)
  void enableProbing(bool on) { probingEnabled_ = on; }
  void resetProbeBudget(int k) const { probeBudget_ = k; }
  void setProbeThreads(int n) { probeThreads_ = (n < 1 ? 1 : n); }
  void setProbeOptLevel(llvm::OptimizationLevel o) { probeOpt_ = o; }

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;

  static bool classof(const CostModel* base) {
    return base->getKind() == CM_CUDA;
  }

private:
  // calibration fit
  void fitFromCache();

  static double interpLUT(const std::vector<std::pair<double, double>>& lut,
                          double x);

  static AccessPattern classifyPattern(const QuantumGate* g);

  static double estimateBytes(int nQubits,
                              Precision p,
                              double coverage,
                              int nSweeps,
                              double bytesFixup);
  static double
  estimateFlops(const QuantumGate* g, int nQubits, double zeroTol);

  // measured data
  std::unique_ptr<CUDAAdvPerfCache> cache_;
  double zeroTol_;

  // fitted params
  std::map<Precision, double> B_peak_GBs_;
  std::map<Precision, double> F_peak_GFLOPs_;
  std::map<AccessPattern, double> f_coal_;
  std::vector<std::pair<double, double>> gB_lut_, gF_lut_;
  double t_launch_s_ = 3e-6;

  // back-compat query context
  int queryBlockSize_ = 256;
  Precision queryPrecision_ = Precision::F64;

  // stage-2 probing
  bool probingEnabled_ = false;
  mutable int probeBudget_ = 0;
  int probeThreads_ = 1;
  llvm::OptimizationLevel probeOpt_ = llvm::OptimizationLevel::O1;
};

using PenaltyParams = CUDAAdvCostModel::Params;

} // namespace cast

#endif // CAST_CUDA_CUDAADVCOSTMODEL_H

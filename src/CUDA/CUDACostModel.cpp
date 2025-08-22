#ifdef CAST_USE_CUDA

#include "cast/CUDA/CUDACostModel.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"

#include "utils/Formats.h"
#include "utils/PrintSpan.h"
#include "utils/utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cast {

static inline double bytesState(int nQubits, Precision p) {
  const uint64_t elemBytes = (p == Precision::F32 ? 8ULL : 16ULL); // complex<float/double>
  return static_cast<double>(elemBytes) * static_cast<double>(1ULL << nQubits);
}

static AccessPattern classifyPatternVec(const std::vector<int>& qs) {
  if (qs.empty()) return AccessPattern::Contiguous;
  int qmin = *std::min_element(qs.begin(), qs.end());
  if (qmin <= 3) return AccessPattern::Contiguous;
  if (qmin <= 8) return AccessPattern::Semi;
  return AccessPattern::Strided;
}

// best-of-N event timing wrapper
static double timeKernelSeconds(const std::function<void()>& launch, int repeats = 5) {
  cudaEvent_t start{}, stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float bestMs = 1e30f;
  for (int i = 0; i < repeats; ++i) {
    cudaEventRecord(start);
    launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    bestMs = std::min(bestMs, ms);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return double(bestMs) * 1e-3;
}

// Driver API occupancy using CUfunction
static double occupancyFromDriver(const CUDADeviceInfo& dev,
                                  CUfunction fn,
                                  int blockSize,
                                  int smemBytes /*dynamic + static*/) {
  if (!fn) return 0.0;
  int activeBlocksPerSM = 0;
  CUresult rc = cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &activeBlocksPerSM, fn, blockSize, smemBytes);
  if (rc != CUDA_SUCCESS) return 0.0;
  const int maxThreads = dev.maxThreadsPerSM;
  const double occ = double(activeBlocksPerSM * blockSize) / double(maxThreads);
  return std::clamp(occ, 0.0, 1.0);
}

// Launch latency measurement via Driver API + in-memory PTX no-op
static bool ensurePrimaryContext() {
  (void)cudaFree(0);
  CUcontext ctx = nullptr;
  cuCtxGetCurrent(&ctx);
  return ctx != nullptr;
}

static CUfunction buildNullKernelPTX(CUmodule& modOut) {
  if (!ensurePrimaryContext()) return nullptr;

  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
  const int sm = prop.major * 10 + prop.minor;

  std::ostringstream oss;
  oss << ".version 7.0\n"
      << ".target sm_" << sm << "\n"
      << ".address_size 64\n"
      << ".visible .entry __cast_null() {\n"
      << "  ret;\n"
      << "}\n";
  const std::string ptx = oss.str();

  CUmodule mod = nullptr;
  CUjit_option opt = CU_JIT_OPTIMIZATION_LEVEL;
  unsigned int lvl = 0;
  void* opts[] = { &lvl };
  if (cuModuleLoadDataEx(&mod, ptx.c_str(), 1, &opt, opts) != CUDA_SUCCESS) return nullptr;

  CUfunction fn = nullptr;
  if (cuModuleGetFunction(&fn, mod, "__cast_null") != CUDA_SUCCESS) {
    cuModuleUnload(mod);
    return nullptr;
  }
  modOut = mod;
  return fn;
}

static double measureLaunchLatencyUs_Driver(int warmup = 20, int trials = 500) {
  CUmodule mod = nullptr;
  CUfunction fn = buildNullKernelPTX(mod);
  if (!fn) return -1.0;

  for (int i = 0; i < warmup; ++i) {
    (void)cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, 0, nullptr, nullptr);
  }
  cuCtxSynchronize();

  cudaEvent_t s{}, e{};
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < trials; ++i) {
    (void)cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, 0, nullptr, nullptr);
  }
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s);
  cudaEventDestroy(e);

  cuModuleUnload(mod);
  return (ms * 1e3) / trials; // microseconds
}

// percentile helper
static double percentile(std::vector<double> v, double q01) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double x = std::clamp(q01, 0.0, 1.0) * double(v.size() - 1);
  const std::size_t i0 = std::size_t(std::floor(x));
  const std::size_t i1 = std::min(v.size() - 1, i0 + 1);
  const double t = x - double(i0);
  return v[i0] * (1.0 - t) + v[i1] * t;
}


// --------------------------- CUDAPerformanceCache -----------------------------
void CUDAPerformanceCache::Item::write(std::ostream& os) const {
  os << nQubitsGate << "," << opCount << ","
     << int(precision) << ","
     << int(variant) << ","
     << int(path) << ","
     << int(pattern) << ","
     << gridDim.x << "," << gridDim.y << "," << gridDim.z << ","
     << blockDim.x << "," << blockDim.y << "," << blockDim.z << ","
     << regsPerThread << ","
     << smemPerBlock << ","
     << occupancy << ","
     << time_s << ","
     << bytes << ","
     << flops << ","
     << Bach_GBs << ","
     << Pach_GFLOPs;
}

void CUDAPerformanceCache::runPreliminaryExperiments(
  const CUDAKernelGenConfig& cfg,
  const CUDADeviceInfo& /*dev*/,
  int nQubits, int nRunsHint,
  std::vector<int>& weights, int verbose)
{
  // Build 2*{k=1..5} dense gates and measure GB/s to bias weights to transition region.
  CUDAKernelManager km;
  std::vector<int> qubits;
  std::vector<std::string> names;

  // generate
  int idx = 0;
  for (int k = 1; k <= 5; ++k) {
    for (int r = 0; r < 2; ++r) {
      utils::sampleNoReplacement(nQubits, k, qubits);
      auto gate = StandardQuantumGate::RandomUnitary(qubits);
      const std::string name = "pre_k" + std::to_string(k) + "_" + std::to_string(r);
      km.genStandaloneGate(cfg, gate, name).consumeError();
      names.push_back(name);
      ++idx;
    }
  }

  // compile & JIT
  km.emitPTX(/*threads*/1, llvm::OptimizationLevel::O1, /*verbose*/0);
  km.initCUJIT(/*threads*/1, /*verbose*/0);

  CUDAStatevector<double> sv(nQubits);
  sv.randomize();

  const double S = bytesState(nQubits, cfg.precision);
  std::vector<double> memSpd(5, 0.0);

  // measure (take best over the two runs for each k)
  for (int k = 1; k <= 5; ++k) {
    double bestGBs = 0.0;
    for (int r = 0; r < 2; ++r) {
      const std::string& nm = "pre_k" + std::to_string(k) + "_" + std::to_string(r);
      const CUDAKernelInfo* ki = km.getKernelByName(nm);
      if (!ki) continue;
      const double t = timeKernelSeconds([&](){
        km.launchCUDAKernel(sv.dData(), nQubits, *ki, cfg.blockSize);
      }, /*repeats*/3);
      const double bytes = 2.0 * S;                 // one sweep, R+W
      bestGBs = std::max(bestGBs, (bytes / t) * 1e-9);
    }
    memSpd[k-1] = bestGBs;
    if (verbose) {
      std::cerr << "[pre] dense " << k << "-qubit @ " << bestGBs << " GB/s\n";
    }
  }

  // weight curve (similar flavor to CPU pre-pass)
  weights.assign(std::max(8, nRunsHint/8), 0);
  if ((int)weights.size() < 5) weights.resize(5, 0);
  weights[0] = 100; // 1q base
  for (int k = 2; k <= 5; ++k) {
    constexpr double ratio = 1.1;
    weights[k-1] = int(double(weights[k-2]) * ((ratio - 1.0) * (memSpd[k-2] / std::max(1e-9,memSpd[k-1])) + 2 - ratio));
  }
  for (int k = 6; k <= (int)weights.size(); ++k) {
    weights[k-1] = int(double(weights[k-2]) * 0.45);
  }
  int maxIdx = 0, maxW = 0;
  for (int i = 0; i < (int)weights.size(); ++i) if (weights[i] > maxW) { maxW = weights[i]; maxIdx = i; }
  for (int i = 0; i < (int)weights.size(); ++i) {
    if (i == maxIdx) continue;
    constexpr double ratio = 1.5;
    double w = double(weights[i]) / std::pow(ratio, std::abs(maxIdx - i));
    weights[i] = int(w);
  }
}

void CUDAPerformanceCache::runExperiments(
  const CUDAKernelGenConfig& cfg,
  const CUDADeviceInfo& dev,
  int nQubits, int nRuns, int verbose)
{
  std::vector<StandardQuantumGatePtr> gates;
  gates.reserve(nRuns);

  // 1) pre-pass weights
  std::vector<int> nQubitsWeights;
  runPreliminaryExperiments(cfg, dev, nQubits, nRuns, nQubitsWeights, verbose);

  auto addRandomGate = [&](float /*erasureProb*/) {
    int sum = std::accumulate(nQubitsWeights.begin(), nQubitsWeights.end(), 0);
    if (sum <= 0) return;
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_int_distribution<int> pick(0, sum-1);
    int r = pick(gen), acc = 0;
    for (int i = 0; i < (int)nQubitsWeights.size(); ++i) {
      acc += nQubitsWeights[i];
      if (r < acc) {
        std::vector<int> tgt; utils::sampleNoReplacement(nQubits, i+1, tgt);
        gates.emplace_back(StandardQuantumGate::RandomUnitary(tgt));
        return;
      }
    }
  };

  // 2) seed with dense 1..5q twice
  for (int q = 1; q <= 5; ++q) {
    std::vector<int> tgt;
    utils::sampleNoReplacement(nQubits, q, tgt);
    gates.emplace_back(StandardQuantumGate::RandomUnitary(tgt));
    utils::sampleNoReplacement(nQubits, q, tgt);
    gates.emplace_back(StandardQuantumGate::RandomUnitary(tgt));
  }
  const int initial = (int)gates.size();
  for (int run = 0; run < nRuns - initial; ++run) addRandomGate(/*eraseProb*/0.0f);

  // 3) JIT
  CUDAKernelManager km;
  int idx = 0;
  for (auto& g : gates) {
    km.genStandaloneGate(cfg, g, "bench_gate_" + std::to_string(idx++)).consumeError();
  }
  km.emitPTX(/*threads*/1, llvm::OptimizationLevel::O3, /*verbose*/0);
  km.initCUJIT(/*threads*/1, /*verbose*/0);

  CUDAStatevector<double> sv(nQubits);
  sv.randomize();

  // 4) Benchmark
  const double S = bytesState(nQubits, cfg.precision);
  const int repeats = 3;

  for (const auto& uptr : km.getAllStandaloneKernels()) {
    const CUDAKernelInfo& k = *uptr;
    const std::vector<int> qs = k.gate->qubits();

    // Access pattern and shape
    const AccessPattern pattern = classifyPatternVec(qs);
    const int kq = k.gate->nQubits();
    const int oc = k.opCount;
    const int blk = (cfg.blockSize > 0) ? cfg.blockSize : 256;

    // Time
    const double time_s = timeKernelSeconds([&](){
      km.launchCUDAKernel(sv.dData(), nQubits, k, blk);
    }, repeats);

    // Occupancy from Driver
    int smemStatic = int(k.sharedMemUsage());
    int blk_for_occ = 0;
    if (k.oneThreadPerBlock) {
        blk_for_occ = 1;
    } else if (k.warpsPerCTA > 0) {
        blk_for_occ = k.warpsPerCTA * dev.warpSize;
    } else {
        int tile = (k.tileSize > 0) ? int(k.tileSize) : 32;
        blk_for_occ = std::max(tile, 32);
    }
    double occ = occupancyFromDriver(dev, k.kernelFunction(), blk_for_occ, k.sharedMemUsage());

    // Derived metrics
    const double bytes = 2.0 * S; // one sweep R+W
    const double groups = double(1ULL << (nQubits - kq));
    const double FLOPS_PER_OP = 8.0;
    const double flops = double(oc) * groups * FLOPS_PER_OP;

    Item it;
    it.nQubitsGate = kq;
    it.opCount = oc;
    it.precision = k.precision;
    it.variant = CUDAVariant::Global; // this manager exposes Global; smem variant can be added
    it.path = CUDAComputePath::ScalarALU;
    it.pattern = pattern;
    it.gridDim = dim3(0,0,0);
    it.blockDim = dim3(blk,1,1);
    it.regsPerThread = int(k.registerUsage());
    it.smemPerBlock = smemStatic;
    it.occupancy = occ;
    it.time_s = time_s;
    it.bytes = bytes;
    it.flops = flops;
    it.Bach_GBs = (bytes / time_s) * 1e-9;
    it.Pach_GFLOPs = (flops / time_s) * 1e-9;

    if (verbose) {
      std::cerr << "[bench] k=" << kq << " occ=" << occ
                << " BW=" << it.Bach_GBs << " GB/s"
                << " GF=" << it.Pach_GFLOPs << " GF/s\n";
    }
    items.emplace_back(std::move(it));
  }

  if (verbose) {
    const double t_us = measureLaunchLatencyUs_Driver();
    if (t_us > 0) std::cerr << "[bench] measured launch latency ~ " << t_us << " us\n";
  }
}

void CUDAPerformanceCache::writeResults(std::ostream& os) const {
  for (const auto& it : items) {
    it.write(os);
    os << "\n";
  }
}


// ------------------------------- CUDACostModel --------------------------------
CUDACostModel::CUDACostModel(std::unique_ptr<CUDAPerformanceCache> cache, double zeroTol)
: CostModel(CM_CUDA), cache_(std::move(cache)), zeroTol_(zeroTol)
{
  fitFromCache();
}

static double safeDiv(double num, double den, double def = 0.0) {
  return (den > 0.0) ? (num / den) : def;
}

void CUDACostModel::fitFromCache() {
  // 1) Bucket achieved GB/s by (precision, variant) and GF/s by (precision, path)
  std::map<std::pair<Precision,CUDAVariant>, std::vector<double>> B_buckets;
  std::map<std::pair<Precision,CUDAComputePath>, std::vector<double>> F_buckets;
  std::map<AccessPattern, std::vector<double>> B_by_pattern;
  std::map<int/*occ*100*/, std::vector<double>> gB_bins, gF_bins;

  for (const auto& it : cache_->items) {
    B_buckets[{it.precision, it.variant}].push_back(it.Bach_GBs);
    F_buckets[{it.precision, it.path}].push_back(it.Pach_GFLOPs);
    B_by_pattern[it.pattern].push_back(it.Bach_GBs);

    int bin = std::clamp(int(std::round(it.occupancy * 100.0)), 0, 100);
    gB_bins[bin].push_back(it.Bach_GBs);
    gF_bins[bin].push_back(it.Pach_GFLOPs);
  }

  // 95th percentile for peaks
  B_peak_GBs_.clear(); F_peak_GFLOPs_.clear();
  for (auto& kv : B_buckets) B_peak_GBs_[kv.first] = percentile(kv.second, 0.95);
  for (auto& kv : F_buckets) F_peak_GFLOPs_[kv.first] = percentile(kv.second, 0.95);

  // f_coal: median ratio vs contiguous median
  double contigMed = 0.0;
  if (auto it = B_by_pattern.find(AccessPattern::Contiguous); it != B_by_pattern.end() && !it->second.empty())
    contigMed = percentile(it->second, 0.50);
  if (contigMed <= 0) contigMed = 1.0;

  f_coal_.clear();
  for (auto& kv : B_by_pattern) {
    const double med = percentile(kv.second, 0.50);
    f_coal_[kv.first] = std::max(0.05, med / contigMed);
  }
  for (AccessPattern p : {AccessPattern::Contiguous, AccessPattern::Semi, AccessPattern::Strided})
    if (!f_coal_.count(p)) f_coal_[p] = (p == AccessPattern::Contiguous ? 1.0 : 0.5);

  // occupancy response curves (normalize medians by 90th in-bin)
  gB_lut_.clear(); gF_lut_.clear();
  for (int b = 0; b <= 100; b += 5) {
    auto itB = gB_bins.find(b), itF = gF_bins.find(b);
    const bool haveB = (itB != gB_bins.end() && !itB->second.empty());
    const bool haveF = (itF != gF_bins.end() && !itF->second.empty());
    if (haveB) {
      const double med = percentile(itB->second, 0.50);
      const double p90 = percentile(itB->second, 0.90);
      if (p90 > 0) gB_lut_.push_back({b/100.0, std::clamp(med/p90, 0.1, 1.0)});
    }
    if (haveF) {
      const double med = percentile(itF->second, 0.50);
      const double p90 = percentile(itF->second, 0.90);
      if (p90 > 0) gF_lut_.push_back({b/100.0, std::clamp(med/p90, 0.1, 1.0)});
    }
  }
  if (gB_lut_.empty()) gB_lut_ = {{0.0,0.4},{0.5,0.9},{1.0,1.0}};
  if (gF_lut_.empty()) gF_lut_ = {{0.0,0.4},{0.5,0.9},{1.0,1.0}};

  // launch latency: prefer direct Driver measurement, otherwise fallback to 5th pct of very small kernels
  double t_us = measureLaunchLatencyUs_Driver(/*warmup*/20, /*trials*/500);
  if (t_us > 0.0) {
    t_launch_s_ = t_us * 1e-6;
  } else {
    std::vector<double> tiny;
    tiny.reserve(cache_->items.size());
    for (const auto& it : cache_->items) if (it.bytes < 1e7) tiny.push_back(it.time_s);
    if (!tiny.empty()) t_launch_s_ = std::max(0.0, percentile(tiny, 0.05));
  }
}

double CUDACostModel::interpLUT(const std::vector<std::pair<double,double>>& lut, double x) {
  if (lut.empty()) return 1.0;
  if (x <= lut.front().first) return lut.front().second;
  if (x >= lut.back().first) return lut.back().second;
  for (std::size_t i = 1; i < lut.size(); ++i) {
    if (x <= lut[i].first) {
      const double x0 = lut[i-1].first, y0 = lut[i-1].second;
      const double x1 = lut[i].first, y1 = lut[i].second;
      const double t = safeDiv(x - x0, x1 - x0, 0.0);
      return y0 * (1.0 - t) + y1 * t;
    }
  }
  return lut.back().second;
}

AccessPattern CUDACostModel::classifyPattern(const QuantumGate* g) {
  return classifyPatternVec(g->qubits());
}

double CUDACostModel::estimateBytes(int nQubits, Precision p, double coverage, int nSweeps, double bytesFixup) {
  const double S = (p == Precision::F32 ? 8.0 : 16.0) * double(1ULL << nQubits);
  return 2.0 * S * std::clamp(coverage, 0.0, 1.0) * std::max(1, nSweeps) + std::max(0.0, bytesFixup);
}

double CUDACostModel::estimateFlops(const QuantumGate* g, int nQubits, double zeroTol) {
  const int k = g->nQubits();
  const int opCount = g->opCount(zeroTol);
  const double groups = double(1ULL << (nQubits - k));
  const double FLOPS_PER_OP = 8.0; // rough complex MAC cost
  return double(opCount) * groups * FLOPS_PER_OP;
}

double CUDACostModel::computeTime(const QuantumGate* gate, const CUDACostQuery& q) const {
  assert(gate);
  assert(q.precision == Precision::F32 || q.precision == Precision::F64);

  const AccessPattern pattern = classifyPattern(gate);

  // Evaluate the "Global" variant (SmemTranspose can be added when exposed)
  const CUDAVariant var = CUDAVariant::Global;

  // Effective bandwidth
  const double Bpeak = [this,&q,var]() {
    auto it = B_peak_GBs_.find({q.precision, var});
    return (it == B_peak_GBs_.end() ? 1.0 : it->second);
  }();
  const double fcoal = [this,pattern]() {
    auto it = f_coal_.find(pattern);
    return (it == f_coal_.end() ? 0.5 : it->second);
  }();
  const double gB = interpLUT(gB_lut_, /*assume mid-high occ*/0.6);
  const double Beff_GBs = std::max(1e-6, Bpeak * fcoal * gB);

  // Effective compute
  const CUDAComputePath path = CUDAComputePath::ScalarALU;
  const double Fpeak = [this,&q,path]() {
    auto it = F_peak_GFLOPs_.find({q.precision, path});
    return (it == F_peak_GFLOPs_.end() ? 1.0 : it->second);
  }();
  const double gF = interpLUT(gF_lut_, 0.6);
  const double Feff_GF = std::max(1e-6, Fpeak * gF);

  // work
  const int nSweeps = 1;
  const double bytes = estimateBytes(q.nQubits, q.precision, q.coverageFraction, nSweeps, q.bytesFixup);
  const double flops = estimateFlops(gate, q.nQubits, zeroTol_);

  // time
  const double t_mem = (bytes * 1e-9) / Beff_GBs; // bytes / (GB/s)
  const double t_cmp = (flops * 1e-9) / Feff_GF;  // flops / (GF/s)
  return std::max(t_mem, t_cmp) + double(q.nLaunches) * t_launch_s_;
}

// Backward-compat API (GiBTime & probing placeholders)

double CUDACostModel::computeGiBTime(const QuantumGate* gate) const {
  // Use pattern + mid occupancy to get seconds/GiB (binary) for this gate shape.
  CUDACostQuery q;
  q.nQubits = std::max(0, 0); // not used directly for per-GiB cost
  q.precision = (queryPrecision_ == Precision::Unknown) ? Precision::F64 : queryPrecision_;
  q.nLaunches = 1;

  const AccessPattern pattern = classifyPattern(gate);
  const double Bpeak = [this,&q]() {
    auto it = B_peak_GBs_.find({q.precision, CUDAVariant::Global});
    return (it == B_peak_GBs_.end() ? 1.0 : it->second);
  }();
  const double fcoal = [this,pattern]() {
    auto it = f_coal_.find(pattern);
    return (it == f_coal_.end() ? 0.5 : it->second);
  }();
  const double gB = interpLUT(gB_lut_, 0.6);
  const double Beff_GBs = std::max(1e-6, Bpeak * fcoal * gB);

  // Convert GB/s (decimal) -> GiB/s (binary)
  const double GiB_per_GB = 1.0 / 1.073741824;
  const double Beff_GiBps = Beff_GBs * GiB_per_GB;
  const double t_mem_perGiB = 1.0 / Beff_GiBps;

  // Compute-path per GiB for this shape (rarely dominant for SV simulators, but keep it)
  const int k = gate->nQubits();
  const int opCount = gate->opCount(zeroTol_);
  const double elemBytes = (q.precision == Precision::F32 ? 8.0 : 16.0);
  const double FLOPS_PER_OP = 8.0;
  const double flops_per_byte = (opCount * FLOPS_PER_OP) / (2.0 * elemBytes * std::ldexp(1.0, k));
  const double Fpeak = [this,&q]() {
    auto it = F_peak_GFLOPs_.find({q.precision, CUDAComputePath::ScalarALU});
    return (it == F_peak_GFLOPs_.end() ? 1.0 : it->second);
  }() * interpLUT(gF_lut_, 0.6);
  const double t_cmp_perGiB = (flops_per_byte * 1073741824.0) / (Fpeak * 1.0e9);

  return std::max(t_mem_perGiB, t_cmp_perGiB);
}

double CUDACostModel::computeGiBTimeStage1(const QuantumGate* g) const {
  // cheap scaling from B_peak; keep for compatibility
  Precision p = (queryPrecision_ == Precision::Unknown) ? Precision::F64 : queryPrecision_;
  const AccessPattern pat = classifyPattern(g);
  const double Bpeak = [this,p]() {
    auto it = B_peak_GBs_.find({p, CUDAVariant::Global});
    return (it == B_peak_GBs_.end() ? 1.0 : it->second);
  }();
  const double fcoal = [this,pat]() {
    auto it = f_coal_.find(pat);
    return (it == f_coal_.end() ? 0.5 : it->second);
  }();
  const double gB = interpLUT(gB_lut_, 0.6);
  const double Beff_GBs = std::max(1e-6, Bpeak * fcoal * gB);
  const double GiB_per_GB = 1.0 / 1.073741824;
  return 1.0 / (Beff_GBs * GiB_per_GB) + params.launchOH;
}

double CUDACostModel::refineWithProbe(const QuantumGate* g,
                                      int /*blockSize*/,
                                      const SkeletonStats& sk) const {
  // Mild occupancy correction (kept for compat with older logic).
  const double base = computeGiBTimeStage1(g) - params.launchOH;
  const double effGiBps = (base > 1e-12) ? (1.0 / base) : 1e12;
  const double occAdj = std::pow(std::max(0.05, sk.occupancy), params.occAlpha);
  const double effGiBpsAdj = effGiBps * occAdj;
  return 1.0 / std::max(1e-9, effGiBpsAdj) + params.launchOH;
}

std::ostream& CUDACostModel::displayInfo(std::ostream& os, int verbose) const {
  os << "CUDA Cost Model (roofline)\n";
  os << "  t_launch ~ " << t_launch_s_ << " s\n";
  os << "  B_peak (GB/s):\n";
  for (const auto& kv : B_peak_GBs_) {
    os << "    prec=f" << int(kv.first.first)
       << " var=" << int(kv.first.second)
       << " => " << kv.second << "\n";
  }
  os << "  F_peak (GFLOP/s):\n";
  for (const auto& kv : F_peak_GFLOPs_) {
    os << "    prec=f" << int(kv.first.first)
       << " path=" << int(kv.first.second)
       << " => " << kv.second << "\n";
  }
  if (verbose) {
    os << "  f_coal(pattern): ";
    for (const auto& kv : f_coal_) os << int(kv.first) << "->" << kv.second << "  ";
    os << "\n";
  }
  return os;
}

} // namespace cast

#endif // CAST_USE_CUDA

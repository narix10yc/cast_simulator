#pragma once
#include <vector>
#include <bitset>
#include <span>

namespace cast {

struct Layout {
  std::vector<int> physPosOfLog; // size = nQubits
  std::vector<int> logOfPhys;    // size = nQubits
};

inline Layout makeIdentityLayout(unsigned n) {
  Layout L; L.physPosOfLog.resize(n); L.logOfPhys.resize(n);
  for (unsigned i=0;i<n;i++){ L.physPosOfLog[i]=i; L.logOfPhys[i]=i; }
  return L;
}

inline bool isContiguousLSB(const Layout& L, std::span<const int> targets) {
  const unsigned k = targets.size();
  std::bitset<64> seen;
  for (int q : targets) {
    int p = L.physPosOfLog[q];
    if (p < 0 || p >= int(k) || seen.test(p)) return false;
    seen.set(p);
  }
  return seen.count() == k;
}

struct AxisSwap { int physA, physB; };

inline std::vector<AxisSwap> planBringTargetsToLSB(Layout L, std::span<const int> targets) {
  const unsigned k = targets.size();
  std::vector<AxisSwap> plan;
  auto& phys = L.physPosOfLog;
  auto& log  = L.logOfPhys;

  for (unsigned p = 0; p < k; ++p) {
    int desiredQ = -1;
    for (int q : targets) {
      int pos = phys[q];
      if (pos >= int(p)) { desiredQ = q; break; }
    }
    if (desiredQ < 0) continue;
    int curP = phys[desiredQ];
    if (curP == int(p)) continue;
    plan.push_back({curP,(int)p});
    int qAtP   = log[p];
    int qAtCur = log[curP];
    std::swap(log[p], log[curP]);
    phys[qAtP]   = curP;
    phys[qAtCur] = p;
  }
  return plan;
}

inline void commitLayout(Layout& L, std::span<const AxisSwap> swaps) {
  for (auto [a,b] : swaps) {
    int qa = L.logOfPhys[a];
    int qb = L.logOfPhys[b];
    std::swap(L.logOfPhys[a], L.logOfPhys[b]);
    std::swap(L.physPosOfLog[qa], L.physPosOfLog[qb]);
  }
}

inline bool subsetFitsLSB(const Layout& L, std::span<const int> targets, unsigned window) {
  for (int q : targets) if (L.physPosOfLog[q] >= (int)window) return false;
  return true;
}

} // namespace cast

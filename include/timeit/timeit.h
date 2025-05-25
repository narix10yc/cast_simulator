#ifndef TIMEIT_TIMEIT_H
#define TIMEIT_TIMEIT_H

#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace timeit {

class TimingResult {
  int repeat, replication;

public:
  std::vector<double> tArr;
  double min, med, q1, q3;
  int n_sig_dig = 4;
  TimingResult()
  : repeat(0), replication(0), tArr(), min(0), med(0), q1(0), q3(0) {}
  
  TimingResult(int repeat, int replication, const std::vector<double>& tArr)
      : repeat(repeat), replication(replication), tArr(tArr) {
    assert(repeat >= 1);
    assert(replication >= 1);
    calcStats();
  }

  static std::string timeToString(double, int);

  std::ostream& display(int nsig = 4, std::ostream& os = std::cerr) const;

  std::string raw_string() const;

  void setNumSignificantDigits(int n) {
    assert(n >= 1);
    n_sig_dig = n;
  }

private:
  void calcStats();
};

/// Total running time will be approximately warmupTime + replication * runTime
class Timer {
  double warmupTime = 0.05;
  double runTime = 0.1;
  int replication;
  int verbose;

public:
  Timer(int replication = 15, int verbose = 0)
      : replication(replication), verbose(verbose) {
    assert(replication >= 1);
  }

  void setWarmupTime(double t) { warmupTime = t; }
  void setRunTime(double t) { runTime = t; }
  void setReplication(int r) {
    assert(r > 0 && r < 100);
    replication = r;
  }

  TimingResult timeit(
      const std::function<void()>& method,
      const std::function<void()>& setup,
      const std::function<void()>& teardown) const;

  TimingResult timeit(const std::function<void()>& method) const {
    return timeit(method, []() {}, []() {});
  }

  TimingResult timeitFixedRepeat(
      const std::function<void()>& method, int _repeat,
      const std::function<void()>& setup = []() {},
      const std::function<void()>& teardown = []() {}) const;
  
  TimingResult timeitPartial(
    const std::function<void()>& preMethod,
    const std::function<void()>& timedMethod,
    const std::function<void()>& postMethod,
    const std::function<void()>& setup,
    const std::function<void()>& teardown) const;
};

} // namespace timeit

#endif // TIMEIT_TIMEIT_H
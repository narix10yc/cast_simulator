#include "timeit/timeit.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace timeit;

namespace {
double getMedian(const std::vector<double>& arr, size_t start, size_t end) {
  auto l = end - start;
  if (l % 2 == 0)
    return 0.5 * (arr[start + l / 2 - 1] + arr[start + l / 2]);
  return arr[start + l / 2];
}
} // namespace

void TimingResult::calcStats() {
  auto l = tArr.size();
  if (repeat == 0) {
    return;
  }
  if (l == 0) {
    return;
  }
  if (l == 1) {
    min = tArr[0];
    med = tArr[0];
    return;
  }
  std::ranges::sort(tArr);
  min = tArr[0] / repeat;
  med = getMedian(tArr, 0, tArr.size()) / repeat;
  q1 = getMedian(tArr, 0, tArr.size() / 2) / repeat;
  q3 = getMedian(tArr, tArr.size() / 2, tArr.size()) / repeat;
}

std::string TimingResult::timeToString(double t, int n_sig_dig) {
  double timeValue;
  std::string unit;

  if (t >= 1) {
    timeValue = t;
    unit = "s";
  } else if (t >= 1e-3) {
    timeValue = t * 1e3;
    unit = "ms";
  } else if (t >= 1e-6) {
    timeValue = t * 1e6;
    unit = "us";
  } else {
    timeValue = t * 1e9;
    unit = "ns";
  }

  // significant digits
  int precision =
      n_sig_dig - 1 - static_cast<int>(std::floor(std::log10(timeValue)));

  std::ostringstream ss;

  ss << std::fixed << std::setprecision(precision) << timeValue << " "
     << unit;
  return ss.str();
}

std::ostream& TimingResult::display(int n_sig_dig, std::ostream& os) const {
  os << replication << " replications (" << repeat << (repeat == 1 ? " repeat each): " : " repeats each): ")
     << "min " << timeToString(min, n_sig_dig) << "; median "
     << timeToString(med, n_sig_dig) << "\n";
  return os;
}

std::string TimingResult::raw_string() const {
  if (tArr.size() == 0) {
    return "[]";
  }
  std::stringstream ss;
  ss << "[" << std::scientific;
  for (size_t i = 0; i < tArr.size() - 1; ++i) {
    ss << (tArr[i] / repeat) << " ";
  }
  ss << (tArr[tArr.size() - 1] / repeat) << "]";
  return ss.str();
}

TimingResult Timer::timeit(
    const std::function<void()>& method,
    const std::function<void()>& setup,
    const std::function<void()>& teardown) const {
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using Duration = std::chrono::duration<double>;

  int repeat = 1;
  std::vector<double> tArr(replication);
  TimePoint tic, toc;
  TimePoint totalT0, totalT1;
  double dur;

  if (verbose > 0) {
    std::cerr << "Desired warm-up/run time: "
              << warmupTime << "/" << runTime << " s;\n";
    std::cerr << "Number of Replication(s): " << replication << "\n";
  }

  setup();

  if (verbose > 0)
    totalT0 = Clock::now();
  // warmup
  tic = Clock::now();
  method();
  toc = Clock::now();
  dur = Duration(toc - tic).count();

  // if running method() once takes longer than runTime, we will skip warmup
  // and record this experiment run as one of the final results
  int r0 = 0;
  if (dur > runTime) {
    r0 = 1;
    tArr[0] = dur;
  }

  // warm-up loop
  while (dur < warmupTime) {
    repeat *= 2;
    tic = Clock::now();
    for (unsigned r = 0; r < repeat; ++r)
      method();
    toc = Clock::now();
    dur = Duration(toc - tic).count();
  }

  // main loop
  if (r0 == 0)
    repeat = static_cast<double>(repeat) * runTime / dur + 1;
  else
    tArr[0] *= repeat;
  for (unsigned r = r0; r < replication; ++r) {
    tic = Clock::now();
    for (unsigned i = 0; i < repeat; ++i)
      method();
    toc = Clock::now();
    dur = Duration(toc - tic).count();
    tArr[r] = dur;
  }
  if (verbose > 0) {
    totalT1 = Clock::now();
    std::cerr << "Actual running time: "
              << Duration(totalT1 - totalT0).count() << " s;\n";
  }

  teardown();

  return TimingResult(repeat, replication, tArr);
}

TimingResult Timer::timeitFixedRepeat(
    const std::function<void()>& method, int _repeat,
    const std::function<void()>& setup,
    const std::function<void()>& teardown) const {
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using Duration = std::chrono::duration<double>;

  std::vector<double> tArr(replication);
  TimePoint tic, toc;
  double dur;

  setup();
  // main loop
  for (unsigned r = 0; r < replication; ++r) {
    tic = Clock::now();
    for (unsigned i = 0; i < _repeat; ++i)
      method();
    toc = Clock::now();
    dur = Duration(toc - tic).count();
    tArr[r] = dur;
  }

  teardown();
  return TimingResult(_repeat, replication, tArr);
}
#include "tests/TestKit.h"

#include <iostream>

using namespace saot::test;

bool TestSuite::displayResult() const {
  std::cerr << BOLDCYAN("Test Result of '" << name << "': ");
  if (failures.empty()) {
    std::cerr << BOLDGREEN(nTests << "/" << nTests << " Passed!\n")
              << GREEN("All tests passed!\n");
    return true;
  }

  unsigned nFailed = failures.size();
  std::cerr << BOLDYELLOW(nTests - nFailed << "/" << nTests << " Passed!\n");
  for (unsigned i = 0; i < nFailed; i++) {
    std::cerr << RED("Failure "
        << i << ": '" << failures[i].title << "' at " << failures[i].info
        << "\n  Reason: " << failures[i].reason << "\n");
  }
  return false;
}

void TestSuite::assertClose(
    float a, float b,
    const std::string& title, const std::string& info, float tol) {
  ++nTests;
  float diff = std::abs(a - b);
  if (diff <= tol)
    return;
  
  std::stringstream ss;
  ss << "LHS " << a << " and RHS " << b
      << " (Diff = " << diff << ") > (Tolerance " << tol << ")";
  failures.emplace_back(title, info, ss.str());
}

void TestSuite::assertClose(
    double a, double b,
    const std::string& title, const std::string& info, double tol) {
  ++nTests;
  double diff = std::abs(a - b);
  if (diff <= tol)
    return;
  
  std::stringstream ss;
  ss << "LHS " << a << ", RHS " << b
      << ", (Diff = " << diff << ") > (Tolerance = " << tol << ")";
  failures.emplace_back(title, info, ss.str());
}

void TestSuite::assertAllClose(
    const std::vector<double>& aVec, const std::vector<double>& bVec,
    const std::string& title, const std::string& info, double tol) {
  ++nTests;
  std::stringstream ss;
  auto size = aVec.size();
  if (size != bVec.size()) {
    ss << "LHS size " << size << " unmatches RHS size " << bVec.size();
    failures.emplace_back(title, info, ss.str());
    return;
  }

  std::vector<int> unmatchPos;
  for (int i = 0; i < size; i++) {
    if (std::abs(aVec[i] - bVec[i]) > tol)
      unmatchPos.push_back(i);
  }
  if (unmatchPos.empty())
    return;
  
  ss << "Unmatch positions: ";
  for (int i = 0, s = unmatchPos.size(); i < s-1; i++)
    ss << unmatchPos[i] << ",";
  ss << unmatchPos.back();
  failures.emplace_back(title, info, ss.str());
}

void TestSuite::assertAllClose(
    const std::vector<float>& aVec, const std::vector<float>& bVec,
    const std::string& title, const std::string& info, float tol) {
  ++nTests;
  std::stringstream ss;
  auto size = aVec.size();
  if (size != bVec.size()) {
    ss << "LHS size " << size << " unmatches RHS size " << bVec.size();
    failures.emplace_back(title, info, ss.str());
    return;
  }

  std::vector<int> unmatchPos;
  for (int i = 0; i < size; i++) {
    if (std::abs(aVec[i] - bVec[i]) > tol)
      unmatchPos.push_back(i);
  }
  if (unmatchPos.empty())
    return;
  
  ss << "Unmatch positions: ";
  for (int i = 0, s = unmatchPos.size(); i < s-1; i++)
    ss << unmatchPos[i] << ",";
  ss << unmatchPos.back();
  failures.emplace_back(title, info, ss.str());
}
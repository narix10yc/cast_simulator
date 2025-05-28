#ifndef SIMULATION_DENSITYMATRIXCPU_H
#define SIMULATION_DENSITYMATRIXCPU_H

#include <cstring>

namespace utils {

template<typename ScalarType>
class DensityMatrixCPU {
  int _nQubits;
  int _nActiveBranches;
  double* _weights;
  ScalarType* _data;
public:
  DensityMatrixCPU(int nQubits) : _nQubits(nQubits) , _nActiveBranches(0) {
    _weights = static_cast<double*>(
      std::aligned_alloc(64, sizeof(double) * (1ULL << nQubits)));
    _data = static_cast<ScalarType*>(
      std::aligned_alloc(64, sizeInBytesActiveBranches()));
    assert(_weights != nullptr);
    assert(_data != nullptr);
  }

  ~DensityMatrixCPU() {
    std::free(_weights);
    std::free(_data);
  }

  DensityMatrixCPU(const DensityMatrixCPU&) = delete;
  DensityMatrixCPU(DensityMatrixCPU&&) = delete;
  DensityMatrixCPU& operator=(const DensityMatrixCPU&) = delete;
  DensityMatrixCPU& operator=(DensityMatrixCPU&&) = delete;

  int nQubits() const { return _nQubits; }
  int nActiveBranches() const { return _nActiveBranches; }

  size_t sizePerBranch() const { return 2ULL << _nQubits; }
  size_t sizeInBytesPerBranch() const {
    return sizePerBranch() * sizeof(ScalarType);
  }

  size_t sizeActiveBranches() const {
    return _nActiveBranches * sizePerBranch();
  }
  size_t sizeInBytesActiveBranches() const {
    return sizeActiveBranches() * sizeof(ScalarType);
  }

  size_t size() const { return 2ULL << (2 * _nQubits); }
  size_t sizeInBytes() const { return size() * sizeof(ScalarType); }

  double weight(int branchIdx) const {
    assert(branchIdx >= 0 && branchIdx < _nActiveBranches);
    return _weights[branchIdx];
  }

  ScalarType* data(int branchIdx) {
    assert(branchIdx >= 0 && branchIdx < _nActiveBranches);
    return _data + branchIdx * sizePerBranch();
  }


}; // class DensityMatrixCPU

extern template class DensityMatrixCPU<float>;
extern template class DensityMatrixCPU<double>;

} // namespace utils

#endif // SIMULATION_DENSITYMATRIXCPU_H
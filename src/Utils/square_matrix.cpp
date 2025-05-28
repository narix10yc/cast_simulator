#include "utils/square_matrix.h"
#include "utils/utils.h"

#include <random>

using namespace utils;

utils::square_matrix<std::complex<double>>
utils::randomUnitaryMatrix(unsigned edgeSize) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, 1.0);

  square_matrix<std::complex<double>> matrix(edgeSize);
  for (unsigned r = 0; r < edgeSize; r++) {
    for (unsigned c = 0; c < edgeSize; c++)
      matrix(r, c) = { dist(gen), dist(gen) };

    // project
    for (unsigned rr = 0; rr < r; rr++) {
      auto coef = utils::inner_product(matrix.row(rr), matrix.row(r), edgeSize);
      // subtract row r by coef * row rr
      for (unsigned cc = 0; cc < edgeSize; cc++)
        matrix(r, cc) -= coef * matrix(rr, cc);
      assert(std::abs(utils::inner_product(matrix.row(r), matrix.row(rr), edgeSize)) < 1e-8);
    }

    // normalize
    utils::normalize(matrix.row(r), edgeSize);
  }
  return matrix;
}

std::ostream& utils::printComplexMatrixF64(
    std::ostream& os,
    const utils::square_matrix<std::complex<double>>& matrix) {
  if (matrix.edgeSize() == 0)
    return os << "[]\n";
  if (matrix.edgeSize() == 1)
    return utils::print_complex(os << "[", matrix(0, 0)) << "]\n";

  // first (edgeSize - 1) rows
  os << "[";
  for (unsigned r = 0; r < matrix.edgeSize() - 1; r++) {
    for (unsigned c = 0; c < matrix.edgeSize(); c++)
      utils::print_complex(os, matrix(r, c)) << ", ";
    os << "\n ";
  }

  // last row
  for (unsigned c = 0; c < matrix.edgeSize(); c++)
    utils::print_complex(os, matrix(matrix.edgeSize() - 1, c)) << ", ";
  return os << " ]\n";
}
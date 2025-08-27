/* cast/Internal/PerfCacheHelper.h
 *
 * Contains a bunch of utility functions to be used in generating performance
 * cache and cost model
 */

#ifndef CAST_INTERNAL_PERFCACHEHELPER_H
#define CAST_INTERNAL_PERFCACHEHELPER_H

#include "cast/Core/Precision.h"
#include "cast/Core/QuantumGate.h"

namespace cast {
namespace internal {

/// @return Speed in gigabytes per second (GiBps)
double calculateMemUpdateSpeed(int nQubits, Precision precision, double t);


// Take the scalar gate matrix representation of the gate and randomly zero
// out some of the elements with probability p. The final matrix is not
// guaranteed to remain unitary. This method is only intended to be used for
// testing purposes.
// For \c StandardQuantumGate, it only applies to the gate
// matrix. For \c SuperopQuantumGate, it applies to the superoperator matrix
// (not implemented yet). This method does not apply completely random removal.
// It keeps the matrix valid, meaning non of the rows or columns will be
// completely zeroed out.
void randRemoveQuantumGate(QuantumGate* quGate, float p);

} // namespace internal
} // namespace cast

#endif // CAST_INTERNAL_PERFCACHEHELPER_H
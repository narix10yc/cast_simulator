#ifndef CAST_CORE_SCALARKIND_H
#define CAST_CORE_SCALARKIND_H

namespace cast {

enum ScalarKind : int {
  SK_Zero = 0,
  SK_One = 1,
  SK_MinusOne = -1,
  SK_General = 2,
  SK_ImmValue = 3,
  SK_Shared = 4,
  SK_SharedNeg = 5,
  SK_Unknown = 100,
}; // enum ScalarKind

}

#endif // CAST_CORE_SCALARKIND_H
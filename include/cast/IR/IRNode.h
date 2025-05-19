#ifndef CAST_IR_IRNODE_H
#define CAST_IR_IRNODE_H

#include <iostream>

namespace cast::ir {

/// @brief Base class for all CAST IR nodes.
class IRNode {
public:
  virtual ~IRNode() = default;

  virtual std::ostream& print(std::ostream& os) const {
    return os << "IRNode @ " << this;
  }
};

} // namespace cast::ir

#endif // CAST_IR_IRNODE_H
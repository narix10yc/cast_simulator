#ifndef CAST_NEW_PARSER_PRETTY_PRINTER_H
#define CAST_NEW_PARSER_PRETTY_PRINTER_H

#include <iostream>
#include <vector>
#include <cassert>

namespace cast {
namespace ast {

class PrettyPrinter {
public:
  std::ostream& os;
  std::vector<int> states;
  std::string prefix;

  PrettyPrinter(std::ostream& os) : os(os), prefix() {
    states.reserve(8);
  }

  void setState(int idx, int value) {
    assert(idx >= 0);
    if (idx >= states.size())
      states.resize(idx + 1);
    assert(states[idx] == 0 && "Trying to set a non-zero state");
    states[idx] = value;
  }

  PrettyPrinter& setPrefix(const std::string& prefix) {
    this->prefix = prefix;
    return *this;
  }

  // prefix will be cleared after write. 
  std::ostream& write(int indent) {
    assert(indent >= 0);
    if (indent == 0)
      return os;
    // col [0, indent - 2]
    for (unsigned i = 0; i < indent - 1; ++i) {
      if (states[i] == 0)
        os << "  ";
      else
        os << "| ";
    }
    // col indent - 1
    if (states[indent - 1] == 0)
      os << "  ";
    else if (states[indent - 1] == 1) {
      states[indent - 1] = 0;
      os << "`-";
    }
    else {
      --states[indent - 1];
      os << "|-";
    }
    if (!prefix.empty()) {
      // Print the prefix in purple
      os << "\033[35m" << prefix << ": \033[0m";
      prefix.clear();
    }
    return os;
  }
  
}; // class PrettyPrinter
  
}; // namespace ast

}; // namespace cast


#endif // CAST_NEW_PARSER_PRETTY_PRINTER_H
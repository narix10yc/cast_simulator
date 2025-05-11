#ifndef CAST_NEW_PARSER_PRETTY_PRINTER_H
#define CAST_NEW_PARSER_PRETTY_PRINTER_H

#include <iostream>
#include <vector>
#include <cassert>

namespace cast::draft {

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

  void setPrefix(const std::string& prefix) {
    this->prefix = prefix;
  }

  // prefix will be cleared after write. 
  std::ostream& write(int indent) {
    assert(indent <= states.size());
    assert(indent >= 0);
    if (indent == 0)
      return os;
    for (unsigned i = 0; i < indent - 1; ++i) {
      if (states[i] == 0)
        os << "  ";
      else
        os << "| ";
    }
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
    os << prefix;
    prefix.clear();
    return os;
  }
  
}; // class PrettyPrinter
  
}; // namespace ast

}; // namespace cast::draft


#endif // CAST_NEW_PARSER_PRETTY_PRINTER_H
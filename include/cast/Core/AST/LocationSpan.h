#ifndef CAST_NEW_PARSER_LOCATION_SPAN_H
#define CAST_NEW_PARSER_LOCATION_SPAN_H

#include <cassert>

namespace cast {

struct LocationSpan {
  const char* begin;
  const char* end;
  
  LocationSpan(const char* begin, const char* end)
    : begin(begin), end(end) { assert(begin <= end); }
}; // sturct LocationSpan

}; // namespace cast

#endif // CAST_NEW_PARSER_LOCATION_SPAN_H
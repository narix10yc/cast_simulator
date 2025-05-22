FILE(GLOB SRC_FILES "ADT/*.cpp" "Transform/*.cpp" "IR/*.cpp")
add_library(adt STATIC ${SRC_FILES})
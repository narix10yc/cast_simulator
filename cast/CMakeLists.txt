file(GLOB SRC_FILES "*.cpp")
add_library(cast ${SRC_FILES})
target_link_libraries(cast utils adt)
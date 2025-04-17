file(GLOB ALL_SRC_FILES "*.cpp")

if (NOT CAST_USE_CUDA)
  list(REMOVE_ITEM ALL_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/testStatevectorCUDA.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/testCudaU.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/testCudaRzParam.cpp")
endif()

add_library(TestKit STATIC ${ALL_SRC_FILES})
target_link_libraries(TestKit timeit)
target_compile_definitions(TestKit PRIVATE
  TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

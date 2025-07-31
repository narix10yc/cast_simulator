file(GLOB CORE_SRC_FILES "Core/*.cpp")
file(GLOB CPU_SRC_FILES "CPU/*.cpp")
file(GLOB CUDA_SRC_FILES "CUDA/*.cpp")
file(GLOB FPGA_SRC_FILES "FPGA/*.cpp")

set(ALL_SRC_FILES ${CORE_SRC_FILES} ${CPU_SRC_FILES} ${FPGA_SRC_FILES})
if(CAST_USE_CUDA)
  list(APPEND ALL_SRC_FILES ${CUDA_SRC_FILES})
endif()

add_executable(unit_test "test_main.cpp" "TestKit.cpp" ${ALL_SRC_FILES})
target_link_libraries(unit_test cast timeit)

target_compile_definitions(unit_test PRIVATE
    TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

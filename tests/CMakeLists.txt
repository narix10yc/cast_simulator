add_library(cast_testkit STATIC "TestKit.cpp")
target_link_libraries(cast_testkit PUBLIC cast timeit)

function(add_cast_unit_test target_name labels)
  add_executable(${target_name} ${ARGN})
  target_link_libraries(${target_name} PRIVATE cast_testkit)
  target_compile_definitions(${target_name} PRIVATE
      TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

  if(BUILD_TESTING)
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES LABELS "${labels}")
  endif()
endfunction()

set(UNIT_TEST_TARGETS)

add_cast_unit_test(
  unit_core
  "core"
  "Core/testComplexSquareMatrix.cpp"
  "Core/testQuantumGate.cpp"
  "Core/testQuantumChannel.cpp"
  "main/unit_core_main.cpp")
list(APPEND UNIT_TEST_TARGETS unit_core)

add_cast_unit_test(
  unit_cpu
  "cpu"
  "CPU/testCpuH.cpp"
  "CPU/testCpuU.cpp"
  "CPU/testFusionCPU.cpp"
  "main/unit_cpu_main.cpp")
list(APPEND UNIT_TEST_TARGETS unit_cpu)

add_cast_unit_test(
  unit_fpga
  "fpga"
  "FPGA/testFpgaGateCategory.cpp"
  "main/unit_fpga_main.cpp")
list(APPEND UNIT_TEST_TARGETS unit_fpga)

if(CAST_USE_CUDA)
  add_cast_unit_test(
    unit_cuda
    "cuda"
    "CUDA/testStatevectorCUDA.cpp"
    "CUDA/testCudaU.cpp"
    "CUDA/testCudaU2.cpp"
    "main/unit_cuda_main.cpp")
  list(APPEND UNIT_TEST_TARGETS unit_cuda)
endif()

add_custom_target(unit_test DEPENDS ${UNIT_TEST_TARGETS})

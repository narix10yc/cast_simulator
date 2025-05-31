add_executable(unit_test "test_main.cpp")
target_link_libraries(unit_test cast TestKit)

add_executable(qasm_to_cast "qasm_to_cast.cpp")

add_executable(qasm_to_qch "qasm_to_qch.cpp")

add_executable(kernel "kernel.cpp")

add_executable(scratch "scratch.cpp")
target_link_libraries(scratch cast)

add_executable(fpga_playground "fpga_playground.cpp")

add_executable(fpga_vary_buffer "fpga_vary_buffer.cpp")

add_executable(cost_model "cost_model.cpp")
target_link_libraries(cost_model cast)

add_executable(blank "blank.cpp")

if (CAST_USE_CUDA)
  add_executable(cuda_playground "cuda_playground.cu")
  add_executable(cost_model_cuda "cost_model_cuda.cpp")
endif()

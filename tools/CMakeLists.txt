add_executable(unit_test "test_main.cpp")
target_link_libraries(unit_test cast TestKit)

add_executable(qasm_to_cast "qasm_to_cast.cpp")

add_executable(qasm_to_qch "qasm_to_qch.cpp")

add_executable(draft_parser "draft_parser.cpp")
target_link_libraries(draft_parser cast)

add_executable(scratch "scratch.cpp")
target_link_libraries(scratch cast)

add_executable(fpga_playground "fpga_playground.cpp")

add_executable(fpga_vary_buffer "fpga_vary_buffer.cpp")

add_executable(cpu_cost_model "cpu_cost_model.cpp")
target_link_libraries(cpu_cost_model cast)

set_source_files_properties("blank.cpp" PROPERTIES COMPILE_FLAGS "-S -emit-llvm")
add_executable(blank "blank.cpp")
# set_target_properties(blank PROPERTIES EXCLUDE_FROM_ALL TRUE)

if (CAST_USE_CUDA)
  add_executable(cuda_playground "cuda_playground.cu")
  add_executable(cost_model_cuda "cost_model_cuda.cpp")
endif()

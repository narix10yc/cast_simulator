add_executable(qasm_to_cast "qasm_to_cast.cpp")

add_executable(qasm_to_qch "qasm_to_qch.cpp")

add_executable(draft_parser "draft_parser.cpp")
target_link_libraries(draft_parser cast)

add_executable(scratch "scratch.cpp")
target_link_libraries(scratch cast)

add_executable(cpu_cost_model "cpu_cost_model.cpp")
target_link_libraries(cpu_cost_model cast)

if (CAST_USE_CUDA)
  add_executable(cuda_cost_model "cuda_cost_model.cpp")
  target_link_libraries(cuda_cost_model cast)
endif()

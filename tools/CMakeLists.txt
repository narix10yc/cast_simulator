add_executable(scratch "scratch.cpp")
target_link_libraries(scratch cast)

add_executable(cpu_cost_model "cpu_cost_model.cpp")
target_link_libraries(cpu_cost_model cast)

if (CAST_USE_CUDA)
  add_executable(cuda_cost_model "cuda_cost_model.cpp")
  target_link_libraries(cuda_cost_model cast)
endif()

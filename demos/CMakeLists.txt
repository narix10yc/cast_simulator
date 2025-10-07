add_executable(demo_cpu_end_to_end "demo_cpu_end_to_end.cpp")
target_link_libraries(demo_cpu_end_to_end PRIVATE cast)

add_executable(demo_cpu_fusion "demo_cpu_fusion.cpp")
target_link_libraries(demo_cpu_fusion PRIVATE cast)

add_executable(cpu_bcmk "cpu_bcmk.cpp")
target_link_libraries(cpu_bcmk PRIVATE cast)

if (CAST_USE_CUDA)
  add_executable(demo_cuda_fusion "demo_cuda_fusion.cpp")
  target_link_libraries(demo_cuda_fusion PRIVATE cast)

  add_executable(demo_cuda_end_to_end "demo_cuda_end_to_end.cpp")
  target_link_libraries(demo_cuda_end_to_end PRIVATE cast)
endif()

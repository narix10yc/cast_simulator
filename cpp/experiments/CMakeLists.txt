add_executable(experi_cpu_profile "experi_cpu_profile.cpp")
target_link_libraries(experi_cpu_profile PRIVATE cast)

if(CAST_USE_CUDA)
  add_executable(experi_cuda_profile "experi_cuda_profile.cpp")
  target_link_libraries(experi_cuda_profile PRIVATE cast)
endif()

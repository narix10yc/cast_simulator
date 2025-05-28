add_executable(demo_fusion "demo_fusion.cpp")

add_executable(cpu_bcmk "cpu_bcmk.cpp")
target_link_libraries(cpu_bcmk PRIVATE cast)

add_executable(demo_kernel_performance "demo_kernel_performance.cpp")

if (CAST_USE_CUDA)
  add_executable(demo_ptx "demo_ptx.cpp")
endif()

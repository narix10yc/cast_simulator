if (CAST_USE_CUDA)
  add_executable(demo_ptx "demo_ptx.cpp")
  target_link_libraries(demo_ptx utils qasm cast simulation timeit)
endif()

add_executable(demo_fusion "demo_fusion.cpp")
target_link_libraries(demo_fusion utils qasm cast simulation timeit)

add_executable(cpu_bcmk "cpu_bcmk.cpp")
target_link_libraries(cpu_bcmk adt new_parser qasm utils cast simulation timeit)

add_executable(demo_kernel_performance "demo_kernel_performance.cpp")
target_link_libraries(demo_kernel_performance utils cast simulation timeit)
if (CAST_USE_CUDA)
  add_executable(bcmk_CudaKernGen "bcmk_CudaKernGen.cpp")
  target_link_libraries(bcmk_CudaKernGen utils qasm cast simulation timeit)

  add_executable(bcmk_CudaKernExec "bcmk_CudaKernExec.cpp")
  target_link_libraries(bcmk_CudaKernExec utils qasm cast simulation timeit)

  add_executable(bcmk_CudaJit "bcmk_CudaJit.cpp")
  target_link_libraries(bcmk_CudaJit utils qasm cast simulation timeit)

  add_executable(simple_bcmk "simple_bcmk.cpp")
  target_link_libraries(simple_bcmk utils qasm cast simulation timeit)
endif()
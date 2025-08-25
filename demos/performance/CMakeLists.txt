if (CAST_USE_CUDA)
  # add_executable(bcmk_CudaKernGen "bcmk_CudaKernGen.cpp")
  # target_link_libraries(bcmk_CudaKernGen utils qasm cast simulation timeit)

  # add_executable(bcmk_CudaKernExec "bcmk_CudaKernExec.cpp")
  # target_link_libraries(bcmk_CudaKernExec utils qasm cast simulation timeit)

  # add_executable(bcmk_CudaJit "bcmk_CudaJit.cpp")
  # target_link_libraries(bcmk_CudaJit utils qasm cast simulation timeit)

  # add_executable(cuda_bcmk_default "cuda_bcmk_default.cpp")
  # target_link_libraries(cuda_bcmk_default utils qasm cast simulation timeit)

  add_executable(cuda_bcmk_imm "cuda_bcmk_imm.cpp")
  target_link_libraries(cuda_bcmk_imm utils qasm cast simulation timeit)

  # add_executable(cuda_bcmk_imm_share "cuda_bcmk_imm_share.cpp")
  # target_link_libraries(cuda_bcmk_imm_share utils qasm cast simulation timeit)

  # add_executable(bcmk_noise "bcmk_noise.cpp")
  # target_link_libraries(bcmk_noise utils qasm cast simulation timeit)
endif()
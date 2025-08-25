add_executable(demo_fusion "demo_fusion.cpp")
target_link_libraries(demo_fusion PRIVATE cast)

add_executable(demo_opt "demo_opt.cpp")
target_link_libraries(demo_opt PRIVATE cast)

add_executable(cpu_bcmk "cpu_bcmk.cpp")
target_link_libraries(cpu_bcmk PRIVATE cast)

add_executable(demo_asm_dump "demo_asm_dump.cpp")
target_link_libraries(demo_asm_dump PRIVATE cast)

# add_executable(bcmk_noise "performance/bcmk_noise.cpp")
# target_link_libraries(bcmk_noise PRIVATE cast)

if (CAST_USE_CUDA)
  add_executable(demo_ptx "demo_ptx.cpp")

  add_executable(cuda_bcmk_imm "performance/cuda_bcmk_imm.cpp")
  target_compile_options(cuda_bcmk_imm PRIVATE "-O3" "-march=native")
  target_link_libraries(cuda_bcmk_imm PRIVATE cast)

  add_executable(cuda_imm_cpu_fusion "performance/cuda_imm_cpu_fusion.cpp")
  target_compile_options(cuda_imm_cpu_fusion PRIVATE "-O3" "-march=native")
  target_link_libraries(cuda_imm_cpu_fusion PRIVATE cast)

  add_executable(cuda_imm_permute "performance/cuda_imm_permute.cpp")
  target_compile_options(cuda_imm_permute PRIVATE "-O3" "-march=native")
  target_link_libraries(cuda_imm_permute PRIVATE cast)
endif()

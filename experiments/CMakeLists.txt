add_executable(exp_simd "exp_simd.cpp")
target_compile_options(exp_simd PRIVATE "-O3" "-march=native")
target_link_libraries(exp_simd PRIVATE cast)

add_executable(exp_single_gate "exp_single_gate.cpp")
target_compile_options(exp_single_gate PRIVATE "-O3" "-march=native")
target_link_libraries(exp_single_gate PRIVATE cast)
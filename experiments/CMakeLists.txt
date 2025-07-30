add_executable(exp_simd "exp_simd.cpp")
target_compile_options(exp_simd PRIVATE "-O3" "-march=native")
target_link_libraries(exp_simd PRIVATE cast)

add_executable(experiment "experiment.cpp")
add_executable(demo_ptx "demo_ptx.cpp")
target_link_libraries(demo_ptx utils qasm cast simulation timeit)

add_executable(demo_fusion "demo_fusion.cpp")
target_link_libraries(demo_fusion utils qasm cast simulation timeit)

add_executable(cpu_bcmk "cpu_bcmk.cpp")
target_link_libraries(cpu_bcmk utils cast simulation timeit)
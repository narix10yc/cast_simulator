set(COMPILE_FLAGS "-fno-exceptions")
if(CAST_PYTHON_BIND)
  list(APPEND COMPILE_FLAGS "-frtti")
else()
  list(APPEND COMPILE_FLAGS "-fno-rtti")
endif()

file(GLOB_RECURSE SRC_FILES "Timeit/*.cpp")
add_library(timeit STATIC ${SRC_FILES})
target_compile_options(timeit PRIVATE -O3 -march=native)

file(GLOB_RECURSE SRC_FILES "OpenQASM/*.cpp")
add_library(openqasm STATIC ${SRC_FILES})
target_compile_options(openqasm PRIVATE ${COMPILE_FLAGS})

file(GLOB_RECURSE SRC_FILES "Utils/*.cpp")
add_library(cast_utils STATIC ${SRC_FILES})
target_compile_options(cast_utils PRIVATE ${COMPILE_FLAGS})
target_link_libraries(cast_utils PUBLIC timeit)

file(GLOB_RECURSE SRC_FILES "Core/*.cpp")
add_library(cast_core STATIC ${SRC_FILES})
target_compile_options(cast_core PRIVATE ${COMPILE_FLAGS})
target_link_libraries(cast_core PUBLIC cast_utils openqasm timeit)

# Grab LLVM libraries
set(LLVM_COMPONENTS
  Support
  Core
  IRReader
  OrcJIT
  native
)
if (CAST_USE_CUDA)
  list(APPEND LLVM_COMPONENTS NVPTX)
endif()
llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_COMPONENTS})
target_link_libraries(cast_core PUBLIC ${LLVM_LIBS})


file(GLOB_RECURSE SRC_FILES "CPU/*.cpp")
add_library(cast_cpu STATIC ${SRC_FILES})
target_compile_options(cast_cpu PRIVATE ${COMPILE_FLAGS})
target_link_libraries(cast_cpu PUBLIC cast_core)

file(GLOB_RECURSE SRC_FILES "FPGA/*.cpp")
add_library(cast_fpga STATIC ${SRC_FILES})
target_compile_options(cast_fpga PRIVATE ${COMPILE_FLAGS})
target_link_libraries(cast_fpga PUBLIC cast_core)


add_library(cast INTERFACE)

if (CAST_USE_CUDA)
  file(GLOB_RECURSE SRC_FILES "CUDA/*.cpp" "CUDA/*.cu")
  add_library(cast_cuda STATIC ${SRC_FILES})
  target_link_libraries(cast_cuda cuda CUDA::cudart cast_core)
  target_compile_options(cast_cuda PRIVATE ${COMPILE_FLAGS})
  target_link_libraries(cast INTERFACE cast_cuda)
endif()

target_link_libraries(cast INTERFACE cast_cpu cast_fpga)
file(GLOB_RECURSE SRC_FILES "Timeit/*.cpp")
add_library(timeit STATIC ${SRC_FILES})
target_compile_options(timeit PRIVATE -O3 -march=native)

file(GLOB_RECURSE SRC_FILES "OpenQASM/*.cpp")
add_library(openqasm STATIC ${SRC_FILES})
target_compile_options(openqasm PRIVATE -fno-exceptions -fno-rtti)

file(GLOB_RECURSE SRC_FILES "Core/*.cpp")
add_library(cast_core STATIC ${SRC_FILES})
target_compile_options(cast_core PRIVATE -fno-exceptions -fno-rtti)

file(GLOB_RECURSE SRC_FILES "CPU/*.cpp")
add_library(cast_cpu STATIC ${SRC_FILES})
target_compile_options(cast_cpu PRIVATE -fno-exceptions -fno-rtti)

file(GLOB_RECURSE SRC_FILES "FPGA/*.cpp")
add_library(cast_fpga STATIC ${SRC_FILES})
target_compile_options(cast_fpga PRIVATE -fno-exceptions -fno-rtti)

file(GLOB_RECURSE SRC_FILES "Utils/*.cpp")
add_library(cast_utils STATIC ${SRC_FILES})
target_compile_options(cast_utils PRIVATE -fno-exceptions -fno-rtti)

add_library(cast INTERFACE)

# Grab LLVM libraries
set(LLVM_COMPONENTS
  Support
  Core
  IRReader
  OrcJIT
  Native
)

if (CAST_USE_CUDA)
  file(GLOB_RECURSE SRC_FILES "CUDA/*.cpp" "CUDA/*.cu")
  add_library(cast_cuda STATIC ${SRC_FILES})
  target_link_libraries(cast_cuda cuda CUDA::cudart)
  target_compile_options(cast_cuda PRIVATE -fno-exceptions -fno-rtti)
  target_link_libraries(cast INTERFACE cast_cuda)
  list(APPEND LLVM_COMPONENTS NVPTX)
endif()

llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_COMPONENTS})

target_link_libraries(cast INTERFACE
  timeit
  openqasm
  cast_core
  cast_cpu
  cast_fpga
  cast_utils
  ${LLVM_LIBS}
)
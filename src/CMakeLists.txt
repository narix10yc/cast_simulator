FILE(GLOB_RECURSE SRC_FILES "Timeit/*.cpp")
add_library(timeit STATIC ${SRC_FILES})
target_compile_options(timeit PRIVATE -O3 -march=native)

FILE(GLOB_RECURSE SRC_FILES "OpenQASM/*.cpp")
add_library(openqasm STATIC ${SRC_FILES})

FILE(GLOB_RECURSE SRC_FILES "Core/*.cpp")
add_library(cast_core STATIC ${SRC_FILES})

FILE(GLOB_RECURSE SRC_FILES "CPU/*.cpp")
add_library(cast_cpu STATIC ${SRC_FILES})

FILE(GLOB_RECURSE SRC_FILES "Legacy/*.cpp")
add_library(cast_legacy STATIC ${SRC_FILES})

FILE(GLOB_RECURSE SRC_FILES "Utils/*.cpp")
add_library(cast_utils STATIC ${SRC_FILES})

add_library(cast INTERFACE)
# Grab LLVM libraries
llvm_map_components_to_libnames(LLVM_LIBS
  support core irreader orcjit ${LLVM_TARGETS_TO_BUILD})

target_link_libraries(cast INTERFACE
  timeit
  openqasm
  cast_core
  cast_cpu
  cast_utils
  cast_legacy
  ${LLVM_LIBS}
)

if (CAST_USE_CUDA)
  FILE(GLOB_RECURSE SRC_FILES "CUDA/*.cpp" "CUDA/*.cu")
  add_library(cast_cuda STATIC ${SRC_FILES})
  target_link_libraries(cast_cuda cuda CUDA::cudart)
  target_link_libraries(cast INTERFACE cast_cuda)
endif()
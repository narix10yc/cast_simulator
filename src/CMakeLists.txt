# if (CAST_USE_CUDA OR CAST_PYTHON_BIND)
#   set(RTTI "-frtti")
# else()
#   set(RTTI "-fno-rtti")
# endif()

# just use RTTI... Performance impact is negligible.
set(RTTI "-frtti")

file(GLOB_RECURSE SRC_FILES "Timeit/*.cpp")
add_library(timeit STATIC ${SRC_FILES})
target_compile_options(timeit PRIVATE -O3 -march=native -fno-exceptions ${RTTI})

file(GLOB_RECURSE SRC_FILES "OpenQASM/*.cpp")
add_library(openqasm STATIC ${SRC_FILES})
target_compile_options(openqasm PRIVATE -fno-exceptions ${RTTI})

file(GLOB_RECURSE SRC_FILES "Utils/*.cpp")
add_library(cast_utils STATIC ${SRC_FILES})
target_compile_options(cast_utils PRIVATE -fno-exceptions ${RTTI})
target_link_libraries(cast_utils PUBLIC timeit)

file(GLOB_RECURSE SRC_FILES "Core/*.cpp")
add_library(cast_core STATIC ${SRC_FILES})
target_compile_options(cast_core PRIVATE -fno-exceptions ${RTTI})
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

# cast_cpu
file(GLOB_RECURSE SRC_FILES "CPU/*.cpp")
add_library(cast_cpu STATIC ${SRC_FILES})
target_compile_options(cast_cpu PRIVATE ${COMPILE_FLAGS})
target_link_libraries(cast_cpu PUBLIC cast_core)

# cast_fpga
file(GLOB_RECURSE SRC_FILES "FPGA/*.cpp")
add_library(cast_fpga STATIC ${SRC_FILES})
target_compile_options(cast_fpga PRIVATE ${COMPILE_FLAGS})
target_link_libraries(cast_fpga PUBLIC cast_core)


# cast (interface library)
add_library(cast INTERFACE)

# cast_cuda (if enabled)
if (CAST_USE_CUDA)
  file(GLOB_RECURSE SRC_FILES "CUDA/*.cpp" "CUDA/*.cu")
  add_library(cast_cuda STATIC ${SRC_FILES})
  target_link_libraries(cast_cuda cuda CUDA::cudart cast_core)
  target_compile_options(cast_cuda PRIVATE ${COMPILE_FLAGS})
  target_link_libraries(cast INTERFACE cast_cuda)

  if(CAST_USE_NCCL)
    target_link_libraries(cast_cuda PRIVATE ${NCCL_LIBRARIES})
  endif()
endif()

target_link_libraries(cast INTERFACE cast_cpu cast_fpga)
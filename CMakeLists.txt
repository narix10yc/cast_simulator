cmake_minimum_required(VERSION 3.20)

##### Handle LLVM Dependency ######
include(scripts/Utils.cmake)
handle_llvm_dep()

##### User Options ######
message("### User Options ###")
option(CAST_USE_LLVM_COMPILER "Use LLVM Clang/Clang++ as compilers" OFF)
message(STATUS "CAST_USE_LLVM_COMPILER: ${CAST_USE_LLVM_COMPILER}")
option(CAST_USE_LLVM_LIBCXX "Use LLVM libc++ as the standard C++ library" OFF)
message(STATUS "CAST_USE_LLVM_LIBCXX: ${CAST_USE_LLVM_LIBCXX}")
option(CAST_USE_CUDA "Use LLVM NVPTX Backend for CUDA Support" OFF)
message(STATUS "CAST_USE_CUDA: ${CAST_USE_CUDA}")
option(CAST_PYTHON_BIND "Bind Python" OFF)
message(STATUS "CAST_PYTHON_BIND: ${CAST_PYTHON_BIND}")
message("#######################")

##### Compiler Setup ######
if(CAST_USE_LLVM_COMPILER)
  use_llvm_compilers()
else()
  message(STATUS "Using system compilers.")
endif()
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

##### Project Setup ######
project(cast_simulator LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if(CAST_USE_LLVM_LIBCXX)
  use_llvm_libcxx()
endif()

if(CAST_USE_CUDA)
  add_compile_definitions(CAST_USE_CUDA)
endif()

# Find LLVM
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
if(CAST_LLVM_DEBUG_ROOT AND CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
  set(LLVM_DIR "${CAST_LLVM_DEBUG_ROOT}/lib/cmake/llvm")
else()
  set(LLVM_DIR "${CAST_LLVM_ROOT}/lib/cmake/llvm")
endif()

message(STATUS "LLVM_DIR is set to: ${LLVM_DIR}")
find_package(LLVM CONFIG REQUIRED)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM is built for targets: ${LLVM_TARGETS_TO_BUILD}")

if(CAST_USE_CUDA AND NOT "NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(WARNING
    "CUDA is enabled in the project, "
    "but LLVM build does not support NVPTX backend.")
endif()

###### CUDA Setup ######

# Find CUDAToolkit (if CAST_USE_CUDA is defined)
if(CAST_USE_CUDA)
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDAToolkit_INCLUDE_DIR: ${CUDAToolkit_INCLUDE_DIR}")
  message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
  message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")

  # FindCUDAToolkit.cmake does not automatically adjust include and link
  # directories. For safely we append it before other include paths.
  # Otherwise, enviroment managers such as conda may interfere with the
  # selected CUDAToolkit version.
  include_directories(BEFORE ${CUDAToolkit_INCLUDE_DIR})
  include_directories(BEFORE ${CUDAToolkit_INCLUDE_DIRS})
  link_directories(BEFORE ${CUDAToolkit_LIBRARY_DIR})
  link_directories(BEFORE ${CUDAToolkit_LIBRARY_DIRS})
  message(STATUS "CUDAToolkit_rt_LIBRARY: ${CUDAToolkit_rt_LIBRARY}")

  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    if(_variableName MATCHES "^CUDAToolkit")
      message(STATUS "${_variableName}=${${_variableName}}")
    endif()
  endforeach()

  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  set(CMAKE_CUDA_ARCHITECTURES native)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
  # Sometimes clang version is too high and nvcc does not support it
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
  enable_language(CUDA)
endif() # if(CAST_USE_CUDA)

# show colored diagnostics in the terminal
add_compile_options("-fdiagnostics-color=always")

if(CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
  add_compile_options("-Wall")
endif()
include_directories(${LLVM_INCLUDE_DIRS} include)
message(STATUS "LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
link_directories(${LLVM_LIBRARY_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# Enable PIC for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(demos)
add_subdirectory(tools)
add_subdirectory(experiments)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/drafts")
  add_subdirectory(drafts)
endif()

if(CAST_PYTHON_BIND)
  message(STATUS "Building Python bindings...")
  add_subdirectory(python)
endif()

cmake_minimum_required(VERSION 3.20)

project(cast_simulator LANGUAGES C CXX)
option(CAST_USE_CUDA "Use LLVM NVPTX Backend for CUDA Support" OFF)
if(CAST_USE_CUDA)
  add_compile_definitions(CAST_USE_CUDA)
endif()

message("############ Custom Setups ############")
set(CAST_LLVM_ROOT "" CACHE PATH "Path to the llvm root directory")

if(NOT CAST_LLVM_ROOT)
  if(DEFINED ENV{cast_llvm_root})
    set(CAST_LLVM_ROOT "$ENV{cast_llvm_root}")
  elseif(DEFINED ENV{CAST_LLVM_ROOT})
    set(CAST_LLVM_ROOT "$ENV{CAST_LLVM_ROOT}")
  endif()
endif()

message(STATUS "CAST_LLVM_ROOT: ${CAST_LLVM_ROOT}")
set(LLVM_RELEASE_INSTALL_ROOT "${CAST_LLVM_ROOT}/release-install")
set(LLVM_DEBUG_INSTALL_ROOT "${CAST_LLVM_ROOT}/debug-install")

# check if clang is installed along with LLVM
if(EXISTS "${LLVM_RELEASE_INSTALL_ROOT}/bin/clang" AND
   EXISTS "${LLVM_RELEASE_INSTALL_ROOT}/bin/clang++")
  message(STATUS "Using compilers under LLVM release-install at: "
                 "${LLVM_RELEASE_INSTALL_ROOT}")
  set(CMAKE_C_COMPILER "${LLVM_RELEASE_INSTALL_ROOT}/bin/clang")
  set(CMAKE_CXX_COMPILER "${LLVM_RELEASE_INSTALL_ROOT}/bin/clang++")
else()
  message(STATUS "No clang install found along with LLVM release-install. "
                 "Using system compilers.")
endif()
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# check if libc++ is installed along with LLVM
set(LIBCXX_INCLUDE_DIR "${LLVM_RELEASE_INSTALL_ROOT}/include/c++/v1")
set(LIBCXX_LIB_DIR "${LLVM_RELEASE_INSTALL_ROOT}/lib")
if(EXISTS "${LIBCXX_INCLUDE_DIR}/vector" AND
   EXISTS "${LIBCXX_LIB_DIR}/libc++.a")
  message(STATUS "Found libc++ at ${LLVM_RELEASE_INSTALL_ROOT}")
  add_compile_options(-stdlib=libc++ -I"${LIBCXX_INCLUDE_DIR}")
  if (NOT APPLE)
    # On macOS, -stdlib=libc++ automatically links libc++, libc++abi, and
    # unwind. Specifying them will cause annoying warnings.
    link_libraries(c++ c++abi unwind)
  endif()
  link_directories("${LIBCXX_LIB_DIR}")
else()
  message(STATUS "No libc++ found along with LLVM release-install, "
                 "using system defaults.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Find LLVM
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
if(CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
  set(LLVM_DIR "${LLVM_DEBUG_INSTALL_ROOT}/lib/cmake/llvm")
else()
  set(LLVM_DIR "${LLVM_RELEASE_INSTALL_ROOT}/lib/cmake/llvm")
endif()

message("############ End Custom Setups ############")

find_package(LLVM CONFIG)
if(NOT LLVM_FOUND)
  message(FATAL_ERROR
    "LLVM not found! CAST_LLVM_ROOT is currently set to '${CAST_LLVM_ROOT}'.\n"
    "We expect there are two sub-directories inside it: "
    "release-install and debug-install.")
endif()
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM is built for targets: ${LLVM_TARGETS_TO_BUILD}")

if(CAST_USE_CUDA AND NOT "NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(WARNING
    "CUDA is enabled in the project, "
    "but LLVM build does not support NVPTX backend.")
endif()

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
  foreach (_variableName ${_variableNames})
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

include_directories(${LLVM_INCLUDE_DIRS} include)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(demos)
add_subdirectory(tools)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/drafts")
  add_subdirectory(drafts)
endif()
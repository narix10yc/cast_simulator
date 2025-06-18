cmake_minimum_required(VERSION 3.20)

project(cast_simulator LANGUAGES C CXX)
option(CAST_USE_CUDA "Use LLVM NVPTX Backend for CUDA Support" OFF)
if(CAST_USE_CUDA)
  add_compile_definitions(CAST_USE_CUDA)
endif()
option(CAST_PYTHON_BIND "Bind Python" OFF)

message("############ Custom Setups ############")
# There are two modes: 
# 1. Enviroment variable CAST_LLVM_ROOT is set: We expect there are two
#    subdirectories under CAST_LLVM_ROOT: 'release-install' and 'debug-install'.
# 2. CAST_LLVM_RELEASE_ROOT: The root directory of the LLVM release-install.
#    In this case, we won't have debug infos in debug builds.
# If both CAST_LLVM_ROOT and CAST_LLVM_RELEASE_ROOT are defined, we issue a
# warning and use the second case.
set(CAST_LLVM_ROOT "" CACHE PATH "Path to the llvm root directory")
set(CAST_LLVM_RELEASE_ROOT "" CACHE PATH "Path to the llvm release-install directory")

if(NOT CAST_LLVM_ROOT)
  if(DEFINED ENV{CAST_LLVM_ROOT})
    set(CAST_LLVM_ROOT "$ENV{CAST_LLVM_ROOT}")
  elseif(DEFINED ENV{cast_llvm_root})
    set(CAST_LLVM_ROOT "$ENV{cast_llvm_root}")
  endif()
endif()

if(NOT CAST_LLVM_RELEASE_ROOT)
  if(DEFINED ENV{CAST_LLVM_RELEASE_ROOT})
    set(CAST_LLVM_RELEASE_ROOT "$ENV{CAST_LLVM_RELEASE_ROOT}")
  elseif(DEFINED ENV{cast_llvm_release_root})
    set(CAST_LLVM_RELEASE_ROOT "$ENV{cast_llvm_release_root}")
  endif()
endif()

if(CAST_LLVM_ROOT AND CAST_LLVM_RELEASE_ROOT)
  message(WARNING
    "Both CAST_LLVM_ROOT and CAST_LLVM_RELEASE_ROOT are defined. "
    "CAST_LLVM_RELEASE_ROOT will be used, ignoring CAST_LLVM_ROOT. "
    "So in debug builds, LLVM-related debug infos will not be available. ")
  set(CAST_LLVM_ROOT "")
endif()

if(NOT CAST_LLVM_ROOT AND NOT CAST_LLVM_RELEASE_ROOT)
  message(FATAL_ERROR
    "Neither CAST_LLVM_ROOT nor CAST_LLVM_RELEASE_ROOT is defined. "
    "Please set one of them to the root directory of your LLVM installation.")
endif()

if(CAST_LLVM_ROOT)
  message(STATUS "Using CAST_LLVM_ROOT: ${CAST_LLVM_ROOT}")
  set(CAST_LLVM_RELEASE_ROOT "${CAST_LLVM_ROOT}/release-install")
  set(CAST_LLVM_DEBUG_ROOT "${CAST_LLVM_ROOT}/debug-install")
  message(STATUS "CAST_LLVM_RELEASE_ROOT is set to: ${CAST_LLVM_RELEASE_ROOT}")
  message(STATUS "CAST_LLVM_DEBUG_ROOT is set to: ${CAST_LLVM_DEBUG_ROOT}")
  if(NOT EXISTS "${CAST_LLVM_RELEASE_ROOT}")
    message(FATAL_ERROR "CAST_LLVM_RELEASE_ROOT does not exist!")
  endif()
  if(NOT IS_DIRECTORY "${CAST_LLVM_RELEASE_ROOT}")
    message(FATAL_ERROR "CAST_LLVM_RELEASE_ROOT is not a directory!")
  endif()
  if(NOT EXISTS "${CAST_LLVM_DEBUG_ROOT}")
    message(FATAL_ERROR "CAST_LLVM_DEBUG_ROOT does not exist!")
  endif()
  if(NOT IS_DIRECTORY "${CAST_LLVM_DEBUG_ROOT}")
    message(FATAL_ERROR "CAST_LLVM_DEBUG_ROOT is not a directory!")
  endif()
else()
  # the second case (CAST_LLVM_RELEASE_ROOT is defined)
  message(STATUS "Using CAST_LLVM_RELEASE_ROOT: ${CAST_LLVM_RELEASE_ROOT}")
  message(STATUS "Notice that In debug builds, "
                 "LLVM-related debug infos will NOT be available.")
endif()

# check if clang is installed along with LLVM
if(EXISTS "${CAST_LLVM_RELEASE_ROOT}/bin/clang" AND
   EXISTS "${CAST_LLVM_RELEASE_ROOT}/bin/clang++")
  message(STATUS "Using compilers under LLVM release-install at: "
                 "${CAST_LLVM_RELEASE_ROOT}")
  set(CMAKE_C_COMPILER "${CAST_LLVM_RELEASE_ROOT}/bin/clang")
  set(CMAKE_CXX_COMPILER "${CAST_LLVM_RELEASE_ROOT}/bin/clang++")
  add_link_options("-fuse-ld=${CAST_LLVM_RELEASE_ROOT}/bin/ld.lld")
else()
  message(STATUS "No clang install found along with LLVM release-install. "
                 "Using system compilers.")
endif()
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# check if libc++ is installed along with LLVM
# include path is almost always
# ${CAST_LLVM_RELEASE_ROOT}/include/c++/v1
set(LIBCXX_INCLUDE_DIR "${CAST_LLVM_RELEASE_ROOT}/include/c++/v1")

# but the library path may vary. For example,
# ${CAST_LLVM_RELEASE_ROOT}/lib
# ${CAST_LLVM_RELEASE_ROOT}/lib/x86_64-unknown-linux-gnu
file(GLOB_RECURSE LIBCXX_LIB_PATHS "${CAST_LLVM_RELEASE_ROOT}/lib/*libc++.*")
if(LIBCXX_LIB_PATHS)
  list(GET LIBCXX_LIB_PATHS 0 LIBCXX_LIB_PATH)
  get_filename_component(LIBCXX_LIB_DIR "${LIBCXX_LIB_PATH}" DIRECTORY)
endif()

if(EXISTS "${LIBCXX_INCLUDE_DIR}/vector" AND
   DEFINED LIBCXX_LIB_DIR)
  message(STATUS "Found libc++ at ${LIBCXX_LIB_DIR}")
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
if(CAST_LLVM_DEBUG_ROOT AND CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
  set(LLVM_DIR "${CAST_LLVM_DEBUG_ROOT}/lib/cmake/llvm")
else()
  set(LLVM_DIR "${CAST_LLVM_RELEASE_ROOT}/lib/cmake/llvm")
endif()

message("############ End Custom Setups ############")

message(STATUS "LLVM_DIR is set to: ${LLVM_DIR}")
find_package(LLVM CONFIG)
if(NOT LLVM_FOUND)
  message(FATAL_ERROR "LLVM not found!")
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
message(STATUS "LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
link_directories(${LLVM_LIBRARY_DIRS})
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

if (CAST_PYTHON_BIND)
  message(STATUS "Building Python bindings...")
  add_subdirectory(python)
endif()
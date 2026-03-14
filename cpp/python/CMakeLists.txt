# CAST Python Bindings

# This will put the dependencies in a top-level _deps directory
# so if we delete the build directory, we don't need to re-download.
set(PYBIND11_FINDPYTHON ON)
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/_deps")
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.1
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python: ${Python3_EXECUTABLE}")
message(STATUS "Python includes: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python3_LIBRARIES}")

FetchContent_MakeAvailable(pybind11)

set(PY_PKG_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind)
set(PY_PKG_LIB_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#### Core library bindings
file(GLOB SRCS ${PY_PKG_SRC_DIR}/core/*.cpp)
pybind11_add_module(pybind_cast MODULE ${SRCS})

target_link_libraries(pybind_cast PRIVATE cast)
target_compile_options(pybind_cast PRIVATE "-fexceptions" "-frtti")

# We put the built library here: cast_root/python/cast/<name>.so 
# because it is along with the __init__.py and .pyi file.
# Most IDEs will be happier in this way.
set_target_properties(pybind_cast PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY ${PY_PKG_LIB_OUTPUT_DIR}/cast)

#### CPU-specific bindings
file(GLOB SRCS ${PY_PKG_SRC_DIR}/cpu/*.cpp)
pybind11_add_module(pybind_cast_cpu MODULE ${SRCS})

target_link_libraries(pybind_cast_cpu PRIVATE cast)
target_compile_options(pybind_cast_cpu PRIVATE "-fexceptions" "-frtti")
set_target_properties(pybind_cast_cpu PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY ${PY_PKG_LIB_OUTPUT_DIR}/cast)

#### CUDA-specific bindings
if(CAST_USE_CUDA)
  file(GLOB SRCS ${PY_PKG_SRC_DIR}/cuda/*.cpp)
  pybind11_add_module(pybind_cast_cuda MODULE ${SRCS})

  target_link_libraries(pybind_cast_cuda PRIVATE cast)
  target_compile_options(pybind_cast_cuda PRIVATE "-fexceptions" "-frtti")
  set_target_properties(pybind_cast_cuda PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${PY_PKG_LIB_OUTPUT_DIR}/cast)
endif()

### cmake target to bind all pythons
add_custom_target(pybind_all)
add_dependencies(pybind_all pybind_cast pybind_cast_cpu)
if(CAST_USE_CUDA)
  add_dependencies(pybind_all pybind_cast_cuda)
endif()

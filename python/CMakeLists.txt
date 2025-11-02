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

file(GLOB SRCS "bind/*.cpp")

pybind11_add_module(cast_python_bind MODULE ${SRCS})

target_link_libraries(cast_python_bind PRIVATE cast)
target_compile_options(cast_python_bind PRIVATE "-fexceptions" "-frtti")
set_target_properties(cast_python_bind PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cast)
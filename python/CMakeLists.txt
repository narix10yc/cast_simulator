# CAST Python Bindings

# This will put the dependencies in a root-level _deps directory
# so if we delete the build directory, we don't lose the dependencies.
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/_deps")

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)

FetchContent_MakeAvailable(pybind11)

file(GLOB SRCS "*.cpp")

pybind11_add_module(cast_python MODULE ${SRCS})

# copy .pyi file alongside the built .so file
set(PYI_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/cast_python.pyi)
set(PYI_TARGET ${CMAKE_CURRENT_BINARY_DIR}/cast_python.pyi)

add_custom_command(
  OUTPUT ${PYI_TARGET}
  DEPENDS ${PYI_SOURCE}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PYI_SOURCE} ${PYI_TARGET}
  COMMENT "Copying cast_python.pyi to output directory"
)

add_custom_target(cast_python_copy_pyi ALL DEPENDS ${PYI_TARGET})
add_dependencies(cast_python cast_python_copy_pyi)
target_link_libraries(cast_python PRIVATE cast)
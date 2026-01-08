### Handle LLVM Dependency ###
function(handle_llvm_dep)
  # Default mode: release-only LLVM.
  #   - Users should pass CAST_LLVM_ROOT via cmake -DCAST_LLVM_ROOT=...
  #   - CAST_LLVM_ROOT is the install prefix of a *release* LLVM build
  #     (e.g. from apt, brew, or a prebuilt tarball).
  #
  # Optional dev mode (debug-enabled development):
  #   - Users can pass CAST_DEV_LLVM_ROOT which is expected to contain BOTH
  #     'release-install' and 'debug-install' subdirectories.
  #   - In this case CAST_LLVM_ROOT is set to <dev>/release-install and
  #     CAST_LLVM_DEBUG_ROOT is set to <dev>/debug-install (unless already 
  #     provided).
  #
  # Environment variables are only used as a fallback when the corresponding
  # CMake variables (cmake -D) are not provided.

  # --- Release root (preferred: CMake -D..., fallback: env) ---
  if(DEFINED CAST_LLVM_ROOT)
    # already set / cached
  elseif(DEFINED ENV{CAST_LLVM_ROOT})
    set(CAST_LLVM_ROOT "$ENV{CAST_LLVM_ROOT}" CACHE PATH
      "Path to the LLVM release install prefix")
  endif()

  # --- Optional dev/debug roots (preferred: CMake -D..., fallback: env) ---
  # If users provide CAST_DEV_LLVM_ROOT we expect both `debug-install` and
  # `release-install` subdirectories.
  if(DEFINED CAST_LLVM_DEBUG_ROOT)
    # already set / cached
  elseif(DEFINED CAST_DEV_LLVM_ROOT)
    if(NOT DEFINED CAST_LLVM_ROOT)
      set(CAST_LLVM_ROOT "${CAST_DEV_LLVM_ROOT}/release-install" CACHE PATH
        "Path to the LLVM release install prefix")
    endif()
    set(CAST_LLVM_DEBUG_ROOT "${CAST_DEV_LLVM_ROOT}/debug-install" CACHE PATH
      "Path to the llvm debug-install directory")
  elseif(DEFINED ENV{CAST_DEV_LLVM_ROOT})
    if(NOT DEFINED CAST_LLVM_ROOT)
      set(CAST_LLVM_ROOT "$ENV{CAST_DEV_LLVM_ROOT}/release-install" CACHE PATH
        "Path to the LLVM release install prefix")
    endif()
    set(CAST_LLVM_DEBUG_ROOT "$ENV{CAST_DEV_LLVM_ROOT}/debug-install" CACHE PATH
      "Path to the llvm debug-install directory")
  endif()

  # --- Validation ---
  if(NOT DEFINED CAST_LLVM_ROOT)
    message(FATAL_ERROR
      "LLVM is not configured. Please set:\n"
      "  -DCAST_LLVM_ROOT=<llvm-install-prefix>\n"
      "Optionally for debug-enabled development builds:\n"
      "  -DCAST_DEV_LLVM_ROOT=<dir with release-install/ and debug-install/>")
  endif()

  if(NOT EXISTS "${CAST_LLVM_ROOT}")
    message(FATAL_ERROR "CAST_LLVM_ROOT: ${CAST_LLVM_ROOT} does not exist!")
  endif()
  if(NOT IS_DIRECTORY "${CAST_LLVM_ROOT}")
    message(FATAL_ERROR "CAST_LLVM_ROOT: ${CAST_LLVM_ROOT} is not a directory!")
  endif()

  message(STATUS "CAST_LLVM_ROOT is set to: ${CAST_LLVM_ROOT}")
  if(DEFINED CAST_LLVM_DEBUG_ROOT)
    message(STATUS "CAST_LLVM_DEBUG_ROOT is set to: ${CAST_LLVM_DEBUG_ROOT}")
  else()
    message(STATUS "CAST_LLVM_DEBUG_ROOT is not set (release-only LLVM mode)")
  endif()
endfunction() # handle_llvm_dep


function(use_llvm_compilers)
  if(EXISTS "${CAST_LLVM_ROOT}/bin/clang" AND
    EXISTS "${CAST_LLVM_ROOT}/bin/clang++")
    message(STATUS "Using compilers under LLVM release-install at "
      "${CAST_LLVM_ROOT}")
    set(CMAKE_C_COMPILER "${CAST_LLVM_ROOT}/bin/clang" PARENT_SCOPE)
    set(CMAKE_CXX_COMPILER "${CAST_LLVM_ROOT}/bin/clang++" PARENT_SCOPE)
  else()
    message(WARNING "Cannot find clang/clang++ under "
      "${CAST_LLVM_ROOT}/bin/, using system defaults.")
  endif()

  # On Mac OS we need to tell the compiler where to find the SDK (detect via 
  # xcrun if not provided)
  if(APPLE)
    if(NOT DEFINED CMAKE_OSX_SYSROOT)
      find_program(XCRUN xcrun)
      if(XCRUN)
        execute_process(
          COMMAND "${XCRUN}" --sdk macosx --show-sdk-path
          OUTPUT_VARIABLE _sdk OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_sdk)
          set(CMAKE_OSX_SYSROOT "${_sdk}" CACHE PATH "macOS SDK" FORCE)
          message(STATUS "CMAKE_OSX_SYSROOT is set to: ${CMAKE_OSX_SYSROOT}")
        endif()
      endif()
    endif()

    # Optional: take deployment target from env if set
    if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
      set(CMAKE_OSX_DEPLOYMENT_TARGET "$ENV{MACOSX_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
    endif()

    # Optional: choose Mach-O lld
    # set(CMAKE_EXE_LINKER_FLAGS_INIT    "-fuse-ld=/opt/homebrew/opt/lld/bin/ld64.lld")
    # set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=/opt/homebrew/opt/lld/bin/ld64.lld")
  endif()
endfunction() # use_llvm_compilers

function(use_llvm_libcxx)
  # check if libc++ is installed along with LLVM
  # include path is almost always
  # ${CAST_LLVM_ROOT}/include/c++/v1
  set(LIBCXX_INCLUDE_DIR "${CAST_LLVM_ROOT}/include/c++/v1")

  # but the library path may vary. For example,
  # ${CAST_LLVM_ROOT}/lib
  # ${CAST_LLVM_ROOT}/lib/x86_64-unknown-linux-gnu
  file(GLOB_RECURSE LIBCXX_LIB_PATHS "${CAST_LLVM_ROOT}/lib/*libc++.*")
  if(LIBCXX_LIB_PATHS)
    list(GET LIBCXX_LIB_PATHS 0 LIBCXX_LIB_PATH)
    get_filename_component(LIBCXX_LIB_DIR "${LIBCXX_LIB_PATH}" DIRECTORY)
  endif()

  if(EXISTS "${LIBCXX_INCLUDE_DIR}/vector" AND
    DEFINED LIBCXX_LIB_DIR)
    message(STATUS "Found libc++ at ${LIBCXX_LIB_DIR}")
    add_compile_options(-stdlib=libc++ -I"${LIBCXX_INCLUDE_DIR}")
    if(NOT APPLE)
      # On macOS, -stdlib=libc++ automatically links libc++, libc++abi, and
      # unwind. Specifying them will cause annoying warnings.
      link_libraries(c++ c++abi unwind)
    endif()
    link_directories("${LIBCXX_LIB_DIR}")
  else()
    message(STATUS "No libc++ found along with LLVM release-install, "
      "using system defaults.")
  endif()
endfunction() # use_llvm_libcxx

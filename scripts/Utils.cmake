
### Handle LLVM Dependency ###
function(handle_llvm_dep)
  # There are two modes: 
  # 1. Enviroment variable CAST_LLVM_ROOT is set: We expect there are two
  #    subdirectories under CAST_LLVM_ROOT: 'release-install' and 'debug-install'.
  # 2. CAST_LLVM_RELEASE_ROOT: The root directory of the LLVM release-install.
  #    In this case, we won't have debug infos in debug builds.
  # If both CAST_LLVM_ROOT and CAST_LLVM_RELEASE_ROOT are defined, we issue a
  # warning and use the second case.

  if(DEFINED CAST_LLVM_RELEASE_ROOT)
    # already cached
  elseif(DEFINED ENV{CAST_LLVM_ROOT})
    # debug + release llvm versions
    set(CAST_LLVM_DEBUG_ROOT "${CAST_LLVM_ROOT}/debug-install" CACHE PATH
      "Path to the llvm debug-install directory")
    set(CAST_LLVM_RELEASE_ROOT "${CAST_LLVM_ROOT}/release-install" CACHE PATH
      "Path to the llvm release-install directory")
    if(NOT EXISTS "${CAST_LLVM_RELEASE_ROOT}")
      message(FATAL_ERROR "${CAST_LLVM_RELEASE_ROOT} does not exist!")
    endif()
    if(NOT IS_DIRECTORY "${CAST_LLVM_RELEASE_ROOT}")
      message(FATAL_ERROR "${CAST_LLVM_RELEASE_ROOT} is not a directory!")
    endif()
    if(NOT EXISTS "${CAST_LLVM_DEBUG_ROOT}")
      message(FATAL_ERROR "${CAST_LLVM_DEBUG_ROOT} does not exist!")
    endif()
    if(NOT IS_DIRECTORY "${CAST_LLVM_DEBUG_ROOT}")
      message(FATAL_ERROR "${CAST_LLVM_DEBUG_ROOT} is not a directory!")
    endif()
  elseif(DEFINED ENV{CAST_LLVM_RELEASE_ROOT})
    # release only llvm version
    set(CAST_LLVM_RELEASE_ROOT "$ENV{CAST_LLVM_RELEASE_ROOT}" CACHE PATH
      "Path to the llvm release-install directory")
  endif()

  message(STATUS "CAST_LLVM_RELEASE_ROOT is set to: ${CAST_LLVM_RELEASE_ROOT}")
  message(STATUS "CAST_LLVM_DEBUG_ROOT is set to: ${CAST_LLVM_DEBUG_ROOT}")

  if(DEFINED ENV{CAST_LLVM_ROOT} AND DEFINED ENV{CAST_LLVM_RELEASE_ROOT})
    message(WARNING
      "Both CAST_LLVM_ROOT and CAST_LLVM_RELEASE_ROOT are defined. "
      "CAST_LLVM_ROOT will be ignored."
      "So in debug builds, LLVM-related debug infos will not be available. ")
  endif()

  if(NOT DEFINED ENV{CAST_LLVM_ROOT} AND NOT DEFINED ENV{CAST_LLVM_RELEASE_ROOT})
    message(FATAL_ERROR
      "Neither CAST_LLVM_ROOT nor CAST_LLVM_RELEASE_ROOT is defined. "
      "Please set one of them to the root directory of your LLVM installation.")
  endif()
endfunction() # handle_llvm_dep


function(use_llvm_compilers)
  if(EXISTS "${CAST_LLVM_RELEASE_ROOT}/bin/clang" AND
    EXISTS "${CAST_LLVM_RELEASE_ROOT}/bin/clang++")
    message(STATUS "Using compilers under LLVM release-install at "
      "${CAST_LLVM_RELEASE_ROOT}")
    set(CMAKE_C_COMPILER "${CAST_LLVM_RELEASE_ROOT}/bin/clang" PARENT_SCOPE)
    set(CMAKE_CXX_COMPILER "${CAST_LLVM_RELEASE_ROOT}/bin/clang++" PARENT_SCOPE)
  else()
    message(WARNING "Cannot find clang/clang++ under "
      "${CAST_LLVM_RELEASE_ROOT}/bin/, using system defaults.")
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

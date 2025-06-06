#!/bin/bash

set -e # exit on error

# arguments
ARG_BUILD_CLANG=0
ARG_BUILD_LIBCXX=0
ARG_RELEASE_ONLY=0
ARG_NATIVE_ONLY=0
ARG_MINIMAL=0
ARC_LLVM_SRC_DIR=""

INFO='\033[0;36m[Info]\033[0m'
WARN='\033[0;33m[Warning]\033[0m'
ERR='\033[0;31m[Error]\033[0m'

# Argument parsing loop
while [[ $# -gt 0 ]]; do
  case "$1" in 
  -build-clang)
    ARG_BUILD_CLANG=1
    shift
    ;;
  -build-libc++)
    ARG_BUILD_LIBCXX=1
    shift
    ;;
  -release-only)
    ARG_RELEASE_ONLY=1
    shift
    ;;
  -native-only)
    ARG_NATIVE_ONLY=1
    shift
    ;;
  -minimal)
    ARG_MINIMAL=1
    shift
    ;;
  -*)
    echo "Unknown option: $1. Available options are:"
    echo "  -build-clang       Build Clang, LLD, and LLDB"
    echo "  -build-libc++      Build libc++ and libc++abi"
    echo "  -release-only      Only build the release version"
    echo "  -native-only       Only build for the native target"
    echo "  -minimal           Minimal build, only release version with native target"
    echo "  -h, --help         Show this help message"
    exit 1
    ;;
  *)
    # llvm source is assumed to be positional
    if [[ -z "$ARC_LLVM_SRC_DIR" ]]; then
      ARC_LLVM_SRC_DIR="$1"
    else
      echo -e "${WARN} Ignoring additional positional argument: $1"
    fi
    shift
    ;;
  esac
done


# Check of argument conflicts
if [[ $ARG_MINIMAL -eq 1 ]]; then
  if [[ $ARG_BUILD_CLANG -eq 1 ]] || [[ $ARG_BUILD_LIBCXX -eq 1 ]]; then
  echo -e "${ERR} Cannot build both Clang and libc++ with minimal build."
  exit 1
  fi
  echo -e "${INFO} Minimal build selected. " \
          "Only building a release-version of LLVM with Native target."
  ARG_RELEASE_ONLY=1
  ARG_NATIVE_ONLY=1
  ARG_BUILD_CLANG=0
  ARG_BUILD_LIBCXX=0
fi

if [[ -z "$ARC_LLVM_SRC_DIR" ]]; then
  echo "Usage: $0 <llvm-src-dir> [-build-libc++] [-build-clang] " \
       "[-release-only] [-native-only]"
  exit 1
fi

if [[ ! -d "$ARC_LLVM_SRC_DIR" ]]; then
  echo -e "${ERR} LLVM source directory '$ARC_LLVM_SRC_DIR' does not exist."
  exit 1
fi

CAST_LLVM_ROOT=$(dirname "$ARC_LLVM_SRC_DIR")
CAST_LLVM_RELEASE_ROOT="${CAST_LLVM_ROOT}/release-install"
CAST_LLVM_DEBUG_ROOT="${CAST_LLVM_ROOT}/debug-install"

# Get paths to cmake and ninja
CMAKE_PATH="$(command -v cmake)"
NINJA_PATH="$(command -v ninja)"
echo -e "$INFO using cmake in $CMAKE_PATH"
echo -e "$INFO using ninja in $NINJA_PATH"
echo -e "$INFO LLVM source is $ARC_LLVM_SRC_DIR"
echo -e "$INFO CAST_LLVM_ROOT is $CAST_LLVM_ROOT"
echo -e "$INFO build-clang is set to $ARG_BUILD_CLANG"
echo -e "$INFO build-libc++ is set to $ARG_BUILD_LIBCXX"
echo -e "$INFO release-only is set to $ARG_RELEASE_ONLY"
echo -e "$INFO native-only is set to $ARG_NATIVE_ONLY"

sleep 2

# if not native-only, we will build for NVPTX as well
ARG_LLVM_TARGETS_TO_BUILD=""
if [[ $ARG_NATIVE_ONLY -eq 1 ]]; then
  ARG_LLVM_TARGETS_TO_BUILD="Native"
else
  ARG_LLVM_TARGETS_TO_BUILD="Native;NVPTX"
fi

# build clang, lld, and lldb if requested
ARG_LLVM_ENABLE_PROJECTS=""
if [[ $ARG_BUILD_CLANG -eq 1 ]]; then
  ARG_LLVM_ENABLE_PROJECTS="-DLLVM_ENABLE_PROJECTS=clang;lld;lldb"
fi

# build libc++ and libc++abi if requested
ARG_LLVM_ENABLE_RUNTIMES=""
if [[ $ARG_BUILD_LIBCXX -eq 1 ]]; then
  ARG_LLVM_ENABLE_RUNTIMES=(
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind"
    -DLIBCXXABI_USE_LLVM_UNWINDER=ON
    -DLIBCXXABI_ENABLE_SHARED=ON
    -DLIBUNWIND_ENABLE_SHARED=ON
  )
fi

cmake -S "${ARC_LLVM_SRC_DIR}/llvm" -G Ninja \
  -B "${CAST_LLVM_ROOT}/release-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_RTTI=ON \
  "-DLLVM_TARGETS_TO_BUILD=${ARG_LLVM_TARGETS_TO_BUILD}" \
  "${ARG_LLVM_ENABLE_PROJECTS[@]}" \
  "${ARG_LLVM_ENABLE_RUNTIMES[@]}"

cmake --build "${CAST_LLVM_ROOT}/release-build"

cmake --install "${CAST_LLVM_ROOT}/release-build" \
      --prefix "${CAST_LLVM_RELEASE_ROOT}"

# end of release build
if [[ $ARG_RELEASE_ONLY -eq 1 ]]; then
  echo -e "${INFO} LLVM release-version installed successfully."
  echo -e "${INFO} LLVM is installed for targets ${ARG_LLVM_TARGETS_TO_BUILD}"
  echo -e "${INFO} Release install is in ${CAST_LLVM_RELEASE_ROOT}"
  echo -e "${INFO} Don't forget to set your environment variables:"
  echo -e "${INFO} export CAST_LLVM_RELEASE_ROOT=${CAST_LLVM_RELEASE_ROOT}"
  echo -e "${INFO} You may remove build directories by running:"
  echo -e "${INFO} rm -rf ${CAST_LLVM_ROOT}/release-build"
  exit 0
fi

# start of debug build
echo -e "${INFO} LLVM release-version installed successfully. "\
          "Continuing with debug build..."

ARG_SET_USE_CLANG=""
if [[ $ARG_BUILD_CLANG -eq 1 ]]; then
  ARG_SET_USE_CLANG=(
    "-DCMAKE_C_COMPILER=${CAST_LLVM_RELEASE_ROOT}/bin/clang"
    "-DCMAKE_CXX_COMPILER=${CAST_LLVM_RELEASE_ROOT}/bin/clang++"
    "-DCMAKE_LINKER=${CAST_LLVM_RELEASE_ROOT}/bin/ld.lld"
  )
fi

ARG_SET_USE_LIBCXX=""
if [[ $ARG_BUILD_LIBCXX -eq 1 ]]; then
  ARG_SET_USE_LIBCXX=(
    -DCMAKE_CXX_FLAGS="-stdlib=libc++ -I${CAST_LLVM_RELEASE_ROOT}/include/c++/v1"
    -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -L${CAST_LLVM_RELEASE_ROOT}/lib"
    -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -L${CAST_LLVM_RELEASE_ROOT}/lib"
    -DCMAKE_INSTALL_RPATH="${CAST_LLVM_RELEASE_ROOT}/lib"
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  )
  # It seems that libc++ is not always installed in release-install/lib. 
  # It might be in release-install/lib/x86_64-unknown-linux-gnu or similar.
  # We will try to find it and set the path accordingly.
  LIBCXX_DIR=$(find "${CAST_LLVM_RELEASE_ROOT}/lib" \
               -type f \( -name "libc++*.so*" -o -name "libc++*.dylib*" \)
               -exec dirname {} \; | head -n 1)
  if [[ -n "$LIBCXX_DIR" ]]; then
    echo -e "${INFO} Found libc++ in directory: ${LIBCXX_DIR}"
  else
    echo -e "${WARN} libc++ not found in ${CAST_LLVM_RELEASE_ROOT}/lib. "\
            "This means we may have not correctly set up runtime lib path "\
            "and second phase of this build may fail."
  fi

  case $(uname) in
    Linux)
      export LD_LIBRARY_PATH="${LIBCXX_DIR}:${LD_LIBRARY_PATH}"
      ;;
    Darwin)
      export DYLD_LIBRARY_PATH="${LIBCXX_DIR}:${DYLD_LIBRARY_PATH}"
      ;;
    *)
      echo -e "${WARN} Platform $(uname) not directly supported. "\
              "Falling back to treating it as Linux."
      export LD_LIBRARY_PATH="${LIBCXX_DIR}:${LD_LIBRARY_PATH}"
      ;;
  esac
fi

cmake -S "${ARC_LLVM_SRC_DIR}/llvm" -G Ninja \
  -B "${CAST_LLVM_ROOT}/debug-build" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_RTTI=ON \
  "-DLLVM_TARGETS_TO_BUILD=${ARG_LLVM_TARGETS_TO_BUILD}" \
  "${ARG_SET_USE_CLANG[@]}" \
  "${ARG_SET_USE_LIBCXX[@]}"

cmake --build "${CAST_LLVM_ROOT}/debug-build"

cmake --install "${CAST_LLVM_ROOT}/debug-build" \
      --prefix "${CAST_LLVM_DEBUG_ROOT}"

echo -e "${INFO} LLVM build completed successfully."
echo -e "${INFO} LLVM is installed for targets ${ARG_LLVM_TARGETS_TO_BUILD}"
echo -e "${INFO} Release install is in ${CAST_LLVM_RELEASE_ROOT}"
echo -e "${INFO} Debug install is in ${CAST_LLVM_DEBUG_ROOT}"
echo -e "${INFO} Don't forget to set your environment variables:"
echo -e "${INFO} export CAST_LLVM_ROOT=${CAST_LLVM_ROOT}"
echo -e "${INFO} You may remove build directories by running:"
echo -e "${INFO} rm -rf ${CAST_LLVM_ROOT}/release-build ${CAST_LLVM_ROOT}/debug-build"

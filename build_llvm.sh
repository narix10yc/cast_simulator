#!/bin/bash

set -e # exit on error

# arguments
BUILD_CLANG=0
BUILD_LIBCXX=0
LLVM_SRC_DIR=""

# Argument parsing loop
while [[ $# -gt 0 ]]; do
  case "$1" in 
  -build-clang)
    BUILD_CLANG=1
    shift
    ;;
  -build-libc++)
    BUILD_LIBCXX=1
    shift
    ;;
  -*)
    echo "Unknown option: $1"
    exit 1
    ;;
  *)
    # llvm source is assumed to be positional
    if [[ -z "$LLVM_SRC_DIR" ]]; then
      LLVM_SRC_DIR="$1"
    else
      OTHER_ARGS+=("$1")
    fi
    shift
    ;;
  esac
done

if [[ -z "$LLVM_SRC_DIR" ]]; then
  echo "Usage: $0 <llvm-src-dir> [-build-libc++] [-build-clang]"
  exit 1
fi

INFO='\033[0;36m[Info]\033[0m'
WARN='\033[0;33m[Warning]\033[0m'
ERR='\033[0;31m[Error]\033[0m'

if [[ ! -d "$LLVM_SRC_DIR" ]]; then
  echo -e "${ERR} LLVM source directory '$LLVM_SRC_DIR' does not exist."
  exit 1
fi

CAST_LLVM_ROOT=$(dirname "$LLVM_SRC_DIR")

# Get paths to cmake and ninja
CMAKE_PATH="$(command -v cmake)"
NINJA_PATH="$(command -v ninja)"
echo -e "$INFO using cmake in $CMAKE_PATH"
echo -e "$INFO using ninja in $NINJA_PATH"
echo -e "$INFO LLVM source is $LLVM_SRC_DIR"
echo -e "$INFO CAST_LLVM_ROOT is $CAST_LLVM_ROOT"
echo -e "$INFO build-clang is set to $BUILD_CLANG"
echo -e "$INFO build-libc++ is set to $BUILD_LIBCXX"

sleep 1

# build clang, lld, and lldb if requested
ARG_LLVM_ENABLE_PROJECTS=""
if [[ $BUILD_CLANG -eq 1 ]]; then
  ARG_LLVM_ENABLE_PROJECTS="-DLLVM_ENABLE_PROJECTS=clang;lld;lldb"
fi

# build libc++ and libc++abi if requested
ARG_LLVM_ENABLE_RUNTIMES=""
if [[ $BUILD_LIBCXX -eq 1 ]]; then
  ARG_LLVM_ENABLE_RUNTIMES=(
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind"
    -DLIBCXXABI_USE_LLVM_UNWINDER=ON
    -DLIBCXXABI_ENABLE_SHARED=ON
    -DLIBUNWIND_ENABLE_SHARED=ON
  )
fi

cmake -S "${LLVM_SRC_DIR}/llvm" -G Ninja \
  -B "${CAST_LLVM_ROOT}/release-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  "${ARG_LLVM_ENABLE_PROJECTS[@]}" \
  "${ARG_LLVM_ENABLE_RUNTIMES[@]}"

cmake --build "${CAST_LLVM_ROOT}/release-build"

cmake --install "${CAST_LLVM_ROOT}/release-build" \
      --prefix "${CAST_LLVM_ROOT}/release-install"

ARG_SET_USE_CLANG=""
if [[ $BUILD_CLANG -eq 1 ]]; then
  ARG_SET_USE_CLANG=(
    "-DCMAKE_C_COMPILER=${CAST_LLVM_ROOT}/release-install/bin/clang"
    "-DCMAKE_CXX_COMPILER=${CAST_LLVM_ROOT}/release-install/bin/clang++"
    "-DCMAKE_LINKER=${CAST_LLVM_ROOT}/release-install/bin/ld.lld"
  )
fi

ARG_SET_USE_LIBCXX=""
if [[ $BUILD_LIBCXX -eq 1 ]]; then
  ARG_SET_USE_LIBCXX=(
    -DCMAKE_CXX_FLAGS="-stdlib=libc++ -I${CAST_LLVM_ROOT}/release-install/include/c++/v1"
    # How to setup linker to use libc++ at ${CAST_LLVM_ROOT}/release-install/lib
    -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -L${CAST_LLVM_ROOT}/release-install/lib"
    # -DCMAKE_STATIC_LINKER_FLAGS="-stdlib=libc++ -L${CAST_LLVM_ROOT}/release-install/lib"
    -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -L${CAST_LLVM_ROOT}/release-install/lib"
    -DCMAKE_BUILD_RPATH="${CAST_LLVM_ROOT}/release-install/lib"
    -DCMAKE_INSTALL_RPATH="${CAST_LLVM_ROOT}/release-install/lib"
  )
fi

cmake -S "${LLVM_SRC_DIR}/llvm" -G Ninja \
  -B "${CAST_LLVM_ROOT}/debug-build" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  "${ARG_SET_USE_CLANG[@]}" \
  "${ARG_SET_USE_LIBCXX[@]}"

cmake --build "${CAST_LLVM_ROOT}/debug-build"

cmake --install "${CAST_LLVM_ROOT}/debug-build" \
      --prefix "${CAST_LLVM_ROOT}/debug-install"

echo -e "${INFO} LLVM build completed successfully."
echo -e "${INFO} Release install is in ${CAST_LLVM_ROOT}/release-install"
echo -e "${INFO} Debug install is in ${CAST_LLVM_ROOT}/debug-install"
echo -e "${INFO} Don't forget to set your environment variables:"
echo -e "${INFO} export CAST_LLVM_ROOT=${CAST_LLVM_ROOT}"
echo -e "${INFO} You may remove build directories by running:"
echo -e "${INFO} rm -rf ${CAST_LLVM_ROOT}/release-build ${CAST_LLVM_ROOT}/debug-build"

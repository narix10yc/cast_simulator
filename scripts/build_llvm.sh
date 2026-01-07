#!/bin/bash

set -e

# arguments
ARG_BUILD_CLANG=0
ARG_BUILD_LIBCXX=0
ARG_RELEASE_ONLY=0
ARG_NATIVE_ONLY=0
ARG_MINIMAL=0
ARG_LLVM_SRC_DIR=""

INFO='\033[0;36m[Info]\033[0m'
WARN='\033[0;33m[Warning]\033[0m'
ERR='\033[0;31m[Error]\033[0m'

usage() {
  cat <<EOF
Usage: $0 <llvm-src-dir> [options]

Options:
  -build-clang       Build Clang, LLD, and LLDB
  -build-libc++      Build libc++ and libc++abi
  -release-only      Only build the release version
  -native-only       Only build for the native target
  -minimal           Minimal build, only release version with native target
  -h, --help         Show this help message
EOF
}

# Argument parsing loop
while [[ $# -gt 0 ]]; do
  case "$1" in 
  -h|--help)
    usage
    exit 0
    ;;
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
    echo "Unknown option: $1"
    usage
    exit 1
    ;;
  *)
    # llvm source is assumed to be positional
    if [[ -z "$ARG_LLVM_SRC_DIR" ]]; then
      ARG_LLVM_SRC_DIR="$1"
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
  echo -e "${ERR} Cannot build Clang or libc++ with minimal build."
  exit 1
  fi
  echo -e "${INFO} Minimal build selected. " \
          "Only building a release-version of LLVM with Native target."
  ARG_RELEASE_ONLY=1
  ARG_NATIVE_ONLY=1
  ARG_BUILD_CLANG=0
  ARG_BUILD_LIBCXX=0
fi

if [[ -z "$ARG_LLVM_SRC_DIR" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$ARG_LLVM_SRC_DIR" ]]; then
  echo -e "${ERR} LLVM source directory '$ARG_LLVM_SRC_DIR' does not exist."
  exit 1
fi

# Accept either llvm-project root or llvm-project/llvm.
LLVM_SRC_DIR=""
if [[ -f "${ARG_LLVM_SRC_DIR}/llvm/CMakeLists.txt" ]]; then
  LLVM_SRC_DIR="${ARG_LLVM_SRC_DIR}/llvm"
  LLVM_ROOT="${ARG_LLVM_SRC_DIR}"
elif [[ -f "${ARG_LLVM_SRC_DIR}/CMakeLists.txt" ]]; then
  LLVM_SRC_DIR="${ARG_LLVM_SRC_DIR}"
  LLVM_ROOT="$(dirname "${ARG_LLVM_SRC_DIR}")"
else
  echo -e "${ERR} Could not find llvm source at '${ARG_LLVM_SRC_DIR}'."
  echo -e "${ERR} Expected llvm/CMakeLists.txt or CMakeLists.txt."
  exit 1
fi

RELEASE_INSTALL_ROOT="${LLVM_ROOT}/release-install"
RELEASE_BUILD_ROOT="${LLVM_ROOT}/release-build"
DEBUG_INSTALL_ROOT="${LLVM_ROOT}/debug-install"
DEBUG_BUILD_ROOT="${LLVM_ROOT}/debug-build"


# Get paths to cmake and ninja
CMAKE_PATH="$(command -v cmake)"
NINJA_PATH="$(command -v ninja)"
if [[ -z "${CMAKE_PATH}" ]]; then
  echo -e "${ERR} cmake not found in PATH."
  exit 1
fi
if [[ -z "${NINJA_PATH}" ]]; then
  echo -e "${ERR} ninja not found in PATH."
  exit 1
fi
echo -e "$INFO using cmake in $CMAKE_PATH"
echo -e "$INFO using ninja in $NINJA_PATH"
echo -e "$INFO LLVM source is $LLVM_SRC_DIR"
echo -e "$INFO LLVM_ROOT is $LLVM_ROOT"
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
ARG_LLVM_ENABLE_PROJECTS=()
if [[ $ARG_BUILD_CLANG -eq 1 ]]; then
  ARG_LLVM_ENABLE_PROJECTS=(
    "-DLLVM_ENABLE_PROJECTS=clang;lld;lldb"
  )
fi

# build libc++ and libc++abi if requested
ARG_LLVM_ENABLE_RUNTIMES=()
if [[ $ARG_BUILD_LIBCXX -eq 1 ]]; then
  ARG_LLVM_ENABLE_RUNTIMES=(
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind"
    -DLIBCXXABI_USE_LLVM_UNWINDER=ON
    -DLIBCXXABI_ENABLE_SHARED=ON
    -DLIBUNWIND_ENABLE_SHARED=ON
  )
fi

# release build

cmake -S "${LLVM_SRC_DIR}" -G Ninja \
  -B "${RELEASE_BUILD_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_RTTI=ON \
  "-DLLVM_TARGETS_TO_BUILD=${ARG_LLVM_TARGETS_TO_BUILD}" \
  ${ARG_LLVM_ENABLE_PROJECTS:+${ARG_LLVM_ENABLE_PROJECTS[@]}} \
  ${ARG_LLVM_ENABLE_RUNTIMES:+${ARG_LLVM_ENABLE_RUNTIMES[@]}}

cmake --build "${RELEASE_BUILD_ROOT}"
cmake --install "${RELEASE_BUILD_ROOT}" --prefix "${RELEASE_INSTALL_ROOT}"

# exit if only release build was requested
if [[ $ARG_RELEASE_ONLY -eq 1 ]]; then
  echo -e "${INFO} LLVM release-version installed successfully."
  echo -e "${INFO} LLVM is installed for targets ${ARG_LLVM_TARGETS_TO_BUILD}"
  echo -e "${INFO} Release install root: ${RELEASE_INSTALL_ROOT}"
  echo -e "${INFO} You may remove build directories by running:"
  echo -e "${INFO} rm -rf ${RELEASE_BUILD_ROOT}"
  exit 0
fi

echo -e "${INFO} LLVM release-version installed successfully. "\
          "Continuing with debug build..."

ARG_SET_USE_CLANG=()
if [[ $ARG_BUILD_CLANG -eq 1 ]]; then
  ARG_SET_USE_CLANG=(
    "-DCMAKE_C_COMPILER=${RELEASE_INSTALL_ROOT}/bin/clang"
    "-DCMAKE_CXX_COMPILER=${RELEASE_INSTALL_ROOT}/bin/clang++"
    "-DCMAKE_LINKER=${RELEASE_INSTALL_ROOT}/bin/ld.lld"
  )
fi

ARG_SET_USE_LIBCXX=()
if [[ $ARG_BUILD_LIBCXX -eq 1 ]]; then
  ARG_SET_USE_LIBCXX=(
    -DCMAKE_CXX_FLAGS="-stdlib=libc++ -I${RELEASE_INSTALL_ROOT}/include/c++/v1"
    -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -L${RELEASE_INSTALL_ROOT}/lib"
    -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -L${RELEASE_INSTALL_ROOT}/lib"
    -DCMAKE_INSTALL_RPATH="${RELEASE_INSTALL_ROOT}/lib"
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  )
  # It seems that libc++ is not always installed in release-install/lib. 
  # It might be in release-install/lib/x86_64-unknown-linux-gnu or similar.
  # We will try to find it and set the path accordingly.
  LIBCXX_DIR=$(find "${RELEASE_INSTALL_ROOT}/lib" \
               -type f \( -name "libc++*.so*" -o -name "libc++*.dylib*" \)
               -exec dirname {} \; | head -n 1)
  if [[ -n "$LIBCXX_DIR" ]]; then
    echo -e "${INFO} Found libc++ in directory: ${LIBCXX_DIR}"
  else
    echo -e "${WARN} libc++ not found in ${RELEASE_INSTALL_ROOT}/lib. "\
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

# debug build

cmake -S "${LLVM_SRC_DIR}" -G Ninja \
  -B "${DEBUG_BUILD_ROOT}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_RTTI=ON \
  "-DLLVM_TARGETS_TO_BUILD=${ARG_LLVM_TARGETS_TO_BUILD}" \
  ${ARG_SET_USE_CLANG:+${ARG_SET_USE_CLANG[@]}} \
  ${ARG_SET_USE_LIBCXX:+${ARG_SET_USE_LIBCXX[@]}}

cmake --build "${DEBUG_BUILD_ROOT}"

cmake --install "${DEBUG_BUILD_ROOT}" --prefix "${DEBUG_INSTALL_ROOT}"

echo -e "${INFO} LLVM *debug* build completed successfully."
echo -e "${INFO} LLVM is installed for targets ${ARG_LLVM_TARGETS_TO_BUILD}"
echo -e "${INFO} Debug install root: ${DEBUG_INSTALL_ROOT}"
echo -e "${INFO} You may remove build directories by running:"
echo -e "${INFO} rm -rf ${DEBUG_BUILD_ROOT}"

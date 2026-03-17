#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

WITH_DEBUG=0
WITH_CLANG_TOOLS=0
WITH_LIBCXX=0
CPU_ONLY=0
CLEAN=0
VERIFY_ONLY=0
JOBS=""
BUILD_ROOT=""
INSTALL_ROOT=""
LLVM_INPUT=""

if [[ -t 1 ]]; then
  INFO=$'\033[0;36m[info]\033[0m'
  WARN=$'\033[0;33m[warn]\033[0m'
  ERR=$'\033[0;31m[error]\033[0m'
else
  INFO='[info]'
  WARN='[warn]'
  ERR='[error]'
fi

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} <llvm-src-dir> [options]

Builds LLVM from source into a local install layout suitable for CAST.

Defaults:
  - Build a release LLVM only
  - Build targets: Native;NVPTX
  - Install under the source root's parent directory:
      <root>/release-build
      <root>/release-install

Options:
  --with-debug          Also build and install a debug LLVM
  --cpu-only            Build Native target only
  --clean               Remove release-build/ and debug-build/ under the chosen build root
  --verify-targets      Verify release-install/bin/llvm-config and target support, then exit
  --with-clang-tools    Build clang, lld, lldb, and clang-tools-extra
  --with-libcxx         Build libc++, libc++abi, and libunwind
  --jobs N              Pass --parallel N to cmake --build
  --build-root DIR      Directory for release-build/ and debug-build/
  --install-root DIR    Directory for release-install/ and debug-install/
  -h, --help            Show this help message

Examples:
  ${SCRIPT_NAME} ~/llvm/22.1.1/llvm-project-22.1.1.src
  ${SCRIPT_NAME} ~/llvm/22.1.1/llvm-project-22.1.1.src --cpu-only
  ${SCRIPT_NAME} ~/llvm/22.1.1/llvm-project-22.1.1.src --verify-targets
  ${SCRIPT_NAME} ~/llvm/22.1.1/llvm-project-22.1.1.src --clean
  ${SCRIPT_NAME} ~/llvm/22.1.1/llvm-project-22.1.1.src --with-debug --with-clang-tools
EOF
}

info() {
  printf '%s %s\n' "${INFO}" "$*"
}

warn() {
  printf '%s %s\n' "${WARN}" "$*" >&2
}

die() {
  printf '%s %s\n' "${ERR}" "$*" >&2
  exit 1
}

run() {
  info "Running: $*"
  "$@"
}

require_tool() {
  local tool="$1"
  command -v "${tool}" >/dev/null 2>&1 || die "Required tool not found in PATH: ${tool}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-debug)
        WITH_DEBUG=1
        shift
        ;;
      --cpu-only)
        CPU_ONLY=1
        shift
        ;;
      --clean)
        CLEAN=1
        shift
        ;;
      --verify-targets)
        VERIFY_ONLY=1
        shift
        ;;
      --with-clang-tools)
        WITH_CLANG_TOOLS=1
        shift
        ;;
      --with-libcxx)
        WITH_LIBCXX=1
        shift
        ;;
      --jobs)
        [[ $# -ge 2 ]] || die "--jobs requires a value"
        JOBS="$2"
        shift 2
        ;;
      --build-root)
        [[ $# -ge 2 ]] || die "--build-root requires a value"
        BUILD_ROOT="$2"
        shift 2
        ;;
      --install-root)
        [[ $# -ge 2 ]] || die "--install-root requires a value"
        INSTALL_ROOT="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      -*)
        die "Unknown option: $1"
        ;;
      *)
        if [[ -n "${LLVM_INPUT}" ]]; then
          die "Unexpected extra positional argument: $1"
        fi
        LLVM_INPUT="$1"
        shift
        ;;
    esac
  done
}

resolve_source_layout() {
  [[ -n "${LLVM_INPUT}" ]] || die "Missing llvm source directory. See --help."
  [[ -d "${LLVM_INPUT}" ]] || die "LLVM source directory does not exist: ${LLVM_INPUT}"

  if [[ -f "${LLVM_INPUT}/llvm/CMakeLists.txt" ]]; then
    LLVM_SOURCE_DIR="${LLVM_INPUT}/llvm"
    DEFAULT_LAYOUT_ROOT="$(cd "${LLVM_INPUT}/.." && pwd)"
  elif [[ -f "${LLVM_INPUT}/CMakeLists.txt" ]]; then
    LLVM_SOURCE_DIR="${LLVM_INPUT}"
    DEFAULT_LAYOUT_ROOT="$(cd "${LLVM_INPUT}/.." && pwd)"
  else
    die "Could not find LLVM sources under ${LLVM_INPUT}. Expected llvm/CMakeLists.txt or CMakeLists.txt."
  fi

  BUILD_ROOT="${BUILD_ROOT:-${DEFAULT_LAYOUT_ROOT}}"
  INSTALL_ROOT="${INSTALL_ROOT:-${DEFAULT_LAYOUT_ROOT}}"

  RELEASE_BUILD_DIR="${BUILD_ROOT}/release-build"
  DEBUG_BUILD_DIR="${BUILD_ROOT}/debug-build"
  RELEASE_INSTALL_DIR="${INSTALL_ROOT}/release-install"
  DEBUG_INSTALL_DIR="${INSTALL_ROOT}/debug-install"
}

validate_args() {
  if [[ -n "${JOBS}" ]] && ! [[ "${JOBS}" =~ ^[0-9]+$ ]]; then
    die "--jobs expects a non-negative integer, got: ${JOBS}"
  fi

  if [[ "${CLEAN}" -eq 1 ]] && [[ "${VERIFY_ONLY}" -eq 1 ]]; then
    die "--clean and --verify-targets are mutually exclusive"
  fi

  if [[ "${CLEAN}" -eq 0 ]] && [[ "${VERIFY_ONLY}" -eq 0 ]]; then
    require_tool cmake
    require_tool ninja
  fi

  if [[ "${WITH_LIBCXX}" -eq 1 ]]; then
    warn "--with-libcxx is optional and less frequently exercised than the default configuration."
  fi
}

targets_to_build() {
  if [[ "${CPU_ONLY}" -eq 1 ]]; then
    printf 'Native'
  else
    printf 'Native;NVPTX'
  fi
}

cmake_build_args() {
  if [[ -n "${JOBS}" ]]; then
    printf '%s\0%s\0' --parallel "${JOBS}"
  fi
}

libcxx_runtime_dir() {
  local prefix="$1"

  if [[ ! -d "${prefix}/lib" ]]; then
    return 0
  fi

  find "${prefix}/lib" -type f \
    \( -name 'libc++*.so*' -o -name 'libc++*.dylib*' \) \
    -exec dirname {} \; 2>/dev/null | head -n 1 || true
}

run_with_runtime_env() {
  local runtime_dir="$1"
  shift

  if [[ -z "${runtime_dir}" ]]; then
    run "$@"
    return
  fi

  case "$(uname)" in
    Darwin)
      info "Using DYLD_LIBRARY_PATH=${runtime_dir}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
      env DYLD_LIBRARY_PATH="${runtime_dir}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" "$@"
      ;;
    *)
      info "Using LD_LIBRARY_PATH=${runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      env LD_LIBRARY_PATH="${runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "$@"
      ;;
  esac
}

configure_release() {
  local targets
  local -a args

  targets="$(targets_to_build)"
  args=(
    -S "${LLVM_SOURCE_DIR}"
    -G Ninja
    -B "${RELEASE_BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DLLVM_ENABLE_RTTI=ON
    "-DLLVM_TARGETS_TO_BUILD=${targets}"
  )

  if [[ "${WITH_CLANG_TOOLS}" -eq 1 ]]; then
    args+=("-DLLVM_ENABLE_PROJECTS=clang;lld;lldb;clang-tools-extra")
  fi
  if [[ "${WITH_LIBCXX}" -eq 1 ]]; then
    args+=(
      "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi;libunwind"
      -DLIBCXXABI_USE_LLVM_UNWINDER=ON
      -DLIBCXXABI_ENABLE_SHARED=ON
      -DLIBUNWIND_ENABLE_SHARED=ON
    )
  fi

  run cmake "${args[@]}"
}

install_release() {
  local -a build_args

  build_args=()
  while IFS= read -r -d '' token; do
    build_args+=("${token}")
  done < <(cmake_build_args)

  if [[ "${#build_args[@]}" -gt 0 ]]; then
    run cmake --build "${RELEASE_BUILD_DIR}" "${build_args[@]}"
  else
    run cmake --build "${RELEASE_BUILD_DIR}"
  fi
  run cmake --install "${RELEASE_BUILD_DIR}" --prefix "${RELEASE_INSTALL_DIR}"
}

configure_debug() {
  local targets
  local runtime_dir
  local -a args

  targets="$(targets_to_build)"
  args=(
    -S "${LLVM_SOURCE_DIR}"
    -G Ninja
    -B "${DEBUG_BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Debug
    -DLLVM_ENABLE_RTTI=ON
    "-DLLVM_TARGETS_TO_BUILD=${targets}"
  )

  if [[ -x "${RELEASE_INSTALL_DIR}/bin/clang" ]] && [[ -x "${RELEASE_INSTALL_DIR}/bin/clang++" ]]; then
    args+=(
      "-DCMAKE_C_COMPILER=${RELEASE_INSTALL_DIR}/bin/clang"
      "-DCMAKE_CXX_COMPILER=${RELEASE_INSTALL_DIR}/bin/clang++"
    )
    if [[ -x "${RELEASE_INSTALL_DIR}/bin/ld.lld" ]]; then
      args+=("-DCMAKE_LINKER=${RELEASE_INSTALL_DIR}/bin/ld.lld")
    fi
  fi

  runtime_dir=""
  if [[ "${WITH_LIBCXX}" -eq 1 ]]; then
    runtime_dir="$(libcxx_runtime_dir "${RELEASE_INSTALL_DIR}")"
    if [[ -n "${runtime_dir}" ]]; then
      args+=(
        "-DCMAKE_CXX_FLAGS=-stdlib=libc++ -I${RELEASE_INSTALL_DIR}/include/c++/v1"
        "-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++ -L${runtime_dir}"
        "-DCMAKE_SHARED_LINKER_FLAGS=-stdlib=libc++ -L${runtime_dir}"
        "-DCMAKE_INSTALL_RPATH=${runtime_dir}"
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
      )
    else
      warn "Could not find libc++ under ${RELEASE_INSTALL_DIR}/lib; continuing without libc++ bootstrap flags."
    fi
  fi

  run_with_runtime_env "${runtime_dir}" cmake "${args[@]}"
}

install_debug() {
  local runtime_dir
  local -a build_args

  runtime_dir=""
  if [[ "${WITH_LIBCXX}" -eq 1 ]]; then
    runtime_dir="$(libcxx_runtime_dir "${RELEASE_INSTALL_DIR}")"
  fi

  build_args=()
  while IFS= read -r -d '' token; do
    build_args+=("${token}")
  done < <(cmake_build_args)

  if [[ "${#build_args[@]}" -gt 0 ]]; then
    run_with_runtime_env "${runtime_dir}" cmake --build "${DEBUG_BUILD_DIR}" "${build_args[@]}"
  else
    run_with_runtime_env "${runtime_dir}" cmake --build "${DEBUG_BUILD_DIR}"
  fi
  run_with_runtime_env "${runtime_dir}" cmake --install "${DEBUG_BUILD_DIR}" --prefix "${DEBUG_INSTALL_DIR}"
}

verify_install() {
  local llvm_config="${RELEASE_INSTALL_DIR}/bin/llvm-config"
  local targets_output
  local expected_target
  local concrete_target
  local missing_targets=()

  [[ -x "${llvm_config}" ]] || die "Release install did not produce ${llvm_config}"
  run "${llvm_config}" --version
  targets_output="$("${llvm_config}" --targets-built)"
  info "llvm-config --targets-built: ${targets_output}"

  IFS=';' read -r -a expected_targets <<< "$(targets_to_build)"
  for expected_target in "${expected_targets[@]}"; do
    concrete_target="${expected_target}"
    if [[ "${expected_target}" == "Native" ]]; then
      concrete_target="$(native_backend_target "${llvm_config}")"
    fi

    if [[ -z "${concrete_target}" ]]; then
      warn "Could not resolve Native to a concrete LLVM backend; skipping strict Native target verification."
      continue
    fi

    if [[ " ${targets_output} " != *" ${concrete_target} "* ]]; then
      missing_targets+=("${concrete_target}")
    fi
  done

  if [[ "${#missing_targets[@]}" -gt 0 ]]; then
    die "Installed LLVM is missing expected targets: ${missing_targets[*]}"
  fi
}

native_backend_target() {
  local llvm_config="$1"
  local host_arch

  host_arch="$("${llvm_config}" --host-target 2>/dev/null | cut -d- -f1)"
  case "${host_arch}" in
    aarch64|arm64)
      printf 'AArch64'
      ;;
    x86_64|amd64|i386|i486|i586|i686)
      printf 'X86'
      ;;
    riscv32|riscv64)
      printf 'RISCV'
      ;;
    powerpc|powerpc64|powerpc64le|ppc64|ppc64le)
      printf 'PowerPC'
      ;;
    systemz|s390x)
      printf 'SystemZ'
      ;;
    loongarch32|loongarch64)
      printf 'LoongArch'
      ;;
    mips|mipsel|mips64|mips64el)
      printf 'Mips'
      ;;
    sparc|sparcv9)
      printf 'Sparc'
      ;;
    wasm32|wasm64)
      printf 'WebAssembly'
      ;;
    bpf|bpfeb|bpfel)
      printf 'BPF'
      ;;
    *)
      printf ''
      ;;
  esac
}

remove_dir_if_present() {
  local dir="$1"

  if [[ ! -e "${dir}" ]]; then
    info "Skipping missing directory: ${dir}"
    return
  fi

  if [[ ! -d "${dir}" ]]; then
    die "Refusing to remove non-directory path: ${dir}"
  fi

  case "${dir}" in
    */release-build|*/debug-build)
      ;;
    *)
      die "Refusing to remove unexpected path: ${dir}"
      ;;
  esac

  run rm -rf "${dir}"
}

clean_build_dirs() {
  info "Cleaning build directories under ${BUILD_ROOT}"
  remove_dir_if_present "${RELEASE_BUILD_DIR}"
  remove_dir_if_present "${DEBUG_BUILD_DIR}"
}

print_summary() {
  local targets
  local llvm_config

  targets="$(targets_to_build)"
  llvm_config="${RELEASE_INSTALL_DIR}/bin/llvm-config"

  printf '\n'
  info "LLVM source      : ${LLVM_SOURCE_DIR}"
  info "Build root       : ${BUILD_ROOT}"
  info "Install root     : ${INSTALL_ROOT}"
  info "Targets          : ${targets}"
  info "Release install  : ${RELEASE_INSTALL_DIR}"
  if [[ "${WITH_DEBUG}" -eq 1 ]]; then
    info "Debug install    : ${DEBUG_INSTALL_DIR}"
  else
    info "Debug install    : not requested"
  fi
  printf '\n'
  info "Set this for Rust builds:"
  printf 'export LLVM_CONFIG="%s"\n' "${llvm_config}"
  printf '\n'
  info "Legacy CMake variables, if still needed:"
  printf 'export CAST_LLVM_ROOT="%s"\n' "${RELEASE_INSTALL_DIR}"
  if [[ "${WITH_DEBUG}" -eq 1 ]]; then
    printf 'export CAST_DEV_LLVM_ROOT="%s"\n' "${INSTALL_ROOT}"
  fi
}

main() {
  parse_args "$@"
  validate_args
  resolve_source_layout

  info "LLVM source     : ${LLVM_SOURCE_DIR}"
  info "Release build   : ${RELEASE_BUILD_DIR}"
  info "Release install : ${RELEASE_INSTALL_DIR}"
  if [[ "${WITH_DEBUG}" -eq 1 ]]; then
    info "Debug build     : ${DEBUG_BUILD_DIR}"
    info "Debug install   : ${DEBUG_INSTALL_DIR}"
  fi
  info "Targets         : $(targets_to_build)"

  if [[ "${CLEAN}" -eq 1 ]]; then
    clean_build_dirs
    exit 0
  fi

  if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
    verify_install
    print_summary
    exit 0
  fi

  configure_release
  install_release

  if [[ "${WITH_DEBUG}" -eq 1 ]]; then
    configure_debug
    install_debug
  fi

  verify_install
  print_summary
}

main "$@"

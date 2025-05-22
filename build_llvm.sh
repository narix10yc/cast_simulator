#! /bin/shell

cast_llvm_root=$1
llvm_version=$2

echo "cast_llvm_root=$cast_llvm_root"
echo "llvm_version=$llvm_version"

cmake -S "${cast_llvm_root}/llvm-project-${llvm_version}.src/llvm" -G Ninja \
-B "${cast_llvm_root}/release-build" \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_RTTI=ON \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
-DLLVM_ENABLE_PROJECTS="clang;lld;lldb" \
-DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind;compiler-rt"
 # -DLLDB_ENABLE_PYTHON=ON \
  # -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  # -DPython3_INCLUDE_DIR="$CONDA_PREFIX/include/python3.12" \
  # -DPython3_LIBRARY="$CONDA_PREFIX/lib/libpython3.12.so"

cmake --build "${cast_llvm_root}/release-build"

cmake --install "${cast_llvm_root}/release-build" \
      --prefix "${cast_llvm_root}/release-install"

cmake -S "${cast_llvm_root}/llvm-project-${llvm_version}.src/llvm" -G Ninja \
-B "${cast_llvm_root}/debug-build" \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
-DLLVM_ENABLE_RTTI=ON \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
-DCMAKE_C_COMPILER="${cast_llvm_root}/release-install/bin/clang" \
-DCMAKE_CXX_COMPILER="${cast_llvm_root}/release-install/bin/clang++" \
-DCMAKE_LINKER="${cast_llvm_root}/release-install/bin/ld.lld"

cmake --build "${cast_llvm_root}/debug-build"

cmake --install "${cast_llvm_root}/debug-build" \
      --prefix "${cast_llvm_root}/debug-install"
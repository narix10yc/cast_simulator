#! /bin/shell

cast_llvm_root = ($1)
llvm_version = ($2)

cmake -S ${cast_llvm_root}/llvm-project-${llvm_version}.src/llvm -G Ninja \
-B release-build \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_RTTI=ON \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
-DLLVM_ENABLE_PROJECTS="clang;lld;lldb" \
-DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind;compiler-rt"

cmake --build release-build

cmake --install release-build --prefix "${cast_llvm_root}/release-install"

cmake -S ${cast_llvm_root}/llvm-project-${llvm_version}.src/llvm -G Ninja \
-B debug-build \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
-DLLVM_ENABLE_RTTI=ON \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
-DCMAKE_C_COMPILER="${cast_llvm_root}/release-install/bin/clang" \
-DCMAKE_CXX_COMPILER="${cast_llvm_root}/release-install/bin/clang++" \
-DLLVM_USE_LINKER="${cast_llvm_root}/release-install/bin/ld.lld"

cmake --build debug-build

cmake --install debug-build --prefix "debug-install"
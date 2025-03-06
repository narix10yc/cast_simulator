#! /bin/shell


# Download the installer and run the following commands in your terminal
# bash Miniconda3-latest-Linux-x86_64.sh  # For Linux

# # Initialize conda
# conda init

# # Close and reopen the terminal, then create and activate your environment
# conda create -n myenv
# conda activate myenv

# # Install the compilers package
# conda install -c conda-forge compilers
# # unicode/ucnv.h file
# conda install icu

## To remove all cuda packages in conda enviroment
## After activate the enviroment, run
# conda remove $(conda list | grep cuda | awk '{print $1}')
## To also clean cache, run
# conda clean --all

export llvm_root=`pwd`
export llvm_version=19.1.0

cmake -S llvm-project-${llvm_version}.src/llvm -G Ninja \
-B release-build \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_RTTI=ON \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
-DLLVM_ENABLE_PROJECTS="clang;lld;lldb" \
-DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind"
  # -DLLDB_ENABLE_PYTHON=ON \
  # -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  # -DPython3_INCLUDE_DIR="$CONDA_PREFIX/include/python3.12" \
  # -DPython3_LIBRARY="$CONDA_PREFIX/lib/libpython3.12.so"

cmake --build release-build

cmake --install release-build --prefix "release-install"

cmake -S llvm-project-${llvm_version}.src/llvm -G Ninja \
-B debug-build \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
-DLLVM_ENABLE_RTTI=ON \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
-DCMAKE_C_COMPILER="$llvm_root/release-install/bin/clang" \
-DCMAKE_CXX_COMPILER="$llvm_root/release-install/bin/clang++" \
-DLLVM_USE_LINKER="$llvm_root/release-install/bin/ld.lld"

cmake --build debug-build

cmake --install debug-build --prefix "debug-install"
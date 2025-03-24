## Environment Setup

CAST is built with cmake and depends on the [LLVM project](https://github.com/llvm/llvm-project).

CAST is under active developments. The development enviromnent of CAST requires a specific structure of LLVM installation, and we recommand compiling and installing the LLVM project afresh.

### Setup LLVM
Go to the [LLVM release](https://releases.llvm.org/) page and find a happy version (We mostly developed on 17.0.6 and 19.1.0. Newer versions should be backward compatible). Download the file `llvm-project-${version}.src.tar.xz`, which should be around 100 - 150 MiB.

After unzipping we should get a folder `llvm-project-${version}.src`. Find a happy place to store it and run our provided shell script `build_llvm.sh` with 

```
build_llvm.sh <dir> <version>
```
where `<dir>` is the directory in which `llvm-project-${version}.src` is put and `<version>` is the version of the LLVM version.

As a specific example, say we downloaded the LLVM project version 19.1.0 and unzip it into `$HOME/llvm/19.1.0`, with file structure
```
$HOME/llvm/19.1.0
  |- llvm-project-19.1.0.src
```
Then running
```
build_llvm.sh $HOME/llvm/19.1.0 19.1.0
```
will use `cmake`, `ninja`, and the native compiler (in system path) to build two version of LLVM: (1) release build, with build directory `release-build` and install directory `release-install`, and (2). debug build, with build directory `debug-build` and install directory `debug-install`.

During the process, we will first build a release version of LLVM, including the `clang` compiler, the `lld` linker, the `lldb` debugger, and the LLVM standard library and runtime. Then we use the freshly built `clang` compiler to build a debug version. Both builds target towards native and NVPTX (for Nvidia GPU) backends.

After the script finishes, the file structure should look like
```
$HOME/llvm/19.1.0
  |- llvm-project-19.1.0.src
  |- debug-build
  |- debug-install
  |- release-build
  |- release-install
```
The setup is now complete, and you can safely delete `llvm-project-19.1.0.src`, `debug-build`, and `release-build`. 

Now `CAST_LLVM_ROOT` should be set to `$HOME/llvm/19.1.0` in this example.

### CMake Commands
CAST uses CMake to configure build. Supported commands include
- `-DCAST_USE_CUDA=<bool>` Enable CUDA support in CAST. This commands requires LLVM to be build with NVPTX backend.
- `-DCAST_LLVM_ROOT=<path>` Specify the installation directory of the LLVM project. Alternatively you can cache it as an environment variable `cast_llvm_root` or `CAST_LLVM_ROOT`, and our top-level `CMakeLists.txt` will handle it.

Some useful tips:
- When setting `-DCAST_USE_CUDA=True`, CAST needs to find a CUDA installation for the `<cuda.h>` and `<cuda_runtime.h>` headers. You can specify where to find CUDA by `-DCUDAToolkit_ROOT=<path>`. It is often somewhere in `/usr/local/cuda-<version>`. 

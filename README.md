## Update 23th June, 2025
Code refactoring in process. Python binding in progress. Check the yl5619-dev branch. Stay tuned... :)

For the main branch, CPU part does work. To test out, disable `CAST_USE_CUDA` and run
```
ninja unit_test && ./unit_test
```
Some demos also run
```
ninja cpu_bcmk && ./cpu_bcmk
```

To enable python binding, add `-DCAST_PYTHON_BIND=True` when configuring cmake.

## Environment Setup
CAST is built with cmake and depends on the [LLVM project](https://github.com/llvm/llvm-project).

CAST is under active developments. We provide two presets to compile CAST, controlled by enviroment variable.

- Either set enviroment variable `CAST_LLVM_RELEASE_ROOT` to the LLVM installation directory. This is the quickest method.
- Or set enviroment variable `CAST_LLVM_ROOT` to a specific structure (detailed below). This will create a development enviroment for CAST.
- If both are set, `CAST_LLVM_RELEASE_ROOT` will be prioritized. 

The development enviromnent of CAST requires a specific structure of LLVM installation, and we recommand compiling and installing the LLVM project afresh.

## Setup LLVM Afresh
Go to the [LLVM release](https://releases.llvm.org/) page and find a happy version (We mostly developed on 19.1.0. Newer versions should be backward compatible). Download the file `llvm-project-${version}.src.tar.xz`, which should be around 100 - 150 MiB.

### Fastest Setup
After unzipping we should get a folder `llvm-project-${version}.src`. Find a happy place to store it and run our provided shell script `build_llvm.sh` with 

```
./build_llvm.sh <dir> -minimal

# For CPU-only
./build_llvm.sh <dir> -minimal -native-only
```
where `<dir>` is the pull path to `llvm-project-${version}.src`.

Then call
```
export CAST_LLVM_RELEASE_ROOT=<dir>/release-install
```

Build CAST
```
mkdir build-debug
cd build-debug
cmake ..
```
Pass in `-DCAST_USE_CUDA=True` if intending to use CUDA backend.

You may adjust the cmake command according to needs. For example,
```
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug ..
```

### Setup for Developing CAST
The shell script will attempt to do the following things:

1. Use `cmake`, `ninja`, and the system compiler to build a release version of LLVM. Optionally build `clang`, `lld,` and `lldb` when specifying argument `-build-clang`. Optionally build `libc++`, `libc++abi`, and `libunwind`, (the LLVM standard library and runtime) when specifying argument `-build-libc++`.  
2. Build a debug version of LLVM. If `clang` and/or `libc++` is built in the first step, use them to compile this debug version. Otherwise, use the system compiler and runtime. 

Known issue: On certain platforms, `-build-libc++` will contract system library. LLVM will be built and installed successfully, but we have encountered several issues when building CAST. Building clang only is often okay.

Arguments supported:
- `-release-only` Only build and install the release version of LLVM. Default to be false. If false, build a release version and a debug version of LLVM.
- `-build-clang` Enables LLVM projects `clang`, `lld`, and `lldb` (all in release version).
- `-build-libc++` Enables LLVM runtimes `libc++`, `libc++abi`, `libunwind` (all in release version).
- `-minimal` Set `-release-only=True -build-clang=False -build-libc++=False`.
- `-native-only` Default to false. If true, build LLVM with target `native` only. Otherwise, build LLVM with target `native` and `NVPTX`. To use CAST with CUDA backend, `-native-only` must be set to false.


As a specific example, say we downloaded the LLVM project version 19.1.0 and unzip it into `$HOME/llvm/19.1.0`, with file structure
```
$HOME/llvm/19.1.0
  |- llvm-project-19.1.0.src
```
Then running
```
source build_llvm.sh $HOME/llvm/19.1.0
```
will use `cmake`, `ninja`, and the native compiler (in system path) to build two version of LLVM: (1) release build, with build directory `release-build` and install directory `release-install`, and (2). debug build, with build directory `debug-build` and install directory `debug-install`.

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
- `-DCAST_USE_CUDA=<bool>` Enable CUDA support in CAST. This commands requires LLVM to be built with NVPTX backend.

Some useful tips:
- We suggest using the Ninja builder by adding `-GNinja` option.
- When setting `-DCAST_USE_CUDA=True`, cmake needs to find a CUDA installation. You can specify where to find CUDA by `-DCUDAToolkit_ROOT=<path>` (this is not controlled by CAST). It is often found somewhere in `/usr/local/cuda-<version>`. 

### Example
Say we installed LLVM version 19.1.0 in `$HOME/llvm/19.1.0` with file structure
```
$HOME/llvm/19.1.0
  |- debug-install
  |- release-install
```
We can configure the project by entering the `cast_simulator` directory, and run
```
mkdir build-debug && \
cd build-debug && \
cmake -GNinja \
-DCMAKE_BUILD_TYPE=Debug \
-DCAST_USE_CUDA=True \
-DCUDAToolkit_ROOT=/usr/local/cuda-12.3
```

Then run `ninja unit_test && ./unit_test` to run some unit tests, and confirm it compiles and runs correctly.

## Running Demos
We provide several demos for testing purposes.

### Demo: `cpu_bcmk`
To compile it, run `ninja cpu_bcmk`. This will give an executable `cpu_bcmk` in the build directory.

The demo generates each one of 1-qubit, 2-qubit, 3-qubit, and 4-qubit random unitary gates as well as 1-qubit, 2-qubit, 3-qubit, and 4-qubit Hadamard gates, for a total of 8 kernels. Then it applies each of these gates to a (defaulted) 28-qubit statevector and reports the time and effective memory update speed of the run. The experiment runs both single-precision and double-precision modes.

You may try out different thread configuration by
```
# assume in the build directory
CAST_NUM_THREADS=<N> ./cpu_bcmk
```
This command basically sets enviroment variable `CAST_NUM_THREADS` which overrides CAST's query to the number of threads `cast::get_cpu_num_threads()`.


## Running Experiments
Update 23 June, 2025: To add README here
<!-- To use our provided CostModel class to conduct benchmarks for a cost model specialized to your hardware platform follow the steps below: -->

<!-- ### CPU
To perform benchmarks on your CPU, run a command such as the following from inside the build-debug folder:
```
ninja cost_model && ./cost_model -o cost_model.csv -T4 -N 10 -simd-s 2
```
Note: you can substitute the arguments provided, or include additional flags. Run `ninja cost_model && ./cost_model --help` to see the manual.

### GPU:
To run benchmarks on your Nvidia GPU, run the following command from inside the build-debug folder:
```
ninja cost_model_cuda && ./cost_model_cuda -o cost_model_cuda.csv --blockSize 128 -N 10 -workerThreads 8
```
Note: you can customise the flags above, or include additional ones. Run `ninja cost_model_cuda && ./cost_model_cuda --help` to see the manual.

### Apply Cost Model
*Check: have access to qasm files?

```
ninja demo_ptx && ./demo_ptx ../examples/qft/qft-28-cp.qasm -T4 --run-no-fuse --run-naive-fuse --run-adaptive-fuse --run-cuda-fuse --model cost_model.csv --cuda-model cost_model_cuda.csv --blocksize 128
``` -->

## Citing

If you find our work interesting, please consider citing it:
```
@article{lu2025versatile,
  title={Versatile Cross-platform Compilation Toolchain for Schr$\backslash$" odinger-style Quantum Circuit Simulation},
  author={Lu, Yuncheng and Liang, Shuang and Fan, Hongxiang and Guo, Ce and Luk, Wayne and Kelly, Paul HJ},
  journal={arXiv preprint arXiv:2503.19894},
  year={2025}
}
```
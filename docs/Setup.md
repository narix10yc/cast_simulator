## Quick Start
CAST is built with cmake and depends on the [LLVM project](https://github.com/llvm/llvm-project). To compile locally, CAST needs to know where to find LLVM. The straightforward way is to set the environment variable `CAST_LLVM_RELEASE_ROOT` to the LLVM installation directory.

### Mac OS
One straightforward way to get LLVM on Mac OS is via [homebrew](https://brew.sh/). Just run
```
brew install llvm
```
And you should find it at `/opt/homebrew/opt/llvm` (optionally with a version number). Set environment variable `CAST_LLVM_RELEASE_ROOT` to that path and everything should be working.

Optionally you may want to use the LLVM compiler and libc++ that come with the brew install. To enable this, pass in `-DCAST_USE_LLVM_COMPILER=ON` and `-DCAST_USE_LLVM_LIBCXX=ON` when running cmake.

### Linux

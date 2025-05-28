# Code Structure
## Libraries
CAST has several components, each packed into a static library.
- `libtimeit` is a lightweight library that provides several utility functions for timing.
- `libcast_core` consists of core data structures, such as `QuantumGate`, `CircuitGraph`, etc.
- `libcast_cpu` consists of CPU-based simulation methods and utilities, such as the `KernelManagerCPU` class.
- `libcast_cuda` consists of CUDA-based simulation methods and utilities, such as the `KernelManagerCUDA` class.
- `libopenqasm` consists of basic support to the OpenQASM2.0 format of quantum circuits.
- `libcast_utils` consists of utility functions such as formatter, printer, IO colors, etc.
- `libcast` is the top-level library that statically links all aforementioned components.
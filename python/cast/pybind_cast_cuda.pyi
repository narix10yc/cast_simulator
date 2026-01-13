# pybind_cast_cuda.pyi
from __future__ import annotations

from enum import Enum
from .pybind_cast import CircuitGraph, Precision, QuantumGate


class CUDAStatevectorFP32:
    """
    CUDA statevector with FP32 amplitude storage.

    Provides basic inspection and normalization utilities. Indexing returns a
    complex amplitude for the basis state at the given index.
    """

    def __init__(self, num_qubits: int, initialize: bool = True) -> None:
        """
        Parameters
        ----------
        num_qubits:
            Number of qubits in the statevector.
        initialize:
            Whether to initialize internal buffers/state on construction.
        """
        ...

    def __getitem__(self, idx: int) -> complex:
        """
        Return amplitude of computational basis state `|idx⟩`.

        Raises
        ------
        IndexError
            If `idx` is out of range.
        """
        ...

    def initialize(self) -> None:
        """Initialize internal buffers/state."""
        ...

    def randomize(self) -> None:
        """Randomize amplitudes (implementation-defined), then typically normalize."""
        ...

    def num_qubits(self) -> int:
        """Number of qubits in this statevector."""
        ...

    def probability(self, qubit: int) -> float:
        """Return probability of measuring `1` on the given qubit."""
        ...

    def norm(self) -> float:
        """Return L2 norm of the statevector."""
        ...

    def get_device_ptr(self) -> int:
        """Return the underlying CUDA device pointer for the statevector."""
        ...


class CUDAStatevectorFP64:
    """
    CUDA statevector with FP64 amplitude storage.

    Same interface as `CUDAStatevectorFP32` but with double precision.
    """

    def __init__(self, num_qubits: int, initialize: bool = True) -> None: ...
    def __getitem__(self, idx: int) -> complex: ...
    def initialize(self) -> None: ...
    def randomize(self) -> None: ...
    def num_qubits(self) -> int: ...
    def probability(self, qubit: int) -> float: ...
    def norm(self) -> float: ...
    def get_device_ptr(self) -> int: ...


class CUDAMatrixLoadMode(Enum):
    """
    Strategy for loading gate matrix elements in generated CUDA kernels.
    """

    UseMatImmValues = ...
    LoadInDefaultMemSpace = ...
    LoadInConstMemSpace = ...


class CUDAKernelGenConfig:
    """
    Configuration for CUDA kernel generation.

    Controls precision, tolerance thresholds, and matrix loading strategy.
    """

    def __init__(self, precision: Precision) -> None:
        """
        Parameters
        ----------
        precision:
            Numeric precision for the generated kernels.
        """
        ...

    zero_tol: float
    one_tol: float
    matrix_load_mode: CUDAMatrixLoadMode

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about this configuration.

        Parameters
        ----------
        verbose:
            Verbosity level (higher means more detail).
        """
        ...


class CudaKernelHandler:
    """
    Handle to a generated CUDA kernel.
    """

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about this kernel.

        Parameters
        ----------
        verbose:
            Verbosity level (higher means more detail).
        """
        ...

    def get_ptx(self) -> str:
        """
        Return PTX text for the kernel, if available.
        """
        ...


class LaunchTaskHandler:
    """
    Handle to an enqueued CUDA kernel execution.
    """
    def get_exec_time(self) -> float:
        """
        Return the execution time in seconds, if timing is enabled.
        """
        ...


class CUDAKernelManager:
    """
    Manager for generating and launching CUDA kernels.
    """

    def __init__(self) -> None: ...

    def print_info(self, verbose: int = 1) -> None:
        """Print diagnostic information about the kernel manager."""
        ...

    def gen_gate(
        self,
        config: CUDAKernelGenConfig,
        gate: QuantumGate,
        func_name: str = "",
    ) -> CudaKernelHandler:
        """
        Generate a CUDA kernel for a single gate.

        Parameters
        ----------
        config:
            Kernel generation configuration.
        gate:
            Quantum gate to compile.
        func_name:
            Optional kernel name.
        """
        ...

    def enqueue_kernel_execution(
        self,
        kernel: CudaKernelHandler,
    ) -> LaunchTaskHandler:
        """
        Enqueue a kernel for execution.

        Parameters
        ----------
        kernel:
            Kernel handler returned by `gen_gate`.
        """
        ...

    def gen_graph_gates(
        self,
        config: CUDAKernelGenConfig,
        graph: CircuitGraph,
        pool_name: str,
    ) -> None:
        """
        Generate CUDA kernels for gates contained in a circuit graph into a named pool.
        """
        ...

    def sync_compilation(self) -> None:
        """
        Block until all enqueued kernel compilations finish.
        """
        ...

    def sync_kernel_execution(self) -> None:
        """
        Synchronize pending kernel executions.
        """
        ...

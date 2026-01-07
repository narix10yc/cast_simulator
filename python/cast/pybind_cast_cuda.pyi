# pybind_cast_cuda.pyi
from __future__ import annotations

from enum import Enum
from typing import List

from .pybind_cast import CircuitGraph, OptimizerBase, Precision, QuantumGate


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


class CUDAKernelInfo:
    """
    Metadata for a generated CUDA kernel.
    """

    gate: QuantumGate
    precision: Precision

    def has_ptx(self) -> bool:
        """Whether this kernel has PTX attached."""
        ...

    def has_cubin(self) -> bool:
        """Whether this kernel has a CUBIN attached."""
        ...

    def get_ptx(self) -> str:
        """Return PTX text for the kernel (if available)."""
        ...

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about this kernel.

        Parameters
        ----------
        verbose:
            Verbosity level (higher means more detail).
        """
        ...


class CUDAKernelExecutionResult:
    """
    Result metadata for a CUDA kernel execution.
    """

    kernel_name: str

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about this execution result.

        Parameters
        ----------
        verbose:
            Verbosity level (higher means more detail).
        """
        ...

    def get_compile_time(self) -> float:
        """Return kernel compile time (implementation-defined units; typically seconds)."""
        ...

    def get_kernel_time(self) -> float:
        """Return kernel execution time (implementation-defined units; typically seconds)."""
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
    ) -> None:
        """
        Generate a CUDA kernel for a single gate.
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

    def set_launch_config(
        self,
        device_ptr: int,
        num_qubits: int,
        block_size: int = 64,
    ) -> None:
        """
        Configure launch parameters for subsequent kernel launches.
        """
        ...

    def get_kernels_in_pool(self, pool_name: str) -> List[CUDAKernelInfo]:
        """
        List kernels currently stored in a named pool.
        """
        ...

    def launch_kernel_fp32(
        self,
        sv: CUDAStatevectorFP32,
        kernel: CUDAKernelInfo,
    ) -> CUDAKernelExecutionResult:
        """
        Enqueue an FP32 kernel launch for the given statevector.
        """
        ...

    def launch_kernel_fp64(
        self,
        sv: CUDAStatevectorFP64,
        kernel: CUDAKernelInfo,
    ) -> CUDAKernelExecutionResult:
        """
        Enqueue an FP64 kernel launch for the given statevector.
        """
        ...

    def sync_kernel_execution(self, progress_bar: bool = False) -> None:
        """
        Synchronize pending kernel executions.
        """
        ...


class CUDAOptimizer(OptimizerBase):
    """
    CUDA-specific optimizer for circuit graph transformations.
    """

    def __init__(self) -> None: ...

    def print_info(self, verbose: int = 1) -> None:
        """Print diagnostic information about this optimizer."""
        ...

    def enable_fusion(self, enable: bool = True) -> None:
        """Enable or disable fusion passes."""
        ...

    def enable_canonicalization(self, enable: bool = True) -> None:
        """Enable or disable canonicalization passes."""
        ...

    def enable_cfo(self, enable: bool = True) -> None:
        """Enable or disable CFO passes."""
        ...

    def set_sizeonly_fusion_config(self, size: int) -> None:
        """Set fusion configuration based on size thresholds."""
        ...

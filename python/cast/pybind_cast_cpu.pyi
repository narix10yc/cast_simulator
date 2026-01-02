# pybind_cast_cpu.pyi
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from .pybind_cast import CostModel, OptimizerBase, Precision, QuantumGate, CircuitGraph


class CPUSimdWidth(Enum):
    """
    SIMD width configuration for CPU statevector/kernels.

    Values correspond to vector register widths (in bits). Usually we can
    auto-detect the system's supported SIMD width.
    """

    W0 = ...
    W64 = ...
    W128 = ...
    W256 = ...
    W512 = ...


class CPUStatevectorFP32:
    """
    CPU statevector with FP32 amplitude storage.

    Provides basic inspection and normalization utilities. Indexing returns a
    complex amplitude for the basis state at the given index.
    """

    def __init__(self, num_qubits: int, simd_width: CPUSimdWidth) -> None:
        """
        Parameters
        ----------
        num_qubits:
            Number of qubits in the statevector.
        simd_width:
            SIMD width used by the underlying implementation.
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

    @property
    def num_qubits(self) -> int:
        """Number of qubits in this statevector."""
        ...

    def normSquared(self) -> float:
        """Return squared L2 norm of the statevector."""
        ...

    def norm(self) -> float:
        """Return L2 norm of the statevector."""
        ...

    def probability(self, qubit: int) -> float:
        """
        Return probability of measuring `1` on the given qubit (convention per implementation).
        """
        ...

    def initialize(self, num_threads: int = 0) -> None:
        """
        Initialize internal buffers/state.

        Parameters
        ----------
        num_threads:
            Thread count hint. `0` typically means "use default".
        """
        ...

    def normalize(self, num_threads: int = 0) -> None:
        """
        Normalize the statevector to unit norm.

        Parameters
        ----------
        num_threads:
            Thread count hint. `0` typically means "use default".
        """
        ...

    def randomize(self, num_threads: int = 0) -> None:
        """
        Randomize amplitudes (implementation-defined), then typically normalize.

        Parameters
        ----------
        num_threads:
            Thread count hint. `0` typically means "use default".
        """
        ...


class CPUStatevectorFP64:
    """
    CPU statevector with FP64 amplitude storage.

    Same interface as `CPUStatevectorFP32` but with double precision.
    """

    def __init__(self, num_qubits: int, simd_width: CPUSimdWidth) -> None: ...
    def __getitem__(self, idx: int) -> complex: ...
    @property
    def num_qubits(self) -> int: ...
    def normSquared(self) -> float: ...
    def norm(self) -> float: ...
    def probability(self, qubit: int) -> float: ...
    def initialize(self, num_threads: int = 0) -> None: ...
    def normalize(self, num_threads: int = 0) -> None: ...
    def randomize(self, num_threads: int = 0) -> None: ...


class CPUMatrixLoadMode(Enum):
    """
    Strategy for loading gate matrix elements in generated CPU kernels.
    """

    UseMatImmValues = ...
    StackLoadMatElems = ...


class CPUKernelGenConfig:
    """
    Configuration for CPU kernel generation.

    Controls precision, SIMD width, instruction selection (e.g. FMA/FMS),
    tolerance thresholds, and matrix loading strategy.
    """

    def __init__(self, precision: Precision) -> None:
        """
        Parameters
        ----------
        precision:
            Numeric precision for the generated kernels.
        """
        ...

    simd_width: CPUSimdWidth
    precision: Precision
    use_fma: bool
    use_fms: bool
    use_pdep: bool
    zero_tol: float
    one_tol: float
    matrix_load_mode: CPUMatrixLoadMode

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about this configuration.

        Parameters
        ----------
        verbose:
            Verbosity level (higher means more detail).
        """
        ...


class CPUKernelInfo:
    """
    Metadata for a generated CPU kernel.

    Holds the LLVM function name, associated gate, precision, and timing
    statistics (if available).
    """

    precision: Precision
    llvm_func_name: str
    gate: QuantumGate

    @property
    def has_executable(self) -> bool:
        """Whether this kernel currently has a compiled executable attached."""
        ...

    def get_jit_time(self) -> float:
        """Return JIT compilation time (implementation-defined units; typically seconds)."""
        ...

    def get_exec_time(self) -> float:
        """Return execution time measured/recorded by the system (if available)."""
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


class CPUKernelManager:
    """
    Manager for generating, compiling, and executing CPU kernels.

    Typical workflow:
    1) Create a `CPUKernelGenConfig`
    2) `gen_gate(...)` or `gen_graph_gates(...)`
    3) `compile_*`
    4) `apply_kernel_fp32/fp64(...)`
    """

    def __init__(self) -> None: ...

    def print_info(self, verbose: int = 1) -> None:
        """Print diagnostic information about the kernel manager."""
        ...

    def gen_gate(
        self,
        config: CPUKernelGenConfig,
        gate: QuantumGate,
        func_name: str = "",
    ) -> Optional[CPUKernelInfo]:
        """
        Generate a CPU kernel for a single gate.

        Parameters
        ----------
        config:
            Kernel generation configuration.
        gate:
            Gate to generate a kernel for.
        func_name:
            Optional function name hint.

        Returns
        -------
        CPUKernelInfo | None
            The most recently generated kernel info in the default pool,
            or None if none is available (should be rare).
        """
        ...

    def gen_graph_gates(
        self,
        config: CPUKernelGenConfig,
        graph: CircuitGraph,
        pool_name: str,
    ) -> None:
        """
        Generate CPU kernels for gates contained in a circuit graph into a named pool.

        Parameters
        ----------
        config:
            Kernel generation configuration.
        graph:
            Circuit graph whose gates will be lowered into kernels.
        pool_name:
            Name of the kernel pool to populate.
        """
        ...

    def get_ir(self, func_name: str) -> str:
        """
        Return LLVM IR for a generated function.

        Parameters
        ----------
        func_name:
            LLVM function name.

        Returns
        -------
        str
            LLVM IR text for the function.
        """
        ...

    def get_kernel_by_name(self, func_name: str) -> Optional[CPUKernelInfo]:
        """
        Look up a kernel by its LLVM function name.

        Returns
        -------
        CPUKernelInfo | None
            Kernel info if found, otherwise None.
        """
        ...

    def compile_default_pool(self, opt_level: int) -> None:
        """
        Compile kernels in the default pool.

        Parameters
        ----------
        opt_level:
            Optimization level as an integer (mapped internally to LLVM levels).
        """
        ...

    def compile_pool(self, pool_name: str, opt_level: int) -> None:
        """
        Compile kernels in a named pool.

        Parameters
        ----------
        pool_name:
            Pool to compile.
        opt_level:
            Optimization level as an integer.
        """
        ...

    def compile_all_pools(self, opt_level: int = 1, verbose: int = 0) -> None:
        """
        Compile kernels in all pools.

        Parameters
        ----------
        opt_level:
            Optimization level as an integer.
        verbose:
            Verbosity level for compilation diagnostics.
        """
        ...

    def get_kernels_in_pool(self, pool_name: str) -> List[CPUKernelInfo]:
        """
        List kernels currently stored in a named pool.

        Parameters
        ----------
        pool_name:
            Pool name.

        Returns
        -------
        list[CPUKernelInfo]
            Kernel infos in that pool.
        """
        ...

    def apply_kernel_fp32(
        self,
        sv: CPUStatevectorFP32,
        gate: CPUKernelInfo,
        num_threads: int = 1,
    ) -> None:
        """
        Apply a compiled FP32 kernel to an FP32 statevector.

        Parameters
        ----------
        sv:
            Statevector to mutate in-place.
        gate:
            Kernel info to execute (must have FP32 precision).
        num_threads:
            Number of threads to use.
        """
        ...

    def apply_kernel_fp64(
        self,
        sv: CPUStatevectorFP64,
        gate: CPUKernelInfo,
        num_threads: int = 1,
    ) -> None:
        """
        Apply a compiled FP64 kernel to an FP64 statevector.

        Parameters
        ----------
        sv:
            Statevector to mutate in-place.
        gate:
            Kernel info to execute (must have FP64 precision).
        num_threads:
            Number of threads to use.
        """
        ...


class CPUPerformanceCache:
    """
    Cache of measured CPU kernel performance.

    Can be loaded from/saved to disk and can run experiments to populate
    performance measurements.
    """

    def __init__(self) -> None: ...

    def load_from_file(self, filename: str) -> None:
        """
        Load cache data from a file.

        Parameters
        ----------
        filename:
            Path to a cache file.
        """
        ...

    def run_experiments(
        self,
        cpu_config: CPUKernelGenConfig,
        n_qubits: int,
        n_threads: int,
        n_runs: int,
        verbose: int = 1,
    ) -> None:
        """
        Run performance experiments and populate the cache.

        Parameters
        ----------
        cpu_config:
            Kernel generation configuration used for experiments.
        n_qubits:
            Number of qubits for the benchmark statevector(s).
        n_threads:
            Number of threads to use for execution.
        n_runs:
            Number of repeated runs per benchmark.
        verbose:
            Verbosity level.
        """
        ...

    def raw(self) -> str:
        """
        Serialize the cache to a string.

        Returns
        -------
        str
            Cache contents in an implementation-defined text format.
        """
        ...

    def save(self, filename: str, overwrite: bool) -> None:
        """
        Save cache to a file.

        Parameters
        ----------
        filename:
            Output file path.
        overwrite:
            Whether to overwrite existing files.
        """
        ...


class CPUCostModel(CostModel):
    """
    CPU-specific cost model built from a `CPUPerformanceCache`.

    Used to evaluate/estimate costs for circuit optimization decisions.
    """

    def __init__(
        self,
        query_num_threads: int,
        query_precision: Precision,
        zero_tol: float = 1e-8,
    ) -> None:
        """
        Parameters
        ----------
        query_num_threads:
            Thread count assumed when querying costs.
        query_precision:
            Precision assumed when querying costs.
        zero_tol:
            Numerical tolerance used for sparsity/zero decisions.
        """
        ...

    def print_info(self, verbose: int = 1) -> None:
        """Print diagnostic information about this cost model."""
        ...

    def show_entries(self, n_lines: int = 5) -> None:
        """
        Print a short view of cache/cost entries.

        Parameters
        ----------
        n_lines:
            Number of entries/lines to show.
        """
        ...

    def clear_cache(self) -> None:
        """Clear any cached state inside the cost model."""
        ...

    def load_cache(self, cache: CPUPerformanceCache) -> None:
        """
        Load a performance cache into the cost model.

        Parameters
        ----------
        cache:
            Performance cache to ingest.
        """
        ...


class CPUOptimizer(OptimizerBase):
    """
    CPU-specific circuit optimizer.

    Inherits `run_circuit(...)` and `run_circuitgraph(...)` from `OptimizerBase`
    and provides CPU-specific optimization toggles.
    """

    def __init__(self) -> None: ...

    def print_info(self, verbose: int = 1) -> None:
        """Print diagnostic information about this optimizer."""
        ...

    def enable_fusion(self, enable: bool = True) -> None:
        """
        Enable/disable fusion passes.

        Parameters
        ----------
        enable:
            Whether to enable fusion.
        """
        ...

    def enable_canonicalization(self, enable: bool = True) -> None:
        """
        Enable/disable canonicalization passes.

        Parameters
        ----------
        enable:
            Whether to enable canonicalization.
        """
        ...

    def enable_cfo(self, enable: bool = True) -> None:
        """
        Enable/disable CFO (implementation-defined) passes.

        Parameters
        ----------
        enable:
            Whether to enable CFO.
        """
        ...

    def set_sizeonly_fusion_config(self, size: int) -> None:
        """
        Configure size-only fusion.

        Parameters
        ----------
        size:
            Size parameter for size-only fusion (meaning per implementation).
        """
        ...
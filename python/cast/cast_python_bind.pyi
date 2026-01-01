import numpy as np

def parse_circuitgraph_from_qasm_file(file_name: str) -> CircuitGraph:
  """
  Parses a circuit graph from a QASM file.
  """
  ...
  
class ComplexSquareMatrix:
  """
  Represents a complex square matrix. Matrix data is arranged in row-major
  order with separate real and imaginary parts.
  """
  @property
  def edgesize(self) -> int:
    """
    Read-only. Edge size of the square matrix.
    """
    ...
  
  def to_numpy(self) -> np.ndarray:
    """
    Converts the matrix to a NumPy array. This method creates a copy of the
    matrix data.
    """
    ...
  
class GateMatrix:
  """
  Represents a gate matrix. This is the base class of all gate matrices.
  There are currently two types of gate matrices:
  - ScalarGateMatrix: Represented by a complex square matrix.
  - UnitaryPermutationGateMatrix: Represented by a unitary permutation matrix.
  """
  ...
  
class ScalarGateMatrix(GateMatrix):
  """
  Represents a scalar gate matrix, which is represented by a complex square
  matrix.
  """
  @property
  def matrix(self) -> ComplexSquareMatrix:
    """
    Read-only. The complex square matrix representing the gate matrix.
    """
    ...

class QuantumGate:
  """
  Represents a quantum gate in a quantum circuit. This class represents the base
  class of all quantum gates. There are currently three types of quantum gates:
  - StandardGate: Represented by a gate matrix and an optional noise channel.
  - SuperopQuantumGate: Represented by a superoperator.
  - ParametrizedQuantumGate: Represented by a parameterized gate matrix and an
    optional noise channel (not in use yet).
  """

  @property
  def num_qubits(self) -> int:
    """
    Read-only. Number of qubits.
    """
    ...
  
  @property
  def qubits(self) -> list[int]:
    """
    Read-only. List of qubit indices the gate acts on.
    """
    ...

class StandardQuantumGate(QuantumGate):
  """
  Standard quantum gates are representated by a gate matrix and an optional 
  noise channel.
  """

  @staticmethod
  def random_unitary(qubits: list[int]) -> StandardQuantumGate:
    """
    Generates a random unitary gate acting on the given qubits.
    """
    ...
    
  @property
  def gatematrix(self) -> ScalarGateMatrix:
    """
    Read-only. The gate matrix of the quantum gate.
    """
    ...


class SuperopQuantumGate(QuantumGate):
  """
  Represents a quantum gate in its superoperator form.
  """
  ...
  
class Circuit:
  def __str__(self) -> str:
    ...

  def get_all_circuit_graphs(self) -> list[CircuitGraph]:
    """Returns all circuit graphs in the circuit."""
    ...
  
class CircuitGraph:
  def __str__(self) -> str:
    ...

  def print_info(self, verbose = 1) -> None:
    """Prints information about the circuit graph."""
    ...

  def visualize(self) -> None:
    """Visualizes the circuit graph."""
    ...

  def get_all_gates(self) -> list[QuantumGate]:
    """Returns all gates in the circuit graph in order."""
    ...

from enum import Enum
class CPUMatrixLoadMode(Enum):
  """
  Enum for matrix load types.
  """
  UseMatImmValues   : CPUMatrixLoadMode
  StackLoadMatElems : CPUMatrixLoadMode

class CPUSimdWidth(Enum):
  """
  Enum for SIMD widths.
  """
  W64  : CPUSimdWidth
  W128 : CPUSimdWidth
  W256 : CPUSimdWidth
  W512 : CPUSimdWidth

class Precision(Enum):
  """
  Enum for precision types.
  """
  FP32 : Precision
  FP64 : Precision
  Unknown : Precision
  
class CPUKernelGenConfig:
  """
  Configuration for CPU kernel generation.
  """
  simd_width : int
  precision : int
  use_fma : bool
  use_fms : bool
  use_pdep : bool
  zero_tol : float
  one_tol : float
  matrix_load_mode : CPUMatrixLoadMode

  def __init__(self, precision: Precision) -> None:
    ...
  
  def print_info(self, verbose: int = 1) -> None:
    """
    Prints information about the CPU kernel generation configuration.
    """
    ...

class CPUKernelInfo:
  """
  Information about a CPU kernel. This is a read-only class and should not be 
  returned by other methods, such as CPUKernelManager.get_kernels().
  """
  
  def print_info(self, verbose: int = 1) -> None:
    """Prints information about the CPU kernel."""
    ...

  @property
  def precision(self) -> int:
    ...

  @property
  def llvm_func_name(self) -> str:
    ...

  @property
  def gate(self) -> QuantumGate:
    """
    Returns the quantum gate associated with this kernel.
    """
    ...
  
  def has_executable(self) -> bool:
    """
    Returns True if the kernel has been JIT compiled and is executable.
    """
    ...
    
  def get_jit_time(self) -> float:
    """
    Get JIT time in seconds.
    """
    ...
    
  def get_exec_time(self) -> float:
    """
    Get execution time in seconds.
    """
    ...

class CPUKernelManager:
  """
  Manages CPU kernel generation and execution.
  """
  def __init__(self) -> None:
    ...
  
  def print_info(self, verbose: int = 1) -> None:
    """
    Prints information about this CPU kernel manager.
    """
    ...

  def get_kernel_by_name(self, llvm_func_name: str) -> CPUKernelInfo | None:
    """
    Returns the CPU kernel with the given LLVM function name, or None if it 
    does not exist.
    """
    ...

  def gen_gate(self,
               config: CPUKernelGenConfig,
               gate: QuantumGate,
               func_name: str = "") -> CPUKernelInfo | None:
    """
    Generates a CPU kernel for the given quantum gate. The generated kernel is
    put into the default pool. 
    
    - func_name: Name of the generated LLVM function. If empty, a unique name
      will be generated automatically.
    """
    ...
  
  def gen_graph_gates(self, 
                      config: CPUKernelGenConfig, 
                      graph: CircuitGraph,
                      graph_name: str) -> None:
    """
    Generates CPU kernels for all gates in the given circuit graph.
    graph_name should be unique for every circuit graph given to this method.
    """
    ...
  
  def init_jit(self,
               num_threads: int = 1,
               opt_level: int = 1,
               use_lazy_jit: bool = False,
               verbose: int = 0) -> None:
    """
    Initializes the JIT compiler with the given configuration.
    - num_threads: Number of threads to use for JIT compilation.
    - optimization_level: Optimization level for JIT compilation. This option
      corresponds to llvm::OptimizationLevel. For example, 0 means 
      llvm::OptimizationLevel::O0, 1 means llvm::OptimizationLevel::O1, etc
    - use_lazy_jit: If True, uses lazy JIT compilation.
    - verbose: Show progress bar if > 0.
    """
    ...

  def apply_kernel_f32(self,
                       sv: CPUStatevectorFP32,
                       kenrel: CPUKernelInfo,
                       num_threads: int) -> None:
    """
    Apply a single-precision kernel.
    """
    ...

  def apply_kernel_f64(self,
                       sv: CPUStatevectorFP64,
                       kenrel: CPUKernelInfo,
                       num_threads: int) -> None:
    """
    Apply a double-precision kernel.
    """
    ...

class CPUStatevectorFP32:
  """
  Represents a statevector in single precision (float).
  """
  def __init__(self, num_qubits: int, simd_s: int) -> None:
    """
    Initializes the statevector with the given number of qubits and SIMD size.
    """
    ...
  
  def __getitem__(self, index: int) -> complex:
    """
    Returns the amplitude at the given index.
    """
    ...

  @property
  def num_qubits(self) -> int:
    """
    Returns the number of qubits in the statevector.
    """
    ...
    
  def initialize(self) -> None:
    """
    Initializes the statevector to the |0...0> state.
    """
    ...
    
  def randomize(self, num_threads: int = 1) -> None:
    """
    Randomizes the statevector.
    """
    ...
  
class CPUStatevectorFP64:
  """
  Represents a statevector in double precision (double).
  """
  def __init__(self, num_qubits: int, simd_s: int) -> None:
    """
    Initializes the statevector with the given number of qubits and SIMD size.
    """
    ...
  
  def __getitem__(self, index: int) -> complex:
    """
    Returns the amplitude at the given index.
    """
    ...

  @property
  def num_qubits(self) -> int:
    """
    Returns the number of qubits in the statevector.
    """
    ...

  def initialize(self) -> None:
    """
    Initializes the statevector to the |0...0> state.
    """
    ...

  def randomize(self, num_threads: int = 1) -> None:
    """
    Randomizes the statevector.
    """
    ...
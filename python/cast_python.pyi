def parse_circuit_from_qasm_file(file_name: str) -> Circuit:
  """Parses a circuit from a QASM file."""
  ...

class QuantumGate:
  """
  Represents a quantum gate in a quantum circuit. This class is binded from a
  shared pointer in C++ so should not be instantiated directly.
  """
  def get_info(self, verbose=1) -> str:
    """Returns information about the quantum gate."""
    ...
  
  def get_superop_gate(self) -> SuperopQuantumGate:
    """
    Returns the superoperator representation of this quantum gate.
    """
    ...

  @property
  def num_qubits(self) -> int:
    """
    Returns the number of qubits this quantum gate acts on.
    """
    ...
  
  @property
  def qubits(self) -> list[int]:
    """
    Returns the qubits this quantum gate acts on.
    """
    ...

class StandardGate(QuantumGate):
  """
  Standard quantum gates are representated by a gate matrix and an optional 
  noise channel.
  """
  def set_noise_spc(self, p: float) -> None:
    """
    Set the noise to be the Symmetric Pauli Channel with probability p. 
    Effectively this is saying we have a probability of p/3 to apply each of
    the Pauli gates (X, Y, Z) as error and a probability of 1-p to have no 
    error.
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

  def get_visualization(self, verbose=1) -> str:
    """Returns a visualization of the circuit graph."""
    ...

  def get_all_gates(self) -> list[QuantumGate]:
    """Returns all gates in the circuit graph in order."""
    ...

from enum import Enum
class MatrixLoadMode(Enum):
  """
  Enum for matrix load types.
  """
  UseMatImmValues   : MatrixLoadMode
  StackLoadMatElems : MatrixLoadMode

# Also expose the enum values directly
UseMatImmValues   : MatrixLoadMode
StackLoadMatElems : MatrixLoadMode

class CPUKernelGenConfig:
  """
  Configuration for CPU kernel generation.
  """
  simd_s : int
  precision : int
  useFMA : bool
  useFMS : bool
  usePDEP : bool
  zeroTol : float
  oneTol : float
  matrixLoadMode : MatrixLoadMode

  def __init__(self) -> None:
    ...
  
  def get_info(self) -> str:
    """
    Get the information about the CPU kernel generation configuration, returned
    as a string.
    """
    ...

class CPUKernelInfo:
  """
  Information about a CPU kernel. This is a read-only class and should not be 
  returned by other methods, such as CPUKernelManager.get_kernels().
  """
  @property
  def precision(self) -> int:
    ...

  @property
  def llvm_func_name(self) -> str:
    ...

  @property
  def gate(self) -> QuantumGate:
    ...
  
  @property
  def executable(self) -> bool:
    ...

  def get_info(self) -> str:
    """Returns information about the CPU kernel."""
    ...

class CPUKernelManager:
  """
  Manages CPU kernel generation and execution.
  """
  def __init__(self) -> None:
    ...
  
  def get_info(self) -> str:
    """
    Returns information about this CPU kernel manager.
    """
    ...

  def get_kernel_by_name(self, llvm_func_name: str) -> CPUKernelInfo | None:
    """
    Returns the CPU kernel with the given LLVM function name, or None if it 
    does not exist.
    """
    ...

  def gen_cpu_gate(self, 
                   config: CPUKernelGenConfig, 
                   gate: QuantumGate,
                   func_name: str = "") -> None:
    """
    Generates a CPU kernel for the given quantum gatex. This method throws 
    RuntimeError if there already exists kernels with name func_name.
    """
    ...
  
  def gen_cpu_gates_from_graph(self, 
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
                       sv: CPUStatevectorF32,
                       kenrel: CPUKernelInfo,
                       num_threads: int) -> None:
    """
    Apply a single-precision kernel.
    """
    ...

  def apply_kernel_f64(self,
                       sv: CPUStatevectorF64,
                       kenrel: CPUKernelInfo,
                       num_threads: int) -> None:
    """
    Apply a double-precision kernel.
    """
    ...

class CPUStatevectorF32:
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
  
class CPUStatevectorF64:
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
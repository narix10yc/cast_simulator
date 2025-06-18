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

  def get_all_gates(self) -> list[QuantumGate]
    """Returns all gates in the circuit graph in order."""
    ...
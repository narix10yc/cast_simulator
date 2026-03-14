from .pybind_cast import *
from .pybind_cast_cpu import *

try:
    from .pybind_cast_cuda import *
except ImportError:
    print("CUDA bindings not found. CUDA backend will be unavailable.")

def random_unitary(qubits: list[int]) -> StandardQuantumGate:
    """
    Generates a random unitary gate acting on the given qubits.
    Internally this is just `StandardQuantumGate.random_unitary`.
    """
    return StandardQuantumGate.random_unitary(qubits)
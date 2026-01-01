from .cast_python_bind import *

def random_unitary(qubits: list[int]) -> StandardQuantumGate:
    """
    Generates a random unitary gate acting on the given qubits.
    Internally this is just `StandardQuantumGate.random_unitary`.
    """
    return StandardQuantumGate.random_unitary(qubits)
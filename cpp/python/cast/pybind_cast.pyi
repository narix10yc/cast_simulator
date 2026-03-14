# pybind_cast.pyi
from __future__ import annotations

from enum import Enum
from typing import List, Sequence

import numpy as np


class ComplexSquareMatrix:
    """
    Dense complex square matrix used to represent quantum operators from C++.
    It stores a square matrix of size (2^n x 2^n) implicitly via edge size.
    Real and imaginary parts are stored as separate contiguous arrays, so we do
    not expose a direct ndarray view. We may call `to_numpy` to create a copy as
    a NumPy array (over complex numbers) if needed.
    """

    def __init__(self, edge_size: int) -> None:
        """
        Create a complex square matrix.

        Parameters
        ----------
        edge_size:
            The dimension of the matrix (number of rows / columns).
        """
        ...

    @property
    def edgesize(self) -> int:
        """
        Size of one dimension of the square matrix.
        """
        ...

    def to_numpy(self) -> np.ndarray:
        """
        Convert the matrix into a NumPy array.

        Returns
        -------
        numpy.ndarray
            A dense complex-valued NumPy array with shape
            (edgesize, edgesize).
        """
        ...


class GateMatrix:
    """
    Abstract representation of a gate matrix acting on one or more qubits.
    """

    @property
    def num_qubits(self) -> int:
        """
        Number of qubits this gate matrix acts on.
        """
        ...

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about the gate matrix.
        """
        ...


class ScalarGateMatrix(GateMatrix):
    """
    Concrete gate matrix represented explicitly as a dense complex matrix.
    """

    @property
    def matrix(self) -> ComplexSquareMatrix:
        """
        Underlying dense complex matrix.
        """
        ...

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about the scalar gate matrix.
        """
        ...


class Precision(Enum):
    """
    Numerical precision used in simulation or gate representation.
    """

    FP32 = ...
    FP64 = ...
    Unknown = ...


class QuantumGate:
    """
    Abstract quantum gate acting on a fixed set of qubits.

    A QuantumGate may represent a unitary, superoperator, or other
    quantum operation.
    """

    @property
    def num_qubits(self) -> int:
        """
        Number of qubits this gate acts on.
        """
        ...

    @property
    def qubits(self) -> List[int]:
        """
        Indices of qubits the gate acts on.
        """
        ...

    def op_count(self, zero_tol: float) -> int:
        """
        Estimate the number of non-zero operations in the gate.

        Parameters
        ----------
        zero_tol:
            Threshold below which values are treated as zero.

        Returns
        -------
        int
            Operation count estimate.
        """
        ...

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about the gate.
        """
        ...


class StandardQuantumGate(QuantumGate):
    """
    Standard unitary quantum gate.

    This includes common single- and multi-qubit unitary gates.
    """

    @staticmethod
    def random_unitary(qubits: Sequence[int]) -> StandardQuantumGate:
        """
        Construct a random unitary gate acting on the given qubits.
        """
        ...

    @staticmethod
    def I1(q: int) -> StandardQuantumGate:
        """
        Identity gate acting on a single qubit.
        """
        ...

    @staticmethod
    def H(q: int) -> StandardQuantumGate:
        """
        Hadamard gate acting on a single qubit.
        """
        ...

    def set_noise_spc(self, p: float) -> None:
        """
        Attach a stochastic noise specification to the gate.

        Parameters
        ----------
        p:
            Noise probability parameter.
        """
        ...

    @property
    def gatematrix(self) -> GateMatrix:
        """
        Matrix representation of this gate.
        """
        ...

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about the standard quantum gate.
        """
        ...


class SuperopQuantumGate(QuantumGate):
    """
    Quantum gate represented as a superoperator.

    Typically used for noisy or non-unitary operations.
    """

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about the superoperator gate.
        """
        ...


class CircuitGraph:
    """
    Graph-based representation of a quantum circuit.

    Nodes correspond to gates and edges encode dependencies.
    """

    def __str__(self) -> str:
        """
        String representation of the circuit graph.
        """
        ...

    def num_qubits(self) -> int:
        """
        Number of qubits used by this circuit graph.
        """
        ...

    def get_all_gates(self) -> List[QuantumGate]:
        """
        Return all gates contained in the circuit graph.
        """
        ...

    def print_info(self, verbose: int = 1) -> None:
        """
        Print diagnostic information about the circuit graph.
        """
        ...


class Circuit:
    """
    High-level quantum circuit container.

    A Circuit may contain one or more CircuitGraphs, e.g. after
    decomposition or optimization.
    """

    def __str__(self) -> str:
        """
        String representation of the circuit.
        """
        ...

    def get_all_circuit_graphs(self) -> List[CircuitGraph]:
        """
        Return all circuit graphs associated with this circuit.
        """
        ...


def parse_circuitgraph_from_qasm_file(path: str) -> CircuitGraph:
    """
    Parse a QASM file and construct a CircuitGraph.

    Parameters
    ----------
    path:
        Path to the QASM file.

    Returns
    -------
    CircuitGraph
        Parsed circuit graph.
    """
    ...


class CostModel:
    """
    Abstract cost model used for circuit evaluation or optimization.
    """
    ...


class OptimizerBase:
    """
    Base class for circuit optimizers.
    """

    def run_circuit(self, circuit: Circuit, verbose: int = 1) -> None:
        """
        Run the optimizer on a full circuit.
        """
        ...

    def run_circuitgraph(
        self, circuitgraph: CircuitGraph, verbose: int = 1
    ) -> None:
        """
        Run the optimizer on a single circuit graph.
        """
        ...
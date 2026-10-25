"""Operator solutions made of Pauli observables.

A quantum (operator) solution of Ax = b over Z_2 assigns to each variable x_j a
Hermitian unitary X_j such that

1. the observables of each equation pairwise commute, and
2. the product of the observables of equation i equals (-1)^{b_i} I.

Such a solution on n qubits gives a perfect strategy for the game G(A, b) when
Alice and Bob share n maximally entangled qubit pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Dict, List

import numpy as np
from qiskit.quantum_info import Pauli

from .system import LinearSystem

ATOL = 1e-9


@dataclass(frozen=True)
class PauliSolution:
    """Signed Pauli labels for each variable, e.g. {0: "XI", 6: "-XZ"}.

    Labels use Qiskit's convention: the rightmost character acts on qubit 0.
    """

    labels: Dict[int, str]

    @property
    def num_qubits(self) -> int:
        return Pauli(next(iter(self.labels.values()))).num_qubits

    def pauli(self, j: int) -> Pauli:
        return Pauli(self.labels[j])

    def matrix(self, j: int) -> np.ndarray:
        return self.pauli(j).to_matrix()


def check_operator_solution(system: LinearSystem, solution: PauliSolution) -> List[str]:
    """Return a list of problems; an empty list means the solution is valid."""
    problems: List[str] = []
    missing = [j for j in range(system.num_variables) if j not in solution.labels]
    if missing:
        return [f"no observable given for variables {missing}"]
    if len({solution.pauli(j).num_qubits for j in solution.labels}) != 1:
        return ["observables act on different numbers of qubits"]

    dim = 2 ** solution.num_qubits
    identity = np.eye(dim)
    for j in range(system.num_variables):
        m = solution.matrix(j)
        if not np.allclose(m, m.conj().T, atol=ATOL):
            problems.append(f"{system.name(j)} is not Hermitian")
        if not np.allclose(m @ m, identity, atol=ATOL):
            problems.append(f"{system.name(j)} does not square to the identity")

    for i in range(system.num_equations):
        vars_i = system.support(i)
        for a in vars_i:
            for c in vars_i:
                if a < c and not solution.pauli(a).commutes(solution.pauli(c)):
                    problems.append(f"{system.name(a)} and {system.name(c)} in equation {i} do not commute")
        prod = reduce(lambda acc, j: acc @ solution.matrix(j), vars_i, identity)
        target = (-1) ** system.b[i] * identity
        if not np.allclose(prod, target, atol=ATOL):
            problems.append(f"product of equation {i} is not (-1)^{system.b[i]} I")
    return problems

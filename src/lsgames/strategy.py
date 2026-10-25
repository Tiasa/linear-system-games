"""Perfect quantum strategies for linear system games, as Qiskit circuits.

Given a Pauli operator solution {X_j} on n qubits, Alice and Bob share n
maximally entangled pairs |Phi> = 2^{-n/2} sum_x |x>_A |x>_B.

* On equation i, Alice measures the commuting observables {X_k : k in V(i)}
  jointly and answers with their eigenvalues (+1 -> 0, -1 -> 1). The product
  constraint guarantees her answer satisfies equation i.
* On variable j, Bob measures the transpose X_j^T. Because
  (M (x) I)|Phi> = (I (x) M^T)|Phi>, his outcome always matches Alice's value
  for x_j.

Joint measurements are implemented by rotating each register into a common
eigenbasis and measuring in the computational basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from .operators import PauliSolution, check_operator_solution
from .system import LinearSystem, Question

ATOL = 1e-8


@dataclass(frozen=True)
class MeasurementBasis:
    """A unitary basis change plus the answer attached to each outcome."""

    basis: np.ndarray  # columns are the measurement basis vectors
    answers: Tuple[Dict[int, int], ...]  # outcome index -> {variable: bit}


def _eigenvalue_bit(vec: np.ndarray, observable: np.ndarray) -> int:
    value = np.vdot(vec, observable @ vec).real
    if abs(abs(value) - 1) > ATOL:
        raise ValueError("basis vector is not an eigenvector of the observable")
    return 0 if value > 0 else 1


def _common_eigenbasis(observables: List[np.ndarray], seed: int) -> np.ndarray:
    """Orthonormal common eigenbasis of commuting Hermitian matrices.

    A generic real combination of commuting observables has eigenspaces that
    refine every joint eigenspace, so its eigenvectors diagonalize all of them.
    """
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(1.0, 2.0, size=len(observables))
    combination = sum(c * m for c, m in zip(coefficients, observables))
    _, vectors = np.linalg.eigh(combination)
    return vectors


class QuantumStrategy:
    """The entangled strategy defined by a Pauli operator solution."""

    def __init__(self, system: LinearSystem, solution: PauliSolution, seed: int = 7):
        problems = check_operator_solution(system, solution)
        if problems:
            raise ValueError("not an operator solution: " + "; ".join(problems))
        self.system = system
        self.solution = solution
        self.num_qubits = solution.num_qubits
        self._seed = seed
        self._alice: Dict[int, MeasurementBasis] = {}
        self._bob: Dict[int, MeasurementBasis] = {}

    # Measurements -------------------------------------------------------

    def alice_measurement(self, i: int) -> MeasurementBasis:
        """Joint measurement of the observables in equation i."""
        if i not in self._alice:
            variables = self.system.support(i)
            mats = [self.solution.matrix(k) for k in variables]
            vectors = _common_eigenbasis(mats, self._seed + i)
            answers = tuple(
                {k: _eigenvalue_bit(vectors[:, s], m) for k, m in zip(variables, mats)}
                for s in range(vectors.shape[1])
            )
            self._alice[i] = MeasurementBasis(vectors, answers)
        return self._alice[i]

    def bob_measurement(self, j: int) -> MeasurementBasis:
        """Measurement of the transposed observable for variable j."""
        if j not in self._bob:
            transposed = self.solution.matrix(j).T
            _, vectors = np.linalg.eigh(transposed)
            answers = tuple({j: _eigenvalue_bit(vectors[:, t], transposed)} for t in range(vectors.shape[1]))
            self._bob[j] = MeasurementBasis(vectors, answers)
        return self._bob[j]

    # Circuits -----------------------------------------------------------

    def circuit(self, question: Question, measure: bool = True) -> QuantumCircuit:
        """Circuit for one round: entangle, rotate each register, measure.

        Alice holds qubits 0..n-1 and Bob holds n..2n-1. With measurement,
        a single 2n-bit register records Alice's outcome in the low n bits
        and Bob's in the high n bits.
        """
        i, j = question
        n = self.num_qubits
        alice, bob = QuantumRegister(n, "alice"), QuantumRegister(n, "bob")
        qc = QuantumCircuit(alice, bob, name=f"eq{i}_var{j}")
        for k in range(n):
            qc.h(alice[k])
            qc.cx(alice[k], bob[k])
        qc.append(UnitaryGate(self.alice_measurement(i).basis.conj().T, label=f"A_eq{i}"), list(alice))
        qc.append(UnitaryGate(self.bob_measurement(j).basis.conj().T, label=f"B_{self.system.name(j)}"), list(bob))
        if measure:
            out = ClassicalRegister(2 * n, "out")
            qc.add_register(out)
            qc.measure(list(alice) + list(bob), list(out))
        return qc

    def wins(self, question: Question, alice_outcome: int, bob_outcome: int) -> bool:
        i, j = question
        a = self.alice_measurement(i).answers[alice_outcome]
        c = self.bob_measurement(j).answers[bob_outcome][j]
        return self.system.satisfies(i, a) and a[j] == c

    # Win probabilities --------------------------------------------------

    def exact_win_probability(self, question: Question) -> float:
        """Win probability on one question pair, from the statevector."""
        n = self.num_qubits
        probs = Statevector(self.circuit(question, measure=False)).probabilities()
        grid = probs.reshape(2**n, 2**n)  # grid[bob_outcome, alice_outcome]
        return float(sum(grid[t, s] for t in range(2**n) for s in range(2**n) if self.wins(question, s, t)))

    def win_probability(self, distribution: Optional[Dict[Question, Fraction]] = None) -> float:
        """Exact overall win probability under a question distribution."""
        pi = distribution or self.system.uniform_distribution()
        return sum(float(p) * self.exact_win_probability(q) for q, p in pi.items())

    def sampled_win_rate(
        self,
        shots: int = 1000,
        seed: Optional[int] = None,
        distribution: Optional[Dict[Question, Fraction]] = None,
    ) -> float:
        """Estimate the win probability by sampling each question pair's circuit."""
        pi = distribution or self.system.uniform_distribution()
        questions = list(pi)
        n = self.num_qubits
        sampler = StatevectorSampler(seed=seed)
        results = sampler.run([self.circuit(q) for q in questions], shots=shots).result()
        rate = 0.0
        for q, result in zip(questions, results):
            counts = result.data.out.get_counts()
            won = sum(c for key, c in counts.items() if self.wins(q, int(key, 2) % 2**n, int(key, 2) >> n))
            rate += float(pi[q]) * won / shots
        return rate

"""Linear systems over Z_2 and their nonlocal games.

A linear system ``Ax = b`` over Z_2 has an associated two-player nonlocal game
G(A, b): the referee sends Alice an equation ``i`` and Bob a variable ``j`` in
the support of that equation. Alice answers with an assignment to the variables
of equation ``i``; Bob answers with a value for ``x_j``. They win if Alice's
assignment satisfies equation ``i`` and agrees with Bob on ``x_j``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Dict, Iterator, List, Sequence, Tuple

Question = Tuple[int, int]  # (equation index i, variable index j)


@dataclass(frozen=True)
class LinearSystem:
    """An m x n linear system Ax = b over Z_2.

    Attributes:
        A: m rows of length n with entries in {0, 1}.
        b: right-hand side of length m with entries in {0, 1}.
        variable_names: optional display names for the n variables.
    """

    A: Tuple[Tuple[int, ...], ...]
    b: Tuple[int, ...]
    variable_names: Tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.A:
            raise ValueError("A must have at least one row")
        n = len(self.A[0])
        if any(len(row) != n for row in self.A):
            raise ValueError("all rows of A must have the same length")
        if len(self.b) != len(self.A):
            raise ValueError("b must have one entry per row of A")
        if any(v not in (0, 1) for row in self.A for v in row) or any(v not in (0, 1) for v in self.b):
            raise ValueError("entries of A and b must be 0 or 1")
        if self.variable_names and len(self.variable_names) != n:
            raise ValueError("variable_names must have one entry per column of A")
        for i in range(len(self.A)):
            if not self.support(i):
                raise ValueError(f"equation {i} has no variables")

    @classmethod
    def from_equations(
        cls,
        equations: Sequence[Sequence[int]],
        b: Sequence[int],
        num_variables: int,
        variable_names: Sequence[str] = (),
    ) -> "LinearSystem":
        """Build a system from the list of variable indices in each equation."""
        rows = []
        for eq in equations:
            row = [0] * num_variables
            for j in eq:
                row[j] ^= 1
            rows.append(tuple(row))
        return cls(tuple(rows), tuple(int(v) % 2 for v in b), tuple(variable_names))

    @property
    def num_equations(self) -> int:
        return len(self.A)

    @property
    def num_variables(self) -> int:
        return len(self.A[0])

    def support(self, i: int) -> List[int]:
        """V_A(i): the variables appearing in equation i."""
        return [j for j, a in enumerate(self.A[i]) if a]

    def name(self, j: int) -> str:
        return self.variable_names[j] if self.variable_names else f"x{j}"

    def questions(self) -> List[Question]:
        """All question pairs (i, j) with j in the support of equation i."""
        return [(i, j) for i in range(self.num_equations) for j in self.support(i)]

    def uniform_distribution(self) -> Dict[Question, Fraction]:
        """The uniform distribution on question pairs."""
        qs = self.questions()
        return {q: Fraction(1, len(qs)) for q in qs}

    def satisfies(self, i: int, assignment: Dict[int, int]) -> bool:
        """Whether an assignment to the support of equation i satisfies it."""
        return sum(assignment[j] for j in self.support(i)) % 2 == self.b[i]

    def satisfying_assignments(self, i: int) -> Iterator[Dict[int, int]]:
        """All assignments to the support of equation i that satisfy it."""
        vars_i = self.support(i)
        for values in product((0, 1), repeat=len(vars_i)):
            assignment = dict(zip(vars_i, values))
            if self.satisfies(i, assignment):
                yield assignment

    def has_classical_solution(self) -> bool:
        """Whether Ax = b has a solution over Z_2 (Gaussian elimination)."""
        rows = [list(row) + [bit] for row, bit in zip(self.A, self.b)]
        n = self.num_variables
        pivot_row = 0
        for col in range(n):
            pivot = next((r for r in range(pivot_row, len(rows)) if rows[r][col]), None)
            if pivot is None:
                continue
            rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
            for r in range(len(rows)):
                if r != pivot_row and rows[r][col]:
                    rows[r] = [x ^ y for x, y in zip(rows[r], rows[pivot_row])]
            pivot_row += 1
        return not any(all(v == 0 for v in row[:n]) and row[n] == 1 for row in rows)

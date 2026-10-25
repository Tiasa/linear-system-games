"""The two standard linear system games with perfect quantum strategies."""

from __future__ import annotations

from typing import Tuple

from .operators import PauliSolution
from .system import LinearSystem


def magic_square() -> Tuple[LinearSystem, PauliSolution]:
    """The Mermin-Peres magic square: 9 variables, 6 equations, 2 qubits.

    Variables x_{rc} fill a 3 x 3 grid. Each row sums to 0 and each column
    sums to 1 mod 2, which has no classical solution: the rows force the total
    to be 0 and the columns force it to be 1.
    """
    names = tuple(f"x{r}{c}" for r in range(1, 4) for c in range(1, 4))
    rows = [[3 * r + c for c in range(3)] for r in range(3)]
    cols = [[3 * r + c for r in range(3)] for c in range(3)]
    system = LinearSystem.from_equations(rows + cols, [0, 0, 0, 1, 1, 1], 9, names)
    grid = [
        ["XI", "IX", "XX"],
        ["IZ", "ZI", "ZZ"],
        ["-XZ", "-ZX", "YY"],
    ]
    solution = PauliSolution({3 * r + c: grid[r][c] for r in range(3) for c in range(3)})
    return system, solution


def magic_pentagram() -> Tuple[LinearSystem, PauliSolution]:
    """Mermin's magic pentagram: 10 variables on 5 lines of 4, 3 qubits.

    Every variable lies on exactly two lines, so the left-hand sides sum to 0,
    while the right-hand sides sum to 1: no classical solution.
    """
    labels = ["XXX", "XYY", "YXY", "YYX", "XII", "IXI", "IIX", "IYI", "IIY", "YII"]
    lines = [
        [0, 1, 2, 3],
        [0, 4, 5, 6],
        [1, 4, 7, 8],
        [2, 9, 5, 8],
        [3, 9, 7, 6],
    ]
    system = LinearSystem.from_equations(lines, [1, 0, 0, 0, 0], 10, tuple(labels))
    solution = PauliSolution(dict(enumerate(labels)))
    return system, solution


EXAMPLES = {
    "magic-square": magic_square,
    "magic-pentagram": magic_pentagram,
}

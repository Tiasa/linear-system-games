"""Classical value of a linear system game, by exhaustive search.

Shared randomness never helps the players beyond the best deterministic
strategy, so the classical value is a maximum over deterministic strategies.
For each assignment ``x`` Bob could use, Alice's best reply to equation ``i`` is
the satisfying assignment that agrees with ``x`` on the most question pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Dict, Optional, Tuple

from .system import LinearSystem, Question

MAX_VARIABLES = 22


@dataclass(frozen=True)
class ClassicalResult:
    value: Fraction
    bob_assignment: Tuple[int, ...]
    alice_assignments: Tuple[Dict[int, int], ...]


def classical_value(
    system: LinearSystem, distribution: Optional[Dict[Question, Fraction]] = None
) -> ClassicalResult:
    """Exact classical value and an optimal deterministic strategy.

    Args:
        system: the linear system defining the game.
        distribution: probabilities of question pairs; uniform by default.
    """
    if system.num_variables > MAX_VARIABLES:
        raise ValueError(f"exhaustive search supports at most {MAX_VARIABLES} variables")
    pi = distribution or system.uniform_distribution()
    satisfying = [list(system.satisfying_assignments(i)) for i in range(system.num_equations)]

    best: Optional[ClassicalResult] = None
    for x in product((0, 1), repeat=system.num_variables):
        total = Fraction(0)
        replies = []
        for i in range(system.num_equations):
            def score(a: Dict[int, int], i: int = i) -> Fraction:
                return sum((pi.get((i, j), Fraction(0)) for j in system.support(i) if a[j] == x[j]), Fraction(0))

            reply = max(satisfying[i], key=score)
            total += score(reply)
            replies.append(reply)
        if best is None or total > best.value:
            best = ClassicalResult(total, tuple(x), tuple(replies))
    assert best is not None
    return best

"""Linear system nonlocal games: classical values and perfect quantum strategies."""

from .classical import ClassicalResult, classical_value
from .examples import EXAMPLES, magic_pentagram, magic_square
from .operators import PauliSolution, check_operator_solution
from .strategy import QuantumStrategy
from .system import LinearSystem

__all__ = [
    "ClassicalResult",
    "EXAMPLES",
    "LinearSystem",
    "PauliSolution",
    "QuantumStrategy",
    "check_operator_solution",
    "classical_value",
    "magic_pentagram",
    "magic_square",
]

__version__ = "0.1.0"

from fractions import Fraction

import pytest

from lsgames import LinearSystem, classical_value, magic_pentagram, magic_square


def test_from_equations_builds_rows():
    system = LinearSystem.from_equations([[0, 1], [1, 2]], [1, 0], 3)
    assert system.A == ((1, 1, 0), (0, 1, 1))
    assert system.support(1) == [1, 2]
    assert system.questions() == [(0, 0), (0, 1), (1, 1), (1, 2)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"A": ((1, 0), (1,)), "b": (0, 0)},
        {"A": ((1, 2),), "b": (0,)},
        {"A": ((1, 1),), "b": (0, 1)},
        {"A": ((0, 0),), "b": (0,)},
    ],
)
def test_invalid_systems_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LinearSystem(**kwargs)


def test_satisfiable_system_has_classical_value_one():
    system = LinearSystem.from_equations([[0, 1], [1, 2], [0, 2]], [1, 1, 0], 3)
    assert system.has_classical_solution()
    assert classical_value(system).value == 1


def test_magic_square_is_not_classically_satisfiable():
    system, _ = magic_square()
    assert not system.has_classical_solution()
    # Best deterministic strategy satisfies 5 of 6 equations and loses one of 18 question pairs.
    assert classical_value(system).value == Fraction(17, 18)


def test_magic_pentagram_is_not_classically_satisfiable():
    system, _ = magic_pentagram()
    assert not system.has_classical_solution()
    assert classical_value(system).value == Fraction(19, 20)


def test_classical_value_respects_distribution():
    system, _ = magic_square()
    # Only ask about the first row: Alice and Bob can always win.
    pi = {(0, j): Fraction(1, 3) for j in system.support(0)}
    assert classical_value(system, pi).value == 1

import numpy as np
import pytest

from lsgames import PauliSolution, QuantumStrategy, check_operator_solution, magic_pentagram, magic_square
from lsgames.strategy import MeasurementBasis

GAMES = [magic_square, magic_pentagram]


@pytest.mark.parametrize("game", GAMES)
def test_pauli_solutions_are_valid(game):
    system, solution = game()
    assert check_operator_solution(system, solution) == []


def test_broken_solution_is_reported():
    system, solution = magic_square()
    labels = dict(solution.labels)
    labels[8] = "XY"  # no longer commutes with its row and column
    problems = check_operator_solution(system, PauliSolution(labels))
    assert any("do not commute" in p for p in problems)
    with pytest.raises(ValueError):
        QuantumStrategy(system, PauliSolution(labels))


@pytest.mark.parametrize("game", GAMES)
def test_alice_answers_always_satisfy_their_equation(game):
    system, solution = game()
    strategy = QuantumStrategy(system, solution)
    for i in range(system.num_equations):
        for answer in strategy.alice_measurement(i).answers:
            assert system.satisfies(i, answer)


@pytest.mark.parametrize("game", GAMES)
def test_measurement_bases_are_unitary(game):
    system, solution = game()
    strategy = QuantumStrategy(system, solution)
    dim = 2**strategy.num_qubits
    for i in range(system.num_equations):
        u = strategy.alice_measurement(i).basis
        assert np.allclose(u.conj().T @ u, np.eye(dim))


@pytest.mark.parametrize("game", GAMES)
def test_quantum_strategy_is_perfect(game):
    system, solution = game()
    strategy = QuantumStrategy(system, solution)
    for question in system.questions():
        assert strategy.exact_win_probability(question) == pytest.approx(1.0, abs=1e-9)
    assert strategy.win_probability() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("game", GAMES)
def test_sampled_win_rate_is_perfect(game):
    system, solution = game()
    strategy = QuantumStrategy(system, solution)
    assert strategy.sampled_win_rate(shots=200, seed=11) == pytest.approx(1.0)


def test_bob_must_measure_the_transpose():
    """Without the transpose, Bob's single-Y observables give the wrong sign."""
    system, solution = magic_pentagram()
    strategy = QuantumStrategy(system, solution)

    def untransposed(j):
        m = solution.matrix(j)
        _, vectors = np.linalg.eigh(m)
        answers = tuple(
            {j: 0 if np.vdot(vectors[:, t], m @ vectors[:, t]).real > 0 else 1} for t in range(vectors.shape[1])
        )
        return MeasurementBasis(vectors, answers)

    strategy.bob_measurement = untransposed  # type: ignore[method-assign]
    y_variable = next(j for j, label in solution.labels.items() if label == "IYI")
    equation = next(i for i in range(system.num_equations) if y_variable in system.support(i))
    assert strategy.exact_win_probability((equation, y_variable)) == pytest.approx(0.0, abs=1e-9)
    assert strategy.win_probability() < 1.0

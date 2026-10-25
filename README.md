# linear-system-games

Classical values and perfect quantum strategies for **linear system nonlocal games**, with the quantum strategies built and simulated as Qiskit circuits.

## Background

A linear system `Ax = b` over Z₂ defines a two-player game. The referee sends Alice an equation `i` and Bob a variable `j` from that equation. Alice answers with values for every variable in equation `i`, Bob answers with a value for `x_j`, and they win if Alice's values satisfy the equation and agree with Bob's on `x_j`. They cannot communicate once the questions are sent.

If the system has no solution, no classical strategy always wins. Some unsatisfiable systems, the Mermin–Peres magic square and Mermin's magic pentagram among them, still have a perfect *quantum* strategy. The strategy comes from an **operator solution**: an assignment of ±1-valued observables to the variables such that the observables in each equation commute and multiply to (−1)^{b_i}·I (Cleve–Mittal 2014; Cleve–Liu–Slofstra 2017).

This package:

- computes the exact classical value of a small game by exhaustive search over deterministic strategies;
- checks that a set of Pauli observables is an operator solution;
- turns an operator solution into a Qiskit circuit for each question pair: shared EPR pairs, a joint measurement of Alice's commuting observables, and Bob's measurement of the transposed observable;
- computes the exact win probability from the statevector and estimates it by sampling with Qiskit's `StatevectorSampler`.

| Game | Equations | Variables | Classical value | Quantum (2 or 3 EPR pairs) |
|---|---|---|---|---|
| Magic square | 6 | 9 | 17/18 | 1 |
| Magic pentagram | 5 | 10 | 19/20 | 1 |

Classical values are for the uniform distribution on (equation, variable) pairs.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and Qiskit 1.0 or later.

## Usage

```bash
lsgames magic-square --shots 1000 --seed 1 --draw
```

```
Game: magic-square  (6 equations, 9 variables)
Classically satisfiable: False
Classical value: 17/18 = 0.9444
Quantum strategy: 2 shared EPR pairs
Quantum win probability (exact): 1.000000
Quantum win rate (sampled, 1000 shots per question): 1.0000
```

From Python:

```python
from lsgames import QuantumStrategy, classical_value, magic_pentagram

system, solution = magic_pentagram()
print(classical_value(system).value)          # 19/20
strategy = QuantumStrategy(system, solution)
print(strategy.win_probability())             # 1.0
print(strategy.circuit((0, 0)).draw())        # circuit for equation 0, variable 0
```

To try your own game, build a `LinearSystem` (for example with `LinearSystem.from_equations`) and a `PauliSolution` mapping each variable to a signed Pauli label such as `"-XZ"`. `QuantumStrategy` raises an error if the labels are not an operator solution.

## How the quantum strategy works

Alice and Bob share n EPR pairs, |Φ⟩ = 2^{−n/2} Σₓ |x⟩|x⟩.

- **Alice** rotates her register into a common eigenbasis of the observables in her equation, found by diagonalizing a generic real combination of them, and measures. Her answers are the eigenvalues. Because the observables multiply to (−1)^{b_i}·I, her answers always satisfy the equation.
- **Bob** measures the transpose X_jᵀ of his variable's observable. Since (M ⊗ I)|Φ⟩ = (I ⊗ Mᵀ)|Φ⟩, his outcome always matches Alice's value for x_j. The test suite checks that measuring X_j instead of X_jᵀ loses on the pentagram's single-Y observables.

## Layout

```
src/lsgames/
  system.py      linear systems over Z_2 and their games
  classical.py   exact classical value by exhaustive search
  operators.py   Pauli operator solutions and their validation
  strategy.py    Qiskit circuits, exact and sampled win probabilities
  examples.py    magic square and magic pentagram
  cli.py         command-line report
tests/           pytest suite
```

## Tests

```bash
pytest
```

## License

MIT

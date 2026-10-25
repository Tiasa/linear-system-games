"""Command-line report: classical value vs. simulated quantum win rate."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .classical import classical_value
from .examples import EXAMPLES
from .strategy import QuantumStrategy


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lsgames",
        description="Compare classical and quantum strategies for a linear system game.",
    )
    parser.add_argument("game", choices=sorted(EXAMPLES), help="which example game to analyse")
    parser.add_argument("--shots", type=int, default=1000, help="shots per question pair (default 1000)")
    parser.add_argument("--seed", type=int, default=None, help="sampler seed for reproducible runs")
    parser.add_argument("--draw", action="store_true", help="print the circuit for the first question pair")
    args = parser.parse_args(argv)

    system, solution = EXAMPLES[args.game]()
    print(f"Game: {args.game}  ({system.num_equations} equations, {system.num_variables} variables)")
    print(f"Classically satisfiable: {system.has_classical_solution()}")

    classical = classical_value(system)
    print(f"Classical value: {classical.value} = {float(classical.value):.4f}")

    strategy = QuantumStrategy(system, solution)
    print(f"Quantum strategy: {solution.num_qubits} shared EPR pairs")
    print(f"Quantum win probability (exact): {strategy.win_probability():.6f}")
    rate = strategy.sampled_win_rate(shots=args.shots, seed=args.seed)
    print(f"Quantum win rate (sampled, {args.shots} shots per question): {rate:.4f}")

    if args.draw:
        first = system.questions()[0]
        print(f"\nCircuit for question {first} (equation, variable):")
        print(strategy.circuit(first).draw(output="text"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

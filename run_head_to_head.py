"""
run_head_to_head.py
Corre solo los duelos directos JUAN vs SANTI y JUAN vs GUTI.
Genera results_h2h.json para el notebook.
"""

import json
from generate_data import (
    FVMCPolicy,
    FlatMonteCarlo,
    QLearningAgent,
    exp7_vs_flatmc,
    exp8_vs_qlearning,
)

N_GAMES = 200  # cambia este valor si quieres mas partidas

results = {
    "exp7_vs_flatmc": exp7_vs_flatmc(FVMCPolicy, FlatMonteCarlo, n_games=N_GAMES),
    "exp8_vs_qlearning": exp8_vs_qlearning(FlatMonteCarlo, QLearningAgent, n_games=N_GAMES),
}

with open("results_h2h.json", "w") as f:
    json.dump(results, f, indent=2)

print("Listo -> results_h2h.json")

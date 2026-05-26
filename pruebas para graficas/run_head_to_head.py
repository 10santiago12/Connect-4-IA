"""
run_head_to_head.py
Corre solo los duelos directos JUAN vs SANTI y JUAN vs GUTI.
Genera results_h2h.json para el notebook.
"""

import importlib.util
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from connect4.connect_state import ConnectState

def _load_policy_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

FVMCPolicy = _load_policy_class("groups/Group C SANTI/policy.py", "FVMCPolicy")
FlatMonteCarlo = _load_policy_class("groups/Group A JUAN/policy.py", "FlatMonteCarlo")
QLearningAgent = _load_policy_class("groups/Group B GUTI/policy.py", "QLearningAgent")

N_GAMES = 10 # cambia este valor si quieres mas partidas
FLATMC_N = 200  # rollouts por accion en FlatMonteCarlo (menor = mas rapido)

def evaluate_head_to_head(agent_a, agent_b, name_a, name_b, n_games=200):
    """Enfrenta dos agentes ya montados, alternando colores."""
    results = {name_a: 0, name_b: 0, "Draw": 0}
    detail = []
    for i in range(n_games):
        if i % 2 == 0:
            red, red_name = agent_a, name_a
            yel, yel_name = agent_b, name_b
        else:
            red, red_name = agent_b, name_b
            yel, yel_name = agent_a, name_a

        state = ConnectState()
        while not state.is_final():
            col = red.act(state.board) if state.player == -1 else yel.act(state.board)
            state = state.transition(col)

        w = state.get_winner()
        winner = red_name if w == -1 else (yel_name if w == 1 else "Draw")
        results[winner] += 1
        detail.append(winner)

    return results, detail

def _cumulative_wr(detail, winner_name):
    running = 0
    out = []
    for i, w in enumerate(detail, 1):
        if w == winner_name:
            running += 1
        out.append(round(running / i, 4))
    return out

def exp7_vs_flatmc(n_games=200):
    print("\n[7] FVMC vs Flat Monte Carlo (JUAN)...")
    fvmc = FVMCPolicy(n_vs_random=40_000, n_self_play=80_000,
                      heuristic_w=0.4, cache=None, seed=42)
    flatmc = FlatMonteCarlo(N=FLATMC_N)
    fvmc.mount()
    flatmc.mount()
    results, detail = evaluate_head_to_head(fvmc, flatmc, "FVMC", "FlatMC", n_games)
    print(f"  FVMC={results['FVMC']}  FlatMC={results['FlatMC']}  Draw={results['Draw']}")
    return {
        "results": results,
        "cumulative_wr_flatmc": _cumulative_wr(detail, "FlatMC"),
        "n_games": n_games
    }

def exp8_vs_qlearning(n_games=200):
    print("\n[8] Flat Monte Carlo (JUAN) vs Q-Learning (GUTI)...")
    flatmc = FlatMonteCarlo(N=FLATMC_N)
    ql = QLearningAgent()
    flatmc.mount()
    ql.mount()
    results, detail = evaluate_head_to_head(flatmc, ql, "FlatMC", "QLearning", n_games)
    print(f"  FlatMC={results['FlatMC']}  QLearning={results['QLearning']}  Draw={results['Draw']}")
    return {
        "results": results,
        "cumulative_wr_flatmc": _cumulative_wr(detail, "FlatMC"),
        "n_games": n_games
    }

if os.path.exists("results.json"):
    with open("results.json") as f:
        cached = json.load(f)
    if "exp7_vs_flatmc" in cached and "exp8_vs_qlearning" in cached:
        results = {
            "exp7_vs_flatmc": cached["exp7_vs_flatmc"],
            "exp8_vs_qlearning": cached["exp8_vs_qlearning"],
        }
    else:
        results = {
            "exp7_vs_flatmc": exp7_vs_flatmc(n_games=N_GAMES),
            "exp8_vs_qlearning": exp8_vs_qlearning(n_games=N_GAMES),
        }
else:
    results = {
        "exp7_vs_flatmc": exp7_vs_flatmc(n_games=N_GAMES),
        "exp8_vs_qlearning": exp8_vs_qlearning(n_games=N_GAMES),
    }

with open("results_h2h.json", "w") as f:
    json.dump(results, f, indent=2)

print("Listo -> results_h2h.json")
